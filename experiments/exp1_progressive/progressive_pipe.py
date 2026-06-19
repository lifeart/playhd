"""E1 -- Progressive play-while-processing core (READ-ONLY reuse of server/ + prototype/).

This module builds a *fragmented MP4* (fMP4) byte stream that interleaves the upscaled
instant-mode video with the matching source audio, in ONE container, emitted INCREMENTALLY
so a browser can start playing after a short lead buffer while processing continues.

Why fragmented MP4 (the delivery decision, proven in measure.py):
  * The current single-file path uses `movflags=+faststart`, which moves the moov atom to
    the FRONT -- but ffmpeg/PyAV can only compute that moov AFTER the whole clip is encoded
    (it needs every sample's size/offset), then rewrites the file. So `+faststart` is a FINAL
    post-process: a chunked progressive download of it CANNOT start before EOF.
  * A fragmented MP4 (`movflags=empty_moov+frag_keyframe+default_base_moof`) writes a tiny
    init `moov` (no samples) UP FRONT, then self-contained `moof`+`mdat` fragments as each GOP
    completes -- no final rewrite, no backward seek. A plain `<video src>` (progressive HTTP)
    OR a Media Source Extensions `SourceBuffer.appendBuffer` can both consume this byte stream
    and begin playback after the first fragment. Same PyAV encode path, just different movflags
    + a small encoder GOP + audio interleaved into the same container.

Audio sync (vs server/pipeline_api._mux_av which muxes only at the END):
  We open the source ONCE, demux its (compressed) audio packets, and feed them into the SAME
  output container, kept slightly AHEAD of the running video PTS so each video fragment can be
  flushed with its covering audio. AAC sources are copied (no re-encode, like _mux_av); non-AAC
  are transcoded to AAC best-effort. av_interleaved_write_frame orders the two streams by dts.

Two producers:
  * InstantProducer  -- the REAL instant fast path (anchor-only SR + GPU recon + GPU grain),
                        imported read-only from pipeline_api. Honest TTFF / sustain numbers.
  * BicubicProducer  -- a GPU-FREE cv2 bicubic x2 upscale. Used to exercise the delivery path
                        and the byte-stream validator at length WITHOUT touching the shared GPU.

Nothing here mutates a shared file; everything imports server/pipeline_api + prototype READ-ONLY.
"""

import os
import sys
import gc
import time
from fractions import Fraction

import numpy as np
import av
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_SERVER = os.path.join(_REPO, "server")
_PROTO = os.path.join(_REPO, "prototype")
for _p in (_SERVER, _PROTO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_api as pipe   # noqa: E402  READ-ONLY: stream_gops, MODE_CONFIG, encoders, consts

try:
    import torch as _torch     # noqa: E402  used only to free the MPS allocator between chunks
except Exception:
    _torch = None


# --------------------------------------------------------------------------- #
# Tunables (local to this experiment; do NOT shadow pipeline_api's).
# --------------------------------------------------------------------------- #
# Encoder GOP / keyframe interval. The fragmented muxer cuts a fragment at every keyframe, so
# this == the fragment length. Smaller -> lower time-to-first-fragment (lower TTFF) + finer
# buffer granularity, at a small bitrate cost (more keyframes). 12 frames ~= 0.5 s at 24 fps.
FRAG_GOP = 12

# Keep the source audio fed this many seconds AHEAD of the current video PTS so a just-finished
# video fragment always has its covering audio available to flush (else the interleaver stalls
# the fragment waiting for audio).
AUDIO_LOOKAHEAD_S = 1.0

_FRAG_MOVFLAGS = "empty_moov+frag_keyframe+default_base_moof"


def _free_gpu():
    gc.collect()
    if _torch is not None:
        try:
            if _torch.backends.mps.is_available():
                _torch.mps.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Non-seekable byte sink: a PyAV custom-IO target that just accumulates bytes. The fragmented
# mp4 muxer never seeks backward (that is the whole point of empty_moov), so we deliberately
# expose NO seek()/tell() -> libav treats it as a non-seekable pipe and streams sequentially.
# drain() hands the accumulated bytes to the HTTP response and clears the buffer.
# --------------------------------------------------------------------------- #
class ByteSink:
    def __init__(self):
        self._buf = bytearray()
        self.total = 0

    def write(self, b):
        b = bytes(b)
        self._buf += b
        self.total += len(b)
        return len(b)

    def drain(self):
        if not self._buf:
            return b""
        out = bytes(self._buf)
        self._buf.clear()
        return out


# --------------------------------------------------------------------------- #
# Audio iterators (mirror pipeline_api._copy_audio_iter / _transcode_audio_iter, but yield
# (t_seconds, packet) WITHOUT a bound stream -- the muxer binds aout itself).
# --------------------------------------------------------------------------- #
def _copy_audio_pairs(scont, ain):
    """Yield (t_seconds, aac_packet) for every source audio packet, in order."""
    for pkt in scont.demux(ain):
        if pkt.dts is None:
            continue
        t = float(pkt.pts * pkt.time_base) if pkt.pts is not None else 0.0
        yield t, pkt


def _transcode_audio_pairs(scont, ain, aout):
    """Decode -> resample -> AAC-encode the source audio, chunked to the encoder frame size.
    Yields (t_seconds, aac_packet). Only exercised for non-AAC sources (the test clip is AAC)."""
    cc = aout.codec_context
    resampler = av.AudioResampler(format=cc.format, layout=cc.layout, rate=cc.sample_rate)
    fifo = av.AudioFifo()
    for frame in scont.decode(ain):
        frame.pts = None
        for rf in resampler.resample(frame):
            fifo.write(rf)
        fs = cc.frame_size or 1024
        while fifo.samples >= fs:
            ch = fifo.read(fs)
            for pkt in aout.encode(ch):
                t = float(pkt.pts * pkt.time_base) if pkt.pts is not None else 0.0
                yield t, pkt
    rem = fifo.read()
    if rem is not None:
        for pkt in aout.encode(rem):
            t = float(pkt.pts * pkt.time_base) if pkt.pts is not None else 0.0
            yield t, pkt
    for pkt in aout.encode():           # flush
        t = float(pkt.pts * pkt.time_base) if pkt.pts is not None else 0.0
        yield t, pkt


# --------------------------------------------------------------------------- #
# Fragmented muxer: ONE output container holding the fresh HD video stream + the source audio,
# written to a non-seekable ByteSink as fragments. Audio is kept ahead of the video PTS so each
# video fragment flushes with its audio.
# --------------------------------------------------------------------------- #
class FragmentMuxer:
    def __init__(self, sink, fps, src_audio_path, w_hd, h_hd, codec=None, gop=FRAG_GOP):
        self.sink = sink
        self.fps = float(fps)
        self.gop = gop
        self.out = av.open(sink, "w", format="mp4", options={"movflags": _FRAG_MOVFLAGS})

        # ---- fresh HD video stream (we re-encode the upscaled frames) ----
        if codec is None:
            codec = pipe._HW_CODEC if pipe._hw_encode_available() else "libx264"
        self.codec = codec
        self.vst = self.out.add_stream(codec, rate=Fraction(fps).limit_denominator(100000))
        self.vst.width, self.vst.height, self.vst.pix_fmt = w_hd, h_hd, "yuv420p"
        self.vst.codec_context.gop_size = gop
        if codec == pipe._HW_CODEC:
            self.vst.bit_rate = int(w_hd * h_hd * float(fps) * pipe._HW_BPP)
            self.vst.options = {"realtime": "1", "allow_sw": "1"}
        else:
            self.vst.options = {"crf": "20", "g": str(gop), "keyint_min": str(gop)}

        # ---- source audio (copy AAC / transcode / none) ----
        self.scont = av.open(src_audio_path)
        self.ain = self.scont.streams.audio[0] if self.scont.streams.audio else None
        self.audio_note = "none (source has no audio)"
        self.aout = None
        self._a_iter = None
        self._a_pending = None
        if self.ain is not None:
            if self.ain.codec_context.name == "aac":
                self.aout = self.out.add_stream_from_template(self.ain)
                self.audio_note = "copied (aac, no re-encode)"
                self._a_iter = _copy_audio_pairs(self.scont, self.ain)
            else:
                self.aout = self.out.add_stream(
                    "aac", rate=self.ain.codec_context.sample_rate or 44100)
                self.audio_note = f"transcoded ({self.ain.codec_context.name} -> aac)"
                self._a_iter = _transcode_audio_pairs(self.scont, self.ain, self.aout)
            self._a_pending = next(self._a_iter, None)

        self.n_video = 0

    def _video_time(self):
        return self.n_video / self.fps

    def _feed_audio(self, upto_t):
        """Mux all pending source-audio packets with pts <= upto_t (keeps audio ahead of video)."""
        if self.aout is None:
            return
        while self._a_pending is not None and self._a_pending[0] <= upto_t:
            _, pkt = self._a_pending
            pkt.stream = self.aout
            self.out.mux(pkt)
            self._a_pending = next(self._a_iter, None)

    def write_frame(self, rgb_uint8):
        """Encode one upscaled HD frame, mux it, then feed audio up to (videoPTS + lookahead)."""
        vf = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(rgb_uint8, dtype=np.uint8), format="rgb24")
        for pkt in self.vst.encode(vf):
            pkt.stream = self.vst
            self.out.mux(pkt)
        self.n_video += 1
        self._feed_audio(self._video_time() + AUDIO_LOOKAHEAD_S)

    def close(self):
        for pkt in self.vst.encode():        # flush video encoder
            pkt.stream = self.vst
            self.out.mux(pkt)
        if self.aout is not None:            # drain remaining audio up to the real video end
            self._feed_audio(float("inf"))
        try:
            self.out.close()
        finally:
            self.scont.close()


# --------------------------------------------------------------------------- #
# Producers: yield HD RGB uint8 frames in display order.
# --------------------------------------------------------------------------- #
class BicubicProducer:
    """GPU-FREE cv2 bicubic x2 upscale. Used to validate the delivery path + buffer math at
    length without contending for the shared MPS GPU. NOT the real SR -- delivery test only."""

    def __init__(self, scale=pipe.INSTANT_SCALE):
        self.scale = scale
        self.w_hd = self.h_hd = None

    def frames(self, input_path, max_frames=None, soft_cap=24):
        for chunk in pipe.stream_gops(input_path, max_frames=max_frames, soft_cap=soft_cap):
            if self.w_hd is None:
                h_lr, w_lr = chunk[0][1].shape[:2]
                self.w_hd, self.h_hd = w_lr * self.scale, h_lr * self.scale
            for (_pt, lr_rgb, _mvs) in chunk:
                yield cv2.resize(lr_rgb, (self.w_hd, self.h_hd),
                                 interpolation=cv2.INTER_CUBIC)


class InstantProducer:
    """The REAL instant fast path, mirroring pipeline_api.process_clip lines ~843-891 exactly
    (anchor-only SR cache -> GPU-resident reconstruct -> adaptive B-leaf patch -> GPU grain ->
    single GPU->host download). Imported read-only; produces honest instant-mode HD frames."""

    def __init__(self):
        self.cfg = pipe.MODE_CONFIG["instant"]
        self.eff_scale = pipe.INSTANT_SCALE
        self.w_hd = self.h_hd = None
        if pipe._gpu_ops is None:
            raise RuntimeError("gpu_ops unavailable -> instant fast path cannot run on this box")

    def frames(self, input_path, max_frames=None, soft_cap=24):
        cfg = self.cfg
        derisk = pipe.derisk
        anchor_sr = pipe.anchor_sr
        fast_grain = pipe.fast_grain
        ggrain = None
        done = 0
        chunk_iter = pipe.stream_gops(input_path, max_frames=max_frames, soft_cap=soft_cap)
        for chunk in chunk_iter:
            if self.w_hd is None:
                h_lr, w_lr = chunk[0][1].shape[:2]
                self.w_hd, self.h_hd = w_lr * self.eff_scale, h_lr * self.eff_scale
            perframe_cache, _ac, sr_set = anchor_sr.build_anchor_cache(
                chunk, self.w_hd, self.h_hd, cfg["sr_mode"], occ_mode=cfg["occ"],
                fallback_thresh=pipe.INSTANT_FALLBACK_THRESH,
                tile=pipe.INSTANT_TILE_SR, gpu_cache=pipe.INSTANT_GPU_CACHE)
            _, R = derisk.reconstruct(
                chunk, None, self.eff_scale, True, cfg["occ"], perframe_cache, set(),
                backend=cfg["backend"], collect_metrics=False, download_output=False)
            anchor_sr.patch_high_fallback(
                chunk, R, self.w_hd, self.h_hd, cfg["sr_mode"],
                fallback_thresh=pipe.INSTANT_FALLBACK_THRESH, skip=sr_set,
                tile=pipe.INSTANT_TILE_SR)
            if ggrain is None and cfg["grain"] != "off":
                ggrain = fast_grain.GpuGrain(self.h_hd, self.w_hd, pipe._gpu_ops.device())
            for i in range(len(chunk)):
                recon_t = R[i]["recon"]
                if cfg["grain"] != "off":
                    recon_t = ggrain.apply(recon_t, done, cfg["grain"])
                recon = fast_grain.download_rgb(recon_t)
                yield recon
                done += 1
            del perframe_cache, R, chunk
            _free_gpu()


# --------------------------------------------------------------------------- #
# The streaming generator: produce -> encode/interleave -> drain bytes -> yield. Single-threaded
# (the GPU stays single-threaded as required); naturally back-pressured by the consumer awaiting
# each yield. Records per-frame + first-fragment timing into `timing` for the TTFF measurement.
# --------------------------------------------------------------------------- #
def stream_fragmented(producer, input_path, src_audio_path, fps, *, max_frames=None,
                      soft_cap=24, codec=None, gop=FRAG_GOP, timing=None):
    """Yield fMP4 byte chunks (init+moov first, then moof+mdat fragments) as they are produced.

    `timing` (optional dict) is filled in-place: t_start, t_first_bytes (init available),
    t_first_fragment (first moof+mdat available == first playable frame), per_frame[] wall
    timestamps, n_frames, total_bytes, audio_note.
    """
    t0 = time.perf_counter()
    if timing is None:
        timing = {}
    timing.update(t_start=t0, t_first_bytes=None, t_first_fragment=None,
                  per_frame=[], n_frames=0, total_bytes=0, audio_note=None)

    sink = ByteSink()
    muxer = None
    try:
        gen = producer.frames(input_path, max_frames=max_frames, soft_cap=soft_cap)
        for rgb in gen:
            if muxer is None:                      # first frame determines HD size -> open muxer
                h_hd, w_hd = rgb.shape[:2]
                muxer = FragmentMuxer(sink, fps, src_audio_path, w_hd, h_hd,
                                      codec=codec, gop=gop)
                timing["audio_note"] = muxer.audio_note
            muxer.write_frame(rgb)
            timing["per_frame"].append(time.perf_counter() - t0)
            timing["n_frames"] += 1
            data = sink.drain()
            if data:
                if timing["t_first_bytes"] is None:
                    timing["t_first_bytes"] = time.perf_counter() - t0
                # First emission that carries a media fragment (moof) == first playable frame.
                if timing["t_first_fragment"] is None and b"moof" in data:
                    timing["t_first_fragment"] = time.perf_counter() - t0
                timing["total_bytes"] += len(data)
                yield data
    finally:
        if muxer is not None:
            muxer.close()
            tail = sink.drain()
            if tail:
                timing["total_bytes"] += len(tail)
                yield tail
    timing["t_end"] = time.perf_counter() - t0


def probe_fps(path):
    return float(pipe._probe_fps(path))
