"""playhd Stage-1 pipeline wrapper -- STREAMING / constant-memory rebuild.

A thin, read-only wrapper around the validated prototype (`prototype/derisk.py` +
`sr.py` / `region_quality.py` / `grain.py`). It does NOT reimplement any of the
NEMO warp/mask/blend math -- it mirrors exactly what `derisk.run()` wires up, but
processes the WHOLE clip in **GOP-sized chunks** so memory stays bounded (one chunk)
regardless of clip length, ENCODES the upscaled HD frames into the output video
stream INCREMENTALLY, and muxes the SOURCE AUDIO into the result (copy if AAC, else
transcode to AAC) so the upscaled clip keeps its sound, in sync.

Why chunks: the old wrapper held the whole window in memory at HD (~10 MB/frame * N)
and opened `avcodec` on a window so large it hit `avcodec_open2` EAGAIN / OOM. Here we
open the input container ONCE, stream frames in display order, and cut a new chunk at
every I-frame (a self-contained backbone for `derisk.reconstruct`). A long GOP is
subdivided at P-frame boundaries (the cut P becomes a forced fresh anchor -- NEMO-style
re-anchoring, still self-contained) so a single chunk never exceeds SOFT_CAP_FRAMES.
Never more than one chunk of HD frames is alive at a time.

Two modes:
  * "instant" -- compact anchor (realesr-general-x4v3), backend=torch, occ=adaptive,
                 + per-frame film grain. The fast / real-time-style path (~0.4 s/frame).
  * "quality" -- heavy x4plus anchor (RealESRGAN_x4plus), region-aware detail blend,
                 + per-frame film grain. The slow / buffered path (~2.2 s/frame SR).

Public API:
    process_clip(input_path, mode, max_frames=None, out_path=None) -> out_path
    list_sources() / resolve_source(name)
    get_progress()                      -- live frames-done/total + ETA (for the UI)
    try_begin_job()/end_job()/is_busy() -- single-job lock (one process at a time)
"""

import os
import sys
import gc
import time
import threading
from fractions import Fraction

import numpy as np
import av

# --------------------------------------------------------------------------- #
# Import the prototype READ-ONLY. The prototype modules import each other by bare
# name (`import sr`, `import grain`, `import region_quality`), so the prototype dir
# must be on sys.path. `sr.py` resolves its weights via its own __file__, so cwd
# does not matter (weights already live in prototype/models/).
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_PROTO = os.path.join(_REPO, "prototype")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

import derisk            # noqa: E402  (the validated prototype)
import grain as _grain   # noqa: E402  (per-frame film-grain final pass)

try:                     # torch is pulled in by the prototype; used only to free MPS memory
    import torch as _torch   # noqa: E402
except Exception:
    _torch = None


def _free_gpu():
    """Release the MPS caching allocator + Python garbage BETWEEN chunks. Without this the
    MPS allocator's freed-but-cached memory creeps up over a long clip (it is NOT bounded by
    the per-chunk `del`s) and eventually fails an allocation with BlockingIOError/EAGAIN under
    memory pressure (the frame-630 crash). Active tensors are untouched; only cached memory is
    returned to the OS. Cheap (~ms), called once per processed chunk."""
    gc.collect()
    if _torch is not None:
        try:
            if _torch.backends.mps.is_available():
                _torch.mps.empty_cache()
        except Exception:
            pass
import layered_api as _layered  # noqa: E402  (LAYERED mode: two-pass-per-scene helpers)
import scene_detect              # noqa: E402  (robust scene-CUT detector -> forced fresh anchors)

SAMPLE_MP4 = os.path.join(_REPO, "sample.mp4")
OUTPUTS_DIR = os.path.join(_HERE, "outputs")
UPLOADS_DIR = os.path.join(_HERE, "uploads")
TESTDATA_DIR = os.path.join(_HERE, "testdata")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Every real SR net in sr.py is an x4 model -> the pipeline always runs at scale 4
# (matches the README: "--sr realesrgan ... it is an x4 net => use --scale 4").
SCALE = 4

# Max frames per processing chunk. A new chunk always starts at an I-frame; if a GOP is
# longer than this, we also cut at the next P-frame (a forced fresh anchor) so a single
# chunk -- and thus peak HD memory -- stays bounded. 48 == the validated prototype window.
SOFT_CAP_FRAMES = 48

# Mode -> exactly the flag combination the handoff recommends for each regime.
MODE_CONFIG = {
    "instant": dict(sr_mode="realesrgan",          # compact realesr-general-x4v3 (~0.13 s/frame SR)
                    backend="torch",               # MPS fast path (recon stays GPU-resident)
                    occ="adaptive",                # full quality both regimes, cheaper than full
                    region_aware=False,
                    grain="med",
                    label="Instant (compact anchor, torch/MPS, adaptive occ, grain)"),
    "quality": dict(sr_mode="realesrgan-x4plus",   # heavy RRDBNet x4plus (~2.2 s/frame, +61% sharper)
                    backend="torch",               # region-aware integration is tested on torch
                    occ="adaptive",
                    region_aware=True,             # OUTPUT-only motion-gated heavy/compact blend
                    grain="med",
                    label="Quality (x4plus anchor, region-aware blend, grain)"),
}

# The LAYERED mode is a SEPARATE third quality path (SR the static background plate ONCE per
# scene, manage only the moving foreground per frame). It is intentionally NOT a MODE_CONFIG
# entry so the instant/quality streaming body stays byte-for-byte untouched -- process_clip
# branches to _run_layered() for it. ALL_MODES gates the API validation + the single-job lock.
LAYERED_LABEL = "Quality — Layered (talking-head, static camera)"
ALL_MODES = tuple(MODE_CONFIG) + ("layered",)


def is_valid_mode(mode):
    return mode in ALL_MODES


# Stats from the most recent process_clip call (the server reads this to report timing).
LAST_STATS = {}


# --------------------------------------------------------------------------- #
# Single-job lock: only one (GPU-bound) process at a time. Concurrent callers get a
# clean BusyError -> the server returns 409, never a crash.
# --------------------------------------------------------------------------- #
_JOB_LOCK = threading.Lock()


class BusyError(RuntimeError):
    """Raised when a process is already running and another is requested."""


def try_begin_job():
    """Acquire the single-job lock without blocking. Returns True if acquired."""
    return _JOB_LOCK.acquire(blocking=False)


def end_job():
    if _JOB_LOCK.locked():
        try:
            _JOB_LOCK.release()
        except RuntimeError:
            pass


def is_busy():
    return _JOB_LOCK.locked()


# --------------------------------------------------------------------------- #
# Live progress (frames-done / total + ETA). Read by GET /api/progress.
# --------------------------------------------------------------------------- #
_PROGRESS = {
    "state": "idle",        # idle | probing | processing | muxing | done | error
    "done": 0,
    "total": 0,
    "mode": None,
    "elapsed_s": 0.0,
    "eta_s": None,
    "ms_per_frame": None,
    "message": "",
}
_PROG_LOCK = threading.Lock()


def _set_progress(**kw):
    with _PROG_LOCK:
        _PROGRESS.update(kw)


def get_progress():
    with _PROG_LOCK:
        return dict(_PROGRESS)


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
# Rough processing cost per mode (ms/frame, measured) -> the UI shows an estimate so the
# user never blind-launches a 12-hour job on a 34-min source. The pipeline is ~10x slower
# than real-time, so a long clip is a long render; this surfaces that BEFORE Process & Play.
MODE_MS_PER_FRAME = {"instant": 400.0, "quality": 2900.0, "layered": 470.0}


def _source_meta(path):
    """(n_frames, duration_s) for a source -- powers the UI's processing-time estimate."""
    try:
        n = probe_total_frames(path)
        fps = float(_probe_fps(path))
        dur = round(n / fps, 1) if (n and fps) else None
        return n, dur
    except Exception:
        return None, None


def _source_item(name, path):
    n, dur = _source_meta(path)
    est = ({m: round(n * ms / 1000.0) for m, ms in MODE_MS_PER_FRAME.items()}
           if n else {})
    return {"name": name, "path": path,
            "size_mb": round(os.path.getsize(path) / 1e6, 2),
            "n_frames": n, "duration_s": dur, "est_s": est}


def list_sources():
    """Available source mp4s the UI can offer without an upload: the repo sample +
    anything in server/uploads/ or server/testdata/. Each item carries duration + a
    per-mode processing-time estimate so the UI can warn before a long render."""
    items = []
    seen = set()
    if os.path.exists(SAMPLE_MP4):
        items.append(_source_item("sample.mp4", SAMPLE_MP4))
        seen.add("sample.mp4")
    for d in (TESTDATA_DIR, UPLOADS_DIR):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".mp4") and fn not in seen:
                items.append(_source_item(fn, os.path.join(d, fn)))
                seen.add(fn)
    return items


def resolve_source(name):
    """Map a UI source name to a server-side path (sample / testdata / upload).
    Raises ValueError for anything outside the known dirs (no path traversal)."""
    if name in (None, "", "sample.mp4"):
        return SAMPLE_MP4
    base = os.path.basename(name)               # strip any path components
    for d in (UPLOADS_DIR, TESTDATA_DIR):
        cand = os.path.join(d, base)
        if os.path.exists(cand):
            return cand
    if os.path.basename(SAMPLE_MP4) == base and os.path.exists(SAMPLE_MP4):
        return SAMPLE_MP4
    raise ValueError(f"unknown source {name!r}")


# --------------------------------------------------------------------------- #
# Container probing
# --------------------------------------------------------------------------- #
def _probe_fps(path):
    cont = av.open(path)
    try:
        vs = cont.streams.video[0]
        r = vs.average_rate or vs.base_rate or vs.guessed_rate
        return Fraction(r) if r else Fraction(25, 1)
    finally:
        cont.close()


def probe_total_frames(path, max_frames=None):
    """Total video frame count for the progress bar: nb_frames if the container
    reports it, else duration*fps. Capped by max_frames (test cap)."""
    cont = av.open(path)
    try:
        vs = cont.streams.video[0]
        n = int(vs.frames or 0)
        if n <= 0:
            dur = None
            if vs.duration is not None:
                dur = float(vs.duration * vs.time_base)
            elif cont.duration is not None:
                dur = float(cont.duration) / 1e6
            fps = float(vs.average_rate or vs.guessed_rate or 25)
            n = int(round(dur * fps)) if dur else 0
    finally:
        cont.close()
    if max_frames is not None:
        n = min(n, max_frames) if n else max_frames
    return n


# --------------------------------------------------------------------------- #
# Streaming GOP decoder: open the container ONCE, yield self-contained chunks in
# display order. Mirrors derisk.decode_lr_and_mvs (export_mvs, int pict_type map)
# but never re-decodes from 0 per chunk.
# --------------------------------------------------------------------------- #
def stream_gops(path, max_frames=None, soft_cap=SOFT_CAP_FRAMES, detect_cuts=True):
    """Yield lists of (ptype, lr_rgb, mvs) in display order. A new chunk starts at:
      * every codec I-frame (a self-contained backbone), AND
      * every DETECTED SCENE CUT (scene_detect: luma-diff + I-frame flag + hysteresis) --
        so a cut that is NOT a clean I-frame ALSO forces a fresh anchor; without this a
        mid-GOP cut would leave the chunk spanning it and derisk.reconstruct would warp the
        pre-cut anchor across the cut = a cross-cut smear, AND
      * the next backbone (I/P) frame once a GOP exceeds `soft_cap` (bounds peak HD memory).

    Each chunk therefore lies within ONE scene and is a self-contained backbone for
    derisk.reconstruct: its first frame is force-anchored (no in-chunk predecessor -> fresh
    per-frame SR, drift/smear reset). A scene cut that lands on a B-frame is handled cleanly --
    the new chunk's leading B-leaves have no in-chunk past anchor so they warp ONLY from the
    FUTURE (post-cut) anchor, and the previous chunk's trailing B-leaves lose their future
    (post-cut) reference so they warp past-only (old scene); neither bridges the cut.

    `detect_cuts=False` restores the OLD I-frame-only chunking (used by the BEFORE/AFTER
    verification to reproduce the cross-cut smear)."""
    cont = av.open(path)
    try:
        vs = cont.streams.video[0]
        vs.codec_context.options = {"flags2": "+export_mvs"}
        det = scene_detect.StreamingCutDetector() if detect_cuts else None
        chunk = []
        produced = 0
        for frame in cont.decode(vs):
            if max_frames is not None and produced >= max_frames:
                break
            ptype = {1: "I", 2: "P", 3: "B"}.get(int(frame.pict_type), "?")
            img = frame.to_ndarray(format="rgb24")
            # Scene-cut signal for THIS frame (cut between produced-1 and produced). Computed
            # BEFORE the boundary decision so the new chunk STARTS at the cut frame.
            is_cut = det.update(produced, ptype, img) if det is not None else False
            # Boundary BEFORE adding this frame so the new chunk STARTS at the boundary frame
            # (I-frame, scene cut, or -- at the soft cap -- the next backbone P).
            if chunk and (ptype == "I" or is_cut
                          or (len(chunk) >= soft_cap and ptype == "P")):
                yield chunk
                chunk = []
            try:
                sd = frame.side_data.get(derisk.SDType.MOTION_VECTORS)
            except Exception:
                sd = None
            mvs = sd.to_ndarray() if sd is not None else None
            chunk.append((ptype, img, mvs))
            produced += 1
        if chunk:
            yield chunk
    finally:
        cont.close()


# --------------------------------------------------------------------------- #
# Audio muxing: copy (AAC) or transcode (-> AAC), interleaved by time with the
# already-encoded video temp file. Bounded memory (streaming two-way merge).
# --------------------------------------------------------------------------- #
def _copy_audio_iter(scont, ain, aout, video_dur):
    """Yield (t_seconds, packet, aout) for source audio packets up to video_dur."""
    for pkt in scont.demux(ain):
        if pkt.dts is None:
            continue
        t = float(pkt.pts * pkt.time_base) if pkt.pts is not None else 0.0
        if t >= video_dur:
            break
        yield t, pkt, aout


def _transcode_audio_iter(scont, ain, aout, video_dur):
    """Best-effort decode->resample->AAC-encode of the source audio, chunked to the
    encoder frame size via an AudioFifo. Yields (t_seconds, packet, aout). NOTE: the
    verified test clip is AAC (copy path); this transcode path is exercised only for
    non-AAC sources."""
    cc = aout.codec_context
    resampler = av.AudioResampler(format=cc.format, layout=cc.layout, rate=cc.sample_rate)
    fifo = av.AudioFifo()
    for frame in scont.decode(ain):
        if frame.pts is not None and float(frame.pts * frame.time_base) >= video_dur:
            break
        frame.pts = None
        for rf in resampler.resample(frame):
            fifo.write(rf)
        fs = cc.frame_size or 1024
        while fifo.samples >= fs:
            chunk = fifo.read(fs)
            for pkt in aout.encode(chunk):
                t = float(pkt.pts * pkt.time_base) if pkt.pts is not None else 0.0
                yield t, pkt, aout
    rem = fifo.read()                      # drain remainder
    if rem is not None:
        for pkt in aout.encode(rem):
            t = float(pkt.pts * pkt.time_base) if pkt.pts is not None else 0.0
            yield t, pkt, aout
    for pkt in aout.encode():              # flush encoder
        t = float(pkt.pts * pkt.time_base) if pkt.pts is not None else video_dur
        yield t, pkt, aout


def _merge_mux(out, vit, ait):
    """Two-way time-ordered merge of two (t, packet, out_stream) iterators -> out."""
    vp = next(vit, None)
    ap = next(ait, None)
    while vp is not None or ap is not None:
        tv = vp[0] if vp is not None else float("inf")
        ta = ap[0] if ap is not None else float("inf")
        if tv <= ta:
            _, pkt, st = vp
            pkt.stream = st
            out.mux(pkt)
            vp = next(vit, None)
        else:
            _, pkt, st = ap
            pkt.stream = st
            out.mux(pkt)
            ap = next(ait, None)


def _mux_av(video_tmp, src_path, out_path, n_video_frames, fps):
    """Mux the encoded video temp + the source audio into the final mp4. Returns a
    short human note about what happened to the audio (copied / transcoded / none)."""
    vcont = av.open(video_tmp)
    scont = av.open(src_path)
    # +faststart moves the moov atom to the FRONT so browsers can start playback
    # progressively (a moov-at-end mp4 makes <video> stall at readyState 0 until the
    # whole file is fetched). This is what makes the result reliably web-playable.
    out = av.open(out_path, "w", options={"movflags": "+faststart"})
    try:
        vin = vcont.streams.video[0]
        vout = out.add_stream_from_template(vin)
        ain = scont.streams.audio[0] if scont.streams.audio else None
        video_dur = n_video_frames / float(fps)

        if ain is None:                                  # video-only, no crash
            for pkt in vcont.demux(vin):
                if pkt.dts is None:
                    continue
                pkt.stream = vout
                out.mux(pkt)
            return "none (source has no audio)"

        def vit():
            for pkt in vcont.demux(vin):
                if pkt.dts is None:
                    continue
                t = float(pkt.pts * pkt.time_base) if pkt.pts is not None else 0.0
                yield t, pkt, vout

        if ain.codec_context.name == "aac":              # mp4-compatible -> copy, no re-encode
            aout = out.add_stream_from_template(ain)
            note = "copied (aac, no re-encode)"
            ait = _copy_audio_iter(scont, ain, aout, video_dur)
        else:                                            # transcode to aac
            aout = out.add_stream("aac", rate=ain.codec_context.sample_rate or 44100)
            note = f"transcoded ({ain.codec_context.name} -> aac)"
            ait = _transcode_audio_iter(scont, ain, aout, video_dur)

        _merge_mux(out, vit(), ait)
        return note
    finally:
        out.close()
        vcont.close()
        scont.close()


# --------------------------------------------------------------------------- #
# Incremental HD video encoder (kept open across chunks).
# --------------------------------------------------------------------------- #
class _VideoWriter:
    def __init__(self, path, fps):
        self.cont = av.open(path, "w")
        self.fps = fps
        self.st = None

    def _ensure(self, w_hd, h_hd):
        if self.st is None:
            st = self.cont.add_stream("libx264", rate=self.fps)
            st.width, st.height, st.pix_fmt = w_hd, h_hd, "yuv420p"
            st.options = {"crf": "18"}
            self.st = st
        return self.st

    def write(self, rgb_uint8):
        h_hd, w_hd = rgb_uint8.shape[:2]
        st = self._ensure(w_hd, h_hd)
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb_uint8, dtype=np.uint8),
                                        format="rgb24")
        for pkt in st.encode(vf):
            self.cont.mux(pkt)

    def close(self):
        if self.st is not None:
            for pkt in self.st.encode():       # flush
                self.cont.mux(pkt)
        self.cont.close()


# --------------------------------------------------------------------------- #
# LAYERED mode: two-pass-per-scene, bounded-memory composite.
#
#   PASS 0  segment the clip into scenes (one lightweight streaming decode; holds 1 frame).
#   PASS A  per scene: build the temporal-median background plate from a CAPPED sample,
#           heavy-SR it ONCE (x4plus), SPILL the HD plate to disk; static-camera check ->
#           a MOVING scene is flagged for the region-aware fallback (no plate).
#   PASS B  stream every frame in GOP chunks: for a STATIC scene composite
#           alpha*fg_hd + (1-alpha)*plate_hd (RVM state threaded per scene); a MOVING scene
#           runs the regular 'quality' region-aware path. Encode incrementally, free.
#
# Two decode passes + the GOP stream; peak memory is one HD plate + one frame's working set.
# --------------------------------------------------------------------------- #
def _emit_frame_progress(done, total, t0, t_passB):
    """Live per-frame progress for PASS B. ms/frame & ETA are measured over PASS B only
    (so the one-off plate-building cost doesn't skew the streaming rate)."""
    now = time.perf_counter()
    rate = (now - t_passB) / max(1, done)
    eta = rate * (total - done) if total and done <= total else None
    _set_progress(done=done, elapsed_s=round(now - t0, 1),
                  eta_s=(round(eta, 1) if eta is not None else None),
                  ms_per_frame=round(rate * 1000.0, 1))


def _split_chunk_by_scene(chunk, start_idx, segs):
    """Split a GOP chunk (global display indices start_idx..) into runs that each lie in a
    SINGLE scene. Yields (sid, sub_chunk, sub_start_idx). Almost always one run (scene cuts
    are I-frames == chunk starts); only a rare mid-GOP RGB-diff cut produces two."""
    out = []
    cur_sid, cur, cur_start = None, [], start_idx
    for j, item in enumerate(chunk):
        idx = start_idx + j
        sid = _layered.scene_of(idx, segs)
        if sid != cur_sid:
            if cur:
                out.append((cur_sid, cur, cur_start))
            cur_sid, cur, cur_start = sid, [], idx
        cur.append(item)
    if cur:
        out.append((cur_sid, cur, cur_start))
    return out


def _quality_subchunk(sub, writer, done, total, t0, t_passB):
    """Run the regular 'quality' (region-aware) path on one self-contained sub-chunk -- the
    static-camera FALLBACK for a MOVING scene. Mirrors the quality branch of process_clip."""
    cfg = MODE_CONFIG["quality"]
    h_lr, w_lr = sub[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    perframe_cache = derisk.build_perframe_cache(sub, w_hd, h_hd, cfg["sr_mode"])
    region_gate = derisk._build_region_gate(sub, w_hd, h_hd, SCALE)
    _, R = derisk.reconstruct(
        sub, None, SCALE, True, cfg["occ"], perframe_cache, set(),
        backend=cfg["backend"], collect_metrics=False, download_output=True,
        region_gate=region_gate,
    )
    for i in range(len(sub)):
        recon = R[i]["recon"]
        if cfg["grain"] != "off":
            recon = _grain.apply_grain(recon, done, cfg["grain"])
        writer.write(recon)
        done += 1
        _emit_frame_progress(done, total, t0, t_passB)
    del perframe_cache, region_gate, R
    return done


def _run_layered(input_path, out_path, video_tmp, max_frames, t0):
    """LAYERED mode entry (called by process_clip under the single-job lock)."""
    plate_dir = out_path + ".plates"
    fps = _probe_fps(input_path)

    try:
        # ---- PASS 0: scene segmentation (bounded; holds 1 frame) ----
        _set_progress(state="segmenting", mode="layered", done=0, total=0, elapsed_s=0.0,
                      eta_s=None, ms_per_frame=None, message="segmenting scenes")
        segs, total = _layered.segment_scenes(input_path, max_frames=max_frames)
        n_scenes = len(segs)

        # ---- PASS A: build + heavy-SR one plate per scene, spilled to disk ----
        _set_progress(state="plate", done=0, total=n_scenes,
                      message=f"building {n_scenes} scene plate(s)")
        t_passA = time.perf_counter()
        model = _layered.load_matting_model()
        device = getattr(model, "_rvm_device", "cpu")

        def _plate_progress(done_s, tot_s):
            _set_progress(state="plate", done=done_s, total=tot_s,
                          elapsed_s=round(time.perf_counter() - t0, 1),
                          message=f"scene plate {done_s}/{tot_s} (heavy SR once per scene)")

        plates = _layered.build_scene_plates(
            input_path, segs, plate_dir, model, max_frames=max_frames,
            progress_cb=_plate_progress)
        passA_s = time.perf_counter() - t_passA
        plate_sr_total_s = sum(p.get("plate_sr_ms", 0.0) for p in plates.values()) / 1000.0
        fallback_sids = [sid for sid, p in plates.items() if p["fallback"]]

        # ---- PASS B: stream every frame, composite (static) or fall back (moving) ----
        _set_progress(state="processing", total=total, done=0,
                      message="compositing frames (foreground per frame, plate reused)")
        t_passB = time.perf_counter()
        writer = _VideoWriter(video_tmp, fps)
        done = 0
        n_chunks = 0
        w_lr = h_lr = w_hd = h_hd = None
        ratio = None
        cur_plate_sid, cur_plate = None, None
        rvm_sid, rec = None, [None] * 4
        try:
            for chunk in stream_gops(input_path, max_frames=max_frames):
                n_chunks += 1
                if w_lr is None:
                    h_lr, w_lr = chunk[0][1].shape[:2]
                    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
                    ratio = _layered.downsample_ratio(h_lr, w_lr)
                for sid, sub, _start in _split_chunk_by_scene(chunk, done, segs):
                    info = plates[sid]
                    if info["fallback"]:
                        rvm_sid = None            # next static scene re-anchors RVM state
                        done = _quality_subchunk(sub, writer, done, total, t0, t_passB)
                        continue
                    if cur_plate_sid != sid:      # load ONE HD plate at a time (bounded)
                        cur_plate = np.load(info["plate_path"])
                        cur_plate_sid = sid
                    if rvm_sid != sid:            # thread RVM recurrent state per scene
                        rec, rvm_sid = [None] * 4, sid
                    for (_ptype, img, _mvs) in sub:
                        pha, rec = _layered.matte_frame_np(model, img, rec, ratio, device)
                        out = _layered.composite_frame(img, pha, cur_plate, w_hd, h_hd)
                        out = _grain.apply_grain(out, done, "med")
                        writer.write(out)
                        done += 1
                        _emit_frame_progress(done, total, t0, t_passB)
                _free_gpu()                 # release MPS cache between chunks (see _free_gpu)
        finally:
            writer.close()
        passB_s = time.perf_counter() - t_passB

        if done == 0:
            raise RuntimeError(f"no frames decoded from {input_path}")

        # ---- mux source audio (copy AAC / transcode / none) ----
        _set_progress(state="muxing", message="muxing audio")
        audio_note = _mux_av(video_tmp, input_path, out_path, done, fps)

        total_s = time.perf_counter() - t0
        verdicts = {sid: p["verdict"] for sid, p in plates.items()}
        layered_note = (
            f"{n_scenes} scene(s); plate heavy-SR x4plus once/scene "
            f"({plate_sr_total_s:.1f}s SR total); "
            + ("all static -> layered" if not fallback_sids
               else f"scenes {fallback_sids} MOVING -> region-aware fallback")
            + "; matte: RVM (CC BY-NC-SA, non-commercial)"
        )
        LAST_STATS.clear()
        LAST_STATS.update({
            "mode": "layered", "label": LAYERED_LABEL, "input": input_path,
            "out_path": out_path, "n_frames": done, "n_chunks": n_chunks,
            "soft_cap": SOFT_CAP_FRAMES, "fps": float(fps), "scale": SCALE,
            "src_resolution": f"{w_lr}x{h_lr}", "out_resolution": f"{w_hd}x{h_hd}",
            "audio": audio_note,
            "n_scenes": n_scenes, "n_fallback": len(fallback_sids),
            "fallback_scenes": fallback_sids, "scene_verdicts": verdicts,
            "plate_sample_cap": _layered.PLATE_SAMPLE_CAP,
            "t_sr_s": round(plate_sr_total_s, 2),      # heavy plate SR (amortized once/scene)
            "t_recon_s": round(passB_s, 2),            # streaming composite pass
            "t_pass0_segment_s": None,                 # folded into passA wall below
            "t_passA_plate_s": round(passA_s, 2),
            "t_passB_composite_s": round(passB_s, 2),
            "t_total_s": round(total_s, 2),
            "ms_per_frame": round(total_s * 1000.0 / max(1, done), 1),
            "layered_note": layered_note,
        })
        _set_progress(state="done", done=done, total=total, elapsed_s=round(total_s, 1),
                      eta_s=0.0, ms_per_frame=round(total_s * 1000.0 / max(1, done), 1),
                      message="done")
        return out_path
    finally:
        import shutil
        shutil.rmtree(plate_dir, ignore_errors=True)   # drop spilled HD plates


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def process_clip(input_path, mode, max_frames=None, out_path=None, detect_cuts=True):
    """STREAMING upscale of the WHOLE clip (or the first `max_frames` for testing).

    Processes the input in GOP-sized chunks (bounded memory), encoding each chunk's
    HD frames into the output video stream incrementally, then muxes the source audio
    in (copy if AAC, else transcode). Returns out_path. Timing/metadata -> LAST_STATS.

    `detect_cuts` (default True): also cut a chunk at every detected SCENE CUT so no chunk
    warps across a cut (see stream_gops). Set False ONLY to reproduce the legacy I-frame-only
    chunking (the BEFORE case in the cross-cut-smear verification).

    Holds the single-job lock for its duration (raises BusyError if one is running)."""
    if not is_valid_mode(mode):
        raise ValueError(f"unknown mode {mode!r}; choices: {list(ALL_MODES)}")
    if not try_begin_job():
        raise BusyError("a clip is already being processed; please wait")

    cfg = MODE_CONFIG.get(mode)             # None for 'layered' (handled by _run_layered)
    if out_path is None:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(OUTPUTS_DIR, f"{stem}_{mode}.mp4")
    video_tmp = out_path + ".video.tmp.mp4"

    t0 = time.perf_counter()
    try:
        if mode == "layered":
            return _run_layered(input_path, out_path, video_tmp, max_frames, t0)
        _set_progress(state="probing", mode=mode, done=0, total=0, elapsed_s=0.0,
                      eta_s=None, ms_per_frame=None, message="probing input")
        fps = _probe_fps(input_path)
        total = probe_total_frames(input_path, max_frames)
        _set_progress(state="processing", total=total, message="upscaling")

        writer = _VideoWriter(video_tmp, fps)
        done = 0
        n_chunks = 0
        w_lr = h_lr = w_hd = h_hd = None
        t_sr = t_recon = 0.0
        try:
            for chunk in stream_gops(input_path, max_frames=max_frames):
                n_chunks += 1
                if w_lr is None:
                    h_lr, w_lr = chunk[0][1].shape[:2]
                    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE

                # 1) per-frame SR cache for this chunk (anchor / fallback / baseline source)
                ts = time.perf_counter()
                perframe_cache = derisk.build_perframe_cache(chunk, w_hd, h_hd, cfg["sr_mode"])
                # 2) region-aware gate (quality only): temporally-stable motion gate + the
                #    per-frame COMPACT source for the OUTPUT-only blend (this chunk's frames).
                region_gate = (derisk._build_region_gate(chunk, w_hd, h_hd, SCALE)
                               if cfg["region_aware"] else None)
                t_sr += time.perf_counter() - ts

                # 3) reconstruct this chunk (I/P backbone + B leaves). anchor_set=set()
                #    => I-frames-only backbone (the real-footage path). The first backbone
                #    frame of the chunk is force-anchored (no in-chunk predecessor).
                tr = time.perf_counter()
                _, R = derisk.reconstruct(
                    chunk, None, SCALE, True, cfg["occ"], perframe_cache, set(),
                    backend=cfg["backend"], collect_metrics=False, download_output=True,
                    region_gate=region_gate,
                )
                t_recon += time.perf_counter() - tr

                # 4) grain (global frame index seed => temporally independent across chunks)
                #    + ENCODE incrementally, then free the chunk's HD frames.
                for i in range(len(chunk)):
                    recon = R[i]["recon"]
                    if cfg["grain"] != "off":
                        recon = _grain.apply_grain(recon, done, cfg["grain"])
                    writer.write(recon)
                    done += 1
                    elapsed = time.perf_counter() - t0
                    rate = elapsed / done
                    eta = rate * (total - done) if total and done <= total else None
                    _set_progress(done=done, elapsed_s=round(elapsed, 1),
                                  eta_s=(round(eta, 1) if eta is not None else None),
                                  ms_per_frame=round(rate * 1000.0, 1))

                del perframe_cache, region_gate, R, chunk
                _free_gpu()                         # release MPS cache between chunks (see _free_gpu)
        finally:
            writer.close()

        if done == 0:
            raise RuntimeError(f"no frames decoded from {input_path}")

        # 5) mux audio into the final container (copy AAC / transcode / none).
        _set_progress(state="muxing", message="muxing audio")
        audio_note = _mux_av(video_tmp, input_path, out_path, done, fps)

        total_s = time.perf_counter() - t0
        LAST_STATS.clear()
        LAST_STATS.update({
            "mode": mode, "label": cfg["label"], "input": input_path, "out_path": out_path,
            "n_frames": done, "n_chunks": n_chunks, "soft_cap": SOFT_CAP_FRAMES,
            "fps": float(fps), "scale": SCALE,
            "src_resolution": f"{w_lr}x{h_lr}", "out_resolution": f"{w_hd}x{h_hd}",
            "audio": audio_note,
            "t_sr_s": round(t_sr, 2), "t_recon_s": round(t_recon, 2),
            "t_total_s": round(total_s, 2),
            "ms_per_frame": round(total_s * 1000.0 / max(1, done), 1),
        })
        _set_progress(state="done", done=done, elapsed_s=round(total_s, 1),
                      eta_s=0.0, ms_per_frame=round(total_s * 1000.0 / max(1, done), 1),
                      message="done")
        return out_path
    except Exception as e:
        _set_progress(state="error", message=f"{type(e).__name__}: {e}")
        raise
    finally:
        if os.path.exists(video_tmp):
            try:
                os.remove(video_tmp)
            except OSError:
                pass
        end_job()


# --------------------------------------------------------------------------- #
# Verification helper: re-decode the produced mp4 -> confirm valid H.264, frame
# count, HD resolution, AND an audio stream whose duration ~= the video duration.
# --------------------------------------------------------------------------- #
def _verify_mp4(path, expect_n, expect_res):
    cont = av.open(path)
    try:
        vs = cont.streams.video[0]
        codec = vs.codec_context.name
        w, h = vs.codec_context.width, vs.codec_context.height
        v_dur = float(vs.duration * vs.time_base) if vs.duration else None
        has_audio = len(cont.streams.audio) > 0
        a_codec = a_dur = None
        if has_audio:
            a = cont.streams.audio[0]
            a_codec = a.codec_context.name
            a_dur = float(a.duration * a.time_base) if a.duration else None
        n = sum(1 for _ in cont.decode(vs))
    finally:
        cont.close()
    sync_ok = (v_dur is not None and a_dur is not None and abs(v_dur - a_dur) <= 0.5)
    ok = (codec == "h264" and n == expect_n and f"{w}x{h}" == expect_res
          and has_audio and sync_ok)
    return ok, {"codec": codec, "frames": n, "resolution": f"{w}x{h}",
                "video_dur_s": round(v_dur, 3) if v_dur else None,
                "audio_codec": a_codec,
                "audio_dur_s": round(a_dur, 3) if a_dur else None,
                "sync_ok": sync_ok}


# --------------------------------------------------------------------------- #
# CLI: `python3 server/pipeline_api.py instant|quality [--input F] [--max-frames N]`
# Foreground verification helper. With --mem it samples RSS to prove bounded memory.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=list(ALL_MODES))
    ap.add_argument("--input", default=os.path.join(TESTDATA_DIR, "short.mp4"))
    ap.add_argument("--max-frames", type=int, default=None,
                    help="test cap; default None = whole clip")
    ap.add_argument("--mem", action="store_true", help="sample RSS to prove bounded memory")
    a = ap.parse_args()

    peak = {"rss": 0}
    samples = []
    stop = threading.Event()

    def _sampler():
        try:
            import psutil
            proc = psutil.Process()
        except Exception:
            return
        while not stop.is_set():
            rss = proc.memory_info().rss
            peak["rss"] = max(peak["rss"], rss)
            p = get_progress()
            samples.append((p.get("done", 0), rss))
            time.sleep(0.25)

    sampler = None
    if a.mem:
        sampler = threading.Thread(target=_sampler, daemon=True)
        sampler.start()

    print(f"[pipeline] mode={a.mode}  input={a.input}  max_frames={a.max_frames}")
    out = process_clip(a.input, a.mode, max_frames=a.max_frames)
    stop.set()
    if sampler:
        sampler.join(timeout=1)

    s = LAST_STATS
    print(f"[pipeline] wrote {out}")
    print(f"[pipeline] stats: {s}")
    ok, info = _verify_mp4(out, s["n_frames"], s["out_resolution"])
    print(f"[pipeline] re-decode check: ok={ok}  {info}")
    if a.mem and peak["rss"]:
        print(f"[pipeline] peak RSS: {peak['rss']/1e6:.0f} MB over {len(samples)} samples "
              f"(chunks={s['n_chunks']}, cap={s['soft_cap']})")
        # a few (frames_done, rss_MB) samples to show it stays flat across chunks
        step = max(1, len(samples) // 8)
        thin = [(d, round(r / 1e6)) for d, r in samples[::step]]
        print(f"[pipeline] (frames_done, RSS_MB) samples: {thin}")
    sys.exit(0 if ok else 2)
