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
from dataclasses import dataclass, asdict

import numpy as np
import av
import cv2

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
import background_plate as _bp    # noqa: E402  (AUTO mode: static-vs-moving camera verdict)

# Instant-mode speedup (Levers 1-4). These live in server/ and import the prototype READ-ONLY.
import anchor_sr                 # noqa: E402  (Lever 1: anchor-only SR cache + adaptive safeguard)
import fast_grain                # noqa: E402  (GPU/MPS film grain)
import pipe_encode               # noqa: E402  (Lever 2: threaded encode + chunk prefetch overlap)
import interp_pass               # noqa: E402  (R7-E1: "smooth 2x" MV-reuse frame interpolation, default OFF)
try:
    import gpu_ops as _gpu_ops   # noqa: E402  (Lever 4: GPU-resident recon -> single host download)
except Exception:
    _gpu_ops = None

# Adaptive-safeguard threshold for the instant path: a non-anchor frame whose occlusion-fallback
# fraction exceeds this gets a real SR call (bicubic upscale handles the rest). At the 720p instant
# tier bicubic fallback is visually fine (it's the fast/lower-quality tier), so this is set HIGH
# (0.50) -> the safeguard fires only on catastrophic >50%-fallback frames (rare), keeping SR pure
# anchor-only (~8 ms/frame) for ~10x / 24 fps. Lower it (e.g. 0.08) to trade speed for crisper
# occlusion regions on high-motion content; verified clean on talking-head at 0.50.
INSTANT_FALLBACK_THRESH = 0.50

# E2 (research round R1) -- MOTION-KEYED fallback threshold (default OFF). On HIGH-motion content
# the 0.50 safeguard leaves ~24% of P-frame pixels on bicubic occlusion fallback; experiment E2
# found that escalating only the high-motion frames (mean LR-MV magnitude > INSTANT_MOTION_GATE) to
# a lower threshold (INSTANT_FALLBACK_THRESH_HI) halves the window-A bicubic weak spot (7.71%->3.65%)
# at ZERO cost on talking-head (self-gating: it has no frame above 20% fallback). The honest tradeoff
# (E2 report): on high motion this RAISES tOF (bicubic fallback is temporally smooth; fresh-SR
# fallback shimmers) -- so it is OFF by default (the steady-but-soft baseline is the better default
# for the fast tier) and exposed for sharpness-priority content. Instant-only; OFF -> byte-identical.
INSTANT_MOTION_KEYED_FALLBACK = False
INSTANT_FALLBACK_THRESH_HI = 0.20      # threshold used on high-motion frames when the flag is ON
INSTANT_MOTION_GATE = 1.0              # mean LR-MV magnitude (px/frame) above which a frame is "high motion"

# R4-E3 -- FALLBACK-SATURATION CAP (default ON, intra-gated). Fixes the low-light/noise real-time
# cliff: noisy content makes the encoder intra-code ~99% of blocks -> ~95-99% occlusion-fallback on
# EVERY frame -> all frames blow the 0.50 safeguard -> a full per-frame compact-SR runs for ZERO
# propagation benefit (nothing to warp from) -> ms/frame 31->121 (3.9x). The cap declines SR
# escalation on a frame that has HIGH fallback AND is ~all-intra (intra-fraction > the gate = the
# encoder gave up = no MVs to propagate) -> bicubic floor -> real-time held. The INTRA gate (not the
# old crude motion gate) is the principled "nothing to propagate" signal: it fires ONLY on genuine
# noise, NOT on a title-card reveal/dissolve (which intra-codes only a LOCAL region) -- so it is safe
# DEFAULT ON with no clean-content regression (verified byte-identical on short.mp4 + clean; real-time
# held on c3 noise). Set CAP=1.0 to disable.
INSTANT_FALLBACK_SATURATION_CAP = 0.70   # fallback fraction above which SR escalation is declined (1.0 = OFF)
INSTANT_SAT_CAP_INTRA_GATE = 0.80        # intra-coded fraction above which a high-fallback frame = NOISE
INSTANT_SAT_CAP_MOTION_GATE = 8.0        # (legacy; kept for back-compat, no longer used by the cap)

# Lever 3 (tile-SR safeguard): SR only the bounding box of a high-fallback frame's occlusion
# region instead of the full 2560x1280. DISABLED by default after measurement: on this real
# footage the occlusion fallback is spatially SCATTERED (camera + complex motion), so a single
# bounding box covers ~97% of the frame and even a 32x16 grid only reaches ~46% coverage -- which
# would need hundreds of tiny SR forward passes whose launch overhead erases the area saving.
# The premise (a compact moving-edge occlusion) doesn't match; kept available for content where
# it does (set True). See _quality_instant.py / the grid-coverage probe.
INSTANT_TILE_SR = False
# Lever 4a: keep the per-frame SR/bicubic fallback cache GPU-RESIDENT. Non-anchor frames are
# bicubic-upscaled on-device (F.interpolate) straight from the small LR upload instead of a CPU
# cv2.resize + a full 9.8 MB HD host->device upload per frame -> kills reconstruct's upload_perframe.
INSTANT_GPU_CACHE = True

# R3-E3 -- HF-only temporal-EMA soft-occlusion (default OFF). Replaces the hard patch_high_fallback
# SR-patch with a feathered, temporally-smoothed HF injection that escapes the high-motion
# tOF<->fallback% frontier (eff-bic 7.70->6.35% at tOF +2.0% vs the hard switch's +20%; verified
# experiments/r3_e3_softocc_wire). OFF -> byte-identical to today. ON runs ~1 compact-SR call per
# non-anchor frame (a quality knob, NOT real-time) -> bound with SOFTOCC_MOTION_GATE (None = every
# frame = full escape; 1.0 = fewer SR calls, shallower escape). Instant-only.
INSTANT_SOFTOCC = False
SOFTOCC_GAIN = 0.6
SOFTOCC_BETA = 0.85
SOFTOCC_FEATHER = 31
SOFTOCC_MOTION_GATE = None

# R7-E1 -- "SMOOTH 2x" MV-reuse frame interpolation (default OFF). When ON, the instant fast path emits
# a motion-compensated MIDPOINT before each real recon frame -- it warps BOTH neighbours by the codec
# 'past' MV field we ALREADY extract (build_lr_flow) + an intra-hole-only blend (interp_pass) -- which
# DOUBLES the output frame rate (out_fps = fps*2). It is OUTPUT-ONLY: it only READS R[t]['recon'] AFTER
# reconstruct() returns and emits an EXTRA frame between neighbours; the midpoint is NEVER stored back
# into R[] so it can never become a propagation reference (GOTCHA #16). Two NON-optional ship-blockers
# baked into interp_pass and verified in experiments/r4_e2_interp_wire: (1) intra-hole routing ONLY (the
# full Ruder/reactive mask over-flags large motion -> re-ghosts); (2) scene-cut guard -- a connecting
# field whose intra-hole fraction exceeds INTERP_CUT_THRESH (scene cut / chaotic intro) -> FRAME-DUP, not
# a ghosting blend. Quality reproduces R3-E1 exactly (+3.6..+8.9 dB PSNR over dup/linear-blend). Instant-
# ONLY; OFF -> n_emit==done, out_fps==fps, real-frame grain seed unchanged -> BYTE-IDENTICAL to today.
# NOTE: NOT free real-time -- ~halves instant throughput (2 warps + blend per inserted frame).
INSTANT_INTERP_2X = False
INTERP_CUT_THRESH = 0.5                 # intra-hole fraction above which the midpoint DUPLICATES (cut guard)
_MID_GRAIN_SEED_BASE = 1 << 20          # midpoint grain seeds live ABOVE real-frame seeds (no collision)

SAMPLE_MP4 = os.path.join(_REPO, "sample.mp4")
OUTPUTS_DIR = os.path.join(_HERE, "outputs")
UPLOADS_DIR = os.path.join(_HERE, "uploads")
TESTDATA_DIR = os.path.join(_HERE, "testdata")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Every real SR net in sr.py is an x4 model -> the pipeline always runs at scale 4
# (matches the README: "--sr realesrgan ... it is an x4 net => use --scale 4").
SCALE = 4

# Instant is the FAST / lower-quality tier -> it renders at HALF scale (x2 = ~720p, 1280x640
# from 640x320) instead of full QHD x4. ~4x fewer output pixels means recon/grain/encode all
# drop ~4x, taking instant to ~real-time (the recon warp on a 3.3 MP QHD frame was the floor).
# SR still runs the x4 net and downscales (`upscale_to`) -> a sharp 720p, better than a native 2x.
# Quality + Layered stay at full QHD (SCALE). The whole instant path is scale-parameterized
# (anchor_sr derives `scale = w_hd // w_lr`; reconstruct takes `scale`), so this is just eff_scale.
INSTANT_SCALE = 2

# Max frames per processing chunk. A new chunk always starts at an I-frame; if a GOP is
# longer than this, we also cut at the next P-frame (a forced fresh anchor) so a single
# chunk -- and thus peak HD memory -- stays bounded. 48 == the validated prototype window.
SOFT_CAP_FRAMES = 48

# V1 (improvement loop): the layered background plate is a STATIC image, so it gets ONE fixed
# grain (this seed) baked in -> filmic texture WITHOUT per-frame flicker. Per-frame grain is then
# applied only to the moving foreground (gated by alpha), so the layered mode's ~167x steadier
# background is actually visible instead of being masked by full-frame per-frame grain.
_PLATE_GRAIN_SEED = 12345

# Mode -> exactly the flag combination the handoff recommends for each regime.
MODE_CONFIG = {
    "instant": dict(sr_mode="realesrgan",          # compact realesr-general-x4v3 (~0.13 s/frame SR)
                    backend="torch",               # MPS fast path (recon stays GPU-resident)
                    occ="reactive",                # Lever 1: drop the fwd-bwd softmax splat. Instant
                                                   # is the FAST tier; Step-7 established reactive ==
                                                   # full-mask quality on low-motion talking-head and
                                                   # loses only slightly on high-motion. The splat was
                                                   # 12 ms of recon (fired 63/82 mask calls on real
                                                   # footage); reactive cuts recon ~58->47 ms AND
                                                   # shrinks hole_frac so fewer safeguard SR upgrades.
                    region_aware=False,
                    grain="med",
                    label="Instant (720p, compact anchor-only SR, ~real-time ~24 fps)"),
    "quality": dict(sr_mode="realesrgan-x4plus",   # heavy RRDBNet x4plus (~2.2 s/frame, +61% sharper)
                    backend="torch",               # region-aware integration is tested on torch
                    occ="adaptive",
                    region_aware=True,             # OUTPUT-only motion-gated heavy/compact blend
                    grain="med",
                    fp16=True,                     # E4: fp16 x4plus anchor (~1.24x faster, PSNR 72-76
                                                   # dB vs fp32 = visually identical). GPU-guarded in
                                                   # sr.load_model (no-op/fp32 on CPU).
                    grain_motion=True,             # E3 V2: freeze grain on static (region-gated) pixels,
                                                   # fresh per-frame grain only on motion -> removes the
                                                   # ~100% grain-induced static-region flicker.
                    label="Quality (x4plus anchor, region-aware blend, grain)"),
}

# The LAYERED mode is a SEPARATE third quality path (SR the static background plate ONCE per
# scene, manage only the moving foreground per frame). It is intentionally NOT a MODE_CONFIG
# entry so the instant/quality streaming body stays byte-for-byte untouched -- process_clip
# branches to _run_layered() for it. ALL_MODES gates the API validation + the single-job lock.
LAYERED_LABEL = "Quality — Layered (talking-head, static camera)"
# "auto" is a meta-mode: process_clip resolves it (via the cheap recommend_mode probe) to one of the
# real modes BEFORE rendering, so it is valid at the API but never reaches the render path itself.
ALL_MODES = tuple(MODE_CONFIG) + ("layered", "auto")


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
# layered is CONTENT-DEPENDENT: a static-camera scene composites cheaply (~470 ms/frame) but a
# MOVING scene falls back to the region-aware quality path (~2900 ms/frame). R5-E1 measured ~1382
# ms/frame on mixed content (4/7 moving scenes) -> the old 470 estimate under-promised ~3x, so the
# UI estimate uses this conservative mixed value (an all-static clip finishes sooner; a moving clip
# is better routed to instant/quality by the R4-E4 auto-mode).
MODE_MS_PER_FRAME = {"instant": 130.0, "quality": 2900.0, "layered": 1400.0}


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
# HARDWARE ENCODE (Lever 3): on Apple silicon, h264_videotoolbox is a dedicated media-engine
# H.264 encoder -- it offloads the encode off the CPU entirely (~5 ms/frame vs ~44 ms for
# libx264 software). We prefer it and FALL BACK to libx264 if it is unavailable or fails to
# open. VideoToolbox has no CRF; quality is set via a generous target bitrate (a function of
# pixel count) tuned to be visually lossless vs the libx264 crf18 baseline (verified by PSNR in
# bench_instant.py). faststart + the audio mux are unchanged (they happen later, in _mux_av).
_HW_CODEC = "h264_videotoolbox"
# bits-per-pixel-per-second target for the HW encoder, tuned in bench_instant.py so the decoded
# VideoToolbox output matches libx264 crf18 (the BEFORE baseline) for this grained HD content:
# 0.70 bpp -> ~37.7 dB decode PSNR (== x264's 37.97) at ~14 MB/48f (== x264's size), and encode
# stays ~10 ms/frame (HW encode time is ~flat in bitrate), so quality parity costs no speed.
_HW_BPP = 0.70


def _hw_encode_available():
    try:
        av.codec.Codec(_HW_CODEC, "w")
        return True
    except Exception:
        return False


class _VideoWriter:
    """Incremental HD encoder kept open across chunks. Prefers the VideoToolbox HW encoder
    (Lever 3) and transparently falls back to libx264 software. `codec=None` => auto (HW if
    available); pass 'libx264' to force the software baseline (used by the BEFORE benchmark)."""

    def __init__(self, path, fps, codec=None):
        self.cont = av.open(path, "w")
        self.fps = fps
        self.st = None
        if codec is None:
            codec = _HW_CODEC if _hw_encode_available() else "libx264"
        self.codec = codec
        self.encoder = None        # the codec actually used (set on first frame; may fall back)

    def _ensure(self, w_hd, h_hd):
        if self.st is not None:
            return self.st
        want = self.codec
        for cand in ([want] if want == "libx264" else [want, "libx264"]):
            try:
                st = self.cont.add_stream(cand, rate=self.fps)
                st.width, st.height, st.pix_fmt = w_hd, h_hd, "yuv420p"
                if cand == _HW_CODEC:
                    # VideoToolbox: target-bitrate quality (no CRF). realtime=1 lets the media
                    # engine run unthrottled; allow_sw=1 keeps it working if the HW path is busy.
                    st.bit_rate = int(w_hd * h_hd * float(self.fps) * _HW_BPP)
                    st.options = {"realtime": "1", "allow_sw": "1"}
                else:
                    st.options = {"crf": "18"}
                self.st = st
                self.encoder = cand
                return self.st
            except Exception:
                if cand == "libx264":
                    raise
                continue
        raise RuntimeError("no usable H.264 encoder")

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
    perframe_cache = derisk.build_perframe_cache(
        sub, w_hd, h_hd, cfg["sr_mode"], half=cfg.get("fp16", False))   # E4 fp16 anchor
    region_gate = derisk._build_region_gate(sub, w_hd, h_hd, SCALE)
    grain_static_hd = (cv2.resize(region_gate["a_lr"], (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
                       if cfg.get("grain_motion") else None)            # E3 V2 motion gate
    _, R = derisk.reconstruct(
        sub, None, SCALE, True, cfg["occ"], perframe_cache, set(),
        backend=cfg["backend"], collect_metrics=False, download_output=True,
        region_gate=region_gate,
    )
    for i in range(len(sub)):
        recon = R[i]["recon"]
        if cfg["grain"] != "off":
            if grain_static_hd is not None:
                recon = _grain.apply_grain_motion(recon, done, grain_static_hd, cfg["grain"])
            else:
                recon = _grain.apply_grain(recon, done, cfg["grain"])
        writer.write(recon)
        done += 1
        _emit_frame_progress(done, total, t0, t_passB)
    del perframe_cache, region_gate, R
    return done


# Process-wide matte model cache. The AUTO probe (recommend_mode) and the LAYERED render share ONE
# loaded RVM/seg model (the weights are identical and stateless across calls -- the recurrent state
# lives in the per-call `rec` list, never on the model), so an auto->layered job loads the matte
# ONCE, not twice. A load failure (offline / missing weights) is REMEMBERED, not retried, and never
# silently swallowed: the probe surfaces it as matte_unavailable; the layered render re-raises.
_MATTE_MODEL = {"model": None, "tried": False, "err": None}


def _get_matte_model():
    """Return the shared, lazily-loaded matte model (None if it could not be loaded)."""
    if _MATTE_MODEL["tried"]:
        return _MATTE_MODEL["model"]
    _MATTE_MODEL["tried"] = True
    try:
        _MATTE_MODEL["model"] = _layered.load_matting_model()
    except Exception as e:                       # offline / no weights -> remember, surface, never crash here
        _MATTE_MODEL["err"] = repr(e)
        _MATTE_MODEL["model"] = None
    return _MATTE_MODEL["model"]


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
        model = _get_matte_model()              # shared with the AUTO probe (loaded once)
        if model is None:                       # layered REQUIRES a matte -> surface, never swallow
            raise RuntimeError(f"layered mode needs the matte model but it failed to load: "
                               f"{_MATTE_MODEL['err']}")
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
        writer = _VideoWriter(video_tmp, fps, codec="libx264")   # layered: keep software encoder
        done = 0
        n_chunks = 0
        w_lr = h_lr = w_hd = h_hd = None
        ratio = None
        cur_plate_sid, cur_plate = None, None
        plate_base = None                 # R4-E1: per-scene plate-validity bg-PSNR baseline (EMA)
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
                        # V1: bake a FIXED grain into the plate ONCE per scene -> stable filmic bg.
                        cur_plate = _grain.apply_grain(np.load(info["plate_path"]),
                                                       _PLATE_GRAIN_SEED, "med")
                        cur_plate_sid = sid
                        plate_base = None         # R4-E1: reset the per-scene plate-validity baseline
                    if rvm_sid != sid:            # thread RVM recurrent state per scene
                        rec, rvm_sid = [None] * 4, sid
                    for (_ptype, img, _mvs) in sub:
                        pha, rec = _layered.matte_frame_np(model, img, rec, ratio, device)
                        # R4-E1: plate-validity guard. If the plate does NOT match this frame's
                        # background (a MISSED scene cut -> wrong fixed plate -> silent corruption),
                        # composite_frame_guarded returns the faithful full-frame compact SR instead.
                        comp, _bgp, _bad = _layered.composite_frame_guarded(
                            img, pha, cur_plate, w_hd, h_hd, plate_base)
                        if not _bad and _bgp == _bgp:   # finite bg-PSNR -> update per-scene baseline
                            plate_base = _bgp if plate_base is None else (
                                _layered.PLATE_GUARD_EMA * _bgp
                                + (1.0 - _layered.PLATE_GUARD_EMA) * plate_base)
                        # V1: per-frame grain ONLY on the moving foreground (gate by alpha). The
                        # background keeps the plate's fixed grain (stable); the subject gets fresh
                        # per-frame grain -> the layered mode's steady background is now visible.
                        comp_g = _grain.apply_grain(comp, done, "med")
                        a = _layered.lp.alpha_to_hd(pha, (h_hd, w_hd))   # (H,W,1) soft alpha
                        if a.ndim == 2:
                            a = a[..., None]
                        out = (a * comp_g.astype(np.float32)
                               + (1.0 - a) * comp.astype(np.float32)).clip(0, 255).astype(np.uint8)
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
# AUTO mode -- a CHEAP probe (R4-E4) that recommends instant | quality | layered from signals
# computable in ONE light decode pass (codec MVs + a few decoded frames), WITHOUT a per-mode render.
# It reuses the SAME shared code the product renders with (stream_gops chunking, anchor_sr occlusion
# fallback, derisk LR flow, scene_detect cuts, background_plate static-camera verdict, the shared
# matte model) so the cheap decision sees exactly what the render will. See experiments/r4_e4_automode.
# --------------------------------------------------------------------------- #
# Thresholds re-measured on R3-E2's authored clips (see r4_e4_automode/REPORT validation table).
AUTO_FB_NOISE = 50.0       # % reactive fallback mean -> low-light/noise -> quality        (c3=94.9)
AUTO_FB_HI = 15.0          # % reactive fallback mean -> high local occlusion -> quality    (c1,c6)
AUTO_MOTION_HI = 10.0      # MEDIAN per-frame LR-MV magnitude (px) -> above = high motion. MEDIAN (not
                           #   mean) so a single cut-frame MV spike (codec predicts huge MVs across a
                           #   cut, e.g. c5b max 207) does NOT inflate the signal.
AUTO_SMEAR_MOTION = 3.0    # a MISSED cut only smears instant when there is motion to warp across it.
AUTO_STATIC_THRESH_PX = 0.6   # |median camera MV| below this = static (matches layered STATIC_THRESH_PX)
AUTO_HUMAN_LO = 0.03       # matte-coverage band for a plausible talking-head (3%..85%)
AUTO_HUMAN_HI = 0.85
AUTO_HIDDEN_CUT_CHROMA = 18.0  # mean |dRGB| between consecutive sampled frames the luma-only cut
                               #   detector can MISS (similar-luma + cool-tint splice) -> reject layered.
AUTO_PLATE_RESID_MAX = 12.0    # mean |frame - median-plate| over BACKGROUND px (luma levels): small on a
                               #   truly static bg, large on a moving/parallax/wrong (hidden-cut) plate.
AUTO_MATTE_K = 3           # frames to matte for the human-coverage check (kept tiny -> cheap)
# Default frames the AUTO probe scans to decide (one soft-cap GOP, ~2 s @24fps; covers the authored
# clips' cut@28). The render itself still processes the WHOLE clip; only the DECISION is windowed.
AUTO_PROBE_FRAMES = 48


@dataclass
class _AutoSignals:
    n_frames: int = 0
    n_chunks: int = 0
    n_scenes: int = 1
    mv_mag_mean: float = 0.0
    mv_mag_median: float = 0.0      # robust motion signal (cut-spike-immune); drives the routing
    mv_mag_max: float = 0.0
    edge_density_mean: float = 0.0
    fb_react_mean: float = 0.0
    fb_react_max: float = 0.0
    camera_verdict: str = "UNKNOWN"          # STATIC | MOVING | UNKNOWN
    global_vec_mag_px: float = float("nan")
    hidden_cut_suspected: bool = False
    chroma_diff_max: float = 0.0
    human_coverage: object = None    # None = not probed / matte unavailable
    plate_resid: object = None       # mean bg residual to the median plate (luma levels)
    probe_s: float = 0.0


@dataclass
class AutoRecommendation:
    mode: str
    reason: str
    signals: dict
    matte_unavailable: bool = False


def _auto_scan(clip_path, n, stride):
    """ONE light decode pass over the product's chunking: motion + edges + exact occlusion-fallback."""
    mags, edges, fbs = [], [], []
    decoded = []                  # flat (ptype,lr,mvs) for estimate_global_motion (reuses these MVs)
    samples = []                  # a few RGB frames (chroma-cut check + matte)
    n_chunks = 0
    n_frames = 0
    for chunk in stream_gops(clip_path, max_frames=n):
        n_chunks += 1
        anchors, backbone = anchor_sr.anchor_indices(chunk)
        h_lr, w_lr = chunk[0][1].shape[:2]
        for i, (pt, lr, mvs) in enumerate(chunk):
            decoded.append((pt, lr, mvs))
            if n_frames % stride == 0:                  # sample subset for the cheap signals
                g = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY)
                edges.append(float((cv2.Canny(g, 80, 160) > 0).mean()))
                if mvs is not None and len(mvs):
                    fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want="all")
                    m = np.sqrt(fx * fx + fy * fy)
                    mags.append(float(np.nanmean(m)) if np.isfinite(m).any() else 0.0)
                else:
                    mags.append(0.0)
                if i in anchors:
                    fbs.append(0.0)
                else:
                    fbs.append(anchor_sr._lr_fallback_fraction(chunk, i, backbone, "reactive"))
                samples.append(lr)          # stride-spaced across the WHOLE window so the chroma
                                            # hidden-cut check + matte span any mid-window cut
            n_frames += 1
    return mags, edges, fbs, decoded, samples, n_chunks, n_frames


def _auto_chroma_hidden_cut(samples):
    """Max mean |dRGB| between consecutive SAMPLED frames -- chroma-sensitive, so a similar-LUMA cut
    (the luma-only detector can miss) still shows up. Returns (max_chroma_diff, suspected)."""
    if len(samples) < 2:
        return 0.0, False
    diffs = []
    for a, b in zip(samples[:-1], samples[1:]):
        if a.shape != b.shape:
            diffs.append(255.0)
            continue
        diffs.append(float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32)))))
    mx = max(diffs) if diffs else 0.0
    return mx, (mx > AUTO_HIDDEN_CUT_CHROMA)


def _auto_human_coverage(samples):
    """Matte K sampled frames -> (coverage_fraction, plate_resid). coverage None if the matte model
    is unavailable. plate_resid = mean |frame - temporal-median-plate| over BACKGROUND px (luma
    levels): small on a truly static bg, large on a moving/parallax/wrong-plate bg."""
    model = _get_matte_model()
    if model is None:
        return None, None
    dev = _layered._device()
    pick = samples[:: max(1, len(samples) // AUTO_MATTE_K)][:AUTO_MATTE_K]
    covs, resid = [], []
    luma = [0.299 * s[..., 0] + 0.587 * s[..., 1] + 0.114 * s[..., 2] for s in pick]
    plate_ref = np.median(np.stack([l.astype(np.float32) for l in luma], 0), axis=0)
    rec = [None] * 4
    for s, l in zip(pick, luma):
        ratio = _layered.downsample_ratio(*s.shape[:2])
        pha, rec = _layered.matte_frame_np(model, s, rec, ratio, dev)
        covs.append(float((pha > 0.5).mean()))
        bg = (pha <= 0.5)                                   # background pixels only
        if bg.any():
            resid.append(float(np.mean(np.abs(l.astype(np.float32) - plate_ref)[bg])))
    _free_gpu()
    cov = float(np.median(covs)) if covs else 0.0
    pr = float(np.mean(resid)) if resid else None
    return cov, pr


def recommend_mode(input_path, max_frames: int = AUTO_PROBE_FRAMES, stride: int = 1,
                   verbose: bool = False) -> AutoRecommendation:
    """CHEAP probe -> AutoRecommendation(mode in {instant,quality,layered}, reason, signals dict).
    Picks per content WITHOUT a per-mode render. The matte (the only neural step) runs ONLY on a
    static, single-scene, low-motion candidate -> it never fires on high-motion/noisy/multi-cut
    clips, keeping the probe cheap. If the matte is unavailable (offline), layered is conservatively
    DROPPED and we fall through to instant (real-time, corruption-free) -- never crash."""
    t0 = time.perf_counter()
    s = _AutoSignals()

    mags, edges, fbs, decoded, samples, n_chunks, n_frames = _auto_scan(input_path, max_frames, stride)
    s.n_frames, s.n_chunks = n_frames, n_chunks
    s.mv_mag_mean = round(float(np.mean(mags)), 3) if mags else 0.0
    s.mv_mag_median = round(float(np.median(mags)), 3) if mags else 0.0
    s.mv_mag_max = round(float(np.max(mags)), 3) if mags else 0.0
    s.edge_density_mean = round(float(np.mean(edges)), 4) if edges else 0.0
    s.fb_react_mean = round(float(np.mean(fbs)) * 100, 2) if fbs else 0.0
    s.fb_react_max = round(float(np.max(fbs)) * 100, 2) if fbs else 0.0

    # scene-cut count (cheap luma pass) -> plate safety
    try:
        cuts, _total = scene_detect.find_cuts(input_path, max_frames=max_frames)
        s.n_scenes = len(cuts) + 1
    except Exception as e:
        s.n_scenes = 1
        if verbose:
            print("  [warn] find_cuts:", repr(e))

    # static-camera verdict (reuses the MVs already decoded) -> plate safety
    try:
        gm = _bp.estimate_global_motion(decoded, static_thresh=AUTO_STATIC_THRESH_PX)
        s.camera_verdict = gm["verdict"]
        s.global_vec_mag_px = round(float(gm["global_vec_mag_px"]), 3)
    except Exception as e:
        s.camera_verdict = "UNKNOWN"
        if verbose:
            print("  [warn] estimate_global_motion:", repr(e))

    s.chroma_diff_max, s.hidden_cut_suspected = _auto_chroma_hidden_cut(samples)
    s.chroma_diff_max = round(s.chroma_diff_max, 2)

    matte_unavailable = False
    # a MISSED cut = a strong chroma discontinuity the luma cut-detector did NOT split on (n_scenes
    # stays 1). stream_gops splits at DETECTED cuts (instant re-anchors cleanly), so only a MISSED
    # cut leaves a chunk spanning it -> instant warps the pre-cut anchor across = smear.
    missed_cut = s.hidden_cut_suspected and s.n_scenes == 1

    # ---------------- decision rule (R4-E4) ----------------
    if s.fb_react_mean > AUTO_FB_NOISE:
        mode = "quality"
        reason = (f"low-light/noisy: reactive fallback {s.fb_react_mean:.0f}% > {AUTO_FB_NOISE:.0f}% "
                  f"-> unreliable MVs collapse instant to per-frame SR (real-time breaks); "
                  f"plate would denoise-corrupt. Safe default = quality.")
    elif s.mv_mag_median > AUTO_MOTION_HI or s.fb_react_mean > AUTO_FB_HI:
        mode = "quality"
        reason = (f"high motion / occlusion: median mvMag {s.mv_mag_median:.1f} (>{AUTO_MOTION_HI:.0f}) "
                  f"or fb {s.fb_react_mean:.0f}% (>{AUTO_FB_HI:.0f}%) -> instant softens/flickers "
                  f"(tOF 3-8x) & breaks real-time. Escalate to quality.")
    elif missed_cut and s.mv_mag_median > AUTO_SMEAR_MOTION:
        mode = "quality"
        reason = (f"MISSED cut: chroma discontinuity {s.chroma_diff_max:.0f} (>{AUTO_HIDDEN_CUT_CHROMA:.0f}) "
                  f"not split by the luma detector (n_scenes=1) + motion (median mvMag "
                  f"{s.mv_mag_median:.1f} > {AUTO_SMEAR_MOTION:.0f}) -> instant warps across the un-split "
                  f"cut = smear. Escalate to quality.")
    else:
        # ---- layered candidate: static, single scene, low motion/fb. Confirm human + plate. ----
        static_ok = s.camera_verdict in ("STATIC", "UNKNOWN")
        if static_ok and not s.hidden_cut_suspected:
            cov, pr = _auto_human_coverage(samples)
            s.human_coverage = None if cov is None else round(cov, 3)
            s.plate_resid = None if pr is None else round(pr, 2)
            human = (cov is not None and AUTO_HUMAN_LO <= cov <= AUTO_HUMAN_HI)
            if cov is None:
                matte_unavailable = True
                mode = "instant"
                reason = (f"static single non-multi-cut scene but matte unavailable "
                          f"({_MATTE_MODEL['err']}) -> can't confirm human; instant is real-time & "
                          f"corruption-free (layered dropped for safety).")
            elif not human:                                  # non-human (incl. coverage 0)
                mode = "instant"
                reason = (f"static single scene, low motion (mvMag {s.mv_mag_mean:.1f}) & fb "
                          f"{s.fb_react_mean:.0f}%, but human coverage {cov:.2f} outside "
                          f"[{AUTO_HUMAN_LO},{AUTO_HUMAN_HI}] (non-human) -> instant: real-time & acceptable.")
            elif pr is not None and pr > AUTO_PLATE_RESID_MAX:   # human but bg not static enough
                mode = "quality"
                reason = (f"static+human (coverage {cov:.2f}) but plate residual {pr:.1f} > "
                          f"{AUTO_PLATE_RESID_MAX} levels -> bg not truly static / hidden cut; plate "
                          f"unsafe -> quality.")
            else:                                            # human + safe static plate
                mode = "layered"
                reason = (f"static camera ({s.camera_verdict}), single scene, human coverage "
                          f"{cov:.2f} in [{AUTO_HUMAN_LO},{AUTO_HUMAN_HI}], plate residual {pr:.1f} <= "
                          f"{AUTO_PLATE_RESID_MAX} levels -> safe static-bg plate win.")
        else:
            mode = "instant"
            why = "moving camera" if not static_ok else "suspected hidden cut (chroma)"
            reason = (f"single scene, low motion (mvMag {s.mv_mag_mean:.1f}) & fb {s.fb_react_mean:.0f}% "
                      f"but {why} -> layered unsafe; instant is real-time & acceptable.")

    s.probe_s = round(time.perf_counter() - t0, 2)
    if verbose:
        print(f"  signals: {asdict(s)}")
    return AutoRecommendation(mode=mode, reason=reason, signals=asdict(s),
                              matte_unavailable=matte_unavailable)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def _motion_keyed_thresh_fn(chunk, base_thresh):
    """Per-frame fallback threshold for the instant fast path (fed to anchor_sr.build_anchor_cache +
    patch_high_fallback). Returns None ONLY when BOTH features are off (=> scalar base_thresh =>
    byte-identical). Composes: E2 motion-keyed (lower thresh on high motion, default OFF) and the
    R4-E3 saturation cap (decline SR on high-fallback + LOW-motion noise frames, default ON). The
    index i is the same display-order 0-based index used everywhere (chunk/frames/R); MVs/fallback
    fractions are reused/memoized -> ~free."""
    cap_on = INSTANT_FALLBACK_SATURATION_CAP < 1.0
    if not INSTANT_MOTION_KEYED_FALLBACK and not cap_on:
        return None
    h_lr, w_lr = chunk[0][1].shape[:2]
    _anchors, _backbone = anchor_sr.anchor_indices(chunk)
    _mmemo, _fmemo, _imemo = {}, {}, {}

    def _mag(i):
        if i not in _mmemo:
            mvs = chunk[i][2]
            if mvs is None or len(mvs) == 0:
                _mmemo[i] = 0.0
            else:
                fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want="all")
                mg = np.sqrt(fx * fx + fy * fy)
                _mmemo[i] = float(np.nanmean(mg)) if np.isfinite(mg).any() else 0.0
        return _mmemo[i]

    def _frac(i):
        if i not in _fmemo:
            _fmemo[i] = (0.0 if i in _anchors
                         else anchor_sr._lr_fallback_fraction(chunk, i, _backbone, "reactive"))
        return _fmemo[i]

    def _intra(i):
        # fraction of the frame with NO motion vector at all (intra-coded blocks). On NOISE the
        # encoder intra-codes ~99% of blocks (inter-prediction on noise is RD-worthless) -> ~1.0;
        # a title-card reveal / dissolve intra-codes only the LOCAL new region -> well below 0.8.
        # This is the principled "nothing to propagate" signal that separates noise from a reveal.
        if i not in _imemo:
            mvs = chunk[i][2]
            if mvs is None or len(mvs) == 0:
                _imemo[i] = 1.0
            else:
                fx, _ = derisk.build_lr_flow(mvs, h_lr, w_lr, want="all")
                _imemo[i] = float(np.mean(~np.isfinite(fx)))
        return _imemo[i]

    def thr(i):
        b = base_thresh
        if INSTANT_MOTION_KEYED_FALLBACK and chunk[i][0] != "I" and _mag(i) > INSTANT_MOTION_GATE:
            b = INSTANT_FALLBACK_THRESH_HI
        # R4-E3 saturation cap: a NOISE-saturated frame (high fallback AND ~all-intra = no MVs to
        # propagate from) -> return an unreachable threshold so SR escalation is DECLINED (bicubic
        # floor, real-time held). The intra-fraction gate fires ONLY on genuine noise, NOT on a
        # title-reveal/dissolve (localized intra) -> safe to default ON with no clean-content regression.
        if cap_on and _frac(i) > INSTANT_FALLBACK_SATURATION_CAP and _intra(i) > INSTANT_SAT_CAP_INTRA_GATE:
            return 2.0
        return b

    return thr


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

    t0 = time.perf_counter()
    video_tmp = None                        # set once the (resolved) out path is known
    auto_info = None                        # populated only when mode == "auto"
    try:
        # AUTO: resolve to a concrete mode via the CHEAP probe FIRST, then run EXACTLY as today.
        # The probe runs under the single-job lock (it is part of this job) and inside this try, so a
        # probe failure still releases the lock. mode/out_path/cfg are derived from the RESOLVED mode,
        # so the rest of process_clip is byte-for-byte the existing instant/quality/layered run.
        if mode == "auto":
            _set_progress(state="probing", mode="auto", done=0, total=0, elapsed_s=0.0,
                          eta_s=None, ms_per_frame=None, message="auto: analyzing content")
            probe_n = min(AUTO_PROBE_FRAMES, max_frames) if max_frames else AUTO_PROBE_FRAMES
            rec = recommend_mode(input_path, max_frames=probe_n)
            mode = rec.mode
            auto_info = {"auto_requested": True, "auto_chosen": rec.mode,
                         "auto_reason": rec.reason, "auto_signals": rec.signals,
                         "auto_matte_unavailable": rec.matte_unavailable,
                         "auto_probe_frames": probe_n}

        cfg = MODE_CONFIG.get(mode)             # None for 'layered' (handled by _run_layered)
        if out_path is None:
            stem = os.path.splitext(os.path.basename(input_path))[0]
            out_path = os.path.join(OUTPUTS_DIR, f"{stem}_{mode}.mp4")
        video_tmp = out_path + ".video.tmp.mp4"

        if mode == "layered":
            out_path = _run_layered(input_path, out_path, video_tmp, max_frames, t0)
            if auto_info:                       # surface the auto decision alongside layered stats
                LAST_STATS.update(auto_info)
            return out_path
        _set_progress(state="probing", mode=mode, done=0, total=0, elapsed_s=0.0,
                      eta_s=None, ms_per_frame=None, message="probing input")
        fps = _probe_fps(input_path)
        total = probe_total_frames(input_path, max_frames)
        _set_progress(state="processing", total=total, message="upscaling")

        # The instant mode takes the FAST path (Levers 1-4): anchor-only SR, GPU-resident
        # reconstruct (no per-frame readback), GPU film grain, single host download, HW encode.
        # The quality mode keeps the original full-SR + region-aware + CPU-grain + libx264 path
        # untouched (HW encode is scoped to instant so quality/layered cannot regress).
        fast = (mode == "instant" and cfg["backend"] == "torch" and _gpu_ops is not None)
        eff_scale = INSTANT_SCALE if fast else SCALE   # instant=720p (x2), quality=QHD (x4)
        # R7-E1: "smooth 2x" is instant-fast-path-only + default OFF. When ON, every real frame is
        # preceded by an MV-interp midpoint -> the encoded stream carries 2x the frames and must run at
        # DOUBLE fps (out_fps) so it plays back over the SAME wall-clock duration (audio stays in sync).
        # OFF -> smooth is False -> out_fps==fps and the loop below never inserts a frame (byte-identical).
        smooth = fast and INSTANT_INTERP_2X
        out_fps = fps * 2 if smooth else fps

        writer = _VideoWriter(video_tmp, out_fps, codec=(None if fast else "libx264"))
        if fast:
            # Lever 2: encode on a worker thread so the VideoToolbox media engine runs frame i
            # while the GPU produces frame i+1 (the encode no longer sits on the critical path).
            writer = pipe_encode.ThreadedEncoder(writer, maxsize=8)
        done = 0
        n_emit = 0                          # R7-E1: TOTAL frames written (real + inserted); == done when OFF
        interp_carry = None                 # R7-E1: previous chunk's LAST recon (cross-chunk midpoint left)
        mid_count = 0                       # R7-E1: inserted-frame counter (midpoint grain seed offset)
        n_interp = n_interp_dup = 0         # R7-E1: inserted total / how many fell back to a duplicate
        n_chunks = 0
        w_lr = h_lr = w_hd = h_hd = None
        t_sr = t_recon = t_grain = t_encode = 0.0
        n_sr_calls = 0
        n_upgrades = 0
        ggrain = None

        def _emit():
            elapsed = time.perf_counter() - t0
            rate = elapsed / max(1, done)
            eta = rate * (total - done) if total and done <= total else None
            _set_progress(done=done, elapsed_s=round(elapsed, 1),
                          eta_s=(round(eta, 1) if eta is not None else None),
                          ms_per_frame=round(rate * 1000.0, 1))

        chunk_iter = stream_gops(input_path, max_frames=max_frames)
        if fast:
            # Lever 2: decode the next GOP chunk on a worker thread while the GPU works the current.
            chunk_iter = pipe_encode.prefetch_chunks(chunk_iter, maxsize=2)
        try:
            for chunk in chunk_iter:
                n_chunks += 1
                if w_lr is None:
                    h_lr, w_lr = chunk[0][1].shape[:2]
                    w_hd, h_hd = w_lr * eff_scale, h_lr * eff_scale

                if fast:
                    # E2 (default OFF): per-frame motion-keyed threshold -> escalate bicubic
                    #    occlusion fallback to compact-SR only on high-motion frames. None when OFF.
                    tfn = _motion_keyed_thresh_fn(chunk, INSTANT_FALLBACK_THRESH)
                    # 1) Lever 1: anchor-only SR cache -- SR the anchors + any high-fallback
                    #    BACKBONE frame (so its detail propagates); cv2 bicubic everywhere else.
                    ts = time.perf_counter()
                    perframe_cache, _ac, sr_set = anchor_sr.build_anchor_cache(
                        chunk, w_hd, h_hd, cfg["sr_mode"], occ_mode=cfg["occ"],
                        fallback_thresh=INSTANT_FALLBACK_THRESH,
                        tile=INSTANT_TILE_SR, gpu_cache=INSTANT_GPU_CACHE, thresh_fn=tfn)
                    t_sr += time.perf_counter() - ts

                    # 2) Lever 4: reconstruct GPU-resident (download_output=False) -- the HD recon
                    #    chain stays on-device; no per-frame host round-trip in reconstruct.
                    tr = time.perf_counter()
                    _, R = derisk.reconstruct(
                        chunk, None, eff_scale, True, cfg["occ"], perframe_cache, set(),
                        backend=cfg["backend"], collect_metrics=False, download_output=False)
                    t_recon += time.perf_counter() - tr

                    # 3) adaptive safeguard for the B LEAVES (post-reconstruct, correct since a
                    #    leaf is never a reference): SR-patch any B frame's fallback pixels above
                    #    threshold. hole_frac is exact + anchor-invariant -> no extra mask scan.
                    ts = time.perf_counter()
                    if INSTANT_SOFTOCC:
                        # R3-E3: HF-EMA soft-occlusion (escapes the high-motion frontier) instead of
                        # the hard SR-patch. Output-only; resets the EMA at every I-frame/cut/chunk start.
                        _anch, _bb = anchor_sr.anchor_indices(chunk)
                        p_info = anchor_sr.softocc_patch(
                            chunk, R, w_hd, h_hd, cfg["sr_mode"],
                            anchors=_anch, backbone=_bb,
                            reset_idx=anchor_sr.softocc_reset_indices(chunk),
                            gain=SOFTOCC_GAIN, beta=SOFTOCC_BETA, feather_k=SOFTOCC_FEATHER,
                            occ_mode=cfg["occ"], skip=sr_set, motion_gate=SOFTOCC_MOTION_GATE)
                    else:
                        p_info = anchor_sr.patch_high_fallback(
                            chunk, R, w_hd, h_hd, cfg["sr_mode"],
                            fallback_thresh=INSTANT_FALLBACK_THRESH, skip=sr_set,
                            tile=INSTANT_TILE_SR, thresh_fn=tfn)
                    t_sr += time.perf_counter() - ts
                    n_sr_calls += p_info["n_sr_calls"]
                    n_upgrades += p_info["n_adaptive_upgrades"]

                    if ggrain is None and cfg["grain"] != "off":
                        ggrain = fast_grain.GpuGrain(h_hd, w_hd, _gpu_ops.device())

                    # 4) Lever 2 grain on-device + the single GPU->host download (HW encode needs
                    #    a CPU frame) + Lever 3 HW encode, then free the chunk. R7-E1: when `smooth`,
                    #    emit an MV-interp MIDPOINT (output-only) BEFORE each real frame -> 2x output fps.
                    for i in range(len(chunk)):
                        recon_t = R[i]["recon"]                  # [1,3,H,W] float, GPU-resident (RAW)
                        # ---- R7-E1 INSERTED midpoint (default OFF): the half-step that PRECEDES real
                        #      frame i. left = prev chunk's last recon (i==0) else R[i-1]['recon'] (RAW --
                        #      grain writes a NEW tensor, never mutates R[]); connecting field = frame i's
                        #      codec 'past' MV (reused, zero new flow). OUTPUT-ONLY: reads R[], never
                        #      writes it, so it is structurally incapable of entering the ref chain.
                        if smooth:
                            left = interp_carry if i == 0 else R[i - 1]["recon"]
                            if left is not None:
                                fx, fy = interp_pass.connecting_flow(
                                    chunk, i, h_lr, w_lr, _build_lr_flow=derisk.build_lr_flow)
                                mid_t, minfo = interp_pass.midpoint_torch(
                                    left, recon_t, fx, fy, eff_scale,
                                    cut_thresh=INTERP_CUT_THRESH, _G=_gpu_ops)
                                if cfg["grain"] != "off":
                                    mid_t = ggrain.apply(mid_t, _MID_GRAIN_SEED_BASE + mid_count,
                                                         cfg["grain"])
                                writer.write(fast_grain.download_rgb(mid_t))
                                n_emit += 1
                                mid_count += 1
                                n_interp += 1
                                if minfo["duplicated"]:
                                    n_interp_dup += 1
                        # ---- real frame i (UNCHANGED from today's pipeline; grain seed == done) ----
                        tg = time.perf_counter()
                        rt = ggrain.apply(recon_t, done, cfg["grain"]) if cfg["grain"] != "off" else recon_t
                        recon = fast_grain.download_rgb(rt)       # contiguous-HWC GPU->host (~5x)
                        t_grain += time.perf_counter() - tg
                        te = time.perf_counter()
                        writer.write(recon)
                        t_encode += time.perf_counter() - te
                        done += 1
                        n_emit += 1
                        _emit()
                    # R7-E1: carry THIS chunk's last recon (clone BEFORE freeing R) so the NEXT chunk's
                    #        first frame interpolates from it. None when OFF -> no extra GPU memory held.
                    interp_carry = R[len(chunk) - 1]["recon"].clone() if smooth else None
                    del perframe_cache, R, chunk
                    _free_gpu()
                    continue

                # ---- QUALITY (region-aware) path -- full-SR pipeline ----
                # 1) per-frame SR cache for this chunk (anchor / fallback / baseline source).
                #    fp16 (E4) speeds the x4plus anchor ~1.24x, visually identical (default fp16=OFF
                #    on instant keeps that path byte-identical).
                ts = time.perf_counter()
                perframe_cache = derisk.build_perframe_cache(
                    chunk, w_hd, h_hd, cfg["sr_mode"], half=cfg.get("fp16", False))
                # 2) region-aware gate (quality only): temporally-stable motion gate + the
                #    per-frame COMPACT source for the OUTPUT-only blend (this chunk's frames).
                region_gate = (derisk._build_region_gate(chunk, w_hd, h_hd, SCALE)
                               if cfg["region_aware"] else None)
                # E3 V2: HD motion gate (1=static) for grain freezing -- reuse the region gate's a_lr.
                grain_static_hd = None
                if cfg.get("grain_motion") and region_gate is not None:
                    grain_static_hd = cv2.resize(region_gate["a_lr"], (w_hd, h_hd),
                                                 interpolation=cv2.INTER_LINEAR)
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
                #    + ENCODE incrementally, then free the chunk's HD frames. E3 V2: when a motion
                #    gate is available, freeze grain on static pixels (kills the grain-induced
                #    static flicker) while keeping fresh per-frame grain on motion.
                for i in range(len(chunk)):
                    recon = R[i]["recon"]
                    if cfg["grain"] != "off":
                        if grain_static_hd is not None:
                            recon = _grain.apply_grain_motion(recon, done, grain_static_hd, cfg["grain"])
                        else:
                            recon = _grain.apply_grain(recon, done, cfg["grain"])
                    writer.write(recon)
                    done += 1
                    _emit()

                del perframe_cache, region_gate, grain_static_hd, R, chunk
                _free_gpu()                         # release MPS cache between chunks (see _free_gpu)
            # R7-E1: trailing midpoint AFTER the global last real frame. It has no successor to warp
            #        toward, so it is a plain DUPLICATE of the last frame -> the stream closes at exactly
            #        2x (n_emit == 2*done). Only fires when smooth and a clip was actually produced.
            if smooth and interp_carry is not None:
                rt = (ggrain.apply(interp_carry, _MID_GRAIN_SEED_BASE + mid_count, cfg["grain"])
                      if cfg["grain"] != "off" else interp_carry)
                writer.write(fast_grain.download_rgb(rt))
                n_emit += 1
                mid_count += 1
                n_interp += 1
                n_interp_dup += 1
        finally:
            writer.close()

        if done == 0:
            raise RuntimeError(f"no frames decoded from {input_path}")

        # 5) mux audio into the final container (copy AAC / transcode / none). R7-E1: when smooth, the
        #    encoded stream has n_emit (==2*done) frames at out_fps (==2*fps) -> SAME wall duration as the
        #    real frames, so the audio stays in sync. OFF -> mux_n==done & out_fps==fps -> byte-identical.
        _set_progress(state="muxing", message="muxing audio")
        mux_n = n_emit if smooth else done
        audio_note = _mux_av(video_tmp, input_path, out_path, mux_n, out_fps)

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
            # R7-E1: total ENCODED frames + the rate they play at. OFF -> n_video_frames==n_frames &
            # out_fps==fps, so existing readers (and _verify_mp4 in __main__) are unchanged.
            "n_video_frames": mux_n, "out_fps": float(out_fps),
        })
        if smooth:                                 # R7-E1 "smooth 2x" extras (instant-only, opt-in)
            LAST_STATS.update({
                "smooth_2x": True, "n_interp": n_interp, "n_interp_dup": n_interp_dup,
            })
        if fast:                                   # instant fast-path extras (Levers 1-4)
            LAST_STATS.update({
                "t_grain_io_s": round(t_grain, 2),     # GPU grain + the single host download
                "t_encode_s": round(t_encode, 2),      # HW (VideoToolbox) encode
                "video_encoder": getattr(writer, "encoder", None),
                "n_sr_calls": n_sr_calls,              # anchors + adaptive upgrades
                "sr_calls_per_frame": round(n_sr_calls / max(1, done), 3),
                "n_adaptive_upgrades": n_upgrades,
            })
        _set_progress(state="done", done=done, elapsed_s=round(total_s, 1),
                      eta_s=0.0, ms_per_frame=round(total_s * 1000.0 / max(1, done), 1),
                      message="done")
        if auto_info:                           # surface the auto decision in LAST_STATS
            LAST_STATS.update(auto_info)
        return out_path
    except Exception as e:
        _set_progress(state="error", message=f"{type(e).__name__}: {e}")
        raise
    finally:
        if video_tmp and os.path.exists(video_tmp):
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
    ok, info = _verify_mp4(out, s.get("n_video_frames", s["n_frames"]), s["out_resolution"])
    print(f"[pipeline] re-decode check: ok={ok}  {info}")
    if a.mem and peak["rss"]:
        print(f"[pipeline] peak RSS: {peak['rss']/1e6:.0f} MB over {len(samples)} samples "
              f"(chunks={s['n_chunks']}, cap={s['soft_cap']})")
        # a few (frames_done, rss_MB) samples to show it stays flat across chunks
        step = max(1, len(samples) // 8)
        thin = [(d, round(r / 1e6)) for d, r in samples[::step]]
        print(f"[pipeline] (frames_done, RSS_MB) samples: {thin}")
    sys.exit(0 if ok else 2)
