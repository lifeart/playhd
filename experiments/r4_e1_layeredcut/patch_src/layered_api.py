"""layered_api.py -- LAYERED quality mode helpers (two-pass-per-scene, bounded memory).

This module holds the heavy lifting for the server's third quality mode, "layered":
SR the STATIC background ONCE per scene (a long-lived HD plate) and manage only the
moving FOREGROUND per frame. It is the streaming/constant-memory realization of the
prototype's layered idea (matting.py + background_plate.py + layered_pipeline.py).

It imports the validated prototype modules READ-ONLY (never modifies them) and adds
NO new matte / SR / plate math -- it only orchestrates them so the WHOLE clip (any
length) is processed with bounded peak memory:

  * PASS 0  segment_scenes()      -- one lightweight streaming decode that records the
                                     scene-boundary frame indices (mirrors the rule in
                                     background_plate.find_scene_cuts: a mid-stream I-frame
                                     OR a large RGB jump starts a new scene). Holds ONE
                                     previous frame, never the whole clip.
  * PASS A  build_scene_plates()  -- for each scene, decode a CAPPED, evenly-sampled
                                     subset (<= cap frames), matte it (RVM), build the
                                     temporal-median background plate, heavy-SR it ONCE
                                     (x4plus), and SPILL the HD plate to disk. Also runs
                                     the static-camera check (estimate_global_motion); a
                                     MOVING scene is flagged as a fallback (no plate).
  * PASS B is driven by pipeline_api (streaming GOP chunks): for a STATIC scene it
                                     composites alpha*fg_hd + (1-alpha)*plate_hd per frame
                                     (matte_frame_np threads RVM state per scene); a MOVING
                                     scene falls back to the regular region-aware path.

Two decode passes total (PASS 0 + the sampled PASS A share nothing held long-term; PASS B
is the third streaming read driven by the caller). Peak memory is bounded by (one capped
sample set + one HD plate) in PASS A and (one HD plate + one frame's working set) in PASS B.

LICENSE NOTE: the matte (Robust Video Matting) is CC BY-NC-SA 4.0 -- NON-COMMERCIAL. The
layered mode is a research/demo path; a commercial deployment needs a different matte source.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
import av

# Prototype on sys.path (read-only). pipeline_api also does this, but layered_api must be
# import-safe on its own (the prototype modules import each other by bare name).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_PROTO = os.path.join(_REPO, "prototype")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)
if _HERE not in sys.path:          # server/ on the path -> `import scene_detect` (sibling) resolves
    sys.path.insert(0, _HERE)

import scene_detect                  # noqa: E402  (shared scene-CUT detector -- one source of truth)
import derisk                        # noqa: E402  (decode/MVs, SDType, reconstruct)
import matting                       # noqa: E402  (L1: RVM matte)
import background_plate as bp        # noqa: E402  (L2: plate build + heavy SR + motion check)
import layered_pipeline as lp        # noqa: E402  (L3: alpha_to_hd / composite / fg budgets)
import sr                            # noqa: E402  (compact + x4plus nets)

# R3-E4 matte-source flag. LAYERED_MATTE selects which matte feeds the WHOLE layered path (PASS A
# plate gates AND PASS B per-frame composite). Default "rvm" keeps the current best-visual demo
# BYTE-IDENTICAL (Robust Video Matting, CC BY-NC-SA 4.0, NON-COMMERCIAL). "deeplab"/"lraspp" REBIND
# `matting` to the BSD-3 permissive (COMMERCIAL-OK) seg adapter (R2-E1 + R3-E4: plate matches RVM
# cov~74%/hole~26%, RVM-parity temporal stability via the display-order alpha-EMA, 0.8-1.0x latency;
# loses only wispy hair). The adapter re-exports fg_mask_lr/composite/auto_downsample_ratio VERBATIM
# from matting and implements matte_sequence (PASS A) + matte_frame (PASS B) with the SAME shapes, so
# this ONE rebind makes every `matting.*` call site consistent.
LAYERED_MATTE = os.environ.get("LAYERED_MATTE", "rvm").strip().lower()
LAYERED_MATTE_EMA = float(os.environ.get("LAYERED_MATTE_EMA", "0.5"))
_SEG_VARIANTS = {"deeplab": "deeplabv3_mobilenetv3", "lraspp": "lraspp_mobilenetv3"}
if LAYERED_MATTE in _SEG_VARIANTS:
    import seg_matte_layered as matting   # noqa: F811,E402  permissive BSD-3 (commercial-OK)

COMPACT = lp.COMPACT                 # "realesrgan"        (compact per-frame foreground SR)
HEAVY = lp.HEAVY                     # "realesrgan-x4plus" (heavy plate SR, once per scene)
SCALE = lp.SCALE                     # 4

# Per-scene plate is built from at most this many evenly-sampled frames -- enough to reveal
# the background behind the moving subject + denoise via temporal median, without ever
# holding a whole scene. The plate is heavy-SR'd ONCE; this cap bounds PASS A memory.
PLATE_SAMPLE_CAP = 64
FG_DILATE = 3                        # grow the FG gate so the matte-edge band stays foreground
STATIC_THRESH_PX = 0.6               # |median MV| above this (px) => camera moves => fallback
# R2-E3 seam-halo reduction (ON: it is a verified strict win -- seam ratio -> the uniform-x4plus
# ceiling, halo -34%, subject core EXACTLY unchanged, ~+5 ms/frame). PASS A bakes a band-localized
# plate-ring restore into the static plate ONCE per scene; PASS B feathers the matte alpha per
# frame. lp.composite stays byte-identical with its defaults, so only the layered path changes.
LAYERED_SEAM_FIX = True

# R4-E1 PLATE-VALIDITY GUARD. The per-scene background plate is laid under (1-alpha) every frame. If
# a scene cut is MISSED (similar-luma -- scene_detect's luma signal stays below threshold) the plate
# spans two scenes and the WRONG background is painted over a whole scene: a SILENT corruption (the
# wrong plate is temporally stable, so tOF is BLIND; only fidelity-vs-LR exposes it -- LR-consistency
# 33.8->14.7 dB on the c7 repro). This guard cheaply checks, PER FRAME, whether the plate actually
# matches THIS frame's background region, and FALLS BACK (to the full-frame compact SR of the real
# frame -- already computed inside the composite, so a tripped frame costs NO extra SR) when it does
# not. It is layered-scoped and robust to ANY missed cut, independent of the detector. R4-E1 verified:
# on c7 the post-cut LR-consistency is restored 14.7->42.5 dB; on a normal static talking-head the
# guard trips 0 frames and the plate path is byte-identical.
PLATE_GUARD_ENABLE = True
PLATE_GUARD_PSNR_DB = 24.0   # ABSOLUTE floor: a plate below this bg-PSNR is wrong -> fall back
                             #   (corrupt c7 frames 14.7 dB; worst legit plate frame 30.3 dB)
PLATE_GUARD_DROP_DB = 8.0    # RELATIVE cliff: also fall back on a sudden >8 dB drop below the plate's
                             #   own established per-scene level (protects a uniformly LOWER-but-correct
                             #   plate -- textured/low-light/grainy bg -- while catching a mid-scene
                             #   plate that was good then went wrong)
PLATE_GUARD_ERODE = 4        # erode the bg mask (px) to drop the soft matte-edge band
PLATE_GUARD_MIN_BG = 0.02    # need at least this fraction of the frame as bg to judge (else trust plate)
PLATE_GUARD_EMA = 0.30       # per-scene bg-PSNR baseline EMA weight (matches scene_detect)

# Scene-cut detection now lives in scene_detect (ONE source of truth, shared with
# pipeline_api.stream_gops). These constants are the layered defaults forwarded to it; the
# detector adds an adaptive-baseline (hysteresis) relative test on top of the same two signals
# (a near-total frame-to-frame jump, OR a smaller jump corroborated by a codec I-frame) and the
# same minimum-scene-length merge. Kept here so segment_scenes' public signature is unchanged.
CUT_THRESH = scene_detect.CUT_THRESH          # 60.0  mean |dluma| above this (any frame) => cut
IFRAME_CUT_THRESH = scene_detect.IFRAME_THRESH  # 45.0  at an I-frame, a smaller jump counts too
MIN_SCENE_LEN = scene_detect.MIN_SCENE_LEN    # 24  frames; cuts closer than this are merged


def _device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
# Flat streaming decoder: open the container ONCE, yield (idx, ptype, lr_rgb, mvs) for
# every frame in DISPLAY order. Mirrors derisk.decode_lr_and_mvs' export_mvs setup but
# never holds more than the caller keeps. Used by PASS 0 and PASS A.
# --------------------------------------------------------------------------- #
def stream_frames(path, max_frames=None):
    cont = av.open(path)
    try:
        vs = cont.streams.video[0]
        vs.codec_context.options = {"flags2": "+export_mvs"}
        idx = 0
        for frame in cont.decode(vs):
            if max_frames is not None and idx >= max_frames:
                break
            ptype = {1: "I", 2: "P", 3: "B"}.get(int(frame.pict_type), "?")
            img = frame.to_ndarray(format="rgb24")
            try:
                sd = frame.side_data.get(derisk.SDType.MOTION_VECTORS)
            except Exception:
                sd = None
            mvs = sd.to_ndarray() if sd is not None else None
            yield idx, ptype, img, mvs
            idx += 1
    finally:
        cont.close()


# --------------------------------------------------------------------------- #
# PASS 0: scene segmentation (bounded -- scene_detect holds ONE previous luma frame).
# The cut DETECTION is delegated to scene_detect.find_cuts (the SAME StreamingCutDetector that
# pipeline_api.stream_gops uses to force fresh anchors -> one source of truth). This function
# only assembles the per-scene [a,b) segments and merges a too-short TRAILING scene (the greedy
# minimum-scene-length between cuts is already applied inside the detector). Returns
# (segments, total_frames). A periodic keyframe (small diff) is NOT a cut; a real cut is a
# near-total content change OR an I-frame-corroborated jump OR a strong relative spike.
# --------------------------------------------------------------------------- #
def segment_scenes(path, max_frames=None, cut_thresh=CUT_THRESH,
                   iframe_thresh=IFRAME_CUT_THRESH, min_scene_len=MIN_SCENE_LEN):
    cuts, total = scene_detect.find_cuts(
        path, max_frames=max_frames,
        cut_thresh=cut_thresh, iframe_thresh=iframe_thresh, min_scene_len=min_scene_len)
    bounds = [0] + [c for c in cuts if 0 < c < total] + [total]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if segs and (b - a) < min_scene_len:  # merge a too-short trailing scene into prev
            segs[-1] = (segs[-1][0], b)
        else:
            segs.append((a, b))
    return segs, total


def scene_of(idx, segs):
    """Scene id (index into segs) containing frame `idx`. segs are contiguous & sorted."""
    for sid, (a, b) in enumerate(segs):
        if a <= idx < b:
            return sid
    return len(segs) - 1


def sample_indices(s0, s1, cap=PLATE_SAMPLE_CAP):
    """<= cap evenly-spaced frame indices in [s0, s1). Whole scene if it already fits."""
    n = s1 - s0
    if n <= cap:
        return list(range(s0, s1))
    step = n / float(cap)
    return sorted(set(s0 + int(k * step) for k in range(cap)))


# --------------------------------------------------------------------------- #
# Matting (RVM) -- recurrent, threaded per scene in display order.
# --------------------------------------------------------------------------- #
def load_matting_model():
    """Load the layered-path matte model on MPS (CPU fallback). Source = LAYERED_MATTE:
    "rvm" (default) -> Robust Video Matting (CC BY-NC-SA 4.0, NON-COMMERCIAL; byte-identical to the
    pre-R3-E4 demo). "deeplab"/"lraspp" -> BSD-3 torchvision person-seg + alpha-EMA (LAYERED_MATTE_EMA)
    via the seg_matte_layered rebind (COMMERCIAL-OK; R2-E1/R3-E4)."""
    if LAYERED_MATTE in _SEG_VARIANTS:
        return matting.load_seg(_device(), _SEG_VARIANTS[LAYERED_MATTE], ema=LAYERED_MATTE_EMA)
    return matting.load_rvm(_device())


def downsample_ratio(h, w):
    """RVM internal coarse-pass ratio for this resolution (matting.auto_downsample_ratio)."""
    return matting.auto_downsample_ratio(h, w)


def _frame_tensor(img, device):
    """uint8 HxWx3 RGB -> float [1,3,H,W] in [0,1] on `device` (matches matting._to_src_tensor)."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0)
    return t.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


def matte_frame_np(model, img, rec, ratio, device):
    """One recurrent RVM step on a single LR frame. Threads `rec` (list of 4 state tensors,
    start [None]*4) in display order -- the CALLER must reset rec at each scene boundary.
    Returns (pha_np HxW float32 in [0,1], rec). Uses the public matting.matte_frame."""
    src = _frame_tensor(img, device)
    _fgr, pha, rec = matting.matte_frame(model, src, rec, ratio)
    pha_np = pha[0, 0].clamp(0, 1).float().cpu().numpy()
    return pha_np, rec


# --------------------------------------------------------------------------- #
# PASS A: build + heavy-SR ONE background plate per scene, SPILLED TO DISK (bounded).
# One streaming decode keeps only the capped, evenly-sampled frames of the CURRENT scene;
# the moment we cross into the next scene we matte that scene's samples, run the static
# camera check, build+SR the plate, write it to disk, and free the samples.
# --------------------------------------------------------------------------- #
def build_scene_plates(
    path,
    segs,
    plate_dir,
    model,
    max_frames=None,
    cap=PLATE_SAMPLE_CAP,
    dilate=FG_DILATE,
    sr_model=HEAVY,
    static_thresh=STATIC_THRESH_PX,
    progress_cb=None,
):
    """Returns plates: dict sid -> {
        fallback: bool,                 # True => MOVING camera, use the region-aware path
        verdict: 'STATIC'|'MOVING'|'UNKNOWN',
        plate_path: str|None,           # HD plate .npy on disk (None if fallback)
        seg: (s0, s1), n_frames, n_samples,
        global_vec_mag_px, bg_block_median_mag_px,
        coverage_pct, hole_pct, plate_sr_ms, build_s,
    }.
    A scene is layered only when the static-camera check passes (or is UNKNOWN -- treated as
    static, since layered targets talking-heads); a materially MOVING camera falls back so we
    never composite a wrong fixed plate."""
    os.makedirs(plate_dir, exist_ok=True)
    device = getattr(model, "_rvm_device", _device())
    sample_sets = {sid: set(sample_indices(a, b, cap)) for sid, (a, b) in enumerate(segs)}
    collected: Dict[int, List[Tuple[str, np.ndarray, object]]] = {}
    plates: Dict[int, dict] = {}
    n_scenes = len(segs)

    def finalize(sid):
        t0 = time.perf_counter()
        samples = collected.pop(sid, [])
        s0, s1 = segs[sid]
        info = {"fallback": True, "verdict": "UNKNOWN", "plate_path": None,
                "seg": (s0, s1), "n_frames": s1 - s0, "n_samples": len(samples),
                "global_vec_mag_px": float("nan"), "bg_block_median_mag_px": float("nan"),
                "coverage_pct": 0.0, "hole_pct": 0.0, "plate_sr_ms": 0.0, "build_s": 0.0}
        if not samples:
            plates[sid] = info
            return
        imgs = [im for (_p, im, _m) in samples]
        h, w = imgs[0].shape[:2]
        # L1 matte the sampled subset (recurrence weak across sparse samples, but per-frame
        # alpha is all the plate median needs to EXCLUDE the subject).
        res = matting.matte_sequence(model, imgs)
        phas = [p for (_f, p) in res]
        gates = [matting.fg_mask_lr(p, lr_hw=(h, w), soft=False, thresh=0.5, dilate=dilate)
                 for p in phas]
        # static-camera verdict from the codec MVs, cross-checked on background blocks.
        gm = bp.estimate_global_motion(samples, gates=gates, static_thresh=static_thresh)
        info["verdict"] = gm["verdict"]
        info["global_vec_mag_px"] = gm["global_vec_mag_px"]
        info["bg_block_median_mag_px"] = gm["bg_block_median_mag_px"]
        if gm["verdict"] == "MOVING":
            info["fallback"] = True            # camera moves -> fixed plate would be wrong
            info["build_s"] = round(time.perf_counter() - t0, 2)
            plates[sid] = info
            return
        # STATIC (or UNKNOWN -> assume static): build the temporal-median plate + heavy-SR ONCE.
        plate_lr, coverage, hole_mask = bp.build_plate(imgs, gates, min_samples=1,
                                                       hole_fill="inpaint")
        rep = bp.coverage_report(coverage, hole_mask)
        plate_hd = bp.sr_plate(plate_lr, scale=SCALE, model=sr_model)
        if LAYERED_SEAM_FIX:
            # R2-E3: restore the soft near-subject plate ring ONCE per scene over the swept
            # alpha-union band (amortized like the plate SR). Union the sampled LR alphas, upscale
            # to HD, and re-contrast the BG-side ring band toward the deep-BG level.
            union_lr = np.maximum.reduce([p.astype(np.float32) for p in phas])
            union_hd = cv2.resize(union_lr, (plate_hd.shape[1], plate_hd.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)[..., None]
            plate_hd = lp.restore_plate_ring(plate_hd, union_hd, strength=0.8)
        plate_sr_ms = sr.last_latency_ms(sr_model)
        plate_path = os.path.join(plate_dir, f"plate_s{sid}.npy")
        np.save(plate_path, plate_hd)          # SPILL to disk -> PASS B holds one plate at a time
        info.update(fallback=False, plate_path=plate_path,
                    coverage_pct=round(rep["pct_ge1"], 1), hole_pct=round(rep["hole_pct"], 2),
                    plate_sr_ms=round(plate_sr_ms, 1))
        info["build_s"] = round(time.perf_counter() - t0, 2)
        plates[sid] = info

    last_sid = None
    for idx, ptype, img, mvs in stream_frames(path, max_frames=max_frames):
        sid = scene_of(idx, segs)
        if sid != last_sid:
            if last_sid is not None:
                finalize(last_sid)             # we just crossed a scene boundary
                if progress_cb:
                    progress_cb(len(plates), n_scenes)
            last_sid = sid
        if idx in sample_sets[sid]:
            collected.setdefault(sid, []).append((ptype, img, mvs))
    if last_sid is not None:
        finalize(last_sid)
        if progress_cb:
            progress_cb(len(plates), n_scenes)
    return plates


# --------------------------------------------------------------------------- #
# PASS B helper: composite one STATIC-scene frame (alpha*fg_hd + (1-alpha)*plate_hd).
# pipeline_api drives the streaming loop and the MOVING-scene fallback; this keeps the
# per-frame layered math in one place (delegating to layered_pipeline READ-ONLY).
# --------------------------------------------------------------------------- #
def composite_frame(img, pha, plate_hd, w_hd, h_hd):
    """alpha*compact_fg_hd + (1-alpha)*plate_hd for one frame. Returns uint8 HxWx3 RGB."""
    fg_hd, _ms = lp.foreground_compact(img)                 # compact per-frame FG SR
    alpha_hd = lp.alpha_to_hd(pha, (h_hd, w_hd))            # soft hair-edge alpha at HD
    # R2-E3: feather the matte alpha (recover hair wisps, core untouched). The plate ring was
    # already restored ONCE per scene in PASS A, so seam_restore stays 0 here.
    out, _c = lp.composite(fg_hd, alpha_hd, plate_hd, feather=LAYERED_SEAM_FIX)
    return out


# --------------------------------------------------------------------------- #
# R4-E1 PLATE-VALIDITY GUARD (fix a) -- cheap per-frame check + auto-fallback.
# --------------------------------------------------------------------------- #
def plate_bg_psnr(img_lr, pha_lr, plate_hd, erode=PLATE_GUARD_ERODE, min_bg=PLATE_GUARD_MIN_BG):
    """PSNR(plate-downscaled-to-LR, decoded-LR) over the BACKGROUND region only -- the honest
    fidelity-vs-LR check for THIS bug (tOF is blind). img_lr: uint8 HxWx3 decoded LR. pha_lr: HxW
    float matte in [0,1]. plate_hd: uint8 HD plate. The plate IS supposed to be this frame's
    background, so over background pixels (alpha<0.5, eroded to drop the soft matte edge) the
    plate downscaled by INTER_AREA (the area-average inverse of the SR upscale) should match the
    decoded LR closely (~30-40 dB); a WRONG-scene plate craters (~12-16 dB). Returns dB, or +inf
    when there is too little background to judge (-> never trips)."""
    h, w = img_lr.shape[:2]
    plate_lr = cv2.resize(plate_hd, (w, h), interpolation=cv2.INTER_AREA)
    bg = np.asarray(pha_lr, np.float32) < 0.5
    if erode > 0:
        bg = cv2.erode(bg.astype(np.uint8), np.ones((erode, erode), np.uint8)) > 0
    if int(bg.sum()) < max(64, int(min_bg * bg.size)):
        return float("inf")
    diff = (plate_lr.astype(np.float32) - img_lr.astype(np.float32))[bg]
    mse = float(np.mean(diff * diff))
    return 99.0 if mse < 1e-6 else float(10.0 * np.log10(255.0 ** 2 / mse))


def plate_is_bad(bg_psnr, baseline, abs_db=PLATE_GUARD_PSNR_DB, drop_db=PLATE_GUARD_DROP_DB):
    """True => the plate does NOT match this frame's background (fall back). `baseline` is the
    per-scene EMA of bg_psnr over frames that PASSED (None until seeded). Trips on the ABSOLUTE
    floor OR a sudden CLIFF below the plate's own established level."""
    if not np.isfinite(bg_psnr):
        return False                                  # too little bg to judge -> trust the plate
    if bg_psnr < abs_db:
        return True
    if baseline is not None and bg_psnr < baseline - drop_db:
        return True
    return False


def composite_frame_guarded(img, pha, plate_hd, w_hd, h_hd, plate_baseline=None):
    """Guarded PASS B composite. Computes fg_hd ONCE; if the plate fails the bg-validity check it
    returns the full-frame compact SR (faithful to the real frame -- kills the missed-cut plate
    corruption) instead of the plate composite. Returns (out_uint8, bg_psnr, plate_bad). The caller
    threads `plate_baseline` (per-scene EMA of bg_psnr over passing frames; reset at each scene)."""
    fg_hd, _ms = lp.foreground_compact(img)
    if PLATE_GUARD_ENABLE:
        bg_psnr = plate_bg_psnr(img, pha, plate_hd)
        if plate_is_bad(bg_psnr, plate_baseline):
            return fg_hd, bg_psnr, True               # FALL BACK: real content, no stale plate
    else:
        bg_psnr = float("nan")
    alpha_hd = lp.alpha_to_hd(pha, (h_hd, w_hd))
    out, _c = lp.composite(fg_hd, alpha_hd, plate_hd, feather=LAYERED_SEAM_FIX)
    return out, bg_psnr, False
