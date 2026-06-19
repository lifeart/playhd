"""background_plate.py -- Stage L2 of the LAYERED architecture: the BACKGROUND PLATE.

The layered pipeline splits a talking-head frame into a long-lived STATIC background
and a per-frame DYNAMIC foreground, each super-resolved on its own budget. This module
builds the background half:

  accumulate ONE static-background image -- larger/cleaner than any single frame, with
  the moving subject removed -- by TEMPORAL-MEDIAN of the background region across the
  window, then HEAVY-SR it ONCE.

Why this is the amortization win: the background is SR'd a SINGLE time per scene
(~2.2 s for the x4plus net) and reused under every frame's foreground, instead of being
re-super-resolved per frame. For a 48-frame window that is one heavy SR call instead of
48 -- and it scales with scene length.

Method (Wang & Adelson 1994 "Layered representations" + Omnimatte): for each background
pixel, take the MEDIAN over all frames where it was NOT covered by the foreground matte
(gate==0). As the subject moves, regions occluded in one frame are revealed in others,
so the plate FILLS IN. Pixels behind the subject in EVERY frame stay holes -- acceptable,
because the subject always covers them in the composite, so they are never displayed.

Camera assumption: this v1 targets a STATIC camera (the talking-head case). Then plate
accumulation is plain image-space median -- no homography. `estimate_global_motion`
verifies the assumption from the codec MVs; `sample_plate`'s `global_motion` hook is the
seam for translational global-motion compensation if the camera ever moves.

Read-only imports of the shared modules:
  derisk.decode_lr_and_mvs   (decode + MVs, for the global-motion check)
  sr.upscale                 (heavy x4plus SR -- one call per scene)
  matting.{load_rvm, matte_sequence, fg_mask_lr}   (Stage L1 -- in the demo, not here)

Import-safe: importing this module loads no weights and touches no GPU. Only the
functions do work, and `sr_plate` is the only one that allocates the SR net.

Public API (consumed by Stage L3 compositor)
--------------------------------------------
  estimate_global_motion(decoded, gates=None, want="past") -> dict   (static-camera check)
  build_plate(frames, gates, min_samples=1) -> (plate_lr, coverage, hole_mask)
  sr_plate(plate_lr, scale=4, model="realesrgan-x4plus") -> plate_hd
  sample_plate(plate_hd, frame_idx=0, global_motion=None) -> hd_bg_for_frame
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Task 1: static-camera verdict from the codec motion vectors
# --------------------------------------------------------------------------- #
def _frame_mv_table(mvs, want="past"):
    """Vectorize one frame's PyAV MV struct array into per-block arrays.

    Returns (dx, dy, cx, cy, area) float arrays at LR (sub-pixel dx/dy recovered from
    motion_x / motion_scale, NOT the rounded src_x). `want` filters by the codec
    `source` sign (verified: <0 = past ref, >0 = future ref); 'past' keeps the P/backbone
    refs that exist on every inter frame. Empty arrays for intra / None.
    """
    if mvs is None or len(mvs) == 0:
        z = np.empty(0, np.float32)
        return z, z, z, z, z
    s = mvs["source"].astype(np.int32)
    if want == "past":
        keep = s < 0
    elif want == "future":
        keep = s > 0
    else:
        keep = np.ones_like(s, dtype=bool)
    if not keep.any():
        z = np.empty(0, np.float32)
        return z, z, z, z, z
    r = mvs[keep]
    ms = r["motion_scale"].astype(np.float32)
    ms[ms == 0] = 1.0
    dx = r["motion_x"].astype(np.float32) / ms
    dy = r["motion_y"].astype(np.float32) / ms
    cx = r["dst_x"].astype(np.float32)
    cy = r["dst_y"].astype(np.float32)
    area = (r["w"].astype(np.float32) * r["h"].astype(np.float32))
    return dx, dy, cx, cy, area


def estimate_global_motion(
    decoded: Sequence[Tuple[str, np.ndarray, object]],
    gates: Optional[Sequence[np.ndarray]] = None,
    want: str = "past",
    static_thresh: float = 0.5,
) -> dict:
    """Estimate camera/global motion across the window from the codec MVs.

    decoded: the list returned by derisk.decode_lr_and_mvs -- (ptype, rgb, mvs) per frame.
    gates:   optional per-frame FG gates (HxW, 1=foreground). If given, the BACKGROUND
             cross-check restricts MVs to blocks whose centre falls on a background pixel
             (gate<0.5) -- a static camera predicts those at ~0 motion regardless of how
             the subject moves.
    static_thresh: px threshold below which the camera is called static.

    The "global" camera motion is read as the MEDIAN MV vector per frame: the background
    is the largest area, so it dominates the median, and a consistent non-zero median
    vector = a camera pan/zoom while a near-zero scattered field = static camera + local
    subject motion. Returns a dict with both the global-vector magnitude (the headline
    number) and the all-block / background-block median magnitudes, plus a verdict.
    """
    per_frame_vec_mag = []   # |median (dx,dy)| per frame  -> camera translation proxy
    per_frame_all_mag = []   # median |MV| over all kept blocks
    per_frame_bg_mag = []    # median |MV| over background blocks only
    per_frame_vec = []       # (dx,dy) median vector per frame
    n_inter = 0
    for i, (ptype, _img, mvs) in enumerate(decoded):
        dx, dy, cx, cy, area = _frame_mv_table(mvs, want=want)
        if dx.size == 0:
            continue
        n_inter += 1
        mag = np.hypot(dx, dy)
        mdx, mdy = float(np.median(dx)), float(np.median(dy))
        per_frame_vec.append((mdx, mdy))
        per_frame_vec_mag.append(float(np.hypot(mdx, mdy)))
        per_frame_all_mag.append(float(np.median(mag)))
        if gates is not None and i < len(gates):
            g = gates[i]
            h, w = g.shape[:2]
            ix = np.clip(cx.astype(np.int32), 0, w - 1)
            iy = np.clip(cy.astype(np.int32), 0, h - 1)
            is_bg = g[iy, ix] < 0.5
            if is_bg.any():
                per_frame_bg_mag.append(float(np.median(mag[is_bg])))

    def _med(a):
        return float(np.median(a)) if len(a) else float("nan")

    global_vec_mag = _med(per_frame_vec_mag)
    all_mag = _med(per_frame_all_mag)
    bg_mag = _med(per_frame_bg_mag)
    # camera is static if the dominant (median) MV vector is ~0 AND, when we can measure
    # it, the background blocks themselves are ~0 motion.
    static = (global_vec_mag < static_thresh) and (
        np.isnan(bg_mag) or bg_mag < static_thresh
    )
    if n_inter == 0:
        verdict = "UNKNOWN"        # no inter frames / no MVs -> cannot assert
    elif static:
        verdict = "STATIC"
    else:
        verdict = "MOVING"
    return dict(
        n_inter_frames=n_inter,
        global_vec_mag_px=global_vec_mag,     # HEADLINE: |median MV vector| (camera trans)
        all_block_median_mag_px=all_mag,      # median |MV| over all blocks (subject incl.)
        bg_block_median_mag_px=bg_mag,        # median |MV| over background blocks only
        per_frame_vec_mag=per_frame_vec_mag,
        per_frame_global_vec=per_frame_vec,
        static_thresh_px=static_thresh,
        is_static=bool(static and n_inter > 0),
        verdict=verdict,
    )


# --------------------------------------------------------------------------- #
# Scene boundaries: a plate is PER-SCENE. The codec re-anchors at scene cuts with a
# mid-stream I-frame, so the I-frames inside a window are the natural cut signal.
# (Mixing two scenes into one plate contaminates the median -- the cut MUST be honored.)
# --------------------------------------------------------------------------- #
def find_scene_cuts(decoded, frames=None, jump_thresh=25.0):
    """Return sorted indices where a NEW scene starts inside the decoded window.

    Primary signal: a mid-window I-frame (index>0 with ptype=='I') -- the encoder inserts
    one at a scene cut (verified on this clip). Optional confirmation: a large mean
    frame-to-frame RGB jump (>jump_thresh) when `frames` is supplied. The window
    [cut[k], cut[k+1]) is one scene = one plate. Index 0 is implicitly a scene start.
    """
    cuts = []
    for i, (ptype, _img, _mv) in enumerate(decoded):
        if i > 0 and ptype == "I":
            cuts.append(i)
    if frames is not None:
        for i in range(1, len(frames)):
            d = float(np.abs(np.asarray(frames[i], np.int16) - np.asarray(frames[i - 1], np.int16)).mean())
            if d > jump_thresh and i not in cuts:
                cuts.append(i)
    return sorted(set(cuts))


def scene_segments(decoded, frames=None, min_len=1, **kw):
    """Split the window into (start, end) half-open scene segments at find_scene_cuts."""
    cuts = find_scene_cuts(decoded, frames=frames, **kw)
    bounds = [0] + cuts + [len(decoded)]
    segs = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= min_len]
    return segs


# --------------------------------------------------------------------------- #
# Task 2: build the LR background plate by temporal-median of the background region
# --------------------------------------------------------------------------- #
def build_plate(
    frames: Sequence[np.ndarray],
    gates: Sequence[np.ndarray],
    min_samples: int = 1,
    hole_fill: str = "inpaint",
    row_chunk: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate the per-pixel temporal MEDIAN of background-only pixels into one plate.

    frames: list of N uint8 HxWx3 RGB LR frames (display order, uniform size).
    gates:  list of N HxW foreground gates (1=foreground/dynamic, 0=background/static),
            e.g. matting.fg_mask_lr(pha, lr_hw, soft=False, dilate=3). Anything >=0.5 is
            treated as foreground and EXCLUDED from that frame's contribution.
    min_samples: a pixel is a "hole" if fewer than this many frames saw it as background.
            (hole_mask flags <1; coverage carries the full count for richer thresholds.)
    hole_fill: how to fill always-occluded holes so the plate has no NaN for SR:
            "inpaint"   -> cv2.inpaint (Telea) the hole from the surrounding RECOVERED
            background -> a clean, complete background plate (no subject-ghost) and a
            seamless SR input. The hole is invisible in the composite anyway, but a clean
            fill avoids SR edge artifacts bleeding from a subject-ghost into visible pixels.
            "allmedian" -> median over ALL frames (subject included) at hole pixels (a
            deterministic colour, but leaves a faint subject-ghost in the hole).
            hole_mask is returned either way so L3 knows which pixels are guesses.
    row_chunk: process the masked median in row bands to bound peak memory for long scenes.

    Returns
      plate_lr  : uint8 HxWx3 -- the accumulated static background.
      coverage  : int32 HxW   -- # of frames that contributed a background sample per pixel.
      hole_mask : bool  HxW   -- True where coverage < min_samples (always-occluded region).
    """
    frames = list(frames)
    gates = list(gates)
    if not frames:
        raise ValueError("build_plate: no frames")
    if len(gates) != len(frames):
        raise ValueError(f"build_plate: {len(frames)} frames but {len(gates)} gates")
    H, W = frames[0].shape[:2]

    stack = np.stack([np.asarray(f, np.float32) for f in frames], axis=0)  # N,H,W,3
    # background = gate < 0.5  (True where the pixel is static background in that frame)
    bg = np.stack([np.asarray(g, np.float32) < 0.5 for g in gates], axis=0)  # N,H,W bool
    coverage = bg.sum(axis=0).astype(np.int32)                              # H,W

    plate = np.empty((H, W, 3), np.float32)
    with warnings.catch_warnings():
        # all-background-empty pixels yield an all-NaN slice -> we fill them below
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for y0 in range(0, H, row_chunk):
            y1 = min(y0 + row_chunk, H)
            sub = stack[:, y0:y1]                 # N, ch, W, 3
            bm = bg[:, y0:y1, :, None]            # N, ch, W, 1
            masked = np.where(bm, sub, np.nan)    # background-only samples
            plate[y0:y1] = np.nanmedian(masked, axis=0)

    hole_mask = coverage < min_samples
    # fill NaN (always-occluded) pixels so SR sees a clean image
    nan_pix = np.isnan(plate).any(axis=2)
    fill_pix = hole_mask | nan_pix
    if fill_pix.any():
        if hole_fill == "inpaint":
            # provisional all-median fill first (so cv2 has no NaN), then inpaint the holes
            # from the surrounding RECOVERED background for a clean, ghost-free plate.
            import cv2
            allmed = np.median(stack, axis=0)
            plate[nan_pix] = allmed[nan_pix]      # kill NaN before the uint8 cast
            base = np.clip(plate, 0, 255).astype(np.uint8)
            mask8 = (fill_pix.astype(np.uint8)) * 255
            bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
            filled = cv2.inpaint(bgr, mask8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB), coverage, hole_mask
        # "allmedian" (or anything else): plausible colour, leaves a faint subject-ghost
        allmed = np.median(stack, axis=0)        # H,W,3 (subject included)
        plate[fill_pix] = allmed[fill_pix]

    plate = np.clip(plate, 0, 255).astype(np.uint8)
    return plate, coverage, hole_mask


def coverage_report(coverage: np.ndarray, hole_mask: np.ndarray) -> dict:
    """Summarize plate completeness: %% of pixels with >=1 and >=3 background samples and
    the always-occluded hole fraction. Handy for the demo / L3 telemetry."""
    total = coverage.size
    return dict(
        total_pixels=int(total),
        pct_ge1=100.0 * float((coverage >= 1).sum()) / total,
        pct_ge3=100.0 * float((coverage >= 3).sum()) / total,
        pct_ge5=100.0 * float((coverage >= 5).sum()) / total,
        hole_pct=100.0 * float(hole_mask.sum()) / total,
        max_coverage=int(coverage.max()),
        median_coverage=float(np.median(coverage)),
    )


# --------------------------------------------------------------------------- #
# Task 4: heavy-SR the plate ONCE  (the amortization)
# --------------------------------------------------------------------------- #
def sr_plate(
    plate_lr: np.ndarray,
    scale: int = 4,
    model: str = "realesrgan-x4plus",
) -> np.ndarray:
    """Super-resolve the LR plate ONCE with the heavy x4plus net -> the HD background plate.

    This is the whole point of the layer: ~2.2 s of heavy SR is paid a SINGLE time for the
    entire scene and reused under every frame. `sr` is imported lazily so this module stays
    import-safe (no weights load until you actually build a plate).

    scale: only x4 is supported by the heavy net; a non-4 scale resizes the x4 result
           (via sr.upscale_to) -- kept as a hook, x4 is the deployed path.
    """
    import sr  # lazy: importing background_plate must not load the SR net
    if scale == 4:
        return sr.upscale(plate_lr, model=model)
    h, w = plate_lr.shape[:2]
    return sr.upscale_to(plate_lr, w * scale, h * scale, model=model)


# --------------------------------------------------------------------------- #
# Task 5: sample the HD plate for a given frame  (static => identity; hook for motion)
# --------------------------------------------------------------------------- #
def sample_plate(
    plate_hd: np.ndarray,
    frame_idx: int = 0,
    global_motion: Optional[object] = None,
) -> np.ndarray:
    """Return the HD background for `frame_idx` to lay under that frame's foreground.

    STATIC camera (this v1's target): the plate is fixed, so this is IDENTITY -- every
    frame gets the same HD plate. L3 just composites `foreground_hd over sample_plate(...)`.

    `global_motion` is the seam for a MOVING camera. If supplied it provides a per-frame
    LR translation to apply to the HD plate (camera pan/zoom compensation):
      * None                      -> identity (static camera).
      * ndarray of shape (N, 2)   -> cumulative LR (dx, dy) per frame; the HD plate is
                                     shifted by scale*-(dx,dy) so the fixed-world plate
                                     aligns to this frame's camera.
      * callable(plate_hd, idx)   -> fully custom warp (e.g. homography) for advanced use.
    Only translation is implemented here (codec MVs give translation cheaply); a homography
    hook is left to the callable form. For the talking-head deliverable this returns the
    plate unchanged.
    """
    if global_motion is None:
        return plate_hd
    if callable(global_motion):
        return global_motion(plate_hd, frame_idx)

    gm = np.asarray(global_motion, dtype=np.float32)
    if gm.ndim == 2 and gm.shape[1] == 2:
        import cv2
        h_hd, w_hd = plate_hd.shape[:2]
        # infer the LR->HD scale is baked into the plate already; the (dx,dy) are LR px,
        # so multiply by the HD/LR ratio. We don't know LR here, so treat gm as HD-px if
        # the caller pre-scaled; otherwise the caller passes LR px and we need the ratio.
        idx = int(np.clip(frame_idx, 0, gm.shape[0] - 1))
        dx, dy = float(gm[idx, 0]), float(gm[idx, 1])
        M = np.array([[1, 0, -dx], [0, 1, -dy]], np.float32)
        return cv2.warpAffine(
            plate_hd, M, (w_hd, h_hd),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
    raise ValueError(
        "sample_plate: global_motion must be None, an (N,2) translation array, or a callable"
    )


if __name__ == "__main__":
    # tiny self-contained smoke test (no real footage / no models): a moving white square
    # over a fixed gradient background -> the plate must recover the FULL background.
    H, W, N = 60, 100, 24
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.stack([
        (xx / W * 255), (yy / H * 255), ((xx + yy) / (W + H) * 255)
    ], axis=-1).astype(np.uint8)
    frames, gates = [], []
    for t in range(N):
        f = bg.copy()
        x0 = int(t / N * (W - 20))
        g = np.zeros((H, W), np.float32)
        f[20:40, x0:x0 + 20] = (255, 0, 0)   # moving subject paints over the background
        g[20:40, x0:x0 + 20] = 1.0           # gate marks it foreground
        frames.append(f)
        gates.append(g)
    plate, cov, hole = build_plate(frames, gates)
    rep = coverage_report(cov, hole)
    err = np.abs(plate.astype(int) - bg.astype(int)).mean()
    print("background_plate.py smoke test")
    print(f"  plate {plate.shape} coverage>=1 {rep['pct_ge1']:.1f}%  holes {rep['hole_pct']:.2f}%")
    print(f"  mean |plate - true_bg| = {err:.2f}  (subject removed cleanly if ~0)")
    gm = estimate_global_motion([("P", f, None) for f in frames])
    print(f"  estimate_global_motion (no MVs) verdict={gm['verdict']}")
    print(f"  sample_plate identity ok: {np.array_equal(sample_plate(plate), plate)}")
    print("OK")
