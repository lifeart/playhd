"""R4-E1 fix (a) -- the per-frame PLATE-VALIDITY GUARD (reference implementation).

The layered composite lays a per-scene background plate under (1-alpha). If a scene cut is
MISSED (similar-luma) the plate spans two scenes and the WRONG background is painted over a whole
scene (silent corruption: LR-consistency 33.8->14.7 dB; tOF blind). This guard cheaply checks, per
frame, whether the plate actually matches THIS frame's background, and falls back when it does not.

Signal: downscale the (grained) HD plate to LR (INTER_AREA, the area-average inverse of the SR
upscale) and PSNR it against the decoded LR frame over the BACKGROUND region (matte alpha < 0.5,
eroded so the soft matte edge does not contaminate). When the plate is the RIGHT scene's
background this is high (~30-40 dB, it IS a temporal median of these very frames); when the plate
is the WRONG scene's background it craters (~12-16 dB). The fallback output is the full-frame
COMPACT SR of the actual frame -- already computed inside the composite, so the tripped frame costs
NO extra SR -- which is faithful to the real content (high LR-consistency), killing the corruption.

This module mirrors what lands in server/layered_api.py. It is import-light (numpy+cv2 only); the
verify harness monkeypatches layered_api.composite_frame with `composite_frame_guarded`.
"""
from __future__ import annotations

import numpy as np
import cv2

# --- guard thresholds (calibrated in verify_guard.py; see REPORT) ---------------------- #
PLATE_GUARD_ENABLE = True
PLATE_GUARD_PSNR_DB = 24.0      # ABSOLUTE floor: a plate below this bg-PSNR is wrong -> fall back.
                                #   (corrupt c7 frames = 14.7 dB; worst legit plate frame = 30.3 dB
                                #    -> 6-9 dB margin either side.)
PLATE_GUARD_DROP_DB = 8.0       # RELATIVE cliff: also fall back on a sudden >8 dB drop below the
                                #   plate's own established per-scene level. Protects a uniformly
                                #   LOWER-but-correct plate (textured/low-light/grainy bg) from the
                                #   absolute floor, while still catching a mid-scene plate that was
                                #   good then went wrong. (c7's missed cut is a ~19 dB cliff.)
PLATE_GUARD_ERODE = 4           # erode the bg mask (px) to drop the soft matte-edge band
PLATE_GUARD_MIN_BG = 0.02       # need at least this fraction of the frame as bg to judge
PLATE_GUARD_EMA = 0.30          # per-scene bg-PSNR baseline EMA weight (matches scene_detect)


def plate_bg_psnr(img_lr, pha_lr, plate_hd, erode=PLATE_GUARD_ERODE,
                  min_bg=PLATE_GUARD_MIN_BG):
    """PSNR(plate-downscaled-to-LR, decoded-LR) over the BACKGROUND region only.
    img_lr: uint8 HxWx3 decoded LR. pha_lr: HxW float matte in [0,1]. plate_hd: uint8 HD plate.
    Returns dB (inf when there is too little background to judge -> never trips)."""
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


def plate_is_bad(bg_psnr, baseline,
                 abs_db=PLATE_GUARD_PSNR_DB, drop_db=PLATE_GUARD_DROP_DB):
    """True => the plate does NOT match this frame's background (fall back). `baseline` is the
    per-scene EMA of bg_psnr over frames that PASSED (None until seeded). Trips on the absolute
    floor OR a sudden cliff below the plate's own established level."""
    if not np.isfinite(bg_psnr):
        return False                         # too little bg to judge -> trust the plate
    if bg_psnr < abs_db:
        return True
    if baseline is not None and bg_psnr < baseline - drop_db:
        return True
    return False
