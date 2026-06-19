#!/usr/bin/env python3
"""
exp3_visual / motion_grain.py -- V2: MOTION-MODULATED film grain (OUTPUT-ONLY pass).

Problem: prototype/grain.apply_grain adds full-strength, FRESH-every-frame grain to every
pixel. On STATIC regions that the propagation chain has stabilised, the per-frame grain flips
sign every frame and re-injects flicker (raises |Delta F| / tOF) -- partly undoing the temporal
stability the warp+residual propagation worked to achieve.

Fix (output-only, mirrors grain.apply_grain exactly except for the per-pixel field): modulate
grain TEMPORALLY by the region-aware motion gate already computed in quality mode
(region_quality.window_static_weight via derisk._build_region_gate). a in [0,1], 1=STATIC.
  * STATIC (a=1):  use a FROZEN grain field (a fixed frame seed) -> spatially identical filmic
                   texture every frame -> contributes ~0 temporal flicker.
  * MOVING (a=0):  use FRESH per-frame grain -> independent frame-to-frame (filmic, decorrelated).
  * Between:       a*frozen + (1-a)*fresh, RENORMALISED to unit variance so grain amplitude is
                   spatially uniform across the seam (no visible grain-density step).
The frozen/fresh fields are both grain._frame_grain unit templates (content-independent), so the
ONLY change vs apply_grain on a static pixel is "same seed every frame" instead of "seed=index".

CRITICAL grain rules honoured:
  * grain is the FINAL pass, added to a COPY of the recon; never warped / propagated / fed to R[].
  * temporal-independence is measured on the RAW additive grain FIELD (returned here), never on
    Y(grained)-Y(recon) (the YCrCb round-trip's quantisation spuriously inflates correlation).

Reduced-amplitude variant (mode="reduced") kept for the report: fresh grain everywhere but its
amplitude scaled down on static (floor..1). Frozen is the primary (zero static flicker, full
spatial texture); reduced is the ablation (proportional, but still flickers on static).
"""
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.join(_HERE, "..", "..", "prototype")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

import grain as _grain     # READ-ONLY: STRENGTHS, _frame_grain, _LUT, _LUMA_BLUR_SIGMA


def _amp_map(y):
    """Local-luma amplitude map, identical to grain.apply_grain."""
    y_local = cv2.GaussianBlur(y, (0, 0), _grain._LUMA_BLUR_SIGMA)
    return _grain._LUT[np.clip(y_local, 0, 255).astype(np.uint8)]


def apply_grain_motion(rgb_uint8, frame_idx, static_w_hd, strength="med", template=None,
                       frozen_idx=0, static_floor=0.0, mode="frozen", return_grain=False):
    """Motion-modulated grain. `static_w_hd` is the HxW motion gate (1=static, 0=moving), already
    upsampled to the HD frame size. Output-only. Returns the grained uint8 frame (and, if
    return_grain=True, the RAW additive luma grain FIELD for an artifact-free independence check)."""
    sigma = _grain.STRENGTHS.get(strength, 0.0) if isinstance(strength, str) else float(strength)
    h, w = rgb_uint8.shape[:2]
    if sigma <= 0.0:
        z = np.zeros((h, w), np.float32)
        return (rgb_uint8.copy(), z) if return_grain else rgb_uint8.copy()
    ycc = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycc[:, :, 0]
    amp = _amp_map(y)
    a = np.clip(static_w_hd, 0.0, 1.0).astype(np.float32)
    fresh = _grain._frame_grain(h, w, frame_idx, template)
    if mode == "frozen":
        frozen = _grain._frame_grain(h, w, frozen_idx, template)
        unit = a * frozen + (1.0 - a) * fresh
        norm = np.sqrt(a * a + (1.0 - a) ** 2)          # keep unit variance across the seam
        unit = unit / np.maximum(norm, 1e-6)
    elif mode == "reduced":
        scale = static_floor + (1.0 - static_floor) * (1.0 - a)   # static->floor, moving->1
        unit = fresh * scale
    else:
        raise ValueError(f"unknown mode {mode!r}")
    grain = unit * sigma * amp
    ycc[:, :, 0] = np.clip(y + grain, 0, 255)
    out = cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return (out, grain) if return_grain else out


def raw_grain_field(rgb_uint8, frame_idx, static_w_hd, strength="med", template=None,
                    frozen_idx=0, static_floor=0.0, mode="frozen"):
    """Just the RAW additive luma grain field (no clipping/round-trip) -- for independence checks."""
    _, g = apply_grain_motion(rgb_uint8, frame_idx, static_w_hd, strength, template,
                              frozen_idx, static_floor, mode, return_grain=True)
    return g
