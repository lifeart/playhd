#!/usr/bin/env python3
"""
exp3_visual / V3 -- graphic/text-edge detector (OUTPUT-ONLY pinning support).

Detects "graphic" regions = title cards / lower-thirds / captions: near-binary text (bright
glyphs on a flat dark field, or vice-versa) with SHARP edges, BOLTED to the frame (low/zero
codec motion). These shimmer under MV warp because the propagation jitters their hard edges
frame to frame. The detector output feeds an output-only PIN (replace those pixels with the
per-frame SR, which is temporally stable on a static overlay).

Discriminator (calibrated on sample.mp4's "USACHEV TODAY" card vs the talking-head face):
  1. BIMODALITY (primary, the false-positive guard): in a local window, the fraction of
     near-WHITE pixels AND the fraction of near-DARK pixels are BOTH high -- text always puts
     pure-bright glyphs next to a pure-dark field. Natural content (a lit face, furniture) is
     mid-tone/continuous and is NEVER densely bimodal -> score ~0. This is what keeps the
     detector OFF natural high-detail content (measured: 0.00% on the talking-head face).
  2. LOW MOTION: codec MV magnitude < thr; intra/no-MV blocks treated as static (graphics are
     intra/skip-coded). Excludes a moving subject even if it had high contrast.
The edge-magnitude map (Sobel) is exported separately so the report can measure |Delta F| on
the actual hard EDGE pixels (where the warp shimmer lives).

READ-ONLY: uses region_quality.motion_mag_lr for the codec-MV magnitude. New code only.
"""
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.join(_HERE, "..", "..", "prototype")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

import region_quality as rq      # READ-ONLY: motion_mag_lr


def _luma(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)


def edge_magnitude(rgb_hd):
    """Sobel gradient magnitude of the HD luma (the hard-edge map; shimmer lives on these px)."""
    y = _luma(rgb_hd)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def bimodal_score(rgb_hd, white_thr=235, dark_thr=20, ksize=51):
    """Local min(frac near-white, frac near-dark): high only where a neighborhood holds BOTH
    pure-bright and pure-dark densely (text on a flat field). ~0 on natural / continuous tone."""
    y = _luma(rgb_hd)
    fb = cv2.boxFilter((y > white_thr).astype(np.float32), -1, (ksize, ksize))
    fd = cv2.boxFilter((y < dark_thr).astype(np.float32), -1, (ksize, ksize))
    return np.minimum(fb, fd)


def detect_graphic_mask(rgb_hd, mvs, h_lr, w_lr, scale,
                        bimodal_thr=0.06, motion_thr=0.25, grad_thr=300.0,
                        bimodal_ksize=51, dilate=15, bm=None, grad=None):
    """Per-frame graphic/text region mask (HD bool) + the hard-edge mask (HD bool).
      bimodal_thr : min local min(white-frac, dark-frac) to count as graphic (primary guard).
      motion_thr  : LR px/frame below which a pixel is 'static' (intra/no-MV => static).
      grad_thr    : Sobel-magnitude threshold defining a 'hard edge' (for the edge mask).
      dilate      : grow the region (HD px) to cover glyph bodies + a margin around edges.
    Returns (region_hd_bool, edge_hd_bool)."""
    w_hd, h_hd = w_lr * scale, h_lr * scale
    if bm is None:
        bm = bimodal_score(rgb_hd, ksize=bimodal_ksize)
    if grad is None:
        grad = edge_magnitude(rgb_hd)
    mag, no_mv = rq.motion_mag_lr(mvs, h_lr, w_lr, want="all")
    mag = np.where(no_mv, 0.0, mag).astype(np.float32)        # intra/skip => static (graphics)
    motion_hd = cv2.resize(mag, (w_hd, h_hd), interpolation=cv2.INTER_NEAREST)
    region = (bm > bimodal_thr) & (motion_hd < motion_thr)
    if dilate >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        region = cv2.dilate(region.astype(np.uint8), k) > 0
    edge = grad > grad_thr
    return region, edge
