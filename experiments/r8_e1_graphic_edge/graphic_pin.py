#!/usr/bin/env python3
"""R8-E1 -- MOVING graphic-edge stabilization (OUTPUT-ONLY pin). Default-OFF integration support.

The shipped instant path (occ=reactive, region_aware=False, anchor-only SR) propagates the warped
anchor's HF text edges along RD-optimal codec MVs; on a MOVING high-contrast graphic those edges
wobble frame-to-frame (measured: registered-dF 1.45-4.81x the per-frame-SR floor; NOT routed to
occlusion fallback -- reactive fb 2.5-19%). Pinning the detected MOVING-graphic pixels to the
TEMPORALLY-STABLE per-frame source (instant's already-cached bicubic, or compact SR) removes the
wobble at ~zero extra SR.

CRITICAL detector difference vs exp3/graphic_detect (which gated on LOW motion to find STATIC
cards): here we require *non-zero* motion. R1-E3 settled that STATIC cards must NOT be pinned
(zero-MV skip-coding makes propagation an identity copy that already out-stabilizes per-frame SR).
So this detector fires ONLY on bimodal high-contrast regions that ALSO carry motion -> it is
byte-identical on the static USACHEV card (motion gate excludes it). bimodality stays as the
motion-INDEPENDENT false-positive guard (measured 0.00% on the talking-head face).

OUTPUT-ONLY: reads R[i]['recon'], writes a pinned COPY; never feeds back into the reference chain.
"""
import os
import sys

import cv2
import numpy as np

_PROTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prototype")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)
import region_quality as rq      # READ-ONLY: motion_mag_lr


def _luma(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)


def bimodal_score_lr(rgb_lr, white_thr=235, dark_thr=20, ksize=13):
    """Local min(frac near-white, frac near-dark) at LR (FP guard; value-distribution => resolution
    robust). High only where a neighborhood densely holds BOTH pure-bright and pure-dark (text on a
    flat field). ~0 on natural / continuous-tone content. ksize~13 at LR mirrors 51 at x4 HD."""
    y = _luma(rgb_lr)
    fb = cv2.boxFilter((y > white_thr).astype(np.float32), -1, (ksize, ksize))
    fd = cv2.boxFilter((y < dark_thr).astype(np.float32), -1, (ksize, ksize))
    return np.minimum(fb, fd)


def moving_graphic_mask_lr(rgb_lr, mvs, h_lr, w_lr,
                           bimodal_thr=0.06, motion_thr=0.6, bimodal_ksize=13, dilate=9):
    """LR bool mask of MOVING high-contrast graphic pixels.
      bimodal_thr : min local min(white-frac,dark-frac)  -> the false-positive guard (text only).
      motion_thr  : min codec |MV| (LR px/frame) to count as MOVING. >0 EXCLUDES the static card
                    (the R1-E3 NO-GO) so the pin is byte-identical there. ~0.6 px sits above codec
                    skip/quarter-pel jitter and below a real crawl (>=1.5 px).
      dilate      : grow to cover glyph bodies + a margin (LR px).
    Returns mask_lr (bool). Empty (all-False) => caller leaves recon untouched (byte-identical)."""
    bm = bimodal_score_lr(rgb_lr, ksize=bimodal_ksize)
    mag, no_mv = rq.motion_mag_lr(mvs, h_lr, w_lr, want="all")
    mag = np.where(no_mv, 0.0, mag).astype(np.float32)        # intra/skip => 0 motion => excluded
    region = (bm > bimodal_thr) & (mag > motion_thr)
    if dilate >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        region = cv2.dilate(region.astype(np.uint8), k) > 0
    return region


def upscale_mask(mask_lr, scale):
    h, w = mask_lr.shape
    return cv2.resize(mask_lr.astype(np.uint8), (w * scale, h * scale),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def apply_pin_np(recon_hd, base_hd, mask_hd):
    """OUTPUT-ONLY pin (numpy reference): recon copy with the moving-graphic pixels replaced by the
    temporally-stable per-frame `base_hd` (instant's cached bicubic, or compact SR). If mask_hd has
    no True pixels this returns an exact copy (byte-identical to recon)."""
    out = recon_hd.copy()
    if mask_hd.any():
        out[mask_hd] = base_hd[mask_hd]
    return out


def pin_graphic_torch(recon_t, base_t, mask_hd_bool, _torch=None):
    """GPU-path twin for the instant pipeline (recon is a GPU-resident [1,3,H,W] tensor; base_t is
    perframe_cache[i], the SAME-device bicubic tensor). OUTPUT-ONLY: returns a NEW tensor, never
    mutates recon_t (so it cannot enter the reference chain). No-op (returns recon_t) if empty.
    Seam: mask_hd_bool is a numpy HxW bool at HD; recon_t/base_t are [1,3,H,W] on `device`."""
    if _torch is None:
        import torch as _torch
    if not mask_hd_bool.any():
        return recon_t
    m = _torch.from_numpy(mask_hd_bool.astype("float32")).to(recon_t.device)[None, None]
    return recon_t * (1.0 - m) + base_t * m
