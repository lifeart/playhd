#!/usr/bin/env python3
"""
exp3_visual / common.py -- shared READ-ONLY helpers for E3 (V2 grain, V3 pinning).

Imports prototype/ + server/ READ-ONLY. No shared file is modified. Everything here is
either a thin call into derisk/region_quality/grain or a new metric used only by E3.

Honest metrics:
  * dframe_luma(seq, mask)  -- direct |Delta F| on the LUMA channel, restricted to a mask.
                              THIS is the flicker number (mean abs frame-to-frame luma diff).
  * derisk.tof(seq, ref)    -- TecoGAN tOF (Farneback-flow EPE vs a reference), temporal
                              stability. Reused verbatim from derisk (no re-derivation).
"""
import os
import sys
import gc

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.join(_HERE, "..", "..", "prototype")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

import derisk as d                 # READ-ONLY
import region_quality as rq        # READ-ONLY
import grain as _grain             # READ-ONLY

SAMPLE = os.path.join(_HERE, "..", "..", "sample.mp4")


def free_gpu():
    """Free the shared MPS GPU between configs (3 sibling experiments contend for it)."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:                       # never silently swallow -- log + continue
        print(f"  [free_gpu] note: {e}")


def luma(rgb):
    """BT.601 luma (matches cv2 RGB2YCrCb Y, the channel grain is added to)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)


def dframe_luma(seq, mask=None):
    """Mean |Delta F| on luma over consecutive frames, restricted to `mask` (HxW bool) if given.
    THE direct flicker metric: how much the pixels change frame-to-frame. Lower = steadier."""
    vals = []
    prev = luma(seq[0])
    for t in range(1, len(seq)):
        cur = luma(seq[t])
        diff = np.abs(cur - prev)
        vals.append(float(diff[mask].mean()) if mask is not None else float(diff.mean()))
        prev = cur
    return float(np.mean(vals)) if vals else float("nan")


def tof_lr(seq, ref_lr):
    """derisk.tof at LR: downscale the (HD) candidate sequence to LR and compare its Farneback
    flow to the decoded-LR flow (the cleanest motion truth on real footage). Reused verbatim."""
    h_lr, w_lr = ref_lr[0].shape[:2]
    seq_lr = [cv2.resize(r, (w_lr, h_lr)) for r in seq]
    return d.tof(seq_lr, ref_lr)


def decode_window(start, n):
    frames = d.decode_lr_and_mvs(SAMPLE, start, n)
    h_lr, w_lr = frames[0][1].shape[:2]
    types = "".join(f[0][0] for f in frames)
    return frames, h_lr, w_lr, types


def reconstruct_window(frames, scale, sr_mode="realesrgan", occ="full"):
    """Build the SR cache + propagated recon for a window. Returns (R, perframe_cache).
    numpy backend = deterministic (contention-robust). anchor_set=set() => I-frames only."""
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * scale, h_lr * scale
    perframe_cache = d.build_perframe_cache(frames, w_hd, h_hd, sr_mode)
    _, R = d.reconstruct(frames, None, scale, True, occ, perframe_cache, set(),
                         backend="numpy", collect_metrics=False, download_output=True)
    return R, perframe_cache


def amplified_diff(a, b, amp=6):
    """Frame-to-frame difference, amplified + JET-colormapped, for a visual flicker artifact."""
    dd = cv2.absdiff(a, b).max(axis=2)
    return cv2.applyColorMap(np.clip(dd.astype(np.float32) * amp, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_JET)


def label(img_bgr, text):
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out
