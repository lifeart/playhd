#!/usr/bin/env python3
"""
R12-E1 -- codec-RESIDUAL reliability gate (SHIPPING integration of the measured win).

Ported from experiments/r12_e1_residual_gate/patch_src/gated_recon.py (reconstruct_gated /
reliability_lr), residual-ONLY branch (use_fb=False), which is the recommended default
GateCfg(tau_res=10, s_res=5, use_fb=False) == "gate_res_t10".

WHAT IT REPLACES: the HARD per-pixel occlusion fallback in derisk.reconstruct
    recon[occ] = perframe[occ]                     # hard switch (jelly / seam source)
with a SOFT per-pixel reliability lerp
    recon = a * warped_propagation + (1 - a) * perframe_fresh
where a in [0,1] is built from the SAME cheap codec-residual signal the existing
occlusion_mask_lr already forms (|LR_cur - MV_warp(LR_ref)|), smoothed by a sigmoid:
    react = |LR_cur - warp_lr(LR_ref)|             (codec-residual approximation, codes)
    a     = sigmoid((tau_res - react) / s_res)     (1 = trust MV propagation, 0 = fresh SR)
    a     = 0 at intra holes (no MV of this direction) -> always fresh.

DEFAULT OFF (wired behind pipeline_api RESIDUAL_GATE, default False). GATED-GO verdict:
high-motion LPIPS -29% / DISTS -23%, near-no-op on calm, but carries a +5.8% jelly tOF_true
cost that needs a broader multi-clip tOF sweep before flipping default-ON (same graduation
path as beta=0.85 / deblock_pre). See experiments/r12_e1_residual_gate/REPORT.md.

READ-ONLY reuse of derisk primitives (build_lr_flow, warp_lr); derisk is NOT edited by import.
The forward-backward round-trip term (use_fb) is intentionally NOT ported -- the recommended
default is residual-only; a use_fb=True cfg is treated as residual-only here (documented).
"""
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import derisk  # noqa: E402  READ-ONLY -- reuse warp_lr / build_lr_flow


def reliability_lr(fx, fy, lr_cur, lr_ref, cfg):
    """Per-pixel reliability a in [0,1] at LR from the codec-residual approximation.
    1 = trust MV propagation, 0 = use fresh per-frame SR. a=0 at intra holes (no MV)."""
    tau = float(cfg.get("tau_res", 10.0))
    s = float(cfg.get("s_res", 5.0))
    pred = derisk.warp_lr(lr_ref, fx, fy).astype(np.float32)
    react = np.abs(lr_cur.astype(np.float32) - pred).mean(axis=2)   # HxW residual (codes)
    a = 1.0 / (1.0 + np.exp(-(tau - react) / max(s, 1e-6)))
    a[~np.isfinite(fx)] = 0.0                                       # intra hole -> fully fresh
    return a.astype(np.float32)


def reliability_hd(mvs, lr_cur, lr_ref, want, w_hd, h_hd, cfg):
    """HD reliability map for one warp direction. Builds this direction's LR flow, forms the
    LR reliability, and bilinear-upsamples to (w_hd, h_hd) -- the resize of the intra-hole zeros
    feathers reliability toward 0 near holes/borders (matches the gated_recon soft path)."""
    h_lr, w_lr = lr_cur.shape[:2]
    fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want=want)
    a = reliability_lr(fx, fy, lr_cur, lr_ref, cfg)
    return cv2.resize(a, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR).astype(np.float32)
