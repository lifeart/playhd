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

KNOWN CAVEAT (R12 review, accepted while default-OFF): unlike the hard path, whose occlusion
mask includes the Ruder fwd-bwd round-trip term (occ_mode='full'/'adaptive'), the soft gate
weighs warps by the LR residual + intra holes ONLY -- a fwd-bwd-detected occlusion whose LR
residual is coincidentally small (revealed background matching the occluder at LR) keeps near-
full warp weight where the hard path forced fresh SR. This is the mechanism behind the measured
+5.8% jelly tOF cost; the multi-clip sweep gating default-ON should decide residual-only vs
re-adding fb (the experiment's gate_resfb variant) before any default flip.
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
    LR reliability, bilinear-upsamples to (w_hd, h_hd), then HARD-zeroes the intra-hole pixels
    at HD -- exactly matching the measured gated_recon soft path (a_hd[hole] = 0.0). The hole
    mask reproduces derisk.warp_hd's: hole = isnan(INTER_NEAREST-upsampled LR flow), so hole
    pixels fall fully to fresh SR instead of keeping feathered (~0.3-0.5) trust in an un-warped
    stale reference (R12 review: the feathered-only port was NOT the code that produced the
    GATED-GO numbers).

    PERF TODO (opt-in path only): this rebuilds build_lr_flow + one warp_lr that _warp_one just
    computed for the same (mvs, direction); plumbing fx/fy/react through would roughly halve the
    gated per-direction cost. Left as-is to keep the shipped math equal to the measured one."""
    h_lr, w_lr = lr_cur.shape[:2]
    fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want=want)
    a = reliability_lr(fx, fy, lr_cur, lr_ref, cfg)
    a_hd = cv2.resize(a, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    # HD hole == derisk.warp_hd's hole: NaN of the NEAREST-upsampled flow == NEAREST-resized LR hole.
    hole_hd = cv2.resize((~np.isfinite(fx)).astype(np.uint8), (w_hd, h_hd),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    a_hd[hole_hd] = 0.0                              # intra hole in HD -> fully fresh (gated_recon parity)
    return a_hd
