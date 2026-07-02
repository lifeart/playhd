#!/usr/bin/env python3
"""
R12-E1 -- codec-RESIDUAL / forward-backward MV-consistency RELIABILITY GATE.

Self-contained COPY of the backbone+B propagation math from prototype/derisk.py
(reconstruct, numpy path) with ONE change: the hard per-pixel occlusion FALLBACK
    recon[occ] = perframe[occ]              # hard switch (baseline / shipping)
is replaced by a SOFT per-pixel reliability gate (CDA-VSR Residual-Map Gated Fusion)
    recon = a * warped_propagation + (1 - a) * perframe_fresh
where a in [0,1] is a per-pixel *reliability* built from the SAME cheap signals the
existing occlusion_mask_lr already forms:
    react  = |LR_cur - MV_warp(LR_ref)|            (codec-residual approximation)
    rt_err = |w~ + w^|  (Ruder fwd-bwd round-trip)  (MV consistency)
    a = sigmoid((tau_res - react)/s_res) [* sigmoid((tau_fb - rt_err)/s_fb)]
    a = 0 at intra holes (no MV) -> always fresh.

Diagnosis (research item #3): the temporal propagation is a motion-compensated IIR
filter; block-quantized codec MVs that are slightly misaligned drag HF detail across
frames -> the "jelly"/wobble + smear. A HARD threshold either fully trusts a
borderline MV (jelly) or hard-drops it (seam). The SOFT gate suppresses stale
propagated detail *proportionally* to unreliability and pulls in freshly-SR'd detail.

READ-ONLY imports from prototype/derisk.py (build_lr_flow, warp_hd, warp_lr,
_add_res, softmax_splat, occlusion_mask_lr, backbone_indices). derisk is NOT edited.

gate_mode='hard' reproduces derisk.reconstruct's numpy path EXACTLY (parity-checked
in run_gate_ab.py) so the A/B isolates only the blend. gate_mode='soft' is the gate.
"""
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))
import derisk  # noqa: E402  READ-ONLY -- reuse its warp/mask primitives


# --------------------------------------------------------------------------- #
# Gate config
# --------------------------------------------------------------------------- #
class GateCfg:
    """Soft reliability gate parameters. The residual sigmoid is centered on the
    SAME decision point as the hard threshold (tau_res == occlusion_mask_lr's
    tau_react=16.0) so the soft gate is a smoothed version of the shipping hard
    switch -- a clean, defensible A/B (only the softness/FB differ)."""
    def __init__(self, tau_res=16.0, s_res=6.0, use_fb=False,
                 tau_fb=1.5, s_fb=0.75):
        self.tau_res = float(tau_res)   # residual (codes) where reliability = 0.5
        self.s_res = float(s_res)       # residual sigmoid softness (codes)
        self.use_fb = bool(use_fb)      # also gate on fwd-bwd round-trip consistency
        self.tau_fb = float(tau_fb)     # round-trip error (px) where reliability = 0.5
        self.s_fb = float(s_fb)         # fb sigmoid softness (px)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
# Per-direction reliability (LR) -- residual + optional fwd-bwd round-trip
# --------------------------------------------------------------------------- #
def _fb_roundtrip_err(fx, fy, react):
    """Forward-backward round-trip error |w~ + w^| at LR (Ruder et al. 2016), the
    exact quantity occlusion_mask_lr's fwd-bwd branch thresholds -- here returned
    CONTINUOUS. fx,fy = backward flow (w^). A forward flow (w~) is built by softmax-
    splatting the backward MVs (collisions won by lower-residual sources), then
    sampled at the backward-mapped location. Large => disocclusion / inconsistent MV."""
    h, w = fx.shape
    fwd = np.stack([-fx, -fy], axis=-1)
    ff, _ = derisk.softmax_splat(fwd, fx, fy, -react)
    ffx = np.nan_to_num(ff[:, :, 0], nan=1e6).astype(np.float32)
    ffy = np.nan_to_num(ff[:, :, 1], nan=1e6).astype(np.float32)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sx, sy = gx + np.nan_to_num(fx), gy + np.nan_to_num(fy)
    wf_x = cv2.remap(ffx, sx, sy, cv2.INTER_LINEAR, borderValue=1e6)
    wf_y = cv2.remap(ffy, sx, sy, cv2.INTER_LINEAR, borderValue=1e6)
    wb_x, wb_y = np.nan_to_num(fx), np.nan_to_num(fy)
    rt = np.sqrt((wf_x + wb_x) ** 2 + (wf_y + wb_y) ** 2)
    # clamp the 1e6 "no source" sentinels to a large-but-finite penalty
    rt = np.minimum(rt, 1e3).astype(np.float32)
    return rt


def reliability_lr(fx, fy, lr_cur, lr_ref, cfg):
    """Per-pixel reliability a in [0,1] at LR. 1 = trust MV propagation, 0 = use fresh
    detail. Built from the codec-residual approximation (+ optional fwd-bwd). a=0 at
    intra holes (no MV of this direction)."""
    pred = derisk.warp_lr(lr_ref, fx, fy).astype(np.float32)
    react = np.abs(lr_cur.astype(np.float32) - pred).mean(axis=2)   # HxW residual (codes)
    a = _sigmoid((cfg.tau_res - react) / max(cfg.s_res, 1e-6)).astype(np.float32)
    if cfg.use_fb:
        rt = _fb_roundtrip_err(fx, fy, react)
        a = a * _sigmoid((cfg.tau_fb - rt) / max(cfg.s_fb, 1e-6)).astype(np.float32)
    a[~np.isfinite(fx)] = 0.0     # intra hole -> fully fresh
    return a


# --------------------------------------------------------------------------- #
# Per-direction warp (mirrors derisk._warp_one, numpy, no oracle) + gate signals
# --------------------------------------------------------------------------- #
def _warp_one(ref_recon, lr_cur, lr_ref, mvs, want, scale, use_residual,
              occ_mode, gate_mode, cfg, w_hd, h_hd):
    """Warp ONE reference by `want` MVs (+NEMO residual). Returns
    (recon_dir_uint8, occ_hard_hd_bool, a_reliable_hd_float|None). occ_hard is the
    exact derisk hard mask (for gate_mode='hard' parity + intra holes); a_reliable is
    the soft HD gate (gate_mode='soft')."""
    h_lr, w_lr = lr_cur.shape[:2]
    fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want=want)
    res_hd = None
    if use_residual:
        pred_lr = derisk.warp_lr(lr_ref, fx, fy)
        res = lr_cur.astype(np.float32) - pred_lr.astype(np.float32)
        res_hd = cv2.resize(res, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
    warped, hole = derisk.warp_hd(ref_recon, fx, fy, scale)
    # hard mask (identical to derisk): intra hole | reactive-residual | (full) Ruder fwd-bwd
    occ_lr, _ = derisk.occlusion_mask_lr(fx, fy, lr_cur, lr_ref, mode=occ_mode)
    occ_hard = cv2.resize(occ_lr.astype(np.uint8), (w_hd, h_hd),
                          interpolation=cv2.INTER_NEAREST).astype(bool) | hole
    a_hd = None
    if gate_mode == "soft":
        a_lr = reliability_lr(fx, fy, lr_cur, lr_ref, cfg)
        a_hd = cv2.resize(a_lr, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
        a_hd[hole] = 0.0          # intra hole in HD -> fresh
    recon_dir = derisk._add_res(warped, res_hd)
    return recon_dir, occ_hard, a_hd


# --------------------------------------------------------------------------- #
# reconstruct_gated -- numpy backbone (I/P) + B leaves, hard OR soft blend
# --------------------------------------------------------------------------- #
def reconstruct_gated(frames, scale, perframe_cache, *, use_residual=True,
                      occ_mode="full", gate_mode="hard", cfg=None, anchor_set=None):
    """Backbone+B reconstruction (numpy; no oracle -- real footage). Returns
    dict i -> recon uint8 HxWx3. gate_mode='hard' == derisk.reconstruct;
    gate_mode='soft' == per-pixel reliability-gated fusion."""
    if cfg is None:
        cfg = GateCfg()
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * scale, h_lr * scale
    N = len(frames)
    anchor_set = set(anchor_set or ())
    backbone_idx = derisk.backbone_indices(frames)

    def prev_ip(i):
        return max([b for b in backbone_idx if b < i], default=None)

    def next_ip(i):
        return min([b for b in backbone_idx if b > i], default=None)

    R = {}   # i -> dict(recon)

    # ---- PASS 1: I/P reference backbone (forward chain) ----
    for i in backbone_idx:
        pt, lr, mvs = frames[i]
        perframe = perframe_cache[i]
        p = prev_ip(i)
        is_anchor = (pt == "I") or (p is None) or (i in anchor_set)
        if is_anchor:
            R[i] = dict(recon=perframe.copy())
            continue
        recon_dir, occ_hard, a_hd = _warp_one(
            R[p]["recon"], lr, frames[p][1], mvs, "past", scale,
            use_residual, occ_mode, gate_mode, cfg, w_hd, h_hd)
        if gate_mode == "soft":
            a = a_hd[:, :, None]
            recon = np.clip(a * recon_dir.astype(np.float32)
                            + (1.0 - a) * perframe.astype(np.float32), 0, 255).astype(np.uint8)
        else:
            recon = recon_dir.copy()
            recon[occ_hard] = perframe[occ_hard]
        R[i] = dict(recon=recon)

    # ---- PASS 2: B-frame bidirectional leaves ----
    zero = np.zeros((h_hd, w_hd), bool)
    for i in range(N):
        if frames[i][0] != "B":
            continue
        pt, lr, mvs = frames[i]
        perframe = perframe_cache[i]
        p, f = prev_ip(i), next_ip(i)
        pf32 = perframe.astype(np.float32)
        wp = wf = None
        occ_p = occ_f = None
        a_p = a_f = None
        if p is not None:
            wp, occ_p, a_p = _warp_one(R[p]["recon"], lr, frames[p][1], mvs, "past",
                                       scale, use_residual, occ_mode, gate_mode, cfg, w_hd, h_hd)
        if f is not None:
            wf, occ_f, a_f = _warp_one(R[f]["recon"], lr, frames[f][1], mvs, "future",
                                       scale, use_residual, occ_mode, gate_mode, cfg, w_hd, h_hd)
        # temporal-distance weights (closer reference more reliable)
        if p is not None and f is not None:
            dp, df = (i - p), (f - i)
            t_p, t_f = df / (dp + df), dp / (dp + df)
        elif p is not None:
            t_p, t_f = 1.0, 0.0
        else:
            t_p, t_f = 0.0, 1.0

        if gate_mode == "soft":
            wsum = np.zeros((h_hd, w_hd), np.float32)
            acc = np.zeros((h_hd, w_hd, 3), np.float32)
            if wp is not None:
                w_p = (t_p * a_p).astype(np.float32)
                acc += w_p[:, :, None] * wp.astype(np.float32)
                wsum += w_p
            if wf is not None:
                w_f = (t_f * a_f).astype(np.float32)
                acc += w_f[:, :, None] * wf.astype(np.float32)
                wsum += w_f
            wsum = np.clip(wsum, 0.0, 1.0)                    # reliability budget in [0,1]
            acc += (1.0 - wsum)[:, :, None] * pf32           # remainder -> fresh per-frame
            recon = np.clip(acc, 0, 255).astype(np.uint8)
        else:
            valid_p = (~occ_p) if occ_p is not None else zero
            valid_f = (~occ_f) if occ_f is not None else zero
            if p is not None and f is not None:
                a_pw, a_fw = t_p, t_f
            elif p is not None:
                a_pw, a_fw = 1.0, 0.0
            else:
                a_pw, a_fw = 0.0, 1.0
            both = valid_p & valid_f
            only_p = valid_p & ~valid_f
            only_f = valid_f & ~valid_p
            r = pf32.copy()
            if wp is not None:
                r[only_p] = wp.astype(np.float32)[only_p]
            if wf is not None:
                r[only_f] = wf.astype(np.float32)[only_f]
            if wp is not None and wf is not None:
                r[both] = (a_pw * wp.astype(np.float32) + a_fw * wf.astype(np.float32))[both]
            recon = np.clip(r, 0, 255).astype(np.uint8)
        R[i] = dict(recon=recon)

    return {i: R[i]["recon"] for i in range(N)}
