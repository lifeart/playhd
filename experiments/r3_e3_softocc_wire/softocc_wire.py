#!/usr/bin/env python3
"""R3-E3 -- SHIPPABLE wiring of the R2-E2 "HF-only temporal-EMA" soft-occlusion fallback.

Turns the R2-E2 finding (`experiments/r2_e2_softocc/REPORT.md`) into the exact code the lead
will land on the instant fast path. The pass REPLACES the hard `patch_high_fallback` SR-patch
with a feathered HF-EMA blend; it is **default-OFF, output-only** (operates on R[i]['recon']
AFTER derisk.reconstruct, never feeds back into R[]'s reference role -> GOTCHA #16), so OFF is
byte-identical to today.

Per non-anchor frame i (display order):
    ema_HF = beta*ema_HF + (1-beta)*(sr[i] - bic[i])         # ONE HD-float buffer; reset @ I-frame
    a      = gain * feather(R[i]['mask'], k) * conf[i]        # conf from the reactive residual
    R[i]['recon'] = (1-a)*R[i]['recon'] + a*(bic[i] + ema)   # in place; anchors untouched

Defaults are the R2-E2 recommended operating point (c): gain=0.6, beta=0.85, feather=31.

This module imports prototype/ READ-ONLY (`derisk`, `sr`). The numpy core `softocc_patch_np`
is the deterministic verified reference; `softocc_patch_torch` is its on-device twin (the form
that lands in server/anchor_sr.py), exercised end-to-end through reconstruct_torch in verify.py.
"""
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import derisk as D   # noqa: E402  decode / reconstruct / build_lr_flow / warp_lr / tof

# --------------------------------------------------------------------------- #
# Default-OFF flags (mirror INSTANT_FALLBACK_THRESH etc. in server/pipeline_api.py).
# OFF -> byte-identical to today; ON -> the R2-E2 (c) escape.
# --------------------------------------------------------------------------- #
INSTANT_SOFTOCC = False      # master switch (default OFF)
SOFTOCC_GAIN = 0.6           # (c) injection gain
SOFTOCC_BETA = 0.85          # (c) HF temporal-EMA factor (higher = smoother HF)
SOFTOCC_FEATHER = 31         # (c) Gaussian feather kernel, HD px (odd)
# confidence ramp on the reactive residual (mean-abs LR diff, 0..255); derisk's binary tau is 16,
# so this brackets the hard decision (R2-E2 CONF_LO/CONF_HI).
SOFTOCC_CONF_LO = 6.0
SOFTOCC_CONF_HI = 26.0


# --------------------------------------------------------------------------- #
# Shared helpers (identical math for numpy + torch paths)
# --------------------------------------------------------------------------- #
def feather(mask_bool, k):
    """Gaussian-feathered soft mask in [0,1] (R2-E2 _feather). mask_bool: HxW bool (numpy)."""
    if k < 3:
        return mask_bool.astype(np.float32)
    k = int(k) | 1
    return cv2.GaussianBlur(mask_bool.astype(np.float32), (k, k), 0)


def conf_lr(lr_cur, lr_ref, fx, fy, lo=SOFTOCC_CONF_LO, hi=SOFTOCC_CONF_HI):
    """Per-pixel confidence-to-use-SR at LR from the reactive residual (the SAME signal
    occlusion_mask_lr's reactive term uses). 0 = trust the warp (bicubic), 1 = need fresh SR.
    Intra holes (no MV) -> 1.0 (no warp info)."""
    pred = D.warp_lr(lr_ref, fx, fy).astype(np.float32)
    react = np.abs(lr_cur.astype(np.float32) - pred).mean(axis=2)
    c = np.clip((react - lo) / (hi - lo), 0.0, 1.0)
    c[~np.isfinite(fx)] = 1.0
    return c


def build_conf(frames, anchors, backbone, w_hd, h_hd):
    """Per non-anchor frame: HD confidence map in [0,1] (R2-E2 setup()). P frames -> residual
    confidence vs the previous backbone; B-leaves / no-MV -> ones (full-SR need in the small
    'none' fallback set). Anchors omitted (never blended)."""
    h_lr, w_lr = frames[0][1].shape[:2]
    conf = {}
    for i in range(len(frames)):
        if i in anchors:
            continue
        pt, lr_cur, mvs = frames[i]
        prev_bb = max([b for b in backbone if b < i], default=None)
        if pt == "P" and prev_bb is not None and mvs is not None and len(mvs):
            fx, fy = D.build_lr_flow(mvs, h_lr, w_lr, want="past")
            c_lr = conf_lr(lr_cur, frames[prev_bb][1], fx, fy)
            conf[i] = cv2.resize(c_lr, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
        else:
            conf[i] = np.ones((h_hd, w_hd), np.float32)
    return conf


def reset_indices(frames, extra_cuts=()):
    """Display-order indices where the HF-EMA MUST reset: every I-frame + chunk start (0) +
    any detected scene cut. A missed reset -> pre-cut HF bleeds across = cross-cut ghost."""
    r = {0}
    r |= {i for i in range(len(frames)) if frames[i][0] == "I"}
    r |= set(extra_cuts)
    return r


# --------------------------------------------------------------------------- #
# SHIPPABLE numpy core -- the verified reference. Replaces patch_high_fallback.
# --------------------------------------------------------------------------- #
def softocc_patch_np(frames, R, *, bic_provider, sr_provider, conf, anchors, reset_idx,
                     gain=SOFTOCC_GAIN, beta=SOFTOCC_BETA, feather_k=SOFTOCC_FEATHER,
                     run_set=None, enabled=INSTANT_SOFTOCC):
    """HF-EMA soft-occlusion pass (numpy reference twin of the deployed torch fn).

    In place on R[i]['recon'] (numpy uint8 HD). `enabled=False` -> no-op (byte-identical OFF).
    bic_provider(i)/sr_provider(i) -> HD float/uint8 images; conf[i] -> HD [0,1]; reset_idx ->
    frames where the EMA reinitializes. The EMA advances on EVERY frame in `run_set` (anchors
    always seed it) so the temporal HF is continuous; only non-anchor run_set frames are blended.

    `run_set=None` -> every frame (the verified default; the EMA + a 1-SR-call cost on every
    non-anchor frame). Pass a motion-gated set (e.g. anchors + {mean|MV|>1.0 frames}) to BOUND the
    per-frame SR cost: non-run frames hold the EMA and are left unblended (their occlusion mask is
    ~empty so a~=0 there -> output ~unchanged). Anchors are ALWAYS in the run set (free: reuse the
    cached SR, cv2 bicubic)."""
    info = {"enabled": bool(enabled), "blended": [], "reset_at": sorted(reset_idx),
            "n_resets": 0, "n_sr_runs": 0, "ema_rms": {}, "ema_seeded_after_reset": {}}
    if not enabled:
        return info
    ema = None
    for i in range(len(frames)):
        if i in reset_idx:
            ema = None
            info["n_resets"] += 1
        is_anchor = i in anchors or R[i].get("mask") is None
        run = is_anchor or run_set is None or i in run_set
        if run:                                        # advance EMA with this frame's fresh HF
            bic_i = bic_provider(i).astype(np.float32)
            sr_i = sr_provider(i).astype(np.float32)
            hf = sr_i - bic_i
            seeded = ema is None
            ema = hf if seeded else (beta * ema + (1.0 - beta) * hf)
            info["n_sr_runs"] += 1
            if seeded and i in reset_idx:
                info["ema_seeded_after_reset"][i] = float(np.sqrt(np.mean(ema * ema)))
        info["ema_rms"][i] = (float(np.sqrt(np.mean(ema * ema))) if ema is not None else 0.0)
        if is_anchor or not run:
            continue                                   # anchor / gated-out: recon unchanged
        m = R[i]["mask"]
        a = np.clip(gain * feather(m, feather_k) * conf[i], 0.0, 1.0)[..., None]
        recon = R[i]["recon"].astype(np.float32)
        out = (1.0 - a) * recon + a * (bic_i + ema)
        R[i]["recon"] = np.clip(out, 0, 255).astype(np.uint8)
        info["blended"].append(i)
    return info


# --------------------------------------------------------------------------- #
# On-device twin -- the form that lands in server/anchor_sr.py (verified to run end-to-end
# through reconstruct_torch in verify.py; structurally line-for-line with the numpy core, the
# standing numpy/torch-twin convention in derisk.reconstruct / reconstruct_torch).
# --------------------------------------------------------------------------- #
def softocc_patch_torch(frames, R, w_hd, h_hd, sr_mode, *, anchors, backbone, reset_idx,
                        gain=SOFTOCC_GAIN, beta=SOFTOCC_BETA, feather_k=SOFTOCC_FEATHER,
                        enabled=INSTANT_SOFTOCC):
    """HF-EMA soft-occlusion on the GPU-resident recon chain. R[i]['recon'] = [1,3,H,W] float
    device tensor; R[i]['mask'] = HxW bool device tensor (download_output=False path). In place,
    output-only. enabled=False -> no-op. Per non-anchor frame runs ONE compact-SR call (== the
    cost patch_high_fallback paid) and one HD-float EMA add; anchors reuse their cached SR to
    seed the EMA (no extra SR call)."""
    info = {"enabled": bool(enabled), "blended": [], "reset_at": sorted(reset_idx), "n_resets": 0}
    if not enabled:
        return info
    import torch
    import gpu_ops as G
    import sr as _srmod
    h_lr, w_lr = frames[0][1].shape[:2]
    dev = G.device()
    ema = None
    for i in range(len(frames)):
        if i in reset_idx:
            ema = None
            info["n_resets"] += 1
        pt, lr_cur, mvs = frames[i]
        bic_np = cv2.resize(lr_cur, (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
        sr_np = _srmod.upscale_to(lr_cur, w_hd, h_hd, model=sr_mode)
        bic_t = G.img_to_dev(bic_np)                                 # [1,3,H,W] float
        hf = G.img_to_dev(sr_np) - bic_t
        ema = hf if ema is None else (beta * ema + (1.0 - beta) * hf)
        if i in anchors or R[i].get("mask") is None:
            continue
        # feather (GaussianBlur on CPU; one HxW map) + reactive confidence -> a in [0,1].
        m = R[i]["mask"]
        m_np = m.detach().to("cpu").numpy().astype(bool) if torch.is_tensor(m) else np.asarray(m, bool)
        a_np = gain * feather(m_np, feather_k)
        prev_bb = max([b for b in backbone if b < i], default=None)
        if pt == "P" and prev_bb is not None and mvs is not None and len(mvs):
            fx, fy = D.build_lr_flow(mvs, h_lr, w_lr, want="past")
            c_lr = conf_lr(lr_cur, frames[prev_bb][1], fx, fy)
            a_np = a_np * cv2.resize(c_lr, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
        a_t = torch.from_numpy(np.ascontiguousarray(np.clip(a_np, 0.0, 1.0),
                                                    dtype=np.float32)).to(dev)[None, None]
        recon = R[i]["recon"]
        R[i]["recon"] = ((1.0 - a_t) * recon + a_t * (bic_t + ema)).clamp(0, 255)
        info["blended"].append(i)
    return info


# --------------------------------------------------------------------------- #
# Honest metrics (R2-E2 methodology, verbatim): tOF (headline) + eff-bicubic% + |dF| crosscheck.
# --------------------------------------------------------------------------- #
EPS = 1e-3


def honest_metrics(frames, out, mask, bic, srf, anchors, hole, name):
    """out/bic/srf: dict i->HD uint8 (out=post-pass output, srf=full per-frame SR, bic=bicubic);
    mask: dict i->HD bool; hole: dict i->float. Returns the R2-E2 metric row."""
    N = len(frames)
    h_lr, w_lr = frames[0][1].shape[:2]
    h_hd, w_hd = bic[0].shape[:2]
    seq = [cv2.resize(out[i], (w_lr, h_lr)) for i in range(N)]
    lr = [frames[i][1] for i in range(N)]
    tof = D.tof(seq, lr)
    s = [x.astype(np.float32) for x in seq]
    d_recon = float(np.mean([np.abs(s[t] - s[t - 1]).mean() for t in range(1, N)]))
    # fallback-localized |dF| (in-disocclusion shimmer)
    fb_d = []
    for t in range(1, N):
        mu = mask[t] | mask[t - 1]
        if mu.any():
            diff = np.abs(out[t].astype(np.float32) - out[t - 1].astype(np.float32)).mean(axis=2)
            fb_d.append(float(diff[mu].mean()))
    fb_df = float(np.mean(fb_d)) if fb_d else 0.0
    # eff-bicubic% (continuous): inside M, realized-detail ratio r=||out-bic||/||sr-bic||.
    nonanchor = [i for i in range(N) if i not in anchors]
    HW = h_hd * w_hd
    ebw, dtl = [], []
    for i in nonanchor:
        m = mask[i]
        if not m.any():
            ebw.append(0.0); dtl.append(0.0); continue
        b = bic[i].astype(np.float32); sr = srf[i].astype(np.float32); o = out[i].astype(np.float32)
        num = np.linalg.norm((o - b)[m], axis=1)
        den = np.linalg.norm((sr - b)[m], axis=1) + EPS
        r = np.clip(num / den, 0.0, 1.0)
        ebw.append(float((1.0 - r).sum()) / HW)
        dtl.append(float(r.sum()) / HW)
    eff_bic = 100.0 * float(np.mean(ebw))
    detail = 100.0 * float(np.mean(dtl))
    raw_fb = 100.0 * float(np.mean([hole[i] for i in nonanchor]))
    return dict(scheme=name, tof=round(tof, 4), eff_bicubic_pct=round(eff_bic, 3),
                detail_injected_pct=round(detail, 3), raw_fallback_pct=round(raw_fb, 3),
                fb_localized_dF=round(fb_df, 3), d_recon=round(d_recon, 3))
