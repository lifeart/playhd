#!/usr/bin/env python3
"""
R13-E3 -- the two ported mechanisms, as an OUTPUT-ONLY post-pass over a finished
reconstruction sequence.

    M1  FSR3.1 LUMA INSTABILITY FACTOR  (engine-free reimplementation)
        Source: AMD FidelityFX SDK, MIT, `ffx_fsr3upscaler_luma_instability.h`
        (`ComputeLumaInstabilityFactor`).  A 4-deep REPROJECTED luma history per pixel;
        the pixel is flagged unstable when the current luma deviates from L[n-1] in the
        same direction as it deviates from some OLDER L[i] *and* is closer (more similar)
        to that older frame -- i.e. the value keeps returning to where it was = temporal
        OSCILLATION, not real change.  Depth-free in the body (`fFarthestDepthInMeters`
        is passed in and never read), which is why it ports to decoded video at all.

        In FSR the factor's ONLY use (ffx_fsr3upscaler_accumulate.h::RectifyHistory) is
            fHistoryContribution = max(instability, lock) * accumulation * (1-disocclusion)
            fHistoryColor = lerp(clampedHistory, rawHistory, sat(fHistoryContribution))
        = "let the history through the rectification clamp UNRECTIFIED where the pixel
        oscillates".  playhd has NO colour box (R1-E4b: structural NO-GO -- and this is
        NOT one: an oscillation detector is a TEMPORAL test, not a colour-identity test),
        so the equivalent action here is the same *direction*: where the pixel oscillates,
        raise the weight of the (motion-compensated) history instead of taking the fresh
        per-frame value.

    M2  PER-PIXEL MOTION-ADAPTIVE EMA
        alpha = alpha_max * (1 - smoothstep(err_lo, err_hi, err)), where err is the
        per-pixel error between the WARPED PREVIOUS OUTPUT and the CURRENT INPUT, measured
        at LR (downscale the warped output, compare to the decoded LR frame = the only
        real ground truth on real footage) and bilinearly re-upsampled.  Low warp error ->
        full history injection; misaligned / occluded -> none.  This is deliberately a
        DIFFERENT signal from R12-E1's residual gate (which measures a SINGLE-hop
        LR_cur vs warp(LR_ref) codec residual): here the history is the accumulated
        OUTPUT, so the error also sees drift the codec residual cannot.

        NOTE vs R9-E1 (per-CLIP adaptive beta, refuted): this is PER-PIXEL and composes
        with, rather than replaces, the fixed global beta=0.85 anchor blend.

HISTORY CHAIN.  The history advances along the I/P BACKBONE (GOTCHA #12: never chain in
display order once B-frames appear).  For frame i the history is the stabilised output at
i's past backbone reference p, warped into i's grid with i's own past-MV field -- the warp
the pipeline already performs (`derisk.warp_hd`), so the deployable extra cost is one
abs-diff + one smoothstep + one 4-channel warp.  B leaves are stabilised but do NOT update
the history (they are never references).  Anchors (I-frames / forced anchors) RESET the
history -- the R3-E3/GOTCHA lesson that an EMA which survives a cut paints a cross-cut ghost.

OUTPUT-ONLY (GOTCHA #16): the stabilised frame never re-enters the propagation R[] chain;
the stabiliser keeps its OWN history buffer.  `STAB_ENABLED=False` at module level -> a
disabled stabiliser returns the input frames untouched (byte-identical).
"""
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import derisk  # noqa: E402  READ-ONLY

# Module-level default: OFF. Any graduation of this lever must keep this False so the
# shipping pipeline stays byte-identical unless a caller explicitly opts in.
STAB_ENABLED = False


class StabCfg:
    """Stabiliser configuration.

    mode:
      'off'          -> passthrough (byte-identical)
      'ema_fixed'    -> constant-alpha motion-compensated EMA (the naive straw man that
                        the project's tabled 'жele' complaint is about)
      'ema_adaptive' -> M2 only (per-pixel motion-adaptive alpha)
      'instab_only'  -> M1 only (history injected ONLY where the luma oscillates)
      'ema_instab'   -> M2 + M1 (adaptive alpha, raised on oscillating pixels)
    """

    def __init__(self, mode="off", alpha_max=0.5, err_lo=2.0, err_hi=12.0,
                 alpha_instab=0.7, vel_max_4k=20.0, use_occ_gate=True,
                 instab_min_depth=4):
        self.mode = mode
        self.alpha_max = float(alpha_max)      # history weight at zero warp error
        self.err_lo = float(err_lo)            # LR codes: below -> full injection
        self.err_hi = float(err_hi)            # LR codes: above -> no injection
        self.alpha_instab = float(alpha_instab)  # history weight on oscillating pixels
        self.vel_max_4k = float(vel_max_4k)    # FSR's Get4KVelocity/20 gate (4K px/frame)
        self.use_occ_gate = bool(use_occ_gate)  # (1-disocclusion) FSR gate <- occlusion mask
        self.instab_min_depth = int(instab_min_depth)  # need a full 4-deep history

    def __repr__(self):
        return (f"StabCfg({self.mode}, a={self.alpha_max}, err=({self.err_lo},{self.err_hi}), "
                f"a_inst={self.alpha_instab}, vel4k={self.vel_max_4k})")


# --------------------------------------------------------------------------- #
# M1 -- FSR3.1 luma instability factor (faithful, vectorised, engine-free)
# --------------------------------------------------------------------------- #
def _min_div_max(a, b, on_zero):
    m = np.maximum(a, b)
    out = np.full_like(a, on_zero, dtype=np.float32)
    nz = m != 0
    out[nz] = (np.minimum(a, b)[nz] / m[nz]).astype(np.float32)
    return out


def luma_instability_factor(cur_luma, hist4, depth):
    """Port of ComputeLumaInstabilityFactor (FSR3.1, MIT).

    cur_luma : HxW float32 -- the current frame's luma (pre-accumulation sample).
    hist4    : HxWx4 float32 -- REPROJECTED luma history, [...,0]=n-1 ... [...,3]=n-4.
    depth    : how many frames the history actually holds (FSR uses "L[n-4] != 0";
               an explicit counter is the same test without the magic zero).
    Returns  : HxW float32 in {0,1}.
    """
    d0 = cur_luma - hist4[:, :, 0]
    sim0 = _min_div_max(cur_luma, hist4[:, :, 0], 1.0)
    max_sim = sim0.copy()
    s0 = np.sign(d0)
    for i in (1, 2, 3):
        d1 = cur_luma - hist4[:, :, i]
        sim1 = _min_div_max(cur_luma, hist4[:, :, i], 0.0)
        same = np.sign(d1) == s0
        max_sim = np.where(same, np.maximum(max_sim, sim1), max_sim)
    inst = ((sim0 < 1.0) & (max_sim > sim0)).astype(np.float32)
    if depth < 4:
        inst[:] = 0.0
    return inst


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------- #
# the stabiliser
# --------------------------------------------------------------------------- #
def stabilize(frames, recon, cfg, scale=3, anchors=None, masks=None, enabled=None,
              collect=False):
    """Output-only temporal stabiliser over a finished reconstruction.

    frames : the decoded (type, lr, mvs[, qp]) list (READ-ONLY).
    recon  : dict/list i -> HD uint8 RGB reconstruction (derisk.reconstruct output).
    masks  : optional dict i -> HD bool occlusion mask (R[i]['mask']) for the FSR
             (1 - disocclusion) gate.
    Returns (out_list, stats). With STAB_ENABLED False and enabled=None, or cfg.mode
    'off', the input frames are returned UNCHANGED (byte-identical).
    """
    on = STAB_ENABLED if enabled is None else bool(enabled)
    N = len(frames)
    seq = [np.asarray(recon[i]) for i in range(N)]
    if not on or cfg.mode == "off":
        return seq, {"mode": "off", "touched": 0}

    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * scale, h_lr * scale
    backbone = derisk.backbone_indices(frames)
    anchors = set(anchors or [])
    anchors |= {i for i in range(N) if frames[i][0] == "I"}
    anchors.add(0)
    # FSR's velocity gate is expressed in 4K pixels/frame; rescale to this pipeline's HD.
    vel_max_lr = cfg.vel_max_4k * (h_hd / 2160.0) / scale

    hist_rgb = None     # HD float32 stabilised output at the previous backbone frame
    hist_luma = None    # HD float32 HxW x4 reprojected luma history (backbone chain)
    depth = 0
    out = [None] * N
    hist_at = {}        # backbone index -> (stabilised output float32, luma history, depth)
    stats = dict(mode=cfg.mode, touched=0, alpha_mean=[], instab_frac=[], frames=[])

    def _blend(i, hist_w, hole, lum_w, hdepth):
        """Common per-frame blend. Returns (out_uint8, alpha_mean, instab_frac)."""
        cur = seq[i].astype(np.float32)
        # M2: per-pixel warp error, warped PREVIOUS OUTPUT vs CURRENT INPUT (at LR --
        # the decoded LR frame is the only ground truth on real footage).
        hw_lr = cv2.resize(hist_w, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
        err_lr = np.abs(hw_lr - frames[i][1].astype(np.float32)).mean(axis=2)
        err_hd = cv2.resize(err_lr, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
        if cfg.mode == "ema_fixed":
            alpha = np.full((h_hd, w_hd), cfg.alpha_max, np.float32)
        elif cfg.mode == "instab_only":
            alpha = np.zeros((h_hd, w_hd), np.float32)
        else:
            alpha = cfg.alpha_max * (1.0 - _smoothstep(err_hd, cfg.err_lo, cfg.err_hi))
        inst_frac = 0.0
        if cfg.mode in ("instab_only", "ema_instab") and lum_w is not None:
            cur_luma = cv2.cvtColor(seq[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
            inst = luma_instability_factor(cur_luma, lum_w, hdepth)
            fxi, fyi = derisk.build_lr_flow(frames[i][2], h_lr, w_lr, want="all")
            mag = np.sqrt(np.nan_to_num(fxi) ** 2 + np.nan_to_num(fyi) ** 2)
            vel_w = 1.0 - np.clip(cv2.resize(mag, (w_hd, h_hd),
                                             interpolation=cv2.INTER_LINEAR) / vel_max_lr,
                                  0.0, 1.0)
            inst = inst * vel_w                              # FSR's Get4KVelocity/20 gate
            if cfg.use_occ_gate and masks is not None and masks.get(i) is not None:
                inst = inst * (~np.asarray(masks[i], bool)).astype(np.float32)  # 1-disocclusion
            inst[hole] = 0.0
            inst_frac = float((inst > 0.5).mean())
            alpha = np.maximum(alpha, cfg.alpha_instab * inst)
        alpha[hole] = 0.0                                    # no history where there is none
        blended = (1.0 - alpha)[:, :, None] * cur + alpha[:, :, None] * hist_w
        return np.clip(blended, 0, 255).astype(np.uint8), float(alpha.mean()), inst_frac

    # ---- PASS 1: the I/P backbone chain (history advances here only) ----
    for i in backbone:
        if i in anchors or hist_rgb is None:
            out[i] = seq[i].copy()
            hist_rgb = seq[i].astype(np.float32)
            hist_luma = np.repeat(cv2.cvtColor(seq[i], cv2.COLOR_RGB2GRAY
                                               ).astype(np.float32)[:, :, None], 4, axis=2)
            depth = 1
            hist_at[i] = (hist_rgb, hist_luma, depth)
            stats["frames"].append(dict(i=i, anchor=True, alpha=0.0, instab=0.0))
            continue
        fx, fy = derisk.build_lr_flow(frames[i][2], h_lr, w_lr, want="past")
        if not np.isfinite(fx).any():
            out[i] = seq[i].copy()
            hist_at[i] = (hist_rgb, hist_luma, depth)
            stats["frames"].append(dict(i=i, anchor=False, alpha=0.0, instab=0.0,
                                        skipped="no-past-flow"))
            continue
        hist_w, hole = derisk.warp_hd(hist_rgb, fx, fy, scale)
        lum_w, _ = derisk.warp_hd(hist_luma, fx, fy, scale)   # cv2.remap handles 4 channels
        o, am, inf = _blend(i, hist_w, hole, lum_w, depth)
        out[i] = o
        stats["touched"] += 1
        stats["alpha_mean"].append(am)
        stats["instab_frac"].append(inf)
        stats["frames"].append(dict(i=i, anchor=False, alpha=am, instab=inf))
        hist_rgb = o.astype(np.float32)
        new_l = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hist_luma = np.dstack([new_l, lum_w[:, :, 0], lum_w[:, :, 1], lum_w[:, :, 2]])
        depth = min(depth + 1, 4)
        hist_at[i] = (hist_rgb, hist_luma, depth)

    # ---- PASS 2: B leaves -- BIDIRECTIONAL history (never update the chain) ----
    # On this stream B-frames are FUTURE-dominant (handoff Step 1), so a past-only history
    # would have no MV over ~50-80% of a leaf. Mirror derisk's own B handling: pull the
    # history from both the past and the future backbone anchor, temporal-distance weighted.
    for i in range(N):
        if out[i] is not None:
            continue
        p = max([b for b in backbone if b < i], default=None)
        f = min([b for b in backbone if b > i], default=None)
        acc = np.zeros((h_hd, w_hd, 3), np.float32)
        wsum = np.zeros((h_hd, w_hd), np.float32)
        lum_w = None
        for ref, want in ((p, "past"), (f, "future")):
            if ref is None or ref not in hist_at:
                continue
            fx, fy = derisk.build_lr_flow(frames[i][2], h_lr, w_lr, want=want)
            if not np.isfinite(fx).any():
                continue
            hw, hl = derisk.warp_hd(hist_at[ref][0], fx, fy, scale)
            ok = (~hl).astype(np.float32)
            t = (f - i) / ((i - p) + (f - i)) if (p is not None and f is not None) else 1.0
            t = t if want == "past" else (1.0 - t if (p is not None and f is not None) else 1.0)
            acc += (t * ok)[:, :, None] * hw
            wsum += t * ok
            if want == "past":
                lum_w, _ = derisk.warp_hd(hist_at[ref][1], fx, fy, scale)
        hole = wsum <= 1e-6
        hist_w = np.zeros_like(acc)
        nz = ~hole
        hist_w[nz] = acc[nz] / wsum[nz][:, None]
        hdepth = hist_at[p][2] if (p is not None and p in hist_at) else 0
        o, am, inf = _blend(i, hist_w, hole, lum_w, hdepth)
        out[i] = o
        stats["touched"] += 1
        stats["alpha_mean"].append(am)
        stats["instab_frac"].append(inf)
        stats["frames"].append(dict(i=i, anchor=False, alpha=am, instab=inf, leaf=True))

    stats["alpha_mean_all"] = float(np.mean(stats["alpha_mean"])) if stats["alpha_mean"] else 0.0
    stats["instab_frac_all"] = float(np.mean(stats["instab_frac"])) if stats["instab_frac"] else 0.0
    if not collect:
        stats.pop("frames", None)
    return out, stats
