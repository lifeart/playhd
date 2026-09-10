#!/usr/bin/env python3
"""
R13-E2 -- FREE per-block codec CONFIDENCE signals (partition size + MV magnitude).

The H.264 motion-vector side-data this project already extracts (GOTCHA #2/#4) carries two
per-record fields the pipeline currently throws away:

  * `w`,`h`      -- the macroblock PARTITION size the encoder chose (16x16 down to 4x4).
                    A whole-MB 16x16 partition means one motion model explained the whole
                    block; sub-16 partitions mean the RD search had to SPLIT the block, i.e.
                    the motion there is not well described by a single translation.
  * `motion_x/y` -- the MV itself; its MAGNITUDE (in LR px, after /motion_scale) is the
                    distance the block travelled. Large travel = more chance the RD-optimal
                    match is "visually similar but geometrically wrong" (GOTCHA #6) and more
                    chance of a hole/occlusion behind it.

Both are FREE (already decoded, already in the structured array) and both are read at the SAME
place `build_lr_flow` reads `motion_x/motion_y/w/h/dst_x/dst_y`, so painting them costs one
extra pass over the same records.

WHAT THIS MODULE IS FOR: the reliability gate in `residual_gate.py` (R12-E1) currently trusts a
warp purely by the LR RESIDUAL `react = |LR_cur - MV_warp(LR_ref)|` (plus a hard zero at intra
holes). The residual is a DIRECT measurement of the LR warp error -- which makes these two
signals redundant wherever the residual is informative. Their only possible value is where the
residual is BLIND: a geometrically wrong MV whose LR residual is coincidentally small (the
documented mechanism behind R12-E1's +5.8% jelly tOF cost). This module supplies the prior;
`experiments/r13_e2_block_confidence/` supplies the measurement of whether it helps.

READ-ONLY w.r.t. every other prototype module: this file imports `derisk` only for its LR-flow
conventions and is imported by nobody in the shipping path (default-OFF experiment surface).
"""
import numpy as np

# --------------------------------------------------------------------------- #
# Default confidence configuration (the formula settled on in R13-E2; see REPORT.md)
# --------------------------------------------------------------------------- #
BC_DEFAULT = dict(
    # partition-size term: min(w,h) of the covering partition, in LR px.
    part_lo=4.0,        # min-side at/below which the partition term is fully distrusted
    part_hi=16.0,       # min-side at/above which the partition term is fully trusted
    # MV-magnitude term: sigmoid in LR px.
    mv_tau=4.0,         # magnitude at which the MV term crosses 0.5
    mv_s=1.5,           # softness (px)
    # floor: a low-confidence BLOCK is not proof of a bad warp (a clean fast pan is
    # large-MV and perfectly predictable), so the prior never drives trust below c_floor.
    c_floor=0.35,
    mode="tau",         # "tau" = confidence scales the residual threshold; "mul" = scales `a`
)


# --------------------------------------------------------------------------- #
# Signal painting (mirrors build_lr_flow's record loop + last-writer-wins overlap)
# --------------------------------------------------------------------------- #
def block_signals(mvs, h, w, want="all"):
    """Paint the free per-block codec signals to LR pixel maps.

    Returns (minside, mvmag, area) as float32 HxW, NaN where no MV record of the requested
    direction covers the pixel (== exactly the intra-hole set of build_lr_flow(want)).
      minside = min(partition_w, partition_h) in LR px  (16 = whole MB, 4 = finest split)
      mvmag   = |(motion_x, motion_y)| / motion_scale   in LR px
      area    = partition_w * partition_h in LR px^2    (256 = a whole 16x16 MB; agent A's term)

    The record loop, the direction filter and the block extent (dst_x/dst_y are block
    CENTERS, GOTCHA #4) are byte-for-byte the same conventions as derisk.build_lr_flow,
    so the NaN sets of the two functions coincide."""
    minside = np.full((h, w), np.nan, np.float32)
    mvmag = np.full((h, w), np.nan, np.float32)
    area = np.full((h, w), np.nan, np.float32)
    if mvs is None or len(mvs) == 0:
        return minside, mvmag, area
    for r in mvs:
        s = int(r["source"])
        if want == "past" and s >= 0:
            continue
        if want == "future" and s <= 0:
            continue
        ms = float(r["motion_scale"]) or 1.0
        dx = float(r["motion_x"]) / ms
        dy = float(r["motion_y"]) / ms
        bw, bh = int(r["w"]), int(r["h"])
        cx, cy = int(r["dst_x"]), int(r["dst_y"])
        x0, x1 = max(cx - bw // 2, 0), min(cx + bw // 2, w)
        y0, y1 = max(cy - bh // 2, 0), min(cy + bh // 2, h)
        minside[y0:y1, x0:x1] = float(min(bw, bh))
        mvmag[y0:y1, x0:x1] = float(np.hypot(dx, dy))
        area[y0:y1, x0:x1] = float(bw * bh)
    return minside, mvmag, area


# --------------------------------------------------------------------------- #
# R13-E1 (agent A) parameterisation -- THE REFERENCE FORMULATION
# --------------------------------------------------------------------------- #
# Agent A shipped the browser-side twin of this idea as an ADDITIVE threshold penalty applied
# BEFORE the sigmoid:
#
#     tau_eff = max(tau - kp*(1 - area/256) - km*min(|mv|/mvref, 1), 0)
#     a       = sigmoid((tau_eff - react) / s)
#
# with the browser config tau=16, s=3, kp=6, km=6 (area in LR px^2; 256 = a whole 16x16 MB).
# To keep the two sides ONE function (agent E diffs them), that is the reference form here and
# `tau_eff_blockpenalty` implements it verbatim. The multiplicative form below (`confidence_lr`)
# is this experiment's original parameterisation, kept because it is what phase 2 measured; both
# are monotone in the same two signals and R13-E2 measures the server-side optimum of BOTH to be
# "no block term at all" (kp = km = 0), so the server does not need to pick between them --
# it needs to not use either. See experiments/r13_e2_block_confidence/REPORT.md §2 and §7.
#
# `mvref` is NOT specified in agent A's message; it is exposed here and swept in
# phase1b_formula_ab.py (the ranking result is flat in it).
BC_BROWSER = dict(tau_res=16.0, s_res=3.0, kp=6.0, km=6.0, mvref=8.0)


def tau_eff_blockpenalty(area, mvmag, cfg=None):
    """Agent A's (R13-E1) additive block penalty on the residual threshold, verbatim.

    area  : partition area in LR px^2 (record.w * record.h), NaN at intra holes.
    mvmag : |MV| in LR px, NaN at intra holes.
    Returns tau_eff (HxW float32). NaN inputs yield tau_eff = 0 (fully distrusted), which
    combined with the caller's hole rule (a = 0 at holes) is consistent either way."""
    cfg = dict(BC_BROWSER, **(cfg or {}))
    a256 = np.clip(np.nan_to_num(area, nan=0.0) / 256.0, 0.0, 1.0)
    mm = np.nan_to_num(mvmag, nan=np.inf)
    mvr = max(float(cfg.get("mvref", 8.0)), 1e-6)
    mterm = np.minimum(mm / mvr, 1.0)
    tau_eff = float(cfg["tau_res"]) - float(cfg["kp"]) * (1.0 - a256) - float(cfg["km"]) * mterm
    return np.maximum(tau_eff, 0.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# The confidence prior (this experiment's multiplicative parameterisation)
# --------------------------------------------------------------------------- #
def confidence_lr(minside, mvmag, cfg=None):
    """Per-pixel block-confidence prior c in [c_floor, 1] from the two free signals.

        c_part = clip((minside - part_lo) / (part_hi - part_lo), 0, 1)
        c_mv   = sigmoid((mv_tau - mvmag) / mv_s)
        c      = c_floor + (1 - c_floor) * c_part * c_mv

    NaN (intra hole) -> c = 0. The gate zeroes holes anyway; returning 0 keeps the map
    self-consistent for analysis."""
    cfg = dict(BC_DEFAULT, **(cfg or {}))
    lo, hi = float(cfg["part_lo"]), float(cfg["part_hi"])
    c_part = np.clip((np.nan_to_num(minside, nan=lo) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    mm = np.nan_to_num(mvmag, nan=1e6)
    # clip the sigmoid argument: a NaN (hole) sentinel would otherwise overflow exp(); the value
    # is masked to 0 below either way, but the clip keeps the map warning-free and portable
    # (agent E ports this formula to WGSL/torch, where an inf is not equally harmless).
    z = np.clip((float(cfg["mv_tau"]) - mm) / max(float(cfg["mv_s"]), 1e-6), -60.0, 60.0)
    c_mv = 1.0 / (1.0 + np.exp(-z))
    fl = float(cfg["c_floor"])
    c = fl + (1.0 - fl) * c_part * c_mv
    c = np.where(np.isfinite(minside) & np.isfinite(mvmag), c, 0.0)
    return c.astype(np.float32)


def confidence_from_mvs(mvs, h, w, want="all", cfg=None):
    """Convenience: block_signals + confidence_lr in one call."""
    ms, mm, _ar = block_signals(mvs, h, w, want=want)
    return confidence_lr(ms, mm, cfg)


# --------------------------------------------------------------------------- #
# Gate variants -- drop-in replacements for residual_gate.reliability_hd
# --------------------------------------------------------------------------- #
def reliability_lr_bc(fx, fy, lr_cur, lr_ref, minside, mvmag, cfg, area=None):
    """Per-pixel reliability a in [0,1] combining the R12-E1 residual term with the free
    block prior. cfg keys: tau_res, s_res (residual, as residual_gate) + BC_DEFAULT keys.

      react  = mean_c |LR_cur - MV_warp(LR_ref)|                 (as R12-E1)
      c      = confidence_lr(minside, mvmag)                     (this module)
      mode "tau"     : a = sigmoid((tau_res * c - react) / s_res)  <- confidence scales the
                                                                      residual threshold
      mode "mul"     : a = c * sigmoid((tau_res - react) / s_res)
      mode "block"   : a = c                                        (no residual, no warp cost)
      mode "penalty" : a = sigmoid((tau_eff_blockpenalty(area, mvmag) - react) / s_res)
                       == AGENT A's (R13-E1) reference formulation, verbatim (needs `area`)
    a = 0 at intra holes (no MV of this direction), exactly as R12-E1."""
    import derisk  # local import: READ-ONLY reuse of warp_lr; keeps this module import-light
    cfg = dict(BC_DEFAULT, **(cfg or {}))
    mode = cfg.get("mode", "tau")
    c = confidence_lr(minside, mvmag, cfg)
    if mode == "block":
        a = c.copy()
    else:
        tau = float(cfg.get("tau_res", 10.0))
        s = float(cfg.get("s_res", 5.0))
        pred = derisk.warp_lr(lr_ref, fx, fy).astype(np.float32)
        react = np.abs(lr_cur.astype(np.float32) - pred).mean(axis=2)
        if mode == "penalty":
            if area is None:
                raise ValueError("block_confidence: mode='penalty' needs the `area` map "
                                 "(third return of block_signals)")
            tau_eff = tau_eff_blockpenalty(area, mvmag, cfg)
            a = 1.0 / (1.0 + np.exp(-np.clip((tau_eff - react) / max(s, 1e-6), -60.0, 60.0)))
        elif mode == "tau":
            a = 1.0 / (1.0 + np.exp(-(tau * c - react) / max(s, 1e-6)))
        elif mode == "mul":
            a = c * (1.0 / (1.0 + np.exp(-(tau - react) / max(s, 1e-6))))
        else:
            raise ValueError(f"block_confidence: unknown mode {mode!r}")
    a[~np.isfinite(fx)] = 0.0
    return a.astype(np.float32)


def reliability_hd_bc(mvs, lr_cur, lr_ref, want, w_hd, h_hd, cfg):
    """HD reliability map for one warp direction -- signature-compatible with
    residual_gate.reliability_hd so it can be substituted in derisk.reconstruct's gated path.
    Bilinear-upsamples the LR reliability then HARD-zeroes the HD intra holes, matching
    residual_gate.reliability_hd (and derisk.warp_hd's NEAREST hole definition) exactly."""
    import cv2
    import derisk
    h_lr, w_lr = lr_cur.shape[:2]
    fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want=want)
    minside, mvmag, area = block_signals(mvs, h_lr, w_lr, want=want)
    a = reliability_lr_bc(fx, fy, lr_cur, lr_ref, minside, mvmag, cfg, area=area)
    a_hd = cv2.resize(a, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    hole_hd = cv2.resize((~np.isfinite(fx)).astype(np.uint8), (w_hd, h_hd),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    a_hd[hole_hd] = 0.0
    return a_hd
