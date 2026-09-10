#!/usr/bin/env python3
"""
R13-E3 -- TEMPORAL ("jelly"/"желе") METRICS.  The measurement half of the round.

The project tabled the jelly item partly because nobody produced a number: tOF (Farneback
flow deviation) is the only temporal metric in the repo and it is blind to same-flow
high-frequency wobble.  Two metrics are added here, both computed with tools already in
the repo (LPIPS + the codec MVs), no optical-flow / RAFT budget:

1. tLP  -- TecoGAN's perceptual temporal metric (the sibling of tOF):
       tLP_t = | LPIPS(out_{t-1}, out_t) - LPIPS(ref_{t-1}, ref_t) |
   i.e. the EXCESS perceptual change between adjacent output frames over the perceptual
   change the true content actually made.  `ref` = bicubic(decoded LR) -- the honest
   real-footage stand-in for content motion (there is no HD GT; same convention as
   R12-E1's tOF_true).  Wobble that keeps the flow field intact (HF boiling under a
   correct MV) shows up here and NOT in tOF.

2. MCLF -- Motion-Compensated Luma Flicker, warping the previous OUTPUT frame's luma into
   the current frame's grid with the EXISTING codec MVs:
       MCLF_t = mean_valid | Y_t - warp(Y_{t-1}, flow_{t->t-1}) |      (codes, 0..255)
   flow_{t->t-1} is built from codec MVs only (see mc_flow_prev): a frame whose display
   predecessor IS its reference anchor uses its own MV field directly; otherwise the field
   is COMPOSED with the splat-inverted MV field of t-1 (both reference the same anchor),
   reusing derisk.softmax_splat -- the same inversion the Ruder fwd-bwd occlusion test uses.

   HONESTY NOTE (required by the round brief): MCLF is NOT zero for ground-truth video --
   codec MVs are rate-distortion vectors, so a warp residual exists for any content.  It is
   used ONLY as a relative A/B between sequences measured with the SAME flow fields, and is
   always reported next to two calibrating sequences:
       bicubic(LR)      = the content+codec floor (no SR, no propagation)
       per-frame x4plus = the pure spatial-SR flicker level (no propagation at all).

3. drift_vs_flicker() -- the diagnostic the round exists for.  Any per-frame series is
   regressed against FRAMES-SINCE-ANCHOR:
       slope > 0 and dominating       -> DRIFT   (error grows with distance from the anchor)
       slope ~ 0, detrended std large -> FLICKER (frame-to-frame oscillation)
   Reported as (slope per frame, detrended std, R^2, mean-at-d curve).

All read-only w.r.t. prototype/ and server/.
"""
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from common import SCALE  # noqa: E402
import derisk  # noqa: E402  READ-ONLY (warp_hd / build_lr_flow / softmax_splat)
import metrics as M  # noqa: E402  READ-ONLY (R5-E2 LPIPS)

_INVALID = 1e6


def luma(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


# --------------------------------------------------------------------------- #
# flow from the current frame's grid to the DISPLAY-PREVIOUS frame's grid
# --------------------------------------------------------------------------- #
def _past_flow(frames, i):
    h_lr, w_lr = frames[i][1].shape[:2]
    return derisk.build_lr_flow(frames[i][2], h_lr, w_lr, want="past")


def _prev_backbone(backbone, i):
    return max([b for b in backbone if b < i], default=None)


def _flow_to(frames, i, want):
    h_lr, w_lr = frames[i][1].shape[:2]
    return derisk.build_lr_flow(frames[i][2], h_lr, w_lr, want=want)


def _compose_through(frames, i, j, ref, want):
    """flow grid_i -> grid_j, composed through the shared reference frame `ref`:
        flow(x) = f_i(x) + inv(f_j)(x + f_i(x))
    f_i, f_j are the codec MV fields of i and j in direction `want` (both point at `ref`);
    inv(f_j) is the splat-INVERSION of f_j (derisk.softmax_splat, collisions won by the
    lower LR residual -- the same machinery the Ruder fwd-bwd occlusion test uses)."""
    if i == ref:      # degenerate: i IS the shared reference -> f_i = 0 (pure inversion of f_j)
        h_lr, w_lr = frames[i][1].shape[:2]
        fx = np.zeros((h_lr, w_lr), np.float32)
        fy = np.zeros((h_lr, w_lr), np.float32)
    else:
        fx, fy = _flow_to(frames, i, want)
    gx, gy = _flow_to(frames, j, want)
    if not np.isfinite(fx).any() or not np.isfinite(gx).any():
        return None, None, None
    pred = derisk.warp_lr(frames[ref][1], gx, gy).astype(np.float32)
    react = np.abs(frames[j][1].astype(np.float32) - pred).mean(axis=2)
    inv, _ = derisk.softmax_splat(np.stack([-gx, -gy], axis=-1), gx, gy, -react)
    ivx = np.nan_to_num(inv[:, :, 0], nan=_INVALID).astype(np.float32)
    ivy = np.nan_to_num(inv[:, :, 1], nan=_INVALID).astype(np.float32)
    h, w = fx.shape
    mgx, mgy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sx, sy = mgx + np.nan_to_num(fx), mgy + np.nan_to_num(fy)
    ax = cv2.remap(ivx, sx, sy, cv2.INTER_LINEAR, borderValue=_INVALID)
    ay = cv2.remap(ivy, sx, sy, cv2.INTER_LINEAR, borderValue=_INVALID)
    valid = (np.isfinite(fx) & np.isfinite(fy)
             & (np.abs(ax) < 1e3) & (np.abs(ay) < 1e3))
    return (fx + ax).astype(np.float32), (fy + ay).astype(np.float32), valid


def mc_flow_prev(frames, i, backbone=None):
    """Backward flow from frame i's grid to frame (i-1)'s grid, from CODEC MVs ONLY.
    Returns (fx, fy, valid) at LR; NaN/False where no codec MV chain covers the pixel.

    Case A (exact): frame i-1 IS i's past reference anchor -> i's own past-MV field.
    Case B (composed, past chain): i and i-1 share the past anchor p -> compose through p.
    Case C (composed, future chain): i and i-1 share the next backbone anchor f -> compose
        through f.  B-frames on this stream are FUTURE-DOMINANT (Step 1), so the past
        chain alone covers only ~10-50% of a leaf; the union of both chains is what makes
        the metric cover most of the frame.  Past wins where both are valid.
    Otherwise -> all-invalid (never guessed).
    """
    if i <= 0:
        return None, None, None
    if backbone is None:
        backbone = derisk.backbone_indices(frames)
    p = _prev_backbone(backbone, i)
    if p is not None and i - 1 == p:                   # case A -- exact
        fx, fy = _past_flow(frames, i)
        return fx, fy, np.isfinite(fx) & np.isfinite(fy)

    cands = []
    if p is not None and _prev_backbone(backbone, i - 1) == p:
        cands.append(_compose_through(frames, i, i - 1, p, "past"))
    nxt = min([b for b in backbone if b > i], default=None)
    nxt_prev = min([b for b in backbone if b > i - 1], default=None)
    if nxt_prev == i:
        # case D: frame i IS i-1's FUTURE anchor (a P right after its B leaves) -> the
        # flow is the pure splat-inversion of i-1's future field. This is the mirror of
        # case A and is what keeps P-frame pairs from being measured on ~10% of pixels.
        cands.append(_compose_through(frames, i, i - 1, i, "future"))
    elif nxt is not None and nxt == nxt_prev:
        cands.append(_compose_through(frames, i, i - 1, nxt, "future"))
    cands = [c for c in cands if c[0] is not None]
    if not cands:
        return None, None, None
    ox = np.full_like(cands[0][0], np.nan)
    oy = np.full_like(cands[0][1], np.nan)
    valid = np.zeros(ox.shape, bool)
    for cx, cy, cv_ in reversed(cands):                # first candidate (past) wins
        take = cv_ & np.isfinite(cx) & np.isfinite(cy)
        ox[take], oy[take] = cx[take], cy[take]
        valid |= take
    return ox, oy, valid


def build_mc_flows(frames, scale=SCALE):
    """Precompute, once per window, the HD warp maps + validity for every adjacent pair.
    Returned list is indexed by t (entry 0 is None). Shared by every variant so the A/B
    is measured with IDENTICAL flow fields."""
    backbone = derisk.backbone_indices(frames)
    h_lr, w_lr = frames[0][1].shape[:2]
    out = [None]
    for i in range(1, len(frames)):
        fx, fy, valid = mc_flow_prev(frames, i, backbone)
        if fx is None:
            out.append(None)
            continue
        vhd = cv2.resize(valid.astype(np.uint8), (w_lr * scale, h_lr * scale),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append(dict(fx=fx, fy=fy, valid_hd=vhd,
                        cover=float(valid.mean())))
    return out


def mclf_series(seq, flows, scale=SCALE):
    """Per-frame motion-compensated luma flicker (codes). seq = HD uint8 RGB frames."""
    vals = [np.nan]
    for t in range(1, len(seq)):
        fl = flows[t]
        if fl is None:
            vals.append(np.nan)
            continue
        warped, hole = derisk.warp_hd(seq[t - 1], fl["fx"], fl["fy"], scale)
        m = fl["valid_hd"] & ~hole
        if m.sum() < 1000:
            vals.append(np.nan)
            continue
        d = np.abs(luma(seq[t]) - luma(warped))
        vals.append(float(d[m].mean()))
    return vals


def _mw(img, width=768):
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def adjacent_lpips(seq, width=768):
    """LPIPS between adjacent frames (the perceptual 'how much did the picture change')."""
    return [np.nan] + [M.lpips_dist(_mw(seq[t - 1], width), _mw(seq[t], width))
                       for t in range(1, len(seq))]


def tlp_series(seq_lp, ref_lp):
    """TecoGAN tLP from two adjacent-LPIPS series (candidate, content reference)."""
    return [abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan
            for a, b in zip(seq_lp, ref_lp)]


def dF_series(seq):
    """Raw (NOT motion-compensated) adjacent luma difference -- context only."""
    return [np.nan] + [float(np.mean(np.abs(luma(seq[t]) - luma(seq[t - 1]))))
                       for t in range(1, len(seq))]


# --------------------------------------------------------------------------- #
# the diagnostic: drift (grows with distance from anchor) vs flicker (oscillates)
# --------------------------------------------------------------------------- #
def frames_since_anchor(frames):
    backbone = derisk.backbone_indices(frames)
    anchors = [i for i in range(len(frames)) if frames[i][0] == "I" or i == 0]
    d = []
    for i in range(len(frames)):
        a = max([x for x in anchors if x <= i], default=0)
        d.append(i - a)
    return d, backbone, anchors


def drift_vs_flicker(series, dist, dmin=1):
    """Regress a per-frame series against frames-since-anchor.
    Returns slope (per frame), intercept, R^2, detrended std, mean-at-d curve."""
    x, y = [], []
    for s, d in zip(series, dist):
        if np.isfinite(s) and d >= dmin:
            x.append(d)
            y.append(s)
    if len(x) < 4:
        return dict(slope=float("nan"), intercept=float("nan"), r2=float("nan"),
                    detrended_std=float("nan"), curve={}, n=len(x))
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.polyfit(x, y, 1)
    fit = np.polyval(A, x)
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    curve = {int(d): float(np.mean(y[x == d])) for d in sorted(set(x.tolist()))}
    return dict(slope=float(A[0]), intercept=float(A[1]),
                r2=(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                detrended_std=float(np.std(y - fit)), mean=float(y.mean()),
                curve=curve, n=int(len(x)))
