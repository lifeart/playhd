#!/usr/bin/env python3
"""
R13-E4 calibration: give the FSR2 temporal-MV-divergence term its BEST SHOT before
judging it (the round's burden of proof is on deletion, so the challenger must be
tuned, not strawmanned).

Sweeps, per window, per warp direction:
  * history definition -- what "last frame's MV field" means for codec MVs:
      ref       : the REFERENCE frame's own backward flow (default; the surface's
                  motion state in the frame we pull history from)
      ref-raw   : same, WITHOUT temporal-distance normalisation (literal FSR2)
      disp      : the previous DISPLAY frame's flow (want='all')
  * pxDistance gate: FSR2's saturate((pxDist/20)^3) with the 20 expressed in LR px
      gate_lr in {0.5, 1, 2, 5, 10}   (5 == FSR2's literal 20 DISPLAY px at scale 4)
      plus 'off' (ungated 1 - |mv_prev|/|mv_cur|)
  * nan_prev handling: flag | skip

Scored by AUC against BOTH ground truths (see run_diag.py), plus, at the threshold
that matches the Ruder term's own extra mask area, recall/precision -- an
area-matched comparison so "bigger mask" cannot masquerade as "better mask".
"""
import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import harness as H          # noqa: E402
import fsr2_mask as F        # noqa: E402
import derisk as D           # noqa: E402
from run_diag import auc, prf, hd_to_lr   # noqa: E402

HISTS = ["ref", "ref-raw", "disp"]
GATES = [("off", None), ("g0.5", 0.5), ("g1", 1.0), ("g2", 2.0), ("g5", 5.0), ("g10", 10.0)]
DILATE = [False, True]      # FSR2 feeds the op DILATED MVs (see fsr2_mask.dilate_mv)


def ruder_margin(fx, fy, lr, lr_ref, tau=16.0):
    """The Ruder test as a CONTINUOUS score (lhs - rhs; >0 == flagged), so its
    signal can be AUC'd on the same footing as FSR2's continuous divergence."""
    h, w = fx.shape
    pred = D.warp_lr(lr_ref, fx, fy).astype(np.float32)
    react = np.abs(lr.astype(np.float32) - pred).mean(axis=2)
    fwd = np.stack([-fx, -fy], axis=-1)
    ff, _ = D.softmax_splat(fwd, fx, fy, -react)
    ffx = np.nan_to_num(ff[:, :, 0], nan=1e6).astype(np.float32)
    ffy = np.nan_to_num(ff[:, :, 1], nan=1e6).astype(np.float32)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sx, sy = gx + np.nan_to_num(fx), gy + np.nan_to_num(fy)
    wf_x = cv2.remap(ffx, sx, sy, cv2.INTER_LINEAR, borderValue=1e6)
    wf_y = cv2.remap(ffy, sx, sy, cv2.INTER_LINEAR, borderValue=1e6)
    wb_x, wb_y = np.nan_to_num(fx), np.nan_to_num(fy)
    lhs = (wf_x + wb_x) ** 2 + (wf_y + wb_y) ** 2
    rhs = 0.01 * (wf_x ** 2 + wf_y ** 2 + wb_x ** 2 + wb_y ** 2) + 0.5
    return lhs - rhs


def history(frames, backbone, i, want, kind):
    h_lr, w_lr = frames[0][1].shape[:2]
    ref = (max([b for b in backbone if b < i], default=None) if want == "past"
           else min([b for b in backbone if b > i], default=None))
    if ref is None:
        return None, None, 1.0, 1.0
    dist_cur = float(abs(i - ref))
    if kind == "disp":
        j = i - 1
        if j < 0 or frames[j][2] is None or not len(frames[j][2]):
            return None, None, dist_cur, 1.0
        pfx, pfy = D.build_lr_flow(frames[j][2], h_lr, w_lr, want="all")
        # the previous display frame's own hop length is unknown in general; treat as 1
        return pfx, pfy, dist_cur, 1.0
    rprev = max([b for b in backbone if b < ref], default=None)
    if rprev is None or frames[ref][2] is None or not len(frames[ref][2]):
        return None, None, dist_cur, 1.0
    pfx, pfy = D.build_lr_flow(frames[ref][2], h_lr, w_lr, want="past")
    if kind == "ref-raw":
        return pfx, pfy, 1.0, 1.0          # literal FSR2 (no distance normalisation)
    return pfx, pfy, dist_cur, float(ref - rprev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="A,C")
    ap.add_argument("--tag", default="calib")
    args = ap.parse_args()
    scale = H.SCALE
    out = {}
    for wname in args.windows.split(","):
        frames, gt = H.window(wname)
        h_lr, w_lr = frames[0][1].shape[:2]
        w_hd, h_hd = w_lr * scale, h_lr * scale
        heavy, compact = H.sr_caches(wname, frames, w_hd, h_hd)
        cache = D.blend_anchor_cache({k: v.copy() for k, v in heavy.items()}, compact, 0.85)
        rows, R = D.reconstruct(frames, None, scale, True, "naive", cache, set(),
                                backend="numpy", collect_metrics=True)
        backbone = D.backbone_indices(frames)

        def prev_ip(i):
            return max([b for b in backbone if b < i], default=None)

        def next_ip(i):
            return min([b for b in backbone if b > i], default=None)

        acc = {}
        ndirs = 0
        for i in range(len(frames)):
            pt, lr, mvs = frames[i][:3]
            dirs = []
            if pt == "P" and prev_ip(i) is not None and not R[i]["is_anchor"]:
                dirs = [("past", prev_ip(i))]
            elif pt == "B":
                if prev_ip(i) is not None:
                    dirs.append(("past", prev_ip(i)))
                if next_ip(i) is not None:
                    dirs.append(("future", next_ip(i)))
            for want, ref in dirs:
                fx, fy = D.build_lr_flow(mvs, h_lr, w_lr, want=want)
                if not np.isfinite(fx).any():
                    continue
                ndirs += 1
                lr_ref = frames[ref][1]
                rec_dir, _, _ = D._warp_one(R[ref]["recon"], None, lr, lr_ref, mvs, want,
                                            scale, True, "naive", w_hd, h_hd)
                e_prop = np.abs(rec_dir.astype(np.int16) - gt[i].astype(np.int16)).mean(axis=2)
                e_pf = np.abs(R[i]["perframe"].astype(np.int16)
                              - gt[i].astype(np.int16)).mean(axis=2)
                gtB = hd_to_lr(e_prop > e_pf, scale)
                react_m, _ = F._ORIG_OCC_NP(fx, fy, lr, lr_ref, 16.0, "reactive")
                full_m, _ = F._ORIG_OCC_NP(fx, fy, lr, lr_ref, 16.0, "full")
                ruder_extra = full_m & ~react_m
                target_area = float(ruder_extra.mean())      # area-matched operating point
                # Ruder as a continuous signal
                rm = ruder_margin(fx, fy, lr, lr_ref)
                _bump(acc, "ruder_margin|-|auc", auc(rm, gtB))
                r, p, a = prf(ruder_extra, gtB)
                _bump(acc, "ruder_extra|-|recall", r)
                _bump(acc, "ruder_extra|-|prec", p)
                _bump(acc, "ruder_extra|-|area", a)
                r, p, a = prf(react_m | ruder_extra, gtB)
                _bump(acc, "react+ruder|-|recall", r)
                _bump(acc, "react+ruder|-|prec", p)
                _bump(acc, "react+ruder|-|area", a)
                r, p, a = prf(react_m, gtB)
                _bump(acc, "react|-|recall", r)
                _bump(acc, "react|-|prec", p)
                _bump(acc, "react|-|area", a)
                for hk in HISTS:
                  pfx, pfy, dc, dp = history(frames, backbone, i, want, hk)
                  for dil in DILATE:
                    for gname, glr in GATES:
                        if glr is None:
                            div = F.temporal_mv_divergence(fx, fy, pfx, pfy, scale,
                                                           dist_cur=dc, dist_prev=dp,
                                                           gate_px=1e-9, min_px=0.0,
                                                           normalize=True, dilate=dil)
                        else:
                            div = F.temporal_mv_divergence(fx, fy, pfx, pfy, scale,
                                                           dist_cur=dc, dist_prev=dp,
                                                           gate_px=glr * scale, min_px=1.0,
                                                           normalize=True, dilate=dil)
                        key = f"{hk}{'+dil' if dil else ''}|{gname}"
                        _bump(acc, f"{key}|auc", auc(div, gtB))
                        # area-matched threshold: take the top `target_area` fraction of div
                        if target_area > 0:
                            q = float(np.quantile(div, 1.0 - min(target_area, 0.99)))
                            m = div > max(q, 1e-9)
                            r, p, a = prf(m, gtB)
                            _bump(acc, f"{key}|am_recall", r)
                            _bump(acc, f"{key}|am_prec", p)
                            _bump(acc, f"{key}|am_area", a)
                        for thr in (0.3, 0.5):
                            r, p, a = prf(div > thr, gtB)
                            _bump(acc, f"{key}|t{thr}_recall", r)
                            _bump(acc, f"{key}|t{thr}_prec", p)
                            _bump(acc, f"{key}|t{thr}_area", a)
        res = {k: float(np.nanmean(v)) for k, v in acc.items()}
        res["_ndirs"] = ndirs
        out[wname] = res
        print(f"\n[{wname}] {ndirs} directions   (GT-B: fallback beats propagation)")
        print(f"  incumbent: react area {res.get('react|-|area', float('nan')):.4f} "
              f"rec {res.get('react|-|recall', float('nan')):.3f} "
              f"prec {res.get('react|-|prec', float('nan')):.3f}   |   "
              f"ruder-extra area {res.get('ruder_extra|-|area', float('nan')):.4f} "
              f"rec {res.get('ruder_extra|-|recall', float('nan')):.3f} "
              f"prec {res.get('ruder_extra|-|prec', float('nan')):.3f} "
              f"AUC(margin) {res.get('ruder_margin|-|auc', float('nan')):.3f}")
        print(f"  {'hist|gate':16} {'AUC':>6} {'am_rec':>7} {'am_prec':>8} {'t0.5_area':>10}")
        for hk in HISTS:
          for dil in DILATE:
            for gname, _ in GATES:
                k = f"{hk}{'+dil' if dil else ''}|{gname}"
                print(f"  {k:16} {res.get(k+'|auc', float('nan')):6.3f} "
                      f"{res.get(k+'|am_recall', float('nan')):7.3f} "
                      f"{res.get(k+'|am_prec', float('nan')):8.3f} "
                      f"{res.get(k+'|t0.5_area', float('nan')):10.4f}")
        H._free_gpu()
    H.save(out, os.path.join(H.OUT, f"{args.tag}.json"))


def _bump(acc, k, v):
    acc.setdefault(k, []).append(v)


if __name__ == "__main__":
    main()
