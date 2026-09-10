#!/usr/bin/env python3
"""
R13-E4: reproduce the PRIOR CLAIM's table and show what it measures.

The claim (window A frames 0-60, window C frames 5000-5120) was:

    window  mode                recall  precision  mask area
    A       full (+Ruder splat) 0.992   0.173      24.0%
    A       reactive only       0.991   0.227      17.2%
    C       full (+Ruder splat) 0.983   0.426      19.3%
    C       reactive only       0.980   0.540      16.5%

From those four numbers the implied GROUND-TRUTH area is recoverable:
    P = TP/recall = (area * precision) / recall
      A: (0.240*0.173)/0.992 = 4.19%   and (0.172*0.227)/0.991 = 3.94%
      C: (0.193*0.426)/0.983 = 8.36%   and (0.165*0.540)/0.980 = 9.09%
i.e. the "ground truth" is a set of ~4% (A) / ~8.7% (C) of pixels that is almost
entirely CONTAINED in the reactive mask (recall ~0.99 for a mask that IS the
reactive term). This script measures that containment directly, at the tau that
reproduces those GT areas, and reports what recall/precision look like against a
ground truth that is NOT a re-thresholding of the reactive signal (GT-B).
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
from run_diag import prf, hd_to_lr, auc     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="A,C")
    ap.add_argument("--tag", default="claim")
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
        acc = {}
        for i in range(len(frames)):
            pt, lr, mvs = frames[i][:3]
            dirs = []
            p = max([b for b in backbone if b < i], default=None)
            f_ = min([b for b in backbone if b > i], default=None)
            if pt == "P" and p is not None and not R[i]["is_anchor"]:
                dirs = [("past", p)]
            elif pt == "B":
                if p is not None:
                    dirs.append(("past", p))
                if f_ is not None:
                    dirs.append(("future", f_))
            for want, ref in dirs:
                fx, fy = D.build_lr_flow(mvs, h_lr, w_lr, want=want)
                if not np.isfinite(fx).any():
                    continue
                lr_ref = frames[ref][1]
                react_m, _ = F._ORIG_OCC_NP(fx, fy, lr, lr_ref, 16.0, "reactive")
                full_m, _ = F._ORIG_OCC_NP(fx, fy, lr, lr_ref, 16.0, "full")
                # --- prior-style GT: luma, INTEGER-rounded MVs, several taus ---
                fxi = np.round(np.nan_to_num(fx)).astype(np.float32)
                fyi = np.round(np.nan_to_num(fy)).astype(np.float32)
                ya = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY).astype(np.float32)
                yr = cv2.cvtColor(lr_ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
                gx, gy = np.meshgrid(np.arange(w_lr, dtype=np.float32),
                                     np.arange(h_lr, dtype=np.float32))
                pred = cv2.remap(yr, gx + fxi, gy + fyi, cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_REPLICATE)
                err = np.abs(ya - pred)
                for tau in (12, 16, 20, 24, 28, 32, 40):
                    g = (err > tau) | (~np.isfinite(fx))
                    for mname, m in (("react", react_m), ("full", full_m)):
                        r, pr, a = prf(m, g)
                        _b(acc, f"tau{tau}|{mname}|recall", r)
                        _b(acc, f"tau{tau}|{mname}|prec", pr)
                        _b(acc, f"tau{tau}|{mname}|area", a)
                    _b(acc, f"tau{tau}|gt_area", float(g.mean()))
                    # CONTAINMENT: what fraction of the "ground truth" already lies
                    # inside the reactive mask (the term being judged)?
                    _b(acc, f"tau{tau}|contain_react",
                       float((g & react_m).sum()) / max(float(g.sum()), 1e-9))
                # --- decision-relevant GT ---
                rec_dir, _, _ = D._warp_one(R[ref]["recon"], None, lr, lr_ref, mvs, want,
                                            scale, True, "naive", w_hd, h_hd)
                e_prop = np.abs(rec_dir.astype(np.int16) - gt[i].astype(np.int16)).mean(axis=2)
                e_pf = np.abs(R[i]["perframe"].astype(np.int16)
                              - gt[i].astype(np.int16)).mean(axis=2)
                gB = hd_to_lr(e_prop > e_pf, scale)
                for mname, m in (("react", react_m), ("full", full_m)):
                    r, pr, a = prf(m, gB)
                    _b(acc, f"gtB|{mname}|recall", r)
                    _b(acc, f"gtB|{mname}|prec", pr)
                    _b(acc, f"gtB|{mname}|area", a)
                _b(acc, "gtB|gt_area", float(gB.mean()))
                _b(acc, "gtB|contain_react",
                   float((gB & react_m).sum()) / max(float(gB.sum()), 1e-9))
                # severity-weighted: how much ERROR does each mask's extra area repair?
                extra = full_m & ~react_m
                ex_hd = np.repeat(np.repeat(extra, scale, 0), scale, 1)
                gain = (e_prop - e_pf)[ex_hd]        # >0 == fallback was the better call
                _b(acc, "ruder_extra|mean_gain", float(gain.mean()) if gain.size else 0.0)
                _b(acc, "ruder_extra|frac_helpful",
                   float((gain > 0).mean()) if gain.size else 0.0)
                _b(acc, "ruder_extra|area", float(extra.mean()))
                re_hd = np.repeat(np.repeat(react_m, scale, 0), scale, 1)
                gain_r = (e_prop - e_pf)[re_hd]
                _b(acc, "react|mean_gain", float(gain_r.mean()) if gain_r.size else 0.0)
                _b(acc, "react|frac_helpful",
                   float((gain_r > 0).mean()) if gain_r.size else 0.0)
        res = {k: float(np.nanmean(v)) for k, v in acc.items()}
        out[wname] = res
        print(f"\n[{wname}] prior-style GT (luma, integer MVs) -- recall/precision/area")
        print(f"  {'tau':>4} {'GT area':>8} {'contain':>8} | "
              f"{'react rec':>9} {'prec':>6} {'area':>6} | {'full rec':>9} {'prec':>6} {'area':>6}")
        for tau in (12, 16, 20, 24, 28, 32, 40):
            print(f"  {tau:>4} {res[f'tau{tau}|gt_area']:8.3f} "
                  f"{res[f'tau{tau}|contain_react']:8.3f} | "
                  f"{res[f'tau{tau}|react|recall']:9.3f} {res[f'tau{tau}|react|prec']:6.3f} "
                  f"{res[f'tau{tau}|react|area']:6.3f} | "
                  f"{res[f'tau{tau}|full|recall']:9.3f} {res[f'tau{tau}|full|prec']:6.3f} "
                  f"{res[f'tau{tau}|full|area']:6.3f}")
        print(f"  DECISION-relevant GT (fallback beats propagation): area "
              f"{res['gtB|gt_area']:.3f}, contained in reactive {res['gtB|contain_react']:.3f}")
        print(f"    react rec {res['gtB|react|recall']:.3f} prec {res['gtB|react|prec']:.3f} "
              f"area {res['gtB|react|area']:.3f} | full rec {res['gtB|full|recall']:.3f} "
              f"prec {res['gtB|full|prec']:.3f} area {res['gtB|full|area']:.3f}")
        print(f"    Ruder EXTRA pixels: area {res['ruder_extra|area']:.4f}  "
              f"helpful {100*res['ruder_extra|frac_helpful']:.1f}%  "
              f"mean gain {res['ruder_extra|mean_gain']:+.3f} /255  "
              f"(reactive's own: helpful {100*res['react|frac_helpful']:.1f}%, "
              f"gain {res['react|mean_gain']:+.3f})")
        H._free_gpu()
    H.save(out, os.path.join(H.OUT, f"{args.tag}.json"))


def _b(acc, k, v):
    acc.setdefault(k, []).append(v)


if __name__ == "__main__":
    main()
