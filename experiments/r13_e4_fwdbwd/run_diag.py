#!/usr/bin/env python3
"""
R13-E4 diagnostics: reproduce the prior recall/precision claim PROPERLY, and show
what it is actually measuring.

Two ground truths, computed on the same frames, same production sub-pixel flows:

  GT-A  "per-pixel warp failure" the way the prior analysis defined it: LUMA-only
        prediction error of an INTEGER-rounded MV warp at LR, thresholded.
        -> This is definitionally (b), the REACTIVE term, up to luma-vs-RGB and MV
           rounding. Scoring masks against it is close to scoring them against the
           reactive term itself, which structurally favours 'reactive' and
           structurally penalises any term that fires where COLOUR happens to match
           but GEOMETRY is wrong -- which is exactly the failure codec MVs have
           (GOTCHA #6: RD-optimised MVs point at "visually similar but geometrically
           wrong" content) and exactly what a fwd-bwd check exists to catch.

  GT-B  DECISION-RELEVANT: at HD, against the real ground truth, a pixel SHOULD be
        masked iff falling back to per-frame SR is closer to GT than the propagated
        pixel:  |perframe - GT| < |propagated - GT|.
        This is the thing the mask is actually for.

For each direction of each non-anchor frame we score: reactive, Ruder-extra,
FSR2-temporal (continuous -> AUC, and thresholded -> recall/precision), plus the
union variants. Pure propagation (occ='naive') is used as the chain so the
propagated pixel is genuinely un-repaired.
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


def auc(score, label, cap=200000, seed=0):
    """Rank-based ROC AUC on a subsample (labels bool, score float)."""
    s = score.reshape(-1).astype(np.float64)
    y = label.reshape(-1).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    if s.size > cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(s.size, cap, replace=False)
        s, y = s[idx], y[idx]
        if y.all() or not y.any():
            return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    # average ranks over ties
    su = s[order]
    i = 0
    while i < su.size:
        j = i
        while j + 1 < su.size and su[j + 1] == su[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def prf(mask, gt):
    tp = float(np.sum(mask & gt))
    fp = float(np.sum(mask & ~gt))
    fn = float(np.sum(~mask & gt))
    rec = tp / max(tp + fn, 1e-9)
    pre = tp / max(tp + fp, 1e-9)
    return rec, pre, float(mask.mean())


def hd_to_lr(mask_hd, scale):
    """Majority-downsample an HD bool mask to LR (block mean > 0.5)."""
    h, w = mask_hd.shape
    m = mask_hd.reshape(h // scale, scale, w // scale, scale).mean(axis=(1, 3))
    return m > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="A,B,C")
    ap.add_argument("--tau-react", type=float, default=16.0)
    ap.add_argument("--tag", default="diag")
    args = ap.parse_args()

    scale = H.SCALE
    out = {}
    for wname in args.windows.split(","):
        frames, gt = H.window(wname)
        h_lr, w_lr = frames[0][1].shape[:2]
        w_hd, h_hd = w_lr * scale, h_lr * scale
        heavy, compact = H.sr_caches(wname, frames, w_hd, h_hd)
        cache = D.blend_anchor_cache({k: v.copy() for k, v in heavy.items()}, compact, 0.85)
        # PURE propagation chain (intra holes only) -> un-repaired propagated pixels
        rows, R = D.reconstruct(frames, None, scale, True, "naive", cache, set(),
                                backend="numpy", collect_metrics=True)
        backbone = D.backbone_indices(frames)

        def prev_ip(i):
            return max([b for b in backbone if b < i], default=None)

        def next_ip(i):
            return min([b for b in backbone if b > i], default=None)

        recs = []
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
                lr_ref = frames[ref][1]
                # --- the propagated pixel for THIS direction (production op) ---
                rec_dir, _, hole_hd = D._warp_one(R[ref]["recon"], None, lr, lr_ref, mvs,
                                                  want, scale, True, "naive", w_hd, h_hd)
                pf = R[i]["perframe"]
                e_prop = np.abs(rec_dir.astype(np.int16) - gt[i].astype(np.int16)).mean(axis=2)
                e_pf = np.abs(pf.astype(np.int16) - gt[i].astype(np.int16)).mean(axis=2)
                gtB_hd = (e_prop > e_pf)
                gtB = hd_to_lr(gtB_hd, scale)
                # --- GT-A: the prior analysis' definition (luma, INTEGER MVs) ---
                fxi = np.round(np.nan_to_num(fx)).astype(np.float32)
                fyi = np.round(np.nan_to_num(fy)).astype(np.float32)
                ya = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY).astype(np.float32)
                yr = cv2.cvtColor(lr_ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
                gxg, gyg = np.meshgrid(np.arange(w_lr, dtype=np.float32),
                                       np.arange(h_lr, dtype=np.float32))
                pred = cv2.remap(yr, gxg + fxi, gyg + fyi, cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_REPLICATE)
                gtA = (np.abs(ya - pred) > args.tau_react) | (~np.isfinite(fx))
                # --- candidate masks / scores (production ops) ---
                react_m, _ = F._ORIG_OCC_NP(fx, fy, lr, lr_ref, args.tau_react, "reactive")
                full_m, _ = F._ORIG_OCC_NP(fx, fy, lr, lr_ref, args.tau_react, "full")
                ruder_m = full_m & ~react_m
                pfx, pfy, dc, dp = _history(frames, backbone, i, want)
                div = F.temporal_mv_divergence(fx, fy, pfx, pfy, scale, dist_cur=dc,
                                               dist_prev=dp, gate_px=20.0, normalize=True)
                div_ng = F.temporal_mv_divergence(fx, fy, pfx, pfy, scale, dist_cur=dc,
                                                  dist_prev=dp, gate_px=1e-9, min_px=0.0,
                                                  normalize=True)  # ungated ratio term
                # residual "score" forms for AUC (continuous, threshold-free)
                predr = D.warp_lr(lr_ref, fx, fy).astype(np.float32)
                react_s = np.abs(lr.astype(np.float32) - predr).mean(axis=2)
                rec_entry = dict(
                    frame=i, ftype=pt, want=want, ref=ref,
                    area_react=float(react_m.mean()), area_ruder=float(ruder_m.mean()),
                    area_full=float(full_m.mean()),
                    gtA_area=float(gtA.mean()), gtB_area=float(gtB.mean()),
                )
                for gname, g in (("A", gtA), ("B", gtB)):
                    for mname, m in (("react", react_m), ("full", full_m),
                                     ("ruder_only", ruder_m)):
                        r, p, a = prf(m, g)
                        rec_entry[f"{gname}_{mname}_recall"] = r
                        rec_entry[f"{gname}_{mname}_prec"] = p
                    rec_entry[f"{gname}_auc_react"] = auc(react_s, g)
                    rec_entry[f"{gname}_auc_fsr2"] = auc(div, g)
                    rec_entry[f"{gname}_auc_fsr2_ungated"] = auc(div_ng, g)
                    for thr in (0.1, 0.3, 0.5, 0.7):
                        r, p, a = prf(div > thr, g)
                        rec_entry[f"{gname}_fsr2@{thr}_recall"] = r
                        rec_entry[f"{gname}_fsr2@{thr}_prec"] = p
                        rec_entry[f"{gname}_fsr2@{thr}_area"] = a
                        r, p, a = prf((react_m | (div > thr)), g)
                        rec_entry[f"{gname}_react+fsr2@{thr}_recall"] = r
                        rec_entry[f"{gname}_react+fsr2@{thr}_prec"] = p
                        rec_entry[f"{gname}_react+fsr2@{thr}_area"] = a
                rec_entry["div_p50"] = float(np.percentile(div, 50))
                rec_entry["div_p95"] = float(np.percentile(div, 95))
                rec_entry["div_p99"] = float(np.percentile(div, 99))
                rec_entry["div_max"] = float(div.max())
                recs.append(rec_entry)
        out[wname] = recs
        print(f"[{wname}] {len(recs)} directions scored")
        _summarize(wname, recs)
        H._free_gpu()
    H.save(out, os.path.join(H.OUT, f"{args.tag}.json"))


def _history(frames, backbone, i, want):
    ref = (max([b for b in backbone if b < i], default=None) if want == "past"
           else min([b for b in backbone if b > i], default=None))
    if ref is None:
        return None, None, 1.0, 1.0
    rprev = max([b for b in backbone if b < ref], default=None)
    if rprev is None:
        return None, None, float(abs(i - ref)), 1.0
    h_lr, w_lr = frames[0][1].shape[:2]
    pfx, pfy = D.build_lr_flow(frames[ref][2], h_lr, w_lr, want="past")
    return pfx, pfy, float(abs(i - ref)), float(ref - rprev)


def _summarize(wname, recs):
    def m(k):
        v = [r[k] for r in recs if k in r and np.isfinite(r[k])]
        return float(np.mean(v)) if v else float("nan")
    print(f"  GT-A (prior-style luma/int-MV warp failure, area {m('gtA_area'):.3f}):")
    print(f"    react  recall {m('A_react_recall'):.3f} prec {m('A_react_prec'):.3f} "
          f"area {m('area_react'):.3f}")
    print(f"    full   recall {m('A_full_recall'):.3f} prec {m('A_full_prec'):.3f} "
          f"area {m('area_full'):.3f}")
    print(f"    AUC: react-residual {m('A_auc_react'):.3f}  fsr2 {m('A_auc_fsr2'):.3f} "
          f" fsr2-ungated {m('A_auc_fsr2_ungated'):.3f}")
    print(f"  GT-B (decision-relevant: fallback beats propagation, area {m('gtB_area'):.3f}):")
    print(f"    react  recall {m('B_react_recall'):.3f} prec {m('B_react_prec'):.3f}")
    print(f"    full   recall {m('B_full_recall'):.3f} prec {m('B_full_prec'):.3f}")
    print(f"    ruder-only-extra recall {m('B_ruder_only_recall'):.3f} "
          f"prec {m('B_ruder_only_prec'):.3f} area {m('area_ruder'):.4f}")
    print(f"    AUC: react-residual {m('B_auc_react'):.3f}  fsr2 {m('B_auc_fsr2'):.3f} "
          f" fsr2-ungated {m('B_auc_fsr2_ungated'):.3f}")
    print(f"    div pct: p50 {m('div_p50'):.3f} p95 {m('div_p95'):.3f} "
          f"p99 {m('div_p99'):.3f} max {m('div_max'):.3f}")


if __name__ == "__main__":
    main()
