#!/usr/bin/env python3
"""
R13-E3 analysis: turn the two runs into the report's numbers.

    python analyze.py [jelly calm]

* EXCESS flicker.  MCLF is not zero for ground-truth video (codec MVs are rate-distortion
  vectors), so the honest quantity is EXCESS = MCLF(sequence) - MCLF(bicubic(LR)), measured
  with the SAME flow fields -- what the SR+propagation stack ADDS over the content itself.
* Drift vs flicker on that excess (slope per frame-since-anchor, R^2, detrended std).
* Periodicity: excess grouped by frame type and by phase within the GOP cadence.
* Corrected hard-switch energy attribution (NaN-safe).
"""
import os
import sys
import json

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import jelly_metrics as JM  # noqa: E402
OUT = os.path.join(_HERE, "out")


def _fit(y, x):
    return JM.drift_vs_flicker(list(y), list(x))


def analyze(name):
    dj = json.load(open(os.path.join(OUT, f"diag_{name}.json")))
    rows = dj["rows"]
    t = np.array([r["t"] for r in rows], float)
    d = np.array([r["dist"] for r in rows], float)
    mclf = np.array([r["mclf"] for r in rows], float)
    bicf = np.array([r["mclf_bic"] for r in rows], float)
    hi = np.array([r["mclf_hi"] for r in rows], float)
    lo = np.array([r["mclf_lo"] for r in rows], float)
    flip = np.array([r["flip_frac"] for r in rows], float)
    mask = np.array([r["mask_frac"] for r in rows], float)
    types = [r["type"] for r in rows]
    excess = mclf - bicf

    print(f"\n================  {name.upper()}  ({len(rows)} adjacent pairs)")
    print(f"MCLF          mean {mclf.mean():6.3f}   bicubic(LR) content floor {bicf.mean():6.3f}"
          f"   EXCESS {excess.mean():+6.3f} codes ({100*excess.mean()/bicf.mean():+.1f}% of floor)")
    print(f"hard-switch   fresh-SR mask {100*mask.mean():5.1f}% of pixels, "
          f"MC-flip {100*flip.mean():5.1f}% of pixels per frame pair")

    # --- energy attribution, NaN-safe ---
    def term(vk, ak):
        v = np.nan_to_num(np.array([r[vk] for r in rows], float))
        a = np.nan_to_num(np.array([r[ak] for r in rows], float))
        return float(np.sum(v * a))
    e_flip = term("mclf_flip", "area_flip")
    e_prop = term("mclf_prop", "area_prop")
    e_fb = term("mclf_fb", "area_fb")
    tot = e_flip + e_prop + e_fb
    a_flip = float(np.nanmean([r["area_flip"] for r in rows]))
    a_prop = float(np.nanmean([r["area_prop"] for r in rows]))
    a_fb = float(np.nanmean([r["area_fb"] for r in rows]))
    print(f"flicker ENERGY share:  flip {100*e_flip/tot:5.1f}% (area {100*a_flip:4.1f}%)"
          f"   stable-propagated {100*e_prop/tot:5.1f}% (area {100*a_prop:4.1f}%)"
          f"   stable-fallback {100*e_fb/tot:5.1f}% (area {100*a_fb:4.1f}%)")
    mf = np.nanmean([r["mclf_flip"] for r in rows])
    mp = np.nanmean([r["mclf_prop"] for r in rows])
    print(f"per-pixel flicker:     flip set {mf:6.2f} codes vs stable-propagated {mp:5.2f} "
          f"=> {mf/mp:.1f}x")

    # --- drift vs flicker ---
    for lab, y in (("MCLF (raw)", mclf), ("MCLF EXCESS over content", excess),
                   ("MCLF high-freq", hi), ("MCLF low-freq", lo), ("flip fraction", flip)):
        f = _fit(y, d)
        span = f["slope"] * (d.max() - 1)
        print(f"  {lab:26s} slope {f['slope']:+.4f}/frame  R2 {f['r2']:5.3f}  "
              f"detrended-std {f['detrended_std']:6.3f}  trend-over-window {span:+6.3f}  "
              f"mean {np.nanmean(y):6.3f}")

    # --- periodicity: by frame type and by GOP phase ---
    for ft in ("P", "B"):
        m = [i for i, x in enumerate(types) if x == ft]
        if m:
            print(f"  frame type {ft}: n={len(m):2d} MCLF {mclf[m].mean():6.3f} "
                  f"excess {excess[m].mean():+6.3f} flip {100*flip[m].mean():5.1f}% "
                  f"mask {100*mask[m].mean():5.1f}%")
    ph = {}
    for i, r in enumerate(rows):
        ph.setdefault(int(r["dist"]) % 4, []).append(excess[i])
    print("  excess by GOP phase (dist mod 4): "
          + "  ".join(f"{k}:{np.mean(v):+.3f}" for k, v in sorted(ph.items())))

    # --- variant table (from run_ab) ---
    fp = os.path.join(OUT, f"results_{name}.json")
    if os.path.exists(fp):
        R = json.load(open(fp))
        b = R["metrics"]["baseline_hard"]
        bic_m = R["metrics"]["bicubic"]
        print(f"\n{'variant':17s} {'MCLF':>7} {'excess':>7} {'d-exc':>7} {'tLP':>8} {'dtLP':>7} "
              f"{'LPIPSx4':>8} {'dLP':>7} {'DISTSx4':>8} {'dDI':>7} {'tOF':>7} {'alpha':>6}")
        b_exc = b["mclf"] - bic_m["mclf"]
        for s, m in R["metrics"].items():
            exc = m["mclf"] - bic_m["mclf"]
            st = R["stab_stats"].get(s, {})
            print(f"{s:17s} {m['mclf']:7.3f} {exc:7.3f} "
                  f"{100*(exc-b_exc)/b_exc:+6.1f}% {m['tlp']:8.4f} "
                  f"{100*(m['tlp']-b['tlp'])/b['tlp']:+6.1f}% {m['lpips_vs_x4']:8.4f} "
                  f"{100*(m['lpips_vs_x4']-b['lpips_vs_x4'])/b['lpips_vs_x4']:+6.1f}% "
                  f"{m['dists_vs_x4']:8.4f} "
                  f"{100*(m['dists_vs_x4']-b['dists_vs_x4'])/b['dists_vs_x4']:+6.1f}% "
                  f"{m['tof_true']:7.3f} "
                  f"{st.get('alpha_mean_all', float('nan')):6.3f}")
        # drift/flicker of the variants' tLP + MCLF
        print(f"\n{'variant':17s} {'MCLF slope':>11} {'R2':>6} {'detrStd':>8} | "
              f"{'tLP slope':>10} {'R2':>6} {'detrStd':>8}")
        for s, m in R["metrics"].items():
            a, c = m["mclf_drift"], m["tlp_drift"]
            print(f"{s:17s} {a['slope']:+11.4f} {a['r2']:6.3f} {a['detrended_std']:8.3f} | "
                  f"{c['slope']:+10.5f} {c['r2']:6.3f} {c['detrended_std']:8.5f}")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["jelly", "calm"]):
        analyze(nm)
