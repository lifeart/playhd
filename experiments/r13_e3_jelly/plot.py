#!/usr/bin/env python3
"""R13-E3 plots: the drift-vs-flicker diagnostic + the mechanism A/B.

    python plot.py            # reads out/results_*.json, writes out/*.png
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "out")

DIAG = ["baseline_hard", "compact_perframe", "x4plus_perframe", "bicubic"]
VARS = ["baseline_hard", "ema_fixed_50", "ema_adapt_50", "ema_adapt_75",
        "instab_fsr", "instab_relaxed", "ema_instab_50"]
COL = {"baseline_hard": "tab:red", "compact_perframe": "tab:orange",
       "x4plus_perframe": "tab:purple", "bicubic": "tab:gray",
       "ema_fixed_50": "tab:blue", "ema_adapt_50": "tab:green",
       "ema_adapt_75": "tab:olive", "instab_fsr": "tab:cyan",
       "instab_relaxed": "tab:brown", "ema_instab_50": "black"}


def _load(name):
    fp = os.path.join(OUT, f"results_{name}.json")
    return json.load(open(fp)) if os.path.exists(fp) else None


def diagnostic(names):
    fig, axes = plt.subplots(2, len(names), figsize=(7 * len(names), 8), squeeze=False)
    for c, nm in enumerate(names):
        R = _load(nm)
        if R is None:
            continue
        d = np.asarray(R["dist"], float)
        for r, key, lab in ((0, "mclf", "MC luma flicker (codes)"),
                            (1, "tlp", "tLP (excess perceptual change)")):
            ax = axes[r][c]
            for s in DIAG:
                y = np.asarray(R["series"][s][key], float)
                ax.plot(d, y, "o-", ms=3, lw=1, color=COL[s], label=s, alpha=0.85)
                fit = R["metrics"][s][f"{key}_drift"]
                if np.isfinite(fit["slope"]):
                    xs = np.array([1, d.max()])
                    ax.plot(xs, fit["slope"] * xs + fit["intercept"], "--", lw=1.2,
                            color=COL[s], alpha=0.6)
            ax.set_xlabel("frames since anchor")
            ax.set_ylabel(lab)
            ax.set_title(f"{nm}: {lab}\n(dashed = fit vs frames-since-anchor)")
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fp = os.path.join(OUT, "diagnostic_drift_vs_flicker.png")
    fig.savefig(fp, dpi=110)
    print("wrote", fp)


def ab(names):
    fig, axes = plt.subplots(1, len(names) * 2, figsize=(7 * len(names), 4.2), squeeze=False)
    for c, nm in enumerate(names):
        R = _load(nm)
        if R is None:
            continue
        have = [v for v in VARS if v in R["metrics"]]
        base = R["metrics"]["baseline_hard"]
        ax = axes[0][2 * c]
        x = np.arange(len(have))
        for k, (key, off) in enumerate((("mclf", -0.2), ("tlp", 0.0), ("lpips_vs_x4", 0.2))):
            vals = [100 * (R["metrics"][v][key] - base[key]) / base[key] for v in have]
            ax.bar(x + off, vals, width=0.2,
                   label={"mclf": "MCLF", "tlp": "tLP", "lpips_vs_x4": "LPIPS vs x4plus"}[key])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(have, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("% vs baseline (lower = better)")
        ax.set_title(f"{nm}: mechanism A/B")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
        ax2 = axes[0][2 * c + 1]
        for v in have:
            ax2.scatter(100 * (R["metrics"][v]["lpips_vs_x4"] - base["lpips_vs_x4"]) / base["lpips_vs_x4"],
                        100 * (R["metrics"][v]["mclf"] - base["mclf"]) / base["mclf"],
                        color=COL.get(v, "k"), s=45, label=v)
        ax2.axhline(0, color="k", lw=0.8)
        ax2.axvline(0, color="k", lw=0.8)
        ax2.set_xlabel("LPIPS vs x4plus ceiling  (% vs baseline; right = worse quality)")
        ax2.set_ylabel("MCLF (% vs baseline; down = steadier)")
        ax2.set_title(f"{nm}: steadiness vs quality trade")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=6)
    fig.tight_layout()
    fp = os.path.join(OUT, "mechanism_ab.png")
    fig.savefig(fp, dpi=110)
    print("wrote", fp)


def table(names):
    for nm in names:
        R = _load(nm)
        if R is None:
            continue
        b = R["metrics"]["baseline_hard"]
        print(f"\n=== {nm.upper()}  n={R['n']} types={R['types']} mean|MV|={R['mv_mean']:.2f} "
              f"flow-cover={100*R['flow_cover']:.0f}%  OFF-parity={R['off_parity_codes']} codes")
        print(f"{'sequence':17s} {'MCLF':>7} {'dMCLF':>7} {'tLP':>8} {'dtLP':>7} "
              f"{'LPIPSx4':>8} {'dLP':>7} {'DISTSx4':>8} {'tOF':>7} {'dF':>7} "
              f"{'slopeMCLF':>10} {'R2':>6} {'detrStd':>8}")
        for s, m in R["metrics"].items():
            dr = m["mclf_drift"]
            print(f"{s:17s} {m['mclf']:7.3f} "
                  f"{100*(m['mclf']-b['mclf'])/b['mclf']:+6.1f}% {m['tlp']:8.4f} "
                  f"{100*(m['tlp']-b['tlp'])/b['tlp']:+6.1f}% {m['lpips_vs_x4']:8.4f} "
                  f"{100*(m['lpips_vs_x4']-b['lpips_vs_x4'])/b['lpips_vs_x4']:+6.1f}% "
                  f"{m['dists_vs_x4']:8.4f} {m['tof_true']:7.3f} {m['dF']:7.3f} "
                  f"{dr['slope']:10.4f} {dr['r2']:6.3f} {dr['detrended_std']:8.3f}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["jelly", "calm"]
    table(names)
    diagnostic(names)
    ab(names)
