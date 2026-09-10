#!/usr/bin/env python3
"""Render results_<window>.json as the markdown table used in REPORT.md (deltas vs hard4)."""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def pct(new, base):
    if base == 0:
        return "n/a"
    d = 100.0 * (new - base) / base
    return f"{d:+.1f}%"


for win in sys.argv[1:] or ["calm", "jelly"]:
    p = os.path.join(_HERE, f"results_{win}.json")
    if not os.path.exists(p):
        print(f"(no results for {win})")
        continue
    r = json.load(open(p))
    rows = {x["variant"]: x for x in r["rows"]}
    base = rows.get("hard4")
    print(f"\n### {win} ({r['frames']} frames, {r['res']}, propagated-frame metrics; ref = SR every frame)\n")
    print("| variant | mean a | PSNR↑ | LPIPS↓ | ΔLPIPS | DISTS↓ | ΔDISTS | tOF(LR)↓ | ΔtOF | warp ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for v, x in rows.items():
        a = "—" if x["gate_mean_a"] is None else f"{x['gate_mean_a']:.3f}"
        print(f"| {v} | {a} | {x['psnr_warp']:.2f} | {x['lpips_warp']:.4f} | "
              f"{pct(x['lpips_warp'], base['lpips_warp']) if base else ''} | {x['dists_warp']:.4f} | "
              f"{pct(x['dists_warp'], base['dists_warp']) if base else ''} | {x['tof_lr']:.4f} | "
              f"{pct(x['tof_lr'], base['tof_lr']) if base else ''} | {x['ms_warp_median']:.3f} |")
