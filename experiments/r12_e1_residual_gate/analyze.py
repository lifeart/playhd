#!/usr/bin/env python3
"""Print the R12-E1 A/B table (baseline HARD vs reliability-gate variants).
LPIPS/DISTS lower=closer to x4plus ceiling; tOF_true lower=less jelly (flow matches
true content motion); dF=raw temporal energy (context). Deltas are vs baseline_hard."""
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["compact_perframe", "baseline_hard", "gate_res_t16", "gate_res_t10",
         "gate_res_t7", "gate_resfb_t10", "bicubic"]


def pct(v, b):
    return f"{100*(v-b)/b:+.1f}%" if b else "  n/a"


for name in ("jelly", "calm"):
    fp = os.path.join(_HERE, "out", f"results_{name}.json")
    if not os.path.exists(fp):
        print(f"[{name}] no results yet"); continue
    R = json.load(open(fp))
    m = R["metrics"]
    b = m["baseline_hard"]
    print(f"\n=== {name.upper()}  (frames {R['start']}..{R['start']+R['n']-1}, "
          f"parity max|d|={R['parity_max_codes']:.1f} codes)  types={R['types']}")
    print(f"{'method':16s} {'LPIPS':>8} {'dLP':>7} {'DISTS':>8} {'dDI':>7} "
          f"{'tOFtrue':>8} {'dTOF':>7} {'tOF_x4':>7} {'dF':>7} {'distrust':>8}")
    for k in ORDER:
        if k not in m:
            continue
        r = m[k]
        du = f"{r.get('distrust_frac', float('nan')):.3f}" if 'distrust_frac' in r else "   -"
        print(f"{k:16s} {r['lpips']:8.4f} {pct(r['lpips'],b['lpips']):>7} "
              f"{r['dists']:8.4f} {pct(r['dists'],b['dists']):>7} "
              f"{r['tof_truemotion']:8.3f} {pct(r['tof_truemotion'],b['tof_truemotion']):>7} "
              f"{r['tof_vs_x4']:7.3f} {r['dF']:7.3f} {du:>8}")
