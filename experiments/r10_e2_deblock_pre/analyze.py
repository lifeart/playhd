#!/usr/bin/env python3
"""Aggregate R10-E2 A/B: deblock-preprocessor vs plain x4plus on REAL H.264 crops.
Arbiter = LPIPS & DISTS (both must improve for a GO). var-Lap = fake/over-smooth flag only."""
import json, os, sys
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
recs = json.load(open(os.path.join(HERE, "results.json")))
MODELS = ["bicubic","compact","x4plus","scunet_x4plus","scunet_x4plus_b85","bilat_x4plus","h264db_x4plus"]

def agg(rows):
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("lpips","dists","psnr","varlap"):
            d[r["model"]][k].append(r[k])
    return d

def mean(d, m, k):
    return float(np.mean(d[m][k])) if d[m][k] else float("nan")

print(f"\n=== OVERALL MEAN ({len(recs)} records) ===")
d = agg(recs)
gtvl = float(np.mean([r["gt_varlap"] for r in recs]))
print(f"{'model':<20} {'LPIPS↓':>8} {'DISTS↓':>8} {'PSNR↑':>7} {'varLap':>8}  (GT varLap={gtvl:.0f})")
base_l = mean(d,"x4plus","lpips"); base_d = mean(d,"x4plus","dists")
for m in MODELS:
    if not d[m]["lpips"]: continue
    L,D,P,V = mean(d,m,"lpips"),mean(d,m,"dists"),mean(d,m,"psnr"),mean(d,m,"varlap")
    tag=""
    if m not in ("bicubic","compact","x4plus"):
        dl=(L-base_l)/base_l*100; dd=(D-base_d)/base_d*100
        win = (L<base_l and D<base_d)
        tag=f"  ΔLPIPS{dl:+.1f}% ΔDISTS{dd:+.1f}% {'<-- WIN both' if win else ''}"
    print(f"{m:<20} {L:>8.4f} {D:>8.4f} {P:>7.2f} {V:>8.0f}{tag}")

# per-CRF (codec-dependence)
for crf in ("moderate","heavy"):
    rows=[r for r in recs if r["crf"]==crf]
    if not rows: continue
    d=agg(rows); bl=mean(d,"x4plus","lpips"); bd=mean(d,"x4plus","dists")
    print(f"\n=== CRF={crf} (n={len(rows)}) ===")
    print(f"{'model':<20} {'LPIPS↓':>8} {'DISTS↓':>8} {'PSNR↑':>7}")
    for m in MODELS:
        if not d[m]["lpips"]: continue
        L,D,P=mean(d,m,"lpips"),mean(d,m,"dists"),mean(d,m,"psnr")
        win=(m not in ("bicubic","compact","x4plus") and L<bl and D<bd)
        print(f"{m:<20} {L:>8.4f} {D:>8.4f} {P:>7.2f}{'  WIN' if win else ''}")

# per-window x CRF win count vs x4plus on BOTH metrics
print(f"\n=== per-(window,CRF) cells where config beats x4plus on BOTH LPIPS & DISTS ===")
cells=defaultdict(lambda: defaultdict(dict))
for r in recs:
    key=(r["window"],r["crf"])
    cells[key][r["model"]].setdefault("lpips",[]).append(r["lpips"]) if False else None
# build per-cell means
cm=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in recs:
    cm[(r["window"],r["crf"])][r["model"]]["lpips"].append(r["lpips"])
    cm[(r["window"],r["crf"])][r["model"]]["dists"].append(r["dists"])
cand=["scunet_x4plus","scunet_x4plus_b85","bilat_x4plus","h264db_x4plus"]
wins=defaultdict(int); ncells=0
for key,mm in sorted(cm.items()):
    ncells+=1
    bl=np.mean(mm["x4plus"]["lpips"]); bd=np.mean(mm["x4plus"]["dists"])
    line=[f"{key[0]:<12}{key[1]:<9}"]
    for c in cand:
        L=np.mean(mm[c]["lpips"]); D=np.mean(mm[c]["dists"])
        w = (L<bl and D<bd)
        wins[c]+= 1 if w else 0
        line.append(f"{c.split('_')[0]}:{'W' if w else ('lp' if L<bl else ('dt' if D<bd else '..'))}")
    print("  "+"  ".join(line))
print(f"\nBOTH-metric wins vs x4plus over {ncells} cells:")
for c in cand:
    print(f"  {c:<20} {wins[c]}/{ncells}")
