#!/usr/bin/env python3
"""Aggregate R12-E2 QP-gated A/B. Find the QP crossover where deblock flips
win->loss, calibrate the gate threshold, and prove GATED fires on heavy / skips light.

deblock "wins" a frame iff ON beats OFF on BOTH LPIPS and DISTS (R10 arbiter rule)."""
import json, os
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
recs = json.load(open(os.path.join(HERE, "ab_results.json")))


def wins(r):
    return (r["on_lpips"] < r["off_lpips"]) and (r["on_dists"] < r["off_dists"])


# ---- per-CRF: mean QP + deblock win-rate, split smooth vs texture ----
print("=== deblock ON vs OFF by CRF (win = ON beats OFF on LPIPS AND DISTS) ===")
print(f"{'CRF':>4} {'QPmean':>7} | {'ALL win%':>8} {'ΔLPIPS%':>8} {'ΔDISTS%':>8} {'ΔPSNR':>6} "
      f"| {'smooth win%':>11} {'texture win%':>12}")
by_crf = defaultdict(list)
for r in recs:
    by_crf[r["crf"]].append(r)
for crf in sorted(by_crf):
    rs = by_crf[crf]
    qp = np.mean([r["qp_mean"] for r in rs])
    w = np.mean([wins(r) for r in rs]) * 100
    dl = np.mean([(r["on_lpips"] - r["off_lpips"]) / r["off_lpips"] for r in rs]) * 100
    dd = np.mean([(r["on_dists"] - r["off_dists"]) / r["off_dists"] for r in rs]) * 100
    dp = np.mean([r["on_psnr"] - r["off_psnr"] for r in rs])
    sm = [r for r in rs if r["smooth"]]
    tx = [r for r in rs if not r["smooth"]]
    ws = np.mean([wins(r) for r in sm]) * 100
    wt = np.mean([wins(r) for r in tx]) * 100
    print(f"{crf:>4} {qp:>7.1f} | {w:>7.0f}% {dl:>+7.1f}% {dd:>+7.1f}% {dp:>+6.2f} "
          f"| {ws:>10.0f}% {wt:>11.0f}%")

# ---- QP binned win-rate (find crossover) ----
print("\n=== win-rate vs exact per-frame QP (crossover search) ===")
edges = [20, 28, 32, 36, 40, 44, 60]
print(f"{'QP bin':>12} {'n':>4} {'win%':>6} {'ΔLPIPS%':>8} {'ΔDISTS%':>8}")
for lo, hi in zip(edges[:-1], edges[1:]):
    rs = [r for r in recs if lo <= r["qp_mean"] < hi]
    if not rs:
        continue
    w = np.mean([wins(r) for r in rs]) * 100
    dl = np.mean([(r["on_lpips"] - r["off_lpips"]) / r["off_lpips"] for r in rs]) * 100
    dd = np.mean([(r["on_dists"] - r["off_dists"]) / r["off_dists"] for r in rs]) * 100
    print(f"{lo:>4}-{hi:<7} {len(rs):>4} {w:>5.0f}% {dl:>+7.1f}% {dd:>+7.1f}%")

# ---- choose threshold: maximize net benefit ----
# For each candidate THR, GATED chooses ON iff qp>=THR. Score = mean regret vs the
# per-frame ORACLE (best of on/off). Also report aggregate LPIPS/DISTS/PSNR.
def config_metrics(pick):  # pick(r) -> True means use ON
    L = np.mean([r["on_lpips"] if pick(r) else r["off_lpips"] for r in recs])
    D = np.mean([r["on_dists"] if pick(r) else r["off_dists"] for r in recs])
    P = np.mean([r["on_psnr"] if pick(r) else r["off_psnr"] for r in recs])
    return L, D, P

print("\n=== threshold calibration (GATED = deblock iff qp_mean >= THR) ===")
print(f"{'THR':>4} | {'fire%':>6} {'LPIPS':>7} {'DISTS':>7} {'PSNR':>6} | "
      f"{'wrong-fire(light)':>16} {'missed(heavy-win)':>17}")
best = None
for thr in range(28, 46, 2):
    pick = lambda r, t=thr: r["qp_mean"] >= t
    fire = np.mean([pick(r) for r in recs]) * 100
    L, D, P = config_metrics(pick)
    # wrong-fire: fired but deblock did NOT win that frame
    wf = sum(1 for r in recs if pick(r) and not wins(r))
    # missed: deblock WOULD have won but gate skipped
    ms = sum(1 for r in recs if (not pick(r)) and wins(r))
    score = L + D  # lower is better (LPIPS+DISTS both in [0,~1])
    print(f"{thr:>4} | {fire:>5.0f}% {L:>7.4f} {D:>7.4f} {P:>6.2f} | {wf:>16} {ms:>17}")
    if best is None or score < best[1]:
        best = (thr, score, L, D, P)
THR = best[0]

# ---- final 3-way comparison at the chosen threshold ----
offL, offD, offP = config_metrics(lambda r: False)
onL, onD, onP = config_metrics(lambda r: True)
gL, gD, gP = config_metrics(lambda r: r["qp_mean"] >= THR)
oracL = np.mean([min(r["on_lpips"], r["off_lpips"]) for r in recs])
print(f"\n=== FINAL 3-way (THR = qp_mean >= {THR}) ===")
print(f"{'config':<10} {'LPIPS↓':>8} {'DISTS↓':>8} {'PSNR↑':>7}")
for name, (L, D, P) in [("OFF", (offL, offD, offP)), ("ON(always)", (onL, onD, onP)),
                        ("GATED", (gL, gD, gP))]:
    print(f"{name:<10} {L:>8.4f} {D:>8.4f} {P:>7.2f}")
print(f"{'ORACLE-LP':<10} {oracL:>8.4f}  (per-frame best-of; lower bound)")

# ---- fire-rate by CRF: proves the gate separates light vs heavy ----
print(f"\n=== gate fire-rate by CRF (THR={THR}) — must be ~0% light, ~100% heavy ===")
print(f"{'CRF':>4} {'QPmean':>7} {'fire%':>6}")
for crf in sorted(by_crf):
    rs = by_crf[crf]
    fire = np.mean([r["qp_mean"] >= THR for r in rs]) * 100
    print(f"{crf:>4} {np.mean([r['qp_mean'] for r in rs]):>7.1f} {fire:>5.0f}%")

# ---- the blockiness confound: show QP is content-independent, blockiness isn't ----
print("\n=== blockiness vs QP by content @ CRF27 (moderate, deblock should SKIP) ===")
print(f"{'window':<12} {'QPmean':>7} {'blockiness':>11} {'gt_varlap':>10}")
for win in ["talkinghead", "highmotion", "texture18k", "texture24k", "texture46k"]:
    rs = [r for r in recs if r["window"] == win and r["crf"] == 27]
    if not rs:
        continue
    print(f"{win:<12} {np.mean([r['qp_mean'] for r in rs]):>7.1f} "
          f"{np.mean([r['blockiness'] for r in rs]):>11.3f} "
          f"{np.mean([r['gt_varlap'] for r in rs]):>10.0f}")
