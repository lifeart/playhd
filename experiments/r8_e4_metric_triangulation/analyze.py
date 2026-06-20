#!/usr/bin/env python3
"""
R8-E4 analysis: build the triangulated table and flag every cell where DISTS
(texture-aware) DISAGREES with LPIPS (the project's lead) on the x4plus-vs-compact
verdict. Also report the fixed-0.5 blend arm and PSNR (the non-learned anchor).

A "verdict" per (window,degrade) = which of {compact, x4plus} a metric prefers.
- LOWER is better for LPIPS / DISTS; HIGHER for PSNR.
- We call a metric a TIE if |Δ| is within a small band (relative), else it picks a winner.
- DISAGREE cell = LPIPS-alex winner != DISTS winner (both non-tie, opposite).
Win-RATE (per-frame) is reported for the headline x4plus-vs-compact contrast so a flip
is not a mean artifact (mirrors R6-E1's 8/8 reporting).
"""
import os, json, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(_HERE, "results.json")))
res = R["results"]
WINDOWS = R["windows"]
DEGRADES = R["degrades"]

TIE_REL = 0.03   # within 3% relative -> tie (same spirit as R6-E1's "tie" calls)


def winner(a, b, lower_better=True):
    """Return ('compact'|'x4plus'|'tie', delta) for metric values a=compact,b=x4plus."""
    denom = max(abs(a), abs(b), 1e-9)
    rel = (a - b) / denom
    if abs(rel) < TIE_REL:
        return "tie", a - b
    if lower_better:
        return ("x4plus" if b < a else "compact"), a - b
    return ("x4plus" if b > a else "compact"), a - b


def winrate(per_c, per_x, lower_better=True):
    """Fraction of frames where x4plus beats compact."""
    c = np.array(per_c); x = np.array(per_x)
    wins = (x < c) if lower_better else (x > c)
    return float(wins.mean())


print("=" * 110)
print("TRIANGULATED TABLE  (compact / x4plus ; LPIPS-alex & DISTS lower=better, PSNR higher=better)")
print("=" * 110)
hdr = f"{'window':12s} {'degrade':9s} | {'LPIPSa c/x':>16s} {'win':>4s} | {'DISTS c/x':>16s} {'win':>4s} | {'PSNR c/x':>14s} {'win':>4s} | flag"
print(hdr); print("-" * len(hdr))

disagree_cells = []
agree_cells = []
rows = []
for w in WINDOWS:
    for d in DEGRADES:
        cell = res[w][d]
        c, x = cell["compact"], cell["x4plus"]
        lw, ld = winner(c["lpips"], x["lpips"], True)
        dw, dd = winner(c["dists"], x["dists"], True)
        pw, pd = winner(c["psnr"], x["psnr"], False)
        # disagreement = LPIPS and DISTS pick OPPOSITE non-tie winners
        flag = ""
        if lw != "tie" and dw != "tie" and lw != dw:
            flag = "<<DISAGREE>>"; disagree_cells.append((w, d, lw, dw))
        elif lw != "tie" and dw != "tie" and lw == dw:
            agree_cells.append((w, d, lw))
        wr_l = winrate(c["per"]["lpips"], x["per"]["lpips"], True)
        wr_d = winrate(c["per"]["dists"], x["per"]["dists"], True)
        rows.append(dict(w=w, d=d, lw=lw, dw=dw, pw=pw,
                         lc=c["lpips"], lx=x["lpips"], dc=c["dists"], dx=x["dists"],
                         pc=c["psnr"], px=x["psnr"], b=cell["blend05"],
                         wr_l=wr_l, wr_d=wr_d))
        print(f"{w:12s} {d:9s} | {c['lpips']:.4f}/{x['lpips']:.4f} {lw[:4]:>4s} | "
              f"{c['dists']:.4f}/{x['dists']:.4f} {dw[:4]:>4s} | "
              f"{c['psnr']:5.2f}/{x['psnr']:5.2f} {pw[:4]:>4s} | {flag}")

print("\n" + "=" * 110)
print("x4plus-vs-compact WIN-RATE (per-frame, fraction x4plus better) — is any flip a mean artifact?")
print("=" * 110)
print(f"{'window':12s} {'degrade':9s} | {'LPIPSa winrate':>15s} | {'DISTS winrate':>14s}")
for r in rows:
    print(f"{r['w']:12s} {r['d']:9s} | {r['wr_l']*100:13.0f}% | {r['wr_d']*100:12.0f}%")

# ---- blend arm vs both bases ----
print("\n" + "=" * 110)
print("FIXED-0.5 BLEND arm vs compact and x4plus (LPIPSa | DISTS | PSNR) — does blend ever win either metric?")
print("=" * 110)
for w in WINDOWS:
    for d in DEGRADES:
        cell = res[w][d]
        b, c, x = cell["blend05"], cell["compact"], cell["x4plus"]
        best_l = min(c["lpips"], x["lpips"])
        best_d = min(c["dists"], x["dists"])
        bl = "BLEND-WIN" if b["lpips"] < best_l - 1e-4 else ""
        bd = "BLEND-WIN" if b["dists"] < best_d - 1e-4 else ""
        print(f"{w:12s} {d:9s} | blend L={b['lpips']:.4f} D={b['dists']:.4f} P={b['psnr']:.2f} "
              f"| bestbase L={best_l:.4f} D={best_d:.4f} | {bl} {bd}")

# ---- LPIPS-alex vs LPIPS-pyiqa(vgg) cross-check (shared-bias sanity) ----
print("\n" + "=" * 110)
print("LPIPS-alex vs pyiqa-LPIPS cross-check (do the two LPIPS impls agree on the x4plus/compact winner?)")
print("=" * 110)
mismatch = 0
for w in WINDOWS:
    for d in DEGRADES:
        cell = res[w][d]
        c, x = cell["compact"], cell["x4plus"]
        a_w, _ = winner(c["lpips"], x["lpips"], True)
        v_w, _ = winner(c["lpips_vgg"], x["lpips_vgg"], True)
        if a_w != "tie" and v_w != "tie" and a_w != v_w:
            mismatch += 1
            print(f"  {w} {d}: alex->{a_w}  pyiqa->{v_w}  MISMATCH")
print(f"  LPIPS-impl winner mismatches: {mismatch}/{len(WINDOWS)*len(DEGRADES)}")

print("\n" + "=" * 110)
print("SUMMARY")
print("=" * 110)
print(f"DISAGREE (LPIPS vs DISTS opposite winners): {len(disagree_cells)} cells")
for w, d, lw, dw in disagree_cells:
    print(f"   {w} {d}: LPIPS->{lw}  DISTS->{dw}")
print(f"AGREE (both non-tie, same winner): {len(agree_cells)} cells")
from collections import Counter
ag = Counter(w for *_x, w in agree_cells)
print(f"   agree winners: {dict(ag)}")
print(f"timing ratio x4plus/compact = {R['timing_ms_per_frame']['x4plus']/R['timing_ms_per_frame']['compact']:.1f}x")
print(f"selfcheck: {R['selfcheck']}")
