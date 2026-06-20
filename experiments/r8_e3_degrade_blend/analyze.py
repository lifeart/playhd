#!/usr/bin/env python3
"""R8-E3 analysis: per-cell LPIPS/PSNR table (x4plus arbiter), per-cell fixed-beta
ORACLE (best fixed beta + whether it can beat x4plus), the adaptive variants vs the
STRICT win condition (<= x4plus on EVERY cell, strictly < on moderate/smooth), and
per-frame win-rate vs x4plus. Reads results.json from eval_blend.py."""
import os, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(_HERE, "results.json")))
res = R["results"]
SMOOTH = {"talkinghead", "highmotion"}
CELLS = list(res.keys())
FIXED = ["fix0.00", "fix0.25", "fix0.50", "fix0.75", "fix1.00"]
ADAPT = [k for k in next(iter(res.values())) if k.startswith("adapt_")]


def lp(cell, m): return res[cell][m]["lpips"]
def ps(cell, m): return res[cell][m]["psnr"]


print("=" * 110)
print("PER-CELL LPIPS (down=better). x4plus = arbiter. fix* = fixed-beta blend. *=beats x4plus")
print("=" * 110)
hdr = f"{'cell':22s}{'compact':>9s}{'x4plus':>9s}{'fix.25':>9s}{'fix.50':>9s}{'fix.75':>9s}"
for a in ADAPT:
    hdr += f"{a.replace('adapt_',''):>11s}"
print(hdr)
for c in CELLS:
    x = lp(c, "x4plus")
    row = f"{c:22s}{lp(c,'compact'):9.4f}{x:9.4f}"
    for m in ["fix0.25", "fix0.50", "fix0.75"]:
        v = lp(c, m); row += f"{v:8.4f}{'*' if v < x - 1e-9 else ' '}"
    for a in ADAPT:
        v = lp(c, a); row += f"{v:9.4f}{'*' if v < x - 1e-9 else ' '}{res[c][a]['beta_mean']:.2f}"[:11]
    print(row)

print("\n" + "=" * 110)
print("PER-CELL FIXED-BETA ORACLE: best fixed beta (incl. 1.0=x4plus) and the GAIN vs x4plus")
print("=" * 110)
print(f"{'cell':22s}{'argmin_beta':>12s}{'best_lpips':>12s}{'x4plus':>10s}{'gain%':>9s}  note")
for c in CELLS:
    x = lp(c, "x4plus")
    vals = [(b, lp(c, f"fix{b:.2f}")) for b in (0.0, 0.25, 0.5, 0.75, 1.0)]
    bb, bv = min(vals, key=lambda t: t[1])
    note = "x4plus IS optimal (textured-style)" if bb >= 1.0 - 1e-9 else \
           ("blend helps" if bv < x - 1e-9 else "")
    print(f"{c:22s}{bb:12.2f}{bv:12.4f}{x:10.4f}{100*(x-bv)/x:8.1f}%  {note}")

print("\n" + "=" * 110)
print("STRICT WIN-CONDITION CHECK per method (fixed + adaptive):")
print("  PASS = LPIPS <= x4plus on EVERY cell  AND  strictly < x4plus on >=1 smooth-moderate cell")
print("=" * 110)
methods = ["fix0.50", "fix0.75"] + ADAPT
for m in methods:
    regress = []   # cells where m is WORSE than x4plus (regression)
    wins = []      # cells where m strictly beats x4plus
    for c in CELLS:
        x, v = lp(c, "x4plus"), lp(c, m)
        if v > x + 1e-4:
            regress.append((c, v - x))
        elif v < x - 1e-4:
            wins.append((c, x - v))
    sm_mod_win = any(c.split("|")[0] in SMOOTH and c.endswith("moderate") for c, _ in wins)
    verdict = "PASS" if (not regress and sm_mod_win) else "FAIL"
    print(f"\n  [{m}] -> {verdict}")
    if regress:
        worst = sorted(regress, key=lambda t: -t[1])[:6]
        print(f"    REGRESSIONS ({len(regress)}): " +
              ", ".join(f"{c}(+{d:.4f})" for c, d in worst))
    else:
        print("    REGRESSIONS: none")
    if wins:
        best = sorted(wins, key=lambda t: -t[1])[:6]
        print(f"    WINS ({len(wins)}): " + ", ".join(f"{c}(-{d:.4f})" for c, d in best))

print("\n" + "=" * 110)
print("PER-FRAME win-rate vs x4plus (fraction of frames where method LPIPS strictly < x4plus)")
print("=" * 110)
for m in ["fix0.50"] + ADAPT:
    line = f"  {m:18s}"
    for c in CELLS:
        xp = res[c]["x4plus"]["lpips_per"]; mp = res[c][m]["lpips_per"]
        wr = np.mean([1.0 if a < b else 0.0 for a, b in zip(mp, xp)])
        line += f" {wr*100:3.0f}"
    print(line)
print("  cells order:", " ".join(c.replace("talkinghead","TH").replace("highmotion","HM")
      .replace("texture","tx").replace("|"," ") for c in CELLS))

print("\n" + "=" * 110)
print("MODERATE-vs-GRITTY decisive numbers (smooth=talkinghead, textured=texture24k)")
print("=" * 110)
for c in ["talkinghead|moderate", "talkinghead|heavy", "talkinghead|gritty",
          "texture24k|moderate", "texture24k|heavy", "texture24k|gritty"]:
    print(f"  {c:22s} compact={lp(c,'compact'):.4f} x4plus={lp(c,'x4plus'):.4f} "
          f"fix.50={lp(c,'fix0.50'):.4f} tex={lp(c,'adapt_tex'):.4f} "
          f"texdeg={lp(c,'adapt_tex_degrade'):.4f} disag={lp(c,'adapt_disag'):.4f}")
