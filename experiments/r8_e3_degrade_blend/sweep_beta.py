#!/usr/bin/env python3
"""R8-E3 fine global-beta sweep: find the SAFEST fixed beta (out = compact +
beta*(x4plus-compact)) that NEVER regresses vs x4plus while strictly beating it on
smooth-moderate. Reads the cached SR; LEAD = TRUE LPIPS. Also reports per-frame
no-regression rate (frames where blend LPIPS <= x4plus) for the chosen beta."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "experiments", "r5_e2_quality"))
import metrics as M  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_HERE, "cache")
meta = json.load(open(os.path.join(CACHE, "meta.json")))
windows, degrades = list(meta["windows"].keys()), meta["degrades"]
SMOOTH = {"talkinghead", "highmotion"}

cells = {}
for w in windows:
    for d in degrades:
        z = np.load(os.path.join(CACHE, f"{w}_{d}.npz"))
        cells[(w, d)] = {k: z[k] for k in ("gt", "compact", "x4plus")}


def blend(c, x, b):
    return np.clip(np.round(c.astype(np.float32) + b * (x.astype(np.float32) - c.astype(np.float32))),
                   0, 255).astype(np.uint8)


# Per-cell per-frame x4plus LPIPS (compute once).
x4_per = {}
for (w, d), cell in cells.items():
    x4_per[(w, d)] = [M.lpips_dist(x, g) for x, g in zip(cell["x4plus"], cell["gt"])]

BETAS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]
print(f"{'beta':>5s} {'max_regress':>12s} {'#beat':>6s} {'#tie':>5s} {'#regress':>9s} "
      f"{'TH-mod gain':>12s} {'mean gain%':>11s}  verdict")
rows = {}
for b in BETAS:
    maxreg = -1e9; nbeat = ntie = nreg = 0; gains = []
    th_mod_gain = None
    per_cell = {}
    for (w, d), cell in cells.items():
        bl = [blend(c, x, b) for c, x in zip(cell["compact"], cell["x4plus"])]
        lp = float(np.mean([M.lpips_dist(r, g) for r, g in zip(bl, cell["gt"])]))
        xl = float(np.mean(x4_per[(w, d)]))
        per_cell[f"{w}|{d}"] = dict(blend=lp, x4plus=xl)
        reg = lp - xl
        maxreg = max(maxreg, reg)
        gains.append((xl - lp) / xl * 100)
        if reg > 1e-4: nreg += 1
        elif reg < -1e-4: nbeat += 1
        else: ntie += 1
        if w == "talkinghead" and d == "moderate":
            th_mod_gain = (xl - lp) / xl * 100
    safe = (maxreg <= 1e-4)
    strict = safe and th_mod_gain is not None and th_mod_gain > 0
    rows[b] = dict(per_cell=per_cell, maxreg=maxreg, nbeat=nbeat, ntie=ntie, nreg=nreg,
                   mean_gain=float(np.mean(gains)))
    print(f"{b:5.2f} {maxreg:+12.4f} {nbeat:6d} {ntie:5d} {nreg:9d} {th_mod_gain:11.1f}% "
          f"{np.mean(gains):10.1f}%  {'STRICT-PASS' if strict else ('safe' if safe else 'REGRESSES')}")

# Per-frame no-regression detail for beta=0.75 (the candidate).
print("\nPer-frame detail at beta=0.75 (frac frames blend LPIPS <= x4plus; n=8/cell):")
B = 0.75
for (w, d), cell in cells.items():
    bl = [blend(c, x, B) for c, x in zip(cell["compact"], cell["x4plus"])]
    blp = [M.lpips_dist(r, g) for r, g in zip(bl, cell["gt"])]
    xp = x4_per[(w, d)]
    nore = np.mean([1.0 if a <= b + 1e-6 else 0.0 for a, b in zip(blp, xp)])
    beat = np.mean([1.0 if a < b - 1e-6 else 0.0 for a, b in zip(blp, xp)])
    print(f"  {w:12s}|{d:8s}  no-regress={nore*100:3.0f}%  strict-beat={beat*100:3.0f}%  "
          f"(blend {np.mean(blp):.4f} vs x4 {np.mean(xp):.4f})")

json.dump(rows, open(os.path.join(_HERE, "sweep_beta.json"), "w"), indent=2)
print("\n[done] -> sweep_beta.json")
