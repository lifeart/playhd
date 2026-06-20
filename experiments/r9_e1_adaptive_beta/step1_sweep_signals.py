#!/usr/bin/env python3
"""R9-E1 step 1 (CPU, reuses the R8-E3 cached SR -> ZERO GPU): for every (window x
degrade) cell, compute (a) the fine per-cell beta-sweep TRUE-LPIPS curve and the
oracle-best beta, and (b) the GLOBAL no-reference signal battery. Dump for analysis of
whether ANY no-reference signal tracks the oracle-best beta (the core falsifier)."""
import os, sys, json
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import signals as S  # noqa: E402

CACHE = os.path.join(_HERE, "..", "r8_e3_degrade_blend", "cache")
meta = json.load(open(os.path.join(CACHE, "meta.json")))
windows, degrades = list(meta["windows"].keys()), meta["degrades"]

rows = {}
for w in windows:
    for d in degrades:
        z = np.load(os.path.join(CACHE, f"{w}_{d}.npz"))
        cell = {k: z[k] for k in z.files}
        sweep, x4_per = S.sweep_cell_lpips(cell)
        sig = S.cell_signals(cell)
        best_b = min(sweep, key=sweep.get)
        rows[f"{w}|{d}"] = dict(
            sweep=sweep, x4_per=x4_per, sig=sig,
            best_beta=best_b, lpips_best=sweep[best_b],
            lpips_x4=sweep[1.0], lpips_085=sweep[0.85],
            headroom_vs_085=sweep[0.85] - sweep[best_b],
        )
        print(f"  [{w:12s}|{d:8s}] best_b={best_b:.2f} L*={sweep[best_b]:.4f} "
              f"L.85={sweep[0.85]:.4f} Lx4={sweep[1.0]:.4f} | "
              f"edge={sig['edge']:.4f} noise_mad={sig['noise_mad']:.2f} "
              f"immerk={sig['immerk']:.2f} disag={sig['disag_hr']:.2f} hf={sig['hf_ratio']:.3f}")

json.dump(dict(meta=meta, rows=rows), open(os.path.join(_HERE, "step1_synth.json"), "w"), indent=2)

# ---- correlation of each signal with the oracle-best beta (the falsifier) ---- #
print("\n[signal -> oracle-best-beta correlation across 15 cells]")
keys = list(next(iter(rows.values()))["sig"].keys())
bb = np.array([rows[k]["best_beta"] for k in rows])
for sk in keys:
    sv = np.array([rows[k]["sig"][sk] for k in rows])
    # Pearson + Spearman
    pear = np.corrcoef(sv, bb)[0, 1]
    rs = np.argsort(np.argsort(sv)); rb = np.argsort(np.argsort(bb))
    spear = np.corrcoef(rs, rb)[0, 1]
    print(f"  {sk:16s} pearson={pear:+.3f} spearman={spear:+.3f}")
print("[done] -> step1_synth.json")
