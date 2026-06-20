#!/usr/bin/env python3
"""R9-E1 step 5 (CPU): the TWO-GATE per-clip beta selector (content gate = tex_comp,
degrade gate = disag), which fixes step4's edge failure. Three analyses:

  (A) HELD-OUT  : calibrate thresholds on synthetic {moderate,heavy}; test on the
                  held-out synthetic {gritty} + all real-H.264 cells (fit-to-test avoided).
  (B) ALL-CELL ORACLE: fit thresholds to ALL 23 cells under a HARD zero-regression
                  constraint, maximize gain -> the UPPER BOUND on what this estimator
                  family can do (does a no-regression positive-gain map even EXIST?).
  (C) ROBUST hand-set map (gaps): content t in [9,14], degrade d in [3.0,4.5] -- the
                  deployable map whose d_lo is informed by the measured H.264 disag floor.

beta = 0.85 - 0.15 * s(texC) * d(disag);  s=1 when SMOOTH (low texC), d=1 when high disag.
Default 0.85 (shipped); only ever pulled DOWN (clean content never wants beta>0.85)."""
import os, sys, json, itertools
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))

synth = json.load(open(os.path.join(_HERE, "step1_synth.json")))["rows"]
ood = json.load(open(os.path.join(_HERE, "step3_ood.json")))["rows"]
BETAS = np.array([round(0.50 + 0.05 * i, 2) for i in range(11)])


def lp_at(sw, b):
    xs = np.array(sorted(float(k) for k in sw))
    ys = np.array([sw[k] for k in sorted(sw, key=float)])
    return float(np.interp(b, xs, ys))


def beta_of(sig, p):
    t_lo, t_hi, d_lo, d_hi = p
    s = np.clip((t_hi - sig["tex_comp"]) / max(t_hi - t_lo, 1e-6), 0, 1)
    d = np.clip((sig["disag_hr"] - d_lo) / max(d_hi - d_lo, 1e-6), 0, 1)
    return float(0.85 - 0.15 * s * d)


cells = {}
for k, r in synth.items():
    op = k.split("|")[1]
    cells[k] = dict(sw=r["sweep"], sig=r["sig"], l085=r["sweep"]["0.85"], bb=r["best_beta"],
                    head=r["headroom_vs_085"],
                    split="calib" if op in ("moderate", "heavy") else "test_gritty")
for k, r in ood.items():
    cells[k] = dict(sw=r["sweep"], sig=r["sig"], l085=r["sweep"]["0.85"], bb=r["best_beta"],
                    head=r["headroom_vs_085"], split="test_h264")

T_LO = [7, 8, 9, 10]; T_HI = [12, 13, 14, 16]
D_LO = [1.5, 2.0, 2.5, 3.0, 3.1]; D_HI = [3.5, 4.0, 4.5, 5.0]


def search(train_cells, tol):
    best = None
    for p in itertools.product(T_LO, T_HI, D_LO, D_HI):
        if p[0] >= p[1] or p[2] >= p[3]:
            continue
        gain = 0.0; ok = True
        for c in train_cells:
            b = beta_of(c["sig"], p)
            reg = lp_at(c["sw"], b) - c["l085"]
            gain += (c["l085"] - lp_at(c["sw"], b))
            if reg > tol:
                ok = False; break
        if ok and (best is None or gain > best[1]):
            best = (p, gain)
    return best


def evaluate(p, tag):
    print(f"\n===== {tag}  params t=[{p[0]},{p[1]}] d=[{p[2]},{p[3]}] =====")
    print(f"{'cell':22s} {'split':11s} {'texC':>6s} {'disag':>6s} {'beta':>5s} "
          f"{'L.85':>7s} {'Ladpt':>7s} {'dLPIPS':>8s} verdict")
    agg = {}
    maxreg = -9; rows = {}
    for k, c in cells.items():
        b = beta_of(c["sig"], p)
        lp = lp_at(c["sw"], b)
        d = lp - c["l085"]
        agg.setdefault(c["split"], []).append(d)
        maxreg = max(maxreg, d)
        v = "WIN" if d < -1e-4 else ("REGRESS" if d > 1e-4 else "tie")
        rows[k] = dict(beta=round(b, 3), dl=round(d, 5), split=c["split"], verdict=v)
        print(f"{k:22s} {c['split']:11s} {c['sig']['tex_comp']:6.2f} {c['sig']['disag_hr']:6.2f} "
              f"{b:5.3f} {c['l085']:7.4f} {lp:7.4f} {d:+8.4f} {v}")
    print("  --aggregate--")
    for sp, ds in agg.items():
        ds = np.array(ds)
        print(f"    {sp:11s} n={len(ds):2d} mean={ds.mean():+.4f} maxreg={ds.max():+.5f} "
              f"#win={int((ds<-1e-4).sum())} #reg={int((ds>1e-4).sum())}")
    print(f"  >> GLOBAL max regression = {maxreg:+.5f}  "
          f"({'NO-REGRESSION (PASS)' if maxreg <= 1e-4 else 'REGRESSES (FAIL)'})")
    return rows, maxreg


# (A) held-out
calib = [c for c in cells.values() if c["split"] == "calib"]
pA, gA = search(calib, tol=3e-4)
print(f"[A held-out] calib-fit params={pA} calib_gain={gA:.4f}")
rowsA, mrA = evaluate(pA, "(A) HELD-OUT (calib=synth mod+heavy)")

# (B) all-cell oracle, zero-regression
allc = list(cells.values())
pB, gB = search(allc, tol=1e-4)
print(f"\n[B oracle] all-cell zero-regress fit params={pB} gain={gB:.4f}")
rowsB, mrB = evaluate(pB, "(B) ALL-CELL ORACLE (zero-regress)")

# (C) robust hand-set
pC = (9, 14, 3.0, 4.5)
rowsC, mrC = evaluate(pC, "(C) ROBUST hand-set d_lo=3.0 (H.264-floor informed)")

json.dump(dict(A=dict(p=pA, rows=rowsA, maxreg=mrA),
               B=dict(p=pB, rows=rowsB, maxreg=mrB),
               C=dict(p=pC, rows=rowsC, maxreg=mrC)),
          open(os.path.join(_HERE, "step5_twogate.json"), "w"), indent=2)
print("\n[done] -> step5_twogate.json")
