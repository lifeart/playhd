#!/usr/bin/env python3
"""R9-E1 step 4 (CPU): calibrate a NO-REFERENCE per-clip beta selector and test it on a
HELD-OUT operator family (avoid fitting to the test).

ESTIMATOR (per-clip GLOBAL beta; default is the shipped 0.85, only ever pulled DOWN):
    beta = 0.85 - 0.15 * s(edge) * d(disag)            # in [0.70, 0.85]
    s(edge)  = clip((e_hi - edge)/(e_hi - e_lo), 0, 1)  # 1 = SMOOTH content (low edge)
    d(disag) = clip((disag - d_lo)/(d_hi - d_lo), 0, 1) # 1 = high compact<->x4plus disagreement
Rationale (measured): the ONLY cells with headroom below 0.85 are SMOOTH + heavily
(synthetically) degraded; there x4plus over-sharpens noise the compact net avoids ->
high disagreement. Real-H.264 smooth clips are CLEAN -> low disagreement -> stay at 0.85
(R8-E3 OOD: beta<0.85 regresses there). Textured clips (high edge) stay at 0.85 (optimal).
edge = recommend_mode's Canny LR density; disag = mean |x4plus-compact| (both NO-ref).

CALIBRATION split: fit on synthetic {moderate,heavy}; TEST on synthetic {gritty} (held-out
operator) + all real-H.264 (held-out codec). beta(cell) is a CONTINUOUS function, evaluated
by LINEAR INTERPOLATION of the measured per-cell beta-sweep (verified exactly in step5)."""
import os, sys, json, itertools
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))

synth = json.load(open(os.path.join(_HERE, "step1_synth.json")))["rows"]
ood = json.load(open(os.path.join(_HERE, "step3_ood.json")))["rows"]
BETAS = np.array([round(0.50 + 0.05 * i, 2) for i in range(11)])


def lp_at(rowsweep, b):
    """Linear-interpolate measured LPIPS at beta b from the {beta:lpips} sweep dict."""
    xs = BETAS
    ys = np.array([rowsweep[f"{x:g}"] if f"{x:g}" in rowsweep else rowsweep[str(x)] for x in xs])
    return float(np.interp(b, xs, ys))


def beta_of(sig, p):
    e_lo, e_hi, d_lo, d_hi = p
    s = np.clip((e_hi - sig["edge"]) / max(e_hi - e_lo, 1e-6), 0, 1)
    d = np.clip((sig["disag_hr"] - d_lo) / max(d_hi - d_lo, 1e-6), 0, 1)
    return float(0.85 - 0.15 * s * d)


# ---- assemble cells with split tags ----
cells = {}
for k, r in synth.items():
    op = k.split("|")[1]
    cells[k] = dict(sweep=r["sweep"], sig=r["sig"], l085=r["sweep"]["0.85"],
                    split="calib" if op in ("moderate", "heavy") else "test_synth_gritty")
for k, r in ood.items():
    cells[k] = dict(sweep=r["sweep"], sig=r["sig"], l085=r["sweep"]["0.85"], split="test_ood_h264")

calib = {k: v for k, v in cells.items() if v["split"] == "calib"}

# ---- grid-search params on CALIB: maximize total gain s.t. no calib cell regresses > tol ----
E_LO = [0.05, 0.06, 0.07, 0.08]
E_HI = [0.10, 0.105, 0.11, 0.12]
D_LO = [1.5, 2.0, 2.5, 3.0]
D_HI = [3.5, 4.0, 4.5, 5.0]
TOL = 3e-4   # allowed per-cell regression vs 0.85 on CALIB (LPIPS)

best = None
for p in itertools.product(E_LO, E_HI, D_LO, D_HI):
    if p[0] >= p[1] or p[2] >= p[3]:
        continue
    gain = 0.0; maxreg = -1e9; ok = True
    for k, c in calib.items():
        b = beta_of(c["sig"], p)
        lp = lp_at(c["sweep"], b)
        reg = lp - c["l085"]
        maxreg = max(maxreg, reg)
        gain += (c["l085"] - lp)
        if reg > TOL:
            ok = False
    if ok and (best is None or gain > best[1]):
        best = (p, gain, maxreg)

P = best[0]
print(f"[calib] best params e_lo={P[0]} e_hi={P[1]} d_lo={P[2]} d_hi={P[3]}  "
      f"calib_total_gain={best[1]:.4f} calib_maxreg={best[2]:+.5f}")

# ---- evaluate FROZEN map on every cell ----
print(f"\n{'cell':22s} {'split':18s} {'edge':>6s} {'disag':>6s} {'beta':>5s} "
      f"{'L.85':>7s} {'Ladpt':>7s} {'dLPIPS':>8s}  verdict")
summary = {"calib": [], "test_synth_gritty": [], "test_ood_h264": []}
maxreg_test = -1e9
out_rows = {}
for k, c in cells.items():
    b = beta_of(c["sig"], P)
    lp = lp_at(c["sweep"], b)
    d = lp - c["l085"]
    summary[c["split"]].append(d)
    if c["split"] != "calib":
        maxreg_test = max(maxreg_test, d)
    v = "WIN" if d < -1e-4 else ("REGRESS" if d > 1e-4 else "tie")
    out_rows[k] = dict(beta=b, l085=c["l085"], ladpt=lp, dl=d, split=c["split"], verdict=v)
    print(f"{k:22s} {c['split']:18s} {c['sig']['edge']:6.4f} {c['sig']['disag_hr']:6.2f} "
          f"{b:5.3f} {c['l085']:7.4f} {lp:7.4f} {d:+8.4f}  {v}")

print("\n[aggregate]")
for sp, ds in summary.items():
    ds = np.array(ds)
    print(f"  {sp:18s} n={len(ds):2d}  mean dLPIPS={ds.mean():+.4f}  "
          f"max regress={ds.max():+.5f}  #win={int((ds<-1e-4).sum())} "
          f"#tie={int((np.abs(ds)<=1e-4).sum())} #reg={int((ds>1e-4).sum())}")
print(f"  >> max regression on ANY held-out cell = {maxreg_test:+.5f} "
      f"({'NO-REGRESSION' if maxreg_test <= 1e-4 else 'REGRESSES'})")

json.dump(dict(params=dict(e_lo=P[0], e_hi=P[1], d_lo=P[2], d_hi=P[3]),
               rows=out_rows), open(os.path.join(_HERE, "step4_calib.json"), "w"), indent=2)
print("[done] -> step4_calib.json")
