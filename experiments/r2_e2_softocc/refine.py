#!/usr/bin/env python3
"""R2-E2 refinement: trace the escape frontier near bicubic's tOF; pin the operating point.
Reuses softocc.setup / policies / run_scheme / metrics (READ-ONLY of prototype as before)."""
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import softocc as S

def linf(bic, hard, eff):
    """linear-interpolated tOF on the bicubic<->hard frontier at a given eff-bic%."""
    return bic["tof"] + (hard["tof"]-bic["tof"])*(bic["eff_bicubic_pct"]-eff)/(
        bic["eff_bicubic_pct"]-hard["eff_bicubic_pct"])

def main():
    W = S.setup()
    FE = 21
    grid = []
    grid.append(("bicubic", S.pol_bicubic(W)))
    grid.append(("HARD-SR all", S.pol_hard(W, feather=0)))
    # (b3) HF-EMA finer beta
    for b in (0.7, 0.8, 0.85, 0.9, 0.95):
        grid.append((f"(b3) ema-HF b={b}", S.pol_b3_ema_hf(W, b, FE, graded=True)))
    # (c) combo: gain x beta grid near bicubic tOF
    for g in (0.4, 0.5, 0.6, 0.75, 0.85, 1.0):
        for b in (0.8, 0.85, 0.9):
            grid.append((f"(c) g={g} b={b}", S.pol_c_combo(W, g, b, FE)))
    # feather sensitivity at the chosen knee
    for fe in (9, 31, 51):
        grid.append((f"(c) g=0.6 b=0.85 fe={fe}", S.pol_c_combo(W, 0.6, 0.85, fe)))

    res = []
    for name, pol in grid:
        out, leak = S.run_scheme(W, pol)
        r = S.metrics(W, out, name); r["leak_pct"] = round(100*leak, 2)
        res.append(r); del out; S._free_gpu()
    bic = next(r for r in res if r["scheme"]=="bicubic")
    hard = next(r for r in res if r["scheme"]=="HARD-SR all")
    print(f"\nbicubic: tOF={bic['tof']:.4f} effBic%={bic['eff_bicubic_pct']:.3f}   "
          f"HARD: tOF={hard['tof']:.4f} effBic%={hard['eff_bicubic_pct']:.3f}")
    print(f"{'scheme':26s}{'tOF':>8}{'effBic%':>9}{'detail%':>9}{'frontier_tOF':>13}{'gain(-)':>9}{'fbdF':>7}")
    for r in res:
        ft = linf(bic, hard, r["eff_bicubic_pct"])
        gain = ft - r["tof"]                                 # >0 => BELOW the frontier (escape)
        flag = "  ESCAPE" if gain > 0.01 and r["eff_bicubic_pct"] < bic["eff_bicubic_pct"]-0.2 else ""
        print(f"{r['scheme']:26s}{r['tof']:>8.4f}{r['eff_bicubic_pct']:>9.3f}"
              f"{r['detail_injected_pct']:>9.3f}{ft:>13.4f}{gain:>9.4f}{r['fb_localized_dF']:>7.2f}{flag}")
    json.dump(dict(bic=bic, hard=hard, results=res),
              open(os.path.join(_HERE,"refine_results.json"),"w"), indent=2)
    print(f"\nwrote {os.path.join(_HERE,'refine_results.json')}")

if __name__=="__main__":
    main()
