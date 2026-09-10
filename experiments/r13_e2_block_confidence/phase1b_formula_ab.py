#!/usr/bin/env python3
"""
R13-E2 phase 1b -- run AGENT A's (R13-E1) exact block-penalty formulation through the same
server-side HD-ground-truth ranking test as this experiment's own formulation, so the two sides
are compared as ONE function rather than two opinions.

Agent A's form (browser, shipped):   tau_eff = max(tau - kp*(1 - area/256) - km*min(|mv|/mvref, 1), 0)
This experiment's form (server):     tau_eff = tau * c,  c = c_floor + (1-c_floor)*c_part*c_mv

Both lower the residual threshold using the same two free signals. The test asks the only
server-relevant question: ranking codec partitions by predicted unreliability, does either form
capture more true HD propagation-error mass than the residual alone?

Reuses the phase-1 block table (out/phase1_blocks.npz), so this is seconds, not minutes.
Run: python3 experiments/r13_e2_block_confidence/phase1b_formula_ab.py
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
import block_confidence as bc   # noqa: E402

OUT = os.path.join(HERE, "out")


def gain_curve(score, err, npix, fracs=(0.05, 0.10, 0.20, 0.30)):
    o = np.argsort(-score)
    ca = np.cumsum(npix[o]) / np.sum(npix)
    cm = np.cumsum((err * npix)[o]) / np.sum(err * npix)
    return {f"top{int(f*100)}": float(np.interp(f, ca, cm)) for f in fracs}


def main():
    d = np.load(os.path.join(OUT, "phase1_blocks.npz"))["pooled"]
    react, err, minside, mvmag, npix = [d[:, k] for k in range(5)]
    area = minside * 16.0          # the probes only ever emit 16x16 and 8x16/16x8 -> area = 16*minside
    rows = {}
    rows["residual_only (shipping)"] = gain_curve(react, err, npix)
    rows["oracle"] = gain_curve(err, err, npix)

    # --- agent A's formulation. Higher distrust == LOWER tau_eff, so rank by -tau_eff blended
    # with the residual exactly as the gate would: score = react - tau_eff (the sigmoid argument,
    # negated -- a monotone transform of `a`, so it ranks identically to 1-a).
    for mvref in (4.0, 8.0, 16.0, 32.0):
        for kp, km in ((6.0, 6.0), (3.0, 3.0), (12.0, 12.0), (6.0, 0.0), (0.0, 6.0)):
            cfg = dict(bc.BC_BROWSER, kp=kp, km=km, mvref=mvref)
            tau_eff = bc.tau_eff_blockpenalty(area, mvmag, cfg)
            rows[f"agentA kp={kp} km={km} mvref={mvref}"] = gain_curve(react - tau_eff, err, npix)
    # kp=km=0 is the pure-residual control inside agent A's own parameterisation
    cfg0 = dict(bc.BC_BROWSER, kp=0.0, km=0.0)
    rows["agentA kp=0 km=0 (== residual)"] = gain_curve(
        react - bc.tau_eff_blockpenalty(area, mvmag, cfg0), err, npix)

    # --- this experiment's multiplicative form, same test
    for cf in (0.35, 0.6, 0.0):
        c = bc.confidence_lr(minside, mvmag, dict(bc.BC_DEFAULT, c_floor=cf))
        rows[f"R13-E2 mul c_floor={cf}"] = gain_curve(react - 10.0 * c, err, npix)

    base = rows["residual_only (shipping)"]["top10"]
    print(f"{'predictor':44s} {'top5':>7s} {'top10':>7s} {'top20':>7s} {'top30':>7s}   vs residual")
    for k, v in rows.items():
        print(f"{k:44s} {v['top5']:7.4f} {v['top10']:7.4f} {v['top20']:7.4f} {v['top30']:7.4f}   "
              f"{100*(v['top10']-base)/base:+.2f}%")
    json.dump(rows, open(os.path.join(OUT, "phase1b_formula_ab.json"), "w"), indent=2)
    best = max((v["top10"], k) for k, v in rows.items() if k != "oracle")
    print(f"\nbest non-oracle predictor: {best[1]} ({best[0]:.4f})")


if __name__ == "__main__":
    sys.exit(main())
