#!/usr/bin/env python3
"""
R8-E4 addendum: re-score the R5-E2 GRAIN A/B with DISTS (texture-aware) alongside
LPIPS+PSNR. Question (c): does a texture-aware FR metric STILL say grain hurts
fidelity, or does DISTS (lenient to texture statistics) credit grain's texture?

Reuses the EXACT R6-E1 restore path: compact-restore the `moderate` LR, then apply
prototype.grain.apply_grain (the deployed final pass) at off/low/med/high, score vs
the pseudo-HD GT. 3 windows (smooth + 2 textured). READ-ONLY imports; no empty catch.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r5_e2_quality"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r6_e1_srdecision"))

import metrics as M       # noqa
import run_matrix as R6   # noqa
import metrics_extra as MX  # noqa
import grain as GRAIN     # noqa  prototype/grain.py (read-only)

WINS = {"talkinghead": 5000, "texture24k": 24000, "texture46k": 46000}
LEVELS = ["off", "low", "med", "high"]


def main():
    n = 6
    print("[grain] decode + compact-restore (moderate degrade) ...")
    out = {}
    for wname, start in WINS.items():
        gt = R6.decode_window(R6.SAMPLE, start, n)
        h, w = gt[0].shape[:2]
        base = []
        for i, g in enumerate(gt):
            lr = R6.degrade(g, "moderate", seed=1000 + i)
            base.append(R6.restore(lr, w, h, "compact"))
        for lvl in LEVELS:
            grained = [GRAIN.apply_grain(b, i, strength=lvl) for i, b in enumerate(base)]
            lp = float(np.mean([M.lpips_dist(r, g) for r, g in zip(grained, gt)]))
            ds = float(np.mean([MX.dists(r, g) for r, g in zip(grained, gt)]))
            ps = float(np.mean([M.psnr(r, g) for r, g in zip(grained, gt)]))
            out.setdefault(wname, {})[lvl] = dict(lpips=lp, dists=ds, psnr=ps)
            print(f"  {wname:11s} grain={lvl:4s}  LPIPS={lp:.4f}  DISTS={ds:.4f}  PSNR={ps:.2f}")
    json.dump(out, open(os.path.join(_HERE, "grain_results.json"), "w"), indent=2)
    print("\n[grain] monotone-hurt check (does DISTS rise with grain, like LPIPS?):")
    for wname in WINS:
        d = out[wname]
        print(f"  {wname:11s} LPIPS {d['off']['lpips']:.3f}->{d['high']['lpips']:.3f}  "
              f"DISTS {d['off']['dists']:.3f}->{d['high']['dists']:.3f}  "
              f"PSNR {d['off']['psnr']:.1f}->{d['high']['psnr']:.1f}")


if __name__ == "__main__":
    main()
