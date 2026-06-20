#!/usr/bin/env python3
"""R8-E3 seam verification for anchor_blend.patch. Proves:
  (1) blend_anchor_cache(beta=None) and beta=1.0 leave the heavy cache BYTE-IDENTICAL
      (the default-OFF guarantee).
  (2) the patch helper's math == the measured eval_blend.blend math (caller==handler==metric).
  (3) re-scoring one cell through the helper reproduces the sweep_beta.json LPIPS.
No shared files are imported-and-mutated; the helper is a verbatim copy of the patch body."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "experiments", "r5_e2_quality"))
import metrics as M  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_HERE, "cache")


def blend_anchor_cache(heavy_cache, compact_cache, beta):     # verbatim from anchor_blend.patch
    if beta is None or float(beta) >= 1.0 - 1e-9:
        return heavy_cache
    b = float(beta)
    for i in list(heavy_cache.keys()):
        h = heavy_cache[i].astype(np.float32)
        c = compact_cache[i].astype(np.float32)
        heavy_cache[i] = np.clip(np.round(c + b * (h - c)), 0, 255).astype(np.uint8)
    return heavy_cache


def eval_blend(c, x, b):    # == eval_blend.py / sweep_beta.py measured op
    return np.clip(np.round(c.astype(np.float32) + b * (x.astype(np.float32) - c.astype(np.float32))),
                   0, 255).astype(np.uint8)


z = np.load(os.path.join(CACHE, "texture24k_gritty.npz"))
heavy = {i: z["x4plus"][i].copy() for i in range(z["x4plus"].shape[0])}
compact = {i: z["compact"][i].copy() for i in range(z["compact"].shape[0])}
gt = z["gt"]

# (1) default-OFF == byte-identical
for off in (None, 1.0):
    h2 = {i: z["x4plus"][i].copy() for i in range(z["x4plus"].shape[0])}
    blend_anchor_cache(h2, compact, off)
    ok = all(np.array_equal(h2[i], z["x4plus"][i]) for i in h2)
    print(f"(1) beta={str(off):4s}: heavy cache byte-identical to x4plus -> {ok}")
    assert ok, "DEFAULT-OFF VIOLATION"

# (2) patch-helper math == measured eval math, at beta=0.85
h3 = {i: z["x4plus"][i].copy() for i in range(z["x4plus"].shape[0])}
blend_anchor_cache(h3, compact, 0.85)
same = all(np.array_equal(h3[i], eval_blend(compact[i], z["x4plus"][i], 0.85)) for i in h3)
print(f"(2) beta=0.85: blend_anchor_cache == eval_blend (measured op) -> {same}")
assert same, "SEAM MISMATCH: integration math != measured math"

# (3) re-score through the helper reproduces the sweep LPIPS for this cell
lp = float(np.mean([M.lpips_dist(h3[i], gt[i]) for i in h3]))
sweep = json.load(open(os.path.join(_HERE, "sweep_beta.json")))
ref = sweep["0.85"]["per_cell"]["texture24k|gritty"]["blend"]
print(f"(3) texture24k|gritty beta=0.85 LPIPS via helper={lp:.4f}  sweep={ref:.4f}  "
      f"match={abs(lp-ref) < 1e-4}")
assert abs(lp - ref) < 1e-4, "RE-SCORE MISMATCH"
print("[seam-verify] ALL PASS")
