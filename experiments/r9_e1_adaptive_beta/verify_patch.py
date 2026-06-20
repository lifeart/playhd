#!/usr/bin/env python3
"""R9-E1 patch seam verification (no prototype edit required -- reimplements the patched
select_anchor_beta exactly as the diff, then checks):
  (1) params=None  -> returns None  (caller falls back to fixed cfg beta -> byte-identical).
  (2) the selector's beta on each cached cell == step5/step6's measured beta (the math the
      eval scored), recomputed from the two caches the same way the patch will at runtime.
  (3) blend(... beta=None) is a no-op (the existing R8-E3 helper) -> OFF == byte-identical."""
import os, sys, json
import numpy as np, cv2
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import signals as S  # noqa: E402

PARAMS = dict(t_lo=9, t_hi=14, d_lo=3.0, d_hi=4.5)   # map (C), the deployable selector


def select_anchor_beta(heavy_cache, compact_cache, params):
    """EXACT copy of the patched derisk.select_anchor_beta body."""
    if params is None:
        return params
    t_lo, t_hi = float(params["t_lo"]), float(params["t_hi"])
    d_lo, d_hi = float(params["d_lo"]), float(params["d_hi"])
    texs, disags = [], []
    for i in list(heavy_cache.keys()):
        yc = cv2.cvtColor(compact_cache[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        yh = cv2.cvtColor(heavy_cache[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        mu = cv2.boxFilter(yc, -1, (7, 7))
        var = np.maximum(cv2.boxFilter(yc * yc, -1, (7, 7)) - mu * mu, 0.0)
        texs.append(float(np.sqrt(var).mean()))
        disags.append(float(np.abs(yh - yc).mean()))
    tex_comp = float(np.mean(texs)); disag = float(np.mean(disags))
    s = min(max((t_hi - tex_comp) / max(t_hi - t_lo, 1e-6), 0.0), 1.0)
    d = min(max((disag - d_lo) / max(d_hi - d_lo, 1e-6), 0.0), 1.0)
    return float(0.85 - 0.15 * s * d)


def beta_ref(sig, p=(9, 14, 3.0, 4.5)):
    t_lo, t_hi, d_lo, d_hi = p
    s = np.clip((t_hi - sig["tex_comp"]) / (t_hi - t_lo), 0, 1)
    d = np.clip((sig["disag_hr"] - d_lo) / (d_hi - d_lo), 0, 1)
    return float(0.85 - 0.15 * s * d)


# (1) OFF -> None
assert select_anchor_beta({}, {}, None) is None, "params=None must return None"
print("[1] params=None -> None (byte-identical fallback to fixed cfg beta) ... PASS")

# (2) selector beta == measured beta, on every cached cell
SYN = os.path.join(_HERE, "..", "r8_e3_degrade_blend", "cache")
OOD = os.path.join(_HERE, "ood_cache")
meta = json.load(open(os.path.join(SYN, "meta.json")))
items = [(f"{w}|{d}", os.path.join(SYN, f"{w}_{d}.npz"))
         for w in meta["windows"] for d in meta["degrades"]]
items += [(f"{w}|crf{c}", os.path.join(OOD, f"{w}_crf{c}.npz"))
          for w, c in json.load(open(os.path.join(OOD, "jobs.json")))]
maxerr = 0.0
for name, path in items:
    z = np.load(path)
    heavy = {i: z["x4plus"][i] for i in range(z["x4plus"].shape[0])}   # dict[int]->HxWx3 (runtime shape)
    comp = {i: z["compact"][i] for i in range(z["compact"].shape[0])}
    b_sel = select_anchor_beta(heavy, comp, PARAMS)
    b_ref = beta_ref(S.cell_signals({k: z[k] for k in z.files}))
    maxerr = max(maxerr, abs(b_sel - b_ref))
assert maxerr < 1e-6, f"selector beta diverges from measured map (maxerr={maxerr})"
print(f"[2] selector beta == measured map on all {len(items)} cells (max|d|={maxerr:.2e}) ... PASS")

# (3) blend(beta=None) no-op already proven in R8-E3 verify_patch; re-assert helper identity
a = np.random.RandomState(0).randint(0, 256, (4, 4, 3), np.uint8)
assert np.array_equal(S.blend(a, a, 0.85), a), "blend of identical arrays must be identity"
print("[3] blend helper identity sanity ... PASS")
print("\nALL SEAM CHECKS PASS")
