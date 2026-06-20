#!/usr/bin/env python3
"""R9-E1 step 6 (CPU): EXACT verification (no interpolation) of the chosen per-clip beta
maps on the cached pixels, with the full metric triple: TRUE AlexNet LPIPS (arbiter) +
DISTS (pyiqa, corroborator) + PSNR (context). Confirms (1) the interpolated beta-sweep
matched exact LPIPS, (2) DISTS agrees with LPIPS on the win/tie/regress verdict.

Maps: fixed-0.85 (shipped default) vs adaptive map (C) (robust hand-set, the deployable
no-regression selector) vs x4plus(beta=1.0)."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import signals as S  # noqa: E402
sys.path.insert(0, os.path.join(_HERE, "..", "r5_e2_quality"))
import metrics as M  # noqa: E402
import pyiqa  # noqa: E402

SYN_CACHE = os.path.join(_HERE, "..", "r8_e3_degrade_blend", "cache")
OOD_CACHE = os.path.join(_HERE, "ood_cache")
_DISTS = pyiqa.create_metric("dists", device="cpu")


@torch.no_grad()
def dists(a, b):
    ta = torch.from_numpy(np.ascontiguousarray(a)).float().div(255).permute(2, 0, 1).unsqueeze(0)
    tb = torch.from_numpy(np.ascontiguousarray(b)).float().div(255).permute(2, 0, 1).unsqueeze(0)
    return float(_DISTS(ta, tb))


def beta_C(sig, p=(9, 14, 3.0, 4.5)):
    t_lo, t_hi, d_lo, d_hi = p
    s = np.clip((t_hi - sig["tex_comp"]) / (t_hi - t_lo), 0, 1)
    d = np.clip((sig["disag_hr"] - d_lo) / (d_hi - d_lo), 0, 1)
    return float(0.85 - 0.15 * s * d)


def load(tag):
    if tag.endswith("_h264"):
        pass
    p = os.path.join(SYN_CACHE if "|" not in tag else "", "")  # unused
    return None


def score(cell, betas):
    gt, comp, x4 = cell["gt"], cell["compact"], cell["x4plus"]
    out = {}
    for name, b in betas.items():
        seq = [S.blend(c, x, b) for c, x in zip(comp, x4)]
        out[name] = dict(
            lpips=float(np.mean([M.lpips_dist(r, g) for r, g in zip(seq, gt)])),
            dists=float(np.mean([dists(r, g) for r, g in zip(seq, gt)])),
            psnr=float(np.mean([M.psnr(r, g) for r, g in zip(seq, gt)])),
            beta=b)
    return out


# assemble all cells (synthetic + OOD)
meta = json.load(open(os.path.join(SYN_CACHE, "meta.json")))
items = []
for w in meta["windows"]:
    for d in meta["degrades"]:
        items.append((f"{w}|{d}", os.path.join(SYN_CACHE, f"{w}_{d}.npz")))
for w, crf in json.load(open(os.path.join(OOD_CACHE, "jobs.json"))):
    items.append((f"{w}|crf{crf}", os.path.join(OOD_CACHE, f"{w}_crf{crf}.npz")))

print(f"{'cell':22s} {'beta':>5s} | {'LPIPS .85':>9s} {'LPIPS adp':>9s} {'dL':>8s} | "
      f"{'DISTS.85':>8s} {'DISTSadp':>8s} {'dD':>8s} | {'PSNR.85':>7s} {'PSNRadp':>7s}")
res = {}
for name, path in items:
    z = np.load(path)
    cell = {k: z[k] for k in z.files}
    sig = S.cell_signals(cell)
    b = beta_C(sig)
    sc = score(cell, {"f085": 0.85, "adpt": b, "x4": 1.0})
    dL = sc["adpt"]["lpips"] - sc["f085"]["lpips"]
    dD = sc["adpt"]["dists"] - sc["f085"]["dists"]
    res[name] = dict(beta=b, **{k: sc[k] for k in sc})
    v = "WIN" if dL < -1e-4 else ("REG" if dL > 1e-4 else "tie")
    print(f"{name:22s} {b:5.3f} | {sc['f085']['lpips']:9.4f} {sc['adpt']['lpips']:9.4f} "
          f"{dL:+8.4f} | {sc['f085']['dists']:8.4f} {sc['adpt']['dists']:8.4f} {dD:+8.4f} | "
          f"{sc['f085']['psnr']:7.2f} {sc['adpt']['psnr']:7.2f}  {v}")

# aggregate + LPIPS<->DISTS agreement
dLs = np.array([res[n]["adpt"]["lpips"] - res[n]["f085"]["lpips"] for n in res])
dDs = np.array([res[n]["adpt"]["dists"] - res[n]["f085"]["dists"] for n in res])
print(f"\nmax LPIPS regress vs .85 = {dLs.max():+.5f}  "
      f"({'NO-REGRESSION' if dLs.max() <= 1e-4 else 'REGRESSES'})")
print(f"max DISTS regress vs .85 = {dDs.max():+.5f}")
agree = np.mean(np.sign(dLs[np.abs(dLs) > 1e-4]) == np.sign(dDs[np.abs(dLs) > 1e-4]))
print(f"LPIPS<->DISTS sign agreement on non-tie cells = {agree*100:.0f}%")
json.dump(res, open(os.path.join(_HERE, "step6_verify.json"), "w"), indent=2)
print("[done] -> step6_verify.json")
