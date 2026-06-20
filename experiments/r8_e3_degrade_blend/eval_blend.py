#!/usr/bin/env python3
"""
R8-E3 step 2 (CPU, iterates on the cached SR): evaluate the LOCAL degrade-adaptive
compact<->x4plus anchor blend against x4plus-alone and the fixed-0.5 blend, on the
R6-E1 degrade-restore matrix. LEAD = TRUE AlexNet LPIPS (lower=better); PSNR for
perception-distortion context (GOTCHA #23: NR var-Lap is NEVER the arbiter).

Blend op (reuses region_quality.blend_region_aware's math: out = c + b*(x - c),
per-pixel b in [0,1]). b->0 = compact, b->1 = x4plus.

Signals (all cheap, GT-free, from already-computed sources):
  tex   = temporal-mean local luma STD of the COMPACT HR output (R7-E2 a_texture).
  lrhf  = temporal-mean local luma STD of the degraded LR (resized to HR) -- a
          DEGRADE proxy: high HF = lightly degraded; low HF = heavily degraded.
  disag = temporal-mean local |x4plus - compact| (luma) -- where the heavy net
          departs from compact.

Methods scored per (window,degrade): compact, x4plus(=arbiter), bicubic, fixed-b
sweep {0,.25,.5,.75,1}, and the adaptive-b variants defined in BETA_VARIANTS.

WIN CONDITION (strict): adaptive LPIPS <= x4plus on EVERY cell (no regression) AND
strictly < x4plus on the moderate/smooth cells.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r5_e2_quality"))
import metrics as M    # noqa: E402  TRUE LPIPS lead + PSNR
CACHE = os.path.join(_HERE, "cache")

SMOOTH = ["talkinghead", "highmotion"]
TEXTURED = ["texture18k", "texture24k", "texture46k"]


# ----------------------------- local signal helpers ------------------------- #
def _luma(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _local_std(y, k):
    mu = cv2.boxFilter(y, -1, (k, k))
    var = np.maximum(cv2.boxFilter(y * y, -1, (k, k)) - mu * mu, 0.0)
    return np.sqrt(var)


def signal_maps(cell, k=7):
    """Temporal-mean HR signal maps for a cell. cell = dict of [n,H,W,3] arrays."""
    comp, x4, lr = cell["compact"], cell["x4plus"], cell["lr"]
    n, H, W = comp.shape[:3]
    tex = np.zeros((H, W), np.float32)
    disag = np.zeros((H, W), np.float32)
    lrhf = np.zeros((H, W), np.float32)
    for i in range(n):
        yc = _luma(comp[i])
        tex += _local_std(yc, k)
        disag += _local_std(np.abs(x4[i].astype(np.float32) - comp[i].astype(np.float32)).mean(2), k)
        ylr = _luma(lr[i])
        s = _local_std(ylr, k)
        lrhf += cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)
    return dict(tex=tex / n, disag=disag / n, lrhf=lrhf / n)


# ----------------------------- blend ---------------------------------------- #
def blend(comp, x4, beta):
    """out = comp + beta*(x4 - comp). beta: scalar or [H,W] (broadcast over RGB)."""
    c = comp.astype(np.float32)
    x = x4.astype(np.float32)
    b = beta if np.isscalar(beta) else beta[..., None]
    return np.clip(np.round(c + b * (x - c)), 0, 255).astype(np.uint8)


def ramp(v, lo, hi):
    return np.clip((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


# ----------------------------- adaptive beta variants ----------------------- #
# Each returns a per-pixel beta map in [0,1] from the signal maps + params.
def beta_tex(sig, p):
    """Content-only: smooth->floor, textured->1. NO degrade term."""
    t = ramp(sig["tex"], p["t_lo"], p["t_hi"])
    return p["floor"] + (1.0 - p["floor"]) * t


def beta_tex_degrade(sig, p):
    """smooth AND lightly-degraded -> floor; textured OR heavily-degraded -> 1.
    drop = smoothness * light_degrade ; beta = 1 - drop*(1-floor)."""
    smoothness = 1.0 - ramp(sig["tex"], p["t_lo"], p["t_hi"])
    light = ramp(sig["lrhf"], p["d_lo"], p["d_hi"])     # high = light degrade
    drop = smoothness * light
    return 1.0 - drop * (1.0 - p["floor"])


def beta_disag(sig, p):
    """smooth AND low-disagreement -> floor (x4 barely departs -> trust blend);
    high-disagreement (x4 adds a lot) -> 1. drop = smoothness*(1-disag_ramp)."""
    smoothness = 1.0 - ramp(sig["tex"], p["t_lo"], p["t_hi"])
    agree = 1.0 - ramp(sig["disag"], p["g_lo"], p["g_hi"])
    drop = smoothness * agree
    return 1.0 - drop * (1.0 - p["floor"])


BETA_VARIANTS = {
    "tex":         (beta_tex,         dict(floor=0.5, t_lo=8.0, t_hi=22.0)),
    "tex_degrade": (beta_tex_degrade, dict(floor=0.5, t_lo=8.0, t_hi=22.0, d_lo=6.0, d_hi=16.0)),
    "disag":       (beta_disag,       dict(floor=0.5, t_lo=8.0, t_hi=22.0, g_lo=2.0, g_hi=8.0)),
}
FIXED_BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def score_seq(restored, gt):
    lp = [M.lpips_dist(r, g) for r, g in zip(restored, gt)]
    ps = [M.psnr(r, g) for r, g in zip(restored, gt)]
    return dict(lpips=float(np.mean(lp)), psnr=float(np.mean(ps)),
                lpips_per=[float(v) for v in lp])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", action="store_true", help="print signal diagnostics and exit")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    windows, degrades = list(meta["windows"].keys()), meta["degrades"]

    cells = {}
    for w in windows:
        for d in degrades:
            z = np.load(os.path.join(CACHE, f"{w}_{d}.npz"))
            cells[(w, d)] = {k: z[k] for k in z.files}

    if args.diag:
        print(f"{'window':12s} {'degrade':8s} {'tex(med/p90)':>16s} {'lrhf(med/p90)':>16s} "
              f"{'disag(med/p90)':>16s}")
        for w in windows:
            for d in degrades:
                sig = signal_maps(cells[(w, d)])
                f = lambda m: f"{np.median(m):5.1f}/{np.percentile(m,90):5.1f}"
                print(f"{w:12s} {d:8s} {f(sig['tex']):>16s} {f(sig['lrhf']):>16s} {f(sig['disag']):>16s}")
        return

    results = {}
    for (w, d), cell in cells.items():
        gt = cell["gt"]
        sig = signal_maps(cell)
        r = {}
        # base models
        r["compact"] = score_seq(cell["compact"], gt)
        r["x4plus"] = score_seq(cell["x4plus"], gt)
        r["bicubic"] = score_seq(cell["bicubic"], gt)
        # fixed-beta sweep (oracle diagnostic)
        for b in FIXED_BETAS:
            seq = [blend(c, x, b) for c, x in zip(cell["compact"], cell["x4plus"])]
            r[f"fix{b:.2f}"] = score_seq(seq, gt)
        # adaptive variants
        for name, (fn, p) in BETA_VARIANTS.items():
            bmap = fn(sig, p)
            seq = [blend(c, x, bmap) for c, x in zip(cell["compact"], cell["x4plus"])]
            rr = score_seq(seq, gt)
            rr["beta_mean"] = float(bmap.mean())
            r[f"adapt_{name}"] = rr
        results[f"{w}|{d}"] = r
        print(f"  [{w:12s}|{d:8s}] x4={r['x4plus']['lpips']:.4f} "
              f"fix.50={r['fix0.50']['lpips']:.4f} "
              f"tex={r['adapt_tex']['lpips']:.4f}(b{r['adapt_tex']['beta_mean']:.2f}) "
              f"texdeg={r['adapt_tex_degrade']['lpips']:.4f}(b{r['adapt_tex_degrade']['beta_mean']:.2f})")

    json.dump(dict(meta=meta, variants={k: v[1] for k, v in BETA_VARIANTS.items()},
                   results=results),
              open(os.path.join(_HERE, "results.json"), "w"), indent=2)
    print("[done] -> results.json")


if __name__ == "__main__":
    main()
