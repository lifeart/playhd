#!/usr/bin/env python3
"""
R8-E4: perceptual-metric TRIANGULATION. Re-score the R6-E1 SR A/B (and add a
fixed-0.5 compact/x4plus blend arm) with a TEXTURE-AWARE FR metric (DISTS) on top
of LPIPS + PSNR, to test whether ANY shipped quality decision rests on LPIPS alone.

Protocol is IDENTICAL to R6-E1 (experiments/r6_e1_srdecision/run_matrix.py): same
5 windows, same 3 degrade operators, same restore path, same per-(window,degrade,
frame) precomputed LR fed to every arm -> directly comparable. The ONLY change is
the metric set: we add DISTS (pyiqa) + a pyiqa-LPIPS cross-check, keep AlexNet
LPIPS (the project's lead) and PSNR (the non-learned anchor), and add the blend arm.

READ-ONLY import of prototype/sr.py, R5-E2 metrics.py, R6-E1 run_matrix.py (decode/
degrade/restore reused verbatim). All outputs stay in THIS dir. No empty catch.
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))                       # read-only
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r5_e2_quality"))    # read-only metrics
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r6_e1_srdecision"))  # reuse harness verbatim

import sr as SR            # noqa: E402
import metrics as M        # noqa: E402  (LPIPS-alex lead + PSNR + SSIM)
import run_matrix as R6    # noqa: E402  (decode_window/degrade/restore/free_gpu/WINDOWS/DEGRADES)
import metrics_extra as MX  # noqa: E402  (DISTS + pyiqa-LPIPS)

SAMPLE = R6.SAMPLE
WINDOWS = R6.WINDOWS                     # 5 windows, identical to R6-E1
DEGRADES = R6.DEGRADES                   # moderate / heavy / gritty


def blend_half(compact_hr, x4plus_hr):
    """Fixed-0.5 linear blend (R6-E3 blend_linear, gain=0.5): compact+0.5*(x4plus-compact)."""
    c = compact_hr.astype(np.float32)
    x = x4plus_hr.astype(np.float32)
    return np.clip(np.round(0.5 * (c + x)), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    # ---- preprocessing-validity gate (must pass before any DISTS-vs-LPIPS verdict) ----
    sc = MX.selfcheck()
    print(f"[selfcheck] DISTS(x,x)={sc['dists_self']:.5f}  DISTS(blur,x)={sc['dists_blur']:.4f}  "
          f"LPIPS(x,x)={sc['lpips_self']:.5f}  LPIPS(blur,x)={sc['lpips_blur']:.4f}")
    assert sc["dists_self"] < 1e-3, "DISTS identity not ~0 -> preprocessing/range bug"
    assert sc["lpips_self"] < 1e-3, "LPIPS identity not ~0 -> preprocessing/range bug"
    assert sc["dists_blur"] > sc["dists_self"], "DISTS not monotone w/ blur -> range bug"

    print(f"[setup] decoding {len(WINDOWS)} windows (n={args.n}) ...")
    windows = {k: R6.decode_window(SAMPLE, s, args.n) for k, s in WINDOWS.items()}
    for k, v in windows.items():
        import cv2
        gu = cv2.cvtColor(v[0], cv2.COLOR_RGB2GRAY)
        print(f"  {k:11s} @{WINDOWS[k]:6d}: {len(v)}f  GT varLap={cv2.Laplacian(gu,cv2.CV_64F).var():.0f}")

    # Precompute the degraded LR ONCE per (window,degrade,frame): identical input to every arm.
    lr_cache = {}
    for wname, gt in windows.items():
        for dmode in DEGRADES:
            for i, g in enumerate(gt):
                lr_cache[(wname, dmode, i)] = R6.degrade(g, dmode, seed=1000 + i)

    # Restore HR for each base model, cache (window,degrade,model)->[HR] for the blend arm.
    hr = {}
    timing = {}
    for model in ("bicubic", "compact", "x4plus"):
        t_all = []
        for wname, gt in windows.items():
            h, w = gt[0].shape[:2]
            for dmode in DEGRADES:
                seq = []
                for i, g in enumerate(gt):
                    lr = lr_cache[(wname, dmode, i)]
                    t0 = time.perf_counter()
                    r = R6.restore(lr, w, h, model)
                    t_all.append((time.perf_counter() - t0) * 1000.0)
                    seq.append(r)
                hr[(wname, dmode, model)] = seq
        timing[model] = float(np.mean(t_all))
        R6.free_gpu({"compact": "realesrgan", "x4plus": "realesrgan-x4plus"}.get(model))
        print(f"  [restored] {model:8s} {timing[model]:.1f} ms/frame (ratio only)")

    # Build the blend arm from cached HR (no extra SR; CPU combine only).
    for wname, gt in windows.items():
        for dmode in DEGRADES:
            c = hr[(wname, dmode, "compact")]
            x = hr[(wname, dmode, "x4plus")]
            hr[(wname, dmode, "blend05")] = [blend_half(ci, xi) for ci, xi in zip(c, x)]
    timing["blend05"] = timing["compact"] + timing["x4plus"]

    ARMS = ("bicubic", "compact", "x4plus", "blend05")

    # Score every arm with per-frame {LPIPS-alex, DISTS, LPIPS-pyiqa, PSNR, SSIM}.
    results = {}
    for wname, gt in windows.items():
        for dmode in DEGRADES:
            for arm in ARMS:
                seq = hr[(wname, dmode, arm)]
                per = {"lpips": [], "dists": [], "lpips_vgg": [], "psnr": [], "ssim": []}
                for r, g in zip(seq, gt):
                    per["lpips"].append(M.lpips_dist(r, g))     # AlexNet (project lead)
                    per["dists"].append(MX.dists(r, g))         # texture-aware FR (NEW)
                    per["lpips_vgg"].append(MX.lpips_pyiqa(r, g))  # same-backend cross-check
                    per["psnr"].append(M.psnr(r, g))            # non-learned anchor
                    per["ssim"].append(M.ssim(r, g))
                cell = {k: float(np.mean(v)) for k, v in per.items()}
                cell["per"] = per                                # keep per-frame for win-rate
                results.setdefault(wname, {}).setdefault(dmode, {})[arm] = cell
                print(f"  [{arm:8s}|{dmode:8s}] {wname:11s} "
                      f"LPIPSa={cell['lpips']:.4f} DISTS={cell['dists']:.4f} "
                      f"LPIPSv={cell['lpips_vgg']:.4f} PSNR={cell['psnr']:.2f} SSIM={cell['ssim']:.4f}")
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    out = dict(n=args.n, windows=WINDOWS, degrades=DEGRADES, arms=list(ARMS),
               results=results, timing_ms_per_frame=timing, selfcheck=sc)
    with open(os.path.join(_HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n[done] -> results.json  timing(ratio): "
          f"compact=1x x4plus={timing['x4plus']/timing['compact']:.1f}x")


if __name__ == "__main__":
    main()
