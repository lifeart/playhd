#!/usr/bin/env python3
"""
R8-E3 step 1: cache the SR outputs (compact, x4plus, bicubic), the degraded LR,
and the GT for the R6-E1 matrix so the (CPU) adaptive-blend tuning can iterate
WITHOUT re-running the heavy x4plus SR. Reuses R6-E1's EXACT decode/degrade/restore
(imported read-only from run_matrix) so my x4plus-alone numbers must reproduce
R6-E1's results.json (a built-in seam check).

Windows (5) x degrades (3) x n frames. Output: cache/<window>_<degrade>.npz with
arrays gt[n,H,W,3], lr[n,h,w,3], compact[n,H,W,3], x4plus[n,H,W,3], bicubic[n,H,W,3].
GPU(MPS) shared -> empty_cache between models. READ-ONLY on prototype/ and r6_e1.
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r5_e2_quality"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r6_e1_srdecision"))
import sr as SR                                  # noqa: E402
from run_matrix import (WINDOWS, DEGRADES, SAMPLE,  # noqa: E402  reuse R6-E1 harness
                        decode_window, degrade, restore, free_gpu)

CACHE = os.path.join(_HERE, "cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)   # R6-E1 used n=8
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    print(f"[setup] decoding {len(WINDOWS)} windows (n={args.n}) ...")
    windows = {k: decode_window(SAMPLE, s, args.n) for k, s in WINDOWS.items()}

    # Precompute LR once per (window,degrade,frame) -- identical input to all models
    # (R6-E1 seed convention: seed=1000+i).
    lr_cache = {}
    for wname, gt in windows.items():
        for dmode in DEGRADES:
            for i, g in enumerate(gt):
                lr_cache[(wname, dmode, i)] = degrade(g, dmode, seed=1000 + i)

    # Run model-by-model (load once, free between) to keep MPS pressure low.
    out = {}   # (wname,dmode) -> dict of arrays
    for model in ("bicubic", "compact", "x4plus"):
        t0 = time.perf_counter()
        for wname, gt in windows.items():
            h, w = gt[0].shape[:2]
            for dmode in DEGRADES:
                res = []
                for i in range(len(gt)):
                    lr = lr_cache[(wname, dmode, i)]
                    res.append(restore(lr, w, h, model))
                out.setdefault((wname, dmode), {})[model] = np.stack(res)
        free_gpu({"compact": "realesrgan", "x4plus": "realesrgan-x4plus"}.get(model))
        print(f"  [{model:8s}] done in {time.perf_counter()-t0:.1f}s")

    for (wname, dmode), d in out.items():
        gt = np.stack(windows[wname])
        lr = np.stack([lr_cache[(wname, dmode, i)] for i in range(len(windows[wname]))])
        path = os.path.join(CACHE, f"{wname}_{dmode}.npz")
        np.savez_compressed(path, gt=gt, lr=lr,
                            compact=d["compact"], x4plus=d["x4plus"], bicubic=d["bicubic"])
    meta = dict(n=args.n, windows=WINDOWS, degrades=DEGRADES,
                gt_varlap={k: float(__import__("cv2").Laplacian(
                    __import__("cv2").cvtColor(windows[k][0], __import__("cv2").COLOR_RGB2GRAY),
                    __import__("cv2").CV_64F).var()) for k in WINDOWS})
    json.dump(meta, open(os.path.join(CACHE, "meta.json"), "w"), indent=2)
    print(f"[done] cached {len(out)} cells -> {CACHE}")


if __name__ == "__main__":
    main()
