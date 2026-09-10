#!/usr/bin/env python3
"""
R13-E4 cost measurement -- the whole case for deleting the Ruder splat is its price,
so the price is measured, not quoted.

Two levels, both at PRODUCTION resolution (real sample.mp4 LR 640x320, quality mode
scale 4 -> 2560x1280):

  (1) per-call mask op: derisk.occlusion_mask_lr (numpy) and gpu_ops.occlusion_mask_lr
      (torch/MPS, G.sync'd) for reactive / full(+Ruder splat) / reactive+FSR2, on REAL
      codec flows from the clip. Best-of-N, medians reported (thermal swing is ~25%,
      so ratios are the robust takeaway -- handoff Step-7 convention).

  (2) end-to-end reconstruct() ms/frame for occ in {naive, reactive, full, adaptive,
      fsr2} with a bicubic per-frame cache (SR is identical across arms, so excluding
      it isolates the recon cost the mask sits in).
"""
import os
import sys
import time
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import harness as H          # noqa: E402
import fsr2_mask as F        # noqa: E402
import derisk as D           # noqa: E402

SAMPLE = H.SAMPLE
PROD_SCALE = 4               # quality mode: 640x320 -> 2560x1280


def med_interleaved(fns, n, warm=2):
    """Time several closures INTERLEAVED (arm A, arm B, arm A, ... per repetition)
    instead of arm-at-a-time. Three sibling agents were hammering this box (load
    average ~20-29), so a sequential best-of-N would attribute a load swing to
    whichever arm ran during it; interleaving makes every arm see the same load
    distribution, and the RATIO between arms stays meaningful even when the
    absolutes are inflated. Returns {name: (median_ms, min_ms)}."""
    ts = {k: [] for k in fns}
    for _ in range(warm):
        for k, fn in fns.items():
            fn()
    for _ in range(n):
        for k, fn in fns.items():
            t0 = time.perf_counter()
            fn()
            ts[k].append((time.perf_counter() - t0) * 1000.0)
    return {k: (float(np.median(v)), float(np.min(v))) for k, v in ts.items()}


def bench_calls(frames, reps, scale):
    h_lr, w_lr = frames[0][1].shape[:2]
    backbone = D.backbone_indices(frames)
    # pick a mid-window non-anchor P direction with real motion
    cases = []
    for i in backbone:
        p = max([b for b in backbone if b < i], default=None)
        if p is None or frames[i][0] == "I":
            continue
        fx, fy = D.build_lr_flow(frames[i][2], h_lr, w_lr, want="past")
        if np.isfinite(fx).mean() > 0.5:
            pfx, pfy = D.build_lr_flow(frames[p][2], h_lr, w_lr, want="past") \
                if frames[p][2] is not None and len(frames[p][2]) else (None, None)
            cases.append((i, p, fx, fy, pfx, pfy))
        if len(cases) >= 3:
            break
    if not cases:
        raise RuntimeError("no usable P direction found for the microbench")
    out = {}
    i, p, fx, fy, pfx, pfy = cases[len(cases) // 2]
    lr, lr_ref = frames[i][1], frames[p][1]
    print(f"[calls] frame {i} <- {p}  LR {w_lr}x{h_lr}  MV coverage "
          f"{np.isfinite(fx).mean():.2f}")
    # ---- numpy (interleaved) ----
    npr = med_interleaved({
        "np_reactive": lambda: F._ORIG_OCC_NP(fx, fy, lr, lr_ref, 16.0, "reactive"),
        "np_full": lambda: F._ORIG_OCC_NP(fx, fy, lr, lr_ref, 16.0, "full"),
        "np_fsr2_term": lambda: F.temporal_mv_divergence(fx, fy, pfx, pfy, scale,
                                                         dilate=True),
    }, reps)
    out.update(npr)
    out["np_splat_only"] = (out["np_full"][0] - out["np_reactive"][0],
                            out["np_full"][1] - out["np_reactive"][1])
    out["np_react+fsr2"] = (out["np_reactive"][0] + out["np_fsr2_term"][0],
                            out["np_reactive"][1] + out["np_fsr2_term"][1])
    # ---- torch / MPS ----
    try:
        import torch                      # noqa: F401
        import gpu_ops as G
    except ImportError as e:
        print(f"[calls] torch path unavailable: {e!r}")
        return out
    if not G.device():
        print("[calls] no GPU device")
        return out
    fxd, fyd = G.flow_to_dev(fx, fy)
    lrd, lrrd = G.img_to_dev(lr), G.img_to_dev(lr_ref)
    pfxd, pfyd = (G.flow_to_dev(pfx, pfy) if pfx is not None else (None, None))
    orig_t = F._ORIG_OCC_T or G.occlusion_mask_lr   # pristine gpu_ops fn (unpatched)

    def t_react():
        orig_t(fxd, fyd, lrd, lrrd, 16.0, "reactive")
        G.sync()

    def t_full():
        orig_t(fxd, fyd, lrd, lrrd, 16.0, "full")
        G.sync()

    def t_fsr2():
        F.temporal_mv_divergence_torch(fxd, fyd, pfxd, pfyd, scale, dilate=True)
        G.sync()

    tr = med_interleaved({"t_reactive": t_react, "t_full": t_full,
                          "t_fsr2_term": t_fsr2}, reps)
    out.update(tr)
    out["t_splat_only"] = (out["t_full"][0] - out["t_reactive"][0],
                           out["t_full"][1] - out["t_reactive"][1])
    out["t_react+fsr2"] = (out["t_reactive"][0] + out["t_fsr2_term"][0],
                           out["t_reactive"][1] + out["t_fsr2_term"][1])
    return out


def bench_e2e(frames, scale, backends, variants, reps=3):
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * scale, h_lr * scale
    cache = D.build_perframe_cache(frames, w_hd, h_hd, "bicubic")
    res = {}
    for backend in backends:
        for v in variants:
            F.patch()
            F.set_variant(v if v in F._BASE_MODE else "reactive", scale=scale, thr=0.5)
            occ = "naive" if v == "naive" else F.occ_mode_for(v)
            best = 1e9
            for _ in range(reps):
                H._free_gpu()
                t0 = time.perf_counter()
                D.reconstruct(frames, None, scale, True, occ, cache, set(),
                              backend=backend, collect_metrics=False,
                              download_output=False)
                dt = (time.perf_counter() - t0) * 1000.0 / len(frames)
                best = min(best, dt)
            res[f"{backend}/{v}"] = best   # best-of-N: least contended pass
            print(f"  [e2e {backend:5}/{v:11}] {best:7.2f} ms/frame  "
                  f"(splat {D.MASK_FIRES[0]}/{D.MASK_FIRES[1]})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--scale", type=int, default=PROD_SCALE)
    ap.add_argument("--tag", default="bench")
    args = ap.parse_args()
    print(f"[bench] real sample.mp4 window start={args.start} n={args.n} scale={args.scale}")
    frames = D.decode_lr_and_mvs(SAMPLE, start_frame=args.start, max_frames=args.n)
    calls = bench_calls(frames, args.reps, args.scale)
    for k, v in calls.items():
        print(f"  [{k:16}] median {v[0]:7.2f} ms   best {v[1]:7.2f} ms")
    e2e = bench_e2e(frames, args.scale, ["numpy", "torch"],
                    ["naive", "reactive", "full", "adaptive", "fsr2"])
    H.save(dict(config=vars(args), calls={k: list(v) for k, v in calls.items()}, e2e=e2e),
           os.path.join(H.OUT, f"{args.tag}.json"))


if __name__ == "__main__":
    main()
