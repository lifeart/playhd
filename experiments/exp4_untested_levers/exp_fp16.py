#!/usr/bin/env python3
"""
E4 Lever B -- fp16 for the SR network (UNTESTED in prior steps).

Step 7 rejected fp16 for the MASK ops (kernel-launch-bound; scatter_add slower in fp16).
But the SR network itself (a compute-bound conv net) was never fp16-tested. This script
casts the loaded fp32 SR nets to fp16 on MPS (a deepcopy -- sr.py is imported READ-ONLY and
never mutated) and measures, per model:

  (1) latency RATIO fp16 vs fp32 -- best-of-N (min, least-contended) + median, under shared-MPS
      contention. Reported as a SPEEDUP = t_fp32 / t_fp16.
  (2) PSNR(fp16, fp32) on the uint8 SR OUTPUT  -- the PRIMARY fidelity check (a GO needs the
      output visually identical to fp32, i.e. very high PSNR). var-Laplacian sharpness is
      SECONDARY (never trust an NR-sharpness metric alone to judge SR).

Runs on real frames decoded from sample.mp4 (window A). No shared file is modified.
"""
import os
import sys
import gc
import copy
import time

import numpy as np
import cv2
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", "..", "prototype"))
sys.path.insert(0, _PROTO)

import sr as srmod                 # READ-ONLY import
from derisk import decode_lr_and_mvs

SAMPLE = os.path.abspath(os.path.join(_HERE, "..", "..", "sample.mp4"))
ART = os.path.join(_HERE, "artifacts")
os.makedirs(ART, exist_ok=True)

DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
N_REPS = 7                          # best-of-N timing reps per frame (contention-robust: take min)


def free_gpu():
    gc.collect()
    if DEV.type == "mps":
        torch.mps.empty_cache()


def var_lap(img_u8):
    """Variance-of-Laplacian NR sharpness (SECONDARY metric)."""
    g = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def psnr(a_u8, b_u8):
    a = a_u8.astype(np.float64)
    b = b_u8.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


@torch.no_grad()
def forward_dtype(net, rgb_u8, dtype):
    """Replicate sr.upscale's forward at an explicit dtype. Returns (out_u8, dt_ms).
    Raises on OOM (never silently swallowed) -- caller empties cache + retries smaller."""
    t = torch.from_numpy(np.ascontiguousarray(rgb_u8)).to(DEV)
    t = t.permute(2, 0, 1).unsqueeze(0).to(dtype).div(255.0)
    if DEV.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    out = net(t)
    if DEV.type == "mps":
        torch.mps.synchronize()
    dt = (time.perf_counter() - t0) * 1000.0
    out = out.float().clamp(0.0, 1.0).mul_(255.0).round_()
    out = out.squeeze(0).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return np.ascontiguousarray(out), dt


def timed_bestof(net, rgb_u8, dtype, reps=N_REPS):
    """Warm once (graph compile), then reps timed runs -> (out_u8, min_ms, median_ms)."""
    out, _ = forward_dtype(net, rgb_u8, dtype)     # warm
    times = []
    for _ in range(reps):
        try:
            out, dt = forward_dtype(net, rgb_u8, dtype)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "EAGAIN" in str(e):
                free_gpu()
                # retry once at smaller input (half each dim) -- never silently swallow
                small = cv2.resize(rgb_u8, (rgb_u8.shape[1] // 2, rgb_u8.shape[0] // 2))
                print(f"    [OOM] retried at {small.shape} after empty_cache")
                out, dt = forward_dtype(net, small, dtype)
            else:
                raise
        times.append(dt)
    return out, float(np.min(times)), float(np.median(times))


def run_model(name, frames, frame_idxs):
    print(f"\n=== {name} ===")
    srmod.load_model(name)
    net32 = srmod._MODELS[name]                      # the cached fp32 net (not mutated)
    net16 = copy.deepcopy(net32).half().eval().to(DEV)   # fp16 twin (separate object)

    rows = []
    for fi in frame_idxs:
        rgb = frames[fi][1]
        out32, min32, med32 = timed_bestof(net32, rgb, torch.float32)
        free_gpu()
        out16, min16, med16 = timed_bestof(net16, rgb, torch.float16)
        free_gpu()
        finite16 = np.isfinite(out16.astype(np.float64)).all()
        p = psnr(out16, out32)
        vl32, vl16 = var_lap(out32), var_lap(out16)
        # max per-pixel abs diff (0..255) -- worst-case visible deviation
        maxdiff = int(np.abs(out16.astype(np.int16) - out32.astype(np.int16)).max())
        rows.append(dict(fi=fi, min32=min32, med32=med32, min16=min16, med16=med16,
                         psnr=p, vl32=vl32, vl16=vl16, maxdiff=maxdiff, ok=finite16,
                         speed_best=min32 / min16, speed_med=med32 / med16))
        print(f"  f{fi:>2}: fp32 {med32:7.1f}ms(min {min32:7.1f})  "
              f"fp16 {med16:7.1f}ms(min {min16:7.1f})  "
              f"speedup x{med32/med16:.2f}(best x{min32/min16:.2f})  "
              f"PSNR(16,32)={p:6.2f}dB  maxdiff={maxdiff}  "
              f"varLap 32={vl32:.0f}/16={vl16:.0f}  finite={finite16}")
        # save a visual crop pair on the first frame
        if fi == frame_idxs[0]:
            h, w = out32.shape[:2]
            cy, cx = h // 2, w // 2
            crop32 = out32[cy - 128:cy + 128, cx - 128:cx + 128]
            crop16 = out16[cy - 128:cy + 128, cx - 128:cx + 128]
            diff = np.clip(np.abs(crop16.astype(np.int16) - crop32.astype(np.int16)) * 8,
                           0, 255).astype(np.uint8)
            strip = np.concatenate([crop32, crop16, diff], axis=1)
            cv2.imwrite(os.path.join(ART, f"fp16_{name}_f{fi}_fp32_fp16_diffx8.png"),
                        cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    del net16
    free_gpu()

    # aggregate
    psnrs = [r["psnr"] for r in rows if np.isfinite(r["psnr"])]
    agg = dict(
        name=name,
        speed_best=float(np.median([r["speed_best"] for r in rows])),
        speed_med=float(np.median([r["speed_med"] for r in rows])),
        psnr_min=float(np.min(psnrs)) if psnrs else float("inf"),
        psnr_med=float(np.median(psnrs)) if psnrs else float("inf"),
        maxdiff_max=max(r["maxdiff"] for r in rows),
        vl32=float(np.median([r["vl32"] for r in rows])),
        vl16=float(np.median([r["vl16"] for r in rows])),
        all_finite=all(r["ok"] for r in rows),
        med32=float(np.median([r["med32"] for r in rows])),
        med16=float(np.median([r["med16"] for r in rows])),
    )
    return agg, rows


def main():
    print(f"device={DEV}  N_REPS={N_REPS}  (best-of-N = min, contention-robust)")
    print("decoding window A (start 0) ...")
    frames = decode_lr_and_mvs(SAMPLE, 0, 12)
    print(f"  decoded {len(frames)} LR frames {frames[0][1].shape}")
    frame_idxs = [0, 4, 8]          # a few real frames spread across the window

    results = {}
    # heavy x4plus FIRST (the quality-mode anchor, the most relevant target), then compact
    for name in ["realesrgan-x4plus", "realesrgan"]:
        agg, rows = run_model(name, frames, frame_idxs)
        results[name] = agg

    print("\n================ SUMMARY ================")
    print(f"{'model':<20} {'fp32 ms':>8} {'fp16 ms':>8} {'speedup(med)':>12} "
          f"{'speedup(best)':>13} {'PSNR med':>9} {'PSNR min':>9} {'maxdiff':>7} {'finite':>6}")
    for name, a in results.items():
        print(f"{name:<20} {a['med32']:>8.1f} {a['med16']:>8.1f} "
              f"x{a['speed_med']:>10.2f} x{a['speed_best']:>11.2f} "
              f"{a['psnr_med']:>8.2f} {a['psnr_min']:>8.2f} {a['maxdiff_max']:>7} "
              f"{str(a['all_finite']):>6}")

    # machine-readable dump for the report
    import json
    with open(os.path.join(ART, "fp16_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nartifacts -> {ART}")


if __name__ == "__main__":
    main()
