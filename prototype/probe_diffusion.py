#!/usr/bin/env python3
"""
Stream-3 feasibility-spike driver. Writes everything to out_diffusion/.

1. MPS feasibility of the SD2/OSEDiff one-step compute pattern (sr_diffusion.mps_feasibility_probe):
   runs N times to bound contention variance, fp16 and fp32.
2. Extracts ONE real 128x128 LR tile from ../sample.mp4 (talking-head anchor frame), and builds
   the quality bar the diffusion anchor would have to beat: bicubic x4 vs RealESRGAN_x4plus (heavy
   GAN anchor, sr.py, local weights) -- crops + var-of-Laplacian sharpness. The diffusion column
   is COMPUTE-verified on MPS but real weights are not downloaded (logistics blocker, documented).
"""
import os
import json
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_diffusion")
os.makedirs(OUT, exist_ok=True)

CROP = 128                   # 128 LR -> 512 HD output tile (x4)
CAND_SECONDS = [240, 360, 480, 600]   # seek targets; 25fps clip -> ~frames 6000/9000/12000/15000


def var_lap(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def best_crop(rgb, c=CROP, stride=32):
    """Return the most-textured cxc crop (max var-of-Laplacian) + its xywh."""
    H, W = rgb.shape[:2]
    best, bxy = -1.0, (0, 0)
    for y in range(0, H - c + 1, stride):
        for x in range(0, W - c + 1, stride):
            v = var_lap(rgb[y:y + c, x:x + c])
            if v > best:
                best, bxy = v, (x, y)
    x, y = bxy
    return np.ascontiguousarray(rgb[y:y + c, x:x + c]), (x, y, c, c), best


def extract_tile():
    """Seek to several timestamps, decode a few frames each, keep the most-textured crop overall."""
    import av
    container = av.open(os.path.join(HERE, "..", "sample.mp4"))
    stream = container.streams.video[0]
    tb = stream.time_base
    best = dict(score=-1.0)
    for sec in CAND_SECONDS:
        try:
            container.seek(int(sec / tb), stream=stream, any_frame=False, backward=True)
        except Exception:
            continue
        for n, frame in enumerate(container.decode(stream)):
            if n > 8:
                break
            rgb = frame.to_ndarray(format="rgb24")
            crop, xywh, score = best_crop(rgb)
            if score > best["score"]:
                best = dict(score=score, full=rgb, crop=crop, xywh=xywh, sec=sec)
    container.close()
    if best["score"] < 0:
        raise RuntimeError("could not decode any candidate frame")
    print(f"[probe] picked crop {best['xywh']} @~{best['sec']}s  var-Lap={best['score']:.1f}")
    return best["full"], best["crop"], best["xywh"], best["sec"]


def run_quality_bar(lr_crop):
    """bicubic x4 vs x4plus heavy GAN on the same 128->512 tile. Returns dict + saves crops."""
    h, w = lr_crop.shape[:2]
    res = {}
    # bicubic
    bic = cv2.resize(lr_crop, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    res["bicubic"] = dict(var_lap=var_lap(bic))
    cv2.imwrite(os.path.join(OUT, "tile_bicubic_x4.png"), cv2.cvtColor(bic, cv2.COLOR_RGB2BGR))
    # heavy GAN x4plus (local weights via sr.py -- imported read-only)
    try:
        import sr
        x4p = sr.upscale(lr_crop, model="realesrgan-x4plus")
        res["x4plus"] = dict(var_lap=var_lap(x4p),
                             latency_ms=float(sr.last_latency_ms("realesrgan-x4plus")))
        cv2.imwrite(os.path.join(OUT, "tile_x4plus.png"), cv2.cvtColor(x4p, cv2.COLOR_RGB2BGR))
        # compact for reference
        cmp = sr.upscale(lr_crop, model="realesrgan")
        res["compact"] = dict(var_lap=var_lap(cmp),
                              latency_ms=float(sr.last_latency_ms("realesrgan")))
        cv2.imwrite(os.path.join(OUT, "tile_compact.png"), cv2.cvtColor(cmp, cv2.COLOR_RGB2BGR))
    except Exception as e:
        res["x4plus"] = dict(error=f"{type(e).__name__}: {e}")
    cv2.imwrite(os.path.join(OUT, "tile_lr.png"), cv2.cvtColor(lr_crop, cv2.COLOR_RGB2BGR))
    return res


def main():
    import sr_diffusion as sd
    summary = dict(diffusers_available=sd.DIFFUSERS_AVAILABLE,
                   diffusers_version=sd.DIFFUSERS_VERSION,
                   fallback_env=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"))

    n_probe = int(os.environ.get("N_PROBE", "2"))
    prev = {}
    if os.path.exists(os.path.join(OUT, "summary.json")):
        with open(os.path.join(OUT, "summary.json")) as f:
            prev = json.load(f)
    if sd.DIFFUSERS_AVAILABLE and n_probe > 0:
        runs = [sd.mps_feasibility_probe(verbose=True) for _ in range(n_probe)]
        summary["mps_probe_fp16"] = runs
        import torch
        summary["mps_probe_fp32"] = sd.mps_feasibility_probe(dtype=torch.float32, verbose=True)
    else:                                  # reuse the already-measured probe
        summary["mps_probe_fp16"] = prev.get("mps_probe_fp16")
        summary["mps_probe_fp32"] = prev.get("mps_probe_fp32")

    # quality bar on a real, auto-selected detailed tile
    try:
        full, crop, xywh, sec = extract_tile()
        cv2.imwrite(os.path.join(OUT, "anchor_frame.png"), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
        summary["quality_bar"] = run_quality_bar(crop)
        summary["quality_bar"]["tile"] = dict(approx_second=sec, crop_xywh=list(xywh),
                                              crop_var_lap=var_lap(crop))
    except Exception as e:
        import traceback; traceback.print_exc()
        summary["quality_bar"] = dict(error=f"{type(e).__name__}: {e}")

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()
