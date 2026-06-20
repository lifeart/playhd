#!/usr/bin/env python3
"""
R6-E3: a NEW SR-quality lever -- can we BEAT the compact model's perceptual
quality (LPIPS) WITHOUT paying the heavy model's hallucination cost?

Premise (from R5-E2, real measured LPIPS, degrade-and-restore): on sample.mp4 the
COMPACT realesr-general-x4v3 BEATS the heavy x4plus because x4plus synthesizes
more HF but MISALIGNED with the true GT (hallucination) -> worse LPIPS at 10x the
compute. Frontier: get MORE true (aligned) detail -> LOWER LPIPS -- without adding
misaligned HF.

PROTOCOL (reused from R5-E2, degrade-and-restore; project has no true HD GT):
  SD frame (640x320) = pseudo-HD GT -> DEGRADE 2x (`real`: codec-soften + 2x down
  + JPEG q40 + noise -- what the SR nets are BUILT to invert) -> RESTORE 2x ->
  SCORE vs GT. LEAD = TRUE LPIPS (AlexNet, `lpips` pkg). var-Lap NR SECONDARY ONLY
  (GOTCHA #23: a lever that only raises var-Lap but not LPIPS is a FAIL).

LEVERS (each vs compact-alone baseline):
  1. TTA / anchor self-ensemble: SR the input + its geometric transforms (4-/8-way
     D4 group), invert + AVERAGE -- averaging cancels the UNCORRELATED hallucinated
     HF (differs per orientation) while reinforcing the ALIGNED true detail. On BOTH
     compact and x4plus. Cost = N x SR (anchor-only -> amortized in propagation).
  2. compact+x4plus blend: out = compact + gain*(x4plus - compact), gain sweep 0..1;
     plus a FREQUENCY-GATED variant (add x4plus HF only where it AGREES in sign with
     compact's HF -> a touch of true detail, suppress divergent hallucination).
  3. Mild unsharp post-sharpen of compact (near-free baseline for the learned levers).

GPU (MPS) SHARED with 2 siblings -> small windows (n=6), free GPU between models,
timing reported as RATIOS. READ-ONLY import of prototype/sr.py + R5-E2 metrics;
outputs only in this dir. No empty catch; ffmpeg CLI broken -> PyAV.
"""
import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import av
import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))                 # read-only
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r5_e2_quality"))  # reuse metrics

import sr as SR        # noqa: E402  prototype/sr.py
import metrics as M    # noqa: E402  experiments/r5_e2_quality/metrics.py (LPIPS lead)

SAMPLE = os.path.join(_ROOT, "sample.mp4")


# --------------------------------------------------------------------------- #
# decode + degrade (reused/aligned with R5-E2 `real` operator)
# --------------------------------------------------------------------------- #
def decode_window(path, start_frame, n):
    cont = av.open(path)
    vs = cont.streams.video[0]
    out, idx = [], 0
    for frame in cont.decode(vs):
        if idx < start_frame:
            idx += 1
            continue
        if len(out) >= n:
            break
        out.append(frame.to_ndarray(format="rgb24"))
        idx += 1
    cont.close()
    return out


def degrade_real(gt):
    """R5-E2 `real`: the representative deployed degrade (nets are built to invert)."""
    x = cv2.GaussianBlur(gt, (0, 0), 0.8)
    h, w = gt.shape[:2]
    x = cv2.resize(x, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 40])
    if ok:
        x = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    rng = np.random.default_rng(12345)
    x = np.clip(x.astype(np.float32) + rng.normal(0, 2.0, x.shape), 0, 255).astype(np.uint8)
    return x


def free_gpu(model_name=None, half=None):
    if model_name is not None:
        for key in list(SR._MODELS.keys()):
            if key[0] == model_name and (half is None or key[1] == half):
                del SR._MODELS[key]
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


# --------------------------------------------------------------------------- #
# D4 geometric self-ensemble (EDSR-style TTA): forward transform, SR, inverse.
# t is a 3-bit code: bit0=hflip, bit1=vflip, bit2=transpose. Each op is an
# involution; the inverse applies them in reverse order. The 8 codes are the
# 8 distinct dihedral-group orientations.
# --------------------------------------------------------------------------- #
def _fwd(x, t):
    if t & 1:
        x = x[:, ::-1]
    if t & 2:
        x = x[::-1, :]
    if t & 4:
        x = np.swapaxes(x, 0, 1)
    return np.ascontiguousarray(x)


def _inv(y, t):
    if t & 4:
        y = np.swapaxes(y, 0, 1)
    if t & 2:
        y = y[::-1, :]
    if t & 1:
        y = y[:, ::-1]
    return np.ascontiguousarray(y)


TTA_SETS = {
    1: [0],                      # identity == plain SR (sanity == baseline)
    4: [0, 1, 2, 4],             # id + hflip + vflip + transpose (spread of orientations)
    8: [0, 1, 2, 3, 4, 5, 6, 7],  # full D4
}


def sr_x4(lr, model, half):
    """x4 SR (1280x640 from a 320x160 LR), uint8."""
    return SR.upscale(lr, model=model, half=half)


def restore_tta(lr, w, h, model, half, n_aug):
    """TTA self-ensemble at the x4 level, then cubic-resize to (w,h)."""
    acc = None
    for t in TTA_SETS[n_aug]:
        lr_t = _fwd(lr, t)
        sr_t = sr_x4(lr_t, model, half)        # x4 in transformed frame
        sr_back = _inv(sr_t, t).astype(np.float64)
        acc = sr_back if acc is None else acc + sr_back
    avg = np.clip(np.round(acc / float(n_aug)), 0, 255).astype(np.uint8)
    if avg.shape[1] != w or avg.shape[0] != h:
        avg = cv2.resize(avg, (w, h), interpolation=cv2.INTER_CUBIC)
    return avg


def restore_plain(lr, w, h, model, half):
    """x4 SR then cubic-resize (== prototype.sr.upscale_to). Returns HR uint8."""
    out = sr_x4(lr, model, half)
    if out.shape[1] != w or out.shape[0] != h:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


# --------------------------------------------------------------------------- #
# Blend levers (operate on already-restored HR frames at (w,h))
# --------------------------------------------------------------------------- #
def blend_linear(compact_hr, x4plus_hr, gain):
    """out = compact + gain*(x4plus - compact). gain=0 -> compact, gain=1 -> x4plus."""
    c = compact_hr.astype(np.float32)
    x = x4plus_hr.astype(np.float32)
    out = c + gain * (x - c)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def blend_freqgated(compact_hr, x4plus_hr, gain, sigma=1.0):
    """Add x4plus's HF only where it AGREES IN SIGN with compact's HF (per channel).
    HF = image - Gaussian(image). Where x4plus & compact disagree on the local HF
    sign (x4plus inventing detail that compact does not see) the extra HF is dropped
    -> keep aligned true detail, suppress divergent hallucination."""
    c = compact_hr.astype(np.float32)
    x = x4plus_hr.astype(np.float32)
    hf_c = c - cv2.GaussianBlur(c, (0, 0), sigma)
    hf_x = x - cv2.GaussianBlur(x, (0, 0), sigma)
    agree = (np.sign(hf_c) == np.sign(hf_x)).astype(np.float32)   # per-pixel,per-channel gate
    out = c + gain * hf_x * agree
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def unsharp(compact_hr, radius, amount):
    """out = img + amount*(img - Gaussian(img, radius)). Near-free CPU post-sharpen."""
    c = compact_hr.astype(np.float32)
    blur = cv2.GaussianBlur(c, (0, 0), radius)
    out = c + amount * (c - blur)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
def score(restored, gt):
    m = M.mean_full_ref(restored, gt)
    m["tof"] = M.tof(restored, gt)
    return m


def fmt(tag, m, extra=""):
    print(f"  {tag:30s} LPIPS={m['lpips']:.4f} PSNR={m['psnr']:.2f} "
          f"SSIM={m['ssim']:.4f} gradFid={m['grad_fid']:.2f} "
          f"varLap={m['varlap']:.0f}{extra}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--gains", type=str, default="0.0,0.25,0.5,0.75,1.0")
    args = ap.parse_args()
    gains = [float(g) for g in args.gains.split(",")]

    print(f"[setup] decoding windows (n={args.n}) ...")
    windows = {
        "talkinghead": decode_window(SAMPLE, 5000, args.n),
        "detailed": decode_window(SAMPLE, 30000, args.n),
    }
    lrs = {}
    for k, gt in windows.items():
        lrs[k] = [degrade_real(g) for g in gt]
        h, w = gt[0].shape[:2]
        print(f"  {k}: {len(gt)} @ {w}x{h}  GT var-Lap="
              f"{np.mean([M.var_laplacian(g) for g in gt]):.0f}")

    results = {}          # window -> { config_label -> metrics }
    timing = {}           # config_label -> mean restore ms/frame (ratio use)
    hr_cache = {}         # (window, 'compact'|'x4plus') -> [HR frames] for blends

    # ---------- Lever 1: TTA on compact and x4plus ({1,4,8}-way) ------------- #
    for model, half, mlabel in [("realesrgan", False, "compact"),
                                ("realesrgan-x4plus", True, "x4plus")]:
        for n_aug in (1, 4, 8):
            label = f"{mlabel}_tta{n_aug}"
            tall = []
            for wname, gt in windows.items():
                h, w = gt[0].shape[:2]
                restored = []
                for lr in lrs[wname]:
                    t0 = time.perf_counter()
                    if n_aug == 1:
                        r = restore_plain(lr, w, h, model, half)
                    else:
                        r = restore_tta(lr, w, h, model, half, n_aug)
                    tall.append((time.perf_counter() - t0) * 1000.0)
                    restored.append(r)
                if n_aug == 1:                         # cache plain HR for the blend lever
                    hr_cache[(wname, mlabel)] = restored
                m = score(restored, gt)
                results.setdefault(wname, {})[label] = m
                fmt(f"[{wname}] {label}", m)
            timing[label] = float(np.mean(tall))
        free_gpu(model, half=half)

    # ---------- Lever 2: compact+x4plus blend (linear sweep + freq-gated) ---- #
    for wname, gt in windows.items():
        h, w = gt[0].shape[:2]
        c_hr = hr_cache[(wname, "compact")]
        x_hr = hr_cache[(wname, "x4plus")]
        for g in gains:
            lab = f"blend_lin_g{g:.2f}"
            restored = [blend_linear(c, x, g) for c, x in zip(c_hr, x_hr)]
            results[wname][lab] = score(restored, gt)
            fmt(f"[{wname}] {lab}", results[wname][lab])
        for g in gains:
            if g == 0.0:
                continue
            lab = f"blend_fg_g{g:.2f}"
            restored = [blend_freqgated(c, x, g) for c, x in zip(c_hr, x_hr)]
            results[wname][lab] = score(restored, gt)
            fmt(f"[{wname}] {lab}", results[wname][lab])
    # blend cost = one compact + one x4plus pass (no extra SR for the CPU combine)
    timing["blend_lin"] = timing["compact_tta1"] + timing["x4plus_tta1"]
    timing["blend_fg"] = timing["blend_lin"]

    # ---------- Lever 3: unsharp post-sharpen of compact (near-free) -------- #
    sharp_grid = [(0.8, 0.3), (0.8, 0.6), (1.2, 0.3), (1.2, 0.6), (1.6, 0.5)]
    for wname, gt in windows.items():
        c_hr = hr_cache[(wname, "compact")]
        for (rad, amt) in sharp_grid:
            lab = f"sharp_r{rad}_a{amt}"
            restored = [unsharp(c, rad, amt) for c in c_hr]
            results[wname][lab] = score(restored, gt)
            fmt(f"[{wname}] {lab}", results[wname][lab])
    timing["sharp"] = timing["compact_tta1"]   # ~= compact + negligible CPU

    # ---------- save ---------- #
    out = dict(n=args.n, gains=gains, windows=list(windows),
               results=results, timing_ms_per_frame=timing,
               baseline="compact_tta1")
    with open(os.path.join(_HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n[done] wrote results.json")


if __name__ == "__main__":
    main()
