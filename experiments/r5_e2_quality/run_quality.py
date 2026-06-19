#!/usr/bin/env python3
"""
R5-E2: honest perceptual quality numbers for the playhd SR configs + grain.

PROTOCOL (degrade-and-restore -- the project has no true HD GT for sample.mp4):
  * Take the SD frames (640x320) as pseudo-HD GROUND TRUTH.
  * DEGRADE: downscale 2x with INTER_AREA -> 320x160 "SD-to-upscale".
  * RESTORE: re-upscale 2x back to 640x320 through the SAME path the deployed
    pipeline uses for non-x4 scales -- prototype.sr.upscale_to (x4 SR then
    INTER_CUBIC resize to the target). bicubic baseline = cv2 INTER_CUBIC 2x.
  * SCORE restored vs the SD GT with FULL-REFERENCE metrics led by TRUE LPIPS
    (AlexNet), plus PSNR / SSIM / MS-SSIM / gradient-fidelity, var-Lap (NR,
    secondary), and tOF (temporal, secondary).

Two windows: talking-head (start 5000) and high-motion (start 0).
SR configs: bicubic | compact (realesr-general-x4v3) | x4plus (RRDBNet) |
            x4plus-fp16 (R2-E4).
Then grain A/B (off/low/med/high) on each mode's SR config, scored vs GT.

GPU (MPS) is SHARED with a sibling -> small windows, empty_cache between SR
models, timing reported as RATIOS not absolute fps.
READ-ONLY import of prototype/sr.py + grain.py; outputs only in this dir.
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
sys.path.insert(0, os.path.join(_ROOT, "prototype"))   # read-only prototype import

import sr as SR          # noqa: E402  prototype/sr.py
import grain as GRAIN    # noqa: E402  prototype/grain.py
import metrics as M      # noqa: E402  local

SAMPLE = os.path.join(_ROOT, "sample.mp4")


# --------------------------------------------------------------------------- #
def decode_window(path, start_frame, n):
    """Decode n RGB frames (640x320) starting at start_frame, display order."""
    cont = av.open(path)
    vs = cont.streams.video[0]
    out = []
    idx = 0
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


def degrade(gt, mode="clean"):
    """SD GT (HxW) -> degraded LR at half size.

    mode="clean": a clean INTER_AREA 2x antialias downscale. This is OUT OF
        DISTRIBUTION for Real-ESRGAN (trained to invert blur+noise+JPEG), so it
        isolates the SR nets' HALLUCINATION COST against a pristine low-pass GT --
        a hard, slightly-unfair-to-SR fidelity floor.
    mode="real": emulates how SD is ACTUALLY delivered (the deployed scenario) --
        codec softening (Gaussian blur) -> 2x downscale -> low-bitrate blocking/
        ringing (JPEG q40) -> sensor/grain noise. This is what the SR nets are
        BUILT to invert, so it is the representative test for playhd's real input.
    Both are deterministic/reproducible. GT stays the pristine 640x320 SD frame."""
    h, w = gt.shape[:2]
    if mode == "clean":
        return cv2.resize(gt, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    if mode == "real":
        x = cv2.GaussianBlur(gt, (0, 0), 0.8)                       # codec/lens softening
        x = cv2.resize(x, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), 40])  # blocking/ringing
        if ok:
            x = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        rng = np.random.default_rng(12345)                          # deterministic noise
        x = np.clip(x.astype(np.float32) + rng.normal(0, 2.0, x.shape), 0, 255).astype(np.uint8)
        return x
    raise ValueError(mode)


def restore(lr, w, h, config):
    """Re-upscale degraded LR back to (w,h) by config."""
    if config == "bicubic":
        return cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    if config == "compact":
        return SR.upscale_to(lr, w, h, model="realesrgan", half=False)
    if config == "x4plus":
        return SR.upscale_to(lr, w, h, model="realesrgan-x4plus", half=False)
    if config == "x4plus-fp16":
        return SR.upscale_to(lr, w, h, model="realesrgan-x4plus", half=True)
    raise ValueError(config)


def free_gpu(model_name=None, half=None):
    """Drop a cached SR net + empty MPS cache (good GPU citizen; sibling shares it)."""
    if model_name is not None:
        for key in list(SR._MODELS.keys()):
            if key[0] == model_name and (half is None or key[1] == half):
                del SR._MODELS[key]
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


SR_CONFIGS = ["bicubic", "compact", "x4plus", "x4plus-fp16"]


# --------------------------------------------------------------------------- #
def run_sr_ab(windows, n, dmode):
    """SR-config A/B on the degrade-restore task. Returns nested dict + keeps the
    restored x4plus fp32/fp16 sequences for the fp16-identity check."""
    results = {}            # window -> config -> metric dict
    timing = {}             # config -> mean restore ms/frame (ratio use only)
    restored_keep = {}      # (window, config) -> [restored frames]  (for grain+fp16)

    for cfg in SR_CONFIGS:
        t_all = []
        for wname, gt in windows.items():
            h, w = gt[0].shape[:2]
            restored = []
            for g in gt:
                lr = degrade(g, dmode)
                t0 = time.perf_counter()
                r = restore(lr, w, h, cfg)
                t_all.append((time.perf_counter() - t0) * 1000.0)
                restored.append(r)
            restored_keep[(wname, cfg)] = restored
            mt = M.mean_full_ref(restored, gt)
            mt["tof"] = M.tof(restored, gt)
            results.setdefault(wname, {})[cfg] = mt
            print(f"  [{dmode}|{cfg:12s}] {wname:11s} "
                  f"LPIPS={mt['lpips']:.4f} PSNR={mt['psnr']:.2f} SSIM={mt['ssim']:.4f} "
                  f"MS-SSIM={mt['ms_ssim']:.4f} gradFid={mt['grad_fid']:.2f} "
                  f"tOF={mt['tof']:.3f} varLap={mt['varlap']:.0f}")
        timing[cfg] = float(np.mean(t_all))
        if cfg in ("compact",):
            free_gpu("realesrgan")
        if cfg == "x4plus":
            free_gpu("realesrgan-x4plus", half=False)
        if cfg == "x4plus-fp16":
            free_gpu("realesrgan-x4plus", half=True)
    return results, timing, restored_keep


def run_fp16_identity(restored_keep, windows):
    """Verify fp16 is perceptually identical to fp32 (R2-E4 claimed PSNR ~72 dB).
    Compare x4plus-fp16 vs x4plus-fp32 restored frames directly (NOT vs GT)."""
    out = {}
    for wname in windows:
        a = restored_keep[(wname, "x4plus")]
        b = restored_keep[(wname, "x4plus-fp16")]
        ps = float(np.mean([M.psnr(x, y) for x, y in zip(a, b)]))
        lp = float(np.mean([M.lpips_dist(x, y) for x, y in zip(a, b)]))
        ss = float(np.mean([M.ssim(x, y) for x, y in zip(a, b)]))
        out[wname] = dict(psnr=ps, lpips=lp, ssim=ss)
        print(f"  fp16-vs-fp32 [{wname:11s}] PSNR={ps:.1f}dB LPIPS={lp:.5f} SSIM={ss:.5f}")
    return out


GRAIN_LEVELS = ["off", "low", "med", "high"]
# mode -> SR config used in that deployed mode
MODE_CFG = {"instant": "compact", "quality": "x4plus", "layered": "x4plus"}


def run_grain_ab(restored_keep, windows):
    """Grain off/low/med/high applied to each mode's SR-config restoration, scored
    vs GT. Grain is the FINAL pass (prototype.grain.apply_grain), so we re-grain the
    already-restored frames. Question: does a little grain HELP the perceptual
    metric (mask SR banding) even though it adds noise (must hurt PSNR)?"""
    out = {}   # mode -> window -> level -> metric dict
    for mode, cfg in MODE_CFG.items():
        for wname, gt in windows.items():
            base = restored_keep[(wname, cfg)]
            for lvl in GRAIN_LEVELS:
                grained = [GRAIN.apply_grain(b, i, strength=lvl) for i, b in enumerate(base)]
                mt = M.mean_full_ref(grained, gt)
                mt["tof"] = M.tof(grained, gt)
                out.setdefault(mode, {}).setdefault(wname, {})[lvl] = mt
                print(f"  grain[{mode:8s}/{cfg:7s}] {wname:11s} {lvl:4s} "
                      f"LPIPS={mt['lpips']:.4f} PSNR={mt['psnr']:.2f} "
                      f"SSIM={mt['ssim']:.4f} tOF={mt['tof']:.3f} varLap={mt['varlap']:.0f}")
    return out


def save_crops(windows, restored_keep, grain_out):
    """Save a side-by-side strip (GT | bicubic | compact | x4plus) for one frame of
    the talking-head window for the visual record."""
    wname = "talkinghead"
    if wname not in windows:
        wname = list(windows)[0]
    gt = windows[wname][0]
    h, w = gt.shape[:2]
    cy, cx = h // 2, w // 2
    crop = lambda im: im[max(0, cy - 70):cy + 70, max(0, cx - 70):cx + 70]
    tiles = [("GT", gt)]
    for cfg in SR_CONFIGS:
        tiles.append((cfg, restored_keep[(wname, cfg)][0]))
    panels = []
    for label, im in tiles:
        c = cv2.resize(crop(im), (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(c)
    strip = np.hstack(panels)
    cv2.imwrite(os.path.join(_HERE, "crops_sr_ab.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="frames per window")
    args = ap.parse_args()

    print(f"[setup] decoding windows from {SAMPLE} (n={args.n}) ...")
    windows = {
        "talkinghead": decode_window(SAMPLE, 5000, args.n),
        "highmotion": decode_window(SAMPLE, 0, args.n),
    }
    for k, v in windows.items():
        print(f"  {k}: {len(v)} frames @ {v[0].shape[1]}x{v[0].shape[0]} "
              f"(degraded to {v[0].shape[1]//2}x{v[0].shape[0]//2})")

    sr_res, timing, grain_out, restored_all = {}, {}, {}, {}
    for dmode in ("clean", "real"):
        print(f"\n[1:{dmode}] SR-config A/B (degrade-restore, full-ref vs SD GT) ------")
        sr_res[dmode], timing[dmode], rk = run_sr_ab(windows, args.n, dmode)
        restored_all[dmode] = rk
        print(f"\n[3:{dmode}] Grain A/B (off/low/med/high per mode, full-ref vs GT) ---")
        grain_out[dmode] = run_grain_ab(rk, windows)

    print("\n[2] fp16 == fp32 identity check (R2-E4; on 'real' degrade) ------------")
    fp16 = run_fp16_identity(restored_all["real"], windows)

    save_crops(windows, restored_all["real"], grain_out)

    out = dict(n=args.n, sr_ab=sr_res, timing_ms_per_frame=timing,
               fp16_identity=fp16, grain_ab=grain_out, mode_cfg=MODE_CFG)
    with open(os.path.join(_HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n[done] wrote results.json + crops_sr_ab.png")


if __name__ == "__main__":
    main()
