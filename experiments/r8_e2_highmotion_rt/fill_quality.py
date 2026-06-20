#!/usr/bin/env python3
"""R8-E2 step 3 (Hypothesis 3): is there a CHEAPER-THAN-SR fallback fill that beats bicubic?

The fallback band is filled by upscaling the CURRENT LR frame (warp is invalid there). Options
on the cost/sharpness spectrum: bicubic (deployed, softest, tOF-optimal) -> lanczos -> bicubic+
unsharp -> compact-SR (sharpest, shimmers). A GO needs FULL-REF improvement at tOF <= bicubic.

FULL-REFERENCE proxy via synthetic downscale (clean, isolates fill quality from warp):
  HD_truth = decoded sample frame (640x320);  LR = INTER_AREA downscale /2 (320x160);
  fill LR->640x320, compare to HD_truth: PSNR, SSIM, LPIPS  +  temporal dF (mean |frame_t -
  frame_{t-1}| of the fill sequence -- the raw flicker the fallback injects).
Run on the instant-relevant high-motion windows (A=0, H2=2352) + extreme (H1=7392).

READ-ONLY. compact-SR via prototype/sr.py (downscaled to the 2x target via upscale_to).
"""
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

import derisk as D    # noqa: E402
import sr as SR       # noqa: E402
import torch          # noqa: E402
import lpips          # noqa: E402

CLIP = os.path.join(_REPO, "sample.mp4")
N = 32
WINDOWS = [("A(0)", 0), ("H2(2352)", 2352), ("H1(7392)", 7392)]
_LPIPS = None


def _lp():
    global _LPIPS
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net="alex", verbose=False)
    return _LPIPS


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a, b):
    """Gaussian-windowed SSIM (grayscale), cv2 only (no skimage)."""
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k, s = (11, 11), 1.5
    mu_a = cv2.GaussianBlur(a, k, s); mu_b = cv2.GaussianBlur(b, k, s)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.GaussianBlur(a * a, k, s) - mu_a2
    sb = cv2.GaussianBlur(b * b, k, s) - mu_b2
    sab = cv2.GaussianBlur(a * b, k, s) - mu_ab
    ss = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return float(ss.mean())


def lpips_dist(a, b):
    def t(x):
        return torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        return float(_lp()(t(a), t(b)).item())


def fills(lr, w, h):
    """LR uint8 -> dict of fill_name -> HD uint8 (target w x h)."""
    bic = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    lan = cv2.resize(lr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    # bicubic + light unsharp (amount 0.5, radius from a 1.0-sigma gaussian) -- a cheap HF boost
    blur = cv2.GaussianBlur(bic, (0, 0), 1.0)
    uns = cv2.addWeighted(bic, 1.5, blur, -0.5, 0).astype(np.uint8)
    sr = SR.upscale_to(lr, w, h, model="realesrgan")
    return {"bicubic": bic, "lanczos": lan, "unsharp": uns, "compactSR": sr}


def main():
    SR.load_model("realesrgan")
    out = {"config": dict(N=N), "windows": {}}
    for label, start in WINDOWS:
        frames = D.decode_lr_and_mvs(CLIP, start, N)
        hd = [f[1] for f in frames]                    # 640x320 = HD_truth
        H, W = hd[0].shape[:2]
        lw, lh = W // 2, H // 2
        lr = [cv2.resize(f, (lw, lh), interpolation=cv2.INTER_AREA) for f in hd]

        seq = {k: [] for k in ("bicubic", "lanczos", "unsharp", "compactSR")}
        for i in range(N):
            for k, v in fills(lr[i], W, H).items():
                seq[k].append(v)

        rec = {}
        for k in seq:
            ps = float(np.mean([psnr(seq[k][i], hd[i]) for i in range(N)]))
            ss = float(np.mean([ssim(seq[k][i], hd[i]) for i in range(N)]))
            lp = float(np.mean([lpips_dist(seq[k][i], hd[i]) for i in range(N)]))
            # temporal flicker of the FILL sequence (downscaled to LR to match motion truth scale)
            dF = float(np.mean([np.abs(cv2.resize(seq[k][t], (lw, lh)).astype(np.float32)
                                       - cv2.resize(seq[k][t - 1], (lw, lh)).astype(np.float32)).mean()
                                for t in range(1, N)]))
            rec[k] = dict(psnr=round(ps, 3), ssim=round(ss, 4), lpips=round(lp, 4), dF=round(dF, 3))
        # reference: temporal dF of the true HD (downscaled) -- the motion floor
        dF_truth = float(np.mean([np.abs(cv2.resize(hd[t], (lw, lh)).astype(np.float32)
                                         - cv2.resize(hd[t - 1], (lw, lh)).astype(np.float32)).mean()
                                  for t in range(1, N)]))
        rec["_dF_truth"] = round(dF_truth, 3)
        out["windows"][label] = rec

        print(f"\n=== {label}  (HD {W}x{H} truth, LR {lw}x{lh}) ===")
        print(f"   {'fill':>10}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'dF':>8}")
        for k in ("bicubic", "lanczos", "unsharp", "compactSR"):
            r = rec[k]
            print(f"   {k:>10}{r['psnr']:>8.2f}{r['ssim']:>8.4f}{r['lpips']:>8.4f}{r['dF']:>8.2f}")
        print(f"   {'(truth dF)':>10}{'':>8}{'':>8}{'':>8}{dF_truth:>8.2f}")

    with open(os.path.join(_HERE, "fill_quality.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(_HERE, 'fill_quality.json')}")


if __name__ == "__main__":
    main()
