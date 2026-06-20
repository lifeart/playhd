#!/usr/bin/env python3
"""R8-E2 step 6: real-time cost of the unsharp fill -- on-device (the deployed GPU-resident
cache path) and CPU, best-of-N (shared MPS, report as the per-non-anchor add-on vs the ~31ms
720p warp floor). The kill criterion is ms/frame.
"""
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
H, W = 640, 1280   # 720p instant HD (640x320 LR -> x2)


def _gauss1d(sigma, dev):
    r = max(1, int(3 * sigma + 0.5))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=dev)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    return (k / k.sum()).view(1, 1, -1)


def gpu_unsharp(bic, amount, sigma=1.0):
    """bic: [1,3,H,W] float (0..255) on device. Separable gaussian blur + lerp; clamp."""
    k = _gauss1d(sigma, bic.device)
    c = bic.shape[1]
    kx = k.expand(c, 1, 1, k.shape[-1])
    ky = k.view(1, 1, -1, 1).expand(c, 1, k.shape[-1], 1)
    pad = k.shape[-1] // 2
    b = F.conv2d(bic, kx, padding=(0, pad), groups=c)
    b = F.conv2d(b, ky, padding=(pad, 0), groups=c)
    return (bic + amount * (bic - b)).clamp_(0.0, 255.0)


def best_ms(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    if DEV.type == "mps":
        torch.mps.synchronize()
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if DEV.type == "mps":
            torch.mps.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def main():
    bic = (torch.rand(1, 3, H, W, device=DEV) * 255.0)
    # reference: the existing per-non-anchor bicubic upscale (F.interpolate from LR)
    lr = torch.rand(1, 3, H // 2, W // 2, device=DEV) * 255.0
    interp = lambda: F.interpolate(lr, size=(H, W), mode="bicubic", align_corners=False)
    uns05 = lambda: gpu_unsharp(bic, 0.5)
    full = lambda: gpu_unsharp(F.interpolate(lr, size=(H, W), mode="bicubic", align_corners=False), 0.5)

    ms_interp = best_ms(interp)
    ms_unsharp = best_ms(uns05)
    ms_full = best_ms(full)
    print(f"device={DEV}  HD={W}x{H} (720p instant tier)")
    print(f"  existing bicubic F.interpolate (per non-anchor): {ms_interp:.3f} ms")
    print(f"  gpu_unsharp alone (blur+lerp):                    {ms_unsharp:.3f} ms")
    print(f"  bicubic+unsharp combined:                         {ms_full:.3f} ms")
    print(f"  ADD-ON of unsharp over today's bicubic:           {ms_full - ms_interp:+.3f} ms/non-anchor")
    print(f"  vs ~31 ms 720p warp floor -> +{100*(ms_full-ms_interp)/31.0:.2f}% per non-anchor frame")

    # CPU reference (cv2) for the non-gpu_cache path
    import numpy as _np
    bic_np = (_np.random.rand(H, W, 3) * 255).astype(_np.uint8)
    def cpu_unsharp():
        bl = cv2.GaussianBlur(bic_np, (0, 0), 1.0)
        return cv2.addWeighted(bic_np, 1.5, bl, -0.5, 0)
    best = float("inf")
    for _ in range(20):
        t0 = time.perf_counter(); cpu_unsharp(); best = min(best, (time.perf_counter() - t0) * 1000)
    print(f"  CPU cv2 unsharp (non-gpu_cache path):             {best:.3f} ms/non-anchor")


if __name__ == "__main__":
    main()
