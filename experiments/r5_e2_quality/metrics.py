#!/usr/bin/env python3
"""
Full-reference quality metrics for R5-E2 (degrade-restore SR quality).

LEAD metric is a TRUE perceptual one: LPIPS (AlexNet backbone, the `lpips` pip
package -- installed for this experiment; weights cached). The NR var-Laplacian
("sharpness") is computed too but reported ONLY as a secondary number, never the
headline -- per GOTCHA #23 (diffusion faked "13x sharper" by var-Lap while
hallucinating). PSNR / SSIM / MS-SSIM(3-scale) / gradient-fidelity are the
full-reference structural numbers against the SD pseudo-GT.

All images are uint8 HxWx3 RGB. LPIPS runs on CPU (keep MPS free for the SR nets,
which are shared with a sibling process).
"""
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch

# ----------------------------- PSNR ----------------------------------------- #
def psnr(a, b, maxval=255.0):
    """PSNR over RGB (dB). a,b uint8 HxWx3."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(maxval * maxval / mse))


# ----------------------------- SSIM (Gaussian window, luma) ----------------- #
_C1 = (0.01 * 255) ** 2
_C2 = (0.03 * 255) ** 2


def _ssim_map(x, y):
    """SSIM mean over a single-scale luma pair (float32 HxW, 0..255)."""
    win = (11, 11)
    sig = 1.5
    mu_x = cv2.GaussianBlur(x, win, sig)
    mu_y = cv2.GaussianBlur(y, win, sig)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = cv2.GaussianBlur(x * x, win, sig) - mu_x2
    sy = cv2.GaussianBlur(y * y, win, sig) - mu_y2
    sxy = cv2.GaussianBlur(x * y, win, sig) - mu_xy
    ssim = ((2 * mu_xy + _C1) * (2 * sxy + _C2)) / (
        (mu_x2 + mu_y2 + _C1) * (sx + sy + _C2)
    )
    return float(ssim.mean())


def _luma(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def ssim(a, b):
    """Single-scale luma SSIM (0..1, higher=better)."""
    return _ssim_map(_luma(a), _luma(b))


def ms_ssim(a, b, scales=3):
    """3-scale MS-SSIM proxy (renormalized weights for 640x320-class frames; full
    5-scale needs >=161px min-dim which 320-height fails). Higher=better."""
    w = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])[:scales]
    w = w / w.sum()
    x, y = _luma(a), _luma(b)
    vals = []
    for i in range(scales):
        vals.append(_ssim_map(x, y))
        if i < scales - 1:
            x = cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2), interpolation=cv2.INTER_AREA)
            y = cv2.resize(y, (y.shape[1] // 2, y.shape[0] // 2), interpolation=cv2.INTER_AREA)
    vals = np.clip(np.array(vals), 1e-6, 1.0)
    return float(np.prod(vals ** w))


# ------------------- gradient / edge-fidelity (full-ref proxy) -------------- #
def grad_fidelity(a, b):
    """PSNR-in-dB of the Sobel gradient-MAGNITUDE maps (full-reference EDGE
    fidelity). Unlike NR var-Lap this is referenced to the GT, so it cannot be
    gamed by hallucinated HF -- fake edges that are not in the GT LOWER it."""
    ga = _grad_mag(_luma(a))
    gb = _grad_mag(_luma(b))
    mse = np.mean((ga - gb) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _grad_mag(y):
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


# ----------------------------- var-Laplacian (NR, SECONDARY) ---------------- #
def var_laplacian(rgb):
    """No-reference sharpness. SECONDARY ONLY -- never the headline (GOTCHA #23)."""
    return float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


# ----------------------------- LPIPS (TRUE perceptual, lead) ---------------- #
_LPIPS_NET = None


def _lpips_net():
    global _LPIPS_NET
    if _LPIPS_NET is None:
        import lpips
        _LPIPS_NET = lpips.LPIPS(net="alex", verbose=False).eval()
    return _LPIPS_NET


def _to_lpips(rgb):
    t = torch.from_numpy(np.ascontiguousarray(rgb)).float().div_(255.0)
    return t.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W in [0,1]


@torch.no_grad()
def lpips_dist(a, b):
    """TRUE perceptual LPIPS distance (AlexNet). LOWER = perceptually closer. CPU."""
    net = _lpips_net()
    da = net(_to_lpips(a), _to_lpips(b), normalize=True)  # normalize=True -> expects [0,1]
    return float(da)


# ----------------------------- tOF (temporal, secondary) -------------------- #
def _farneback(a, b):
    return cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0)


def tof(seq, ref):
    """TecoGAN tOF: mean Farneback-flow EPE between candidate and reference
    sequence (lower=steadier). Matches prototype/derisk.tof exactly."""
    vals = []
    for t in range(1, len(seq)):
        d = _farneback(ref[t - 1], ref[t]) - _farneback(seq[t - 1], seq[t])
        vals.append(float(np.mean(np.sqrt(np.sum(d * d, axis=-1)))))
    return float(np.mean(vals)) if vals else float("nan")


def mean_full_ref(restored, gt):
    """All single-frame full-ref + NR metrics averaged over a sequence."""
    out = {k: [] for k in ("psnr", "ssim", "ms_ssim", "grad_fid", "lpips", "varlap")}
    for r, g in zip(restored, gt):
        out["psnr"].append(psnr(r, g))
        out["ssim"].append(ssim(r, g))
        out["ms_ssim"].append(ms_ssim(r, g))
        out["grad_fid"].append(grad_fidelity(r, g))
        out["lpips"].append(lpips_dist(r, g))
        out["varlap"].append(var_laplacian(r))
    return {k: float(np.mean(v)) for k, v in out.items()}
