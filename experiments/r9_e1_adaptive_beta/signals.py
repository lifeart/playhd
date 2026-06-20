#!/usr/bin/env python3
"""R9-E1 shared module: the blend op, the per-cell fine beta-sweep scorer, and the
NO-REFERENCE degrade/content signal battery (all computed from the LR input + the two
cheap SR caches that the product already has -- NEVER from the GT).

Signals are GLOBAL per-clip scalars (one number per cell), per the R9-E1 hypothesis
(a per-CLIP global beta, NOT per-pixel). LEAD quality metric is TRUE AlexNet LPIPS.
"""
import os, sys
import numpy as np, cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r5_e2_quality"))
import metrics as M  # noqa: E402  TRUE LPIPS + PSNR

BETAS = [round(0.50 + 0.05 * i, 2) for i in range(11)]  # 0.50..1.00 step .05


# ----------------------------- blend (== shipped derisk.blend_anchor_cache math) ---- #
def blend(c, x, b):
    return np.clip(np.round(c.astype(np.float32) + b * (x.astype(np.float32) - c.astype(np.float32))),
                   0, 255).astype(np.uint8)


def _luma(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


# ----------------------------- NO-REFERENCE signal battery -------------------------- #
def _immerkaer_sigma(y):
    """Immerkaer (1996) noise std estimate (texture-suppressing Laplacian mask)."""
    N = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(y, cv2.CV_32F, N, borderType=cv2.BORDER_REPLICATE)
    H, W = y.shape
    return float(np.sqrt(np.pi / 2.0) / (6.0 * (W - 2) * (H - 2)) * np.abs(conv).sum())


def _noise_mad(y):
    """Texture-robust noise proxy: MAD of (y - median3x3) over the FLATTEST 25% of
    7x7 windows (low local-std regions = where noise dominates over texture)."""
    med = cv2.medianBlur(y.astype(np.uint8), 3).astype(np.float32)
    resid = y - med
    mu = cv2.boxFilter(y, -1, (7, 7))
    var = np.maximum(cv2.boxFilter(y * y, -1, (7, 7)) - mu * mu, 0.0)
    lstd = np.sqrt(var)
    thr = np.percentile(lstd, 25)
    flat = lstd <= thr
    r = resid[flat]
    if r.size < 16:
        return float(np.median(np.abs(resid - np.median(resid))) * 1.4826)
    return float(np.median(np.abs(r - np.median(r))) * 1.4826)


def _hf_ratio(y):
    """Fraction of spectral energy above 0.25 Nyquist (sharpness/HF content)."""
    F = np.fft.fftshift(np.abs(np.fft.fft2(y - y.mean())))
    H, W = y.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    tot = F.sum() + 1e-9
    return float(F[r > 0.25].sum() / tot)


def _edge_density(y):
    """recommend_mode's signal: Canny edge density on LR (content/texture)."""
    return float((cv2.Canny(y.astype(np.uint8), 80, 160) > 0).mean())


def _tenengrad(y):
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def cell_signals(cell):
    """All GLOBAL no-reference signals for a cell (dict of arrays gt/lr/compact/x4plus).
    Computed on the LR (the only thing available at inference) + the two SR caches."""
    lr, comp, x4 = cell["lr"], cell["compact"], cell["x4plus"]
    n = lr.shape[0]
    sig = dict(edge=0.0, lapvar=0.0, immerk=0.0, noise_mad=0.0, hf_ratio=0.0,
               lrhf=0.0, tenengrad=0.0, disag_hr=0.0,
               edge_comp=0.0, tex_comp=0.0)
    for i in range(n):
        ylr = _luma(lr[i])
        # degrade-ROBUST content signals: measured on the DENOISED compact SR output
        ycomp = _luma(comp[i])
        sig["edge_comp"] += _edge_density(ycomp)
        muc = cv2.boxFilter(ycomp, -1, (7, 7))
        varc = np.maximum(cv2.boxFilter(ycomp * ycomp, -1, (7, 7)) - muc * muc, 0.0)
        sig["tex_comp"] += float(np.sqrt(varc).mean())
        sig["edge"] += _edge_density(ylr)
        sig["lapvar"] += float(cv2.Laplacian(ylr, cv2.CV_32F).var())
        sig["immerk"] += _immerkaer_sigma(ylr)
        sig["noise_mad"] += _noise_mad(ylr)
        sig["hf_ratio"] += _hf_ratio(ylr)
        mu = cv2.boxFilter(ylr, -1, (7, 7))
        var = np.maximum(cv2.boxFilter(ylr * ylr, -1, (7, 7)) - mu * mu, 0.0)
        sig["lrhf"] += float(np.sqrt(var).mean())
        sig["tenengrad"] += _tenengrad(ylr)
        # global model disagreement (HR scale) -- mean |x4 - compact| luma
        sig["disag_hr"] += float(np.abs(_luma(x4[i]) - _luma(comp[i])).mean())
    for k in sig:
        sig[k] /= n
    # derived: disagreement normalised by content texture (degrade-ish per unit edge)
    sig["disag_per_edge"] = sig["disag_hr"] / (sig["edge"] * 1000 + 1e-6)
    sig["noise_per_edge"] = sig["noise_mad"] / (sig["edge"] * 100 + 1e-3)
    return sig


# ----------------------------- per-cell beta sweep (LPIPS) -------------------------- #
def sweep_cell_lpips(cell, betas=BETAS):
    """Return {beta: mean LPIPS}, plus per-frame x4plus LPIPS for no-regression tests."""
    gt, comp, x4 = cell["gt"], cell["compact"], cell["x4plus"]
    out = {}
    for b in betas:
        seq = [blend(c, x, b) for c, x in zip(comp, x4)]
        out[b] = float(np.mean([M.lpips_dist(r, g) for r, g in zip(seq, gt)]))
    x4_per = [M.lpips_dist(x, g) for x, g in zip(x4, gt)]
    return out, x4_per


def lpips_at_beta_perframe(cell, b):
    gt, comp, x4 = cell["gt"], cell["compact"], cell["x4plus"]
    seq = [blend(c, x, b) for c, x in zip(comp, x4)]
    return [M.lpips_dist(r, g) for r, g in zip(seq, gt)]
