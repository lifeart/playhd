#!/usr/bin/env python3
"""
R10-E2 deliverable: codec-artifact-removal PREPROCESSOR for the x4plus anchor.

Verdict (see REPORT.md): GATED GO. Running a 1x deblock/restoration pass (SCUNet
real_psnr) on a HEAVILY-compressed LR frame BEFORE x4plus is a real ceiling-raise --
on heavy H.264 + low/mid-detail content it beats plain x4plus on LPIPS (-13%), DISTS
(-17%) AND PSNR (+0.5 dB), 4/4 per-frame, with var-Lap == x4plus (artifact-removal,
not blur). It must be GATED: on light compression it strips real detail (loses both
metrics), and on dense photographic texture even at heavy CRF it over-smooths (the
DISTS guard catches it). Hence default-OFF + a compression gate.

This module is standalone (no shared-code edits). It is wired into the anchor SR path
via build_perframe_cache.patch, default-OFF -> byte-identical when off.

Config (read from MODE_CONFIG["quality"]["deblock_pre"], absent/None => disabled):
    {"model": "scunet_color_real_psnr.pth",  # path under this dir's models/
     "gate":  "qp"|"blockiness"|"always",
     "qp_min": 30,            # apply only when bitstream QP >= this (heavy compression)
     "block_min": 1.30,       # OR LR blockiness proxy >= this (when QP unavailable)
     "skip_texture_varlap": null}  # optional: skip if LR var-Lap above X (dense texture guard; off by default)
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_NET = {}


def _device():
    import torch
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_deblocker(model_fname="scunet_color_real_psnr.pth"):
    """Load the SCUNet (or any spandrel scale-1 restoration) deblocker, cached."""
    if model_fname in _NET:
        return _NET[model_fname]
    import torch
    from spandrel import ModelLoader
    path = os.path.join(_HERE, "models", model_fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"deblock weights missing: {path}")
    dev = _device()
    md = ModelLoader(device=dev).load_from_file(path)
    if md.scale != 1:
        raise ValueError(f"deblock model must be scale-1, got scale={md.scale}")
    net = md.model.eval().to(dev)
    _NET[model_fname] = net
    return net


def blockiness(rgb, q=8):
    """Cheap LR-only proxy for H.264 compression severity: ratio of gradient energy AT
    the q-pixel DCT-block grid to the overall gradient energy. > ~1.3 => heavy blocking."""
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    dv = np.abs(np.diff(g, axis=1)); dh = np.abs(np.diff(g, axis=0))
    cols = np.arange(q - 1, dv.shape[1], q); rows = np.arange(q - 1, dh.shape[0], q)
    grid = (dv[:, cols].mean() + dh[rows, :].mean()) / 2.0
    allg = (dv.mean() + dh.mean()) / 2.0
    return float(grid / (allg + 1e-6))


def should_deblock(rgb, cfg, qp=None):
    """Gate: True iff the frame is heavily-enough compressed to benefit from deblocking.

    R12 review: the 'qp' gate is STRICT -- qp=None (no venc_params / non-H.264 source) means
    SKIP, never a silent fall-through to the blockiness proxy (the proxy is the R10-refuted
    unreliable gate; it conflates texture with compression and mis-fired on light content).
    Use gate='blockiness' EXPLICITLY if the proxy is genuinely wanted."""
    gate = cfg.get("gate", "qp")
    if gate == "always":
        ok = True
    elif gate == "qp":
        ok = (qp is not None) and qp >= cfg.get("qp_min", 30)
    else:  # gate == "blockiness": the explicit LR proxy (synthetic harnesses without QP)
        ok = blockiness(rgb) >= cfg.get("block_min", 1.30)
    if not ok:
        return False
    tex = cfg.get("skip_texture_varlap")
    if tex is not None:
        vl = cv2.Laplacian(cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY),
                           cv2.CV_64F).var()
        if vl >= tex:        # dense-texture guard (the texture46k over-smooth case)
            return False
    return True


def deblock_neural(rgb, model_fname="scunet_color_real_psnr.pth"):
    import torch
    net = load_deblocker(model_fname)
    dev = _device()
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(rgb)).to(dev).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        out = net(t).clamp_(0, 1).mul_(255.0).round_()
        out = out.squeeze(0).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return np.ascontiguousarray(out)


_LOAD_FAILED = set()   # model_fname -> warned+disabled (R12 review: no mid-render crash, no silence)


def apply(rgb_lr, cfg, qp=None):
    """Preprocessor entrypoint. cfg None/empty -> identity (byte-identical OFF path).
    Returns a (possibly) deblocked uint8 RGB at the SAME size as the input LR.

    R12 review: a loader failure (missing gitignored SCUNet weights / spandrel not installed)
    WARNS ONCE and no-ops instead of killing the whole render mid-clip with FileNotFoundError
    (the repo ships code-only; a fresh clone has no weights)."""
    if not cfg:
        return rgb_lr
    if not should_deblock(rgb_lr, cfg, qp=qp):
        return rgb_lr
    model = cfg.get("model", "scunet_color_real_psnr.pth")
    if model in _LOAD_FAILED:
        return rgb_lr
    try:
        return deblock_neural(rgb_lr, model)
    except (FileNotFoundError, ImportError, OSError) as e:
        _LOAD_FAILED.add(model)
        import sys
        print(f"[deblock_pre] WARNING: cannot load '{model}' ({type(e).__name__}: {e}) -> "
              f"deblock DISABLED for this run (frames pass through unchanged)",
              file=sys.stderr, flush=True)
        return rgb_lr
