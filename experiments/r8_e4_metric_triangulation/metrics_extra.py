#!/usr/bin/env python3
"""
R8-E4 metric-triangulation extras: a TEXTURE-AWARE full-reference metric (DISTS)
and a re-implemented LPIPS via the SAME pyiqa backbone, so DISTS and LPIPS share
preprocessing and we can attribute any divergence to the metric, not the pipeline.

DISTS (Ding et al. 2020, "Image Quality Assessment: Unifying Structure and Texture
Similarity"): a learned FR metric that explicitly models TEXTURE similarity via the
global means of VGG feature maps -> tolerant of texture *resampling* (where SSIM/PSNR
over-penalize) but still structure-aware. This is precisely the axis LPIPS is known
to be weak on, so it is the independent texture-sensitive check this thread needs.

All metrics here are FULL-REFERENCE (GOTCHA #23: no NR sharpness as a verdict).
Inputs are uint8 HxWx3 RGB. Everything runs on CPU to keep the shared MPS free for
the SR nets. NO empty catch: if a metric backend is missing we raise loudly.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

_DEV = "cpu"   # keep MPS free for the SR nets (3 sibling agents contend for it)

_METRICS = {}   # name -> pyiqa metric module (lazy, cached)


def _to_tensor(rgb_uint8):
    """uint8 HxWx3 RGB -> 1x3xHxW float tensor in [0,1] (pyiqa FR convention)."""
    t = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).float().div_(255.0)
    return t.permute(2, 0, 1).unsqueeze(0).contiguous()


def _get(name):
    global _METRICS
    if name not in _METRICS:
        import pyiqa
        # as_loss=False -> returns the quality SCORE (DISTS: lower=better, like LPIPS)
        _METRICS[name] = pyiqa.create_metric(name, device=_DEV, as_loss=False).eval()
    return _METRICS[name]


@torch.no_grad()
def dists(a, b):
    """DISTS distance (texture+structure). LOWER = perceptually closer. a=cand,b=ref."""
    m = _get("dists")
    return float(m(_to_tensor(a), _to_tensor(b)).item())


@torch.no_grad()
def lpips_pyiqa(a, b):
    """LPIPS via pyiqa (VGG backbone by default) -- a same-backend cross-check of the
    `lpips` (AlexNet) package number, so DISTS vs LPIPS divergence is metric-attributable."""
    m = _get("lpips")
    return float(m(_to_tensor(a), _to_tensor(b)).item())


def selfcheck():
    """Range/preprocessing validity gate run before any verdict:
       (1) DISTS(x,x) ~= 0 and LPIPS(x,x) ~= 0   (identity);
       (2) DISTS(blurred,x) > DISTS(x,x)          (monotone w/ degradation);
       (3) DISTS in plausible [0, ~0.6] band.
    Returns a dict; caller asserts the identity ~0 invariant before trusting flips."""
    import cv2
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (96, 160, 3), dtype=np.uint8)
    # add real structure so feature maps are non-degenerate
    x = cv2.GaussianBlur(x, (0, 0), 0.7)
    blur = cv2.GaussianBlur(x, (0, 0), 2.0)
    return {
        "dists_self": dists(x, x),
        "dists_blur": dists(blur, x),
        "lpips_self": lpips_pyiqa(x, x),
        "lpips_blur": lpips_pyiqa(blur, x),
    }


if __name__ == "__main__":
    print("[selfcheck]", selfcheck())
