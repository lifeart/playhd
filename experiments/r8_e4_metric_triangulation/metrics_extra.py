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


# --------------------------------------------------------------------------- #
# R12-E3 adoption: VMAF-NEG guardrail, re-exported so every harness that already
# does `import metrics_extra` also gets MX.vmaf_neg / MX.vmaf_neg_single /
# MX.vmaf_available alongside DISTS+LPIPS. Single source of truth stays in
# experiments/r12_e3_quality_probes/vmaf_neg.py. USE AS A GUARDRAIL COLUMN ONLY
# (anti-hallucination), never an optimisation target (VMAF-NEG is itself gameable).
# --------------------------------------------------------------------------- #
import os as _os   # noqa: E402
_VMAF_FILE = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                           "r12_e3_quality_probes", "vmaf_neg.py")
try:
    # R12 review [18]: load the exact FILE via importlib instead of sys.path.insert -- the
    # r12_e3 dir also holds generic names (analyze.py, run_ab.py) that a path insert would
    # let shadow other harnesses' imports as a side effect of importing metrics_extra.
    import importlib.util as _ilu   # noqa: E402
    _spec = _ilu.spec_from_file_location("_r12e3_vmaf_neg", _VMAF_FILE)
    _vm = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_vm)
    vmaf_neg, vmaf_neg_single, vmaf_available = _vm.vmaf_neg, _vm.vmaf_neg_single, _vm.available
except Exception as _e:   # loud (no silent swallow): the guardrail column is simply unavailable
    _VMAF_IMPORT_ERR = _e
    def vmaf_neg(*_a, **_k):
        raise RuntimeError(f"vmaf_neg unavailable: {_VMAF_IMPORT_ERR!r}")
    def vmaf_neg_single(*_a, **_k):
        raise RuntimeError(f"vmaf_neg unavailable: {_VMAF_IMPORT_ERR!r}")
    def vmaf_available():
        return False


if __name__ == "__main__":
    print("[selfcheck]", selfcheck())
    print("[vmaf] available:", vmaf_available())
