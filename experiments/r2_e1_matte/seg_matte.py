"""seg_matte.py -- PERMISSIVE (commercial-OK) foreground matte adapter.

DROP-IN replacement for the matte source in the LAYERED pipeline (Stage L1). It
exposes the SAME public surface as prototype/matting.py so that layered_pipeline /
layered_api can swap the matte source with a one-line import change:

    # import matting                  # RVM, CC BY-NC-SA 4.0 (NON-COMMERCIAL)
    import seg_matte as matting       # permissive person segmentation (commercial-OK)

API parity (matting.py -> seg_matte.py)
---------------------------------------
  load_rvm(device, variant)            ->  load_seg(device, variant)     (alias: load_rvm)
  matte_sequence(model, frames, ...)   ->  matte_sequence(model, frames, ...)   SAME SIG
  fg_mask_lr(pha, lr_hw, soft, ...)    ->  fg_mask_lr(...)               (re-exported verbatim)
  composite(fgr, pha, bg)             ->  composite(...)                (re-exported verbatim)
  auto_downsample_ratio(h, w)          ->  auto_downsample_ratio(...)    (re-exported)

GOTCHA #17 (RVM is RECURRENT + human-only): RVM threads a ConvGRU state across
frames for temporal coherence. The permissive candidates here are torchvision
person-segmentation nets (LR-ASPP / DeepLabV3, COCO+VOC `person` class) which are
**STATELESS** -- one independent forward per frame. matte_sequence keeps RVM's
"feed frames IN ORDER + thread state" interface but simply IGNORES the recurrent
state (it is created and discarded, never used). To recover some of RVM's temporal
coherence on a stateless net we add an optional, cheap **alpha EMA** smoother
(thread the *previous alpha* instead of a GRU state): pha_t = a*pha_t + (1-a)*pha_{t-1}.
That EMA IS the "recurrent state" for these models and is threaded in display order,
preserving the contract's spirit.

LICENSE: torchvision model code + weights are BSD-3-Clause (PyTorch). The `person`
class comes from COCO/Pascal-VOC supervision. This is commercially deployable, unlike
RVM. MediaPipe Selfie Segmentation (Apache-2.0) is the recommended production target
(see run_compare.py report); this torchvision net is the runnable, same-license-tier
PROXY used to prove the layered pipeline survives a permissive, stateless matte.

Import-safe: importing this module loads no weights and touches no GPU. `load_seg`
is the only thing that downloads / allocates.
"""
from __future__ import annotations

import sys
import os
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Re-use the model-agnostic helpers from the RVM module VERBATIM (read-only import).
# fg_mask_lr / composite / auto_downsample_ratio operate purely on the alpha array and
# carry no RVM dependency, so the gate output shape is byte-identical to the RVM path.
_PROTO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "prototype")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)
from matting import fg_mask_lr, composite, auto_downsample_ratio  # noqa: E402  (re-export)

try:
    import torch
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None
    _TORCH_IMPORT_ERR = e

# Pascal-VOC class index for "person" in torchvision COCO_WITH_VOC_LABELS weights.
_VOC_PERSON = 15
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_VARIANTS = {
    # name -> (constructor, weights enum attr)
    "lraspp_mobilenetv3": ("lraspp_mobilenet_v3_large", "LRASPP_MobileNet_V3_Large_Weights"),
    "deeplabv3_mobilenetv3": ("deeplabv3_mobilenet_v3_large", "DeepLabV3_MobileNet_V3_Large_Weights"),
    "deeplabv3_resnet50": ("deeplabv3_resnet50", "DeepLabV3_ResNet50_Weights"),
}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_seg(device: str = "mps", variant: str = "lraspp_mobilenetv3",
             seg_res: int = 512, ema: float = 0.0):
    """Load a permissive person-segmentation net, move to `device`, eval mode.

    variant: "lraspp_mobilenetv3" (3.2M, fastest -- closest to MediaPipe Selfie in
             spirit/speed), "deeplabv3_mobilenetv3" (11M, sharper edges), or
             "deeplabv3_resnet50" (42M, quality ceiling). All BSD-3-Clause weights.
    seg_res: longer-side resolution the net runs at (it was trained ~520px; we resize
             the LR frame to this for the forward, then resize the person-prob back to
             LR -- mirrors RVM's internal ~512 coarse pass for a fair comparison).
    ema:     alpha EMA factor in [0,1) for temporal smoothing (0 = stateless/off). This
             is the stateless net's stand-in for RVM's recurrent state.

    Returns the nn.Module with helper attrs stashed (so the helpers stay subclass-free).
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError(f"torch unavailable: {_TORCH_IMPORT_ERR}")
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(_VARIANTS)}")
    from torchvision.models import segmentation as tvseg
    ctor_name, weights_name = _VARIANTS[variant]
    weights = getattr(tvseg, weights_name).DEFAULT  # COCO_WITH_VOC_LABELS_V1
    model = getattr(tvseg, ctor_name)(weights=weights)
    model = model.eval()
    if device:
        model = model.to(device)
    model._rvm_device = device       # same attr name as matting.load_rvm -> drop-in helpers
    model._rvm_variant = variant
    model._seg_res = int(seg_res)
    model._seg_ema = float(ema)
    model._seg_person_idx = _VOC_PERSON
    return model


# back-compat alias so call-sites literally written `load_rvm(...)` still work after
# `import seg_matte as matting`.
def load_rvm(device: str = "mps", variant: str = "lraspp_mobilenetv3", **kw):
    return load_seg(device=device, variant=variant, **kw)


# --------------------------------------------------------------------------- #
# Tensor <-> image conversion (ImageNet-normalized; seg nets need it, RVM did not)
# --------------------------------------------------------------------------- #
def _to_norm_tensor(frame, device, seg_res):
    """uint8 HxWx3 RGB -> normalized float [1,3,Hs,Ws] on `device`, longer side = seg_res.
    Returns (tensor, (H,W)) so the caller can resize the prob back to native LR."""
    arr = np.ascontiguousarray(frame)
    H, W = arr.shape[:2]
    if arr.dtype == np.uint8:
        t = torch.from_numpy(arr).float().div_(255.0)
    else:
        t = torch.from_numpy(arr.astype(np.float32))
    t = t.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)  # [1,3,H,W]
    scale = seg_res / float(max(H, W))
    if scale < 1.0:
        hs, ws = int(round(H * scale)), int(round(W * scale))
        t = F.interpolate(t, size=(hs, ws), mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t, (H, W)


def _person_prob(model, src_tensor, hw_native):
    """One stateless forward -> person-probability alpha at NATIVE LR, [1,1,H,W] float."""
    H, W = hw_native
    with torch.no_grad():
        out = model(src_tensor)["out"]                 # [1,21,Hs,Ws] logits
        prob = torch.softmax(out, dim=1)[:, model._seg_person_idx: model._seg_person_idx + 1]
        prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
    return prob.clamp(0, 1)                              # [1,1,H,W]


def _pha_to_float(pha) -> np.ndarray:
    return pha[0, 0].clamp(0, 1).float().cpu().numpy()


# --------------------------------------------------------------------------- #
# Core: matte an ordered sequence. SAME SIGNATURE as matting.matte_sequence.
# Stateless model -> the "recurrent state" is an optional alpha EMA (display order).
# --------------------------------------------------------------------------- #
def matte_sequence(
    model,
    frames: Sequence,
    downsample_ratio: Optional[float] = None,   # accepted for API parity; unused (seg has no coarse ratio)
    device: Optional[str] = None,
    return_tensors: bool = False,
    sync_each: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Matte an ordered sequence; returns list of (fgr, pha) like matting.matte_sequence.

    fgr: uint8 HxWx3 RGB == the SOURCE frame (these seg nets do not estimate a
         decontaminated foreground; the layered pipeline never uses RVM's fgr anyway --
         it composites the real frame * alpha -- so returning the source is contract-safe).
    pha: float32 HxW person-probability alpha in [0,1] at native LR (same as RVM's pha).

    `downsample_ratio` is accepted and ignored (stateless net, no coarse pass). Frames
    are still consumed IN ORDER so the optional alpha-EMA (model._seg_ema>0) threads
    correctly -- mismatched order would break the smoother, mirroring RVM's order rule.
    """
    if device is None:
        device = getattr(model, "_rvm_device", "mps")
    seg_res = getattr(model, "_seg_res", 512)
    ema = getattr(model, "_seg_ema", 0.0)
    frames = list(frames)
    if not frames:
        return []

    prev = None
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for f in frames:
        src, hw = _to_norm_tensor(f, device, seg_res)
        pha = _person_prob(model, src, hw)              # [1,1,H,W]
        if ema > 0.0 and prev is not None:
            pha = ema * pha + (1.0 - ema) * prev
        prev = pha
        if sync_each and device == "mps":
            torch.mps.synchronize()
        if return_tensors:
            out.append((f, pha))
        else:
            out.append((np.ascontiguousarray(f), _pha_to_float(pha)))
    return out


# --------------------------------------------------------------------------- #
# Timing helper (mirrors matting.benchmark; stateless -> no rec threading)
# --------------------------------------------------------------------------- #
def benchmark(model, frames, downsample_ratio=None, warmup=4, device=None):
    if device is None:
        device = getattr(model, "_rvm_device", "mps")
    seg_res = getattr(model, "_seg_res", 512)
    frames = list(frames)
    h, w = frames[0].shape[:2]
    srcs = [_to_norm_tensor(f, device, seg_res) for f in frames]
    if device == "mps":
        torch.mps.synchronize()
    times = []
    for i, (src, hw) in enumerate(srcs):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = _person_prob(model, src, hw)
        if device == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    times = np.array(times)
    return dict(median_ms=float(np.median(times)), mean_ms=float(times.mean()),
                p90_ms=float(np.percentile(times, 90)), n=int(times.size), h=h, w=w,
                downsample_ratio=float("nan"))


if __name__ == "__main__":
    print("seg_matte.py smoke test")
    m = load_seg("mps", "lraspp_mobilenetv3")
    n = sum(p.numel() for p in m.parameters())
    print(f"  loaded {m._rvm_variant}: {n/1e6:.2f}M params on {m._rvm_device}")
    frames = [np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(8)]
    res = matte_sequence(m, frames)
    fgr, pha = res[0]
    print(f"  matte_sequence -> {len(res)} frames; fgr {fgr.shape} {fgr.dtype}, "
          f"pha {pha.shape} {pha.dtype} range[{pha.min():.3f},{pha.max():.3f}]")
    gate = fg_mask_lr(pha, lr_hw=(360, 640), soft=False, thresh=0.5, dilate=2)
    print(f"  fg_mask_lr -> {gate.shape} {gate.dtype} fg_frac={gate.mean():.3f}")
    print("OK")
