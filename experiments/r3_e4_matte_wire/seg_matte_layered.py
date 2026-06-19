"""seg_matte_layered.py -- R3-E4 COMPLETED permissive matte adapter (PASS A + PASS B).

This is the R2-E1 `seg_matte.py` adapter, COMPLETED so it satisfies EVERY call site the
server's LAYERED path (`server/layered_api.py`) makes on the `matting` module:

    layered_api call               provided by                       used in
    ----------------------------   -------------------------------   -----------------
    matting.load_rvm(device)       load_rvm (alias -> load_seg)      load_matting_model (PASS A/B)
    matting.auto_downsample_ratio  auto_downsample_ratio (re-export) downsample_ratio   (PASS B)
    matting.matte_sequence(...)    matte_sequence                    build_scene_plates (PASS A)
    matting.fg_mask_lr(...)        fg_mask_lr (re-export)            build_scene_plates (PASS A)
    matting.matte_frame(...)       matte_frame  <-- NEW (this file)  matte_frame_np     (PASS B)

The ONLY thing R2-E1's adapter lacked was `matte_frame`, the STATELESS per-frame step the
server's PASS B drives (`layered_api.matte_frame_np` -> `matting.matte_frame(model, src, rec,
ratio)`). RVM's `matte_frame` threads a 4-tensor ConvGRU `rec`; the seg net is STATELESS, so we
honour the contract's SHAPE (return `(fgr, pha, rec)` with `pha` == `[1,1,H,W]`, the same as RVM)
and thread the display-order **alpha-EMA** (GOTCHA #17) as the recurrent-state stand-in: `rec[0]`
carries the previous native-LR alpha tensor, and `pha_t = a*pha_t + (1-a)*pha_{t-1}`. The seg net
IGNORES the GRU semantics of `rec` (it is stateless) but the EMA threaded through `rec` recovers
RVM-parity temporal stability at ~0 cost, exactly as the PASS-A `matte_sequence` already does.

LICENSE: torchvision model code + weights are BSD-3-Clause (commercial-OK), unlike RVM
(CC BY-NC-SA 4.0, NON-COMMERCIAL). `person` class = COCO/Pascal-VOC supervision. Human-only,
same scope as RVM. Recommended runnable pick: DeepLabV3-MobileNetV3-Large + alpha-EMA(0.5)
(R2-E1: cleanest plate, RVM-parity hole%/sharpness, 0.85x latency).

Import-safe: importing loads no weights and touches no GPU. `load_seg`/`load_rvm` allocate.

Location-independent: it resolves `prototype/matting.py` whether dropped in experiments/ or
server/ (it first tries the already-on-path `matting`, then walks up to find prototype/).
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Re-use the model-agnostic helpers from the RVM module VERBATIM (read-only).
# fg_mask_lr / composite / auto_downsample_ratio operate purely on the alpha array, carry
# no RVM dependency, and so the gate output shape is byte-identical to the RVM path. Resolve
# prototype/matting.py robustly: try the caller's path first (layered_api already inserts
# prototype/ on sys.path), else walk up from here to find a sibling prototype/matting.py.
# --------------------------------------------------------------------------- #
def _ensure_matting_on_path():
    try:
        import matting  # noqa: F401  (already importable -> caller put prototype/ on path)
        return
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):                       # walk up looking for <repo>/prototype/matting.py
        cand = os.path.join(d, "prototype")
        if os.path.isfile(os.path.join(cand, "matting.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    raise ImportError("could not locate prototype/matting.py for fg_mask_lr/composite re-export")


_ensure_matting_on_path()
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

    variant: "lraspp_mobilenetv3" (3.2M, fastest), "deeplabv3_mobilenetv3" (11M, sharper
             edges -- the R2-E1 recommended pick), "deeplabv3_resnet50" (42M, ceiling/slow).
    seg_res: longer-side resolution the net runs at (resize LR -> seg_res for the forward,
             then resize the person-prob back to LR -- mirrors RVM's ~512 coarse pass).
    ema:     alpha-EMA factor in [0,1) (0 = stateless/off). The stateless net's stand-in for
             RVM's recurrent state; threaded in display order (PASS A matte_sequence AND
             PASS B matte_frame). R2-E1 recommends 0.5 for RVM-parity temporal stability.
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


def load_rvm(device: str = "mps", variant: str = "lraspp_mobilenetv3", **kw):
    """back-compat alias: call-sites literally written `load_rvm(...)` still work after the
    `import seg_matte_layered as matting` swap. layered_api.load_matting_model uses this."""
    return load_seg(device=device, variant=variant, **kw)


# --------------------------------------------------------------------------- #
# Tensor <-> image conversion (ImageNet-normalized; seg nets need it, RVM did not).
# Two entry points share ONE resize+normalize core so the numpy (PASS A) and tensor
# (PASS B / matte_frame) paths are byte-identical given the same pixels.
# --------------------------------------------------------------------------- #
def _resize_norm(t, device, seg_res):
    """t: [1,3,H,W] float in [0,1] (on any device) -> ([1,3,Hs,Ws] normalized on `device`, (H,W))."""
    t = t.to(device)
    _, _, H, W = t.shape
    scale = seg_res / float(max(H, W))
    if scale < 1.0:
        hs, ws = int(round(H * scale)), int(round(W * scale))
        t = F.interpolate(t, size=(hs, ws), mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (t - mean) / std, (H, W)


def _to_norm_tensor(frame, device, seg_res):
    """uint8/float HxWx3 RGB -> ([1,3,Hs,Ws] normalized on `device`, (H,W)). PASS-A numpy path."""
    arr = np.ascontiguousarray(frame)
    if arr.dtype == np.uint8:
        t = torch.from_numpy(arr).float().div_(255.0)
    else:
        t = torch.from_numpy(arr.astype(np.float32))
    t = t.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,3,H,W] in [0,1]
    return _resize_norm(t, device, seg_res)


def _person_prob(model, src_norm, hw_native):
    """One stateless forward -> person-probability alpha at NATIVE LR, [1,1,H,W] float in [0,1]."""
    H, W = hw_native
    with torch.no_grad():
        out = model(src_norm)["out"]                   # [1,21,Hs,Ws] logits
        prob = torch.softmax(out, dim=1)[:, model._seg_person_idx: model._seg_person_idx + 1]
        prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
    return prob.clamp(0, 1)                              # [1,1,H,W]


def _pha_to_float(pha) -> np.ndarray:
    return pha[0, 0].clamp(0, 1).float().cpu().numpy()


# --------------------------------------------------------------------------- #
# NEW (the R2-E1 gap): per-frame stateless step. SAME contract as matting.matte_frame.
# --------------------------------------------------------------------------- #
def matte_frame(model, src_tensor, rec, downsample_ratio=None):
    """One per-frame seg step with the display-order alpha-EMA as the recurrent-state stand-in.

    DROP-IN for matting.matte_frame (RVM). The server's PASS B calls
        _fgr, pha, rec = matting.matte_frame(model, src, rec, ratio)   # layered_api.matte_frame_np
    where `src` = `_frame_tensor(img, device)` = [1,3,H,W] float in [0,1] on the model's device
    (RAW, UN-normalized -- the same tensor RVM consumed). We normalize/resize INTERNALLY.

    Args
      model        : a load_seg(...) net (stashes _rvm_device / _seg_res / _seg_ema).
      src_tensor   : [1,3,H,W] float in [0,1] on device (or [3,H,W] -> unsqueezed).
      rec          : opaque recurrent-state list (start [None]*4, as RVM/the caller threads it).
                     rec[0] holds the PREVIOUS native-LR alpha tensor (the EMA state); the
                     stateless seg net IGNORES the GRU meaning of rec[1:] entirely.
      downsample_ratio : accepted for RVM API parity and IGNORED (seg has its own seg_res pass).

    Returns (fgr, pha, rec)  -- ALL on-device tensors, SHAPES identical to matting.matte_frame:
      fgr = src_tensor  (seg nets don't decontaminate FG; layered composites real-frame*alpha,
                         never RVM's fgr -- so returning the source is contract-safe).
      pha = [1,1,H,W] person-prob alpha at native LR, in [0,1]  (caller does pha[0,0] -> HxW).
      rec = [pha, None, None, None]  (the EMA state threaded into the next call; same display
                                      order + per-scene-reset rule the caller already enforces).
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError(f"torch unavailable: {_TORCH_IMPORT_ERR}")
    device = getattr(model, "_rvm_device", None) or src_tensor.device
    seg_res = getattr(model, "_seg_res", 512)
    ema = float(getattr(model, "_seg_ema", 0.0))

    t = src_tensor
    if t.dim() == 3:
        t = t.unsqueeze(0)
    src_norm, hw = _resize_norm(t, device, seg_res)
    pha = _person_prob(model, src_norm, hw)             # [1,1,H,W]

    prev = rec[0] if (rec is not None and len(rec) > 0) else None
    if ema > 0.0 and prev is not None and tuple(prev.shape) == tuple(pha.shape):
        pha = ema * pha + (1.0 - ema) * prev            # display-order EMA == "recurrent state"

    new_rec = [pha, None, None, None]                   # 4-slot to mirror RVM's rec shape
    return t, pha, new_rec


# --------------------------------------------------------------------------- #
# Core: matte an ordered sequence. SAME SIGNATURE as matting.matte_sequence.
# PASS A (build_scene_plates) consumes this. EMA threaded across the (sparse) samples.
# --------------------------------------------------------------------------- #
def matte_sequence(
    model,
    frames: Sequence,
    downsample_ratio: Optional[float] = None,   # accepted for API parity; unused (no coarse ratio)
    device: Optional[str] = None,
    return_tensors: bool = False,
    sync_each: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Matte an ordered sequence; returns list of (fgr, pha) like matting.matte_sequence.

    fgr: HxWx3 == the SOURCE frame (seg nets don't estimate a decontaminated FG; layered
         composites real-frame * alpha, so the source is contract-safe).
    pha: float32 HxW person-prob alpha in [0,1] at native LR.
    Frames are consumed IN ORDER so the optional alpha-EMA threads correctly (display order).
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
        if ema > 0.0 and prev is not None and tuple(prev.shape) == tuple(pha.shape):
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
    print("seg_matte_layered.py smoke test (adapter parity incl. NEW matte_frame)")
    m = load_seg("mps" if (torch and torch.backends.mps.is_available()) else "cpu",
                 "deeplabv3_mobilenetv3", ema=0.5)
    dev = m._rvm_device
    n = sum(p.numel() for p in m.parameters())
    print(f"  loaded {m._rvm_variant}: {n/1e6:.2f}M params on {dev}, ema={m._seg_ema}")
    frames = [np.random.randint(0, 255, (320, 640, 3), dtype=np.uint8) for _ in range(6)]
    # PASS A surface
    res = matte_sequence(m, frames)
    fgr, pha = res[0]
    print(f"  matte_sequence -> {len(res)} frames; fgr {fgr.shape}, pha {pha.shape} {pha.dtype} "
          f"range[{pha.min():.3f},{pha.max():.3f}]")
    gate = fg_mask_lr(pha, lr_hw=(320, 640), soft=False, thresh=0.5, dilate=3)
    print(f"  fg_mask_lr -> {gate.shape} fg_frac={gate.mean():.3f}")
    # PASS B surface (NEW matte_frame), threaded like layered_api.matte_frame_np
    ratio = auto_downsample_ratio(320, 640)
    rec = [None] * 4
    shapes = []
    for f in frames:
        t = torch.from_numpy(f).float().div_(255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        _fgr, p, rec = matte_frame(m, t, rec, ratio)
        shapes.append(tuple(p.shape))
    print(f"  matte_frame x{len(frames)} -> pha shape {shapes[0]} (const={len(set(shapes))==1}); "
          f"rec len={len(rec)} rec0_is_alpha={rec[0] is not None}")
    print("OK")
