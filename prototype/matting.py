"""matting.py -- fast, temporally-stable foreground matting for the LAYERED pipeline.

Stage L1 of the layered architecture. The layered design SRs a static background
PLATE and a dynamic FOREGROUND with separate budgets; that split needs a clean,
temporally-coherent foreground matte. This module wraps **Robust Video Matting**
(PeterL1n/RobustVideoMatting, WACV 2022) -- an auxiliary-free (no trimap / no
greenscreen), RECURRENT (ConvGRU state threaded across frames -> temporal
coherence) human matting network -- and runs it on Apple MPS.

LICENSE NOTE: RVM weights/code are **CC BY-NC-SA 4.0 (NonCommercial)**. Fine for
this research prototype; a commercial deployment needs a different matte source
(re-train / license / swap model). RVM is also HUMAN-ONLY.

Public API
----------
  load_rvm(device="mps", variant="mobilenetv3")            -> model (eval)
  matte_sequence(model, frames, downsample_ratio=None, ...) -> [(fgr, pha), ...]
  fg_mask_lr(pha, lr_hw=None, soft=True, thresh=0.5, dilate=0) -> mask  (the L2 gate)
  composite(fgr, pha, bg=(0,255,0))                        -> uint8 HxWx3 (viz)
  auto_downsample_ratio(h, w)                              -> float

Import-safe: importing this module does NOT load weights or touch the GPU.
`load_rvm` is the only thing that downloads / allocates.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover - torch is expected present
    torch = None
    _TORCH_IMPORT_ERR = e


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_rvm(device: str = "mps", variant: str = "mobilenetv3"):
    """Load Robust Video Matting and move it to `device` in eval mode.

    variant: "mobilenetv3" (3.75M params, fast) or "resnet50" (heavier, sharper).
    Load path: torch.hub.load("PeterL1n/RobustVideoMatting", variant). Weights
    (~14.5MB for mobilenetv3) are cached under ~/.cache/torch/hub. Requires net
    on first call only.

    Returns the nn.Module. Raises loudly on failure (never returns a broken stub).
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError(f"torch unavailable: {_TORCH_IMPORT_ERR}")
    model = torch.hub.load(
        "PeterL1n/RobustVideoMatting", variant, trust_repo=True
    )
    model = model.eval()
    if device:
        model = model.to(device)
    # stash a couple of attributes the helpers want, without subclassing
    model._rvm_device = device
    model._rvm_variant = variant
    return model


def auto_downsample_ratio(h: int, w: int) -> float:
    """RVM's recommended internal coarse-pass ratio: keep the longer side of the
    downsampled pass near 512 px (what RVM was trained around). Clamped to <=1."""
    return float(min(512.0 / max(h, w), 1.0))


# --------------------------------------------------------------------------- #
# Tensor <-> image conversion
# --------------------------------------------------------------------------- #
def _to_src_tensor(frame, device):
    """uint8 HxWx3 RGB  ->  float [1,3,H,W] in [0,1] on `device`.
    Accepts an already-batched float tensor unchanged (must be [1,3,H,W])."""
    if torch is not None and isinstance(frame, torch.Tensor):
        t = frame
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t.to(device)
    arr = np.ascontiguousarray(frame)
    if arr.dtype == np.uint8:
        t = torch.from_numpy(arr).float().div_(255.0)
    else:
        t = torch.from_numpy(arr.astype(np.float32))
    # HWC -> CHW -> NCHW
    t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
    return t.to(device)


def _fgr_to_uint8(fgr) -> np.ndarray:
    """[1,3,H,W] float in [0,1] -> HxWx3 uint8 RGB."""
    a = fgr[0].clamp(0, 1).mul(255).round().to(torch.uint8)
    return a.permute(1, 2, 0).cpu().numpy()


def _pha_to_float(pha) -> np.ndarray:
    """[1,1,H,W] float in [0,1] -> HxW float32."""
    return pha[0, 0].clamp(0, 1).float().cpu().numpy()


# --------------------------------------------------------------------------- #
# Core: recurrent matting over a sequence  (state threaded frame-to-frame)
# --------------------------------------------------------------------------- #
def matte_frame(model, src_tensor, rec, downsample_ratio):
    """One recurrent step. src_tensor: [1,3,H,W] float on the model's device.
    rec: list of 4 recurrent-state tensors (start with [None]*4). Returns
    (fgr, pha, rec) -- all still on-device tensors so the caller controls sync.
    RVM is recurrent: you MUST feed frames in temporal order and thread `rec`."""
    with torch.no_grad():
        fgr, pha, *rec = model(src_tensor, *rec, downsample_ratio)
    return fgr, pha, rec


def matte_sequence(
    model,
    frames: Sequence,
    downsample_ratio: Optional[float] = None,
    device: Optional[str] = None,
    return_tensors: bool = False,
    sync_each: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Matte an ordered sequence of frames, threading RVM's recurrent state.

    frames: list of uint8 HxWx3 RGB arrays (LR or HD), in TEMPORAL (display) order.
            Mixed resolution is NOT allowed within one call (recurrent state is
            resolution-tied) -- pass a uniform-size list.
    downsample_ratio: RVM coarse-pass ratio; None -> auto_downsample_ratio(H,W).
    return_tensors: if True, yield on-device (fgr, pha) tensors instead of numpy.
    sync_each: MPS-synchronize after every frame (for honest per-frame timing).

    Returns list of (fgr, pha): by default fgr = uint8 HxWx3 RGB, pha = float32
    HxW in [0,1]. Recurrent state is created fresh here and discarded at the end.
    """
    if device is None:
        device = getattr(model, "_rvm_device", "mps")
    frames = list(frames)
    if not frames:
        return []
    h, w = frames[0].shape[:2]
    if downsample_ratio is None:
        downsample_ratio = auto_downsample_ratio(h, w)

    rec = [None] * 4
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for f in frames:
        src = _to_src_tensor(f, device)
        fgr, pha, rec = matte_frame(model, src, rec, downsample_ratio)
        if sync_each and device == "mps":
            torch.mps.synchronize()
        if return_tensors:
            out.append((fgr, pha))
        else:
            out.append((_fgr_to_uint8(fgr), _pha_to_float(pha)))
    return out


# --------------------------------------------------------------------------- #
# The LAYER GATE: foreground mask at LR for Stage L2 (background plate)
# --------------------------------------------------------------------------- #
def fg_mask_lr(
    pha: np.ndarray,
    lr_hw: Optional[Tuple[int, int]] = None,
    soft: bool = True,
    thresh: float = 0.5,
    dilate: int = 0,
) -> np.ndarray:
    """Turn an alpha matte into the foreground gate the layered pipeline consumes.

    pha:    HxW float in [0,1] (alpha; from matte_sequence). May be HD or LR.
    lr_hw:  (h_lr, w_lr) to resize the gate to LR. None -> keep pha's resolution.
    soft:   True -> return the (resized) soft alpha in [0,1]; False -> binary mask
            (alpha >= thresh) as float32 {0,1}.
    dilate: grow the (binary) foreground by N px so the gate covers matte-edge
            uncertainty -- background pixels just outside the subject stay BG, but
            the hair/edge band is claimed by FG (where flicker lives). Only applies
            when soft=False.

    Returns HxW float32: 1 = foreground (dynamic layer), 0 = background (static
    plate). Stage L2 uses (1 - mask) to decide which pixels belong to the
    long-lived background plate vs. the per-frame foreground budget.
    """
    import cv2

    m = np.asarray(pha, dtype=np.float32)
    if lr_hw is not None and (m.shape[0], m.shape[1]) != tuple(lr_hw):
        h_lr, w_lr = lr_hw
        m = cv2.resize(m, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
    if soft:
        return np.clip(m, 0.0, 1.0)
    binm = (m >= thresh).astype(np.float32)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        binm = cv2.dilate(binm, k)
    return binm


def composite(fgr: np.ndarray, pha: np.ndarray, bg=(0, 255, 0)) -> np.ndarray:
    """Composite foreground over a flat background colour (visualization helper).
    fgr: HxWx3 uint8 RGB. pha: HxW float [0,1]. bg: RGB tuple. -> HxWx3 uint8."""
    a = pha[..., None].astype(np.float32)
    bg_arr = np.array(bg, dtype=np.float32)[None, None, :]
    com = fgr.astype(np.float32) * a + bg_arr * (1.0 - a)
    return np.clip(com, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Timing helper (median per-frame MPS-synced latency, after warmup)
# --------------------------------------------------------------------------- #
def benchmark(model, frames, downsample_ratio=None, warmup=4, device=None):
    """Median per-frame matting latency (ms), MPS-synced, recurrent-state threaded.
    Returns dict(median_ms, mean_ms, p90_ms, n, h, w, downsample_ratio)."""
    if device is None:
        device = getattr(model, "_rvm_device", "mps")
    frames = list(frames)
    h, w = frames[0].shape[:2]
    if downsample_ratio is None:
        downsample_ratio = auto_downsample_ratio(h, w)
    # pre-upload sources so we time matting, not host->device copies
    srcs = [_to_src_tensor(f, device) for f in frames]
    if device == "mps":
        torch.mps.synchronize()

    rec = [None] * 4
    times = []
    for i, src in enumerate(srcs):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fgr, pha, *rec = model(src, *rec, downsample_ratio)
        if device == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
    times = np.array(times)
    return dict(
        median_ms=float(np.median(times)),
        mean_ms=float(times.mean()),
        p90_ms=float(np.percentile(times, 90)),
        n=int(times.size),
        h=h,
        w=w,
        downsample_ratio=float(downsample_ratio),
    )


if __name__ == "__main__":
    # tiny smoke test on random input (no real footage needed)
    print("matting.py smoke test")
    m = load_rvm("mps")
    n = sum(p.numel() for p in m.parameters())
    print(f"  loaded RVM mobilenetv3: {n/1e6:.2f}M params on mps")
    frames = [np.random.randint(0, 255, (320, 640, 3), dtype=np.uint8) for _ in range(8)]
    res = matte_sequence(m, frames)
    fgr, pha = res[0]
    print(f"  matte_sequence -> {len(res)} frames; fgr {fgr.shape} {fgr.dtype}, "
          f"pha {pha.shape} {pha.dtype} range[{pha.min():.3f},{pha.max():.3f}]")
    gate = fg_mask_lr(pha, lr_hw=(320, 640), soft=False, thresh=0.5, dilate=2)
    print(f"  fg_mask_lr -> {gate.shape} {gate.dtype} fg_frac={gate.mean():.3f}")
    bench = benchmark(m, frames)
    print(f"  benchmark @ {bench['w']}x{bench['h']} ratio={bench['downsample_ratio']:.2f}: "
          f"median {bench['median_ms']:.1f} ms (n={bench['n']})")
    print("OK")
