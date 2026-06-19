#!/usr/bin/env python3
"""
GPU (torch/MPS) implementations of the playhd hot path — warp + occlusion mask.

Step 6: profiling showed the bottleneck shifted from the SR net to warp+mask. These are
torch/MPS ports of derisk.py's two hot ops, used behind `--backend torch`. The numpy path
in derisk.py stays the default regression guard; this module is the fast path.

Convention notes (verified empirically against cv2 / a true-float bilinear reference):
  * WARP  == cv2.remap(INTER_LINEAR, BORDER_REPLICATE) is reproduced by
    F.grid_sample(mode='bilinear', padding_mode='border', align_corners=True). This matches a
    true float bilinear to ~1e-4; cv2.remap itself quantizes sub-pixel maps to fixed-point
    (1/32 px), so the torch warp is actually MORE accurate -- the two differ only by cv2's
    quantization (~PSNR 50+ dB between them), well inside tolerance.
  * The LR block-flow is densified to HD on-device (nearest * scale), matching warp_hd().
  * SPLAT (softmax forward splat in occlusion_mask_lr): np.add.at scatter -> scatter_add_,
    float32 (numpy used float64). Same bilinear-splat / softmax-weight / Ruder math.

All tensors are float32 on the MPS device; images are [1,3,H,W] in 0..255 (NOT /255) so the
NEMO residual arithmetic stays in the same units as the numpy path. Flow fields are [H,W]
with NaN preserved for "no MV here" (intra holes).
"""
import numpy as np
import torch
import torch.nn.functional as F

_DEV = None


def device():
    global _DEV
    if _DEV is None:
        _DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return _DEV


def sync():
    """MPS-synchronize (no-op on cpu). Needed for honest GPU timing."""
    if device().type == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------------- #
# Host <-> device image / flow transfer
# --------------------------------------------------------------------------- #
def img_to_dev(np_u8):
    """HxWx3 uint8 RGB -> [1,3,H,W] float32 (0..255) on device. If given an already-resident
    [1,3,H,W] tensor (e.g. a GPU-resident perframe cache that models on-GPU SR output), it is
    returned as-is -- this lets the profiler measure the no-upload deployment case."""
    if isinstance(np_u8, torch.Tensor):
        return np_u8
    t = torch.from_numpy(np.ascontiguousarray(np_u8)).to(device())
    return t.permute(2, 0, 1).unsqueeze(0).float()


def img_to_host(t):
    """[1,3,H,W] float (0..255) -> HxWx3 uint8 RGB (clamp+round, == numpy clip().astype)."""
    t = t.clamp(0, 255).round_().squeeze(0).permute(1, 2, 0).to("cpu", torch.uint8)
    return np.ascontiguousarray(t.numpy())


def flow_to_dev(fx_lr, fy_lr):
    """LR flow (HxW float32, NaN = no MV) -> two [H,W] tensors on device, NaN preserved."""
    return (torch.from_numpy(np.ascontiguousarray(fx_lr)).to(device()),
            torch.from_numpy(np.ascontiguousarray(fy_lr)).to(device()))


# --------------------------------------------------------------------------- #
# Warp (grid_sample) -- matches cv2.remap(INTER_LINEAR, BORDER_REPLICATE)
# --------------------------------------------------------------------------- #
def _base_grid(h, w):
    gy, gx = torch.meshgrid(
        torch.arange(h, device=device(), dtype=torch.float32),
        torch.arange(w, device=device(), dtype=torch.float32), indexing="ij")
    return gx, gy


def _sample(src_1chw, mapx, mapy, padding_mode):
    """grid_sample at pixel coords (mapx,mapy). align_corners=True == integer-pixel-center.
    MPS does NOT implement grid_sample padding_mode='border', so BORDER_REPLICATE is emulated
    by clamping the sample coords to the edge (== replicate) and using 'zeros' padding."""
    _, _, h, w = src_1chw.shape
    if padding_mode == "border":
        mapx = mapx.clamp(0, w - 1)
        mapy = mapy.clamp(0, h - 1)
        padding_mode = "zeros"
    gxn = mapx * 2.0 / (w - 1) - 1.0
    gyn = mapy * 2.0 / (h - 1) - 1.0
    grid = torch.stack([gxn, gyn], dim=-1).unsqueeze(0)  # [1,H,W,2]
    return F.grid_sample(src_1chw, grid, mode="bilinear",
                         padding_mode=padding_mode, align_corners=True)


def densify(fx_lr, fy_lr, scale):
    """LR flow -> HD flow (nearest * scale) + HD hole mask. == warp_hd()'s cv2.resize NEAREST."""
    h, w = fx_lr.shape
    hole_lr = torch.isnan(fx_lr)
    fx0 = torch.nan_to_num(fx_lr)
    fy0 = torch.nan_to_num(fy_lr)
    fx_hd = F.interpolate(fx0[None, None], scale_factor=scale, mode="nearest")[0, 0] * scale
    fy_hd = F.interpolate(fy0[None, None], scale_factor=scale, mode="nearest")[0, 0] * scale
    hole_hd = F.interpolate(hole_lr[None, None].float(), scale_factor=scale,
                            mode="nearest")[0, 0] > 0.5
    return fx_hd, fy_hd, hole_hd


def warp_hd(ref_1c3hw, fx_lr, fy_lr, scale):
    """Warp an HD reference [1,3,Hhd,Whd] by an LR flow. Returns (warped, hole_mask_hd)."""
    fx_hd, fy_hd, hole = densify(fx_lr, fy_lr, scale)
    h_hd, w_hd = fx_hd.shape
    gx, gy = _base_grid(h_hd, w_hd)
    warped = _sample(ref_1c3hw, gx + fx_hd, gy + fy_hd, "border")
    return warped, hole


def warp_lr(ref_1c3hw, fx_lr, fy_lr):
    """Warp an LR image [1,3,H,W] by an LR flow (NaN->0). == warp_lr() numpy."""
    _, _, h, w = ref_1c3hw.shape
    gx, gy = _base_grid(h, w)
    fx0 = torch.nan_to_num(fx_lr)
    fy0 = torch.nan_to_num(fy_lr)
    return _sample(ref_1c3hw, gx + fx0, gy + fy0, "border")


def residual_hd(lr_cur_1c3hw, lr_ref_1c3hw, fx_lr, fy_lr, scale):
    """NEMO residual: (LR_cur - motion_comp(LR_ref)) bilinear-upscaled to HD. [1,3,Hhd,Whd]."""
    pred = warp_lr(lr_ref_1c3hw, fx_lr, fy_lr)
    res = lr_cur_1c3hw - pred
    _, _, h, w = res.shape
    return F.interpolate(res, size=(h * scale, w * scale), mode="bilinear", align_corners=False)


def add_res(warped, res_hd):
    """warped + optional residual, clamped to 0..255 (kept float; caller rounds on download)."""
    if res_hd is None:
        return warped.clone()
    return (warped + res_hd).clamp(0, 255)


# --------------------------------------------------------------------------- #
# Forward softmax splat (scatter_add) + Ruder fwd-bwd occlusion mask (all at LR)
# --------------------------------------------------------------------------- #
def _softmax_splat_flow(fwd_x, fwd_y, fx, fy, weight):
    """Forward bilinear splat of the forward-flow field (fwd_x,fwd_y) from source (x,y) to
    target (x+fx,y+fy), weighted by bilinear * softmax(weight). scatter_add_ replaces
    np.add.at. Returns (ffx, ffy) with NaN where nothing landed, and den. float32."""
    dev = device()
    h, w = fx.shape
    gx, gy = _base_grid(h, w)
    tx = gx + fx
    ty = gy + fy
    wsm = torch.exp(weight - weight[torch.isfinite(weight)].max())  # weight finite (= -react)
    valid0 = torch.isfinite(fx) & torch.isfinite(fy)
    x0 = torch.floor(tx)
    y0 = torch.floor(ty)
    fwd_x0 = torch.nan_to_num(fwd_x)   # 0 where invalid; wb is 0 there too (avoid 0*NaN=NaN)
    fwd_y0 = torch.nan_to_num(fwd_y)
    num_x = torch.zeros(h * w, device=dev)
    num_y = torch.zeros(h * w, device=dev)
    den = torch.zeros(h * w, device=dev)
    for ddx in (0, 1):
        for ddy in (0, 1):
            xs = x0 + ddx
            ys = y0 + ddy
            wb = (1 - (tx - xs).abs()).clamp(0, 1) * (1 - (ty - ys).abs()).clamp(0, 1) * wsm
            inb = valid0 & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (wb > 0)
            wbm = torch.where(inb, wb, torch.zeros_like(wb)).reshape(-1)
            idx = (ys.long().clamp(0, h - 1) * w + xs.long().clamp(0, w - 1)).reshape(-1)
            den.scatter_add_(0, idx, wbm)
            num_x.scatter_add_(0, idx, wbm * fwd_x0.reshape(-1))
            num_y.scatter_add_(0, idx, wbm * fwd_y0.reshape(-1))
    den = den.reshape(h, w)
    nz = den > 0
    nan = torch.full((h, w), float("nan"), device=dev)
    ffx = torch.where(nz, num_x.reshape(h, w) / torch.clamp_min(den, 1e-12), nan)
    ffy = torch.where(nz, num_y.reshape(h, w) / torch.clamp_min(den, 1e-12), nan)
    return ffx, ffy, den


def _sample_const(field_hw, mapx, mapy, const):
    """cv2.remap(field, INTER_LINEAR, borderValue=const). grid_sample has no constant pad, so
    sample (field-const) with zeros padding and add const back (linear => commutes)."""
    f = (field_hw - const)[None, None]
    g = _sample(f, mapx, mapy, "zeros")[0, 0]
    return g + const


ADAPTIVE_TAU = 0.06   # Step-7: reactive-fallback fraction above which 'adaptive' fires fwd-bwd
# (tuned in tune_adaptive.py; see derisk.ADAPTIVE_TAU for the rationale).


def occlusion_mask_lr(fx, fy, lr_cur_1c3hw, lr_prev_1c3hw, tau_react=16.0, mode="full",
                      adaptive_tau=None):
    """Three cheap LR occlusion signals unioned into one 'unreliable pixel' mask:
    (a) intra holes (NaN flow), (b) reactive residual, (c) Ruder 2016 fwd-bwd consistency
    via a softmax-splatted forward flow. Mirrors derisk.occlusion_mask_lr exactly.
    `mode`: 'full' = all three; 'reactive' = drop (c) (the splat ablation); 'adaptive' = run (c)
    ONLY when the reactive-fallback fraction exceeds `adaptive_tau` (Step-7 per-direction switch:
    pays for the splat only on motion-stressed frames). Returns (mask[H,W] bool, used_fwdbwd)."""
    h, w = fx.shape
    # (b) reactive residual
    pred = warp_lr(lr_prev_1c3hw, fx, fy)
    react = (lr_cur_1c3hw - pred).abs().mean(dim=1).squeeze(0)  # [H,W]
    occ = (~torch.isfinite(fx)) | (react > tau_react)
    if mode == "adaptive":
        tau = ADAPTIVE_TAU if adaptive_tau is None else adaptive_tau
        use_fwdbwd = float(occ.float().mean().item()) > tau   # one device->host sync
    else:
        use_fwdbwd = (mode == "full")
    if use_fwdbwd:
        # (c) forward flow via softmax splat (collisions won by lower-residual sources)
        ffx, ffy, _ = _softmax_splat_flow(-fx, -fy, fx, fy, -react)
        ffx = torch.nan_to_num(ffx, nan=1e6)
        ffy = torch.nan_to_num(ffy, nan=1e6)
        gx, gy = _base_grid(h, w)
        sx = gx + torch.nan_to_num(fx)
        sy = gy + torch.nan_to_num(fy)
        wf_x = _sample_const(ffx, sx, sy, 1e6)   # w~ (fwd flow at the mapped location)
        wf_y = _sample_const(ffy, sx, sy, 1e6)
        wb_x = torch.nan_to_num(fx)              # w^ (backward flow)
        wb_y = torch.nan_to_num(fy)
        lhs = (wf_x + wb_x) ** 2 + (wf_y + wb_y) ** 2
        rhs = 0.01 * (wf_x ** 2 + wf_y ** 2 + wb_x ** 2 + wb_y ** 2) + 0.5
        occ = occ | (lhs > rhs)
    return occ, use_fwdbwd


def upsample_bool(mask_hw, scale):
    """Nearest-upsample a bool LR mask to HD. == cv2.resize(uint8, NEAREST).astype(bool)."""
    up = F.interpolate(mask_hw[None, None].float(), scale_factor=scale, mode="nearest")[0, 0]
    return up > 0.5


def frac_true(mask_hw):
    return float(mask_hw.float().mean().item())
