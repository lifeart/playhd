#!/usr/bin/env python3
"""
R13-E4: FSR2 `ComputeTemporalMotionDivergence` ported to codec MVs, + the variant
wiring that lets us A/B it against this repo's incumbent Ruder softmax-splat
forward-backward check WITHOUT editing any shipped file.

THE PORTED OP (AMD FidelityFX FSR2, MIT, ffx_fsr2_depth_clip.h:136-147):

    fUv              = (px + 0.5) / RenderSize
    fMotionVector    = LoadDilatedMotionVector(px)              # UV units, cur -> prev
    fReprojectedUv   = clamp(fUv + fMotionVector)
    fPrevMotionVector= SamplePreviousDilatedMotionVector(fReprojectedUv)
    fPxDistance      = length(fMotionVector * DisplaySize)      # DISPLAY pixels
    return fPxDistance > 1
           ? lerp(0, 1 - saturate(len(fPrevMV)/len(fMV)), saturate((fPxDistance/20)^3))
           : 0

Splat-free: one gather (sample last frame's MV field at the reprojected position),
no scatter/softmax-splat. It fires where a pixel is moving NOW but the content it
reprojects onto was NOT moving before -- i.e. history there is not this pixel's
history (a disocclusion / newly-revealed surface).

PORT DECISIONS (all measured, all switchable so the report can defend them):
  * codec MVs are BACKWARD (dst-indexed) and already in LR pixels, so
    `fMotionVector` == our (fx, fy) and the reprojected position is (x+fx, y+fy)
    -- no UV round-trip needed (that is FSR2's texture-sampling convention only).
  * "last frame's MV field" == the MV field of the frame we are WARPING FROM (the
    reference's own backward flow). That is the exact analogue of FSR2's previous
    RENDERED frame's dilated MVs: the motion state of the surface we are pulling
    history out of.
  * temporal-distance NORMALISATION (`normalize=True`, default): unlike a game's
    fixed frame delta, a P-frame's reference can be ~6 display frames back while
    that reference's own hop is a different length. Magnitudes are divided by
    their display-frame distance so the ratio compares per-frame VELOCITY, not
    raw displacement. `normalize=False` reproduces the literal FSR2 formula.
  * `fPxDistance` is measured at DISPLAY resolution in FSR2, so we multiply the LR
    flow magnitude by `scale` (the pipeline's LR->HD factor). `gate_px` (FSR2: 20)
    and `min_px` (FSR2: 1) stay in display pixels.
  * FSR2's output is a CONTINUOUS reactive weight; this pipeline's occlusion mask
    is BINARY, so we threshold it (`thr`). thr is swept.
  * NaN previous flow (the reference frame was intra-coded there) is a real signal:
    `nan_prev="flag"` treats it as zero previous motion (=> divergence 1 => flag,
    the conservative reading), `nan_prev="skip"` treats it as no information.

NO shipped file is edited: `patch()` swaps `derisk.occlusion_mask_lr` /
`gpu_ops.occlusion_mask_lr` for a wrapper that CALLS THE ORIGINAL for the base
(intra + reactive [+ Ruder]) signals and only unions in the extra term, so the
incumbent path stays bit-for-bit the production code.
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if os.path.join(_ROOT, "prototype") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "prototype"))

import cv2                      # noqa: E402
import derisk as D              # noqa: E402  READ-ONLY prototype import


# --------------------------------------------------------------------------- #
# The op itself (numpy)
# --------------------------------------------------------------------------- #
def dilate_mv(fx, fy):
    """FSR2 feeds `ComputeTemporalMotionDivergence` DILATED motion vectors: within a
    3x3 neighbourhood it takes the MV of the sample CLOSEST TO THE CAMERA, so a
    moving object's MV bleeds one pixel outward and the divergence fires on the
    disoccluding EDGE rather than one pixel late. Decoded video has no depth
    (GOTCHA #8), so the codec analogue is the LARGEST-MAGNITUDE MV of the 3x3
    neighbourhood (the moving foreground dominates a static background), with NaN
    treated as magnitude 0 so real MVs win over intra holes."""
    h, w = fx.shape
    mag = np.where(np.isfinite(fx) & np.isfinite(fy),
                   np.sqrt(np.nan_to_num(fx) ** 2 + np.nan_to_num(fy) ** 2), -1.0)
    best_m = mag.copy()
    best_x = fx.copy()
    best_y = fy.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            sm = np.roll(np.roll(mag, dy, axis=0), dx, axis=1)
            sx = np.roll(np.roll(fx, dy, axis=0), dx, axis=1)
            sy = np.roll(np.roll(fy, dy, axis=0), dx, axis=1)
            take = sm > best_m
            best_m = np.where(take, sm, best_m)
            best_x = np.where(take, sx, best_x)
            best_y = np.where(take, sy, best_y)
    return best_x.astype(np.float32), best_y.astype(np.float32)


def temporal_mv_divergence(fx, fy, prev_fx, prev_fy, scale,
                           dist_cur=1.0, dist_prev=1.0,
                           gate_px=20.0, min_px=1.0, normalize=True,
                           nan_prev="flag", dilate=False):
    """FSR2 ComputeTemporalMotionDivergence on an LR codec-MV field.

    fx, fy         : current backward flow (LR px, NaN where no MV).
    prev_fx/prev_fy: the REFERENCE frame's own backward flow (LR px, NaN where none)
                     or None (no history yet -> returns zeros, like FSR2 frame 0).
    scale          : LR->display factor (fPxDistance is a DISPLAY-pixel quantity).
    dist_cur/prev  : display-frame distances of the two hops (velocity normalisation).
    Returns float32 HxW divergence in [0,1] (0 = history trustworthy)."""
    h, w = fx.shape
    if prev_fx is None:
        return np.zeros((h, w), np.float32)
    if dilate:
        fx, fy = dilate_mv(fx, fy)
        prev_fx, prev_fy = dilate_mv(prev_fx, prev_fy)
    cx = np.nan_to_num(fx, nan=0.0).astype(np.float32)
    cy = np.nan_to_num(fy, nan=0.0).astype(np.float32)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mapx, mapy = gx + cx, gy + cy                      # reprojected position (clamped by remap)
    pvalid = (np.isfinite(prev_fx) & np.isfinite(prev_fy)).astype(np.float32)
    px = cv2.remap(np.nan_to_num(prev_fx, nan=0.0).astype(np.float32), mapx, mapy,
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    py = cv2.remap(np.nan_to_num(prev_fy, nan=0.0).astype(np.float32), mapx, mapy,
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pv = cv2.remap(pvalid, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dc = float(dist_cur) if normalize else 1.0
    dp = float(dist_prev) if normalize else 1.0
    mag_c = np.sqrt(cx * cx + cy * cy) / max(dc, 1e-6)
    mag_p = np.sqrt(px * px + py * py) / max(dp, 1e-6)
    ratio = np.where(mag_c > 1e-6, mag_p / np.maximum(mag_c, 1e-6), 1.0)
    div = 1.0 - np.clip(ratio, 0.0, 1.0)
    px_dist = np.sqrt(cx * cx + cy * cy) * float(scale) / max(dc, 1e-6)
    gate = np.clip((px_dist / float(gate_px)) ** 3, 0.0, 1.0)
    out = np.where(px_dist > float(min_px), div * gate, 0.0).astype(np.float32)
    if nan_prev == "flag":
        # prev flow undefined at the reprojection (reference was intra there):
        # zero previous motion => full divergence, still gated by pxDistance.
        out = np.where(pv < 0.5, np.where(px_dist > float(min_px), gate, 0.0), out)
    elif nan_prev == "skip":
        out = np.where(pv < 0.5, 0.0, out)
    else:
        raise ValueError(f"nan_prev={nan_prev!r}")
    return out.astype(np.float32)


def dilate_mv_torch(fx, fy):
    """Torch twin of dilate_mv (3x3 max-magnitude MV dilation)."""
    import torch
    mag = torch.where(torch.isfinite(fx) & torch.isfinite(fy),
                      torch.sqrt(torch.nan_to_num(fx) ** 2 + torch.nan_to_num(fy) ** 2),
                      torch.full_like(fx, -1.0))
    bm, bx, by = mag.clone(), fx.clone(), fy.clone()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            sm = torch.roll(mag, shifts=(dy, dx), dims=(0, 1))
            sx = torch.roll(fx, shifts=(dy, dx), dims=(0, 1))
            sy = torch.roll(fy, shifts=(dy, dx), dims=(0, 1))
            take = sm > bm
            bm = torch.where(take, sm, bm)
            bx = torch.where(take, sx, bx)
            by = torch.where(take, sy, by)
    return bx, by


def temporal_mv_divergence_torch(fx, fy, prev_fx, prev_fy, scale,
                                 dist_cur=1.0, dist_prev=1.0,
                                 gate_px=20.0, min_px=1.0, normalize=True,
                                 nan_prev="flag", dilate=False):
    """Torch/MPS twin (same math, same knobs). fx/fy/prev_* are [H,W] tensors."""
    import torch
    import gpu_ops as G
    h, w = fx.shape
    if prev_fx is None:
        return torch.zeros((h, w), device=fx.device)
    if dilate:
        fx, fy = dilate_mv_torch(fx, fy)
        prev_fx, prev_fy = dilate_mv_torch(prev_fx, prev_fy)
    cx = torch.nan_to_num(fx)
    cy = torch.nan_to_num(fy)
    gx, gy = G._base_grid(h, w)
    sx, sy = gx + cx, gy + cy
    pvalid = (torch.isfinite(prev_fx) & torch.isfinite(prev_fy)).float()
    px = G._sample(torch.nan_to_num(prev_fx)[None, None], sx, sy, "border")[0, 0]
    py = G._sample(torch.nan_to_num(prev_fy)[None, None], sx, sy, "border")[0, 0]
    pv = G._sample(pvalid[None, None], sx, sy, "border")[0, 0]
    dc = float(dist_cur) if normalize else 1.0
    dp = float(dist_prev) if normalize else 1.0
    mag_raw = torch.sqrt(cx * cx + cy * cy)
    mag_c = mag_raw / max(dc, 1e-6)
    mag_p = torch.sqrt(px * px + py * py) / max(dp, 1e-6)
    ratio = torch.where(mag_c > 1e-6, mag_p / torch.clamp_min(mag_c, 1e-6),
                        torch.ones_like(mag_c))
    div = 1.0 - ratio.clamp(0.0, 1.0)
    px_dist = mag_raw * float(scale) / max(dc, 1e-6)
    gate = ((px_dist / float(gate_px)) ** 3).clamp(0.0, 1.0)
    out = torch.where(px_dist > float(min_px), div * gate, torch.zeros_like(div))
    if nan_prev == "flag":
        out = torch.where(pv < 0.5,
                          torch.where(px_dist > float(min_px), gate, torch.zeros_like(gate)),
                          out)
    elif nan_prev == "skip":
        out = torch.where(pv < 0.5, torch.zeros_like(out), out)
    else:
        raise ValueError(f"nan_prev={nan_prev!r}")
    return out


# --------------------------------------------------------------------------- #
# Frame registry + call context (how the mask hook learns which frame it is on)
# --------------------------------------------------------------------------- #
class _Registry:
    """Maps the `mvs` array object a warp is currently using back to its display
    index, so the patched mask can fetch the REFERENCE frame's own flow field.

    `derisk.build_lr_flow(mvs, h, w, want=...)` is called immediately before the
    mask in BOTH backends' warp_one, with `frames[i][2]` -- so intercepting it is
    a backend-agnostic way to know (i, want) at mask time. Nothing shipped is
    modified: build_lr_flow is wrapped, the original does the work."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = None          # strong ref: keeps id()s alive
        self.idx = {}               # id(mvs) -> display index
        self.flow_past = {}         # i -> (fx, fy) cached LR flow (past refs)
        self.flow_future = {}
        self.backbone = []
        self.ctx = None             # (i, want) of the in-flight warp
        self.hits = 0
        self.misses = 0

    def register(self, frames):
        self.reset()
        self.frames = frames
        for i, f in enumerate(frames):
            mvs = f[2]
            if mvs is not None and len(mvs):
                self.idx[id(mvs)] = i
        self.backbone = D.backbone_indices(frames)

    def prev_ip(self, i):
        return max([b for b in self.backbone if b < i], default=None)

    def next_ip(self, i):
        return min([b for b in self.backbone if b > i], default=None)

    def flow(self, i, want):
        cache = self.flow_past if want == "past" else self.flow_future
        if i not in cache:
            f = self.frames[i]
            h, w = f[1].shape[:2]
            cache[i] = _ORIG_BUILD_FLOW(f[2], h, w, want=want)
        return cache[i]

    def history_for(self, i, want):
        """(prev_fx, prev_fy, dist_cur, dist_prev) for frame i's `want` direction."""
        if want == "past":
            ref = self.prev_ip(i)
        else:
            ref = self.next_ip(i)
        if ref is None:
            return None, None, 1.0, 1.0
        dist_cur = abs(i - ref)
        rprev = self.prev_ip(ref)
        if rprev is None:
            return None, None, float(dist_cur), 1.0
        pfx, pfy = self.flow(ref, "past")     # the reference's OWN backward flow
        return pfx, pfy, float(dist_cur), float(ref - rprev)


REG = _Registry()
_ORIG_BUILD_FLOW = D.build_lr_flow
_ORIG_OCC_NP = D.occlusion_mask_lr
_ORIG_RECON = D.reconstruct
_ORIG_OCC_T = None      # gpu_ops.occlusion_mask_lr, captured at patch() time
_PATCHED = False

# active variant config, set by set_variant()
CFG = dict(variant="full", thr=0.5, gate_px=20.0, min_px=1.0, normalize=True,
           nan_prev="flag", scale=4, ruder_a=0.01, ruder_b=0.5, dilate=False)

# variants:
#   'reactive'    (a)+(b)                      -- splat deleted
#   'full'        (a)+(b)+Ruder splat          -- incumbent
#   'adaptive'    production quality default   -- Ruder splat gated by ADAPTIVE_TAU
#   'fsr2'        (a)+(b)+FSR2 temporal        -- REPLACE the splat
#   'fsr2+ruder'  (a)+(b)+Ruder+FSR2           -- ADD to the splat
#   'ruder-tuned' (a)+(b)+Ruder w/ retuned thresholds
_BASE_MODE = {"reactive": "reactive", "full": "full", "adaptive": "adaptive",
              "fsr2": "reactive", "fsr2+ruder": "full", "ruder-tuned": "reactive",
              "fsr2+adaptive": "adaptive"}

STATS = dict(fsr2_area=[], base_area=[], calls=0)


def set_variant(variant, **kw):
    if variant not in _BASE_MODE:
        raise ValueError(f"unknown variant {variant!r}")
    CFG["variant"] = variant
    CFG.update(kw)
    STATS["fsr2_area"] = []
    STATS["base_area"] = []
    STATS["calls"] = 0


def occ_mode_for(variant):
    """The `occ` string to hand the production pipeline for this variant (the hook
    rewrites the semantics, but the pipeline must still take an occ code path)."""
    return _BASE_MODE[variant]


# --------------------------------------------------------------------------- #
# The patches
# --------------------------------------------------------------------------- #
def _build_flow_hook(mvs, h, w, want="all"):
    i = REG.idx.get(id(mvs))
    if i is not None:
        REG.ctx = (i, want)
    return _ORIG_BUILD_FLOW(mvs, h, w, want=want)


def _ruder_np(fb_x, fb_y, lr_cur, lr_prev, tau_react, a, b):
    """derisk's Ruder term with tunable thresholds (a=0.01, b=0.5 == shipped)."""
    h, w = fb_x.shape
    pred = D.warp_lr(lr_prev, fb_x, fb_y).astype(np.float32)
    react = np.abs(lr_cur.astype(np.float32) - pred).mean(axis=2)
    fwd = np.stack([-fb_x, -fb_y], axis=-1)
    ff, _ = D.softmax_splat(fwd, fb_x, fb_y, -react)
    ffx = np.nan_to_num(ff[:, :, 0], nan=1e6).astype(np.float32)
    ffy = np.nan_to_num(ff[:, :, 1], nan=1e6).astype(np.float32)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    sx, sy = gx + np.nan_to_num(fb_x), gy + np.nan_to_num(fb_y)
    wf_x = cv2.remap(ffx, sx, sy, cv2.INTER_LINEAR, borderValue=1e6)
    wf_y = cv2.remap(ffy, sx, sy, cv2.INTER_LINEAR, borderValue=1e6)
    wb_x, wb_y = np.nan_to_num(fb_x), np.nan_to_num(fb_y)
    lhs = (wf_x + wb_x) ** 2 + (wf_y + wb_y) ** 2
    rhs = a * (wf_x ** 2 + wf_y ** 2 + wb_x ** 2 + wb_y ** 2) + b
    return lhs > rhs


def _occ_hook_np(fb_x, fb_y, lr_cur, lr_prev, tau_react=16.0, mode="full",
                 adaptive_tau=None):
    """Patched derisk.occlusion_mask_lr. Calls the ORIGINAL for the production
    signals, then unions the variant's extra term."""
    v = CFG["variant"]
    base_mode = _BASE_MODE[v]
    if v == "ruder-tuned":
        occ, used = _ORIG_OCC_NP(fb_x, fb_y, lr_cur, lr_prev, tau_react, "reactive",
                                 adaptive_tau)
        extra = _ruder_np(fb_x, fb_y, lr_cur, lr_prev, tau_react,
                          CFG["ruder_a"], CFG["ruder_b"])
        STATS["base_area"].append(float(occ.mean()))
        STATS["fsr2_area"].append(float((extra & ~occ).mean()))
        STATS["calls"] += 1
        return (occ | extra).astype(bool), True
    occ, used = _ORIG_OCC_NP(fb_x, fb_y, lr_cur, lr_prev, tau_react, base_mode, adaptive_tau)
    if not v.startswith("fsr2"):
        return occ, used
    ctx = REG.ctx
    if ctx is None:
        REG.misses += 1
        return occ, used
    REG.hits += 1
    i, want = ctx
    pfx, pfy, dc, dp = REG.history_for(i, want)
    div = temporal_mv_divergence(fb_x, fb_y, pfx, pfy, CFG["scale"],
                                 dist_cur=dc, dist_prev=dp, gate_px=CFG["gate_px"],
                                 min_px=CFG["min_px"], normalize=CFG["normalize"],
                                 nan_prev=CFG["nan_prev"], dilate=CFG["dilate"])
    extra = div > CFG["thr"]
    STATS["base_area"].append(float(occ.mean()))
    STATS["fsr2_area"].append(float((extra & ~occ).mean()))
    STATS["calls"] += 1
    return (occ | extra).astype(bool), used


def _make_occ_hook_torch(orig):
    def _hook(fx, fy, lr_cur, lr_prev, tau_react=16.0, mode="full", adaptive_tau=None):
        v = CFG["variant"]
        occ, used = orig(fx, fy, lr_cur, lr_prev, tau_react, _BASE_MODE[v], adaptive_tau)
        if not v.startswith("fsr2"):
            return occ, used
        ctx = REG.ctx
        if ctx is None:
            REG.misses += 1
            return occ, used
        REG.hits += 1
        i, want = ctx
        pfx, pfy, dc, dp = REG.history_for(i, want)
        if pfx is None:
            return occ, used
        import gpu_ops as G
        pfx_t, pfy_t = G.flow_to_dev(pfx, pfy)
        div = temporal_mv_divergence_torch(fx, fy, pfx_t, pfy_t, CFG["scale"],
                                           dist_cur=dc, dist_prev=dp,
                                           gate_px=CFG["gate_px"], min_px=CFG["min_px"],
                                           normalize=CFG["normalize"],
                                           nan_prev=CFG["nan_prev"], dilate=CFG["dilate"])
        return occ | (div > CFG["thr"]), used
    return _hook


def _recon_hook(frames, *a, **kw):
    REG.register(frames)
    return _ORIG_RECON(frames, *a, **kw)


def patch(torch_too=True):
    """Install the hooks (idempotent). Nothing on disk is modified."""
    global _PATCHED
    if _PATCHED:
        return
    D.build_lr_flow = _build_flow_hook
    D.occlusion_mask_lr = _occ_hook_np
    D.reconstruct = _recon_hook
    if torch_too:
        try:
            import gpu_ops as G
        except ImportError as e:              # loud: torch path simply not available here
            print(f"[fsr2_mask] torch path NOT patched: {e!r}")
        else:
            # gpu_ops has no build_lr_flow of its own -- reconstruct_torch calls
            # derisk's module-global one, which is already hooked above.
            global _ORIG_OCC_T
            _ORIG_OCC_T = G.occlusion_mask_lr
            G.occlusion_mask_lr = _make_occ_hook_torch(_ORIG_OCC_T)
    _PATCHED = True


def unpatch():
    global _PATCHED
    D.build_lr_flow = _ORIG_BUILD_FLOW
    D.occlusion_mask_lr = _ORIG_OCC_NP
    D.reconstruct = _ORIG_RECON
    _PATCHED = False
