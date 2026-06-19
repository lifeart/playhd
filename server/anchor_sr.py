"""Anchor-only SR cache + adaptive safeguard -- Lever 1 of the instant-mode speedup.

derisk.build_perframe_cache runs the SR net on EVERY frame of a chunk, but
derisk.reconstruct(anchor_set=set()) reads the full SR image only for ANCHOR frames
(every I-frame + the chunk's first backbone frame -- no in-window predecessor). For every
other frame the SR image is consumed ONLY at occlusion-fallback pixels:
  * a non-anchor backbone P:  recon = where(occ, perframe, warp(prev_recon)+residual)
  * a B leaf:                 recon = pf.clone(); warps overwrite all but the `none`
                              (both-directions-occluded) pixels.
So SR'ing the 27 non-anchor frames of a 28-frame chunk is wasted everywhere except a small
fallback fraction. We run the real SR net only on the anchors and bicubic-upscale the rest
(used only at fallback pixels) -> the PROPAGATED (warped) pixels trace back to the same SR'd
anchors and are unchanged; only the fallback pixels differ (bicubic vs compact-SR).

ADAPTIVE SAFEGUARD -- so soft bicubic fallback never dominates a high-motion frame. The
per-frame occlusion-fallback fraction (hole_frac) is ANCHOR-INVARIANT: it depends only on a
frame's MVs vs its reference, NOT on the recon content (derisk._adaptive_fallback relies on
this). So ONE reconstruct pass with the cheap bicubic cache returns the EXACT fallback fraction
for every frame for free (no separate mask scan). Any non-anchor frame whose fraction exceeds
`fallback_thresh` (default 8%, the high-motion regime) gets a real SR call whose detail is
patched into exactly its fallback pixels (patch_high_fallback). derisk + sr are READ-ONLY.
"""
import time

import cv2
import numpy as np

import derisk

# MPS graph compilation for the SR net is a one-off cost; warm it only the FIRST time a model is
# used (per process), not once per chunk -- a per-chunk warmup wastes a full SR forward per chunk.
_WARMED = set()


def _warm_sr(srmod, sr_mode, frame):
    if sr_mode not in _WARMED:
        srmod.upscale(frame, model=sr_mode)        # compile the MPS graph on the real frame size
        srmod.reset_latency(sr_mode)
        _WARMED.add(sr_mode)


def anchor_indices(frames):
    """The frames derisk.reconstruct(anchor_set=set()) reads the FULL SR image from: every
    I-frame + the chunk's first backbone (I/P) frame (it has no in-window predecessor)."""
    backbone = derisk.backbone_indices(frames)
    first = backbone[0] if backbone else None
    return {i for i in backbone if frames[i][0] == "I" or i == first}, backbone


# --------------------------------------------------------------------------- #
# Lever 4a: GPU-resident bicubic upscale (avoids the CPU cv2.resize + the 9.8 MB HD upload).
# --------------------------------------------------------------------------- #
def _gpu_bicubic(lr_u8, w_hd, h_hd):
    """LR uint8 HxWx3 -> [1,3,h_hd,w_hd] float32 (0..255) on device, bicubic. Resident; consumed
    directly by reconstruct_torch (gpu_ops.img_to_dev passes a tensor through untouched)."""
    import torch
    import torch.nn.functional as F
    import gpu_ops as G
    t = G.img_to_dev(lr_u8)                                   # [1,3,h,w] float, single small upload
    up = F.interpolate(t, size=(h_hd, w_hd), mode="bicubic", align_corners=False)
    return up.clamp_(0.0, 255.0)


# --------------------------------------------------------------------------- #
# Lever 3: tile super-resolution -- SR only the bounding box of a frame's occlusion-fallback
# region. The SR there is consumed ONLY at fallback pixels (a backbone P reads pf at occ pixels;
# a B leaf overwrites all but the both-occluded pixels), so the rest of the frame never reads it.
# --------------------------------------------------------------------------- #
_TILE_PAD_LR = 12   # LR-pixel halo SR'd around the bbox then discarded, so the kept interior is
# free of the compact net's conv-border effect (verified PSNR-vs-full-SR in _quality_instant.py).


def _bbox_of(mask_lr, pad, w_lr, h_lr):
    """Padded bounding box (x0,y0,x1,y1) of a True LR mask, or None if empty."""
    ys, xs = np.where(mask_lr)
    if ys.size == 0:
        return None
    x0 = max(int(xs.min()) - pad, 0); x1 = min(int(xs.max()) + 1 + pad, w_lr)
    y0 = max(int(ys.min()) - pad, 0); y1 = min(int(ys.max()) + 1 + pad, h_lr)
    return x0, y0, x1, y1


def _sr_tile(lr_u8, bbox_lr, scale, sr_mode):
    """SR the LR bbox tile (x4) -> (tile_hd_u8, hd_bbox). The halo is SR'd for border accuracy
    then cropped away to the exact bbox so only well-conditioned interior pixels are kept."""
    import sr as _srmod
    x0, y0, x1, y1 = bbox_lr
    tile = lr_u8[y0:y1, x0:x1]
    sr = _srmod.upscale(tile, model=sr_mode)                 # ((y1-y0)*scale)x((x1-x0)*scale)x3
    hx0, hy0, hx1, hy1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
    return sr[:hy1 - hy0, :hx1 - hx0], (hx0, hy0, hx1, hy1)


def _place_tile(hd_base, sr_tile, hd_bbox):
    """Write an SR tile into an HD frame at hd_bbox. hd_base may be numpy (CPU cache) or a
    [1,3,H,W] device tensor (gpu_cache); the SR tile is uint8 HxWx3."""
    hx0, hy0, hx1, hy1 = hd_bbox
    if isinstance(hd_base, np.ndarray):
        hd_base[hy0:hy1, hx0:hx1] = sr_tile
        return hd_base
    import torch
    t = torch.from_numpy(np.ascontiguousarray(sr_tile)).to(hd_base.device)
    hd_base[0, :, hy0:hy1, hx0:hx1] = t.permute(2, 0, 1).float()
    return hd_base


def build_anchor_cache(frames, w_hd, h_hd, sr_mode, occ_mode="adaptive", fallback_thresh=0.08,
                       tile=False, gpu_cache=False, thresh_fn=None):
    """HYBRID anchor cache (the instant-mode default). Runs the SR net on:
      * the ANCHORS (I + first backbone), AND
      * any non-anchor BACKBONE (I/P) frame whose LR occlusion-fallback fraction exceeds
        `fallback_thresh` -- decided by a CHEAP scan of the ~backbone frames ONLY (a small
        fraction of the chunk; B leaves are NOT scanned here).
    Bicubic for everything else. Because the backbone is the propagation chain, putting a
    high-fallback backbone frame's compact-SR INTO the cache (before reconstruct) makes its
    detail propagate correctly down the chain (no bicubic contamination of later frames) --
    this is what the cheap post-reconstruct patch could not do. B-frame fallback (leaves) is
    handled afterwards by patch_high_fallback.

    `tile` (Lever 3): an UPGRADED backbone frame is SR'd only over the bounding box of its
    occlusion-fallback region (the only place its cache is read -- the rest is overwritten by
    the warp) and bicubic elsewhere, not full-frame. `gpu_cache` (Lever 4a): the cache holds
    GPU-resident [1,3,H,W] tensors (bicubic done on-device from the small LR upload), so
    reconstruct_torch never does the per-frame HD host->device upload. Anchors are ALWAYS full
    SR (read everywhere, drift 0). Returns (cache, info, sr_set)."""
    import sr as _srmod
    N = len(frames)
    h_lr, w_lr = frames[0][1].shape[:2]
    scale = w_hd // w_lr
    anchors, backbone = anchor_indices(frames)

    # ---- cheap scan of the BACKBONE chain only -> which P frames to SR in the cache ----
    t_scan0 = time.perf_counter()
    bb_fracs, sr_set, bb_masks = {}, set(anchors), {}
    for i in backbone:
        if i in anchors:
            continue
        m = _lr_fallback_mask(frames, i, backbone, occ_mode)
        bb_fracs[i] = float(m.mean())
        # E2 (motion-keyed fallback): thresh_fn(i) returns a per-frame threshold (lower on
        # high-motion frames) so a high-occlusion backbone frame gets a real compact-SR cache
        # entry only where it matters; thresh_fn=None (default) -> the scalar threshold = today's
        # behavior exactly (instant byte-identical).
        thr = thresh_fn(i) if thresh_fn is not None else fallback_thresh
        if bb_fracs[i] > thr:
            sr_set.add(i)
            bb_masks[i] = m
    t_scan = time.perf_counter() - t_scan0

    _srmod.load_model(sr_mode)
    _warm_sr(_srmod, sr_mode, frames[0][1])            # one-off MPS graph compile (per process)
    t0 = time.perf_counter()
    cache, tile_area = {}, []

    def base_hd(i):
        return _gpu_bicubic(frames[i][1], w_hd, h_hd) if gpu_cache else \
            cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)

    def full_sr(i):
        sr = _srmod.upscale_to(frames[i][1], w_hd, h_hd, model=sr_mode)
        import gpu_ops as G
        return G.img_to_dev(sr) if gpu_cache else sr

    for i in range(N):
        if i in anchors:
            cache[i] = full_sr(i)                       # anchors: always full SR
        elif i in sr_set:
            if tile:                                    # Lever 3: SR only the fallback bbox
                bbox = _bbox_of(bb_masks[i], _TILE_PAD_LR, w_lr, h_lr)
                if bbox is None:
                    cache[i] = base_hd(i)
                else:
                    sr_tile, hd_bbox = _sr_tile(frames[i][1], bbox, scale, sr_mode)
                    cache[i] = _place_tile(base_hd(i), sr_tile, hd_bbox)
                    tile_area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (w_lr * h_lr))
            else:
                cache[i] = full_sr(i)
        else:
            cache[i] = base_hd(i)                       # bicubic fallback source
    info = {"n_frames": N, "n_anchors": len(anchors), "anchors": sorted(anchors),
            "backbone_upgrades": sorted(sr_set - anchors), "bb_fallback_fracs": bb_fracs,
            "tile": tile, "gpu_cache": gpu_cache,
            "mean_tile_area_frac": round(float(np.mean(tile_area)), 4) if tile_area else 0.0,
            "t_scan_s": round(t_scan, 4), "t_cache_s": round(time.perf_counter() - t0, 4)}
    return cache, info, sr_set


def patch_high_fallback(frames, R, w_hd, h_hd, sr_mode, fallback_thresh=0.08, skip=None,
                        collect_info=True, tile=False, thresh_fn=None):
    """Adaptive safeguard for the LEAF frames (post-reconstruct). For each frame whose exact,
    anchor-invariant occlusion-fallback fraction (R[i]['hole_frac']) exceeds `fallback_thresh`
    and is NOT already SR'd in the cache (`skip` = the cache's sr_set), run the real SR net and
    patch its detail into EXACTLY that frame's fallback pixels (R[i]['mask']). For a B leaf this
    is fully correct (it is never a reference, so nothing propagates from it); the warped pixels
    came from the already-correct backbone. Operates in place on the GPU-resident recon tensor.
    `tile` (Lever 3): SR only the bounding box of the frame's fallback mask (the patch only
    writes inside the mask anyway) instead of the full 2560x1280.
    Returns the upgrade accounting + per-frame fallback fractions for the quality report."""
    import torch
    import gpu_ops as G
    import sr as _srmod
    h_lr, w_lr = frames[0][1].shape[:2]
    scale = w_hd // w_lr
    anchors, _ = anchor_indices(frames)
    skip = set(skip or ()) | set(anchors)
    N = len(frames)
    upgraded, fracs = [], {}
    for i in range(N):
        hf = float(R[i].get("hole_frac", 0.0))
        fracs[i] = hf
        if i in skip or R[i].get("mask") is None:
            continue
        thr = thresh_fn(i) if thresh_fn is not None else fallback_thresh   # E2 motion-keyed
        if hf > thr:
            mask = R[i]["mask"]
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(np.ascontiguousarray(mask)).to(R[i]["recon"].device)
            if tile:
                # SR only the fallback bbox; place it into a copy of the recon, then masked-select.
                ys, xs = torch.where(mask)
                if ys.numel() == 0:
                    continue
                hx0 = int(xs.min()); hx1 = int(xs.max()) + 1
                hy0 = int(ys.min()); hy1 = int(ys.max()) + 1
                # map HD bbox back to LR (+halo), SR the LR tile, place into a full HD canvas.
                lx0 = max(hx0 // scale - _TILE_PAD_LR, 0); lx1 = min(-(-hx1 // scale) + _TILE_PAD_LR, w_lr)
                ly0 = max(hy0 // scale - _TILE_PAD_LR, 0); ly1 = min(-(-hy1 // scale) + _TILE_PAD_LR, h_lr)
                sr_tile, hd_bbox = _sr_tile(frames[i][1], (lx0, ly0, lx1, ly1), scale, sr_mode)
                sr_hd = R[i]["recon"].clone()
                _place_tile(sr_hd, sr_tile, hd_bbox)
            else:
                sr_hd = G.img_to_dev(_srmod.upscale_to(frames[i][1], w_hd, h_hd, model=sr_mode))
            R[i]["recon"] = torch.where(mask[None, None], sr_hd, R[i]["recon"])
            upgraded.append(i)
    info = {}
    if collect_info:
        n_cache_sr = len(set(skip))
        info = {
            "leaf_upgrades": upgraded,
            "n_adaptive_upgrades": len(upgraded) + (n_cache_sr - len(anchors)),
            "n_sr_calls": n_cache_sr + len(upgraded),
            "sr_calls_per_frame": round((n_cache_sr + len(upgraded)) / max(1, N), 4),
            "fallback_fracs": {i: round(fracs[i], 5) for i in range(N)},
            "max_fallback_frac": round(max(fracs.values()) if fracs else 0.0, 5),
            "fallback_thresh": fallback_thresh,
        }
    return info


# --------------------------------------------------------------------------- #
# Optional PRE-SCAN path (correct propagation for upgraded backbone frames, at the cost of a
# cheap LR occlusion scan). Kept for the A/B comparison in bench_instant.py.
# --------------------------------------------------------------------------- #
def _lr_fallback_mask(frames, i, backbone_idx, occ_mode):
    """Frame i's occlusion-fallback mask at LR (no HD warp). Mirrors the mask reconstruct()
    builds: non-anchor backbone P -> past occlusion; B leaf -> the `none` (past AND future
    occluded). Used for the backbone scan (fraction + tile bbox) and diagnostics."""
    pt, lr, mvs = frames[i]
    h_lr, w_lr = lr.shape[:2]

    def occ_dir(ref_i, want):
        fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want=want)
        occ, _ = derisk.occlusion_mask_lr(fx, fy, lr, frames[ref_i][1], mode=occ_mode)
        return occ

    prev_ip = max([b for b in backbone_idx if b < i], default=None)
    next_ip = min([b for b in backbone_idx if b > i], default=None)
    if pt in ("I", "P"):
        return (occ_dir(prev_ip, "past") if prev_ip is not None
                else np.zeros((h_lr, w_lr), bool))
    occ_p = occ_dir(prev_ip, "past") if prev_ip is not None else np.ones((h_lr, w_lr), bool)
    occ_f = occ_dir(next_ip, "future") if next_ip is not None else np.ones((h_lr, w_lr), bool)
    return occ_p & occ_f


def _lr_fallback_fraction(frames, i, backbone_idx, occ_mode):
    """Frame i's LR occlusion-fallback fraction (mask mean). Used by the pre-scan path."""
    return float(_lr_fallback_mask(frames, i, backbone_idx, occ_mode).mean())


def build_anchor_cache_prescan(frames, w_hd, h_hd, sr_mode, occ_mode="adaptive",
                               fallback_thresh=0.08):
    """Pre-scan variant: SR the anchors AND every frame whose LR-scanned fallback fraction
    exceeds `fallback_thresh` BEFORE reconstruct, so an upgraded backbone frame's compact-SR
    detail propagates correctly down the chain. Returns (cache, info)."""
    import sr as _srmod
    N = len(frames)
    anchors, backbone = anchor_indices(frames)
    t_scan0 = time.perf_counter()
    fracs, sr_idx = {}, set(anchors)
    for i in range(N):
        if i in anchors:
            fracs[i] = 0.0
            continue
        fracs[i] = _lr_fallback_fraction(frames, i, backbone, occ_mode)
        if fracs[i] > fallback_thresh:
            sr_idx.add(i)
    t_scan = time.perf_counter() - t_scan0

    _srmod.load_model(sr_mode)
    _warm_sr(_srmod, sr_mode, frames[0][1])
    t0 = time.perf_counter()
    cache = {}
    for i in range(N):
        cache[i] = (_srmod.upscale_to(frames[i][1], w_hd, h_hd, model=sr_mode) if i in sr_idx
                    else cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC))
    info = {
        "n_frames": N, "n_anchors": len(anchors), "anchors": sorted(anchors),
        "n_sr_calls": len(sr_idx), "n_adaptive_upgrades": len(sr_idx - anchors),
        "adaptive_upgrades": sorted(sr_idx - anchors),
        "sr_calls_per_frame": round(len(sr_idx) / max(1, N), 4),
        "fallback_fracs": {i: round(fracs[i], 5) for i in range(N)},
        "max_fallback_frac": round(max(fracs.values()) if fracs else 0.0, 5),
        "fallback_thresh": fallback_thresh,
        "t_scan_s": round(t_scan, 4), "t_build_s": round(time.perf_counter() - t0, 4),
    }
    return cache, info


# --------------------------------------------------------------------------- #
# R3-E3: HF-only temporal-EMA soft-occlusion fallback (default-OFF, output-only). REPLACES the hard
# patch_high_fallback SR-patch on the instant path. Per non-anchor frame keep ONE HD-float EMA of
# (sr - bicubic) [reset at every I-frame / scene-cut / chunk start] and blend
#   R[i]['recon'] = (1-a)*recon + a*(bicubic + ema_HF),   a = gain*feather(mask)*conf
# -> low-freq stays fresh (tOF-safe), only flickery HF is temporally smoothed. Escapes the
# high-motion tOF<->fallback% frontier (eff-bic 7.70->6.35% at tOF +2.0% vs the hard switch's +20%;
# verified experiments/r3_e3_softocc_wire). Output-only: runs AFTER reconstruct, never feeds R[]'s
# reference role (GOTCHA #16). Cost ~1 compact-SR call per non-anchor frame -> a quality knob, not
# real-time; bound with motion_gate. Returns patch_high_fallback-compatible stats keys.
# --------------------------------------------------------------------------- #
_SOFTOCC_CONF_LO, _SOFTOCC_CONF_HI = 6.0, 26.0   # reactive-residual confidence ramp


def softocc_reset_indices(frames, extra_cuts=()):
    """Indices where the HF-EMA MUST reset: chunk start (0) + every I-frame + any scene cut.
    A missed reset -> pre-cut HF bleeds across = cross-cut ghost (verified 4.6-RMS in R3-E3)."""
    return {0} | {i for i in range(len(frames)) if frames[i][0] == "I"} | set(extra_cuts)


def _softocc_feather(mask_bool, k):
    if k < 3:
        return mask_bool.astype(np.float32)
    k = int(k) | 1
    return cv2.GaussianBlur(mask_bool.astype(np.float32), (k, k), 0)


def _softocc_mean_mv(mvs, h_lr, w_lr):
    if mvs is None or len(mvs) == 0:
        return 0.0
    fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want="all")
    mag = np.sqrt(fx * fx + fy * fy)
    return float(np.nanmean(mag)) if np.isfinite(mag).any() else 0.0


def _softocc_conf(frames, i, w_hd, h_hd):
    """HD confidence-to-use-SR in [0,1] from the reactive residual (the same signal
    occlusion_mask_lr's reactive term forms). P -> residual vs prev backbone; B/no-MV -> ones."""
    pt, lr_cur, mvs = frames[i]
    h_lr, w_lr = lr_cur.shape[:2]
    prev_bb = max([b for b in derisk.backbone_indices(frames) if b < i], default=None)
    if pt != "P" or prev_bb is None or mvs is None or len(mvs) == 0:
        return np.ones((h_hd, w_hd), np.float32)
    fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want="past")
    pred = derisk.warp_lr(frames[prev_bb][1], fx, fy).astype(np.float32)
    react = np.abs(lr_cur.astype(np.float32) - pred).mean(axis=2)
    c = np.clip((react - _SOFTOCC_CONF_LO) / (_SOFTOCC_CONF_HI - _SOFTOCC_CONF_LO), 0.0, 1.0)
    c[~np.isfinite(fx)] = 1.0
    return cv2.resize(c, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)


def softocc_patch(frames, R, w_hd, h_hd, sr_mode, *, anchors, backbone, reset_idx,
                  gain=0.6, beta=0.85, feather_k=31, occ_mode="reactive", skip=None,
                  motion_gate=None):
    """HF-EMA soft-occlusion on the GPU-resident recon chain (drop-in for patch_high_fallback; same
    return keys). R[i]['recon'] = [1,3,H,W] float device tensor; R[i]['mask'] = HxW bool device
    tensor (download_output=False). In place, output-only. The EMA advances on every RUN frame
    (anchors always seed it, reusing the cache's SR -> no extra call); each non-anchor RUN frame
    costs ONE compact-SR call. `motion_gate` (e.g. 1.0): run only where mean|MV| exceeds it (fewer
    calls, shallower escape); None = every non-anchor frame (the full verified escape)."""
    import torch
    import gpu_ops as G
    import sr as _srmod
    h_lr, w_lr = frames[0][1].shape[:2]
    dev = G.device()
    skip = set(skip or ()) | set(anchors)
    ema, blended, n_sr_runs = None, [], 0
    for i in range(len(frames)):
        if i in reset_idx:
            ema = None
        pt, lr_cur, mvs = frames[i]
        is_anchor = i in anchors or R[i].get("mask") is None
        run = is_anchor or motion_gate is None or _softocc_mean_mv(mvs, h_lr, w_lr) > motion_gate
        if run:
            bic_t = G.img_to_dev(cv2.resize(lr_cur, (w_hd, h_hd), interpolation=cv2.INTER_CUBIC))
            hf = G.img_to_dev(_srmod.upscale_to(lr_cur, w_hd, h_hd, model=sr_mode)) - bic_t
            ema = hf if ema is None else (beta * ema + (1.0 - beta) * hf)
            if not is_anchor:
                n_sr_runs += 1
        if is_anchor or not run or ema is None:
            continue
        m = R[i]["mask"]
        m_np = m.detach().to("cpu").numpy().astype(bool) if torch.is_tensor(m) else np.asarray(m, bool)
        a_np = np.clip(gain * _softocc_feather(m_np, feather_k) * _softocc_conf(frames, i, w_hd, h_hd),
                       0.0, 1.0).astype(np.float32)
        a_t = torch.from_numpy(np.ascontiguousarray(a_np)).to(dev)[None, None]
        R[i]["recon"] = ((1.0 - a_t) * R[i]["recon"] + a_t * (bic_t + ema)).clamp(0, 255)
        blended.append(i)
    n_cache_sr = len(skip)
    return {"softocc": True, "leaf_upgrades": blended, "n_softocc_sr_runs": n_sr_runs,
            "n_adaptive_upgrades": len(blended), "n_sr_calls": n_cache_sr + n_sr_runs,
            "sr_calls_per_frame": round((n_cache_sr + n_sr_runs) / max(1, len(frames)), 4)}
