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


def build_anchor_cache(frames, w_hd, h_hd, sr_mode, occ_mode="adaptive", fallback_thresh=0.08):
    """HYBRID anchor cache (the instant-mode default). Runs the SR net on:
      * the ANCHORS (I + first backbone), AND
      * any non-anchor BACKBONE (I/P) frame whose LR occlusion-fallback fraction exceeds
        `fallback_thresh` -- decided by a CHEAP scan of the ~backbone frames ONLY (a small
        fraction of the chunk; B leaves are NOT scanned here).
    Bicubic for everything else. Because the backbone is the propagation chain, putting a
    high-fallback backbone frame's compact-SR INTO the cache (before reconstruct) makes its
    detail propagate correctly down the chain (no bicubic contamination of later frames) --
    this is what the cheap post-reconstruct patch could not do. B-frame fallback (leaves) is
    handled afterwards by patch_high_fallback. Returns (cache, info, sr_set)."""
    import sr as _srmod
    N = len(frames)
    anchors, backbone = anchor_indices(frames)

    # ---- cheap scan of the BACKBONE chain only -> which P frames to SR in the cache ----
    t_scan0 = time.perf_counter()
    bb_fracs, sr_set = {}, set(anchors)
    for i in backbone:
        if i in anchors:
            continue
        bb_fracs[i] = _lr_fallback_fraction(frames, i, backbone, occ_mode)
        if bb_fracs[i] > fallback_thresh:
            sr_set.add(i)
    t_scan = time.perf_counter() - t_scan0

    _srmod.load_model(sr_mode)
    _warm_sr(_srmod, sr_mode, frames[0][1])            # one-off MPS graph compile (per process)
    t0 = time.perf_counter()
    cache = {}
    for i in range(N):
        cache[i] = (_srmod.upscale_to(frames[i][1], w_hd, h_hd, model=sr_mode) if i in sr_set
                    else cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC))
    info = {"n_frames": N, "n_anchors": len(anchors), "anchors": sorted(anchors),
            "backbone_upgrades": sorted(sr_set - anchors), "bb_fallback_fracs": bb_fracs,
            "t_scan_s": round(t_scan, 4), "t_cache_s": round(time.perf_counter() - t0, 4)}
    return cache, info, sr_set


def patch_high_fallback(frames, R, w_hd, h_hd, sr_mode, fallback_thresh=0.08, skip=None,
                        collect_info=True):
    """Adaptive safeguard for the LEAF frames (post-reconstruct). For each frame whose exact,
    anchor-invariant occlusion-fallback fraction (R[i]['hole_frac']) exceeds `fallback_thresh`
    and is NOT already SR'd in the cache (`skip` = the cache's sr_set), run the real SR net and
    patch its detail into EXACTLY that frame's fallback pixels (R[i]['mask']). For a B leaf this
    is fully correct (it is never a reference, so nothing propagates from it); the warped pixels
    came from the already-correct backbone. Operates in place on the GPU-resident recon tensor.
    Returns the upgrade accounting + per-frame fallback fractions for the quality report."""
    import torch
    import gpu_ops as G
    import sr as _srmod
    anchors, _ = anchor_indices(frames)
    skip = set(skip or ()) | set(anchors)
    N = len(frames)
    upgraded, fracs = [], {}
    for i in range(N):
        hf = float(R[i].get("hole_frac", 0.0))
        fracs[i] = hf
        if i in skip or R[i].get("mask") is None:
            continue
        if hf > fallback_thresh:
            sr_hd = G.img_to_dev(_srmod.upscale_to(frames[i][1], w_hd, h_hd, model=sr_mode))
            mask = R[i]["mask"]
            if not torch.is_tensor(mask):
                mask = torch.from_numpy(np.ascontiguousarray(mask)).to(sr_hd.device)
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
def _lr_fallback_fraction(frames, i, backbone_idx, occ_mode):
    """Estimate frame i's occlusion-fallback fraction at LR (no HD warp). Mirrors the mask
    reconstruct() builds: non-anchor backbone P -> past occlusion fraction; B leaf -> the
    `none` (past AND future occluded) fraction. Used only by the pre-scan path / diagnostics."""
    pt, lr, mvs = frames[i]
    h_lr, w_lr = lr.shape[:2]

    def occ_dir(ref_i, want):
        fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want=want)
        occ, _ = derisk.occlusion_mask_lr(fx, fy, lr, frames[ref_i][1], mode=occ_mode)
        return occ

    prev_ip = max([b for b in backbone_idx if b < i], default=None)
    next_ip = min([b for b in backbone_idx if b > i], default=None)
    if pt in ("I", "P"):
        return 0.0 if prev_ip is None else float(occ_dir(prev_ip, "past").mean())
    occ_p = occ_dir(prev_ip, "past") if prev_ip is not None else np.ones((h_lr, w_lr), bool)
    occ_f = occ_dir(next_ip, "future") if next_ip is not None else np.ones((h_lr, w_lr), bool)
    return float((occ_p & occ_f).mean())


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
