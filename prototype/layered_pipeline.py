"""layered_pipeline.py -- Stage L3 of the LAYERED architecture: COMPOSE the layers.

The layered render splits a static-camera talking-head frame into

    out_hd[i] = alpha_hd[i] * foreground_hd[i] + (1 - alpha_hd[i]) * plate_hd

where
  * plate_hd          = L2 heavy-SR'd STATIC background plate, sampled per frame
                        (sample_plate; static camera => IDENTITY => the background is
                        FIXED, zero flicker).
  * alpha_hd[i]       = L1 RVM alpha matte, upsampled to HD (soft edges for hair).
  * foreground_hd[i]  = SR of the MOVING subject, on one of two budgets:
        (a) "compact"      -- compact per-frame SR of the whole frame (~real-time);
        (b) "x4plus-bbox"  -- the heavy x4plus net run ONLY on the foreground
                              bounding box (~18-25% of the frame => cheap because
                              the region is small), composited over a cheap bicubic
                              base (the base is masked away by (1-alpha) anyway).
  * film grain        = per-frame, temporally-independent, added as the FINAL pass.

This module ONLY composes pieces built and validated upstream; it imports them
READ-ONLY and adds no new SR / matte math:
    derisk             (decode + scene split helpers live in the demo)
    matting            (L1: load_rvm, matte_sequence, fg_mask_lr)
    background_plate   (L2: build_plate, sr_plate, sample_plate, scene_segments)
    sr                 (compact + x4plus nets; latency accounting)
    grain              (final-pass film grain)

Import-safe: importing this module loads no weights and touches no GPU. Only the
functions do work (and only the SR / matte calls allocate nets).

Public API (consumed by the demo / a deployment)
------------------------------------------------
  matte_scene(model, frames)                         -> (phas, gates)
  build_background(frames, gates, sr_model)          -> (plate_lr, plate_hd, cov, hole, sr_ms)
  alpha_to_hd(pha, hw_hd)                             -> HxWx1 float32 soft alpha at HD
  foreground_compact(frame_lr)                       -> (fg_hd, ms)
  foreground_x4plus_bbox(frame_lr, pha, pad, thr)    -> (fg_hd, bbox, area_frac, ms)
  composite(fg_hd, alpha_hd, plate_hd)               -> (out_hd, ms)
  add_grain(out_hd, idx, strength, template)         -> (grained, ms)
  render_scene(...)                                  -> dict (frames + per-stage timings)
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import matting                 # READ-ONLY (L1)
import background_plate as bp  # READ-ONLY (L2)
import sr                      # READ-ONLY (compact + x4plus)
import grain                   # READ-ONLY (final pass)

COMPACT = "realesrgan"
HEAVY = "realesrgan-x4plus"
SCALE = 4


# --------------------------------------------------------------------------- #
# L1: matte the scene -> per-frame alpha + (dilated, binary) background gate
# --------------------------------------------------------------------------- #
def matte_scene(
    model,
    frames: Sequence[np.ndarray],
    dilate: int = 3,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Run RVM over the scene (recurrent, temporal order). Returns
      phas  : list of HxW float32 alpha mattes (LR), soft edges for hair.
      gates : list of HxW float32 binary FG gates (alpha>=0.5, dilated) for the
              L2 plate (FG is EXCLUDED from the background median)."""
    h, w = frames[0].shape[:2]
    res = matting.matte_sequence(model, frames)
    phas = [p for (_f, p) in res]
    gates = [matting.fg_mask_lr(p, lr_hw=(h, w), soft=False, thresh=0.5, dilate=dilate)
             for p in phas]
    return phas, gates


# --------------------------------------------------------------------------- #
# L2: build + heavy-SR the static background plate ONCE for the scene
# --------------------------------------------------------------------------- #
def build_background(
    frames: Sequence[np.ndarray],
    gates: Sequence[np.ndarray],
    sr_model: str = HEAVY,
    min_samples: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Accumulate the LR background plate (temporal median of background-only pixels)
    and heavy-SR it ONCE. Returns (plate_lr, plate_hd, coverage, hole_mask, sr_ms)."""
    plate_lr, coverage, hole_mask = bp.build_plate(frames, gates, min_samples=min_samples)
    plate_hd = bp.sr_plate(plate_lr, scale=SCALE, model=sr_model)
    sr_ms = sr.last_latency_ms(sr_model)
    return plate_lr, plate_hd, coverage, hole_mask, sr_ms


# --------------------------------------------------------------------------- #
# Alpha matte -> HD (soft, for hair edges)
# --------------------------------------------------------------------------- #
def alpha_to_hd(pha: np.ndarray, hw_hd: Tuple[int, int]) -> np.ndarray:
    """Upsample an LR alpha matte to HD as a soft [0,1] field. LINEAR keeps the
    hair-edge gradient (a hard nearest upsample would blockify the matte). Returns
    HxWx1 float32 so it broadcasts against HxWx3 colour."""
    h_hd, w_hd = hw_hd
    a = cv2.resize(np.asarray(pha, np.float32), (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
    return np.clip(a, 0.0, 1.0)[..., None]


# --------------------------------------------------------------------------- #
# Foreground budget (a): compact per-frame full-frame SR (~real-time)
# --------------------------------------------------------------------------- #
def foreground_compact(frame_lr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compact SR of the whole LR frame -> HD foreground. Returns (fg_hd, ms)."""
    fg = sr.upscale(frame_lr, model=COMPACT)
    return fg, sr.last_latency_ms(COMPACT)


# --------------------------------------------------------------------------- #
# Foreground budget (b): heavy x4plus on the FG bounding box ONLY
# --------------------------------------------------------------------------- #
def fg_bbox_lr(pha: np.ndarray, thr: float = 0.05, pad: int = 8) -> Optional[Tuple[int, int, int, int]]:
    """LR bounding box (x0,y0,x1,y1) of the foreground (alpha>thr), padded by `pad`
    px so the soft hair band around the subject is inside the box. thr is LOW (0.05),
    not 0.5, so wispy hair (low alpha) is covered by the heavy net. None if no FG."""
    m = np.asarray(pha, np.float32) >= thr
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    h, w = pha.shape[:2]
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + 1 + pad, w)
    y1 = min(int(ys.max()) + 1 + pad, h)
    return x0, y0, x1, y1


def foreground_x4plus_bbox(
    frame_lr: np.ndarray,
    pha: np.ndarray,
    pad: int = 8,
    thr: float = 0.05,
    base: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], float, float]:
    """Heavy x4plus SR on the FG bbox ONLY, pasted onto a cheap bicubic HD base.

    Only the foreground pixels are ever shown (the composite multiplies fg by alpha,
    and alpha~0 outside the subject), so the rest of the frame does NOT need the heavy
    net -- a bicubic upscale is fine there and costs ~nothing. The heavy net therefore
    runs on ~18-25% of the frame instead of 100%.

    Returns (fg_hd, bbox_lr, bbox_area_frac, x4plus_ms). x4plus_ms is the heavy-net
    latency on the crop ONLY (the headline per-frame FG cost for this budget)."""
    h, w = frame_lr.shape[:2]
    if base is None:
        base = cv2.resize(frame_lr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    bbox = fg_bbox_lr(pha, thr=thr, pad=pad)
    if bbox is None:
        return base, None, 0.0, 0.0
    x0, y0, x1, y1 = bbox
    crop = np.ascontiguousarray(frame_lr[y0:y1, x0:x1])
    crop_hd = sr.upscale(crop, model=HEAVY)            # x4 of the crop, exact integer scale
    ms = sr.last_latency_ms(HEAVY)
    fg = base.copy()
    fg[y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] = crop_hd
    area_frac = float((x1 - x0) * (y1 - y0)) / float(h * w)
    return fg, bbox, area_frac, ms


# --------------------------------------------------------------------------- #
# The composite + the grain final pass
# --------------------------------------------------------------------------- #
# --- R2-E3 seam-halo reduction helpers (OUTPUT-ONLY, default-preserving) --------------- #
# The matte seam halo is a SOFT-BACKGROUND problem: the near-subject plate ring is real
# background softened by low temporal coverage + matte-edge contamination (var-Lap ~10.8 vs
# ~15.4 deep BG). restore_plate_ring re-contrasts that existing texture (fabricates nothing);
# feather_alpha recovers hair wisps with the subject core untouched. See
# experiments/r2_e3_seam/REPORT.md.
def _alpha_grad_softness(a):
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy)
    band = (a > 0.02) & (a < 0.98)
    p90 = max(np.percentile(g[band], 90), 1e-3) if band.any() else 1e-3
    soft = np.clip(1.0 - np.clip(g / p90, 0, 1), 0, 1) * band
    return cv2.GaussianBlur(soft, (0, 0), 2.0)


def feather_alpha(a_hd, soft_sigma=4.0, lift=0.35):
    """Alpha-aware feather: wider feather + faint-wisp lift where the alpha ramp is gentle (soft
    hair), tight at hard jaw/shoulder edges. Recovers wisps; subject core untouched."""
    a = a_hd[..., 0].astype(np.float32)
    soft = _alpha_grad_softness(a)
    a_wide = cv2.GaussianBlur(a, (0, 0), soft_sigma)
    a_f = soft * a_wide + (1.0 - soft) * a
    if lift > 0:
        lifted = np.power(np.clip(a_f, 0, 1), 1.0 / (1.0 + lift))
        a_f = soft * lifted + (1.0 - soft) * a_f
    return np.clip(a_f, 0, 1)[..., None]


def _bgside_band_weight(a, center=0.22, width=0.18):
    w = np.exp(-((a - center) / width) ** 2).astype(np.float32)
    w[a < 0.015] = 0.0
    w[a > 0.55] = 0.0
    return cv2.GaussianBlur(w, (0, 0), 2.0)


def restore_plate_ring(plate_hd, a_hd, strength=0.5, amount=0.6, sigma=2.0):
    """Blend a band-localized plate unsharp (BG side of the matte) toward the plate at `strength`.
    strength~0.5 lands the near-subject ring at the deep-BG level; strength=0 -> identity. Targets
    the background, not the subject (budget-independent). Cheapest run ONCE per scene on the static
    plate (see prepare_scene_plate)."""
    if strength <= 0:
        return plate_hd
    a = a_hd[..., 0].astype(np.float32)
    w = _bgside_band_weight(a)[..., None]
    p = plate_hd.astype(np.float32)
    hf = p - cv2.GaussianBlur(p, (0, 0), sigma)
    sh = np.clip(p + amount * w * hf, 0, 255).astype(np.uint8)
    return cv2.addWeighted(plate_hd, 1.0 - strength, sh, strength, 0.0)


def prepare_scene_plate(plate_hd, alphas_hd, strength=0.8):
    """Restore the soft near-subject plate ring ONCE per scene (the plate is static -> amortized to
    ~0/frame like the plate SR). `alphas_hd` = a list of per-frame HxWx1 HD alphas OR a single
    union alpha; the union footprint covers every frame's ring. Call after build_plate/sr_plate and
    pass the result as plate_hd to composite() with seam_restore=0."""
    union = None
    for a in (alphas_hd if isinstance(alphas_hd, (list, tuple)) else [alphas_hd]):
        union = a.astype(np.float32) if union is None else np.maximum(union, a)
    return restore_plate_ring(plate_hd, union, strength=strength)


def composite(fg_hd: np.ndarray, alpha_hd: np.ndarray, plate_hd: np.ndarray,
              seam_restore: float = 0.0, feather: bool = False) -> Tuple[np.ndarray, float]:
    """out = alpha*fg + (1-alpha)*plate  (HD). alpha_hd is HxWx1 float in [0,1].

    OUTPUT-ONLY seam-halo reduction (R2-E3), DEFAULT-PRESERVING:
      seam_restore=0.0, feather=False -> byte-identical to the original composite.
      feather=True   -> alpha-aware feather (recover hair wisps, core untouched).
      seam_restore>0 -> restore the soft near-subject plate ring (kill the halo moat); 0.5 =
                        deep-BG-matched. Prefer prepare_scene_plate ONCE/scene + feather per frame."""
    t0 = time.perf_counter()
    if feather:
        alpha_hd = feather_alpha(alpha_hd)
    if seam_restore > 0:
        plate_hd = restore_plate_ring(plate_hd, alpha_hd, strength=seam_restore)
    out = alpha_hd * fg_hd.astype(np.float32) + (1.0 - alpha_hd) * plate_hd.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, (time.perf_counter() - t0) * 1000.0


def add_grain(out_hd: np.ndarray, idx: int, strength: str = "med", template=None) -> Tuple[np.ndarray, float]:
    """Per-frame film grain (temporally independent) as the FINAL pass."""
    t0 = time.perf_counter()
    g = grain.apply_grain(out_hd, idx, strength=strength, template=template)
    return g, (time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------------- #
# Orchestrator: render a whole scene one budget at a time, timing every stage
# --------------------------------------------------------------------------- #
def render_scene(
    frames: Sequence[np.ndarray],
    phas: Sequence[np.ndarray],
    plate_hd: np.ndarray,
    fg_budget: str = "compact",
    grain_strength: Optional[str] = None,
    matte_refresh: int = 1,
    plate_sample_motion=None,
    pad: int = 8,
    thr: float = 0.05,
) -> Dict:
    """Render the scene with the chosen foreground budget.

    fg_budget       : "compact" (per-frame compact SR) or "x4plus_bbox".
    grain_strength  : None/"off" -> no grain; else {"low","med","high"}.
    matte_refresh   : reuse the alpha matte for this many frames (amortize L1). 1 =
                      per-frame (the measured default). The plate/gates are still built
                      from per-frame mattes upstream; this only governs the ALPHA used
                      in the composite, to show matte cost can be amortized.
    plate_sample_motion : passed to sample_plate (None = static identity).

    Returns dict(frames, frames_grained, timings{...lists...}, bbox_area_fracs).
    """
    h, w = frames[0].shape[:2]
    h_hd, w_hd = h * SCALE, w * SCALE
    tmpl = grain.make_template(h_hd, w_hd) if grain_strength and grain_strength != "off" else None

    out_frames, out_grained = [], []
    t_sample, t_alpha, t_fg, t_comp, t_grain = [], [], [], [], []
    bbox_fracs = []

    base_bicubic = None
    alpha_hd_cur = None
    for i, (frame, pha) in enumerate(zip(frames, phas)):
        # plate sample (static => identity, ~free)
        t0 = time.perf_counter()
        plate_i = bp.sample_plate(plate_hd, frame_idx=i, global_motion=plate_sample_motion)
        t_sample.append((time.perf_counter() - t0) * 1000.0)

        # alpha to HD (amortizable across matte_refresh frames)
        t0 = time.perf_counter()
        if alpha_hd_cur is None or (i % matte_refresh == 0):
            alpha_hd_cur = alpha_to_hd(pha, (h_hd, w_hd))
        t_alpha.append((time.perf_counter() - t0) * 1000.0)

        # foreground SR (the budget)
        if fg_budget == "compact":
            fg_hd, fg_ms = foreground_compact(frame)
            bbox_fracs.append(1.0)
        elif fg_budget == "x4plus_bbox":
            base_bicubic = cv2.resize(frame, (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
            fg_hd, bbox, frac, fg_ms = foreground_x4plus_bbox(
                frame, pha, pad=pad, thr=thr, base=base_bicubic)
            bbox_fracs.append(frac)
        else:
            raise ValueError(f"unknown fg_budget {fg_budget!r}")
        t_fg.append(fg_ms)

        # composite
        out, c_ms = composite(fg_hd, alpha_hd_cur, plate_i)
        t_comp.append(c_ms)
        out_frames.append(out)

        # grain final pass
        if grain_strength and grain_strength != "off":
            g, g_ms = add_grain(out, i, strength=grain_strength, template=tmpl)
            t_grain.append(g_ms)
            out_grained.append(g)
        else:
            t_grain.append(0.0)

    return dict(
        frames=out_frames,
        frames_grained=out_grained,
        bbox_area_fracs=bbox_fracs,
        timings=dict(
            plate_sample_ms=t_sample,
            alpha_ms=t_alpha,
            fg_sr_ms=t_fg,
            composite_ms=t_comp,
            grain_ms=t_grain,
        ),
    )


if __name__ == "__main__":
    # tiny self-contained smoke test (NO real footage / NO heavy nets): a moving white
    # square over a fixed gradient. We synthesize an alpha matte + a "plate_hd" directly
    # and check the composite identity and shapes, so importing/plumbing is verified
    # without downloading RVM or the SR weights.
    H, W, N = 40, 64, 6
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.stack([(xx / W * 255), (yy / H * 255), ((xx + yy) / (W + H) * 128)], -1).astype(np.uint8)
    frames, phas = [], []
    for t in range(N):
        f = bg.copy()
        x0 = int(t / N * (W - 16))
        f[10:30, x0:x0 + 16] = (255, 0, 0)
        a = np.zeros((H, W), np.float32)
        a[10:30, x0:x0 + 16] = 1.0
        frames.append(f); phas.append(a)
    plate_hd = cv2.resize(bg, (W * SCALE, H * SCALE), interpolation=cv2.INTER_CUBIC)
    # composite the synthetic FG (bicubic) over the plate using our alpha path
    fg_hd = cv2.resize(frames[2], (W * SCALE, H * SCALE), interpolation=cv2.INTER_CUBIC)
    a_hd = alpha_to_hd(phas[2], (H * SCALE, W * SCALE))
    out, ms = composite(fg_hd, a_hd, plate_hd)
    print("layered_pipeline.py smoke test")
    print(f"  composite {out.shape} {out.dtype} in {ms:.2f} ms")
    g, gms = add_grain(out, 0, "med")
    print(f"  grain pass {g.shape} in {gms:.2f} ms; differs from input: {not np.array_equal(g, out)}")
    bbox = fg_bbox_lr(phas[2], thr=0.05, pad=4)
    print(f"  fg bbox (LR) = {bbox}  (subject at x~{int(2/N*(W-16))})")
    print("OK (plumbing verified; real run is demo_layered.py)")
