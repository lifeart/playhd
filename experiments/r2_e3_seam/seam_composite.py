"""seam_composite.py -- R2-E3 DELIVERABLE: the concrete, OUTPUT-ONLY, default-preserving
change to prototype/layered_pipeline.py composite().

Drop the helpers + the two new kwargs into layered_pipeline.py. With the defaults
(seam_restore=0.0, feather=False) the output is BYTE-IDENTICAL to the current composite()
(verified below) -- so it is safe to land dark. Turn it on per-render to reduce the matte
seam halo + recover hair wisps:

    out, ms = composite(fg_hd, alpha_hd, plate_hd, seam_restore=0.5, feather=True)

What it does (measured in run_sweep.py / ring_fill_compare.py on the talking-head scene):
  * feather=True            -> alpha-aware feather: wider feather + faint-wisp lift where the
                               alpha ramp is GENTLE (soft hair), tight at hard jaw/shoulder
                               edges. Recovers wisps (hair fg-bias -3.20 -> -2.72) with the
                               subject CORE untouched (core |Lap| 4.08 -> 4.08).
  * seam_restore in [0,1]   -> band-localized plate-ring restore on the BG side of the matte.
                               The near-subject plate ring is REAL background softened by low
                               temporal coverage + matte-edge contamination (var-Lap ~10.8 vs
                               ~15.4 deep BG); 0.5 restores it to the deep-BG level, killing
                               the soft 'moat' that reads as a halo. x4plus-bbox seam ratio
                               5.02 -> 3.22 (== uniform-x4plus ceiling 3.45); halo moat width
                               11.7 -> 7.7 px. It RE-CONTRASTS existing softened texture (no
                               fabricated structure; the ~8% always-occluded inpaint holes
                               have ~0 HF so they stay soft, not invented).

NOT done: softening the FG edge (cuts the seam ratio only by SMEARING the subject -- core
|Lap| 4.08 -> 4.04 -- rejected); swapping the Telea inpaint (lever c) -- the always-occluded
holes are only ~8% of the visible ring and always sit behind the subject (gotcha #19), and NS/
wider-Telea fills are SOFTER not sharper, so a better inpaint does not move the seam.
"""
from __future__ import annotations
import time
import cv2
import numpy as np


# ---- helpers (add these to layered_pipeline.py) -------------------------------------- #
def _alpha_grad_softness(a):
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy)
    band = (a > 0.02) & (a < 0.98)
    p90 = max(np.percentile(g[band], 90), 1e-3) if band.any() else 1e-3
    soft = np.clip(1.0 - np.clip(g / p90, 0, 1), 0, 1) * band
    return cv2.GaussianBlur(soft, (0, 0), 2.0)


def feather_alpha(a_hd, soft_sigma=4.0, lift=0.35):
    """Alpha-aware feather: wider feather + faint-wisp lift where the alpha ramp is gentle."""
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
    """Blend a band-localized plate unsharp (BG side of the matte) toward the plate at
    `strength`. strength~0.5 lands the near-subject ring at the deep-BG level. strength=0
    -> identity. Budget-independent (targets the background, not the subject)."""
    if strength <= 0:
        return plate_hd
    a = a_hd[..., 0].astype(np.float32)
    w = _bgside_band_weight(a)[..., None]
    p = plate_hd.astype(np.float32)
    hf = p - cv2.GaussianBlur(p, (0, 0), sigma)
    sh = np.clip(p + amount * w * hf, 0, 255).astype(np.uint8)
    return cv2.addWeighted(plate_hd, 1.0 - strength, sh, strength, 0.0)


# ---- cost-optimized integration: restore the plate ring ONCE per scene --------------- #
def prepare_scene_plate(plate_hd, alphas_hd, strength=0.8):
    """Restore the soft near-subject plate ring ONCE per scene (the plate is static, so this
    is amortized to ~0/frame like the plate SR). `alphas_hd` = the per-frame HxWx1 HD alphas
    (or a single union alpha); the union footprint covers every frame's ring. strength~0.8
    matches the per-frame restore (the wider union band needs a touch more). Call this after
    build_background; pass the result as plate_hd to composite() with seam_restore=0."""
    union = None
    for a in (alphas_hd if isinstance(alphas_hd, (list, tuple)) else [alphas_hd]):
        union = a.astype(np.float32) if union is None else np.maximum(union, a)
    return restore_plate_ring(plate_hd, union, strength=strength)


# ---- the proposed composite() (replaces layered_pipeline.composite) ------------------ #
def composite(fg_hd, alpha_hd, plate_hd, seam_restore: float = 0.0, feather: bool = False):
    """out = alpha*fg + (1-alpha)*plate (HD). alpha_hd is HxWx1 float in [0,1].

    OUTPUT-ONLY seam-halo reduction (R2-E3), DEFAULT-PRESERVING:
      seam_restore=0.0, feather=False -> byte-identical to the original composite.
      feather=True       -> alpha-aware feather (recover hair wisps, core untouched).
      seam_restore>0      -> restore the soft near-subject plate ring (kill the halo moat);
                             0.5 = deep-BG-matched (recommended)."""
    t0 = time.perf_counter()
    if feather:
        alpha_hd = feather_alpha(alpha_hd)
    if seam_restore > 0:
        plate_hd = restore_plate_ring(plate_hd, alpha_hd, strength=seam_restore)
    out = alpha_hd * fg_hd.astype(np.float32) + (1.0 - alpha_hd) * plate_hd.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, (time.perf_counter() - t0) * 1000.0


if __name__ == "__main__":
    # default-preserving (byte-identical) check + added-cost measurement on an HD frame.
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import seam_lib as L
    D = L.load_inputs(); H, W = D["plate_hd"].shape[:2]; i = 16
    a = L.alpha_to_hd(D["phas"][i], (H, W)); fg = D["x4"][i]; pl = D["plate_hd"]
    out_def, _ = composite(fg, a, pl)                                 # defaults OFF
    out_orig = L.composite(fg, a, pl)                                 # the current pipeline math
    print("default byte-identical to current composite():", np.array_equal(out_def, out_orig))
    # timing (best of 5) for default vs recommended on the 2560x1280 HD frame
    def best(fn, n=5):
        ts = []
        for _ in range(n):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000.0)
        return min(ts)
    t_def = best(lambda: composite(fg, a, pl))
    t_rec = best(lambda: composite(fg, a, pl, seam_restore=0.5, feather=True))
    print(f"composite cost @2560x1280: default {t_def:.1f} ms  |  per-frame seam {t_rec:.1f} ms  "
          f"(+{t_rec - t_def:.1f} ms)")
    # COST-OPTIMIZED path: restore the plate ONCE/scene + feather the LR alpha (cheap)
    alphas = [L.alpha_to_hd(D["phas"][k], (H, W)) for k in range(D["x4"].shape[0])]
    t_once = best(lambda: prepare_scene_plate(pl, alphas, strength=0.8), n=3)
    aLR = D["phas"][i][..., None]
    def amort_frame():
        af = feather_alpha(aLR)
        afh = cv2.resize(af[..., 0], (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        return composite(fg, afh, pl)          # pl already pre-restored in practice
    t_amort = best(amort_frame)
    print(f"COST-OPT: prepare_scene_plate {t_once:.0f} ms ONCE/scene (~{t_once/300:.2f} ms/frame @300f) "
          f"+ LR-feather composite {t_amort:.1f} ms/frame  => ~+{t_amort - t_def + t_once/300:.1f} ms/frame")
