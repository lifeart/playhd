#!/usr/bin/env python3
"""
Per-frame film-grain pass for playhd (Step 8).

Recipe (AV1 / AFGS1, the unanimous deep-research recipe):
  * Temporally INDEPENDENT: grain is regenerated for EVERY output frame from a per-frame
    seed (= frame index). Deterministic but DIFFERENT every frame, so it does NOT freeze
    onto moving content -- the No.1 film-grain failure mode. NEVER warp/propagate grain, and
    NEVER feed grained frames back into the anchor / reference / propagated chain.
  * Spatially CORRELATED, content-INDEPENDENT template: a white-Gaussian noise field, blurred
    by a small Gaussian kernel, then re-normalized to unit std. The blur gives the grain a
    finite "grain size" (correlation length) so it reads as filmic, not per-pixel salt-and-
    pepper / blocky. (AFGS1's full spec is a 2D auto-regressive model fit per content; a
    Gaussian-blurred template is the agreed v1 stand-in and is content-independent by design.)
  * Amplitude modulated by LOCAL LUMA via a LUT: film grain is most visible in mid-tones and
    rolls off in deep shadow and bright highlight. The LUT scales the grain template's
    amplitude per pixel by the (slightly blurred = "local") luma. This modulates VISIBILITY,
    not position.
  * Added as the FINAL pass, AFTER upscaling, in LUMA, in gamma space (the uint8 sRGB frame is
    already gamma-encoded, so we operate directly on its Y channel). Chroma is left untouched.

The grain pass is the LAST thing before display; it is applied to a COPY and never to any
frame that is warped or used as a reference, so it cannot be frozen onto content by the codec
motion vectors.

Public API:
    apply_grain(rgb_uint8, frame_idx, strength="med", template=None) -> rgb_uint8
    STRENGTHS = {"off","low","med","high"}
A reusable grain template (same H,W) can be passed to avoid rebuilding it every call; it is
re-seeded/rolled per frame_idx internally, so passing one is purely a speed optimization and
does NOT make the grain temporally correlated.
"""
import cv2
import numpy as np

# Base grain sigma in 8-bit luma code values, per strength. These are the std of the grain
# at the LUT's peak (mid-tone) before luma modulation.
STRENGTHS = {
    "off": 0.0,
    "low": 2.5,
    "med": 5.0,
    "high": 9.0,
}
# Gaussian blur sigma (px) applied to the white-noise template => grain correlation length.
# ~0.8 px gives a fine, 35mm-like grain at HD; larger = coarser/clumpier grain.
_GRAIN_BLUR_SIGMA = 0.8
# Blur sigma (px) for the LOCAL-luma modulation map: averages luma over a small neighbourhood
# so grain amplitude tracks region brightness, not per-pixel edges (avoids edge shimmer).
_LUMA_BLUR_SIGMA = 2.0


def _luma_lut():
    """Amplitude-scale as a function of 8-bit luma (0..255). Filmic visibility curve: ~0.45 in
    deep shadow / bright highlight, peaking at 1.0 around mid-tone. A smooth sine hump keeps it
    monotone-in/monotone-out with no hard knees. Content-independent."""
    y = np.arange(256, dtype=np.float32) / 255.0
    hump = np.sin(np.pi * y)              # 0 at black/white, 1 at mid
    return (0.45 + 0.55 * hump).astype(np.float32)


_LUT = _luma_lut()


def make_template(h, w, blur_sigma=_GRAIN_BLUR_SIGMA, seed=0):
    """Spatially-correlated unit-variance Gaussian grain template (content-independent)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w)).astype(np.float32)
    g = cv2.GaussianBlur(noise, (0, 0), blur_sigma)
    s = float(g.std())
    if s > 1e-6:
        g /= s                            # re-normalize to unit std after blur
    return g


def _frame_grain(h, w, frame_idx, template=None):
    """A unit-variance grain field UNIQUE to this frame index. If a base template is supplied we
    re-roll it per frame (a per-frame pixel shift + sign from the index) so the field differs
    every frame WITHOUT rebuilding+blurring noise each call. Otherwise build fresh from the seed.
    Either way the result is deterministic in frame_idx and temporally independent frame-to-frame."""
    if template is None:
        return make_template(h, w, seed=frame_idx * 2654435761 % (2**32))
    th, tw = template.shape
    # deterministic per-frame roll (decorrelates consecutive frames) + occasional sign flip
    rng = np.random.default_rng(frame_idx * 2654435761 % (2**32))
    sy, sx = int(rng.integers(0, th)), int(rng.integers(0, tw))
    g = np.roll(template, (sy, sx), axis=(0, 1))
    if h != th or w != tw:
        g = g[:h, :w] if (h <= th and w <= tw) else cv2.resize(g, (w, h))
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return (g * sign).astype(np.float32)


def apply_grain(rgb_uint8, frame_idx, strength="med", template=None, return_grain=False):
    """Add per-frame film grain to an HxWx3 uint8 RGB frame's LUMA. Returns a new uint8 frame
    (or, if return_grain=True, also the raw additive luma grain FIELD as float HxW -- the exact
    noise added to Y before clipping/round-trip, for an artifact-free temporal-independence check).
    `frame_idx` seeds the grain => deterministic but DIFFERENT every frame (temporally
    independent). `strength` in {off,low,med,high}. Grain is luma-only, gamma-space, final pass."""
    sigma = STRENGTHS.get(strength, 0.0) if isinstance(strength, str) else float(strength)
    h, w = rgb_uint8.shape[:2]
    if sigma <= 0.0:
        return (rgb_uint8.copy(), np.zeros((h, w), np.float32)) if return_grain else rgb_uint8.copy()
    ycc = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycc[:, :, 0]
    # local-luma amplitude modulation (LUT of a slightly blurred luma => "local" brightness)
    y_local = cv2.GaussianBlur(y, (0, 0), _LUMA_BLUR_SIGMA)
    amp = _LUT[np.clip(y_local, 0, 255).astype(np.uint8)]          # HxW per-pixel scale
    grain = _frame_grain(h, w, frame_idx, template) * sigma * amp  # unit grain -> code values
    ycc[:, :, 0] = np.clip(y + grain, 0, 255)
    out = cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return (out, grain) if return_grain else out


def apply_grain_motion(rgb_uint8, frame_idx, static_w_hd, strength="med", template=None,
                       frozen_idx=0, return_grain=False):
    """MOTION-MODULATED grain (experiment E3 V2): identical to apply_grain EXCEPT the per-pixel
    unit grain field is gated by the region-aware motion weight `static_w_hd` (HxW, 1=static,
    0=moving, already upsampled to the HD frame size; from derisk._build_region_gate's `a_lr`):
      * STATIC pixels (a=1) use a FROZEN grain field (fixed `frozen_idx` seed) -> spatially full
        filmic texture but IDENTICAL every frame -> ~0 temporal flicker, so the propagation chain's
        stability survives instead of grain re-injecting ~4.5/frame of static flicker.
      * MOVING pixels (a=0) use FRESH per-frame grain (seed=frame_idx) -> independent, filmic.
      * Between: a*frozen + (1-a)*fresh, RENORMALISED to unit variance so grain density is uniform
        across the seam (no visible amplitude step).
    Degenerate static_w_hd==0 everywhere -> exactly apply_grain. Output-only: never warp/propagate
    this or feed it into the reference chain R[]. Measure independence on the RAW grain FIELD."""
    sigma = STRENGTHS.get(strength, 0.0) if isinstance(strength, str) else float(strength)
    h, w = rgb_uint8.shape[:2]
    if sigma <= 0.0:
        return (rgb_uint8.copy(), np.zeros((h, w), np.float32)) if return_grain else rgb_uint8.copy()
    ycc = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycc[:, :, 0]
    y_local = cv2.GaussianBlur(y, (0, 0), _LUMA_BLUR_SIGMA)
    amp = _LUT[np.clip(y_local, 0, 255).astype(np.uint8)]
    a = np.clip(static_w_hd, 0.0, 1.0).astype(np.float32)
    fresh = _frame_grain(h, w, frame_idx, template)
    frozen = _frame_grain(h, w, frozen_idx, template)
    unit = a * frozen + (1.0 - a) * fresh
    unit = unit / np.maximum(np.sqrt(a * a + (1.0 - a) ** 2), 1e-6)   # unit variance across the seam
    grain = unit * sigma * amp
    ycc[:, :, 0] = np.clip(y + grain, 0, 255)
    out = cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    return (out, grain) if return_grain else out


if __name__ == "__main__":
    # self-test: (1) grain visible & luma-modulated on a ramp; (2) temporally independent.
    H, W = 256, 512
    ramp = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
    ramp_rgb = np.stack([ramp] * 3, axis=-1)
    g0 = apply_grain(ramp_rgb, 0, "high")
    g1 = apply_grain(ramp_rgb, 1, "high")
    # residual std per luma bucket (should peak mid-tone, dip at black/white = LUT working)
    res0 = g0[:, :, 0].astype(np.float32) - ramp.astype(np.float32)
    dark = float(res0[:, :W // 8].std())
    mid = float(res0[:, W * 7 // 16:W * 9 // 16].std())
    bright = float(res0[:, -W // 8:].std())
    print(f"grain std  dark={dark:.2f}  mid={mid:.2f}  bright={bright:.2f}  "
          f"(LUT ok if mid > dark and mid > bright)")
    # temporal independence: two consecutive frames must differ (grain re-rolled per index)
    diff = float(np.abs(g0.astype(np.int16) - g1.astype(np.int16)).mean())
    same = float(np.abs(apply_grain(ramp_rgb, 5, "high").astype(np.int16)
                        - apply_grain(ramp_rgb, 5, "high").astype(np.int16)).mean())
    print(f"frame0 vs frame1 mean|delta| = {diff:.2f} (must be >0 => independent); "
          f"frame5 vs frame5 = {same:.2f} (must be 0 => deterministic)")
