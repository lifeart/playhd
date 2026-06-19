"""seam_lib.py -- R2-E3 OUTPUT-ONLY composite tweaks + seam metrics (GPU-free).

All tweaks are pure functions of (fg_hd, alpha_hd, plate_hd) [+ alpha-gradient], so they
drop into layered_pipeline.composite without touching the upstream SR/matte/plate build.
Metrics reuse the demo's var-Laplacian RING definition for comparability; var-Laplacian is
used here ONLY as a RELATIVE seam-continuity measure, NOT an absolute SR-quality claim.
"""
import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.abspath(os.path.join(HERE, "..", "..", "prototype"))
CACHE = os.path.join(HERE, "cache")
OUT_LAYERED = os.path.join(PROTO, "out_layered")
SCALE = 4


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_inputs():
    frames = np.load(os.path.join(CACHE, "frames_lr.npy"))          # (N,h,w,3) uint8 RGB
    phas = np.load(os.path.join(CACHE, "phas.npy"))                 # (N,h,w) float32
    gates = np.load(os.path.join(CACHE, "gates.npy"))              # (N,h,w) float32
    plate_raw = np.load(os.path.join(CACHE, "plate_raw.npy"))      # (h,w,3) float32 + NaN
    fill_mask = np.load(os.path.join(CACHE, "fill_mask.npy"))      # (h,w) bool
    x4 = np.load(os.path.join(OUT_LAYERED, "cache", "sr_5000_0_32_realesrgan-x4plus.npy"))
    cp = np.load(os.path.join(OUT_LAYERED, "cache", "sr_5000_0_32_realesrgan.npy"))
    plate_bgr = cv2.imread(os.path.join(OUT_LAYERED, "plate_hd.png"))
    plate_hd = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)          # (H,W,3) uint8 RGB
    return dict(frames=frames, phas=phas, gates=gates, plate_raw=plate_raw,
                fill_mask=fill_mask, x4=x4, cp=cp, plate_hd=plate_hd)


def alpha_to_hd(pha, hw_hd):
    h_hd, w_hd = hw_hd
    a = cv2.resize(np.asarray(pha, np.float32), (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
    return np.clip(a, 0.0, 1.0)[..., None]


def composite(fg_hd, alpha_hd, plate_hd):
    out = alpha_hd * fg_hd.astype(np.float32) + (1.0 - alpha_hd) * plate_hd.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# var-Laplacian helpers (RELATIVE seam-continuity metric only)
# --------------------------------------------------------------------------- #
def vlap(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F)


def ring_var(lap, ring):
    return float(lap[ring].var()) if ring.any() else float("nan")


# --------------------------------------------------------------------------- #
# METRIC 1: seam discontinuity ratio (demo definition: FGring / BGring var-Lap)
# --------------------------------------------------------------------------- #
def seam_ratio(out_hd, a_hd):
    a = a_hd[..., 0]
    bg_ring = (a > 0.02) & (a < 0.2)
    fg_ring = (a > 0.8) & (a < 0.98)
    lap = vlap(out_hd)
    sb = ring_var(lap, bg_ring)
    sf = ring_var(lap, fg_ring)
    return sf, sb, (sf / sb if sb > 1e-6 else float("nan"))


# --------------------------------------------------------------------------- #
# METRIC 2: halo width -- depth (px, HD) the excess edge-contrast leaks into the BG.
# local sharpness S = boxfilter(|Laplacian|); signed distance d from the alpha=0.5
# contour (d>0 into BG). halo width = contiguous px from the boundary into the BG over
# which S(d) stays above K x (far-BG baseline). Averaged via a distance-binned profile.
# --------------------------------------------------------------------------- #
def _local_sharp(rgb, ksize=9):
    lap = np.abs(vlap(rgb)).astype(np.float32)
    return cv2.boxFilter(lap, -1, (ksize, ksize), normalize=True)


def signed_dist(a_hd):
    fg = (a_hd[..., 0] >= 0.5).astype(np.uint8)
    d_out = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 5)   # >0 in BG
    d_in = cv2.distanceTransform(fg, cv2.DIST_L2, 5)        # >0 in FG
    return d_out - d_in                                     # >0 BG, <0 FG


def halo_profile(out_hd, a_hd, dmax=40):
    """Return (bins, S(d)) for integer signed distance d in [-dmax, dmax]."""
    S = _local_sharp(out_hd)
    d = signed_dist(a_hd)
    di = np.round(d).astype(np.int32)
    bins = np.arange(-dmax, dmax + 1)
    prof = np.full(bins.shape, np.nan, np.float32)
    for k, b in enumerate(bins):
        m = di == b
        if m.any():
            prof[k] = float(S[m].mean())
    return bins, prof


def halo_width(out_hd, a_hd, dmax=40, K=1.3):
    bins, prof = halo_profile(out_hd, a_hd, dmax=dmax)
    # far-BG baseline = mean S over the outer BG band [dmax-10, dmax]
    far = np.nanmean(prof[(bins >= dmax - 10) & (bins <= dmax)])
    thr = K * far
    # walk from the boundary (d=1) outward into BG; count contiguous bins above thr
    width = 0
    for b in range(1, dmax + 1):
        k = np.where(bins == b)[0][0]
        v = prof[k]
        if np.isnan(v):
            continue
        if v >= thr:
            width = b
        else:
            break
    peak = float(np.nanmax(prof))
    return float(width), peak, far, (bins, prof)


# --------------------------------------------------------------------------- #
# METRIC 3: hair-region detail -- does the soft-alpha band look more like the FG
# subject (wisps recovered) or like the smooth plate (wisps lost)? Plus a subject-core
# guard (a>0.95) that must NOT lose detail (no smearing the face).
# --------------------------------------------------------------------------- #
def hair_detail(out_hd, fg_hd, plate_hd, a_hd):
    a = a_hd[..., 0]
    hair = (a > 0.05) & (a < 0.6)
    core = a > 0.95
    og = cv2.cvtColor(out_hd, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fgg = cv2.cvtColor(fg_hd, cv2.COLOR_RGB2GRAY).astype(np.float32)
    plg = cv2.cvtColor(plate_hd, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # closer-to-FG-than-plate in the hair band => wisps shown rather than swallowed by plate
    olap = np.abs(cv2.Laplacian(og, cv2.CV_32F))
    mae_fg = float(np.abs(og[hair] - fgg[hair]).mean()) if hair.any() else float("nan")
    mae_pl = float(np.abs(og[hair] - plg[hair]).mean()) if hair.any() else float("nan")
    hair_struct = float(olap[hair].mean()) if hair.any() else float("nan")
    core_struct = float(olap[core].mean()) if core.any() else float("nan")
    return dict(mae_fg=mae_fg, mae_pl=mae_pl, fg_pl_bias=mae_pl - mae_fg,
                hair_struct=hair_struct, core_struct=core_struct)


# =========================================================================== #
# OUTPUT-ONLY TWEAKS  (pure fns of fg_hd / alpha_hd / plate_hd -- composite-time)
# =========================================================================== #
def _alpha_grad_softness(a, p90_floor=1e-3):
    """softness map in [0,1]: 1 where the alpha ramp is GENTLE (soft/wispy hair),
    0 where it is STEEP (hard jaw/shoulder edge). Confined to the transition band."""
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    g = np.hypot(gx, gy)
    band = (a > 0.02) & (a < 0.98)
    if band.any():
        p90 = max(np.percentile(g[band], 90), p90_floor)
    else:
        p90 = p90_floor
    gn = np.clip(g / p90, 0, 1)
    soft = np.clip(1.0 - gn, 0, 1) * band
    return cv2.GaussianBlur(soft, (0, 0), 2.0)


# --- (a) alpha-aware feather: wider feather + faint-wisp lift where alpha is soft --- #
def feather_alpha(a_hd, soft_sigma=4.0, lift=0.35):
    a = a_hd[..., 0].astype(np.float32)
    soft = _alpha_grad_softness(a)
    a_wide = cv2.GaussianBlur(a, (0, 0), soft_sigma)
    a_f = soft * a_wide + (1.0 - soft) * a
    if lift > 0:
        # gamma<1 lifts faint hair (low alpha) so wisps show through; only in soft band
        lifted = np.power(np.clip(a_f, 0, 1), 1.0 / (1.0 + lift))
        a_f = soft * lifted + (1.0 - soft) * a_f
    return np.clip(a_f, 0, 1)[..., None]


def _bgside_band_weight(a, center=0.22, width=0.18):
    """soft bump peaking in the BG-side transition ring (a~center), 0 in deep BG / subject."""
    w = np.exp(-((a - center) / width) ** 2).astype(np.float32)
    w[a < 0.015] = 0.0          # keep deep BG untouched
    w[a > 0.55] = 0.0           # subject side dominated by fg anyway
    return cv2.GaussianBlur(w, (0, 0), 2.0)


# --- (b)/(c) sharpness-matched seam band: restore the SOFT near-subject plate ring --- #
def sharpen_plate_band(plate_hd, a_hd, amount=1.4, sigma=2.0):
    """Band-localized unsharp of the plate on the BG side of the matte. The near-subject
    ring is REAL background softened by low temporal coverage + matte-edge contamination
    (var-Lap ~9 vs ~15 deep BG); restoring its contrast bridges the discontinuity. This
    re-contrasts EXISTING (softened) texture -- it does not fabricate structure (the ~8%
    pure-inpaint holes have ~0 HF, so they stay soft rather than being invented)."""
    a = a_hd[..., 0].astype(np.float32)
    w = _bgside_band_weight(a)[..., None]
    p = plate_hd.astype(np.float32)
    hf = p - cv2.GaussianBlur(p, (0, 0), sigma)
    out = p + amount * w * hf
    return np.clip(out, 0, 255).astype(np.uint8)


# --- (b-alt) soften the FG matte-edge band (lower the FG side toward the plate) --- #
def soften_fg_band(fg_hd, a_hd, sigma=1.6, lo=0.5, hi=0.97):
    a = a_hd[..., 0].astype(np.float32)
    band = ((a > lo) & (a < hi)).astype(np.float32)
    w = cv2.GaussianBlur(band, (0, 0), 2.0)[..., None]
    f = fg_hd.astype(np.float32)
    fb = cv2.GaussianBlur(f, (0, 0), sigma)
    out = w * fb + (1.0 - w) * f
    return np.clip(out, 0, 255).astype(np.uint8)


def halo_deficit_width(out_hd, a_hd, ref_out, dmax=30):
    """Halo width = px-extent (HD) of the near-edge BG band where the layered composite is
    SOFTER than the per-frame uniform-x4plus CEILING (ref_out, no FG/BG discontinuity by
    construction). Contiguous from the boundary (d=1) outward while S_layered(d)<S_ref(d).
    baseline -> wide soft moat; a perfectly matched seam -> 0. Correctly polarized: lower
    = less halo. Averaged over edge pixels via the distance-binned profile."""
    _, pl = halo_profile(out_hd, a_hd, dmax=dmax)
    _, pr = halo_profile(ref_out, a_hd, dmax=dmax)
    bins = np.arange(-dmax, dmax + 1)
    width = 0
    for b in range(1, dmax + 1):
        k = np.where(bins == b)[0][0]
        if np.isnan(pl[k]) or np.isnan(pr[k]):
            continue
        if pl[k] < pr[k]:
            width = b
        else:
            break
    return float(width)


def restore_plate_ring(plate_hd, a_hd, strength=0.5, amount=0.6, sigma=2.0):
    """RECOMMENDED (b)/(c): blend a band-localized plate unsharp toward the plate at
    `strength` in [0,1]. strength gives continuous control (the raw unsharp has a uint8
    quantization step); strength~0.5 lands the near-subject BG ring at the deep-BG level
    (~15 var-Lap), matching the ring to its OWN background instead of leaving a soft moat.
    Budget-independent (targets deep-BG, not the subject). strength=0 -> identity."""
    if strength <= 0:
        return plate_hd
    sh = sharpen_plate_band(plate_hd, a_hd, amount=amount, sigma=sigma)
    return cv2.addWeighted(plate_hd, 1.0 - strength, sh, strength, 0.0)
