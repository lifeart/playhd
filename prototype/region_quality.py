#!/usr/bin/env python3
"""
Stream-1 / region_quality.py -- PER-PIXEL motion-aware detail policy.

THE FINDING WE ACT ON (Step 8, verified): a heavy perceptual anchor (RealESRGAN_x4plus)
adds real high-frequency detail, but:
  * on LOW-motion / STATIC content that detail PROPAGATES well (warps are near-identity,
    it rides along undegraded for the whole GOP), and
  * on HIGH-motion content it is ERODED in ~1 frame (warp-blur + occlusion) AND the
    extra hallucinated HF is TEMPORALLY UNSTABLE -- uniform x4plus flickers more than the
    compact net (Step 8 measured tOF 0.66 vs 0.33 on the talking-head chain).
So heavy detail HELPS static regions and is WASTED (and flickery) on moving regions.

THIS MODULE gives the static-region detail benefit WITHOUT the uniform-x4plus flicker
penalty, by GATING detail per pixel with a motion map derived from the codec MVs:

  motion map (at LR, from build_lr_flow):  m(x,y) = |MV| magnitude per block.
      low  m  -> STATIC   -> keep the HEAVY-anchor propagated detail (persists, stable).
      high m  -> DYNAMIC  -> prefer the temporally-stable COMPACT source (heavy detail is
                            eroded anyway and its fresh re-injection is what flickers).
      no MV (disocclusion) inside a P/B -> treated as DYNAMIC (new content -> unstable).
      anchor frame (fresh full-frame SR) -> all-static (use heavy everywhere).

  static-weight  a(x,y) = clamp((HI - m)/(HI - LO), 0, 1)   (spatially smoothed; 1=static).
  region-aware recon     = a * recon_heavy + (1 - a) * recon_compact   (per pixel, per frame).

We reconstruct one real mixed-motion window THREE ways and measure, SPLIT BY REGION
(static vs dynamic masks built from the temporal-mean motion magnitude) and overall:
  * sharpness  = variance of the luma Laplacian inside the region (HF/detail proxy), and
  * tOF        = TecoGAN temporal-OF flicker vs the decoded LR (cleanest motion truth).

TARGET to demonstrate: region-aware keeps STATIC-region sharpness near uniform-x4plus while
keeping OVERALL tOF near uniform-compact -- i.e. recover most of the heavy detail without
most of the flicker. We report the ACTUAL deltas + an honest verdict.

Imports derisk/sr READ-ONLY (no shared files edited). Numpy backend (deterministic). Foreground.

    python3 region_quality.py --probe                 # fast: motion stats only, calibrate LO/HI
    python3 region_quality.py                          # full run (heavy SR; writes out_region/)
    python3 region_quality.py --heavy bicubic --compact bicubic --window-frames 8  # plumbing smoke
"""
import argparse
import csv
import os

import cv2
import numpy as np

import derisk as d            # READ-ONLY import (decode/build_lr_flow/reconstruct/tof/...)

_HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(_HERE, "..", "sample.mp4")
OUT = os.path.join(_HERE, "out_region")


# --------------------------------------------------------------------------- #
# Motion map: per-block MV magnitude at LR, from the codec MVs (build_lr_flow)
# --------------------------------------------------------------------------- #
def motion_mag_lr(mvs, h_lr, w_lr, want="all"):
    """Per-pixel MV magnitude at LR from the codec motion vectors. Returns (mag, no_mv):
      mag[y,x]   = sqrt(dx^2 + dy^2) in LR pixels (the codec block MV that covers the pixel),
      no_mv[y,x] = True where NO MV of `want` covers the pixel (intra block / disocclusion).
    NaN flow (no MV) is returned as mag=NaN; callers decide how to treat disocclusion."""
    fx, fy = d.build_lr_flow(mvs, h_lr, w_lr, want=want)
    mag = np.sqrt(fx * fx + fy * fy)               # NaN where no MV
    return mag, np.isnan(fx)


def _alpha_from_mag(m, lo, hi):
    """Map a motion-magnitude map (LR px) -> static-weight a in [0,1]: 1 below `lo` (static ->
    HEAVY), 0 above `hi` (dynamic -> COMPACT), linear ramp between."""
    a = (hi - m) / max(hi - lo, 1e-6)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def window_static_weight(motion_stat, lo, hi, feather=0):
    """TEMPORALLY-STABLE gate: one static-weight map for the whole window, from the aggregated
    per-pixel motion statistic (`motion_stat`, smoothed temporal-mean |MV|). FIXED in time, so the
    heavy/compact blend seam does not move frame-to-frame (no per-frame seam jitter). `feather`
    (odd LR-px Gaussian kernel) widens the heavy->compact transition: a wider, gentler sharpness
    gradient produces a smaller spatial-OF artifact at the seam (a sharp seam tears against moving
    content). Physical reading of hi=1.0 LR px/frame: mean motion above ~1 px/frame cannot carry the
    heavy anchor's HF stably under warp -> fall back to the stable compact."""
    a = _alpha_from_mag(motion_stat, lo, hi)
    if feather and feather >= 3:
        k = int(feather) | 1
        a = cv2.GaussianBlur(a, (k, k), 0)
    return a


def static_weight_perframe(mvs, h_lr, w_lr, is_anchor, lo, hi, blur=9):
    """ABLATION gate: per-frame INSTANTANEOUS motion. Adapts to the current frame's motion, but the
    seam between heavy/compact MOVES every frame -> it RE-INTRODUCES flicker (shown in the table).
      anchor frame -> 1 everywhere (fresh full-frame SR; no propagation flicker -> use heavy).
      else         -> a from the smoothed instantaneous |MV|; disocclusion (no MV) -> hi (=> compact)."""
    if is_anchor:
        return np.ones((h_lr, w_lr), np.float32)
    mag, no_mv = motion_mag_lr(mvs, h_lr, w_lr, want="all")
    m = np.where(no_mv, hi, mag).astype(np.float32)       # disocclusion -> dynamic
    if blur and blur >= 3:
        m = cv2.GaussianBlur(m, (blur, blur), 0)          # blocks are coarse -> smooth edges
    return _alpha_from_mag(m, lo, hi)


def region_masks(frames, h_lr, w_lr, pct_static, pct_dynamic):
    """STATIC / DYNAMIC spatial masks for the window, from the TEMPORAL-MEAN motion magnitude
    (per pixel, averaged over the P/B frames that carry MVs). Data-driven thresholds:
      static  = mean motion below the `pct_static` percentile,
      dynamic = mean motion above the `pct_dynamic` percentile.
    Returns (static_lr, dynamic_lr, meanmag, info) -- masks guarantee non-trivial coverage; we
    report the actual mean motion inside each so the separation is shown to be real, not a slice."""
    acc = np.zeros((h_lr, w_lr), np.float64)
    cnt = np.zeros((h_lr, w_lr), np.float64)
    for pt, _lr, mvs, *_ in frames:            # R12-E2: tolerate the optional QP 4th element
        if pt == "I" or mvs is None or len(mvs) == 0:
            continue
        mag, no_mv = motion_mag_lr(mvs, h_lr, w_lr, want="all")
        valid = ~no_mv
        acc[valid] += mag[valid]
        cnt[valid] += 1.0
    meanmag = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0).astype(np.float32)
    meanmag_s = cv2.GaussianBlur(meanmag, (9, 9), 0)
    t_static = float(np.percentile(meanmag_s, pct_static))
    t_dynamic = float(np.percentile(meanmag_s, pct_dynamic))
    static_lr = meanmag_s <= t_static
    dynamic_lr = meanmag_s >= max(t_dynamic, t_static + 1e-3)
    info = dict(t_static=t_static, t_dynamic=t_dynamic,
                static_cov=float(static_lr.mean()), dynamic_cov=float(dynamic_lr.mean()),
                static_motion=float(meanmag_s[static_lr].mean()) if static_lr.any() else 0.0,
                dynamic_motion=float(meanmag_s[dynamic_lr].mean()) if dynamic_lr.any() else 0.0,
                mean_all=float(meanmag_s.mean()), p50=float(np.percentile(meanmag_s, 50)),
                p90=float(np.percentile(meanmag_s, 90)), p99=float(np.percentile(meanmag_s, 99)))
    return static_lr, dynamic_lr, meanmag_s, info


# --------------------------------------------------------------------------- #
# Region-aware blend
# --------------------------------------------------------------------------- #
def cached_perframe(frames, w_hd, h_hd, model, window_start, out):
    """Build the per-frame SR cache, memoized to disk so repeated runs skip the (heavy) SR pass.
    Heavy x4plus is ~2.2 s/frame on MPS; caching the stacked uint8 HD frames to .npy lets the gate/
    feather/measurement be re-run cheaply. Keyed by (window, frame count, model)."""
    cdir = os.path.join(out, "cache")
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"sr_{window_start}_{len(frames)}_{model}.npy")
    if os.path.exists(path):
        arr = np.load(path)
        if arr.shape[0] == len(frames) and arr.shape[1] == h_hd and arr.shape[2] == w_hd:
            print(f"  [cache] {model}: loaded {path}")
            return {i: np.ascontiguousarray(arr[i]) for i in range(len(frames))}
    cache = d.build_perframe_cache(frames, w_hd, h_hd, model)
    np.save(path, np.stack([cache[i] for i in range(len(frames))]))
    return cache


def blend_region_aware(recon_heavy, recon_compact, a_lr, scale):
    """Per-pixel blend: a*heavy + (1-a)*compact, with the LR static-weight upsampled to HD.
    Static (a~1) -> heavy detail; dynamic (a~0) -> compact (temporally stable)."""
    h_hd, w_hd = recon_heavy.shape[:2]
    a_hd = cv2.resize(a_lr, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)[..., None]
    out = a_hd * recon_heavy.astype(np.float32) + (1.0 - a_hd) * recon_compact.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Region-split metrics
# --------------------------------------------------------------------------- #
def _lap(rgb):
    return cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F)


def sharp_regionsplit(recon_list, static_hd, dynamic_hd):
    """Mean over frames of var-of-Laplacian inside STATIC / DYNAMIC / ALL pixels (sharpness)."""
    vs, vd, va = [], [], []
    for r in recon_list:
        lap = _lap(r)
        if static_hd.any():
            vs.append(float(lap[static_hd].var()))
        if dynamic_hd.any():
            vd.append(float(lap[dynamic_hd].var()))
        va.append(float(lap.var()))
    f = lambda x: float(np.mean(x)) if x else float("nan")
    return f(vs), f(vd), f(va)


def tof_regionsplit(recon_list, ref_lr_list, static_lr, dynamic_lr):
    """TecoGAN tOF (Farneback-flow EPE vs decoded-LR motion) averaged inside STATIC / DYNAMIC /
    ALL pixels at LR. Lower = steadier / less flicker. recon downscaled to LR for the flow."""
    h_lr, w_lr = ref_lr_list[0].shape[:2]
    seq = [cv2.resize(r, (w_lr, h_lr)) for r in recon_list]
    es, ed, ea = [], [], []
    for t in range(1, len(seq)):
        df = d._farneback(ref_lr_list[t - 1], ref_lr_list[t]) - d._farneback(seq[t - 1], seq[t])
        epe = np.sqrt((df * df).sum(axis=-1))      # HxW per-pixel EPE
        if static_lr.any():
            es.append(float(epe[static_lr].mean()))
        if dynamic_lr.any():
            ed.append(float(epe[dynamic_lr].mean()))
        ea.append(float(epe.mean()))
    f = lambda x: float(np.mean(x)) if x else float("nan")
    return f(es), f(ed), f(ea)


# --------------------------------------------------------------------------- #
# Visuals
# --------------------------------------------------------------------------- #
def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _motion_heat(a_lr, w_hd, h_hd):
    """Static-weight map -> RGB heatmap (red = static/heavy, blue = dynamic/compact)."""
    a = cv2.resize(a_lr, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)   # BGR
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def write_visuals(out, vi, frames, scale, R_heavy, R_compact, ra, a_lr_of, static_lr, dynamic_lr,
                  meanmag, info):
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * scale, h_lr * scale

    # (1) motion-map panel for the chosen frame: weight heatmap + the three full frames
    heat = _motion_heat(a_lr_of[vi], w_hd, h_hd)
    panels = [_label(heat, f"motion map a (red=static/heavy) f{vi}"),
              _label(R_compact[vi]["recon"], "uniform compact"),
              _label(R_heavy[vi]["recon"], "uniform x4plus (heavy)"),
              _label(ra[vi], "REGION-AWARE")]
    # downscale the big montage for a manageable file
    montage = np.concatenate([cv2.resize(p, (w_hd // 2, h_hd // 2)) for p in panels], axis=0)
    cv2.imwrite(os.path.join(out, f"motionmap_frame{vi:03d}.png"),
                cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

    # (2) 1:1 center crops so the HF detail is visible: compact | x4plus | region-aware
    cs = 360
    y0, x0 = max(0, h_hd // 2 - cs // 2), max(0, w_hd // 2 - cs // 2)
    crop = lambda im, lab: _label(im[y0:y0 + cs, x0:x0 + cs], lab)
    crops = np.concatenate([crop(R_compact[vi]["recon"], "compact"),
                            crop(R_heavy[vi]["recon"], "x4plus"),
                            crop(ra[vi], "region-aware")], axis=1)
    cv2.imwrite(os.path.join(out, f"crops_frame{vi:03d}.png"),
                cv2.cvtColor(crops, cv2.COLOR_RGB2BGR))

    # (3) region masks overlay on the recon (green=static, red=dynamic)
    base = ra[vi].copy()
    s_hd = cv2.resize(static_lr.astype(np.uint8), (w_hd, h_hd), interpolation=cv2.INTER_NEAREST).astype(bool)
    d_hd = cv2.resize(dynamic_lr.astype(np.uint8), (w_hd, h_hd), interpolation=cv2.INTER_NEAREST).astype(bool)
    ov = base.copy()
    ov[s_hd] = (0.5 * ov[s_hd] + np.array([0, 180, 0])).clip(0, 255).astype(np.uint8)
    ov[d_hd] = (0.5 * ov[d_hd] + np.array([200, 0, 0])).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out, "region_masks.png"),
                cv2.cvtColor(_label(cv2.resize(ov, (w_hd // 2, h_hd // 2)),
                                    f"static(green) {100*info['static_cov']:.0f}%  "
                                    f"dynamic(red) {100*info['dynamic_cov']:.0f}%"),
                             cv2.COLOR_RGB2BGR))

    # (4) raw temporal-mean motion magnitude heatmap
    mm = meanmag / max(float(meanmag.max()), 1e-6)
    mmh = cv2.applyColorMap((mm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(out, "mean_motion.png"),
                cv2.resize(mmh, (w_hd // 2, h_hd // 2)))


def write_barchart(out, table, info, window_tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["compact", "x4plus", "ra-stable", "ra-wide", "ra-perframe"]
    colors = {"compact": "tab:blue", "x4plus": "tab:red", "ra-stable": "tab:green",
              "ra-wide": "tab:cyan", "ra-perframe": "tab:olive"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    regions = ["static", "dynamic", "overall"]
    x = np.arange(len(regions))
    bw = 0.16
    for j, m in enumerate(methods):
        axes[0].bar(x + (j - 2) * bw, [table[m]["sharp"][r] for r in regions], bw,
                    label=m, color=colors[m])
        axes[1].bar(x + (j - 2) * bw, [table[m]["tof"][r] for r in regions], bw,
                    label=m, color=colors[m])
    axes[0].set_title("Sharpness (var-of-Laplacian) by region -- higher = more detail")
    axes[0].set_ylabel("var-of-Laplacian")
    axes[1].set_title("tOF (flicker vs decoded LR) by region -- lower = steadier")
    axes[1].set_ylabel("tOF")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9)
    fig.suptitle(f"Region-aware detail gating ({window_tag}): keep static detail, cut dynamic flicker",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out, "region_split.png"), dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-start", type=int, default=5000, help="first display frame (talking-head)")
    ap.add_argument("--window-frames", type=int, default=48)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--heavy", default="realesrgan-x4plus", help="static-region SR model")
    ap.add_argument("--compact", default="realesrgan", help="dynamic-region (stable) SR model")
    ap.add_argument("--occ", default="full", choices=["naive", "full", "reactive", "adaptive"])
    ap.add_argument("--lo", type=float, default=0.2, help="static-weight: motion px at/below => a=1 (heavy)")
    ap.add_argument("--hi", type=float, default=1.0, help="static-weight: motion px at/above => a=0 (compact)")
    ap.add_argument("--pct-static", type=float, default=45.0, help="region mask: static <= this pct")
    ap.add_argument("--pct-dynamic", type=float, default=80.0, help="region mask: dynamic >= this pct")
    ap.add_argument("--feather", type=int, default=9, help="stable-gate seam feather (LR px)")
    ap.add_argument("--feather-wide", type=int, default=61, help="wide-seam stable-gate feather (LR px)")
    ap.add_argument("--probe", action="store_true", help="decode + motion stats only (fast, no SR)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if not os.path.exists(SAMPLE):
        raise SystemExit(f"sample clip not found: {SAMPLE}")
    frames = d.decode_lr_and_mvs(SAMPLE, args.window_start, args.window_frames)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * args.scale, h_lr * args.scale
    types = "".join(f[0][0] for f in frames)
    window_tag = f"start {args.window_start}, {len(frames)}f, x{args.scale}"
    print(f"window: {window_tag}  LR {w_lr}x{h_lr} -> HD {w_hd}x{h_hd}  types={types}")

    static_lr, dynamic_lr, meanmag, info = region_masks(frames, h_lr, w_lr,
                                                        args.pct_static, args.pct_dynamic)
    print(f"motion (LR px/frame): mean={info['mean_all']:.2f} p50={info['p50']:.2f} "
          f"p90={info['p90']:.2f} p99={info['p99']:.2f}")
    print(f"region masks: STATIC cov={100*info['static_cov']:.0f}% (mean motion "
          f"{info['static_motion']:.2f}px)  DYNAMIC cov={100*info['dynamic_cov']:.0f}% "
          f"(mean motion {info['dynamic_motion']:.2f}px)  [thresholds "
          f"{info['t_static']:.2f}/{info['t_dynamic']:.2f}px]")
    if args.probe:
        # report a per-frame motion summary so LO/HI for the static-weight can be calibrated
        print("\nper-frame motion magnitude (non-anchor frames):")
        for i, (pt, _, mvs) in enumerate(frames):
            if pt == "I" or mvs is None or len(mvs) == 0:
                print(f"  f{i:02d} {pt}: (anchor / no MV)")
                continue
            mag, no_mv = motion_mag_lr(mvs, h_lr, w_lr, want="all")
            v = mag[~no_mv]
            print(f"  f{i:02d} {pt}: nMV%={100*(~no_mv).mean():4.0f}  motion p50={np.percentile(v,50):.2f} "
                  f"p90={np.percentile(v,90):.2f} max={v.max():.2f}px  no-MV={100*no_mv.mean():.1f}%")
        return

    os.makedirs(args.out, exist_ok=True)

    # ---- SR caches (computed ONCE per model; reused for all reconstructions) ----
    print(f"\nbuilding per-frame SR caches (heavy={args.heavy}, compact={args.compact}) ...")
    heavy_cache = cached_perframe(frames, w_hd, h_hd, args.heavy, args.window_start, args.out)
    compact_cache = cached_perframe(frames, w_hd, h_hd, args.compact, args.window_start, args.out)

    # ---- TWO uniform reconstructions: identical warp/occlusion, different SR source ----
    anchor_set = set()                                   # I-frames only (the natural GOP)
    print("reconstructing uniform-x4plus (heavy everywhere) ...")
    _, R_heavy = d.reconstruct(frames, None, args.scale, True, args.occ, heavy_cache,
                               anchor_set, backend="numpy")
    print("reconstructing uniform-compact (compact everywhere) ...")
    _, R_compact = d.reconstruct(frames, None, args.scale, True, args.occ, compact_cache,
                                 anchor_set, backend="numpy")

    # ---- THIRD: region-aware per-pixel blends. TWO gates:
    #   ra-stable   (PRIMARY): one fixed window gate from the aggregated motion map -> seam does
    #                          NOT move -> no seam flicker.
    #   ra-perframe (ABLATION): per-frame instantaneous gate -> the moving seam re-adds flicker. ----
    N = len(frames)
    a_stable = window_static_weight(meanmag, args.lo, args.hi, feather=args.feather)
    a_wide = window_static_weight(meanmag, args.lo, args.hi, feather=args.feather_wide)
    a_pf_of, ra_stable, ra_wide, ra_pf = {}, {}, {}, {}
    for i in range(N):
        pt, _, mvs = frames[i]
        ra_stable[i] = blend_region_aware(R_heavy[i]["recon"], R_compact[i]["recon"],
                                          a_stable, args.scale)
        ra_wide[i] = blend_region_aware(R_heavy[i]["recon"], R_compact[i]["recon"],
                                        a_wide, args.scale)
        a_pf_of[i] = static_weight_perframe(mvs, h_lr, w_lr, R_heavy[i]["is_anchor"], args.lo, args.hi)
        ra_pf[i] = blend_region_aware(R_heavy[i]["recon"], R_compact[i]["recon"],
                                      a_pf_of[i], args.scale)

    # ---- measure: sharpness + tOF, split by static/dynamic region + overall ----
    static_hd = cv2.resize(static_lr.astype(np.uint8), (w_hd, h_hd),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    dynamic_hd = cv2.resize(dynamic_lr.astype(np.uint8), (w_hd, h_hd),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    ref_lr = [frames[i][1] for i in range(N)]
    seqs = {"compact": [R_compact[i]["recon"] for i in range(N)],
            "x4plus": [R_heavy[i]["recon"] for i in range(N)],
            "ra-stable": [ra_stable[i] for i in range(N)],
            "ra-wide": [ra_wide[i] for i in range(N)],
            "ra-perframe": [ra_pf[i] for i in range(N)]}
    methods = ("compact", "x4plus", "ra-stable", "ra-wide", "ra-perframe")
    table = {}
    for m, seq in seqs.items():
        ss, sd, sa = sharp_regionsplit(seq, static_hd, dynamic_hd)
        ts, td, ta = tof_regionsplit(seq, ref_lr, static_lr, dynamic_lr)
        table[m] = dict(sharp=dict(static=ss, dynamic=sd, overall=sa),
                        tof=dict(static=ts, dynamic=td, overall=ta))

    # ---- print the region-split table ----
    print(f"\n================ REGION-SPLIT TABLE ({window_tag}) ================")
    print("                  STATIC region        DYNAMIC region       OVERALL")
    print("method          sharp     tOF        sharp     tOF        sharp     tOF")
    for m in methods:
        t = table[m]
        print(f"{m:14s}  {t['sharp']['static']:7.1f}  {t['tof']['static']:6.3f}    "
              f"{t['sharp']['dynamic']:7.1f}  {t['tof']['dynamic']:6.3f}    "
              f"{t['sharp']['overall']:7.1f}  {t['tof']['overall']:6.3f}")

    # ---- deltas / verdict.  PRIMARY (recommended) = ra-wide: a STABLE, WIDELY-FEATHERED gate.
    # ra-stable (sharp seam) and ra-perframe (moving seam) are ABLATIONS that isolate the seam. ----
    c, x = table["compact"], table["x4plus"]
    rw, r, rp = table["ra-wide"], table["ra-stable"], table["ra-perframe"]
    den_sharp = x["sharp"]["static"] - c["sharp"]["static"]
    den_tof = x["tof"]["overall"] - c["tof"]["overall"]
    recov = (rw["sharp"]["static"] - c["sharp"]["static"]) / den_sharp if abs(den_sharp) > 1e-6 else float("nan")
    flick_avoid = (x["tof"]["overall"] - rw["tof"]["overall"]) / den_tof if abs(den_tof) > 1e-6 else float("nan")
    fa_narrow = (x["tof"]["overall"] - r["tof"]["overall"]) / den_tof if abs(den_tof) > 1e-6 else float("nan")
    fa_pf = (x["tof"]["overall"] - rp["tof"]["overall"]) / den_tof if abs(den_tof) > 1e-6 else float("nan")
    print("\n---- deltas (PRIMARY = ra-wide: stable gate + wide seam) ----")
    print(f"STATIC sharpness: compact {c['sharp']['static']:.1f} -> x4plus {x['sharp']['static']:.1f} "
          f"(heavy adds +{den_sharp:.1f}); ra-wide {rw['sharp']['static']:.1f} "
          f"=> recovers {100*recov:.0f}% of the heavy STATIC detail")
    print(f"OVERALL tOF:      compact {c['tof']['overall']:.3f}  x4plus {x['tof']['overall']:.3f} "
          f"(heavy adds +{den_tof:.3f} flicker); ra-wide {rw['tof']['overall']:.3f} "
          f"=> avoids {100*flick_avoid:.0f}% of x4plus's excess flicker (<=0 = at/below compact)")
    print(f"DYNAMIC tOF:      compact {c['tof']['dynamic']:.3f}  x4plus {x['tof']['dynamic']:.3f}  "
          f"ra-wide {rw['tof']['dynamic']:.3f}")
    print(f"ABLATION (SEAM is the lever): sharp seam ra-stable OVERALL tOF {r['tof']['overall']:.3f} "
          f"(avoids {100*fa_narrow:.0f}%) vs wide seam ra-wide {rw['tof']['overall']:.3f} "
          f"(avoids {100*flick_avoid:.0f}%) -- widening the heavy/compact transition removes the "
          f"seam tear")
    print(f"ABLATION (gate stability): per-frame gate ra-perframe OVERALL tOF {rp['tof']['overall']:.3f} "
          f"(avoids {100*fa_pf:.0f}%) -- the MOVING per-frame seam keeps too much heavy on the "
          f"dynamic region")
    good_detail = np.isfinite(recov) and recov >= 0.6
    good_flicker = np.isfinite(flick_avoid) and flick_avoid >= 0.5
    verdict = ("YES -- recovers most static detail AND keeps tOF at/below compact" if good_detail and good_flicker
               else "PARTIAL -- see deltas" if good_detail or good_flicker
               else "NO -- did not separate detail from flicker on this window")
    print(f"\nVERDICT: region-aware (stable+wide gate) {verdict}.")

    # ---- visuals + chart + csv (visual uses the PRIMARY ra-wide result + its gate) ----
    vi = max((i for i in range(N) if not R_heavy[i]["is_anchor"]),
             key=lambda i: float((a_pf_of[i] < 0.5).mean()), default=N // 2)  # most-dynamic frame
    a_wide_of = {i: a_wide for i in range(N)}
    write_visuals(args.out, vi, frames, args.scale, R_heavy, R_compact, ra_wide, a_wide_of,
                  static_lr, dynamic_lr, meanmag, info)
    write_barchart(args.out, table, info, window_tag)
    with open(os.path.join(args.out, "region_split.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["method", "region", "sharpness_varlap", "tof"])
        for m in methods:
            for rg in ("static", "dynamic", "overall"):
                wr.writerow([m, rg, round(table[m]["sharp"][rg], 3), round(table[m]["tof"][rg], 4)])
    print(f"\nwrote -> {args.out}/  (region_split.png/.csv, motionmap_frame{vi:03d}.png, "
          f"crops_frame{vi:03d}.png, region_masks.png, mean_motion.png)")


if __name__ == "__main__":
    main()
