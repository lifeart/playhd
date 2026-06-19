#!/usr/bin/env python3
"""
E2 -- Content-adaptive fallback/anchoring for the high-motion instant weak spot.

READ-ONLY driver: imports prototype/ (derisk, sr, region_quality) + server/ (anchor_sr)
without modifying any shared file. Measures three content-adaptive policies against the
deployed instant baseline (720p, INSTANT_SCALE=2, INSTANT_FALLBACK_THRESH=0.50, occ=reactive)
on the high-motion stress window A (start 0) and the realistic talking-head window C (start
5000), N=48, torch/MPS backend.

WHY THIS IS A WARP-ONLY SWEEP (cheap):
  The expensive thing is the SR network. We run it ONCE per (window, scale) to build the
  COMPACT-SR cache for EVERY frame (derisk.build_perframe_cache(...,'realesrgan')). Every
  policy is then just a per-frame CHOICE between bicubic-upscale (cheap fallback, the weak
  spot) and that cached compact-SR -- re-run through derisk.reconstruct (warp/blend only,
  zero SR re-runs). Putting the cached compact-SR into the cache for an "escalated" frame is
  EXACTLY what the deployed instant path does:
    * for a BACKBONE (I/P) frame, build_anchor_cache puts compact-SR in the cache so its
      detail propagates down the chain;
    * for a B LEAF, patch_high_fallback SR-patches the fallback ('none') pixels post-recon,
      which is identical to reconstruct reading perframe=compact-SR at those `none` pixels.
  So a single unified cache {i: compact_sr[i] if escalate(i) else bicubic[i]} reproduces the
  deployed (build_anchor_cache + reconstruct + patch_high_fallback) result for any threshold,
  with no SR re-run across the sweep.

HONEST METRICS (decisions made ONLY on these):
  * tOF (PRIMARY temporal): TecoGAN tOF = mean Farneback-flow EPE between the propagated HD
    recon (downscaled to LR) and the DECODED LR sequence (the cleanest motion ground-truth).
    Lower = steadier / tracks true motion. Deterministic -> GPU-contention-robust.
  * effective-bicubic% (PRIMARY weak-spot): mean over non-anchor frames of the fraction of
    pixels STILL served by cheap bicubic = hole_frac(i) if frame i was NOT escalated to
    compact-SR, else 0. This is the honest "how much of the frame is the soft fallback".
    raw-fallback% (= mean hole_frac, policy-independent) is reported alongside for context.
  * direct |dF| (cross-check): mean abs frame-to-frame LR change of the recon vs of the
    decoded LR (a raw flicker sanity check; tOF is the headline).
NOT used for decisions: LR-consistency (fallback is trivially LR-consistent) and
PSNR-vs-perframe (noisy / non-monotonic) -- per the methodology.

  SR-calls/frame (COST) = (#anchors + #escalated non-anchor frames) / N  (full-frame SR, the
  tile=False deployed default).

Run:  cd experiments/exp2_highmotion && python3 exp2.py
Out:  results.json (+ console tables).  REPORT.md is written from these by hand.
"""
import gc
import json
import os
import sys
import time

import cv2
import numpy as np

# ---- READ-ONLY imports of the validated prototype + server fast path ----------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_PROTO = os.path.join(_REPO, "prototype")
_SERVER = os.path.join(_REPO, "server")
for p in (_PROTO, _SERVER):
    if p not in sys.path:
        sys.path.insert(0, p)

import derisk as D                 # noqa: E402  decode / build_lr_flow / reconstruct / tof
import sr as SR                    # noqa: E402  compact SR net (latency accounting)
import region_quality as RQ        # noqa: E402  per-frame MV-magnitude (motion signal)
import torch                       # noqa: E402  MPS free between configs

CLIP = os.path.join(_REPO, "sample.mp4")
N = 48
SR_MODEL = "realesrgan"            # the deployed instant compact net (realesr-general-x4v3)
OCC = "reactive"                   # MODE_CONFIG['instant'] ships occ='reactive'
INSTANT_SCALE = 2                  # 720p tier
QHD_SCALE = 4                      # QHD tier (policy c)
BASELINE_THRESH = 0.50             # server.pipeline_api.INSTANT_FALLBACK_THRESH (safeguard ~off)

WINDOWS = [("A", 0, "high-motion (stress)"), ("C", 5000, "talking-head (realistic)")]


def _free_gpu():
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:                       # surface, never silently swallow
        print(f"  [warn] mps empty_cache failed: {e}")


# --------------------------------------------------------------------------- #
# Honest metrics
# --------------------------------------------------------------------------- #
def tof_vs_lr(recon_by_i, frames, w_lr, h_lr):
    """tOF of the propagated recon (downscaled to LR) vs the decoded LR (motion ground-truth).
    Reuses derisk.tof: tof(seq, ref) = mean EPE(Farneback(ref), Farneback(seq))."""
    n = len(frames)
    sm = (w_lr, h_lr)
    seq = [cv2.resize(recon_by_i[i], sm) for i in range(n)]
    lr = [frames[i][1] if frames[i][1].shape[1::-1] == sm else cv2.resize(frames[i][1], sm)
          for i in range(n)]
    return D.tof(seq, lr)


def direct_dframe_lr(recon_by_i, frames, w_lr, h_lr):
    """Raw mean abs frame-to-frame LR change of the recon and of the decoded LR (flicker
    cross-check; recon change well above the LR change => excess flicker)."""
    n = len(frames)
    sm = (w_lr, h_lr)
    seq = [cv2.resize(recon_by_i[i], sm).astype(np.float32) for i in range(n)]
    lr = [(frames[i][1] if frames[i][1].shape[1::-1] == sm
           else cv2.resize(frames[i][1], sm)).astype(np.float32) for i in range(n)]
    dr = float(np.mean([np.abs(seq[t] - seq[t - 1]).mean() for t in range(1, n)]))
    dl = float(np.mean([np.abs(lr[t] - lr[t - 1]).mean() for t in range(1, n)]))
    return dr, dl


# --------------------------------------------------------------------------- #
# Per-window setup: decode, compact-SR cache (once), bicubic cache, baseline hole_frac+motion
# --------------------------------------------------------------------------- #
def anchors_of(frames):
    """The frames derisk.reconstruct(anchor_set=set()) reads FULL SR from: every I-frame +
    the first backbone (I/P) frame (no in-window predecessor). Mirrors anchor_sr.anchor_indices."""
    backbone = D.backbone_indices(frames)
    first = backbone[0] if backbone else None
    return {i for i in backbone if frames[i][0] == "I" or i == first}, backbone


def per_frame_motion(frames, h_lr, w_lr):
    """Per-frame mean codec-MV magnitude (LR px/frame) over MV-covered pixels, + the no-MV
    (intra/disocclusion) fraction. The FREE motion signal for the motion-keyed policy."""
    mag_of, nomv_of = {}, {}
    for i, (pt, _, mvs) in enumerate(frames):
        if pt == "I" or mvs is None or len(mvs) == 0:
            mag_of[i], nomv_of[i] = 0.0, (1.0 if pt != "I" else 0.0)
            continue
        mag, no_mv = RQ.motion_mag_lr(mvs, h_lr, w_lr, want="all")
        valid = ~no_mv
        mag_of[i] = float(mag[valid].mean()) if valid.any() else 0.0
        nomv_of[i] = float(no_mv.mean())
    return mag_of, nomv_of


def reconstruct_warp_only(frames, scale, cache, anchor_set):
    """One warp/blend-only torch reconstruct (NO SR; cache already built). Returns R with
    GPU->host-downloaded recon + per-frame hole_frac. collect_metrics=False (no SSIM/PSNR)."""
    torch.mps.synchronize() if torch.backends.mps.is_available() else None
    t0 = time.perf_counter()
    _, R = D.reconstruct(frames, None, scale, True, OCC, cache, anchor_set,
                         backend="torch", collect_metrics=False, download_output=True)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    return R, dt


def setup_window(start, scale):
    """Decode + build the (single) compact-SR cache and bicubic cache at `scale`, and the
    anchor-only baseline reconstruction (gives per-frame hole_frac, the escalation trigger)."""
    frames = D.decode_lr_and_mvs(CLIP, start, N)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * scale, h_lr * scale
    anchors, backbone = anchors_of(frames)

    SR.reset_latency(SR_MODEL)
    t0 = time.perf_counter()
    compact = D.build_perframe_cache(frames, w_hd, h_hd, SR_MODEL)     # the ONE SR pass
    t_sr = time.perf_counter() - t0
    bicubic = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
               for i in range(N)}

    # baseline cache == deployed instant @ thresh 0.50: anchors=compact-SR, else bicubic
    base_cache = {i: (compact[i] if i in anchors else bicubic[i]) for i in range(N)}
    R0, _ = reconstruct_warp_only(frames, scale, base_cache, set())
    hole = {i: float(R0[i]["hole_frac"]) for i in range(N)}
    mag, nomv = per_frame_motion(frames, h_lr, w_lr)
    del R0
    _free_gpu()
    return dict(frames=frames, h_lr=h_lr, w_lr=w_lr, w_hd=w_hd, h_hd=h_hd,
                anchors=anchors, backbone=backbone, compact=compact, bicubic=bicubic,
                hole=hole, mag=mag, nomv=nomv, t_sr=round(t_sr, 2),
                sr_ms=round(SR.median_latency_ms(SR_MODEL), 1))


# --------------------------------------------------------------------------- #
# Build an escalation set + the matching cache, reconstruct, measure
# --------------------------------------------------------------------------- #
def escalate_set(W, thresh, mgate=None):
    """Non-anchor frames to escalate bicubic-fallback -> compact-SR. thresh on hole_frac;
    optional mgate gates on the FREE per-frame mean-|MV| motion signal (the motion-keyed
    policy escalates only genuinely-high-motion frames)."""
    out = set()
    for i in range(N):
        if i in W["anchors"]:
            continue
        if W["hole"][i] > thresh and (mgate is None or W["mag"][i] > mgate):
            out.add(i)
    return out


def measure(W, scale, esc):
    """Reconstruct with cache = {compact-SR for anchors|escalated, bicubic else}; return the
    honest metric record. esc = escalated non-anchor set; anchors always compact-SR."""
    frames = W["frames"]
    sr_frames = W["anchors"] | esc
    cache = {i: (W["compact"][i] if i in sr_frames else W["bicubic"][i]) for i in range(N)}
    R, dt = reconstruct_warp_only(frames, scale, cache, set())
    recon = {i: R[i]["recon"] for i in range(N)}
    tof = tof_vs_lr(recon, frames, W["w_lr"], W["h_lr"])
    d_recon, d_lr = direct_dframe_lr(recon, frames, W["w_lr"], W["h_lr"])

    nonanchor = [i for i in range(N) if i not in W["anchors"]]
    raw_fb = float(np.mean([W["hole"][i] for i in nonanchor]))
    eff_bic = float(np.mean([(0.0 if i in esc else W["hole"][i]) for i in nonanchor]))
    sr_calls = len(sr_frames)
    del R, recon
    _free_gpu()
    return dict(tof=round(tof, 4), eff_bicubic_pct=round(100 * eff_bic, 3),
                raw_fallback_pct=round(100 * raw_fb, 3),
                sr_calls=sr_calls, sr_calls_per_frame=round(sr_calls / N, 4),
                n_escalated=len(esc), escalated=sorted(esc),
                recon_ms=round(1000 * dt / N, 1),
                d_recon=round(d_recon, 3), d_lr=round(d_lr, 3))


# --------------------------------------------------------------------------- #
# Policy (b): adaptive re-anchoring (promote high-fallback BACKBONE P -> fresh anchor)
# --------------------------------------------------------------------------- #
def reanchor_measure(W, scale, budget):
    """derisk adaptive re-anchoring at fallback `budget` (frame-equiv accumulated fallback).
    Promoted P-anchors use fresh compact-SR (drift reset). Fallback pixels of the REMAINING
    non-anchors stay bicubic (no per-frame escalation) -- this isolates the re-anchoring lever.
    SR-calls = anchors (I+first+promoted)."""
    frames = W["frames"]
    # compute_anchor_set runs one cheap 'none' pass internally to read hole_frac; the SR cache
    # is already built so it is warp-only.
    aset = D.compute_anchor_set(frames, "adaptive", 1.0, budget, "fallback", None,
                                scale, True, OCC, W["compact"], backend="torch")
    all_anchors = W["anchors"] | set(aset)
    cache = {i: (W["compact"][i] if i in all_anchors else W["bicubic"][i]) for i in range(N)}
    R, dt = reconstruct_warp_only(frames, scale, cache, aset)
    recon = {i: R[i]["recon"] for i in range(N)}
    tof = tof_vs_lr(recon, frames, W["w_lr"], W["h_lr"])
    # after re-anchoring, promoted frames have hole_frac 0; remaining non-anchors keep bicubic.
    nonanchor = [i for i in range(N) if i not in all_anchors]
    raw_fb = float(np.mean([W["hole"][i] for i in nonanchor])) if nonanchor else 0.0
    eff_bic = raw_fb                                   # nothing escalated; all fallback is bicubic
    del R, recon
    _free_gpu()
    return dict(budget=budget, n_anchors=len(all_anchors), promoted=sorted(set(aset) - W["anchors"]),
                tof=round(tof, 4), eff_bicubic_pct=round(100 * eff_bic, 3),
                raw_fallback_pct=round(100 * raw_fb, 3),
                sr_calls=len(all_anchors), sr_calls_per_frame=round(len(all_anchors) / N, 4),
                recon_ms=round(1000 * dt / N, 1))


# --------------------------------------------------------------------------- #
def main():
    out = {"config": dict(N=N, sr_model=SR_MODEL, occ=OCC, instant_scale=INSTANT_SCALE,
                          qhd_scale=QHD_SCALE, baseline_thresh=BASELINE_THRESH, clip=CLIP),
           "windows": {}}

    # ---- policies (a) + (b): both windows at the 720p instant tier (scale 2) ----
    THRESHS = [0.50, 0.30, 0.20, 0.12, 0.08]          # 0.50 == deployed baseline
    MGATES = [None, 1.0, 2.0]                          # motion gate on mean-|MV| (LR px/frame)
    BUDGETS = [3.0, 2.0, 1.5, 1.0, 0.5]               # adaptive re-anchoring fallback budgets

    setups = {}
    for name, start, desc in WINDOWS:
        print(f"\n{'='*78}\n=== window {name} ({desc})  start={start}  scale={INSTANT_SCALE} (720p) ===")
        W = setup_window(start, INSTANT_SCALE)
        setups[name] = W
        types = "".join(f[0][0] for f in W["frames"])
        print(f"    types={types}")
        print(f"    anchors={sorted(W['anchors'])}  backbone={len(W['backbone'])}/{N}")
        print(f"    compact-SR cache: 1 pass, {W['t_sr']}s, median {W['sr_ms']} ms/frame")
        hv = [W["hole"][i] for i in range(N) if i not in W["anchors"]]
        mv = [W["mag"][i] for i in range(N) if i not in W["anchors"]]
        print(f"    non-anchor hole_frac: mean={100*np.mean(hv):.2f}%  max={100*max(hv):.2f}%  "
              f"#>0.08={sum(1 for x in hv if x>0.08)} #>0.20={sum(1 for x in hv if x>0.20)} "
              f"#>0.50={sum(1 for x in hv if x>0.50)}")
        print(f"    non-anchor mean|MV| (LR px/f): mean={np.mean(mv):.2f} max={max(mv):.2f}")

        wrec = {"desc": desc, "start": start, "types": types,
                "anchors": sorted(W["anchors"]), "n_backbone": len(W["backbone"]),
                "sr_ms": W["sr_ms"],
                "per_frame": {i: dict(type=W["frames"][i][0], hole=round(W["hole"][i], 4),
                                      mag=round(W["mag"][i], 3), nomv=round(W["nomv"][i], 4))
                              for i in range(N)},
                "policy_a_global": [], "policy_a_motion": [], "policy_b_reanchor": []}

        # policy (a): global threshold sweep (motion gate OFF)
        print(f"\n  -- policy (a) global hole_frac threshold (motion gate OFF) --")
        print(f"     {'thresh':>7}{'SR/f':>7}{'#esc':>6}{'tOF':>8}{'effBic%':>9}{'rawFb%':>8}{'reMs':>7}")
        for th in THRESHS:
            esc = escalate_set(W, th, mgate=None)
            r = measure(W, INSTANT_SCALE, esc)
            r["thresh"] = th
            wrec["policy_a_global"].append(r)
            tag = "  <= baseline" if th == BASELINE_THRESH else ""
            print(f"     {th:>7.2f}{r['sr_calls_per_frame']:>7.3f}{r['n_escalated']:>6}"
                  f"{r['tof']:>8.3f}{r['eff_bicubic_pct']:>9.2f}{r['raw_fallback_pct']:>8.2f}"
                  f"{r['recon_ms']:>7.1f}{tag}")

        # policy (a): motion-keyed (escalate only high-motion frames)
        print(f"  -- policy (a) motion-keyed (escalate iff hole>thresh AND mean|MV|>mgate) --")
        print(f"     {'thresh':>7}{'mgate':>7}{'SR/f':>7}{'#esc':>6}{'tOF':>8}{'effBic%':>9}")
        for th in [0.20, 0.12, 0.08]:
            for mg in [g for g in MGATES if g is not None]:
                esc = escalate_set(W, th, mgate=mg)
                r = measure(W, INSTANT_SCALE, esc)
                r["thresh"], r["mgate"] = th, mg
                wrec["policy_a_motion"].append(r)
                print(f"     {th:>7.2f}{mg:>7.1f}{r['sr_calls_per_frame']:>7.3f}{r['n_escalated']:>6}"
                      f"{r['tof']:>8.3f}{r['eff_bicubic_pct']:>9.2f}")

        # policy (b): adaptive re-anchoring
        print(f"  -- policy (b) adaptive re-anchoring (fallback budget) --")
        print(f"     {'budget':>7}{'#anch':>7}{'SR/f':>7}{'tOF':>8}{'effBic%':>9}{'rawFb%':>8}{'reMs':>7}")
        for b in BUDGETS:
            r = reanchor_measure(W, INSTANT_SCALE, b)
            wrec["policy_b_reanchor"].append(r)
            print(f"     {b:>7.1f}{r['n_anchors']:>7}{r['sr_calls_per_frame']:>7.3f}"
                  f"{r['tof']:>8.3f}{r['eff_bicubic_pct']:>9.2f}{r['raw_fallback_pct']:>8.2f}"
                  f"{r['recon_ms']:>7.1f}  promoted={r['promoted']}")

        out["windows"][name] = wrec
        # keep window A around for policy (c); free C
        if name == "C":
            del setups["C"]
            _free_gpu()

    # ---- policy (c): QHD-instant escalation on window A (scale 4 vs scale 2) ----
    print(f"\n{'='*78}\n=== policy (c) QHD escalation -- window A, scale {QHD_SCALE} (QHD) vs {INSTANT_SCALE} (720p) ===")
    WA2 = setups["A"]
    # scale-2 baseline tOF (anchor-only, bicubic fallback) for the apples-to-apples contrast
    base2 = measure(WA2, INSTANT_SCALE, set())
    # build the x4 compact-SR cache for A (one extra SR pass) and reconstruct at scale 4
    print(f"    building QHD (x4) compact-SR cache for window A (1 extra SR pass) ...")
    WA4 = setup_window(0, QHD_SCALE)
    base4 = measure(WA4, QHD_SCALE, set())
    # recon-time ratio (best-of-3, honest GPU sync), shared-GPU => report as a ratio
    def best_recon_ms(W, scale, n=3):
        cache = {i: (W["compact"][i] if i in W["anchors"] else W["bicubic"][i]) for i in range(N)}
        best = float("inf")
        for _ in range(n):
            _, dt = reconstruct_warp_only(W["frames"], scale, cache, set())
            best = min(best, 1000 * dt / N)
            _free_gpu()
        return round(best, 1)
    ms2 = best_recon_ms(WA2, INSTANT_SCALE)
    ms4 = best_recon_ms(WA4, QHD_SCALE)
    crec = dict(tof_720p=base2["tof"], tof_qhd=base4["tof"],
                eff_bicubic_pct=base4["eff_bicubic_pct"],           # scale-independent (hole_frac)
                recon_ms_720p=ms2, recon_ms_qhd=ms4,
                recon_ratio_qhd_over_720p=round(ms4 / ms2, 2) if ms2 else None,
                sr_ms_720p=WA2["sr_ms"], sr_ms_qhd=WA4["sr_ms"])
    print(f"    720p(x2): tOF={base2['tof']:.3f}  recon best {ms2} ms/f  SR {WA2['sr_ms']} ms/f")
    print(f"    QHD (x4): tOF={base4['tof']:.3f}  recon best {ms4} ms/f  SR {WA4['sr_ms']} ms/f")
    print(f"    recon-time ratio QHD/720p = {crec['recon_ratio_qhd_over_720p']}x  "
          f"(tOF measured at common LR; effBic% identical -- hole_frac is scale-free)")
    out["windows"]["A"]["policy_c_qhd"] = crec

    with open(os.path.join(_HERE, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(_HERE, 'results.json')}")


if __name__ == "__main__":
    main()
