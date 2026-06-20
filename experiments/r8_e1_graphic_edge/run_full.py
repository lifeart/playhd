#!/usr/bin/env python3
"""R8-E1 consolidated measurement (one compact-SR pass per clip; GPU-contention-friendly).

Answers, per moving-graphic clip, on the GRAPHIC-region pixels (motion-compensated):
  Q1 PROBLEM?    propagation reg-dF vs per-frame SR reg-dF (>1 => propagation shimmers more).
  Q2 SELF-HEAL (instant)?  reactive/full FALLBACK% on the bar (high => already routed to fresh SR).
  Q3 SELF-HEAL (quality)?  region-aware a_lr on the bar + region-aware-output reg-dF
                           (a~0 => output==compact per-frame SR == the fix => REDUNDANT in quality).
  Q4 metric cross-check:   LR-tOF (blind to HF) vs HD-tOF on the bar crop (sees HF).
  Q5 SEAM:                 pinned composite reg-dF on a ring around the bar (new flicker?).
  Q6 DETECTOR FP:          bimodality-only (no motion gate) coverage on the moving bar vs the
                           talking-head face (must fire on bar, stay ~0 on face).
"""
import os
import sys
import numpy as np
import cv2

import exp_common as E
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prototype"))
import derisk as d
import region_quality as rq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp3_visual"))
import graphic_detect as gd

START, N = 5000, 48
SCALE = 4
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ART, exist_ok=True)


def region_gate(frames):
    """Quality-mode region gate a_lr (shipped params lo=0.2,hi=1.0,feather=61), reusing
    region_quality exactly as derisk._build_region_gate does (no x4plus needed here)."""
    h_lr, w_lr = frames[0][1].shape[:2]
    _, _, meanmag, _ = rq.region_masks(frames, h_lr, w_lr, 45.0, 80.0)
    a_lr = rq.window_static_weight(meanmag, 0.2, 1.0, feather=61)
    return a_lr, meanmag


def hd_tof_bar(seq, ref_cubic, mask_hd):
    """HD-tOF restricted to the bar bounding box (sees HF edge wobble that LR-tOF blurs away)."""
    ys, xs = np.where(mask_hd)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = [r[y0:y1, x0:x1] for r in seq]
    rf = [r[y0:y1, x0:x1] for r in ref_cubic]
    return d.tof(s, rf)


def analyze_ticker(name, v_lr, **enc):
    rgb, h, w = E.decode_clean_rgb(START, N)
    mod, mask_lr, v = E.overlay_ticker(rgb, h, w, v_lr=v_lr)
    path = os.path.join(E.TMP, f"full_{name}.mp4")
    E.encode_h264(mod, path, **enc)
    frames = E.decode_mvs(path, N)
    h_lr, w_lr = frames[0][1].shape[:2]
    mask_hd = E.upscale_mask(mask_lr, SCALE)
    v_hd = v * SCALE
    ref_lr = [frames[i][1] for i in range(N)]
    cubic = [cv2.resize(frames[i][1], (w_lr * SCALE, h_lr * SCALE), interpolation=cv2.INTER_CUBIC)
             for i in range(N)]

    print(f"\n############### TICKER {name}  v_lr={v}  v_hd={v_hd}  enc={enc} ###############")
    # --- one compact-SR pass; reuse cache for both occ modes (numpy recon is cheap) ---
    Rr, pf = E.build_recon(frames, SCALE, sr_mode="realesrgan", occ="reactive")
    recon_r = [Rr[i]["recon"] for i in range(N)]
    perframe = [pf[i] for i in range(N)]
    _, Rf = d.reconstruct(frames, None, SCALE, True, "full", pf, set(),
                          backend="numpy", collect_metrics=False, download_output=True)
    recon_f = [Rf[i]["recon"] for i in range(N)]

    # --- Q3 quality region-aware self-heal (heavy proxy = compact propagation; on bar a~0 the
    #     heavy choice is irrelevant, output -> compact per-frame SR) ---
    a_lr, meanmag = region_gate(frames)
    a_bar = a_lr[mask_lr]
    rows_bar = a_lr[mask_lr.any(axis=1)]            # a per bar row (feather-bleed check)
    region_aware = [rq.blend_region_aware(recon_r[i], perframe[i], a_lr, SCALE) for i in range(N)]

    seqs = {
        "per-frame compact SR (=pin)": perframe,
        "LR cubic (ref)": cubic,
        "propagation occ=reactive": recon_r,
        "propagation occ=full": recon_f,
        "quality region-aware blend": region_aware,
    }
    print(f"  bar={100*mask_lr.mean():.1f}% of frame | meanmag(bar)={meanmag[mask_lr].mean():.2f} "
          f"px | region-aware a_lr(bar) mean={a_bar.mean():.3f} max={a_bar.max():.3f}")
    print(f"  {'sequence':30s} {'reg-dF':>8s} {'raw|dF|':>8s} {'LR-tOF':>7s} {'HDtOF-bar':>9s}")
    res = {}
    for k, seq in seqs.items():
        rdf = E.registered_dframe(seq, mask_hd, v_hd)
        raw = E.raw_dframe(seq, mask_hd)
        ltof = E.tof_lr(seq, ref_lr)
        htof = hd_tof_bar(seq, cubic, mask_hd)
        res[k] = rdf
        print(f"  {k:30s} {rdf:8.3f} {raw:8.3f} {ltof:7.4f} {htof:9.4f}")
    ffr, _ = E.fallback_frac_on_mask(Rr, mask_hd, range(1, N))
    fff, _ = E.fallback_frac_on_mask(Rf, mask_hd, range(1, N))
    print(f"  Q2 FALLBACK% on bar: reactive={100*ffr:.1f}%  full={100*fff:.1f}%")
    print(f"  -> propagation/per-frame reg-dF ratio = "
          f"{res['propagation occ=reactive']/res['per-frame compact SR (=pin)']:.2f}x "
          f"| region-aware/per-frame = "
          f"{res['quality region-aware blend']/res['per-frame compact SR (=pin)']:.2f}x")

    # --- Q5 SEAM: pinned composite = propagation with bar replaced by per-frame SR. Measure new
    #     flicker on a RING just OUTSIDE the bar (registered by v, though the ring is background). ---
    pinned = []
    for i in range(N):
        o = recon_r[i].copy(); o[mask_hd] = perframe[i][mask_hd]; pinned.append(o)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    ring = (cv2.dilate(mask_hd.astype(np.uint8), k) > 0) & ~mask_hd
    seam_prop = E.raw_dframe(recon_r, ring)
    seam_pin = E.raw_dframe(pinned, ring)
    print(f"  Q5 SEAM raw|dF| on bar-ring: propagation={seam_prop:.3f} pinned={seam_pin:.3f} "
          f"(delta {seam_pin-seam_prop:+.3f})")

    # --- visual: registered amplified diff on a bar crop, propagation vs per-frame SR ---
    save_visual(name, recon_r, perframe, mask_hd, v_hd)
    E.free_gpu()
    return res, ffr, a_bar.mean()


def save_visual(name, prop, perframe, mask_hd, v_hd, t=20):
    ys, xs = np.where(mask_hd)
    y0, x0 = ys.min(), xs.min()
    cs_h = min(160, ys.max() - y0); cs_w = 900
    crop = lambda im: im[y0:y0 + cs_h, x0:x0 + cs_w]
    M = np.float32([[1, 0, v_hd], [0, 1, 0]])

    def reg_amp(seq):
        a = crop(seq[t - 1]).astype(np.float32)
        cur = cv2.warpAffine(seq[t].astype(np.float32), M, (seq[t].shape[1], seq[t].shape[0]))
        b = crop(cur.astype(np.uint8)).astype(np.float32)
        dd = np.abs(a - b).max(axis=2)
        return cv2.applyColorMap(np.clip(dd * 12, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    panels = [
        E_label(reg_amp(prop), "propagation registered dF x12"),
        E_label(reg_amp(perframe), "per-frame SR registered dF x12"),
    ]
    cv2.imwrite(os.path.join(ART, f"{name}_regdiff.png"), np.concatenate(panels, axis=0))


def E_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def detector_fp_check():
    """Q6: bimodality (the motion-INDEPENDENT FP guard) must fire on the moving bar and stay ~0 on
    the natural talking-head face. Run on CHEAP cubic-upscaled HD (bimodality is value-distribution,
    SR-choice-robust). This shows the existing detector's FP guard survives DROPPING the motion gate
    (needed because a moving ticker has motion >> the 0.25 px gate)."""
    print("\n############### Q6 DETECTOR (bimodality-only, motion gate DROPPED) ###############")
    # moving bar clip
    rgb, h, w = E.decode_clean_rgb(START, N)
    mod, mask_lr, v = E.overlay_ticker(rgb, h, w, v_lr=2.0)
    path = os.path.join(E.TMP, "fp_bar.mp4")
    E.encode_h264(mod, path, crf=20, preset="medium", g=64, bf=2)
    frames = E.decode_mvs(path, N)
    hh, ww = frames[0][1].shape[:2]
    cov_bar = []
    for i in range(5, N, 6):
        hd = cv2.resize(frames[i][1], (ww * SCALE, hh * SCALE), interpolation=cv2.INTER_CUBIC)
        bm = gd.bimodal_score(hd)
        cov_bar.append(100 * (bm > 0.06).mean())
    # natural face window (no graphic)
    face = d.decode_lr_and_mvs(E.SAMPLE, START, N)
    cov_face = []
    for i in range(0, 28, 6):           # window-C face frames (card enters ~32)
        hd = cv2.resize(face[i][1], (ww * SCALE, hh * SCALE), interpolation=cv2.INTER_CUBIC)
        bm = gd.bimodal_score(hd)
        cov_face.append(100 * (bm > 0.06).mean())
    print(f"  moving-bar bimodal coverage %: max={max(cov_bar):.2f} (fires on the graphic)")
    print(f"  face       bimodal coverage %: max={max(cov_face):.2f} (FP guard => must stay ~0)")


def analyze_lowerthird():
    """Translucent lower-third (softer edges) sliding in then HOLDING -> static-vs-moving split,
    AND the quality-gate failure mode: temporal-mean motion is dominated by the static hold so
    a_lr~1 over the bar -> region-aware does NOT route the brief slide-in to compact (gap)."""
    print("\n############### LOWER-THIRD (slide-in then hold; translucent) ###############")
    rgb, h, w = E.decode_clean_rgb(START, N)
    mod, masks, tops = E.overlay_lowerthird(rgb, h, w)
    path = os.path.join(E.TMP, "full_lt.mp4")
    E.encode_h264(mod, path, crf=20, preset="medium", g=64, bf=2)
    frames = E.decode_mvs(path, N)
    h_lr, w_lr = frames[0][1].shape[:2]
    masks_hd = [E.upscale_mask(m, SCALE) for m in masks]
    Rr, pf = E.build_recon(frames, SCALE, sr_mode="realesrgan", occ="reactive")
    recon_r = [Rr[i]["recon"] for i in range(N)]
    perframe = [pf[i] for i in range(N)]
    a_lr, meanmag = region_gate(frames)
    region_aware = [rq.blend_region_aware(recon_r[i], perframe[i], a_lr, SCALE) for i in range(N)]
    settled_mask = masks[-1]
    print(f"  settled-bar meanmag={meanmag[settled_mask].mean():.2f} px | a_lr(bar) "
          f"mean={a_lr[settled_mask].mean():.3f}  (high => region-aware keeps HEAVY here)")

    # split: slide-in (frames 1..13) vs hold (15..N-1), raw |dF| on the per-frame bar mask
    def split_raw(seq, lo, hi):
        vals = []
        for t in range(lo, hi):
            m = masks_hd[t]
            vals.append(float(np.abs(E.luma(seq[t]) - E.luma(seq[t - 1]))[m].mean()))
        return float(np.mean(vals))
    for ph, lo, hi in [("slide-in", 2, 13), ("hold", 16, N)]:
        pf_v = split_raw(perframe, lo, hi)
        pr_v = split_raw(recon_r, lo, hi)
        ra_v = split_raw(region_aware, lo, hi)
        print(f"  {ph:9s} raw|dF| bar: per-frame SR={pf_v:.3f}  propagation={pr_v:.3f}  "
              f"region-aware={ra_v:.3f}")
    E.free_gpu()


def main():
    r1, fb1, a1 = analyze_ticker("int2", 2.0, crf=20, preset="medium", g=64, bf=2)
    r2, fb2, a2 = analyze_ticker("sub17", 1.7, crf=20, preset="medium", g=64, bf=2)
    detector_fp_check()
    analyze_lowerthird()
    print("\n==== SUMMARY ====")
    print(f"int2 : prop/perframe reg-dF={r1['propagation occ=reactive']/r1['per-frame compact SR (=pin)']:.2f}x "
          f"reactive-fb={100*fb1:.1f}% a_lr(bar)={a1:.3f}")
    print(f"sub17: prop/perframe reg-dF={r2['propagation occ=reactive']/r2['per-frame compact SR (=pin)']:.2f}x "
          f"reactive-fb={100*fb2:.1f}% a_lr(bar)={a2:.3f}")


if __name__ == "__main__":
    main()
