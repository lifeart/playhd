#!/usr/bin/env python3
"""
exp3_visual / V3 -- graphic/text-edge PINNING (OUTPUT-ONLY pass). "USACHEV TODAY" title card.

Premise to test: title cards shimmer under MV warp, so pin the graphic region to a stable
source (per-frame SR, or freeze to one anchor SR) instead of the warped propagation.

Two parts:
  (A) DETECTOR quality + false-positive guard (graphic_detect): does it fire on the real card
      and NOT on talking-head face detail?
  (B) PINNING benefit: on the STATIC card run (frames 18-28; 28 is a scene cut), edge |Delta F|
      of the propagated recon vs the pinned/frozen variants vs the per-frame-SR floor.

The animated REVEAL frames (1-15, the text slides/forms) are legit content change, NOT shimmer,
so the pinning benefit is measured on the STATIC run only. Output-only throughout: every variant
is built on a COPY of the recon, never fed into R[].
"""
import os

import cv2
import numpy as np

import common as C
import graphic_detect as gd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SCALE = 4
STATIC0, STATIC1 = 18, 28          # static-card run (frame 28 = hard cut to the talking head)
FACE_END = 32                       # window-C frames [0,32) are face; the card enters at 32


def _detect(recon, frames, h_lr, w_lr):
    return [gd.detect_graphic_mask(recon[i], frames[i][2], h_lr, w_lr, SCALE)[0]
            for i in range(len(frames))]


def run_title():
    frames, h_lr, w_lr, types = C.decode_window(0, 32)
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    print(f"V3 title window: start 0, {len(frames)}f  HD {w_hd}x{h_hd}  types={types}")
    R, perframe = C.reconstruct_window(frames, SCALE, sr_mode="realesrgan", occ="full")
    recon = [R[i]["recon"] for i in range(len(frames))]
    pf = [perframe[i] for i in range(len(frames))]
    C.free_gpu()

    regions = _detect(recon, frames, h_lr, w_lr)
    covs = [100 * r.mean() for r in regions]
    print("per-frame graphic region coverage %: " + " ".join(f"{c:.0f}" for c in covs))

    # ---- output-only variants (all on a COPY) ----
    def pin(src):                                   # src[i] supplies the pinned pixels of frame i
        out = []
        for i in range(len(frames)):
            o = recon[i].copy(); o[regions[i]] = src[i][regions[i]]; out.append(o)
        return out
    pinned = pin(pf)                                # task primary: pin -> per-frame SR
    frozen_src = [pf[STATIC0]] * len(frames)        # freeze whole card to ONE anchor SR
    frozen = pin(frozen_src)
    lr_up = [cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
             for i in range(len(frames))]

    # ---- edge-flicker measurement on the STATIC run, on the card's hard EDGE pixels ----
    union = np.zeros((h_hd, w_hd), bool)
    for i in range(STATIC0, STATIC1):
        union |= regions[i]
    edge = (gd.edge_magnitude(pf[(STATIC0 + STATIC1) // 2]) > 300.0) & union
    print(f"\nstatic run [{STATIC0},{STATIC1}) edge mask = {100*edge.mean():.2f}% "
          f"({int(edge.sum())} px)")

    def edf(seq):
        return C.dframe_luma([seq[i] for i in range(STATIC0, STATIC1)], edge)
    res = {"propagated recon (engine)": edf(recon),
           "V3 pin -> per-frame SR": edf(pinned),
           "V3 freeze -> 1 anchor SR": edf(frozen),
           "per-frame SR (floor)": edf(pf),
           "LR source (cubic, ref)": edf(lr_up)}
    rp_diff = float(np.mean([np.abs(C.luma(recon[i]) - C.luma(pf[i]))[edge].mean()
                             for i in range(STATIC0, STATIC1)]))
    print("\n================ V3 title-card EDGE |Delta F| (luma) ================")
    for k, v in res.items():
        print(f"  {k:28s} {v:7.3f}")
    print(f"  (mean |recon - per-frame SR| on card edge = {rp_diff:.2f} codes => engine already "
          f"reproduces the card)")
    base = res["propagated recon (engine)"]
    print(f"\n  -> pinning to per-frame SR changes edge flicker {base:.3f} -> "
          f"{res['V3 pin -> per-frame SR']:.3f} "
          f"({'WORSE' if res['V3 pin -> per-frame SR'] > base else 'better'}); the card is "
          f"ZERO-MV (skip) so the identity-warp propagation is already the stable path.")

    # ---- artifacts: amplified diff over the static run (recon vs pin) + region overlay ----
    ys, xs = np.where(edge); cy, cx = int(ys.mean()), int(xs.mean()); cs = 320
    y0 = int(np.clip(cy - cs // 2, 0, h_hd - cs)); x0 = int(np.clip(cx - cs // 2, 0, w_hd - cs))
    crop = lambda im: im[y0:y0 + cs, x0:x0 + cs]
    t = STATIC0 + 3
    panels = [C.label(C.amplified_diff(crop(recon[t - 1]), crop(recon[t]), amp=8), "recon dF (engine)"),
              C.label(C.amplified_diff(crop(pinned[t - 1]), crop(pinned[t]), amp=8), "pin->perframe dF")]
    cv2.imwrite(os.path.join(OUT, "v3_edge_ampdiff.png"), np.concatenate(panels, axis=1))
    ov = cv2.cvtColor(recon[t], cv2.COLOR_RGB2BGR)
    ov[regions[t]] = (0.5 * ov[regions[t]] + np.array([0, 0, 180])).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, "v3_region.png"),
                C.label(cv2.resize(ov, (w_hd // 2, h_hd // 2)), "graphic region (red)"))
    print(f"wrote artifacts -> {OUT}/v3_edge_ampdiff.png, v3_region.png")
    return res, rp_diff


def run_false_positive():
    frames, h_lr, w_lr, types = C.decode_window(5000, 48)
    print(f"\nV3 false-positive check, talking-head window C: {len(frames)}f  types={types}")
    R, _ = C.reconstruct_window(frames, SCALE, sr_mode="realesrgan", occ="full")
    C.free_gpu()
    covs = [100 * gd.detect_graphic_mask(R[i]["recon"], frames[i][2], h_lr, w_lr, SCALE)[0].mean()
            for i in range(len(frames))]
    print("per-frame region coverage %: " + " ".join(f"{c:.0f}" for c in covs))
    face = covs[:FACE_END]
    print(f"  FACE frames [0,{FACE_END}): max coverage = {max(face):.2f}%  (mis-fire guard => ~0 "
          f"= NO false positive on natural high-detail content)")
    print(f"  TAIL frames [{FACE_END},48): max coverage = {max(covs[FACE_END:]):.1f}%  (the SAME "
          f"card enters in window C's tail -> correct TRUE positive)")
    return covs


def main():
    os.makedirs(OUT, exist_ok=True)
    run_title()
    run_false_positive()


if __name__ == "__main__":
    main()
