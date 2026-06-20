#!/usr/bin/env python3
"""R8-E1 probe: confirm the re-encoded moving caption gets NON-ZERO codec MVs that track the
text, and measure the cheap reactive-fallback fraction on the graphic region (no SR needed)."""
import os
import numpy as np
import cv2

import exp_common as E
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prototype"))
import derisk as d
import region_quality as rq

START, N = 5000, 48
SCALE = 4


def reactive_frac_lr(frames, mask_lr, tau=16.0):
    """Per P-frame, the reactive-occlusion fraction inside mask_lr: |LR_cur - warp(LR_prev)|>tau
    OR intra-hole. This is exactly the (a)+(b) signal the shipped reactive mask uses (LR, before
    HD resize). Anchors excluded (they are fresh SR). Returns list per frame."""
    bb = d.backbone_indices(frames)
    prev_ip = {}
    for k, b in enumerate(bb):
        prev_ip[b] = bb[k - 1] if k > 0 else None
    out = []
    for i in range(len(frames)):
        pt, lr, mvs = frames[i]
        if pt != "P":
            out.append(None); continue
        p = prev_ip.get(i)
        if p is None:
            out.append(None); continue
        fx, fy = d.build_lr_flow(mvs, *lr.shape[:2], want="past")
        pred = d.warp_lr(frames[p][1], fx, fy).astype(np.float32)
        react = np.abs(lr.astype(np.float32) - pred).mean(axis=2)
        occ = (~np.isfinite(fx)) | (react > tau)
        out.append(float(occ[mask_lr].mean()))
    return out


def run_variant(name, mod, mask_lr, v_lr, **enc):
    path = os.path.join(E.TMP, f"{name}.mp4")
    E.encode_h264(mod, path, **enc)
    frames = E.decode_mvs(path, N)
    types = "".join(f[0] for f in frames)
    h_lr, w_lr = frames[0][1].shape[:2]
    print(f"\n=== {name}  enc={enc}  v_lr={v_lr} ===")
    print(f"  GOP types: {types}")
    mx, nbad = d.scan_source_magnitude(frames)
    print(f"  max|source|={mx} nbad={nbad}")
    # MV magnitude INSIDE the graphic bar vs OUTSIDE, on P frames
    inside, outside = [], []
    for i in range(len(frames)):
        if frames[i][0] != "P":
            continue
        mag, nomv = rq.motion_mag_lr(frames[i][2], h_lr, w_lr, want="all")
        mag = np.where(nomv, np.nan, mag)
        mi = np.nanmean(mag[mask_lr]); mo = np.nanmean(mag[~mask_lr])
        inside.append(mi); outside.append(mo)
    print(f"  P-frame mean|MV| INSIDE bar = {np.nanmean(inside):.2f} px/frame   "
          f"OUTSIDE = {np.nanmean(outside):.2f}  (authored v_lr={v_lr})")
    nomv_in = []
    for i in range(len(frames)):
        if frames[i][0] != "P":
            continue
        _, nomv = rq.motion_mag_lr(frames[i][2], h_lr, w_lr, want="all")
        nomv_in.append(float(nomv[mask_lr].mean()))
    print(f"  P-frame intra/no-MV frac INSIDE bar = {np.nanmean(nomv_in)*100:.1f}%")
    rf = reactive_frac_lr(frames, mask_lr)
    rfv = [x for x in rf if x is not None]
    print(f"  REACTIVE fallback frac INSIDE bar (P frames) = {np.nanmean(rfv)*100:.1f}% "
          f"(min {np.nanmin(rfv)*100:.1f} / max {np.nanmax(rfv)*100:.1f})  <-- self-healing check")
    return frames


def main():
    rgb, h, w = E.decode_clean_rgb(START, N)
    print(f"clean bg window start={START} n={N} LR {w}x{h}")

    # Variant T-int: integer 2px/frame ticker (encoder should find exact MV)
    modA, maskA, vA = E.overlay_ticker(rgb, h, w, v_lr=2.0)
    run_variant("ticker_int2", modA, maskA, vA, crf=20, preset="medium", g=64, bf=2)

    # Variant T-sub: sub-pixel 1.7px/frame ticker (forces quarter-pel rounding -> imperfect MV)
    modB, maskB, vB = E.overlay_ticker(rgb, h, w, v_lr=1.7)
    run_variant("ticker_sub17", modB, maskB, vB, crf=20, preset="medium", g=64, bf=2)

    # Variant T-fast: faster, fast preset (coarser RD MV search)
    modC, maskC, vC = E.overlay_ticker(rgb, h, w, v_lr=3.3)
    run_variant("ticker_fast33", modC, maskC, vC, crf=23, preset="fast", g=64, bf=2)

    # Variant LT: translucent lower-third over moving video (slide-in then hold)
    modD, masksD, topsD = E.overlay_lowerthird(rgb, h, w)
    # use the settled-phase bar region as the static mask for the probe
    maskD = masksD[-1]
    run_variant("lowerthird", modD, maskD, "slide", crf=20, preset="medium", g=64, bf=2)


if __name__ == "__main__":
    main()
