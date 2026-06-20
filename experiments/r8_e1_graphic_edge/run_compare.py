#!/usr/bin/env python3
"""R8-E1 core comparison: on the MOVING ticker's graphic region, is propagation shimmerier than
per-frame SR (=> a fix helps) or steadier (=> pin is a regression, like the static card)?
AND does the shipped reactive occlusion already route the edges to per-frame SR (self-healing)?

Sequences compared on the bar region (motion-compensated rDF + tOF + fallback%):
  (1) per-frame SR (the pin target / floor of fresh re-SR each frame)
  (2) propagation recon, occ=reactive  (shipped INSTANT)
  (3) propagation recon, occ=full       (shipped QUALITY-ish upper occlusion)
  (4) LR cubic                           (reference)
The PIN fix == (1) inside the bar, so (2)-vs-(1) IS the GO/NO-GO test."""
import os
import sys
import numpy as np
import cv2

import exp_common as E
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prototype"))

START, N = 5000, 48
SCALE = 4


def compare_ticker(name, v_lr, **enc):
    rgb, h, w = E.decode_clean_rgb(START, N)
    mod, mask_lr, v = E.overlay_ticker(rgb, h, w, v_lr=v_lr)
    path = os.path.join(E.TMP, f"cmp_{name}.mp4")
    E.encode_h264(mod, path, **enc)
    frames = E.decode_mvs(path, N)
    h_lr, w_lr = frames[0][1].shape[:2]
    mask_hd = E.upscale_mask(mask_lr, SCALE)
    v_hd = v * SCALE
    ref_lr = [frames[i][1] for i in range(N)]

    print(f"\n========== TICKER {name}  v_lr={v}  enc={enc} ==========")
    results = {}
    fb = {}
    for occ in ("reactive", "full"):
        R, pf = E.build_recon(frames, SCALE, sr_mode="realesrgan", occ=occ)
        recon = [R[i]["recon"] for i in range(N)]
        if occ == "reactive":             # build the shared per-frame & cubic refs once
            perframe = [pf[i] for i in range(N)]
            cubic = [cv2.resize(frames[i][1], (w_lr * SCALE, h_lr * SCALE),
                                interpolation=cv2.INTER_CUBIC) for i in range(N)]
            results["per-frame SR (=pin)"] = perframe
            results["LR cubic (ref)"] = cubic
        results[f"propagation occ={occ}"] = recon
        ff, perfr = E.fallback_frac_on_mask(R, mask_hd, range(1, N))
        fb[occ] = (ff, perfr)
        E.free_gpu()

    rows = []
    for k, seq in results.items():
        rdf = E.registered_dframe(seq, mask_hd, v_hd)
        raw = E.raw_dframe(seq, mask_hd)
        tof = E.tof_lr(seq, ref_lr)
        rows.append((k, rdf, raw, tof))
    print(f"  bar mask = {100*mask_lr.mean():.1f}% of frame; v_hd={v_hd}px")
    print(f"  {'sequence':28s} {'reg-dF':>8s} {'raw|dF|':>8s} {'tOF':>7s}")
    for k, rdf, raw, tof in rows:
        print(f"  {k:28s} {rdf:8.3f} {raw:8.3f} {tof:7.4f}")
    print(f"  -- FALLBACK% on bar (per-frame SR routed): reactive={100*fb['reactive'][0]:.1f}%  "
          f"full={100*fb['full'][0]:.1f}%   (self-healing check)")
    # verdict line
    pf_rdf = [r for r in rows if r[0].startswith("per-frame")][0][1]
    pr_rdf = [r for r in rows if "reactive" in r[0]][0][1]
    verd = "PROPAGATION STEADIER (pin = regression, NO-GO)" if pr_rdf < pf_rdf else \
           "per-frame SR steadier (pin could help)"
    print(f"  => reg-dF: propagation(reactive) {pr_rdf:.3f} vs per-frame SR {pf_rdf:.3f} "
          f"-> {verd}")
    return rows, fb


def main():
    compare_ticker("int2", 2.0, crf=20, preset="medium", g=64, bf=2)
    compare_ticker("sub17", 1.7, crf=20, preset="medium", g=64, bf=2)


if __name__ == "__main__":
    main()
