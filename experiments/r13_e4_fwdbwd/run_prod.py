#!/usr/bin/env python3
"""
R13-E4 production-RESOLUTION arm: the real clip, the real scale, the shipped
quality-mode config -- no degrade-restore, therefore no GT, therefore only the
project's honest no-GT decision metrics (GOTCHA #13): tOF + fallback%.

  input   : sample.mp4 LR 640x320 with its OWN codec MVs (no re-encode)
  restore : derisk.reconstruct, scale 4 -> 2560x1280, x4plus fp16 anchors,
            anchor-blend beta=0.85, region-aware output blend, torch backend
            == server MODE_CONFIG["quality"] minus grain (grain is a final
            output pass, identical across arms, and would only add noise here)
  metrics : tOF(prop vs decoded-LR) [lower = steadier], fallback%, ms/frame

This complements the GT-bearing tier-1 rig (harness.py), whose LR is 320x160.
If the two disagree, the report says so.
"""
import os
import sys
import time
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import harness as H          # noqa: E402
import fsr2_mask as F        # noqa: E402
import derisk as D           # noqa: E402

SCALE = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="A,C")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--variants", default="reactive,full,adaptive,fsr2,fsr2+ruder")
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--gate-px", type=float, default=20.0)
    ap.add_argument("--nan-prev", default="flag")
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--tag", default="prod")
    args = ap.parse_args()

    out = {}
    for wname in args.windows.split(","):
        start = H.WINDOWS[wname]["start"]
        frames = D.decode_lr_and_mvs(H.SAMPLE, start_frame=start, max_frames=args.n)
        h_lr, w_lr = frames[0][1].shape[:2]
        w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
        print(f"[{wname}] start={start} n={len(frames)} LR {w_lr}x{h_lr} -> {w_hd}x{h_hd} "
              f"{''.join(f[0] for f in frames)}")
        heavy = D.build_perframe_cache(frames, w_hd, h_hd, "realesrgan-x4plus", half=True)
        H._free_gpu()
        compact = D.build_perframe_cache(frames, w_hd, h_hd, "realesrgan")
        H._free_gpu()
        rg = D._build_region_gate(frames, w_hd, h_hd, SCALE, compact_cache=compact)
        cache = D.blend_anchor_cache({k: v.copy() for k, v in heavy.items()}, compact, 0.85)
        del heavy
        res = {}
        for v in args.variants.split(","):
            F.patch()
            F.set_variant(v if v in F._BASE_MODE else "reactive", scale=SCALE,
                          thr=args.thr, gate_px=args.gate_px, dilate=bool(args.dilate),
                          nan_prev=args.nan_prev)
            occ = "naive" if v == "naive" else F.occ_mode_for(v)
            D.MASK_FIRES[0] = D.MASK_FIRES[1] = 0
            t0 = time.perf_counter()
            rows, R = D.reconstruct(frames, None, SCALE, True, occ, cache, set(),
                                    backend=args.backend, collect_metrics=True,
                                    region_gate=rg)
            ms = (time.perf_counter() - t0) * 1000.0 / len(frames)
            t = D._tof_from_R(frames, None, R, w_lr, h_lr)
            holes = [float(np.mean(R[i]["mask"])) if R[i]["mask"] is not None else 0.0
                     for i in range(len(frames))]
            res[v] = dict(tof_prop_vs_lr=t["prop_vs_lr"],
                          tof_prop_vs_perframe=t["prop_vs_perframe"],
                          tof_perframe_vs_lr=t["perframe_vs_lr"],
                          fallback_pct=100.0 * float(np.mean(holes)),
                          fallback_max_pct=100.0 * float(np.max(holes)),
                          splat_fires=f"{D.MASK_FIRES[0]}/{D.MASK_FIRES[1]}",
                          extra_area_pct=100.0 * (float(np.mean(F.STATS["fsr2_area"]))
                                                  if F.STATS["fsr2_area"] else 0.0),
                          ms_per_frame=ms)
            print(f"  [{v:11}] tOF(prop|LR) {res[v]['tof_prop_vs_lr']:.4f}  "
                  f"fb {res[v]['fallback_pct']:.2f}%  splat {res[v]['splat_fires']}  "
                  f"extra {res[v]['extra_area_pct']:.2f}%  {ms:.0f} ms/frame")
            del R, rows
            H._free_gpu()
        out[wname] = res
        del cache, compact, rg, frames
        H._free_gpu()
    H.save(dict(config=vars(args), results=out), os.path.join(H.OUT, f"{args.tag}.json"))


if __name__ == "__main__":
    main()
