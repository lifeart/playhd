#!/usr/bin/env python3
"""
R13-E1 scoring: hard binary occlusion switch vs soft confidence gate, on the frames the
headless browser harness actually rendered (harness.ts dumps raw RGBA HD frames).

REFERENCE (the ceiling the propagation approximates) = the SAME browser pipeline with
`--mode ref` (SR every frame, no propagation at all). Using the pipeline's own pure-SR
output as the arbiter keeps the comparison inside the browser's world: both arms differ
from it ONLY in how propagated frames are filled.

Metrics (repo convention, R10/R12: LPIPS/DISTS are the arbiters, tOF is the flicker axis):
  PSNR / LPIPS / DISTS vs the pure-SR reference   (lower LPIPS/DISTS = closer to the ceiling)
  tOF(candidate, bicubic-LR)                      (flow deviation from TRUE content motion)
  dF                                              (raw temporal energy; context)
Scored twice: over ALL frames, and over PROPAGATED frames only (anchors are bit-identical
between the arms, so the all-frames number is diluted -- the warp-only number is the signal).

Run:  python3 eval.py --scratch <dir> --window calm --variants hard4,soft_t16s3,...
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r5_e2_quality"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r8_e4_metric_triangulation"))
import metrics as M            # noqa: E402  psnr / lpips_dist / tof
import metrics_extra as ME     # noqa: E402  dists (pyiqa)

METRIC_W = 512                 # LPIPS/DISTS width (R12-E1 used 768; 512 halves CPU cost and
                               # only rescales the absolute distances -- the A/B ordering is unchanged)
TOF_W = 640                    # tOF Farneback at the LR width


def load_seq(d, n, wd, hd):
    out = []
    for i in range(n):
        a = np.fromfile(os.path.join(d, f"hd_{i}.bin"), dtype=np.uint8)
        out.append(a.reshape(hd, wd, 4)[:, :, :3].copy())
    return out


def _rs(img, width, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(round(h * width / w))), interpolation=interp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric-w", type=int, default=METRIC_W)
    ap.add_argument("--skip-dists", action="store_true")
    ap.add_argument("--skip-lpips", action="store_true")   # PSNR+tOF only: a fast screen for parameter sweeps
    args = ap.parse_args()
    mw = args.metric_w

    data = os.path.join(args.scratch, args.window)
    meta = json.load(open(os.path.join(data, "meta.json")))
    n, w, h = meta["n"], meta["w"], meta["h"]
    refdir = os.path.join(args.scratch, f"out_{args.window}_ref")
    run = json.load(open(os.path.join(refdir, "run.json")))
    wd, hd = [int(x) for x in run["res"].split("->")[1].split("x")]
    ref = load_seq(refdir, n, wd, hd)

    # LR content sequence (the tOF anchor: the true motion the codec encoded)
    lr = [np.fromfile(os.path.join(data, f"rgb_{i}.bin"), dtype=np.uint8).reshape(h, w, 3) for i in range(n)]
    lr_small = [_rs(x, TOF_W) for x in lr]
    ref_small = [_rs(x, TOF_W) for x in ref]
    ref_m = [_rs(x, mw) for x in ref]

    rows = []
    for name in args.variants.split(","):
        d = os.path.join(args.scratch, f"out_{args.window}_{name}")
        rj = json.load(open(os.path.join(d, "run.json")))
        print(f"  scoring {name}...", flush=True)
        seq = load_seq(d, n, wd, hd)
        kinds = rj["kinds"]
        warp_idx = [i for i, k in enumerate(kinds) if k == "W"]
        per = []
        for i in range(n):
            c_m = _rs(seq[i], mw)
            per.append(dict(
                i=i, kind=kinds[i],
                psnr=M.psnr(seq[i], ref[i]),
                lpips=(float("nan") if args.skip_lpips else M.lpips_dist(c_m, ref_m[i])),
                dists=(float("nan") if args.skip_dists else ME.dists(c_m, ref_m[i])),
            ))
        seq_small = [_rs(x, TOF_W) for x in seq]
        tof_lr = M.tof(seq_small, lr_small)
        tof_ref = M.tof(seq_small, ref_small)
        dF = float(np.mean([np.abs(seq_small[t].astype(np.float32) - seq_small[t - 1].astype(np.float32)).mean()
                            for t in range(1, n)]))

        def agg(key, idx=None):
            v = [p[key] for p in per] if idx is None else [per[i][key] for i in idx]
            return float(np.mean(v))

        rows.append(dict(
            variant=name, gate=rj["gate"], reanchor=rj["reanchor"],
            ms_warp_median=rj["ms_warp_median"], ms_sr_median=rj["ms_sr_median"],
            gate_mean_a=rj.get("gate_mean_a"), gate_frac_a_lt_05=rj.get("gate_frac_a_lt_05"),
            gate_frac_a_gt_95=rj.get("gate_frac_a_gt_95"),
            psnr_all=round(agg("psnr"), 3), psnr_warp=round(agg("psnr", warp_idx), 3),
            lpips_all=round(agg("lpips"), 5), lpips_warp=round(agg("lpips", warp_idx), 5),
            dists_all=round(agg("dists"), 5), dists_warp=round(agg("dists", warp_idx), 5),
            tof_lr=round(tof_lr, 4), tof_ref=round(tof_ref, 4), dF=round(dF, 4),
        ))
        print(json.dumps(rows[-1]), flush=True)

    out = args.out or os.path.join(_HERE, f"results_{args.window}.json")
    json.dump(dict(window=args.window, frames=n, res=f"{wd}x{hd}", metric_w=mw, rows=rows), open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
