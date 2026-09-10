#!/usr/bin/env python3
"""
R13-E3 diagnostic #2: WHERE does the temporal energy come from, and is it low- or
high-frequency?  Run after run_ab.py (reuses the same baseline reconstruction).

    python diag_extra.py jelly calm

Three questions, all answered on the SHIPPING reconstruction (hard occlusion fallback):

  A. HARD-SWITCH OSCILLATION.  R[i]['mask'] is the set of pixels taken from the fresh
     per-frame SR instead of the propagated warp (P: occlusion mask; B: "neither
     direction valid").  Motion-compensate the previous frame's mask into this frame's
     grid and count the pixels whose state FLIPS.  A flipping pixel alternates between
     two different image sources frame to frame -- the textbook boiling/jelly mechanism.

  B. ATTRIBUTION.  Split the motion-compensated luma flicker into the flip set, the
     always-propagated set and the always-fallback set.  If the flip set carries flicker
     far above its area share, the hard switch IS the jelly source.

  C. FREQUENCY.  Same flicker on a low-pass (Gaussian sigma=2) and on the high-pass
     residual of the luma.  Drift-type error is low-frequency; boiling is high-frequency.
"""
import os
import sys
import json

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from common import SCALE, OUT, decode, hd_size, bicubic_seq, sr_cache  # noqa: E402
import jelly_metrics as JM   # noqa: E402
import derisk                # noqa: E402  READ-ONLY


def _lp(y, sigma=2.0):
    return cv2.GaussianBlur(y, (0, 0), sigma)


def run(name):
    frames = decode(name)
    N = len(frames)
    compact = sr_cache(name, "realesrgan")
    _, R = derisk.reconstruct(frames, None, SCALE, True, "full",
                              {i: compact[i] for i in range(N)}, set(), backend="numpy")
    seq = [R[i]["recon"] for i in range(N)]
    bic = bicubic_seq(frames)
    flows = JM.build_mc_flows(frames, SCALE)
    dist, backbone, anchors = JM.frames_since_anchor(frames)
    w_hd, h_hd = hd_size(frames)

    masks = []
    for i in range(N):
        m = R[i].get("mask")
        masks.append(np.zeros((h_hd, w_hd), bool) if m is None else np.asarray(m, bool))

    rows = []
    for t in range(1, N):
        fl = flows[t]
        if fl is None:
            continue
        prev_m, hole_m = derisk.warp_hd(masks[t - 1].astype(np.uint8) * 255,
                                        fl["fx"], fl["fy"], SCALE)
        prev_m = prev_m > 127
        wprev, hole = derisk.warp_hd(seq[t - 1], fl["fx"], fl["fy"], SCALE)
        wprev_b, _ = derisk.warp_hd(bic[t - 1], fl["fx"], fl["fy"], SCALE)
        valid = fl["valid_hd"] & ~hole
        if valid.sum() < 1000:
            continue
        cur_m = masks[t]
        flip = (cur_m != prev_m) & valid
        both_prop = (~cur_m) & (~prev_m) & valid
        both_fb = cur_m & prev_m & valid

        y_c, y_p = JM.luma(seq[t]), JM.luma(wprev)
        d = np.abs(y_c - y_p)
        # frequency split (on the SAME motion-compensated pair)
        lo = np.abs(_lp(y_c) - _lp(y_p))
        hi = np.abs((y_c - _lp(y_c)) - (y_p - _lp(y_p)))
        # content floor: the same three numbers on the bicubic(LR) sequence
        yb_c, yb_p = JM.luma(bic[t]), JM.luma(wprev_b)
        db = np.abs(yb_c - yb_p)

        def mean_on(mask_):
            return float(d[mask_].mean()) if mask_.sum() > 500 else float("nan")

        rows.append(dict(
            t=t, dist=dist[t], type=frames[t][0],
            mask_frac=float(cur_m.mean()), flip_frac=float(flip.sum() / max(valid.sum(), 1)),
            mclf=float(d[valid].mean()), mclf_bic=float(db[valid].mean()),
            mclf_lo=float(lo[valid].mean()), mclf_hi=float(hi[valid].mean()),
            mclf_flip=mean_on(flip), mclf_prop=mean_on(both_prop), mclf_fb=mean_on(both_fb),
            area_flip=float(flip.sum() / max(valid.sum(), 1)),
            area_prop=float(both_prop.sum() / max(valid.sum(), 1)),
            area_fb=float(both_fb.sum() / max(valid.sum(), 1)),
        ))
        print(f"  t={t:3d} d={dist[t]:2d} {frames[t][0]} mask={rows[-1]['mask_frac']:.3f} "
              f"flip={rows[-1]['flip_frac']:.3f} MCLF={rows[-1]['mclf']:6.2f} "
              f"(bic {rows[-1]['mclf_bic']:5.2f}) lo={rows[-1]['mclf_lo']:5.2f} "
              f"hi={rows[-1]['mclf_hi']:5.2f} | flip {rows[-1]['mclf_flip']:6.2f} "
              f"prop {rows[-1]['mclf_prop']:6.2f} fb {rows[-1]['mclf_fb']:6.2f}", flush=True)

    def col(k):
        return np.array([r[k] for r in rows], float)

    # energy attribution: share of total |dY| carried by each set
    tot = np.nansum(col("mclf_flip") * col("area_flip") + col("mclf_prop") * col("area_prop")
                    + col("mclf_fb") * col("area_fb"))
    summary = dict(
        window=name, n_pairs=len(rows),
        mask_frac=float(np.nanmean(col("mask_frac"))),
        flip_frac=float(np.nanmean(col("flip_frac"))),
        mclf=float(np.nanmean(col("mclf"))), mclf_bic=float(np.nanmean(col("mclf_bic"))),
        mclf_lo=float(np.nanmean(col("mclf_lo"))), mclf_hi=float(np.nanmean(col("mclf_hi"))),
        mclf_flip=float(np.nanmean(col("mclf_flip"))),
        mclf_prop=float(np.nanmean(col("mclf_prop"))),
        mclf_fb=float(np.nanmean(col("mclf_fb"))),
        energy_share_flip=float(np.nansum(col("mclf_flip") * col("area_flip")) / tot) if tot else float("nan"),
        energy_share_prop=float(np.nansum(col("mclf_prop") * col("area_prop")) / tot) if tot else float("nan"),
        energy_share_fb=float(np.nansum(col("mclf_fb") * col("area_fb")) / tot) if tot else float("nan"),
        drift_mclf=JM.drift_vs_flicker([r["mclf"] for r in rows], [r["dist"] for r in rows]),
        drift_flip=JM.drift_vs_flicker([r["flip_frac"] for r in rows], [r["dist"] for r in rows]),
        drift_mclf_hi=JM.drift_vs_flicker([r["mclf_hi"] for r in rows], [r["dist"] for r in rows]),
        drift_mclf_lo=JM.drift_vs_flicker([r["mclf_lo"] for r in rows], [r["dist"] for r in rows]),
        rows=rows)
    with open(os.path.join(OUT, f"diag_{name}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[{name}] mask={summary['mask_frac']:.3f} flip={summary['flip_frac']:.3f} "
          f"MCLF={summary['mclf']:.3f} (bic floor {summary['mclf_bic']:.3f}) "
          f"lo={summary['mclf_lo']:.3f} hi={summary['mclf_hi']:.3f} | "
          f"energy flip {100*summary['energy_share_flip']:.1f}% "
          f"prop {100*summary['energy_share_prop']:.1f}% fb {100*summary['energy_share_fb']:.1f}%")
    print(f"[{name}] drift(MCLF) slope={summary['drift_mclf']['slope']:+.4f}/frame "
          f"R2={summary['drift_mclf']['r2']:.3f} detrended-std={summary['drift_mclf']['detrended_std']:.3f}")
    return summary


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["jelly", "calm"]):
        run(nm)
