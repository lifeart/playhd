#!/usr/bin/env python3
"""
R13-E4 mask TEMPORAL-STABILITY diagnostic (requested by the round lead after agent
C found that this pipeline's instability is FLICKER, and that the flicker energy
concentrates in exactly the pixels whose source flips between the propagated warp
and fresh per-frame SR -- pixels chosen by the occlusion mask under test).

recall/precision are blind to this: they score a mask frame-by-frame as a static
classifier. Two masks with identical recall can behave completely differently if
one keeps a stable membership while the other's pixels blink in and out, because
each blink is a source switch (warp <-> per-frame SR) = a flicker event.

Per variant, per window, this measures:
  area              mean mask area (fallback fraction)
  flip              mean( mask[t] XOR mask[t-1] )                    raw churn
  mc_flip           mean( mask[t] XOR warp(mask[t-1]) ) over pixels covered by a
                    codec-MV chain -- the honest version: a mask that MOVES WITH
                    THE CONTENT is stable, and only the raw metric would punish it.
                    Flow fields come from R13-E3's jelly_metrics.build_mc_flows
                    (read-only reuse, identical fields for every variant).
  churn             mc_flip / area  -- membership turnover per unit of mask
  mclf              motion-compensated luma flicker of the OUTPUT (codes 0..255),
                    R13-E3's metric, computed with the same flow fields
  mclf_flip / mclf_stable
                    MCLF restricted to pixels that flipped membership vs pixels that
                    did not -- C's "flicker energy is concentrated in the flip pixels"
                    measured per variant.
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import harness as H          # noqa: E402  (also puts prototype/ + r5_e2 on the path)
import fsr2_mask as F        # noqa: E402
import derisk as D           # noqa: E402
from run_ab import build_window, run_variant     # noqa: E402  (same production rig)

_JELLY = os.path.join(os.path.dirname(_HERE), "r13_e3_jelly")
if _JELLY not in sys.path:
    sys.path.append(_JELLY)
try:
    import jelly_metrics as JM      # noqa: E402  R13-E3, READ-ONLY reuse
except Exception as e:              # loud: no silent fallback to a weaker metric
    raise RuntimeError(f"cannot import R13-E3 jelly_metrics (needed for the MC flow "
                       f"fields): {e!r}") from e


def mask_stats(masks, flows, seq, scale):
    """masks: list of HD bool (or None for anchors); seq: HD uint8 output frames."""
    areas, flips, mcflips = [], [], []
    e_flip, e_stable, mclf = [], [], []
    prev = None
    for t, m in enumerate(masks):
        if m is None:
            m = np.zeros(seq[0].shape[:2], bool)
        areas.append(float(m.mean()))
        if prev is not None:
            flips.append(float((m ^ prev).mean()))
            fl = flows[t] if t < len(flows) else None
            if fl is not None:
                pw, hole = D.warp_hd(np.repeat(prev[:, :, None], 3, axis=2).astype(np.uint8)
                                     * 255, fl["fx"], fl["fy"], scale)
                pm = pw[:, :, 0] > 127
                valid = fl["valid_hd"] & ~hole
                if valid.sum() > 1000:
                    xor = (m ^ pm)
                    mcflips.append(float(xor[valid].mean()))
                    warped, hole2 = D.warp_hd(seq[t - 1], fl["fx"], fl["fy"], scale)
                    v2 = valid & ~hole2
                    d = np.abs(JM.luma(seq[t]) - JM.luma(warped))
                    mclf.append(float(d[v2].mean()))
                    fm = v2 & xor
                    sm = v2 & ~xor
                    if fm.sum() > 100:
                        e_flip.append(float(d[fm].mean()))
                    if sm.sum() > 100:
                        e_stable.append(float(d[sm].mean()))
        prev = m
    a = float(np.mean(areas))
    mcf = float(np.mean(mcflips)) if mcflips else float("nan")
    return dict(area=a, flip=float(np.mean(flips)) if flips else float("nan"),
                mc_flip=mcf, churn=(mcf / a if a > 0 else float("nan")),
                mclf=float(np.mean(mclf)) if mclf else float("nan"),
                mclf_flip=float(np.mean(e_flip)) if e_flip else float("nan"),
                mclf_stable=float(np.mean(e_stable)) if e_stable else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="A,B,C")
    ap.add_argument("--variants", default="naive,reactive,full,adaptive,fsr2,fsr2+ruder")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--gate-px", type=float, default=10.0)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--no-region", action="store_true")
    ap.add_argument("--tag", default="maskstab")
    args = ap.parse_args()
    out = {}
    for wname in args.windows.split(","):
        W = build_window(wname, "numpy")
        W["no_region"] = bool(args.no_region)
        flows = JM.build_mc_flows(W["frames"], scale=H.SCALE)
        cov = [f["cover"] for f in flows if f]
        print(f"[{wname}] MC-flow coverage {np.mean(cov):.2f} over {len(cov)} pairs")
        out[wname] = {}
        for v in args.variants.split(","):
            seq, st, R = run_variant(W, v, thr=args.thr, gate_px=args.gate_px,
                                     dilate=bool(args.dilate))
            masks = [R[i]["mask"] for i in range(len(W["frames"]))]
            s = mask_stats(masks, flows, seq, H.SCALE)
            s.update(dict(fallback_pct=st["fallback_pct"], splat=st["splat_fires"],
                          extra_area_pct=st["extra_area_pct"]))
            out[wname][v] = s
            print(f"  [{v:11}] area {100*s['area']:6.2f}%  flip {100*s['flip']:6.2f}%  "
                  f"mc_flip {100*s['mc_flip']:6.2f}%  churn {s['churn']:5.2f}  "
                  f"MCLF {s['mclf']:6.3f}  (flip px {s['mclf_flip']:6.3f} vs "
                  f"stable {s['mclf_stable']:6.3f})")
            del seq, R
        del W
        H._free_gpu()
    H.save(out, os.path.join(H.OUT, f"{args.tag}.json"))


if __name__ == "__main__":
    main()
