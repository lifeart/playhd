#!/usr/bin/env python3
"""Cost probe: is the recommended motion-gate (run SR only on mean|MV|>1.0 frames) FREE on
window A? Compare ungated (SR every non-anchor frame) vs motion-gated softocc. READ-ONLY."""
import os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    sys.path.insert(0, _p)
import derisk as D, softocc_wire as W, verify as V  # noqa

GATE = 1.0  # mean LR-MV magnitude (px/frame); == INSTANT_MOTION_GATE in pipeline_api


def mean_mv(frames, i):
    pt, _, mvs = frames[i]
    h_lr, w_lr = frames[0][1].shape[:2]
    if pt == "I" or mvs is None or len(mvs) == 0:
        return 0.0
    fx, fy = D.build_lr_flow(mvs, h_lr, w_lr, want="all")
    mag = np.sqrt(fx * fx + fy * fy)
    return float(np.nanmean(mag)) if np.isfinite(mag).any() else 0.0


def run(S, run_set, label):
    frames, anchors = S["frames"], S["anchors"]
    R = V.clone_recon(S["R"])
    info = W.softocc_patch_np(frames, R, bic_provider=lambda i: S["bic"][i],
                              sr_provider=lambda i: S["srf"][i], conf=S["conf"], anchors=anchors,
                              reset_idx=W.reset_indices(frames), gain=0.6, beta=0.85, feather_k=31,
                              run_set=run_set, enabled=True)
    out = {i: R[i]["recon"] for i in range(len(frames))}
    m = W.honest_metrics(frames, out, S["mask"], S["bic"], S["srf"], anchors, S["hole"], label)
    return m, info


S = V.setup(V.decode(0))
frames, anchors = S["frames"], S["anchors"]
N = len(frames)
mv = {i: mean_mv(frames, i) for i in range(N)}
high = {i for i in range(N) if i not in anchors and mv[i] > GATE}
print(f"window A: {len(high)}/{N - len(anchors)} non-anchor frames have mean|MV|>{GATE} "
      f"(mv range {min(mv.values()):.2f}..{max(mv.values()):.2f})")

m_ung, i_ung = run(S, None, "ungated (SR every non-anchor)")
m_gate, i_gate = run(S, high | anchors, "motion-gated (mean|MV|>1.0)")
for m, info in ((m_ung, i_ung), (m_gate, i_gate)):
    print(f"  {m['scheme']:34s} tOF={m['tof']:.4f}  effBic%={m['eff_bicubic_pct']:.3f}  "
          f"SR_runs(non-anchor)={info['n_sr_runs'] - len(anchors):2d}  blended={len(info['blended'])}")
