#!/usr/bin/env python3
"""Probe: confirm window-A structure + that the all-bicubic numpy reconstruct base reproduces
R2-E2's anchors {0,28} / non-anchor hole mean 7.70%. READ-ONLY import of prototype/."""
import os, sys
import numpy as np, cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "prototype"))
import derisk as D  # noqa: E402

CLIP = os.path.join(_REPO, "sample.mp4")
N, SCALE, OCC = 48, 2, "reactive"


def probe(start):
    frames = D.decode_lr_and_mvs(CLIP, start, N)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    backbone = D.backbone_indices(frames)
    anchors = {i for i in backbone if frames[i][0] == "I" or i == backbone[0]}
    iframes = {i for i in range(N) if frames[i][0] == "I"}
    types = "".join(frames[i][0] for i in range(N))
    bic = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC) for i in range(N)}
    _, R = D.reconstruct(frames, None, SCALE, True, OCC, bic, set(), backend="numpy",
                         collect_metrics=False, download_output=True)
    na = [i for i in range(N) if i not in anchors]
    hole = {i: float(R[i]["hole_frac"]) for i in range(N)}
    print(f"start={start} LR={w_lr}x{h_lr} HD={w_hd}x{h_hd}")
    print(f"  types        = {types}")
    print(f"  backbone[0]  = {backbone[0]}  anchors={sorted(anchors)}  iframes={sorted(iframes)}")
    print(f"  non-anchor hole mean = {100*np.mean([hole[i] for i in na]):.2f}%  "
          f"max={100*max(hole.values()):.1f}%  #>0.20={sum(hole[i]>0.20 for i in na)}")
    return frames, anchors, iframes


if __name__ == "__main__":
    print("=== window A (start 0) ===")
    probe(0)
    print("=== talking-head (start 5000) ===")
    probe(5000)
