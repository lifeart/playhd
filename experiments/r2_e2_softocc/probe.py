#!/usr/bin/env python3
"""R2-E2 probe: characterise window A, confirm base chain + SR/bicubic caches, baseline tOF."""
import os, sys, time
import cv2, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    if p not in sys.path:
        sys.path.insert(0, p)
import derisk as D  # noqa

CLIP = os.path.join(_REPO, "sample.mp4")
N = 48
SCALE = 2
OCC = "reactive"

def main():
    frames = D.decode_lr_and_mvs(CLIP, 0, N)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    types = "".join(f[0][0] for f in frames)
    backbone = D.backbone_indices(frames)
    first = backbone[0]
    anchors = {i for i in backbone if frames[i][0] == "I" or i == first}
    print(f"LR={w_lr}x{h_lr} HD={w_hd}x{h_hd}  types={types}")
    print(f"backbone={len(backbone)}/{N}  anchors={sorted(anchors)}")
    nB = sum(1 for f in frames if f[0]=="B")
    print(f"#I={types.count('I')} #P={types.count('P')} #B={nB}")

    bic = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC) for i in range(N)}
    t0=time.perf_counter()
    rows, R = D.reconstruct(frames, None, SCALE, True, OCC, bic, set(), backend="numpy",
                            collect_metrics=False, download_output=True)
    print(f"base chain (numpy, bicubic) {time.perf_counter()-t0:.2f}s")
    hole = {i: float(R[i]["hole_frac"]) for i in range(N)}
    na = [i for i in range(N) if i not in anchors]
    hv = [hole[i] for i in na]
    print(f"non-anchor hole_frac mean={100*np.mean(hv):.2f}% max={100*max(hv):.2f}% "
          f"#>0.08={sum(x>0.08 for x in hv)} #>0.20={sum(x>0.20 for x in hv)}")
    worst = sorted(na, key=lambda i:-hole[i])[:10]
    print("worst frames (i,type,hole%):", [(i, frames[i][0], round(100*hole[i],1)) for i in worst])
    # baseline tOF vs LR
    sm=(w_lr,h_lr)
    seq=[cv2.resize(R[i]["recon"], sm) for i in range(N)]
    lr=[frames[i][1] for i in range(N)]
    print(f"baseline(bicubic) tOF vs LR = {D.tof(seq, lr):.4f}")
    # confirm mask available + dtype
    print("mask sample (frame", worst[0], "):", R[worst[0]]["mask"].dtype, R[worst[0]]["mask"].shape,
          "true_frac=", round(float(R[worst[0]]["mask"].mean()),4))

if __name__=="__main__":
    main()
