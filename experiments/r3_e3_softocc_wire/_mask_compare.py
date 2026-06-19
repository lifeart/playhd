#!/usr/bin/env python3
"""Measure numpy-reconstruct vs torch-reconstruct occlusion-mask divergence on window A, and how
it propagates to the soft-occ injection weight a=gain*feather(mask). READ-ONLY prototype import."""
import os, sys
import cv2, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    sys.path.insert(0, _p)
import derisk as D, softocc_wire as W, anchor_sr  # noqa
import torch  # noqa

N, SCALE, OCC, SR = 48, 2, "reactive", "realesrgan"
CLIP = os.path.join(_REPO, "sample.mp4")
frames = D.decode_lr_and_mvs(CLIP, 0, N)
h_lr, w_lr = frames[0][1].shape[:2]; w_hd, h_hd = w_lr*SCALE, h_lr*SCALE
bb = D.backbone_indices(frames); anchors = {i for i in bb if frames[i][0]=="I" or i==bb[0]}

cache, _i, _s = anchor_sr.build_anchor_cache(frames, w_hd, h_hd, SR, occ_mode=OCC, fallback_thresh=0.50, gpu_cache=False)
_, Rn = D.reconstruct(frames, None, SCALE, True, OCC, cache, set(), backend="numpy", collect_metrics=False, download_output=True)
_, Rt = D.reconstruct(frames, None, SCALE, True, OCC, cache, set(), backend="torch", collect_metrics=False, download_output=False)

def maskof(R, i, torchbk):
    m = R[i]["mask"]
    if m is None:
        return np.zeros((h_hd, w_hd), bool)
    if torchbk and torch.is_tensor(m):
        return m.to("cpu").numpy().astype(bool)
    return np.asarray(m, bool)

print(f"{'i':>3} {'typ':>3} {'np_area%':>8} {'t_area%':>8} {'IoU':>6} {'a_np_inM':>9} {'a_t_inM':>9}")
ious, dareas = [], []
for i in range(N):
    if i in anchors:
        continue
    mn = maskof(Rn, i, False); mt = maskof(Rt, i, True)
    inter = (mn & mt).sum(); union = (mn | mt).sum()
    iou = inter/union if union else 1.0
    an = W.feather(mn, 31); at = W.feather(mt, 31)
    # mean injection weight (gain*feather) inside each path's own mask (drives detail injected)
    a_np = 0.6*an[mn].mean() if mn.any() else 0.0
    a_t = 0.6*at[mt].mean() if mt.any() else 0.0
    ious.append(iou); dareas.append(abs(mn.mean()-mt.mean()))
    if frames[i][0]=="P" or mn.mean()>0.05 or iou<0.9:
        print(f"{i:>3} {frames[i][0]:>3} {100*mn.mean():>8.2f} {100*mt.mean():>8.2f} {iou:>6.3f} {a_np:>9.4f} {a_t:>9.4f}")
print(f"\nmean IoU(np,torch masks)={np.mean(ious):.3f}  mean |area diff|={100*np.mean(dareas):.3f}%")
