#!/usr/bin/env python3
"""Isolate: apply softocc_patch_np and softocc_patch_torch to the IDENTICAL numpy base recon
(same masks, same SR/bic) -> if outputs match, the torch_parity gap is base/mask GPU divergence,
not a blend bug. READ-ONLY prototype import."""
import os, sys
import cv2, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    sys.path.insert(0, _p)
import derisk as D, softocc_wire as W  # noqa
import torch, gpu_ops as G             # noqa

N, SCALE, OCC, SR = 48, 2, "reactive", "realesrgan"
CLIP = os.path.join(_REPO, "sample.mp4")
frames = D.decode_lr_and_mvs(CLIP, 0, N)
h_lr, w_lr = frames[0][1].shape[:2]; w_hd, h_hd = w_lr*SCALE, h_lr*SCALE
bb = D.backbone_indices(frames); anchors = {i for i in bb if frames[i][0]=="I" or i==bb[0]}
bic = {i: cv2.resize(frames[i][1], (w_hd,h_hd), interpolation=cv2.INTER_CUBIC) for i in range(N)}
srf = D.build_perframe_cache(frames, w_hd, h_hd, SR)
_, Rbase = D.reconstruct(frames, None, SCALE, True, OCC, bic, set(), backend="numpy",
                         collect_metrics=False, download_output=True)
conf = W.build_conf(frames, anchors, bb, w_hd, h_hd)
reset = W.reset_indices(frames)

# numpy core on the numpy base
Rnp = {i: dict(recon=Rbase[i]["recon"].copy(), mask=Rbase[i]["mask"]) for i in range(N)}
W.softocc_patch_np(frames, Rnp, bic_provider=lambda i: bic[i], sr_provider=lambda i: srf[i],
                   conf=conf, anchors=anchors, reset_idx=reset, gain=0.6, beta=0.85,
                   feather_k=31, enabled=True)

# torch core on the SAME numpy base (recon -> tensor, mask -> tensor). softocc_patch_torch runs
# its OWN per-frame SR + bic internally (identical functions), so only the blend path differs.
Rt = {}
for i in range(N):
    m = Rbase[i]["mask"]
    Rt[i] = dict(recon=G.img_to_dev(Rbase[i]["recon"]),
                 mask=(torch.from_numpy(np.ascontiguousarray(m)).to(G.device()) if m is not None else None))
W.softocc_patch_torch(frames, Rt, w_hd, h_hd, SR, anchors=anchors, backbone=bb, reset_idx=reset,
                      gain=0.6, beta=0.85, feather_k=31, enabled=True)

diffs = []
for i in range(N):
    if i in anchors:
        continue
    a = Rnp[i]["recon"].astype(np.float32)
    b = G.img_to_host(Rt[i]["recon"]).astype(np.float32)
    diffs.append((i, float(np.abs(a-b).mean()), float(np.abs(a-b).max())))
mean_mae = np.mean([d[1] for d in diffs])
print(f"numpy-vs-torch blend on SAME base: mean MAE={mean_mae:.4f}  "
      f"worst-frame MAE={max(d[1] for d in diffs):.3f}  worst-px={max(d[2] for d in diffs):.1f}")
for i, mae, mx in diffs:
    if mae > 0.5:
        print(f"  frame {i:2d}: MAE={mae:.3f} maxpx={mx:.1f}")
print("VERDICT:", "BLEND MATCHES (gap is base/mask GPU divergence)" if mean_mae < 0.5
      else "BLEND DIFFERS (torch twin bug)")
