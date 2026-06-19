#!/usr/bin/env python3
"""Definitive: deployed cache base. Path A = numpy reconstruct + softocc_patch_np (== Task 2c).
Path B = torch reconstruct + softocc_patch_torch (== torch_parity). Compare per-frame inside M."""
import os, sys
import cv2, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    sys.path.insert(0, _p)
import derisk as D, softocc_wire as W, anchor_sr  # noqa
import torch, gpu_ops as G  # noqa

N, SCALE, OCC, SR = 48, 2, "reactive", "realesrgan"
CLIP = os.path.join(_REPO, "sample.mp4")
frames = D.decode_lr_and_mvs(CLIP, 0, N)
h_lr, w_lr = frames[0][1].shape[:2]; w_hd, h_hd = w_lr*SCALE, h_lr*SCALE
bb = D.backbone_indices(frames); anchors = {i for i in bb if frames[i][0]=="I" or i==bb[0]}
bic = {i: cv2.resize(frames[i][1], (w_hd,h_hd), interpolation=cv2.INTER_CUBIC) for i in range(N)}
srf = D.build_perframe_cache(frames, w_hd, h_hd, SR)
conf = W.build_conf(frames, anchors, bb, w_hd, h_hd)
reset = W.reset_indices(frames)
cache, _i, _s = anchor_sr.build_anchor_cache(frames, w_hd, h_hd, SR, occ_mode=OCC, fallback_thresh=0.50, gpu_cache=False)

# Path A: numpy
_, Rn = D.reconstruct(frames, None, SCALE, True, OCC, cache, set(), backend="numpy", collect_metrics=False, download_output=True)
Rna = {i: dict(recon=Rn[i]["recon"].copy(), mask=Rn[i]["mask"]) for i in range(N)}
W.softocc_patch_np(frames, Rna, bic_provider=lambda i: bic[i], sr_provider=lambda i: srf[i], conf=conf,
                   anchors=anchors, reset_idx=reset, gain=0.6, beta=0.85, feather_k=31, enabled=True)

# Path B: torch
_, Rt = D.reconstruct(frames, None, SCALE, True, OCC, cache, set(), backend="torch", collect_metrics=False, download_output=False)
Rtb = {i: dict(recon=Rt[i]["recon"].clone(), mask=Rt[i]["mask"]) for i in range(N)}
W.softocc_patch_torch(frames, Rtb, w_hd, h_hd, SR, anchors=anchors, backbone=bb, reset_idx=reset,
                      gain=0.6, beta=0.85, feather_k=31, enabled=True)

def maskof(R, i, tb):
    m = R[i]["mask"]
    if m is None: return np.zeros((h_hd, w_hd), bool)
    return m.to("cpu").numpy().astype(bool) if (tb and torch.is_tensor(m)) else np.asarray(m, bool)

print(f"{'i':>3} {'inM_MAE':>8} {'detA':>7} {'detB':>7}  (detail = mean r inside M)")
for i in range(N):
    if i in anchors: continue
    mn = maskof(Rn, i, False)
    if not mn.any(): continue
    oA = Rna[i]["recon"].astype(np.float32)
    oB = G.img_to_host(Rtb[i]["recon"]).astype(np.float32)
    b = bic[i].astype(np.float32); sr = srf[i].astype(np.float32)
    rA = np.clip(np.linalg.norm((oA-b)[mn],axis=1)/(np.linalg.norm((sr-b)[mn],axis=1)+1e-3),0,1).mean()
    rB = np.clip(np.linalg.norm((oB-b)[mn],axis=1)/(np.linalg.norm((sr-b)[mn],axis=1)+1e-3),0,1).mean()
    mae = np.abs(oA-oB)[mn].mean()
    if mae > 0.5 or abs(rA-rB) > 0.02:
        print(f"{i:>3} {mae:>8.3f} {rA:>7.3f} {rB:>7.3f}")
# overall
outA = {i: Rna[i]["recon"] for i in range(N)}
outB = {i: G.img_to_host(Rtb[i]["recon"]) for i in range(N)}
mae_all = np.mean([np.abs(outA[i].astype(np.float32)-outB[i].astype(np.float32)).mean()
                   for i in range(N) if i not in anchors])
seqA = [cv2.resize(outA[i], (w_lr, h_lr)) for i in range(N)]
seqB = [cv2.resize(outB[i], (w_lr, h_lr)) for i in range(N)]
lr = [frames[i][1] for i in range(N)]
print(f"\noverall full-frame mean MAE (pathA vs pathB) = {mae_all:.3f}")
print(f"tOF  pathA(numpy+softocc_np)={D.tof(seqA, lr):.4f}   pathB(torch+softocc_torch)={D.tof(seqB, lr):.4f}")
# eff-bic / detail aggregated (R2-E2), using numpy masks for both (apples-to-apples)
def aggr(out):
    na=[i for i in range(N) if i not in anchors]; HW=h_hd*w_hd; eb=[]; dt=[]
    for i in na:
        m=maskof(Rn,i,False)
        if not m.any(): eb.append(0); dt.append(0); continue
        b=bic[i].astype(np.float32); sr=srf[i].astype(np.float32); o=out[i].astype(np.float32)
        r=np.clip(np.linalg.norm((o-b)[m],axis=1)/(np.linalg.norm((sr-b)[m],axis=1)+1e-3),0,1)
        eb.append((1-r).sum()/HW); dt.append(r.sum()/HW)
    return 100*np.mean(eb), 100*np.mean(dt)
ebA,dtA=aggr(outA); ebB,dtB=aggr(outB)
print(f"eff-bic% pathA={ebA:.3f} (detail {dtA:.3f})   pathB={ebB:.3f} (detail {dtB:.3f})  [both vs numpy masks]")
