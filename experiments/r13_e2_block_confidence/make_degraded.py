#!/usr/bin/env python3
"""
R13-E2 phase 0 -- build DEGRADE-AND-RESTORE probe clips (the R5-E2 protocol).

WHY: the reliability gate's residual term `react = |LR_cur - MV_warp(LR_ref)|` is itself a
measurement of the LR warp error, so "does partition size / MV magnitude predict warp error"
is CIRCULAR if the error is measured at LR. To ask the only question that matters -- do the
free per-block signals predict the propagation error that the LR residual CANNOT see (the HD
detail warped from a geometrically wrong place) -- we need a true HD ground truth alongside
real codec MVs.

PROTOCOL (established here by R5-E2): take the decoded native-resolution frames as a pseudo-HD
GT, downscale 2x (INTER_AREA) to LR, RE-ENCODE the LR with libx264 (real B-frames, real RD
motion search), decode it back with +export_mvs. Now we have: real codec MVs at LR, a real LR
residual, and an exact 2x GT for every frame -> the HD warp error is directly measurable.

Writes experiments/r13_e2_block_confidence/out/probe/<name>_lr.mp4 (the encoded LR) and
<name>_gt.npy (uint8 N x H x W x 3 pseudo-HD GT, display order, same frame count/order).
"""
import os
import sys
import json

import av
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out", "probe")
os.makedirs(OUT, exist_ok=True)

SAMPLE = os.path.join(ROOT, "sample.mp4")
RCLIPS = os.path.join(ROOT, "experiments", "r3_e2_robustness", "clips")

# (name, source path, start frame, n frames)  -- the project's canonical windows A/B/C of
# sample.mp4 plus the R3-E2 robustness clip set (authored real-H.264 content regimes).
PROBES = [
    ("A_highmotion", SAMPLE, 0, 48),
    ("B_mid", SAMPLE, 2500, 48),
    ("C_talkinghead", SAMPLE, 5032, 48),
    ("D_jelly", SAMPLE, 12000, 48),
    ("c1_fastpan", os.path.join(RCLIPS, "c1_fastpan.mp4"), 0, 40),
    ("c2_graphics", os.path.join(RCLIPS, "c2_graphics.mp4"), 0, 40),
    ("c3_lowlight", os.path.join(RCLIPS, "c3_lowlight.mp4"), 0, 40),
    ("c4_talkinghead", os.path.join(RCLIPS, "c4_talkinghead.mp4"), 0, 40),
]


def decode_rgb(path, start, n):
    """Decode n RGB frames in DISPLAY order starting at display index `start`."""
    cont = av.open(path)
    st = cont.streams.video[0]
    st.thread_type = "AUTO"
    out = []
    idx = 0
    for packet in cont.demux(st):
        for f in packet.decode():
            if idx >= start:
                out.append(f.to_ndarray(format="rgb24"))
            idx += 1
            if len(out) >= n:
                cont.close()
                return out
    cont.close()
    return out


def encode_lr(path, lr_frames, fps=25, crf=20):
    cont = av.open(path, "w")
    vst = cont.add_stream("libx264", rate=fps)
    h, w = lr_frames[0].shape[:2]
    vst.width, vst.height, vst.pix_fmt = w, h, "yuv420p"
    vst.options = {"crf": str(crf)}          # default preset => real B-frames, real RD search
    for fr in lr_frames:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24")
        for p in vst.encode(vf):
            cont.mux(p)
    for p in vst.encode():
        cont.mux(p)
    cont.close()


def main():
    manifest = []
    for name, src, start, n in PROBES:
        if not os.path.exists(src):
            raise FileNotFoundError(f"probe source missing: {src}")
        gt = decode_rgb(src, start, n)
        if len(gt) < n:
            raise RuntimeError(f"{name}: decoded only {len(gt)}/{n} frames from {src}")
        h, w = gt[0].shape[:2]
        # even LR dims (yuv420p)
        w_lr, h_lr = (w // 2) // 2 * 2, (h // 2) // 2 * 2
        gt = [cv2.resize(f, (w_lr * 2, h_lr * 2), interpolation=cv2.INTER_AREA)
              if (w, h) != (w_lr * 2, h_lr * 2) else f for f in gt]
        lr = [cv2.resize(f, (w_lr, h_lr), interpolation=cv2.INTER_AREA) for f in gt]
        lr_path = os.path.join(OUT, f"{name}_lr.mp4")
        encode_lr(lr_path, lr)
        np.save(os.path.join(OUT, f"{name}_gt.npy"), np.stack(gt).astype(np.uint8))
        manifest.append(dict(name=name, src=src, start=start, n=n,
                             gt_hw=[h_lr * 2, w_lr * 2], lr_hw=[h_lr, w_lr], lr=lr_path))
        print(f"[{name}] gt {h_lr*2}x{w_lr*2} lr {h_lr}x{w_lr} n={n} -> {lr_path}")
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print(f"wrote manifest with {len(manifest)} probes")


if __name__ == "__main__":
    sys.exit(main())
