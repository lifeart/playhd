#!/usr/bin/env python3
"""Build side-by-side comparison PNGs + a reliability-gate heatmap for a window.
Reads out/frames_<name>.npz (saved by run_gate_ab.py) and re-derives the gate alpha
for the saved frame by re-decoding the window (cheap). Saves to out/."""
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))
sys.path.insert(0, os.path.join(_HERE, "patch_src"))
import derisk            # noqa
import gated_recon as gr  # noqa

CLIP = os.path.join(_ROOT, "web_spike", "sd600.mp4")
WINDOWS = {"jelly": dict(start=582, n=40), "calm": dict(start=437, n=40)}
SCALE = 3


def label(img, txt):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def row(imgs, labels, wide=560):
    out = []
    for im, lb in zip(imgs, labels):
        h, w = im.shape[:2]
        r = cv2.resize(im, (wide, int(round(h * wide / w))), interpolation=cv2.INTER_AREA)
        out.append(label(r, lb))
    return np.concatenate(out, axis=1)


def make(name):
    z = np.load(os.path.join(_HERE, "out", f"frames_{name}.npz"))
    idx = int(z["idx"])
    panels = [z["bic"], z["compact"], z["baseline"], z["gate"], z["gate_fb"], z["x4"]]
    labels = ["bicubic-LR", "compact/frame", "baseline HARD",
              "gate res t10", "gate resfb t10", "x4plus ref"]
    full = row([cv2.cvtColor(p, cv2.COLOR_RGB2BGR) for p in panels], labels, wide=560)

    # zoom crop: pick the 220px region with max baseline-vs-x4 error (where drift shows)
    err = np.abs(z["baseline"].astype(np.float32) - z["x4"].astype(np.float32)).mean(2)
    H, W = err.shape
    ks = 220
    ii = cv2.boxFilter(err, -1, (ks, ks))
    cy, cx = np.unravel_index(np.argmax(ii), ii.shape)
    y0 = int(np.clip(cy - ks // 2, 0, H - ks)); x0 = int(np.clip(cx - ks // 2, 0, W - ks))
    crops = [p[y0:y0 + ks, x0:x0 + ks] for p in panels]
    zoom = row([cv2.cvtColor(c, cv2.COLOR_RGB2BGR) for c in crops], labels, wide=560)

    stack = np.concatenate([full, zoom], axis=0)
    cv2.imwrite(os.path.join(_HERE, "out", f"compare_{name}.png"), stack)

    # reliability heatmap for this frame (recompute alpha vs prev backbone)
    frames = derisk.decode_lr_and_mvs(CLIP, start_frame=WINDOWS[name]["start"],
                                      max_frames=WINDOWS[name]["n"])
    h_lr, w_lr = frames[0][1].shape[:2]
    bb = derisk.backbone_indices(frames)
    # find nearest backbone P at/after idx with a past ref (for a clean single-direction map)
    cand = [i for i in bb if frames[i][0] == "P" and max([b for b in bb if b < i], default=None) is not None]
    pi = min(cand, key=lambda i: abs(i - idx)) if cand else idx
    p = max([b for b in bb if b < pi])
    fx, fy = derisk.build_lr_flow(frames[pi][2], h_lr, w_lr, want="past")
    a = gr.reliability_lr(fx, fy, frames[pi][1], frames[p][1], gr.GateCfg(tau_res=10.0, s_res=5.0))
    a_hd = cv2.resize(a, (w_lr * SCALE, h_lr * SCALE), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((255 * (1 - a_hd)).astype(np.uint8), cv2.COLORMAP_JET)  # red=distrust
    heat = label(cv2.resize(heat, (700, int(700 * heat.shape[0] / heat.shape[1]))),
                 f"distrust (red=fresh) P#{pi}")
    cv2.imwrite(os.path.join(_HERE, "out", f"gatemap_{name}.png"), heat)
    print(f"[{name}] wrote compare_{name}.png ({stack.shape[1]}x{stack.shape[0]}), "
          f"gatemap_{name}.png; frame={idx} crop=({x0},{y0}) gateP={pi} distrust={float((a<0.5).mean()):.3f}")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["jelly", "calm"]):
        make(nm)
