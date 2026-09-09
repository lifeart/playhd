#!/usr/bin/env python3
"""
R13-E3 shared setup: clip/windows, read-only imports of the prototype, and the on-disk
SR cache (scratchpad, NOT the repo -- ~900 MB of HD uint8 frames).

Windows are IDENTICAL to R12-E1's (`experiments/r12_e1_residual_gate/run_gate_ab.py`) so
every number here is directly comparable to the residual-gate round:
    jelly = frames 582..621 of web_spike/sd600.mp4  (I-anchor at 582, mean|MV| ~6-7.7)
    calm  = frames 437..476                          (I-anchor at 437, mean|MV| ~1.1)

READ-ONLY imports only (prototype/derisk.py, prototype/sr.py are NOT edited).
"""
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "prototype"),
           os.path.join(_ROOT, "experiments", "r5_e2_quality"),
           os.path.join(_ROOT, "experiments", "r8_e4_metric_triangulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import derisk  # noqa: E402  READ-ONLY

CLIP = os.path.join(_ROOT, "web_spike", "sd600.mp4")
SCALE = 3                       # 640x320 -> 1920x960 (matches R12-E1)
CACHE = os.environ.get(
    "R13_CACHE",
    "/private/tmp/claude-501/-Users-lifeart-Repos-playhd/"
    "31ea3233-a1d0-4284-81d1-54f556b10b5b/scratchpad/r13cache")
OUT = os.path.join(_HERE, "out")

WINDOWS = {
    "jelly": dict(start=582, n=40),
    "calm":  dict(start=437, n=40),
}


def decode(name):
    cfg = WINDOWS[name]
    return derisk.decode_lr_and_mvs(CLIP, start_frame=cfg["start"], max_frames=cfg["n"])


def hd_size(frames):
    h_lr, w_lr = frames[0][1].shape[:2]
    return w_lr * SCALE, h_lr * SCALE


def bicubic_seq(frames):
    w_hd, h_hd = hd_size(frames)
    return [cv2.resize(f[1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC) for f in frames]


def sr_cache(name, model):
    """Load (or build) the per-frame SR cache for `model` on window `name`.
    model: 'realesrgan' (compact, the real-time tier) | 'realesrgan-x4plus' (quality ceiling).
    Returns a list of HD uint8 frames. Cached on disk under CACHE (scratchpad)."""
    tag = {"realesrgan": "compact", "realesrgan-x4plus": "x4plus"}[model]
    fp = os.path.join(CACHE, f"{name}_{tag}_s{SCALE}.npy")
    if os.path.exists(fp):
        return list(np.load(fp, mmap_mode="r"))
    import sr as srmod  # noqa: E402  READ-ONLY
    frames = decode(name)
    w_hd, h_hd = hd_size(frames)
    srmod.load_model(model)
    arr = np.stack([srmod.upscale_to(f[1], w_hd, h_hd, model=model) for f in frames])
    os.makedirs(CACHE, exist_ok=True)
    np.save(fp, arr)
    return list(arr)


if __name__ == "__main__":
    import time
    for nm in (sys.argv[1:] or list(WINDOWS)):
        for mdl in ("realesrgan", "realesrgan-x4plus"):
            t0 = time.time()
            c = sr_cache(nm, mdl)
            print(f"[{nm}/{mdl}] {len(c)} frames {c[0].shape} in {time.time()-t0:.1f}s",
                  flush=True)
