#!/usr/bin/env python3
"""
R13-E4 harness: a GT-BEARING, production-path A/B rig for the occlusion mask.

WHY GT-BEARING. The mask decides, per pixel, PROPAGATED-from-anchor vs FALLBACK-to-
per-frame-SR. Every no-GT proxy this project has is biased on exactly that axis:
`psnr_lr_consistency` and `psnr_prop_vs_perframe` both REWARD falling back (GOTCHA
#13), and so would any "distance to the per-frame-SR image". So a mask A/B scored
without ground truth is rigged toward the bigger mask. We therefore use the project's
established degrade-restore protocol (R5-E2/R6-E1/R8-E3/R10/R12), with the important
addition that the LR is a REAL libx264 encode -> real RD-optimised codec MVs, real
B-frames, real blocking:

    GT      = sample.mp4 frames at 640x320 (pseudo true-HD)
    LR      = GT downscaled 2x (INTER_AREA) -> REAL libx264 encode -> decode w/ export_mvs
    RESTORE = the production propagation path (derisk.reconstruct, x4plus anchors,
              region-aware output blend, anchor-blend beta=0.85) at scale 2 -> 640x320
    SCORE   = LPIPS + DISTS + PSNR against GT, tOF against the GT sequence, fallback%

The reconstruction code is the SHIPPED one (derisk.reconstruct / reconstruct_torch);
only the occlusion-mask hook differs between arms (see fsr2_mask.py).

Caveat recorded in the report: the LR here is 320x160, half the production 640x320,
so MV blocks cover 2x more of the frame than in production. The complementary
scale-4 production-resolution arm (real sample.mp4 MVs, 640x320 -> 2560x1280) is run
separately for tOF/fallback%/ms, where no GT is needed.
"""
import os
import sys
import json
import numpy as np
import cv2
import av

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "prototype"),
           os.path.join(_ROOT, "experiments", "r8_e4_metric_triangulation"),
           os.path.join(_ROOT, "experiments", "r5_e2_quality")):
    if _p not in sys.path:
        sys.path.append(_p)

import derisk as D            # noqa: E402  READ-ONLY

SAMPLE = os.path.join(_ROOT, "sample.mp4")
CACHE = os.environ.get(
    "R13_E4_CACHE",
    "/private/tmp/claude-501/-Users-lifeart-Repos-playhd/"
    "31ea3233-a1d0-4284-81d1-54f556b10b5b/scratchpad/r13e4")
os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(_HERE, "out")
os.makedirs(OUT, exist_ok=True)

# The three canonical working windows (handoff "Program progress"): A high-motion,
# B mid-clip (dense texture, the R11/R12 'texture24k' window), C talking-head.
WINDOWS = {
    "A": dict(start=0,     n=48, desc="high-motion stress"),
    "B": dict(start=24000, n=48, desc="mid-clip, dense texture"),
    "C": dict(start=5000,  n=48, desc="talking-head (the realistic case)"),
}
SCALE = 2          # LR 320x160 -> 640x320 == GT resolution
CRF = 26           # the delivered-SD operating point (sample.mp4 is ~0.063 bpp)


# --------------------------------------------------------------------------- #
def decode_rgb(path, start=0, n=None):
    cont = av.open(path)
    vs = cont.streams.video[0]
    out, idx = [], 0
    for frame in cont.decode(vs):
        if idx < start:
            idx += 1
            continue
        if n is not None and len(out) >= n:
            break
        out.append(frame.to_ndarray(format="rgb24"))
        idx += 1
    cont.close()
    return out


def encode_lr(gt, path, crf=CRF, fps=25):
    """GT frames -> 2x INTER_AREA down -> REAL libx264 (B-frames on, GOP 24)."""
    h, w = gt[0].shape[:2]
    c = av.open(path, "w")
    st = c.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w // SCALE, h // SCALE, "yuv420p"
    st.options = {"crf": str(crf), "preset": "medium", "g": "24", "bf": "3"}
    for fr in gt:
        img = cv2.resize(fr, (w // SCALE, h // SCALE), interpolation=cv2.INTER_AREA)
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        for p in st.encode(vf):
            c.mux(p)
    for p in st.encode():
        c.mux(p)
    c.close()
    return path


def window(name, crf=CRF):
    """(frames, gt) for a window: frames = decoded LR + real MVs, gt = pseudo true-HD."""
    cfg = dict(WINDOWS[name])
    cfg["n"] = int(os.environ.get("R13_E4_N", cfg["n"]))
    lr_path = os.path.join(CACHE, f"lr_{name}_n{cfg['n']}_crf{crf}.mp4")
    gt_npy = os.path.join(CACHE, f"gt_{name}_n{cfg['n']}.npy")
    if os.path.exists(gt_npy):
        gt = list(np.load(gt_npy))
    else:
        gt = decode_rgb(SAMPLE, cfg["start"], cfg["n"])
        np.save(gt_npy, np.stack(gt))
    if not os.path.exists(lr_path):
        encode_lr(gt, lr_path, crf)
    frames = D.decode_lr_and_mvs(lr_path)
    n = min(len(frames), len(gt))
    return frames[:n], gt[:n]


# --------------------------------------------------------------------------- #
# SR caches (built ONCE per window, shared by every mask arm -- the whole point of
# derisk's perframe_cache: only warp/mask/blend re-runs across operating points).
# --------------------------------------------------------------------------- #
def sr_caches(name, frames, w_hd, h_hd, fp16=True):
    import sr as SR
    out = {}
    for tag, model in (("x4plus", "realesrgan-x4plus"), ("compact", "realesrgan")):
        # the frame count MUST be in the cache key: a stale shorter cache silently
        # KeyErrors deep inside reconstruct() (hit once; do not remove the n).
        f = os.path.join(CACHE, f"sr_{name}_n{len(frames)}_{tag}_{w_hd}x{h_hd}.npz")
        if os.path.exists(f):
            z = np.load(f)
            out[tag] = {int(k): z[k] for k in z.files}
            continue
        half = fp16 and tag == "x4plus"          # quality mode ships fp16 x4plus
        cache = D.build_perframe_cache(frames, w_hd, h_hd, model, half=half)
        np.savez(f, **{str(k): v for k, v in cache.items()})
        out[tag] = cache
        _free_gpu()
    return out["x4plus"], out["compact"]


def _free_gpu():
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
    except ImportError as e:
        print(f"[harness] torch unavailable, no GPU free: {e!r}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
_LP = {}


def _lpips_net():
    if "net" not in _LP:
        import lpips
        _LP["net"] = lpips.LPIPS(net="alex", verbose=False).eval()
    return _LP["net"]


def lpips_dist(a, b):
    """TRUE LPIPS (AlexNet, the `lpips` package) -- the project's lead perceptual metric."""
    import torch
    net = _lpips_net()
    ta = torch.from_numpy(np.ascontiguousarray(a)).float().div_(127.5).sub_(1.0)
    tb = torch.from_numpy(np.ascontiguousarray(b)).float().div_(127.5).sub_(1.0)
    with torch.no_grad():
        return float(net(ta.permute(2, 0, 1)[None], tb.permute(2, 0, 1)[None]).item())


def score(seq, gt, tag=""):
    """Full-reference perceptual + fidelity + temporal scores of a recon sequence."""
    import metrics_extra as ME
    n = min(len(seq), len(gt))
    lp = [lpips_dist(seq[i], gt[i]) for i in range(n)]
    ds = [ME.dists(seq[i], gt[i]) for i in range(n)]
    ps = [D.psnr(seq[i], gt[i]) for i in range(n)]
    t = D.tof(seq[:n], gt[:n])
    return dict(tag=tag, n=n, lpips=float(np.mean(lp)), dists=float(np.mean(ds)),
                psnr=float(np.mean(ps)), tof=float(t),
                lpips_per_frame=lp, dists_per_frame=ds)


def save(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"[saved] {path}")
