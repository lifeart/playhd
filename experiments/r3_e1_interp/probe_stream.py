#!/usr/bin/env python3
"""R3-E1 probe: inspect frame types + MV reference structure for the two eval windows.
Read-only import of prototype/derisk.py. Decides whether consecutive (t,t+1,t+2) display
triplets have the clean single-step past-reference structure the interpolation test needs."""
import os
import sys
import numpy as np

PROTO = os.path.join(os.path.dirname(__file__), "..", "..", "prototype")
sys.path.insert(0, PROTO)
import derisk  # noqa: E402  (read-only)

CLIP = os.path.join(os.path.dirname(__file__), "..", "..", "sample.mp4")


def probe(start, n, tag):
    frames = derisk.decode_lr_and_mvs(CLIP, start, n)
    types = "".join(f[0] for f in frames)
    print(f"\n=== {tag}: start={start} n={len(frames)} ===")
    print("types:", types)
    smax, nbad = derisk.scan_source_magnitude(frames)
    print(f"max|source|={smax}  records with |source|>1: {nbad}")
    # per-frame: type, #mv, fraction past(<0)/future(>0), mean |mv| in px (LR)
    h, w = frames[0][1].shape[:2]
    for i, (pt, img, mvs) in enumerate(frames[:12]):
        if mvs is None or len(mvs) == 0:
            print(f"  f{i:2d} {pt}  no MVs")
            continue
        s = mvs["source"].astype(int)
        npast = int((s < 0).sum()); nfut = int((s > 0).sum())
        ms = mvs["motion_scale"].astype(np.float32); ms[ms == 0] = 1.0
        mx = mvs["motion_x"].astype(np.float32) / ms
        my = mvs["motion_y"].astype(np.float32) / ms
        mag = np.sqrt(mx * mx + my * my)
        # coverage of past-flow
        fx, fy = derisk.build_lr_flow(mvs, h, w, want="past")
        cov_past = float(np.isfinite(fx).mean())
        print(f"  f{i:2d} {pt}  nmv={len(mvs):4d} past={npast:4d} fut={nfut:4d} "
              f"|mv|mean={mag.mean():5.2f} max={mag.max():5.1f}  past-cov={cov_past:4.2f}")


if __name__ == "__main__":
    probe(0, 16, "HIGH-MOTION (start 0)")
    probe(5000, 16, "TALKING-HEAD (start 5000)")
