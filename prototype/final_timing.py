#!/usr/bin/env python3
"""Step-7 clean deployable timing: both windows, interleaved per-config to share thermal state,
best-of-N reps to reject thermal/scheduling outliers. The warp/mask/blend op timings are
independent of the SR fill content, so bicubic SR is used (fast, low heat) -- the recon wall-clock
is identical to the realesrgan path (SR is amortized separately and excluded here)."""
import statistics as st
import time
import numpy as np

import derisk
import gpu_ops as G
from derisk import PROF

REPS = 12
SCALE = 4
WINDOWS = [("C_talkinghead", 5000), ("A_highmotion", 0)]
CONFIGS = [("full", False), ("reactive", False), ("adaptive", False),  # deployable (no download)
           ("full", True)]                                              # +HD download (with-I/O)


def prep(start):
    frames = derisk.decode_lr_and_mvs("../sample.mp4", start, 48)
    h, w = frames[0][1].shape[:2]
    pf = derisk.build_perframe_cache(frames, w*SCALE, h*SCALE, "bicubic")
    pf_dev = {i: G.img_to_dev(pf[i]) for i in range(len(frames))}   # on-GPU SR output, no upload
    return frames, pf_dev, len(frames)


def time_cfg(frames, pf_dev, n, occ, dl):
    kw = dict(backend="torch", collect_metrics=False, download_output=dl)
    derisk.reconstruct(frames, None, SCALE, True, occ, pf_dev, set(), **kw); G.sync()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        derisk.reconstruct(frames, None, SCALE, True, occ, pf_dev, set(), **kw); G.sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    fires = f"{derisk.MASK_FIRES[0]}/{derisk.MASK_FIRES[1]}"
    return min(ts)/n, st.median(ts)/n, fires


def main():
    PROF.reset(enabled=False)
    W = {name: prep(start) for name, start in WINDOWS}
    res = {}
    # interleave: for each config, time both windows adjacent (shared thermal state)
    for occ, dl in CONFIGS:
        for name, _ in WINDOWS:
            frames, pf_dev, n = W[name]
            res[(name, occ, dl)] = time_cfg(frames, pf_dev, n, occ, dl)
    label = {("full", False): "full        [deployable]",
             ("reactive", False): "reactive    [deployable]",
             ("adaptive", False): "adaptive    [deployable]",
             ("full", True): "full        [with-I/O dl]"}
    print(f"\n=== Step-7 deployable timing (bicubic SR, best-of-{REPS}, ms/frame) ===")
    for name, _ in WINDOWS:
        print(f"\n  window {name}:")
        print(f"    {'config':<26}{'best':>8}{'median':>9}{'fps(best)':>11}{'fwdbwd':>9}{'<=40?':>7}")
        for occ, dl in CONFIGS:
            best, med, fires = res[(name, occ, dl)]
            print(f"    {label[(occ, dl)]:<26}{best:>8.2f}{med:>9.2f}{1000/best:>11.1f}"
                  f"{fires:>9}{('YES' if best <= 40 else 'no'):>7}")


if __name__ == "__main__":
    main()
