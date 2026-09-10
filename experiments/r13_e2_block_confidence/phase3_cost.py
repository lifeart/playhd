#!/usr/bin/env python3
"""
R13-E2 phase 3 -- what does enabling RESIDUAL_GATE actually COST per frame?

derisk.reconstruct routes an enabled gate to the NUMPY path even under backend='torch'
(derisk.py:583 -- "the gate is the numpy path (the measured, parity-verified locus)"), so the
cost of RESIDUAL_GATE=True is not the soft lerp alone: it is (numpy recon - torch recon) PLUS
the gate's own per-direction work (an extra build_lr_flow + warp_lr + sigmoid per warp).

Measured at the SHIPPING QUALITY geometry (sample.mp4 640x320 -> 2560x1280, SCALE=4,
occ='adaptive') on a real window. No SR runs inside reconstruct, so a bicubic per-frame cache
gives identical timing to an x4plus one -- this is a pure recon/gate wall-clock.

Run: python3 experiments/r13_e2_block_confidence/phase3_cost.py [start] [n] [reps]
"""
import os
import sys
import json
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "prototype"))

import derisk                    # noqa: E402
import residual_gate as rgmod    # noqa: E402
import block_confidence as bc    # noqa: E402

SCALE = 4
OCC = "adaptive"
GATE_RES = {"tau_res": 10.0, "s_res": 5.0}


def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.time()
        fn()
        ts.append(time.time() - t)
    return min(ts), float(np.median(ts))


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 5032
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    clip = os.path.join(ROOT, "sample.mp4")
    frames = derisk.decode_lr_and_mvs(clip, start_frame=start, max_frames=n)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    pf = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
          for i in range(len(frames))}
    print(f"window start={start} n={len(frames)} {w_lr}x{h_lr} -> {w_hd}x{h_hd} "
          f"types={''.join(f[0] for f in frames)}")

    def mk(backend, gate, kind="res"):
        def go():
            orig = rgmod.reliability_hd
            if kind == "bc":
                rgmod.reliability_hd = bc.reliability_hd_bc
            try:
                derisk.reconstruct(frames, None, SCALE, True, OCC, pf, set(),
                                   backend=backend, collect_metrics=False,
                                   residual_gate=gate)
            finally:
                rgmod.reliability_hd = orig
        return go

    cases = [
        ("torch_hard  (shipping, RESIDUAL_GATE=False)", mk("torch", None)),
        ("numpy_hard  (gate's backend, gate off)", mk("numpy", None)),
        ("numpy_gate_res  (RESIDUAL_GATE=True)", mk("numpy", GATE_RES)),
        ("numpy_gate_res_block", mk("numpy", dict(GATE_RES, mode="tau"), "bc")),
        ("numpy_gate_block_only", mk("numpy", dict(GATE_RES, mode="block"), "bc")),
    ]
    out = {}
    for label, fn in cases:
        fn()                                   # warm-up (MPS kernels / caches)
        best, med = timeit(fn, reps)
        out[label] = dict(best_ms_per_frame=best / len(frames) * 1000.0,
                          median_ms_per_frame=med / len(frames) * 1000.0)
        print(f"  {label:44s} best {out[label]['best_ms_per_frame']:8.1f} ms/frame  "
              f"median {out[label]['median_ms_per_frame']:8.1f}")
    base = out["torch_hard  (shipping, RESIDUAL_GATE=False)"]["best_ms_per_frame"]
    for k, v in out.items():
        v["x_vs_shipping"] = v["best_ms_per_frame"] / base
    json.dump(dict(start=start, n=len(frames), scale=SCALE, hd=[h_hd, w_hd], cases=out),
              open(os.path.join(HERE, "out", "phase3_cost.json"), "w"), indent=2)
    print("\nslowdown vs the shipping torch path:")
    for k, v in out.items():
        print(f"  {k:44s} {v['x_vs_shipping']:6.2f}x")


if __name__ == "__main__":
    sys.exit(main())
