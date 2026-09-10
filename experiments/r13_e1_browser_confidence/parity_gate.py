#!/usr/bin/env python3
"""
R13-E1 SEAM CHECK: the browser's WGSL reliability pass vs the server twin,
prototype/residual_gate.py::reliability_lr (imported READ-ONLY, never edited).

Both are fed the SAME inputs (the LR frames + codec MVs the harness ran on) and the same
(tau, s); the GPU map is read back from the r16float texture the harness dumps. A match
proves the port is the same function, not a lookalike -- the established technique in this
repo (prior: warp parity mean|delta| 0.0021 codes).

Expected (documented) deviations, both sub-1e-2:
  * cv2.remap INTER_LINEAR is 1/32-px fixed point; the GPU sampler is true float bilinear.
  * the GPU map is stored as r16float (~1e-3 relative).

Run: python3 parity_gate.py --scratch DIR --window jelly --variant soft_t16s3 --tau 16 --s 3
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))
import derisk           # noqa: E402  READ-ONLY
import residual_gate    # noqa: E402  READ-ONLY -- the reference implementation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--tau", type=float, default=16.0)
    ap.add_argument("--s", type=float, default=3.0)
    args = ap.parse_args()

    data = os.path.join(args.scratch, args.window)
    outd = os.path.join(args.scratch, f"out_{args.window}_{args.variant}")
    meta = json.load(open(os.path.join(data, "meta.json")))
    w, h, n = meta["w"], meta["h"], meta["n"]
    cfg = dict(tau_res=args.tau, s_res=args.s)

    stats = []
    for i in range(n):
        f = os.path.join(outd, f"reli_{i}.f16.bin")
        if not os.path.exists(f):
            continue                                   # anchor frame (no gate ran)
        gpu = np.frombuffer(open(f, "rb").read(), dtype=np.float16).astype(np.float32).reshape(h, w)
        lr_cur = np.fromfile(os.path.join(data, f"rgb_{i}.bin"), np.uint8).reshape(h, w, 3)
        lr_prev = np.fromfile(os.path.join(data, f"rgb_{i-1}.bin"), np.uint8).reshape(h, w, 3)
        mvs = np.fromfile(os.path.join(data, f"mv_{i}.bin"), np.int32).reshape(-1, 10)
        # rebuild the same flow the browser built (flow.js buildLrFlow is bit-exact vs build_lr_flow,
        # verified in web_spike/wasm_mv/README.md) from the packed records
        rec = np.zeros(len(mvs), dtype=[(k, "<i4") for k in
                                        ("source", "w", "h", "src_x", "src_y", "dst_x", "dst_y",
                                         "motion_x", "motion_y", "motion_scale")])
        for j, k in enumerate(rec.dtype.names):
            rec[k] = mvs[:, j]
        fx, fy = derisk.build_lr_flow(rec, h, w, want="all")
        cpu = residual_gate.reliability_lr(fx, fy, lr_cur, lr_prev, cfg)
        d = np.abs(cpu - gpu)
        stats.append((i, float(d.mean()), float(d.max()), float(np.mean(d > 0.05))))

    if not stats:
        raise SystemExit(f"no reliability dumps in {outd} — run harness.ts with --mode soft --out")
    arr = np.array([[s[1], s[2], s[3]] for s in stats])
    print(json.dumps(dict(
        window=args.window, variant=args.variant, frames=len(stats), tau=args.tau, s=args.s,
        mean_abs_delta=round(float(arr[:, 0].mean()), 6),
        max_abs_delta=round(float(arr[:, 1].max()), 6),
        frac_pixels_over_0p05=round(float(arr[:, 2].mean()), 6),
    ), indent=2))


if __name__ == "__main__":
    main()
