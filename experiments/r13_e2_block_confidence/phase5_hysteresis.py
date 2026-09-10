#!/usr/bin/env python3
"""
R13-E2 phase 5 -- TEMPORAL HYSTERESIS on the reliability membership.

Three R13 agents converged on the same shape: A (soft gate beats hard), C (flicker energy is
concentrated in the pixels whose SOURCE FLIPS between propagated warp and fresh per-frame SR --
12.4x on calm, 3.5x on high motion), D (the oracle per-pixel ceiling is -29% LPIPS / -20% DISTS
and the whole binary-detector family reaches only 21-28% of it; the headroom is a SOFT,
TEMPORALLY-HYSTERETIC membership). Nobody had measured hysteresis itself. This does.

MECHANISM (the minimal version): motion-compensate the PREVIOUS frame's reliability map into
the current frame's grid with the same LR flow the gate already builds, and EMA it into the
current one:

    a_raw = sigmoid((tau_res - react)/s_res)                 # R12-E1, unchanged
    a     = lam * warp_lr(a_prev, fx, fy) + (1 - lam) * a_raw
    a[intra hole] = 0                                        # unchanged

lam = 0 is EXACTLY the shipping gate (asserted below). The membership then changes smoothly
along a pixel's motion trajectory instead of flipping frame to frame -- which is precisely the
flip-set that C and D measured as carrying 12-27x the flicker.

State is per warp DIRECTION and is reset when PASS 2 (the B leaves) begins, because
derisk.reconstruct runs the whole I/P backbone first and only then the B frames (GOTCHA #12);
the windows used here start ON an I-anchor and contain no other I-frame, so no mid-window
anchor reset is needed (asserted).

Setup, clip, windows, SR caches and metrics are phase4_compose.py's (= agent C's = R12-E1's).
Run: python3 experiments/r13_e2_block_confidence/phase5_hysteresis.py [window ...]
"""
import os
import sys
import json
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(ROOT, "prototype"),
           os.path.join(ROOT, "experiments", "r5_e2_quality"),
           os.path.join(ROOT, "experiments", "r8_e4_metric_triangulation"),
           os.path.join(ROOT, "experiments", "r13_e3_jelly")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import derisk                    # noqa: E402  READ-ONLY
import residual_gate as rgmod    # noqa: E402  READ-ONLY
import metrics as M              # noqa: E402
import metrics_extra as ME       # noqa: E402
import common as C3              # noqa: E402  READ-ONLY (agent C's setup)

OUT = os.path.join(HERE, "out")
SCALE = C3.SCALE
OCC = "full"
TOF_W = 640
METRIC_W = 640
GATE_RES = {"tau_res": 10.0, "s_res": 5.0}


class HysteresisGate:
    """Stateful drop-in for residual_gate.reliability_hd (see module docstring)."""

    def __init__(self, lam):
        self.lam = float(lam)
        self.prev = {}
        self.pass2 = False
        self.calls = 0

    def __call__(self, mvs, lr_cur, lr_ref, want, w_hd, h_hd, cfg):
        if want == "future" and not self.pass2:
            self.pass2 = True          # PASS 2 (B leaves) begins -> drop the backbone history
            self.prev = {}
        self.calls += 1
        h_lr, w_lr = lr_cur.shape[:2]
        fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want=want)
        a = rgmod.reliability_lr(fx, fy, lr_cur, lr_ref, cfg)
        p = self.prev.get(want)
        if self.lam > 0.0 and p is not None and p.shape == a.shape:
            pw = derisk.warp_lr(p, fx, fy).astype(np.float32)
            a = (self.lam * pw + (1.0 - self.lam) * a).astype(np.float32)
        a[~np.isfinite(fx)] = 0.0
        self.prev[want] = a.copy()
        a_hd = cv2.resize(a, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        hole_hd = cv2.resize((~np.isfinite(fx)).astype(np.uint8), (w_hd, h_hd),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        a_hd[hole_hd] = 0.0
        return a_hd


def _rs(img, w):
    h, ww = img.shape[:2]
    return cv2.resize(np.asarray(img), (w, int(round(h * w / ww))), interpolation=cv2.INTER_AREA)


def _luma(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)


def dF(seq):
    return float(np.mean([np.mean(np.abs(_luma(seq[t]) - _luma(seq[t - 1])))
                          for t in range(1, len(seq))]))


def gated(frames, perframe, patch):
    orig = rgmod.reliability_hd
    if patch is not None:
        rgmod.reliability_hd = patch
    try:
        _, R = derisk.reconstruct(frames, None, SCALE, True, OCC, perframe, set(),
                                  backend="numpy", collect_metrics=False,
                                  residual_gate=GATE_RES)
    finally:
        rgmod.reliability_hd = orig
    return [R[i]["recon"] for i in range(len(frames))]


def run_window(name):
    t0 = time.time()
    frames = C3.decode(name)
    n = len(frames)
    types = "".join(f[0] for f in frames)
    if types.count("I") != 1 or types[0] != "I":
        raise RuntimeError(f"{name}: window must start on the only I-frame (types={types}) -- "
                           "the hysteresis state has no mid-window anchor reset")
    compact = C3.sr_cache(name, "realesrgan")
    x4ref = C3.sr_cache(name, "realesrgan-x4plus")
    bic = C3.bicubic_seq(frames)
    perframe = {i: np.asarray(compact[i]) for i in range(n)}
    print(f"[{name}] n={n} types={types} (caches reused) {time.time()-t0:.0f}s", flush=True)

    seqs = {}
    _, Rb = derisk.reconstruct(frames, None, SCALE, True, OCC, perframe, set(),
                               backend="numpy", collect_metrics=False)
    seqs["baseline_hard"] = [Rb[i]["recon"] for i in range(n)]
    seqs["gate_res"] = gated(frames, perframe, None)
    # lam=0 must reproduce the shipping gate EXACTLY (the control that validates the harness)
    ctrl = gated(frames, perframe, HysteresisGate(0.0))
    d0 = max(float(np.max(np.abs(ctrl[i].astype(np.int32) - seqs["gate_res"][i].astype(np.int32))))
             for i in range(n))
    if d0 != 0.0:
        raise RuntimeError(f"hysteresis lam=0 is not identical to the shipping gate ({d0} codes)")
    print(f"[{name}] PARITY hysteresis(lam=0) == gate_res: {d0} codes", flush=True)
    for lam in (0.35, 0.60):
        seqs[f"gate_res_hyst{lam:g}"] = gated(frames, perframe, HysteresisGate(lam))
    print(f"[{name}] recons built ({time.time()-t0:.0f}s)", flush=True)

    bic_t = [_rs(b, TOF_W) for b in bic]
    ref_t = [_rs(x4ref[i], TOF_W) for i in range(n)]
    ref_m = [_rs(x4ref[i], METRIC_W) for i in range(n)]
    res = dict(window=name, n=n, scale=SCALE, occ=OCC, metrics={})
    for sname, seq in seqs.items():
        st = [_rs(x, TOF_W) for x in seq]
        sm = [_rs(x, METRIC_W) for x in seq]
        row = dict(tof_truemotion=M.tof(st, bic_t), dF=dF(seq),
                   lpips_x4=float(np.mean([M.lpips_dist(a, b) for a, b in zip(sm, ref_m)])),
                   dists_x4=float(np.mean([ME.dists(a, b) for a, b in zip(sm, ref_m)])))
        res["metrics"][sname] = row
        print(f"  {sname:20s} tOF_true={row['tof_truemotion']:.4f} LPIPS={row['lpips_x4']:.4f} "
              f"DISTS={row['dists_x4']:.4f} dF={row['dF']:.3f}", flush=True)
    print(f"[{name}] done ({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    os.makedirs(os.path.join(OUT, "phase5"), exist_ok=True)
    for nm in (sys.argv[1:] or ["jelly", "calm"]):
        r = run_window(nm)
        json.dump(r, open(os.path.join(OUT, "phase5", f"{nm}.json"), "w"), indent=2)
        print(f"wrote {OUT}/phase5/{nm}.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
