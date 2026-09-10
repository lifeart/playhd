#!/usr/bin/env python3
"""
R13-E2 phase 2b -- the multi-clip tOF + fallback sweep on REAL, un-degraded footage at the
project's native geometry (the exact thing the R12 review asked for before flipping
RESIDUAL_GATE default-ON, and which was never run).

No ground truth exists for real footage, so this uses the project's OWN honest decision metrics
(GOTCHA #13): tOF against the bicubic-upscaled decoded LR (= the TRUE content motion, which no
variant can game by giving up on propagation) and the hard-path fallback fraction. LPIPS/DISTS
against the per-frame SR sequence are also reported, flagged as the CONFOUNDED arbiter (the
gate's fallback source is that same per-frame SR cache, so those two numbers fall monotonically
with distrust by construction -- see phase2_sweep.py and REPORT.md).

Geometry: sample.mp4 / web_spike sd600.mp4 windows at SCALE=3 (640x320 -> 1920x960), per-frame
x4plus SR + the R8-E3 beta=0.85 anchor blend, occ='adaptive' -- the shipping quality-mode gate
locus.

Run: python3 experiments/r13_e2_block_confidence/phase2b_realtof.py [window ...]
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
sys.path.insert(0, os.path.join(ROOT, "experiments", "r5_e2_quality"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "r8_e4_metric_triangulation"))

import derisk                    # noqa: E402
import sr as srmod               # noqa: E402
import residual_gate as rgmod    # noqa: E402
import block_confidence as bc    # noqa: E402
import metrics as M              # noqa: E402
import metrics_extra as ME       # noqa: E402

OUT = os.path.join(HERE, "out")
SCALE = 3
TOF_W = 640
METRIC_W = 640
OCC = "adaptive"
BETA = 0.85
GATE_RES = {"tau_res": 10.0, "s_res": 5.0}
_ORIG_RELIABILITY_HD = rgmod.reliability_hd   # bound ONCE: the norm variant is installed onto
                                              # rgmod.reliability_hd, so it must not call the
                                              # patched attribute (that would recurse forever).


def reliability_hd_norm(mvs, lr_cur, lr_ref, want, w_hd, h_hd, cfg):
    """ABLATION -- the shipping gate with its trust CEILING removed.

    The R12-E1 reliability is a = sigmoid((tau_res - react)/s_res); at a PERFECT warp
    (react = 0) that is sigmoid(10/5) = 0.881, so the shipping gate takes ~11.9% of EVERY
    pixel from the fresh per-frame SR cache even where propagation is exact. This variant
    rescales a by that ceiling (a/0.881, clipped) so a perfect warp is fully trusted and
    only genuinely-unreliable pixels are diluted. Comparing it against gate_res separates
    'the soft transition helps' from 'globally mixing in per-frame SR helps'."""
    a = _ORIG_RELIABILITY_HD(mvs, lr_cur, lr_ref, want, w_hd, h_hd, cfg)
    ceil = 1.0 / (1.0 + np.exp(-float(cfg.get("tau_res", 10.0))
                               / max(float(cfg.get("s_res", 5.0)), 1e-6)))
    return np.clip(a / ceil, 0.0, 1.0).astype(np.float32)


VARIANTS = {
    "gate_res":       ("res", GATE_RES),
    "gate_res_norm":  ("norm", GATE_RES),
    "gate_res_block": ("bc", dict(GATE_RES, mode="tau")),
    "gate_block":     ("bc", dict(GATE_RES, mode="block")),
}

SAMPLE = os.path.join(ROOT, "sample.mp4")
SD600 = os.path.join(ROOT, "web_spike", "sd600.mp4")
WINDOWS = {
    # the project's canonical sample.mp4 windows (handoff "Step 1")
    "A_highmotion":  dict(clip=SAMPLE, start=0, n=32),
    "B_mid":         dict(clip=SAMPLE, start=2500, n=32),
    "C_talkinghead": dict(clip=SAMPLE, start=5032, n=32),
    "E_mid2":        dict(clip=SAMPLE, start=12000, n=32),
    # the two R12-E1 windows, for direct continuity with the measurement that produced the flag
    "R12_jelly":     dict(clip=SD600, start=582, n=32),
    "R12_calm":      dict(clip=SD600, start=437, n=32),
}


def _rs(img, w):
    h, ww = img.shape[:2]
    return cv2.resize(img, (w, int(round(h * w / ww))), interpolation=cv2.INTER_AREA)


def _luma(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def dF(seq):
    return float(np.mean([np.mean(np.abs(_luma(seq[t]) - _luma(seq[t - 1])))
                          for t in range(1, len(seq))]))


def run_gated(frames, perframe, kind, cfg):
    orig = rgmod.reliability_hd
    if kind == "bc":
        rgmod.reliability_hd = bc.reliability_hd_bc
    elif kind == "norm":
        rgmod.reliability_hd = reliability_hd_norm
    try:
        _, R = derisk.reconstruct(frames, None, SCALE, True, OCC, perframe, set(),
                                  backend="numpy", collect_metrics=False, residual_gate=cfg)
    finally:
        rgmod.reliability_hd = orig
    return {i: R[i]["recon"] for i in R}


def run_window(name):
    t0 = time.time()
    cfg = WINDOWS[name]
    frames = derisk.decode_lr_and_mvs(cfg["clip"], start_frame=cfg["start"],
                                      max_frames=cfg["n"])
    n = len(frames)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    types = "".join(f[0] for f in frames)
    print(f"[{name}] n={n} {w_lr}x{h_lr} -> {w_hd}x{h_hd} types={types}", flush=True)

    srmod.load_model("realesrgan")
    srmod.load_model("realesrgan-x4plus")
    compact = {i: srmod.upscale_to(frames[i][1], w_hd, h_hd, model="realesrgan") for i in range(n)}
    perframe = {i: srmod.upscale_to(frames[i][1], w_hd, h_hd, model="realesrgan-x4plus")
                for i in range(n)}
    derisk.blend_anchor_cache(perframe, compact, BETA)
    bic = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
           for i in range(n)}
    print(f"[{name}] SR caches built ({time.time()-t0:.0f}s)", flush=True)

    recons = {}
    _, Rb = derisk.reconstruct(frames, None, SCALE, True, OCC, perframe, set(),
                               backend="numpy", collect_metrics=False)
    recons["baseline_hard"] = {i: Rb[i]["recon"] for i in Rb}
    hard_fb = float(np.mean([Rb[i]["hole_frac"] for i in Rb if not Rb[i]["is_anchor"]]))
    for vn, (kind, c) in VARIANTS.items():
        recons[vn] = run_gated(frames, perframe, kind, c)
    recons["perframe_sr"] = perframe
    recons["bicubic"] = bic

    bic_t = [_rs(bic[i], TOF_W) for i in range(n)]
    pf_t = [_rs(perframe[i], TOF_W) for i in range(n)]
    pf_m = [_rs(perframe[i], METRIC_W) for i in range(n)]

    res = dict(name=name, start=cfg["start"], n=n, types=types,
               hard_fallback_frac=hard_fb, metrics={})
    for rn in recons:
        seq = [recons[rn][i] for i in range(n)]
        st = [_rs(x, TOF_W) for x in seq]
        sm = [_rs(x, METRIC_W) for x in seq]
        row = dict(
            tof_truemotion=M.tof(st, bic_t),        # HONEST: true content motion, ungameable
            tof_vs_perframe=M.tof(st, pf_t),
            dF=dF(seq),
            lpips_pf=float(np.mean([M.lpips_dist(a, b) for a, b in zip(sm, pf_m)])),
            dists_pf=float(np.mean([ME.dists(a, b) for a, b in zip(sm, pf_m)])),
        )
        res["metrics"][rn] = row
        print(f"  {rn:15s} tOF_true={row['tof_truemotion']:.4f} "
              f"tOF_pf={row['tof_vs_perframe']:.4f} dF={row['dF']:.3f} "
              f"[confounded] LPIPS_pf={row['lpips_pf']:.4f} DISTS_pf={row['dists_pf']:.4f}",
              flush=True)
    print(f"[{name}] hard-fallback={100*hard_fb:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    names = sys.argv[1:] or list(WINDOWS)
    os.makedirs(os.path.join(OUT, "phase2b"), exist_ok=True)
    for nm in names:
        r = run_window(nm)
        json.dump(r, open(os.path.join(OUT, "phase2b", f"{nm}.json"), "w"), indent=2)
        print(f"wrote {OUT}/phase2b/{nm}.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
