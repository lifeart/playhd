#!/usr/bin/env python3
"""
R13-E2 phase 4 -- COMPOSE the R12-E1 residual gate with R13-E3 (agent C)'s per-pixel
motion-adaptive EMA stabiliser, so the two levers are not judged on single-lever evidence.

WHY: both levers spend the SAME currency. C's M2 costs tOF_true +1.7% (jelly) / +5.2% (calm);
R12-E1's gate costs +5.8% (jelly, C's/R12's config). A recommendation about either that ignores
the other risks double-spending the tOF budget. C also measured that this pipeline's temporal
instability is FLICKER, not drift, and that the flicker energy is CONCENTRATED in the pixels
whose source flips between propagated warp and fresh per-frame SR (12.4x on calm, 3.5x on
high-motion) -- which is exactly the discontinuity the residual gate softens. So the composition
is not merely additive bookkeeping: the gate is the lever aimed at C's own diagnosed source.

SETUP is C's / R12-E1's, verbatim, so every number composes with both rounds:
  clip web_spike/sd600.mp4, windows jelly=582..621 and calm=437..476, SCALE=3 (1920x960),
  occ='full', per-frame fallback+anchor cache = COMPACT SR, metric reference = x4plus SR.
  The SR caches are READ from agent C's scratchpad cache (read-only; not rebuilt).

MATRIX: {baseline_hard, gate_res, gate_res_norm} x {stab off, ema_adaptive a=0.5}.
METRICS: tOF_true (vs bicubic-LR = true content motion -- the SHARED currency and the only
one no variant can game), dF, and LPIPS/DISTS vs the x4plus per-frame ceiling (the R12-E1
arbiter, reported for continuity and flagged confounded).

Run: python3 experiments/r13_e2_block_confidence/phase4_compose.py [window ...]
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
import common as C3              # noqa: E402  agent C's shared setup (READ-ONLY)
import stabilizer as ST          # noqa: E402  agent C's module (READ-ONLY, never edited)

OUT = os.path.join(HERE, "out")
SCALE = C3.SCALE
OCC = "full"                     # C's / R12-E1's occlusion mode
TOF_W = 640
METRIC_W = 640
GATE_RES = {"tau_res": 10.0, "s_res": 5.0}
_ORIG_RELIABILITY_HD = rgmod.reliability_hd


def reliability_hd_norm(mvs, lr_cur, lr_ref, want, w_hd, h_hd, cfg):
    """The shipping gate with its 0.881 trust ceiling removed (see REPORT §3.1)."""
    a = _ORIG_RELIABILITY_HD(mvs, lr_cur, lr_ref, want, w_hd, h_hd, cfg)
    ceil = 1.0 / (1.0 + np.exp(-float(cfg.get("tau_res", 10.0))
                               / max(float(cfg.get("s_res", 5.0)), 1e-6)))
    return np.clip(a / ceil, 0.0, 1.0).astype(np.float32)


def _rs(img, w):
    h, ww = img.shape[:2]
    return cv2.resize(img, (w, int(round(h * w / ww))), interpolation=cv2.INTER_AREA)


def _luma(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def dF(seq):
    return float(np.mean([np.mean(np.abs(_luma(seq[t]) - _luma(seq[t - 1])))
                          for t in range(1, len(seq))]))


def build_recons(frames, perframe):
    """baseline_hard + the two gate variants, through the SHIPPING derisk.reconstruct path."""
    out = {}
    masks = {}
    _, Rb = derisk.reconstruct(frames, None, SCALE, True, OCC, perframe, set(),
                               backend="numpy", collect_metrics=False)
    out["baseline_hard"] = {i: Rb[i]["recon"] for i in Rb}
    masks["baseline_hard"] = {i: Rb[i]["mask"] for i in Rb if Rb[i].get("mask") is not None}
    for name, patch in (("gate_res", None), ("gate_res_norm", reliability_hd_norm)):
        orig = rgmod.reliability_hd
        if patch is not None:
            rgmod.reliability_hd = patch
        try:
            _, R = derisk.reconstruct(frames, None, SCALE, True, OCC, perframe, set(),
                                      backend="numpy", collect_metrics=False,
                                      residual_gate=GATE_RES)
        finally:
            rgmod.reliability_hd = orig
        out[name] = {i: R[i]["recon"] for i in R}
        masks[name] = {i: R[i]["mask"] for i in R if R[i].get("mask") is not None}
    return out, masks


def run_window(name):
    t0 = time.time()
    frames = C3.decode(name)
    n = len(frames)
    w_hd, h_hd = C3.hd_size(frames)
    compact = C3.sr_cache(name, "realesrgan")            # fallback + anchors (R12-E1 config)
    x4ref = C3.sr_cache(name, "realesrgan-x4plus")       # metric ceiling (decoupled model)
    bic = C3.bicubic_seq(frames)
    print(f"[{name}] n={n} -> {w_hd}x{h_hd} types={''.join(f[0] for f in frames)} "
          f"(SR caches reused from R13-E3) {time.time()-t0:.0f}s", flush=True)

    perframe = {i: np.asarray(compact[i]) for i in range(n)}
    recons, masks = build_recons(frames, perframe)
    print(f"[{name}] recons built ({time.time()-t0:.0f}s)", flush=True)

    stab_cfg = ST.StabCfg(mode="ema_adaptive", alpha_max=0.50)   # C's landed M2 lever
    off_cfg = ST.StabCfg(mode="off")
    seqs = {}
    for gname, rec in recons.items():
        s_off, _ = ST.stabilize(frames, rec, off_cfg, scale=SCALE,
                                masks=masks.get(gname), enabled=False)
        seqs[f"{gname}"] = s_off
        s_on, st = ST.stabilize(frames, rec, stab_cfg, scale=SCALE,
                                masks=masks.get(gname), enabled=True)
        seqs[f"{gname}+stabM2"] = s_on
        # OFF-path parity: a disabled stabiliser must return the input untouched
        d = max(float(np.max(np.abs(np.asarray(s_off[i]).astype(np.int32)
                                    - np.asarray(rec[i]).astype(np.int32))))
                for i in range(n))
        if d != 0.0:
            raise RuntimeError(f"stabiliser OFF-path is not byte-identical ({d} codes)")
    seqs["perframe_compact"] = [np.asarray(compact[i]) for i in range(n)]
    print(f"[{name}] stabilised ({time.time()-t0:.0f}s)", flush=True)

    bic_t = [_rs(b, TOF_W) for b in bic]
    ref_t = [_rs(np.asarray(x4ref[i]), TOF_W) for i in range(n)]
    ref_m = [_rs(np.asarray(x4ref[i]), METRIC_W) for i in range(n)]

    res = dict(window=name, n=n, scale=SCALE, occ=OCC, metrics={})
    for sname, seq in seqs.items():
        st_ = [_rs(np.asarray(x), TOF_W) for x in seq]
        sm = [_rs(np.asarray(x), METRIC_W) for x in seq]
        row = dict(
            tof_truemotion=M.tof(st_, bic_t),
            tof_vs_x4=M.tof(st_, ref_t),
            dF=dF([np.asarray(x) for x in seq]),
            lpips_x4=float(np.mean([M.lpips_dist(a, b) for a, b in zip(sm, ref_m)])),
            dists_x4=float(np.mean([ME.dists(a, b) for a, b in zip(sm, ref_m)])),
        )
        res["metrics"][sname] = row
        print(f"  {sname:24s} tOF_true={row['tof_truemotion']:.4f} "
              f"tOF_x4={row['tof_vs_x4']:.4f} LPIPS={row['lpips_x4']:.4f} "
              f"DISTS={row['dists_x4']:.4f} dF={row['dF']:.3f}", flush=True)
    print(f"[{name}] done ({time.time()-t0:.0f}s)", flush=True)
    return res


def main():
    os.makedirs(os.path.join(OUT, "phase4"), exist_ok=True)
    for nm in (sys.argv[1:] or ["jelly", "calm"]):
        r = run_window(nm)
        json.dump(r, open(os.path.join(OUT, "phase4", f"{nm}.json"), "w"), indent=2)
        print(f"wrote {OUT}/phase4/{nm}.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
