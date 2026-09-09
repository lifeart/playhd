#!/usr/bin/env python3
"""
R13-E3 runner: DIAGNOSE the jelly (drift vs flicker) and A/B the two ported mechanisms.

    python run_ab.py jelly            # one window
    python run_ab.py all              # jelly + calm

Baseline = the SHIPPING path: derisk.reconstruct(numpy, occ='full', hard occlusion
fallback), compact anchors, scale 3 -- identical setup to R12-E1 so the numbers compose.

Variants (all OUTPUT-ONLY post-passes, see stabilizer.py):
    ema_fixed_XX     constant-alpha motion-compensated EMA (the naive straw man)
    ema_adapt_XX     M2 per-pixel motion-adaptive alpha
    instab_XX        M1 FSR3.1 luma instability only (FSR's own velocity gate)
    instab_relaxed   M1 with the velocity gate relaxed to this content's motion scale
    ema_instab_XX    M2 + M1

Context rows: compact_perframe (fresh SR every frame), x4plus_perframe (quality ceiling),
bicubic (content+codec floor for the relative temporal metrics).

Metrics: tLP (perceptual temporal, vs bicubic content reference), MCLF (motion-compensated
luma flicker on codec MVs), dF (raw), tOF_true (R12-comparable), and the QUALITY GUARD
LPIPS/DISTS vs the x4plus per-frame ceiling -- a temporal smoother that wins the flicker
metrics by blurring MUST lose here, so both are always reported together.
"""
import os
import sys
import json
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from common import SCALE, OUT, decode, hd_size, bicubic_seq, sr_cache  # noqa: E402
import jelly_metrics as JM      # noqa: E402
import stabilizer as ST         # noqa: E402
import derisk                   # noqa: E402  READ-ONLY
import metrics as M             # noqa: E402  READ-ONLY (LPIPS, tOF)
import metrics_extra as ME      # noqa: E402  READ-ONLY (DISTS)

DISTS_STRIDE = 4        # DISTS is ~1.8 s/pair on CPU; a strided mean is the quality guard
TOF_W = 960


def _variants():
    return {
        "ema_fixed_50":   ST.StabCfg(mode="ema_fixed", alpha_max=0.50),
        "ema_adapt_50":   ST.StabCfg(mode="ema_adaptive", alpha_max=0.50),
        "ema_adapt_75":   ST.StabCfg(mode="ema_adaptive", alpha_max=0.75),
        "instab_fsr":     ST.StabCfg(mode="instab_only", alpha_instab=0.70),
        "instab_relaxed": ST.StabCfg(mode="instab_only", alpha_instab=0.70, vel_max_4k=100.0),
        "ema_instab_50":  ST.StabCfg(mode="ema_instab", alpha_max=0.50, alpha_instab=0.70,
                                     vel_max_4k=100.0),
    }


def _small(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (TOF_W, int(round(h * TOF_W / w))), interpolation=cv2.INTER_AREA)


def run_window(name):
    t0 = time.time()
    frames = decode(name)
    N = len(frames)
    w_hd, h_hd = hd_size(frames)
    types = "".join(f[0] for f in frames)
    print(f"[{name}] {N} frames -> {w_hd}x{h_hd}  types={types}", flush=True)

    compact = sr_cache(name, "realesrgan")
    x4ref = sr_cache(name, "realesrgan-x4plus")
    bic = bicubic_seq(frames)

    # ---- baseline = shipping reconstruction (hard occlusion fallback) ----
    _, R = derisk.reconstruct(frames, None, SCALE, True, "full",
                              {i: compact[i] for i in range(N)}, set(), backend="numpy")
    base = {i: R[i]["recon"] for i in range(N)}
    masks = {i: (np.asarray(R[i]["mask"]) if R[i].get("mask") is not None else None)
             for i in range(N)}
    print(f"[{name}] baseline recon done ({time.time()-t0:.1f}s)", flush=True)

    # ---- mean |MV| per frame (context for the FSR velocity gate) ----
    h_lr, w_lr = frames[0][1].shape[:2]
    mvmag = []
    for i in range(N):
        fx, fy = derisk.build_lr_flow(frames[i][2], h_lr, w_lr, want="past")
        mag = np.sqrt(fx * fx + fy * fy)
        mvmag.append(float(np.nanmean(mag)) if np.isfinite(mag).any() else 0.0)

    # ---- sequences under test ----
    seqs = {"baseline_hard": [base[i] for i in range(N)]}
    stats = {}
    for vname, cfg in _variants().items():
        s, st = ST.stabilize(frames, base, cfg, scale=SCALE, masks=masks, enabled=True)
        seqs[vname] = s
        stats[vname] = {k: v for k, v in st.items() if k not in ("alpha_mean", "instab_frac")}
        print(f"[{name}] {vname}: alpha_mean={st['alpha_mean_all']:.3f} "
              f"instab={st['instab_frac_all']:.4f}", flush=True)
    # OFF-path parity: the flag default must leave the sequence untouched
    off_seq, _ = ST.stabilize(frames, base, ST.StabCfg(mode="ema_adaptive"), scale=SCALE,
                              masks=masks)   # enabled=None -> STAB_ENABLED (False)
    parity = max(int(np.max(np.abs(off_seq[i].astype(np.int32) - base[i].astype(np.int32))))
                 for i in range(N))
    print(f"[{name}] OFF-path parity vs baseline: max|d| = {parity} codes", flush=True)

    seqs["compact_perframe"] = [compact[i] for i in range(N)]
    seqs["x4plus_perframe"] = [x4ref[i] for i in range(N)]
    seqs["bicubic"] = bic

    # ---- shared temporal machinery (identical flows for every variant) ----
    flows = JM.build_mc_flows(frames, SCALE)
    cover = [f["cover"] for f in flows if f is not None]
    print(f"[{name}] MC flows: {len(cover)}/{N-1} pairs, mean codec-MV coverage "
          f"{100*np.mean(cover):.1f}%", flush=True)
    dist, backbone, anchors = JM.frames_since_anchor(frames)

    bic_lp = JM.adjacent_lpips(bic)
    ref_seq = [x4ref[i] for i in range(N)]

    results = dict(window=name, n=N, types=types, scale=SCALE,
                   mv_mean=float(np.mean(mvmag)), mv_series=mvmag,
                   anchors=anchors, backbone=backbone, dist=dist,
                   flow_cover=float(np.mean(cover)), off_parity_codes=parity,
                   stab_stats=stats, metrics={}, series={})

    for sname, seq in seqs.items():
        t1 = time.time()
        mclf = JM.mclf_series(seq, flows, SCALE)
        adj = JM.adjacent_lpips(seq)
        tlp = JM.tlp_series(adj, bic_lp)
        dfs = JM.dF_series(seq)
        # quality guard vs the x4plus per-frame ceiling
        lp = float(np.mean([M.lpips_dist(JM._mw(seq[i]), JM._mw(ref_seq[i])) for i in range(N)]))
        ds = float(np.mean([ME.dists(JM._mw(seq[i]), JM._mw(ref_seq[i]))
                            for i in range(0, N, DISTS_STRIDE)]))
        tof_true = M.tof([_small(x) for x in seq], [_small(x) for x in bic])
        row = dict(
            mclf=float(np.nanmean(mclf)), tlp=float(np.nanmean(tlp)),
            adj_lpips=float(np.nanmean(adj)), dF=float(np.nanmean(dfs)),
            lpips_vs_x4=lp, dists_vs_x4=ds, tof_true=tof_true,
            mclf_drift=JM.drift_vs_flicker(mclf, dist),
            tlp_drift=JM.drift_vs_flicker(tlp, dist),
        )
        results["metrics"][sname] = row
        results["series"][sname] = dict(mclf=mclf, tlp=tlp, adj_lpips=adj, dF=dfs)
        print(f"  {sname:16s} MCLF={row['mclf']:7.3f} tLP={row['tlp']:.4f} "
              f"LPIPSvx4={lp:.4f} DISTSvx4={ds:.4f} tOF={tof_true:.3f} "
              f"({time.time()-t1:.0f}s)", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"results_{name}.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[{name}] total {time.time()-t0:.1f}s", flush=True)
    return results


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for nm in (["jelly", "calm"] if which == "all" else [which]):
        run_window(nm)
