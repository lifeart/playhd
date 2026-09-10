#!/usr/bin/env python3
"""
R13-E2 phase 2 -- the MULTI-CLIP gate sweep the R12 review asked for, plus the block-confidence
variants, judged against a TRUE ground truth.

TWO arbiters are computed on the SAME reconstructions, deliberately:

  (1) vs TRUE GT       -- the degrade-and-restore pseudo-HD ground truth (make_degraded.py).
                          Independent of both the propagation and the SR model. This is the
                          honest arbiter.
  (2) vs per-frame SR  -- the R12-E1 arbiter (LPIPS/DISTS against the per-frame SR sequence),
                          reported for continuity AND to expose its confound: the soft gate's
                          fallback source IS the per-frame SR cache, so as trust a -> 0 the
                          reconstruction converges ON the reference and this arbiter goes to 0
                          by construction. Any "win" it reports must be corroborated by (1).

Pipeline configuration = the SHIPPING QUALITY mode's gate locus (that is the only mode whose
MODE_CONFIG carries residual_gate="RESIDUAL_GATE"): per-frame x4plus SR cache with the R8-E3
anchor blend beta=0.85 applied (derisk.blend_anchor_cache), occ='adaptive', numpy backend,
scale 2 (the probe clips are 2x). The region-aware OUTPUT pass is intentionally NOT applied so
the gate is isolated (it is an identical post-pass for every variant); see REPORT.md.

Variants
  baseline_hard   derisk.reconstruct(residual_gate=None)          == RESIDUAL_GATE=False (shipping)
  gate_res        residual_gate.reliability_hd, tau=10 s=5        == RESIDUAL_GATE=True  (R12-E1)
  gate_res_block  block_confidence mode='tau'  (residual x block prior)
  gate_res_bmul   block_confidence mode='mul'  (residual * block prior)
  gate_block      block_confidence mode='block' (FREE: no warp, no residual)
  perframe_sr     the per-frame SR cache itself (context / the a->0 limit)
  bicubic         context floor

Run: python3 experiments/r13_e2_block_confidence/phase2_sweep.py [clip ...]
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

import derisk                    # noqa: E402  READ-ONLY
import sr as srmod               # noqa: E402  READ-ONLY
import residual_gate as rgmod    # noqa: E402  READ-ONLY (shipping gate)
import block_confidence as bc    # noqa: E402  (this experiment's module)
import metrics as M              # noqa: E402  lpips_dist, tof, psnr
import metrics_extra as ME       # noqa: E402  dists (pyiqa)

OUT = os.path.join(HERE, "out")
PROBE = os.path.join(OUT, "probe")
SCALE = 2
METRIC_W = 640          # LPIPS/DISTS at 640 wide (CPU-tractable, >= the probe HD width/1)
TOF_W = 480             # tOF Farneback at 480 wide
OCC = "adaptive"
BETA = 0.85             # R8-E3 anchor blend, as shipping quality mode

GATE_RES = {"tau_res": 10.0, "s_res": 5.0}
VARIANTS = {
    "gate_res":       ("res", GATE_RES),
    "gate_res_block": ("bc", dict(GATE_RES, mode="tau")),
    "gate_res_bmul":  ("bc", dict(GATE_RES, mode="mul")),
    "gate_block":     ("bc", dict(GATE_RES, mode="block")),
}
if os.environ.get("R13_LEAN"):          # contention budget: drop the two context-only variants
    VARIANTS.pop("gate_res_bmul")       # (its mode='mul' twin is covered by gate_res_block)


def _mw(img):
    h, w = img.shape[:2]
    if w == METRIC_W:
        return img
    return cv2.resize(img, (METRIC_W, int(round(h * METRIC_W / w))), interpolation=cv2.INTER_AREA)


def _tw(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (TOF_W, int(round(h * TOF_W / w))), interpolation=cv2.INTER_AREA)


def _luma(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def dF(seq):
    return float(np.mean([np.mean(np.abs(_luma(seq[t]) - _luma(seq[t - 1])))
                          for t in range(1, len(seq))]))


def run_gated(frames, perframe, cfg_kind, cfg):
    """Run the SHIPPING gated path (derisk.reconstruct with residual_gate=cfg). For the
    block-confidence variants the gate's reliability function is swapped for this experiment's
    implementation via an explicit, restored monkeypatch -- derisk.py and residual_gate.py are
    NOT edited (R13 file ownership)."""
    orig = rgmod.reliability_hd
    if cfg_kind == "bc":
        rgmod.reliability_hd = bc.reliability_hd_bc
    try:
        _, R = derisk.reconstruct(frames, None, SCALE, True, OCC, perframe, set(),
                                  backend="numpy", collect_metrics=False,
                                  residual_gate=cfg)
    finally:
        rgmod.reliability_hd = orig          # always restored; never silently left patched
    if rgmod.reliability_hd is not orig:
        raise RuntimeError("reliability_hd monkeypatch was not restored")
    return {i: R[i]["recon"] for i in R}


def distrust_stats(frames, cfg_kind, cfg):
    """Mean per-frame distrust (1 - a) over the backbone 'past' direction: the fraction of the
    reconstruction that is taken from fresh per-frame SR rather than from propagation."""
    h_lr, w_lr = frames[0][1].shape[:2]
    bb = derisk.backbone_indices(frames)
    vals = []
    for i in bb:
        p = max([b for b in bb if b < i], default=None)
        if p is None or frames[i][0] == "I":
            continue
        if cfg_kind == "bc":
            a = bc.reliability_hd_bc(frames[i][2], frames[i][1], frames[p][1], "past",
                                     w_lr, h_lr, cfg)
        else:
            a = rgmod.reliability_hd(frames[i][2], frames[i][1], frames[p][1], "past",
                                     w_lr, h_lr, cfg)
        vals.append(float((1.0 - a).mean()))
    return float(np.mean(vals)) if vals else 0.0


def run_clip(name, n_max=int(os.environ.get("R13_NMAX", "40"))):
    t0 = time.time()
    lr_path = os.path.join(PROBE, f"{name}_lr.mp4")
    gt_all = np.load(os.path.join(PROBE, f"{name}_gt.npy"))
    frames = derisk.decode_lr_and_mvs(lr_path)
    n = min(len(frames), len(gt_all), n_max)
    frames = frames[:n]
    gt = [gt_all[i] for i in range(n)]
    h_lr, w_lr = frames[0][1].shape[:2]
    h_hd, w_hd = gt[0].shape[:2]
    types = "".join(f[0] for f in frames)
    print(f"[{name}] n={n} {w_lr}x{h_lr} -> {w_hd}x{h_hd} types={types}")

    # ---- SR caches (shipping quality mode: x4plus per-frame + R8-E3 beta blend) ----
    srmod.load_model("realesrgan")
    srmod.load_model("realesrgan-x4plus")
    compact = {i: srmod.upscale_to(frames[i][1], w_hd, h_hd, model="realesrgan")
               for i in range(n)}
    perframe = {i: srmod.upscale_to(frames[i][1], w_hd, h_hd, model="realesrgan-x4plus")
                for i in range(n)}
    derisk.blend_anchor_cache(perframe, compact, BETA)     # in-place, as pipeline_api does
    bicubic = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
               for i in range(n)}
    print(f"[{name}] SR caches built ({time.time()-t0:.1f}s)")

    # ---- reconstructions ----
    recons = {}
    timings = {}
    t = time.time()
    _, Rb = derisk.reconstruct(frames, None, SCALE, True, OCC, perframe, set(),
                               backend="numpy", collect_metrics=False)
    timings["baseline_hard"] = (time.time() - t) / n * 1000.0
    recons["baseline_hard"] = {i: Rb[i]["recon"] for i in Rb}
    hard_fallback = float(np.mean([Rb[i]["hole_frac"] for i in Rb if not Rb[i]["is_anchor"]]))

    dist = {}
    for vname, (kind, cfg) in VARIANTS.items():
        t = time.time()
        recons[vname] = run_gated(frames, perframe, kind, cfg)
        timings[vname] = (time.time() - t) / n * 1000.0
        dist[vname] = distrust_stats(frames, kind, cfg)
    recons["perframe_sr"] = perframe
    if not os.environ.get("R13_LEAN"):
        recons["bicubic"] = bicubic

    # ---- metrics: vs TRUE GT and vs the per-frame SR reference ----
    gt_m = [_mw(g) for g in gt]
    pf_m = [_mw(perframe[i]) for i in range(n)]
    gt_t = [_tw(g) for g in gt]
    pf_t = [_tw(perframe[i]) for i in range(n)]

    res = dict(name=name, n=n, types=types, lr_hw=[h_lr, w_lr], hd_hw=[h_hd, w_hd],
               hard_fallback_frac=hard_fallback, distrust=dist, ms_per_frame=timings,
               metrics={})
    for rn in recons:
        seq = [recons[rn][i] for i in range(n)]
        sm = [_mw(x) for x in seq]
        st = [_tw(x) for x in seq]
        row = dict(
            lpips_gt=float(np.mean([M.lpips_dist(a, b) for a, b in zip(sm, gt_m)])),
            dists_gt=float(np.mean([ME.dists(a, b) for a, b in zip(sm, gt_m)])),
            psnr_gt=float(np.mean([M.psnr(a, b) for a, b in zip(seq, gt)])),
            tof_gt=M.tof(st, gt_t),
            lpips_pf=float(np.mean([M.lpips_dist(a, b) for a, b in zip(sm, pf_m)])),
            dists_pf=float(np.mean([ME.dists(a, b) for a, b in zip(sm, pf_m)])),
            tof_pf=M.tof(st, pf_t),
            dF=dF(seq),
        )
        res["metrics"][rn] = row
        print(f"  {rn:15s} LPIPS_gt={row['lpips_gt']:.4f} DISTS_gt={row['dists_gt']:.4f} "
              f"tOF_gt={row['tof_gt']:.4f} | LPIPS_pf={row['lpips_pf']:.4f} "
              f"tOF_pf={row['tof_pf']:.4f} | dF={row['dF']:.3f}"
              + (f" distrust={dist[rn]:.3f}" if rn in dist else ""))
    print(f"[{name}] hard-fallback={100*hard_fallback:.2f}%  "
          f"ms/frame {json.dumps({k: round(v, 1) for k, v in timings.items()})}  "
          f"({time.time()-t0:.0f}s)")
    return res


def main():
    manifest = json.load(open(os.path.join(PROBE, "manifest.json")))
    names = sys.argv[1:] or [m["name"] for m in manifest]
    os.makedirs(os.path.join(OUT, "phase2"), exist_ok=True)
    for nm in names:
        r = run_clip(nm)
        json.dump(r, open(os.path.join(OUT, "phase2", f"{nm}.json"), "w"), indent=2)
        print(f"wrote {OUT}/phase2/{nm}.json")


if __name__ == "__main__":
    sys.exit(main())
