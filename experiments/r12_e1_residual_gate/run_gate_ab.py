#!/usr/bin/env python3
"""
R12-E1 A/B: codec-residual / fwd-bwd RELIABILITY GATE vs the shipping HARD occlusion
fallback (== derisk.reconstruct, gate_mode='hard'), on a JELLY (high-motion) window and
a CALM window.

Setup (real footage, no HD GT -> use a decoupled arbiter, per team convention R10):
  * anchors + fresh-detail fallback = COMPACT SR (realesr-general-x4v3)  [the real-time tier].
  * REFERENCE for LPIPS/DISTS = x4plus per-frame SR (quality ceiling). Decoupled from the
    fallback model so the gate cannot trivially "win" by copying its own fallback source.
  * BASELINE = derisk.reconstruct hard occlusion fallback  (recon[occ]=perframe[occ]).
  * GATED    = reconstruct_gated soft per-pixel reliability blend, tau/FB variants.

Metrics per frame/window:
  LPIPS(recon, x4ref)  DISTS(recon, x4ref)   [vs quality ceiling; lower=closer, oversmooth guard]
  tOF(recon, bicubic-LR)   [flow vs TRUE content motion; JELLY = HF wobble => higher; lower=better]
  tOF(recon, x4ref)        [temporal deviation from fresh SR seq]
  dF(recon)                [raw temporal energy; context]
Parity: reconstruct_gated(hard) is asserted byte-identical to derisk.reconstruct.

Run:  python run_gate_ab.py <window_name>     # window_name in {jelly, calm}
      python run_gate_ab.py all
"""
import os
import sys
import json
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r5_e2_quality"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r8_e4_metric_triangulation"))
sys.path.insert(0, os.path.join(_HERE, "patch_src"))

import derisk                       # noqa: E402
import sr as srmod                  # noqa: E402
import metrics as M                 # noqa: E402  lpips_dist, tof, psnr
import metrics_extra as ME          # noqa: E402  dists (pyiqa)
import gated_recon as gr            # noqa: E402

CLIP = os.path.join(_ROOT, "web_spike", "sd600.mp4")
SCALE = 3                            # 640x320 -> 1920x960 (FullHD-ish)
METRIC_W = 768                       # LPIPS/DISTS computed at 768x384 (CPU tractable)
TOF_W = 960                          # tOF Farneback at 960x480 (half res, matches propagation_ab)

WINDOWS = {
    "jelly": dict(start=582, n=40),   # I-anchor at 582; mv~6-7.7 (high motion, jelly-prone)
    "calm":  dict(start=437, n=40),   # I-anchor at 437; mv~1.1 (talking-head calm)
}

# soft-gate variants. tau_res=16 == the hard threshold's decision point; JELLY is caused
# by SUB-threshold MV misalignment, so lower tau catches the drift the hard switch keeps.
VARIANTS = {
    "gate_res_t16":   gr.GateCfg(tau_res=16.0, s_res=6.0, use_fb=False),
    "gate_res_t10":   gr.GateCfg(tau_res=10.0, s_res=5.0, use_fb=False),
    "gate_res_t7":    gr.GateCfg(tau_res=7.0,  s_res=4.0, use_fb=False),
    "gate_resfb_t10": gr.GateCfg(tau_res=10.0, s_res=5.0, use_fb=True, tau_fb=1.5, s_fb=0.75),
}


def _mw(img):
    """Downscale to METRIC_W-wide (INTER_AREA) for LPIPS/DISTS."""
    h, w = img.shape[:2]
    return cv2.resize(img, (METRIC_W, int(round(h * METRIC_W / w))), interpolation=cv2.INTER_AREA)


def _small_rgb(img):
    """Downscale to TOF_W-wide RGB. M.tof does the RGB->gray Farneback conversion itself."""
    h, w = img.shape[:2]
    return cv2.resize(img, (TOF_W, int(round(h * TOF_W / w))), interpolation=cv2.INTER_AREA)


def _luma(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def dF(seq):
    return float(np.mean([np.mean(np.abs(_luma(seq[t]) - _luma(seq[t - 1])))
                          for t in range(1, len(seq))]))


def perceptual(recon_seq, ref_seq):
    lp = [M.lpips_dist(_mw(r), _mw(g)) for r, g in zip(recon_seq, ref_seq)]
    ds = [ME.dists(_mw(r), _mw(g)) for r, g in zip(recon_seq, ref_seq)]
    return float(np.mean(lp)), float(np.mean(ds))


def run_window(name):
    cfg = WINDOWS[name]
    t0 = time.time()
    frames = derisk.decode_lr_and_mvs(CLIP, start_frame=cfg["start"], max_frames=cfg["n"])
    N = len(frames)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    types = "".join(f[0] for f in frames)
    print(f"[{name}] decoded {N} frames ({w_lr}x{h_lr} -> {w_hd}x{h_hd})  types={types}")

    # ---- SR caches: compact (anchor+fallback) + x4plus (reference) ----
    srmod.load_model("realesrgan")
    srmod.load_model("realesrgan-x4plus")
    compact = {}
    x4ref = {}
    for i in range(N):
        compact[i] = srmod.upscale_to(frames[i][1], w_hd, h_hd, model="realesrgan")
        x4ref[i] = srmod.upscale_to(frames[i][1], w_hd, h_hd, model="realesrgan-x4plus")
    print(f"[{name}] SR caches built ({time.time()-t0:.1f}s)")

    # bicubic-LR (true-motion reference for tOF) + fresh-compact-per-frame (context)
    bic = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC) for i in range(N)}

    # ---- reconstructions ----
    recons = {}
    # baseline: derisk hard occlusion fallback (== shipping)
    _, Rbase = derisk.reconstruct(frames, None, SCALE, True, "full", compact, set(), backend="numpy")
    recons["baseline_hard"] = {i: Rbase[i]["recon"] for i in range(N)}
    # parity check: reconstruct_gated(hard) must equal derisk
    hard = gr.reconstruct_gated(frames, SCALE, compact, use_residual=True,
                                occ_mode="full", gate_mode="hard")
    parity = max(float(np.max(np.abs(recons["baseline_hard"][i].astype(np.int32) - hard[i].astype(np.int32))))
                 for i in range(N))
    print(f"[{name}] PARITY reconstruct_gated(hard) vs derisk: max|d|={parity:.1f} codes")
    # gated variants + distrust fraction
    distrust = {}
    for vname, vcfg in VARIANTS.items():
        recons[vname] = gr.reconstruct_gated(frames, SCALE, compact, use_residual=True,
                                             occ_mode="full", gate_mode="soft", cfg=vcfg)
    # context anchors
    recons["compact_perframe"] = {i: compact[i] for i in range(N)}   # fresh compact every frame
    recons["bicubic"] = bic

    # ---- distrust fraction (mean a<0.5) for the reported gate ----
    def distrust_frac(vcfg):
        fr = []
        bb = derisk.backbone_indices(frames)
        for i in bb:
            p = max([b for b in bb if b < i], default=None)
            if frames[i][0] == "I" or p is None:
                continue
            fx, fy = derisk.build_lr_flow(frames[i][2], h_lr, w_lr, want="past")
            a = gr.reliability_lr(fx, fy, frames[i][1], frames[p][1], vcfg)
            fr.append(float((a < 0.5).mean()))
        return float(np.mean(fr)) if fr else 0.0

    # ---- metrics ----
    ref_seq = [x4ref[i] for i in range(N)]
    bic_seq = [bic[i] for i in range(N)]
    g_ref_tof = [_small_rgb(x) for x in ref_seq]
    g_bic_tof = [_small_rgb(bic[i]) for i in range(N)]

    results = {"name": name, "start": cfg["start"], "n": N, "types": types,
               "parity_max_codes": parity, "metrics": {}}
    for rn, rseq_d in recons.items():
        rseq = [rseq_d[i] for i in range(N)]
        lp, ds = perceptual(rseq, ref_seq)
        g_cand = [_small_rgb(x) for x in rseq]
        tof_bic = M.tof(g_cand, g_bic_tof)
        tof_ref = M.tof(g_cand, g_ref_tof)
        row = dict(lpips=lp, dists=ds, tof_truemotion=tof_bic, tof_vs_x4=tof_ref, dF=dF(rseq))
        if rn in VARIANTS:
            row["distrust_frac"] = distrust_frac(VARIANTS[rn])
        results["metrics"][rn] = row
        print(f"  {rn:16s} LPIPS={lp:.4f} DISTS={ds:.4f} tOF_true={tof_bic:.3f} "
              f"tOF_x4={tof_ref:.3f} dF={row['dF']:.3f}"
              + (f" distrust={row['distrust_frac']:.3f}" if 'distrust_frac' in row else ""))

    # ---- save reconstructions for PNGs (pick highest-hard-occ frame) ----
    occfr = []
    bb = derisk.backbone_indices(frames)
    for i in range(N):
        m = None
        if frames[i][0] in ("P",) and i in Rbase and Rbase[i].get("mask") is not None:
            m = Rbase[i]["mask"]
        occfr.append((float(m.mean()) if m is not None else 0.0, i))
    hi = max(occfr)[1] if occfr else N // 2
    np.savez_compressed(os.path.join(_HERE, "out", f"frames_{name}.npz"),
                        idx=hi,
                        lr=frames[hi][1], bic=bic[hi], compact=compact[hi], x4=x4ref[hi],
                        baseline=recons["baseline_hard"][hi],
                        gate=recons["gate_res_t10"][hi],
                        gate_fb=recons["gate_resfb_t10"][hi])
    results["png_frame"] = hi
    print(f"[{name}] saved frames npz (frame {hi})  total {time.time()-t0:.1f}s")
    json.dump(results, open(os.path.join(_HERE, "out", f"results_{name}.json"), "w"), indent=2)
    return results


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(WINDOWS) if which == "all" else [which]
    for nm in names:
        run_window(nm)
