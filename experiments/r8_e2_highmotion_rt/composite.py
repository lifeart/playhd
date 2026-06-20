#!/usr/bin/env python3
"""R8-E2 step 4: COMPOSITE temporal test of the unsharp fallback fill through the real warp.

fill_quality.py (isolated, full-reference) showed an unsharp-mask fill beats bicubic on
PSNR/SSIM/LPIPS at truth-matching dF. This validates it through the DEPLOYED warp path on the
REAL codec LR (no synthetic downscale), measuring exp2's headline temporal metrics:
  * tOF (Farneback EPE of recon vs decoded LR) -- bicubic is the tOF-optimal baseline; the test
    is whether unsharp RAISES it (does the added HF shimmer in the fallback band under motion?).
  * band-localized dF -- mean |R_t - R_{t-1}| restricted to the union fallback band (where the fill
    is actually read); the precise flicker the fill injects.
  * eff-fallback% (unchanged across fills -- same anchors/threshold; sanity).
Warp-only methodology (exp2): SR net runs ONCE/window to cache compact-SR; each fill is a per-frame
cache choice for NON-ANCHORS (read only at fallback pixels), reconstructed warp-only (zero SR re-run).

Fills compared (non-anchor cache; anchors always compact-SR, as deployed @ thresh 0.50):
  bicubic (deployed) | unsharp (bicubic + 0.5*(bic-blur)) | compactSR (escalate-all upper bound).
Also times the unsharp op itself (HD 1280x640) -> the real-time cost.

READ-ONLY imports of prototype/ + server/.
"""
import gc
import json
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

import derisk as D    # noqa: E402
import sr as SR       # noqa: E402
import anchor_sr as A # noqa: E402
import torch          # noqa: E402

CLIP = os.path.join(_REPO, "sample.mp4")
N = 48
OCC = "reactive"
SCALE = 2
WINDOWS = [("A(0)", 0), ("H2(2352)", 2352)]


def _free():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def unsharp_hd(bic_u8, amount=0.5, sigma=1.0):
    blur = cv2.GaussianBlur(bic_u8, (0, 0), sigma)
    return cv2.addWeighted(bic_u8, 1.0 + amount, blur, -amount, 0)


def tof_vs_lr(recon, frames, w_lr, h_lr):
    sm = (w_lr, h_lr)
    seq = [cv2.resize(recon[i], sm) for i in range(len(frames))]
    lr = [frames[i][1] if frames[i][1].shape[1::-1] == sm else cv2.resize(frames[i][1], sm)
          for i in range(len(frames))]
    return D.tof(seq, lr)


def reconstruct_warp(frames, cache):
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    t0 = time.perf_counter()
    _, R = D.reconstruct(frames, None, SCALE, True, OCC, cache, set(),
                         backend="torch", collect_metrics=False, download_output=True)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return R, time.perf_counter() - t0


def band_dF(recon, R, frames, w_lr, h_lr):
    """Mean |R_t - R_{t-1}| restricted to the fallback band (mask) of either frame, downscaled to
    LR. Isolates the flicker the FILL injects (the band is where the fills differ)."""
    sm = (w_lr, h_lr)
    seq = [cv2.resize(recon[i], sm).astype(np.float32) for i in range(len(frames))]
    masks = []
    for i in range(len(frames)):
        m = R[i].get("mask")
        if m is None:
            masks.append(np.zeros((h_lr, w_lr), bool))
        else:
            mm = m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
            masks.append(cv2.resize(mm.astype(np.uint8), sm, interpolation=cv2.INTER_NEAREST) > 0)
    vals = []
    for t in range(1, len(frames)):
        band = masks[t] | masks[t - 1]
        if band.any():
            vals.append(float(np.abs(seq[t] - seq[t - 1])[band].mean()))
    return float(np.mean(vals)) if vals else 0.0


def main():
    SR.load_model("realesrgan")
    out = {"config": dict(N=N, occ=OCC, scale=SCALE), "windows": {}}
    for label, start in WINDOWS:
        frames = D.decode_lr_and_mvs(CLIP, start, N)
        h_lr, w_lr = frames[0][1].shape[:2]
        w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
        anchors, backbone = A.anchor_indices(frames)

        compact = D.build_perframe_cache(frames, w_hd, h_hd, "realesrgan")
        bic = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC) for i in range(N)}
        # time the unsharp op (HD) -> per-non-anchor real-time cost
        t_us = []
        uns = {}
        for i in range(N):
            t0 = time.perf_counter()
            uns[i] = unsharp_hd(bic[i])
            t_us.append((time.perf_counter() - t0) * 1000.0)
        unsharp_ms = round(float(np.median(t_us)), 3)

        rec = {}
        caches = {
            "bicubic": {i: (compact[i] if i in anchors else bic[i]) for i in range(N)},
            "unsharp": {i: (compact[i] if i in anchors else uns[i]) for i in range(N)},
            "compactSR_all": {i: compact[i] for i in range(N)},
        }
        for name, cache in caches.items():
            R, dt = reconstruct_warp(frames, cache)
            recon = {i: R[i]["recon"] for i in range(N)}
            tof = tof_vs_lr(recon, frames, w_lr, h_lr)
            bdf = band_dF(recon, R, frames, w_lr, h_lr)
            nonanchor = [i for i in range(N) if i not in anchors]
            eff_fb = float(np.mean([(0.0 if name == "compactSR_all" else float(R[i]["hole_frac"]))
                                    for i in nonanchor]))
            rec[name] = dict(tof=round(tof, 4), band_dF=round(bdf, 3),
                             eff_fallback_pct=round(100 * eff_fb, 3),
                             recon_ms=round(1000 * dt / N, 1))
            del R, recon
            _free()
        rec["unsharp_op_ms_per_frame"] = unsharp_ms
        out["windows"][label] = rec

        print(f"\n=== {label}  anchors={sorted(anchors)} ===")
        print(f"   {'fill':>14}{'tOF':>9}{'bandDF':>9}{'effFb%':>8}{'reconMs':>9}")
        for name in ("bicubic", "unsharp", "compactSR_all"):
            r = rec[name]
            print(f"   {name:>14}{r['tof']:>9.4f}{r['band_dF']:>9.3f}"
                  f"{r['eff_fallback_pct']:>8.2f}{r['recon_ms']:>9.1f}")
        print(f"   unsharp op cost: {unsharp_ms} ms/frame (HD {w_hd}x{h_hd}, CPU cv2)")
        del compact, bic, uns
        _free()

    with open(os.path.join(_HERE, "composite.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(_HERE, 'composite.json')}")


if __name__ == "__main__":
    main()
