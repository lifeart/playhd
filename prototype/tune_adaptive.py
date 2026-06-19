#!/usr/bin/env python3
"""Step-7 adaptive-mask tau tuning. For each window, sweep the fwd-bwd trigger threshold and
report fire-rate + fallback% so we can pick a tau that fires rarely on talking-head (where
reactive == full quality) and often on high-motion (where fwd-bwd earns its cost). Fallback% and
fire-rate are SR-independent (the mask is computed at LR), so this uses bicubic SR for speed."""
import numpy as np
import derisk
from derisk import PROF

WINDOWS = {"A_highmotion": 0, "C_talkinghead": 5000}
TAUS = [0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 1.0]


def run_window(name, start):
    frames = derisk.decode_lr_and_mvs("../sample.mp4", start, 48)
    h_lr, w_lr = frames[0][1].shape[:2]
    scale = 4
    pf = derisk.build_perframe_cache(frames, w_lr*scale, h_lr*scale, "bicubic")
    types = "".join(f[0] for f in frames)
    print(f"\n=== window {name} (start={start}) types={types} ===")
    PROF.reset(enabled=False)

    def recon(occ, tau=None):
        if tau is not None:
            derisk.ADAPTIVE_TAU = tau
            import gpu_ops as G
            G.ADAPTIVE_TAU = tau
        rows, R = derisk.reconstruct(frames, None, scale, True, occ, pf, set(), backend="torch")
        hf = [r["hole_frac"] for r in rows]
        tofr = derisk._tof_from_R(frames, None, R, w_lr, h_lr)
        return (100*np.mean(hf), 100*max(hf), tofr["prop_vs_lr"],
                derisk.MASK_FIRES[0], derisk.MASK_FIRES[1])

    fbm, fbx, ftof, _, ntot = recon("full")
    rfm, rfx, rtof, _, _ = recon("reactive")
    print(f"  full     : fallback {fbm:5.2f}/{fbx:5.2f}%  tOF(prop/LR) {ftof:.4f}  (fwdbwd {ntot}/{ntot})")
    print(f"  reactive : fallback {rfm:5.2f}/{rfx:5.2f}%  tOF(prop/LR) {rtof:.4f}  (fwdbwd 0/{ntot})")
    print(f"  {'tau':>6}{'fire%':>8}{'fallback% mean/max':>22}{'tOF(prop/LR)':>14}")
    for tau in TAUS:
        fm, fx, tof, fires, tot = recon("adaptive", tau)
        print(f"  {tau:>6.2f}{100*fires/tot:>7.0f}%{fm:>13.2f} /{fx:>7.2f}{tof:>14.4f}")


if __name__ == "__main__":
    for name, start in WINDOWS.items():
        run_window(name, start)
