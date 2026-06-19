#!/usr/bin/env python3
"""Recompute the PRE-FIX (Step 1) reconstruction exactly, to get a clean BEFORE for both
fallback% and the new LR-consistency metric. Old behavior: build_lr_flow kept source<0 only
(want='past'), warped the DISPLAY-PREV recon, and the first frame of the window OR any
I-frame was a bicubic anchor. This is old run() minus the oracle/true-HD metrics."""
import numpy as np
import cv2
from derisk import (decode_lr_and_mvs, build_lr_flow, warp_hd, warp_lr, _add_res,
                    occlusion_mask_lr, psnr_lr_consistency)

SCALE = 3


def before_window(path, start, n, use_residual=True, occ_mode="full"):
    frames = decode_lr_and_mvs(path, start, n)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    recon_prev, lr_prev = None, None
    out = []  # (ptype, fallback_frac, lr_consistency_db)
    for pt, lr, mvs in frames:
        perframe = cv2.resize(lr, (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
        if pt == "I" or recon_prev is None:
            recon = perframe.copy()
            out.append((pt, 0.0, psnr_lr_consistency(recon, lr)))
        else:
            fx, fy = build_lr_flow(mvs, h_lr, w_lr, want="past")  # OLD: source<0 only
            res_hd = None
            if use_residual:
                pred_lr = warp_lr(lr_prev, fx, fy)
                res = lr.astype(np.float32) - pred_lr.astype(np.float32)
                res_hd = cv2.resize(res, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
            warped, intra = warp_hd(recon_prev, fx, fy, SCALE)
            if occ_mode == "full":
                occ_lr, _ = occlusion_mask_lr(fx, fy, lr, lr_prev)
                mask = cv2.resize(occ_lr.astype(np.uint8), (w_hd, h_hd),
                                  interpolation=cv2.INTER_NEAREST).astype(bool) | intra
            else:
                mask = intra
            recon = _add_res(warped, res_hd)
            recon[mask] = perframe[mask]
            out.append((pt, float(mask.mean()), psnr_lr_consistency(recon, lr)))
        recon_prev, lr_prev = recon, lr
    return out


if __name__ == "__main__":
    path = "/Users/lifeart/Repos/playhd/sample.mp4"
    for name, start in [("A", 0), ("B", 30), ("C", 5000)]:
        rows = before_window(path, start, 48)
        by = {}
        for pt, hf, lc in rows:
            by.setdefault(pt, []).append((hf, lc))
        print(f"\n=== BEFORE window {name} (start={start}) ===")
        for t in ("I", "P", "B"):
            if t not in by:
                continue
            hf = [x[0] for x in by[t]]
            lc = [x[1] for x in by[t]]
            print(f"  {t}: n={len(hf):>3}  fallback%% mean={100*np.mean(hf):6.2f} max={100*max(hf):6.2f}"
                  f"   LR-consist mean={np.mean(lc):6.2f} min={min(lc):6.2f}")
