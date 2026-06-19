#!/usr/bin/env python3
"""Empirically determine, per frame, the true single-step reference: warp the candidate
reference LR_{o+1} by its INVERSE past-MV field and see if it reconstructs a held-out
predecessor LR_o (high PSNR => frame o+1's MVs reference o, i.e. a clean single display
step). This validates MV semantics instead of assuming them, and tells us which held-out
triplets are usable in each window. Read-only import of derisk."""
import os
import sys
import numpy as np

PROTO = os.path.join(os.path.dirname(__file__), "..", "..", "prototype")
sys.path.insert(0, PROTO)
import derisk  # noqa: E402

CLIP = os.path.join(os.path.dirname(__file__), "..", "..", "sample.mp4")


def inv_warp(lr_ref, mvs, h, w):
    """Reconstruct the frame REFERENCED by lr_ref's MVs: warp lr_ref by the negated past flow."""
    fx, fy = derisk.build_lr_flow(mvs, h, w, want="past")
    return derisk.warp_lr(lr_ref, -fx, -fy), np.isfinite(fx).mean()


def probe(start, n, tag):
    frames = derisk.decode_lr_and_mvs(CLIP, start, n)
    h, w = frames[0][1].shape[:2]
    print(f"\n=== {tag}: types={''.join(f[0] for f in frames)} ===")
    print("  o  type  PSNR(inv-warp(o) vs o-1)  vs o-2   vs o-3   past-cov  -> ref@")
    for o in range(3, len(frames)):
        pt, lr_o, mvs = frames[o]
        if mvs is None or len(mvs) == 0:
            continue
        recon, cov = inv_warp(lr_o, mvs, h, w)
        ps = []
        for d in (1, 2, 3):
            ps.append(derisk.psnr(recon, frames[o - d][1]))
        best = int(np.argmax(ps)) + 1
        print(f"  {o:2d}  {pt}    {ps[0]:6.2f}                {ps[1]:6.2f}  {ps[2]:6.2f}    "
              f"{cov:4.2f}    -> {best if max(ps)>20 else '?'} "
              f"({'CLEAN-1step' if best==1 and ps[0]>22 else 'distant/intra'})")


if __name__ == "__main__":
    probe(0, 16, "HIGH-MOTION")
    probe(5000, 24, "TALKING-HEAD")
