# R3-E1: MV-reuse motion-compensated frame interpolation — VERDICT: GO (optional "smooth 2× fps")

## What I built
Prototype reuses `prototype/derisk.py` READ-ONLY: `build_lr_flow` (codec MV → dense LR fetch-flow),
`warp_lr`/`warp_hd`, `occlusion_mask_lr`, `psnr/ssim/tof` — **no new optical flow**. A midpoint is
synthesized by warping the earlier frame forward and the later backward by the codec MVs, then
occlusion-aware (intra-hole) blending.

## Honest protocol
No real frame exists at a true half-step → PSNR/SSIM measured against an **exact held-out real frame at
LR (640×320, decoded = ground truth)**. `fwdbwd` (all-P) reconstructs real `o` from `o-1`(+MV) and
`o+1`(−MV); `bidir` (B-pyramid) reconstructs a real B from its real anchors via its own source≷0 MVs
(= playhd PASS-2). Per-warp motion here is ~one FULL codec step ≈ 2× the deployment half-step → reported
numbers are a **conservative lower bound**. HD quality inferred (LR is clean GT); cost measured at HD.

## Quality (LR GT; PSNR dB / SSIM; tOF lower=truer)
| window | n | dup | linear-blend | **MV-blend** | tOF dup / lin / blend |
|---|---|---|---|---|---|
| talking-head (start 5000, ~0.6px) | 24 | 27.96/.895 | 31.09/.928 | **40.03/.9914** | 0.518 / 0.431 / **0.102** |
| moderate (start 12000, ~1px) | 29 | 18.98/.852 | 22.95/.909 | **30.60/.9882** | 1.63 / 1.06 / **0.243** |
| high-motion coherent (start 30000, ~4.5–7px) | 14 | 17.78/.815 | 22.02/.877 | **29.38/.9547** | 5.19 / 3.64 / **2.18** |
| intro chaotic (start 0, ~10px, 9dB raw) | 14 | 9.16/.773 | 11.74/.724 | **15.36/.8713** | 14.3 / 17.1 / **9.76** |

MV-blend wins on PSNR, SSIM, AND tOF in every window: **+3.6 to +8.9 dB** over the best trivial baseline,
tOF cut ~2–4×. Visual (`halfstep_012_vs_linear.png`): linear blend ghosts on a fast hand; MV-interp
places it at a single coherent intermediate position. (The "start 0" window is a chaotic title intro,
near-uninterpolatable for any method — the coherent stress case is start 30000.)

## Cost (1920×960 HD, scale 3)
**torch/MPS ~17 ms per inserted frame** (2× `warp_hd` + blend; zero SR, zero new flow). Base recon hot
loop is ~38–40 ms/real-frame → an interpolated frame is **~2.3× cheaper than a real one** → **2× output
fps for ~+42% compute**. NOT free real-time 50 fps (the base pipeline already saturates one GPU at 25 fps)
→ ships as an optional smooth/render mode.

## Where it breaks (measured)
1. Scene cuts / chaotic content → ~15 dB (but linear/dup also collapse; MV still wins). Guard: intra-hole
   fraction >~0.5 or low MV coverage → fall back to frame duplication.
2. The inverse (−MV backward) warp is the weak direction (codec MVs are RD-optimized backward predictors,
   GOTCHA #6, not invertible); the blend's hole-routing rescues it; the half-step makes it less broken.
3. **The project's full Ruder/reactive occlusion mask HURTS interpolation** (29.4→27.8 dB high motion): it
   over-flags large-motion blocks → routes them to ghosting linear blend. Interp MUST use **intra-hole
   routing only**, keep warping in high-residual regions.
4. High intra-hole fraction at high motion (54%) — bidirectional routing fills from the complementary
   direction (why blend beats each single direction by ~+7 dB).

## Integration sketch (GO) — output-only pass, never feeds the reference chain
After `reconstruct()` makes HD recon `R[t]`,`R[t+1]`:
1. `fx,fy = build_lr_flow(frames[t+1][2], h_lr, w_lr, want="past")` (reuse the field already built during
   recon of t+1 — cache it; zero new compute).
2. `wf = gpu_ops.warp_hd(R[t], 0.5*fx, 0.5*fy, scale)`; `wb = gpu_ops.warp_hd(R[t+1], -0.5*fx,-0.5*fy, scale)`.
3. Blend `0.5*(wf+wb)`, `where(intra-hole both → linear)` (intra-hole fallback ONLY — not the full Ruder mask).
4. Emit `R[t], mid, R[t+1], …` at 2× rate, behind `--smooth/--interp-2x`; auto-disable on intra-hole >0.5.
Cost slot ~17 ms/inserted-frame on MPS, between `reconstruct` and the encode sink.

## Executive summary
MV-reuse interpolation works and is cheap: warp each neighbor by the codec MVs we already extract +
occlusion-aware (intra-hole) blend beats frame-dup / linear-blend by **+3.6 to +8.9 dB PSNR / +0.06–0.10
SSIM**, cuts tOF 2–4× across talking-head, moderate, and genuine high-motion (verified vs exact held-out
real frames + visually removes the linear ghost), at **~17 ms/inserted-frame on MPS** (~2.3× cheaper than
a real frame → 2× fps for ~+42% compute). **GO as an optional "smooth 2×" output**, not free real-time 50
fps. Respect two ship-blockers: intra-hole routing only (the full mask re-introduces ghosting), and a
scene-cut guard (intra-hole >0.5 → duplication).
