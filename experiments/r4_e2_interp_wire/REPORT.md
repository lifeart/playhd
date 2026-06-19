# R4-E2 — Wire + verify R3-E1 MV interpolation as an optional "smooth 2×" output — PASS

Reproduces R3-E1 quality exactly through the shippable wire; both ship-blockers work; output-only,
byte-identical when OFF; produces a real in-sync 2×-fps mp4. Default-OFF landing diff delivered.
Artifacts: `interp_pass.py` (lead-landable module — READ-ONLY reuse of `derisk.build_lr_flow` +
`gpu_ops.warp_hd`), `verify.py`, `run_instant_interp.py`, `smooth2x_demo.mp4`.

## Quality (interp-vs-held-out-real, exact LR GT, through the wire's intra-hole blend)
| window | n | dup | linear-blend | **MV-blend (wire)** | tOF blend | gain vs best trivial |
|---|---|---|---|---|---|---|
| talking-head (5000) | 24 | 27.96/.895 | 31.09/.928 | **40.03/.9914** | **0.102** | **+8.94 dB / +0.063 SSIM** |
| moderate (12000) | 29 | 18.98/.852 | 22.95/.909 | **30.60/.9882** | **0.243** | **+7.65 dB / +0.080** |
| high-motion (30000) | 14 | 17.78/.815 | 22.02/.877 | **29.38/.9547** | **2.175** | **+7.36 dB / +0.078** |
| intro-chaotic (0) | 14 | 9.17/.773 | 11.74/.724 | **15.36/.8713** | **9.757** | **+3.62 dB / +0.099** |

Byte-for-byte the R3-E1 numbers — the wire's intra-hole-only routing IS the validated path. Wins PSNR,
SSIM AND tOF in every window.

## Ship-blockers + cost
- **#1 intra-hole only:** `blend_intra_hole_np` distrusts a warp ONLY at NaN-flow holes; the over-flagging
  Ruder/reactive mask is never invoked (the occf variant that regressed 29.4→27.8 dB is absent).
- **#2 scene-cut guard (intra-hole frac > 0.5 → duplicate):** cross-cut I-frame (100% hole) → exact
  duplicate (`array_equal`), no ghost; fires on exactly the >0.5 frames; within-scene low-hole →
  interpolates. PASS.
- **Cost (MPS, ms/inserted-frame):** 1920×960 ×3 → 12.7–17.4 (reproduces R3-E1 ~17); **1280×640 instant
  tier ×2 → 7.8–15.1** (cheaper). No new flow, no SR — 2 warps + blend; guard short-circuits cuts.

## End-to-end seam (faithful copy of the instant fast path)
- OFF copy == real `process_clip(instant)`: **40/40 byte-identical**.
- Exact 2×: 80==2×40 (trailing dup closes the sequence).
- **Output-only:** ON's even frames == OFF's frames 40/40 → the midpoint never altered a real frame and
  never entered `R[]` (GOTCHA #16).
- Real 2×-fps mp4: h264 1280×640, 400 frames, out_fps=50, video 8.0s / audio 8.034s synced.
- On a 200-frame mixed window: 165/200 (82.5%) genuine interpolations; 35 (17.5%) safe duplications on
  intro/cuts/unreliable-high-motion-B.

## Integration (default-OFF; lead lands)
Copy `interp_pass.py` → `server/`. `pipeline_api.py`: import it; add `INSTANT_INTERP_2X=False`,
`INTERP_CUT_THRESH=0.5`; when `smooth = fast and INSTANT_INTERP_2X`, set `out_fps = fps*2`, and in the
instant emission loop emit `interp_pass.midpoint_torch(left, recon_t, fx, fy, eff_scale, cut_thresh)`
BEFORE each real frame (left = prev chunk's last recon for i==0 else `R[i-1]['recon']`; connecting field
= frame i's codec 'past' MV, reused), carry the chunk's last recon across chunks, trailing-dup to close,
mux with `n_emit`/`out_fps`. Full diff in the agent output. OFF → `n_emit==done`, `out_fps==fps`,
real-frame grain seed unchanged → byte-identical (verified). NOT free real-time (≈halves instant throughput).

## Executive summary
MV-reuse interpolation is now shippable default-OFF code. The lead-landable `interp_pass.py` reuses the
codec MVs + HD warp we already compute to synthesize a half-step midpoint, doubling output fps as an
optional "smooth 2×" mode. Quality exactly reproduces R3-E1 (+3.6 to +8.9 dB over dup/linear-blend, tOF
2–4× lower); intra-hole-only routing + scene-cut guard both verified; output-only (byte-for-byte never
touches a real frame or `R[]`); OFF byte-identical (40/40). ~8–15 ms/inserted-frame at the 1280×640 tier.
PASS — ready to land.
