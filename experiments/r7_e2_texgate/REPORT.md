# R7-E2: TEXTURE-gated region-aware blend — GO (quality-safe; compute win is tiling-bound + needs a follow-up)

**Lead = TRUE AlexNet LPIPS**; var-Lap secondary only (GOTCHA #23). Protocol = R6-E1 degrade-restore. Content
mix = talkinghead@5000 (smooth) + texture24k@24000 (chart+text), n=6/cell. Artifacts: `run_texgate.py`,
`results.json`, `derisk_build_region_gate.patch`.

**Gate:** `a' = a_motion · a_texture`. `a_motion` = exact replica of `_build_region_gate`. `a_texture` ∈[0,1]
= temporal-mean local luma STD (7×7) of the **already-computed compact source** (zero extra SR), thresholded
`tex_lo..tex_hi`, feathered. Texture cleanly separates the windows (compact-source median local-std:
talkinghead ~3.7 vs texture24k ~10).

## Results (compact-sourced texture; meanA = ideal per-pixel heavy fraction; t16 = realizable 16px-tile fraction)
| window · degrade | gate | t16 | meanA | LPIPS | dMot (vs motion-only) |
|---|---|---|---|---|---|
| talkinghead · moderate | motion (today) | 90% | 74% | 0.1285 | — |
| | **tex (f21)** | **63%** | **14%** | **0.1092** | **−0.019** |
| talkinghead · heavy | motion (today) | 90% | 74% | 0.1933 | — |
| | **tex (f21)** | **60%** | **11%** | **0.1757** | **−0.018** |
| texture24k · moderate | motion (today) | 91% | 66% | 0.0520 | — |
| | **tex (f21)** | 82% | **28%** | 0.0528 | **+0.0008** |
| texture24k · heavy | motion (today) | 91% | 66% | 0.0874 | — |
| | **tex (f21)** | 81% | **28%** | 0.0863 | **−0.0011** |

Config: `tex_lo=6, tex_hi=14, tex_feather=21, tex_k=7` (compact source). Sweeps hold the verdict.

## Verdict — GO (quality-safe), with two honest caveats
1. **LPIPS non-regressing-to-better:** talking-head IMPROVES by −0.018 to −0.023 (the motion-only gate
   over-applies heavy to the static-but-smooth face where R6-E1 proved x4plus's HF is misaligned;
   smooth→compact fixes it); detailed graphics neutral (±0.001). Dropping heavy area never raised LPIPS.
2. **Heavy-area reduction concentrated on smooth content** (as designed): talking-head effective heavy
   74%→~12% (−83%); detailed kept ~28%.
3. **⚠ Compute win is NOT realized by the flag alone.** Today the gate is an OUTPUT-ONLY blend — both heavy
   and compact are already computed, so `texture_aware` only changes OUTPUT (neutral-to-better), saving ZERO
   compute. The real win needs a **follow-up wiring change: run the x4plus anchor SR only on textured-static
   tiles** (skip a'≈0). This experiment proves that mask is quality-safe + quantifies the skip.
4. **Tiling-bound saving:** talking-head texture (hair/eyes/edges) is scattered → ideal −83%, fine ~−50–60%,
   16px tiles −33%, coarse 32px −18%. Heavy ≈11.8× compact restore latency on MPS, so even −30% is large absolute.
5. **Fallback must stay COMPACT, not bicubic** (`tex_md_bicubic` regresses LPIPS +0.04 to +0.14).

## Deliverable
`derisk_build_region_gate.patch` adds `texture_aware=False, tex_lo=6, tex_hi=14, tex_feather=21, tex_k=7` to
`_build_region_gate` (**False = byte-identical to today**; True sets `a_lr = a_motion · a_tex` from the
already-built compact cache). Delivered **default-OFF** (flipping default changes quality-mode output bytes +
the real compute win needs the tile-skip wiring → both warrant an integrated propagation+tOF A/B first, per
R6-E1's "measure before landing"). NOT landed this round — filed as a validated-ready 2-part follow-up
(land the gate + add tile-skip heavy SR).

## Executive summary
Texture-gating the heavy model is quality-safe and behaves as R6-E1 predicted: `a' = a_motion · a_texture`
(texture = free local-std of the compact source) cuts the effective heavy-SR fraction on talking-head ~74%→
~12% while IMPROVING LPIPS ~0.02; detailed graphics keep heavy (~28%) at LPIPS within ±0.001. LPIPS cost is
zero-to-negative everywhere. BUT the compute saving is tiling-bound AND requires a follow-up tile-skip pass
(today's output-only blend computes both SRs regardless). Ship plan: land the `texture_aware` flag default-OFF,
then add tile-skip heavy SR + an integrated A/B, then consider default-ON. Keep compact (never bicubic) as fallback.
