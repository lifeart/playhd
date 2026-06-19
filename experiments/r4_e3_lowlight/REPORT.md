# R4-E3: Instant-mode low-light/noise quality cliff — FIXED

## Root cause (sharper than the R3-E2 diagnosis)
Decomposing the c3 fallback mask: **99.0% is INTRA-hole (no MV at all)**, ~0.1% reactive. The encoder,
facing heavy noise on dim low-contrast content, **intra-codes ~99% of blocks** (inter-prediction on
noise is RD-worthless) → there are *no MVs to propagate from*. The anchor-propagation premise is
**structurally void**, so the safeguard escalates a full per-frame compact-SR on all 24/24 frames for
zero benefit → the 3.9× cliff. Two assumption-flips: (1) the compact net **denoises rather than
hallucinates** here (so dropping to bicubic loses that denoising — "noise-SR hallucinates" is false on
this content); (2) **pre-denoise can't restore real-time** (denoising pixels can't create MVs — verified:
intra stays 99%, n_sr stays 24, ms/frame stays ~123).

## Results — noisy c3_lowlight (24f, fb 95%/99% intra)
| config | ms/frame | n_sr | tOF/clean | PSNR/clean | tnoise | note |
|---|---|---|---|---|---|---|
| baseline (today) | **121 (3.9×, NOT RT)** | 24 | 0.119 | 33.17 | 1.98 | full per-frame SR; blows the budget = the cliff |
| **cap0.70 (PRIMARY)** | **31 (RT ✓)** | 1 | 0.913 | 32.83 | 4.99 | bicubic floor; real-time held |
| denoise alone | 123 (NOT RT) | 24 | 0.115 | 35.71 | 1.95 | quality↑ but ZERO speed (intra-structural) |
| **cap0.70+bilat (knob)** | **32 (RT ✓)** | 1 | 0.612 | **35.67** | 3.08 | real-time + denoised floor — best balance |

## No-regression (clean + fast-motion controls)
| config | PSNR-vs-baseline | verdict |
|---|---|---|
| c4 talkinghead (fb 0.25%) cap0.70 | **361 dB (BYTE-IDENTICAL)** | cap never fires (fb≪0.70) |
| c1 fastpan (mv 35.7) cap0.70+mgate8 | **361 dB (BYTE-IDENTICAL)** | motion gate spares genuine fast-motion |

## Recommendation
**PRIMARY — motion-gated fallback-saturation cap, default ON** (`CAP=0.70`, `MOTION_GATE=8.0`): restores
real-time on noise (121→31 ms/frame, SR 24→1), BYTE-IDENTICAL on clean + fast-motion (fires only on
high-fallback + low-motion = noise). Quality on noise = bicubic floor (meets the goal: "no worse than
today's bicubic-fallback regions"). Honest cost: forgoes the per-frame SR's accidental denoising
(tnoise 1.98→4.99) — but that cost 3.9× real-time and isn't the fast tier's job.
**OPTIONAL quality knob — gated bilateral denoise, default OFF**: real-time + beats baseline fidelity
(PSNR 33.2→35.7, tnoise 4.99→3.08). OFF because denoise softens clean content (−0.13 dB) → gate to
cap-fired frames only.
**AUTO-ROUTE signal**: surface `noise_saturated` (fb-mean>0.7 AND mv-mag<8 AND intra-frac>0.8) so the UI
recommends quality mode for users wanting max denoising (only the buffered path can afford per-frame SR-denoise).

## Integration (pipeline_api.py only; anchor_sr untouched)
The cap routes through the existing `thresh_fn` hook (return `2.0` → both `build_anchor_cache` +
`patch_high_fallback` decline escalation). New consts `INSTANT_FALLBACK_SATURATION_CAP=0.70`,
`INSTANT_SAT_CAP_MOTION_GATE=8.0`; the `_motion_keyed_thresh_fn` body composes E2 (motion-keyed, OFF) +
R4-E3 (cap, ON): `if _frac(i) > CAP and _mag(i) < GATE: return 2.0`. Hardening: a more principled gate is
**intra-fraction > 0.8** (the true "nothing to propagate" signal) if the motion gate under-fires on real
low-light. Optional `INSTANT_NOISE_DENOISE=False` bilateral gate before `build_anchor_cache`.

## Executive summary
Cliff fixed. Root cause: noise → encoder intra-codes ~99% of blocks → no MVs → the safeguard wastes a
full per-frame SR per frame (121 ms/frame, 3.9× over budget). A motion-gated fallback-saturation cap
(0.70/8px) declines that escalation, restoring real-time (121→31 ms/frame, SR 24→1), byte-identical on
clean + fast-motion (safe default-ON). Quality on noise = bicubic floor (goal met). Pre-denoise can't help
speed (verified); cap + gated bilateral denoise is real-time AND beats today's fidelity (PSNR 33.2→35.7) —
a default-OFF quality knob. Max-denoise users → auto-route to quality. ≤32 ms/frame, no clean regression.
