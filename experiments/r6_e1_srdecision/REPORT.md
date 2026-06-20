# R6-E1: Stress-testing "compact beats x4plus" — REFUTED as a broad claim

**Lead metric = TRUE AlexNet LPIPS** (real); var-Lap NR/secondary (GOTCHA #23 respected). Artifacts in
`experiments/r6_e1_srdecision/` (`run_matrix.py`, `analyze.py`, `results.json`, `crop_texture24k_gritty.png`).

## Method (R5-E2 protocol, broadened)
SD frame = pseudo-HD GT → degrade 2× → restore via `sr.upscale_to` → LPIPS vs GT. **5 windows** (found by a
var-Lap scan of all 50,805 frames): `talkinghead@5000` (smooth, GT varLap 131), `highmotion@0` (title card,
1076), + 3 genuinely detailed broadcast windows `texture18k/24k/46k` (news headline/chart/photo, varLap
3444–4856). **3 degrade operators** escalating into x4plus's training regime: `moderate` (=R5-E2 `real`),
`heavy` (blur1.5/q25/σ4), `gritty` (2nd-order RealESRGAN-style). Identical degraded LR to every model, n=8/cell.

## Matrix — compact vs x4plus, TRUE LPIPS (↓)
| window | GT varLap | moderate (c/x) | heavy (c/x) | gritty (c/x) |
|---|---|---|---|---|
| talkinghead (smooth) | 131 | **0.108**/0.125 | 0.183/**0.167** | 0.241/0.242 tie |
| highmotion (titlecard) | 1076 | 0.0050/0.0057 tie | 0.0060/0.0068 tie | 0.0124/**0.0097** |
| texture18k | 4856 | 0.0362/0.0363 tie | 0.078/**0.062** | 0.172/**0.122** |
| texture24k (chart+text) | 4508 | 0.075/**0.051** | 0.131/**0.091** | 0.251/**0.145** |
| texture46k | 3444 | 0.118/**0.101** | 0.210/**0.166** | 0.349/**0.231** |

**Cell winners (15): x4plus 9, compact 1, tie 5.** On all 3 textured windows at heavy & gritty, **x4plus
wins 100% of 8/8 frames** (not a mean artifact). Textured Δ(compact−x4plus) = +0.013/+0.034/+0.092 — x4plus's
lead GROWS with both texture and grit (gritty texture46k: 0.231 vs 0.349, ~34% lower). Perception-distortion
split (the crux): on textured content x4plus often LOSES PSNR yet WINS LPIPS → its synthesized HF lands on
REAL GT structure; on the smooth face it produces more HF but HIGHER LPIPS → misaligned. var-Lap stays a
useless arbiter; LPIPS cleanly separates good HF (texture) from bad HF (smooth). x4plus ≈ 23× compact compute.

## Verdict: REFUTE the broad R5-E2 claim — it was an artifact of testing only the 2 LOWEST-detail windows
R5-E2 concluded "compact beats x4plus" from a smooth face + a flat title card (≈no recoverable HF, where
x4plus's prior is pure misalignment cost). That does NOT generalize. On the detailed/graphics/text/photo
content that dominates the rest of `sample.mp4` — and that a "quality" mode exists to serve — **x4plus wins
LPIPS decisively and consistently, more so under heavier (realistic) degradation.** R5-E2's own caveat
("x4plus may pull ahead on grittier sources / more recoverable texture") is CONFIRMED and is the COMMON case.

## Quality-mode SR recommendation: KEEP x4plus + region-aware (NO config change); withdraw "lean compact"
`MODE_CONFIG["quality"]` = `sr_mode="realesrgan-x4plus"`, `region_aware=True`, `fp16=True` is correct and
should stay byte-identical. The region-aware blend (heavy on static-detail, compact on moving/low-detail)
already routes each model to the regime the matrix says it wins. Switching to compact would be a measured
perceptual regression (up to +0.12 LPIPS, ~8/8 frames) on exactly quality mode's target content. fp16 stays.
x4plus stays right for the layered static plate. Keep instant=compact (its smooth/fast domain).

**Optional follow-up (OFFER, do NOT land unmeasured):** the gate keys on MOTION, but the true discriminator
is TEXTURE×degrade. A static-but-smooth region (still face/sky) gets 23× heavy SR for ~0 LPIPS gain. A
local-detail term in `derisk._build_region_gate` (heavy only where static AND high-texture) could cut
quality-mode compute at ≈zero LPIPS cost — a compute optimization gated on an integrated A/B.

## Executive summary
Compact does NOT broadly beat x4plus — R5-E2's headline was an artifact of its 2 lowest-detail windows. On
genuinely detailed content (graphics/text/charts/photos — the realistic upscaling workload), **x4plus beats
compact on TRUE LPIPS in 9/15 cells, 100% of frames on every textured heavy/gritty cell**, lead growing with
degradation (up to −34%). Clean perception-distortion win; var-Lap useless, LPIPS decisive. **Change nothing:
keep quality = x4plus + region-aware + fp16; withdraw R5-E2's "lean compact".** Optional R7 compute
optimization: texture-gate the heavy model (measure integrated before landing).
