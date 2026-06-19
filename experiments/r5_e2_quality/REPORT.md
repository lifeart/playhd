# R5-E2: Honest perceptual quality numbers for the SR configs + grain

## Metric (honest disclosure)
**TRUE perceptual metric, not a proxy:** installed `lpips` (AlexNet, weights cached) → **LPIPS is real,
measured, and the headline.** Alongside: PSNR, SSIM, 3-scale MS-SSIM, full-reference gradient/edge-fidelity
(Sobel-mag PSNR — hallucinated edges LOWER it). var-Laplacian secondary only (GOTCHA #23); tOF temporal
secondary. VMAF not used (no python pkg; broken ffmpeg CLI — LPIPS is the better perceptual metric anyway).
**Protocol (the reference number the project lacked):** SD frames = pseudo-HD GT → degrade 2× → restore 2×
through `sr.upscale_to` → score vs GT. Two degrade operators: `clean` (INTER_AREA, OOD for Real-ESRGAN —
isolates hallucination cost) and **`real`** (codec-soften + 2× down + JPEG q40 + noise — what the nets are
built to invert; the representative test). Talking-head (start 5000) + high-motion (start 0), n=12/window.

## 1. SR-config A/B (talking-head, `real` degrade — representative)
| config | LPIPS↓ | PSNR↑ | SSIM↑ | MS-SSIM↑ | gradFid↑ | var-Lap (NR) |
|---|---|---|---|---|---|---|
| bicubic | 0.220 | 30.07 | 0.903 | 0.960 | 18.4 | 23 |
| **compact** | **0.108** | **30.26** | **0.915** | **0.965** | 18.6 | 93 |
| x4plus | 0.123 | 28.55 | 0.883 | 0.953 | 17.1 | 256 |
| x4plus-fp16 | 0.123 | 28.55 | 0.884 | 0.953 | 17.1 | 255 |

(`clean` degrade: bicubic wins — nothing to restore, nets only add misaligned HF.)

**Three honest surprises:**
1. **On real-degraded talking-head, COMPACT BEATS x4plus on every full-reference metric** (LPIPS 0.108 vs
   0.123, PSNR +1.7 dB, SSIM +0.03; also on high-motion 0.0077 vs 0.0088). x4plus synthesizes 2.7× more HF
   (var-Lap 256 vs 93) but it is *misaligned* with the true GT → costs perceptual quality at **10× the
   compute**. The heavy model is NOT a quality win on this content.
2. **Live GOTCHA #23:** x4plus is "sharper" by var-Lap at every setting yet worse by LPIPS/PSNR/SSIM. An
   NR metric would rank x4plus #1; the full-reference metric correctly ranks it below compact.
3. **SR is worth it vs bicubic — but only the perceptual metric shows it:** compact cuts LPIPS 0.220→0.108
   (**−51%**) while PSNR is identical (30.26 vs 30.07). PSNR alone said "SR ≈ bicubic"; LPIPS shows it
   halves perceived distortion. The perception-distortion gap the project never measured.

## 2. fp16 == fp32 (re-confirmed on the perceptual metric)
talking-head LPIPS(fp16,fp32) **0.00005** (PSNR 58.4), high-motion **0.00002** (62.4). Perceptually
identical. fp16 ≈ **1.23×** faster than fp32 x4plus (matches R2-E4's 1.24×). Free speedup, GO confirmed.
(Timing ratio-only, MPS shared: compact ≈1×, x4plus ≈10× compact, x4plus-fp16 ≈8×.)

## 3. Grain A/B (LPIPS↓, talking-head / real degrade)
| mode | off | low | med | high |
|---|---|---|---|---|
| instant (compact) | **0.108** | 0.184 | 0.300 | 0.419 |
| quality (x4plus) | **0.123** | 0.185 | 0.277 | 0.385 |
| high-motion (compact) | **0.008** | 0.020 | 0.078 | 0.213 |

**Opposite of the "grain masks banding helps" hypothesis** — grain monotonically HURTS every full-reference
metric (it's additive noise uncorrelated with the clean GT). Fidelity sweet spot = OFF. Grain's
banding-masking benefit is real but aesthetic/NR — invisible to (and penalized by) a fidelity-to-source
metric. If wanted aesthetically, `low` is the minimum-cost dose (~+0.07 LPIPS). Minor counter-note: on
high-motion, `low` grain reduces tOF (4.86→3.97) — a touch can stabilize the temporal metric on motion.

## Per-mode recommendation (with the number)
| mode | recommended | headline LPIPS (real, TH / HM) | rationale |
|---|---|---|---|
| **instant** | compact, grain OFF | **0.108 / 0.008** | compact is the fast AND perceptual winner; −51% LPIPS vs bicubic |
| **quality** | lean COMPACT (keep region-aware heavy only in static-detail); grain OFF | compact 0.108 vs x4plus 0.123 | x4plus does NOT beat compact here — over-hallucinates; the region-aware blend is the right hedge, lean it toward compact |
| **layered** | x4plus-fp16 for the static plate, grain OFF on plate | plate ≈ 0.123 / 0.009 | x4plus cost amortizes to ~0 on a once-computed plate + its HF doesn't flicker (tOF 0 on static) — the one place the heavy model is defensible; fp16 makes it free |

**Caveat (held honestly):** one clip, moderate `real` operator. x4plus is built for HEAVIER degradation and
may pull ahead on grittier sources / content with more recoverable fine texture. The reference metric does
not justify its 10× cost on `sample.mp4`.

## Executive summary
First true perceptual reference numbers (measured AlexNet LPIPS): instant **0.108 / 0.008**, quality
**0.108–0.123**, layered-plate **0.123 / 0.009**. **Biggest finding:** the heavy x4plus is NOT a quality
win — on real-degraded `sample.mp4` the compact model beats it on every full-reference metric at 1/10 the
compute, while being "sharper" by var-Lap (live GOTCHA #23). fp16 perceptually identical (free 1.23×). Grain
helps no reference metric (aesthetic, default OFF). PSNR rated SR ≈ bicubic; LPIPS showed SR *halves*
perceived distortion — the gap the project had never measured.
