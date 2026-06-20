# R6-E3: SR-quality levers — beating compact-alone on LPIPS

**Protocol** (R5-E2 degrade-and-restore): SD frame (640×320) = pseudo-HD GT → `real` degrade 2×
(codec-soften + JPEG q40 + noise) → restore → score vs GT. **Lead = real measured LPIPS (AlexNet)**;
var-Lap NR secondary only. Two windows × n=6: talking-head (start 5000, GT var-Lap 133) + detailed
(start 30000, GT var-Lap 731). GPU shared → ratios. Code: `experiments/r6_e3_srlever/`.

## Results (LPIPS↓ headline; cost = × a single compact pass)
| Lever | TH LPIPS | DET LPIPS | Δ vs compact | var-Lap (TH/DET) | cost | verdict |
|---|---|---|---|---|---|---|
| **compact-alone (baseline)** | 0.1069 | 0.0666 | — | 91 / 1057 | **1.0×** | baseline |
| compact TTA-4 | 0.1101 | 0.0662 | +3.0% / −0.6% | 84 / 1024 | 2.3× | FAIL (smooth content hurt) |
| compact TTA-8 | 0.1110 | 0.0656 | +3.8% / −1.5% | 85 / 1018 | 4.7× | FAIL |
| x4plus-alone | 0.1196 | 0.0626 | +11.9% / −6.0% | 247 / 1051 | 7.6× | mixed (TH worse) |
| x4plus TTA-4 | 0.1036 | 0.0614 | −3.1% / −7.8% | 135 / 991 | 31× | wins but expensive |
| x4plus TTA-8 | 0.1068 | 0.0621 | −0.1% / −6.8% | 123 / 987 | 73× | no gain over TTA-4 |
| **blend lin g=0.5** | **0.0956** | **0.0586** | **−10.6% / −12.0%** | 128 / 1010 | **8.6×** | **WINNER** |
| blend freq-gated g=0.25 | 0.1061 | 0.0701 | −0.7% / +5.3% | 126 / 1286 | 8.6× | FAIL |
| unsharp r0.8 a0.3 | 0.1074 | 0.0720 | +0.5% / +8.1% | 121 / 1315 | 1.0× | FAIL (hallucination trap) |

Fine sweep `out = compact + g·(x4plus − compact)`: TH min at **g=0.5** (0.0982/0.0956/0.0970 for 0.3/0.5/0.7);
DET plateau g=0.5–0.7 (0.0586). **g=0.5 is the robust shared operating point.**

## Verdict
**The conservative linear blend `compact + 0.5·(x4plus − compact)` beats compact-alone by −10.6% (TH) /
−12.0% (DET) LPIPS, at 8.6× compact compute** (one compact + one x4plus-fp16 pass per anchor). Lowest LPIPS
of EVERY config on BOTH windows, with moderate var-Lap → a real aligned-detail win, not a sharpness artifact.
- **TTA:** confirmed the hallucination-cancel hypothesis on x4plus (var-Lap 247→135, LPIPS 0.1196→0.1036,
  beats compact) — but costs 31× and is dominated by the blend; TTA-8 adds nothing over TTA-4. TTA on
  COMPACT FAILS (compact barely hallucinates → averaging only blurs).
- **Blend:** linear lerp WINS; the frequency-gated variant FAILS (keeps x4plus HF at full amplitude where
  signs match → inflates var-Lap, worsens LPIPS). The benefit is scaling the WHOLE residual down (lerp), not
  spatial gating.
- **Unsharp FAILS** both windows (raises var-Lap, hurts LPIPS — textbook GOTCHA #23).
- **Why the blend beats x4plus-TTA:** both decorrelate x4plus's hallucination by averaging two estimates —
  but the blend's second estimate is the better-aligned compact net, not a flipped x4plus. Better + cheaper.

## Integration proposal (lead; interacts with R6-E1's quality-SR decision)
The blend is an **anchor output-pass**: on each anchor frame run compact SR + x4plus-fp16 SR, then
`anchor = compact + 0.5·(x4plus − compact)` (per-pixel lerp, ~free CPU); propagated by MV warp as today.
- **quality / layered** (already run x4plus-fp16 anchor, 7.6×): +1.0× = **+13% anchor cost for −10 to −12%
  LPIPS** — a strict upgrade that ALSO kills x4plus's hallucination penalty (TH x4plus-alone 0.1196 > compact
  0.1069; blend 0.0956 beats both). **Recommend adopting g=0.5 here.**
- **instant** (compact-only anchor): blend is the full 8.6× → worth it only if the anchor duty cycle (~2–12%)
  keeps amortized cost in budget; else keep compact-alone.

**Measured vs inferred:** all LPIPS/PSNR/SSIM/var-Lap measured (real AlexNet, n=6×2, `real` degrade).
Inferred: the amortization (anchor duty cycle from prior rounds) + generalization beyond 1 clip / 2 windows.
Win is consistent in sign + magnitude across two very different windows and monotone in the sweep → robust,
but n=6 on one clip is the honest scope.

## Executive summary
Compact-alone is NOT the perceptual ceiling — **`compact + 0.5·(x4plus − compact)` beats it by −10.6%/−12.0%
LPIPS** (and beats x4plus-alone), the lowest LPIPS of every config on both windows, moderate var-Lap (real
aligned detail). Best LPIPS-per-compute lever. TTA confirms the hallucination-cancel mechanism on x4plus but
costs 31×; TTA-on-compact, freq-gated blend, and unsharp all FAIL (the latter two = GOTCHA #23 live).
**Recommend the g=0.5 blend on the anchor in quality/layered (+13% over the x4plus they already run, a strict
upgrade); keep compact-alone for instant.** Synthesize with R6-E1.
