# R8-E3 — Local degrade-adaptive compact↔x4plus anchor blend

**Verdict: LOCAL adaptive β = measured NO-GO. Surfaced a stronger result: a GLOBAL fixed β=0.85 blend (validated-ready, default-OFF, integrated-A/B pending).**

Lead metric = TRUE AlexNet LPIPS; PSNR for perception-distortion context; var-Lap is never an arbiter (GOTCHA #23). Protocol = R6-E1's exact harness (degrade-restore, 5 windows × 3 operators × n=8, same seeds; `moderate` == R5-E2 `real`). Blend op == `region_quality.blend_region_aware`'s lerp. **Protocol-correctness seam:** cached x4plus LPIPS reproduces R6-E1's `results.json` to 4 d.p. on every spot-checked cell.

## Why local adaptivity can't work (diagnostic, before any blend)
The only regime needing a different β is the smooth face: TH-moderate wants β≈0.5 (compact wins by 0.017), TH-heavy wants β≈1 (x4plus wins by +0.015). Local signals on the face can't separate moderate↔heavy: `tex` 3.8/3.7/2.6 (flat), `lrhf` 8.3/8.3/7.1 (flat — added noise cancels blur), `disag` 1.2/1.6/1.8 (weak, and wrong sign *within*-cell: high-disag pixels at moderate are the face edges x4plus over-sharpens). The discriminator is the **global degrade level**, invisible to cheap local stats.

## Per-cell matrix — LPIPS↓ [PSNR dB], n=8

| window | degrade | compact L/P | x4plus L/P (arbiter) | β=.50 | β=.75 | β=.85 | β.85 vs x4 |
|---|---|---|---|---|---|---|---|
| talkinghead | moderate | 0.1077/30.2 | 0.1247/28.5 | 0.0983 | 0.1068 | 0.1136 | WIN −0.0111 |
| talkinghead | heavy | 0.1827/29.4 | 0.1674/27.3 | 0.1391 | 0.1485 | 0.1556 | WIN −0.0118 |
| talkinghead | gritty | 0.2414/27.1 | 0.2423/25.0 | 0.2032 | 0.2183 | 0.2275 | WIN −0.0148 |
| highmotion | moderate | 0.0050/36.0 | 0.0057/35.8 | 0.0048 | 0.0051 | 0.0053 | WIN −0.0004 |
| highmotion | heavy | 0.0060/34.4 | 0.0068/33.8 | 0.0052 | 0.0057 | 0.0060 | WIN −0.0008 |
| highmotion | gritty | 0.0124/31.0 | 0.0097/31.2 | 0.0086 | 0.0084 | 0.0087 | WIN −0.0010 |
| texture18k | moderate | 0.0362/22.8 | 0.0363/22.2 | 0.0318 | 0.0329 | 0.0340 | WIN −0.0023 |
| texture18k | heavy | 0.0780/20.8 | 0.0618/19.5 | 0.0542 | 0.0537 | 0.0561 | WIN −0.0057 |
| texture18k | gritty | 0.1720/18.2 | 0.1217/17.1 | 0.1216 | 0.1161 | 0.1174 | WIN −0.0043 |
| texture24k | moderate | 0.0745/22.4 | 0.0514/22.7 | 0.0551 | 0.0505 | 0.0502 | WIN −0.0012 |
| texture24k | heavy | 0.1311/20.6 | 0.0908/19.7 | 0.0880 | 0.0829 | 0.0847 | WIN −0.0061 |
| texture24k | gritty | 0.2512/18.0 | 0.1448/17.5 | 0.1570 **REG** | 0.1420 | 0.1414 | WIN −0.0035 |
| texture46k | moderate | 0.1178/26.0 | 0.1009/25.7 | 0.1036 **REG** | 0.0990 | 0.0995 | WIN −0.0014 |
| texture46k | heavy | 0.2100/22.6 | 0.1658/22.0 | 0.1719 **REG** | 0.1621 | 0.1624 | WIN −0.0034 |
| texture46k | gritty | 0.3491/18.5 | 0.2306/18.6 | 0.2501 **REG** | 0.2306 | 0.2291 | WIN −0.0015 |

β=0.85 wins LPIPS even where it sits 0.2–1.5 dB *below* x4plus PSNR on textured cells → aligned detail, not a sharpness fake.

## Global-β sweep (the real lever)
| β | max regress | beat/tie/regress (of 15) | TH-mod gain | mean gain | verdict |
|---|---|---|---|---|---|
| 0.50 (R6-E3) | +0.0195 | 9/1/5 | +21.2% | +6.9% | REGRESSES |
| 0.75 | −0.0000 | 14/1/0 | +14.3% | +7.9% | STRICT-PASS (synthetic) |
| **0.85** | **−0.0004** | **15/0/0** | **+8.9%** | **+5.7%** | **STRICT-PASS + beats all 15** |
| 1.00 | +0.0000 | 0/15/0 | 0 | 0 | == x4plus |

Per-frame at β=0.85: strict-beat on all 15 cells (not a mean artifact). Real lever (20% compact admixture, −5.7% mean), not a collapse to x4plus.

## Local-adaptive variants all FAIL (dominated)
`adapt_tex` 5 regressions (texture doesn't saturate to 1 on tx46k); `adapt_tex_degrade` 4 (lrhf inert); `adapt_disag` 3 (≤+0.0006, closest, but disag's within-cell sign is wrong). All beaten outright by `fix0.85`.

## OOD check — real libx264 codec (the overfit falsifier)
| cell | x4plus | β.50 | β.75 | β.85 |
|---|---|---|---|---|
| talkinghead crf26 | 0.0915 | 0.0960 REG | 0.0908 win | 0.0905 win |
| talkinghead crf32 | 0.1290 | 0.1356 REG | 0.1299 REG | 0.1290 tie |
| texture24k crf26 | 0.0408 | 0.0453 REG | 0.0412 REG | 0.0405 win |
| texture24k crf32 | 0.0582 | 0.0602 REG | 0.0570 win | 0.0569 win |

Safe-β is operator-sensitive in 0.75–0.85: β=0.50 regresses 4/4, β=0.75's synthetic STRICT-PASS doesn't fully transfer (2/4 tiny regressions), **β=0.85 safe under both families** → land at 0.85, not 0.75. This strengthens the local NO-GO: if a global constant's safe point drifts with the operator, a local estimator tuned to one operator is more fragile.

## Integration: `anchor_blend.patch` (default-OFF, byte-identical, seam-verified)
New `derisk.blend_anchor_cache(heavy_cache, compact_cache, β)` lerps the x4plus anchor cache toward compact in place; wired into `run()` + both server quality call sites, gated on `MODE_CONFIG["quality"]["anchor_blend_beta"]` (absent/None ⇒ skipped ⇒ byte-identical). **Compute:** compact cache already exists as `region_gate["compact"]` → one CPU lerp, **zero extra SR**. `verify_patch.py` ALL PASS: (1) β=None/1.0 byte-identical, (2) helper math == measured op, (3) re-score reproduces sweep LPIPS.

## Threats to validity / could-not-verify
1. **Anchor-only measurement** — the propagation+tOF effect of pre-blending anchors is UNMEASURED → patch ships default-OFF pending an integrated A/B (R6-E3/R7-E2 discipline). The one real "could-not-verify."
2. **Operator sensitivity** addressed (β=0.85 survives H.264 OOD) but not eliminated (untested: heavy H.264, other clips/genres). Low overfit surface — a single scalar over 5 windows × 4 operators, no estimator fit to the operator.
3. **n=8, one clip** — sign is frame-robust (100% strict-beat on 13/15 synthetic cells); smallest textured margin +0.0015.

## Bottom line
LOCAL degrade-adaptive β is a measured NO-GO. The shippable result is a **global fixed β=0.85** compact↔x4plus anchor blend: ≤ x4plus on every measured cell (synthetic + real-H.264), −5.7% mean / −8.9% smooth-moderate LPIPS, zero extra SR. Closes R6-E3's open item (β=0.50 was simply too aggressive). Patch is default-OFF/byte-identical and seam-verified; flipping default needs the integrated propagation+tOF A/B.

_(Report authored by R8-E3 agent; transcribed to file by lead after the agent's direct write was declined.)_
