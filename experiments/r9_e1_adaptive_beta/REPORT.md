# R9-E1 — Per-clip-adaptive global β (refine R8-E3)

**Verdict: NO-GO for a default change. Per-clip-adaptive β is ≤ fixed-0.85 on every cell but only *beats* it on a narrow smooth-heavily-degraded slice (n=8, one content) and is NOT robustly calibratable from the synthetic operators → fixed β=0.85 (R8-E3) stays the robust default.** The patch is delivered default-OFF/byte-identical as a documented opt-in, NOT landed.

Lead metric = TRUE AlexNet LPIPS; DISTS corroborator; PSNR context; var-Lap never an arbiter (GOTCHA #23). Protocol = R8-E3/R6-E1 degrade-restore matrix (5 windows × 3 synthetic operators) + a real-libx264 OOD (4 CRF levels). Selector: `β = 0.85 − 0.15·s(tex_comp)·d(disag)`, a single per-clip scalar in [0.70, 0.85], pulled toward 0.70 (more compact) ONLY on smooth + heavily-degraded clips — the only cells with measured LPIPS headroom below 0.85. Both signals are CPU reductions over the two caches the quality path already holds (`perframe_cache`=x4plus, `region_gate["compact"]`=compact) → **zero extra SR**.

- `tex_comp` = mean 7×7 local-luma-std of the COMPACT HR cache (degrade-ROBUST content measure: real texture survives heavy degrade where Canny edges collapse — the falsifier that sank an LR-edge variant: texture46k|gritty edge 0.091 < talkinghead|moderate edge 0.087 yet wants the opposite β; tex_comp 17.97 vs 7.67 separates them 2×).
- `disag` = mean |heavy − compact| luma (degrade drive: x4plus over-sharpens synthetic noise/compression the compact net avoids → high disagreement; a CLEAN real-H.264 round-trip keeps both nets in agreement → low disagreement).

## Per-cell result — LPIPS↓, adaptive vs fixed-0.85 vs x4plus (n=8/cell)

| cell | β(adpt) | adpt | fixed-0.85 | x4plus | vs 0.85 |
|---|---|---|---|---|---|
| talkinghead\|moderate | 0.83 | 0.1113 | 0.1136 | 0.1247 | **WIN −0.0023** |
| talkinghead\|heavy | 0.70 | 0.1455 | 0.1556 | 0.1674 | **WIN −0.0102** |
| talkinghead\|gritty | 0.70 | 0.2140 | 0.2275 | 0.2423 | **WIN −0.0134** |
| highmotion\|{mod,heavy,gritty} | 0.85 | = | = | (0.0057–0.0097) | tie |
| texture18k\|{mod,heavy,gritty} | 0.85 | = | = | — | tie |
| texture24k\|{mod,heavy,gritty} | 0.85 | = | = | — | tie |
| texture46k\|{mod,heavy,gritty} | 0.85 | = | = | — | tie |
| talkinghead\|crf{18,26,32,40} (real OOD) | 0.85 | = | = | — | tie |
| highmotion\|crf28, texture24k/46k\|crf{26,32} (real OOD) | 0.85 | = | = | — | tie |

**Summary: 3 WIN / 20 TIE / 0 REGRESS (of 23 cells).** Adaptive meets the *strict numeric* win condition (≥ fixed-0.85 everywhere, > on the heavily-degraded face), DISTS-corroborated (talkinghead|heavy DISTS 0.1210 adpt vs 0.1308 fixed). On every textured cell **0.85 is already the per-cell optimum** (β stays 0.85 → tie); on every real-H.264 OOD cell the disagreement is low → β stays 0.85 → tie.

## Why NO-GO for default (the honest threats that make fixed-0.85 strictly more robust)
1. **The win rides on ONE smooth-face content (n=8).** No independent smooth clip confirms it; the entire measured headroom below 0.85 is the talking-head face at heavy/gritty synthetic degrade.
2. **The degrade-gate floor is NOT learnable from the synthetic operators alone.** `d` must turn on just ABOVE the real-H.264 disagreement level (~2.96); a held-out *synthetic* calibration of that floor REGRESSES on real H.264 (+0.0006 LPIPS). The numeric "0 regress" above used a hand-set gate (`t=[9,14]`, `d=[3.0,4.5]`) tuned with knowledge of the real-H.264 level — i.e. the selector is not safely auto-calibratable, which is the whole point of a no-reference estimator.
3. **Anchor-only measurement** (propagation/tOF effect unmeasured — same caveat as R8-E3, which I separately closed for the fixed β with a propagation A/B).

The fixed-0.85 default already captures the textured + real-H.264 cells optimally (ties) and is robust without a fragile, non-auto-calibratable gate. The adaptive selector's only real gain is a narrow, single-content, synthetic-heavy-degrade slice.

## Integration (delivered, NOT landed)
`adaptive_beta.patch`: `derisk.select_anchor_beta(heavy_cache, compact_cache, params)` → returns a float β fed straight into the EXISTING `blend_anchor_cache` (R8-E3, unchanged), gated on `MODE_CONFIG["quality"]["anchor_blend_adaptive"]`. `params=None` → the fixed `anchor_blend_beta` path (byte-identical to R8-E3); both None → blend skipped (byte-identical to pre-R8). Seam-verified (`verify_patch.py`). **Not landed**: per the threats above, fixed-0.85 is the more robust default and an un-auto-calibratable opt-in flag is config surface for a narrow win.

## Bottom line
**R9-E1 CONFIRMS R8-E3's fixed β=0.85 is the robust optimum.** Per-clip adaptivity helps only a narrow smooth-heavily-degraded slice and cannot be safely calibrated from synthetic degrade alone. Combined with R9-E2 (degradation-aware anchors don't beat x4plus), the quality program has reached a well-validated local optimum for the obtainable tools.

_(Experiment ran to completion — all 6 steps + patch + `verify_patch.py`; the agent hit a transient 401 before writing this file. Report synthesized by the lead from the salvaged `step{1..6}_*.json` + `adaptive_beta.patch`; numbers re-derived directly from `step6_verify.json`.)_
