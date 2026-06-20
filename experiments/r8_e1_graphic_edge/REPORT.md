# R8-E1 — Moving graphic/text-edge stabilization (the open V3 item)

**Verdict: the MOVING-graphic edge shimmer is REAL and robust, but NO-GO for default integration; shipped as a validated default-OFF instant hook (`INSTANT_GRAPHIC_PIN`).**

Three numbers drive it:
1. **Propagation shimmers 1.45–4.8× the per-frame-SR floor** on moving high-contrast text (registered-ΔF, motion-compensated; robust across 5 windows / text / vertical+horizontal motion; visually every letter edge wobbles in `artifacts/int2_regdiff.png`).
2. **Instant does NOT self-heal it** — reactive occlusion fallback is only **2.9%** on integer-motion text (95% propagated).
3. **But QUALITY already self-heals it for free**: the region-aware blend gives **a_lr≈0.085** on the high-motion bar → output = compact per-frame SR → reg-ΔF within 1.0–1.3× of floor. Redundant there.

The only *effective* instant fix (pin→compact-SR crop, 0.486→0.274) costs ~17 ms/graphic-frame (~24→17 fps on ticker clips); the free bicubic pin is a wash (0.471). It also trades tOF (which mildly favors propagation — it's blind to HF edge wobble). Hence default-OFF.

Integration one-liner: `instant_graphic_pin.patch` adds `INSTANT_GRAPHIC_PIN=False` + `_graphic_pin_one` in `server/pipeline_api.py` (additions-only → byte-identical OFF; motion-gated detector is byte-identical on the static USACHEV card, 0% on faces).

## Method
`sample.mp4` LR 640×320 → HD 2560×1280 (scale 4). To get the real RD-MV artifact: decode a clean window → overlay a synthetic high-contrast caption at a KNOWN sub-pixel velocity → **re-encode H.264** (PyAV/libx264 `crf20 preset medium g64 bf2` → single I + long P/B run = max propagation drift) → re-decode with `+export_mvs` → run shipped `derisk.reconstruct` (numpy, deterministic), compact `realesr-general-x4v3`. Encoder confirmed to produce RD-MVs that **OVERSHOOT** true motion (mean |MV| inside bar 4.46/5.91/9.24 px vs authored 2.0/1.7/3.3) — the "RD-optimal not true-flow" block MV.

Metric = **registered-ΔF**: consecutive |luma ΔF| on the bar after compensating the known velocity (isolates HF edge wobble from legitimate motion). Raw |ΔF| is useless here (~35 codes for all sequences, motion-dominated). Plus tOF (LR + HD-bar) and occlusion fallback%.

## Q1 — Problem is real (instant occ=reactive)
| ticker int2 (v=2.0) | reg-ΔF | LR-tOF | HDtOF-bar |
|---|---|---|---|
| per-frame compact SR (=pin target) | **0.222** | 0.282 | 0.764 |
| LR cubic | 0.228 | 0.099 | — |
| **propagation reactive (shipped)** | **0.486** | 0.235 | 0.709 |
| propagation full | 0.486 | 0.230 | 0.708 |

Propagation = 2.19× the floor (int2), 1.85× (sub17: 3.156 vs 1.707). `artifacts/int2_regdiff.png` shows every "BREAKING NEWS" letter edge wobbling under propagation vs near-black for per-frame SR. Mechanism: the NEMO residual fixes the LF/position, but the SR'd HF edge rides the warped anchor at the jittery MV position.

## Q2 — Occlusion does NOT self-heal (instant)
Reactive fallback on bar = **2.9%** (int2) / 19.1% (sub17). The "occlusion already routes edges to per-frame SR" redundancy hypothesis is **refuted for instant** — a hard edge moved by a near-exact MV has low residual and passes through.

## Q3 — Quality self-heals it (fix redundant there)
`a_lr = window_static_weight(meanmag, lo=0.2, hi=1.0)`; bar meanmag 2.7–3.5 ≫ hi=1.0 → **a_lr(bar)=0.085** → output ≈ compact per-frame SR.
| | propagation | region-aware (quality) | floor |
|---|---|---|---|
| int2 | 0.486 | **0.286** (1.29×) | 0.222 |
| sub17 | 3.156 | **1.744** (1.02×) | 1.707 |

## Refutation (robust, not a single-setup artifact)
prop/per-frame reg-ΔF ratio: **A** window12k horizontal **3.15×** (fb 6.8%); **B** vertical credits roll **4.81×** (fb 2.5%); **C** window8k fast4 **1.45×** (fb 6.3%). All >1.3.

## Q4 — Metric tension (threat to validity)
tOF (LR and HD-bar) mildly FAVORS propagation (it rewards smooth warp-field consistency and is blind to HF edge-intensity wobble). The fix lowers reg-ΔF (0.486→0.27) but raises LR-tOF (0.235→0.282) — same trade class as the shipped `INSTANT_MOTION_KEYED_FALLBACK`.

## Pin validation (`test_pin.py`)
| int2, on detected region | reg-ΔF |
|---|---|
| propagation | 0.486 |
| **pin→compact SR (effective)** | **0.274** |
| pin→bicubic (free) | 0.471 ← wash |
| floor | 0.222 |

- Free bicubic pin is a wash at realistic 70% recall (9.4% coverage). Effective fix needs the compact-SR crop.
- **STATIC USACHEV card: 0 pixels fired → byte-identical (`max|Δ|=0`)** — R1-E3 NO-GO structurally preserved (detector requires |MV|>0.6, the inverse of exp3's gate).
- FACE: 0 pixels (bimodality FP guard, motion-independent: 29.7% on bar vs 0.00% on face).
- Seam: pinned-vs-prop raw|ΔF| on bar-ring = Δ+0.000.

## Integration — `instant_graphic_pin.patch`
Instant-only. `server/pipeline_api.py`: adds `INSTANT_GRAPHIC_PIN=False` (+ thresholds) next to `INSTANT_MOTION_KEYED_FALLBACK`, helper `_graphic_pin_one(recon_t, frame, h_lr, w_lr, scale, sr_mode)` (OUTPUT-ONLY, returns a NEW tensor — never writes `R[]`), and an `if INSTANT_GRAPHIC_PIN:` call right after `recon_t = R[i]["recon"]`. Seam-verified by the agent: patch applies clean, **additions-only (0 deletions → byte-identical OFF)**, compiles, all caller symbols in scope, `_gpu_ops.img_to_dev` returns `[1,3,H,W] float 0..255` = recon_t layout, `sr.upscale` signature matches. **Cost:** detection ~1–2 ms; compact-SR crop ~17 ms/graphic-frame (~24→17 fps on persistent-ticker clips); zero on graphic-free frames.

## Threats to validity
1. tOF doesn't corroborate (favors propagation) — GO rests on registered-ΔF + visual; if tOF is the sole headline, it's a NO-GO outright.
2. Synthetic captions; opaque scrolling text is the STRONG case — the translucent lower-third was WEAK (slide-in 22.2 vs 21.9; hold 0.73 vs 0.68 ≈ static NO-GO).
3. ON-path torch seam is numpy-validated (0.274) + layout-confirmed but NOT executed in a live server (no harness here); OFF-path byte-identity is guaranteed.
4. Quality's region-aware self-heal needs *persistent* motion (high temporal-mean); a brief slide-then-hold lower-third keeps a_lr≈0.33 → mostly HEAVY during the slide-in (un-healed window, short, not isolated).
5. Compact-SR MPS jitter ~1e-3 codes ≪ the 1.45–4.8× gaps.

## Bottom line
Moving graphic edge shimmer is REAL (prop 1.45–4.8× per-frame-SR floor, not occlusion-self-healed on instant) but NO-GO for default: **quality mode already self-heals it via the region-aware blend (a_lr≈0.085)** — a quality-mode strength worth recording. Delivered as a validated default-OFF `INSTANT_GRAPHIC_PIN` instant hook (byte-identical OFF, static-card-safe).

_(Report authored by R8-E1 agent; transcribed to file by lead after the agent's direct write was declined.)_
