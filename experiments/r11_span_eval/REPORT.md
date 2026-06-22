# R11 — Does 2xLiveActionV1_SPAN actually beat the compact on real H.264? (measured, not assumed)

The web_spike claim **"SPAN beats compact on LPIPS+DISTS"** came from the model card, never measured on
real codec output. R10-E1's validated harness, run on `2xLiveActionV1_SPAN` (the exact web_spike anchor):
GT = sample.mp4 256 crop; LR = 2× INTER_AREA → real libx264 CRF{27,35} → decode; restore net 2× (SPAN
native 128→256; 4× models 128→512→256); arbiter = pyiqa LPIPS(Alex)+DISTS+PSNR vs GT.

## Overall (3 windows × 3 frames × 2 CRF)

| model | LPIPS↓ | DISTS↓ | PSNR↑ | varLap | latency 128→out |
|---|---:|---:|---:|---:|---:|
| bicubic | 0.1994 | 0.2117 | 24.06 | 300 | — |
| **compact** (realesr-general-x4v3) | **0.1160** | **0.1671** | 25.13 | 1953 | 21 ms |
| x4plus (RRDBNet, ceiling) | 0.1026 | 0.1595 | 25.32 | 1934 | 360 ms |
| **span** (2xLiveActionV1) | 0.1217 | 0.1876 | 25.02 | 1136 | 84 ms |
| GT | — | — | — | 2378 | |

**Overall, the blanket claim is REFUTED: compact beats SPAN on both LPIPS (0.116 vs 0.122) and DISTS
(0.167 vs 0.188).** x4plus is the ceiling (but 17× compact's latency).

## But it's content-dependent — and SPAN wins exactly where it's deployed

SPAN vs compact head-to-head (Δ = span − compact; negative = SPAN better):

| window / crf | ΔLPIPS | ΔDISTS | ΔPSNR | winner |
|---|---:|---:|---:|:--|
| **talkinghead / moderate** | **−0.0305** | **−0.0175** | +1.54 | **span** |
| **talkinghead / heavy** | **−0.0254** | **−0.0107** | +0.62 | **span** |
| highmotion / moderate | +0.0052 | +0.0547 | −2.70 | compact |
| highmotion / heavy | +0.0132 | +0.0200 | −0.94 | compact |
| texture24k / moderate | +0.0090 | +0.0252 | +0.72 | compact |
| texture24k / heavy | +0.0629 | +0.0516 | +0.12 | compact |

**SPAN wins talking-head faces decisively** (~20% lower LPIPS, both CRF, +PSNR) — and that is the demo
clip and the user's actual content. It **loses on texture and high-motion** (its smoothing drops real
texture; compact's grit scores closer to a textured GT).

## Why (var-Lap inversion confirms the mechanism)
- Clean native LR (fabrication check): SPAN keeps the most detail and never exceeds GT (talkinghead
  span vL=78 vs compact 29 vs GT 100; texture span 4418 vs compact 1616 vs GT 5696) → **not fabricating,
  just sharper-and-faithful.**
- Codec-degraded restore: SPAN var-Lap (1136) < compact (1953) → SPAN (codec-trained) **suppresses**
  blocking/ringing; compact **amplifies** it into fake high-freq. On faces, suppression wins (smooth
  GT). On texture, the amplified grit happens to score closer to the textured GT → compact wins.

`2xLiveActionV1` is a *live-action specialist*: trained on film/video, it excels on faces/people and
under-performs on synthetic texture — exactly the measured pattern.

## Verdict
- The web_spike "SPAN > compact" claim was **wrong as stated** (overall it's the reverse).
- **SPAN is the right anchor for live-action / talking-head content** (the demo + the user's clip),
  where it beats compact by ~20% LPIPS at both compression levels — so the player default is *defensible
  for its target content*, but it is a specialist, not a universal upgrade.
- Truly correct design = **content-adaptive** (SPAN for faces/smooth, compact for texture/high-motion;
  compact is also 4× cheaper) — the R6 Auto-mode idea applied to model choice.
