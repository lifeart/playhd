# R3-E2: Content-Robustness QA Sweep — Findings

**8 authored H.264 clips × 3 modes (instant/quality/layered), 24 cells.** Honest metrics: tOF (output→LR
vs decoded-LR), exact occlusion-fallback % (`anchor_sr._lr_fallback_*`), per-frame LR-consistency
(supporting), output validity/audio-sync. Artifacts: `make_clips.py`, `qa_sweep.py`, `clips/`, `out/`,
`samples/`, `results.json`.

## Content × Mode — tOF (↓) | ms/frame | valid
| content (signal) | instant | quality | layered |
|---|---|---|---|
| c1 fastpan (mvMag 35.7, fb 17/79%) | 3.18 \| 88 | 1.21 \| 1524 | 1.21 \| 1639 (→MOVING fb) |
| c2 graphics (edgeDens 0.069) | 0.36 \| 32 | **0.069** \| 1943 | 0.61 \| 470 (plate) |
| c3 lowlight (fb **95/99.8%**) | 0.91 \| **124** | 0.90 \| 1906 | 0.90 \| 1493 (→MOVING fb) |
| c4 talkinghead (mvMag 0.06) | 0.39 \| 34 | **0.015** \| 2056 | 0.034 \| 454 (plate) +mp3→aac |
| c5 scenecut **missed** (Δluma 36) | 1.01 \| 46 | 0.33 \| 1440 | 0.33 \| 1535 (→fb) |
| c5b scenecut **detected** (Δluma 105) | 0.79 \| 33 (split) | 0.27 \| 1782 | 0.27 \| 1833 |
| c6 oddres 642×362@23.976 (fb 15%) | 1.14 \| 75 (1284×724) | 0.29 \| 2601 | 0.29 \| 3234 |
| c7 **static-cut missed** (Δluma 6) | 0.43 \| 42 | 0.056 \| 1516 | 0.12 \| 522 **CORRUPT bg** |

**Zero crashes; every cell valid H.264 at correct frame-count/resolution.** Odd resolution (non-mult-16),
odd fps (23.976, vdur preserved), and the mp3→aac **transcode** mux all work + stay in sync.

## Failure-mode catalog (ranked)
1. **[HIGH — real bug] Layered paints the WRONG background over a whole scene when a cut is missed.**
   Trigger: two static scenes, similar-luma cut (c7, Δluma 6). `segment_scenes` misses it → one plate
   spans both scenes → scene-A plate composited over scene-B. LR-consistency **33.8→14.7 dB** post-cut
   (quality holds 37.6); visually confirmed. **tOF does NOT catch it** (wrong plate is temporally stable)
   — only fidelity-vs-LR exposed it. Fix: per-frame plate-validity guard (cross-check
   composite-vs-LR-consistency → auto-fallback to region-aware), and/or a chroma/structural term in the
   cut detector, and/or a within-plate-window cut check.
2. **[HIGH — quality cliff] Instant degenerates to per-frame-SR on low-light/noisy content.** c3: noise →
   unreliable MVs → 95% pixel fallback (max 99.8%) → 24/24 frames exceed the 0.50 safeguard → 24 SR
   upgrades, ms/frame 34→**124 (3.6×)**, real-time broken. Mitigation: pre-denoise before MV/occlusion;
   or cap SR upgrades on fallback-saturated frames (accept bicubic, hold fps); or auto-route to quality.
3. **[MED — quality cliff] Instant soft+flickery on high global motion.** c1: tOF **3.18 (8× the 0.39
   baseline)**, 17%/79% bicubic fallback left soft. Known fast-tier tradeoff, now quantified.
4. **[MED — detection gap] Similar-lit / fast (<24f) scene cuts missed.** Detector is frame-avg-|Δluma|:
   c5(Δ36)+c7(Δ6) missed; c5b(Δ105) detected+split (fallback 100%→12%). Harmless for instant/quality
   (per-frame source is correct content) — it only feeds #1 in layered. Fix: add a chroma/structural term;
   reconsider MIN_SCENE_LEN=24 for fast-cut content.
5. **[LOW] Layered mis-flags noisy static cameras as MOVING** (c3: spurious-noise median |MV| 2.5 > 0.6 →
   region-aware fallback, forfeits the plate denoising exactly where it'd help). Mitigation: denoise before
   global-motion estimation.
6. **[LOW] Layered adds flicker on motion-graphics** (c2 layered tOF 0.61 > quality 0.069). Graphics belong
   on instant/quality, never layered.
7. **[LOW — caveat, not a bug] Video-only sources** produce valid silent output but `_verify_mp4` returns
   ok=False (it requires an audio stream). Verification-helper quirk.

*Caveat:* RVM is human-trained; the synthetic subjects aren't human, so absolute matte quality on real
talking-heads isn't validated here. Findings #1/#5/#6 are matte-independent.

## Per-content mode recommendation + auto-select signals
Signals: **motion** = mean LR-MV mag; **graphic-edge density** = Canny fraction; **plate safety** =
detected-scene-count + static verdict + plate-vs-frame residual.
- Near-static talking-head (mvMag<1, fb<2%): **layered** if single static human scene + low plate residual, else **quality**.
- Graphics/text (edgeDens>0.05, low motion): **instant** or **quality**; never layered.
- High motion / fast pan (mvMag>~10 or fb-react>15%): **quality** (instant tOF 3–8× worse).
- Low-light/noisy (fb-react mean>50%): **quality**; off instant (real-time breaks) and off the layered plate unless denoised.
- Rule of thumb: instant only when mvMag<~8 AND fb-react-mean<~15%; layered only when scenes are
  detected-static, single-scene-per-plate, human, low plate residual — else quality is the safe default.

## Executive summary
24 runs, never crashed, always valid H.264 (odd res, odd fps, mp3→aac all OK). **Top 3 failure modes:**
(1) layered paints the wrong background over a whole scene on a missed similar-luma cut (33.8→14.7 dB; tOF
blind, only fidelity-vs-LR caught it — a true bug); (2) instant collapses to per-frame-SR on low-light
noise (95% fallback → 3.6× slower, real-time broken); (3) instant softens/flickers on high motion (tOF
3.18 vs 0.39). **Biggest risk:** the silent layered missed-cut plate corruption. **Next fix:** a per-frame
plate-validity guard in layered + a chroma/structural term in the cut detector.
