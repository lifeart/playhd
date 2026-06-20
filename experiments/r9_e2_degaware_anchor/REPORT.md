# R9-E2: Can a degradation-aware anchor BEAT x4plus on REAL H.264 SD? — NO-GO

**Verdict: NO-GO. x4plus (RealESRGAN_x4plus, RRDBNet 16.7M) stays the validated quality ceiling.**
No obtainable non-diffusion degradation-aware model beats x4plus on **LPIPS AND DISTS** on real
H.264-degraded crops. The strongest available challenger — **4x-UltraSharp** (ESRGAN RRDB **16.7M,
capacity-EQUAL to x4plus**, community real-world-degradation-trained) — *loses* on both arbiter
metrics and reproduces the **GOTCHA #23 fake-detail trap** (a *different mechanism* from diffusion:
GAN over-sharpening on clean input -> plasticky faces; over-denoising on real codec input -> soft
output). Both failure modes lose real (full-reference) quality.

Artifacts: `run_ab.py`, `analyze.py`, `results.json` (240 records), `out/fab_*.png` (native-crop
fabrication panels + 512 pixel-peep crops). Weights under `models/` (NOT prototype/models/).

---

## Models obtained (clean path: `pip install spandrel` -> MPS auto-arch-detect)
Network IS available on this box (pip + HTTPS work). `spandrel 0.4.2` installed cleanly and
auto-identifies + runs each .pth on MPS.

| candidate | source | arch (spandrel) | params | status |
|---|---|---|---|---|
| **x4plus** (baseline/ceiling) | already in prototype/models | RRDBNet x23 | 16.70M | tested |
| compact (reference) | already in prototype/models | RealESRGAN Compact | 1.21M | tested |
| **4x-UltraSharp** | HF uwg/upscaler | ESRGAN (RRDB x23) | **16.70M** | tested (capacity-equal challenger) |
| **realesr-general-wdn-x4v3** | RealESRGAN v0.2.5.0 | RealESRGAN Compact | 1.21M | tested as DNI denoise blend (w=0.5, 0.0) |
| 4xNomos8kSC | Phhofm/models GH release | ESRGAN/Span family | ~16M | **BLOCKED - see below** |

**Blocker (1 model):** `4xNomos8kSC.pth` is a **67,076,350-byte** GitHub release asset; this box's
network **truncates the connection at ~14-40 MB every attempt** (HTTP/2 200 starts, drops mid-stream).
`curl -C -` resume crawls (~2 MB/poll) and did not complete within the bounded window (the corrupt
partial fails `PytorchStreamReader: failed finding central directory`). A future run on an
unthrottled connection can drop it straight into the same `spandrel` path - no code change needed.
**This does not change the verdict:** Nomos is the *same ESRGAN/real-world-degradation family* as the
already-tested 4x-UltraSharp, which lost decisively.

The diffusion route (OSEDiff/StableSR) was deliberately **not** pursued - it is the prior settled
NO-GO (GOTCHA #23) and is out of scope (the lighter non-diffusion bet was the whole point of R9-E2).

---

## Protocol (synthesis of the two prior methodologies in handoff.md)
Real SD has **no true HR**, so a full-reference arbiter needs a pseudo-GT. I combined the diffusion
NO-GO's *real-codec* spirit with R6-E1's *decoded-frame-as-GT* convention, swapping R6-E1's **synthetic**
degrade operator (blur+JPEG+noise - which the task forbids) for a **REAL libx264 encode**:

1. **GT** = decoded `sample.mp4` 256x256 crop (R6-E1's 5 var-Lap-scan windows: talkinghead@5000,
   highmotion@0, texture18k/24k/46k). Best-texture crop for texture windows; upper-center for faces. n=4 frames/window.
2. **LR** = 2x INTER_AREA down (->128px) -> **PyAV `libx264` encode** at CRF in {27 moderate, 35 heavy}
   -> decode -> genuine H.264 artifacts (blocking/ringing/4:2:0 chroma). *Real* codec degradation, the
   exact regime x4plus (trained on synthetic bicubic+noise) is *not* trained on and a real-codec
   degradation-aware model should win - if the thesis held. (System ffmpeg is broken/missing libx265;
   PyAV bundles its own libs and encodes in-process.)
3. **Restore** = model x4 (128->512) -> INTER_AREA 512->256 (matches R6-E1 `sr.upscale_to`). Identical for all.
4. **Arbiter** = full-reference **LPIPS (AlexNet)** + **DISTS** + **PSNR** (pyiqa, MPS) vs GT.
   **var-Lap = FAKE-detail flag ONLY** (GOTCHA #23), never the verdict.
5. **Fabrication check** = each model on the *clean native crop* (no codec) x4->512, saved for visual +
   var-Lap - the diffusion-trap detector.

---

## Result - OVERALL MEAN (5 windows x 2 CRF x 4 frames = 40 cells/model)
| model | LPIPS (lo) | DISTS (lo) | PSNR (hi) | var-Lap |
|---|---|---|---|---|
| bicubic (floor) | 0.2308 | 0.2173 | 22.64 | 461 |
| compact 1.2M | 0.1275 | 0.1684 | 23.63 | 3036 |
| **x4plus 16.7M (CEILING)** | **0.1179** | **0.1614** | 23.78 | 3120 |
| 4x-UltraSharp 16.7M | 0.1357 | 0.1808 | 23.55 | 2304 |
| wdn-dni 0.5 (denoise) | 0.1277 | 0.1742 | 23.63 | 2908 |
| wdn-dni 0.0 (pure WDN) | 0.1304 | 0.1809 | 23.63 | 2623 |

**x4plus has the best (lowest) LPIPS AND DISTS overall, by a clear margin.** UltraSharp - the
capacity-equal degradation-aware challenger - is **+15% LPIPS and +12% DISTS WORSE**. wdn denoise
blends nudge PSNR (distortion) but lose LPIPS+DISTS (the classic perception-distortion trade, on the
lower-capacity compact arch anyway).

### Per-cell winners vs x4plus (the GO bar = beat x4plus on **LPIPS AND DISTS**)
- **LPIPS:** x4plus is the *best non-bicubic model in 8/10 cells.* UltraSharp beats x4plus in **1/10**
  (texture46k/heavy, delta=0.0004 - a tie - and there by being the *softest* output, not by recovering detail).
- **DISTS:** x4plus best in 5/10; UltraSharp beats it in 2/10 (both texture46k, again via over-smoothing).
- **Cells where ANY candidate beats x4plus on BOTH LPIPS+DISTS:** exactly two -
  `highmotion/heavy` (compact; the low-detail title-card domain R6-E1 already assigned to compact) and
  `texture46k/heavy` (UltraSharp; LPIPS *tie* + DISTS, achieved by over-smoothing the hardest crush at a
  -0.5 dB PSNR cost). **Neither raises the ceiling on detailed content.** On every other textured cell,
  x4plus wins LPIPS and usually DISTS.

---

## Fabrication verdict (mandatory - the diffusion trap)
UltraSharp reproduces GOTCHA #23 in **two** content-dependent ways, both of which the arbiter rejects:

1. **Clean input -> over-sharpening (the var-Lap fab signature).** On the *clean native* talking-head
   crop, UltraSharp var-Lap = **160 vs x4plus 53** (3x). **Visual pixel-peep** (`out/fab_talkinghead_*_512.png`)
   confirms this is **fabricated/plasticky**: the face/hand render with an artificial waxy "3D-render"
   edge-enhanced look; x4plus renders natural photographic skin + fabric weave. The var-Lap "win" is
   FAKE - and full-reference agrees: UltraSharp *loses* LPIPS (0.1413 vs 0.1321) and DISTS
   (0.1723 vs 0.1508) on this window. **Exactly the trap that killed the diffusion anchor.**
2. **Real H.264 input -> over-denoising.** On the actual codec-degraded LR, UltraSharp is consistently
   the **softest** model (lowest var-Lap on *every* window: 76 / 883 / 3728 / 3765 / 3071, all below GT)
   - it reads codec artifacts as noise and scrubs them along with real detail, losing LPIPS+DISTS.

Either way: **no real-metric gain over x4plus.** var-Lap, as always, is the fake-detail detector, never the arbiter.

---

## Integration scope/cost (had it been a GO - for the record)
A spandrel-loaded anchor is cheap to wire: `pip install spandrel`, add a `MODE_CONFIG` model option,
load via `ModelLoader().load_from_file(path)` (auto-arch). UltraSharp is the *same RRDB arch already
hand-written in `sr.py`*, so it would also load with `strict=True` and no new dependency. Per-anchor
latency ~= x4plus (same 16.7M RRDB) - neutral, amortized over ~2-12% anchor frames. **But the measured
quality is a regression, so there is nothing to integrate.** Keep `quality = x4plus + region-aware + fp16`.

## What a future attempt needs to flip this
1. An unthrottled box to finish 4xNomos8kSC (and try a genuinely *codec-trained* video model, e.g. a
   SwinIR/DAT finetuned on H.264/HEVC artifacts, not generic Real-ESRGAN 2nd-order synthetic degradation).
2. The honest reason this is hard: the obtainable real-world SR GANs are trained on *synthetic*
   blur+noise+JPEG degradation - when they meet *real* H.264 they either over-sharpen (clean regions ->
   fabrication) or over-denoise (compressed regions -> softness). Beating x4plus needs a model whose
   degradation prior actually *matches* H.264, which none of the community checkpoints here provide.

**Bottom line: x4plus remains the validated quality ceiling. NO-GO on the degradation-aware anchor.**
