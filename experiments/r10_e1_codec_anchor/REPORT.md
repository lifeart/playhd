# R10-E1: Can a CODEC/compression-trained anchor BEAT x4plus on REAL H.264 SD? — NO-GO

**Verdict: NO-GO. RealESRGAN_x4plus (RRDBNet 16.7M) remains the validated quality ceiling.**
Four architecturally-NOVEL, compression-trained x4 models — **DAT2, ATD, HAT-L** transformers
**+ ESRGAN** (the whole point of R10: go beyond R9-E2's RRDB family) — every one **LOSES to x4plus
on LPIPS AND DISTS** on real libx264-degraded `sample.mp4` crops. None beats x4plus on **both**
arbiter metrics in **any** of the 10 cells. This settles the model-replacement frontier across the
obtainable compression-trained space (RRDB tried in R9-E2; transformers + Span tried here).

This run also **resolves R9-E2's only blocker:** `4xNomos8kSC` (the 67 MB GitHub asset that truncated
on R9-E2's throttled box) downloaded cleanly via `huggingface_hub` (`Phips/4xNomos8kSC`, 33.5 MB
safetensors) — and lost too, confirming R9-E2's prediction.

Artifacts: `run_ab.py`, `analyze.py`, `results.json` (280 records + latency + spandrel arch),
`out/fab_*.png` (native fabrication panels + per-model 512 pixel-peep crops). Weights under `models/`.

---

## Models obtained — all via HuggingFace Hub (resumable, zero truncation)
`huggingface_hub.hf_hub_download` from `Phips/*` repos; `spandrel 0.4.2` auto-detected each arch on MPS.

| candidate | HF repo | spandrel arch | size | training degradation (model card) | status |
|---|---|---|---|---|---|
| **x4plus** (CEILING) | prototype/models | RRDBNet | 67 MB | Real-ESRGAN 2nd-order **synthetic** (blur+noise+**JPEG**) | baseline |
| compact (ref) | prototype/models | SRVGGNetCompact | 4.9 MB | realesr-general | ref |
| **4xRealWebPhoto_v4_dat2** | Phips/4xRealWebPhoto_v4_dat2 | **DAT** (DAT-2) | 140 MB | lens blur + ludvae200 noise + **JPG+WEBP compression (40-95)** + multi-kernel resize | tested |
| **4xNomos8k_atd_jpg** | Phips/4xNomos8k_atd_jpg | **ATD** | 82 MB | **OTF JPG compression->40 + re-compression->40** + blur kernels (preserves noise) | tested |
| **4xNomos8kHAT-L_otf** | Phips/4xNomos8kHAT-L_otf | **HAT** (HAT-L) | 165 MB | OTF: jpg compression + blur + noise | tested |
| **4xNomos8kSC** | Phips/4xNomos8kSC | **ESRGAN** (Span family) | 33 MB | OTF jpg compression + slight blur (R9-E2's blocker — now obtained) | tested |

**Honest scope caveat (matters for "what a future attempt needs"):** every obtainable "codec/compression"
model is trained on **JPEG/WebP (DCT-block) compression**, the closest available proxy to H.264 *intra*
blocking — **not** on true libx264 degradation (deblock-filter smoothing, motion-comp residual, GOP
structure). No photographic model trained on actual H.264/HEVC video codec degradation is publicly
obtainable. (Philip Hofmann's "AVC"=H.264 series exists but is **anime-only + 2x**, wrong domain/scale.)

---

## Protocol (identical to R9-E2 — direct harness reuse, the validated methodology)
GT = decoded `sample.mp4` 256^2 crop (R6-E1 var-Lap windows, n=4 frames/window). LR = 2x INTER_AREA
down (->128) -> **REAL PyAV libx264 encode** @ CRF {27 moderate, 35 heavy} -> decode -> genuine H.264
artifacts. Restore = model x4 (128->512) -> INTER_AREA 512->256. **Arbiter = full-reference LPIPS(AlexNet)
+ DISTS + PSNR (pyiqa, MPS). var-Lap = FAKE-detail flag ONLY (GOTCHA #23), NEVER the verdict.**
Fabrication check = each model on the CLEAN native crop (no codec) + var-Lap + visual pixel-peep.
5 windows x 2 CRF x 4 frames x 7 backends = 280 records.

---

## Result — OVERALL MEAN (lower=better LPIPS/DISTS, higher=better PSNR)
| model | arch | LPIPS lo | DISTS lo | PSNR hi | var-Lap | anchor lat (128->512) |
|---|---|---|---|---|---|---|
| bicubic (floor) | — | 0.2308 | 0.2173 | 22.64 | 461 | — |
| compact 1.2M | — | 0.1275 | 0.1684 | 23.63 | 3036 | 32 ms |
| **x4plus 16.7M (CEILING)** | RRDB | **0.1179** | **0.1614** | 23.78 | 3120 | **463 ms** |
| 4xRealWebPhoto_v4_dat2 | DAT2 | 0.1445 | 0.2011 | 23.35 | 1792 | 1380 ms (3.0x) |
| 4xNomos8k_atd_jpg | ATD | 0.1282 | 0.1833 | **24.02** | 1733 | 3691 ms (8.0x) |
| 4xNomos8kHAT-L_otf | HAT-L | 0.1380 | 0.2061 | 22.89 | 1637 | 3217 ms (6.9x) |
| 4xNomos8kSC | ESRGAN | 0.1320 | 0.1952 | 23.02 | 1959 | 442 ms (1.0x) |
| *(GT reference)* | | | | | *3312* | |

**x4plus has the best (lowest) LPIPS AND DISTS by a clear margin.** The closest perceptual challenger,
**ATD**, is **+9% LPIPS and +14% DISTS WORSE**; DAT2/HAT-L are +18-23% DISTS worse. Per-cell: a
challenger beats x4plus on **BOTH LPIPS+DISTS in exactly ZERO of 10 cells** (the lone both-win cell,
`highmotion/heavy`, belongs to **compact** — the existing low-detail title-card domain R6-E1 already
assigned it, not a codec model and not detailed content). Paired win-rate (40 matched samples each):
DAT2 0/40 both-better, ATD 0/40, HAT-L 0/40, SC 1/40.

---

## Fabrication verdict (mandatory — the GOTCHA #23 trap)
**This time the trap fires as OVER-SMOOTHING, not over-sharpening — opposite mechanism, same loss.**
On real H.264 LR, all four candidates' var-Lap (1637-1959) sits **below GT (3312) and below x4plus
(3120)**: they are NOT inventing fake high-frequency detail — the **JPEG-compression prior reads H.264
residual *and real texture* as compression noise to scrub**, producing a softer image. The tell:
**ATD wins PSNR (24.02 > 23.78, 8/10 cells)** — a more L2-faithful but perceptually *softer* output —
the textbook perception-distortion trade, and LPIPS+DISTS (the arbiter) reject it. **Pixel-peep
(`out/fab_texture18k_{x4plus,nomos_atd_jpg}_512.png`):** x4plus renders the news headline + tiny
masthead line crisp and legible; ATD smears the fine text. Confirmed: not a hidden win.
(Only DAT2 mildly over-sharpens *clean faces* — var-Lap 81 vs x4plus 53 — but still loses LPIPS+DISTS
on that window, and over-smooths everything textured.) **var-Lap stayed a flag, never the verdict — and
correctly flagged the softness.** Three models died over-sharpening (diffusion, UltraSharp-on-clean);
these four die over-denoising. Both directions lose the full-reference arbiter to x4plus.

### Per-candidate
- **DAT2 (RealWebPhoto):** worst LPIPS (0.1445); over-smooths texture, mild clean-face over-sharpen. **FAIL.**
- **ATD (Nomos8k_atd_jpg):** strongest challenger, wins PSNR by over-smoothing, **loses LPIPS+DISTS**; visibly softer text. **FAIL.**
- **HAT-L (Nomos8k):** most over-smoothed (var-Lap 1637), **worst DISTS 0.2061**. **FAIL.**
- **ESRGAN (Nomos8kSC):** anchor-cheap (442 ms ~ x4plus) but loses LPIPS (0.132) + DISTS (0.195). R9-E2's blocker, now settled. **FAIL.**

---

## Latency (second, independent disqualifier for the transformers)
At 128->512 the transformers cost **3.0x (DAT2) / 6.9x (HAT-L) / 8.0x (ATD)** x4plus's per-anchor time,
and transformer cost scales *worse* than RRDB with crop area -> at real SD anchor sizes they are **not
anchor-affordable** even amortized over ~2-12% of frames. Only ESRGAN/Span is latency-neutral, and it
loses on quality. So even a quality tie would not have justified integration for DAT2/ATD/HAT-L.

---

## Integration scope/cost (N/A — for the record, had any been a GO)
A spandrel-loaded anchor is cheap to wire: add a `MODE_CONFIG` model option + a `MODELS[...]` entry in
`sr.py` loading via `spandrel.ModelLoader().load_from_file(path)` (auto-arch, no per-arch code). **But
every measured candidate is a quality regression on the arbiter, and three add 3-8x latency, so there is
nothing to integrate.** Keep `quality = x4plus + region-aware + fp16`.

## What a future attempt needs to flip this
1. A model trained on **genuine libx264/libx265 photographic** degradation (deblock filter +
   motion-comp residual + GOP), not JPEG/WebP DCT-block compression. No such public photographic
   checkpoint exists today; it would need a **custom finetune** of a strong backbone (HAT-L/DAT2) with
   a PyAV-libx264 degradation operator (this harness already produces exactly that LR — it could
   double as the training degradation).
2. The honest reason this keeps failing: obtainable real-world SR models are tuned to **remove**
   compression artifacts. Fed real H.264 they over-denoise (this round) or, on clean regions,
   over-sharpen (R9-E2/diffusion). x4plus's generic-synthetic prior happens to **preserve** more true
   perceptual texture on this codec regime than any compression-specialist's prior **erases**.

**Bottom line: across RRDB (R9-E2) + DAT2/ATD/HAT-L/ESRGAN (R10), no obtainable compression-trained
anchor beats x4plus on real H.264. x4plus remains the validated quality ceiling. NO-GO.**
