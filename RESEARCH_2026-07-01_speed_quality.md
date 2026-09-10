# playhd — Deep Research: Speed & Upscale-Quality Frontiers (2026-07-01)

Four parallel web-grounded research streams (Opus agents) on how to push playhd's **speed** and
**upscale quality** beyond the R11 local optimum (x4plus + SPAN-on-faces + β=0.85, 183 ms conv floor).
Each stream was instructed to separate measured results from model-card marketing, and to respect
playhd's hard-won constraints: the **fake-detail trap** (synthetic-degradation SOTA over-smooths real
H.264), LPIPS+DISTS+PSNR measurement discipline, browser real-time target, and license-sensitivity.

## The two frontiers (both cheaper to move than expected)

- **Speed frontier = WebGPU.** The 183 ms conv floor is almost certainly a WASM/CPU number. Measured
  comparables put a compact SR net at 640×360 at **~10–25 ms on WebGPU** (ONNX Runtime Web WebGPU EP) —
  a **7–15× win from a backend swap alone**, before any kernel work.
- **Quality frontier = codec-matched training.** The reason SPAN wins on faces is that it was trained on
  real H.264 degradations, not architecture magic. Released tooling to reproduce this ourselves already
  exists, and the SPAN *architecture* is Apache-2.0 (only the specific weights are non-commercial).

Three conclusions were reached *independently* by ≥2 streams (higher confidence): (1) codec-matched
training is THE quality frontier; (2) browser SR is memory-bandwidth-bound and our amortization is the
moat that lets us afford a heavier anchor than any per-frame competitor; (3) our architecture is
validated by 2024–2026 peer-reviewed work (TMP, CDA-VSR, PnP-VCVE, Palantir, AIM-2024 AV1 track).

---

## Ranked actions (unified across all four streams)

### #1 — Retrain the SPAN *architecture* on a codec-matched libx264 degradation pipeline  ·  QUALITY + LICENSE  ·  highest leverage
- SPAN *arch* = **Apache-2.0** (`hongyuanyu/SPAN`, arXiv 2311.12770); only `2xLiveActionV1_SPAN` *weights* are CC-BY-NC-SA.
- Released tooling does **real ffmpeg libx264 encode→decode round-trips**: APISR `degradation/video_compression/h264.py`, VCISR `degradation/`.
- Verified: stock Real-ESRGAN/x4plus degradation uses `DiffJPEG` only — **x4plus has never seen an H.264 artifact.**
- Recipe: fine-tune a permissive real-time backbone (SPAN-arch, or RLFN/compact for max license-safety) on HR→libx264-LR pairs at **our exact CRF/preset/GOP** (self-encode from REDS/Vimeo-90K/BVI-DVC). Match GOP — keyframe interval changes I- vs P-frame artifacts.
- Evidence it works: StreamSR 2026 measured RLFN **2.17→2.69 subjective** from real-pair training; a fine-tuned small net beat NVIDIA's heavy VSR in **77.4%** of a 3,822-person study.
- Fixes the non-commercial demo blocker AND the fake-detail trap in one move. This is our own R10 open item, now externally backed.
- Cost: one offline training run, zero inference/arch change. Validate on held-out real clips, not synthetic.

### #2 — Port the anchor model to ONNX Runtime Web (WebGPU EP) and measure  ·  SPEED  ·  cheapest high-info move
- Strong prior: land ~10–25 ms (≥7× under 183 ms). fp16 + IO-binding + graph capture.
- ORT-Web WebGPU EP ships native `Conv`/`ConvTranspose`/`FusedConv`/`DepthToSpace(=PixelShuffle)` — exactly our graph.
- Keep the current tuned pass as the **WASM-SIMD fallback** (iOS<26, Adreno-no-shader-f16, no-WebGPU).
- Reference impl to crib: `xororz/web-realesrgan` (TF.js WebGPU, fp16, 12 px-overlap tiling).

### #3 — Replace the temporal-EMA blend with a codec-residual reliability gate  ·  QUALITY (untables "желе" + softens high-motion)
- Root cause of jelly: EMA is a motion-compensated IIR low-pass filter dragging high-freq detail through block-quantized (misaligned) MVs.
- Fix: CDA-VSR **Residual-Map Gated Fusion** — per-pixel suppress propagation where the codec residual is high (occlusion / failed MC), fall back to current frame. Near-free (residual from same bitstream).
- Fallback signal if residuals are hard to get: **forward-backward MV consistency** (round-trip warp error → occlusion mask).
- The *specific gap* between playhd and CDA-VSR/TMP (2024–2026 peer-reviewed, ~our architecture) IS this EMA blend.

### #4 — Fix the `deblock_pre` gate with exact per-frame QP from ffmpeg  ·  QUALITY (dissolves R10 blocker)
- "PyAV hides QP" is gone: verified in FFmpeg source `h264_export_enc_params()` (`libavcodec/h264dec.c`) exports per-MB QP as `AV_FRAME_DATA_VIDEO_ENC_PARAMS` via codec option `export_side_data=venc_params`. Ready tool: `tools/venc_data_dump.c`.
- This is the **actual bitstream quantizer**, free at decode time — turns `deblock_pre` (LPIPS −13% / DISTS −17% / +0.5 dB on heavily-compressed anchors) from gated-unreliable into reliably-gated.
- Adding `venc_params` export to the WASM-libav build is the same class of change as the MV export we already ship.
- Browser-only fallback: DCT-coefficient-histogram first-peak QP estimate (Qstep doubles every +6 QP).
- Verify first with a 10-min `venc_data_dump` run on one real clip. Do NOT use the dead `ffmpeg -debug qp` hack or unmerged ffprobe patch.

### #5 — fp16 on WebGPU / int8-QAT (post-reparam-fuse) on WASM  ·  SPEED
- Reparam conv nets (SPAN, compact) lose only **~0.2–0.28 dB int8** with QAT — but **fuse reparam branches before quantizing** (else accuracy craters). Transformers crater regardless (−3.76 dB naive 4-bit).
- fp16 near-lossless + halves download; query `adapter.features.has('shader-f16')` (absent on Qualcomm/Adreno).
- int8 is a **trap on WebGPU** (no speedup, artifacts) — it's a WASM/XNNPACK/mobile-NPU story only.

### #6 — Graded frame-type compute + texture-weighted patch anchoring  ·  SPEED
- CDA-VSR FTAR (heavy-I / light-P): 78 vs 176 GMACs (−55.7%).
- Palantir (MMSys 2025) patch-level anchoring: up to **80% less SR overhead** vs NeuroScaler — concentrate the SPAN budget on faces/high-texture patches (leverages R11 "SPAN wins faces"). We already probe texture → a heuristic captures most of it.

### #7 — This-week cheap probes
- A/B **VCISR** released weights (real-libx264-trained SISR) vs x4plus on real H.264 (LPIPS/DISTS/PSNR + VMAF-NEG). Fastest test of the "codec-trained SR wins" hypothesis. GPL-3.0 weights = non-commercial (probe only).
- Evaluate **PnP-VCVE** (4.56M params, 47 GFLOPs, 28 fps, real H.264 CRF 15–48, bitstream-conditioned) as (a) pre-net before SR and (b) its own ×4 head vs x4plus. Closest external validation of playhd's whole thesis.
- Add **VMAF-NEG** as a hallucination anti-cheat guardrail (LPIPS/DISTS *rank* well on compressed content but are blind to hallucination: corr ~0.23 / −0.09). Do NOT optimize against it (gameable ~+22%).

### #8 — Hand-written WGSL kernels  ·  SPEED  ·  last, only if still short after #2 with a frozen arch
- Templates: `Anime4K-WebGPU` (<3 ms/720p on 4090/3070Ti), `sona1111/webgpu-super-resolution` (Real-ESRGAN in WGSL). 2.5–30× vs TFLite but the gap is smaller vs ORT-WebGPU; high eng cost.

---

## Validated NO-GOs (don't spend time here)

- **Diffusion SR/restorers** (OSEDiff, SUPIR, DiffBIR, SDATC @ 11.6 s/frame, FlashVSR) — hallucinate + 100–1000× too slow.
- **Mamba/state-space SR** (MambaIR, DVMSR, Hi-Mamba) — sequential selective-scan can't run fast on WebGPU (no efficient scan kernel); synthetic-benchmark-only.
- **Big transformer SR as a deployed model** — slow in WASM, quantizes badly. Fine only as an *offline distillation teacher*.
- **Optical-flow / VFI nets as an MV *replacement*** — destroys the amortization; codec MVs are unbeatable on cost-vs-quality for common motion (measured 0.25–0.4 px EPE, free). MVs *do* break down at medium/large motion (5.6–53 px EPE) and intra/SKIP blocks — handle via the residual gate + block-sparse local refine, not replacement.
- **FBCNN/QGAC/SCUNet-style QP transfer for the gate** — tied to JPEG 8×8 DCT, doesn't transfer to H.264. SCUNet being a JPEG proxy is *why* the gate was unreliable — use `venc_params`.
- **NR metrics (MUSIQ/CLIP-IQA/NIQE) as a decision criterion** — measurably reward oversharpening. LPIPS+DISTS+PSNR choice is externally vindicated.
- **Shipping SAFMN (no license file = all-rights-reserved) or `*_SPAN` community weights (CC-BY-NC) commercially** — retrain, don't reship.
- **wonnx** (Rust ONNX-on-wgpu) — repo archived May 2025, dead. **WebNN** — Chromium-only, behind flags, no Safari/Firefox; future NPU probe, not a 2026 dependency.

---

## Key numbers & prior art (citation spine)

- **Browser inference:** IMG.LY conv net WebGPU fp16 ~100 ms vs WASM-16thr ~2000 ms on M3 Max (16–20×); Anime4K-WebGPU <3 ms/720p; free.upscaler.video 15 fps@720p (100k-param) on M4. WebGPU ships Chrome/Edge 113+, FF 141/145, Safari 26 (~70–78% coverage). WebGPU ≈ ½ native. `maxStorageBufferBindingSize` default 128 MiB → must tile 1080p (≥12 px halo).
- **Temporal/MV:** TMP (TIP 2024) 3.1M params, 25 ms/frame, 40 fps@720p; CDA-VSR (2026) 10.8 ms/frame, 93 fps@320×180, RMGF + frame-type-aware; AV1 MV fidelity 0.25 px EPE (small motion) → 46–53 px (large); Palantir 80% overhead cut.
- **Architectures:** AIM-2024 Efficient VSR for AV1 (360p→1080p ×3, real SVT-AV1 CRF 31–63) — SuperBicubic++ 0.05M/2.91 GMACs, BVI-RTVSR 0.062M/3.9 GMACs (reparam CNNs, the design template). SPAN = NTIRE-2024 runtime winner, plain-conv + parameter-free attention. int8 QAT: reparam CNN −0.2/−0.28 dB, ~72% latency cut.
- **Codec restoration:** StreamSR 2026 (5,200 real YouTube clips, 3,822-person study) — small nets on real pairs beat heavy VSR. PnP-VCVE real H.264 bitstream-conditioned. Real-ESRGAN degradation is JPEG-only (verified). FFmpeg `venc_params` = exact per-MB QP (verified in `h264dec.c`).

## Primary sources (fetched/verified)

- ORT-Web WebGPU: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html · IMG.LY 20× measured: https://img.ly/blog/browser-background-removal-using-onnx-runtime-webgpu/ · web-realesrgan: https://github.com/xororz/web-realesrgan · Anime4K-WebGPU: https://github.com/Anime4KWebBoost/Anime4K-WebGPU
- CDA-VSR: https://arxiv.org/html/2603.07694v1 · TMP: https://arxiv.org/html/2312.09909v2 (repo https://github.com/xtudbxk/TMP) · Palantir: https://arxiv.org/html/2408.06152v1 · AV1 MV fidelity: https://arxiv.org/html/2510.17427v1
- SPAN (Apache-2.0): https://arxiv.org/abs/2311.12770 · https://github.com/hongyuanyu/SPAN · `2xLiveActionV1_SPAN`: https://openmodeldb.info/models/2x-LiveActionV1-SPAN · AIM-2024 AV1 track: https://arxiv.org/html/2409.17256v1/ · SeemoRe (Apache-2.0): https://arxiv.org/html/2402.03412v1 · 2DQuant: https://arxiv.org/html/2406.06649v1
- StreamSR: https://arxiv.org/html/2602.11339v2 · PnP-VCVE: https://arxiv.org/html/2504.15380 (https://github.com/ZeldaM1/PnP-VCVE) · VCISR: https://github.com/Kiteretsu77/VCISR-official · APISR (degradation code): https://github.com/Kiteretsu77/APISR · Real-ESRGAN degradation (JPEG-only): https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/models/realesrgan_model.py · FFmpeg h264dec.c: https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/libavcodec/h264dec.c · venc_data_dump.c: https://ffmpeg.org/doxygen/7.0/venc__data__dump_8c_source.html · SCUNet: https://github.com/cszn/SCUNet
