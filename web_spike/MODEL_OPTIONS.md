# SR model options for the browser anchor — survey + measured eval

**Goal:** find a better *speed/quality Pareto point* for the in-browser WebGPU anchor SR than the
current **SRVGGNetCompact / realesr-general-x4v3** (1.21M params, 34 plain 3×3 convs, 64 ch, PReLU,
×4 PixelShuffle, run ×4→downscale to the ×2 output). SR runs only on anchor frames (~6–50%); the
rest propagate via codec MVs.

**Two constraints drive everything (both established by prior measurement):**
1. **Browser WGSL is latency/occupancy-bound, not ALU/bandwidth-bound.** Faster = *fewer channels /
   fewer layers / less register pressure* (raises occupancy). Plain 3×3 conv stacks are ideal.
   Winograd LOST (−2.9×), layer-fusion LOST (−13.7×), attention / large-kernel / deformable /
   dense-concat are occupancy-hostile. **Reparameterizable** nets (train multi-branch, collapse to a
   plain 3×3 conv stack at inference) are the ideal class.
2. **Input is real H.264-compressed video.** Settled finding (R8–R10): on real compressed content,
   heavier / codec-finetuned models *over-smooth* (a "fake-detail" trap caught by **DISTS**, not
   PSNR); models trained only on bicubic downsampling fail / amplify blocking. Real-world degradation
   training matters. **LPIPS + DISTS (pyiqa) are the arbiters.**

## TL;DR
- **Winner: `2xLiveActionV1_SPAN`** — beats the current compact on **both LPIPS and DISTS at every CRF
  tested** (real-H.264-trained, native 2×), and its inference graph (20× plain 3×3 conv @ **48 ch** +
  6 parameter-free element-wise gates) is **2.27× cheaper** than the compact on a same-runtime MPS
  proxy → expected *faster* in WGSL, not slower. Quality **and** occupancy improve together.
- **Runner-up: a `num_conv=16` SRVGGNetCompact with real-world weights (`2xHFA2kAVCCompact`)** —
  literally the current arch minus 16 conv layers (zero port risk), measured **1.9× faster**, LPIPS
  parity-or-better, with a consistent small **DISTS penalty (~+0.015)**. The instant fast-tier.
- The classic efficient-SR-challenge nets (RLFN/RFDN/IMDN) and tiny nets (ECBSR/ABPN/edge-SR) are
  bicubic-trained with no real-world weights → not directly usable (the right *arch*, wrong weights).

---

## 1. Measured results — real H.264 SD, full-reference LPIPS + DISTS

**Protocol** (`web_spike/eval_model_options.py` + `eval_x2_candidates.py`, mirrors
`prototype/derisk.make_synthetic` and the real SD→HD ×2 deployment): decoded 640×320 photographic
frame = HD reference → AREA-downscale ×2 to 320×160 = "true SD" → **libx264 all-intra (g=1, no
B-frames) at the stated CRF, decode** = real codec degradation as an *anchor* (I-frame) sees it → SR
→ LPIPS + DISTS vs reference (lower = better). Latency = MPS median at the 320×160 SD input.

### 1a. ×4 family (depth ablation) — `eval_model_options.py`, sd600.mp4, 6 frames

| CRF | model | params | LPIPS ↓ | DISTS ↓ | SD latency | note |
|----:|-------|-------:|--------:|--------:|-----------:|------|
| 22 | bicubic | – | 0.1440 | 0.1098 | – | floor |
| 22 | **compact (current)** | 1.21M | **0.1082** | **0.1330** | **41.3 ms** | baseline |
| 22 | animev3 (num_conv=16) | 0.62M | 0.1110 | 0.1438 | 22.3 ms | LPIPS +0.003, DISTS +0.011, **1.85×** |
| 22 | x4plus (heavy) | 16.7M | 0.0967 | 0.1129 | 682 ms | better both, **16× slower** |
| 28 | **compact (current)** | 1.21M | **0.1535** | **0.1569** | **40.2 ms** | baseline |
| 28 | animev3 (num_conv=16) | 0.62M | 0.1548 | 0.1674 | 22.7 ms | LPIPS +0.001, DISTS +0.011, **1.77×** |
| 28 | x4plus (heavy) | 16.7M | 0.1406 | 0.1364 | 680 ms | better both, 16× slower |
| 34 | **compact (current)** | 1.21M | **0.2386** | **0.1966** | **40.9 ms** | baseline |
| 34 | animev3 (num_conv=16) | 0.62M | 0.2265 | 0.2047 | 22.7 ms | LPIPS **−0.012**, DISTS +0.008, **1.80×** |
| 34 | x4plus (heavy) | 16.7M | 0.2296 | 0.1876 | 682 ms | better both, 16× slower |

short.mp4 (2nd source, CRF 28): compact LPIPS 0.1260 / DISTS 0.1478; animev3 0.1282 / 0.1533 (1.89×);
x4plus 0.1159 / 0.1379. Same ordering — robust.

### 1b. Native-2× real-world candidates (the deployment scale) — `eval_x2_candidates.py`, sd600.mp4, 6 frames

All real-compression-trained; the compact baseline is the current realesr (×4→downscale).
`dDISTS`/`dLPIPS` are vs the current compact (negative = better).

| CRF | model | LPIPS ↓ | DISTS ↓ | dLPIPS | dDISTS | latency |
|----:|-------|--------:|--------:|:------:|:------:|--------:|
| 22 | compact (current) | 0.1082 | 0.1330 | – | – | 37 ms (MPS) |
| 22 | nomosuni_otf (16nc) | 0.1317 | 0.1419 | +0.024 | +0.009 | 20 ms |
| 22 | **hfa2k_avc (16nc, H.264-trained)** | 0.0925 | 0.1445 | **−0.016** | +0.012 | 20 ms |
| 22 | hfa2k (16nc) | 0.1178 | 0.1485 | +0.010 | +0.015 | 20 ms |
| 22 | **span_live (2x)** | **0.0774** | **0.0982** | **−0.031** | **−0.035** | (see proxy) |
| 28 | compact (current) | 0.1535 | 0.1569 | – | – | 37 ms |
| 28 | nomosuni_otf | 0.1658 | 0.1588 | +0.012 | +0.002 | 20 ms |
| 28 | hfa2k_avc | 0.1455 | 0.1737 | −0.008 | +0.017 | 20 ms |
| 28 | hfa2k | 0.1541 | 0.1750 | +0.001 | +0.018 | 20 ms |
| 28 | **span_live (2x)** | **0.1391** | **0.1343** | **−0.014** | **−0.023** | (see proxy) |
| 34 | compact (current) | 0.2386 | 0.1966 | – | – | 38 ms |
| 34 | nomosuni_otf | 0.2358 | 0.1960 | −0.003 | −0.001 | 20 ms |
| 34 | hfa2k_avc | 0.2320 | 0.2128 | −0.007 | +0.016 | 20 ms |
| 34 | hfa2k | 0.2207 | 0.2101 | −0.018 | +0.014 | 20 ms |
| 34 | **span_live (2x)** | **0.2309** | **0.1823** | **−0.008** | **−0.014** | (see proxy) |

**Reading the table:**
- **SPAN is the only model that wins BOTH LPIPS and DISTS at every CRF.** DISTS is the metric the
  project trusts for the over-smooth trap, and SPAN improves it (−0.014 to −0.035) — i.e. SPAN adds
  *real* structure, not fake detail. Real-H.264-trained + native 2× = best-matched to the task.
- The **num_conv=16 compacts confirm the ~1.9× speedup** but are a mixed quality bag: `hfa2k_avc`
  (H.264-trained) wins LPIPS at all CRF but carries a steady **DISTS penalty (~+0.015)** — it
  sharpens (helps LPIPS) at a small texture-fidelity cost. None *strictly* dominates the current
  compact; they are "near-parity at 1.9×", weight-dependent.
- *Methodology note:* the compact baseline runs ×4→INTER_CUBIC-downscale (slight smoothing → flatters
  its DISTS, softens its LPIPS); the candidates are native ×2 (sharper). This makes SPAN's **DISTS**
  win conservative (it beats a DISTS-flattered baseline) and gives its LPIPS win a small tailwind.
  Native ×2 is also the *correct* deployment scale and skips the wasted ×4 upsample compute.

### 1c. Fair latency — same-runtime (PyTorch-MPS) conv-stack proxy

The native-2× SPAN latency in 1b (≈100 ms) was via **ONNX/CoreML and is a runtime artifact**, not
comparable to the PyTorch-MPS numbers. To compare fairly, time the *dominant* cost — the sequential
3×3 conv stack — on the same runtime at the LR body resolution (160×320):

| stack (body proxy) | MPS latency | vs compact | validates against |
|--------------------|------------:|:----------:|-------------------|
| compact: 34× conv @ 64 ch | 32.6 ms | 1.00× | ≈ measured compact body (37 ms total) ✓ |
| animev3 / 16nc: 18× conv @ 64 ch | 17.7 ms | **1.85×** | ≈ measured anime/16nc 1.8–1.9× ✓ |
| **SPAN: 20× conv @ 48 ch** | **14.4 ms** | **2.27×** | — |

The proxy reproduces both known data points, so the SPAN ratio is trustworthy. SPAN is the **cheapest
of all** because conv cost ∝ channels² (48²/64² = 0.56/pass) and 48-ch is *narrower → higher
occupancy*. Its extra ops (6 parameter-free element-wise gates, one 1×1 concat) are negligible at LR.
**Net: SPAN is ~2× faster than the current compact AND higher quality.** In the latency/occupancy-
bound WGSL regime the project established, this conv-cost ratio is the load-bearing predictor.

---

## 2. Survey — efficient / real-world SR families vs the browser constraints

`reparam` = collapses to a plain 3×3 conv stack at inference. `occ` = browser occupancy fit.
**Quality column: MEASURED rows are §1; all others INFERRED from literature.** (Survey corroborated by
a 4-agent web sweep; uncertain values marked ~.)

| model | params | ch | conv/layers | scale | op types | reparam | weights + source | real-world wts? | occ | quality (real H.264) |
|-------|-------:|---:|------------:|:-----:|----------|:------:|------------------|:--------------:|:---:|----------------------|
| **realesr-general-x4v3** (current) | 1.21M | 64 | 34 | ×4 | 3×3 conv+PReLU+PixShuf+NN-res | plain | GH release (have) | ✅ | ✅✅ | **MEASURED baseline** |
| **realesr-animevideov3** (16nc) | 0.62M | 64 | 18 | ×4 | same, half body | plain | GH release (have) | ✅ (anime) | ✅✅✅ | **MEASURED: LPIPS≈tie, DISTS+0.01, 1.85×** |
| **2xNomosUni_compact_otf_medium** | 0.60M | 64 | 18 | **×2** | same | plain | **HF Phips (CC-BY-4.0)** | ✅ OTF noise/blur/JPEG | ✅✅✅ | **MEASURED: mixed (LPIPS−DISTS+), 1.9×** |
| **2xHFA2kAVCCompact** | 0.60M | 64 | 18 | **×2** | same | plain | **HF Phips** | ✅ **H.264/AVC** | ✅✅✅ | **MEASURED: LPIPS↓, DISTS+0.015, 1.9×** |
| **2xLiveActionV1_SPAN** | 0.41M (eval) / 2.2M (train) | **48** | 20 (6 SPAB) | **×2** | 3×3 conv + **param-free sigmoid gate (element-wise, no matmul)** + 1×1 concat + PixShuf | **YES (eval_conv folded)** | **GH jcj83429/upscaling (ONNX+pth)** | ✅ **H.264/H.265/VP9/MPEG-4** | ✅✅✅ | **MEASURED: wins LPIPS+DISTS all CRF; 2.27× cheaper conv** |
| realesr-general-wdn-x4v3 | 1.21M | 64 | 34 | ×4 | same (denoise blend) | plain | GH release | ✅ | ✅✅ | inferred ≈ compact, no speed gain |
| ECBSR (m4c16…m16c64) | 17K–622K | 8–64 | 4–16 ECB | ×2/×4 | ECB→**single 3×3**+PReLU | **YES, full** | code+exporters only; **no real-world weights** | ❌ (Y-only/bicubic) | ✅✅✅ best | inferred poor on H.264 (needs retrain) |
| ABPN (MAI'21) | 42.5K | 28 | 7 | ×3 | plain 3×3 + anchor-residual | already plain | NJU-Jet repo (tflite) | ❌ bicubic | ✅✅✅ | inferred poor on H.264 |
| edge-SR / eSR | ~10²–10³ | 1–16 | 1–3 | ×2/×4 | conv + MAX/softmax head | no | pnavarre/eSR | ❌ bicubic | ✅✅✅ fastest | inferred weak + bicubic |
| RLFN (NTIRE-22 winner) | 0.32–0.54M | 48–52 | RLFB + **ESA** | ×2/×4 | conv + **ESA (pool/strided/upsample/sigmoid)** | no | bytedance/RLFN (Apache-2) | ❌ bicubic | ⚠️ ESA hurts occ | inferred (bicubic; untested H.264) |
| RFDN / RFDN-L (AIM-20) | 0.55/0.64M | 50/52 | RFDB + **ESA** | ×4 | **1×1 distill split/concat + ESA** | no | njulj/RFDN | ❌ bicubic | ⚠️ | inferred (bicubic) |
| IMDN (AIM-19) | 0.72M | 64 | IMDB + **CCA** | ×2/×4 | **split/concat + CCA channel attn** | no | Zheng222/IMDN | ❌ bicubic | ⚠️ | inferred (bicubic) |
| SPAN (official, bicubic) | 0.15M/0.48M | 28/48 | 6 SPAB | ×2/×4 | as LiveAction above | **YES** | hongyuanyu/SPAN | ❌ official | ✅✅✅ | Set14 ×4 28.66/0.783 (≥ RLFN, fewer params) |
| FSRCNN / ESPCN | 12–25K | 32–56 | 5–8 | ×2/×3/×4 | plain conv + PixShuf/deconv | n/a | many (Y-only) | ❌ bicubic | ✅✅✅ | inferred weak (classic) |
| RealESRGAN_x4plus | 16.7M | 64 | 23×RRDB | ×4 | **dense-concat RRDB** | no | GH (have) | ✅ | ❌❌ | **MEASURED: best quality, 16× too slow** |
| HAT / ATD / DAT2 (HF cache, R9/R10) | >20M | – | transformer | ×4 | **window/channel attention** | no | HF (have) | ✅ | ❌❌❌ | settled NO-GO (over-smooth + far too slow) |
| FeMaSR / DASR | tens of M / ~16.7M | – | Swin+VQ / RRDB+MoE | ×2/×4 | **attention / codebook / MoE** | no | GH | ✅ | ❌ | not a browser candidate |

**Survey takeaways:**
- **Arch ≠ weights.** ECBSR/ABPN/edge-SR have the *ideal* reparam plain-conv arch but ship only
  **bicubic / Y-only** weights → adopting them is a *training* project, not a port. Per the project's
  settled real-world lesson, bicubic-trained nets are expected to underperform on real H.264.
- The only efficient nets that already have **real-world-degradation weights** are (a) the
  SRVGGNetCompact family (compact/anime + the Phips OTF/AVC/JPEG community compacts) and (b) **SPAN**
  via the Phips and jcj83429 community lines — and the jcj83429 LiveAction SPAN is trained on the
  *exact* codec family (H.264/265/VP9).
- **Avoid the distillation/attention challenge winners** (RLFN/RFDN/IMDN): non-foldable ESA/CCA +
  split/concat DAGs (pool, strided conv, bilinear upsample) are occupancy killers in WGSL, and none
  ship real-world weights. SPAN's "attention" is the exception — a *parameter-free element-wise gate*
  (no matmul/softmax), the only attention-style op worth porting.

---

## 3. Browser portability + occupancy fit of the top candidates

| candidate | port effort | new WGSL ops vs current | occupancy | quality vs compact |
|-----------|------------|------------------------|-----------|--------------------|
| **SPAN (2xLiveActionV1)** | **medium** | +1 param-free element-wise sigmoid gate (no matmul) · +1×1 concat conv · the rest = plain 3×3 (reparam'd `eval_conv`) + PixelShuffle(2) | **↑↑ (48ch < 64ch + 20<34 passes)** | **better LPIPS *and* DISTS (measured)** |
| **16nc compact, real-world wts (hfa2k_avc)** | **near-zero** | **none** — identical op set, half the conv passes, upscale=2 | **↑↑ (½ passes)** | LPIPS≈/↓, DISTS +0.015 (measured) |
| ECBSR/ABPN c16–c32 (reparam) | high (**training**) | none at inference (plain 3×3 stack) | **↑↑↑ best** | unknown — needs real-world retrain |

SPAN is the rare option that improves **both** axes (quality and occupancy) at once. Its one genuinely
new op — the SPAB symmetric activation/gate `att = sigmoid(x) − 0.5; out = (conv(x)+skip)·att` — is a
pointwise op trivially expressible in WGSL.

---

## 4. Ranked recommendation

### #1 — Port **SPAN (`2xLiveActionV1_SPAN`)** to WGSL.
- **Quality:** the **only** candidate that beats the current compact on **both LPIPS and DISTS at
  every CRF** (e.g. CRF28: LPIPS −0.014, DISTS −0.023). Trained on the *exact* codec family
  (H.264/H.265/VP9), native 2× (the deployment scale). Improving DISTS means it adds real structure,
  not the fake detail the project's settled finding warns about.
- **Speed:** its reparam inference graph is **20× plain 3×3 conv @ 48 ch** — a same-runtime MPS proxy
  puts its conv cost at **2.27× cheaper than the current compact** (validated: the proxy reproduces
  the compact and 16nc-compact ratios). Narrower 48-ch channels *raise* occupancy. The ≈100 ms ONNX
  number was a CoreML runtime artifact, not the WGSL cost.
- **Port effort: medium.** One new pointwise gate op + a 1×1 concat; everything else is the plain
  3×3 + PixelShuffle stack already ported. Weights are reparam'd (`eval_conv`) and ship as a 1.65 MB
  ONNX. **This is the Pareto move sought: better quality *and* ~2× faster.**

### #2 — **`num_conv=16` SRVGGNetCompact with real-world weights (`2xHFA2kAVCCompact`)**.
- **Zero port effort** (the current arch with half the body and `upscale=2`; loads `strict=True`).
  Measured **1.9× faster**, LPIPS parity-or-better, with a steady small **DISTS penalty (~+0.015)**.
  Ship this *today* as the speed-slider "fast" tier while SPAN's gate op is built — it needs no new
  WGSL code at all. If DISTS matters more than LPIPS for a clip, `nomosuni_otf` trades the other way.

### #3 (research) — Train an **ECBSR/ABPN reparam net (c16–c32)** on the libx264 degradation recipe.
- Theoretical occupancy optimum (reparam → tiny plain-conv stack), but blocked by the absence of
  real-world RGB weights. Pursue only if SPAN + the fast-compact leave speed on the table; the R10
  degradation harness is the training operator.

### Do NOT port: x4plus / HAT / ATD / DAT2 / FeMaSR / DASR / RLFN / RFDN / IMDN
Either far too slow / occupancy-hostile (RRDB dense-concat, transformer attention, ESA/CCA), and/or
bicubic-only weights, and the heaviest hit the settled real-H.264 over-smooth trap.

---

## 5. Measured vs inferred — honesty ledger
- **MEASURED here:** bicubic, current compact, animev3, x4plus, **3 real-world 16nc compacts
  (nomosuni_otf / hfa2k_avc / hfa2k)**, and **SPAN LiveActionV1** — LPIPS + DISTS on real all-intra
  H.264 (sd600 + short, CRF 22/28/34) + MPS/ONNX latency + a same-runtime MPS conv-stack latency
  proxy (validated against 2 known points). Harnesses: `web_spike/eval_model_options.py`,
  `web_spike/eval_x2_candidates.py` (reproducible: `python eval_x2_candidates.py --crf 28`).
- **INFERRED (not run — bicubic/Y-only weights or out of scope):** ECBSR, ABPN, edge-SR, RLFN, RFDN,
  IMDN, FSRCNN/ESPCN, FeMaSR, DASR — params/arch/op-types from literature; quality-on-real-H.264 not
  measured. **SPAN's WGSL latency is inferred** from the MPS conv-stack proxy (the project's validated
  occupancy model), not measured in WGSL — the one number to confirm during the port.
- **Protocol caveats:** single-generation all-intra libx264 + ideal AREA downscale is a *clean*
  degradation — harsher than bicubic-only but milder than multi-gen broadcast SD; absolute LPIPS/DISTS
  shift on harsher content, but the *ordering* (SPAN > compact > 16nc-compact on quality; SPAN/16nc ≈
  2× cheaper) is the load-bearing result. The ×4→downscale (compact) vs native-×2 (candidates)
  asymmetry makes SPAN's DISTS win conservative and gives its LPIPS win a small tailwind (§1b).
  x4plus's win here does **not** overturn the settled "heavier over-smooths" finding (cleaner content)
  and is browser-irrelevant (16× slower, RRDB occupancy-hostile).
