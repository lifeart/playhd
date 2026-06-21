# Advanced WebGPU acceleration for the compact-SR conv: dp4a & cooperative-matrix

Honest feasibility study of two "next-tier" WebGPU acceleration paths for the SR anchor
kernel — **packed int8 dot products (dp4a)** and **cooperative / subgroup matrix
(tensor-core) ops** — on the GPUs where they would help most (NVIDIA / AMD / Intel desktop).

**Critical measurement caveat (applies to every number in this doc):** this machine is
**Apple-GPU only**. Nothing here is measured-here. dp4a and cooperative-matrix give their
advantage on NVIDIA/AMD/Intel desktop hardware that cannot be tested on this Mac. Every
performance figure below is **inferred or vendor/source-cited**, never measured on this
project's kernel. The only thing we *have* measured for this conv is the shipped baseline:
**f16 weight-tiled "combo" = 121.7 ms in Chrome/Dawn on Apple (11.8× over naive, already
below the ~130 ms native reference)** — see `../PLATFORM_PERF.md`.

The kernel under study: SRVGGNetCompact / realesr-general-x4v3, **34× 3×3 convs**, channels
mostly **64→64** (first `in_c=3`, last `out_c=48` feeding a ×4 pixelshuffle), conv body runs
at **256×256** spatial, planar layout. It is **floating-point**; fp16 is validated as
visually identical (LPIPS-identical, mean 0.016 / max 7 codes vs PyTorch). The project's
quality program explicitly fights *fake detail* and has validated quality **only down to
fp16** — that bar matters enormously below.

---

## TL;DR — two verdicts

| Path | Shippable to end users today? | Needs model change? | Quality risk | Where it helps | **Verdict** |
|---|---|---|---|---|---|
| **A. dp4a (`dot4{U,I}8Packed`)** | **Yes** — stable since Chrome 123, all devices | **Yes — full int8 quantization** (model is fp, not int8-trained) | **High & unvalidated** against the LPIPS/DISTS/anti-fake-detail bar | NVIDIA Pascal+, AMD Vega20/RDNA2+, Intel Gen11+ (4× int8 ALU) | **NO-GO now / conditional-FUTURE** — gated entirely on int8-SR quality, not on the kernel |
| **B. cooperative / subgroup matrix** | **No** — flag-only experimental, unstandardized, **absent on D3D12** | **No — keeps validated f16 model** | **Low** (stays fp16) | NVIDIA tensor, AMD WMMA (RDNA3+), Intel XMX, Apple simdgroup_matrix | **FUTURE** — strongest path, quality-safe, but not deployable until it ships unflagged with desktop-backend coverage |

The asymmetry is the whole story: **dp4a is available but quality-risky; cooperative-matrix
is quality-safe but unavailable.**

---

## Path A — Packed integer dot products (dp4a)

### A.1 Availability (well-established — this part is GO)

- **WGSL builtins `dot4U8Packed(u32,u32)->u32` and `dot4I8Packed(u32,u32)->i32`**, plus
  `pack4x{I,U}8[Clamp]` / `unpack4x{I,U}8`, are the `packed_4x8_integer_dot_product` WGSL
  language feature. Detect with
  `navigator.gpu.wgslLanguageFeatures.has('packed_4x8_integer_dot_product')`; gate the
  shader with `requires packed_4x8_integer_dot_product;`.
  [W3C WGSL spec](https://www.w3.org/TR/WGSL/),
  [MDN WGSLLanguageFeatures](https://developer.mozilla.org/en-US/docs/Web/API/WGSLLanguageFeatures).
- **Chrome/Dawn: shipped & stable since Chrome 123** (Feb 2024). Chrome implements it on
  **all devices** — hardware DP4A where present, **a software polyfill (no speedup) where
  not**. So presence of the feature ≠ presence of the speedup.
  [Chrome 123 blog](https://developer.chrome.com/blog/new-in-webgpu-123),
  [Web-AI part 2](https://developer.chrome.com/blog/io24-webassembly-webgpu-2).
- **Firefox/wgpu:** implemented — [wgpu #7494](https://github.com/gfx-rs/wgpu/pull/7494)
  (merged Apr 2025, polyfills across SPIR-V/HLSL/GLSL/MSL/WGSL), with follow-up #7574 adding
  *specialized hardware intrinsics* on SPIR-V/HLSL. So both engines expose it; Firefox is
  recent. [wgpu #7481](https://github.com/gfx-rs/wgpu/issues/7481).
- **Hardware mapping (the 4× int8 lever):** NVIDIA **DP4A** since Pascal GP102–GP106 / sm_61
  (all modern GeForce; *not* GP100); AMD **V_DOT4** via Rapid Packed Math on Vega20 (Radeon
  VII) and **RDNA2 RX-6000+** (RDNA1 lacks it); Intel since Gen11/Xe. On these, int8 runs at
  ~4× f32 ALU throughput. [NVIDIA mixed-precision/CUDA 8](https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/),
  [gpuweb #2677](https://github.com/gpuweb/gpuweb/issues/2677).
  **Apple GPUs have no exposed DP4A** → on this Mac the builtin is polyfilled, i.e. **the
  win cannot even appear here**, let alone be measured.

### A.2 Feasibility for THIS fp SR conv — the quantization reality (this is the catch)

dp4a consumes **INT8** operands. The model is floating-point and **not int8-trained**, so
using dp4a means quantizing it — and because there is no training pipeline in this project,
that means **post-training quantization (PTQ)**. A workable scheme:

- **Weights:** symmetric **per-output-channel** int8 (the standard, robust choice).
- **Activations:** per-tensor (cheap) or per-channel (better) int8 with calibration. **This
  is the hard part for SR.** SR activations have asymmetric, long-tailed, high-dynamic-range
  distributions; per-tensor scales clip the tails and **band smooth gradients** — precisely
  the artifact class this project's quality program is built to avoid.
  ([2DQuant](https://arxiv.org/html/2406.06649v1) documents SR's awkward activation
  distributions explicitly.)
- **Per-layer dequant/requant:** the int32 dp4a accumulator must be rescaled back to int8
  between **every one of the 34 layers**, folding in bias, the **PReLU** (negative slope
  breaks the symmetric-zero assumption → needs careful requant), and the global **residual
  skip** (wide dynamic range). First (`in_c=3`) and last (`out_c=48` → pixelshuffle) layers
  are degenerate for int8 packing (a 4-lane dot wants K multiple of 4; `K=3` wastes lanes).

**Is int8 SR quality acceptable here?** *Maybe on PSNR, but PSNR is the wrong question.*

- 8-bit is far more forgiving than the 2–4-bit regime most SR-quant papers study (2DQuant
  4-bit ≈ **−0.28 dB**, 2-bit ≈ **−2.15 dB**; 8-bit is usually near-PSNR-lossless and not
  even tabulated). [QuantSR (NeurIPS'23)](https://proceedings.neurips.cc/paper_files/paper/2023/file/b2169d573d75ff90c7b12dc3a5fc2898-Paper-Conference.pdf),
  [2DQuant](https://arxiv.org/html/2406.06649v1).
- **But** ESRGAN-family models are documented as **quantization-sensitive** ("greatly reduced
  performance"), and getting within ~0.2 dB generally requires **quantization-aware
  retraining**, not PTQ. [Multi-precision SR quantization survey](https://www.researchgate.net/publication/354394805_Super-Resolution_Model_Quantized_in_Multi-Precision).
- AMD *did* ship an int8 Real-ESRGAN ([amd/realesrgan-…-amdnpu](https://huggingface.co/amd/realesrgan-512x512-tiles-amdnpu)) — so int8 SR is **feasible** — but for an NPU target, with undisclosed (likely retrained)
  methodology and some reported metric movement. That is a different quality contract than
  this project's.
- **Decisive point:** this project's bar is **LPIPS/DISTS perceptual fidelity + the
  anti-fake-detail program, validated only to fp16.** int8 PTQ injects an *unvalidated*
  quantization-noise floor on top of a model whose quality was tuned and signed off at fp16.
  No part of the existing validation covers it; it would all have to be re-run (per-clip,
  the robustness sweep, the cut/low-light cases) before int8 could ship.

### A.3 Does the win even survive the requant — and on which GPUs?

- Cited dp4a wins are **1.7–2.9× over f16** — but those are **matrix-vector / LLM** kernels
  ([Web-AI part 2](https://developer.chrome.com/blog/io24-webassembly-webgpu-2)), which are
  ALU-and-bandwidth-dominated in a way that favors int8 on both axes. **This conv is different.**
- Per `../conv_opt/PORTABILITY.md`, the measured f16 win on *this* kernel was largely
  **bandwidth + register relief, not pure ALU**. dp4a's 4× is an **ALU-only** lever. Where the
  kernel isn't ALU-bound, the 4× is heavily discounted — and the **per-layer requantize
  (scale, round, clamp) adds non-trivial int work between every conv**, eating further into it.
- Net inferred outcome: a **partial** win (maybe ~1.3–2× over f16) **only on** NVIDIA/AMD/Intel
  desktop with real DP4A/V_DOT4 hardware; **zero on Apple** (polyfill) — i.e. zero on the only
  platform we can test, and zero on much of mobile.

### A.4 Effort & verdict

- **Effort: high.** Build a PTQ pipeline (calibration set, per-channel weight + activation
  scales, PReLU/residual/bias folding), rewrite all 34 layers as packed-int8 dp4a kernels with
  per-layer requant, handle the degenerate first/last layers, **and** re-run the full
  perceptual + robustness validation suite to prove no quality regression. Plus a runtime
  detect (`packed_4x8_integer_dot_product` *and* real HW, not polyfill).
- **Verdict: NO-GO now / conditional-FUTURE.** The blocker is **not** the kernel or the API
  (both are ready and stable) — it's that int8 introduces an **unvalidated quality risk
  against a perceptual bar deliberately set at fp16**, the speedup is ALU-only and partly eaten
  by requant, and it forks the model. **Only pursue if** a quantization-aware int8 SR model is
  trained *and proven LPIPS/DISTS-neutral* against the existing validation suite. Until then
  it trades the project's hard-won quality story for a partial, desktop-only, unmeasurable-here
  speedup. Not worth it.

---

## Path B — Cooperative / subgroup matrix (tensor cores)

### B.1 Availability (this is the catch for Path B)

- **WebGPU "subgroup matrix" is a proposal, not a shipped feature.**
  [gpuweb #4195](https://github.com/gpuweb/gpuweb/issues/4195) (open, Milestone-2/WGSL) and
  ["What's next for WebGPU"](https://developer.chrome.com/blog/next-for-webgpu) list it as a
  **prioritized next-gen feature** to "take advantage of fixed-size matrix-multiplication
  hardware next to shader cores" — explicitly still in proposal.
- **Dawn prototype exists but is flag-only/experimental:**
  `wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix` /
  `enable chromium_experimental_subgroup_matrix`, gated behind Experimental Web Platform
  Features / a Dawn toggle. **Not enabled for end users.** Notably, **Chrome 134 shipped plain
  *subgroups* as stable but says nothing about subgroup matrix** — confirming matrix is a
  separate, still-experimental track. [Chrome 134 blog](https://developer.chrome.com/blog/new-in-webgpu-134),
  [ORT PR #23729](https://github.com/microsoft/onnxruntime/pull/23729).
- **Backend coverage — the decisive limitation:** Dawn's subgroup matrix is **NOT available
  on D3D12** (this explicitly blocks Intel XMX on Windows). It works on **Metal** (maps to
  **MSL 3.1 `simdgroup_matrix`**); the Vulkan path maps to **`SPV_KHR_cooperative_matrix`**
  (NVIDIA/AMD/Intel). So on **Windows — the dominant NVIDIA/AMD consumer surface, which Chrome
  drives through D3D12 — it is unavailable today.** [ORT PR #23729](https://github.com/microsoft/onnxruntime/pull/23729),
  [Intel WebGPU AI article](https://www.intel.com/content/www/us/en/developer/articles/community/boost-ai-inference-performance-with-webgpu.html).
- **Types:** the underlying cooperative-matrix HW supports **f16 / bf16 / u8 / i8** inputs with
  f16/f32 accumulate (NVIDIA tensor, AMD WMMA RDNA3+, Intel XMX, Apple simdgroup_matrix).
  [VK_KHR_cooperative_matrix](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_cooperative_matrix.html),
  [gpuweb #4195](https://github.com/gpuweb/gpuweb/issues/4195).
  **The pivotal fact: f16 is a first-class input type → Path B needs NO quantization → it
  keeps the validated fp16 model and carries no int8 quality risk.**

### B.2 Feasibility for THIS conv — conv-as-GEMM at 3×3×64→64

Map each middle conv to a GEMM (im2col or implicit/"conv2d_mm" GEMM):

- **M = H·W = 256·256 = 65,536** (output pixels), **K = 9·Cᵢₙ = 576**, **N = Cₒᵤₜ = 64**.
- Work/layer = M·N·K ≈ **2.4 G MACs ≈ 4.8 GFLOP**; ×~32 middle layers ≈ **~155 GFLOP / frame**.
  fp16 tensor cores (tens–hundreds of TFLOP/s) clear this in **~ms theoretical** — so the conv
  is genuinely a **legitimate, compute-heavy matrix-core candidate**, not too small in
  aggregate.
- **The real question is shape, not total size.** It's a *tall-skinny* GEMM: **M enormous, but
  N = 64 = only 4 tiles of 16, K = 576 = 36 tiles of 16.** Matrix cores want all three dims
  large for fragment reuse; **N=64 limits N-direction reuse**, so efficiency is good (tile M
  heavily, keep the small 576×64 weight matrix resident) but **below peak**. Still clearly net
  positive for the inner 32 layers.
- **Overheads that erode it:** im2col materializes a 9× activation blow-up (M×K fp16 ≈ **75 MB
  per layer**) → use **implicit GEMM** (index the stencil directly into matrix loads) to avoid
  it, which is the harder kernel to write. The **first (Cᵢₙ=3) and last (Cₒᵤₜ=48) layers are
  degenerate** for matrix tiling and stay on the existing f16 kernel.
- Cited evidence is **LLM GEMM/GEMV**, not conv: ORT-Web SubgroupMatrix MatMulNBits gave
  **~3× on Phi-3.5 1K-prefill (15 s→5.4 s) on Metal**; matrix-vector **2.3–2.9× vs int dot
  products**. [ORT PR #23729](https://github.com/microsoft/onnxruntime/pull/23729),
  [Web-AI part 2](https://developer.chrome.com/blog/io24-webassembly-webgpu-2). For a tall-skinny
  N=64 conv the realized factor would be **smaller** than a square LLM GEMM, but a **meaningful
  multi-× over a generic WGSL conv is plausible on real tensor hardware** (inferred, not measured).

### B.3 Which GPUs benefit — and the irony

- **Benefit:** NVIDIA tensor cores, AMD **WMMA (RDNA3+)**, Intel **XMX** — *exactly* the
  desktop GPUs the brief flags as "where they'd help most," and where a generic WGSL conv
  leaves the matrix units idle (note `../PLATFORM_PERF.md`: on NVIDIA, generic WGSL f16 is
  ~1:1 with f32 because it never touches tensor cores — matrix ops are the *only* way to reach
  that silicon from WebGPU).
- **The irony:** those NVIDIA/AMD desktop GPUs are reached through **Chrome → Dawn → D3D12 on
  Windows**, where subgroup matrix **isn't implemented yet**. Today the only reachable path is
  **Metal (Apple)** — where the project already runs and the f16 kernel is already **below
  native (121.7 ms)**, and where the tensor advantage is smallest. So Path B's value is real
  but **stranded behind backend coverage** for now.

### B.4 Effort & verdict

- **Effort: high, and partly blocked by externals.** Write an implicit-GEMM conv over the
  subgroup-matrix builtins, tile for N=64, handle degenerate first/last layers, behind a
  capability+flag detect — **and you can't even run it for users until Chrome ships it
  unflagged with D3D12/Vulkan coverage.** Until then it's a research prototype only.
- **Verdict: FUTURE.** This is the **technically strongest and quality-safest** path: it keeps
  the validated fp16 model (no quantization risk), and the conv really is a GEMM that the
  hardware is built for. But it is **flag-only, unstandardized, and absent on D3D12**, so it is
  **not shippable to end users today**, and the NVIDIA/AMD desktop GPUs it would help most can't
  reach it through Chrome yet. **Revisit when subgroup/cooperative matrix ships unflagged with
  D3D12 (WaveMatrix/SM6.8) + Vulkan coverage** — then prototype on a real NVIDIA/AMD box (not
  this Mac) and gate behind the first-load microbench already designed in `PORTABILITY.md §2d`.

---

## Cross-cutting: why neither can be decided here

- **Apple-only machine.** dp4a is *polyfilled* on Apple (no HW DP4A) → zero win visible here.
  Subgroup matrix on Apple Metal works but Apple is already near-native and is the
  smallest-advantage case. **Every speedup claim above is from vendor docs / LLM benchmarks /
  architectural reasoning — none is this conv on this hardware.** Any future GO must be
  prototyped on a real NVIDIA/AMD/Intel desktop GPU and validated against the project's
  perceptual suite before shipping (the `PORTABILITY.md` narrow-then-microbench + per-device
  cache pattern is the right gate).
- **Both are desktop-NVIDIA/AMD-centric**, while the pipeline's amortized design (heavy SR on
  ~2–8% of frames) already makes the anchor cost modest. The ROI of either path is therefore
  bounded by how often the SR anchor is the actual bottleneck on those GPUs.

---

## Recommendation (1 paragraph)

For a **shippable web product**, neither path is worth integrating **now**, and the two are
not equally promising. **Cooperative/subgroup matrix is the one to pursue — but only when it
ships unflagged with D3D12 + Vulkan coverage**: it keeps the already-validated fp16 model (no
quantization quality risk), the 3×3×64→64 conv is a genuine compute-heavy GEMM that maps to
NVIDIA/AMD/Intel matrix hardware, and it's the *only* way a WebGPU app reaches those tensor
units at all — so it is the right long-term bet, just blocked today by experimental status and
missing backends. **dp4a should be pursued only if int8 SR quality first proves acceptable**:
the API is stable and the 4× int8 lever is real on desktop, but the model isn't int8-trained,
PTQ injects an unvalidated noise floor against a perceptual bar deliberately set at fp16, the
win is ALU-only and eroded by per-layer requant, and it's polyfilled (zero gain) on Apple. In
short: **cooperative-matrix once it ships unflagged (quality-safe, just wait for it); dp4a only
behind a proven int8-SR quality result (kernel is ready, the model isn't).**

---

## Sources

- W3C WGSL spec — packed integer dot product: https://www.w3.org/TR/WGSL/
- MDN WGSLLanguageFeatures: https://developer.mozilla.org/en-US/docs/Web/API/WGSLLanguageFeatures
- Chrome 123 — DP4a shipped: https://developer.chrome.com/blog/new-in-webgpu-123
- Chrome Web-AI part 2 (DP4a perf, cooperative matrix, subgroup perf): https://developer.chrome.com/blog/io24-webassembly-webgpu-2
- Chrome 134 — subgroups stable (no subgroup-matrix): https://developer.chrome.com/blog/new-in-webgpu-134
- "What's next for WebGPU" — subgroup matrix prioritized proposal: https://developer.chrome.com/blog/next-for-webgpu
- gpuweb #2677 (DP4a builtin proposal): https://github.com/gpuweb/gpuweb/issues/2677
- gpuweb #4195 (Subgroup matrix proposal; Metal simdgroup_matrix / SPV_KHR_cooperative_matrix / SM6.8 WaveMatrix): https://github.com/gpuweb/gpuweb/issues/4195
- wgpu #7494 / #7481 (Firefox/wgpu dot4{U,I}8Packed, merged Apr 2025): https://github.com/gfx-rs/wgpu/pull/7494
- ONNX-Runtime PR #23729 (Dawn ChromiumExperimentalSubgroupMatrix MatMulNBits, Metal-only, not D3D12, 3× Phi-3.5 prefill): https://github.com/microsoft/onnxruntime/pull/23729
- Intel — WebGPU subgroup-matrix / XMX: https://www.intel.com/content/www/us/en/developer/articles/community/boost-ai-inference-performance-with-webgpu.html
- VK_KHR_cooperative_matrix (NVIDIA/AMD/Arm; types incl. u8/i8/f16): https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_cooperative_matrix.html
- NVIDIA mixed-precision / DP4A (Pascal): https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/
- 2DQuant — PTQ SR, 4-bit −0.28 dB / 2-bit −2.15 dB, SR activation distributions: https://arxiv.org/html/2406.06649v1
- QuantSR (NeurIPS'23) — low-bit SR quantization: https://proceedings.neurips.cc/paper_files/paper/2023/file/b2169d573d75ff90c7b12dc3a5fc2898-Paper-Conference.pdf
- Multi-precision SR quantization (ESRGAN sensitivity, ~0.2 dB w/ QAT): https://www.researchgate.net/publication/354394805_Super-Resolution_Model_Quantized_in_Multi-Precision
- AMD int8 Real-ESRGAN (NPU): https://huggingface.co/amd/realesrgan-512x512-tiles-amdnpu

*Companion docs: `PORTABILITY.md` (per-GPU auto-tuning + microbench gate), `../PLATFORM_PERF.md`
(measured Apple baseline, subgroups null result).*
