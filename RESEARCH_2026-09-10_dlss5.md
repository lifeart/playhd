# playhd — Deep Research: DLSS 5, FSR/XeSS, and what actually ports to WebGPU (2026-09-10)

Triggered by a simple question: *NVIDIA shipped DLSS 5 — can we use the papers, or port parts of it to
WebGPU?* Three parallel research streams (Opus agents) plus a direct read of NVIDIA's own technical
report. The answer splits cleanly: **the model is unportable, but the report corrects a load-bearing
assumption in our own notes.** Round R13 then tested the transferable half; results in
`experiments/r13_e*/REPORT.md` and summarised in §5.

## 1. What DLSS 5 actually is

Not an upscaler. A *generative appearance enhancer* that runs as a rendering stage **after** DLSS Super
Resolution, at output resolution, in games. From NVIDIA's technical report (2026-09-01):

- **~154M-parameter transformer**, **one-step deterministic diffusion**, **pixel space with no VAE**.
  They reject latent space explicitly on preservation grounds — the encode/decode round trip shifts
  edges and thin structures: *"even the individual pixel aliasing of the rendered input survives generation."*
- **Causal frame-wise streaming**, O(1) memory. They explicitly reject chunk-based temporal attention
  (Wan, HunyuanVideo, Cosmos 3) for costing K−1 frame intervals of latency and O(K) memory.
- **~8 ms per 3840×2160 frame on an RTX 5090, 731 MB peak** (§4.3.3). Digital Foundry independently
  measured 128.4 → 62.7 fps at 4K — ~51% of frame time on the fastest consumer GPU that exists.
- **FP8 GEMMs + Blackwell TMA**; shipped cubins are `sm_120`-only.
- **Closed.** Streamline 2.14.1's changelog announces `sl.dlss_nr` but the release does not ship it. The
  runtime is a 158 MB DLL that is ~90% one opaque weight blob (not ONNX). Every third-party
  reimplementation reads NVIDIA's weights out of the user's own DLL.

## 2. THE finding — "3D-guided" is a training-time property, not a runtime contract

This corrects `handoff.md:1227` ("FSR2/DLSS depth-based disocclusion does NOT transfer — decoded video
has no depth buffer"). True of FSR2 and DLSS 4. **Not true of DLSS 5.** From the report, twice:

> Abstract: "At inference, the model is conditioned on the current rendered frame, engine motion
> vectors, carried temporal state, and artistic-direction values. **During training**, consistency
> supervision from renderer-derived scene attributes further grounds generation…"

> §2.1: "During training, we use renderer-derived albedo, surface normal, and lighting information to
> provide attribute-specific consistency supervision… **These signals are used only during training and
> are not inputs to the shipped model.**"

**DLSS 5's runtime interface is RGB + motion vectors + carried temporal state — exactly playhd's input
set.** The 3D knowledge is baked in through training supervision, not consumed at inference.

*Caveat:* the shipped `nvngx_dlssnr.dll` does expose `DLSSNR.Depth` as a core NGX parameter — an
unexplained gap between the report and the binary. Widely-repeated claims that DLSS 5 consumes
G-buffers at runtime come from misreading the GeForce page's "trained to recognize engine data such
as…" as an input list; they are wrong.

## 3. Why a port is off the table (three independent blockers)

1. **Nothing to port.** No weights, no ONNX, no shaders. Reimplementations consume NVIDIA's weight
   blob — a non-starter for a repo made commercial-clean when SPAN was dropped over CC-BY-NC.
2. **Compute, by our own numbers.** Our compact anchor is 1.21M params at 121.7 ms (f16, Chrome/Dawn,
   640×320→1280×640). 154M params is ~127× the weights; at equal resolution that is **seconds per
   frame**. Grant 5–10× for better kernels and it is still seconds. WebGPU has no FP8 and no
   cooperative-matrix path — precisely where NVIDIA's 2× comes from.
3. **Wrong task.** It adds photoreal lighting/materials to *clean synthetic renders*. Its stated
   limitation is that output is hard-bounded by input quality and it *"preserves the geometric
   structure of the input exactly — including defects such as aliasing."* Our input is
   compression-degraded natural video; preserving defects exactly is the opposite of the job.

## 4. What genuinely transfers

- **The training recipe (highest leverage, untested).** Supervise with auxiliary signals you will not
  have at inference. playhd's analogue: train the anchor SR with codec-side data (per-MB QP, residual
  energy, MV fields) as consistency supervision, shipping a model that takes RGB alone. This composes
  with R10's conclusion that our sharp external frontier is a libx264-finetuned model, and R10's
  harness is already the degradation operator such a finetune needs.
- **Two of our NO-GOs independently confirmed.** Our "Diffusion anchor — NO-GO" rejected iterative
  sampling + TAESD latent. NVIDIA reached the same two conclusions for a strongly-conditioned
  enhancement task. Confirmatory — and it *narrows* rather than reopens the door, since R9/R10 already
  showed generative "fake detail" losing on real H.264 under DISTS.
- **Our architecture is on the right side of their latency argument.** Their rejection of chunked joint
  spatiotemporal processing is the same reasoning that made us choose anchor SR + MV propagation over
  BasicVSR-style windows. R13-E3 then *measured* this for the first time: propagation is **3.7× steadier
  than per-frame x4plus** on calm content (excess MC-flicker +0.258 vs +0.952).
- **Art direction as learned conditioning** (§3.3): structure-strength and tone-strength scalars folded
  into the network rather than a runtime multi-frequency decomposition. Our `anchor_blend_beta=0.85` is
  a hand-rolled global structure-strength knob; per-pixel β remains untested (R9-E1 refuted per-*clip*).

## 5. What R13 tested, and what happened

| lever | source | verdict |
|---|---|---|
| Browser soft confidence gate + per-block codec signals | port of `residual_gate.py` to WGSL | **GATED-GO**, default-OFF. calm LPIPS −4.2%/DISTS −8.7%, jelly −19.5%/−9.7%. `km` (MV magnitude) carries the win; `kp` within noise |
| Per-pixel motion-adaptive EMA | arXiv:2607.09125 mechanism | **GATED-GO**, validated-ready, not wired. Excess flicker −37.5%/−57.6% with LPIPS *and* DISTS improving and var-Lap held |
| FSR3.1 luma instability factor | FSR3.1 MIT source | **NO-GO** on two windows. In FSR its only job is to bypass the colour-box clamp, which we structurally cannot have (R1-E4b) — it arrives with nothing to unlock |
| Server-side per-block confidence | this round's own measurements | **NO-GO**. Conditioned on the residual the signals collapse; best combination weight w=0 |
| `RESIDUAL_GATE` default flip | R12-E1 follow-up | **NO** — DISTS corroborates only 5/8 clips (+27.3% worse on high motion); the project's triangulation rule is not met |
| FSR2 temporal MV divergence | FSR2 MIT source | **NO-GO**. AUC 0.50–0.52 (chance) across 36 configs |
| Ruder softmax-splat (deletion candidate) | challenge to an incumbent | Perceptually inert (|ΔLPIPS| ≤ 0.0001 on 3 windows), but **flip held** — its production arm never ran |
| Temporal hysteresis of gate membership | R13 cross-agent ask | **NO-GO** in the symmetric-EMA form; asymmetric/sticky form untested |

**R13's most useful non-lever result:** the temporal instability we call "желе" is **flicker, not drift**
— excess flicker is flat in frames-since-anchor, and sequences with *no history at all* (bicubic,
per-frame x4plus) show the same slope. It concentrates in pixels whose source flips between propagated
warp and fresh per-frame SR (12.4× the per-pixel flicker of a stable pixel; a 6.25 Hz GOP-cadence pulse
on high motion). This retro-explains why R1-E2b's adaptive re-anchoring was a NO-GO: **it fixes drift we
do not have.**

**A new methodological GOTCHA, and it is general:** *whenever the gate's fallback source coincides with
the metric's reference, that metric is a distrust meter, not an arbiter.* Browser (fallback = bilinear
LR) ⇒ PSNR and tOF-vs-LR disqualified. Server (fallback = per-frame SR) ⇒ LPIPS/DISTS-vs-per-frame-SR
disqualified. Applying it retroactively: **R12-E1's headline −29% LPIPS is confounded** (LPIPS is
strictly monotone in distrust and the a→0 endpoint is the best in the table) — but the **−23% DISTS is
not**, since DISTS is non-monotone with an interior optimum that beats the a→0 endpoint by 3.8%.

## 6. Provenance warning — three of five research measurements did not survive

The FSR/video-research stream produced five original measurements on `sample.mp4`. R13 re-derived them
in the production harness. **Do not cite the originals:**

| claim | fate |
|---|---|
| "Ruder splat adds ~0 recall for +17–40% mask area" | **REFUTED as structurally invalid** — its ground truth was a re-thresholding of the reactive signal itself (containment 0.83–0.96; `react recall ≡ containment` to 3 decimals). Recall ≈0.99 was arithmetic |
| "FSR2 temporal MV divergence AUC 0.813/0.741" | **NOT REPRODUCIBLE** — measured 0.50–0.52 (chance) across 36 configs including MV dilation |
| per-block triple: partition 8.6×, MV magnitude ~75×, hole 18.6× | **DO NOT REPRODUCE** — partition 4.82×; MV magnitude *non-monotonic* (no 75× spread); hole 1.1–14.8× per clip, so the pooled figure is not a stable property |
| ANVIL MV smoothing rejection (+98% warp error) | untested by R13; plausible, unverified |
| FSR2 *spatial* MV divergence AUC 0.39 (below chance) | consistent with R13-E4's independent finding; treat as corroborated |

The *direction* of the block signals survives (R13-E1's block-augmented browser gate was the only
variant regressing on nothing) — it is the specific ratios that do not.

## 7. Platform status (verified 2026-09-09)

- **WebGPU cooperative/subgroup matrix: behind an experimental flag, not shippable.** `gpuweb#4195` open
  since 2023; Dawn ships it as `ChromiumExperimentalSubgroupMatrix` requiring `--enable-unsafe-webgpu`.
  There is no Tensor-Core-equivalent path in the browser, which is the second blocker in §3.
- **f16 is available everywhere that matters** (Chrome 120+, Safari, ~95% field coverage).
- **`requiredLimits` is a measured no-op for us today** — we get 16 KiB/256 by default and Apple grants
  32 KiB/1024 on request, but `combo.wgsl` already fits in 256 invocations / ~3.6 KiB.
- **Chrome quantizes timestamp queries to 100 µs by default** — launch with
  `--disable-dawn-features=timestamp_quantization` or time N repeats.
- **WebCodecs still exposes no motion vectors**, and *zero* issues have ever been filed proposing it.
  WASM libav `+export_mvs` remains the only path; unchanged through 2025–26.

## 8. Legal flags

- **FSR2/FSR3 shaders: per-file MIT, verified.** Usable — R13 read them for exactly this reason.
- **FSR4 sources: leaked-and-withdrawn.** AMD's SDK license forbids reverse engineering. Read for ideas,
  **do not vendor.**
- **XeSS: binary-only**, license bans disassembly. Not a source option.
- **AMD holds patents on the FSR temporal upscaler** (e.g. US 12293485) even where shader source is MIT.
  Readable source ≠ patent freedom. Not legal advice — flag for review before shipping derived work.
- **DLSS 5 itself:** closed weights; every reimplementation depends on NVIDIA's DLL. Off-limits.

## 9. Primary sources (fetched and verified)

- **DLSS 5 technical report** — `https://research.nvidia.com/labs/adlr/files/DLSS5_Report.pdf`
  (24 pp., 2026-09-01; NVIDIA's own `[Paper]` link 403s — the asset lives one directory up).
  No arXiv preprint, no SIGGRAPH paper; presented at NVIDIA's SIGGRAPH 2026 research keynote.
- **DLSS 5: 3D-Guided Neural Rendering** — `https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/`
- **NVIDIA newsroom announcement** (GTC, 2026-03-16) —
  `https://nvidianews.nvidia.com/news/nvidia-dlss-5-delivers-ai-powered-breakthrough-in-visual-fidelity-for-games`
- **Streamline SDK** — `https://github.com/NVIDIA-RTX/Streamline` (2.14.1 announces `sl.dlss_nr`, does not ship it)
- **FSR3.1 luma instability / FSR2 depth clip** — FidelityFX-SDK v1.1.4, per-file MIT
- Lineage the report cites itself: **PixelDiT** (arXiv:2511.20645), **EDM** (2206.00364),
  **EDM2** (2312.02696), **vid2vid** (1808.06601), **simple diffusion** (2301.11093);
  **Cosmos 3** (2606.02800) cited as the counterexample it rejects.

## 10. Bottom line

DLSS 5 is unportable and aimed at a different problem, but its report is the best available statement
that **an engine-free runtime interface (RGB + MVs + carried temporal state) is sufficient**, provided
the 3D knowledge goes in during training. That is the shape playhd already has. The open frontier is
unchanged from R10 and now better motivated: **a codec-matched trained anchor** — not a better runtime
heuristic. R13 tested the runtime-heuristic half of the space fairly thoroughly and retired most of it.
