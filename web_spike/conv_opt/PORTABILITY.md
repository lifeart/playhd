# Cross-GPU portability analysis — compact-SR 3×3 conv WebGPU kernel

Scope: the two Apple-tuned kernels `candidate_wtile.ts` (f32, bit-exact) and
`candidate_combo.ts` (f16) were optimized **only** on this Mac (Apple GPU, 32-wide
SIMD). This document quantifies how their tuning parameters behave across a broad
sweep, designs the runtime auto-selection a shippable web pipeline should use to
pick params per GPU family, and gives a safe fallback for unknown/constrained GPUs.

Measurement caveat up front: **all numbers below are Deno/wgpu → Metal on the
Apple GPU** (the same backend Firefox uses; Chrome uses Dawn and differs — this
kernel is 12.9× in Deno/wgpu vs 6.8× in Chrome/Dawn, so Firefox is likely faster
than Chrome). Treat Deno numbers as a **second data point, not ground truth**, and
treat every non-Apple per-family recommendation as **architecture-inferred**, to be
confirmed by a first-load micro-benchmark on real hardware.

Harness: `sweep.ts` (single process, all configs back-to-back, 2 passes × best-of-5,
normalized to an in-process anchor). Generator: `sweepgen.ts` (env-driven, reproduces
both shipped kernels bit-for-bit: f32 parity 3.0e-7, f16 parity 2.7e-3).

---

## 1. Parameter-sensitivity sweep (Deno/wgpu → Metal, SIZE=256, 34 layers)

Anchor = `f32 OCB32 16×16 DB` (the wtile config). Anchor was re-timed each pass:
293 ms then 296 ms → **<1% drift**, so the relative column is contention-robust.
`rel` = config_ms / anchor_ms (lower = faster). `regA` = accumulator register-
pressure proxy/thread (vec4<f32>=2 units, vec4<f16>=1 unit; f16 halves it at equal
OCB). `smem` = workgroup shared bytes (double-buffered configs hold 2×).

```
config                          ms     rel   thr   smem   regA      mad   status
-- f32 (bit-exact, parity 3e-7) --------------------------------------------------
f32 OCB16 8x8   DB            358.4   1.23    64   1952    32   3.0e-7   OK
f32 OCB16 16x16 DB            347.3   1.19   256   3744    32   3.0e-7   OK
f32 OCB16 8x32  DB            346.2   1.19   256   3872    32   3.0e-7   OK
f32 OCB16 32x8  DB            346.3   1.19   256   3872    32   3.0e-7   OK
f32 OCB32 8x8   DB            279.9   0.96    64   3104    64   3.0e-7   OK
f32 OCB32 16x16 DB  (anchor) 292.1   1.00   256   4896    64   3.0e-7   OK
f32 OCB32 8x32  DB            289.8   0.99   256   5024    64   3.0e-7   OK
f32 OCB32 32x8  DB            290.7   1.00   256   5024    64   3.0e-7   OK
f32 OCB64 8x8   DB            257.2   0.88    64   5408   128   3.0e-7   OK  <- fastest f32
f32 OCB64 16x16 DB           353.7   1.21   256   7200   128   3.0e-7   OK  <- SAME OCB, +37% slower
f32 OCB64 8x32  DB           353.0   1.21   256   7328   128   3.0e-7   OK
f32 OCB64 32x8  DB           334.5   1.15   256   7328   128   3.0e-7   OK
-- f16 (combo family, parity 2.7e-3) ---------------------------------------------
f16 OCB16 8x8   DB           275.2   0.94    64    976    16   2.7e-3   OK
f16 OCB16 16x16 DB           252.5   0.86   256   1872    16   2.7e-3   OK
f16 OCB16 8x32  DB           253.8   0.87   256   1936    16   2.7e-3   OK
f16 OCB16 32x8  DB           253.5   0.87   256   1936    16   2.7e-3   OK
f16 OCB32 8x8   DB           194.3   0.67    64   1552    32   2.7e-3   OK
f16 OCB32 16x16 DB           181.4   0.62   256   2448    32   2.7e-3   OK
f16 OCB32 8x32  DB           182.5   0.62   256   2512    32   2.7e-3   OK
f16 OCB32 32x8  DB           182.9   0.63   256   2512    32   2.7e-3   OK
f16 OCB64 8x8   DB           159.8   0.55    64   2704    64   2.7e-3   OK
f16 OCB64 16x16 DB           150.5   0.52   256   3600    64   2.7e-3   OK  <- GLOBAL WINNER (combo)
f16 OCB64 8x32  DB           151.5   0.52   256   3664    64   2.7e-3   OK
f16 OCB64 32x8  DB           152.8   0.52   256   3664    64   2.7e-3   OK
-- accumulator precision at the f16 sweet spot (OCB64 16x16 DB) -------------------
f16 OCB64 16x16 DB ACC=f32   212.2   0.73   256   3600   128   4.0e-4   OK  (tightest f16 parity)
f16 OCB64 16x16 DB ACC=hybr  324.7   1.11   256   3600   128   7.5e-4   OK  (NO-GO: slowest)
-- double-buffer OFF at the two winners ------------------------------------------
f32 OCB32 16x16 noDB         302.9   1.04   256   2448    64   3.0e-7   OK  (DB = +3.6%)
f16 OCB64 16x16 noDB         154.5   0.53   256   1800    64   2.7e-3   OK  (DB = +2.6%)
-- oversize workgroups (>256 threads) --------------------------------------------
f32 OCB32 32x16=512  DB        FAIL         512   7200    64            "workgroup size exceeds..."
f32 OCB32 32x32=1024 DB        FAIL        1024  11552    64            "workgroup size exceeds..."
```

### What is ROBUST vs KNIFE-EDGE

**Robust (safe to ship as a default everywhere):**
- **f16 vs f32 precision choice.** f16 is a clean ~1.6–1.9× win at every OCB/tile,
  with parity 2.7e-3 (passes the project's mean<1e-2 gate with 3.7× margin). On any
  GPU with `shader-f16`, f16 is the right base. This is the single most transferable
  parameter — it wins on ALU throughput AND halves bandwidth AND halves register
  pressure, so it helps most exactly where you're constrained.
- **256-thread ceiling.** Never exceed 256 threads/workgroup (see failure mode below).
- **OCB16 is always too small.** gz=4 input reloads dominate; slowest in every block.
- **Tile *shape* at fixed thread count barely matters** (16×16 vs 8×32 vs 32×8 are
  within ~1% everywhere). Only the *thread count* (256 vs 64) and OCB matter.

**Knife-edge (the optimum flips depending on the GPU's register budget):**
- **OCB × thread-count interaction — the central portability hazard.** OCB sets
  accumulator registers/thread; thread-count multiplies total register demand. The
  same OCB is great or terrible depending on workgroup size:
  - `f32 OCB64 8×8` (64 threads) = **0.88 rel, the fastest f32 config** — but
  - `f32 OCB64 16×16` (256 threads) = **1.21 rel, +37% slower** at the *same OCB*.
  At 64 threads the big register block (128 reg-units/thread) fits; at 256 threads
  it collapses occupancy. The shipped "OCB=32 is the sweet spot" conclusion was
  conditioned on 256-thread workgroups; it is **not** a portable constant.
- **Why f16 dodges this:** `f16 OCB64 16×16` (64 reg-units) is the global winner,
  while `f32 OCB64 16×16` (128 reg-units) is slow. f16 halving the accumulator
  footprint is exactly what lets big-OCB (max reuse, gz=1) coexist with high
  occupancy. On a small-register GPU this matters more, not less.
- **Accumulator precision:** ACC=f16 (150 ms) ≫ ACC=f32 (212 ms, but 6× tighter
  parity 4.0e-4) ≫ ACC=hybrid (325 ms — **NO-GO**, the per-ic f16 tap-sum + f32
  flush costs more than it saves). Keep f16-acc as default; expose f32-acc as a
  "tight-parity / debugging" switch only.
- **Double-buffer:** a flat ~3% win on Apple (292→303, 150→154 ms). Marginal, and it
  **doubles shared memory** — a bad trade on tile-memory-constrained mobile GPUs.
  Safe to drop; not worth defending.

### Failure modes observed / predicted

1. **>256 threads → hard pipeline failure (CONFIRMED).** 512- and 1024-thread
   workgroups fail with *"Shader entry point's workgroup size … exceeds …"*. The
   adapter advertises `maxComputeInvocationsPerWorkgroup = 1024`, but a default
   `requestDevice({})` caps it at the **WebGPU spec default 256** — and on this Apple
   GPU via wgpu, **512 still fails even after explicitly requesting the 1024 limit**
   (verified with a trivial empty kernel). Lesson: **do not trust adapter limits
   above 256 for compute workgroup size; 256 is the only universally safe ceiling.**
2. **Register-spill / occupancy collapse → soft slowdown (OBSERVED as 1.21 rel).**
   On Apple this is a graceful ~37% penalty (Metal spills to thread-local memory).
   On a **small-register tile GPU (Mali/Adreno) the same OCB64-at-256-threads config
   is predicted to spill catastrophically (multi-× slowdown) or fail to compile**
   (inferred — not testable on this Mac). This is the #1 ship risk of the Apple
   default; see §2.
3. **Shared-memory over budget → not triggered here.** Max config used 7.2 KB f32 /
   3.6 KB f16, well under this device's 32 KB. But the WebGPU **spec minimum is only
   16 KB** (`maxComputeWorkgroupStorageSize`), and double-buffering doubles usage.
   On a 16 KB GPU, `f32 OCB64 DB` (7.2 KB) is fine but a larger-tile or higher-OCB
   variant could exceed it → request must be validated against the device limit.

---

## 2. Adapter-aware auto-selection design (the real deliverable)

### 2a. What to query

At init, after `requestAdapter()`:

| Query | Use |
|---|---|
| `adapter.features.has("shader-f16")` | **Gate the f16 path.** No f16 → fall back to f32 (and its OCB×thread caution). |
| `adapter.features.has("subgroups")` + `adapter.info` `minSubgroupSize`/`maxSubgroupSize` (Chrome ≥125, behind the `subgroups` feature) | Detect SIMD width (Apple/NV 32, AMD 32/64, Intel 8–32, Adreno 64/128). Width informs the *thread-count* choice; **do not require it** — it is absent in Deno's wgpu build and may be masked. |
| `adapter.limits.maxComputeInvocationsPerWorkgroup` | Clamp threads. **Treat 256 as the real ceiling** regardless of the reported value (see failure mode 1). |
| `adapter.limits.maxComputeWorkgroupStorageSize` | Validate the chosen tile's `smem` fits; if not, drop double-buffer or OCB. Assume **16 KB** when planning conservatively. |
| `adapter.limits.maxStorageBufferBindingSize` | Sanity for planar feature buffers (not a kernel-param driver here). |
| `adapter.features.has("subgroups")` + `adapter.info.subgroupMinSize` / `subgroupMaxSize` (shipped Chrome **134** stable, default-on; absent in Deno's wgpu) | SIMD width hint (Apple/NV 32, AMD 32/64, Intel 8–32, Adreno 64/128). Informs thread-count; **do not require it**. Field names are `subgroupMin/MaxSize` (not `min/maxSubgroupSize`). |
| `adapter.info.vendor` / `adapter.info.architecture` | Coarse family hint **only**. `requestAdapterInfo()` was removed (Chrome 131); `adapter.info` is now a **sync attribute** (no gesture). The spec does **not** standardize `architecture` values and a UA may return **empty strings** for privacy — prefer `vendor`, design for empty, and never make a config *load-bearing* on a name match. Always pair with a capability + micro-bench fallback. |

**Request the device deliberately:** `requiredLimits` must explicitly ask for the
storage size you need (default is the 16 KB minimum, not the adapter max), and you
must *not* assume >256 threads even if granted.

### 2b. Per-family decision table (Apple = MEASURED; others = architecture-inferred)

The transferable physics from the sweep: **pick the largest OCB whose total
accumulator register demand (OCB-regs × threads) the family's register file holds at
useful occupancy; prefer f16 (halves that demand); cap threads at 256; on tile-based
small-register GPUs, shrink OCB and/or threads and drop double-buffer.**

| Family | SIMD/wave | Reg file | f16 ALU | OCB | tile (threads) | double-buf | Reasoning |
|---|---|---|---|---|---|---|---|
| **Apple** | 32 | large, unified | gen-dep (≈2× pre-A15, ~par M1–M4) | **64** | 16×16 (256) | on | **MEASURED winner: f16 OCB64 16×16 = 150 ms (rel 0.52).** Note the measured f16 win is largely **bandwidth + register-relief**, not pure ALU — which is why it transfers even to families with 1:1 f16 ALU. |
| **NVIDIA** | warp 32 | very large (64K 32-bit regs/SM = 256 KB) | **~1:1 on the vector ALU** (2× is Tensor-Core-only) | **64** | 16×16 (256) | on | Biggest register file → high OCB at full occupancy even in f32. Use f16 anyway (halves bandwidth + accumulator regs → still net win on this memory/register-bound kernel) but **do NOT expect 2× ALU** — a generic WGSL conv doesn't hit Tensor Cores. Dawn backend. |
| **AMD RDNA** | wave32 (def) | large VGPR (~128 KB/SIMD32) | **2× (packed/RPM)** | **64** | 16×16 (256) | on | wave32 like NV; ample VGPRs; packed-f16 (RPM since Vega) gives real 2× ALU. Mirror NVIDIA but f16 helps more. (GCN/wave64 legacy → treat as Intel-tier if detected.) |
| **Intel** | EU SIMD 8/16/32 | 4 KB GRF/thread (128 regs, SRM) | **2× (double-rate)** | **32** | 16×16 (256) | on | Small-Register-Mode is the common case → OCB64-at-256 risks spill; OCB32 is the safe high-occupancy point. Large OCB / >73 B SLM-per-lane can **shrink SIMD width** — another reason to stay at OCB32. Variable SIMD → no subgroup-width assumptions. |
| **Mali** (Valhall/Bifrost) | warp 16 | **small; 33+ regs = ½ occupancy** | **2× but only via packed `f16vec2/4`** (scalar f16 ≠ 2×) | **16** | **8×8 (64)** | **off** | **Danger zone.** Tile-based with **no dedicated shared memory** (workgroup "shared" = cached system RAM) → the kernel's load-weights-into-shared premise is *weak* here; consider a shared-memless variant (future work). OCB64×256-threads would blow the 33-reg occupancy cliff. Smallest OCB + 64-thread workgroup + packed-f16; drop DB (its 2× smem buys nothing without real shared mem). |
| **Adreno** | wave 64/128 | tile, ≤32 KB local; wave128 dropped under reg pressure | **2× (double-rate, 128-wide FP16)** | **32** | 8×8 or 16×16 (≤256) | **off** | High OCB forces narrow waves (wave128→wave64) → loses the FP16 throughput. Conservative OCB32 + packed-f16 to keep wide waves; drop DB. |
| **unknown / SwiftShader / llvmpipe / masked** | ? | ? | maybe | **32** (16 if smem<16K) | 16×16 (256) | on→off if tight | **Capability-only path.** Use f16 if `shader-f16` present else f32; OCB32 16×16 is the broadly-safe middle (64 reg-units, 4.9 KB f32 / 2.4 KB f16, both under the 16 KB min). Then let the micro-bench (§2d) confirm/override. |

### 2c. Safe FALLBACK (the config you ship when in doubt)

**Universal safe config = the bit-exact wtile: `f32 OCB32 16×16 256-thread DB`.**
- Parity guaranteed (3e-7, bit-identical to naive) — no precision risk on any GPU.
- 256 threads (the safe ceiling), 4.9 KB smem (under the 16 KB spec minimum),
  64 reg-units/thread (moderate — fit on every family tested-or-inferred).
- For a **constrained** unknown GPU (smem reported <16 KB, or known small-register
  mobile, or f32 only): step down to **`OCB16 16×16`** (32 reg-units, 3.7 KB) and/or
  **double-buffer OFF** (halves smem to 2.4 KB). Slower (rel ~1.19) but the most
  spill-proof, parity-exact config available.
- Decision order at runtime: `shader-f16?` → choose f16 vs f32 base · family hint →
  pick OCB/tile/DB from the table · **validate** `smem ≤ device limit` and
  `threads ≤ 256` (shrink OCB/drop DB until both hold) · micro-bench the 2–3
  surviving candidates (§2d) · cache the winner per `adapter.info` signature.

### 2d. Static table vs runtime micro-benchmark — recommendation

**Hybrid: static table to NARROW to 2–3 candidates, micro-bench to PICK.** The sweep
shows the optimum is genuinely knife-edge in a way a static `limits`-only table
cannot predict:
- The `f32 OCB64 8×8 > f32 OCB32` surprise (faster at smaller workgroup) is invisible
  to any rule keyed on advertised limits — it emerges from the OCB×threads×register
  interaction, which no WebGPU limit exposes.
- Dawn (Chrome) vs wgpu (Firefox/Deno) differ enough that this kernel reports 6.8×
  vs 12.9× — a static per-family constant baked from one backend can mis-rank on the
  other.
- `adapter.info.architecture` can be privacy-masked, so name-based selection alone
  can silently fall through to a bad default.

A **first-load micro-benchmark is cheap and decisive**: time the 2–3 table-shortlisted
configs on a *small* tile (e.g. one 128×128 conv layer, a few iterations via
`timestamp-query` exactly as `sweep.ts` does) — tens of ms total, once — and keep the
min. Cache by `adapter.info` signature so it's paid once per device. Pure static is
fragile (knife-edge + backend gap + masking); micro-benching the *full* grid on load
is too slow; **narrow-then-time** captures the wins the table can't while staying
fast. Note the precedent survey (§3) found that **no shipping WebGPU project does
this for conv** — ORT-Web's runtime-timing "TunableOp" is wired only to its ROCm/CUDA
backends, not WebGPU — so a narrow first-load micro-bench would put this pipeline
*ahead* of the ecosystem, not merely match it. Tellingly, the two untuned SR libraries
(Anime4K-WebGPU, websr) both independently converged on `8×8` workgroups — the
mobile-safe point — which is also our measured small-workgroup f32 winner.

---

## 3. Real-world precedent (cross-GPU WebGPU kernel selection)

**Headline: of four shipping WebGPU ML/SR projects, _none_ do meaningful per-GPU-family
conv/param adaptation.** Three do zero GPU introspection and hardcode one workgroup
size; only ONNX-Runtime-Web reads limits and has a handful of narrow vendor branches,
and even it sizes conv/matmul tiles with static shape heuristics — not autotuning.
There is no precedent for sophisticated cross-GPU OCB/workgroup selection; the
adapter-aware + micro-bench design above would be ahead of the field.

- **Anime4K-WebGPU** (`github.com/Anime4KWebBoost/Anime4K-WebGPU`): bare
  `requestAdapter()`→`requestDevice()` with **no** `requiredFeatures`/`requiredLimits`;
  never queries `adapter.limits`/`info`/`features`. **Every** WGSL uses hardcoded
  `@workgroup_size(8,8)`, dispatch `ceil(dim/8)`. No vendor branches. f32 compute,
  with half-precision *storage* via the core `rgba16float` texture format (no
  `shader-f16` feature). Sources: src/renderer/index.ts, src/pipelines/helpers/Conv2d/index.ts,
  src/pipelines/upscale/CNNx2VL/shaders/conv2dtf.wgsl on that repo.
- **websr** (`github.com/sb2702/websr`, npm `@websr/websr`): hand-written WGSL CNNs,
  zero runtime deps. **No tiling** (whole frame in one pass); hardcoded
  `@workgroup_size(8,8)` (`num_work_groups=8`); no `adapter.info/limits/features`, no
  `shader-f16`, f32 compute. Sources: src/layers/base_compute_layer.ts, src/renderer.ts,
  src/main.ts on that repo.
- **ONNX-Runtime-Web (WebGPU EP) + transformers.js**: the most sophisticated, still
  **static**. `backend-webgpu.ts` reads `maxComputeWorkgroupStorageSize/InvocationsPerWorkgroup/SizeX`
  and passes them as `requiredLimits`, **but the packed conv/matmul shaders don't use
  them** — `matmul_packed_webgpu.ts` / `conv2d_mm_webgpu.ts` fix `workgroupSize=[8,8,1]`
  with `elementsPerThread = dimAOuter<=8 ? [4,1,1] : [4,4,1]` (a shape branch, with a
  literal `// TODO: fine tune size`). A few narrow vendor branches exist
  (`matmulnbits.ts`: `isVendor('intel') && isArchitecture('gen-12lp')`; an Intel-tuned
  "wide tile" MatMulNBits path, PR #23908). Subgroups are used only for select contrib
  ops (FlashAttention, PR #22932 — branches on subgroup size, requires ≥16, degrades
  gracefully), **not** generic conv/matmul. ORT's runtime-timing **TunableOp is
  ROCm/CUDA-only — not wired to the WebGPU EP** (absence-of-evidence, corroborated by
  the static-heuristic design). transformers.js delegates entirely to ORT-Web
  (`device:'webgpu'`→`executionProviders:['webgpu']`; recommends `q4f16` for WebGPU).
  Sources: microsoft/onnxruntime js/web/lib/wasm/jsep/{backend-webgpu.ts,
  webgpu/ops/3rd-party/matmul_packed_webgpu.ts, .../conv2d_mm_webgpu.ts,
  webgpu/ops/matmulnbits.ts}, PRs #22932/#23908.
- **wonnx** (`github.com/webonnx/wonnx`, Rust/wgpu): **hardcodes spec-default caps**
  (`MAX_WORKGROUP_SIZE_X/Y=256`, `MAX_COMPUTE_WORKGROUPS_PER_DIMENSION=65535`) as
  constants and creates the device with `Limits::default()` — it **never inspects real
  hardware limits**. Workgroup sizing = ceiling-split against those caps (Tera-templated
  `@workgroup_size`), erroring if exceeded. No vendor code, f32-only. Sources:
  wonnx/src/{compiler.rs,gpu.rs,resource.rs}, templates/matrix/gemm.wgsl.

**Spec-default limits (W3C WebGPU §3.6.2, confirmed):**
`maxComputeInvocationsPerWorkgroup = 256`, `maxComputeWorkgroupStorageSize = 16384` B,
`maxComputeWorkgroupSizeX/Y = 256/256`, `…SizeZ = 64`, `maxComputeWorkgroupsPerDimension
= 65535`. Adapters may report more (desktop often 32 KB storage) but only via explicit
`requiredLimits`. Source: w3.org/TR/webgpu/#limits, MDN GPUSupportedLimits.

**Family register/SIMD/f16 facts used in §2b (research-sourced, several flagged
approximate):** NVIDIA 64K 32-bit regs/SM (256 KB), consumer vector f16 ≈1:1 with f32
(double-rate is Tensor-Core/datacenter only) — NVIDIA Pascal tuning guide & mixed-
precision blog. AMD RDNA ~128 KB VGPR/SIMD32, packed-f16 RPM 2× — GPUOpen RDNA guide.
Apple ~208 KB reg file/core (~6.5 KB/thread, microbench-derived), API-enforced 32 KB
threadgroup memory, f16 rate generation-dependent — metal-benchmarks. Mali: no
dedicated shared memory (cached system RAM), warp 16, **33+ regs → half occupancy**,
recommended workgroup 64 / 8×8, packed-`f16vec2` for 2× (scalar f16 ≠ 2×) — ARM arch
notes. Adreno: tile-based, ≤32 KB local, wave 64/128 with wave128 dropped under
register pressure, double-rate FP16 — Chips&Cheese Snapdragon analyses. Intel: 4 KB
GRF/thread Small-Register-Mode, SIMD8 native, large SLM-per-lane shrinks SIMD — Intel
oneAPI GPU optimization guide. Subgroups shipped Chrome 134 stable; `subgroupMin/MaxSize`
on `adapter.info`. **Caveat:** Dawn-vs-wgpu compute perf-gap numbers and architecture-
string registry (`apple-g13`/`rdna-2`) could not be confirmed from primary sources —
only NVIDIA `"turing"` is spec/MDN-confirmed; design for empty `architecture`.

---

## Reproduce

```
# f16 candidates need the newer Deno (system deno can't compile `enable f16`):
/tmp/deno_latest/deno run --allow-read --unstable-webgpu sweep.ts 256        # full sweep
# one config through the existing harness:
OCB=64 TW=16 TH=16 F16=1 ACC=f16 PROD=f16 DB=1 \
  /tmp/deno_latest/deno run --allow-read --allow-env --unstable-webgpu bench.ts sweepgen.ts 256
# adapter capabilities:
/tmp/deno_latest/deno run --allow-read --unstable-webgpu plat_probe.ts
```
Files: `sweep.ts` (in-process sweep harness), `sweepgen.ts` (env-driven generator,
reproduces both shipped kernels), `bench.ts` (single-candidate harness), `plat_probe.ts`.
