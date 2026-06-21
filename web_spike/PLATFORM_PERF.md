# Platform-related performance — playhd in-browser instant tier

How the optimized anchor conv (and the surrounding pipeline) behaves across **GPU backends, browser
engines, precision, power envelopes, and GPU families** — and where the next platform-specific wins are.
Companion docs: `conv_opt/PORTABILITY.md` (per-GPU-family auto-tuning) and the subgroup investigation.

Baseline for everything below: the 34-layer compact-SR conv, 256→1024, measured in Chrome on this Apple GPU —
naive 1437 ms → **wtile (f32, bit-exact) 210 ms (6.8×)** → **combo (f16, visually identical) 121.7 ms (11.8×,
below the native ~130 ms)**. The live pipeline (`gop_live.html`) ships the combo by default, wtile auto-fallback.

## 1. Browser engine / GPU backend matters as much as the GPU itself

WebGPU is one API over three very different native backends:

| browser | WebGPU backend | native API (macOS) |
|---|---|---|
| Chrome / Edge | **Dawn** | Metal |
| Firefox | **wgpu** (wgpu-rs) | Metal |
| Safari | **WebKit** (own impl) | Metal |
| Deno / Node (our headless harness) | **wgpu** | Metal |

**MEASURED backend gap (same Apple GPU, same WGSL):** the wtile conv was **12.9× in Deno/wgpu** but **6.8× in
Chrome/Dawn**; the combo was **1.91× over wtile in Deno/wgpu** but **1.73× in Chrome/Dawn**. wgpu schedules this
kernel (heavy shared-mem + register-blocked) noticeably better than Dawn on the same hardware. Deno's wgpu also
advertises `maxComputeInvocationsPerWorkgroup: 1024` where Dawn effectively tops out at 256 on Apple.

**INFERENCE (not yet browser-tested):** since Firefox uses the *same wgpu backend* as our Deno harness, **Firefox
likely runs this anchor ~1.5–2× faster than Chrome** — plausibly ~70–110 ms. Safari (Metal-native) is unknown but
has no reason to be slow. **Action:** the kernels are plain WGSL with no backend-specific code, so they *run*
unchanged in every engine — only timing differs. Verify by opening `sr_combo.html` / `sr_wtile.html` in Firefox &
Safari and reading `window.__c.convMs` / `window.__w.convMs`. (Caveat: wgpu's reported limits are its own
abstraction; real Apple-HW caps still apply — don't push 1024-thread workgroups in production.)

**Takeaway:** "is it real-time?" is partly a *browser* question. On Chrome it's below-native today; on Firefox it
is very likely comfortably faster. Don't tune solely against Chrome.

## 2. Streaming overhead — the non-conv cost of an anchor (the next real win)

**MEASURED stage breakdown** (`sr_breakdown.html`, f16 combo, persistent buffers, after the fixes below; the conv
itself reads ~121 ms clean — `sr_combo.html`):

| stage | measured | in production? | fix |
|---|---|---|---|
| **LR decode** (`createImageBitmap` + `getImageData`) | ~13–29 ms | **NO** | the LR frame arrives from the **WASM decoder as raw pixels** (the same software decode that exports the MVs) → direct `writeBuffer`, zero image decode |
| **setup** (CPU f16-pack + upload of the seed) | **76 ms → 1.8 ms** ✅ | yes (now tiny) | **FIXED & shipped:** packed all 64 feature channels when layer 0 reads only `in_c=3` → pack just the 3 LR channels (channels 3..63 are zero-init and unread). Applied to `gop_live` |
| **conv** (34 layers) | ~121 ms (clean) | yes — the real work | the optimized combo kernel (this whole effort) |
| **pixelshuffle** (residual + ×4 shuffle draw) | ~5 ms | yes | already cheap |
| **readback** (`copyTextureToBuffer` + `mapAsync`) | ~13 ms | **NO** | **test-only** parity check — production renders straight to the canvas/anchor texture; delete it |
| per-anchor buffer allocation | (folded above) | minimize | allocate A/B/uniforms/bind-groups **once, reuse across anchors** (fixed size; only contents change) — `sr_breakdown.html` already does this |

**Result:** the original 220 ms `runSR` was **mostly harness artifact**. The 76 ms "setup" was a real bug (64-channel
pack) — now **1.8 ms and shipped to `gop_live`**. Decode (~20 ms) and readback (~13 ms) vanish in production (raw
upload, no readback). **Production per-anchor ≈ conv + ~7 ms ≈ ~128 ms in Chrome** (and likely ~70–110 ms in
Firefox/wgpu — §1). So the anchor cost is now essentially *just the conv*, which is below native. The remaining
pipeline-level work is pure engineering (persistent resources + raw decoder upload + drop readback), no feasibility.

## 3. Precision / power / mobile

- The combo's **f16** halves feature+weight memory traffic and buffer footprint → less bandwidth, **less power**
  (matters on laptops/phones/battery), smaller VRAM. Output visually identical (the project validated fp16 SR as
  LPIPS-identical; here mean 0.016 / max 7 codes vs PyTorch).
- **Mobile GPUs (Mali/Adreno) are tile-based with small register files** but often have *strong* f16 throughput —
  so f16 is doubly attractive there, but the **OCB=64 register-blocking (16 vec4 accumulators) may spill** on a
  small register file. The portability layer must drop OCB on mobile (see `PORTABILITY.md`).
- `shader-f16` is widely but not universally available → `gop_live` already **auto-falls-back to the bit-exact f32
  wtile** when the adapter lacks the feature (the `F16` flag gates the whole f16 path). Correctness is preserved
  everywhere; only speed/precision degrade gracefully.
- The whole architecture is inherently **power-efficient on every platform**: heavy SR runs on ~2–8% of frames
  (the amortized anchor), and the per-frame cost is a cheap MV warp — the browser's worst energy cost (per-frame
  CNN) is structurally avoided.

## 4. Cross-GPU portability (Apple-tuned today) → full analysis in `conv_opt/PORTABILITY.md`

The wtile/combo params were tuned ONLY on this Apple GPU. A broad parameter sweep (OCB × tile × precision ×
double-buffer, Deno/wgpu) produced these portability rules:

- **f16 is the most transferable parameter** — a clean **1.6–1.9× win at *every* OCB/tile**, because it helps on
  three axes at once (ALU throughput, halved bandwidth, **halved accumulator register pressure**) — i.e. it helps
  *most* exactly where a GPU is constrained. On any GPU with `shader-f16`, f16 is the right base. (Already the
  live default, with f32 fallback.)
- **256 threads/workgroup is the only universally safe ceiling.** >256 **hard-fails** (`workgroup size exceeds…`)
  even when the adapter advertises `maxComputeInvocationsPerWorkgroup: 1024` — on Apple/wgpu, 512 fails even after
  explicitly requesting the higher limit. **Never trust that limit above 256 for compute.**
- **The central portability hazard = OCB × thread-count register interaction.** OCB sets accumulator registers
  per thread; thread-count multiplies total demand. The *same* OCB flips from best to worst: `f32 OCB64` is the
  **fastest f32 config at 64 threads (8×8)** but **+37% slower at 256 threads (16×16)** — occupancy collapse from
  register pressure. So the shipped "OCB=32 sweet spot" was conditioned on 256-thread Apple workgroups; it is
  **not a portable constant**. f16 *dodges* this (half the accumulator footprint lets big-OCB + high occupancy
  coexist) — which is why it matters more on small-register GPUs, not less.
- **Mobile (Mali/Adreno) is the danger zone** — small per-thread register files + tile memory. The Apple
  OCB64-at-256-threads default is predicted to spill catastrophically there → the table drops them to OCB16–32,
  smaller workgroups (8×8), f16 (mobile f16 is strong), and **double-buffer OFF** (DB is only ~3% on Apple but
  doubles shared memory — a bad mobile trade).
- **Recommendation: hybrid — a static `adapter`-keyed table narrows to 2–3 candidates, then a first-load
  micro-benchmark picks the winner** (time the shortlist on one small conv layer via `timestamp-query`, tens of
  ms once, cache per `adapter.info`). Pure-static is too fragile (the knife-edge OCB×threads optimum isn't
  exposed by any WebGPU limit; Dawn-vs-wgpu disagree 6.8× vs 12.9×; `adapter.info.architecture` can be
  privacy-masked) — this matches ONNX-Runtime-Web's WebGPU tunable-op autotuning.
- **Safe fallback when in doubt = the bit-exact `f32 OCB32 16×16 256-thread` wtile** (parity-guaranteed, 4.9 KB
  shared < the 16 KB spec minimum, moderate registers); step down to OCB16 / no-DB on a constrained unknown GPU.

Full sweep table + per-family decision table (Apple measured; NVIDIA/AMD/Intel/Mali/Adreno architecture-inferred)
+ the query/validation logic are in `conv_opt/PORTABILITY.md`.

## 5. Subgroups (cross-lane ops without shared memory) → NULL on Apple (and a tooling trap)

Apple SIMD = 32 lanes. Investigated whether `subgroupBroadcast`/`subgroupShuffle` can beat the shared-memory
weight/halo cache (lower latency, no barrier).

**Tooling trap (decisive for the headless harness):** `subgroups` is a **phantom feature** in Deno's wgpu — the
adapter *advertises* it but naga rejects `enable subgroups;` ("not yet implemented", wgpu #5555). So the Deno
timestamp arbiter **cannot time any subgroup kernel** (a naive probe silently reads zeros — same class of bug as
the earlier vacuous-parity trap). Chrome's Tint *does* implement subgroups, so it's the only place to measure them.

**Design analysis (Apple):** three strategies were designed; only **weight-broadcast** is viable (a 32-lane subgroup
loads the per-channel weights once and `subgroupBroadcast`s them, freeing the ~4.6 KB shared weight tile).
Horizontal input-shuffle fails (32 lanes = a 2-row strip, but the 3×3 stencil's *vertical* reuse crosses subgroup
boundaries) and `subgroupAdd` channel-reduction fails (destroys the register-blocking that feeds 64 outputs per
input load). **Expectation: no win on Apple** — the combo is **register-bound** (OCB=64 = 16 vec4 accumulators), so
freeing shared memory doesn't lift occupancy; Apple weight reads are already free uniform threadgroup-broadcasts;
and the explicit broadcasts *add* ~144 ops/channel. The lane-indexing was verified correct via an emulation twin
(`candidate_subgroup_emu.ts`, parity 3e-7 = bit-identical to wtile), so `candidate_subgroup.ts` is correct — just
unmeasured in Deno. **Plausibly useful on NVIDIA/AMD** (cheaper cross-lane shuffle, costlier shared-mem bank
conflicts) — but there the bigger lever is **dp4a / cooperative-matrix (tensor) ops**, not subgroups.

**Chrome confirmation (MEASURED, `sr_subgroup.html`):** the weight-broadcast subgroup conv runs in Chrome at
**556.9 ms, bit-exact** (parity 5e-6, max 1 — confirms the lane-indexing is correct) — i.e. **4.6× SLOWER than the
combo (121.7 ms)** and 2.6× slower than wtile (210 ms). The null is now empirical, not just predicted. Two
compounding reasons: (1) subgroups can't be combined with f16 here (`subgroups-f16` isn't exposed), so the kernel
is stuck on the **f32 OCB64** base — which the portability sweep shows is *already* register-throttled at 256
threads (rel 1.21); and (2) the 144 explicit `subgroupBroadcast`s/channel add overhead the threadgroup-broadcast
path didn't pay. **Verdict: subgroups are a dead end for this kernel on Apple** (the f16 register-relief, which
subgroups can't access, is what actually wins). They may help on NVIDIA/AMD, but there the real lever is
dp4a / cooperative-matrix (tensor) ops — a separate future investigation. Artifacts kept for the record:
`conv_opt/candidate_subgroup.ts` (+ `_emu` correctness twin), `webgpu_warp/{sr_subgroup.html, subgroup.wgsl}`.
