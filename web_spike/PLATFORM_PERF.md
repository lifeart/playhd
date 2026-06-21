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

## 4. Cross-GPU portability (Apple-tuned today) → `conv_opt/PORTABILITY.md`

The wtile/combo params (256-thread 16×16 workgroup, OCB=32/64, fully-unrolled vec4 accumulators) were tuned ONLY
on this Apple GPU. NVIDIA (warp 32, large register file), AMD (wave 64), Intel (SIMD 8/16/32) and mobile differ in
SIMD width, register file size, and shared-memory budget — so the optimal OCB/tile/precision differ, and an
Apple-optimal config can spill or collapse occupancy elsewhere. The adapter-aware selection table (query
`adapter.limits`/`info` → pick params per family, with a conservative fallback) is in `PORTABILITY.md`.

## 5. Subgroups (cross-lane ops without shared memory) → (investigation in progress)

Apple SIMD = 32 lanes; `subgroups` is exposed (added to the harness). Whether `subgroupBroadcast`/`subgroupShuffle`
can beat the shared-memory weight/halo cache (lower latency, no barrier) is under test — results appended here.
