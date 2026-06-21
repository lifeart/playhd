# playhd — web-only (browser/WebGPU) port: spike + P1 pipeline (all browser-verified)

Investigation of whether the whole pipeline (codec-MV propagation + sparse anchor SR) can run
**entirely in a browser**, no server. Feasibility doc: `../WEB_ONLY_FEASIBILITY.md`. **Outcome: every
component of the instant-tier pipeline now runs and is parity-verified in real Chrome WebGPU, with the
motion vectors as the only remaining offline input.** The architecture is a *better* fit for the browser
than native — it amortizes away per-frame heavy SR, the browser's single worst SR cost.

## The pieces (run each: `python -m http.server` here, open the page in a WebGPU browser)

| # | artifact | what it proves | result |
|---|---|---|---|
| spike | `native_mv_throughput.py` + `node_bench.mjs` | codec MVs are reachable + real-time in WASM (`+export_mvs`) | WASM SD decode+MVs **~850–1000 fps** (~34–40× real-time); native 2392 fps; penalty only ~2.4× |
| warp | `webgpu_warp/index.html` (+ `extract_warp_data.py`) | the codec-MV backward warp matches the prototype `warp_hd` | parity **0.002 codes** (x2), **0.000** (x4); warp **0.096 ms = ~10,400/s** |
| chain | `webgpu_warp/gop.html` (+ `extract_gop_data.py`) | full GOP propagation chain (SR 1 anchor, warp 15) + playback | chain parity **avg 0.016 / worst 0.065** over 16 frames |
| SR | `webgpu_warp/sr.html` (+ `export_compact_weights.py`) | the compact net (SRVGGNetCompact, 34 conv passes) runs on-GPU | parity vs PyTorch **0.0000035 codes** = bit-exact |
| **live** | `webgpu_warp/gop_live.html` (+ `extract_gop_live.py`) | **the whole instant tier**: on-GPU SR anchor + warp chain + reactive occlusion + playback | full-chain parity **avg 0.007 / worst 0.020**; MVs the only offline input |

`window.__warp` / `__gop` / `__sr` / `__live` expose the measured results on each page.

## What's faithful to the prototype
- **Warp** = `derisk.warp_hd` exactly (nearest LR-flow ×scale → bilinear clamp-to-edge sample = cv2.remap
  + BORDER_REPLICATE).
- **Occlusion** = `derisk.occlusion_mask_lr` *reactive* mode ((a) intra holes ∪ (b) `mean_c|LR_cur −
  warp_lr(LR_prev)| > 16`). The adaptive fwd-bwd splat (mode (c)) is not ported (high-motion-only refinement).
- **SR** = the exact `realesr-general-x4v3` weights + RGB/[0,1] preprocessing; bit-exact.
- **Anchor cadence** = SR the I-frame, propagate the rest (6.25% SR on the 16-frame GOP — the NEMO ratio).

## Three WebGPU gotchas found here (carry forward for any WGSL porting)
1. **`textureSample` only in uniform control flow** — calling it inside a data-dependent `if` is a compile
   error that silently invalidates the pipeline (renders black). Use **`textureSampleLevel(…, 0.0)`** (no
   derivatives; identical when there are no mips).
2. **A *sampled* texture needs `TEXTURE_BINDING`** in its usage (else the bind group is invalid → draw dropped).
3. **A `copyExternalImageToTexture` destination needs `RENDER_ATTACHMENT`** (else the upload silently yields a
   black texture — this one cost a real debug pass: the LR fallback sampled black).
   Diagnosis tooling that worked: `dev.pushErrorScope("validation")` + `shaderModule.getCompilationInfo()`,
   plus splitting parity error by region/condition (hole vs non-hole) to localize.

## Performance characterization
- **Warp / occlusion / playback**: negligible (warp ~0.1 ms; these are texture-sample passes) — real-time with huge headroom.
- **Anchor SR (the cost)**: ~1.3 s for 256×256→1024×1024 with a **naive** conv. It is a one-time **amortized**
  anchor (SR ~1 frame in 12–48; the warp carries the rest), so fine for a **render-then-play** UX.

### Conv-perf optimization — SOLVED: **6.8× in Chrome, bit-exact** (multi-agent + headless Deno harness)
My solo first pass (fp16, naive input-tiling) found no quick win and mis-diagnosed the bottleneck. The real fix
came from a **4-agent fleet** (opus, distinct strategies) iterating against a headless **Deno+WebGPU harness**
(`conv_opt/bench.ts`, GPU-timestamp timing + parity) — agents self-test without a browser; the manager confirms
the winner in Chrome.

**Clean Deno results @256 (contention-free, real parity after fixing a vacuous-parity bug an agent caught):**
| candidate | strategy | speedup (Deno) | parity vs naive |
|---|---|---|---|
| **wtile** | weight+input shared-mem tile, OCB=32, register-blocked vec4 acc, double-buffered | **12.9×** | 3e-7 (≈bit-exact) |
| regblock | input-tile + 2×2-pixel×8-ch register micro-tile | 9.7× | 0 (bit-exact) |
| vec | interleaved layout + vec4 dot + register tile | 8.1× | 5.5e-7 |
| fp16 | shared-tile + f16 storage/MAC, f32 accumulate | 7.7× | 6.6e-4 |

**Authoritative Chrome confirmation (`sr_wtile.html`, the real SR pipeline):** the **wtile** kernel runs the
34-layer conv in **210 ms** (vs naive ~1437 ms = **6.8×**) with the **full SR output bit-exact vs PyTorch**
(mean|Δ| 5e-6). Speedup is lower than Deno's 12.9× because Dawn≠wgpu, but it's a real, verified 6.8×.

**What actually mattered (corrects my first-pass diagnosis):** the win is **caching the weights *as vec4 over 4
oc* + the input halo in shared memory, reused across a 16×16 pixel tile**, with **fully-unrolled vec4 register
accumulators** (dynamic-indexed accumulator arrays spill → 0.6×). My earlier "weight-caching is a dead end"
was wrong — it's the *primary* fix when the accumulators stay register-resident. The kernel is
latency/occupancy-bound; vec4-vectorization and pure-fp16 each give little alone (Metal already vectorizes;
not bandwidth-bound), but **weight+input tiling + register blocking** breaks the ~65 GFLOP/s wall.

**Combination round — wtile + fp16 = BELOW NATIVE (`sr_combo.html`, `combo.wgsl`):** grafting fp16 onto wtile's
register-blocked structure (OCB=64 now fits since 16×vec4&lt;f16&gt; = same 32 regs as 8×vec4&lt;f32&gt; → gz=1,
each input loaded once; f16-accumulate, both mul+add 2× on Apple) gives **121.7 ms in Chrome = 11.8× over naive,
below the native compact-SR (~130 ms)**, full SR output **visually identical** vs PyTorch (mean 0.016 / max 7
codes — f16-level, and the project already validated fp16 SR as LPIPS-identical). A switchable `ACC="f32"` mode
is bit-tighter (parity 4e-4) at 212 ms if ever needed.

**Impact:** the anchor SR goes from ~1437 ms → **210 ms bit-exact** (wtile, integrated into `gop_live`) or **121.7 ms
visually-identical** (wtile+fp16) — **the in-browser instant tier now streams in real-time, matching (and at f16,
beating) the native architecture's amortized anchor**. The live pipeline ships the bit-exact wtile by default
(anchor `runSR` ~1357→516 ms, full-chain parity unchanged); the below-native fp16 combo is proven and available to
swap in. Harness + candidates in `conv_opt/`; Chrome proofs `webgpu_warp/{sr_wtile.html, sr_combo.html}`
(+ `wtile.wgsl`, `combo.wgsl`).

**Results ladder (Chrome, 256→1024, 34-layer conv, timestamp best-of-5):**
| kernel | conv time | vs naive | vs native (~130 ms) | full-SR parity | status |
|---|---|---|---|---|---|
| naive | 1437 ms | 1.0× | 11× slower | bit-exact | (baseline) |
| **wtile** (f32) | **210 ms** | **6.8×** | 1.6× slower | **bit-exact** (max 1) | **integrated in live** |
| **combo** (wtile+f16) | **121.7 ms** | **11.8×** | **below native** | visually identical (max 7) | proven, available |

## Roadmap to a shippable in-browser tier (all engineering, no open feasibility)
1. **WASM MV-binding** (the long pole; needs `emsdk` — parked): expose
   `av_frame_get_side_data(MOTION_VECTORS)` from a WASM libav build so the flow comes live from in-browser
   decode instead of the offline `.bin`. The spike proved the throughput; this is marshalling.
2. **Conv perf**: fp16 / tiled conv (or reuse a tuned WGSL CNN) → real-time anchor.
3. **Adaptive fwd-bwd occlusion** (mode (c)): the softmax-splat consistency check, high-motion only.
4. **Full-frame (uncropped) run** at production resolution; the GOP frame loop driven by `requestVideoFrameCallback`.

> Reproducibility note: each `*_data/` dir + downloaded weights are **git-ignored** (regenerate with the
> matching `extract_*.py` / `export_*.py`; weights download from HF on first use). The Python scripts,
> WGSL/HTML pages, and `node_bench.mjs` are the committed artifacts.
