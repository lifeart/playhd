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

### Conv-perf investigation (measured; `sr.html` naive, `sr_f16.html` fp16, `sr_tiled.html` shared-mem)
Profiled empirically (no GPU profiler available) by trying the obvious levers and measuring:
| variant | time | parity vs PyTorch fp32 | takeaway |
|---|---|---|---|
| naive fp32 | ~1330 ms | **0.0000035** (bit-exact) | baseline |
| batch 34 submits → 1 | ~1330 ms | bit-exact | **not** submit/overhead-bound |
| **fp16** (f16 storage, f32 accum) | ~1270 ms | 0.0015 (visually identical) | **not** bandwidth-bound (halving load traffic ≈ no change) |
| shared-mem **input tile** | ~1586 ms (slower) | bit-exact | **not** input-latency-bound; the tiled structure made each thread do all 64 oc → far fewer threads → **occupancy regression** |

**Diagnosis:** the kernel runs at ~65 GFLOP/s — far below the GPU's TFLOP peak — so it's **latency/occupancy-bound**,
not bandwidth- or compute-peak-bound. The naive structure's high thread count (W×H×OC, one output channel each)
is actually reasonable; input-tiling *reduced* parallelism. fp16 helps only memory (which isn't the bottleneck).
**A real speedup needs the right thing cached (the weights, reused across all pixels — but 144 KB/layer >
32 KB shared, so it needs oc-chunked weight tiling) or an interleaved-channel layout enabling vec4 MACs — i.e.
a profiler-guided rewrite, or (the pragmatic path) reuse websr / Anime4K-WebGPU's hand-tuned WGSL conv kernels
rather than hand-rolling.** Conclusion: the naive conv is the correctness reference (bit-exact); the
amortized 1.3 s anchor is fine for render-then-play; a real-time anchor is a tuned-kernel integration, not a
one-line change. fp16 is validated (0.0015) and ready if a 2× memory win ever matters.

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
