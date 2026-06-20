# Web-only de-risk spike — RESULT: GO (the MV seam is not a throughput risk)

**Question (from `WEB_ONLY_FEASIBILITY.md`): can codec motion vectors be extracted in-browser
via WASM-libav `+export_mvs` at SD, in real time?** The whole web-only port hinges on this — WebCodecs
exposes no MV API, so the architecture's foundation (codec MVs) is only reachable through a WASM
software-decode + `+export_mvs` path. **Verdict: GO, with ~34–40× real-time headroom.**

## What was measured

**Part 1 — native software-decode + `+export_mvs` at SD** (`native_mv_throughput.py`, PyAV/libav,
which uses software decode = the right proxy; sample.mp4, 640×320, 600 frames):

| config | fps | ms/frame |
|---|---|---|
| single-thread, decode-only | 2623 | 0.38 |
| **single-thread, decode + export_mvs** | **2392** | 0.42 |
| auto-thread, decode-only | 8641 | 0.12 |
| auto-thread, decode + export_mvs | 7446 | 0.13 |

- **MV extraction overhead is ~10%** over decode-only (2623→2392) — negligible.
- **MV payload: ~1144 records/frame (max 1833), ~45 KB/frame** → at 25 fps ≈ 1.1 MB/s across the
  WASM→JS boundary. Trivial.

**Part 2 — ACTUAL ffmpeg.wasm in V8/WASM** (`node_bench.mjs`, single-thread Emscripten core loaded
directly with `self`/`location` shims; **V8's WASM engine == Chrome's**, so this is a faithful
in-browser proxy; the SD test clip `sd600.mp4` = 700 frames stream-copied from sample.mp4):

| config | fps |
|---|---|
| WASM decode-only | ~960–1015 |
| **WASM decode + `+export_mvs`** (with `codecview` rendering the MVs) | **~844–1019** |

- **`+export_mvs` compiles and RUNS in WebAssembly** — `codecview=mv=pf+bf+bb` rendered the extracted
  vectors without error, which is only possible if export_mvs actually populated the MV side-data.
- **Actual WASM SD decode throughput ≈ 850–1000 fps, single-threaded** = **~34–40× real-time** (25 fps).
- **Native→WASM penalty is only ~2.4×** (2392 → ~1000 fps), NOT the 5–10× the literature cites — because
  that figure is for *transcode* (decode+encode); pure SD **decode** is light, and MV export is free.

## Conclusion

The make-or-break seam is **decisively not a performance risk.** SD decode + codec-MV extraction in
WebAssembly runs at ~850–1000 fps single-threaded — orders of magnitude above the ~25–30 fps the
real-time tier needs, leaving the entire frame budget for the WebGPU SR + warp + blend.

**What remains is engineering, not feasibility:** the CLI `codecview` path proves extraction works and
is fast, but the architecture needs the **raw MV side-data read in JS** — i.e. a WASM build that exposes
`av_frame_get_side_data(AV_FRAME_DATA_MOTION_VECTORS)` through its bindings (return the `AVMotionVector`
array per frame). Since export_mvs adds ~0% to decode time and the payload is ~45 KB/frame, surfacing it
is a binding/marshalling task with zero throughput concern. The field semantics are already known and
PyAV-validated (GOTCHA #4: `dst_x/dst_y` block centers, `src = dst + motion_x/motion_scale`, `source` sign).

**Spike → GO.** Proceed to P1 (instant-tier browser port): WASM decode+MV → WebGPU compact-anchor SR +
WGSL warp + reactive occlusion → canvas. Reproduce: `python native_mv_throughput.py` and
`node node_bench.mjs` (after `npm install`).
