# playhd — web-only feasibility (browser/WebGPU port)

> **Exploration, 2026-06-20.** Can the whole pipeline (codec-MV propagation + sparse anchor SR) run
> **entirely in a browser**, no server? Verdict below is from current-web-platform research (June 2026)
> + this repo's architecture. **TL;DR: FEASIBLE — and the architecture is a *better* fit for the browser
> than for native**, because it amortizes away exactly the cost the browser is worst at (per-frame SR).
> One make-or-break seam needed a de-risk spike: getting codec motion vectors in-browser.
>
> **✅ SPIKE DONE — GO (`web_spike/SPIKE_RESULT.md`).** Measured the seam: **`+export_mvs` runs in real
> WebAssembly** (ffmpeg.wasm Emscripten core, V8/WASM == Chrome's engine; codecview rendered the extracted
> MVs), and **SD decode + MV extraction hits ~850–1000 fps single-threaded = ~34–40× real-time** (native
> 2392 fps; WASM penalty only ~2.4× because the heavy ffmpeg.wasm cost is *encode*, not decode; MV export
> adds ~0%; payload ~45 KB/frame ≈ 1.1 MB/s). **The MV seam is not a throughput risk** — what remains is a
> binding task (expose the raw MV side-data to JS), not a feasibility question. Proceed to P1.

## Why this architecture *wants* to be in a browser

The browser's single biggest SR weakness is that **per-frame heavy SR is far too slow in WebGPU** —
Real-ESRGAN ×4 measures **~1–2 s per 1080p frame** in-browser (a 15 s/30 fps clip ≈ 10 min). That is the
wall every naive "AI upscale in the browser" tool hits. **This project's entire thesis is to *not* run
per-frame heavy SR** — it SRs only the sparse anchors (~2–12 % of frames; NEMO: 1.79–9.74 %) and propagates
the rest with the codec's own motion vectors (a cheap GPU warp). So the architecture **amortizes the browser's
worst cost by 10–50×** — the slow WebGPU Real-ESRGAN becomes affordable when it runs on 1 frame in ~10–48, and
the per-frame work collapses to a warp+blend that GPUs do trivially. The thing that makes this repo novel on
native (codec-MV propagation vs. per-frame baselines) is *even more* valuable on the web.

WebGPU shipped across **Chrome, Firefox, Safari, Edge as of Nov 2025**, so the GPU substrate is finally
universal.

## The component map: native → browser

| pipeline stage (native) | browser path | maturity |
|---|---|---|
| H.264 decode → frames | **WebCodecs `VideoDecoder`** (hardware) OR WASM libav (software) | shipping |
| **codec MV extraction** (PyAV `+export_mvs`) | **WASM libav `+export_mvs`** (only viable path — see below) | **the de-risk seam** |
| heavy anchor SR (x4plus / RRDBNet) | **WebGPU compute shaders** (web-realesrgan proves RRDBNet runs in WebGPU; amortized over anchors) | proven, slow-but-amortized |
| compact tier (realesr-general / SRVGGNet) | **Anime4K-WebGPU / websr** (hand-written WebGPU CNN shaders, real-time) | proven, real-time |
| MV warp / propagation (`cv2.remap`, torch) | **WebGPU** gather/sample shader (textbook GPU work; already Metal in `gpu_ops.py`) | straightforward |
| occlusion mask (intra/reactive/fwd-bwd) | **WebGPU** compute (it's per-pixel LR math) | straightforward |
| region-aware blend, grain, β-blend | **WebGPU** (all per-pixel) | straightforward |
| encode/mux output | not needed (render straight to `<canvas>`) OR `VideoEncoder` to save | shipping |

The bulk of the per-frame pipeline (warp + occlusion + blend) is exactly the gather/remap/blend work GPUs
excel at and is **already GPU-resident** in the prototype (`reconstruct_torch`, `gpu_ops.py`) — porting the
Metal/torch ops to WGSL is mechanical, not research.

## The make-or-break seam: motion vectors in the browser

**WebCodecs does NOT expose motion vectors.** `VideoDecoder` hands back decoded pixels (`VideoFrame`) only;
there is no MV side-data API. This is the one hard constraint, and the whole architecture rests on MVs.

**The only viable path is a WASM libav build with `+export_mvs`** — the *exact* mechanism the Python prototype
uses (`flags2=+export_mvs` → `av_frame_get_side_data(MOTION_VECTORS)`; GOTCHA #2). `ffmpeg.wasm` / `libav.wasm`
exist and run libav in WebAssembly; the canonical `doc/examples/extract_mvs.c` is the reference. Implications:
- It must be **software decode** (WASM can't reach the hardware decoder to export MVs). ffmpeg.wasm software
  decode benchmarks **~40 fps at 720p**. **At SD (640×320 ≈ ¼ the pixels of 720p) software decode is plausibly
  ~100–180 fps → real-time-capable** — and SD is exactly this product's input. The penalty native pays for HW
  decode is largely recovered by the small frame size.
- A stock ffmpeg.wasm build does **not** expose MV side-data through its JS bindings by default — you build
  libav with the example wired up (or patch the bindings to surface `av_frame_get_side_data`). Precedent
  exists natively (LukasBommes `mv-extractor`, MV-Tractus) but those "patch FFmpeg internals"; doing it in the
  WASM build is the **single highest-risk engineering task** of the port.
- Single decode yields **both frames and MVs** (like PyAV does), so the WASM-software-decode path replaces
  WebCodecs entirely rather than running alongside it — simpler, and avoids decoding twice.

## Recommended de-risk spike (do this first, ~days not weeks)

Before any port work, prove the one unproven thing:
1. Build/obtain a WASM libav with `+export_mvs` exposed; in a worker, software-decode a **SD H.264** clip and
   read `MOTION_VECTORS` side-data per frame. **Confirm (a) the MV field semantics match PyAV** (GOTCHA #4:
   `dst_x/dst_y` block centers, `src = dst + motion_x/motion_scale`, `source` sign), and **(b) decode+MV
   throughput ≥ ~30 fps at SD** on a mid laptop. If both hold, the architecture is web-portable; if MV
   throughput is the bottleneck, fall back to a WebGPU optical-flow estimator (loses the "free codec MV"
   thesis and adds cost — a real downgrade, so prove the WASM path first).
2. Port one warp+occlusion+anchor-SR GOP to WebGPU and A/B the *output* against the Python prototype on the
   same clip (seam-verify the WGSL warp == `cv2.remap`/`reconstruct_torch` to within rounding) — reuses the
   repo's existing regression discipline.

## Phased plan (after the spike passes)

- **P0 — spike** (above): WASM-libav `+export_mvs` at SD + a single-GOP WebGPU warp parity check.
- **P1 — instant tier, real-time**: WASM decode+MV → WebGPU compact-anchor SR (Anime4K/websr-class shader) +
  WGSL warp + reactive occlusion + render to canvas. Target the same ~24 fps the native instant tier hits.
  **✅ CORE BROWSER-VERIFIED (`web_spike/webgpu_warp/P1_RESULT.md`):** the WGSL MV-backward-warp matches the
  prototype `warp_hd` to **0.002 codes mean / 1 max** in real Chrome WebGPU, at **0.096 ms/frame ≈ ~430×
  real-time** (1280×640). Both halves of the pipeline are now measured in-browser (WASM decode+MV ~850–1000 fps;
  WebGPU warp ~10,400 fps). Remaining: on-GPU compact SR (websr/Anime4K precedent), reactive occlusion (per-pixel),
  the WASM MV-binding, and the GOP frame loop.
- **P2 — quality tier**: add the WebGPU RRDBNet/x4plus anchor (web-realesrgan proves it runs), amortized over
  anchors; region-aware blend + β=0.85 + (optional) deblock as WGSL passes. Not real-time (like native), used
  for render-and-watch with progressive output.
- **P3 — parity/polish**: scene-cut detection (luma-diff, trivial in JS), Auto mode (the `recommend_mode`
  probe is cheap JS), progressive playback is *native* to the canvas (no fMP4/MSE plumbing needed — just keep
  drawing frames).

## Risks / honest unknowns

1. **WASM-libav `+export_mvs` engineering** — the one genuinely novel build task; everything else has precedent.
   If it can't hit SD real-time, the whole "free codec MV" advantage degrades to in-browser optical flow.
2. **HEVC in WASM** — H.264 is clean; H.265 adds patent/build complexity. Target H.264 first (it's the repo's
   validated path anyway).
3. **WebGPU SR throughput at SD anchor size** — un-benchmarked here; amortization makes it tractable but P2's
   heavy anchor may cap the quality-tier frame budget. Measure in the spike.
4. **Cross-origin isolation** — WASM threads need `SharedArrayBuffer` → COOP/COEP headers (a deploy detail, not
   a blocker).
5. **No server means no offload** — the heaviest device (a phone, 3–5× slower per the WebGPU SR benchmarks)
   sets the floor; the instant/compact tier is the realistic mobile target, quality tier is desktop.

## Bottom line

Web-only is **feasible and architecturally favorable**: the codec-MV-propagation design specifically dodges the
browser's central SR bottleneck (per-frame heavy SR), the propagation/occlusion/blend math is standard
WebGPU work already GPU-resident in the prototype, and the SR tiers have proven WebGPU precedents
(web-realesrgan for the heavy anchor, Anime4K-WebGPU/websr for the compact tier). **The entire bet reduces to
one spike: codec motion vectors via WASM-libav `+export_mvs` at SD, real-time.** If that holds, this is a
better browser product than a native one — fully local, zero-install, privacy-preserving SD→FullHD upscaling
that beats the per-frame baselines every other in-browser upscaler uses.

## Sources
- WebCodecs API + browser video codec support — [MDN WebCodecs](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API), [Chrome WebCodecs](https://developer.chrome.com/docs/web-platform/best-practices/webcodecs)
- ffmpeg.wasm + software-decode perf — [ffmpeg.wasm](https://github.com/ffmpegwasm/ffmpeg.wasm), [WebCodecs vs ffmpeg.wasm](https://burnsub.com/blog/webcodecs-vs-ffmpeg-wasm/), [libav.wasm](https://github.com/qiweicao/libav.wasm)
- MV extraction via `+export_mvs` — [FFmpeg extract_mvs.c example](https://ffmpeg.org/doxygen/8.0/extract_mvs_8c-example.html), [LukasBommes/mv-extractor](https://github.com/LukasBommes/mv-extractor)
- WebGPU SR perf + repos — [xororz/web-realesrgan](https://github.com/xororz/web-realesrgan), [Anime4K-WebGPU](https://github.com/Anime4KWebBoost/Anime4K-WebGPU), [sb2702/websr](https://github.com/sb2702/websr), [WebGPU benchmarks 2025](https://www.mayhemcode.com/2025/12/gpu-acceleration-in-browsers-webgpu.html)
