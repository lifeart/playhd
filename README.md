# playhd

### Browser-native SD→HD video upscaling, paid for by the codec's own motion vectors

[![▶ Live Demo](https://img.shields.io/badge/▶_Live_Demo-lifeart.github.io%2Fplayhd-2ea44f?style=for-the-badge)](https://lifeart.github.io/playhd/)
&nbsp;
[![Code License: MIT](https://img.shields.io/badge/code-MIT-blue?style=for-the-badge)](LICENSE)
[![🤗 Runtime assets](https://img.shields.io/badge/🤗_runtime_assets-playhd--web--assets-FFD21E?style=for-the-badge)](https://huggingface.co/datasets/lifeart/playhd-web-assets)
[![status: research spike](https://img.shields.io/badge/status-research_spike-orange?style=for-the-badge)](web_spike/WRITEUP.md)

> **▶ One-click live demo (runs 100% in your browser): <https://lifeart.github.io/playhd/>**
> &nbsp; *(needs a WebGPU browser — Chrome/Edge with `shader-f16`)*

<!-- TODO: add a demo GIF here -->

---

## TL;DR

Super-resolving every video frame in a browser is too expensive. So don't — super-resolve only **sparse
anchor frames** on the GPU (WebGPU/WGSL), and reconstruct everything in between by **warping the previous
frame with the codec's own H.264 motion vectors** (the encoder already did the motion search — reuse it).
This is the [NEMO](https://dl.acm.org/doi/10.1145/3372224.3419185) idea moved into the browser. The
enabler nobody documents: **getting H.264 motion vectors in the browser at all.** Everything — decode, MV
extraction, warp, SR, composite — runs **client-side**: no server, no upload.

> **Status: research spike, not production.** A de-risking prototype: everything here runs and is
> measured, but it is not a shipped product or a trained model. Full technical write-up:
> [`web_spike/WRITEUP.md`](web_spike/WRITEUP.md).

---

## 1. The load-bearing trick — codec MVs in the browser

`WebCodecs` (`VideoDecoder`) hands you decoded frames and nothing about *how* they were predicted — no
motion vectors. But the H.264 bitstream is full of MVs the encoder spent real effort computing. FFmpeg
exposes them (`flags2=+export_mvs` → `av_frame_get_side_data(AV_FRAME_DATA_MOTION_VECTORS)` — exactly what
PyAV's `export_mvs` does). So the move is a **minimal FFmpeg → WASM** build with that path enabled.

### ⚠️ The build trap (the valuable part)

The obvious minimal build —

```sh
./configure --disable-everything --enable-decoder=h264 ...
```

— **decodes 700 frames perfectly and emits ZERO motion vectors**, with `+export_mvs` set the whole time.
No error. The cause: H.264's MV-export call (`ff_print_debug_info2` in `libavcodec/h264dec.c`) is wrapped
in `#if CONFIG_MPEGVIDEODEC`, and `--disable-everything` sets `CONFIG_MPEGVIDEODEC=0` — so the export is
**compiled out of the H.264 decoder** even though H.264 itself is enabled. The fix is a one-liner that
looks unrelated:

```sh
--enable-decoder=mpeg2video   # _selects mpegvideodec → CONFIG_MPEGVIDEODEC=1 → re-enables H.264's MV export
```

You never use the mpeg2 decoder; you enable it purely to flip the config gate.

### Verified byte-identical to native PyAV

The WASM MVs aren't "close" to a native FFmpeg/PyAV decode — they're **exact**:

| check | result |
|---|---|
| WASM MV CSV vs PyAV reference | **22,429 MVs over 30 frames, exact — 30/30 frames match** |
| full `sd600.mp4` (640×320, 689 frames) | **802,611 MVs, frame-for-frame match** to the PyAV reference |
| `mv_decode` clean API (RGB + MVs) | 30/30 MV counts; **RGB24 bit-exact** (mean\|Δ\|=0.000, max=0) |
| `flow.js` (MVs → dense per-pixel flow) | **bit-exact** (28160/28160 holes, max\|Δ\|=0) |

The WASM module is **~1.9 MB** (libavcodec/format/util only; no asm, no x264/x265, no threads).
Single-thread SD decode ran **~850–1000 fps** in the spike — decode is *not* the bottleneck. Recipe +
verification: [`web_spike/wasm_mv/README.md`](web_spike/wasm_mv/README.md),
[`web_spike/wasm_mv/build_ffmpeg_wasm.sh`](web_spike/wasm_mv/build_ffmpeg_wasm.sh).

---

## 2. Architecture

```
            ┌──────────────── per frame ────────────────┐
  .mp4  →   │  WASM libav decode → RGB + motion vectors  │
            └────────────┬───────────────────┬──────────┘
                         │ anchor?            │ non-anchor
                   ┌─────▼─────┐        ┌─────▼──────────────────────┐
                   │ GPU SR    │        │ MV → dense flow (flow.js)  │
                   │ (compact  │        │ → WebGPU warp prev frame   │
                   │  BSD-3)   │        │ + reactive occlusion fill  │
                   └─────┬─────┘        └─────┬──────────────────────┘
                         └────────┬───────────┘
                                  ▼
                       HD frame → split-screen display (vs bicubic)
```

- **Anchor SR** — every Nth frame (plus I-frames, which carry no MVs) gets a full WebGPU super-resolution pass.
- **MV-warp propagation** — in-between frames are reconstructed by warping the previous reconstruction with
  the codec MVs — a texture fetch per pixel. Validated live: warping frame 0 into frame 1 by codec MVs
  reconstructs it at **mean\|Δ\|=0.49 code values** over covered pixels (visually exact).
- **Occlusion fallback** — a reactive mask falls back to the current LR frame where the warp has holes (disocclusion).

The SR models are **hand-ported to WGSL** — no ONNX runtime, no tfjs. The port is **bit-faithful to
PyTorch** (SPAN graph Conv3XC → 6×SPAB → conv_cat → pixel-shuffle: f32 mean\|Δ\|=1.46e-7,
**f16 mean\|Δ\|=6.17e-4**). The hot 3×3 conv is occupancy-bound (Winograd *lost* 2.9×, layer-fusion *lost*
13.7×); the measured optimum is the tiled register-blocked kernel at **OCB=48**. The **shipped demo runs the
permissive `compact` port** (realesr-general-x4v3, BSD-3); the SPAN port stays in the repo for the eval below
and local use.

---

## 3. Results — we measured the model-card claim, and it was wrong

The SPAN model card claims it beats a generic compact SR on perceptual metrics. We had been repeating
that — then measured it on **real libx264 degradation** (the regime that actually matters here, and the
one most SR models were *not* trained on): GT = a 256-px crop; LR = 2× INTER_AREA down → real libx264
encode (CRF 27 "moderate" / 35 "heavy") → decode; arbiter = `pyiqa` LPIPS(AlexNet) + DISTS + PSNR.
Source: [`experiments/r11_span_eval/REPORT.md`](experiments/r11_span_eval/REPORT.md).

### Overall — the blanket claim is **refuted**

| model | LPIPS ↓ | DISTS ↓ | PSNR ↑ | latency (128→out) |
|---|---:|---:|---:|---:|
| bicubic | 0.1994 | 0.2117 | 24.06 | — |
| **compact** (realesr-general-x4v3) | **0.1160** | **0.1671** | 25.13 | 21 ms |
| x4plus (RRDBNet, ceiling) | 0.1026 | 0.1595 | 25.32 | 360 ms |
| **SPAN** (2xLiveActionV1) | 0.1217 | 0.1876 | 25.02 | 84 ms |

Averaged across content, **compact beats SPAN on both LPIPS and DISTS.** x4plus is the quality ceiling but
at ~17× the compact's latency — not anchor-affordable in a browser.

### …but it's content-dependent, and SPAN wins exactly where it's used

SPAN vs compact, per window (Δ = SPAN − compact; **negative = SPAN better**):

| window / CRF | ΔLPIPS | ΔDISTS | winner |
|---|---:|---:|:--|
| **talking-head / moderate** | **−0.0305** | **−0.0175** | **SPAN** |
| **talking-head / heavy** | **−0.0254** | **−0.0107** | **SPAN** |
| high-motion / moderate | +0.0052 | +0.0547 | compact |
| high-motion / heavy | +0.0132 | +0.0200 | compact |
| texture / moderate | +0.0090 | +0.0252 | compact |
| texture / heavy | +0.0629 | +0.0516 | compact |

**Takeaway:** SPAN is a *content-dependent live-action specialist* — it wins talking-head faces by ~20%
LPIPS at both compression levels, and loses on texture and high-motion. **The shipped demo runs only the
permissive `compact` model (BSD-3).** SPAN (CC-BY-NC-SA-4.0) was dropped from the public player so the demo
is commercially usable — compact wins overall on LPIPS/DISTS anyway (table above); the talking-head edge is
the accepted cost. The SPAN WGSL port remains in git history for local, non-commercial use.

### Performance ladder

This ladder was measured with the SPAN anchor (native ×2, so a ×2 output runs on the full source,
640×320 → 1280×640) — **~165–183 ms/anchor**, i.e. pure per-frame SR ≈ **6 fps**. The shipped `compact`
anchor is a lighter net (see the crop-latency column above: 21 ms vs SPAN's 84 ms), so it's at least as
fast. Real time comes from **propagation** — the "SR every N frames" knob warps more frames between sharp
anchors (640×320→1280×640; live GPU-synced one-off measurement):

| SR every N | ms/frame | fps | |
|---:|---:|---:|---|
| 1 (pure SR) | 213 | 5 | max quality |
| 2 | 81 | 12 | |
| **4** | **43** | **23** | real-time, still clean |
| 8 | 21 | 48 | |

Aggressive propagation doesn't smear here because the anchors are sharp and codec-faithful and the
inter-anchor chains are short.

---

## 4. Gotchas worth knowing

- **`maxComputeWorkgroupsPerDimension` is 65,535.** A 1-D dispatch over `channels·H·W/64` silently exceeds
  it at modest resolution (48·320·640/64 = 153,600 > 65,535) → the pass **no-ops with no error** → the
  graph cascades to black. Fix: dispatch a 3-D grid `(ceil(W/8), ceil(H/8), channels)`.
- **`requestAnimationFrame` pauses in background tabs** — a rAF-driven player looks "frozen" under
  headless/occluded testing and runs fine when foregrounded. Test in a visible tab.
- **Media elements need HTTP range support** — `<video>`/`<audio>` pointed at a dev server without
  `Accept-Ranges` stalls forever in `readyState=0` with no error. Use a `blob:` URL for local files, and a
  CORS- + range-enabled host for remote clips.

More (the `rgba32float`-not-renderable trap, occluded-canvas readback) in
[`web_spike/WRITEUP.md` §5](web_spike/WRITEUP.md).

---

## 5. Run it

**Easiest — the live demo (one click, nothing to install):** **<https://lifeart.github.io/playhd/>**
&nbsp;(WebGPU browser; runs entirely client-side).

**Locally** — needs a **WebGPU browser** (Chrome/Edge with `shader-f16`) and an **H.264 SD `.mp4`** clip:

```sh
# from the repo root, serve web_spike/ over http (the player fetches sibling files)
cd web_spike && python3 -m http.server 8767
# open http://localhost:8767/wasm_mv/player.html  and DRAG IN an H.264 SD .mp4
```

Drag the divider to compare playhd vs bicubic; the **"SR every N"** slider trades quality for speed
(propagation). The anchor SR is the permissive **`compact`** model (BSD-3). You can also **open your own
clip** via a CORS-enabled URL: `?clip=<https-url>`.

### Get an openly-licensed test clip

No clip is bundled (the dev clips are copyrighted). Any short H.264 SD `.mp4` works (live-action / faces
upscale best). Open options:

- **Big Buck Bunny** — CC-BY 3.0 (Blender Foundation). Small SD/H.264 clips: <https://test-videos.co.uk/bigbuckbunny/mp4-h264>
- **Blender open movies** (Sintel, Tears of Steel, …) — CC-BY: <https://studio.blender.org/films/>

---

## 6. What's NOT included, and licensing

This **git repo** holds only the **MIT application code** — no weights, no clips, no compiled WASM. The
**live demo's runtime assets** are hosted separately on a HuggingFace dataset and fetched at runtime
(CORS-enabled): **🤗 [lifeart/playhd-web-assets](https://huggingface.co/datasets/lifeart/playhd-web-assets)**.
Every asset the shipped demo fetches is **permissively licensed** (BSD-3 / LGPL / CC-BY) — each keeps its
own license:

| asset | license | source / credit |
|---|---|---|
| `realesr-general-x4v3` ("compact") weights — the anchor SR | **BSD-3-Clause** | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (Xintao Wang et al.) |
| `mv_decode.wasm` (FFmpeg→WASM decoder) | **LGPL-2.1+** | [FFmpeg](https://ffmpeg.org); build recipe: `web_spike/wasm_mv/build_ffmpeg_wasm.sh` |
| demo clip | **CC-BY 3.0** | Big Buck Bunny © [Blender Foundation](https://peach.blender.org) |

> **Dropped from the shipped demo (commercial-licensing decision):** the SPAN `2xLiveActionV1` weights
> (**CC-BY-NC-SA-4.0**, non-commercial — jcj83429 · [OpenModelDB](https://openmodeldb.info/models/2x-LiveActionV1-SPAN)).
> SPAN wins talking-head faces (§3), but its non-commercial license would taint the whole demo, so the
> public player now runs the permissive `compact` model only. The SPAN WGSL port is still in git history and
> can be re-enabled locally for non-commercial use.

Because every fetched asset is BSD-3 / LGPL / CC-BY, **the demo as a whole is commercially usable**. To run
locally instead, obtain the compact model weights from their source and regenerate the runtime data with the
`export_*` scripts in `web_spike/`, then serve `web_spike/` (the player falls back to sibling files when not
on `*.github.io`).

**License.** Original code in this repo: **MIT** (see [`LICENSE`](LICENSE)). The third-party runtime assets
(model weights, WASM decoder, demo clip) are **not** covered by it — each keeps its own license (table
above); all of them are permissive (BSD-3 / LGPL / CC-BY), so **the hosted demo is commercially usable**.

---

## 7. Repo map

| path | what |
|---|---|
| [`web_spike/WRITEUP.md`](web_spike/WRITEUP.md) | **the full technical write-up** — architecture, the MV build trap, the honest model eval, perf, gotchas |
| `web_spike/wasm_mv/` | the browser player + WASM-libav motion-vector decode API + [build recipe](web_spike/wasm_mv/README.md) + the MV→flow port |
| `web_spike/webgpu_warp/`, `web_spike/conv_opt/` | the WebGPU/WGSL SR drivers (SPAN + compact) + parity/timing harnesses |
| [`experiments/`](experiments/) | the R1–R11 research rounds (scripts + REPORTs), incl. [`r11_span_eval/`](experiments/r11_span_eval/REPORT.md) — the real-H.264 LPIPS/DISTS model eval |
| `prototype/` | the Python research core (decode/MV/warp/occlusion/SR — the offline pipeline the browser port is based on) |
| `server/` | an alternate server-rendered player |

---

*Prior art: [NEMO (MobiCom '20)](https://dl.acm.org/doi/10.1145/3372224.3419185) — neural-enhanced video
delivery by super-resolving anchors and propagating with cached info. playhd is a browser-native take, with
the codec's own motion vectors as the propagation signal.*
