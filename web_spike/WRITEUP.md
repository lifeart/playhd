# Real-time-ish SD→HD upscaling in the browser, paid for by the codec's own motion vectors

> **Status: research spike, not production.** Everything below runs and is measured, but it's a
> de-risking prototype — not a shipped product, not a trained model, not tuned for every clip.
> Numbers are from this repo's harnesses and reports; where a figure is a one-off live measurement
> it's marked as such.

## TL;DR

Super-resolving every frame of video in a browser is too expensive. So don't. Super-resolve only
**sparse anchor frames** on the GPU, and reconstruct everything in between by **warping the previous
frame with the codec's own motion vectors** — the H.264 encoder already did the motion search; we
reuse its output for free. This is the [NEMO](https://dl.acm.org/doi/10.1145/3372224.3419185) idea,
moved into the browser.

The one thing that makes it possible — and the part nobody documents — is **getting H.264 motion
vectors in the browser at all.** `WebCodecs` exposes none. The only path is a custom WASM `libav`
build, and that build has a silent trap that makes it emit *zero* motion vectors while looking like
it works. The recipe is below.

End to end the pipeline decodes, extracts MVs, warps, super-resolves anchors, and composites entirely
client-side — no server, no upload. Honest performance: pure per-frame SR is ~6 fps at 640×320→1280×640;
real-time comes from leaning on propagation (a "SR every N frames" knob, ~23 fps at N=4).

---

## 1. Codec motion vectors in the browser (the load-bearing trick)

`WebCodecs` (`VideoDecoder`) gives you decoded frames and nothing about *how* they were predicted —
no motion vectors, no macroblock info. But the H.264 bitstream is full of motion vectors the encoder
spent real effort computing. FFmpeg exposes them: set `flags2=+export_mvs` on the decoder and read
`av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS)` per frame (this is exactly what PyAV's
`export_mvs` does). So the move is: compile a **minimal FFmpeg to WASM** with that path enabled.

```c
// mv_decode.c — the essential bit
AVDictionary *opts = NULL;
av_dict_set(&opts, "flags2", "+export_mvs", 0);   // THE switch
avcodec_open2(dec, codec, &opts);
...
AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
if (sd) {
    const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
    int n = sd->size / sizeof(AVMotionVector);   // per-MB src/dst + motion_x/y/scale
}
```

### The build trap (this is the valuable part)

The obvious minimal build —

```sh
./configure --disable-everything --enable-decoder=h264 --enable-demuxer=mov,h264 ...
```

— **decodes 700 frames perfectly and emits ZERO motion vectors**, with `+export_mvs` set the whole time.
No error. You burn an afternoon assuming your side-data read is wrong.

The cause: H.264's MV-export call (`ff_print_debug_info2` in `libavcodec/h264dec.c`) is wrapped in
`#if CONFIG_MPEGVIDEODEC`. `--disable-everything` sets `CONFIG_MPEGVIDEODEC=0`, so the export is
**compiled out of the H.264 decoder** even though H.264 itself is enabled. The fix is a one-liner that
looks unrelated:

```sh
--enable-decoder=mpeg2video    # _selects mpegvideodec -> CONFIG_MPEGVIDEODEC=1 -> re-enables H.264's MV export
```

You don't use the mpeg2 decoder at all; you enable it purely so the config gate flips and H.264's
existing export path is compiled in. Diagnosed by instrumenting the wrapper to print `frames_decoded`
plus the flag state, then tracing the export call to its `#if` gate.

### Verified byte-identical to native

The WASM MVs are not "close" to a native FFmpeg/PyAV decode — they're exact:

| check | result |
|---|---|
| `mv_wasm` CSV vs PyAV reference | **22,429 MVs over 30 frames, exact; 30/30 frames match** |
| full `sd600.mp4` (640×320, 689 frames) | **802,611 MVs, frame-for-frame match** to `mv_reference.json` |
| `mv_decode` clean API (RGB + MVs) | 30/30 MV counts; **RGB24 bit-exact vs PyAV** (mean\|Δ\|=0.000, max=0) |
| `flow.js` (MVs → dense per-pixel flow) | **bit-exact** to the Python reference (28160/28160 holes, max\|Δ\|=0) |

The WASM module is ~1.9 MB (libavcodec/format/util only; H.264 decode + MOV/h264/mkv demux + `export_mvs`;
no asm, no x264/x265, no threads). Single-thread SD decode ran ~850–1000 fps in the spike — decode is
not the bottleneck. See `wasm_mv/README.md`, `wasm_mv/build_ffmpeg_wasm.sh`, `wasm_mv/mv_decode.c`.

---

## 2. Architecture

```
            ┌──────────────── per frame ────────────────┐
  .mp4  →   │  WASM libav decode → RGB + motion vectors  │
            └────────────┬───────────────────┬──────────┘
                         │ anchor?            │ non-anchor
                   ┌─────▼─────┐        ┌─────▼──────────────────────┐
                   │ GPU SR    │        │ MV → dense flow (flow.js)  │
                   │ (SPAN /   │        │ → WebGPU warp prev frame   │
                   │  compact) │        │ + reactive occlusion fill  │
                   └─────┬─────┘        └─────┬──────────────────────┘
                         └────────┬───────────┘
                                  ▼
                       HD frame → split-screen display (vs bicubic)
```

- **Anchors** (every Nth frame, plus I-frames which carry no MVs) get a full WebGPU super-resolution pass.
- **In-between frames** are reconstructed by warping the previous reconstruction with the codec's
  motion vectors — cheap (a texture fetch per pixel) — with a reactive occlusion mask falling back to
  the current LR frame where the warp has holes (disocclusion).

**Why this is *favorable* in a browser specifically.** The browser's worst cost is per-frame heavy SR.
Propagation amortizes that ~10–50× by only paying it on sparse anchors; warping is bandwidth-bound and
trivial on a GPU. The expensive thing the encoder already did (motion estimation) is reused instead of
recomputed. The chain decode→MV→flow→warp was validated live: warping frame 0 into frame 1 by codec MVs
reconstructs it at **mean\|Δ\|=0.49 code values** over covered pixels (visually exact).

**The SR models are hand-ported to WebGPU/WGSL** — no ONNX runtime, no tfjs:

- **SPAN** (`2xLiveActionV1`, native ×2) and a **compact** SRVGG (`realesr-general-x4v3`, ×4) both run as
  WGSL compute. The SPAN graph (Conv3XC → 6×SPAB → conv_cat → pixel-shuffle) is **bit-faithful to PyTorch**:
  f32 mean\|Δ\|=1.46e-7, **f16 mean\|Δ\|=6.17e-4** vs the reference output.
- The 3×3 conv is the hot kernel. It is **occupancy-bound, not ALU- or bandwidth-bound** — established by
  measurement: Winograd *lost* 2.9× and layer-fusion *lost* 13.7× (both collapse occupancy). The tiled
  register-blocked kernel with **OCB=48** (all 48 channels in one workgroup-z) is the measured optimum;
  a sweep at the deployment resolution 640×320 confirms it (OCB 16/24/48/64 = 254.5 / 210.9 / **185.6** /
  215.1 ms — 48 fastest).

See `wasm_mv/player.html` (orchestration), `conv_opt/span_driver_fast.ts` (SPAN WGSL + the persistent
per-frame runner), `webgpu_warp/kernel_gen.js` (the conv generator).

---

## 3. Honest evaluation — does the fancy model actually help?

The SPAN model card claims it beats a generic compact SR on perceptual metrics. We had been repeating
that. So we measured it on **real H.264 degradation**, which is the regime that actually matters here and
the one most SR models were *not* trained on.

**Harness** (`experiments/r11_span_eval/`, reusing a validated earlier rig): GT = a 256-px crop;
LR = 2× INTER_AREA down → **real libx264 encode** (PyAV, CRF 27 "moderate" and 35 "heavy") → decode;
restore to net ×2 (SPAN does 128→256 natively; ×4 models do 128→512 then INTER_AREA→256); arbiter =
`pyiqa` LPIPS(AlexNet) + DISTS + PSNR against GT. Three content windows × 3 frames × 2 CRF.

### Overall — the blanket claim is **refuted**

| model | LPIPS ↓ | DISTS ↓ | PSNR ↑ | latency (128→out) |
|---|---:|---:|---:|---:|
| bicubic | 0.1994 | 0.2117 | 24.06 | — |
| **compact** (realesr-general-x4v3) | **0.1160** | **0.1671** | 25.13 | 21 ms |
| x4plus (RRDBNet, ceiling) | 0.1026 | 0.1595 | 25.32 | 360 ms |
| **SPAN** (2xLiveActionV1) | 0.1217 | 0.1876 | 25.02 | 84 ms |

Averaged across content, **compact beats SPAN on both LPIPS and DISTS.** x4plus is the quality ceiling
but at ~17× the compact's latency — not anchor-affordable in a browser.

### But it's content-dependent, and SPAN wins exactly where it's used

SPAN vs compact, per window (Δ = SPAN − compact; negative = SPAN better):

| window / CRF | ΔLPIPS | ΔDISTS | winner |
|---|---:|---:|:--|
| **talking-head / moderate** | **−0.0305** | **−0.0175** | **SPAN** |
| **talking-head / heavy** | **−0.0254** | **−0.0107** | **SPAN** |
| high-motion / moderate | +0.0052 | +0.0547 | compact |
| high-motion / heavy | +0.0132 | +0.0200 | compact |
| texture / moderate | +0.0090 | +0.0252 | compact |
| texture / heavy | +0.0629 | +0.0516 | compact |

**SPAN wins talking-head faces by ~20% LPIPS at both compression levels, and loses on texture and
high-motion.** It's a *live-action specialist* (trained on film/people), not a universal upgrade.

**Mechanism — and a metric caveat.** A `var-Lap` (variance-of-Laplacian) sharpness proxy *inverts*
between clean and codec-degraded input: on clean LR, SPAN preserves the most detail without exceeding
GT (not fabricating); on codec-degraded LR, SPAN's var-Lap (1136) is *below* compact's (1953) because
SPAN **suppresses** blocking/ringing while compact **amplifies** it into fake high-frequency. That fake
grit happens to score closer to a textured GT — which is exactly why **`var-Lap` is a fake-detail flag,
never the verdict.** LPIPS/DISTS are the arbiters. Full numbers: `experiments/r11_span_eval/REPORT.md`.

**Takeaway:** the player ships SPAN as default *because the target content is talking-head video*, and
exposes a runtime model selector so you can switch to the cheaper compact for texture/general content.
The genuinely correct design is content-adaptive model choice — left as future work.

---

## 4. Performance, honestly

SPAN is native ×2, so for a ×2 output it must run on the **full source** (640×320 → 1280×640) — about
**165–183 ms per anchor** in the browser. That's the floor: the conv is at its occupancy-bound optimum
(§2), and the per-frame CPU input-pack is only ~4 ms. So **pure per-frame SR ≈ 6 fps.** Real time is not
free here, and we don't pretend otherwise.

Real-time playback comes from **propagation**: a "SR every N frames" slider trades quality for speed by
warping more frames between anchors. Measured ladder on the talking-head demo (640×320→1280×640;
live GPU-synced one-off measurement):

| SR every N | ms/frame | fps | |
|---:|---:|---:|---|
| 1 (pure SR) | 213 | 5 | max quality |
| 2 | 81 | 12 | |
| **4** | **43** | **23** | real-time, still clean (sharp SPAN anchors → short warp chains stay clean) |
| 8 | 21 | 48 | |

The reason aggressive propagation doesn't smear here is that the anchors are *sharp and codec-faithful*
(SPAN) and the inter-anchor chains are short — the failure mode of earlier attempts (weak anchors + long
chains) is avoided.

---

## 5. Gotchas worth knowing (WebGPU + browser media)

- **`maxComputeWorkgroupsPerDimension` is 65,535.** A 1-D elementwise dispatch over `channels·H·W/64`
  silently exceeds it at modest resolution (48·320·640/64 = 153,600 > 65,535) → the dispatch is invalid →
  the pass **no-ops with no error** → the whole graph cascades to black. This passed at 160×320 (38,400)
  and broke at 320×640. **Fix: dispatch a 3-D grid** `(ceil(W/8), ceil(H/8), channels)` and index
  `c*H*W + y*W + x` in the shader. Don't put large flat extents in a single dimension.
- **`requestAnimationFrame` pauses in background tabs** — a player driven by rAF looks "frozen" under
  headless/occluded testing and runs fine when foregrounded. Test in a visible tab.
- **Reading a WebGPU canvas while it's occluded** (e.g., `drawImage` from a background tab) can return
  empty. Read back via `copyTextureToBuffer`, and build any `ImageData` *before* `unmap()` (the mapped
  range detaches on unmap).
- **`rgba32float` is not renderable** — giving a float flow texture `RENDER_ATTACHMENT` usage makes it
  invalid → invalid bind group → the render is silently dropped (black). Sampled-only usage for it.
- **Media elements need HTTP range support** — an `<audio>`/`<video>` pointed at a dev server without
  `Accept-Ranges` stalls forever in `readyState=0` with no error. Use a `blob:` URL (the browser
  range-serves it internally) for local files.

---

## 6. Limitations & licensing

- **Research spike.** De-risking prototype. Not hardened, not cross-browser-matrixed, no audio/video
  A/V-sync guarantees (audio tracks the displayed frame, so it's choppy at slow pure-SR settings).
- **No model was trained here.** The SR nets (SPAN `2xLiveActionV1`, compact `realesr-general-x4v3`,
  `RealESRGAN_x4plus`) are **third-party weights** — check each model's license before redistributing
  the weights (e.g. bundling them into a hosted demo). This repo ports them to WGSL; it doesn't relicense
  them.
- **The bundled sample clip is likely copyrighted.** Ship a clip you have rights to, or a synthetic one.
- **WebGPU + `shader-f16` required.** Falls back to f32 (slower, larger buffers) where f16 is absent.

---

## 7. Reproduce / file map

**Run the demo** (WebGPU browser): serve `web_spike/` and open
`wasm_mv/player.html?demo` (bundled clip) or `?clip=<name.mp4>`. Drag the divider to compare playhd vs
bicubic; the "SR every N" slider trades speed for quality; the model dropdown switches SPAN ↔ compact.

**Key files**

| file | role |
|---|---|
| `wasm_mv/player.html` | the player: decode → SR anchor / MV-warp → split display + audio + selector |
| `wasm_mv/mv_decode.c`, `mv_decode.mjs` | WASM libav decode API (RGB + packed MVs per frame) |
| `wasm_mv/build_ffmpeg_wasm.sh` | the minimal FFmpeg→WASM build recipe (incl. the `mpeg2video` fix) |
| `wasm_mv/flow.js` | codec MVs → dense per-pixel warp flow (bit-exact to the Python reference) |
| `conv_opt/span_driver_fast.ts` | SPAN graph in WGSL + the persistent per-frame runner |
| `webgpu_warp/kernel_gen.js` | the tiled/register-blocked 3×3 conv generator (OCB-parametric) |
| `experiments/r11_span_eval/` | the real-H.264 LPIPS/DISTS/PSNR model evaluation (+ REPORT.md) |

**Verification harnesses** (`conv_opt/`, run with an f16-capable Deno):

```sh
deno run --allow-read test_runner.ts    # SPAN WGSL parity vs PyTorch: f16 mean|Δ| ≈ 6.17e-4
deno run --allow-read test_runner2.ts   # 160×320 and 320×640 both non-zero (the 3-D dispatch canary)
deno run --allow-read span_sweep.ts     # conv timing across OCB @640×320 (OCB=48 fastest)
node ../wasm_mv/test_mv_decode.mjs       # WASM MVs + RGB vs native PyAV (30/30, bit-exact)
```

---

*Prior art: [NEMO (MobiCom '20)](https://dl.acm.org/doi/10.1145/3372224.3419185) — neural-enhanced video
delivery by super-resolving anchors and propagating with cached info. This is a browser-native take on
that idea, with the codec's own motion vectors as the propagation signal.*
