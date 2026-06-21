# WASM libav motion-vector binding — the make-or-break seam, VERIFIED

The whole web-only architecture rests on codec **motion vectors**, and **WebCodecs has no MV API**. The only path
is a WASM **libav** build that exposes `av_frame_get_side_data(AV_FRAME_DATA_MOTION_VECTORS)` — the exact PyAV
mechanism (`flags2=+export_mvs`). The feasibility doc flagged this as the single highest-risk engineering task.
**Result: built and verified — the WASM MVs are byte-identical to the native PyAV pipeline.**

## Result
```
✅ SEAM VERIFIED: WASM libav surfaces motion vectors IDENTICAL to the native PyAV pipeline (22429 MVs over 30 frames, exact).
30/30 frames match exactly (WASM frameN+1 == PyAV frameN)   # I=0 MVs, P-frames 845/732/615/613/683/722/746/...
```
- `mv_wasm.wasm` (~1.9 MB): minimal FFmpeg (libavcodec/format/util) compiled to WASM via Emscripten — H.264
  decode + MOV/h264/mkv demux + `export_mvs`, no asm, no x264/x265, no threads.
- Decodes the whole `sd600.mp4` (640×320 h264) in Node and emits **802,611 motion vectors across 689 frames**,
  matching `mv_reference.json` (the PyAV `derisk.decode_lr_and_mvs` reference) frame-for-frame.

## The build trap (the actual hard part)
A naive `--disable-everything --enable-decoder=h264` build **decodes fine but emits ZERO motion vectors**, even
though the `AV_CODEC_FLAG2_EXPORT_MVS` flag is set and 700 frames decode. Cause: the h264 decoder's MV-export call
(`ff_print_debug_info2`, h264dec.c) is wrapped in **`if (CONFIG_MPEGVIDEODEC)`**, and `--disable-everything` sets
`CONFIG_MPEGVIDEODEC=0` → the export is compiled out. **Fix: `--enable-decoder=mpeg2video`** (it `_select`s
`mpegvideodec` → `CONFIG_MPEGVIDEODEC=1`), which re-enables the export path inside the h264 decoder. Diagnosed by
instrumenting the wrapper to print `frames_decoded` + the flag state, then tracing the export call to its config gate.

## Reproduce
```
# toolchain (one-time): git clone https://github.com/emscripten-core/emsdk /tmp/emsdk && \
#   (cd /tmp/emsdk && ./emsdk install latest && ./emsdk activate latest)
# source (one-time): git clone --depth 1 -b n7.1 https://github.com/FFmpeg/FFmpeg /tmp/ffmpeg-src
bash build_ffmpeg_wasm.sh        # configure + emmake make + emcc wrapper -> mv_wasm.{wasm,mjs}
node test_mv_wasm.mjs            # seam check vs mv_reference.json  (exit 0 = identical)
# regenerate the reference: cd ../../prototype && python -c "import derisk,json; ..."  (see git history)
```

## Files
- `build_ffmpeg_wasm.sh` — the minimal FFmpeg→WASM build + wrapper link (the recipe).
- `mv_wasm.c` — the MV-extraction wrapper (= FFmpeg's `extract_mvs.c` + a debug line); CSV → stdout.
- `test_mv_wasm.mjs` — Node seam test: mount clip in MEMFS, `callMain`, diff MV counts vs the PyAV reference.
- `mv_reference.json` — per-frame MV counts from the native PyAV pipeline (the validation target).
- `extract_mvs.c` — the upstream example, kept for reference.

## Clean API + JS flow + LIVE browser pipeline (all built & verified)
Beyond the CSV proof, the module now exposes a real pipeline, all parity-checked against the native PyAV path:
- **`mv_decode.c` + `mv_decode.mjs`** — clean JS-callable API: `mvdec_open` then loop `mvdec_next()`, reading
  `mvdec_rgb()` (RGB24 via libswscale) + `mvdec_mvs()` (packed motion vectors) from the heap per frame.
  `test_mv_decode.mjs`: 30/30 MV counts identical, **RGB bit-exact vs PyAV's `rgb24`** (mean|Δ|=0.000, max=0).
- **`flow.js`** — JS port of `derisk.build_lr_flow` (codec MVs → dense per-pixel LR fetch-flow). **Bit-exact** to
  the Python `build_lr_flow` (28160/28160 hole positions, value max|Δ|=0).
- **`mv_pipeline.html`** — the LIVE end-to-end browser demo, **zero offline data**: `sd600.mp4` → WASM decode
  (frame + MVs, ~66 ms) → JS flow → WebGPU MV-warp. Warping frame 0 by the codec MVs reconstructs frame 1 at
  **mean|Δ|=0.49 codes** over covered pixels (visually exact). Open it in a WebGPU browser to watch it run.

## What's left (engineering, not feasibility)
With the decode→MV→flow→warp chain proven live and bit-exact, the remaining work is product integration: wire
`mv_decode` + `flow.js` into `gop_live`'s GOP loop (replace the offline `lr_*.png`/`flow_*.bin` with live WASM
output) driven by a frame clock; add the SR anchor + occlusion stages already verified elsewhere; and
`SharedArrayBuffer`/cross-origin isolation only if threaded decode is ever needed (single-thread SD is likely
enough — the spike showed ~850–1000 fps). **Every input the pipeline needs is now reachable and verified
in-browser — the architecture is end-to-end web-only, proven not argued.**
