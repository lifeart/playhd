# playhd — browser-native SD→HD video upscaling, paid for by the codec's own motion vectors

> **Status: research spike, not production.** A de-risking prototype — everything runs and is measured,
> but it isn't a shipped product or a trained model. See [`web_spike/WRITEUP.md`](web_spike/WRITEUP.md)
> for the full technical write-up.

Super-resolving every frame of video in a browser is too expensive. So don't — super-resolve only **sparse
anchor frames** on the GPU (WebGPU/WGSL), and reconstruct everything in between by **warping the previous
frame with the codec's own motion vectors** (the H.264 encoder already did the motion search; reuse it).
This is the [NEMO](https://dl.acm.org/doi/10.1145/3372224.3419185) idea, moved into the browser.

The load-bearing trick — and the part nobody documents — is **getting H.264 motion vectors in the browser
at all**: `WebCodecs` exposes none, so the only path is a custom WASM `libav` build with `+export_mvs`, which
has a silent trap that makes it emit *zero* motion vectors while looking like it works. The recipe (and the
fix) is in the write-up and [`web_spike/wasm_mv/`](web_spike/wasm_mv/).

Everything runs **client-side** — no server, no upload: decode → extract MVs → warp → super-resolve anchors →
composite, all in the browser.

## What's here

| path | what |
|---|---|
| [`web_spike/WRITEUP.md`](web_spike/WRITEUP.md) | the full technical write-up (architecture, the MV build trap, the honest model eval, perf, gotchas) |
| `web_spike/wasm_mv/` | the browser player + WASM-libav motion-vector decode API + build recipe + the MV→flow port |
| `web_spike/webgpu_warp/`, `web_spike/conv_opt/` | the WebGPU/WGSL super-resolution drivers (SPAN + compact) + parity/timing harnesses |
| `experiments/` | the R1–R11 research rounds (scripts + REPORTs), incl. `r11_span_eval/` — the real-H.264 LPIPS/DISTS model evaluation |
| `prototype/` | the Python research core (decode/MV/warp/occlusion/SR, the offline pipeline the browser port is based on) |
| `server/` | an alternate server-rendered player |

## Running the demo

It needs a **WebGPU browser** (Chrome/Edge with `shader-f16`) and an **H.264 SD `.mp4`** clip.

```sh
# from the repo root, serve web_spike/ over http (the player fetches sibling files)
cd web_spike && python3 -m http.server 8767
# open http://localhost:8767/wasm_mv/player.html  and DRAG IN an H.264 SD .mp4
```

Drag the divider to compare playhd vs bicubic; the **"SR every N"** slider trades quality for speed
(propagation); the **model dropdown** switches SPAN ↔ compact. You can also load a CORS-enabled URL with
`?clip=<https-url>`.

### Get an openly-licensed test clip
No clip is bundled (the dev clips are copyrighted). Any short H.264 SD `.mp4` works; SPAN favours
live-action / faces. Open options:
- **Big Buck Bunny** — CC-BY 3.0 (Blender Foundation). Small SD/H.264 clips: <https://test-videos.co.uk/bigbuckbunny/mp4-h264>
- **Blender open movies** (Sintel, Tears of Steel, …) — CC-BY: <https://studio.blender.org/films/>

## What's NOT included (and why)
- **Super-resolution model weights** — SPAN `2xLiveActionV1`, `realesr-general-x4v3` ("compact"),
  `RealESRGAN_x4plus`. These are **third-party** weights with their own licenses; this repo ports them to
  WGSL, it doesn't relicense or redistribute them. Obtain them from their sources (e.g. OpenModelDB /
  the Real-ESRGAN releases), then regenerate the runtime data with the `export_*` scripts in `web_spike/`.
- **Video clips** — copyright. Bring your own / use an open clip (above).

## A note on the honest result
We measured SPAN vs the compact SR on **real libx264 degradation** (`experiments/r11_span_eval/`): SPAN does
**not** beat the compact overall — it's a **content-dependent live-action specialist** (wins talking-head
faces by ~20% LPIPS, loses on texture/high-motion). The player defaults to SPAN because the target content is
talking-head video, and exposes a selector for the rest. Details in the write-up.

## License
Original code: **MIT** (see [`LICENSE`](LICENSE)). Third-party model weights and any video clips are **not**
covered and are **not** redistributed here — see their respective licenses.
