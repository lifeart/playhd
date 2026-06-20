# P1 core — WebGPU MV-warp propagation: BROWSER-VERIFIED, parity-perfect

The architecture's per-frame core is *propagation*: warp an SR'd anchor by the next frame's codec
MVs to reconstruct that frame (instead of per-frame SR). This is the second-biggest web-port risk
after MV extraction. **Built it in WebGPU and verified it in a real browser (Chrome, WebGPU) against
the Python prototype — parity is essentially bit-identical, and the warp is ~430× real-time.**

## What runs
A WGSL fragment shader reproduces `derisk.warp_hd` exactly: HD pixel → nearest LR flow lookup →
`src = dst + flow·scale` → bilinear, clamp-to-edge sample of the anchor texture (= cv2.remap +
BORDER_REPLICATE). Real data from `sample.mp4` (frame 0 I-frame SR'd by the compact net = anchor;
frame 1's 845 codec MVs = the flow), 640×320 → 1280×640 (×2, instant tier).

## Measured (in-browser, Chrome WebGPU)
| metric | value |
|---|---|
| **parity vs prototype `warp_hd`** | **mean \|Δ\| = 0.0021 codes, max = 1 code** → PASS (the 1-code max is bilinear/border rounding between cv2.remap and WGSL `textureSample`) |
| **steady-state warp** | **0.096 ms/frame → ~10,400 warps/sec** at 1280×640 ≈ **~430× real-time** |
| full pipeline (warm) | ~6.6 ms incl. texture upload + readback + parity |
| hole fraction (intra, no past MV) | 13.8% (the occlusion-fallback region, handled by a per-pixel mask in the next step) |

Visual: WebGPU output and the cv2.remap reference are indistinguishable (both warp the "USACHEV"
title card by the codec MVs).

## What this de-risks
Together with the MV-extraction spike (`../SPIKE_RESULT.md`: ~850–1000 fps WASM decode+MVs), **both
halves of the instant-tier pipeline are now measured in-browser**:
- decode + codec MVs (WASM): ~850–1000 fps
- MV warp / propagation (WebGPU): ~10,400 fps
The warp — the novel core — is parity-perfect and consumes ~0.1 ms of the frame budget. The GPU
substrate and the data contract (HD anchor texture + LR rgba32f fetch-flow) are proven.

## Remaining for a full P1 instant tier
1. **Compact-anchor SR in WebGPU** — websr / Anime4K-WebGPU are direct precedents (the compact net
   is an SRVGGNet-class CNN; port or reuse their WGSL conv shaders). Currently the anchor is SR'd
   offline; move it on-GPU.
2. **Reactive occlusion fallback** — per-pixel LR math (intra-hole + residual mask), then fall back
   to bicubic/compact at flagged pixels. Cheap WGSL passes; the 13.8% hole mask is already available.
3. **The WASM MV-binding build** (the engineering long pole) — expose
   `av_frame_get_side_data(MOTION_VECTORS)` to JS so the flow comes live from WASM decode instead of
   the offline file. Scoped in `../SPIKE_RESULT.md`; zero throughput risk, it's a marshalling task.
4. **The frame loop** — chain anchor→warp→warp→… across a GOP, re-anchor at I-frames/cuts, draw each
   to canvas via `requestAnimationFrame`.

## Reproduce
`python extract_warp_data.py` (writes `demo_data/`), then serve `web_spike/` over http
(`python -m http.server`) and open `webgpu_warp/index.html` in a WebGPU browser; the parity + timing
print to the page and to `window.__warp`.

---

# GOP propagation CHAIN — browser-verified (the full NEMO loop)

`gop.html` extends the single warp to a **full GOP**: SR only the anchor (I-frame), reconstruct every
other frame by warping the PREVIOUS recon with that frame's codec MVs, falling back to upscaled-LR at
intra holes — then play it back to canvas. Verified in real Chrome WebGPU against a Python reference
that runs the identical algorithm (`extract_gop_data.py`).

**GOP:** 16 frames, **1 anchor SR'd + 15 propagated** (6.25% SR — the NEMO ratio), 640×320 → 1280×640, 25 fps.

| metric | value |
|---|---|
| **chain parity vs Python reference** | **avg mean \|Δ\| = 0.016 codes, worst-frame = 0.065 (max 4)** over all 16 frames → PASS |
| per-frame drift | frame 1: 0.004 → frame 15: 0.065 — **15 chained bilinear warps accumulate ~nothing** (bounded, not a drift catastrophe) |
| playback | loops the GOP to `<canvas>` at 25 fps via `requestAnimationFrame` (visually coherent: the propagated title card) |

**Bug found + fixed along the way (a real WebGPU gotcha worth recording):** `textureSample` may only be
called from *uniform* control flow — calling it inside the data-dependent `if (hole)` branch is a WGSL
compile error, which silently invalidated the pipeline and rendered every propagated frame **black**
(caught via `pushErrorScope` + `getCompilationInfo`). Fix: **`textureSampleLevel(..., 0.0)`** (explicit
LOD, no derivatives → legal in non-uniform flow; identical result since the textures have no mips). Any
branchy WebGPU sampling (occlusion fallback, region gates) must use `textureSampleLevel`.

**This verifies the whole instant-tier propagation core in-browser:** sparse-anchor SR + chained codec-MV
warp + hole fallback + playback, parity-perfect vs the prototype. Remaining for a live P1: on-GPU compact
SR (so anchors aren't offline-SR'd), the reactive occlusion mask (vs the intra-only holes used here), and
the WASM MV-binding (live flow instead of the offline file). Reproduce: `python extract_gop_data.py` then
open `webgpu_warp/gop.html`; results in `window.__gop`.
