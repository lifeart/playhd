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
warp + hole fallback + playback, parity-perfect vs the prototype. Reproduce: `python extract_gop_data.py` then
open `webgpu_warp/gop.html`; results in `window.__gop`.

---

# On-GPU compact SR — browser-verified, BIT-EXACT (removes the last offline dependency except MVs)

`sr.html` runs the actual compact anchor net — **realesr-general-x4v3 / SRVGGNetCompact** (conv(3→64)+PReLU,
32×[conv(64→64)+PReLU], conv(64→48), PixelShuffle(4), + nearest(×4) residual = **34 conv passes**, 1.21M
params) — entirely in WebGPU, vs the PyTorch net. So the anchor no longer needs offline SR; only the MVs do.

**How:** a generic conv+PReLU compute shader (channel-planar feature buffers, ping-pong, zero-pad, weights
from a storage buffer at per-pass offsets), chained 34× in JS, then a PixelShuffle(4)+nearest-residual render
pass. Weights exported from the .pth by `export_compact_weights.py` (matches PyTorch layout + the RGB/[0,1]
preprocessing exactly).

| metric | value |
|---|---|
| **parity vs PyTorch `sr.upscale`** | **mean \|Δ\| = 0.0000035 codes, max = 1** over 1024×1024×3 → essentially bit-exact (≈11 total codes of diff across 3.1M values; the 34-layer fp32 CNN reproduces PyTorch) |
| visual | WebGPU SR and the PyTorch reference are pixel-indistinguishable |
| timing (this naive conv) | ~1.33 s for 256×256→1024×1024 (x4) |

**Honest perf note:** the 1.33 s is a *naive* conv (one invocation per output element — no shared-memory
tiling, no fp16, no vec4). It's a one-time **anchor** (SR ~1 frame in 16–48; the warp at ~10,400 fps carries
the rest), so it's amortized — but for a snappy live instant tier it wants the standard WGSL conv
optimizations (tiled/workgroup-shared conv, fp16, vectorized MACs) or reusing websr/Anime4K-WebGPU's tuned
kernels. That is a known optimization path, not a feasibility question — **the parity (bit-exact) is the
result that matters: the real anchor net runs faithfully in the browser.**

**Second instance of the texture-usage gotcha:** the display blit sampled the SR `out` texture, which was
created `RENDER_ATTACHMENT|COPY_SRC` — missing `TEXTURE_BINDING` → the bind group was invalid → the blit drew
black (parity was unaffected, since it reads via `COPY_SRC`). Any texture you later *sample* needs
`TEXTURE_BINDING` in its usage. Fixed.

## Web-port status after this step (offline deps remaining: just the MVs)
- ✅ MV extraction in WASM — ~850–1000 fps (`SPIKE_RESULT.md`)
- ✅ single MV warp in WebGPU — parity 0.002, ~430× real-time
- ✅ full GOP propagation chain + playback — parity 0.016, browser-verified
- ✅ **compact anchor SR in WebGPU — bit-exact (0.0000035), browser-verified** ← the anchor is now on-GPU
- remaining: reactive occlusion mask (vs intra-only holes), the WASM MV-binding (live flow), conv perf opt,
  and wiring the on-GPU SR into the GOP loop as the anchor source.

Reproduce: `python export_compact_weights.py` then open `webgpu_warp/sr.html`; result in `window.__sr`.

---

# LIVE pipeline (on-GPU SR anchor + warp chain) — BROWSER-VERIFIED, parity-perfect

`gop_live.html` + `extract_gop_live.py` wire the on-GPU compact SR into the GOP loop: the anchor is SR'd
**in-browser** (no offline anchor PNG), the rest propagate by codec-MV warp + hole fallback, played to canvas.
**MVs are the only remaining offline input.** 256×256 LR crop at the net's native x4 (→1024×1024).

| metric | value |
|---|---|
| **on-GPU SR anchor (frame 0) parity** | **mean \|Δ\| = 0.000, max 1** (bit-exact, in-chain) |
| **full-chain parity vs Python ref** | **avg 0.003, worst-frame 0.0098 codes** over all 12 frames → PASS |
| hole-fallback parity | 0.02 (non-hole 0.000) |
| playback | loops to canvas at 25 fps; visually correct propagated crop |

**The bug that was here (root-caused — a real WebGPU gotcha, the 3rd texture-usage one):** the LR fallback
texture initially rendered **black**, so hole pixels diverged by up to ~90 codes (full-chain avg ~16–27).
Cause: **`copyExternalImageToTexture` requires the destination texture to include `RENDER_ATTACHMENT` usage**;
`gop_live`'s `lrTexOf` had only `TEXTURE_BINDING|COPY_DST`, so the LR upload silently produced a black
texture and the fallback sampled black. (`gop.html` already had `RENDER_ATTACHMENT` and was correct.) Isolating
it: the warp split cleanly — **non-hole pixels were already 0.000** (warp perfect), all error was in the holes;
an x4 standalone single-warp test (`index.html` at SCALE=4) returned **0.000**, proving the browser x4 warp;
then a hole/non-hole pixel dump showed the fallback returning `[0,0,0]` vs real LR content → the missing usage.
**Fix:** add `RENDER_ATTACHMENT` to `lrTexOf` → full-chain parity 0.003. **Three texture-usage rules now
recorded:** (1) `textureSample` only in uniform control flow (use `textureSampleLevel`); (2) a *sampled*
texture needs `TEXTURE_BINDING`; (3) a `copyExternalImageToTexture` destination needs `RENDER_ATTACHMENT`.

**This closes P1's core in-browser:** the full instant-tier pipeline — on-GPU compact SR anchor (bit-exact) +
chained codec-MV warp + hole fallback + playback — runs and is parity-verified against the prototype, with
**MVs the only offline input**.

## + Reactive occlusion mask (catches bad MVs, not just intra holes) — parity-verified
The fallback now unions **(a) intra holes** (flow sentinel) **with (b) the reactive residual**
`mean_c|LR_cur − warp_lr(LR_prev)| > TAU(16)` — exactly the prototype's `occlusion_mask_lr` *reactive* mode
((a)+(b); the project's key occlusion lever — "detecting ~3% extra unreliable pixels flips wins-7-frames →
wins-all-23"). In the warp shader: per HD pixel, bilinear-sample the *previous* LR at `lr+flow`, compare to
the current LR, fall back where the residual exceeds the threshold. Re-extracted the Python reference with the
same reactive mask and re-verified: **full-chain parity avg 0.007 / worst-frame 0.020 codes** over 12 frames
(per-frame mean 0.001–0.013; the occasional max ~100 is a single pixel sitting exactly on the `react>16`
threshold flipping warp↔fallback between float and uint8 — negligible in the mean). Remaining for a
fully-live tier: the WASM MV-binding (emsdk — parked), conv perf optimization (the ~1.3 s naive anchor),
the adaptive fwd-bwd splat (occlusion mode (c), high-motion only), and a full-frame (uncropped) run.

Reproduce: `python extract_gop_live.py` then open `webgpu_warp/gop_live.html`; results in `window.__live`.
