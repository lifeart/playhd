# R13-E1 — browser occlusion gate: hard binary switch → soft per-pixel confidence

**Question owned.** Replace the browser player's hard binary occlusion fallback with a soft
per-pixel confidence gate (the browser port of `prototype/residual_gate.py`), and carry the free
per-block codec signals in the flow texture's unused `.zw`. Does it measurably improve the browser
output?

**VERDICT: GATED-GO** — yes, it measurably improves the browser output, and the per-block signals
are the part that makes it improve *everywhere*.

On the propagated frames the gate exists for, the full configuration
(`?gate=softblk&tau=16&s=3&kp=6&km=6`) beats the shipped hard switch on **all four metrics on both
test windows**, at **+0.03 ms/frame** of GPU and no extra CPU:

| window | PSNR | LPIPS | DISTS | tOF |
|---|---|---|---|---|
| calm (talking-head) | +0.90 dB | **−4.2 %** | **−8.7 %** | **−11.2 %** |
| jelly (high motion) | +0.35 dB | **−19.5 %** | **−9.7 %** | **−12.9 %** |

Residual-only (the straight `residual_gate.py` port) is a smaller win that *costs* tOF on calm
(+4.3 %) — the per-block `.zw` priors are what remove that cost. It still ships **default-OFF**
behind `?gate=` (page byte-identical when off), for the same reason its server twin is default-OFF:
this is two windows of one clip, the best (tau, s) is content-dependent, and the kp/km pair has
perceptual evidence at exactly one setting.

---

## 1. What was built

| file | what |
|---|---|
| `web_spike/wasm_mv/warp_shader.js` | **new** — the single copy of the HD warp WGSL, imported by both the player and the harness. `warpShaderWGSL({gate:"hard"})` reproduces the pre-R13 shader **byte-for-byte**; `{gate:"soft"}` is the confidence lerp. `reliabilityShaderWGSL({blocks})` is the new LR reliability pass. |
| `web_spike/wasm_mv/flow.js` | `buildLrFlow(..., blocks=true)` optionally emits the per-block signals; **new** `packLrFlowRGBA()` writes the `rgba32float` flow texture directly (§5). Default path untouched. |
| `web_spike/wasm_mv/player.html` | imports the shader module; `?gate=` selects the path; `window.__setReanchor()` test hook; `window.__diag()` now reports the gate. |
| `web_spike/wasm_mv/parity_shader.mjs` | **default-OFF guard**: shader text vs git, `buildLrFlow` default path, packer equivalence. |
| `experiments/r13_e1_browser_confidence/` | `prep_and_probe.py` (window extraction + LR signal probe), `harness.ts` (headless Deno/WebGPU A/B of the real pipeline), `eval.py`/`table.py` (scoring), `parity_gate.py` (WGSL vs `residual_gate.py`), `make_crops.py` (visual). |

**Flag and default (repo hard convention):** `?gate=hard` — **DEFAULT, OFF, byte-identical to the
shipped page**. `?gate=soft` / `?gate=softblk` opt in; `?tau= ?s= ?kp= ?km= ?mvref=` tune it.

```
$ deno run --allow-run --allow-read web_spike/wasm_mv/parity_shader.mjs
[1] warp shader gate=hard vs HEAD: IDENTICAL (1215 vs 1215 chars)
[2] buildLrFlow default path: UNCHANGED | .zw block channels: OK
[3] packLrFlowRGBA vs buildLrFlow+interleave: IDENTICAL
PARITY OK — gate=hard is byte-identical to the shipped path.
```

## 2. The gate, and every deviation from `residual_gate.py`

Reference (`prototype/residual_gate.py:48,60`): `react = mean_c|LR_cur − MV_warp(LR_ref)|` in codes,
`a = sigmoid((tau_res − react)/s_res)`, `a = 0` at intra holes, `a` formed at **LR**, **bilinearly**
upsampled to HD, then **hard-zeroed** at HD holes; output `a·warp + (1−a)·fresh`.

The port keeps that structure exactly. A first **LR-resolution render pass** writes `a` into a
filterable `r16float` texture (0 at the intra-hole sentinel `f.x < −1e8`); the HD warp pass samples
it at `pos/(W_hd,H_hd)` — precisely the mapping `cv2.resize(INTER_LINEAR)` uses — hard-zeroes `a`
where the nearest flow texel is a hole, and returns `mix(fresh, warped, a)`.

**Measured parity with the reference implementation** (`parity_gate.py`, same frames/MVs, tau=16
s=3, 30 propagated frames of the jelly window):

```
mean |a_gpu − a_residual_gate| = 0.00138     max = 0.0418     pixels with |Δ| > 0.05 = 0.0 %
```

**Deliberate deviations:**

1. **The `(1−a)` source is bilinear-upsampled LR, not fresh per-frame SR.** The server twin blends a
   fresh SR image there; a propagated browser frame has no per-frame SR at all (not running it *is*
   MV propagation), so the browser blends the same bilinear LR the hard switch already falls back
   to. Consequence: `a < 1` costs sharpness in the browser but is free on the server, so the browser
   needs a gate that **saturates to 1** on well-predicted pixels. This is exactly why the reference's
   own `tau_res=10, s_res=5` (which yields `a = 0.88` even at `react = 0`) is the **wrong** browser
   default — it is the one variant that loses perceptually on calm content (§4).
2. **Residual-only.** The Ruder fwd-bwd term is not ported — the same choice `residual_gate.py`
   makes (`use_fb=False` is its recommended default), and the browser never had a fwd-bwd mask.
   Its documented caveat carries over: an occlusion whose LR residual is coincidentally small keeps
   near-full warp weight.
3. **Reference frame.** The browser chains `prevRecon` in **display order** with `want="all"` flow
   (no I/P backbone — GOTCHA #12 constrains the prototype, not what ships in the browser). Both A/B
   arms share this, so it does not bias the comparison.
4. **Storage precision.** `a` lives in `r16float` (~1e-3) and `react` uses the GPU's true-float
   bilinear sampler vs `cv2.remap`'s 1/32-px fixed point. Both are inside the 0.0014 parity number.

**Per-block priors** (`?gate=softblk`) ride in the flow texture's free channels — `.z` = partition
area in px, `.w` = |MV| in LR px — and lower the threshold for suspicious blocks:
`tau_eff = max(tau − kp·(1 − area/256) − km·min(|mv|/mvref, 1), 0)`. This has **no counterpart in
`residual_gate.py`** (the server has no such term); it is new browser-side surface.

## 3. Setup

Clip `web_spike/sd600.mp4` (the shipped demo clip, 640×320 H.264); the two R12-E1 windows so the
numbers line up with the server twin: **calm** (start 437, talking-head) and **jelly** (start 582,
high motion), 40 frames each. `harness.ts` is a transcription of `player.html`'s `Player()` (same
weights, conv kernel, pixelshuffle, textures, anchor rule) minus canvas/audio/rAF, driven by frames
+ MVs extracted with PyAV `+export_mvs`; it imports the **same** `warp_shader.js` and `flow.js` the
page imports, so there is no second copy of the shader or the flow builder to drift.

Anchor cadence **SR every 4 frames** (the gate only exists on propagated frames).
**Reference for the metrics = the same pipeline in `--mode ref` (SR every frame)** — the ceiling
propagation approximates. Metrics are over **propagated frames only** (anchor frames are
bit-identical between arms, so an all-frames average just dilutes the effect by 25 %).
LPIPS/DISTS at 512-wide, tOF vs the decoded LR content at 640-wide.

**LR signal probe** (`probe_lr.json`, per propagated frame, mean):

| window | intra holes | hard fallback @tau=16 | mean react (codes) | MAE 16×16 vs sub-16 | MAE \|mv\|<1 / 1–4 / ≥4 |
|---|---|---|---|---|---|
| calm | 1.2 % | **9.0 %** | 3.9 | 1.89 vs 16.52 (**8.7×**) | 1.27 / 12.38 / 23.33 (**18×**) |
| jelly | 16.0 % | **37.9 %** | 27.3 | 21.04 vs 40.02 (1.9×) | 11.32 / 17.49 / 37.11 (3.3×) |

So the hard switch dumps **9 % (calm) to 38 % (jelly)** of every propagated frame to bilinear with no
feather — there is plenty for a gate to act on — and the brief's per-block discriminators reproduce
on this clip: strong on calm, weak on jelly (where every block is unreliable).

## 4. Results — soft vs hard (propagated frames, vs the SR-every-frame ceiling)

### calm (talking-head, start 437, 40 frames, 640×320 → 1280×640)

| variant | mean a | PSNR↑ | LPIPS↓ | ΔLPIPS | DISTS↓ | ΔDISTS | tOF(LR)↓ | ΔtOF |
|---|---|---|---|---|---|---|---|---|
| **hard (shipped)** | — | 30.257 | 0.04360 | — | 0.07265 | — | 0.1477 | — |
| soft tau=10 s=5 *(server-parity)* | 0.768 | 31.375 | 0.04394 | **+0.8 %** | 0.07441 | **+2.4 %** | 0.1599 | +8.3 % |
| soft tau=16 s=5 | 0.865 | 30.854 | 0.04274 | −2.0 % | 0.06823 | −6.1 % | 0.1655 | +12.0 % |
| soft tau=16 s=3 | 0.899 | 30.764 | 0.04257 | −2.4 % | 0.06729 | −7.4 % | 0.1540 | +4.3 % |
| soft tau=16 s=2 | 0.906 | 30.693 | 0.04280 | −1.8 % | 0.06800 | −6.4 % | 0.1558 | +5.5 % |
| **softblk tau=16 s=3 kp=6 km=6** | 0.880 | **31.155** | **0.04175** | **−4.2 %** | **0.06636** | **−8.7 %** | **0.1312** | **−11.2 %** |

### jelly (high motion, start 582, 40 frames)

| variant | mean a | PSNR↑ | LPIPS↓ | ΔLPIPS | DISTS↓ | ΔDISTS | tOF(LR)↓ | ΔtOF |
|---|---|---|---|---|---|---|---|---|
| **hard (shipped)** | — | 26.639 | 0.07477 | — | 0.09942 | — | 1.2228 | — |
| soft tau=10 s=5 *(server-parity)* | 0.562 | 27.087 | **0.05927** | **−20.7 %** | **0.08740** | **−12.1 %** | **1.0099** | **−17.4 %** |
| soft tau=16 s=5 | 0.668 | 26.851 | 0.06335 | −15.3 % | 0.09254 | −6.9 % | 1.1341 | −7.3 % |
| soft tau=16 s=3 | 0.701 | 26.754 | 0.06623 | −11.4 % | 0.09502 | −4.4 % | 1.1614 | −5.0 % |
| soft tau=16 s=2 | 0.710 | 26.695 | 0.06853 | −8.3 % | 0.09670 | −2.7 % | 1.1745 | −3.9 % |
| **softblk tau=16 s=3 kp=6 km=6** | 0.654 | **26.990** | 0.06022 | **−19.5 %** | **0.08980** | **−9.7 %** | **1.0649** | **−12.9 %** |

**Reading:**

* **Every soft variant beats the hard switch on high motion, on every metric** — the same direction
  and roughly half the magnitude of R12-E1's server-side result (LPIPS −29 % / DISTS −23 %), which
  is what the weaker `(1−a)` source (bilinear vs fresh SR) predicts.
* **On calm content the parameters decide the sign.** The server's own `tau=10, s=5` is the only
  variant that *loses* perceptually on calm (LPIPS +0.8 %, DISTS +2.4 %) — precisely deviation #1:
  it mixes ~23 % bilinear blur into pixels the warp had right. Moving tau to the browser's existing
  threshold (16) and steepening the ramp (s=3, `a = 0.995` at `react = 0`) turns that into
  **LPIPS −2.4 % / DISTS −7.4 %**.
* **tOF splits by regime**, as it did on the server: high motion improves (−5 % to −17 %), calm
  costs +4.3 % at the recommended setting (the server twin measured +5.8 %). Same known
  residual-only weakness, same size.
* **`tau=16, s=3` is the safe residual-only setting** (wins both windows on both arbiters);
  `tau=10, s=5` is materially better on high motion but negative on calm → a content-adaptive
  (or motion-adaptive) tau is an obvious follow-up.
* **Adding the per-block priors dominates every residual-only variant**: `softblk` reaches
  jelly-LPIPS −19.5 % (within 1.2 pts of the aggressive `tau=10, s=5` setting) while *also* being
  the best calm row, and it is the only variant with **no regression on any metric on any window**
  — including tOF, which every residual-only variant made worse on calm. So the recommended opt-in
  configuration is `?gate=softblk&tau=16&s=3&kp=6&km=6`.

### 4b. Per-block `.zw` priors (`?gate=softblk`)

`tau_eff = tau − kp·(1 − area/256) − km·min(|mv|/8, 1)` on top of `tau=16, s=3`:

| window | variant | mean a | PSNR↑ | tOF(LR)↓ | LPIPS↓ | DISTS↓ |
|---|---|---|---|---|---|---|
| calm | soft (kp=0, km=0) | 0.899 | 30.764 | 0.1540 | 0.04257 | 0.06729 |
| calm | + kp=6 (partition only) | 0.893 | 30.917 | 0.1458 | — | — |
| calm | + km=6 (MV length only) | 0.887 | 31.037 | 0.1381 | — | — |
| **calm** | **+ kp=6, km=6** | 0.880 | **31.155** | **0.1312** | **0.04175 (−1.9 % vs soft)** | **0.06636 (−1.4 % vs soft)** |
| calm | + kp=10, km=10 | 0.866 | 31.305 | 0.1203 | — | — |
| jelly | soft (kp=0, km=0) | 0.701 | 26.754 | 1.1614 | 0.06623 | 0.09502 |
| jelly | + kp=6 (partition only) | 0.698 | 26.771 | 1.1575 | — | — |
| jelly | + km=6 (MV length only) | 0.658 | 26.979 | 1.0615 | — | — |
| **jelly** | **+ kp=6, km=6** | 0.654 | **26.990** | 1.0649 | **0.06022 (−9.1 % vs soft)** | **0.08980 (−5.5 % vs soft)** |
| jelly | + kp=10, km=10 | 0.608 | 27.074 | 1.0287 | — | — |

**The free per-block channels earn their place.** At the one setting with perceptual evidence
(kp=km=6) they improve LPIPS *and* DISTS over the residual-only gate on both windows, and they turn
calm tOF from +4.3 % (worse than hard) into −11.2 % (better than hard). Mechanistically that is what
the probe predicted: the residual alone cannot tell "coincidentally small residual on a bad MV"
from "genuinely good MV", and partition size / MV length are exactly the priors that separate them
(calm: 8.7× and 18× MAE spread). The MV-length prior (km) carries most of the effect; partition size
(kp) alone moves PSNR/tOF only slightly. Stronger priors (kp=km=10) keep improving PSNR and tOF, but
those two metrics are compromised here (see the trap below), so kp=km=10 is **not** claimed as
better — it needs LPIPS/DISTS before anyone uses it.

**METRIC TRAP (worth adding to the GOTCHA list).** In this browser A/B, PSNR-vs-the-SR-reference and
tOF-vs-LR both improve *monotonically* as the gate leans harder on the fallback — because the
fallback is a bilinear upsample of the LR frame, which is trivially LR-flow-consistent (the same
insensitivity GOTCHA #13 records for `psnr_lr_consistency`) and, at K=4, often closer to the
reference than a misaligned warp. **Only LPIPS/DISTS can tell "the seam got fixed" from "the frame
got softer."** The kp/km rows above with `—` are a PSNR/tOF *screen*, not a verdict.

## 5. Cost

**Timings are measured on a CONTENDED box** (three sibling agents; load average 25–37 during every
measurement, recorded in the run logs). Absolute ms are therefore NOT comparable to this project's
quiet-box baselines (121.7 ms f16 combo conv, 0.096 ms warp). Both A/B arms were measured
back-to-back in the same sessions, so the **deltas** are the trustworthy part; the absolutes should
be re-taken on a quiet box before anyone quotes them.

GPU, timestamp-query timing (Deno/wgpu, f32 `wtile` kernel — Deno's Naga cannot compile
`enable f16`, so the harness uses the f32 path; Chrome uses f16 for both arms alike), 400 repeats,
warm-up quarter dropped, 3 runs each:

| gate | warp pass min | median | p90 |
|---|---|---|---|
| hard (shipped) | 0.107 ms | 0.215 ms | 0.216 ms |
| soft | 0.126 ms | **0.248 ms** | 0.250 ms |
| softblk | 0.143 ms | **0.248 ms** | 0.251 ms |

→ the extra LR reliability pass costs **≈ +0.03 ms/frame** (+15 % of a stage that is ~0.1 % of an
anchor frame: the SR anchor measured 188–194 ms in the same runs). The per-block channels cost
**nothing extra on the GPU** (already-allocated `rgba32float`, same sampler work).

CPU (main thread, JS, same contended box, two sessions):

| flow build + pack, 640×320 | session A | session B |
|---|---|---|
| shipped: `buildLrFlow` planes + interleave (.xy) | 2.0 ms | 3.4 ms |
| planes + interleave, with .zw | 6.6 ms | 4.5 ms |
| **new `packLrFlowRGBA` (.xy)** | 1.10 ms | 1.57 ms |
| **new `packLrFlowRGBA` (.xy + .zw)** | **1.21 ms** | **1.38 ms** |

→ the naive way to carry the per-block signals is **NOT free** (it adds a second full-image pass and
two extra planes). Packing the flow texture directly from the MV records makes them free **and is
~2× faster than the shipped flow build even without them** — an unrelated CPU win now sitting behind
the gate flag (the default path was deliberately left untouched). Verified equivalent two ways:
`parity_shader.mjs` check [3], and a full 40-frame re-render whose dumped HD frames were
**bit-identical** (max |Δ| = 0) to the pre-packer run.

**WebGPU limits side-question.** `player.html` requests no `requiredLimits`, so it gets the defaults
(16 KiB workgroup storage / 256 invocations) while this adapter offers 32 KiB / 1024. Measured with
`--spike limits` (3 alternating runs): SR anchor 189.7 / 190.9 / 188.0 ms at the defaults vs
191.4 / 187.2 / 190.5 ms raised — **identical within noise, as expected**: `combo.wgsl` uses
`workgroup_size(16,16)` = 256 invocations and ~3.6 KiB of workgroup storage, i.e. it already fits
under the default caps. Raising the limits is free headroom for a *future* kernel (a >256-invocation
or bigger-tile variant), not a win today. (An earlier pair of runs showed 415 vs 263 ms and would
have "shown" a 1.6× win — that was pure load noise. Recorded as a warning.)

## 6. Visual

`crops_jelly_f7.png`, `crops_calm_f7.png` — the worst-error 256×256 box of a propagated frame:
reference (SR every frame) | hard | soft | softblk, with each arm's |error| map. The hard switch's
error map is blocky (macroblock-sized patches dumped to bilinear); the soft arms' are feathered
ramps. Mean |err| in that worst box: jelly 13.93 (hard) → 13.59 (soft) → 13.3 (softblk); calm 13.28
→ 12.65 → 12.5. Over the WHOLE propagated frames (all 30, not just the worst box):

| window | hard | soft t16s3 | softblk kp6km6 |
|---|---|---|---|
| calm | 2.812 | 2.681 (−4.7 %) | **2.528 (−10.1 %)** |
| jelly | 6.101 | 6.058 (−0.7 %) | **5.673 (−7.0 %)** |

## 7. Browser verification (Chrome, WebGPU, f16 path)

Served `web_spike/` over `python3 -m http.server`, loaded `player.html?demo` with each gate:

| URL | result |
|---|---|
| `?demo` (default) | `__diag().gate = "hard"`, 24 frames stepped at SR-every-4 (6 anchors / 18 warps), no console errors — **the default page is unaffected**. |
| `?demo&gate=soft&tau=16&s=3` | `gate="soft"`, gateCfg echoed, 8 frames stepped (2 anchors / 6 warps), output mean 133.7 vs source 131.6, no errors. |
| `?demo&gate=softblk&tau=16&s=3&kp=6&km=6` | `gate="softblk"`, 24 frames stepped (6 anchors / 18 warps), no errors. **Re-verified after the `packLrFlowRGBA` swap** (16 frames, 4 anchors / 12 warps, no console output) — the seam was re-exercised in the real browser, not just in the harness. |

So the new `r16float` reliability target, the two-pass encoder and the `.zw` packing all work on the
real Chrome/Dawn f16 path, not just on Deno/wgpu f32. (Frame-rate numbers from that tab are
meaningless — background tab + readback hooks + a load-37 box — so none are quoted.)

## 8. Honest limits of this result

* **Two windows of ONE clip** (the demo clip, which is also `sample.mp4` content). The server twin
  is default-OFF pending exactly this kind of multi-clip sweep; the browser port inherits that gate.
* **Blast radius is the propagation path only.** The shipped page defaults to `SR every 1 frame`
  (pure per-frame SR), where no warp — and therefore no gate — ever runs. The gate changes output
  only when the "SR every N frames" slider is raised, i.e. the propagation mode this architecture is
  actually about.
* The kp/km grid has **only PSNR/tOF** evidence (see the metric trap above) except at kp=km=6.
* Timings are contended (§5).
* `a` is quantized to `r16float` and the reliability pass re-reads `lrPrev`/`lrCur` that the warp
  pass also samples — a fused single-pass variant would remove one texture read per LR pixel; not
  attempted (the measured cost is 0.03 ms).

## 9. Recommended next steps

1. **Multi-clip sweep** of `tau/s` (and kp/km) on more content before any default flip — the same
   graduation path as the server's `RESIDUAL_GATE`, β=0.85 and `deblock_pre`.
2. **Motion-adaptive tau**: the data says calm wants tau≈16/s=3 and high motion wants tau≈10/s=5;
   the per-frame mean |MV| (already free in `.w`) is an obvious selector.
3. **LPIPS/DISTS the kp/km grid** rather than trusting the PSNR/tOF screen.
4. Consider landing `packLrFlowRGBA` on the default path as a standalone CPU win (needs its own
   default-path parity run; the rendered-frame bit-identity check above already exists).
5. **For the seam pass (agent E):** R13-E2 (`prototype/block_confidence.py`,
   `experiments/r13_e2_block_confidence/`) is the SERVER-side twin of the same per-block idea,
   written in parallel. The two arrived independently at the same two signals and the same
   hypothesis (the residual is blind exactly where a geometrically wrong MV has a small residual).
   Worth diffing the two formulations directly: this experiment's browser form is
   `tau_eff = max(tau − kp·(1 − area/256) − km·min(|mv|/mvref, 1), 0)` with kp=km=6, tau=16, s=3,
   mvref=8, applied to the LR reliability BEFORE the sigmoid; if R13-E2 normalizes or combines them
   differently, one of the two should move so the browser and server gates stay the same function.

## 10. Reproduce

```bash
# 1. windows + LR probe (writes rgb_/mv_ bins to a scratch dir)
python3 experiments/r13_e1_browser_confidence/prep_and_probe.py --scratch $SCR
# 2. render the A/B arms headlessly (Deno + WebGPU)
cd experiments/r13_e1_browser_confidence
deno run --unstable-webgpu --allow-read --allow-write --allow-env harness.ts \
  --data $SCR/calm --mode ref  --out $SCR/out_calm_ref
deno run ... --data $SCR/calm --mode hard    --reanchor 4                  --out $SCR/out_calm_hard4
deno run ... --data $SCR/calm --mode soft    --reanchor 4 --tau 16 --s 3   --out $SCR/out_calm_soft_t16s3
deno run ... --data $SCR/calm --mode softblk --reanchor 4 --tau 16 --s 3 --kp 6 --km 6 --out $SCR/out_calm_softblk_kp6km6
# 3. score / tables / visuals / parity
python3 eval.py --scratch $SCR --window calm --variants hard4,soft_t16s3,softblk_kp6km6
python3 table.py calm jelly
python3 make_crops.py --scratch $SCR --window jelly --frame 7 --variants hard4,soft_t16s3,softblk_kp6km6
python3 parity_gate.py --scratch $SCR --window jelly --variant soft_t16s3 --tau 16 --s 3
deno run --allow-run --allow-read ../../web_spike/wasm_mv/parity_shader.mjs
# 4. in a browser
python3 -m http.server 8123    # from web_spike/
#   http://127.0.0.1:8123/wasm_mv/player.html?demo&gate=soft&tau=16&s=3
```
