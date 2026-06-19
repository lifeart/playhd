# playhd — de-risk prototype

Validates the single riskiest seam of the NEMO-style real-time SR architecture:
**are H.264 motion vectors clean enough to warp a super-resolved keyframe onto later
frames, instead of running the SR network every frame?**

Steps 1–4 are complete: the prototype now runs on **real H.264 footage with real B-frames**
(`../sample.mp4`), upscales the anchor with a **real SR network** (`realesr-general-x4v3`),
and sweeps an **adaptive re-anchoring quality-vs-anchor-fraction curve**. Default flags keep
the original synthetic experiment byte-identical.

## Run

```bash
# Synthetic clip (pan + moving occluder), 360p->1080p x3, bicubic anchor, occ full, residual on
python3 derisk.py
python3 derisk.py --occ naive          # ablate occlusion masking
python3 derisk.py --no-residual        # ablate the NEMO residual

# Real H.264 (no true-HD metrics then). --start-frame / --max-frames window the clip:
python3 derisk.py --input ../sample.mp4 --start-frame 5000 --max-frames 48 --scale 3

# Real SR network on the anchor (realesr-general-x4v3 is an x4 model => use --scale 4):
python3 derisk.py --input ../sample.mp4 --start-frame 5000 --max-frames 48 --sr realesrgan --scale 4

# Adaptive re-anchoring (greedy, NEMO-style) or a fixed interval:
python3 derisk.py --input ../sample.mp4 --start-frame 0 --max-frames 48 --sr realesrgan --scale 4 --reanchor adaptive
python3 derisk.py --input ../sample.mp4 --start-frame 0 --max-frames 48 --reanchor interval:4

# Step 4 quality-vs-anchor-fraction curve (caches SR once per window, sweep is warp-only):
python3 anchor_sweep.py
```

### Flags

| flag | meaning |
|---|---|
| `--input` | H.264 file; omit for the synthetic clip |
| `--scale` | upscale factor (default 3; **use 4 with `--sr realesrgan`** — it is an x4 net) |
| `--start-frame` / `--max-frames` | (real only) window the clip; decode runs from 0 but skips conversion before the window and breaks after |
| `--no-residual` | warp only, skip the NEMO residual |
| `--occ {naive,full,reactive,adaptive}` | occlusion mask: `naive`=intra blocks only; `full`=+softmax-splat fwd-bwd+reactive; `reactive`=intra+reactive only (drops the fwd-bwd splat); `adaptive`=reactive + fwd-bwd only on motion-stressed directions (Step 7, `ADAPTIVE_TAU=0.06`) |
| `--sr {bicubic,realesrgan,realesrgan-x4plus}` | anchor/fallback upscaler. **Default `bicubic` is byte-identical to prior synthetic runs.** `realesrgan` = `realesr-general-x4v3` (compact, 1.21M params, ~130 ms/frame MPS); `realesrgan-x4plus` = `RealESRGAN_x4plus` (RRDBNet x23, 16.7M params, **heavy perceptual anchor, Step 8**, ~2.1–2.25 s/frame MPS). All x4 (use `--scale 4`); weights auto-download to `models/`, run on MPS |
| `--grain {off,low,med,high}` | per-frame film-grain final pass (**Step 8**, default `off`). Regenerated per output frame from the frame index (temporally independent), spatially-correlated Gaussian template, luma-modulated via a LUT, added to luma in gamma space AFTER reconstruction. Never warped / propagated / fed to references |
| `--region-aware` | **Step 9 (Stream-1)** OUTPUT-ONLY region-aware detail gating (default OFF => byte-identical regression). **Requires `--sr realesrgan-x4plus`.** The propagation chain stays single-model (heavy); the final per-output-frame blend is `out = a_hd*recon_heavy + (1-a_hd)*compact_source`, where `a_hd` is a temporally-stable, widely-feathered motion gate (lo=0.2, hi=1.0, feather=61 LR px) from the free per-frame MV magnitude, and `compact_source` is the per-frame compact SR. Static regions keep the heavy detail (propagates well); dynamic regions fall to the temporally-stable compact. Applied to the OUTPUT copy only (single `torch.lerp` on the GPU path), never into the propagation reference chain `R[]`. Reuses `region_quality.py` |
| `--reanchor {none,interval:K,adaptive}` | anchor policy: `none`=I-frames only; `interval:K`=every K-th backbone frame; `adaptive`=greedy re-anchoring |
| `--quality-margin` | (adaptive `--adapt-metric psnr`) dB floor on PSNR(prop, per-frame-SR) |
| `--fallback-budget` / `--adapt-metric {fallback,psnr}` | adaptive driver knobs (default `fallback` is content-fair) |

Outputs land in `out/` (or `--out DIR`): `metrics.csv`, `curves.png`, `erosion.png`, and sample
frames. See [`## Artifacts`](#artifacts) for what every `out*/` dir contains.

## How it works

- **MV extraction** via **PyAV** (`flags2=+export_mvs` → `Type.MOTION_VECTORS`). No `mvextractor`
  build needed; PyAV ships its own FFmpeg. Quarter-pel is decoded as `motion_x/motion_scale`;
  `dst_x/dst_y` is the block **center**. **`source` is a reference-LIST index, not a display
  distance** — `source<0` selects a past ref, `source>0` a future one (do NOT divide by `|n−m|`).
- **B-frame-aware reconstruction (Step 2).** `run()` / `reconstruct()` is two passes:
  1. **I/P reference backbone** — each P warps the **previous I/P anchor's** recon by its
     `source<0` MVs (`build_lr_flow(..., want="past")`). Drift ("cache erosion") accumulates
     along this chain, exactly like NEMO. The display-order `recon_prev` chaining used pre-Step-2
     is invalid once B-frames appear.
  2. **B-frames = bidirectional leaves** — each B warps the nearest **past** anchor (`want="past"`)
     AND the nearest **future** anchor (`want="future"`), blended per-pixel (temporal-distance
     weighted; one valid → use it; neither → bicubic fallback). B-frames are **never** used as a
     reference.
- **Anchor SR (Step 3).** An anchor is upscaled by `--sr`: `bicubic` (placeholder, isolates warp
  error) or `realesrgan` (`sr.py` = hand-written SRVGGNetCompact loading `realesr-general-x4v3.pth`,
  strict=True, on MPS). Per-frame SR is also computed for the baseline and to fill the fallback.
- **Residual** (NEMO-style, self-computed): `LR_cur − motion_comp(LR_ref)`, bilinear-upscaled and
  added to the warped HD frame (per-direction for B-frames).
- **Disocclusion / unreliable pixels** → filled from per-frame SR. Three cheap LR signals unioned
  (`--occ full`): intra holes (no MV), reactive residual, and softmax-splat fwd-bwd consistency
  (Ruder 2016 motion-adaptive test). Fallback fraction is reported per frame.
- **Adaptive re-anchoring (Step 4).** `compute_anchor_set()` resolves a `--reanchor` policy into a
  set of promoted backbone anchors; `anchor_set=∅` (`none`) is byte-identical to the Step-3
  I-frames-only backbone. `adaptive` greedily promotes anchors once the accumulated fallback budget
  (or PSNR floor) is exceeded.

### Metrics

- **`psnr_oracle_vs_true` / `psnr_perframe_vs_true`** (synthetic only, true HD available) — the
  economic argument: propagating a perfect anchor vs per-frame bicubic.
- **`psnr_prop_vs_perframe`** — NEMO cache-erosion metric (propagated-SR vs per-frame-SR).
  **NOISY/non-monotonic, and MISLEADING on real footage** (a high value can mean "dumped to
  bicubic"). Do not use as a decision metric.
- **`psnr_lr_consistency`** (Step 2) — PSNR(downscale(recon)→LR vs the true decoded LR). Useful as
  a sanity check but **too INSENSITIVE for decisions**: the per-frame-SR fallback is trivially
  LR-consistent, so it sits near the ceiling even when propagation has given up.
- **tOF** (TecoGAN Farneback flow EPE; lower=steadier). Synthetic: prop vs true. Real: prop-vs-LR
  and per-frame-SR-vs-LR (decoded LR = cleanest motion truth) plus prop-vs-per-frame-SR.
- **Decision rule:** use **tOF + fallback%**. They are the only honest signals (see GOTCHAs #12–13
  in `../handoff.md`).

## Result — synthetic (fast pan, x3, residual ON)

| metric | `--occ naive` | `--occ full` |
|---|---|---|
| perfect-anchor propagation vs true HD | 37.5 → 26.8 dB | **41.08 → 31.45 dB** |
| per-frame bicubic vs true HD (baseline) | ~27 dB flat | ~27 dB flat |
| **propagation beats per-frame for** | first **7** frames | **all 23** frames |
| warp+drift error (prop vs per-frame) | 49 → 36 dB | 50 → 41 dB |
| fallback pixels (SR re-run) | mean 2.4% | mean **5.84%** |
| tOF flicker (lower=steadier) | — | **0.091** prop vs **0.106** per-frame |

Residual OFF collapses the win window 7 → 3 frames, confirming both the **residual** and the
**occlusion mask** are load-bearing. `python3 derisk.py` (default bicubic) is the regression
guard — it must print oracle **41.08→31.45 dB**, **23/23** wins, fallback **5.84%**, tOF
**0.091/0.106**.

### Occlusion masking (`--occ full`) is the big lever

`--occ naive` only falls back on intra-coded blocks (true disocclusion). `--occ full` adds two
cheap LR signals: a **reactive mask** (high prediction residual flags bad/occluded MVs) and
**forward-backward consistency** (forward flow built by softmax-splatting the backward MV field,
checked with the **Ruder et al. 2016** motion-adaptive test `|w̃+ŵ|² > 0.01(|w̃|²+|ŵ|²)+0.5`).
Detecting these ~3% extra unreliable pixels turns "beats baseline for 7 frames" into "beats for
the entire 24-frame GOP" (+4.7 dB at the tail). This is FSR2-style history rejection transferred
to codec-MV-only input (FSR2's primary depth-based detector does NOT transfer).

## Result — real footage (Steps 2–4, `../sample.mp4`, 640×320, real B-frames)

**Step 2 — B-frame fix collapsed the fallback** (old display-order chaining dumped 60–95% of
B-pixels to bicubic; B-frames are ~5/6 of the GOP):

| window | B-frame fallback before → after |
|---|---|
| A (high-motion) | 42% → **0.96%** mean (97% → 4.7% max) |
| B (mid) | 19% → **4.3%** |
| C (talking-head) | 21% → **1.4%** |

The talking-head B-frame mask went from whole-subject-white to ~0.8% specks. The remaining
**~24% P-frame fallback in window A is honest large-motion occlusion** (the ~6-frame P→anchor
hop), not a bug.

**Step 3 — economic thesis validated on talking-head.** Per-frame SR latency is **~128–138
ms/frame on MPS** (~7.3–7.8 fps — the real-time bottleneck). Propagated-SR ≈ per-frame-SR
(**45–46 dB** PSNR-vs-PF) at ~7% of SR compute = **~14× fewer SR calls**, and adds temporal
stability. (Fallback pulls from the full per-frame-SR, computed anyway, so the % is the
**deployable amortized cost, not literal**.)

**Step 4 — adaptive re-anchoring + quality-vs-anchor-fraction curve** (`out_anchor_sweep/`):

| | anchors @ budget 0.5 | anchors @ budget 1.0 |
|---|---|---|
| A (high-motion) | 9 / 18.8% | 6 / 12.5% |
| C (talking-head) | 2 / 4.2% | 2 / 4.2% |

The adaptive policy spends **3–4.5× more anchors on high-motion than talking-head at the same
budget** — the core behavior works. **Talking-head free-rides the encoder I-frames** (4.2%
anchors within ~0.8 dB of ceiling, tOF flat ~0.13, fallback ~3%, **~24× fewer SR calls**).
**High-motion genuinely needs anchors** (adaptive budget 1.0 → 12.5% anchors, fallback 11%→9%,
tOF 0.68→0.53, **~6× speedup**, knee around K=2–4).

**REAL-TIME takeaway:** propagation makes SR no longer the bottleneck (amortized **5–16
ms/frame** vs ~128 ms per-frame); the bottleneck **SHIFTS to warp+mask+decode, which has not yet
been profiled.** That profile is the recommended next step.

## Temporal consistency (the real selling point)

Production real-time upscalers (NVIDIA RTX VSR, animejanai/Real-ESRGAN, Anime4K) are **all pure
per-frame spatial SR** — they flicker because each frame is upscaled independently. Propagation
is temporally coherent by construction (synthetic tOF 0.091 vs 0.106; on talking-head it tracks
true motion more closely than per-frame SR). No production tool exploits codec motion vectors for
temporal propagation on arbitrary video — that is this project's novelty.

## Result — Step 7 (real-time close: transfers + adaptive mask, both regimes)

Deployable per-frame wall-clock (torch/MPS, GPU-resident SR output + recon kept on the GPU; no
host transfers; best-of-12, ms/frame; absolute ms swings ±~25% with laptop thermal state, so the
ratios/verdicts are the robust takeaway). `profile_e2e.py` reports both this **deployable** number
and a **with-I/O** number (HD readback included) for honesty.

| window | full | reactive | adaptive | full +I/O (readback) |
|---|---|---|---|---|
| C talking-head | 42.4 ms (24 fps) | **27.7 ms (36 fps)** | **39.5 ms (25 fps)** | 67.2 ms (15 fps) |
| A high-motion  | 40.1 ms (25 fps) | **32.6 ms (31 fps)** | **38.3 ms (26 fps)** | 66.3 ms (15 fps) |

Removing the transfers (per-frame `upload_perframe` + HD `download`) takes the full path ~67 → ~42
ms/frame. **Which mask per regime:** on **talking-head** the fwd-bwd mask does not earn its cost —
**reactive == full quality** (fallback −0.76 pt, tOF(prop/LR) −0.020) → use reactive. On
**high-motion** reactive genuinely loses (fallback −3.36 pt of un-flagged bad-MV pixels, tOF +0.052)
→ need full; **adaptive recovers full quality** (tOF 1.3996 vs full 1.3994) at lower cost.
**Adaptive is ≤40 ms on BOTH windows** and is the safe single auto-policy. fp16 was measured and
rejected (the dominant mask op is kernel-launch-bound; `scatter_add` is slower in fp16). torch
matches numpy within float tolerance (recon PSNR 64–68 dB). Artifacts: `out_profile_{C,A}/`,
`tune_adaptive.py`, `final_timing.py`.

## Result — Step 8 (upscaling quality: heavy anchor + film grain + detail-drift)

Three additions, all behind new flags that default OFF / to the existing compact model (synthetic
regression stays byte-identical: 41.08→31.45, 23/23, 5.84%, 0.091/0.106):

**(1) Heavy perceptual anchor — `--sr realesrgan-x4plus`** (`RRDBNet`, 23 RRDBs, 16.7M params, loads
`RealESRGAN_x4plus.pth` `params_ema` strict=True on MPS). On an anchor crop it is **+61% sharper**
(var-of-Laplacian 28.4 vs compact 17.6 vs bicubic 2.5) — visibly more texture/edge detail (A/B in
`out_quality/ab_anchor_5031_crop_*.png`). **MPS latency ~2.1–2.25 s/frame** (640×320→2560×1280) vs
compact ~0.13 s. Affordable because SR runs only on sparse anchors: **amortized at 1 anchor/48
frames = ~45 ms/frame** (≈ the warp pipeline's per-frame cost); at a talking-head 4.2% anchor
fraction ≈ ~95 ms/frame amortized.

**(2) Per-frame film grain — `--grain {off,low,med,high}`** (`grain.py`): per OUTPUT frame, a
spatially-correlated Gaussian template re-seeded from the frame index (deterministic but different
every frame), amplitude scaled by local luma via a LUT, added to luma in gamma space as the FINAL
pass after reconstruction — never warped, never fed to references. Verified: visible & filmic
(not blocky), luma-modulated (grain std mid 9.1 > dark/bright 4.8 on a ramp; suppressed in shadows),
and **temporally INDEPENDENT** — raw grain-field correlation frame-to-frame **+0.0012** (≈0, not
frozen onto content) while self-correlation is **+1.0000** (deterministic). Visuals: `out_quality/
grain_ab.png`, `grain_consec.png`, `grain_field.png`.

**(3) THE KEY EXPERIMENT — does heavy-anchor detail survive propagation?** (`detail_drift.py` →
`out_quality/detail_drift.png`, `detail_drift.csv`). On clean single-anchor all-P chains (one I
anchor, then consecutive P-frames; **pure propagation** = SR the anchor only, fill occlusion holes
with bicubic, NO fresh per-frame SR re-injection), measured sharpness (var-of-Laplacian) and the
heavy-vs-compact advantage vs distance from the anchor, for both models. **Result is sharply
motion-dependent:**

| window | dist-0 heavy advantage | half-life of the advantage (pure warp) |
|---|---|---|
| talking-head (low motion) | +38 var-Lap (+21%) | **persists past 11 frames** (still +30 at d11) |
| high-motion | +31 var-Lap (+14%) | **≈ 1 frame** (→ +3.7 at d2, → ~0 by d3) |

On **low-motion** content the warps are near-identity, so the heavy anchor's extra detail rides
along almost undegraded for the whole GOP — a heavy anchor every ~24–48 frames genuinely delivers
its detail to non-anchor frames. On **high-motion** content warp-blur + occlusion erase it almost
immediately (both models collapse toward bicubic-smooth, var-Lap 220→27→13 by d2–d3): you would
need to anchor every 1–2 frames to keep it, which defeats the amortization. So the heavy anchor
helps the **static/slow** parts of a frame and is wasted on the **moving** parts — exactly where the
codec already needs frequent fallback/anchors anyway (and that fallback path is per-frame SR, which
the heavy model also improves). **Honest caveats:** the deployable system's fallback re-injects
fresh per-frame SR (21–40% of pixels here on these stretches), which sustains the advantage by
*re-running SR*, not by propagation (`gap_deployable` column / dashed curve). And the heavy anchor's
extra hallucinated texture is **less temporally stable** under warp on low-motion (tOF 0.66 vs
compact 0.33) — the NR/sharpness gain trades against temporal stability, as the metrics caveat warns.

**Verdict:** heavier anchor is worth it for low-motion/static content at a sparse anchor interval
(≤ ~half the GOP); it is NOT worth it for high-motion content (anchor interval would have to be
~1–2). No diffusion anchor tried (MPS-unverified — future work). Artifacts: `out_quality/`,
`sr.py` (RRDBNet), `grain.py`, `ab_anchor.py`, `detail_drift.py`, `grain_demo.py`.

## Result — Step 9 (3 parallel quality streams: region-aware INTEGRATED; pipelining; diffusion)

Three parallel streams each acted on the Step-8 motion-dependent-detail finding. Each was a NEW
module importing the shared code READ-ONLY; a seam-verification + integration pass then wired the
viable one and verified every interface (all seams matched — no caller/handler mismatches).

**(1) Stream-1 `region_quality.py` — INTEGRATED as `--region-aware`.** The Step-8 finding: a heavy
x4plus anchor adds real HF detail that PROPAGATES well on static content but is eroded in ~1 frame
on motion AND flickers (uniform x4plus tOF 0.66 vs compact 0.33). So gate detail PER PIXEL by a
motion map from the (free) codec MVs: static -> keep heavy, dynamic -> temporally-stable compact.
Wired as an **OUTPUT-ONLY final pass** (like grain): the propagation chain stays single-model
(heavy); the per-output-frame blend `out = a_hd*recon_heavy + (1-a_hd)*compact_source` is applied
to the OUTPUT copy only — a single `torch.lerp` (~1–2 ms) in `reconstruct_torch`, the numpy twin
reusing `region_quality.blend_region_aware`. `a_hd` is the temporally-stable, widely-feathered
(lo=0.2/hi=1.0/feather=61 LR px) gate; `compact_source` is the per-frame compact SR. **It NEVER
enters the reference chain `R[]`** (verified). Default OFF keeps the synthetic regression
**byte-identical** (41.08→31.45, 23/23, 5.84%, 0.091/0.106). Talking-head window (start 5000, 48f,
x4), region-split sharpness (var-Laplacian) / overall tOF:

| method | STATIC sharp | OVERALL tOF |
|---|---|---|
| uniform compact | 96.2 | 0.209 |
| uniform x4plus (heavy) | 119.5 | 0.231 |
| **region-aware** | **118.4** | **0.211** |

Region-aware **recovers 95% of the heavy static-region detail** (118.4 vs heavy 119.5, ≈ x4plus
sharpness) while keeping **overall tOF at the compact floor** (0.211 vs compact 0.209, far below
x4plus's 0.231) — i.e. the static-detail benefit without the uniform-x4plus flicker. These match
`region_quality.py`'s standalone `ra-wide` result (static 118.4, recovers 95%); the standalone's
slightly lower overall tOF (0.201) comes from blending a fully-propagated compact chain, whereas
the integration blends the cheaper already-computed per-frame compact SR per the hook. torch matches
numpy within the standing tolerance (region-aware recon PSNR 53.3 dB; `torch.lerp` == the numpy
`blend_region_aware` at 55.2 dB). Artifacts: `region_quality.py`, `out_region/`, `out_region_e2e/`.

**(2) Stream-2 `anchor_pipeline.py` — analysis artifact (buffered-only verdict).** A lookahead/
pipelining scheduler MODEL + threaded producer/consumer demo for the ~2.2 s x4plus anchor. Verdict
(single Apple-Silicon GPU): you can **buffer the LATENCY** (`B_min = ceil(L*F)` ≈ L seconds of
pre-rendered output) **but cannot beat the THROUGHPUT ceiling** `r + L/K ≤ 1/F` — per-frame recon
already consumes most of the 40 ms budget, so live x4plus at 25 fps is **not feasible at a useful
anchor interval** on one GPU. Feasible only with a dedicated anchor accelerator (2nd GPU/ANE,
K_min = L*F ≈ GOP), F=15 fps + reactive recon, or a cheaper heavy anchor. Kept standalone (no
derisk edits). Artifacts: `anchor_pipeline.py`, `out_pipeline/`.

**(3) Stream-3 `sr_diffusion.py` — analysis artifact (diffusion NO-GO on this box).** A feasibility
spike for a one-step real-world diffusion SR anchor (OSEDiff/SD2): an MPS probe of the exact SD2.1
UNet + VAE-decode per-tile compute graph (random weights = valid timing/op-coverage/memory, no
multi-GB download), plus a real-anchor hook that **raises loudly if no real model is wired** (never
a silent no-op). On this box no real diffusion SR is wired (basicsr dep + multi-GB gated weights +
disk), so it is import-safe but `--sr diffusion` is **NO-GO**; the probe quantifies the MPS cost for
the record. Use `--sr realesrgan-x4plus` for the heavy anchor. Artifacts: `sr_diffusion.py`.

## Known limitations / next

- ✅ **DONE — upscaling quality** (Step 8): heavy `realesrgan-x4plus` anchor + per-frame film grain
  + the detail-drift measurement (above). Open: a motion-adaptive anchor *model* choice (heavy
  anchor only where local motion is low enough for it to propagate); a GPU/torch grain pass (current
  `grain.py` is numpy — straightforward to port for the GPU-resident output path); diffusion anchors
  (MPS-unverified).
- ✅ **DONE — end-to-end fps profile + GPU acceleration** (Step 6) and **real-time close** (Step 7,
  above): adaptive mask + transfer removal put the full-quality path at ≤40 ms (25 fps) on both
  talking-head and high-motion. Residual levers: `build_flow` is ~5 ms/frame of CPU (vectorize the
  numpy MV-scatter / overlap with the GPU), and a cheaper fwd-bwd (nearest-splat reverse flow).
- **High-motion regime** still pays ~11% P-frame fallback (honest occlusion) and needs more anchors;
  the adaptive mask now keeps it at full quality, but a shorter P→anchor hop is still a lever.
- **No color-box clamping yet** (`--clamp`) — the one transferable FSR2 technique still to add.
- **No WebGPU / lighter-SR path** benchmarked — relevant if the target is the browser (ABPN,
  web-realesrgan, Anime4K-WebGPU, websr).
- **HEVC** — PyAV exports HEVC MVs too, but verify `motion_scale`/PU partitioning per stream.

## Artifacts

What every output dir / scratch script holds. **Keep all `out*/` dirs, the helper scripts,
`sr.py`, and `models/`** — they are the recorded evidence for Steps 1–4.

### Output dirs

| dir | step | what it is |
|---|---|---|
| `out/` | regression | default synthetic run (`metrics.csv`, `curves.png`, `erosion.png`, frames). Regenerated by `python3 derisk.py`. |
| `out_naive/`, `out_warponly/`, `out_baseline_synth/`, `out_synth_after/` | 1–2 | synthetic ablation / before-after scratch runs |
| `out_abl_naive/`, `out_abl_nores/`, `out_abl_nores_naive/` | — | synthetic ablations (`--occ naive`, `--no-residual`, both) |
| `out_abl_real_naive/`, `out_abl_real_nores/` | — | real-clip ablations |
| `out_real_A/`, `out_real_B/`, `out_real_C/` | 1 | first real-window runs (pre-B-frame-fix) for windows A (high-motion), B (mid), C (talking-head) |
| `out_real_A_after/`, `out_real_B_after/`, `out_real_C_after/` | 2 | the same windows after the B-frame backbone+leaf fix (the fallback-collapse evidence) |
| `out_sr_A/`, `out_sr_C/` | 3 | real-SR-network side-by-sides (bicubic vs propagated-SR vs per-frame-SR crops) for windows A and C |
| `out_anchor_sweep/` | 4 | `anchor_curve.png`, `sweep_A.csv`, `sweep_C.csv`, `summary.txt` — the quality-vs-anchor-fraction curve |
| `out_quality/` | 8 | `ab_anchor_*` (compact vs x4plus A/B crops), `detail_drift.png` + `detail_drift.csv` (heavy-anchor detail-survival vs distance), `grain_ab.png` / `grain_consec.png` / `grain_field.png` (film-grain visuals) |
| `out_x4plus_e2e/` | 8 | end-to-end x4plus + torch-backend + grain run on a real window (no-crash evidence) |
| `out_region/` | 9 | **Stream-1** standalone region-aware run: `region_split.png`/`.csv` (sharpness+tOF by region), `crops_*`, `motionmap_*`, `region_masks.png`, `mean_motion.png`; `cache/` = memoized heavy/compact SR `.npy` |
| `out_region_e2e/` | 9 | end-to-end `--region-aware --backend torch` run on the real talking-head window (integration no-crash evidence) |
| `out_pipeline/` | 9 | **Stream-2** scheduler model: `sweep.csv`, `pipeline_overview.png`, `summary.txt` (buffered-only verdict), `threaded_demo.txt` |
| `out_regcheck/`, `out_regcheck2/`, `out_regcheck_step8*/` | verify | synthetic regression-guard runs (confirm each step's edits stayed byte-identical) |

### Helper scripts

| script | what it does |
|---|---|
| `derisk.py` | the experiment (self-contained; all CLI flags above) |
| `sr.py` | the real SR networks — `SRVGGNetCompact`/`realesr-general-x4v3` (compact) **and `RRDBNet`/`RealESRGAN_x4plus` (heavy, Step 8)**, both x4, MPS; weights auto-download to `models/` |
| `grain.py` | **Step 8** per-frame film-grain pass (`apply_grain`); temporally-independent, luma-modulated, spatially-correlated Gaussian template. Self-test in `__main__` |
| `ab_anchor.py` | **Step 8** — A/B visual of compact vs x4plus on a real anchor frame → `out_quality/ab_anchor_*` |
| `detail_drift.py` | **Step 8** KEY EXPERIMENT — heavy-anchor detail survival vs distance-from-anchor (pure warp propagation, both models, two windows) → `out_quality/detail_drift.png` + `.csv` |
| `grain_demo.py` | **Step 8** — film-grain before/after + 3-consecutive-frame temporal-independence proof on real recon frames → `out_quality/grain_*` |
| `region_quality.py` | **Step 9 / Stream-1** — per-pixel motion-aware detail gating (the module wired into derisk as `--region-aware`). `region_masks`/`window_static_weight`/`blend_region_aware` are reused by derisk; standalone `main` measures region-split sharpness+tOF → `out_region/` |
| `anchor_pipeline.py` | **Step 9 / Stream-2** ANALYSIS ARTIFACT — heavy-anchor lookahead/pipelining scheduler model + threaded demo (buffered-LATENCY-yes / throughput-ceiling-no verdict). Standalone, not wired into derisk → `out_pipeline/` |
| `sr_diffusion.py` | **Step 9 / Stream-3** ANALYSIS ARTIFACT — MPS feasibility probe for a one-step diffusion SR anchor + import-safe, loud-on-failure hook. Diffusion NO-GO on this box (no real model wired); not exposed as `--sr` |
| `anchor_sweep.py` | Step 4 driver — sweeps anchor density on windows A & C, caches SR once per window (warp-only sweep), writes `out_anchor_sweep/` |
| `before_fallback.py` | recomputes the **pre-fix (Step 1)** reconstruction exactly, to get a clean BEFORE for fallback% and LR-consistency |
| `probe_gop.py` | probes a real clip's GOP structure, MV `source` signs, `|source|` range, and duplicate frames |
| `models/` | downloaded SR weights (`realesr-general-x4v3.pth`, `RealESRGAN_x4plus.pth`) |
