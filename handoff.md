# playhd — handoff

Real-time SD→FullHD video upscaling "on the fly". Status as of 2026-06-19.

## TL;DR

Goal: upscale low-res video to HD in real time by running an **expensive neural SR network
only on sparse anchor/keyframes**, then reconstructing every other frame cheaply by **warping
the super-resolved anchor with codec motion vectors + residuals**. This is the **NEMO**
(MobiCom 2020) architecture; we are porting its idea from VP9 to **H.264/H.265**.

**Current state:** the architecture is validated end-to-end on real H.264 and **GPU-accelerated to
real-time on Apple Silicon**. **5 deep-research passes** done; **Steps 1–9 complete** + a **layered-
architecture track (L1–L4) in progress** (see below). Under **git** (init 2026-06-19; commits
`7520d7e` Steps 1–8, `ee53c13` Step 9; `.gitignore` excludes `sample.mp4`/`models/`/`out*/`).

Headline results:
- **Economic thesis validated:** on a talking-head clip, propagated-SR ≈ per-frame-SR (45–46 dB) at
  ~7% of the SR compute (~14–24× fewer SR calls). No production upscaler (RTX VSR, Anime4K, animejanai)
  exploits codec MVs for temporal propagation → this is the novelty.
- **Real-time reached** (Steps 6–7): numpy 252 ms/frame → torch/MPS **38–40 ms (25 fps)** via GPU warp+
  mask+blend, transfer removal, and `--occ adaptive` (full quality both regimes) or `--occ reactive`
  (faster, full quality on low-motion). RECOMMENDED LIVE CONFIG: `--backend torch --occ adaptive --grain`.
- **Quality axis** (Steps 8–9): heavy x4plus anchor (+61% sharper, 2.2 s/frame so buffered/quality-mode
  only — live anchor stays compact); per-frame film grain; `--region-aware` motion-gated detail (recovers
  95% of heavy static detail at compact flicker). Diffusion anchor = NO-GO on this box (7–14× too slow,
  VAE-decode bound — needs TAESD); live heavy anchor = throughput-bound on one GPU.
- **Layered track ("video as layers" idea) — DONE (L1–L4):** decompose into a static background plate
  (RVM matte L1 + temporal-median plate L2, SR'd ONCE/scene) + per-frame foreground, composite + grain (L3).
  RESULT: uniquely delivers a **rock-stable, denoised, x4plus-sharp background (~167× less flicker than
  per-frame x4plus)** — but it is **NOT a speed win** (~4–5× slower; the moving foreground can't be
  propagated so it needs fresh per-frame SR), and the earlier "layered may be cheaper" hypothesis was
  WRONG. Verdict: a quality/stability OPTION to kill background shimmer, not the default path. Real-time
  path stays `--region-aware` propagation.

## Repo layout

```
sample.mp4             # real test clip: H.264 640x320 25fps ~34min (~50k frames), real B-frames
prototype/derisk.py    # the de-risk experiment (self-contained; see prototype/README.md)
prototype/sr.py        # real SR nets: SRVGGNetCompact/realesr-general-x4v3 (compact) + RRDBNet/RealESRGAN_x4plus (heavy, Step 8), x4, MPS
prototype/grain.py     # Step 8 per-frame film-grain pass (temporally independent, luma-modulated)
prototype/region_quality.py # Step 9 Stream-1: per-pixel motion-aware detail gating (WIRED into derisk as --region-aware)
prototype/anchor_pipeline.py # Step 9 Stream-2 ANALYSIS: heavy-anchor pipelining/buffer scheduler model (standalone)
prototype/sr_diffusion.py    # Step 9 Stream-3 ANALYSIS: MPS diffusion-SR feasibility probe + loud-fail hook (standalone, NO-GO)
prototype/detail_drift.py  # Step 8 KEY EXPERIMENT: heavy-anchor detail survival vs distance-from-anchor
prototype/ab_anchor.py # Step 8 compact-vs-x4plus A/B visual; grain_demo.py = grain before/after + temporal-independence proof
prototype/anchor_sweep.py   # Step 4 quality-vs-anchor-fraction sweep driver
prototype/before_fallback.py # Step 1 "before" recon (clean BEFORE for fallback%/LR-consistency)
prototype/probe_gop.py      # GOP/MV-source-sign/duplicate-frame probe for a real clip
prototype/matting.py        # LAYERED L1: RVM foreground matting on MPS (load_rvm/matte_sequence/fg_mask_lr)
prototype/background_plate.py # LAYERED L2: static-camera background plate (scene_segments/build_plate/sr_plate/sample_plate)
prototype/layered_pipeline.py # LAYERED L3: composite plate + per-frame FG + grain (in progress)
prototype/demo_matting.py / demo_background_plate.py  # layered demos
prototype/README.md    # how the prototype works + result tables + ARTIFACTS index
prototype/models/      # auto-downloaded SR weights (realesr-general-x4v3.pth, RealESRGAN_x4plus.pth) + RVM via torch.hub cache
prototype/out_quality/ # Step 8 artifacts: detail_drift.png/.csv, ab_anchor_*, grain_{ab,consec,field}.png
prototype/out_matting/ out_plate/ out_layered/  # LAYERED L1/L2/L3 artifacts
prototype/out*/        # generated artifacts (see prototype/README.md ## Artifacts) — all gitignored
handoff.md             # this file
```
**Under git** since 2026-06-19 (`master`): `7520d7e` Steps 1–8, `ee53c13` Step 9. `.gitignore`
excludes `sample.mp4` (78 MB), `prototype/models/` (weights), `out*/` (~340 MB), `__pycache__`, logs.
Layered-track modules (matting/background_plate/layered_pipeline + the analysis modules
anchor_pipeline/sr_diffusion) are committed after the layered track's seam-verification pass.
Persistent project memory (loaded each Claude session) lives outside the repo at
`~/.claude/projects/-Users-lifeart-Repos-playhd/memory/` → `playhd-project.md`,
`playhd-research-state.md`.

Raw research reports (full JSON, more detail than summarized here) are in the task outputs under
`/private/tmp/claude-501/.../tasks/`: `wo4me2syo` (pass 1: NEMO/architecture), `wgdeoh3u5`
(pass 2: H.264 MV extraction), `wy0mun7qt` (pass 3: occlusion/FSR2/metrics/baselines), `wau8fdg60`
(pass 4: better-upscaling — heavy/diffusion anchors, grain, propagation-stabilizes-detail),
`wce7evoli` (pass 5: layered/compositional video SR — Wang&Adelson, RVM, MV-free segmentation/plate).

## How to run

```bash
cd prototype
python3 derisk.py                       # synthetic clip, 360p->1080p x3, occ full, residual on
python3 derisk.py --occ naive           # ablate occlusion masking
python3 derisk.py --no-residual         # ablate the NEMO residual
# real H.264 (no true-HD metrics then). --start-frame/--max-frames window a real clip:
python3 derisk.py --input ../sample.mp4 --start-frame 5000 --max-frames 48 --scale 3
# real SR network on the anchor (x4 model => use --scale 4):
python3 derisk.py --input ../sample.mp4 --start-frame 5000 --max-frames 48 --sr realesrgan --scale 4
# adaptive re-anchoring (greedy NEMO-style), or a fixed interval:
python3 derisk.py --input ../sample.mp4 --start-frame 0 --max-frames 48 --sr realesrgan --scale 4 --reanchor adaptive
python3 derisk.py --input ../sample.mp4 --start-frame 0 --max-frames 48 --reanchor interval:4
python3 anchor_sweep.py                 # Step 4 quality-vs-anchor-fraction curve (out_anchor_sweep/)
```
Current flags: `--input`, `--scale` (x4 for the real SR net), `--frames` (synthetic length),
`--start-frame`/`--max-frames` (window a real clip; decode runs from 0 but skips conversion
before the window and breaks after), `--no-residual`, `--occ {naive,full}`,
`--sr {bicubic,realesrgan}` (default bicubic = byte-identical to prior synthetic runs),
`--reanchor {none,interval:K,adaptive}`, `--quality-margin` (dB floor for the psnr adaptive
driver), `--fallback-budget` + `--adapt-metric {fallback,psnr}` (adaptive driver knobs).

Env: Python 3.11 (miniconda). Deps already present: `av` (PyAV 17), `cv2` 4.11, `numpy`,
`scipy`, `matplotlib`, `torch` (MPS). Missing: `skimage` (SSIM is hand-rolled), `mvextractor`
(not needed — PyAV is used instead). SR weights auto-download to `prototype/models/` on first
`--sr realesrgan` run.

## Validated result (synthetic, fast pan + moving occluder, x3, residual ON, occ full)

| metric | value |
|---|---|
| perfect-anchor propagation vs true HD | 41.1 dB (f1) → 31.4 dB (f23) |
| per-frame bicubic baseline | ~27 dB flat |
| **propagation beats per-frame for** | **all 23 frames of the GOP** |
| tOF flicker (lower=steadier) | 0.091 (propagation) vs 0.106 (per-frame SR) |
| fallback pixels (SR re-run) | mean 5.8% |

Ablations: `--occ naive` → beats baseline only 7 frames; `--no-residual` → only 3 frames.
So **both the NEMO residual and the occlusion masking are load-bearing.**

## Program progress — Steps 1–4 (real footage)

**Test clip:** `sample.mp4` = H.264, 640×320, 25 fps, ~34 min (~50k frames). Has **real
B-frames** (`...PPPP BBBBBP BBBBBP I ...` cadence). `|source|` is only ±1; `motion_scale`
uniformly 4. Three working windows used throughout: **A** (start 0, high-motion stress),
**B** (mid), **C** (start ~5000, talking-head, the realistic case).

**Step 1 — windowing + B-frame diagnosis.** Added `--start-frame`/`--max-frames` (decode runs
from 0 but skips conversion before the window, breaks after). Diagnosed: **`source` is a
REFERENCE-LIST INDEX, not a display distance.** P's `source` is always −1; B-frames carry both
−1 (past) and +1 (future), **future-dominant**. The old `build_lr_flow` skipped future refs →
**60–95% of B-pixels dumped to bicubic fallback** (B-frames are ~5/6 of the GOP). Display-order
`recon_prev` chaining is **invalid** once B-frames appear (a P's true reference is the previous
I/P anchor, e.g. ~6 display-frames back). `psnr_prop_vs_perframe` is **MISLEADING on real
footage** (a high value can mean "gave up to bicubic"). The encoder re-anchors at mid-stream
I-frames (scene cuts are handled by the codec).

**Step 2 — B-frame fix (backbone + bidirectional leaves).** `run()` restructured into:
(1) **I/P reference backbone** — each P warps the PREVIOUS I/P anchor's recon by its `source<0`
MVs; (2) **B-frames = bidirectional leaves** — warp nearest past anchor (`source<0`) AND nearest
future anchor (`source>0`), blend per-pixel (temporal-distance weighted; one-valid → use it;
neither → bicubic), and are **never used as a reference**. Added
`build_lr_flow(..., want="all"|"past"|"future")`. New metric **`psnr_lr_consistency`** =
PSNR(downscale(recon)→LR vs the true decoded LR). RESULT: B-frame fallback collapsed —
window A 42%→**0.96%** mean (97%→4.7% max), B 19%→**4.3%**, C (talking-head) 21%→**1.4%**.
Visual: the talking-head B-frame mask went from whole-subject-white to ~0.8% specks. Synthetic
regression byte-identical. CAVEAT: the remaining **P-frame fallback ~24% in window A is now
HONEST large-motion occlusion** (the ~6-frame P→anchor hop), not a bug.

**Step 3 — real SR network on the anchor.** `prototype/sr.py` = hand-written
**SRVGGNetCompact** loading `realesr-general-x4v3.pth` (strict=True, runs on **MPS**, weights
auto-download to `prototype/models/`). `--sr {bicubic,realesrgan}`, default bicubic (regression
preserved). It is an **x4** model → comparison runs at `--scale 4`. SR latency **~128–138
ms/frame on MPS** → per-frame SR is only **~7.3–7.8 fps** (the real-time bottleneck). Economic
thesis **VALIDATED on talking-head**: propagated-SR ≈ per-frame-SR (**45–46 dB** PSNR-vs-PF) at
~7% of SR compute = **~14× fewer SR calls**, and adds temporal stability. Visual crop confirms
propagated-SR edges are crisp = per-frame-SR, ≫ bicubic. Honest caveat: fallback pulls from the
full per-frame-SR (computed anyway), so the % is the **DEPLOYABLE amortized cost, not literal**.
Weak case: high-motion (24% P-fallback, only ~7× fewer).

**Step 4 — adaptive re-anchoring + quality-vs-anchor-fraction curve.** Code:
`reconstruct(frames, ..., anchor_set)`, `compute_anchor_set(...)`,
`--reanchor {none,interval:K,adaptive}`, `--quality-margin`. Sweep driver `anchor_sweep.py` →
`out_anchor_sweep/{anchor_curve.png, sweep_A.csv, sweep_C.csv, summary.txt}`; **SR is cached
once per window, the sweep is warp-only** (~2 SR passes total, not per operating point).
RESULTS: the **adaptive policy spends 3–4.5× more anchors on high-motion (A) than talking-head
(C) at the same budget** (budget 0.5: A=9/18.8% vs C=2/4.2%; budget 1.0: A=6/12.5% vs
C=2/4.2%) — the core behavior works. **Talking-head free-rides the encoder I-frames** (4.2%
anchors within ~0.8 dB of ceiling, tOF flat ~0.13, fallback ~3%, **~24× fewer SR calls**).
**High-motion genuinely needs anchors** (adaptive budget 1.0 → 12.5% anchors, fallback 11%→9%,
tOF 0.68→0.53, **~6× speedup**, knee around K=2–4).
**REAL-TIME:** propagation makes SR no longer the bottleneck (amortized **5–16 ms/frame** vs
~128 ms per-frame); the bottleneck **SHIFTS to warp+mask+decode, which was NOT profiled.**

**METHODOLOGICAL LESSON (record prominently):** **LR-consistency is too INSENSITIVE as a
decision metric** — fallback filled from per-frame-SR is trivially LR-consistent, so it looks
near-ceiling even when propagation isn't doing the work. **PSNR-vs-per-frame-SR is NOISY and
non-monotonic.** The **honest decision metrics are tOF + fallback%.**

**Step 6 — end-to-end profile + MPS GPU acceleration of warp+mask+blend.** First REAL per-frame
profile (talking-head window C, start 5000, 48 frames, 640×320→2560×1280 x4, realesrgan).
Big new finding the rough estimates missed: the **B-frame BLEND** (numpy boolean fancy-indexing
+ full-HD float casts) is the single largest numpy op at **~160 ms/B-frame**, bigger than warp
(25/39 ms P/B) or mask (47/73 ms). Real numpy deployable cost ≈ **252 ms/frame (4 fps)**, not the
~65–90 ms the pre-profile guess assumed. Added `gpu_ops.py` (torch/MPS twins: warp = `grid_sample`
border/align_corners=True; mask = `scatter_add` softmax-splat, float32) + `reconstruct_torch`
(recon chain stays RESIDENT on GPU across frames) behind **`--backend {numpy,torch}`** (default
numpy = the byte-identical regression guard) and a **`--occ reactive`** mask ablation. RESULT
(single-sync deployable wall-clock, same window): numpy-full **252 ms → torch-full 63 ms (3.99×,
16 fps)** → **torch + reactive mask 48 ms (21 fps)** → **torch + reactive + on-GPU-SR-output
~38–40 ms (25–26 fps) = real-time at the margin.** ≤40 ms (25 fps) is reached only in the most
favorable config; the **full fwd-bwd mask stays ~55–63 ms (16–18 fps)**. Per-op GPU gains: warp
39→13 ms, mask 73→20 ms, blend 163→5 ms (B-frame). Correctness: torch matches numpy within
tolerance — PSNR(torch,numpy) recon 45–68 dB (grid_sample is true-float-bilinear, so marginally
*better* than cv2.remap's 1/32-px fixed-point), fallback% delta ≤0.37 pts; synthetic numpy
default still byte-identical (41.08→31.45, 23/23, 5.84%, 0.091/0.106). Residual bottleneck = the
mask (even reactive ~6 ms numpy / ~17 ms GPU at the per-op-sync inflation) + host↔device
transfers (perframe upload ~9 ms is an experiment artifact removed by on-GPU SR; output download
~10 ms removable if rendering from the GPU texture). The **reactive-only ablation** (drop the
~29 ms fwd-bwd softmax-splat) costs ≈0 quality on talking-head (fallback 2.89%→2.22%, tOF 0.209→
0.228) and is the cheapest path to real-time. Artifacts: `prototype/out_profile/`
(`summary.txt`, `components.csv`, `quality.csv`), `prototype/profile_e2e.py`, `prototype/gpu_ops.py`.

**Step 7 — close the real-time gap (transfers + adaptive mask) + high-motion profile.** Three
changes, all behind the same `--backend torch` fast path (numpy default still byte-identical):
(1) **Removed the deployment-artifact transfers.** Added `download_output` to `reconstruct(_torch)`:
the DEPLOYABLE path keeps the HD recon RESIDENT on the GPU (rendered from a Metal texture, no
per-frame readback) and uses a GPU-resident SR/anchor output (no per-frame `upload_perframe`).
`profile_e2e.py` now reports a true **deployable** number (transfers excluded) AND a **with-I/O**
number (HD readback included) for honesty. Removing both transfers takes the full-mask path from
**~67 ms → ~42 ms/frame** (the HD download alone is ~8–25 ms, the perframe upload ~7–12 ms;
absolute values swing ±~25% with laptop thermal state, so report best-of-N + ratios).
(2) **Adaptive occlusion mask** (`--occ adaptive`, numpy+torch): always run the cheap reactive
mask, fire the costly fwd-bwd softmax-splat ONLY per-direction when the reactive-fallback fraction
> `ADAPTIVE_TAU` (tuned **0.06** via `tune_adaptive.py`). Mirrors adaptive anchoring. **fp16 was
measured and rejected** (microbench: only the compute-bound HD-warp `grid_sample` gains ~13%
6.25→5.42 ms; `scatter_add` — the splat core — is *slower* in fp16 0.19→0.36 ms, and the dominant
mask op is kernel-launch-bound so fp16 can't touch it → ≤2 ms/frame projected gain at fp16
precision risk; not worth it).
(3) **High-motion (window A, start 0) profiled** for the first time (Step 6 was talking-head only).
RESULTS (clean best-of-12, deployable ms/frame; fps): **talking-head C** full 42.4 (24) / reactive
**27.7 (36)** / **adaptive 39.5 (25)**; **high-motion A** full 40.1 (25) / reactive **32.6 (31)** /
**adaptive 38.3 (26)**. **WHICH MASK PER REGIME:** on talking-head the fwd-bwd mask does NOT earn
its cost — **reactive == full quality** (fallback −0.76 pt, tOF(prop/LR) −0.020, i.e. as-good-or-
better) → **use reactive** (36 fps, real-time). On high-motion **reactive genuinely LOSES**
(fallback −3.36 pt = 3.4% of bad-MV pixels left un-flagged, tOF +0.052) → need full; **adaptive
recovers full quality** (tOF 1.3996 vs full 1.3994, fallback −0.14 pt) at less cost. The
per-direction reactive trigger also trips on talking-head B-frames (far-future-anchor direction),
so a single global tau can't make C as cheap as pure reactive — adaptive lands between reactive and
full on C. So: **reactive for low-motion, full/adaptive for high-motion; adaptive is the safe
single auto-policy** (≤40 ms on BOTH windows, full-quality where it matters). **REAL-TIME VERDICT:**
the full-quality path now reaches ≤40 ms (25 fps) on BOTH windows via the adaptive mask + transfer
removal (C 39.5, A 38.3); reactive clears it with margin on both (27.7 / 32.6); the literal full
fwd-bwd mask sits right at the line (42.4 / 40.1, ~24 fps). Residual levers if more headroom is
needed: `build_flow` is ~5 ms/frame of CPU (numpy MV-scatter loop — vectorize/overlap with GPU),
and a cheaper fwd-bwd (nearest-splat reverse flow) would cut the splat further. torch matches numpy
within float tolerance (recon PSNR(torch,numpy) full 64.9–67.7 dB, adaptive 63.8–65.3 dB, fallback
delta ≤0.16 pt). numpy default still byte-identical (41.08→31.45, 23/23, 5.84%, 0.091/0.106).
Artifacts: `prototype/out_profile_{C,A}/`, `prototype/{profile_e2e.py,tune_adaptive.py,final_timing.py,gpu_ops.py}`.

**Step 8 — upscaling quality: heavy anchor + per-frame film grain + the detail-drift experiment.**
Three additions, all behind flags that default OFF / to the compact model (synthetic regression
stays byte-identical 41.08→31.45, 23/23, 5.84%, 0.091/0.106):
(1) **Heavy perceptual anchor** `--sr realesrgan-x4plus`: `sr.py` now has **`RRDBNet`** (23 RRDBs,
16.7M params) loading `RealESRGAN_x4plus.pth` (`params_ema`, strict=True, MPS) alongside the compact
`realesr-general-x4v3`. On an anchor crop it is **+61% sharper** (var-of-Laplacian 28.4 vs compact
17.6 vs bicubic 2.5) — visibly more texture. **MPS latency ~2.1–2.25 s/frame** (640×320→2560×1280)
vs compact ~0.13 s; affordable because SR runs only on sparse anchors → **amortized at 1 anchor/48
frames ≈ 45 ms/frame** (≈ the warp pipeline cost). `sr.py` API is now model-named
(`upscale(rgb, model=...)`, per-model latency); old `upscale(rgb)`/`load_model()` still default to
the compact net (back-compat).
(2) **Per-frame film grain** `--grain {off,low,med,high}` (`grain.py`): per OUTPUT frame, a
spatially-correlated Gaussian template re-seeded from the frame index, amplitude scaled by local
luma via a LUT, added to luma in gamma space as the FINAL pass — never warped/propagated/fed to
references. Verified temporally INDEPENDENT (raw grain-field corr frame-to-frame **+0.0012**, self
**+1.0000**), visible & filmic, luma-modulated (suppressed in shadow/highlight). The deterministic
per-frame seed is the whole trick: regenerate, never warp (warping would freeze grain onto content).
(3) **KEY EXPERIMENT — `detail_drift.py`:** does heavy-anchor detail SURVIVE MV propagation? On
clean single-anchor all-P chains, **pure propagation** (SR the anchor only; occlusion holes →
bicubic, NO fresh per-frame SR re-injection) measured sharpness vs distance-from-anchor for both
models. **Sharply motion-dependent half-life of the heavy advantage:** low-motion (talking-head)
**persists past 11 frames** (advantage +38→+30 var-Lap, warps are near-identity so nothing erodes);
high-motion **half-life ≈ 1 frame** (advantage +31→+3.7 by d2→~0 by d3; both models collapse toward
bicubic as warp-blur+occlusion destroy all SR detail). **VERDICT:** the heavy anchor is worth it for
low-motion/static content at a sparse interval (≤ ~half the GOP) but NOT for high-motion (would need
to anchor every 1–2 frames → defeats amortization). It helps the static parts of a frame, is wasted
on moving parts (where the codec already needs fallback = per-frame SR anyway). **Honest caveats:**
the deployable fallback re-injects fresh per-frame SR (21–40% of pixels on these stretches) which
sustains the advantage by *re-running SR*, not propagation; and the heavy anchor's extra hallucinated
texture is **less temporally stable** under warp on low motion (tOF 0.66 vs compact 0.33) — the
NR-sharpness-vs-temporal tradeoff the research warned about. Artifacts: `prototype/out_quality/`
(`detail_drift.png`+`.csv`, `ab_anchor_*`, `grain_{ab,consec,field}.png`), `prototype/{sr.py,
grain.py,ab_anchor.py,detail_drift.py,grain_demo.py}`, `models/RealESRGAN_x4plus.pth`.

**Step 9 — three parallel quality streams + a seam-verification/integration pass.** Each stream
was a NEW module importing the shared `derisk.py`/`sr.py` API READ-ONLY; a SEAM-VERIFICATION pass
then checked every cross-module call (name / arg-shape / return-shape) and INTEGRATED the viable one.
**Seam result: all interfaces matched — zero caller/handler mismatches** (the parallel-agent failure
mode did not occur here). region_quality→derisk (`build_lr_flow`(→2-tuple), `build_perframe_cache`,
`decode_lr_and_mvs`, `_farneback`, `reconstruct`(→`(rows,R)`)), anchor_pipeline→sr
(`upscale(model=)`, `load_model`, `*_latency_ms`), and sr_diffusion (standalone, mirrors
`sr.upscale`) all line up; all 3 import clean; `sr_diffusion` is import-safe with diffusion weights
absent and loud-fails only when actually called.
(1) **Stream-1 `region_quality.py` — INTEGRATED as `--region-aware`** (default OFF => byte-identical
regression). Acts on the Step-8 finding (heavy detail propagates on static content, erodes+flickers
on motion). Wired as an **OUTPUT-ONLY final pass, exactly like grain**: the propagation chain stays
single-model (heavy); the per-output-frame blend `out = a_hd*recon_heavy + (1-a_hd)*compact_source`
(a_hd = temporally-stable, widely-feathered motion gate, lo=0.2/hi=1.0/feather=61 LR px from the free
per-frame MV magnitude; compact_source = the per-frame compact SR) re-paints the OUTPUT copy only and
**NEVER enters the reference chain `R[]`**. On-GPU it is a single `torch.lerp` (~1–2 ms) in
`reconstruct_torch` (new optional `region_gate` arg on `reconstruct`/`reconstruct_torch`); the numpy
twin reuses `region_quality.blend_region_aware` (no math duplicated). RESULT (talking-head, start
5000, 48f, x4): region-aware **recovers 95% of the heavy STATIC-region detail** (var-Lap 118.4 vs
x4plus 119.5, compact 96.2) while **overall tOF stays at the compact floor** (0.211 vs compact 0.209,
far below x4plus 0.231) — static detail without the uniform-x4plus flicker. Matches
`region_quality.py` standalone `ra-wide` (static 118.4, recovers 95%); the standalone's lower overall
tOF (0.201) is from blending a fully-propagated compact chain vs the integration's cheaper per-frame
compact source (per the hook). torch==numpy within tolerance (region-aware recon PSNR 53.3 dB;
`torch.lerp` == numpy `blend_region_aware` 55.2 dB). Synthetic default-off byte-identical
(41.08→31.45, 23/23, 5.84%, 0.091/0.106). Artifacts: `region_quality.py`, `out_region/`,
`out_region_e2e/`.
(2) **Stream-2 `anchor_pipeline.py` — ANALYSIS ARTIFACT (pipelining = buffered-only).** Lookahead/
buffer scheduler model + threaded demo for the ~2.2 s x4plus anchor. VERDICT (single Apple-Silicon
GPU): you can **buffer the LATENCY** (B_min = ceil(L*F) ≈ L s of pre-rendered output) **but cannot
beat the THROUGHPUT ceiling** `r + L/K ≤ 1/F` — recon already eats most of the 40 ms budget, so live
x4plus at 25 fps is NOT feasible at a useful anchor interval on one GPU. Feasible only via a dedicated
anchor accelerator (2nd GPU/ANE), F=15 + reactive recon, or a cheaper heavy anchor. Standalone (no
derisk edits). Artifacts: `anchor_pipeline.py`, `out_pipeline/`.
(3) **Stream-3 `sr_diffusion.py` — ANALYSIS ARTIFACT (diffusion NO-GO on this box).** MPS feasibility
probe of the SD2/OSEDiff one-step UNet+VAE-decode per-tile graph (random weights → valid timing/op-
coverage/memory, no download) + an import-safe, **loud-on-failure** real-anchor hook (raises, never
silently no-ops). No real diffusion SR is wired on this box (basicsr + multi-GB gated weights + disk),
so `--sr diffusion` is NO-GO; the probe is retained for the record. Standalone. Artifacts:
`sr_diffusion.py`.

## Step 10 — LAYERED architecture ("video as a composition of layers") — IN PROGRESS

Motivated by the recurring finding that **everything splits by motion** (static = heavy detail
persists + propagation near-free; dynamic = detail erodes + needs fresh SR). Idea: decompose the
frame into a **static background layer** (SR'd ONCE per scene with a heavy model → amortized over
hundreds of frames, dissolving the live-heavy-anchor throughput ceiling) + a **dynamic foreground
layer** (~18% of frame, SR'd per-frame/anchor), then composite + grain. Research pass 5 (`wce7evoli`)
validated it for **static-camera talking-head** content; Wang&Adelson 1994 is the blueprint (background
= one extended plate the camera window slides over, built by temporal-median of motion-compensated
frames). Key correction: **codec-MV segmentation FAILS for near-still talking heads** (lip-only motion →
~zero MVs → misclassified as background) — so use a real matte (RVM); MVs are still free for global-
motion plate registration + scene-cut detection.

- **L1 (DONE) — `matting.py`:** Robust Video Matting on MPS via `torch.hub.load("PeterL1n/RobustVideoMatting","mobilenetv3")`.
  Fully native (zero CPU fallback), **~19–22 ms/frame** (LR→720p), clean & temporally stable matte
  (alpha MAD 0.0105), FG **~18% of frame**. API: `load_rvm`, `matte_sequence` (threads the recurrent
  state — feed frames IN ORDER), `fg_mask_lr(pha,lr_hw,soft,thresh,dilate)` → gate (1=FG, 0=BG). The
  KEY economic insight: in the layered render you DON'T run warp+mask propagation on the background, so
  recon shrinks to the ~18% FG → layered may be CHEAPER than full-frame propagation AND higher quality.
- **L2 (DONE) — `background_plate.py`:** static-camera verified (global MV = 0.000 px). Plate = per-pixel
  temporal-median of background-only pixels (FG masked, dilate 3 px). **Plates are PER-SCENE** — segment
  on codec I-frames (`scene_segments`); the canonical `start 5000` window has a hard cut at frame 32, so
  build on `[0,32)`. Coverage 75.2%, always-occluded hole 24.78% (behind subject, inpainted, never
  visible). Subject cleanly removed. Heavy-SR the plate ONCE (x4plus) = **86 ms/frame over 32 frames →
  ~9 ms/frame for a 300-frame shot** (amortization scales with scene length), bg sharpness 8.86× bicubic.
  API: `estimate_global_motion`, `scene_segments`, `build_plate(hole_fill="inpaint")`, `sr_plate`,
  `sample_plate(static=identity; global_motion hook for camera motion)`.
- **L3 (DONE) — `layered_pipeline.py`:** composite `out = alpha*fg_hd + (1-alpha)*plate_hd` + grain.
  **The background win is REAL & unique:** BG sharpness 15.3 vs x4plus ceiling 13.6 (**112%** — the
  temporal-median plate DENOISES before the single heavy SR, so it beats SR-ing one noisy frame) at
  **direct frame-to-frame flicker |ΔF| = 0.001 vs x4plus 0.167 (~167× steadier)** — neither uniform-x4plus
  nor region-aware achieves this. x4plus-bbox FG gets full subject detail (55.7 ≈ x4plus 57.1), so both
  layers are x4plus-sharp at once. **BUT IT IS NOT A SPEED WIN** (corrects the earlier "layered may be
  cheaper" hypothesis): the foreground is the MOVING subject → can't be propagated (Step-8 detail erodes
  in ~1 frame on motion) → must be SR'd FRESH every frame, which is the cost floor (~130 ms compact full-
  frame). The plate amortizes only the background (which was already cheap via warp). GPU-realistic layered
  compact-FG ≈ **~148 ms (6.8 fps), ~4–5× SLOWER than the 33 ms propagation pipeline.** The bbox trick
  barely helps a centered head (bbox = 50.6% of frame, not 18%; alpha-mass 27% but bounding box huge).
  Metric note: literal "tOF≈0" was REFUTED (tOF measures deviation from decoded-LR jitter; the fixed plate
  deviates from that real jitter, so tOF BG = 0.023 ≈ x4plus 0.025) — the spirit (zero flicker) is true via
  the direct |ΔF| metric, not tOF. FG tOF is WORSE (0.218 vs 0.074) — the moving alpha edge over a static
  plate + disocclusion ring is temporally noisier. Artifacts: `out_layered/`.
- **L4 (folded into L3's honesty section) — failure modes:** lighting/color across the seam is GOOD (plate
  = temporal median of the SAME footage). Halo REAL but subtle (faint hairline/jaw rim; FG/BG sharpness
  discontinuity 5.1–5.8× vs uniform-x4plus 3.4×). Hair bulk fine, fine wisps lose. WORST exactly at the
  disocclusion ring (low-coverage inpainted plate, sharpness 8.7 vs 15.3). Bounded to static-camera +
  human foreground + non-commercial matte.
- **VERDICT:** layered is a **quality/stability OPTION (kills background shimmer with a rock-stable,
  denoised, x4plus-sharp background)**, NOT a speed win and NOT the default real-time path. **Keep
  `--region-aware` propagation as the real-time path; reach for layered only when background shimmer is the
  specific defect to kill** and the ~4× cost + non-commercial matte + static-camera limits are acceptable.

## GOTCHAS (read before touching anything)

1. **System `ffmpeg` is BROKEN** on this machine: `dyld: Library not loaded:
   /opt/homebrew/opt/x265/lib/libx265.215.dylib` (homebrew x265 version mismatch). Do NOT rely
   on the `ffmpeg`/`ffprobe` CLI. **PyAV ships its own libav\*** and works fine — use it for
   decode/encode/MV extraction. (If you ever need the CLI: `brew reinstall x265 ffmpeg`.)

2. **MV extraction is via PyAV, not mvextractor.** Set `stream.codec_context.options =
   {"flags2": "+export_mvs"}` BEFORE decoding, then per frame
   `frame.side_data.get(av.sidedata.sidedata.Type.MOTION_VECTORS).to_ndarray()`.
   - `mvextractor` (LukasBommes) supports **H.264/MPEG-4 only, NOT HEVC**. PyAV exports HEVC
     MVs too, so PyAV is the better choice for the H.265 path.

3. **PyAV `frame.pict_type` is an int enum (1=I, 2=P, 3=B), has NO `.name`.** `getattr(...,
   "name", ...)` silently returns the number → map manually `{1:"I",2:"P",3:"B"}`. (This bit us:
   `ptype == "I"` never matched; only saved by the `recon_prev is None` first-frame fallback.)

4. **MV field semantics (verified empirically):** structured array fields
   `source, w, h, src_x, src_y, dst_x, dst_y, flags, motion_x, motion_y, motion_scale`.
   - `dst_x/dst_y` = block **CENTER** (not top-left). Block spans `[dst-w//2, dst+w//2)`.
   - Sub-pixel source: `src_x = dst_x + motion_x/motion_scale`. `motion_scale=4` ⇒ quarter-pel.
   - `source < 0` = past reference, `> 0` = future (display order). I-frames return shape (0,N).
   - The integer `src_x/src_y` fields are rounded — **recover sub-pixel from
     `motion_x/motion_scale`, not from `src_x`.**

5. **Codec MVs are backward (dst-indexed), so BACKWARD warp (`cv2.remap`) is the natural op** —
   each dst pixel gathers one source, no collisions, no holes except where there's no MV. Don't
   force forward splatting for the main warp. (Forward/softmax splatting is used only to build a
   reverse flow field for the occlusion check — see #7.)

6. **Codec MVs are rate-distortion-optimized, NOT true optical flow.** They are block-level and
   sometimes point to "visually similar but geometrically wrong" content (minimizes residual,
   not motion error). Backward-warping those produces garbage → must be detected (see #7). Median
   EPE ~0.36px (AV1 study); good enough as a proxy, NOT pixel-exact.

7. **Occlusion handling is what makes or breaks quality.** Three cheap signals at LR, unioned
   into one "unreliable pixel" mask; fall back to per-frame SR there (the Ruder `c=0` recipe):
   - **intra holes** — pixels with no MV (NaN flow) = true disocclusion.
   - **reactive** — high prediction residual `|LR_cur − warp(LR_prev)|` = bad/occluded MV.
   - **fwd-bwd consistency** — build a forward flow by **softmax-splatting** the backward MVs
     (collisions won by lower-residual matches), then **Ruder et al. 2016** test
     `|w̃+ŵ|² > 0.01(|w̃|²+|ŵ|²)+0.5`. Threshold is motion-adaptive (don't use a fixed px
     threshold — it won't generalize across motion magnitudes).
   - Detecting ~3% extra unreliable pixels flips "wins 7 frames" → "wins all 23". Huge lever.

8. **FSR2/DLSS depth-based disocclusion does NOT transfer** — decoded video has no depth buffer.
   Only the **RGB-neighborhood color-box clamp** and luma-only lock/shading signals transfer.
   This is why depth (e.g. Depth Pro) was considered — see #10.

9. **Synthetic content matters.** Pure random-noise frames are a pathological SR case (bicubic
   can't recover noise → everything ~19 dB vs true, drowns the warp signal). `make_synthetic`
   uses **structured** content (gradients + sharp shapes + a thin-line resolution band + a moving
   occluder) so PSNR-vs-true is meaningful. Don't revert to noise.

10. **Oracle anchor at frame 0 gives ~inf PSNR** (oracle==true) → it blows up plot y-axis;
    `_plots` clips ylim to the meaningful range. Keep that.

11. **tOF is computed at LR** (frames resized to 640×360) for speed — it's a relative comparison,
    so scale doesn't change the verdict. Full-HD Farneback ×24×2 would be slow.

12. **`source` is a REFERENCE-LIST INDEX, not a display-order distance** (Step 1). On
    `sample.mp4` `|source|` is only ±1 even though a P's true reference is ~6 display-frames
    back. `source<0` = a past ref-list entry, `source>0` = a future one (B-frames carry both,
    future-dominant). NEVER chain reconstruction in display order once B-frames appear, and
    NEVER divide MVs by `|n−m|` using `source` as the distance — walk the **I/P backbone**
    instead (`backbone_indices` / `reconstruct`), warp a P from its previous I/P anchor, and
    treat B-frames as bidirectional leaves (`build_lr_flow(..., want="past"|"future")`).

13. **Metric trap — pick the right decision metric** (Step 4). **`psnr_lr_consistency` is too
    INSENSITIVE**: the bicubic/per-frame-SR fallback is trivially LR-consistent, so the metric
    sits near the per-frame-SR ceiling even when propagation has given up. **`psnr_prop_vs_perframe`
    is NOISY/non-monotonic AND misleading on real footage** (high can mean "dumped to bicubic").
    For any re-anchoring or quality decision use **tOF + fallback%** — those are the honest signals.

14. **Film grain MUST be regenerated per output frame, never warped** (Step 8). The whole point is
    temporal INDEPENDENCE: seed the grain from the frame index (deterministic but different every
    frame) and add it as the FINAL pass to a COPY of the output. If you ever warp/propagate grain
    or feed a grained frame into the anchor/reference/propagation chain, the codec MVs freeze the
    grain onto moving content (it stops looking like grain). Measure independence on the RAW
    additive grain field, NOT on `Y(grained)−Y(recon)` — the RGB↔YCrCb round-trip's quantization is
    content-dependent and ~identical between consecutive (near-duplicate) frames, which spuriously
    inflates the measured correlation to ~0.2 even though the real grain is uncorrelated (~0.001).

15. **Heavy-anchor detail survival is MOTION-DEPENDENT** (Step 8). To measure pure warp erosion you
    MUST disable fresh per-frame-SR re-injection (SR the anchor only; fill occlusion holes with
    bicubic) — otherwise the deployable fallback (21–40% of pixels on these stretches) re-paints
    fresh SR every frame and the "advantage persists" for the wrong reason (re-running SR, not
    propagation). With that isolated: low-motion content keeps the heavy advantage for the whole GOP
    (warps near-identity), high-motion loses it in ~1 frame (warp-blur + occlusion → bicubic). Pick
    the all-P chain by STARTING ON the I-frame (`--start-frame` exactly on an I) so the only anchor
    is dist 0 — `start 5031` gave `PIPPP…` (a 2nd anchor at the I), `start 5032` gives clean `IPPP…`.

16. **`--region-aware` is an OUTPUT-ONLY pass — never feed the blend into `R[]`** (Step 9). Like
    grain, the region-aware blend `a_hd*recon_heavy + (1-a_hd)*compact` is applied AFTER both
    reconstruction passes complete (the heavy recon's reference role is over), into the OUTPUT copy
    only. If you blend it into `R[i]["recon"]` *before* a later frame warps from it, the
    compact/heavy seam propagates and the gate stops being a pure per-output detail policy. In
    `reconstruct_torch` the lerp writes a FRESH tensor, leaving the resident GPU reference untouched.
    Also: `--region-aware` REQUIRES `--sr realesrgan-x4plus` (the propagation chain is the single
    heavy model; the compact source is a separate per-frame compact SR). Default OFF = byte-identical.

17. **RVM matte: non-commercial license + recurrent state** (L1). RVM is **CC BY-NC-SA 4.0** —
    fine for this prototype/research, but a COMMERCIAL product cannot ship it; swap for a
    differently-licensed matte (re-trained RVM, MediaPipe Selfie Segmentation, etc.). Also RVM is
    RECURRENT (ConvGRU) — you MUST thread its state (`rec=[None]*4`) across frames in display order
    (`matte_sequence` does this); calling it per-frame statelessly loses the temporal coherence that
    keeps the matte edge from crawling. Human-only; it can drop a non-human/empty frame (e.g. a title
    card), which the plate's temporal median tolerates as a minority but a mostly-mis-matted scene
    would corrupt the plate.

18. **Background plates are PER-SCENE — segment on codec I-frames** (L2). A plate accumulates one
    static shot; never let frames from across a cut into the same median (the canonical `start 5000`
    window spans a hard cut at frame 32 → a mid-stream I-frame; the naive full-48 plate reports a
    bogus 0% hole because the title-card frames fill the occluded region). Use `scene_segments` /
    `find_scene_cuts` (codec I-frames + an RGB-diff spike) and build per segment.

19. **The plate assumes a STATIC camera** (L2). Verified here (global MV = 0.000 px → `sample_plate`
    is identity). Any pan/zoom breaks the identity plate: the `global_motion` hook handles translation,
    but real camera motion (parallax/zoom) needs per-frame homography warping of the plate (cheap from
    block MVs, ~6–18 ms/field per the research) — NOT implemented in v1. The always-occluded hole
    (~25%) is INPAINTED (a guess) — safe only because the subject always covers it; if the matte ever
    under-covers, the guessed pixels show.

## Known limitations / done since (Steps 1–4) and still NOT done

- ✅ **DONE — real SR network** (Step 3): `prototype/sr.py` SRVGGNetCompact /
  `realesr-general-x4v3` (1.21M params, x4, MPS), `--sr realesrgan --scale 4`. Latency ~128–138
  ms/frame is the per-frame bottleneck the propagation removes.
- ✅ **DONE — real footage** (Steps 1–4): `sample.mp4` windows A/B/C. The mask and drift behave;
  high-motion stresses the P-frame fallback (~24%, honest occlusion), talking-head is clean.
- ✅ **DONE — B-frames** (Step 2): backbone (I/P chain) + bidirectional B leaves; `source` is a
  ref-list index, not a distance (no `÷|n−m|` — see GOTCHA #12). Fallback collapsed to ~1–4%.
- ✅ **DONE — adaptive re-anchoring** (Step 4): `--reanchor {none,interval:K,adaptive}` +
  `--quality-margin`/`--fallback-budget`; quality-vs-anchor-fraction curve in `out_anchor_sweep/`.
  Adaptive spends 3–4.5× more anchors on high-motion than talking-head at the same budget.
- **No color-box clamping yet** — the one transferable FSR2 technique still to add (`--clamp`).
  CAVEAT: FSR assumes history≈current resolution; here the warped anchor is *more* detailed than
  the current LR, so clamping to the current-frame box risks clipping the SR detail you want.
  Needs a loose box (mean ± large γ) + empirical tuning; may help on real ghosting, hurt on clean
  synthetic. Test, don't assume.
- ✅ **DONE — end-to-end fps profile + MPS acceleration** (Step 6): warp/mask/**blend** profiled
  (`prototype/profile_e2e.py`, `out_profile/`); the B-frame blend was the hidden ~160 ms numpy
  cost. GPU-ported via `gpu_ops.py` + `reconstruct_torch` behind `--backend torch` →
  **252 ms→48–63 ms/frame (4→16–21 fps)**, real-time (~38–40 ms, 25 fps) reached at the margin
  with `--occ reactive` + on-GPU SR output. Full fwd-bwd mask stays 16–18 fps.
- **No VMAF** — would need the (broken) ffmpeg CLI `[distorted][reference]libvmaf=...` with the
  distorted upscaled to ref res via bicubic first. Note VMAF is compression-trained, not SR —
  report tOF/tLP alongside.

## Key research findings (3 passes, all adversarially verified)

- **NEMO** (https://chaos5958.github.io/assets/pdf/3372224.3419185.pdf, code
  https://github.com/kaist-ina/nemo): SR on 1.79–9.74% of frames, 45–120 fps 240p→1080p on a
  2019 Snapdragon 855, within 0.41 dB of per-frame SR. Drift ("cache erosion") → ~3 dB loss by
  frame #25 if you anchor only on key frames; solved by greedy quality-bounded anchor placement.
- **NAS** (OSDI 2018) does NOT use MVs (it's per-video overfitted per-frame SR) — don't conflate.
- **Anchor net**: video-SR (BasicVSR++ 77ms/frame, EDVR 378ms) = quality ceiling, too slow.
  Compact single-image SR is real-time: **ABPN** 640×360→1080p in 36.89ms (~27fps) on edge NPU
  INT8; Real-ESRGAN-general-x4v3 1.21M params. INT8 quantization is a severe pitfall (most models
  corrupt; quantization-aware design mandatory).
- **MV extraction**: FFmpeg `flags2=+export_mvs` / PyAV (gotchas #2–#4). REFUTED claim: "per-frame
  MVs too noisy to warp without accumulation" → single-frame MVs ARE usable for warping.
- **Occlusion**: Ruder 2016 fwd-bwd test (https://arxiv.org/pdf/1604.08610) + color-box clamp
  from FSR2 (https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-temporal/).
- **Metrics**: tOF/tLP from TecoGAN (https://github.com/skycrapers/TecoGAN-PyTorch — note tLP is
  only in the original https://github.com/thunil/TecoGAN), VMAF via Netflix/vmaf ffmpeg.md.
- **Baselines are ALL pure per-frame spatial SR** (NVIDIA RTX VSR, animejanai/Real-ESRGAN,
  Anime4K) — none exploit codec MVs for temporal propagation. **That is this project's novelty.**
- **Apple ml-depth-pro**: single-image metric depth, 2.25MP in 0.3s (~3fps, NOT real-time
  per-frame), no temporal, restrictive Apple research license. Verdict: anchor-only auxiliary /
  offline depth+occlusion oracle, not a live dependency. For live, a lighter model (Depth
  Anything V2-Small). Our cheap fwd-bwd+reactive mask already captures most of depth's benefit.

## Still open (would need more research / experiments)

- DLSS/XeSS history-rectification internals (only FSR2 verified).
- Concrete browser/WebGPU fps for real-time video SR (repos to benchmark:
  https://github.com/xororz/web-realesrgan, https://github.com/Anime4KWebBoost/Anime4K-WebGPU,
  https://github.com/sb2702/websr) — relevant if the target platform is the browser.
- ✅ Quality-vs-anchor-fraction tradeoff curve methodology — DONE (Step 4, `anchor_sweep.py` →
  `out_anchor_sweep/anchor_curve.png`). Lesson: use tOF + fallback% as the curve's decision axis,
  not LR-consistency (insensitive) or PSNR-vs-per-frame-SR (noisy).
- Codec residual export for the SR domain on H.264/H.265 (we self-compute the residual from LR
  frames; NEMO pulled it from the codec internals).
- Ruder threshold retuning for lossy block-level codec MVs (constants tuned for dense flow).

## Recommended next step

Steps 1–9 confirmed the architecture on real H.264 AND reached real-time on Apple Silicon. The
active frontier is the **layered architecture** (Step 10) — the most promising direction because it
could deliver quality AND speed at once:

1. **Finish the layered track (L3 → L4).** L3 measures the composite (does the fixed plate give
   x4plus background sharpness at ~zero flicker, and is it cheaper than full-frame propagation?);
   L4 characterizes failure modes (hair/seams, camera motion, the hole, multi-object). If it holds,
   build the streaming layered pipeline (matte at a slow refresh + MV-propagated gate; plate per
   scene; foreground per-anchor) and add the `global_motion` homography path for non-static cameras.
2. **Improve the high-motion regime** (the standing weak spot). Window A pays ~24% honest P-frame
   occlusion fallback, needs ~12.5% anchors at budget 1.0. Levers: better masks, shorter P→anchor
   hops, adaptive triggering keyed on tOF + fallback% (the honest metrics — NOT LR-consistency).
3. **Productionization gaps:** a commercially-licensed matte (RVM is non-commercial); the TAESD
   path to make a diffusion anchor affordable (VAE decode is the bottleneck, not MPS); color-box
   clamping (`--clamp`, untested); a WebGPU / lighter-SR (ABPN) path if the target is the browser
   (web-realesrgan / Anime4K-WebGPU / websr — links in "Still open").
