# playhd — handoff

Real-time SD→FullHD video upscaling "on the fly". Status as of 2026-06-19.

## TL;DR

**PRODUCT TARGET (clarified 2026-06-19): a web app.** Upload / pick an mp4 → **"Process & Play"** →
a **QUALITY toggle**, and watch the upscaled result *with sound*. Underneath: upscale low-res video
to HD by running an **expensive neural SR network only on sparse anchor/keyframes**, then
reconstructing every other frame cheaply by **warping the super-resolved anchor with codec motion
vectors + residuals** (the **NEMO**, MobiCom 2020, architecture, ported from VP9 to **H.264/H.265**).
**Three product modes:** **instant** (720p tier, compact anchor-**only** SR + reactive mask + GPU
grain — **now ~41 ms/frame ≈ 24 fps = real-time** after the improvement loop, see
"## Improvement loop"), **quality** (full-QHD x4plus + region-aware + grain, ~2.9 s/frame), **layered**
(two-pass-per-scene static-bg plate, ~0.47 s/frame, ~167× steadier background — now actually visible).
**Staging: server-side NOW** (wrap the validated Python/MPS prototype), **browser-only (WebGPU) LATER.**

**HARD CONSTRAINT (honest):** the **quality / layered** paths are still **~10× slower than real-time**
(a long render); **instant is no longer** — the improvement loop took it to ~24 fps at 720p, so an
instant clip now processes about as fast as it plays. Long clips on the slow modes still want
**progressive play-while-processing** (HLS/fMP4) + a background-render mode — the next UX stage (and now
unblocked for instant, which keeps up with playback).

**Current state:** the architecture is validated end-to-end on real H.264 and **GPU-accelerated to
real-time on Apple Silicon**; the **layered track is done**; a **Stage-1 product server** (FastAPI
+ a knob-free browser console, 3 quality modes, streaming whole-clip processing + in-sync audio) is
built on top; and a **6-iteration "improvement loop"** then hardened it — **scene-cut detection**, the
**layered grain-gate** (its stable background finally visible), and a perf push that took **instant to
10× / ~24 fps = real-time at a 720p tier** (quality + layered stay full QHD). **5 deep-research passes**
done; **Steps 1–10 complete** (layered L1–L4); **server built (`5dcf75f` + `c45e4c8`)**; **improvement
loop (`397f461`..`d90055e`).** Under **git** (init 2026-06-19; commits `7520d7e` Steps 1–8, `ee53c13`
Step 9, `9ee519e` Step 10/layered, `5dcf75f` Stage-1 server, `c45e4c8` server streaming rebuild +
layered mode + crash fixes + long-video guard, then the loop: `397f461` scene-cut detection + docs +
backlog, `0e97c9e` V1 layered grain-gate, `10de07e` instant real-time stack (2.8×), `5e83ec1` instant
117→91 ms (3.75×), `2b46fc6` instant 720p tier, `d90055e` instant 10× / 24 fps; `.gitignore` excludes
`sample.mp4`/`models/`/`out*/`/`server/{outputs,uploads,testdata}/`).

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
- **Product layer — Stage-1 server (web app), commits `5dcf75f` / `c45e4c8`:** a FastAPI app
  (`server/`) wraps the prototype so you upload/pick an mp4, click **Process & Play**, and get the
  upscaled clip **with in-sync source audio**, knob-free, in **3 modes** (instant / quality / layered).
  Processes the **WHOLE clip** in **constant memory** by streaming GOP-sized chunks + encoding
  incrementally, then muxes audio with **`+faststart`** (browser plays progressively). See
  **"## Product layer — Stage-1 server"** below.

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
prototype/layered_pipeline.py # LAYERED L3: composite plate + per-frame FG + grain
prototype/demo_matting.py / demo_background_plate.py  # layered demos
prototype/README.md    # how the prototype works + result tables + ARTIFACTS index
prototype/models/      # auto-downloaded SR weights (realesr-general-x4v3.pth, RealESRGAN_x4plus.pth) + RVM via torch.hub cache
prototype/out_quality/ # Step 8 artifacts: detail_drift.png/.csv, ab_anchor_*, grain_{ab,consec,field}.png
prototype/out_matting/ out_plate/ out_layered/  # LAYERED L1/L2/L3 artifacts
prototype/out*/        # generated artifacts (see prototype/README.md ## Artifacts) — all gitignored
server/                # PRODUCT LAYER — Stage-1 web app (FastAPI + browser console)
server/app.py          #   FastAPI: GET / (console) + /api/sources, /api/process, /api/progress, /outputs/*.mp4, + R1: /api/stream (progressive), /api/upload
server/progressive.py  #   R1 E1: fragmented-MP4 play-while-process core (GET /api/stream) — server-verified, browser-pending (opt-in)
server/pipeline_api.py #   STREAMING constant-memory GOP-chunk processing of the WHOLE clip + source-audio mux + faststart (instant/quality); instant takes the Levers-1–4 fast path
server/layered_api.py  #   the LAYERED mode (two-pass-per-scene: build+heavy-SR one bg plate/scene, composite the moving FG per frame)
server/scene_detect.py #   IMPROVEMENT-LOOP iter1: robust scene-CUT detector (luma-diff + I-frame-corroborated + relative/hysteresis + min-scene-len). ONE StreamingCutDetector shared by stream_gops + layered segment_scenes
server/anchor_sr.py    #   IMPROVEMENT-LOOP Lever 1: anchor-only SR cache + adaptive catastrophic-fallback safeguard (SR only anchors + >thresh frames; bicubic the rest)
server/fast_grain.py   #   IMPROVEMENT-LOOP Lever 2: GPU/MPS film-grain twin of prototype/grain.py (runs on the GPU-resident HD recon, ~few ms/frame) + a contiguous-HWC download
server/pipe_encode.py  #   IMPROVEMENT-LOOP Lever 2: ThreadedEncoder (overlap VideoToolbox encode w/ GPU) + prefetch_chunks (decode next GOP while GPU works)
server/bench_instant.py#   IMPROVEMENT-LOOP before/after benchmark for the instant speedup (per-component ms/frame + quality parity checks)
server/index.html      #   knob-free UI: pick/upload mp4 → mode → Process & Play → progress + duration/ETA guard → player (with sound)
server/{uploads,testdata,outputs}/  # uploaded sources / test clips / produced mp4s (all gitignored)
IMPROVEMENTS.md        # prioritized perf/visual backlog (the improvement loop worked this list; completed items marked DONE)
handoff.md             # this file
```
**Under git** since 2026-06-19 (`master`): `7520d7e` Steps 1–8, `ee53c13` Step 9, `9ee519e`
Step 10 (layered), `5dcf75f` Stage-1 server, `c45e4c8` server streaming rebuild + layered mode +
crash fixes + long-video guard, then the **improvement loop** `397f461` scene-cut detection + docs +
backlog, `0e97c9e` V1 layered grain-gate, `10de07e` instant real-time stack (2.8×), `5e83ec1` instant
117→91 ms (3.75×), `2b46fc6` instant 720p tier, `d90055e` instant 10× / 24 fps. `.gitignore` excludes
`sample.mp4` (78 MB), `prototype/models/` (weights), `out*/` (~340 MB),
`server/{outputs,uploads,testdata}/`, `__pycache__`, logs.
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

## Step 10 — LAYERED architecture ("video as a composition of layers") — DONE (commit `9ee519e`)

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

## Product layer — Stage-1 server (web app) — DONE (commits `5dcf75f`, `c45e4c8`)

The product target is a **web app**: upload or pick an mp4 → **Process & Play** → a QUALITY toggle,
and watch the upscaled clip **with its original sound**, knob-free, any length. **Stage 1 (now) is
server-side** — a FastAPI app wraps the validated Python/MPS prototype unchanged and streams the
result to a minimal browser player. **Stage 2 (later, once tuned) is a browser-only WebGPU/ORT-Web
port** — deferred because the make-or-break unknown there is codec MVs in the browser (WebCodecs
doesn't expose them); server-side extracts MVs via PyAV exactly as the prototype does, so there is no
port and no browser-MV problem yet.

**Run:**
```bash
cd /Users/lifeart/Repos/playhd
python3 -m uvicorn server.app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/
```

**Files (`server/`):**
- **`app.py`** — FastAPI. `GET /` serves the console; `GET /api/sources` returns sources + the 3 modes
  + each source's duration & per-mode processing-time estimate; `POST /api/process` (`mode` + `source`
  or an uploaded `file`) runs the whole-clip pipeline OFF the event loop (`run_in_threadpool`) and
  returns `{url, source, stats}`; `GET /api/progress` (live `{state, done, total, elapsed_s, eta_s,
  ms_per_frame}`) is served concurrently while a job runs; `/outputs/*.mp4` via StaticFiles (HTTP Range
  → `<video>` can seek). **One job at a time** → 409 if busy. Failures are surfaced to the UI (500 with
  the real error), never swallowed.
- **`pipeline_api.py`** — the **STREAMING, constant-memory** core for instant/quality. Opens the input
  container ONCE, yields **self-contained GOP chunks** (new chunk at every I-frame; a GOP longer than
  `SOFT_CAP_FRAMES=48` is also cut at the next P → a forced fresh anchor, NEMO-style), runs the
  prototype's `build_perframe_cache` + `reconstruct` (+ `_build_region_gate` for quality) per chunk,
  applies grain (global frame-index seed → temporally independent across chunks), and **encodes each
  chunk into the output H.264 stream incrementally** (libx264 crf 18, yuv420p). Never more than one
  chunk of HD frames is alive → peak memory is bounded regardless of clip length. Then **muxes the
  SOURCE AUDIO** in, in sync (copy if AAC, else transcode to AAC; a video-only source doesn't crash),
  and writes the final mp4 with **`+faststart`** (moov atom at the front → the browser starts
  progressive playback instead of stalling until the whole file is fetched).
- **`layered_api.py`** — the **LAYERED** mode (two pass per scene, bounded memory). PASS 0 segments the
  clip into scenes (one lightweight decode; a real cut = big RGB jump OR an I-frame + smaller jump, with
  a `MIN_SCENE_LEN` filter so periodic keyframes / short GOPs don't spawn false scenes). PASS A per
  scene: matte a capped, evenly-sampled subset (RVM, `PLATE_SAMPLE_CAP=64`), run the static-camera
  check, build the temporal-median background plate, **heavy-SR it ONCE (x4plus)**, and **spill the HD
  plate to disk**; a MOVING-camera scene is flagged → it falls back to the quality (region-aware) path
  (a fixed plate would be wrong). PASS B (driven by the streaming GOP loop): per frame composite
  `alpha*compact_fg_hd + (1-alpha)*plate_hd` + grain (RVM recurrent state threaded per scene; one HD
  plate held at a time). Reuses `matting.py`/`background_plate.py`/`layered_pipeline.py` READ-ONLY.
- **`index.html`** — knob-free console: source dropdown (+ size/duration) or upload, the 3 mode cards,
  a **duration/estimate guard** (shows source length + estimated processing time for the chosen mode and
  warns ⚠ before a >3-min render, since the pipeline is ~10× slower than real-time), Process & Play, a
  live progress bar (polls `/api/progress` every 400 ms), then the `<video>` player (autoplay **with
  sound**) + a server-timing stats readout.

**The 3 modes** (mode → the exact prototype flag combo the handoff recommends for that regime):

| mode | config | ~speed (server, ms/frame) |
|---|---|---|
| **instant** | 720p tier (`INSTANT_SCALE=2`), compact `realesr-general-x4v3` **anchor-only** SR, `--backend torch`, `--occ reactive`, GPU grain, HW encode | **~41 ms/frame ≈ 24 fps** (post-loop; was ~0.4 s) |
| **quality** | heavy `RealESRGAN_x4plus` anchor, `--region-aware` blend, `--occ adaptive`, grain | ~2.9 s/frame (2900) |
| **layered** | static-bg plate heavy-SR'd ONCE/scene + per-frame composited moving FG + grain | ~0.47 s/frame (470) |

> **NOTE (post-improvement-loop):** the instant row above reflects the loop's rewrite — see
> **"## Improvement loop"** below. **Quality + layered output full QHD x4** (SCALE=4; 640×320 →
> 2560×1280); **instant now outputs a 720p tier** (`INSTANT_SCALE=2`, x2 = 1280×640) but the SR net is
> still the x4 net + downscale (a *sharp* 720p, not a native 2× net). The server's UI processing-time
> estimate for instant was lowered 400 → 130 ms/frame (a conservative end-to-end figure; the deployable
> hot loop is ~41 ms).

Layered uniquely delivers a **rock-stable, denoised, x4plus-sharp background** (direct frame-to-frame
flicker |ΔF| ≈ 0.001 vs per-frame x4plus 0.167, **~167× steadier** — see Step 10, and now actually
visible after the V1 grain-gate); it needs a roughly static camera + a human subject and uses the
**non-commercial** RVM matte (CC BY-NC-SA).

**Two crash fixes baked into the streaming rebuild (`c45e4c8`):**
1. **Windowed-in-memory OOM / `avcodec_open2` EAGAIN.** The first server (`5dcf75f`) held a whole
   window at HD and opened the codec on a window so large it OOM'd (failed around n=5000). FIXED by the
   GOP-chunk streaming above (open the container once; one bounded chunk alive at a time; encode
   incrementally).
2. **MPS-allocator creep / frame-630 `BlockingIOError`/EAGAIN.** The MPS caching allocator's
   freed-but-cached memory crept up over a long clip (NOT bounded by per-chunk `del`s) and eventually
   failed an allocation under memory pressure. FIXED by `_free_gpu()` (`gc.collect()` +
   `torch.mps.empty_cache()`) once per processed chunk — returns cached memory to the OS, leaves active
   tensors untouched, ~ms cost.

**Long-video guard:** `/api/sources` returns each source's duration + a per-mode processing-time
estimate (`MODE_MS_PER_FRAME` instant **130** (lowered from 400 by the improvement loop) / quality 2900 /
layered 470 ms·frame⁻¹); the UI shows it and warns before a multi-hour render. This is the honest surfacing of the ~10× hard constraint BEFORE the
user blind-launches a multi-hour job on a 34-min source.

## Improvement loop — DONE (commits `397f461`..`d90055e`)

After the Stage-1 server shipped, a **six-iteration improvement loop** worked the `IMPROVEMENTS.md`
backlog (a read-only "poke" of the shipped outputs that surfaced two headline defects: instant ran
full per-frame SR; per-frame grain erased the layered mode's stable-background win). Every new module
lives in `server/` and imports the prototype **READ-ONLY**; the prototype's synthetic regression stayed
intact throughout (41.08→31.45 dB, 23/23). **Headline: instant mode now hits 10× = ~24 fps = real-time**
(at a 720p tier); **quality + layered stay full QHD and unchanged.**

### iter 1 — scene-cut detection (`server/scene_detect.py`, commit `397f461`)

The propagation pipeline reconstructs every non-anchor frame by **warping** a super-resolved reference
with codec MVs — valid only WITHIN one scene. `stream_gops` already cut a fresh chunk (a forced fresh
anchor) at every codec **I-frame**, but a real content cut that the encoder did NOT mark with an I-frame
(two segments spliced + re-encoded so the cut lands mid-GOP, or B-leaves straddling the cut) left a
chunk **spanning the cut** → `derisk.reconstruct` warped the pre-cut anchor across it = a visible
**cross-cut smear**. The new detector adds the missing signal: per-frame mean **|Δluma|** between
consecutive display frames, fired by ANY of (A) **absolute** `d > CUT_THRESH` (60, any frame type), (B)
**I-frame-corroborated** `ptype==I and d > IFRAME_THRESH` (45 — a *periodic* keyframe with small `d` is
NOT a cut, which is the whole reason I-frames alone are insufficient), or (C) **relative/hysteresis**
`d > REL_FLOOR (40) and d > REL_MULT (8×) the EMA motion baseline` (the adaptive baseline rises during
sustained fast motion so a motion BURST does not fragment into tiny scenes, and a 1-frame flash that
returns to baseline does not fire), then a **min-scene-length** (24) greedy filter. On `sample.mp4`'s
`[0,900)` window it scores **1.00 / 1.00 precision/recall** (true cuts `{28,196,341,479,514,563,630,688,810}`
caught; the periodic keyframes and a transient flash correctly NOT fired). Wired into `stream_gops` so a
chunk boundary lands at **every detected cut**, not just every I-frame → reconstruction never warps
across a cut. **Unified as ONE `StreamingCutDetector`** shared by streaming (`stream_gops`) and the
layered batch path (`segment_scenes` / `find_cuts`) — a single source of truth, no extra decode/GPU.

### iter 2 (V1) — layered grain-gate (commit `0e97c9e`)

The layered mode's *only* advantage over quality (it is ~4–5× slower and uses a non-commercial matte) is
its **rock-stable, denoised, x4plus-sharp background plate** (prototype |ΔF| = 0.001). But the shipped
server re-grained the **whole** frame per output frame — including the static plate — so the shipped
background flickered at **|ΔF| ≈ 4.5/frame**, identical to instant/quality: the stability win was 100%
invisible. **Fix:** bake **ONE fixed grain** into each scene's static plate once (seed
`_PLATE_GRAIN_SEED=12345` → filmic texture, but temporally stable), and apply **per-frame grain only to
the moving foreground**, gated by the RVM alpha. Result: background **|ΔF| 4.5 → 0.000** (rock-stable),
the foreground keeps fresh per-frame grain → the **~167× steadier background is finally visible** to the
viewer instead of being masked by full-frame grain. Layered is no longer strictly dominated by quality.

### iters 3–6 — instant mode → 10× / real-time (commits `10de07e`, `5e83ec1`, `2b46fc6`, `d90055e`)

A four-step perf push took the instant server path from **~409 → 41 ms/frame = 10.0× = ~24 fps =
real-time** (best-of-N steady-state; absolute ms swing ±~25% with laptop thermal state, so the ratios
are the robust takeaway — `server/bench_instant.py` is the contention-free before/after harness). The
per-iter levers + cumulative speedup:

1. **Real-time stack (341→117 ms, 2.8×, `10de07e`).** **Anchor-only SR** (`server/anchor_sr.py`,
   Lever 1): `build_anchor_cache` runs the SR net only on the **anchors** (every I-frame + the chunk's
   first backbone frame — the only frames `reconstruct` reads the full SR image from) plus any
   catastrophic-fallback frame; **bicubic-upscale everything else** (consumed only at the small
   occlusion-fallback fraction). The propagated (warped) pixels trace back to the same SR'd anchors → are
   **unchanged**; only fallback pixels differ (bicubic vs compact-SR). The per-frame fallback fraction
   (`hole_frac`) is anchor-invariant, so one reconstruct pass returns the exact fraction for free → the
   adaptive safeguard (`patch_high_fallback`) SR-patches only the frames that need it. Plus **GPU film
   grain** (`fast_grain.py`, Lever 2 — the CPU numpy grain ~73 ms/frame ported to torch/MPS on the HD
   recon tensor already resident on the GPU, ~few ms/frame, recipe-parity ~48 dB), **HW VideoToolbox
   encode** (`h264_videotoolbox`, Apple media engine, ~5–10 ms/frame vs ~44 ms libx264; target-bitrate
   0.70 bpp tuned to match libx264 crf18 PSNR; falls back to libx264), and **GPU-resident recon**
   (`download_output=False` + `INSTANT_GPU_CACHE` bicubic-on-device + a contiguous-HWC download_rgb ~5×
   faster than the strided `img_to_host` → no per-frame HD host round-trip).
2. **Reactive mask + pipeline parallelism + batched sync (117→91 ms, 3.75×, `5e83ec1`).** Instant's
   occlusion mask flipped `adaptive → reactive` (Step 7 established reactive == full-mask quality on
   low-motion talking-head; the fwd-bwd softmax splat was ~12 ms of recon, fired 63/82 mask calls on real
   footage → reactive cuts recon ~58→47 ms AND shrinks `hole_frac` so fewer safeguard SR upgrades).
   **Pipeline-parallel decode/encode** (`pipe_encode.py`): `ThreadedEncoder` runs the VideoToolbox encode
   on a worker (media engine works frame i while the GPU produces i+1) and `prefetch_chunks` decodes the
   next GOP on a CPU worker while the GPU works the current — GPU stays single-threaded on the main thread.
   **Batched `hole_frac` device→host sync** (`prototype/derisk.py`): the per-frame `.item()` on `hole_frac`
   drained the MPS queue ~12 ms/frame; batching that device→host sync removes the stall — **numerically
   identical, regression EXACT.**
3. **720p tier (91→72 ms, 5.7×, `2b46fc6`).** Instant renders at **half scale** (`INSTANT_SCALE=2`,
   x2 = 1280×640 from 640×320) instead of full QHD x4 → ~4× fewer output pixels → recon/grain/encode all
   drop ~4× (the **recon warp wall dropped ~47 → ~12 ms** — the 3.3 MP QHD warp was the floor). The SR
   net is still the **x4 net + downscale** (`upscale_to`) → a *sharp* 720p, better than a native 2× net.
   Quality + layered stay full QHD.
4. **Safeguard-off → pure anchor-only SR (72→41 ms, 10.0× = 24 fps, `d90055e`).**
   `INSTANT_FALLBACK_THRESH` 0.08 → **0.50**: at the 720p tier bicubic occlusion fallback is visually
   fine (it is the fast/lower-quality tier), so the adaptive safeguard fires only on **catastrophic
   >50%-fallback frames** (rare) → SR is essentially **pure anchor-only (~8 ms/frame)**. Verified clean on
   talking-head at 720p.

**The QHD floor was ~91 ms (~4.5×)** — the 3.3 MP warp is a hard floor; **the 720p tier was the unlock**
to a true 10×. **Tile-SR (Lever 3, `INSTANT_TILE_SR=False`) was measured and DISABLED**: SR only the
bounding box of a frame's occlusion-fallback region — but on this real footage the fallback is spatially
**scattered** (camera + complex motion), so a single bbox covers **~97%** of the frame (even a 32×16 grid
reaches only ~46% coverage) → hundreds of tiny SR passes whose launch overhead erases the area saving.
Kept available (set True) for content where a compact moving-edge occlusion actually holds.

## Research round R1 — 4 parallel Opus experiments + seam-verified integration (2026-06-19)

After the improvement loop, a research-lead pass ran **4 parallel Opus subagent experiments** (each
importing prototype/server READ-ONLY, honest metrics only — tOF + fallback% + direct |ΔF|, never
LR-consistency or NR-sharpness alone). Reports + scripts live in `experiments/expN_*/REPORT.md`. The
lead then seam-verified every proposed caller/handler signature against the real pipeline and
integrated the GO findings behind flags; the **synthetic regression stayed byte-identical**
(41.08→31.45 dB, 23/23, 5.84%) throughout. GO/NO-GO board:

| # | experiment | verdict | shipped |
|---|---|---|---|
| **E1** | Progressive play-while-process (fMP4 over chunked HTTP) | **GO** | `server/progressive.py` + `GET /api/stream` + `POST /api/upload` — **server-verified, browser-pending** (opt-in) |
| **E4a** | fp16 SR net | **GO** | `--fp16`/`half=` in `sr.py`+`derisk.py`; quality mode `fp16=True` |
| **E3-V2** | motion-modulated grain (freeze on static) | **GO** | `grain.apply_grain_motion`; quality mode `grain_motion=True` |
| **E2** | high-motion content-adaptive fallback | GO (opt-in) | **documented, default-OFF hook ready** (`thresh_fn`); not wired |
| **E3-V3** | graphic/text-edge detector | util ready | detector validated; **pinning NO-GO** (not wired) |
| **E4b** | FSR2 color-box clamp (`--clamp`) | **NO-GO** | structural — see below |
| **E2b/c** | adaptive re-anchor / QHD-instant escalation | **NO-GO** | wrong lever / breaks real-time |

**Integrated + verified this round:**
- **E4a fp16 (quality mode).** `sr.load_model(name, device=None, half=False)` casts the net to fp16 on
  MPS/CUDA (CPU-guarded → silently fp32, never crashes); the **cache key is now `(name, half)`** so fp16
  and fp32 nets don't collide. `upscale`/`upscale_to`/`derisk.build_perframe_cache` thread `half=`.
  `MODE_CONFIG["quality"]["fp16"]=True`; **instant stays fp32** (`fp16` absent → byte-identical). Measured
  A/B: x4plus **PSNR(fp16,fp32) 71.7 dB**, compact 76.1 dB (≫50 dB = visually identical), maxΔ 1–2/255, no
  NaN; ~1.24× best-of-N speedup on x4plus (E4). End-to-end quality run: 2466 ms/frame (was ~2900).
- **E3-V2 motion grain (quality mode).** `grain.apply_grain_motion(rgb, idx, static_w_hd, ...)` = exactly
  `apply_grain` except the per-pixel unit field is `renorm(a·frozen + (1−a)·fresh)` gated by the
  region-aware static weight (`region_gate["a_lr"]`, already computed for region-aware), upsampled to HD.
  `MODE_CONFIG["quality"]["grain_motion"]=True`. Static-region |ΔF| 5.649→**2.313 ≈ the 2.307 no-grain
  floor** (~100% of grain-induced static flicker removed); moving regions keep independent fresh grain
  (raw-field corr 0.0010). Verified: static gate → frozen (corr 1.0000); moving gate → maxΔ=0 vs
  `apply_grain` (degenerate case preserved). Output-only, never enters `R[]`.
- **E1 progressive playback — SERVER-VERIFIED, BROWSER-PENDING.** `server/progressive.py` (promoted from
  the validated prototype) builds a **fragmented MP4** (`movflags=empty_moov+frag_keyframe+default_base_moof`)
  with the upscaled instant video + source audio interleaved in ONE container, emitted incrementally over a
  `StreamingResponse` so playback can start after one fragment instead of the whole clip. `GET /api/stream`
  (instant|bicubic, 409 if busy) + `POST /api/upload`. **Verified end-to-end at the HTTP/decode level**:
  init (`ftyp`+`moov`) up front, **play-before-EOF** (a 25% byte prefix re-decodes to frames in PyAV),
  whole stream → full frame count + synced AAC, 200/`video/mp4`/no-Content-Length, 409 under concurrency,
  and the **client-disconnect lock-release bug fixed** (see GOTCHA #28). **NOT yet verified in a real
  browser** — Chrome held `readyState=0` on both plain `<video src>` and an MSE `SourceBuffer`, and the
  renderer repeatedly froze under CDP (local instability + a real open question about Chrome's consumption
  of chunked fMP4 — see GOTCHA #29). Because of that, the UI exposes progressive streaming as an **opt-in
  checkbox (default OFF)** so instant keeps its proven buffered path — **no regression**.

**Documented, validated, NOT wired (ready for a follow-up):**
- **E2 motion-keyed fallback** (high-motion instant weak spot). KEY honest finding: on high motion **tOF and
  fallback% are in direct tension** — bicubic fallback is already tOF-optimal, so *no* policy improves both
  (the task's premise was refuted, stated rather than papered over). The one shippable lever is a
  **motion-keyed threshold @ 0.20** (escalate bicubic→compact-SR fallback only when mean·|MV|>1.0): window
  A eff-bicubic 7.71%→3.65% (weak spot halved) at an honest tOF cost (0.847→1.214), **zero cost on
  talking-head** (self-gating). Default-OFF `thresh_fn` hook spec'd in `experiments/exp2_highmotion/REPORT.md`.
- **E3-V3 graphic detector** (`detect_graphic_mask`, bimodal+low-MV): 0.00% mis-fire on 32 talking-head face
  frames, correct true-positives on the title card. Ship as a default-OFF diagnostic util; **pinning is
  NO-GO** — the "USACHEV TODAY" card is **zero-MV skip-coded**, so the engine's identity-warp propagation
  (edge |ΔF| 0.816) is already steadier than per-frame SR (1.009); pinning makes it 2.5× worse (2.061).
  NEMO anchor-reuse already stabilises a static graphic; the "shimmer under warp" premise fails on this clip.

**NO-GO (documented so they aren't re-litigated):**
- **E4b color-box clamp (`--clamp`) — STRUCTURAL NO-GO.** No γ works: loose γ≥2 preserves SR detail (100%
  var-Lap) but doesn't reduce ghosting (HF-divergence flat); tight γ=1 clips real texture (var-Lap→90%,
  PSNR 42 dB) and still doesn't fix it. Root cause: rate-distortion MVs point to **visually-similar**
  content, so the ghost's colours sit *inside* a box built from the same LR neighbourhood — an RGB color box
  provably can't separate ghost from signal (FSR2 works because its ghost is a different surface; H.264
  ghosting is same-content-misplaced). Settles the long-standing "test, don't assume" `--clamp` question.
- **E2b adaptive re-anchor** (raises tOF, barely moves occlusion — the weak spot is disocclusion, not drift)
  and **E2c QHD-instant escalation** (5.27× recon cost, breaks real-time, eff-bicubic% unchanged since the
  fallback is still bicubic just higher-res).

## Research round R2 — 4 parallel Opus experiments (2026-06-20)

Continuing the autonomous experiment loop. Same discipline (READ-ONLY imports, honest metrics,
lead-owned seam-verification, synthetic regression byte-identical). Reports in
`experiments/r2_*/REPORT.md`.

| # | experiment | verdict | status |
|---|---|---|---|
| **R2-E1** | Commercial-licensed matte for layered (replace non-commercial RVM) | **GO** | validated, drop-in `seg_matte.py` ready (not yet server-wired) |
| **R2-E2** | Break the high-motion tOF↔fallback% tension | **frontier ESCAPED** (HF-only EMA) | validated, default-OFF spec ready |
| **R2-E3** | Layered seam-halo / hair (the open V4) | **GO** | **INTEGRATED** (`LAYERED_SEAM_FIX=True`) |
| **R2-E4** | Progressive-streaming hardening | **bug found** | **FIXED + verified** (GOTCHA #28-fix) |

**R2-E4 — fixed a real ship-blocker in the just-shipped progressive feature.** `FragmentMuxer.close()`
did `_feed_audio(float("inf"))`, draining the ENTIRE source audio track — so `GET /api/stream?frames=N`
on a long clip muxed 2032 s of audio over 24 s of video and produced a corrupt, un-decodable file.
Fix: `_feed_audio(self._video_time())` (the streaming loop already fed audio to ~`video_end +
AUDIO_LOOKAHEAD_S`). Lead-verified: capped `frames=600` now decodes 600/600 clean, audio bounded 25 s
(was 2032 s), 7.2 MB (was 19.6 MB corrupt); uncapped still clean. The non-AAC transcode, A/V-sync,
long-clip-bound, and video-only paths all PASS; a constant ~80 ms encoder-side video-start offset (no
edit list) is the only residual, harmless, non-accumulating.

**R2-E3 — INTEGRATED (the layered seam fix; closes the open V4 item).** Key correction to the L4
framing: the matte seam halo is a **soft-BACKGROUND problem, not a sharp-subject one** — the
near-subject plate ring is real background softened by low temporal coverage + matte-edge
contamination (var-Lap ~10.8 vs ~15.4 deep BG), and it is only ~8% inpaint (so a "better inpaint",
lever c, was a NO-GO). Fix = a band-localized **plate-ring sharpness restore** (`restore_plate_ring`,
baked ONCE per scene into the static plate in `layered_api` PASS A) + an **alpha-aware feather**
(`feather_alpha`, per frame in PASS B). Result: x4plus-bbox seam-discontinuity ratio **5.02 → 3.22**
(= the uniform-x4plus ceiling 3.45), halo moat **11.7 → 7.7 px**, subject core **EXACTLY unchanged**
(it re-contrasts existing texture, fabricates nothing), ~+5 ms/frame. **Softening the FG edge was
REJECTED** (wins the ratio only by smearing the subject). `lp.composite` keeps byte-identical defaults
(`seam_restore=0, feather=False`); the server layered path turns it on via `LAYERED_SEAM_FIX=True`.
Verified end-to-end: a layered run on `short.mp4` builds a plate (1 scene/0 fallback) and outputs valid
QHD with the seam fix active.

**R2-E1 — GO, validated, ready to wire (not yet server-integrated).** A permissive **torchvision
DeepLabV3-MobileNetV3-Large (BSD-3) + alpha-EMA** matte replaces the **non-commercial RVM (CC
BY-NC-SA)** with **no layered-plate regression** — plate coverage 74–79% (vs RVM 75.2%), holes 21–26%
(vs 24.8%), sharpness 95–119%, edges 2.8–4.5× steadier with EMA, latency 0.59–0.85× RVM (faster). The
only loss is RVM's wispy hair on the composite FG edge — irrelevant to the background plate (layered's
whole value). Drop-in adapter `experiments/r2_e1_matte/seg_matte.py` covers the **plate path**
(`matte_sequence`/`fg_mask_lr`) but **lacks `matte_frame`** (used by `layered_api` PASS B's per-frame
matte) — add a stateless `matte_frame` adapter method before the server swap. Production target:
**MediaPipe Selfie Segmentation (Apache-2.0)**, run in an isolated env (its protobuf<5 conflicts with
this env). GOTCHA #17 handled via a display-order alpha-EMA standing in for RVM's recurrent state.

**R2-E2 — the high-motion tOF↔fallback% frontier is ESCAPABLE, validated, default-OFF spec ready.**
R1-E2 found a hard wall (no policy improves both); R2-E2 shows it is breakable — but ONLY by
**high-frequency-only temporal smoothing**: `T = bicubic_current + EMA(sr − bicubic)` keeps the
motion-tracking low-freq fresh (tOF-safe) and temporally smooths only the flickery HF detail (which
carries little of the motion energy Farneback locks onto). Recommended `(c) gain=0.6, β=0.85,
feather=31`: high-motion eff-bicubic 7.70 → 6.35% at tOF +2.0% (the R1 hard switch charged +20% for the
same). **Spatial feathering sits on the frontier (no escape); naive temporal reuse — screen-space EMA /
warp-blend — GHOSTS (tOF 2.4–3.7×)** and notably had *lower* |ΔF| while tOF rose (re-confirming tOF is
the only honest temporal headline). The escape is **bounded** (only persistent disocclusions; freshly
revealed pixels must stay bicubic). Default-OFF, instant-only spec (replaces `patch_high_fallback`'s
hard SR with a feathered HF-EMA blend; reset the EMA buffer on I-frames/cuts) in the E2 report.

## Research round R3 — 4 parallel Opus experiments (2026-06-20)

Same loop + discipline. Reports in `experiments/r3_*/REPORT.md`. Two new frontiers + two
"convert-to-shippable" experiments that turn R2's validated-ready findings into landed, tested code.

| # | experiment | verdict | status |
|---|---|---|---|
| **R3-E1** | MV-reuse frame interpolation (2× fps) | **GO** | validated, integration sketch ready (not wired) |
| **R3-E2** | Content-robustness QA sweep (8 clips × 3 modes) | **found a real HIGH bug** | bug documented (top open item) |
| **R3-E3** | Wire+verify the E2 HF-EMA soft-occlusion | **PASS** | **INTEGRATED** (`INSTANT_SOFTOCC`, default-OFF) |
| **R3-E4** | Wire+verify the E1 commercial matte | **PASS** | **INTEGRATED** (`LAYERED_MATTE`, default `"rvm"`) |

**R3-E4 — INTEGRATED (commercial matte, flag-gated, default-preserving).** The R2-E1 adapter completed
with the missing per-frame `matte_frame` (the PASS-B gap) → `server/seg_matte_layered.py`; a
`LAYERED_MATTE` env flag (default `"rvm"` → byte-identical RVM demo; `"deeplab"`/`"lraspp"` → the
BSD-3 permissive commercial path) rebinds the `matting` module once so every call site is consistent.
Verified live: both branches produce valid QHD layered output (plate builds, PASS A + PASS B run); the
permissive matte's plate matches RVM (cov 73.7% vs 75.1%, hole 26.3% vs 24.9%), and the display-order
alpha-EMA recovers RVM-parity temporal stability (a|ΔF| 0.0068 vs 0.0065). **Layered can now ship
commercially.** Production target remains MediaPipe Selfie (Apache-2.0, isolated env per the protobuf
conflict); DeepLab-mv3+EMA is the runnable same-license-tier pick.

**R3-E3 — INTEGRATED (HF-EMA soft-occlusion, default-OFF, instant-only).** The R2-E2 frontier escape,
wired into `server/anchor_sr.py` (`softocc_patch` + helpers, replacing the hard `patch_high_fallback`
SR-patch when `INSTANT_SOFTOCC=True`) + `server/pipeline_api.py` flags. Verified through the real
`derisk.reconstruct`: window A OFF tOF 0.756 / eff-bic 7.70% → ON 0.771 / 6.35% (the escape reproduces
exactly), the EMA reset is proven safe (a deliberately-missed reset = a 4.6-RMS decaying cross-cut
ghost; the reset eliminates it), torch parity 0.18/255 MAE. Lead-verified OFF → instant byte-identical
(`n_sr_calls=2`); ON → the pass runs (48 SR / 46 blends). It costs ~1 compact-SR call per non-anchor
frame → a **quality knob, not a real-time default** (DEFAULT-OFF); `SOFTOCC_MOTION_GATE=1.0` bounds the
cost for a shallower escape.

**R3-E1 — GO, validated, ready to wire (not integrated).** MV-reuse motion-compensated frame
interpolation: warp each neighbour by the codec MVs we already extract + occlusion-aware (intra-hole)
blend → synthesize midpoints. Beats frame-dup / linear-blend by **+3.6 to +8.9 dB PSNR / +0.06–0.10
SSIM**, cuts tOF 2–4× (verified vs exact held-out real frames + visually removes the linear-blend ghost
on fast motion), at **~17 ms/inserted-frame on MPS** (2 warps + blend, zero SR, zero new flow; ~2.3×
cheaper than a real frame → 2× output fps for ~+42% compute). Ships as an **optional "smooth 2×"
output**, NOT free real-time 50 fps (the base recon already saturates one GPU at 25 fps). Two
ship-blockers: use **intra-hole routing only** (the project's full Ruder/reactive mask over-flags →
re-introduces ghosting), and a **scene-cut guard** (intra-hole fraction > 0.5 → fall back to
duplication). Output-only integration sketch (reuse `build_lr_flow` + `gpu_ops.warp_hd`) in the E1 report.

**R3-E2 — robustness sweep: zero crashes across 24 runs, but found a real HIGH bug (TOP OPEN ITEM).**
**Layered paints the WRONG background over an entire scene when a similar-luma scene cut is missed** —
`segment_scenes` (luma-only) misses the cut → one temporal-median plate spans two scenes → scene-A
plate composited over scene-B (LR-consistency collapses 33.8 → 14.7 dB). **tOF is BLIND to it** (the
wrong plate is temporally stable) — only fidelity-vs-LR caught it. Two more cliffs: **instant collapses
to per-frame-SR on low-light noise** (95% fallback → 24/24 SR upgrades → 3.6× slower, real-time broken),
and **instant softens/flickers on high motion** (tOF 3.18 vs 0.39). Plus a per-content
mode-recommendation + cheap auto-select signals (motion mag / graphic-edge density / plate safety). The
layered missed-cut corruption is the **#1 robustness fix** — see `experiments/r3_e2_robustness/REPORT.md`
(fix: a per-frame plate-validity guard cross-checking composite-vs-LR consistency + a chroma/structural
term in the cut detector). Other modes are unaffected by a missed cut (per-frame source is correct).

## Research round R4 — 4 parallel Opus experiments (2026-06-20)

Led by the #1 open bug. Reports in `experiments/r4_*/REPORT.md`.

| # | experiment | verdict | status |
|---|---|---|---|
| **R4-E1** | Fix the layered missed-cut plate corruption (the #1 bug) | **FIXED** | **INTEGRATED + verified** |
| **R4-E2** | Wire+verify MV interpolation ("smooth 2×") | **PASS** | validated, ready-to-land (default-OFF) |
| **R4-E3** | Instant low-light/noise cliff | **FIXED** | **INTEGRATED** (cap, default-OFF opt-in) |
| **R4-E4** | "Auto" mode-selection | **8/8 authored, 9/10** | validated, ready-to-land |

**R4-E1 — INTEGRATED (the silent layered missed-cut corruption is fixed).** Two fixes, defense-in-depth:
(a) a **per-frame plate-validity guard** (`layered_api.composite_frame_guarded` + `PLATE_GUARD_*`, wired
in `pipeline_api` PASS B): per frame, downscale the HD plate to LR and PSNR it against the decoded LR over
the **background region only** (matte α<0.5, eroded); trip on an absolute floor (24 dB) OR a relative
cliff (>8 dB below the per-scene EMA) → fall back to the **full-frame compact SR already computed inside
the composite** (so a tripped frame costs ZERO extra SR). Robust to ANY missed cut, detector-independent.
(b) a **chroma-dominant cut term** in `scene_detect` (fires only when `dChroma > 1.1·dLuma` — the
missed-cut signature; luma path byte-identical). **Lead-verified end-to-end:** on the c7 repro the post-cut
LR-consistency is restored **14.7 → 42.5 dB**; a normal static talking-head trips the guard 0/44 frames
(byte-identical), and `sample.mp4` cut detection stays 1.00/1.00 (zero new false positives). (a) is the
must-have — `segment_scenes` merges a too-short trailing scene back, which can undo (b) alone.
**Honest metric: this bug is invisible to tOF (the wrong plate is temporally stable) — only fidelity-vs-LR
catches it.**

**R4-E3 — INTEGRATED as a default-OFF opt-in (instant low-light/noise cliff).** Root cause (sharper than
R3-E2): noise makes the encoder **intra-code ~99% of blocks** → no MVs to propagate → the safeguard wastes
a full per-frame SR on every frame (121 ms/frame, 3.9× over budget). A **motion-gated fallback-saturation
cap** (`INSTANT_FALLBACK_SATURATION_CAP`, via the existing `thresh_fn` hook) declines SR escalation on
high-fallback + low-motion frames → bicubic floor → real-time (121→31 ms/frame, verified). **DEFAULT OFF
(=1.0)** because it is byte-identical only on LOW-fallback content — it also fires on non-noise
high-fallback + low-motion frames (a title-card reveal: short.mp4 n_sr 3→1), trading their SR fallback for
the bicubic floor (a small but real behaviour change). The product avoids the cliff at the mode level
instead: **R4-E4 auto-mode routes noise → quality**. Set `CAP=0.70` to enable for instant-on-noise; an
intra-fraction>0.8 gate (the agent's hardening note) would spare title reveals. Pre-denoise CANNOT restore
real-time (it can't create MVs — verified).

**R4-E2 — GO, validated, ready to wire (not integrated).** The R3-E1 interpolation wired as an optional
"smooth 2×" output reproduces the quality exactly through the pipeline (+3.6 to +8.9 dB PSNR over
dup/linear-blend), both ship-blockers verified (intra-hole routing only; scene-cut guard → frame-dup on
>0.5 intra-hole), output-only (real frames byte-identical, never enters `R[]`), OFF byte-identical (40/40),
real in-sync 50-fps mp4, ~8–15 ms/inserted-frame at the 720p tier. Lead-landable `interp_pass.py` +
default-OFF diff in `experiments/r4_e2_interp_wire/`.

**R4-E4 — auto-mode works, validated, ready to wire (not integrated).** A ~1 s `recommend_mode(input_path)`
probe (one decode pass: codec-MV magnitude + occlusion-fallback% + Canny edge density + scene-cut count +
static-camera verdict + a gated human matte) picks instant/quality/layered, matching the honest-metric best
on **8/8 authored clips, 9/10 overall** (the lone miss is a safe over-escalation — a content flash mis-read
as a missed cut → quality, costing time not quality). Two measured design choices: **median** MV magnitude
(a cut frame's 207-px spike fakes high motion in the mean), and **detected vs missed** cut (a detected cut
splits the chunk → instant-safe; only a *missed moving* cut escalates). The **human-matte gate**
structurally prevents the R4-E1 layered corruption on all non-human content. Probe is 50–150× cheaper than a
render. Integration sketch (`recommend_mode` + an "auto" mode) in `experiments/r4_e4_automode/`.

## Research round R5 — production readiness + the first perceptual quality numbers (2026-06-20)

A lean 2-experiment round (the new-feature frontier had thinned). Reports in `experiments/r5_*/REPORT.md`.

**R5-E1 — production stress test: ALL 3 MODES ARE PRODUCTION-READY, no blockers.** Stress-tested longer
windows (instant to 8000 frames / 320 s) + edge cases through the shipped `process_clip`. Every run:
valid + correctly-counted + audio-synced output, single-job lock releases cleanly, **instant is
bit-for-bit deterministic** on the MPS fast path, multi-scene clean (the `StreamingCutDetector` caught all
9 cuts in [0,900) exactly → no cross-cut smear), the R4-E1 layered guard classified 7 scenes correctly with
**no false-trips across 6 normal cuts**. **Memory (the headline worry) is RESOLVED:** instant RSS rises in a
~3000-frame **warm-up ramp then PLATEAUS** at ~1.3 GB (proven over 8000 frames; the earlier "169 KB/frame
creep" was just the pre-plateau ramp); quality/layered show a bounded per-chunk **sawtooth** (peaks 1.64 /
2.50 GB). So the bounded-memory claim holds for the full 50k-frame clip on a 17 GB box — more precisely it's
"**bounded after a ~3000-frame warm-up ramp**", not literally "constant from frame 0". **One real defect
(MEDIUM, fixed):** the layered UI time-estimate (`MODE_MS_PER_FRAME["layered"]`) was 470 ms/frame (all-static
assumption) but a MOVING scene falls back to the ~2900 ms quality path → measured **1382 ms/frame** on mixed
content (4/7 moving) = ~3× under-promise → **fixed: estimate bumped to 1400** (conservative mixed value;
content-dependent — moving content is better routed by R4-E4 auto-mode). Verdicts: **instant SHIP** (37
ms/frame, ~1.1× real-time, deterministic), **quality SHIP** (ETA honestly conservative 2517 vs 2900),
**layered SHIP** (ETA caveat now fixed).

**R5-E2 — the project's FIRST perceptual reference numbers (real measured LPIPS, not a proxy).** Degrade-
and-restore protocol (SD = pseudo-HD GT → degrade 2× → restore → score vs GT) with a `real` codec-like
degrade operator. Headline LPIPS (talking-head / high-motion, `real`): instant **0.108 / 0.008**, quality
**0.108–0.123**, layered-plate **0.123 / 0.009**. **Three findings that should shape strategy:**
1. **[REFUTED by R6-E1 — see below] The heavy x4plus appeared NOT to be a quality win** — on R5-E2's two
   windows the COMPACT model beat it on every full-reference metric (LPIPS 0.108 vs 0.123) at 1/10 the
   compute. **⚠ R6-E1 OVERTURNED this: R5-E2 had tested only the clip's two LOWEST-detail windows (smooth
   face + flat title card), where x4plus's prior is pure misalignment. Across 5 windows × 3 degrade levels,
   x4plus BEATS compact decisively on the detailed/graphics/text/photo content the quality mode targets
   (9/15 cells, 100% of frames on every textured heavy/gritty cell, lead growing to −34% LPIPS). So the
   "lean compact" suggestion is WITHDRAWN — KEEP quality = x4plus + region-aware + fp16 unchanged.** The
   live GOTCHA #23 (x4plus "sharper" by var-Lap) still holds — LPIPS is the arbiter, and it now says x4plus
   wins where it matters. x4plus also stays right for the layered static plate.
2. **fp16 is perceptually identical** (LPIPS(fp16,fp32) 0.00002–0.00005) for a free 1.23× — re-confirms R2-E4.
3. **Grain HURTS every full-reference metric** (it's additive noise uncorrelated with the clean GT) — it is
   an aesthetic/NR overlay (default-OFF is right for fidelity); `low` is the minimum dose if wanted, and a
   touch of `low` grain *reduces* tOF on high-motion. Also: PSNR rated SR ≈ bicubic on talking-head while
   LPIPS showed SR *halves* perceived distortion (−51%) — the perception-distortion gap the project never measured.

## Research round R6 — 3 experiments: SR-quality decision + Auto mode shipped (2026-06-20)

Reports in `experiments/r6_*/REPORT.md`. Notable: the loop **self-corrected** — R5-E2's "compact ≥
x4plus" claim was refuted by R6-E1 *before* any quality-mode change shipped (I had documented but
deliberately NOT acted on it, pending confirmation — the adversarial-verify discipline paying off).

| # | experiment | verdict | status |
|---|---|---|---|
| **R6-E1** | Confirm/refute "compact ≥ x4plus" (5 windows × 3 degrades) | **REFUTED** — x4plus wins on detailed content | **no config change** (keep x4plus) |
| **R6-E2** | Wire+verify "Auto" mode-selection | **PASS** | **INTEGRATED + verified** |
| **R6-E3** | New SR-quality lever (TTA / blend) | blend beats both at moderate degrade | content-dependent — NOT a default |

**R6-E1 — REFUTED R5-E2; quality mode unchanged.** R5-E2's "compact beats x4plus" was an artifact of
testing only the two lowest-detail windows. Across 5 windows (smooth face → news graphics/text/charts/
photos) × 3 degrade operators (moderate → gritty 2nd-order), **x4plus beats compact on TRUE LPIPS in 9/15
cells, 100% of frames on every textured heavy/gritty cell, lead GROWING with detail + degradation (up to
−34% LPIPS)** — a clean perception-distortion win (lower PSNR, lower LPIPS; var-Lap useless). So **keep
`MODE_CONFIG["quality"]` = x4plus + region-aware + fp16 unchanged** (switching to compact would be a measured
regression on exactly the content quality mode targets); the region-aware gate already routes heavy→static-
detail, compact→moving. **Optional R7 compute optimization (flagged, unmeasured-at-integration):** the gate
keys on MOTION but the true discriminator is TEXTURE×degrade — a local-detail term in `_build_region_gate`
(heavy only where static AND high-texture) could cut quality compute at ~0 LPIPS cost; A/B before landing.

**R6-E2 — INTEGRATED + verified ("Auto" mode-selection).** `pipeline_api.recommend_mode(input_path)` (a
cheap ~0.1–3.4 s probe: codec-MV magnitude + occlusion-fallback% + Canny edge density + scene-cut count +
static-camera verdict + a gated human matte) resolves `mode="auto"` at the top of `process_clip` → runs the
chosen mode unchanged → surfaces `{auto_chosen, auto_reason, auto_signals}` in `LAST_STATS` + the UI.
`app.py` has an "Auto (recommended)" card (default), `index.html` defaults to it. **Lead-verified:** server
imports + auto registered; instant/quality/layered all valid through the wiring; **non-auto is byte-identical
at the pixel level** (instant n_sr=3 baseline; the agent's double-run pixel-hash check); auto resolves to a
real mode + valid output; the probe is ~0.3 s vs a 42.9 s render. Design carried from R4-E4: **median** MV
magnitude (a cut frame's 207-px spike fakes high motion in the mean), **detected-vs-missed** cut, and a
**human-matte gate** (non-human → layered never selected → a second safety net for the R4-E1 layered
corruption). Routing honest-agreement 8/10 (2 benign misses — one is the real-time-correct instant pick at a
tOF-1.01 boundary; one a safe over-escalation to quality); R4-E4's one historical misroute is now fixed.
Only `pipeline_api.py` + `app.py` + `index.html` changed; the `_run_layered` matte load is now shared with
the probe (loaded once, re-raises on failure). v1: auto uses the buffered POST path (not `/api/stream`).

**R6-E3 — a real but content-dependent quality lever (NOT a default).** `compact + 0.5·(x4plus − compact)`
(an anchor output-pass lerp) beats BOTH compact-alone (−10.6%/−12.0% LPIPS) and x4plus-alone on R6-E3's two
**moderate-degrade** windows, with moderate var-Lap (genuine aligned detail — the freq-gated and unsharp
variants both FAILED by inflating var-Lap while hurting LPIPS, GOTCHA #23 caught twice; TTA confirmed the
hallucination-cancel mechanism on x4plus but costs 31×). **BUT** synthesized against R6-E1: the blend is
halfway to compact, so on the **heavy/gritty textured** content where x4plus's full strength wins (R6-E1),
the blend would *pull back* that HF and lose — so it is **NOT a safe quality default** (it helps moderate
degrade, hurts gritty). Filed as a content/degrade-adaptive option, not landed.

## Research round R7 — ship the last feature + a quality-mode optimization (2026-06-20)

A lean 2-experiment round. Reports in `experiments/r7_*/REPORT.md`.

**R7-E1 — INTEGRATED + verified: "smooth 2×" MV interpolation (default-OFF).** The validated R4-E2
interpolation, landed for real: `server/interp_pass.py` (new) + a 2-flag wire in
`server/pipeline_api.py::process_clip` (`INSTANT_INTERP_2X=False`, `INTERP_CUT_THRESH=0.5`). When ON, the
instant fast path inserts one MV-reuse midpoint between consecutive output frames → 2× output fps. **Lead-
verified through the real `process_clip`:** OFF is **byte-identical** (the whole output mp4 md5 is unchanged
`ef8d4835405a` + 40/40 frame hashes), so instant/quality/layered cannot regress; ON gives exactly 2× frames
at 2× fps with a valid audio-synced mp4; the midpoint is output-only (never enters `R[]`, never called when
OFF); the scene-cut guard duplicates the last real frame (no ghost) at the I-frame boundary. Both R4-E2
ship-blockers (intra-hole-only routing, cut guard) intact. Optional follow-up: a UI toggle (a `smooth_2x`
param on `app.py`'s `process_clip` call; it ~halves instant throughput so it's a render option, not real-time).

**R7-E2 — texture-gated region-aware blend: GO (quality-safe), validated-ready, NOT landed.** Acting on
R6-E1 (x4plus only wins on TEXTURED content): a texture term `a' = a_motion · a_texture` (texture = the FREE
local-std of the already-computed compact source) routes the heavy x4plus only where static AND textured.
**Quality-safe + slightly BETTER:** on the static-but-smooth talking-head face it IMPROVES LPIPS by ~0.02
(fixes the motion-only gate over-applying misaligned heavy HF, exactly R6-E1's prediction) and cuts the
effective heavy-SR fraction 74%→~12%; on detailed graphics it's neutral (LPIPS ±0.001, heavy kept ~28%).
**BUT the compute win is NOT realized by the flag alone** — today the gate is an OUTPUT-ONLY blend (both
heavy + compact are already computed), so it only changes output; the real saving needs a **follow-up
tile-skip wiring** (run the x4plus anchor SR only on textured-static tiles), and the saving is tiling-bound
(scattered talking-head texture → ideal −83%, coarse 16–32px tiles only −18 to −33%). Delivered as a
**default-OFF `texture_aware` flag** diff (`experiments/r7_e2_texgate/derisk_build_region_gate.patch`,
byte-identical when off) — NOT landed (an unused flag whose payoff needs the tile-skip follow-up + an
integrated propagation+tOF A/B before flipping default). Filed as a 2-part follow-up. Keep compact (never
bicubic) as the low-detail fallback.

## Diffusion anchor — NO-GO (research pass 4 `wau8fdg60` + the Q1 real-weights spike)

Whether a one-step / few-step diffusion model should be the heavy anchor was chased twice; it is a
**NO-GO on this content**:
- **MPS feasibility** (Step 9 Stream-3): diffusion DOES run on MPS, but VAE *decode* dominates and the
  per-anchor latency is ~7–14× x4plus → too slow without a tiny VAE (TAESD).
- **The Q1 real-weights A/B** (`stabilityai/stable-diffusion-x4-upscaler`, the diffusers-native, ungated
  model — I ran the A/B myself): on a REAL H.264-compressed SD anchor crop it **HALLUCINATES** — var-of-
  Laplacian face 216 vs x4plus 16 (13×) but the "win" is **FAKE**: it paints smooth skin (a hand) in
  grainy/scaly texture and rewrites the plaid weave. SD-x4-upscaler was trained on CLEAN bicubic LR, so
  it reads H.264 compression noise as signal — the exact NR-metric trap research pass 4 warned about.
**Verdict:** **x4plus stays the max-quality anchor.** Revisiting needs a **degradation-aware** model
(OSEDiff/StableSR — trained/finetuned on real codec degradation) **+ TAESD** (kill the VAE-decode cost)
+ CoreML/ANE. Even then, the Step-8 detail-drift finding caps the benefit to static content/regions.
Visuals: `prototype/out_diffusion_real/`.

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

20. **Whole-clip processing MUST stream GOP chunks — never window-in-memory** (server). The first
    server build held an entire window at HD and opened `avcodec` on it → OOM / `avcodec_open2` EAGAIN
    past ~n=5000. The rebuild opens the container ONCE and processes self-contained GOP chunks (cut at
    every I-frame; a long GOP also cut at the next P = a forced fresh anchor, `SOFT_CAP_FRAMES=48`),
    encoding incrementally so only one chunk of HD frames is ever alive. Peak memory is bounded
    regardless of clip length.

21. **Free the MPS caching allocator BETWEEN chunks** (server). Over a long clip the MPS allocator's
    freed-but-cached memory creeps up (NOT bounded by per-chunk `del`s) and eventually fails an
    allocation with `BlockingIOError`/EAGAIN under memory pressure (the "frame-630" crash). Call
    `gc.collect()` + `torch.mps.empty_cache()` once per processed chunk (`_free_gpu()`): it returns
    cached memory to the OS, leaves active tensors untouched, costs ~ms.

22. **Write the output mp4 with `+faststart`** (server). Without the moov atom moved to the front, a
    `<video>` element stalls at readyState 0 until the whole file is fetched (no progressive play).
    `av.open(out, "w", options={"movflags": "+faststart"})` is what makes the result reliably
    web-playable. (Audio: copy if the source is AAC, else transcode to AAC; muxed in sync up to the
    video duration; a video-only source does NOT crash.)

23. **Diffusion SR hallucinates on real compressed SD** (Q1). SD-x4-upscaler was trained on clean
    bicubic LR; on real H.264 it reads compression noise as signal and invents grainy/scaly texture and
    false weave. Its var-Lap "win" over x4plus is FAKE detail, not fidelity — never trust an
    NR-sharpness metric to pick the anchor. x4plus stays the anchor; a diffusion anchor needs a
    degradation-aware model + TAESD (see "## Diffusion anchor — NO-GO").

24. **Instant runs at 720p; quality + layered run at QHD** (improvement loop). `INSTANT_SCALE=2`
    (1280×640) is the fast/real-time tier; `SCALE=4` (2560×1280) is quality + layered. The whole instant
    path is scale-parameterized (`anchor_sr` derives `scale = w_hd // w_lr`, `reconstruct` takes the
    scale), so it is just `eff_scale` — do NOT hard-code 4. The ~10× speedup depends on the 720p tier:
    the 3.3 MP QHD warp is a hard floor (~91 ms ≈ 4.5×), so QHD instant is NOT real-time on one GPU.

25. **A scene cut MUST force a fresh anchor / chunk boundary** (improvement loop iter 1). `reconstruct`
    warps within one scene only; a chunk that spans a cut smears the pre-cut anchor across it. `stream_gops`
    cuts at every I-frame AND every `scene_detect` cut — never disable `detect_cuts` in production (it
    exists only to reproduce the cross-cut-smear BEFORE case). Periodic keyframes are NOT cuts (small
    |Δluma|); do not anchor on I-frames alone.

26. **Batch the `hole_frac` device→host sync — never `.item()` per frame** (improvement loop iter 4).
    A per-frame `.item()` on the GPU-resident `hole_frac` blocks until the MPS queue drains (~12 ms/frame
    of stall on the instant hot loop). Batch the device→host sync instead — the values are numerically
    identical (the synthetic regression stays EXACT), it just removes the per-frame pipeline bubble.

27. **The 720p instant tier uses the x4 net + downscale, NOT a native 2× net** (improvement loop iter 5).
    `upscale_to(lr, w_hd, h_hd)` runs the x4 SR net then downscales to the 720p target — this yields a
    *sharp* 720p (the x4 net's detail, resampled down), visibly better than running a native 2× net. Do
    not "optimize" instant by swapping in a 2× model; you would lose the detail the downscale preserves.

28. **A *sync* generator wrapped by StreamingResponse is NOT sent GeneratorExit on client disconnect**
    (R1 progressive playback). The first cut acquired the single-job lock and released it in a sync
    generator's `finally` — but on disconnect that finally never ran, so a runaway client left the GPU
    churning the WHOLE clip and held the lock forever (a 34-min `sample.mp4` instant stream kept going
    after `curl` was killed). FIX (`server/app.py::api_stream`): make the response body an **async**
    generator that pulls the sync producer one fragment at a time via `run_in_threadpool`. Starlette's
    `StreamingResponse` runs a `listen_for_disconnect` that **cancels the stream task** on
    `http.disconnect`; the CancelledError lands at the generator's next `await` → its `finally` runs **SYNC**
    cleanup (close the producer = flush+close the muxer + free GPU; release the lock). `run_in_threadpool`
    finishes the in-flight chunk before the cancel lands, so a disconnect stops the GPU within ONE chunk
    (verified: 75 frames then freed, not 50 805). **Do NOT also poll `request.is_disconnected()`** — it
    calls `receive()` concurrently with Starlette's own disconnect listener on the same ASGI channel and
    BREAKS the cancellation (this was the bug before the fix).

29. **Progressive-playback BROWSER consumption is IMPLEMENTED (MSE) but UNVERIFIABLE on this box's Chrome**
    (R1 + R1.1). `index.html` now consumes the stream via a Media Source Extensions `SourceBuffer` (the
    reliable way Chrome plays live fMP4): `MediaSource` + a `fetch().body` ReadableStream, append each chunk,
    `sb.mode="sequence"`, evict already-played buffer on `QuotaExceededError` (so arbitrarily long clips
    play), codec `avc1.640028`/`64001f`/`4d4028`+`mp4a.40.2`, with a plain `<video src>` fallback (Safari
    plays fMP4 natively). The **server bytes are proven valid** (PyAV re-decodes a truncated prefix =
    play-before-EOF). BUT in-browser playback **could not be verified here**: this machine's Chrome
    `MediaSource` **never transitions to "open"** (`sourceopen` never fires, `video.error` is null) and the
    CDP renderer froze repeatedly — a dead local media pipeline, NOT a code defect (MSE init with no error =
    environment). So progressive streaming stays **opt-in (default OFF checkbox)**; instant keeps its proven
    buffered `POST /api/process → /outputs` path — no regression. **To finish E1: load `/` in a WORKING
    browser, check "▶ Play while processing", confirm `currentTime` advances + audio while `/api/progress`
    reports `streaming`; then flip the default by defaulting the checkbox checked** (a one-line change in
    `index.html`). Do NOT flip it default-on until a real browser confirms playback (test what the user sees).

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

## Key research findings (5 passes, all adversarially verified)

(Passes 1–3 detailed below; **pass 4** `wau8fdg60` — better upscaling: heavy/diffusion anchors, film
grain, propagation-stabilizes-detail — and **pass 5** `wce7evoli` — layered/compositional video SR:
Wang&Adelson, RVM, MV-free segmentation/plate — are summarized in Steps 8–10, the layered section, and
"## Diffusion anchor — NO-GO" above.)

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

Steps 1–10 validated the architecture on real H.264, reached real-time on Apple Silicon, and finished
the layered track. The PRODUCT now exists as a **Stage-1 server** (web app, 3 modes), and the
**improvement loop** hardened it (scene-cut detection, the layered grain-gate, and instant → 10× /
24 fps real-time). The frontier is making that product good:

1. ✅ **DONE — scene-cut detection** (improvement loop iter 1, `server/scene_detect.py`): luma-diff +
   I-frame-corroborated + relative/hysteresis + min-scene-len, 1.00/1.00 prec/recall on `sample.mp4`,
   forces a fresh anchor at every cut. One `StreamingCutDetector` shared by `stream_gops` + layered
   `segment_scenes`.
2. ✅ **DONE — make "instant" actually instant** (improvement loop iters 3–6): anchor-only SR + GPU
   grain + HW encode + GPU-resident recon + reactive mask + pipeline parallelism + the 720p tier →
   **~409 → 41 ms/frame = 10.0× = ~24 fps = real-time**. (See "## Improvement loop".)
3. **Progressive play-while-processing — now the top lever.** With instant at ~24 fps it keeps up with
   playback, so streaming output as it is produced (HLS / fMP4 segments) + a background-render mode lets
   the user start watching almost immediately. The quality/layered modes (still ~10× slower) get the same
   UX win for short leads. This is the biggest remaining UX lever.
4. **Improve the high-motion / instant-quality regime** (standing weak spot): ~24% honest P-frame
   occlusion fallback, needs ~12.5% anchors at budget 1.0; at 720p instant the safeguard is off
   (`INSTANT_FALLBACK_THRESH=0.50`) so a high-motion clip leans on bicubic fallback — a content-adaptive
   threshold (or a QHD instant tier when motion is high) is the lever. Plus better masks, shorter
   P→anchor hops, tOF+fallback%-keyed triggering (the honest metrics — NOT LR-consistency).
5. **Productionization / later:** a commercially-licensed matte (RVM is non-commercial); a
   degradation-aware diffusion anchor + TAESD (see "## Diffusion anchor — NO-GO"); color-box clamping
   (`--clamp`, untested); and the **Stage-2 browser-only WebGPU/ORT-Web port** — where
   codec-MVs-in-the-browser becomes the make-or-break unknown (ffmpeg.wasm export_mvs / JS H.264 parser /
   WebGPU optical-flow substitute / keep MV extraction server-side). Benchmark web-realesrgan /
   Anime4K-WebGPU / websr.
