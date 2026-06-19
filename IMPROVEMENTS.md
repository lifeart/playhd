# playhd — prioritized improvement backlog

> **STATUS — improvement loop DONE (commits `397f461`..`d90055e`).** The top items below were
> implemented; see `handoff.md` → "## Improvement loop" for the full write-up.
> - ✅ **V1 — layered grain-gate (DONE, `0e97c9e`):** bake one fixed grain into the static plate +
>   per-frame grain only on the alpha-gated foreground → background **|ΔF| 4.5 → 0.000**; the ~167×
>   steadier background is finally visible.
> - ✅ **P1 — anchor-only SR (DONE, `10de07e`, `server/anchor_sr.py`):** SR only the anchors (+ a
>   catastrophic-fallback safeguard), bicubic the rest; propagated pixels are identical, only fallback
>   pixels differ.
> - ✅ **P2 + the real-time stack / 720p (DONE → 10×, `10de07e`/`5e83ec1`/`2b46fc6`/`d90055e`):** HW
>   VideoToolbox encode + reactive mask + GPU grain (`server/fast_grain.py`) + GPU-resident recon +
>   pipeline-parallel decode/encode (`server/pipe_encode.py`) + batched `hole_frac` sync + a **720p
>   instant tier** (`INSTANT_SCALE=2`) → instant **~409 → 41 ms/frame = 10.0× = ~24 fps = real-time**.
> - ✅ **Scene-cut detection (DONE, `397f461`, `server/scene_detect.py`):** robust cut detector
>   (1.00/1.00 prec/recall on `sample.mp4`) forces a fresh anchor at every cut → no cross-cut smear.
> - ◻ **Still open:** V4 (layered seam/hair), and the BROWSER half of progressive playback.
>
> Tile-SR (P1's Lever 3) was measured and **DISABLED** (`INSTANT_TILE_SR=False`): the occlusion
> fallback is spatially scattered on real footage so a bbox covers ~97% of the frame → no win.
>
> **UPDATE — research round R1 (4 parallel Opus experiments; see `handoff.md` → "## Research round R1"
> and `experiments/expN_*/REPORT.md`):**
> - ✅ **P3 fp16 SR net — DONE (GO):** `half=` in `sr.py`/`derisk.py`, quality mode `fp16=True`. x4plus
>   ~1.24× faster, PSNR(fp16,fp32) 71.7 dB = visually identical; instant stays fp32 (byte-identical).
> - ✅ **V2 motion-modulated grain — DONE (GO):** `grain.apply_grain_motion`, quality mode
>   `grain_motion=True`. Static |ΔF| 5.649→2.313 (≈ no-grain floor); moving keeps fresh grain.
> - ◐ **Progressive play-while-process — SERVER DONE, browser PENDING:** `server/progressive.py` +
>   `GET /api/stream` (fMP4 over chunked HTTP). HTTP/decode-verified (play-before-EOF, audio, 409,
>   disconnect-frees-lock); browser playback unconfirmed → shipped opt-in (default OFF). See GOTCHAs #28–29.
> - ◻ **V3 graphic-edge:** detector validated (0% face mis-fire) but **pinning NO-GO** — the title card is
>   zero-MV skip-coded so NEMO anchor-reuse already out-stabilises per-frame SR. Detector ready as a util.
> - **NO-GO (settled):** color-box clamp `--clamp` (structural — RD-MVs land the ghost inside the box);
>   high-motion adaptive re-anchor / QHD-instant escalation (E2, wrong lever / breaks real-time). On high
>   motion **tOF↔fallback% are in tension** (bicubic fallback is already tOF-optimal) — the motion-keyed
>   threshold @0.20 is a documented default-OFF opt-in (halves window-A weak spot, zero cost on talking-head).

Performance + visual-quality "poke", 2026-06-19. Read-only analysis of existing code, the
documented Step 6/7 profiling, and the shipped `server/outputs/*.mp4`, plus light new
measurement (one 24-frame instant run; a few-frame visual diff on real outputs). No
pipeline/prototype/server code was modified.

Honesty note up front — what is **measured** vs **inferred**:
- MEASURED: the SR-every-frame code path (read directly); the instant-mode time split (one
  short run, see below); the background frame-to-frame flicker across all three shipped modes
  (real outputs); the grain σ; the on-disk Step 6/7 component profiles.
- INFERRED/ESTIMATED: the encode/grain ms sub-breakdown inside "other"; the exact net speedup
  of an anchor-only rewrite (depends on the fallback policy chosen); fp16-SR gains.
- The one new MPS run was under GPU contention (another agent was using the GPU), so absolute
  SR/recon ms are inflated; the **ratios** are what matter and are robust to that.

---

## TL;DR — the two headline findings

1. **PERF (confirmed): "Instant" mode runs the full SR net on EVERY frame.** It is the single
   biggest, most clearly-wasteful cost — ~48% of instant-mode wall time — and the architecture
   needs SR on ~1 frame in 48. Fixing it ≈ halves processing time and is the gateway to
   play-while-processing. **→ item P1.**

2. **VISUAL (confirmed): per-frame film grain ERASES the layered mode's only advantage.** The
   prototype measured the layered background plate at |ΔF| = 0.001 (≈1700× steadier than
   per-frame x4plus). In the **shipped** layered video the background flickers at ~4.5/frame —
   identical to instant/quality — because grain is added over the whole frame, including the
   static plate. The stability win is 100% invisible to the viewer. Cheap fix. **→ item V1.**

---

## PERFORMANCE

### P1 — ✅ DONE (`10de07e`, `server/anchor_sr.py`) — [CRITICAL · impact HIGH · effort MEDIUM] Instant runs full per-frame SR; the architecture needs anchor-only SR
> **Done:** `anchor_sr.build_anchor_cache` SRs only the anchors (+ a catastrophic-fallback safeguard
> above `INSTANT_FALLBACK_THRESH`), bicubic everywhere else; the propagated pixels are byte-identical,
> only the small fallback fraction differs. This was the gateway to the instant 10× / real-time push.

**Issue.** `server/pipeline_api.py:process_clip` → `derisk.build_perframe_cache(chunk, …)`
(`prototype/derisk.py:466-477`) runs the SR network on **every frame of the chunk**
(`for i in range(N): cache[i] = upscale_to(...)`). `derisk.reconstruct`
(`prototype/derisk.py:503`, docstring line 523: *"NO SR is run here"*) then **only consumes** that
cache at (a) the backbone anchors — I-frames + forced anchors, ~1-2 frames per 48-frame chunk —
and (b) occlusion-fallback pixels (`recon[occ] = perframe[occ]`, ~2-10% of pixels). The whole
NEMO thesis of this repo is *SR the sparse anchors, propagate the rest with codec MVs*. The
server pre-computes per-frame SR and then throws ~95% of it away.

**Evidence.**
- Code: `build_perframe_cache` loop over all N frames; `reconstruct`/`reconstruct_torch` upload
  `perframe_cache[i]` for every frame (`derisk.py:729`, `:753`) but only *use* it at anchors +
  `torch.where(occ, pf, recon)` fallback.
- Measured (this poke; `server/pipeline_api.py instant --max-frames 24`, 640×320→2560×1280,
  single chunk, GPU contended): `t_sr = 3.95 s` (**47.5%** of `t_total = 8.31 s`) = 164.6 ms/fr;
  `t_recon = 1.51 s` (18.2%) = 62.9 ms/fr; other = 2.85 s (34.3%) = 118.8 ms/fr; 346 ms/frame.
- On-disk profile `prototype/out_profile_C/summary.txt`: "SR per call: median 143 ms" but
  "anchors=1/48 (2.1%), **amortized SR = 2.99 ms/frame**". So the deployable architecture wants
  ~3 ms/frame of SR; the server spends ~145-165.

**Proposed fix.** Build the per-frame cache **lazily / partially** to match what `reconstruct`
actually consumes:
- Run the SR net only on backbone **anchors** (the frames `reconstruct` promotes anyway).
- For occlusion-**fallback** pixels, use a cheap source: bicubic (`cv2.resize`, ~1 ms) by
  default, and escalate to a real (compact) SR pass **only when the per-frame fallback fraction
  exceeds a tau** — exactly the adaptive policy already implemented for the occlusion mask
  (`ADAPTIVE_TAU`) and for re-anchoring. Low-motion talking-head (fallback ~2-3%) → bicubic
  fallback is invisible; high-motion (fallback 10-40%) → escalate.
- Plumb this as a per-mode flag (e.g. `sr_perframe=False`) so the byte-identical numpy default
  and the prototype regression are untouched.

**Win.** Removes ~150-160 ms/frame (~45% of instant wall time): ~346 → ~190 ms/frame, ~1.8×,
purely from this. It is the prerequisite for real-time / progressive playback. **Necessary but
not sufficient** for true 25 fps (see P2) — but it is the largest and most obviously-correct
lever, and it ships the validated "deployable amortized" path the server currently bypasses.

**Effort.** Medium. The consumer side (`reconstruct`) already only touches anchors + fallback,
so no warp/mask/blend math changes — this is a cache-builder + fallback-source + adaptive-
escalation change, plus a regression check that numpy default stays byte-identical.

---

### P2 — ✅ DONE (`10de07e`/`5e83ec1`/`2b46fc6`/`d90055e`) — [impact HIGH · effort MEDIUM-LOW] Server pays CPU encode + per-frame transfers + CPU grain on top of recon
> **Done (this is the real-time stack):** HW `h264_videotoolbox` encode + reactive mask for instant +
> GPU grain (`server/fast_grain.py`) + GPU-resident recon (`download_output=False`, contiguous-HWC
> download) + pipeline-parallel decode/encode (`server/pipe_encode.py`) + batched `hole_frac` sync, then
> a **720p instant tier** (`INSTANT_SCALE=2`). Together with P1: instant **~409 → 41 ms/frame = 10.0× =
> ~24 fps = real-time**. The QHD warp floor (~91 ms ≈ 4.5×) is why the 720p tier was the unlock.

**Issue.** Even with P1, "other" is ~119 ms/frame. The server runs `reconstruct(..., download_output=True)`
(full HD readback every frame ~11 ms), uploads `perframe_cache[i]` every frame (~9 ms),
applies grain on CPU at 3.3 MP, and encodes with **libx264 crf 18** at 2560×1280 (CPU, default
preset). The Step 6/7 profile shows GPU-resident recon is ~40 ms vs ~65 with these transfers.

**Evidence.** `pipeline_api._VideoWriter` uses `libx264`, `crf 18`; `download_output=True` in both
`process_clip` reconstruct calls; `_grain.apply_grain` runs on the downloaded numpy frame.
Profile `out_profile_C`: "transfers removed: 83 → 59 ms"; "reactive floor 41 ms (24 fps)";
"torch reactive [deployable] 38.7 ms" vs "adaptive 53.8 ms". Output files are 25-40 MB/150 fr
(crf 18 is heavy).

**Proposed fix (independent sub-levers, all small):**
- Encoder: switch to HW `h264_videotoolbox` (Apple) or at least `preset=veryfast` / a higher
  crf for instant — the biggest single chunk of "other".
- Mask: instant mode currently uses `occ=adaptive`; the profile shows **reactive == full quality
  on low-motion talking-head** and is ~15 ms cheaper — use `reactive` for instant, keep
  adaptive/full for high-motion (the handoff's own recommendation).
- Move grain to GPU (torch) so the recon need not round-trip to CPU just to be grained; keep
  recon GPU-resident and feed the encoder without the per-frame full readback where possible.

**Win.** Encode + grain ~90 ms → ~20-30; transfers ~20 ms removable; reactive saves ~15 ms on
instant. Combined with P1, brings instant materially toward play-while-processing.
**Effort.** Medium-low — encoder preset/codec is a 1-3 line change; reactive mask is a config flip;
GPU grain is a port of `grain.py`.

---

### P3 — ◻ OPEN — [impact MEDIUM · effort LOW] fp16 for the SR net itself (not the mask) is untested
> **Status:** still open. Lower priority now that instant SR is anchor-only/amortized; most relevant to
> the quality-mode x4plus anchor (~2.2 s). Cast the SR net + input to fp16 on MPS and A/B sharpness/PSNR.

**Issue.** Step 7 measured fp16 on the **mask** ops and rejected it (kernel-launch-bound,
`scatter_add` slower in fp16). The **SR network** (compute-bound conv net) was not tested in fp16
and would likely gain ~1.5-2×. Matters for the quality-mode x4plus anchor (~2.2 s) and for any
fallback SR pass; less so once instant SR is anchor-only (P1) and already amortized.
**Fix.** Cast the SR net + input to fp16 on MPS; A/B sharpness/PSNR to confirm no quality loss.
**Effort.** Low (a cast), but validate quality before shipping.

---

### Real-time / progressive feasibility (summary)
- The handoff's **~38-40 ms/frame (25 fps)** figure is the *deployable recon-only* path
  (GPU-resident, no per-frame SR, no CPU grain, no libx264). The server is ~10× slower because
  it stacks five costs the deployable path excludes: per-frame SR (P1, the big one), HD
  readback + perframe upload (P2), CPU grain (P2), CPU libx264 (P2), and per-chunk warmup/`empty_cache`.
- Realistic optimized "instant": P1 alone → ~190 ms/frame (~5 fps process rate). P1 + P2 (HW
  encode + GPU grain + reactive mask + GPU-resident) → plausibly ~60-90 ms/frame for the full
  produce-an-mp4 path; the pure recon hot loop is already 40 ms. That makes **play-while-
  processing viable** (process faster than the buffer drains, start playback after a lead buffer)
  even if a literal 25 fps end-to-end with simultaneous encode is tight on one GPU.
- Verdict: P1 is the gateway; P2 is what actually crosses into "play while processing"; neither
  requires new algorithms — both are wiring the server onto paths the prototype already validated.
- ✅ **OUTCOME (improvement loop):** P1 + P2 shipped, plus a **720p instant tier** that the original
  poke did not anticipate, taking instant **all the way to ~41 ms/frame = 10.0× = ~24 fps = real-time**
  (better than the ~60–90 ms estimated here). Progressive play-while-processing is now the open lever
  (instant keeps up with playback at 24 fps).

---

## VISUAL QUALITY

### V1 — ✅ DONE (`0e97c9e`) — [impact HIGH · effort LOW] Per-frame grain destroys the layered mode's stable-background win
> **Done:** `pipeline_api._run_layered` bakes ONE fixed grain into each scene's static plate
> (`_PLATE_GRAIN_SEED=12345`) and applies per-frame grain only to the alpha-gated foreground →
> background **|ΔF| 4.5 → 0.000**; the ~167× steadier background is finally visible. Layered is no
> longer strictly dominated by quality mode.

**Issue.** Layered mode's *only* advantage over quality mode (it is 4-5× slower and uses a
non-commercial matte) is a rock-stable, denoised background plate. The plate is sampled
identically every frame (static camera) → bare composite |ΔF| = 0.001. But grain is added as a
final pass over the **whole** frame (`_run_layered`: `_grain.apply_grain(out, done, "med")`),
including the static plate, so the shipped background is no steadier than instant's.

**Evidence.**
- Prototype claim: `out_layered`, bg |ΔF| = 0.001 vs x4plus 0.167 (~167×), and 112% of x4plus bg
  sharpness — the mode's whole selling point.
- Shipped `server/outputs/short_layered.mp4`, talking-head frames 60-63, background corners
  (mid-tone luma 124-141): consecutive |Δ| = **4.56 / 4.98 / 3.44**. Identical to
  `short_instant.mp4` (4.52 / 5.02 / 3.64) and `short_quality.mp4` (4.57 / 4.98 / 3.47).
- Grain "med" σ = 5.0 at mid-tone (`grain.py:STRENGTHS`); two independent fields differ by
  ~σ·1.1 ≈ 5.5 — i.e. essentially all of the 4.5-5/frame is grain. The plate is ~4500× steadier
  underneath but invisible.
- Amplified diff crops (`/tmp/playhd_frames/th_bg_lay_diff12x.png`,
  `…_inst_diff12x.png`) show pure uniform grain speckle, no warp structure — the background flicker
  is grain in both modes.

**Proposed fix.** Gate grain by the foreground alpha (already computed per frame as `pha`): full
per-frame grain on the foreground, and on the static background use either **frozen grain** (same
seed every frame → still dithers banding on the smooth bokeh plate, zero temporal flicker) or no
grain. `grain.apply_grain` already supports `return_grain=True`; multiply the grain field by
`pha_hd` (FG) + a frozen field by `(1-pha_hd)` (BG) before adding. ~10 lines.

**Win.** Restores the 167×/1700× background-stability advantage that the mode exists to deliver,
at ~zero cost. Without this, layered mode is strictly dominated by quality mode (slower, halos,
non-commercial matte, no visible benefit). This is the single highest-value cheap visual fix.
**Effort.** Low.

---

### V2 — ◻ OPEN — [impact MEDIUM · effort LOW] Grain re-adds flicker over otherwise-stable propagated regions (all modes)
> **Status:** still open for instant/quality (V1 fixed only layered's plate). The motion-gated /
> frozen-on-static grain idea below is unchanged; reuses the region-aware motion gate already in quality.

**Issue.** The same mechanism as V1 in the non-layered modes: on low-motion content the
propagated background is near-static, and per-frame grain re-injects ~4.5/frame of flicker (raises
tOF). Grain is filmic and desirable on detail/motion, but uniform full-strength grain over static
regions trades the temporal stability the propagation pipeline worked to achieve.
**Fix.** Default instant to grain "low" (σ 2.5), or motion-modulate grain amplitude using the
**region-aware motion gate already computed** in quality mode (more grain on moving/FG, less on
static) — reuses existing machinery. Frozen-grain-on-static (as in V1) also applies here.
**Effort.** Low (config + reuse the motion gate).

---

### V3 — ◻ OPEN — [impact MEDIUM · effort MEDIUM] High-contrast graphics/text edges shimmer under propagation
> **Status:** still open. (Scene-cut detection, iter 1, helps the *cut* into a title card but not the
> per-frame edge shimmer *within* a held graphic — that still needs per-graphic pinning/freezing.)

**Issue.** Title cards, lower-thirds and captions (sharp high-contrast edges on flat fields) shimmer
frame-to-frame under MV warp. Observed on `short_*` frames 20-24 (the "USACHEV TODAY" title card):
the amplified diff (`/tmp/playhd_frames/lay_TLbg_diff8x.png`) shows the letter edges flickering
strongly — distinct from grain, which is suppressed on the black field.
**Fix.** Detect high-contrast + low-MV graphic regions and pin them to per-frame SR (or freeze
them) rather than propagating warped edges; the region-aware static branch should cover these but
may currently let them propagate. Could also anchor more aggressively when a graphic overlay is
detected.
**Effort.** Medium (detection + gating). Common content type, so worth it.

---

### V4 — ✅ DONE (R2-E3, `LAYERED_SEAM_FIX`) — Layered matte-edge halo / hair
> **Done (research round R2):** diagnosed as a SOFT-BACKGROUND problem (the near-subject plate ring is
> real background softened by low coverage + matte-edge contamination, only ~8% inpaint — so "better
> inpaint" was a NO-GO). Fix = a band-localized plate-ring sharpness RESTORE baked once per scene
> (`layered_pipeline.restore_plate_ring`, `layered_api` PASS A) + an alpha-aware FEATHER per frame
> (`feather_alpha`, PASS B): x4plus seam ratio 5.02→3.22 (= uniform-x4plus ceiling), halo 11.7→7.7px,
> subject core EXACTLY unchanged, ~+5 ms/frame. Softening the FG edge was REJECTED (smears the subject).
> `lp.composite` defaults stay byte-identical. See `experiments/r2_e3_seam/REPORT.md`.

**Issue.** Already characterized (handoff L4, `out_layered/seam_*`): faint hairline/jaw rim halo
from the FG/BG sharpness discontinuity (5.1-5.8× vs uniform-x4plus 3.4×); fine hair wisps lost;
worst exactly at the inpainted always-occluded ring (plate sharpness 8.7 vs 15.3 there).
**Fix.** Wider alpha feather at hair; match FG/BG sharpness near the seam (slightly soften the FG
edge or sharpen the plate band adjacent to the matte); better hole fill than Telea inpaint at the
ring.
**Effort.** Medium. Lower priority — subtle, and layered is an optional mode (gated behind V1
being worth shipping at all).

---

### V5 — [NO-GO, documented, no action] Diffusion anchor hallucinates on real H.264
`out_diffusion_real`: SD-x4-upscaler invents fake skin/weave texture on real compressed SD
(trained on clean bicubic LR, reads H.264 noise as signal). var-Lap looks 13× sharper but it is
fabricated. x4plus stays the max-quality anchor. No action; recorded so it isn't re-litigated.

### V6 — [impact LOW · effort already-addressed] x4plus heavy-anchor texture is less temporally stable
x4plus's hallucinated HF detail flickers more under warp on low motion (tOF 0.66 vs compact 0.33).
The `--region-aware` gate already mitigates by confining heavy detail to static regions
(`out_region/region_split.png`: ra-wide recovers 95% of static detail at compact-floor tOF). Keep
region-aware on for quality mode (it is). Monitor; no new work.

---

## Suggested order of attack (impact × effort)
1. ✅ **V1** — grain gating in layered — **DONE** (`0e97c9e`).
2. ✅ **P1** — anchor-only SR in instant — **DONE** (`10de07e`); ended up the gateway to the full 10×.
3. ✅ **P2** — HW encode + reactive mask + GPU grain + drop transfers (+ pipeline parallelism + the
   720p tier) — **DONE** (`10de07e`/`5e83ec1`/`2b46fc6`/`d90055e`); instant is now real-time (24 fps).
4. **Progressive play-while-processing** — now the top open lever (instant keeps up with playback).
5. **V2** — motion-modulated / lower grain on static regions (LOW; reuses the region gate).
6. **V3** — pin high-contrast graphic overlays (MEDIUM; common content).
7. **P3 / V4 / V6** — fp16 SR net; layered seam/hair; monitor x4plus stability (lower priority).
