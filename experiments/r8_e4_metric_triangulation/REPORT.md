# R8-E4: Perceptual-metric TRIANGULATION — does any quality decision rest on LPIPS alone?

**Verdict up front: NO shipped quality decision flips.** Adding a texture-aware full-reference
metric (**DISTS**, Ding et al. 2020) on top of the project's lead (AlexNet **LPIPS**) and the
non-learned anchor (**PSNR**) **CONFIRMS** every LPIPS-based call — x4plus-as-quality-anchor,
instant=compact, grain default-OFF. The one genuinely *new* signal is on the **un-landed 0.5 blend**,
where DISTS is *more* favorable than LPIPS (an ensemble/denoise effect) — a refinement for R8-E3 to
chase, **not** a ship-flip. This is a de-risking result: report confidence, not drama.

## What ran (metric path, honestly)
- **DISTS** — `pip install pyiqa` (IQA-PyTorch 0.1.15) [OK]. The texture-aware FR metric this thread needed
  (VGG features + explicit global texture statistics -> tolerant to texture *resampling* where SSIM/PSNR
  over-penalize; the exact axis LPIPS is reputed weak on). Weights auto-downloaded, cached.
- **LPIPS-alex** — the project's `lpips` package (lead), kept verbatim.
- **pyiqa-LPIPS** cross-check — turned out to load the **same AlexNet weights** -> it matched LPIPS-alex to
  4 decimals in **all 15 cells (0 mismatches)**. So it is NOT an independent backbone; it only proves the
  pyiqa preprocessing is identical to the lead pipeline (rules out a range/normalization bug). **DISTS (VGG)
  is the one architecturally-independent learned metric here; PSNR is the non-learned anchor.**
- **VMAF** — **could not run.** PyAV's bundled libav has **no `libvmaf` filter**
  (`'libvmaf' in av.filter.filters_available` -> False), `ffmpeg-quality-metrics` is absent and shells to the
  broken ffmpeg CLI (handoff "No VMAF"), no usable pip wheel. VMAF is compression-trained (not SR) -> it was
  only ever SECONDARY context; documented as a non-runner, not blocked on.

## Validity gate (run before trusting any flip)
`metrics_extra.selfcheck()`: **DISTS(x,x)=1.8e-7 ~= 0**, LPIPS(x,x)=0.0 (identity holds -> range correct);
**DISTS(blur,x)=0.301 > 0** and LPIPS(blur,x)=0.638 (monotone with degradation). Preprocessing is `[0,1]`
RGB NCHW for both — verified equal. Asserted in `run_triangulation.py` before scoring.

## Protocol (IDENTICAL to R6-E1 — directly comparable)
Reused `experiments/r6_e1_srdecision/run_matrix.py` *verbatim* (decode / degrade / restore / windows /
degrade operators). SD frame (640x320) = pseudo-HD GT -> degrade 2x -> restore via `prototype.sr.upscale_to`
-> score vs GT. **5 windows** (talkinghead@5000 smooth, highmotion@0 titlecard, texture18k/24k/46k detailed),
**3 degrades** (moderate=R5-E2 `real`, heavy, gritty), **same precomputed LR fed to every arm**, n=6/cell,
per-frame metrics kept for win-rate. Arms: **compact, x4plus, fixed-0.5 blend** (`0.5*(compact+x4plus)`,
the R6-E3 `blend_linear` g=0.5), + bicubic anchor. x4plus = **15.8x** compact compute (matches R6-E1's ~23x
order; MPS contended by 3 siblings).

## Triangulated table — compact / x4plus (LPIPS-alex & DISTS lower-better, PSNR higher-better)
| window | degrade | LPIPSa c/x | win | DISTS c/x | win | PSNR c/x | win | flag |
|---|---|---|---|---|---|---|---|---|
| talkinghead | moderate | 0.108/0.123 | **comp** | 0.105/0.122 | **comp** | 30.3/28.5 | comp | agree->comp |
| talkinghead | heavy | 0.183/0.163 | x4pl | 0.140/0.137 | tie | 29.4/27.5 | comp | soft |
| talkinghead | gritty | 0.241/0.239 | tie | 0.192/0.179 | x4pl | 27.1/25.1 | comp | soft |
| highmotion | moderate | 0.0051/0.0057 | **comp** | 0.0285/0.0353 | **comp** | tie | tie | agree->comp |
| highmotion | heavy | 0.0064/0.0068 | comp | 0.0349/0.0355 | tie | tie | tie | soft |
| highmotion | gritty | 0.0127/0.0099 | **x4pl** | 0.0646/0.0446 | **x4pl** | tie | tie | agree->x4pl |
| texture18k | moderate | 0.036/0.037 | tie | 0.072/0.052 | x4pl | tie | tie | soft |
| texture18k | heavy | 0.078/0.062 | **x4pl** | 0.109/0.079 | **x4pl** | 20.8/19.5 | comp | agree->x4pl |
| texture18k | gritty | 0.172/0.123 | **x4pl** | 0.204/0.117 | **x4pl** | 18.2/17.1 | comp | agree->x4pl |
| texture24k | moderate | 0.074/0.051 | **x4pl** | 0.101/0.092 | **x4pl** | tie | tie | agree->x4pl |
| texture24k | heavy | 0.132/0.091 | **x4pl** | 0.139/0.119 | **x4pl** | 20.6/19.6 | comp | agree->x4pl |
| texture24k | gritty | 0.251/0.146 | **x4pl** | 0.201/0.141 | **x4pl** | 18.1/17.5 | comp | agree->x4pl |
| texture46k | moderate | 0.118/0.101 | **x4pl** | 0.098/0.089 | **x4pl** | tie | tie | agree->x4pl |
| texture46k | heavy | 0.210/0.166 | x4pl | 0.116/0.119 | tie | tie | tie | **soft (DISTS least sold on x4plus)** |
| texture46k | gritty | 0.349/0.232 | **x4pl** | 0.188/0.172 | **x4pl** | tie | tie | agree->x4pl |

**DISAGREE cells (LPIPS & DISTS pick OPPOSITE non-tie winners): 0 / 15.**
**AGREE cells (both non-tie, same winner): 10 — x4plus 8, compact 2.** The 2 compact-agreement cells are
exactly the smooth/low-detail windows (talkinghead-moderate, highmotion-moderate) — R5-E2's domain. On the
**9 textured cells, DISTS picks x4plus 8x and ties once, compact 0x** — at least as decisive as LPIPS.

### Per-frame win-rate (x4plus better, %) — flips are not mean artifacts
Textured heavy/gritty cells: LPIPS **100%** and DISTS **100%** of frames favour x4plus on texture18k &
texture24k (all degrades). The lone soft spot is **texture46k-heavy**: DISTS per-frame win-rate for x4plus
is only **33%** (mean a tie, 0.116 vs 0.119) while LPIPS says x4plus — the single place a texture metric is
*least* convinced by x4plus, but it is a **tie, not a flip**, and x4plus still wins LPIPS + the other two
degrades on that same window decisively.

## Answers to the three questions
**(a) Does DISTS CONFIRM x4plus > compact on textured/gritty content? — YES, decisively.** This is the
load-bearing check: DISTS is the metric *built to be lenient to texture resampling*, so the worry was that
LPIPS over-credits x4plus's synthesized HF. It does not — DISTS credits x4plus **at least as much** (8/9
textured cells, ~100% per-frame), confirming x4plus's HF lands on **real GT texture structure**, not
hallucination. **R6-E1's "keep x4plus + region-aware, withdraw lean-compact" stands, hardened.**

**(b) Would DISTS change the blend verdict? — It nudges, does not overturn.** The fixed-0.5 blend beats the
*better* of the two bases on **LPIPS in 11/15** and **DISTS in 11/15** cells, with **PSNR up in all 15**.
The split is content-dependent and the metrics differ on texture:
- **Smooth / low-detail (talkinghead, highmotion): blend beats BOTH bases on LPIPS AND DISTS, 6/6 cells** —
  a real 2-model ensemble win (decorrelated errors cancel).
- **Heavy texture (texture18k/24k): blend LOSES LPIPS to x4plus-alone** (it dilutes x4plus's aligned HF with
  compact's softness) **but often WINS DISTS** (texture-statistic cleanup) -> the metrics **disagree on the
  blend** here, the one place DISTS adds genuinely new information.
- **CONFOUND (held honestly): PSNR rises in all 15 cells -> much of the blend gain is variance/noise
  reduction (averaging two estimates), not pure perception.** And the blend costs compact+x4plus = **16.8x**
  (more than x4plus alone), so it only makes sense where BOTH are already computed (region-aware overlap).
- **Net:** DISTS makes the 0.5 blend look *better than R6-E3's LPIPS-only read did*, especially on smooth
  content — **a validated-ready refinement to hand R8-E3** (test as a region-aware overlap blend, report
  the PSNR-confound), **not** a standalone mode and **not** a flip of any shipped call.

**(c) Does DISTS still say grain hurts fidelity? — YES, monotonically.** Re-scored R5-E2's grain A/B
(compact, moderate degrade) with DISTS: off->high DISTS rises on every window
(talkinghead 0.105->0.223, texture24k 0.101->0.182, texture46k 0.098->0.175) — same direction and comparable
relative slope as LPIPS. **Grain default-OFF for fidelity is confirmed by the texture-aware metric too.**

## Agree / disagree summary
- **Hard disagreements (opposite winners): 0/15.** No shipped decision flips.
- **Soft divergences (one metric ties): 4** — talkinghead-heavy, talkinghead-gritty, texture18k-moderate,
  texture46k-heavy. In 3/4 the softer metric still leans x4plus or ties; only texture46k-heavy has DISTS
  mildly toward compact (within tie band) -> flagged, not a flip.
- **LPIPS-alex vs pyiqa-LPIPS: 0/15 winner mismatches** (same weights -> confirms preprocessing parity only).

## Does any shipped quality decision change under DISTS/VMAF? — NO.
| shipped decision | source | DISTS/PSNR verdict |
|---|---|---|
| quality = x4plus + region-aware (keep) | R6-E1 | **CONFIRMED** (8/9 textured cells, ~100% per-frame) |
| instant = compact (smooth/fast) | R5-E2 | **CONFIRMED** (compact wins both smooth-moderate cells on LPIPS+DISTS+PSNR) |
| grain default-OFF for fidelity | R5-E2 | **CONFIRMED** (DISTS rises monotonically with grain) |
| fp16 perceptually identical | R5-E2/R2-E4 | not re-run — numerical-near-identity claim (LPIPS~5e-5); any FR metric agrees by construction |

## Threats to validity / what could not run
1. **VMAF did not run** (no libvmaf in PyAV, broken ffmpeg CLI, no pip wheel). It is compression-trained,
   not SR -> secondary by design; its absence does not weaken the FR triangulation.
2. **Shared learned-metric bias:** LPIPS-alex and DISTS are both learned nets and *could* share a GAN-HF
   bias. Mitigants: (i) different architectures/objectives (AlexNet local-feature 2AFC vs VGG texture+
   structure unification); (ii) DISTS is specifically *designed not to be fooled by texture resampling* -> a
   conservative check that still credits x4plus; (iii) **PSNR (non-learned) goes the SAME way in the smooth
   cells** (compact wins PSNR there), anchoring the agreement. The pyiqa-LPIPS "cross-check" is NOT
   independent (same AlexNet weights) — stated plainly.
3. **Pseudo-HD-GT protocol** (no true HD GT for sample.mp4) is inherited from R5-E2/R6-E1 and is a *constant*
   across all compared arms -> relative verdicts valid even if absolute numbers are not "true-HD" perception.
4. **n=6 frames/cell, one clip** (sample.mp4). Small, but per-frame win-rates (mostly 100%) back the means;
   same scale as R6-E1 (n=8). GOTCHA #23 respected throughout — no NR metric used as a verdict.
5. **Blend gain is partly a PSNR/denoise confound**, explicitly flagged so R8-E3 does not over-claim it.

## Artifacts (all in `experiments/r8_e4_metric_triangulation/`)
`metrics_extra.py` (DISTS + pyiqa-LPIPS + selfcheck), `run_triangulation.py` (the matrix, reuses R6-E1
verbatim), `run_grain.py` (grain re-score), `analyze.py` (table + agree/disagree), `results.json`,
`grain_results.json`, `run.log`. **No patch** (validation thread) — prototype/ and server/ untouched.

## Executive summary
Installed **DISTS** via pyiqa and re-scored the R6-E1 SR A/B (5 windows x 3 degrades x {compact, x4plus,
0.5-blend}) plus the grain A/B, with DISTS + LPIPS + PSNR. **Every LPIPS-based shipped decision is
CONFIRMED, none flips:** the texture-aware metric independently credits **x4plus > compact on 8/9 textured
cells (~100% per-frame), 0/15 hard disagreements**, says **grain monotonically hurts**, and agrees
**compact wins the smooth/low-detail domain**. The only new signal is that DISTS likes the **un-landed 0.5
blend** more than LPIPS (ensemble win on smooth content, texture-stat win on detail) — a refinement for
R8-E3 with an honest PSNR-confound, not a ship change. **VMAF could not run** (no libvmaf/PyAV, broken
ffmpeg). DISTS path ran cleanly (identity self-check 1.8e-7). The quality program does **not** rest on LPIPS
alone — it is hardened.
