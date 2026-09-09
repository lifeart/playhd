# R13-E3 — the jelly round: FSR3.1 luma instability + per-pixel motion-adaptive EMA

**The question owned:** does FSR3.1's luma-instability mechanism plus a per-pixel
motion-adaptive EMA fix the tabled temporal "jelly"/"желе" wobble — and can it finally be
measured with a number instead of vibes?

## Headline

| | verdict |
|---|---|
| **Is the project's jelly DRIFT or FLICKER?** | **FLICKER — measured, on both windows.** The temporal error the pipeline ADDS over the content floor is *flat* in frames-since-anchor (calm: slope +0.0002 codes/frame, R²=0.000; high-motion: +0.0059/frame, R²=0.081, trend +0.225 codes ≈ the frame-to-frame noise σ 0.224) and it is *concentrated* in the pixels whose source flips between the propagated warp and the fresh per-frame SR: **12.4× the per-pixel flicker of a stably-propagated pixel on calm, 3.5× on high-motion**. Drift is not the problem. The **hard occlusion switch** is. |
| **M1 — FSR3.1 luma instability factor** | **NO-GO as a lever.** Standalone it is inside the noise (−3.0% excess flicker with FSR's own velocity gate, +0.3% LPIPS); on top of M2 it buys a little extra steadiness but is worse than plain M2 on tLP and LPIPS. Three structural reasons it does not transfer (§3). **Kept as a free diagnostic** — the port is a working per-pixel oscillation detector. |
| **M2 — per-pixel motion-adaptive EMA** | **GATED-GO** — the first measured jelly reduction in the project: **excess MC-flicker −37.5% (α=0.5) / −57.6% (α=0.75)** on high-motion, with **LPIPS and DISTS vs the x4plus ceiling BOTH improving** (−3.9% / −1.5%) and sharpness held (var-Lap −1.1%). On calm the win is smaller but the same sign: **−20.5% / −29.4% excess flicker**, tLP −12.1%, LPIPS −0.8%, DISTS −0.7%. Cost: **tOF_true +1.7% (jelly) / +5.2% (calm)** — the same currency as R12-E1's +5.8%, so the same graduation path. Landed **default-OFF** (`stabilizer.STAB_ENABLED = False`; OFF-path parity **0 codes** on both windows). |

Measured on real H.264 with the **same clip and windows as R12-E1** (`web_spike/sd600.mp4`,
jelly = frames 582–621 mean|MV| 6.58, calm = 437–476), the same scale-3 numpy reconstruction,
compact anchors, x4plus decoupled arbiter — so the numbers compose with the residual-gate round.
Cross-check: this round's `baseline_hard` reproduces R12-E1's baseline LPIPS to 4 decimals
(0.0353 both).

---

## 1. The measurement problem (half the value of the round)

The repo had exactly one temporal metric, **tOF** (Farneback flow deviation), which is blind to
wobble that leaves the flow field intact. Two were added, both from tools already in the repo —
**no RAFT, no optical-flow budget**:

**tLP** (`jelly_metrics.tlp_series`) — TecoGAN's perceptual temporal metric, the sibling of tOF:

```
tLP_t = | LPIPS(out_{t-1}, out_t) − LPIPS(ref_{t-1}, ref_t) |
```

the *excess perceptual change* between adjacent output frames over the change the content actually
made. `ref` = bicubic(decoded LR), the real-footage stand-in for content motion (the convention
R12-E1 used for `tOF_true`). LPIPS already runs here (R5-E2 `metrics.py`).

**MCLF** (`jelly_metrics.mclf_series`) — motion-compensated luma flicker, warping the previous
**output** luma into the current frame's grid **with the existing codec MVs**:

```
MCLF_t = mean_valid | Y_t − warp(Y_{t−1}, flow_{t→t−1}) |            (codes, 0..255)
```

Building `flow_{t→t−1}` from codec MVs is not trivial on this stream: a frame's MVs point at its
*reference anchor*, not its display predecessor, and B-frames here are **future-dominant**
(handoff Step 1). `mc_flow_prev` therefore has four cases — (A) the predecessor *is* the reference
anchor → use the field directly; (B) both hang off the same past anchor → compose `f_i` with the
**splat-inversion** of `f_{i−1}` (`derisk.softmax_splat`, the machinery the Ruder fwd-bwd test
already uses); (C) the same through the *future* anchor; (D) the current frame *is* the
predecessor's future anchor → pure inversion. Union coverage: **72.2%** of pixels on jelly,
**95.2%** on calm, 39/39 adjacent pairs both (past-chain alone gave 10–50% — cases C/D are what
make the metric usable). Uncovered pixels are excluded, never guessed.

**Honesty guard (as briefed):** MCLF is **not zero for ground-truth video** — codec MVs are
rate-distortion vectors. It is used only as a **relative** number, always beside two calibrating
sequences measured with **identical flow fields**:

| window | bicubic(LR) = content+codec floor | propagated (shipping) | per-frame compact | per-frame x4plus |
|---|---|---|---|---|
| jelly | **3.666** | 3.990 (**excess +0.324**) | 4.172 (+0.506) | 4.208 (+0.543) |
| calm | **0.815** | 1.073 (**excess +0.258**) | 1.255 (+0.440) | 1.767 (+0.952) |

Two things fall straight out: (i) **most of the raw flicker is the content itself** — the pipeline
adds 8.8% (jelly) / 31.7% (calm) on top of the floor; (ii) **propagation is already steadier than
per-frame SR** (excess 0.324 vs 0.543 on jelly), which re-confirms the project's core thesis on a
metric it had never been measured with.

---

## 2. The diagnostic: drift or flicker?

`diag_extra.py` + `analyze.py` on the shipping reconstruction, every series regressed against
**frames-since-anchor**:

| window | series | slope /frame | R² | detrended σ | trend over window | mean |
|---|---|---|---|---|---|---|
| jelly | MCLF raw | +0.0555 | 0.096 | 1.916 | +2.109 | 3.990 |
| jelly | **MCLF excess over content** | **+0.0059** | **0.081** | 0.224 | **+0.225** | **0.324** |
| jelly | MCLF high-freq | +0.0222 | 0.119 | 0.678 | +0.843 | 2.069 |
| jelly | flip fraction | +0.0024 | 0.013 | 0.243 | +0.092 | 0.321 |
| calm | MCLF raw | +0.0081 | 0.032 | 0.504 | +0.309 | 1.073 |
| calm | **MCLF excess over content** | **+0.0002** | **0.000** | 0.143 | **+0.009** | **0.258** |
| calm | MCLF high-freq | +0.0026 | 0.014 | 0.249 | +0.099 | 0.633 |

**FLICKER, not drift.** On calm the excess is *perfectly* flat (R² 0.000): everything the pipeline
adds is there at frame 1 after the anchor and never grows. On high-motion there is a weak upward
trend, but it is the size of the frame-to-frame noise and explains 8% of the variance. The *raw*
MCLF does rise with distance — that rise is **content**, and it disappears when the bicubic floor
is subtracted. (Plot: `out/diagnostic_drift_vs_flicker.png`.)

**The control that settles it.** Run the same regression on sequences that contain *no propagation
at all* and the slope barely moves: MCLF slope/frame is **0.0496 for bicubic(LR), 0.0533 for
per-frame x4plus, 0.0552 for per-frame compact, 0.0555 for the propagated pipeline**. A per-frame
SR sequence cannot drift — it has no history — so a slope that is ~identical across all four is the
*content* getting more energetic later in the window, not the pipeline accumulating error. The same
control applies to tLP (slope 0.00022 propagated vs 0.00016 per-frame compact, 0.00008 per-frame
x4plus): the rise is mostly shared with the no-history sequences.

This matters because the two failure modes want opposite fixes: drift wants more anchors /
re-anchoring — which R1-E2b already measured as a NO-GO — while flicker wants temporal
regularisation. The project has been guessing; the settled re-anchoring NO-GO is retrospectively
consistent with the answer.

### Where the flicker comes from — the hard occlusion switch

`R[i]['mask']` is the set of pixels taken from fresh per-frame SR instead of the propagated warp.
Motion-compensating the previous frame's mask into the current grid and counting state changes:

| window | fresh-SR mask | pixels that FLIP source per pair | flicker of the flip set | of stably-propagated | ratio | flip share of flicker energy (area share) |
|---|---|---|---|---|---|---|
| jelly | 28.0% | **32.1%** | 8.62 codes | 2.43 | **3.5×** | **49.9%** (32.1%) |
| calm | 4.8% | **4.3%** | 10.75 codes | 0.87 | **12.4×** | **20.0%** (4.3%) |

On calm, **4.3% of pixels carry 20% of the temporal energy**. That is the jelly: a per-pixel hard
switch between two *different image sources* (propagated SR detail vs freshly-SR'd detail) that
toggles frame to frame wherever the residual hovers near the occlusion threshold.

On high motion a second, structural mechanism appears: the **P-frames sit at 62.6% fresh-SR fill**
(their reference is 4 display-frames back) while the B leaves between them sit at 17.6% — so the
frame's *source composition* swings at the GOP cadence:

```
excess flicker by GOP phase (frames-since-anchor mod 4)
  jelly:  phase0(P) +0.254   phase1 +0.030   phase2 +0.585   phase3 +0.422    <- 4-frame pulse
  calm:            +0.274          +0.267          +0.252          +0.242     <- flat
```

A 4-frame periodic modulation at 25 fps is a **6.25 Hz pulse** — squarely in the band the eye reads
as "wobbling jelly" — and it is absent on the calm window, matching the subjective report that
jelly shows up on motion.

---

## 3. M1 — FSR3.1 luma instability factor

**Ported faithfully** in `stabilizer.py::luma_instability_factor` from the MIT source
(`ffx_fsr3upscaler_luma_instability.h::ComputeLumaInstabilityFactor`): 4-deep reprojected luma
history; the pixel is flagged when the current luma deviates from `L[n−1]` in the same direction as
from an older `L[i]` **and** is more similar to that older frame — the value keeps returning =
oscillation. `MinDividedByMax` semantics, the sign test, the `sim0 < 1` early-out and the
"history must be full" test are reproduced (the magic `L[n−4] != 0` becomes an explicit depth
counter). FSR's gates are ported too: the `Get4KVelocity/20` velocity weight rescaled to this
pipeline's resolution, and `(1 − disocclusion)` fed from the pipeline's own occlusion mask. Body is
**depth-free**, as briefed; state is one 4-channel (RGBA8-equivalent) HD history.

**Result: NO-GO as a lever** (jelly window; % vs `baseline_hard`):

| variant | fires on | excess MCLF | tLP | LPIPS vs x4plus | tOF_true |
|---|---|---|---|---|---|
| `instab_fsr` (FSR's own velocity gate) | 1.6% of pixels | −3.0% | +0.6% | **+0.3%** | −0.4% |
| `instab_relaxed` (gate ×5) | 7.2% | −14.5% | +1.0% | −0.0% | −0.1% |
| `ema_instab_50` (M1 on top of M2) | 7.5% | −47.2% | −6.3% | −3.5% | +1.7% |

Standalone it is inside the noise. On top of M2 it does add something real — at α_mean 0.339 it
reaches −47.2% excess where plain M2 interpolated to the same α gives ≈ −39% — but it is
simultaneously *worse* than plain M2 on tLP (−6.3% vs −8.1%) and LPIPS (−3.5% vs −3.9%). For an
extra 4-deep HD history and a per-pixel branch, that is not a trade worth landing.

**Three structural reasons it does not transfer** (all measured, not assumed):

1. **The velocity gate is calibrated for game rendering.** FSR kills instability above 20
   4K-pixels/frame ≈ **3 LR px/frame** here. The jelly window runs at mean |MV| **6.58** — FSR's own
   gate switches the mechanism off exactly where this project's jelly is worst. Relaxing it 5×
   raised firing 1.6% → 7.2% and the effect −3.0% → −14.5%, confirming the gate (not the detector)
   was binding.
2. **There is no per-frame history to detect oscillation in.** FSR renders every frame, so
   `L[n−1..n−4]` are consecutive. Here the history can only advance along the **I/P backbone**
   (GOTCHA #12 forbids display-order chaining once B-frames appear), so a 4-deep history spans
   **16 display frames / 640 ms** on this stream — the wrong timescale for a 6.25 Hz pulse — and the
   B leaves, two thirds of the frames, cannot update it at all.
3. **In FSR the factor's only job is to bypass the colour-box clamp** (`RectifyHistory`:
   `lerp(clampedHistory, rawHistory, historyContribution)`). playhd has no clamp — R1-E4b settled
   that as a structural NO-GO — so the mechanism arrives with nothing to unlock. Its residual
   action, "raise the history weight where the pixel oscillates", is what M2 already does from a
   cheaper, better-conditioned signal.

**Keep it as a diagnostic.** The factor is a working per-pixel oscillation map from luma alone, and
it agrees with the hard-switch attribution (it fires 4× more often on the high-motion window). If
jelly localisation is ever wanted in the browser player, it is ~20 lines and one texture.

---

## 4. M2 — per-pixel motion-adaptive EMA

`stabilizer.py::stabilize`, an **output-only** post-pass (GOTCHA #16: the stabilised frame never
re-enters `R[]`; the stabiliser keeps its own history). The history advances along the I/P backbone
and is reprojected with the frame's own past-MV field — **the warp the pipeline already performs**.
B leaves are stabilised **bidirectionally** (past + future anchor histories, temporal-distance
weighted, mirroring `derisk`'s own B handling) because a past-only history has no MV over 50–80% of
a future-dominant leaf. The history **resets at every anchor / I-frame** (the R3-E3
cross-cut-ghost lesson).

```
err   = | downscale_AREA(warp(prev_output)) − decoded_LR_cur |        # LR codes
alpha = alpha_max · (1 − smoothstep(err_lo=2, err_hi=12, err));  alpha = 0 at intra holes
out   = (1−alpha)·recon + alpha·warp(prev_output)
```

Deliberately a **different signal from R12-E1's residual gate**: that gate measures a single-hop
`LR_cur vs warp(LR_ref)` codec residual and *selects a source*; this measures the **accumulated
output** against the decoded LR — so it also sees error the codec residual cannot — and modulates a
*temporal filter*. And per the brief: this is **per-PIXEL** and composes with (does not replace)
the fixed global β=0.85 anchor blend — it is not a re-run of R9-E1's refuted per-CLIP adaptive β.

### Results — jelly window (high motion, mean|MV| 6.58)

| sequence | MCLF | excess | Δexcess | tLP | ΔtLP | LPIPS vs x4 | ΔLPIPS | DISTS vs x4 | ΔDISTS | tOF_true | ᾱ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_hard` (shipping) | 3.990 | +0.324 | — | 0.0114 | — | 0.0353 | — | 0.0338 | — | 1.654 | — |
| `ema_fixed_50` (naive EMA) | 4.258 | +0.592 | **+82.5%** | 0.0207 | +81.8% | 0.0551 | +55.9% | 0.0655 | +93.7% | 1.861 | 0.402 |
| **`ema_adapt_50` (M2)** | 3.868 | +0.203 | **−37.5%** | 0.0105 | −8.1% | **0.0340** | **−3.9%** | **0.0333** | **−1.5%** | 1.681 | 0.324 |
| **`ema_adapt_75` (M2)** | 3.803 | +0.138 | **−57.6%** | 0.0104 | −8.3% | 0.0341 | −3.6% | 0.0336 | −0.7% | 1.719 | 0.486 |
| `instab_fsr` (M1) | 3.980 | +0.314 | −3.0% | 0.0115 | +0.6% | 0.0354 | +0.3% | 0.0339 | +0.1% | 1.647 | 0.013 |
| `instab_relaxed` (M1) | 3.943 | +0.277 | −14.5% | 0.0115 | +1.0% | 0.0353 | −0.0% | 0.0338 | −0.2% | 1.652 | 0.045 |
| `ema_instab_50` (M2+M1) | 3.837 | +0.171 | −47.2% | 0.0107 | −6.3% | 0.0341 | −3.5% | 0.0334 | −1.4% | 1.682 | 0.339 |
| *context:* compact per-frame | 4.172 | +0.506 | +56.1% | 0.0082 | −27.8% | 0.0242 | −31.5% | 0.0292 | −13.6% | 2.243 | — |
| *context:* x4plus per-frame | 4.208 | +0.543 | +67.3% | 0.0038 | −67.0% | 0.0000 | — | 0.0000 | — | 2.115 | — |
| *context:* bicubic(LR) | 3.666 | 0 (floor) | — | 0.0000 | — | 0.0788 | +122.8% | 0.0724 | +113.9% | 0.000 | — |

### Results — calm window (talking-head, mean|MV| ~1.1)

| sequence | MCLF | excess | Δexcess | tLP | ΔtLP | LPIPS vs x4 | ΔLPIPS | DISTS vs x4 | ΔDISTS | tOF_true | ᾱ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_hard` (shipping) | 1.073 | +0.258 | — | 0.0033 | — | 0.0378 | — | 0.0426 | — | 0.058 | — |
| `ema_fixed_50` (naive EMA) | 1.027 | +0.212 | -17.8% | 0.0044 | +33.3% | 0.0441 | +16.7% | 0.0443 | +4.0% | 0.107 | 0.484 |
| **`ema_adapt_50` (M2)** | 1.020 | +0.205 | -20.5% | 0.0029 | -12.1% | 0.0375 | -0.8% | 0.0423 | -0.7% | 0.061 | 0.427 |
| **`ema_adapt_75` (M2)** | 0.997 | +0.182 | -29.4% | 0.0027 | -18.2% | 0.0375 | -0.8% | 0.0424 | -0.5% | 0.064 | 0.641 |
| `instab_fsr` (M1) | 1.063 | +0.248 | -3.9% | 0.0032 | -3.0% | 0.0377 | -0.3% | 0.0425 | -0.2% | 0.060 | 0.013 |
| `instab_relaxed` (M1) | 1.047 | +0.232 | -10.1% | 0.0033 | +0.0% | 0.0377 | -0.3% | 0.0424 | -0.5% | 0.064 | 0.025 |
| *context:* compact per-frame | 1.255 | +0.440 | +70.4% | — | — | — | — | — | — | — | — |
| *context:* x4plus per-frame | 1.767 | +0.952 | +268.7% | — | — | — | — | — | — | — | — |
| *context:* bicubic(LR) | 0.815 | 0 (floor) | — | — | — | — | — | — | — | — | — |

*The calm window's last row (`ema_instab_50`, M2+M1) was still
running when this report was written — the box was carrying four sibling experiments (load avg 20)
and each metric row costs ~2 min there. Both M1 rows landed and repeat the jelly result
(−3.9% / −10.1% excess at −0.3% LPIPS = inside the noise). The M1 verdict rests on the jelly
window anyway, where the mechanism should have been strongest and was not. `python run_ab.py calm`
regenerates the full row set; `analyze.py` / `plot.py` then rebuild these tables from the JSON.*

### Why this is not a blur-win

A temporal smoother can always win flicker metrics by destroying detail. Guards, all passed:

* **LPIPS vs the x4plus per-frame ceiling improves** (−3.9% jelly) — a blur would move *away* from x4plus.
* **DISTS, the texture-aware R8-E4 triangulator, agrees** (−1.5% jelly) — no flip, as in every previous round.
* **var-Laplacian held** (secondary only, GOTCHA #23): jelly 487.7 → 482.5 (-1.1%), calm 62.0 → 60.9 (-1.9%).
* **Set-wise attribution** (`diag_sets.py`): scored on the SAME pixel sets, `ema_adapt_50` moves the stably-propagated set -4.2% (jelly) / -5.5% (calm), the flip set -2.4% / -2.9%, and the always-fallback set -1.3% / -1.4%.

The naive constant-α EMA (`ema_fixed_50`) — the mechanism the tabled "желе" complaint is actually
about — **fails every guard**: excess flicker +82.5%, tLP +81.8%, LPIPS +55.9%, DISTS +93.7%, tOF
+12.5%. The per-pixel gate is the entire difference between the two.

### The honest limit of M2

The set-wise numbers say M2 damps the *stably-propagated* set more than the *flip* set
(prop -4.2%/-5.5% vs flip -2.4%/-2.9%). So M2 lowers the total, but it does **not** attack the concentrated source the
diagnostic identified. That is a real finding, and it points at the right combination:
**soften the switch (R12-E1's residual gate) and damp what is left (M2)** — the two levers are
complementary, and R12-E1 measured its gate to *raise* jelly-tOF while lowering LPIPS, which is
exactly the trade M2 pays back.

### Cost

Measured in **numpy on CPU** at 1920×960, both sides under the same contention (team convention:
report ratios, never absolutes, on a shared box): the M2 post-pass costs **≈0.6–0.8× one
`derisk.reconstruct` numpy pass** (1077 vs 1365 ms/frame contended, 40-frame window). That is an
honest upper bound, not the deployable number: the pass is one warp of the history (which
`reconstruct_torch` already performs for the same frame), one AREA downscale, one smoothstep and one
lerp — all GPU ops that belong in the existing output pass next to `--region-aware`. **No extra SR
call, ever**, and no extra decode.

---

## 5. Verdicts and graduation

**M1 (FSR3.1 luma instability) — NO-GO.** Not landed as a lever; the port stays in
`stabilizer.py::luma_instability_factor` as a documented oscillation *detector*.

**M2 (per-pixel motion-adaptive EMA) — GATED-GO, landed default-OFF.**
`stabilizer.STAB_ENABLED = False`; `stabilize(...)` with the flag off returns the input sequence
untouched — verified **max|Δ| = 0 codes** on both windows against the shipping reconstruction.

Before any default flip (the β=0.85 / `deblock_pre` / `RESIDUAL_GATE` graduation path):

1. **Multi-clip tOF sweep, jointly with `RESIDUAL_GATE`.** M2 costs +1.7% tOF_true at α=0.5 (+4.0%
   at 0.75) on high motion — the same tension R1-E2 documented and the same currency as the gate's
   +5.8%. Sweep them together; they are complementary levers on the same artefact.
2. **A second clip.** Both windows here are from one source (`sd600.mp4`), as in R12-E1.
3. **A GPU port + a real end-to-end run.** Measured in numpy; the shipping locus is
   `reconstruct_torch`'s output pass, where `--region-aware` already lives, reusing the warp.
4. **Interaction with scene-cut detection.** The history resets on anchors/I-frames; a *missed* cut
   would smear one GOP (the R4-E1 plate-validity failure mode, one order milder).

**Not proposed, deliberately:** anything that rectifies history against a colour neighbourhood
(box, k-DOP, YCoCg ellipsoid, variance clipping). R1-E4b's root cause — rate-distortion MVs point
at *visually similar* content, so the ghost sits inside any box built from that neighbourhood — is
about colour identity, not box shape, and defeats all of them. M2 is a temporal filter keyed on a
*warp-error* signal; M1 is an oscillation test. Neither is a colour-identity test.

## Files

```
experiments/r13_e3_jelly/
  common.py         windows + read-only prototype imports + the on-disk SR cache (scratchpad)
  jelly_metrics.py  tLP, MCLF, the codec-MV-only adjacent-frame flow, drift_vs_flicker()
  stabilizer.py     M1 (FSR3.1 port) + M2 (motion-adaptive EMA); STAB_ENABLED = False
  run_ab.py         the A/B over both windows            -> out/results_{window}.json
  diag_extra.py     hard-switch oscillation + attribution + frequency split -> out/diag_*.json
  diag_sets.py      set-wise attribution of a variant + the var-Lap guard
  analyze.py        the report's tables      plot.py    -> out/*.png
  out/              JSONs, logs, diagnostic_drift_vs_flicker.png, mechanism_ab.png  (gitignored)
```

Read-only w.r.t. `prototype/`, `server/`, `web_spike/` — nothing outside this directory was edited.
