# R13-E4 — the forward-backward consistency term: does the Ruder softmax-splat earn its cost, and is FSR2 temporal MV divergence a substitute?

**Owner:** agent D, round R13. All work in `experiments/r13_e4_fwdbwd/`; `prototype/`,
`server/`, `web_spike/` imported READ-ONLY (variants are installed by monkey-patching
`derisk.occlusion_mask_lr` / `gpu_ops.occlusion_mask_lr` **in-process** with wrappers that
call the ORIGINAL for the production signals — the incumbent path is literally the shipped
code, byte-for-byte).

---

## Verdict — two questions, two answers

### 1. Does the Ruder softmax-splat earn its cost? → **NO — but not for the reason the challenge gave.**

The prior recall/precision analysis **reproduces**, and is **structurally incapable of
crediting the term it judges**: its "ground truth" (thresholded warp prediction error) is a
re-thresholding of the *reactive* signal, and is 83–96 % **contained inside the reactive
mask** — so `recall(reactive) ≡ containment` by construction and any additional term can
only add ~0 recall and lower precision. Arithmetically it could not have concluded anything
else.

Re-decided on **perceptual metrics against real ground truth** (LPIPS lead, DISTS
triangulation), the answer lands in the same place by better evidence: the splat is
**perceptually inert** — |ΔLPIPS| ≤ 0.0001 (≤0.25 %), |ΔDISTS| ≤ 0.0006 (≤0.8 %, sign flips
across windows), |ΔPSNR| ≤ 0.01 dB on all three windows — while inflating the fallback mask
by **+30…+67 %**, nearly **doubling the mask's frame-to-frame membership churn**, and adding
**+22 % / +9 % / +0.4 % flicker energy** in the pixels that switch source. It costs
**12.6 ms/frame (GPU) / 63.8 ms (numpy)** per mask call at production resolution.

**Recommendation: DELETE from the default — flip quality mode `occ: "adaptive"` →
`"reactive"`.** Keep the code and the flag (see "what NOT to delete"). Honest caveat: the
saving is real but *small in context* — 24 ms/frame of a ~2900 ms/frame quality-mode frame
(**~0.8 %**), because quality mode is per-frame-x4plus-bound. The strong part of this
result is not "we save time", it is **"we were paying for a signal that changes no pixel a
viewer can see, and slightly destabilises the ones it does change."**

### 2. Is FSR2 `ComputeTemporalMotionDivergence` a viable cheaper substitute? → **NO. REPLACE refuted, ADD refuted.**

Ported faithfully (including **dilated MVs**, which FSR2 feeds it and the prior analysis did
not mention) and swept over **36 configurations** — 3 history definitions × MV-dilation ×
6 pxDistance gates — its AUC for predicting decision-relevant warp failure is **0.50–0.52,
i.e. chance**, on *both* ground-truth definitions. **The prior AUC 0.813 / 0.741 did not
reproduce under any variant tried.** Wired into the production path at an area-matched
operating point it adds 3.5 pt of mask area on high motion for **zero** LPIPS/DISTS change
and *worse* tOF and churn. It is cheaper than the splat (4.96 ms vs 12.6 ms on GPU,
undilated), but cheapness is irrelevant to a chance-level predictor.

---

## Why the prior analysis could not have found anything (the methodological result)

`claim_check.py` recomputes the prior table on the production sub-pixel flows, sweeping the
GT threshold. "contain" = fraction of the "ground truth" already inside the reactive mask.

**Window A** (high-motion, 48 frames, 68 warp directions):

| τ | GT area | **contain(GT ⊆ reactive)** | react recall | react prec | full recall | full prec |
|---|---|---|---|---|---|---|
| 12 | 0.149 | 0.828 | 0.828 | 0.955 | 0.834 | 0.820 |
| 16 | 0.146 | **0.896** | 0.896 | 0.944 | 0.900 | 0.809 |
| 24 | 0.142 | 0.927 | 0.927 | 0.851 | 0.928 | 0.733 |
| 40 | 0.139 | 0.958 | 0.958 | 0.787 | 0.959 | 0.684 |

**Window C** (talking-head): contain 0.856 → 0.947 over the same sweep; react recall tracks
it to three decimals; the Ruder extra adds **+0.000…+0.005 recall** and costs 0.05–0.10
precision — *exactly the shape of the reported +0.003/+0.001 recall claim.*

`react recall ≡ contain` to 3 decimals in every row. That is the whole story: the metric
asks "does your mask contain the high-residual pixels?", the reactive term **is** the
high-residual pixel detector, and the fwd-bwd term exists precisely to catch the pixels
where **residual is LOW but geometry is WRONG** (GOTCHA #6: RD-optimised MVs point at
"visually similar but geometrically wrong" content). Under that GT those true positives are
scored as *false* positives.

### The decision-relevant ground truth

The mask's actual job is choosing, per pixel, propagated-warp vs fresh per-frame SR. So the
honest GT is: **a pixel should be masked iff falling back is closer to the true HD frame**
(`|perframe − GT| < |propagated − GT|`, measured at HD on a pure-propagation chain). Under it:

| window | GT area | contain(GT ⊆ reactive) | react rec / prec | full rec / prec | Ruder-extra: area, % helpful, mean gain |
|---|---|---|---|---|---|
| A | 20.5 % | **0.282** | 0.282 / 0.354 | 0.294 / 0.321 | 3.01 %, **27.5 %** helpful, **+0.88**/255 |
| C | 19.0 % | **0.213** | 0.213 / 0.391 | 0.221 / 0.372 | 0.48 %, **40.6 %** helpful, **+1.18**/255 |

Two things fall out, and they cut in opposite directions — both are reported:
* **Fair to the splat:** its extra pixels are *net positive* (mean gain > 0). It is not
  garbage area; it is weak-but-real area, and its 40.6 % hit rate on window C is close to
  the reactive mask's own 47.1 %.
* **Fatal to the splat:** *no* mask in this family is close to the achievable set —
  reactive covers only **21–28 %** of the pixels that would benefit. The occlusion mask is
  not the lever that is being under-served by the fwd-bwd term; the whole family is far
  from the ceiling (below).

---

## The decision evidence — GT-bearing perceptual A/B (`run_ab.py`)

**Rig.** The project's degrade-restore protocol with a **real codec** in the loop:
GT = `sample.mp4` frames at 640×320 → 2× INTER_AREA down → **real libx264 CRF 26, bf=3,
g=24** → decode with `+export_mvs` → **real RD-optimised MVs, real B-frames** → the shipped
`derisk.reconstruct` at scale 2 → 640×320, scored against GT. Quality-mode config
throughout: **x4plus fp16 anchors + anchor-blend β = 0.85 + region-aware output blend**.
48 frames/window. Windows: **A** = start 0 (high-motion), **B** = start 24000 (mid-clip,
dense texture), **C** = start 5000 (talking-head).

`oracle` = per-pixel best-of(propagated, per-frame-SR) against GT — the ceiling ANY
per-pixel mask could reach on this propagation (diagnostic; it can see the GT).

### Window A — high-motion (the window where the splat is supposed to matter)

| variant | LPIPS ↓ | DISTS ↓ | PSNR ↑ | tOF ↓ | fallback % | splat fires |
|---|---|---|---|---|---|---|
| naive (holes only) | 0.0436 | 0.0791 | 31.83 | 1.6135 | 3.99 | 0/0 |
| **reactive** | 0.0427 | 0.0729 | **31.84** | **1.6058** | 5.07 | 0/68 |
| **full (+Ruder)** | **0.0426** | 0.0727 | 31.83 | 1.6295 | 8.46 | 68/68 |
| **adaptive (shipped)** | **0.0426** | **0.0726** | 31.83 | 1.6297 | 8.34 | 45/68 |
| fsr2 (replace) | 0.0427 | 0.0729 | 31.83 | 1.6222 | 9.33 | 0/68 |
| fsr2+ruder (add) | 0.0426 | 0.0728 | 31.83 | 1.6134 | 11.39 | 68/68 |
| *oracle ceiling* | *0.0303* | *0.0583* | *33.15* | *1.4991* | — | — |

### Window B — mid-clip, dense texture

| variant | LPIPS ↓ | DISTS ↓ | PSNR ↑ | tOF ↓ | fallback % |
|---|---|---|---|---|---|
| naive | 0.0416 | 0.0709 | 24.30 | 0.4189 | 0.35 |
| **reactive** | 0.0389 | 0.0672 | 24.35 | **0.4346** | 3.43 |
| **full** | 0.0389 | **0.0670** | 24.35 | 0.4436 | 4.66 |
| **adaptive** | 0.0389 | 0.0671 | 24.35 | 0.4431 | 4.30 |
| fsr2 | 0.0389 | 0.0673 | 24.35 | 0.4344 | 3.56 |
| fsr2+ruder | 0.0389 | 0.0670 | 24.35 | 0.4434 | 4.76 |
| *oracle* | *0.0347* | *0.0629* | *24.92* | *0.3884* | — |

### Window C — talking-head

| variant | LPIPS ↓ | DISTS ↓ | PSNR ↑ | tOF ↓ | fallback % |
|---|---|---|---|---|---|
| naive | 0.0649 | 0.0716 | 31.03 | 0.3587 | 0.61 |
| **reactive** | 0.0633 | **0.0713** | 31.16 | 0.3405 | 1.29 |
| **full** | 0.0633 | 0.0719 | 31.17 | 0.3398 | 1.88 |
| **adaptive** | **0.0632** | 0.0715 | 31.16 | 0.3442 | 1.53 |
| fsr2 | 0.0633 | 0.0714 | 31.16 | 0.3421 | 1.46 |
| fsr2+ruder | 0.0633 | 0.0719 | 31.17 | 0.3357 | 1.97 |
| *oracle* | *0.0549* | *0.0662* | *32.01* | *0.3337* | — |

**Reading (LPIPS/DISTS carry the decision — see the metric-trap note):**
* Ruder ON vs OFF: LPIPS −0.0001/0/0 (A/B/C), DISTS −0.0002/−0.0002/**+0.0006**. Sign is
  inconsistent and every delta is ~10× smaller than the naive→reactive step (which is a
  real, consistent −0.0009…−0.0027 LPIPS). **The splat's perceptual effect is
  indistinguishable from zero, and DISTS does not corroborate even the sign.**
* The reactive term itself **is** load-bearing (naive → reactive is a consistent gain on
  every window and every metric). This is a verdict on the *third* signal only.
* The oracle ceiling is **−29 % LPIPS / −20 % DISTS / +1.3 dB** below the best real mask.
  The headroom is large and **neither** candidate term touches it.

**Metric-trap note (from agent A, applied):** PSNR/tOF reward "more of the frame came from
the fallback" when the fallback is a bilinear upsample of the true LR. Here the fallback is
the **per-frame x4plus SR**, and tOF is measured against the **GT sequence**, so the bias
runs the *other* way (more fallback = more per-frame SR = more flicker). Either way the
verdict rests on LPIPS/DISTS, which are flat. The tOF column is reported for completeness
and is *mildly against* the splat on A and B.

---

## Mask temporal stability (`mask_stability.py`) — the diagnostic recall/precision cannot see

Requested by the lead after agent C found the flicker energy concentrates in pixels whose
source flips between warp and fresh SR. `mc_flip` = `mask[t] XOR warp(mask[t−1])` over
pixels covered by a codec-MV chain (a mask that *moves with the content* is stable; only the
raw metric would punish it). MC flow fields from R13-E3's `jelly_metrics.build_mc_flows`,
reused read-only, identical fields for every variant. `flipE` = mc_flip × MCLF-on-flip-pixels
= the flicker energy attributable to source switching.

| window | variant | area % | mc_flip % | churn | MCLF | MCLF flip-px / stable-px | **flipE** |
|---|---|---|---|---|---|---|---|
| A | reactive | 5.07 | 5.09 | **1.00** | 1.874 | 26.72 / 1.36 (**19.7×**) | **135.9** |
| A | full | 8.46 | 9.33 | 1.10 | 1.874 | 17.81 / 1.43 | 166.2 (**+22 %**) |
| A | adaptive | 8.34 | 9.16 | 1.10 | 1.874 | 18.91 / 1.43 | 173.3 (+28 %) |
| A | fsr2 | 9.33 | 10.00 | 1.07 | 1.874 | 19.65 / 1.48 | 196.4 (+45 %) |
| B | reactive | 3.43 | 5.14 | 1.50 | 4.237 | 51.23 / 1.97 (**26×**) | 263.5 |
| B | full | 4.66 | 6.97 | 1.49 | 4.238 | 41.08 / 1.98 | 286.4 (+9 %) |
| C | reactive | 1.29 | 2.05 | 1.59 | 0.814 | 16.97 / 0.62 (**27×**) | 34.9 |
| C | full | 1.88 | 3.06 | 1.63 | 0.816 | 11.45 / 0.60 | 35.0 (+0.4 %) |

* **Agent C's finding is confirmed at the mask level**: flipping pixels carry **19.7×–27×**
  the per-pixel motion-compensated luma flicker of stably-propagated pixels.
* The splat's extra area is **not** buying stability — it is *itself* unstable. It raises
  churn-per-unit-area on A and C, raises the absolute flipping area by **+37…+83 %**, and
  raises total source-switch flicker energy by **+22 % / +9 % / +0.4 %**. Whole-frame MCLF is
  unchanged (1.874/1.874, 4.237/4.238, 0.814/0.816) — consistent with the flat LPIPS.
* So the "a bigger but better-shaped mask could still win" escape hatch was measured and
  **does not hold for this term**: bigger, no better shaped, slightly less stable.

---

## Cost (`bench_ms.py`) — production resolution, LR 640×320, scale 4 → 2560×1280

**Contention honesty:** three sibling agents ran heavy jobs on this box for most of the
timebox (load average peaked ~29; ~14 at measurement time). A quiet box was never available
inside the timebox, so per-call arms were timed **INTERLEAVED** (reactive, full, fsr2, …
per repetition, 11 reps, medians) — every arm sees the same load distribution, so the
**ratios** are sound even though absolutes are inflated. End-to-end is best-of-3 (the
least-contended pass). Absolutes should be re-measured on a quiet box before being quoted
elsewhere; the *ratios* and the *ordering* are the load-robust part, and the deletion
argument below is deliberately built on the ratio and on quality, not the absolute.

| op (one mask call, LR 640×320) | numpy median (best) | torch/MPS median (best) |
|---|---|---|
| reactive (intra + residual) | 7.31 (5.52) ms | 2.26 (1.90) ms |
| full (+ Ruder softmax-splat) | 71.10 (50.33) ms | 14.87 (12.11) ms |
| **→ the splat alone** | **63.78 (44.81) ms** | **12.62 (10.21) ms** |
| FSR2 temporal divergence, dilated | 30.78 (24.35) ms | 15.48 (11.04) ms |
| FSR2 temporal divergence, undilated | 10.27 (7.84) ms | 4.96 (3.52) ms |

End-to-end `reconstruct()` at production scale, bicubic per-frame cache (SR identical across
arms), ms/frame, best-of-3:

| backend | naive | reactive | full | adaptive | fsr2 |
|---|---|---|---|---|---|
| numpy | 257.4 | 270.1 | 325.3 | 322.6 | 303.4 |
| **torch (shipped)** | 52.3 | **60.1** | **84.3** | **82.4** | 81.1 |

* The splat is **6.6× the reactive mask** on the GPU and **+24.2 ms/frame (+40 %) of recon**
  at QHD — the docstring's "dominant GPU recon op" is **confirmed** (its "~29 ms numpy" is
  an *under*-statement at this resolution: measured 63.8 ms median / 44.8 ms best).
* **But context matters and is the honest brake on the deletion case:** quality mode is the
  only mode that runs it, and quality mode is ~2900 ms/frame (per-frame x4plus). 24 ms is
  **~0.8 %** of it. Deleting the splat does not make quality mode meaningfully faster.
* Instant (41 ms/frame, real-time) already uses `reactive` — where 12–24 ms *would* be
  30–60 % of the budget. So the expensive op is already excluded from the only place its
  price would hurt.

---

## Calibration — the challenger got its best shot (`calib_fsr2.py`)

Window A, 68 directions, scored against the decision-relevant GT. `am_*` = the operating
point whose mask AREA matches the Ruder term's own extra area (so "bigger mask" cannot
masquerade as "better mask").

| signal | AUC | area-matched recall | area-matched precision |
|---|---|---|---|
| reactive residual, for scale (window C, 75 dirs) | **0.649** (GT-B) / 0.803 (GT-A) | — | — |
| Ruder margin (continuous `lhs−rhs`) | 0.542 | 0.012 | 0.226 |
| FSR2 `ref` history, gate 20 display-px | 0.501 | 0.021 | 0.193 |
| FSR2 `ref` + **MV dilation**, gate 20 dpx | 0.505 | 0.018 | 0.190 |
| FSR2 `ref-raw` (literal, no normalisation) | 0.504 | 0.014 | 0.192 |
| FSR2 `disp` (previous display frame's MVs) | 0.502 | 0.019 | 0.207 |
| FSR2 ungated (`1 − |mv_prev|/|mv_cur|` raw) | 0.510–0.521 | 0.011–0.015 | 0.010–0.019 |

**All 36 configurations land in AUC 0.50–0.52.** Best area-matched precision (0.214) is
still below the Ruder extra's 0.226 and half the reactive mask's 0.354. Under the
prior-style GT the numbers are the same (`run_diag.py`, window C, 75 directions:
FSR2 AUC **0.511** vs the reactive residual's **0.803**; on GT-B 0.514 vs 0.649).

**I could not reproduce AUC 0.813 / 0.741 under any port I tried**, and I tried the three
plausible readings of "last frame's MV field", with and without FSR2's own MV dilation, with
and without temporal-distance normalisation, at six gate settings, against two ground
truths. Stated plainly because the round's standard is measured-not-claimed: either the
prior number used a definition I did not find, or it was not measured the way it reads.

**Why it fails here is structural, and it is the same reason FSR2's *spatial* divergence
failed (AUC 0.397):** FSR2's temporal term detects *"this pixel is moving but its history
was not"* — in a renderer that means a disocclusion, because MVs there are geometrically
exact. Codec MVs are **rate-distortion** vectors: on real footage a block's MV magnitude
changes between frames constantly for coding reasons that have nothing to do with
disocclusion, and a genuine disocclusion is frequently coded as **intra** (no MV at all) —
which the pipeline's signal (a) already catches for free. The temporal term is measuring
codec bitrate decisions, not geometry.

---

## What to change (and what NOT to delete)

1. **Flip `server/pipeline_api.py` `MODE_CONFIG["quality"]["occ"]` from `"adaptive"` to
   `"reactive"`.** Evidence: quality-neutral on LPIPS+DISTS across A/B/C, −30…−67 % fallback
   area, −37…−83 % source-switch area, −22 % switch flicker energy on high motion, −24 ms/frame
   of recon. Instant is already `reactive`, so this makes the two modes consistent.
2. **Do NOT remove `derisk.softmax_splat` / `gpu_ops._softmax_splat_flow` or the `occ="full"`
   / `"adaptive"` code paths.** (a) `softmax_splat` is used *outside* the mask — R13-E3's
   `jelly_metrics.mc_flow_prev` composes MC flow fields with it, and it is the natural
   inversion primitive for the MV-interpolation feature. (b) `"full"` is the prototype's
   ablation lever and part of the synthetic regression's documented behaviour. This is a
   **default flip**, not a code deletion; the risk of the flip is one flag away from zero.
3. **Do not adopt FSR2 temporal MV divergence in any form** (replace or add). Record it with
   the spatial variant as a settled NO-GO for codec MVs, with the shared root cause above.
4. **The real headroom is elsewhere.** The oracle ceiling says a *better* per-pixel decision
   is worth −29 % LPIPS / +1.3 dB, and the whole current family reaches only 21–28 % recall
   of it. Given agent C's flicker finding and the 19.7–27× flicker concentration in
   flipping pixels measured here, the promising direction is not a better binary detector
   but a **soft, temporally-hysteretic** membership (R12-E1's `residual_gate` is exactly
   this shape, currently default-OFF) — a mask that fades rather than switches removes the
   flicker event itself instead of relocating it.

---

## Honest limitations

* **Scale.** The GT-bearing rig runs LR 320×160 → 640×320 (scale 2): the only way to have
  true HD ground truth for `sample.mp4`. MV blocks therefore cover 2× more of the frame than
  in production. The production-resolution arm (`run_prod.py`, real 640×320 MVs, scale 4,
  torch, tOF+fallback only) is **written but was NOT run inside the timebox** — the GPU was
  saturated by sibling agents. Its metrics would be no-GT (tOF/fallback%) and thus weaker
  than what is reported above; it should be run before the default is flipped.
* **Backend.** The perceptual A/B ran on the **numpy** backend (the byte-identical reference
  implementation); quality mode ships **torch**. The mask math is mirrored line-for-line and
  the handoff documents 63–68 dB agreement, but a torch-backend repeat of the A/B was not
  run. The torch mask path *was* exercised in the cost bench (all variants).
* **Timing under contention** — see the cost section. Ratios interleaved and sound;
  absolutes inflated; re-measure on a quiet box before quoting.
* **Content.** One 34-min source clip, three 48-frame windows, one CRF (26). No scene-cut
  window was included; the fwd-bwd term's behaviour across a *missed* cut is untested here
  (the pipeline forces a fresh anchor at detected cuts, so this is a second-order risk, but
  it is the one place a "safety net" argument for the splat could still live).
* `mc_flip` uses codec-MV chains, so it is measured over the 84–91 % of adjacent pairs those
  chains cover, not the whole frame.

---

## Files & reproduction

```
experiments/r13_e4_fwdbwd/
  fsr2_mask.py      FSR2 ComputeTemporalMotionDivergence (numpy + torch) + MV dilation +
                    the in-process variant wiring (patches derisk/gpu_ops; nothing on disk)
  harness.py        GT-bearing degrade-restore rig (real libx264 LR + real MVs) + metrics
  run_ab.py         the perceptual A/B (LPIPS/DISTS/PSNR/tOF/fallback%) + oracle ceiling
  mask_stability.py mask churn / source-switch flicker (reuses r13_e3_jelly read-only)
  claim_check.py    reproduction of the prior recall/precision claim + containment
  calib_fsr2.py     36-config FSR2 sweep (history × dilation × gate), AUC + area-matched
  run_diag.py       both ground truths, AUC of every signal
  bench_ms.py       interleaved per-call + end-to-end ms, numpy and torch
  run_prod.py       production-resolution arm (WRITTEN, NOT RUN — see limitations)
  out/*.json        every number above
```

```bash
cd experiments/r13_e4_fwdbwd
python3 run_ab.py --windows A,B,C --variants naive,reactive,full,adaptive --backend numpy --tag ab_incumbent
python3 run_ab.py --windows A,B,C --variants fsr2,fsr2+ruder --backend numpy --gate-px 10 --dilate 1 --no-oracle --tag ab_fsr2
python3 mask_stability.py --windows A,B,C
python3 claim_check.py --windows A,C
python3 calib_fsr2.py --windows A
python3 bench_ms.py --start 0 --n 24 --reps 11 --scale 4      # RE-RUN ON A QUIET BOX
```
