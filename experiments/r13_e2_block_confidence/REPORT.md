# R13-E2 — free per-block codec confidence, and the `RESIDUAL_GATE` default decision

**Owner:** agent B, round R13. **Branch:** `r13-dlss5-ports`.
**Files owned:** `experiments/r13_e2_block_confidence/**` + the one new module `prototype/block_confidence.py`.
`derisk.py` / `gpu_ops.py` / `residual_gate.py` / `server/**` / `web_spike/**` were imported READ-ONLY and
are unmodified (the gated variants are installed by an explicit, always-restored monkeypatch of
`residual_gate.reliability_hd` inside the experiment harness — see `phase2_sweep.py::run_gated`).

---

## Verdicts

| question | verdict |
|---|---|
| **Q1 — do partition size + MV magnitude improve the reliability gate?** | **NO-GO.** They are ~fully redundant with the residual term the gate already uses. Best-case combination weight measured is **w = 0** (adding them never improves error ranking; it degrades it monotonically as weight rises). |
| **Q2 — should `RESIDUAL_GATE` flip default-ON?** | **NO-GO for a default flip; keep `RESIDUAL_GATE = False`** (it stays a legitimate default-OFF opt-in). On an independent true-GT arbiter across 8 clips LPIPS improves 8/8 but **DISTS corroborates on only 5/8 (+27.3% worse on the high-motion window)** — the project's own triangulation rule (R8-E4 15/15, and both R9-E2 and R10-E1 *rejected* on this same guard) is not met. Separately, the R12-E1 evidence that created the flag is **confounded**: its arbiter and the gate's fallback source are the same per-frame-SR sequence, and in the shipping wiring they are literally the same images. |
| **Q3 (added, D's ask) — is temporal hysteresis on the membership the headroom?** | **NO-GO** in the minimal motion-compensated-EMA form: ≤2.1% LPIPS / ≤0.7% DISTS for 0.2–1.7% more tOF, two windows × two λ. Nowhere near D's −29% oracle ceiling. |
| **Q4 (added, coordinator's ask) — do the gate and agent C's stabiliser double-spend tOF?** | **No.** Composed they cost **+6.3% (jelly) / +0.5% (calm)** tOF_true against algebraic sums of +9.3% / +0.1% — **sub-additive on jelly, additive on calm; amplifying on neither.** Neither lever's cost is inflated by the other's presence. |

Q1 and Q2 are honest negatives; the round's substance is *why*. The deliverables are (a) the reason
Q1 fails, which generalises to any block-metadata gating signal, (b) a **measurement flaw in the
R12-E1 evidence** that would have mis-justified the flip — and which turns out to be the same flaw
agent A hit from the browser side, so it is stated once as a rule (§4.3), (c) two structural
properties of the shipped gate nobody had measured (§3), one of which (`gate_res_norm`, §3.1) halves
its tOF cost for almost nothing, and (d) the two cross-agent measurements above (§5, §6).

**Reproduce:** `make_degraded.py` → `phase1_signals.py` → `phase1b_formula_ab.py` →
`phase2_sweep.py` → `phase2b_realtof.py` → `phase3_cost.py` → `phase4_compose.py` →
`phase5_hysteresis.py` → `aggregate.py`. All artifacts land in `out/` (gitignored per
`.gitignore:23`); `out/summary_tables.md` regenerates every table below.

---

## 0. Why the brief's discrimination table cannot decide Q1

The brief's table measured **warp error at LR** and found partition size discriminating 8.6× and MV
magnitude ~75×. But the gate's residual term **is** the LR warp error:

```
react = mean_c |LR_cur − MV_warp(LR_ref)|          (residual_gate.reliability_lr)
```

So "partition size predicts LR warp error" and "MV magnitude predicts LR warp error" are statements
that these signals correlate with **the number the gate already computes exactly**. A signal can only
be worth adding if it predicts the error the residual *cannot see*: an RD-optimal MV that points at
visually-similar-but-geometrically-wrong content (GOTCHA #6), whose LR residual is small but whose
**HD** detail is warped from the wrong place. That is also the documented mechanism behind R12-E1's
+5.8% jelly tOF cost.

Measuring that requires an HD ground truth alongside real codec MVs, which real footage does not have.

## 1. Method — degrade-and-restore probes (the R5-E2 protocol)

`make_degraded.py` takes the decoded native frames as a pseudo-HD GT, downscales 2× (INTER_AREA),
**re-encodes the LR with libx264** (default preset → real B-frames, real RD motion search) and decodes
it back with `+export_mvs`. That yields real codec MVs, a real LR residual, **and** an exact 2× GT.

8 probes = the project's canonical `sample.mp4` windows (A high-motion / B mid / C talking-head /
D) **plus** the R3-E2 robustness clip set (fastpan / graphics / lowlight / talking-head).

Per MV **record** (per codec partition), with a **perfect anchor** (GT reference — GOTCHA #15's
isolation rule, so the number is pure propagation error and not re-injected fresh SR):

```
err_hd  = | warp_hd(GT_ref, MV) + upsample(LR_cur − warp_lr(LR_ref)) − GT_cur |
react   = mean_c |LR_cur − warp_lr(LR_ref)|        (what the gate already sees)
minside = min(partition_w, partition_h)            (free, unused)
mvmag   = |MV| in LR px                            (free, unused)
```

102 665 MV records across the 8 probes. Raw table: `out/phase1_blocks.npz`; summary
`out/phase1_summary.json`; driver `phase1_signals.py`.

## 2. Q1 results — the block signals are redundant

### 2.1 The brief's table does not reproduce on HD propagation error

| signal | brief (LR MAE) | measured here (HD propagation error, pooled) |
|---|---|---|
| intra hole (already used) | 18.6× | **1.1×–14.8× per clip** (pooled: holes are the worst pixels, confirmed — keep the term) |
| partition 16×16 vs sub-16 | 8.6× | **4.82×** (1.03 vs 4.96); sub-16 = 17.0% of MV area |
| MV magnitude | ~75× spread | **NOT monotonic**: 0–0.25 px 0.71 · 0.25–1 px 3.80 · 1–2 px 4.49 · 2–4 px 5.28 · 4–8 px 4.91 · 8–16 px **2.87** · ≥16 px 4.47 |

MV magnitude is **not** a reliability signal past ~2 px: an 8–16 px MV is a clean global pan and is
*better* predicted than a 1–2 px one. What the magnitude term actually encodes is "is this block
moving at all" (0–0.25 px = 71% of area, err 0.71). The residual encodes that far better.

Per-clip intra-hole vs covered HD error (the term the gate already uses, and the only one of the
three that survives): A_highmotion 3.4× · B_mid 2.4× · C_talkinghead 2.7× · D_jelly 1.5× · c1_fastpan 1.7× · c2_graphics 14.8× · c3_lowlight 1.1× · c4_talkinghead 12.0×. It is real everywhere but its *ratio* is strongly content-dependent —
huge where the covered region is near-perfect (`c2_graphics` 14.8×, `c4_talkinghead` 12.0×), close
to 1× where everything is bad anyway (`c3_lowlight` 1.1×). A single pooled number like "18.6×" is
not a stable property of the signal.

Also worth recording: on these H.264 streams the partition signal is effectively **binary** — only
min-side 8 and 16 ever appear (no 4×4 sub-partitions), 83% / 17% of MV area.

### 2.2 Conditioned on the residual, the block signals collapse

Residual quintiles (q1 is empty: >20% of blocks have `react` exactly 0):

| quintile | react range | err_hd | 16×16 vs sub-16 | mv<4 vs mv≥4 |
|---|---|---|---|---|
| q2 | 0 – 0.02 | 0.032 | 0.029 / 0.389 = **13.4×** | 0.031 / 0.082 = 2.6× |
| q3 | 0.02 – 1.37 | 0.865 | 0.653 / 1.629 = 2.49× | 0.846 / 1.266 = 1.50× |
| q4 | 1.37 – 3.99 | 3.917 | 3.450 / 4.728 = 1.37× | 4.244 / 2.685 = **0.63×** |
| q5 | 3.99 – 252 | 6.646 | 5.742 / 7.593 = 1.32× | 6.754 / 6.426 = 0.95× |

The partition ratio survives only in the quintile where the **absolute error is negligible** (q2:
0.029 vs 0.389 codes — invisible). Where the error actually lives (q4/q5, ~5 of the 6.6-code mean),
the ratio drops to ~1.3× — and the MV-magnitude term **inverts** (large-MV blocks are *better*).

Spearman: `react↔err 0.907`, `minside↔err −0.617`, `mvmag↔err 0.514`, and
`minside↔react −0.622`, `mvmag↔react 0.516` — i.e. the block signals' correlation with error is
almost entirely *mediated* by their correlation with the residual.

### 2.3 The decision-relevant test: the gain curve

Rank blocks by predicted unreliability; what fraction of total HD error **mass** does distrusting the
top-X% of frame area capture?

| predictor | top 5% | top 10% | top 20% | top 30% |
|---|---|---|---|---|
| oracle (err itself) | 0.323 | 0.539 | 0.794 | 0.928 |
| **residual only (shipping)** | **0.273** | **0.444** | **0.692** | **0.904** |
| residual ÷ block-confidence | 0.267 | 0.420 | 0.684 | 0.905 |
| block confidence only | 0.118 | 0.230 | 0.507 | 0.771 |
| random | 0.050 | 0.100 | 0.200 | 0.300 |

The residual alone already captures **82%** of what a per-block oracle could (0.444 / 0.539 at top-10%).
Mixing the block prior in makes it *worse*.

### 2.4 Best-case combination sweep — the answer is "add zero"

Score = `react + w·(1−c)·K` (K = the 90th-percentile residual, so w is dimensionless), gain@top-10%:

| w | 0.0 | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|
| pooled gain@10% | **0.4440** | 0.4434 | 0.4428 | 0.4410 | 0.4367 | 0.4232 | 0.3947 |

Per clip the best `w` is 0.0 on 3/8, 0.05 on 3/8, 0.1 on 1/8, 1.0 on 1/8 — and where a non-zero `w`
"wins" the improvement is ≤ **+0.0004 absolute** (≤0.05% relative), i.e. noise. The one clip that
prefers w = 1.0 (`c4_talkinghead`) has a mean propagation error of **0.125 codes** — nothing to gate.

**→ Q1 = NO-GO.** Partition size and MV magnitude are free, but the gate already has a strictly better
measurement of the same thing. This is CDA-VSR's (CVPR 2026) conclusion arrived at independently:
*residual localizes where motion compensation failed; block metadata only says how hard the encoder
worked.* The same logic that rules out QP for propagation gating rules out partition size.

### 2.5 The one place the block signals are still interesting (secondary result, for agent A)

They cost **no warp**: `block_signals` is one pass over MV records the decoder already produced, while
`react` needs a `warp_lr` + abs-diff per direction. As a *standalone cheap* gate (`gate_block`, i.e.
`a = c`, no residual at all) it retains a usable but clearly inferior share of the ranking power
(gain@10% 0.230 vs 0.444). Relevant only to a compute-starved port (the browser/WASM path); it is not
a server-side option, because the server's residual is nearly free next to the SR it already runs.

## 3. Two structural properties of the shipped gate that had not been measured

### 3.1 The gate has a **trust ceiling of 0.881** — it never fully trusts any warp

`a = sigmoid((tau_res − react)/s_res)` with `tau_res=10, s_res=5` gives, at a *perfect* warp
(`react = 0`), `a = sigmoid(2) = 0.8808`. So **≥11.9% of every pixel** is taken from the fresh
per-frame SR cache even where propagation is exact. Measured mean trust on real windows
(`prototype/residual_gate.reliability_hd`, backbone past direction, 32-frame windows):

| window | mean `a` (trust) | mean fresh-per-frame-SR weight | intra-hole (a=0) area |
|---|---|---|---|
| A_highmotion (sample.mp4 @0) | 0.710 | **29.0%** | 17.2% |
| C_talkinghead (@5032) | 0.842 | 15.9% | 3.8% |
| E_mid2 (@12000) | 0.699 | 30.1% | 15.8% |
| R12_jelly (sd600 @582) | **0.348** | **65.2%** | 50.7% |
| R12_calm (sd600 @437) | 0.772 | 22.8% | 5.1% |

On R12-E1's own headline window the "gated propagation" output is **65% fresh per-frame SR**.

### 3.2 That makes R12-E1's arbiter circular — and in the *shipping* config, totally circular

R12-E1 scored `LPIPS/DISTS(recon, per-frame x4plus SR)` while the gate's fallback source was per-frame
**compact** SR. Those two are decoupled *models*, but they are not decoupled *quantities*: as trust
`a → 0` the reconstruction converges on a fresh per-frame SR sequence, which is much closer to the
per-frame-SR reference than any propagated sequence is. Re-analysing R12-E1's own published JSON
(`out/r12e1_reanalysis.json`) on the jelly window:

| variant | distrust | LPIPS↓ | tOF_true↑ | % of the way from hard→per-frame on LPIPS |
|---|---|---|---|---|
| baseline_hard | — | 0.03535 | 1.6537 | 0% |
| gate_res_t16 | 0.561 | 0.02607 | 1.7291 | **83%** |
| gate_res_t10 (the flag) | 0.579 | 0.02509 | 1.7499 | **92%** |
| gate_res_t7 | 0.600 | 0.02478 | 1.7673 | **95%** |
| gate_resfb_t10 | 0.652 | 0.02455 | 1.8186 | **97%** |
| compact_perframe | (1.0) | 0.02422 | 2.2425 | 100% |

LPIPS is **strictly monotone-decreasing in distrust** and tOF_true **strictly monotone-increasing** —
the four gate settings are just four points on the interpolation line between propagation and
per-frame SR. The reported "−29% LPIPS" is a measurement of *how much propagation was switched off*.

And the shipping wiring is worse than the experiment's: `MODE_CONFIG["quality"]` is the only mode
carrying `residual_gate="RESIDUAL_GATE"` (`server/pipeline_api.py:332`), and quality mode's per-frame
cache **is x4plus** (`build_perframe_cache` runs SR on every frame; only *instant* was moved to
anchor-only SR by P1). So in production the gate's fallback source and R12-E1's metric reference are
**the same images** — under that arbiter `a → 0` scores a perfect zero.

Not everything about the gate is confounded, and the calm window shows it: there,
tOF_true goes 0.0581 (hard) → **0.0514** (gate) while per-frame SR sits at 0.0559 — *better than both
endpoints*, which no interpolation can produce. That is a real effect: softening the hard
`recon[occ] = perframe[occ]` switch removes seam flicker. It is the honest part of the gate's case,
and it is small.

---

## 4. Q2 results — the multi-clip sweep the R12 review asked for

### 4.1 Independent arbiter: true GT (degrade-and-restore), 7 clips

Shipping-quality-mode gate locus: per-frame **x4plus** SR cache with the R8-E3 β=0.85 anchor
blend, `occ='adaptive'`, numpy backend. `RESIDUAL_GATE=True` == `gate_res`. Deltas vs
`baseline_hard` (= `RESIDUAL_GATE=False`, shipping); **negative = better**. Full tables:
`out/summary_tables.md`, raw `out/phase2/*.json`.

| clip | hard fallback | ΔLPIPS_gt | ΔDISTS_gt | ΔtOF_gt |
|---|---|---|---|---|
| A_highmotion (sample @0) | 11.68% | −8.9% | **+27.3%** | −5.8% |
| B_mid (sample @2500) | 6.49% | −2.6% | −1.7% | −6.0% |
| C_talkinghead (sample @5032) | 3.69% | −3.7% | −2.8% | −0.6% |
| D_jelly (sample @12000) | 1.46% | −3.5% | **+3.2%** | −2.0% |
| c1_fastpan (R3-E2) | 30.98% | −28.9% | −23.3% | −25.1% |
| c2_graphics (R3-E2) | 3.26% | −1.6% | −5.6% | −7.6% |
| c3_lowlight (R3-E2) | 39.91% | −5.8% | **+4.1%** | −3.6% |
| c4_talkinghead (R3-E2) | 0.17% | −22.9% | −26.4% | −31.1% |
| **mean** | | **−9.8%** | **−3.2%** | **−10.2%** |
| **win count** | | **8/8** | **5/8** | **8/8** |

**LPIPS says GO on every clip (8/8). DISTS does not corroborate** — it wins on only **5/8**, and
is **+27.3% worse on the high-motion window `A`**, i.e. exactly the regime R12-E1's headline came
from. This is the first LPIPS/DISTS disagreement in the project's record: R8-E4's triangulation
pass confirmed **15/15** LPIPS decisions, and R9/R10 both used a DISTS guard to catch a
plausible-looking win (R10-E1's over-smoothing; R9-E2's fake detail). The same guard fires here.
The mechanism is the familiar one: the gate replaces propagated texture with fresh per-frame SR
texture; LPIPS rewards the structural agreement, DISTS penalises the texture substitution.

Note the shape of the win: it is carried almost entirely by the two clips where propagation is
either barely used (`c4`, 0.17% fallback) or nearly broken (`c1_fastpan`, 30.98% fallback), i.e.
where the gate is closest to "just use per-frame SR". On the four `sample.mp4` windows — the
project's own canonical content — the LPIPS gain is −2.6…−8.9% and DISTS is *negative* on two of
four. A default flip would be justified by the outliers, not by the target content.

*(The block-confidence variant `gate_res_block` tracks `gate_res` to within ~1 pt on every cell —
8/8, 8/8, 5/8, mean −9.7% / −5.6% / −11.4%. It does not change the verdict either way, which is
Q1's answer restated end-to-end.)*

### 4.2 Real footage, native geometry, honest tOF — the "is it just dilution?" test

`sd600.mp4` jelly window (the R12-E1 headline window), 640×320 → 1920×960, shipping quality
config. `tOF_true` = flow vs the bicubic-upscaled decoded LR = **true content motion**; the
per-frame-SR endpoint is *worse* on it, so this metric cannot be gamed by giving up on
propagation. `LPIPS_pf`/`DISTS_pf` are the R12-E1 arbiter — reported to show what it does.

| variant | tOF_true | Δ | LPIPS_pf* | DISTS_pf* | distrust |
|---|---|---|---|---|---|
| baseline_hard | 0.9036 | — | 0.0122 | 0.0266 | — |
| **gate_res (`RESIDUAL_GATE=True`)** | 1.0350 | **+14.5%** | 0.0027 | 0.0093 | 0.65 |
| gate_res_norm (trust ceiling removed) | 0.9630 | **+6.6%** | 0.0039 | 0.0127 | |
| gate_res_block | 1.0728 | +18.7% | 0.0014 | 0.0052 | |
| gate_block | 1.0806 | +19.6% | 0.0024 | 0.0100 | |
| perframe_sr (the a→0 limit) | 1.2469 | +38.0% | **0.0000** | **0.0000** | 1.0 |
| bicubic | 0.0000 | (is the reference) | 0.0499 | 0.0645 | |

Every row is ordered the same way on both columns: **the further a variant leans on the
fallback, the better the R12-E1 arbiter reads and the worse the honest one does.** The gate's
+5.8% tOF cost recorded in R12-E1 (compact fallback) is **+14.5%** in the shipping wiring
(x4plus fallback). And `gate_res_norm` shows **over half of that cost is the global 11.9%
dilution, not the occlusion softening**: removing the trust ceiling cuts the tOF cost from
+14.5% to +6.6% while giving up almost nothing on the perceptual columns.

The same interpolation logic explains the *calm* window's apparent tOF win (§6): there
`perframe_compact` is *better* than propagation on tOF_true (0.0285 vs 0.0327), so leaning on it
reads as a −8% "win". Whether the dilution reads as a win or a loss is a property of the content,
not of the gate.

### 4.3 The general rule (unifies this with agent A's browser finding)

Agent A independently hit the mirror image of this in the browser: there the fallback is
**bilinear LR**, so **PSNR-vs-ref and tOF-vs-LR** improve monotonically as the gate leans on the
fallback, and only LPIPS/DISTS separate "seam fixed" from "frame softened". Same bug, different
metric:

> **Whenever the gate's fallback source coincides with (or is trivially close to) the metric's
> reference, that metric is not an arbiter for the gate — it is a distrust meter.**
> Browser: fallback = bilinear LR ⇒ PSNR-vs-ref and tOF-vs-LR are disqualified.
> Server: fallback = per-frame SR ⇒ LPIPS/DISTS-vs-per-frame-SR are disqualified.
> The only safe arbiters are references built from neither: a true GT (§4.1), or a reference the
> fallback is measurably far from (tOF_true here, where per-frame SR is +38%).

This deserves to be one GOTCHA covering both sides, not two.

## 5. Composition with agent C's stabiliser (R13-E3 M2) — they do NOT double-spend

Both levers spend tOF_true, so judging either alone is unsafe. `phase4_compose.py` runs the
matrix in agent C's / R12-E1's exact setup (sd600, jelly=582..621 / calm=437..476, SCALE=3,
`occ='full'`, **compact** per-frame fallback, x4plus metric reference, C's SR caches reused
read-only). C's landed lever = `StabCfg(mode="ema_adaptive", alpha_max=0.50)`; the disabled path
was asserted byte-identical (0 codes) on both windows.

| window | variant | tOF_true | Δ vs hard | LPIPS(x4) | DISTS(x4) |
|---|---|---|---|---|---|
| jelly | baseline_hard | 0.8044 | — | 0.0304 | 0.0362 |
| jelly | + stabM2 only | 0.8021 | **−0.3%** | 0.0292 | 0.0347 |
| jelly | gate_res only | 0.8814 | +9.6% | 0.0224 | 0.0277 |
| jelly | **gate_res + stabM2** | 0.8551 | **+6.3%** | 0.0223 | 0.0279 |
| jelly | gate_res_norm + stabM2 | 0.8576 | +6.6% | 0.0231 | 0.0291 |
| jelly | perframe_compact | 1.0979 | +36.5% | 0.0221 | 0.0293 |
| calm | baseline_hard | 0.0327 | — | 0.0304 | 0.0431 |
| calm | + stabM2 only | 0.0354 | **+8.1%** | 0.0302 | 0.0431 |
| calm | gate_res only | 0.0301 | **−8.0%** | 0.0279 | 0.0433 |
| calm | **gate_res + stabM2** | 0.0329 | **+0.5%** | 0.0280 | 0.0431 |
| calm | perframe_compact | 0.0285 | −12.8% | 0.0281 | 0.0453 |

**The two levers are complementary; neither amplifies the other's cost.** Stated precisely
against the algebraic sum of the single-lever costs:

| window | stabM2 alone | gate alone | sum (if independent) | **measured composed** |
|---|---|---|---|---|
| jelly | −0.3% | +9.6% | +9.3% | **+6.3%** (sub-additive — the stabiliser *pays back* part of the gate's cost, for free on LPIPS/DISTS) |
| calm | +8.1% | −8.0% | +0.1% | **+0.5%** (additive to within measurement noise — the gate pays back the stabiliser's cost) |

So the coordinator's double-spending worry does not materialise on these two windows: composing
them costs **no more** than adding their individual bills, and on jelly measurably less. Whichever
of the two ships, it does not consume the other's budget. (This does not make either one a GO on
its own — see §10 for why the gate still should not flip.)

Honest discrepancy to record: I measure stabM2-alone at **−0.3% (jelly) / +8.1% (calm)** tOF_true
where agent C reports **+1.7% / +5.2%**. Same clip, windows, scale, occ mode and stabiliser
config; the likely cause is metric geometry (my tOF runs at 640-wide, and my LPIPS baseline is
0.0304 vs R12-E1/C's 0.0353, consistent with a different metric width). **Compare within a table,
not across them**; the signs and the composition conclusion are unaffected.

## 6. Temporal hysteresis on the membership (agents A, C, D all pointed here) — measured, NO-GO

Three agents converged on "soft, temporally stable membership" being the headroom (D measured an
oracle per-pixel ceiling of −29% LPIPS / −20% DISTS that the whole binary-detector family reaches
only 21–28% of; C measured the flip-set carrying 12.4×/3.5× the flicker; A found the soft gate
beat the hard one). Nobody had measured hysteresis itself, so `phase5_hysteresis.py` does:

```
a = lam * warp_lr(a_prev, fx, fy) + (1 - lam) * sigmoid((tau_res - react)/s_res)
a[intra hole] = 0
```
— i.e. the membership is motion-compensated along the pixel's own trajectory and EMA'd, so it
changes smoothly instead of flipping. `lam = 0` is asserted **byte-identical (0 codes)** to the
shipping gate on both windows.

| window | variant | tOF_true | Δ vs gate_res | LPIPS(x4) | Δ | DISTS(x4) | Δ |
|---|---|---|---|---|---|---|---|
| jelly | gate_res | 0.8814 | — | 0.0224 | — | 0.0277 | — |
| jelly | hyst λ=0.35 | 0.8833 | +0.2% | 0.0220 | −2.1% | 0.0276 | −0.4% |
| jelly | hyst λ=0.60 | 0.9283 | +5.3% | 0.0221 | −1.7% | 0.0280 | +0.9% |
| calm | gate_res | 0.0301 | — | 0.0279 | — | 0.0433 | — |
| calm | hyst λ=0.35 | 0.0306 | +1.7% | 0.0277 | −0.8% | 0.0430 | −0.7% |
| calm | hyst λ=0.60 | 0.0317 | +5.3% | 0.0277 | −0.8% | 0.0430 | −0.6% |

**NO-GO.** λ=0.35 buys 0.8–2.1% LPIPS and 0.4–0.7% DISTS for 0.2–1.7% more tOF_true; λ=0.60 is
strictly worse on tOF for no extra perceptual gain. Nothing here is within an order of magnitude
of D's −29% / −20% oracle ceiling. The useful negative for D: **the oracle headroom is not
reachable by making the existing membership smoother in time.** Smoothing a membership derived
from a single-hop LR residual cannot add information the residual never had; closing that gap
needs a *better per-pixel decision*, not a *steadier* one. (Scope: one hysteresis form — a
motion-compensated EMA of `a` — two λ values, two windows. A sticky/asymmetric form
(`a = min(a_raw, λ·a_prev + (1−λ))`, distrust decaying slower than it rises) is untested.)

## 7. Answering agent A directly (one function, two constant sets)

A's shipped browser form is an ADDITIVE threshold penalty; mine was a MULTIPLICATIVE one. To keep
the two sides one function, A's form is now the **reference formulation** in
`prototype/block_confidence.py` (`tau_eff_blockpenalty`, `mode="penalty"`), implemented verbatim:

```
tau_eff = max(tau - kp*(1 - area/256) - km*min(|mv|/mvref, 1), 0)
```
(`mvref` was not given in A's message; it is exposed and swept below — the result is flat in it.)

`phase1b_formula_ab.py` runs A's exact form through the server's HD-GT ranking test (§2.3),
gain@top-10% of true HD propagation-error mass, pooled over 102 665 partitions:

| predictor | top10% | vs residual-only |
|---|---|---|
| oracle | 0.5389 | +21.4% |
| **residual only (shipping)** | **0.4440** | — |
| A's form, **kp=6 km=0** (partition term only), any mvref | 0.4457 | **+0.38%** |
| A's form, kp=3 km=3, mvref 8 | 0.4350 | −2.0% |
| **A's shipped kp=6 km=6, mvref 8** | 0.4198 | **−5.5%** |
| A's form, kp=0 km=6 (MV term only), mvref 8 | 0.4035 | −9.1% |
| A's form, kp=12 km=12, mvref 8 | 0.3869 | −12.9% |
| this experiment's multiplicative form, c_floor 0.35 | 0.4213 | −5.1% |

Three things for A:
1. **The two formulations behave the same** (−5.5% vs −5.1% at comparable strength), so the
   additive/multiplicative choice is not what matters. **A's is the reference form**; the server's
   answer is that both should be off, so there is no divergence to reconcile — only a documented
   difference in constants (server: `kp = km = 0`).
2. **The MV-magnitude term is the harmful half.** `kp=6, km=0` is the only configuration that
   does not lose (+0.38%, i.e. noise); `kp=0, km=6` loses 9.1%. On true HD propagation error MV
   magnitude is **not monotonic** (§2.1) — 8–16 px blocks are *better* predicted than 1–2 px ones,
   because a large MV usually means a clean global pan. If A's browser win survives dropping `km`,
   that is a simpler and better-founded browser config too; if it does not, the reason is worth
   knowing, and A's own metric trap (§4.3) is the first place to look.
3. **A's "different constants for a documented reason" framing is exactly right, and the reason is
   §4.3.** In the browser `(1−a)` costs real sharpness (bilinear LR); on the server it costs
   nothing in compute (per-frame SR is computed anyway in quality mode) but costs *texture
   fidelity* — which is why the server's DISTS guard fires (§4.1) where the browser's PSNR does not.

---

## 8. Compute cost of enabling the gate

`phase3_cost.py`, shipping QUALITY geometry (`sample.mp4` @5032, 24 frames, 640×320 → **2560×1280**,
`occ='adaptive'`, `use_residual=True`), best-of-3 after a warm-up. **The box was heavily contended
(other R13 agents; load average ~40), so read the RATIOS, not the absolute ms** — the project's
standing convention.

| path | best ms/frame | vs shipping |
|---|---|---|
| `backend='torch'`, `RESIDUAL_GATE=False` (**shipping**) | 162.2 | 1.00× |
| `backend='numpy'`, gate off (the gate's forced backend) | 397.8 | 2.45× |
| `backend='numpy'`, **`RESIDUAL_GATE=True`** | 438.9 | **2.71×** |
| `backend='numpy'`, gate + block confidence | 573.8 | 3.54× |
| `backend='numpy'`, block-confidence-only gate (no residual warp) | 326.0 | 2.01× |

Decomposition: of the 2.71× that `RESIDUAL_GATE=True` costs today, **2.45× is not the gate at all** —
it is `derisk.py:583` routing an enabled gate to the numpy path even under `backend='torch'`. The
gate's own math is ~**1.10×** on top of that (it recomputes `build_lr_flow` + `warp_lr` per direction
that `_warp_one` just computed — the `PERF TODO` already noted in `residual_gate.py`).

So the honest cost statement is: **enabling the flag today nearly triples quality-mode render time,
and ~90% of that is a missing GPU port, not the algorithm.** Quality mode is already ~2.9 s/frame
end-to-end, so this is a render-time, not a real-time, concern — but it is not free, and it is the
reason the flag cannot be flipped "for free" even if the quality case were proven.

## 9. The formula — stated precisely for agent E's cross-implementation diff

**The shipped gate (unchanged, `prototype/residual_gate.py`), per warp direction `d ∈ {past, future}`:**

```
fx, fy   = build_lr_flow(mvs, h_lr, w_lr, want=d)          # NaN where no MV of direction d
react    = mean_c |LR_cur − warp_lr(LR_ref, fx, fy)|       # HxW, luma+chroma mean, 0..255 codes
a_lr     = 1 / (1 + exp(−(tau_res − react) / s_res))       # tau_res = 10.0, s_res = 5.0
a_lr[¬isfinite(fx)] = 0                                    # intra hole → fully fresh
a_hd     = INTER_LINEAR_resize(a_lr, w_hd, h_hd)
a_hd[INTER_NEAREST_resize(¬isfinite(fx)) ] = 0             # HD hole → fully fresh (hard zero)
recon    = a_hd · warped_propagation + (1 − a_hd) · perframe_fresh
```
Note `a_lr ≤ sigmoid(tau_res/s_res) = 0.8808` — there is no input for which the gate fully trusts a warp.

**This experiment's block-confidence prior (`prototype/block_confidence.py`) — measured, NO-GO, but
specified so ports can be diffed against it:**

```
# painted with the SAME record loop / direction filter / block extent as build_lr_flow
# (dst_x,dst_y are block CENTRES; NaN sets of the two functions are byte-identical — verified)
minside  = min(record.w, record.h)                         # LR px; only 8 and 16 occur in practice
mvmag    = |(record.motion_x, record.motion_y)| / record.motion_scale     # LR px

c_part   = clip((minside − part_lo) / (part_hi − part_lo), 0, 1)          # part_lo=4,  part_hi=16
c_mv     = 1 / (1 + exp(−clip((mv_tau − mvmag)/mv_s, −60, 60)))           # mv_tau=4.0, mv_s=1.5
c        = c_floor + (1 − c_floor) · c_part · c_mv                        # c_floor=0.35
c        = 0 where minside or mvmag is NaN                                # intra hole
```
combined with the residual in one of three modes (`cfg["mode"]`):
```
"tau"   :  a = sigmoid((tau_res · c − react) / s_res)      # confidence lowers the threshold
"mul"   :  a = c · sigmoid((tau_res − react) / s_res)
"block" :  a = c                                            # free: no warp, no residual
```
then `a[¬isfinite(fx)] = 0` and the same HD upsample + hard hole-zero as the shipped gate.

**Invariants agent E can assert (both verified here):**
1. `c_floor = 1.0` with `mode="tau"` makes `reliability_lr_bc` **bit-identical** to
   `residual_gate.reliability_lr` (measured `max|Δ| = 0.0`). Any port must preserve this.
2. `isfinite(block_signals(...))` equals `isfinite(build_lr_flow(...))` for every frame/direction
   (44/44 exact on a 12-frame window) — the hole sets must not diverge, or the `a=0` hole rule
   silently changes meaning.

---

## 10. THE RECOMMENDATION on `RESIDUAL_GATE`

**Keep `server/pipeline_api.py:245` `RESIDUAL_GATE = False`. Do not flip the default in R13.**

The evidence, weighed:

| for the flip | against the flip |
|---|---|
| True-GT LPIPS improves on **8/8** clips (mean −9.8%) | True-GT **DISTS corroborates on only 5/8** (mean −3.2%, **+27.3% on the high-motion window**). The project's own triangulation rule (R8-E4: 15/15; R9-E2 and R10-E1 both *rejected* on a DISTS guard) is not met. |
| True-GT tOF improves on **8/8** (mean −10.2%) | On **real footage at native geometry** the honest `tOF_true` is **+14.5% worse** on the jelly window, and the whole variant family orders monotonically along the propagation→per-frame-SR line (§4.2). |
| The mechanism is right: it softens the hard `recon[occ]=perframe[occ]` switch, and agents C and D independently measured that flip-set as carrying **12–27×** the flicker | The evidence that created the flag (R12-E1, −29% LPIPS) is **confounded** — its arbiter and the gate's fallback source are the same sequence, and in the *shipping* wiring they are literally the same images (§3.2). |
| It composes with agent C's stabiliser without double-spending tOF (§5) | It costs **2.71× quality-mode render time** today (§8), ~90% of which is the missing GPU port. |
| Win concentrated on clips where propagation barely runs (0.17% / 30.98% fallback) | On the four canonical `sample.mp4` windows the win is small and DISTS-negative on two. |
| | It **overlaps with `region_aware=True`**, which is ON in the same mode and already blends toward per-frame compact on moving pixels — untested in combination (§12.4). |

So the honest verdict is the one the flag already encodes: **GATED-GO as an opt-in, NO-GO as a
default.** What is new is that the *reason* has changed. The R12 review deferred the flip pending
"a broader multi-clip tOF sweep". That sweep has now been run, and it does not settle the
question in the gate's favour — it shows the original measurement could not have settled it
either way, and surfaces a DISTS failure that the original arbiter could not see.

**Concretely, before this is worth re-litigating (in priority order):**
1. **Fix the trust ceiling** (§3.1). `a ≤ 0.881` means ≥11.9% of every pixel is per-frame SR even
   at a perfect warp; `gate_res_norm` cuts the jelly tOF cost from +14.5% to **+6.6%** for almost
   nothing on the perceptual columns. Any future evaluation should evaluate the *fixed* gate.
2. **GPU-port the soft lerp** — 2.45× of the 2.71× cost is `derisk.py:583` forcing numpy.
3. **Measure gate × `region_aware`**, not gate alone. They act on the same pixels.
4. **Then** re-run §4.1 with DISTS as a hard gate, and flip only if DISTS clears on the
   `sample.mp4` windows specifically.

## 11. Answers to the two coordinator inputs, on the record

**(a) The per-block ratios in the kickoff are superseded by this round's own measurement.** I
re-derived them rather than citing them, on HD propagation error against a true GT (§2.1):
partition **4.82×** (not 8.6×), MV magnitude **non-monotonic with no meaningful spread** (not
~75×), intra hole **1.1×–14.8× per clip** (the hole term is confirmed load-bearing; its pooled
ratio depends heavily on content). Where the kickoff numbers and mine disagree, **mine are the
ones measured here and should be used.** The *direction* on partition size is corroborated (it is
the only block term that does not lose in agent A's parameterisation either, §7) — it is the
magnitude of the effect, and the MV-magnitude term entirely, that do not survive.

**(b) Hysteresis was measured, not deferred** — §6. NO-GO in the minimal (motion-compensated EMA)
form: ≤2.1% LPIPS for ≤1.7% more tOF, nowhere near D's −29% oracle ceiling. **Top remaining
follow-up** in this direction: the *asymmetric/sticky* form (distrust decays slower than it
rises, `a = min(a_raw, λ·a_prev + (1−λ))`) — untested here, and the only membership-shape variant
left that the three convergent findings (A/C/D) would predict differently from the symmetric EMA.

---

## 12. Scope, limits and honest caveats

1. **Phase-1's error is measured with a PERFECT anchor.** That isolates warp error (GOTCHA #15) but
   excludes accumulated backbone drift. Drift is invisible to *both* the residual (which is measured
   against the true decoded LR, not the drifted recon) and the block signals, so the *comparison*
   between predictors is unaffected — but neither predictor sees drift, and nothing here claims
   otherwise.
2. **Phase-1/2 run on 2× degrade-and-restore probes** (LR 320×160 / 240×136), not the shipping x4 QHD
   geometry. The protocol is the project's own (R5-E2) and is the only way to obtain a true HD GT with
   real codec MVs, but partition statistics and residual magnitudes at 320×160 are not guaranteed to
   transfer verbatim to a 640×320 encode. Phase 2b is the un-degraded, native-geometry control.
3. **The re-encode is second-generation** (sample.mp4 is already H.264; the probe LR is a second
   libx264 pass). Again the established protocol, and the robustness clips are first-generation.
4. **The region-aware OUTPUT pass is not applied** in the sweeps, so the gate is measured in isolation.
   In shipping quality mode `region_aware=True` blends the output toward per-frame *compact* SR exactly
   where motion is — i.e. it already does part of what the gate does, on the same pixels. A fair
   default-flip decision would have to measure gate-on-top-of-region-aware; this experiment did not,
   and that is an argument for *more* caution about the flip, not less.
5. **Box contention.** All four other R13 agents were running; load averages of 20–43 were normal
   during these runs. Absolute ms are inflated; ratios and metric comparisons (which are contention-
   independent) are the load-bearing numbers.
6. **`c_floor`, `part_lo/hi`, `mv_tau/s` were not tuned per clip.** They were set from the phase-1
   distributions before phase 2 ran. Given that the best-case *ranking* weight measured is w = 0, a
   tuning sweep is very unlikely to rescue the signal, but it was not attempted.

## 13. What should happen next (in priority order)

1. **Leave `RESIDUAL_GATE = False`.** Update the flag's comment in `server/pipeline_api.py:238-248`:
   the "pending a broader multi-clip tOF sweep" note should become "swept in R13-E2; the R12-E1 arbiter
   was confounded; keep default-OFF". *(Not done here — `server/**` is outside this agent's ownership.)*
2. **If anyone wants the gate ON**, the prerequisite is not more clips, it is a **GPU port of the soft
   lerp** (kills 2.45× of the 2.71× cost) plus **`residual_gate.reliability_hd` reusing the `fx/fy` and
   `react` that `_warp_one` already computed** (the module's own `PERF TODO`).
3. **Consider fixing the trust ceiling** (§3.1) before any further gate evaluation. A gate that takes
   ≥11.9% per-frame SR from a perfect warp is not "soft occlusion handling", it is a global blend with
   a soft occlusion term on top; the `gate_res_norm` ablation in `phase2b_realtof.py` is the one-line
   version of the fix and separates the two effects.
4. **Do not spend more rounds on codec block metadata for propagation gating.** Between R12-E2 (QP) and
   this round (partition size, MV magnitude), the pattern is consistent and now has an explanation:
   block metadata describes *how the encoder spent bits*; the residual measures *whether the prediction
   was right*. For gating a prediction, only the second is decision-relevant. CDA-VSR reached the same
   conclusion from the other direction.
