# R13-E5 — SEAM VERIFICATION of the R13 round (agent E)

**Owner:** agent E. **Branch:** `r13-dlss5-ports`. **Mandate:** break the round, not confirm it.
**Files written:** `experiments/r13_e5_seam/**` only. The working tree at the end of this pass is byte-for-byte the tree at the start (§4).

Everything below was **re-run or re-derived here**. Where I could not verify something inside the timebox it says **UNVERIFIED**, never "verified".

## 0. Headline

| claim | verdict |
|---|---|
| A's browser numbers reproduce | **YES — exactly.** All 12 published LPIPS/DISTS cells re-rendered and re-scored from scratch; every one matches to 5 decimals (§1.1). |
| A's headline "softblk kp6/km6 beats hard on all four metrics on both windows" | **SURVIVES**, but only two of the four metrics were ever admissible; on those two it still wins (§2.1). |
| A's mechanism claim "km carries most of the effect, kp only moves PSNR/tOF slightly" | **Made on disqualified metrics** — but I re-measured it on LPIPS/DISTS and **it is true, and stronger than A claimed**: km is ~all of the effect, kp is ~inert (§1.2). |
| B's "the MV-magnitude term (km) is the harmful half; the server's answer is kp=km=0" | **TRUE on B's rig, FALSE in the browser.** In the browser, on the arbiters B itself declares valid there, km is the *entire* win and kp is the inert half. A **real disagreement**, not a constants difference (§1.3). |
| B's reconciliation ("no functional-form divergence, only constants, because the browser's fallback is bilinear LR") | **RATIONALISATION.** It predicts the browser should want *less* distrust; the browser measurably wants *more*, from exactly the term B calls harmful (§1.4). |
| B's methodological rule (fallback ≡ reference ⇒ distrust meter) | **CORRECT and it should become a GOTCHA** (§2). |
| B's "R12-E1's −29% LPIPS is confounded" | **CORRECT for LPIPS.** Verified against R12-E1's own raw JSON. **OVERSTATED for DISTS** — in the same file DISTS is *non-monotone* with an interior optimum and the gate beats the per-frame-SR endpoint by 3.8%, which dilution cannot produce (§2.3). |
| B's Q1 "block signals add zero" | **OVER-CLAIMED.** B's own end-to-end table contains `gate_res_bmul`, which turns the +27.3% DISTS regression that is the whole basis of B's Q2 NO-GO into **−15.1%** on the same window. Absent from B's main tables and verdict (§3.4, defect **D5**). |
| B's two invariants (bit-identity at `c_floor=1`, hole-set equality) | **VERIFIED independently, both exactly** (§3.2). |
| The confidence formula is one function across implementations | **YES for three of them** (WGSL ≡ `block_confidence.py` to 9e-14; `block_confidence(c_floor=1)` ≡ `residual_gate` bit-identically). **The fourth does not exist**: `gpu_ops.py` has no gate at all (§3.1, defect **D3**). |
| All four OFF paths are byte-identical | **VERIFIED**, including real-browser pixel-hash equality for `?gate=hard` (§4). |
| Branch coherent, nobody overwrote anyone, user's `r12_e4` edits intact | **VERIFIED** (§4). |
| One real end-to-end browser run with the gate actually executing | **DONE** — 15 real warps through the soft gate in Chrome, pixel-level A/B, zero console errors (§5). |
| C's M2 "landed default-OFF" | **Nothing was landed.** `stabilizer.py` lives in the experiment dir, imported by no shipping module (§6, defect **D8**). |
| D's recommended default flip (`quality.occ` adaptive→reactive) | **DO NOT LAND IN R13** — untested overlap + D's own production arm never ran (§8). |

**Net:** the round's *measurements* are unusually solid — A reproduced perfectly, B's invariants and headline table reproduced, C's and D's parity claims hold. What does **not** survive is a set of *attributions and reconciliations* layered on top of them, and one cross-agent claim (B §7's answer to A) that is now measurably backwards for the browser.

## 1. PRIORITY 1 — the A-vs-B contradiction, resolved by measurement

### 1.1 First: is A reproducible at all?

Re-ran A's whole pipeline from scratch: `prep_and_probe.py` (fresh window extraction + LR probe), 12 fresh `harness.ts` renders (2 windows × {ref, hard, soft, softblk kp6km6, **kp6km0**, **kp0km6**}), then `eval.py`.

* LR probe reproduces A's §3 table exactly (calm: hole 1.2%, hard fallback 8.96%, react 3.871, MAE 1.889 vs 16.516 = 8.7×, mv 1.272/12.375/23.331).
* `gate_mean_a` reproduces to 4 decimals on every variant (calm soft 0.8992, softblk 0.8799, kp6km0 0.8926, kp0km6 0.8872; jelly soft 0.7008, softblk 0.6543).
* **Every published LPIPS and DISTS cell reproduces to 5 decimals.**

A's *numbers* are not in question anywhere in this report.

### 1.2 The measurement A did not run: LPIPS/DISTS on the kp/km split

A's §4b grid has `lpips_warp: NaN` on every kp/km row (confirmed in `results_{calm,jelly}_grid.json`) — the split was screened on PSNR and tOF only, which A itself disqualified in the same section. I filled it in.

**calm** (propagated frames only, vs the SR-every-frame ceiling):

| variant | PSNR | LPIPS | DISTS | tOF(LR) | A published LPIPS/DISTS |
|---|---|---|---|---|---|
| hard (shipped) | 30.257 | 0.04360 | 0.07265 | 0.1477 | 0.04360 / 0.07265 ✓ |
| soft t16 s3 (kp0 km0) | 30.764 | 0.04257 | 0.06729 | 0.1540 | 0.04257 / 0.06729 ✓ |
| **+ kp6 km0** | 30.917 | **0.04249** | **0.06707** | 0.1458 | *(new)* |
| **+ kp0 km6** | 31.037 | **0.04157** | **0.06657** | 0.1381 | *(new)* |
| + kp6 km6 (A's config) | 31.155 | 0.04175 | 0.06636 | 0.1312 | 0.04175 / 0.06636 ✓ |

**jelly**:

| variant | PSNR | LPIPS | DISTS | tOF(LR) | A published |
|---|---|---|---|---|---|
| hard (shipped) | 26.639 | 0.07477 | 0.09942 | 1.2228 | 0.07477 / 0.09942 ✓ |
| soft t16 s3 (kp0 km0) | 26.754 | 0.06623 | 0.09502 | 1.1614 | 0.06623 / 0.09502 ✓ |
| **+ kp6 km0** | 26.771 | **0.06634** | **0.09455** | 1.1575 | *(new)* |
| **+ kp0 km6** | 26.979 | **0.06004** | **0.09025** | 1.0615 | *(new)* |
| + kp6 km6 (A's config) | 26.990 | 0.06022 | 0.08980 | 1.0649 | 0.06022 / 0.08980 ✓ |

**Each block term against the residual-only soft gate, on the surviving metrics:**

| window | term | ΔLPIPS | ΔDISTS |
|---|---|---|---|
| calm | kp=6 alone | **−0.19 %** | **−0.33 %** |
| calm | km=6 alone | **−2.35 %** | **−1.07 %** |
| calm | both | −1.93 % | −1.38 % |
| jelly | kp=6 alone | **+0.17 %** | **−0.49 %** |
| jelly | km=6 alone | **−9.35 %** | **−5.02 %** |
| jelly | both | −9.07 % | −5.49 % |

**Reading.** In the browser, **`km` (MV magnitude) is essentially the entire block-signal win and `kp` (partition size) is essentially inert** — on LPIPS `kp` alone is −0.19 % / +0.17 %, i.e. noise in both directions, while `km` alone recovers 100 %+ of the combined LPIPS gain. `kp` contributes a small consistent DISTS-only sliver (−0.33 % / −0.49 %), which is why keeping it is still defensible.

**This is not a dilution artifact.** Mean trust *falls* when km is added (calm 0.8992 → 0.8872, jelly 0.7008 → 0.6577) — the gate leans **harder** on the bilinear-LR fallback — and LPIPS/DISTS *vs the SR reference* nevertheless improve. Bilinear LR is far from the SR reference perceptually, so the dilution direction is **against** this metric. Whatever `km` is doing, it is not "softening the frame toward the metric's reference".

### 1.3 So which side is right? Both, about different things — and the disagreement is real

| | agent B (server) | agent A (browser), re-measured here |
|---|---|---|
| quantity measured | *ranking power*: share of true HD propagation-**error mass** captured by the top-10 % of predicted-unreliable area | *rendered output*: perceptual distance of the finished frame from the SR ceiling |
| rig | 2× degrade-and-restore, LR 320×160 / 240×136, **perfect anchor, single hop** | native 640×320 → 1280×640, real compact SR anchors, **4-frame propagation chain** |
| verdict on `km` | **harmful** (kp0/km6 = −9.1 % ranking power) | **the entire win** (kp0/km6 = −9.35 % LPIPS on jelly) |
| verdict on `kp` | the only non-losing term (+0.38 %, noise) | inert (−0.19 % / +0.17 % LPIPS, noise) |

They *agree* on `kp`. They disagree only on `km`, and the sign is flipped. Three mechanisms plausibly explain it; all three are properties of B's rig that B documents as limitations without connecting them to this answer:

1. **B's rig cannot see accumulated drift** — B's own caveat §12.1 ("measured with a PERFECT anchor … neither predictor sees drift"). The browser is a 4-hop display-order chain. MV magnitude is the obvious proxy for "how far this pixel has been dragged", i.e. exactly the quantity B's single-hop rig defines away. That alone can flip a term's sign.
2. **`mvref = 8` is resolution-dependent.** B ran A's form verbatim at LR 320×160/240×136 (its §12.2 caveat), where 8 px is roughly twice the frame fraction it is at 640×320 — and B's own non-monotonicity kicks in exactly at ≥8 px. Re-running A's form at A's geometry was not done.
3. **Ranking power ≠ output quality.** The gate does not *choose* pixels, it *lerps* them; over-distrusting a pixel whose fallback is acceptable costs almost nothing, while under-distrusting a geometrically-wrong warp costs a visible seam.

None of these is settled here. What *is* settled: the constants cannot be transplanted between the two sides, and B's "there is no divergence to reconcile" is wrong.

### 1.4 Is B's stated reconciliation sufficient?

**No — it is a rationalisation, and it points the wrong way.** B's argument is: `(1−a)` is bilinear LR in the browser (costs sharpness) and free per-frame SR on the server, so the browser "needs a gate that saturates to 1". That predicts the browser should want **less** distrust — smaller kp/km, larger tau. The measurement says the opposite: the browser's best configurations distrust **more** (mean a 0.899 → 0.887 → 0.880 calm, 0.701 → 0.658 → 0.654 jelly, LPIPS/DISTS improving monotonically along that direction). The fallback-source asymmetry is a true statement about the two systems; it is not an explanation of this result and should not be recorded as one.

### 1.5 Recommended constant set — one per side

* **Browser (opt-in, default stays OFF):** `?gate=softblk&tau=16&s=3&kp=6&km=6&mvref=8` — **unchanged**, now supported by LPIPS/DISTS rather than PSNR/tOF. Record that **`km` is the load-bearing half and `kp` is within noise on LPIPS**; if simplified, drop `kp`, never `km`. `kp=6,km=0` (B's server answer) is **the worst of the three block configurations in the browser** — indistinguishable from no block term at all.
* **Server:** `kp = km = 0` — **unchanged**. B's ranking evidence stands on its own rig and no server-side end-to-end measurement contradicts it. **Caveat:** B's own end-to-end table says something different for the multiplicative form on high motion (§3.4).
* **Not resolved (needs measurement X):** whether `km` is a drift proxy or a genuine geometric prior. The decisive experiment is B's `phase1b_formula_ab.py` gain curve re-run **at native geometry with a multi-hop propagated reference instead of a perfect anchor** — one parameter change to an existing script, and it settles the sign of `km` for both sides.

## 2. PRIORITY 2 — B's metric rule applied to everyone

### 2.1 Applied to A

Rule: *whenever the gate's fallback source coincides with (or is trivially close to) the metric's reference, that metric is a distrust meter, not an arbiter.* Browser fallback = bilinear LR ⇒ **PSNR-vs-SR-reference and tOF-vs-LR are disqualified.**

| A's claim | status |
|---|---|
| calm PSNR +0.90 dB, jelly +0.35 dB | **DISQUALIFIED** (A says so itself, in §4b's METRIC TRAP box) |
| calm tOF −11.2 %, jelly −12.9 % | **DISQUALIFIED** |
| calm LPIPS −4.2 %, DISTS −8.7 % | **SURVIVES** (reproduced) |
| jelly LPIPS −19.5 %, DISTS −9.7 % | **SURVIVES** (reproduced) |
| "the only variant with **no regression on any metric on any window** — including tOF" | **DOES NOT SURVIVE.** This is the load-bearing argument for choosing `softblk` over `soft t16s3`, and it is built on the disqualified tOF column. |
| "softblk dominates every residual-only variant" | **SURVIVES on the surviving metrics** — softblk beats soft t16s3 on LPIPS *and* DISTS on both windows (calm −1.9 %/−1.4 %, jelly −9.1 %/−5.5 %). |

**Was A's final config chosen on the surviving metrics or the disqualified ones?** The *prose* justification is the disqualified one ("it turns calm tOF from +4.3 % into −11.2 %"). The *choice* happens to be correct on the surviving ones anyway. Both facts belong on the record: the config survives, the argument for it does not. The kp-vs-km attribution was made purely on the disqualified pair — right answer, wrong evidence, until now.

### 2.2 Applied to B, C, D

* **B** applies its own rule to itself consistently (true-GT arbiter where neither endpoint is the reference). ✔
* **C**: its lever is a temporal EMA of the propagated output, not a fallback-selection gate — the EMA pulls toward its own warped history, not toward per-frame SR, so the rule does not bite. Its MCLF/tLP are explicitly calibrated against bicubic-LR, per-frame-compact and per-frame-x4plus endpoints in the same table, which is the right shape. ✔
* **D** flags the rule itself and correctly notes the bias runs the other way in its rig; its verdict rests on LPIPS/DISTS, which are flat. ✔

### 2.3 The R12-E1 "−29 % LPIPS / −23 % DISTS" claim — verified specifically

Verified directly against `experiments/r12_e1_residual_gate/out/results_jelly.json` (R12-E1's own raw artifact, not B's re-analysis):

| variant | distrust | LPIPS | DISTS | tOF_true |
|---|---|---|---|---|
| baseline_hard | — | 0.035349 | 0.037474 | 1.6537 |
| gate_res_t16 | 0.5609 | 0.026071 | 0.029946 | 1.7291 |
| **gate_res_t10 (the flag)** | 0.5791 | **0.025093** (−29.0 %) | **0.028809** (−23.1 %) | 1.7499 |
| gate_res_t7 | 0.5996 | 0.024777 | 0.028742 | 1.7673 |
| gate_resfb_t10 | 0.6523 | 0.024547 | 0.028304 | 1.8186 |
| **compact_perframe (a→0 limit)** | 1.0 | **0.024220** | **0.029956** | 2.2425 |

**On LPIPS, B is right and should be quoted as right.** LPIPS is strictly monotone-decreasing in distrust and the a→0 endpoint is the *best* LPIPS in the table. The flag's −29 % is 92 % of the way from "propagate" to "give up and run per-frame SR". **The −29 % LPIPS number should stop being quoted as a quality win.**

**On DISTS, B is overreaching, and B's own §3.2 table hides it by omitting the DISTS column.** DISTS is **not** monotone in distrust: it falls to 0.028304 at distrust 0.652 then *rises* to 0.029956 at distrust 1.0. The flag setting (0.028809) is **3.8 % better than the a→0 endpoint**. By B's own argument — the one it uses two paragraphs later for the calm tOF win ("better than both endpoints, which no interpolation can produce") — the DISTS −23 % contains a component dilution cannot explain.

**Verdict:** narrow B's blanket sentence to *the LPIPS half of R12-E1's headline is confounded; the DISTS half is not (or not entirely)*. The *decision* B draws (keep `RESIDUAL_GATE = False`) is unaffected — it rests on the independent 8-clip true-GT sweep, not on this re-analysis.

**One sub-claim of B's is exactly right:** "in the shipping wiring they are literally the same images." Verified — `MODE_CONFIG["quality"]` is the only mode carrying the gate sentinel (`server/pipeline_api.py:332`) and it is `sr_mode="realesrgan-x4plus"` (`:305`) fed straight to `derisk.build_perframe_cache` (`:882`). Any future evaluation of the shipping gate against a per-frame-x4plus reference would score `a → 0` as a perfect zero.

## 3. PRIORITY 3 — the confidence formula across implementations

### 3.1 There are three implementations, not four

| # | locus | what it implements | state |
|---|---|---|---|
| 1 | `prototype/residual_gate.py:48-60` (numpy) | `a = sigmoid((tau−react)/s)`, `a=0` at LR hole, INTER_LINEAR → HD, hard-zero at NEAREST HD hole | the reference |
| 2 | `prototype/block_confidence.py:113-125` + `:165-200` (`mode="penalty"`) | A's additive block penalty, verbatim | ✔ equals #3 |
| 3 | `web_spike/wasm_mv/warp_shader.js:89-104` + `:56-72` | the same, in WGSL | ✔ equals #2 |
| 4 | `prototype/gpu_ops.py` | **nothing — there is no gate here at all** | **defect D3** |

`prototype/derisk.py:583` routes *any* enabled gate to the numpy reconstruct. There is no torch twin of `reliability_lr/_hd`, so the shipping server's GPU path cannot run the gate — B's §8 cost finding (2.45× of the 2.71× is this line) stated as a seam: **the only GPU implementation of this formula in the repo is the browser's WGSL one, and no automated test compares it to the numpy reference.** (A's `parity_gate.py` does compare them, at 0.00138 mean / 0.0418 max — but it needs a populated scratch dir and is not part of `parity_shader.mjs`, the guard that actually runs.)

**Functional diff, #2 vs #3, measured not read** (`verify_invariants.py`, 11 frames of the jelly window, `want="all"`, tau=16 s=3 kp=6 km=6 mvref=8):

```
max | a_WGSL(numpy transcription) − block_confidence(mode="penalty") | = 9.36e-14
```

The additive-vs-multiplicative question A raised in its §9.5 hand-off is closed: **A's additive form is the single reference formulation**, and `block_confidence.py` implements it verbatim (`area/256` normalisation, `min(|mv|/mvref, 1)` clamp, `max(·, 0)` floor).

**Deviations found, and whether documented:**

| # | deviation | documented? | severity |
|---|---|---|---|
| 1 | `(1−a)` source = bilinear LR (browser) vs fresh per-frame SR (server) | yes (`warp_shader.js:17-23`, A §2.1) | deliberate |
| 2 | residual-only; no Ruder fwd-bwd term in either soft path | yes | deliberate |
| 3 | display-order `prevRecon` chain vs I/P backbone (GOTCHA #12) | yes (A §2.3) | deliberate, browser-only |
| 4 | `a` in `r16float` + GPU true-float bilinear vs `cv2.remap` 1/32-px | yes (A §2.4) | benign, inside 0.0014 parity |
| 5 | sigmoid arg clamped ±30 (WGSL), ±60 (`block_confidence`), **unclamped** in `residual_gate.reliability_lr` | **no** | benign in float64; matters for a float32 port — worth one comment line |
| 6 | NaN handling: python maps `area`→0 / `mvmag`→∞; WGSL returns early at the hole sentinel | implicitly | equivalent (the 9e-14 number) |
| 7 | HD upsample mapping `textureSampleLevel(reliT, s, pos.xy/(W,H))` ≡ `cv2.resize(INTER_LINEAR)` | yes (`warp_shader.js:63-64`) | correct — checked algebraically (`(dst+0.5)/scale − 0.5` both sides) |

No **undocumented** functional deviation was found between #1, #2 and #3.

### 3.2 B's two invariants — independently verified

`experiments/r13_e5_seam/verify_invariants.py`, 12 frames of `sd600.mp4` @582, all three direction filters, 33 (frame, direction) pairs:

```
INV1  max|reliability_lr_bc(c_floor=1.0, mode="tau") − residual_gate.reliability_lr| = 0.0   PASS
INV2  isfinite(block_signals(...)) == isfinite(build_lr_flow(...))  on 33/33 pairs           PASS
```

Both hold **exactly**, as claimed. (Repro note: run from a directory that is *not* the shared scratchpad — a stray `av.py` arXiv helper there shadows the PyAV package.)

### 3.3 `packLrFlowRGBA` vs the plane path

`parity_shader.mjs` check [3] re-run: **IDENTICAL**. Read the two paths against each other (`flow.js:8-33` vs `:47-77`): same record loop, direction filter, `dst = block CENTRE` extent, last-writer-wins order, `-1e9` hole sentinel. The only behavioural difference is that the packer writes `.zw` (area, |mv|), read only when `blocks:true`. ✔

### 3.4 A block-signal result of B's that B did not report

Re-derived from B's own generated `out/summary_tables.md` (Δ vs `baseline_hard`, negative = better, true-GT arbiter):

| clip | `gate_res` ΔDISTS | `gate_res_block` ΔDISTS | `gate_res_bmul` ΔDISTS |
|---|---|---|---|
| **A_highmotion** | **+27.3 %** | **+6.8 %** | **−15.1 %** |
| C_talkinghead | −2.9 % | −0.5 % | +1.6 % |
| c1_fastpan | −23.3 % | −24.7 % | −24.5 % |
| c3_lowlight | +4.1 % | +5.0 % | +6.5 % |
| c4_talkinghead | −26.3 % | −29.0 % | −32.1 % |

1. B's §4.1 parenthetical — *"`gate_res_block` tracks `gate_res` to within ~1 pt on every cell"* — is **false on the cell that decides the round**: A_highmotion DISTS, +27.3 % vs +6.8 %, a 20.5-point gap.
2. `gate_res_bmul` **reverses the sign** of the +27.3 % DISTS regression that is B's entire stated reason for the Q2 NO-GO, turning it into −15.1 %. It appears nowhere in B's tables or verdicts. It ran on only 5 of 8 clips and is *worse* on `C_talkinghead` and `c3_lowlight`, so it is not a clean win — but "the block prior is redundant, add zero" is not what this data says end-to-end on the failing window.

Same error class B correctly diagnoses in others: **Q1 was decided on a ranking proxy (gain@top-10 %) that measurably disagrees with the arbiter B itself trusts (true-GT DISTS) on the one window the flip decision turns on.** Narrow B's Q1 to "NO-GO *as a ranking improvement*; the multiplicative form's DISTS behaviour on high motion is unexplained and unreported".

## 4. PRIORITY 4 — OFF-path parity, independently re-run

| lever | claim | how verified | result |
|---|---|---|---|
| **A** — `?gate=hard` byte-identical | `parity_shader.mjs` | re-ran it: `[1] IDENTICAL (1215 vs 1215 chars) · [2] UNCHANGED · [3] IDENTICAL · PARITY OK` | ✔ |
| **A** — the *page* is unchanged, at pixel level | not claimed | **real Chrome**: `?demo` and `?demo&gate=hard`, both driven to the identical state → `leftMean 64.7569`, FNV hash `1427851492` **identical in both** | ✔ stronger than claimed |
| **B** — λ=0 byte-identical | asserted in `phase5_hysteresis.py:138-144` | found the recorded log lines: `[jelly] PARITY hysteresis(lam=0) == gate_res: 0.0 codes`, `[calm] … 0.0 codes` | ✔ |
| **B** — `prototype/`/`server/`/`web_spike/` unmodified | monkeypatch-only harness | `git show --stat 460a566`: 11 files, all its own dir + the **new** `prototype/block_confidence.py`; no existing file edited | ✔ |
| **C** — `STAB_ENABLED=False` → 0-code parity | report | `out/results_{calm,jelly}.json` both record `off_parity_codes: 0`; `stabilizer.py:149-153` is a real early return | ✔ — but see D8 |
| **D** — touched no shipped file | assertion | `git show --stat efae862`: 10 files, **all** under `experiments/r13_e4_fwdbwd/`; no D-owned working-tree modification | ✔ verified by diff |

**Branch coherence.** `eca343b`/`d3ae309`/`ead5946` (C) touch only `experiments/r13_e3_jelly/**`; `efae862` (D) only `experiments/r13_e4_fwdbwd/**`; `460a566` (B) only its own dir plus one new file. **No two agents touched the same file.** A's tree is uncommitted as stated: `M web_spike/wasm_mv/flow.js`, `M web_spike/wasm_mv/player.html`, `?? web_spike/wasm_mv/{warp_shader.js,parity_shader.mjs}`, `?? experiments/r13_e1_browser_confidence/`.

**The user's pre-existing `r12_e4` edits are INTACT and untouched.** `experiments/r12_e4_retrain_prep/{run_full.sh,train.py}` still show the same modification, `ab_eval.py` is still the same untracked file, and every file in that directory has an mtime of **Jul 1–10** — nothing in this round wrote there. Re-running `prep_and_probe.py` rewrote `probe_lr.json` with byte-identical values inside A's own untracked directory; nothing else changed.

## 5. PRIORITY 5 — one real end-to-end browser run

Served `web_spike/` with `python3 -m http.server 8123` and drove Chrome directly. The shipped page defaults to `REANCHOR = 1` (`player.html:273`) = SR every frame, where no warp and therefore no gate runs — so every run below **raised the propagation cadence to SR-every-4 first**, and to make the arms comparable each run was first driven to an identical state (40 pure-SR anchor frames, whose HD output has no history dependence) before switching to cadence 4.

| URL | gate per `__diag()` | anchors / warps | left-half mean | FNV hash |
|---|---|---|---|---|
| `?demo` (default) | `hard` | 45 / 15 | 64.7569 | 1427851492 |
| `?demo&gate=hard` | `hard` | 45 / 15 | 64.7569 | **1427851492 (identical)** |
| `?demo&gate=soft&tau=16&s=3` | `soft`, kp0 km0 | 45 / 15 | 64.7644 | 3133670473 |
| `?demo&gate=softblk` (no params) | `softblk`, **kp0 km0** | 45 / 15 | 64.7644 | **3133670473 (identical to `soft`)** |
| `?demo&gate=softblk&tau=16&s=3&kp=6&km=6` | `softblk`, kp6 km6 | 45 / 15 | 64.7661 | 1855822140 |

* **Rendering works and the gate genuinely executes**: 15 real warps per run through the soft path, the `r16float` reliability target, the two-pass encoder and the `.zw` packing all on Chrome/Dawn's f16 path. Screenshot confirms live playback with the slider at "SR every **4** frames" and the badge reading `propagated · codec MV-warp`.
* **Zero console messages** (not just zero errors — the console is empty), no `⚠` in the page, no error/failure text in the DOM. No JS dialogs triggered.
* **`?gate=hard` is pixel-identical to the default page** — the strongest form of A's default-OFF claim, and it holds.
* **`?gate=softblk` with no parameters is pixel-identical to `?gate=soft`** — the block terms default to 0 (defect **D1**).

**Do hard and softblk "differ visibly"?** They differ **measurably and locally**. At the same propagated frame (fi=59, 14 warps, identical cadence), over 800 uniform blocks of the playhd half: **max block-mean Δ = 3.36 codes, mean Δ = 0.047 codes, 18/800 blocks above 0.5 codes.** That is exactly the shape A's report shows (blocky hard-switch patches replaced by feathered ramps, confined to the fallback set) — but on this window I would **not** describe it as visible at normal viewing size, and A does not claim that either: its visual evidence is worst-error 256×256 crops with |error| maps, the honest way to present a localised difference. *(Caveat: this frame is in the demo clip's opening, not the high-motion jelly window where the fallback set is 38 % of the frame; the difference there is ~8× larger by A's whole-frame |err| table.)*

## 6. Defect list

**D1 — `?gate=softblk` alone is a no-op; the recommended configuration is unreachable without explicit parameters.**
`web_spike/wasm_mv/player.html:96` — `kp: gnum("kp", 0), km: gnum("km", 0)`. A's recommended opt-in is `kp=6, km=6`, but the flag's own defaults are 0/0, so `?gate=softblk` is pixel-identical to `?gate=soft`. *Repro:* §5, rows 3–4 (same hash `3133670473`). *Severity:* low-medium (ergonomics trap, not a wrong render; 0/0 is arguably the safer default). *Fix:* default to 6/6, or state in the flag comment and report that `softblk` ≡ `soft` unless `kp`/`km` are passed. *(Silver lining: the shipped browser default for the block terms — `kp=km=0` — already agrees with B's server answer, so the two sides ship the same constants today.)*

**D2 — `?gate=soft&kp=…&km=…` silently ignores the block parameters, and `__diag()` misreports the running configuration.**
`player.html:96` parses `kp`/`km` for any gate; `:208` builds the reliability shader with `blocks: GATE_BLOCKS` (false for `gate=soft`), so `warp_shader.js:97-99` is not emitted and the parameters have no effect; `:314` nevertheless returns `gateCfg: GCFG` including `kp: 6, km: 6`. *Repro:* load `?gate=soft&kp=6&km=6`, call `__diag()` → reports kp 6 / km 6; the render is byte-identical to `?gate=soft`. *Severity:* medium — a diagnostic that reports a configuration the shader is not running is exactly how a seam bug survives review. *Fix:* zero `kp`/`km` in `GCFG` when `!GATE_BLOCKS`, or warn. *(Code-inspection finding; the exact `?gate=soft&kp=6` load is **UNVERIFIED empirically**.)*

**D3 — the confidence gate has no torch/GPU implementation, and the one GPU implementation that exists (WGSL) has no guard in the automated parity script.**
`prototype/gpu_ops.py` — no reliability function of any kind; `prototype/derisk.py:583` forces numpy whenever a gate is passed, even under `backend="torch"`. *Repro:* `grep -n "reliability\|residual_gate" prototype/gpu_ops.py` → no hits. *Consequence:* (a) enabling `RESIDUAL_GATE` costs 2.45× purely from backend routing (B §8 — absolute **UNVERIFIED** here); (b) the numpy reference and the WGSL implementation can drift with nothing in `parity_shader.mjs` to catch it. *Fix:* fold a small fixed-input `reliability_lr` vector (a few hand-written MV records + two tiny LR frames, expected `a` checked in) into `parity_shader.mjs` so WGSL-vs-python equality runs without data prep.

**D4 — with the gate enabled, `occ_mode` no longer affects the output but still pays for the Ruder splat.**
`prototype/derisk.py:620-632`: `_warp_one(..., occ_mode, ...)` still computes `occ` (including the softmax splat under `occ="full"/"adaptive"`), but the gated branch replaces the `recon[occ] = perframe[occ]` hard blend with the reliability lerp; `occ` survives only as `R[i]["mask"]` / `hole_frac` bookkeeping. Same in the B-leaf path at `:667-687`. *Repro:* read the branch — `_resgate is not None` never reads `occ`. *Consequence:* `MODE_CONFIG["quality"]` is `occ="adaptive"` (`server/pipeline_api.py:307`) **and** the only mode carrying the gate sentinel (`:332`). If `RESIDUAL_GATE` is ever flipped ON, quality mode pays D's measured 12.6 ms/mask-call (GPU) / 63.8 ms (numpy) for a mask its output ignores.

**D5 — B's report contradicts B's own generated data on the cell that decides the round.** See §3.4. `experiments/r13_e2_block_confidence/REPORT.md` §4.1 parenthetical vs `out/summary_tables.md`. *Severity:* high for the round's conclusions.

**D6 — A's kp/km attribution was made on the two metrics A disqualified in the same section.** `experiments/r13_e1_browser_confidence/REPORT.md` §4b: every kp/km row except `kp6 km6` has `—` in LPIPS/DISTS (`lpips_warp: NaN` in `results_{calm,jelly}_grid.json`), and the sentence "the MV-length prior (km) carries most of the effect" rests entirely on them. *Status:* the conclusion is now **independently confirmed on LPIPS/DISTS** (§1.2) — but it was not established by the evidence presented.

**D7 — B's §7 reconciliation of the A/B divergence is a rationalisation.** See §1.4. Replace with: "the two rigs measure different quantities (ranking power at a perfect single-hop anchor vs rendered perceptual quality over a multi-hop chain); the constants do not transfer; the sign of `km` is unresolved".

**D8 — C's M2 was not "landed"; nothing in the shipping tree references it.** `experiments/r13_e3_jelly/stabilizer.py:63` (`STAB_ENABLED = False`) is a module-level flag in an *experiment* directory. *Repro:* `grep -rn "stabilizer" prototype server web_spike` → no hits. The "0-code parity" claim is true and the code genuinely early-returns (`:149-153`), but it is parity of a module nothing calls. *Severity:* low as code, real as wording — the next round should not go looking for a flag to flip.

**D9 — a measured NO-GO module now lives in the shipping-adjacent package.** `prototype/block_confidence.py` is imported by nobody outside `experiments/r13_e2_block_confidence/`. Not a defect; flagged so it is a deliberate choice. If the browser keeps `km`, this module is the natural home for the shared reference formulation and should stay — with its header amended to say the *browser* uses it and the *server* does not.

## 7. Untested overlaps

| pair | status |
|---|---|
| `RESIDUAL_GATE` × `region_aware=True` | **UNTESTED.** Both ON in quality mode's config space, both blend the output toward a per-frame SR source on the same moving pixels (`derisk.py:517-540` region gate, `:624-632` reliability lerp). B's §12.4 flags it; the sweeps ran with `region_aware` off. **The largest untested interaction in the round.** |
| `RESIDUAL_GATE` × C's M2 | **MEASURED** (B §5, `phase4_compose.py`, log verified here): composed tOF_true cost +6.3 % (jelly) / +0.5 % (calm) vs algebraic sums +9.3 % / +0.1 % — sub-additive on jelly, additive on calm. Neither amplifies the other. ✔ |
| D's `occ` flip × everything else | **UNTESTED, same pixel set.** D's mask-stability table shows the flip set carries 19.7–27× the flicker; C's stabiliser damps that set; the gate softens it. D measured `occ` alone (with `region_aware` on, correctly, but with the gate off, on numpy, at scale 2). |
| C's M2 × `region_aware` | **UNTESTED.** Both are output-only post-passes writing the same output copy. Neither is default-ON, so nothing ships broken; but if M2 graduates it lands next to `region_aware` in `reconstruct_torch`'s output pass, where the ordering question becomes real. |

## 8. Verdict per lever, and the recommended integration set

**Ship nothing default-ON in R13.**

| lever | owner | recommendation |
|---|---|---|
| Browser occlusion gate | A | **Keep `?gate=hard` as the default** (verified byte-identical in a real browser). Land the code as-is *plus* the D1/D2 fixes. Recommended opt-in stays `?gate=softblk&tau=16&s=3&kp=6&km=6&mvref=8`, now justified on LPIPS/DISTS. Amend the report: **km is the load-bearing half**, kp is within noise on LPIPS, and this is the opposite of the server's ranking answer. |
| `packLrFlowRGBA` on the default path | A | **Do not land in R13.** Genuine ~2× CPU win, bit-identity-checked, but it currently ships only behind the gate flag; landing it on the default path is its own change with its own parity run (A's §9.4 says this). |
| `RESIDUAL_GATE` | B | **Keep `False`.** Reason amended: the *LPIPS* half of R12-E1's headline is confounded (verified), the DISTS half is not (verified), and the 8-clip true-GT DISTS guard fails 3/8 including the high-motion window. Update `server/pipeline_api.py:238-248` to say "the −29 % LPIPS arbiter was confounded; DISTS corroborates on 5/8", not "the evidence was confounded". |
| Block confidence on the server | B | **Do not add** (`kp = km = 0`). Narrow the Q1 verdict per D5: `gate_res_bmul`'s DISTS behaviour on `A_highmotion` is the one open server-side thread. |
| Temporal hysteresis | B | NO-GO in the symmetric-EMA form; λ=0 verified as a 0-code control. The asymmetric/sticky form remains untested. |
| M1 FSR3.1 luma instability | C | NO-GO. Keep as a diagnostic. No shipping surface. |
| M2 per-pixel adaptive EMA | C | **Validated-ready, not landed** (D8). Before graduation: GPU port, a second clip, and the `region_aware` ordering question. |
| Ruder splat / `quality.occ` flip | D | **DO NOT FLIP IN R13.** D's evidence is good and I found no hole in it, but (a) `run_prod.py` was **written and never run**; (b) the A/B ran numpy while quality ships torch; (c) the flip touches the same pixel set as two other proposed levers with no combination measured (§7). Re-propose once `run_prod.py` has run on a quiet box. |
| FSR2 temporal MV divergence | D | Settled NO-GO. Record with the spatial variant. |

**Two cheap follow-ups this pass surfaced:**

1. `residual_gate` + `occ="adaptive"` pays for a mask it discards (D4). If the gate ever ships ON, `occ` should be `reactive` in the same change — D's flip arriving as a *cost* argument rather than a quality one, and the cleanest way to land D's finding.
2. `prototype/residual_gate.py`'s own "KNOWN CAVEAT" names the dropped fwd-bwd term as the suspected mechanism behind the +5.8 % jelly tOF cost. **D's result contradicts that caveat**: the fwd-bwd term is perceptually inert (|ΔLPIPS| ≤ 0.0001 on three windows) and destabilises the flip set. So "re-adding fb" is no longer a plausible fix for the gate's tOF cost; B's trust-ceiling fix (`gate_res_norm`, +14.5 % → +6.6 %) is the better candidate. This is the one place where two agents' results compose into a conclusion neither reached alone.

## 9. What I could not verify inside the timebox

* **Any absolute timing.** Every ms number in all four reports was taken on a contended box; I re-timed nothing. B's 2.71×, D's 12.6/63.8 ms, A's +0.03 ms — all **UNVERIFIED** here (the *ratios* are the load-bearing part per project convention, and I have no reason to doubt them).
* **B's phase-1 block statistics** (102 665 MV records, gain curves, Spearman table). I verified the phase-2 end-to-end table and both invariants; the phase-1 npz was not re-derived.
* **C's entire jelly measurement** (MCLF, tLP, drift-vs-flicker regression, the M2 A/B). I verified only its OFF-path parity, its module's non-wiring, and B's independent composition measurement of `stabM2`. C's own numbers are **UNVERIFIED** by this pass. The honest C-vs-B discrepancy they already recorded (stabM2-alone tOF −0.3 %/+8.1 % vs +1.7 %/+5.2 %) was not re-adjudicated.
* **D's perceptual A/B and the 36-config FSR2 sweep.** Not re-run. D's "no shipped file touched" and its cross-references to C's `jelly_metrics` were verified; its numbers were not.
* **`?gate=soft&kp=6&km=6` in a live browser** (D2 is a code-inspection finding).
* **Whether `km`'s browser win is a drift proxy or a geometric prior** (§1.5 — the one experiment that would settle the A/B disagreement properly).

## 10. Files

```
experiments/r13_e5_seam/
  REPORT.md                this document
  verify_invariants.py     B's two invariants + the WGSL-vs-block_confidence functional diff
  e5_results_calm.json     independent re-render + re-score, 5 variants incl. the kp/km split
  e5_results_jelly.json      (LPIPS/DISTS on kp6km0 and kp0km6 are new to the round)
  rerun_browser_ab.sh      the 12 headless harness runs behind those two JSONs
```

Reproduce the decisive measurement:

```bash
SCR=<scratch>
python3 experiments/r13_e1_browser_confidence/prep_and_probe.py --scratch $SCR
bash    experiments/r13_e5_seam/rerun_browser_ab.sh            # edit SCR at the top
cd experiments/r13_e1_browser_confidence
python3 eval.py --scratch $SCR --window calm  --variants hard4,soft_t16s3,softblk_kp6km6,blk_kp6km0,blk_kp0km6 --out /tmp/calm.json
python3 eval.py --scratch $SCR --window jelly --variants hard4,soft_t16s3,softblk_kp6km6,blk_kp6km0,blk_kp0km6 --out /tmp/jelly.json
# invariants (run from OUTSIDE the shared scratchpad — a stray av.py there shadows PyAV)
python3 experiments/r13_e5_seam/verify_invariants.py
# default-OFF guard
cd web_spike/wasm_mv && deno run --allow-run --allow-read parity_shader.mjs
```
