# E2 — Content-adaptive fallback/anchoring for the high-motion instant weak spot

**Window A** = `--start-frame 0` (high-motion stress) · **Window C** = `--start-frame 5000` (talking-head) · `--max-frames 48` · `--backend torch` (MPS) · `occ=reactive` (what `MODE_CONFIG['instant']` ships) · compact net `realesr-general-x4v3` · 720p tier (`INSTANT_SCALE=2`) except policy (c).

Artifacts (READ-ONLY imports of prototype/server): `exp2.py`, `results.json`, `run.log` in this dir.

## Method (warp-only)
SR net runs **once per (window, scale)** to cache compact-SR for every frame; each policy is a per-frame choice between bicubic (the weak spot) and cached compact-SR, re-run through `derisk.reconstruct` (zero SR re-runs in the sweep). Escalating a frame's cache entry to compact-SR exactly reproduces the deployed path: backbone → `build_anchor_cache` in-cache upgrade; B-leaf → identical to `patch_high_fallback` (leaf reads `perframe` at its `none` pixels). `hole_frac` is anchor- and content-invariant → one baseline pass gives every frame's fallback for free.

**Decision metrics (only these):** **tOF** = Farneback-flow EPE of propagated recon (downscaled to LR) vs decoded LR (motion truth; lower=steadier; deterministic). **eff-bicubic%** = mean over non-anchor frames of pixels *still served by bicubic* (= `hole_frac` if not escalated, else 0) — the honest weak-spot size. **SR-calls/frame** = cost. `|ΔF|` reported as a flicker cross-check. Not used: LR-consistency / PSNR-vs-perframe.

## Window characterisation (measured)
| window | anchors | non-anchor hole_frac mean / max | #>0.08 / >0.20 / >0.50 | mean·\|MV\| (LR px/f) |
|---|---|---|---|---|
| **A** high-motion | {0,28} | **7.70% / 32.6%** | 15 / **7** / 0 | 2.90 (max 27.3) |
| **C** talking-head | {2,32} | **2.32% / 19.0%** | 3 / **0** / 0 | 0.54 (max 2.22) |

Weak spot is real and **occlusion-dominated** on A: 7 P-frames carry 22–33% large-motion disocclusion served by bicubic at the `thresh=0.50` baseline. C never exceeds 20% and has ~5× lower motion.

## KEY FINDING — tOF and fallback% are in **tension** on high motion
Bicubic fallback is temporally **smooth**; compact-SR fallback injects fresh per-frame HF into disocclusion regions, which *shimmers* (independently confirmed by `region_quality`'s thesis that fresh-HF re-injection is what flickers). So **reducing eff-bicubic% necessarily raises tOF — bicubic fallback is already the tOF-optimal operating point.** There is **no** policy that improves both tOF and fallback% on A. The real choice is *steady-but-soft (baseline)* vs *sharper-but-shimmerier (escalate)*, at a compute cost, gated to high-motion content. This reframes the task's "improve both" criterion as not achievable with these levers — stated honestly rather than papered over.

## Policy (a) — fallback threshold

**A (720p), global threshold:**
| thresh | SR/f | #esc | tOF | eff-bic% | raw-fb% |
|---|---|---|---|---|---|
| **0.50 (deployed)** | 0.042 | 0 | **0.847** | **7.71** | 7.71 |
| 0.30 | 0.062 | 1 | 0.965 | 7.00 | 7.71 |
| **0.20 (knee)** | 0.188 | 7 | **1.214** | **3.65** | 7.71 |
| 0.12 | 0.312 | 13 | 1.300 | 1.39 | 7.71 |
| 0.08 | 0.354 | 15 | 1.410 | 0.96 | 7.71 |

Monotone tradeoff. `thresh=0.20` is the knee — escalates only the 7 worst-occlusion P-frames (all mean·|MV| ≥ 2.65, genuinely high-motion), **halving** eff-bicubic% (7.71→3.65) for +43% tOF and 4.5× SR-calls.

**C (720p), same sweep:**
| thresh | SR/f | #esc | tOF | eff-bic% |
|---|---|---|---|---|
| 0.50 / 0.30 / **0.20** | 0.042 | **0** | **0.196** | **2.32** |
| 0.12 | 0.083 | 2 | 0.218 | 1.61 |
| 0.08 | 0.104 | 3 | 0.219 | 1.43 |

**At any thresh ≥ 0.20, C escalates nothing → identical to baseline (zero cost).** C is self-gating: zero frames over 20% fallback. (Below 0.12 it starts paying 2–3 frames.)

**Motion-keyed variant** (escalate iff `hole_frac>thresh` AND mean·|MV|>`mgate`): at `thresh=0.20` the gate is a no-op (all 7 A-frames have mean·|MV| 2.65–16.0 ≫ gate; C has 0). It only bites at `thresh=0.12`, trimming C's accidental escalations (2→1, eff-bic 1.61→2.02). So the gate is a cheap **safety guard** for high-occlusion/low-motion frames, not a lever on this footage.

## Policy (b) — adaptive re-anchoring — **NO-GO** (wrong lever)
**A (720p):** baseline tOF 0.847 / eff-bic 7.71 → budget 3.0: tOF 0.874 / 7.43 (promote [21]) → budget 0.5: tOF **1.296** / eff-bic **5.76** (7 anchors). **C:** budget never triggers at any value (`[]` promoted) → identical to baseline. Re-anchoring **raises tOF** (each fresh anchor is a temporal pop) and barely moves eff-bicubic% (5.76% even at 7 anchors) because it refreshes *propagated* pixels (drift), while the weak spot is **occlusion**. Matches the gotcha: "the lever is fresh per-frame fallback SR, not propagation." **NO-GO.**

## Policy (c) — QHD escalation (window A, x4 vs x2) — **NO-GO** for the goal
| | tOF | eff-bic% | recon (best-of-3) | ratio |
|---|---|---|---|---|
| 720p (x2) | 0.847 | 7.71 | **31.4 ms/f** | 1.0× |
| QHD (x4) | **0.808** | 7.71 | **165.5 ms/f** | **5.27×** |

QHD gives only −4.6% tOF and **identical eff-bicubic%** (fallback is still bicubic, just higher-res → does NOT fix the weak spot), at **5.27×** recon cost. 720p warp floor (31 ms) is already most of the ~42 ms 24-fps budget; QHD (166 ms) blows it ~4×. **NO-GO.** (SR-latency note: the x2 cache read 236 ms/f but that's the process's *first* SR pass — cold-start + sibling-MPS contention; steady-state compact SR ~130 ms/f, corroborated by C@x2=129.5 and A@x4=131.1. Timing as ratios/best-of-N; tOF/fallback% deterministic.) `|ΔF|` cross-check (A): baseline recon 13.96 vs true-LR 13.90 (faithful); thresh-0.08 recon 14.03 — escalation nudges change above true-LR, agreeing with the tOF rise.

## RECOMMENDATION
**Default: keep the bicubic baseline (`INSTANT_FALLBACK_THRESH=0.50`)** — it is tOF-optimal; "soft but steady" disocclusion is a defensible fast-tier tradeoff; no policy improves both metrics.

**Ship as default-OFF, instant-only opt-in: motion-keyed fallback threshold @ 0.20** (the only policy that addresses the weak spot, content-adaptive by construction):
```
thresh(frame) = 0.20  if mean_MV_magnitude(frame) > 1.0 LR px/frame   (high motion)
                0.50  otherwise                                       (steady → baseline)
escalate frame to compact-SR fallback  iff  hole_frac(frame) > thresh(frame)
```
Effect: **A** eff-bicubic 7.71%→**3.65%** (weak spot halved), tOF 0.847→1.214 (honest cost), SR/f 0.042→0.188 (~+19 ms/f amortized → the high-motion *chunk* drops ~24→~15–18 fps, bounded to those frames). **C: exactly zero change** (tOF 0.196, eff-bic 2.32, SR/f 0.042 — requirement (ii) met precisely). Pick 0.20 (not 0.12) so it's self-gating on talking-head; the mean·|MV|>1.0 gate is a free safety guard.

## Integration hook (default-OFF, instant-only) — PROPOSED, not applied
**1) `server/anchor_sr.py`** — add optional `thresh_fn=None` to `build_anchor_cache` and `patch_high_fallback`. When `None` (default) behavior is exactly today's. One-line override at each existing comparison:
```python
# build_anchor_cache(..., thresh_fn=None):  at the backbone scan
thr = thresh_fn(i) if thresh_fn is not None else fallback_thresh
if bb_fracs[i] > thr: sr_set.add(i); bb_masks[i] = m
# patch_high_fallback(..., thresh_fn=None):  at `if hf > fallback_thresh:`
thr = thresh_fn(i) if thresh_fn is not None else fallback_thresh
if hf > thr: ...
```
**2) `server/pipeline_api.py`** — new default-OFF consts `INSTANT_MOTION_KEYED_FALLBACK=False`, `INSTANT_FALLBACK_THRESH_HI=0.20`, `INSTANT_MOTION_GATE=1.0`; build the free per-chunk `thresh_fn` (reuses `derisk.build_lr_flow`, already decoded) and pass it to both calls in the instant fast branch only:
```python
def _motion_keyed_thresh_fn(chunk, base_thresh):
    if not INSTANT_MOTION_KEYED_FALLBACK: return None
    import numpy as _np
    h_lr, w_lr = chunk[0][1].shape[:2]
    def thr(i):
        pt, _, mvs = chunk[i]
        if pt == "I" or mvs is None or len(mvs) == 0: return base_thresh
        fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want="all")
        mag = _np.sqrt(fx*fx + fy*fy)
        m = float(_np.nanmean(mag)) if _np.isfinite(mag).any() else 0.0
        return INSTANT_FALLBACK_THRESH_HI if m > INSTANT_MOTION_GATE else base_thresh
    return thr
# tfn = _motion_keyed_thresh_fn(chunk, INSTANT_FALLBACK_THRESH); pass thresh_fn=tfn to both calls.
```
**Seam contract to verify:** `thresh_fn(i)` is called with the same display-order, 0-based index `i` that indexes `chunk`/`frames`/`R` everywhere else; scalar `fallback_thresh` remains the fallback for `thresh_fn=None` and I/no-MV frames; quality/layered never build a `thresh_fn` (instant-only). Flag OFF → new args `None` → byte-identical regression. Minimal-equivalent on this footage: setting `INSTANT_FALLBACK_THRESH=0.20` alone reproduces the A-vs-C result (C has zero frames >0.20); the gate generalizes to other content.

## Honest measured-vs-inferred
**Measured:** every tOF / eff-bicubic% / SR-calls/frame / recon-ms table; the tOF↔fallback% tension; (b)/(c) NO-GO; C zero-cost at thresh≥0.20. **Inferred:** the ~15–18 fps figure for the escalated chunk (amortized SR estimate, not an end-to-end pipeline timing); the perceptual weight of a 0.847→1.214 tOF rise (numbers given; perceptual call deferred). Timing contention-affected (shared MPS) → ratios/best-of-N; tOF/fallback% deterministic.
