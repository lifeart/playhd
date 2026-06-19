# R2-E2 — Soft-occlusion / temporally-consistent fallback: VERDICT = frontier ESCAPED

**Window A** (`sample.mp4 --start-frame 0 --max-frames 48`), numpy backend (deterministic), 720p tier,
`occ='reactive'`, compact SR. Baseline reproduces R1-E2 window-A exactly: anchors `{0,28}`, non-anchor
hole_frac mean 7.70% / max 32.6%, 7 P-frames >0.20. All schemes are **output-only post-passes** on the
*identical* bicubic-fallback chain (nothing soft fed back as a reference — GOTCHA #16). Artifacts:
`softocc.py`, `refine.py`, `plot_frontier.py`, `probe.py`, `results.json`, `frontier.png`.

**eff-bicubic% (scheme-agnostic generalization of R1's metric):** inside the fallback mask M, realized-
detail ratio `r = ‖out−bic‖/‖sr−bic‖ ∈ [0,1]`; `eff-bic% = 100·mean_nonanchor(Σ_M(1−r)/HW)`. a=0 →
7.70% (=hole_frac, matches R1); hard SR → ~0. Self-consistent: `eff-bic% + detail% = 7.698` for every scheme.

## Frontier table (window A)

| scheme | tOF | eff-bic% | detail% | note |
|---|---|---|---|---|
| **bicubic** (R1 tOF-optimal) | **0.756** | **7.70** | 0 | baseline — soft but steady |
| **HARD-SR all** (R1 lower-fb point) | **1.214** | **3.69** | 4.01 | the R1 tradeoff anchor |
| (a) spatial feather g=1.0 | 1.162 | 4.56 | 3.13 | ~on the R1 line |
| (a) feather g=0.25 | 0.785 | 7.23 | 0.47 | on line, negligible |
| (a′) conf-graded g=0.75 | 1.071 | 5.70 | 2.00 | on/above line — no escape |
| (b1) SR+steady-warp k=0.5 | **2.695** | 2.93 | 4.76 | **GHOST — NO-GO** |
| (b2) screen-space EMA β=0.5 | **2.958** | 1.58 | 6.11 | **GHOST — NO-GO** |
| (b3) HF-only EMA β=0.85 | 0.936 | 5.16 | 2.54 | **ESCAPE** |
| (b3) HF-only EMA β=0.95 | 0.923 | 5.79 | 1.91 | **ESCAPE** |
| **(c) combo g=0.6 β=0.85 fe=31** | **0.771** | **6.35** | 1.34 | **ESCAPE — recommended** |
| (c) combo g=0.5 β=0.8 | 0.784 | 6.60 | 1.10 | escape, conservative |
| (c) combo g=0.75 β=0.9 | 0.827 | 6.01 | 1.69 | escape, deeper |
| (c) combo g=1.0 β=0.85 | 0.936 | 5.16 | 2.54 | escape, aggressive |

## Verdict — YES, a third option escapes the frontier

Recommended **`(c) g=0.6, β=0.85, feather=31`: tOF 0.756→0.771 (+2.0%, ≈ bicubic's tOF) while eff-bicubic
7.70→6.35** — a 1.34-pt (17.5% relative) cut of the soft-fallback weak spot for ~ZERO temporal cost. The
R1 hard switch needed tOF ≈ 0.91 (+20%) for the same eff-bic; this hits it at +2%. Every (b3)/(c) point
lies below the R1 bicubic↔hard line. The escape is **bounded**: EMA can only pre-stabilize HF in regions
that *persist* across frames; freshly-revealed disocclusions must stay bicubic to stay tOF-safe — so
soft+temporal recovers ~1.3 pts free, up to ~2.5 pts cheaply (+24% tOF, still ≪ hard's +60%), but does not
fully close to hard's 3.69%. The frontier is **bent, not erased**.

**Why:**
- **(a)/(a′) spatial feathering & confidence-grading sit ON the frontier** — they smooth only the spatial
  seam; the interior fresh per-frame HF still shimmers (confirms R1's diagnosis).
- **(b1)/(b2) naive temporal reuse GHOST (tOF 2.4–3.7).** Screen-space EMA / warp-blend drags stale
  content across motion → huge spurious flow. b2 has *lower* |ΔF| (38.3 vs 39.5) yet 4× tOF — a textbook
  demonstration that |ΔF| lies and tOF is the honest headline (GOTCHA #13).
- **(b3)/(c) escape via HF-ONLY temporal smoothing** (`T = bicubic_current + EMA(sr − bicubic)`): the
  motion-tracking low-freq stays fresh each frame (tOF-safe); only HF detail is temporally smoothed, and
  HF carries little of the motion energy Farneback locks onto, so it ghosts negligibly. (c) adds confidence-
  graded feathering. Feather sweet spot fe=31 HD px.

## Concrete integration (default-OFF, instant-only)
Structurally the existing output-only `region_gate` post-pass in `prototype/derisk.py:reconstruct`
(runs after both passes → chain consumed the un-blended recon; soft HF never propagates, GOTCHA #16):
```
# per non-anchor frame i (reset ema_HF at I-frames / scene cuts):
ema_HF = beta*ema_HF + (1-beta)*(sr[i] - bic[i])               # beta=0.85, one HD-float buffer of state
a      = gain * feather(R[i]['mask'], k) * conf[i]             # gain=0.6, k=31; conf from reactive residual
R[i]['recon_out'] = (1-a)*R[i]['recon'] + a*(bic[i] + ema_HF)  # OUTPUT copy only, never R[i]['recon']
```
- `conf[i] = clip((react−6)/(26−6),0,1)` on the reactive residual `occlusion_mask_lr` already computes.
- Deployment hook (`server/anchor_sr.py`): **replaces the hard `patch_high_fallback` SR-patch** with the
  feathered HF-EMA blend; backbone chain stays bicubic (no cache escalation → no propagation). Flags
  default-OFF: `INSTANT_SOFTOCC=False`, `SOFTOCC_GAIN=0.6`, `SOFTOCC_BETA=0.85`, `SOFTOCC_FEATHER=31`.
- **Seams to verify:** `mask[i]`, `bic[i]`, `sr[i]`, `conf[i]`, the EMA buffer all display-order-aligned to
  `i`; EMA reset MUST fire on every I-frame/scene-cut (missed reset → cross-cut ghost). Validate torch/MPS
  parity (numpy validated). Gate to high-motion via R1-E2's `mean|MV|>1.0` to bound the per-frame SR cost.

## Executive summary
1. R1's "tOF↔fallback% is a hard wall" is refined: the wall is real for *spatial* tricks, **breakable with
   a *temporal* one**.
2. Naive temporal reuse (b1/b2) is a NO-GO (ghosts; tOF explodes) — and showed |ΔF| can fall while tOF rises.
3. The escape is **HF-only temporal smoothing (b3) + confidence-feather (c)**.
4. Recommended `(c) g=0.6/β=0.85/fe=31`: eff-bicubic 7.70→6.35% at tOF +2.0% — a genuine point OFF the
   frontier (~17.5% of the weak spot removed for ~no temporal cost; hard would charge +20%).
5. The escape is **bounded** (only persistent disocclusions; ~1.3 free, ~2.5 cheap, not a full close).
6. Integration = the existing output-only post-pass pattern, default-OFF, instant-only (GOTCHA #16/#6).
