# R8-E2 — HIGH-MOTION instant quality without breaking real-time

**Goal.** Improve the instant tier's standing weak spot — HIGH-motion occlusion fallback served by
BICUBIC at the 720p safeguard (`INSTANT_FALLBACK_THRESH=0.50`) — at NEAR-ZERO real-time cost. The
two documented levers (`INSTANT_SOFTOCC`, `INSTANT_MOTION_KEYED_FALLBACK`) are tradeoffs, not wins;
not re-proposed. Tested three NEW hypotheses. **READ-ONLY** on `prototype/`+`server/`; integration =
a default-OFF unified diff under this dir, byte-identical when off (verified).

Warp-only methodology of exp2 (R1-E2): the SR net runs ONCE/window to cache compact-SR; each policy
is a per-frame cache choice (read only at fallback pixels), reconstructed warp-only.
Metrics: **tOF** (Farneback EPE of recon↓LR vs decoded LR — bicubic is the tOF-optimal baseline),
**eff-fallback%**, **ms/frame** (kill criterion), plus a **full-reference** proxy (PSNR/SSIM/LPIPS
vs a synthetic /2 ground truth). Backend torch/MPS, occ=reactive, 720p (scale 2); shared MPS →
timing best-of-N / ratios.

## Windows (located by MV-magnitude scan; `probe_windows.py` over 9000 frames)
| window | start | mean·|MV| (LR px/f) | role |
|---|---|---|---|
| **A** | 0 | 2.75 (max 27.3) | exp2's "window A" — moderate high-motion, **instant-relevant** |
| **H2** | 2352 | 10.83 (max 28.6) | fast action, still instant-relevant |
| **H1** | 7392 | **26.37** (max 79.3) | EXTREME motion (strongest in scan) |
| **C** | 4488 | 0.43 | talking-head control |

Critically, **H1 is NOT an instant scenario**: 26 of 45 non-anchor frames exceed the 0.50 safeguard
→ the baseline instant path *already* fires 26 full compact-SR calls (not real-time), and Auto
(`recommend_mode`, median |MV|≫`AUTO_MOTION_HI`) routes it to **quality**. The real "instant
high-motion weak spot" is **window A** (0 frames >0.50 → pure anchor-only → real-time holds; 7
frames >0.20 served by bicubic — exactly exp2's weak spot) and **H2**.

---

## Hypothesis 1 (THE CRUX) — clustered-tile SR under HIGH motion — **NO-GO**
The prior tile-SR NO-GO was measured on general footage (bbox ~97%, 32×16 grid ~46% tiles). I
re-measured the EXACT reactive occlusion-fallback mask's spatial clustering on high-motion frames
(`clustering.py`; aggregated over the >8%-fallback frames, n=15–41 per window — not one frame).

| window | mean hole | #CC | largestCC | **bbox cov** | **g8×4 touched** | **g32×16 touched** |
|---|---|---|---|---|---|---|
| A (mod) | 7.7% | 116 | 63% | 77% | 62.7% | **41.1%** |
| H2 (fast) | 13.7% | **301** | 53% | 100% | 99.2% | **79.7%** |
| H1 (extreme) | **47.5%** | 87 | 82% | 98% | 95.2% | **77.2%** |
| C (talk) | 0.8% | — | — | — (0 hi-frames) | — | — |

**The fallback is NOT clustered on high motion — it gets MORE scattered/larger as motion rises.**
The "fast-moving object → compact disocclusion band" premise fails on real footage: camera+scene
motion disoccludes along MANY moving edges (87–301 connected components), and at extreme motion the
fallback covers ~half the WHOLE frame (47.5%). A coarse grid must SR 77–99% of tiles on H1/H2 (vs
41% on the milder A — which merely **reproduces** the prior general-footage NO-GO). Tile-SR is *more*
hopeless on high motion, not less. **Refuted, with measurement.**

## Hypothesis 2 — content-adaptive anchor budget — **NO-GO**
The weak spot is OCCLUSION (fresh per-frame fallback), not propagation drift. exp2 policy (b)
already measured re-anchoring NO-GO (raises tOF — each fresh anchor is a temporal pop — and barely
moves eff-bicubic, because anchors refresh *propagated* pixels). H1/H2 confirm the regime: on a
47.5%-occlusion frame the fallback *is* the frame, so no anchor budget can fill it. Not re-measured
beyond exp2 + the clustering occlusion split. **NO-GO (lever mismatch).**

## Hypothesis 3 — cheaper-than-SR temporal fill (UNSHARP) — **QUALIFIED GO (default-OFF opt-in)**
The fallback band can only be filled by upscaling the CURRENT LR (warp is invalid there). Spectrum:
bicubic (deployed, softest, tOF-optimal) → unsharp → compact-SR (sharpest, shimmers). A light
**unsharp** of the bicubic fill (one gaussian + lerp, NO net) was measured against a true full
reference (`fill_quality.py`: codec frame = HD truth, INTER_AREA /2 = LR) and through the warp
(`composite.py`, tOF + band-dF on real codec LR).

**Full-reference fill quality (isolated), N=32:**
| window | fill | PSNR | SSIM | LPIPS | dF (truth) |
|---|---|---|---|---|---|
| A | bicubic | 31.05 | 0.9785 | 0.0824 | 20.27 (20.34) |
| A | **unsharp** | **32.56** | **0.9849** | **0.0546** | 20.32 |
| A | compactSR | 34.31 | 0.9876 | 0.0179 | 20.45 |
| H2 | bicubic | 30.00 | 0.9554 | 0.1247 | 10.02 (10.27) |
| H2 | **unsharp** | **30.93** | **0.9589** | **0.0908** | 10.41 |
| H2 | compactSR | **27.69** | 0.9350 | 0.0878 | 11.14 |
| H1 | bicubic | 33.90 | 0.9821 | 0.0708 | 88.53 (88.70) |
| H1 | **unsharp** | **34.53** | **0.9829** | **0.0474** | 88.72 |
| H1 | compactSR | **29.53** | 0.9695 | 0.0597 | 90.84 |

**Composite through the warp (real codec LR, N=48):**
| window | fill | tOF | Δ vs bicubic | band-dF | eff-fb% | recon ms |
|---|---|---|---|---|---|---|
| A | bicubic | 0.8465 | — | 39.70 | 7.71 | ~29 |
| A | **unsharp(0.5)** | 0.8596 | **+1.5%** | 39.92 (+0.6%) | 7.71 | ~26 |
| A | compactSR(all) | 1.3432 | **+58.7%** | 40.47 | 0.00 | ~24 |
| H2 | bicubic | 0.2473 | — | 26.98 | 13.72 | ~30 |
| H2 | **unsharp(0.5)** | 0.2639 | **+6.7%** | 27.35 (+1.4%) | 13.72 | ~29 |
| H2 | compactSR(all) | 0.4136 | **+67.2%** | 28.83 | 0.00 | ~28 |

**Cost (`cost.py`, best-of-20, MPS):** on-device unsharp adds **+1.19 ms/non-anchor** over today's
bicubic `F.interpolate` (CPU cv2: +1.37 ms) → **~+1.1 ms/frame amortized**, **+2.7%** of the ~41 ms
instant budget. **Real-time held — kill criterion PASSED.** (compact-SR escalation costs a full
~130 ms SR per escalated frame — exp2's +19 ms/f amortized.)

**Amount knee (`sweep.py`):** full-ref improves monotonically with amount; band-dF stays nearly flat
(A +0.5%, H2 +2.8% at a=1.0). **a=0.5** captures ~60% of the max full-ref gain (A +1.26 dB,
LPIPS −0.025; H2 +0.94 dB, −0.034) at small tOF cost; a=0.25 is the conservative setting.

### Verdict on H3 (honest)
- It is **NOT** a tOF win (bicubic is tOF-optimal; unsharp is +1.5..6.7% tOF) — so it does **not**
  clear the strict "lower tOF" sub-bar. It IS a full-reference (PSNR+SSIM+LPIPS) win at **near-zero
  real-time cost AND near-zero tOF cost** → it MEETS the stated GO criterion ("improve quality via a
  full-ref proxy at near-zero real-time cost"). Default stays bicubic; ship as opt-in.
- **It strictly supersedes the existing compact-SR escalation lever** (`INSTANT_MOTION_KEYED_FALLBACK`)
  for fallback sharpening: compact-SR HURTS full-ref PSNR on genuine high motion (H2 27.7, H1 29.5 vs
  bicubic 30.0/33.9 — hallucinated detail) and raises tOF ~60–67% at ~130 ms/frame; unsharp is
  fidelity-POSITIVE on every window at +1.2 ms. **This refines exp2's recommendation:** escalating
  the fallback to compact-SR is fidelity-negative on real high motion — prefer unsharp.

---

## Integration (PROPOSED — `r8_e2_unsharp.diff`, default-OFF, byte-identical when off)
One new flag, three seam points (the instant fast path + progressive, both via the same call):

- **`server/anchor_sr.py`** — `build_anchor_cache(..., unsharp=0.0, unsharp_sigma=1.0)`. `base_hd(i)`
  applies `_gpu_unsharp`/`_cpu_unsharp` to the bicubic fill iff `unsharp > 0.0`; else returns EXACTLY
  today's value. Read ONLY at fallback pixels; anchors (full SR) and the occlusion mask are untouched.
- **`server/pipeline_api.py`** — `INSTANT_FALLBACK_UNSHARP = 0.0` (knee ~0.5),
  `INSTANT_FALLBACK_UNSHARP_SIGMA = 1.0`; passed to `build_anchor_cache` in the instant fast branch.
- **`server/progressive.py`** — same two kwargs (the other instant-path `build_anchor_cache` caller).

**Seam contract verified (`verify.py`, all PASS):** (1) OFF → cache **byte-identical** (`torch.equal`)
to the shipped call on BOTH `gpu_cache` paths; (2) ON → anchors unchanged, every non-anchor entry
equals `_unsharp(bicubic)`; (3) `hole_frac`/masks identical ON vs OFF (no threshold/mask change →
eff-fallback% unchanged); (4) flags exposed, default 0.0. Patch applies cleanly (`patch -p1`).

## Threats to validity / what I could NOT verify
- **Full-ref proxy uses a synthetic INTER_AREA /2 LR** (clean). The real pipeline upscales
  codec-decoded LR (artifacts/noise); unsharp would also sharpen mosquito noise/grain — UNMEASURED on
  real codec artifacts (the composite tOF/band-dF ARE on real codec LR and stay benign, but full-ref
  on codec content has no ground truth). Recommend a perceptual A/B before raising the default.
- **tOF carries Farneback variance** (sweep.py is non-monotonic in amount) — the +1.5/6.7% composite
  figures (N=48) are the cleaner read; band-dF corroborates a small, monotonic flicker rise.
- No end-to-end `process_clip` render was run with the flag ON (READ-ONLY constraint; verified at the
  `build_anchor_cache` seam + reconstruct instead). Grain/encode are downstream of the cache and
  unaffected by a fill change.
- H1 (extreme) is quality-routed; the GO is scoped to the instant-relevant regime (A, H2).

## Artifacts (this dir)
`probe_windows.py`→`windows.json`; `clustering.py`→`clustering.json` (the crux); `fill_quality.py`
→`fill_quality.json`; `composite.py`→`composite.json`; `sweep.py`→`sweep.json`; `cost.py`;
`verify.py`; `patch_src/` (patched copies) + **`r8_e2_unsharp.diff`** (the integration patch).
