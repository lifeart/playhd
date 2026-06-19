# E3 — Two visual-quality wins as OUTPUT-ONLY passes

**Files written** (all NEW, `prototype/`+`server/` imported READ-ONLY, no shared file touched):
`common.py, motion_grain.py, graphic_detect.py, v2_grain.py, v3_pin.py` and `artifacts/{v2_static_ampdiff.png, v2_gate.png, v3_region.png, v3_edge_ampdiff.png, v3_probe_titlewindow.png, v3_probe_headtail.png, metrics.csv}`.

**Method.** Recon = `derisk.reconstruct` (numpy = deterministic/contention-robust), compact `realesr-general-x4v3` SR, scale 4, occ=full. Honest metrics: **|ΔF|** = mean abs frame-to-frame **luma** diff over a mask (flicker); **tOF** = `derisk.tof` vs decoded LR (stability). Grain independence measured on the **RAW additive field** (gotcha-safe). MPS freed between configs.

## V2 — motion-modulated grain on static regions → **SHIP**

Talking-head window C (start 5000, 48f), HD 2560×1280, grain=med. Gate `a_lr` = `region_quality.window_static_weight(meanmag, lo=0.2, hi=1.0, feather=61)` — the SAME gate quality mode already builds in `derisk._build_region_gate` (1=static). V2 grain = `apply_grain`'s exact recipe but the per-pixel unit field is `renorm(a·frozen + (1−a)·fresh)`: frozen (fixed seed) on static → identical filmic texture every frame → ~0 temporal flicker; fresh (seed=index) on motion → independent.

| sequence | **bg/STATIC \|ΔF\|** | DYN \|ΔF\| | ALL \|ΔF\| | tOF(LR) |
|---|---|---|---|---|
| recon (no grain) — floor | 2.307 | 9.067 | 4.345 | 0.209 |
| **full grain (current)** | **5.649** | 11.132 | 7.258 | 0.334 |
| **motion grain frozen (V2)** | **2.313** | 11.061 | 5.009 | 0.259 |
| motion grain reduced (floor 0.25) | 3.129 | 10.966 | 5.480 | 0.253 |

- Full grain raises static |ΔF| +145% over the stabilised floor; **V2 frozen 2.313 ≈ 2.307 floor → removes ~100% of grain-induced static flicker.**
- DYNAMIC |ΔF| unchanged (11.061 vs 11.132) → fresh filmic grain kept on motion. tOF: V2 avoids 60% of the grain tOF rise.

**Grain independence (RAW field):** full grain STATIC −0.0000 / DYNAMIC −0.0007. V2 frozen: STATIC **0.9978** (frozen → no flicker, by design), **gate-moving (a<0.1) corr = 0.0010** — exactly the ~0.001 filmic target. (DYNAMIC-mask 0.0383 only because that mask's mean a=0.077, slightly static.)

Artifacts confirm visually: `v2_static_ampdiff.png` (FULL grain speckles the whole static bg; V2 frozen back to the clean recon baseline); `v2_gate.png` (red=static bg frozen, blue=moving subject fresh — correctly isolated).

## V3 — graphic/text-edge pinning → **detector SHIP, pinning NO-GO**

Detector `graphic_detect.detect_graphic_mask`: **bimodality** (local min(frac near-white, frac near-dark) — text packs pure-bright glyphs on a pure-dark field; natural content never does) **AND low codec motion**. The animated reveal (f1–15) is content change, not shimmer; benefit measured on the **static run f18–28** (f28 = scene cut).

**(A) Detector + false-positive guard — excellent:** title f18–27 = 33% coverage (correct), f28–31 = 0% (correct), window-C **face f0–31 = 0.00% max (no mis-fire)**, window-C tail f32–47 = 19–33% (the same card actually enters → correct TRUE positive). Bimodal score ~0.49 on card vs ≤0.02 on face. `v3_region.png` shows it precisely outlining the glyphs.

**(B) Pinning — counterproductive here.** Edge |ΔF| on the card's hard edges, static run:

| sequence | edge \|ΔF\| |
|---|---|
| **propagated recon (engine)** | **0.816** ← steadiest |
| per-frame SR (floor) | 1.009 |
| freeze → 1 anchor SR | 1.200 |
| **pin → per-frame SR (task primary)** | **2.061** ← 2.5× worse |
| LR source (cubic, ref) | 0.428 |

The card is **ZERO-MV (skip-coded, p50=p90=0.00px)** → the MV warp is an identity copy that reuses ONE anchor's SR across the GOP, so the engine's propagation (0.816) is **already steadier than per-frame SR (1.009)**, which re-hallucinates edges each frame. Pinning injects that instability (2.061); freezing adds a recon/frozen seam (1.200). `v3_edge_ampdiff.png`: recon clean, pin streaks the edges. **The "shimmer under warp" premise fails because NEMO anchor-reuse already stabilises a static graphic.** (Inferred, untested: pinning could help a graphic carrying non-zero MVs — a lower-third over moving video — none exists in sample.mp4.)

**Verdict:** ship the detector as a validated guard; HOLD pinning (a measured regression on the only card). V2's motion gate already freezes grain on low-motion graphics for free (a≈1).

## Proposed output-only passes (for SEAM verification)

**P1 (V2) — recommended.** Add to `prototype/grain.py` (additive; existing `apply_grain` untouched → byte-identical regression) the validated `apply_grain_motion(rgb_uint8, frame_idx, static_w_hd, strength="med", template=None, frozen_idx=0, mode="frozen", return_grain=False)` (= `motion_grain.py`). Wire in `server/pipeline_api.py` **quality path** (the gate already exists there): precompute once per chunk `a_hd = cv2.resize(region_gate["a_lr"], (w_hd,h_hd))`, then swap the grain calls at ~lines 916–919 and the twin in `_quality_subchunk` ~631–634 to `_grain.apply_grain_motion(recon, done, a_hd, cfg["grain"])`; add `grain_motion: True` to `MODE_CONFIG["quality"]`. Seam checks: `a_lr` is LR `(h_lr,w_lr)`, `a_hd` matches recon `(h_hd,w_hd)`, `done` is the global frame index (independence seed across chunks — already true), `R[]` untouched. **Instant GPU path** (follow-up, larger): build the free MV-only gate (`region_masks`+`window_static_weight`) per chunk and extend `fast_grain.GpuGrain.apply(..., static_w=None, frozen_idx=0)` to blend frozen+fresh on-device. Ship quality first.

**P2 (V3).** Ship `graphic_detect.detect_graphic_mask(...) -> (region_hd_bool, edge_hd_bool)` as a default-OFF validated guard/diagnostic. **Do not wire the pin** (measured regression). Revisit only with a non-zero-MV graphic test clip.

**Honest notes.** All tabled numbers are measured (deterministic). Inferred/flagged: V2 generalising beyond window C (content-independent mechanism, one window measured); V3 pinning on non-zero-MV graphics (none available); GPU-path V2 parity (by construction, mirrors `fast_grain`, not yet measured). Compact SR used; heavy x4plus would only widen V3's propagation-beats-per-frame gap, not create shimmer.
