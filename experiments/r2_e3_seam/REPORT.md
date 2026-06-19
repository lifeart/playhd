# R2-E3: LAYERED-mode seam HALO reduction + hair recovery

## What I did
Reproduced the demo's seam baseline (x4plus-bbox 5.75× mid / 5.02 mean, compact 5.12×, uniform-x4plus
3.37×), then tested all three output-only levers on the talking-head scene (`sample.mp4 --start-frame
5000`, scene[0]=[0,32)). RVM matte run once + cached; measurement GPU-free, reusing the prototype's
`out_layered/cache`. Prototype imported READ-ONLY; new code in `experiments/r2_e3_seam/`.

## Key diagnostic (corrects the handoff framing)
The discontinuity is driven almost entirely by the **soft BG ring** (var-Lap 10.8 vs deep-bg 15.4,
ceiling 16.9), NOT the FG ring (50.8 vs ceiling 57.9). The soft ring is **only 7.5% inpaint** — 92.5% is
*recovered* temporal-median that is soft from **low coverage + matte-edge contamination** near the subject
boundary. So the brief's lever (c) "better inpaint" addresses <8% of the visible ring; the real fix is
restoring the soft near-subject plate band — lever (b).

## Before/after (x4plus-bbox budget, 32-frame mean; halo over frames 8/16/24)
| config | ratio | FGring | BGring | halo(px) | fg_bias | coreS |
|---|---|---|---|---|---|---|
| baseline | 5.02 | 50.8 | 10.8 | 11.7 | -3.20 | 4.08 |
| (a) feather | 4.72 | 46.1 | 10.3 | 12.0 | -2.68 | 4.08 |
| (b) ringRestore s=0.5 | 3.38 | 50.8 | **15.9** | 8.0 | -3.23 | 4.08 |
| (b-alt) softenFG **[REJECTED]** | 1.39 | 14.0 | 10.8 | 11.7 | -3.36 | 4.04 |
| **(a)+(b) RECOMMENDED** | **3.22** | 46.1 | 14.9 | **7.7** | -2.72 | **4.08** |
| uniform-x4plus (ceiling REF) | 3.45 | 57.9 | 16.9 | 0.0 | — | 4.08 |

- **Seam ratio**: (a)+(b) takes x4plus 5.02 → **3.22 ≈ ceiling 3.45**.
- **Halo width**: 11.7 → **7.7 px (−34%)**.
- **Hair**: feather moves the hair band toward the subject (~15% wisp recovery), subject core untouched
  (coreS 4.08→4.08 exact — no smearing). Confirmed in heatmap crops; softenFG visibly destroys subject-edge.

## Verdict per lever
- **(a) alpha-aware feather — HELPS (mild).** Wider feather + faint-wisp lift keyed on alpha-gradient. Core
  untouched. +4 ms/frame at LR.
- **(b) sharpness-matched plate-ring band — THE FIX.** Band-localized unsharp on the BG side, blended to
  land BGring at deep-bg (~15). Ratio → ceiling, halo −34%, core untouched. Re-contrasts existing texture;
  fabricates nothing (the ~8% pure-inpaint holes have ~0 HF so stay soft).
- **(b-alt) soften FG edge — REJECTED** (wins ratio only by smearing the subject).
- **(c) better disocclusion-ring inpaint — NO-GO** (holes 7.5% of ring, always behind subject; alt fills
  softer not sharper). Lever (b) is the effective "better ring treatment."

## Concrete output-only change to `layered_pipeline.py` (default-preserving)
`experiments/r2_e3_seam/seam_composite.py` is the drop-in: two new kwargs on `composite(fg_hd, alpha_hd,
plate_hd, seam_restore=0.0, feather=False)` — **defaults byte-identical to the current composite (verified
`np.array_equal == True`)**. Helpers: `feather_alpha`, `restore_plate_ring`, `prepare_scene_plate`.

**Cost:** naive per-frame seam pass is +158 ms/frame (too dear). Cost-optimized — ring-restore is
plate-only, so `prepare_scene_plate` bakes it ONCE per scene over the subject's swept alpha-union band
(strength~0.8 → ratio 3.47, halo 8.0): **135 ms once (~0.45 ms/frame @300f)** + LR alpha feather (+4 ms) =
**~+5 ms/frame**, negligible vs the ~209 ms/frame layered budget.

## Executive summary
The layered seam halo is a **soft-background problem, not a sharp-subject problem**: the near-subject plate
ring collapses to var-Lap ~10.8 (vs ~15.4 deep-bg / 16.9 ceiling), 92% recovered-median softened by low
coverage + matte-edge contamination, only ~8% inpaint. Fix = composite-time band-localized **plate-ring
restore** (b): x4plus seam ratio **5.02 → 3.22** (= ceiling 3.45), halo **11.7 → 7.7 px**, subject core
exactly unchanged (re-contrasts existing texture, fabricates nothing). **Alpha-aware feather** (a) adds
~15% hair-wisp recovery free. Softening the FG edge REJECTED; better inpaint NO-GO. Recommended **(a)+(b)**
as once-per-scene plate prep + LR feather, **~+5 ms/frame**. Drop-in `seam_composite.py` (defaults
byte-identical, verified).
