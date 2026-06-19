# R4-E1: Fix the silent LAYERED missed-cut plate corruption — COMPLETE (PASS)

## The bug
LAYERED builds ONE temporal-median plate per **detected** scene. On a similar-luma cut (c7 frame 28,
|Δluma|=26.4 < CUT_THRESH 60 / REL_FLOOR 40), `scene_detect` misses it → one plate spans both scenes →
scene-A plate composited over scene-B. Post-cut LR-consistency collapses **33.8 → 14.72 dB**; **tOF is
blind** (wrong plate is temporally stable). Honest metric = fidelity-vs-LR, not tOF.

## Both fixes (defense-in-depth)
**(a) PRIMARY — per-frame plate-validity guard** (`layered_api.py`, wired in `pipeline_api.py` PASS B).
Per frame, downscale the HD plate to LR (INTER_AREA), PSNR vs decoded LR over the **background region
only** (matte α<0.5, eroded). A right-scene plate scores ~30–40 dB; a wrong-scene plate craters ~12–16 dB.
Trips on an **absolute floor (24 dB)** OR a **relative cliff (>8 dB below the plate's per-scene EMA
baseline)** (the cliff protects a uniformly-lower-but-correct plate — textured/low-light/grainy bg). On a
trip → fall back to the **full-frame compact SR already computed inside the composite** → **zero extra
SR**. Robust to ANY missed cut, detector-independent.
**(b) ROOT-CAUSE — chroma-dominant cut test** (`scene_detect.py`). Luma path **byte-identical**; ADD a
test firing only when chroma moves more than luma (`dChroma > 1.1·dLuma`) — the missed-cut signature.
Separates real similar-luma cuts (c7 dC 34>dL 26; c5 dC 109>dL 47) from sample.mp4's legit non-cuts (all
luma-dominant brightness flashes). Self-calibrating chroma EMA baseline.

## Verification (through the REAL pipeline)
- **c7 corruption GONE:** post-cut LR-consistency 14.72 → **42.47 dB**; guard trips exactly the 12
  post-cut frames.
- **No regression** (static scene `sample.mp4` [5093,5137)): guard trips **0/44**, output LRC identical
  (32.32 mean), plate path runs; ~9 dB margin to the 24 dB floor.
- **Cut detector:** `sample.mp4` [0,900) **1.00/1.00**, byte-identical cut set (zero new FPs); c7 ✓ + c5 ✓
  now caught (chroma-fired); zero new cuts across 5060 real frames. End-to-end on a two-static-scene clip
  (cut dLuma 1.6 / dChroma 103.8, missed by shipped): chroma splits into 2 plates → scene-A LRC 10.09 →
  27.80, scene-B 24.26 → 26.22.
- **Key interaction:** `segment_scenes` merges a too-short trailing scene back (c7's post-cut is 12f <
  MIN_SCENE_LEN 24), so (b) alone can be undone downstream → **(a) is the essential primary fix**; they're
  complementary.

## Integration (3 diffs, apply clean `patch -p1`, all py_compile OK)
`{scene_detect,layered_api,pipeline_api}.diff`. `layered_api` adds guard consts + `plate_bg_psnr()`,
`plate_is_bad()`, `composite_frame_guarded()` (original `composite_frame` untouched). `pipeline_api` PASS B
calls `composite_frame_guarded` + threads a per-scene `plate_base` EMA (reset on scene change).
`scene_detect` adds chroma consts + `chroma_uv()`/`chroma_diff()` + `_chroma_cut()` + a chroma baseline
(luma path byte-identical; `chroma_dom=0` = legacy).

## Executive summary
Fixed: (a) a per-frame plate-validity guard (bg-region LR-consistency → fall back to the already-computed
compact SR) + (b) a chroma-dominant `scene_detect` term. Kills the corruption (c7 post-cut LRC 14.72 →
42.47), zero regression (static scene 0/44 trips; sample.mp4 cut detection unchanged 1.00/1.00), cheap (no
extra SR on tripped frames; one color-convert/frame in the detector). (a) catches ANY missed cut and
survives the short-trailing-scene merge that can undo (b); (b) fixes the root cause + helps instant/quality
anchoring. Ready to land all three; the guard is the must-have.
