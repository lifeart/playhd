# R2-E1: Permissive matte to replace RVM in LAYERED mode

(Two agent runs reached the same GO; this is the more detailed one. Window: `sample.mp4
--start-frame 5000`, scene[0]=frames [0,32), LR 640×320, N=32. MPS shared with siblings → latency as
ratio vs RVM, baseline 18.9 ms/frame. RVM = pseudo-GT. Artifacts in `experiments/r2_e1_matte/`.)

## Comparison table (RVM = pseudo-GT; others BSD-3 torchvision person-seg, commercial-OK)

| candidate | MAD↓ | IoU↑ | FG% | edge\|ΔF\|↓ | a\|ΔF\|↓ | band\|ΔF\|↓ | lat ms | ×RVM | plate cov | hole%↓ | sharp | %RVM | plate-MAD | bleed↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **RVM** (NON-COMM) | 0.000 | 1.00 | 26.8 | 0.0201 | 0.0066 | 0.0446 | 18.9 | 1.00 | 75.2 | 24.78 | 36 | 100 | 0.00 | 0.00 |
| **LRASPP-mv3** | 0.065 | 0.90 | 26.9 | 0.0071 | 0.0104 | 0.0264 | 12.0 | **0.64** | 76.4 | 23.63 | 39 | 109 | 5.08 | 14.91 |
| **LRASPP-mv3+EMA** | 0.067 | 0.90 | 26.9 | 0.0045 | **0.0066** | 0.0173 | 12.1 | **0.64** | 75.9 | 24.11 | 38 | 106 | 5.16 | 15.15 |
| **DeepLab-mv3** | 0.068 | 0.90 | 29.5 | 0.0060 | 0.0105 | 0.0251 | 16.1 | 0.85 | 74.1 | 25.93 | 34 | 95 | **2.20** | **6.41** |
| **DeepLab-r50** | 0.052 | 0.84 | 23.6 | 0.0083 | 0.0113 | 0.0366 | 80.0 | 4.23 | 79.0 | 20.98 | 43 | 119 | 6.00 | 17.64 |

`edge|ΔF|`=alpha-edge crawl, `a|ΔF|`=whole-frame alpha temporal diff, `band|ΔF|`=soft-edge-band diff;
`bleed`=plate code-value diff vs RVM-plate inside RVM union-FG (subject leak; lower=cleaner);
`sharp`=var-of-Laplacian.

### How to read it
- **Matte quality holds:** every candidate agrees with RVM at IoU 0.84–0.90, MAD 0.05–0.07, same FG%.
- **Temporal stability / GOTCHA #17:** stateless seg jitters more than recurrent RVM at the whole-frame
  level (`a|ΔF|` 0.0104 vs 0.0066). A cheap **alpha-EMA (a=0.5) recovers RVM parity exactly** (0.0066)
  at ~0 cost. (Raw `edge|ΔF|` lower for seg is a confound — smoother alpha → smaller edge gradient; lead
  with `a|ΔF|`/`band|ΔF|`.)
- **LAYERED SURVIVES (critical):** plates from the new gates match RVM on coverage (74–79% vs 75.2%),
  hole% (21–26% vs 24.8%), sharpness (95–119%). **DeepLab-mv3 gives the cleanest plate** (bleed 6.41,
  plate-MAD 2.20) — it slightly over-segments the person so dilation keeps subject pixels out of the bg
  median. (DeepLab-r50's higher "sharpness" 119% is partly subject-ghost contamination — worst bleed.)
- **Latency:** lightweight nets are FASTER than RVM (LRASPP 0.64×, DeepLab-mv3 0.85×); r50 4.23× too slow.

## GO / NO-GO per candidate
- **DeepLabV3-MobileNetV3-Large (BSD-3) — GO (recommended runnable pick).** Cleanest plate, hole%≈RVM,
  sharpness 95%, 0.85× latency. Add EMA for edge stability.
- **LRASPP-MobileNetV3 + EMA (BSD-3) — GO (fastest).** 0.64× RVM latency, RVM-parity temporal stability.
- **LRASPP-mv3 no EMA — CONDITIONAL** (ship only with EMA).
- **DeepLab-r50 (BSD-3) — NO-GO** (4.23× latency, worst bleed).
- **MediaPipe Selfie Segmentation (Apache-2.0) — GO as production target, NOT installed here.** Needs
  protobuf<5; this env has protobuf 6.33 → would risk transformers/onnxruntime, so assessed from the
  model card with DeepLab/LRASPP as the same-license-tier runnable proxy. Ship it in an **isolated
  venv/process**.
- **Also-viable permissive (license-verified, not benchmarked):** PP-HumanSeg/PP-MattingV2 (PaddleSeg,
  **Apache-2.0**, true soft alpha matte), BiRefNet (**MIT**), U²-Net (**Apache-2.0**). **Avoid:** MODNet
  (official weights non-commercial), BackgroundMattingV2 (needs a captured bg), RMBG-1.4/2.0 (Bria NC).

## Swap plan for `matting.py` (drop-in; `layered_api.py`/`layered_pipeline.py` unchanged)
Adapter delivered + smoke-tested: `experiments/r2_e1_matte/seg_matte.py`. It re-exports `fg_mask_lr`,
`composite`, `auto_downsample_ratio` **verbatim from `matting.py`** (gate output byte-identical shape)
and provides `load_seg`/`load_rvm`(alias)/`matte_sequence`/`benchmark` with identical signatures.

Two-line swap in the consumer:
```python
# import matting                                   # RVM, CC BY-NC-SA (NON-COMMERCIAL)
import seg_matte as matting                        # permissive, BSD-3 (commercial-OK)
# model = matting.load_rvm("mps")                  # old
model = matting.load_seg("mps", "deeplabv3_mobilenetv3", ema=0.5)   # new (best plate + stable edge)
```
`matte_sequence`/`fg_mask_lr`/`build_plate` work unchanged — `matte_sequence` returns the same
`[(fgr, pha)]` (`fgr`=source frame, unused by layered which composites real-frame × alpha; `pha`=person
alpha at LR in [0,1]). **GOTCHA #17:** frames consumed in display order; the stateless seg ignores RVM's
recurrent state, and the optional alpha-EMA is threaded in display order as the stand-in "recurrent
state". Human-only scope preserved. Plate is built via temporal median over all gates → jitter-robust;
EMA mainly steadies the per-frame composite FG edge.

## Executive summary
1. **The LAYERED pipeline survives the swap to a permissive matte** — plates match RVM on coverage
   (≈75%), hole% (≈25%), sharpness (95–109%); montage shows clean subject-removed plates for all.
2. **Best runnable, license-clean replacement: DeepLabV3-MobileNetV3-Large (BSD-3) + alpha-EMA** —
   cleanest plate, RVM-parity hole%/sharpness, 0.85× latency.
3. **Fastest viable: LRASPP-MobileNetV3 + EMA (BSD-3)** — 0.64× RVM latency, EMA restores RVM-level
   temporal stability.
4. Matte quality vs RVM is high (IoU 0.84–0.90, MAD ≤0.07); the only gap was stateless temporal jitter,
   **fully closed by a free alpha-EMA** (GOTCHA #17 handled).
5. **Production target: MediaPipe Selfie Segmentation (Apache-2.0)** — run in an isolated env (protobuf
   conflict here). For RVM-grade soft hair under a permissive license: PP-MattingV2/PP-HumanSeg
   (Apache-2.0) or BiRefNet (MIT).
6. **Net:** RVM is replaceable today with a two-line import/loader swap → the layered mode can ship
   commercially.
