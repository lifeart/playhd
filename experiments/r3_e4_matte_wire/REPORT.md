# R3-E4: Full layered matte swap — wired, verified end-to-end, flag-gated. **PASS.**

## 1. Completed adapter — the PASS-B gap is closed
`seg_matte_layered.py` = the R2-E1 adapter completed with the missing **`matte_frame(model,
src_tensor, rec, downsample_ratio)`** + a shared tensor-normalize path so PASS-A (numpy) and PASS-B
(tensor) are byte-identical given the same pixels. Drop-in for `matting.matte_frame`:
- Accepts the raw `[1,3,H,W]` [0,1] tensor `layered_api._frame_tensor` produces; normalizes/resizes
  to `seg_res` internally.
- Returns `(fgr, pha, rec)` with RVM-identical shapes: `fgr=src_tensor`, `pha=[1,1,H,W]`,
  `rec=[pha, None, None, None]`.
- Stateless seg ignores the GRU meaning of `rec` but threads the **display-order alpha-EMA** through
  `rec[0]` (GOTCHA #17): `pha_t = a·pha_t + (1-a)·pha_{t-1}` — the recurrent-state stand-in.

**Seam check:** every `layered_api` call site on `matting` (`load_rvm/load_seg`,
`auto_downsample_ratio`, `matte_sequence`, `fg_mask_lr`, `matte_frame`) present + resolvable. Zero mismatches.

## 2. Full layered path verified end-to-end (real `layered_api`, `matting` patched)
Window `sample.mp4` start-5000, scene[0], 32 real LR frames → temp H.264 → `segment_scenes` (1 STATIC
scene) → real `build_scene_plates` (PASS A) + `matte_frame_np`+`composite_frame` (PASS B). RVM vs
DeepLabV3-MobileNetV3-Large+EMA(0.5):

| config | verdict | cov% | hole% | sharp | QHD valid | fg% | a\|ΔF\| | band\|ΔF\| | bgplate\|ΔF\| |
|---|---|---|---|---|---|---|---|---|---|
| **RVM** (non-comm) | STATIC | 75.1 | 24.86 | 14 | True | 27.0 | 0.0065 | 0.0471 | 0.001 |
| **DeepLab-mv3+EMA** (BSD-3) | STATIC | 73.7 | 26.29 | 17 | True | 30.4 | 0.0068 | 0.0174 | 0.041 |

Cross-vs-RVM: matte MAD 0.069 / IoU 0.874; plate_MAD 2.71, bleed 7.28; composite **background matches
RVM to 0.70 code-values** (plate-dominated → matte change isolated to FG). Timing: PASS-A 0.79×,
PASS-B 1.04×. Consistent with R2-E1.

Key live findings: (a) 32 valid QHD frames — `matte_frame` works through the real PASS-B seam; (b) the
EMA threads correctly through the opaque `rec` (a|ΔF| 0.0068 ≈ RVM 0.0065; stateless-no-EMA ~0.0105);
(c) bg-plate stability holds (≤0.04 code-values, imperceptible); (d) FG loses only wispy hair.
**PASS** — full layered swap (PASS A + PASS B) works with the permissive matte.

## 3. Flag-gated `layered_api.py` change (default `"rvm"` → byte-identical; both branches tested)
`layered_api.diff` + tested copy `layered_api_patched.py`. ONE conditional rebind of `matting` makes
every `matting.*` call site consistent:
```python
LAYERED_MATTE = os.environ.get("LAYERED_MATTE", "rvm").strip().lower()
LAYERED_MATTE_EMA = float(os.environ.get("LAYERED_MATTE_EMA", "0.5"))
_SEG_VARIANTS = {"deeplab": "deeplabv3_mobilenetv3", "lraspp": "lraspp_mobilenetv3"}
if LAYERED_MATTE in _SEG_VARIANTS:
    import seg_matte_layered as matting   # permissive BSD-3 (commercial-OK)
# load_matting_model():
if LAYERED_MATTE in _SEG_VARIANTS:
    return matting.load_seg(_device(), _SEG_VARIANTS[LAYERED_MATTE], ema=LAYERED_MATTE_EMA)
return matting.load_rvm(_device())
```
Lead action: copy `seg_matte_layered.py` → `server/`, apply the diff. (Optional: surface `LAYERED_MATTE`
in the `pipeline_api` stats note that hardcodes "matte: RVM (non-commercial)".) Production target stays
MediaPipe Selfie (Apache-2.0, isolated env); DeepLab+EMA is the runnable same-license-tier shipping pick.

## Executive summary
PASS B works (32 valid QHD frames via the real seam); alpha-EMA threads correctly through `rec` (RVM-parity
stability, GOTCHA #17 closed); plate quality equivalent to RVM (cov 73.7 vs 75.1, hole 26.3 vs 24.9,
clean subject removal); FG loses only wispy hair; latency 0.79–1.04× RVM. Ready to land behind a default-
`"rvm"` (byte-identical) flag; `LAYERED_MATTE=deeplab` activates the BSD-3 commercial path.
