# E4 — Two untested levers ("test, don't assume")

**Levers:** (A) FSR2 RGB color-box clamping to suppress ghosting, (B) fp16 for the SR network.
**Window:** high-motion window A (`--start-frame 0 --max-frames 48`), LR 640×320 → HD 2560×1280 (x4).
**Engine imported READ-ONLY** (`prototype/sr.py`, `prototype/derisk.py`, `prototype/gpu_ops.py`); all new code in this dir. MPS shared with 3 sibling experiments → timing is **ratios + best-of-N (min = least-contended)**.

Scripts: `exp_clamp.py`, `exp_fp16.py`. Artifacts in `artifacts/`.

## Lever B — fp16 for the SR net → GO (strong for the heavy x4plus anchor)

Cast a **deepcopy** of the loaded fp32 net to `.half()` (sr.py never mutated) + fp16 input, on real decoded frames. Best-of-N=7 (warm once, 7 timed reps, take **min**; median also shown). PSNR(fp16,fp32) on the uint8 SR **output** is the primary fidelity check; var-Laplacian secondary.

| model | fp32 min ms | fp16 min ms | speedup (best-of-N) | speedup (median)¹ | PSNR(16,32) med / min | max pixel Δ | varLap 32 / 16 | finite |
|---|---|---|---|---|---|---|---|---|
| realesrgan-x4plus (heavy, RRDBNet x23) | ~1790–2430 | ~1771–1789 | **×1.24** | ×1.48 | **75.8 / 71.7 dB** | **2 / 255** | 249 / 250 | ✅ |
| realesrgan (compact, SRVGGNet) | ~143–132 | ~124–114 | **×1.15** | ×1.15 | 80.7 / 76.1 dB | 2 / 255 | 218 / 218 | ✅ |

¹ Median > best-of-N only because **fp32** was hit harder by contention (fp32 median drifted to 2.6–3.4 s while its *min* held ~2.2 s; fp16 *min* was steady ~1.78 s). So **×1.24 (best-of-N) is the conservative honest floor**; uncontended is ≥ that.

**Fidelity (primary):** PSNR(fp16,fp32) = **71–82 dB**, far above the ~50 dB "visually identical" line; worst-case per-pixel Δ = **1–2 LSB / 255**; var-Laplacian identical to integer; no NaN/Inf (RRDB's 0.2 residual scaling keeps activations inside fp16 range). Output visually indistinguishable. Diff crops (`artifacts/fp16_*_fp32_fp16_diffx8.png`) are near-black even at ×8 amplification.

**GO** — real speedup AND fidelity. Most valuable on the quality-mode x4plus anchor (~2.21→1.78 s/anchor, free ~24% cut at the dominant SR cost, zero visible quality loss). Compact gains less (×1.15, it's less compute-bound) but is also free quality-wise.
*Measured vs inferred:* speedups measured under live contention (×1.24 is a floor; couldn't get an isolated-GPU number). Fidelity fully measured (deterministic).

## Lever A — FSR2 color-box clamp (`--clamp`) → NO-GO (no usable operating point)

Applied **post-hoc** to the real propagated recon from `reconstruct()` (numpy backend, occ=full, I/P chain, compact per-frame SR across all 48 frames — cheap + contention-robust; conclusion re-confirmed on a real heavy x4plus anchor). Color box = channel-wise `mean ± γ·std` over a 3×3 LR window, bilinear-upscaled ×4, then `clip(recon_HD, lo, hi)`. Sweep γ ∈ {1, 2, 4, 8, ∞} (∞ = no clamp).

- **Ghosting proxy** = HF divergence `|HF(recon) − HF(perframe)|` over the **warped (non-occluded)** region of the 6 highest-motion frames (`perframe` = per-frame SR of the actual current LR = geometrically correct ⇒ HF-divergence = misplaced fine detail = ghosting). Also restricted to **low-reactive** warped pixels (the visually-similar bad MVs the occlusion mask misses). Plus temporal `|ΔF|` + visual.
- **Detail** = var-Laplacian + PSNR on a static textured crop (LR (80,64), near-zero motion), clamped vs unclamped.

| γ | ghost HF (warped) | ghost HF (low-react) | clip % of warped | \|ΔF\| | static varLap (ret %) | PSNR(clamp,unclamp) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | 0.073 | 0.057 | 23.7% | 34.87 | 764 (90.5%) | 42.7 dB | clips real detail, ghost unchanged |
| 2 | 0.075 | 0.059 | 22.8% | 34.90 | 845 (100%) | 48.2 dB | detail OK, ghost not reduced |
| 4 | 0.075 | 0.060 | 21.8% | 34.93 | 845 (100%) | 49.0 dB | detail OK, ghost not reduced |
| 8 | 0.074 | 0.059 | 20.8% | 34.95 | 845 (100%) | 49.5 dB | detail OK, ghost not reduced |
| ∞ (no clamp) | **0.071** | **0.056** | 0% | 35.03 | 845 (100%) | — | baseline |

Heavy x4plus static-detail re-confirmation (1 real anchor SR, same crop): γ=1 → varLap **89.3%**, PSNR **42.3 dB** (clipped); γ≥2 → 100% / 46–47 dB (untouched). Higher-detail model loses **more** at tight γ, as the docs warned.

Fails both ways: **loose γ≥2** preserves detail (100% varLap) but **does not reduce ghosting** (HF-divergence actually rises 0.071→0.074–0.075, |ΔF| flat) while needlessly clipping ~21–23% of warped pixels; **tight γ=1** clips real SR texture (varLap→90%, PSNR 42 dB) and **still** doesn't fix ghosting. **No γ reduces ghosting without clipping detail → NO-GO.**

**Why (structural, not a tuning miss):** codec MVs are rate-distortion-optimized → they point to *visually-similar* content (minimizes SAD), so the ghosted misplaced HD texture has nearly the same local color statistics as the current LR neighborhood and lands **inside** the box built from that same neighborhood. An RGB color box centered on the current LR cannot separate ghost from signal — it's derived from the content the ghost resembles. Where the box *is* tight enough to clip (flat regions, γ=1) it clips legitimate SR high-frequencies, not the ghost. FSR2's clamp works because its ghost is a *different surface* (different color/depth); H.264 ghosting is same-content-misplaced, so statistics overlap. (The existing reactive + Ruder fwd-bwd mask already catches high-residual bad MVs; the residual low-reactive ghosting it misses is exactly the visually-similar case the color box also can't see.)
*Honesty:* the proxy mixes true ghosting with per-frame SR hallucination variance, but the argument is proxy-independent and the proxy confirms the clamp never moves recon toward the correct reference at any γ. Window-A's highest-motion frames are dark (the ghost crop `artifacts/clamp_ghost_f10_*.png` is dim) → the aggregate metric over the full warped region, not a single crop, is decisive; static-detail crops (`artifacts/clamp_staticdetail_{compact,heavy}_*.png`) show γ≥2 is a visual no-op.

## Integration proposals (for SEAM-VERIFY)

**B — fp16 (RECOMMENDED, default-OFF), cast point in `prototype/sr.py`:**
1. `load_model(name, device=None, half=False)`: after `model.eval().to(_DEVICE)`, `half = half and _DEVICE.type in ("mps","cuda")` (**guard:** fp16 conv is slow/unsupported on CPU); if `half:` `model = model.half()`. **Cache key must include precision** — make `_MODELS`/`_LAT` key on `(name, half)` so fp32 and fp16 nets/latencies don't collide. *Seam:* every `_MODELS[...]`/`_LAT[...]`/`_LAST_MODEL` site (sr.py ~L181–243) keys on the same tuple.
2. `upscale(rgb_uint8, model="realesrgan", half=False)`: input cast `.float().div_(255.0)` (L235) → `.half()` when half; call `load_model(model, half=half)`; insert `out = out.float()` before the uint8 download (L243–244) so clamp/mul/round run in fp32.
3. Thread `half` through `upscale_to` + `build_perframe_cache`; add `--fp16` CLI flag in `derisk.main()` (default False) → `build_perframe_cache(..., half=args.fp16)`. *Seam:* `--fp16` flag name, `half=` kwarg, and the cache-key tuple must match end-to-end. Default OFF keeps the byte-identical regression path.

**A — `--clamp` (NOT RECOMMENDED; design only for seam-checking):** would apply per-direction after `warp_hd()` and before the occlusion blend — numpy `_warp_one` (derisk.py ~L436) and torch `warp_one_t` (~L705, needs a `gpu_ops.color_box_clamp` twin: boxFilter→`F.avg_pool2d`+upsample, clamp via `torch.maximum/minimum`); CLI `--clamp FLOAT` (default `inf` = OFF, byte-identical) + `--clamp-win INT` (default 3). **Do not merge** — measured NO-GO at every γ.
