# R10-E2: Codec-deblock PREPROCESSOR before x4plus — GATED GO

**Verdict: GATED GO.** Removing H.264 artifacts *before* x4plus (a 1× deblock pass, not a
model swap) is a **real ceiling-raise on HEAVILY-compressed, low/mid-detail anchors**:
on real-CRF-35 H.264 it beats plain x4plus on **LPIPS −4.7%, DISTS −6.8%, AND PSNR +0.33 dB**,
4/4 per-frame, with **var-Lap == x4plus** (artifact-removal, not blur). It **must be gated**:
on *light* compression it strips real detail (loses both metrics), and on *dense photographic
texture* even at CRF 35 it over-smooths (the **DISTS guard caught this unanimously**, 0/4
DISTS-wins on texture46k). So: a **default-OFF, compression-gated preprocessor** on the sparse
anchor — NOT an unconditional one. Different/better than R9-E2's flat NO-GO on replacing the
anchor: keep x4plus, feed it a cleaner input on the inputs that need it.

Distinct from R9-E2 (replace x4plus → fabricates/over-denoises, NO-GO). Here x4plus is unchanged.

---

## Method (reuses R9-E2's REAL-H.264 harness exactly)
- **GT** = decoded `sample.mp4` 256-crop, R6-E1's 5 validated windows (talkinghead@5000,
  highmotion@0, texture18k/24k/46k). n=4 frames/window.
- **LR** = 2× INTER_AREA down (→128) → **REAL libx264 (PyAV) encode** @ CRF {27 moderate,
  35 heavy} → decode → genuine 8×8 blocking / ringing / 4:2:0 chroma. (ffmpeg CLI broken; PyAV in-proc.)
- **Pipelines** (all net ×2, 128→512→256, identical tail): `x4plus` (BASELINE),
  `scunet_x4plus` (deblock→x4plus), `scunet_x4plus_b85` (stack on R8-E3's β=0.85 toward compact),
  `bilat_x4plus`, `h264db_x4plus` (classical), + `bicubic`/`compact` refs.
- **Arbiter** = full-reference **LPIPS(Alex)+DISTS+PSNR** (pyiqa, MPS). **DISTS = over-smoothing
  guard.** var-Lap = fake/over-sharpen flag ONLY (GOTCHA #23), never the verdict.
- **Deblock model:** **SCUNet `scunet_color_real_psnr`** (17.9 M, **scale-1** restoration, spandrel
  auto-loaded), from HF `deepinv/scunet` via `hf_hub_download` (72 MB, clean). Its BSRGAN-style
  practical degradation prior **includes JPEG/DCT compression** → transfers to H.264 intra blocks.

### FBCNN note (the task's primary candidate)
FBCNN is **not on HuggingFace Hub** (searched models + 6 mirror repos; official weights are
GitHub-release-only = the throttled path the task says to avoid). SCUNet real_psnr is the on-thesis,
HF-hosted, spandrel-loadable substitute (deblock+denoise on a JPEG-inclusive degradation model).
Classical baselines included regardless (free, integrable): **bilateral** + a **hand-written 8px-grid
weak deblock** (H.264-in-loop-filter spirit: low-pass block boundaries only where the cross-edge jump
is small → preserves true edges).

---

## Results — OVERALL MEAN (280 records, n=4)
| pipeline | LPIPS↓ | DISTS↓ | PSNR↑ | var-Lap | vs x4plus |
|---|---|---|---|---|---|
| bicubic (floor) | 0.2308 | 0.2173 | 22.64 | 461 | |
| compact 1.2M | 0.1275 | 0.1684 | 23.63 | 3036 | |
| **x4plus (BASELINE)** | **0.1179** | **0.1614** | 23.78 | 3120 | — |
| **scunet→x4plus** | 0.1209 | 0.1582 | 23.89 | 3071 | LPIPS +2.6% / DISTS −2.0% |
| scunet→x4plus→β.85 | 0.1199 | 0.1575 | 24.06 | 2953 | LPIPS +1.7% / DISTS −2.4% |
| bilat→x4plus | 0.1251 | 0.1629 | 23.91 | 3048 | LPIPS +6.1% / DISTS +1.0% |
| h264db→x4plus | 0.1197 | 0.1629 | 23.67 | 3096 | LPIPS +1.5% / DISTS +0.9% |

Pooled over both CRFs, scunet improves DISTS but loses LPIPS → **not an unconditional both-win →
unconditional NO-GO.** The signal is entirely in the **CRF split**:

## The CRF split is the whole story
| pipeline | CRF27 LPIPS / DISTS / PSNR | CRF35 LPIPS / DISTS / PSNR |
|---|---|---|
| x4plus (BASELINE) | 0.0726 / 0.1155 / 26.22 | 0.1633 / 0.2073 / 21.35 |
| **scunet→x4plus** | 0.0861 / 0.1232 / 26.10 **(loses both)** | **0.1557 / 0.1932 / 21.68 — WINS BOTH+PSNR** |
| scunet→x4plus→β.85 | 0.0843 / 0.1203 / 26.34 | **0.1555 / 0.1947 / 21.77 — WINS BOTH+PSNR** |

- **Heavy (CRF 35): clean win** — LPIPS −4.7%, DISTS −6.8%, PSNR **+0.33 dB**. PSNR *up* means it's
  not even a perception-distortion trade. **DISTS (texture-sensitive) improving the MOST is the proof
  it removes artifacts, not detail.**
- **Moderate (CRF 27): loses both** (LPIPS +18.6%, DISTS +6.7%) — deblock scrubs real detail the light
  codec preserved. Exactly the task's predicted "light = deblock may hurt."

## Per-frame consistency + the over-smoothing guard firing (CRF 35, n=4)
| window | BOTH-win | LPIPS-win | DISTS-win | reading |
|---|---|---|---|---|
| highmotion | **4/4** | 4/4 | 4/4 | clean win (low-detail title) |
| texture18k | **4/4** | 4/4 | 4/4 | clean win (headline) |
| texture24k | **4/4** | 4/4 | 4/4 | clean win (chart+text) |
| texture46k | **0/4** | 4/4 | **0/4** | **DISTS catches over-smooth** on densest texture |
| talkinghead | 0/4 | 1/4 | 0/4 | faces don't benefit |

The win is unanimous per-frame on graphics/text/low-detail; **texture46k is the textbook trap** —
LPIPS prefers the cleaner look on every frame, **DISTS rejects it on every frame** (it scrubbed real
dark-scene micro-texture; pixel-peep `out/codecpeep_texture46k_crf35.png`). var-Lap is *higher* there
(2959→3011) — exactly why var-Lap is never the arbiter. The win cases keep detail:
**heavy var-Lap x4plus 2893 vs scunet 2881 (both 87% of GT)** — not blur.

## Gated subset where it's a clean ceiling-raise (CRF 35, exclude face + texture46k)
| pipeline | LPIPS↓ | DISTS↓ | PSNR↑ |
|---|---|---|---|
| x4plus | 0.1013 | 0.2029 | 20.13 |
| **scunet→x4plus** | **0.0880 (−13.1%)** | **0.1684 (−17.0%)** | **20.63 (+0.50 dB)** |

Visual: `out/codecpeep_texture24k_crf35.png` — x4plus carries H.264 blocking into the upscale;
scunet→x4plus renders the chart text + background visibly cleaner, text still legible (not smoothed away).

## Classical baselines (free, integrable)
- **bilateral**: over-denoises — loses LPIPS overall (+6.1%), wins only 2/10 cells.
- **h264db (8px-grid)**: ~free, conservative — ~tie overall (+1.5% LPIPS), 2/10 cells (incl.
  texture24k/heavy both-win). Too weak to fix the bulk of artifacts but never harmful.
- **Neural SCUNet is the one that delivers the −13/−17% heavy win.** Classical = a free, weaker fallback.

---

## VERDICT: GATED GO — codec-deblock preprocessor raises the ceiling where x4plus struggles most
- **GO** on **heavy compression (≈CRF ≥ 33–35) + low/mid-detail anchors** (titles, headlines, charts,
  graphics, simple scenes): beats x4plus on LPIPS **and** DISTS **and** PSNR, 4/4 per-frame, not blur.
- **NO-GO** on light/moderate compression (strips real detail) and on dense photographic texture / faces
  even at heavy CRF (DISTS-caught over-smoothing) → **must be gated**, default-OFF.
- **CRF-dependence is monotonic and large** — the more H.264 crushed the frame, the more there is to
  recover; this is the de-risk the task asked for.

### Cost (anchor-affordable)
SCUNet 1× @128px steady **117 ms** vs x4plus 128→512 **196 ms** (isolated MPS; full-run 521/702 ms are
inflated by the shared-GPU sibling). **+~60 % anchor latency, amortized over the ~2–12 % anchor frames**
→ small net cost, and only paid when the gate fires (heavy anchors).

### Integration (default-OFF, byte-identical) — `build_perframe_cache.patch` + `deblock_pre.py`
One-liner: **`build_perframe_cache(..., deblock_cfg=None)` runs `deblock_pre.apply(lr, cfg, qp)` on the
LR frame before `upscale_to`; `deblock_cfg` absent/None ⇒ identity ⇒ byte-identical.** Gate =
**bitstream QP** (the real codec-MV pipeline already decodes it — apply iff QP ≥ qp_min) or a
**blockiness proxy** when QP is unavailable (LR-only; separates CRF27/35 on every window but overlaps
on hard-edged charts, so QP is preferred), plus an optional dense-texture var-Lap skip-guard.
`verify_patch.py` ALL PASS: (1) OFF cfg → byte-identical, (2) blockiness gate ON-heavy/OFF-light,
(3) QP gate, (4) texture guard. Stacking R8-E3's β=0.85 is neutral-to-slightly-better on heavy
(DISTS −2.4% overall) and inherits its own default-OFF safety.

### Threats / could-not-verify
1. **One clip, 256-crops, n=4** — but the heavy win is 4/4 per-frame on 3 distinct windows (sign-robust).
2. **Synthetic 2×-down + real-encode** (no true HR) — same pseudo-GT convention as R6-E1/R8-E3/R9-E2.
3. **Gate threshold not tuned per-genre**; blockiness overlaps on graphics → production should gate on
   true QP (available) and keep texture46k-style content out via the var-Lap skip-guard or QP-only.
4. **Anchor-only** — propagation/tOF effect of cleaner anchors is unmeasured (ships default-OFF, like R8-E3).
5. FBCNN itself untested (not on HF); SCUNet is the loadable on-thesis stand-in. An actual H.264/HEVC-
   *finetuned* deblocker would likely widen the heavy win and shrink the moderate loss.

**Bottom line:** codec-artifact removal **is** a missing piece x4plus needs — but only on **heavily-
compressed, non-dense** anchors. As a **QP-gated, default-OFF preprocessor** it's a measured ceiling-raise
(−13% LPIPS / −17% DISTS / +0.5 dB on the gated heavy subset) at ~+60% amortized anchor cost. The DISTS
guard did its job: it caught the over-smoothing trap (texture46k, faces) an LPIPS-only read would ship.

Artifacts: run_ab.py, analyze.py, results.json (280), latency.json, deblock_pre.py,
build_perframe_cache.patch, verify_patch.py, out/fab_*.png + out/codecpeep_*.png, models/scunet_color_real_psnr.pth.

---

## GRADUATION ADDENDUM (lead, post-integration) — propagation-validated, but stays default-OFF (gate-reliability blocker)

To decide a default-ON flip, the lead ran the two graduation checks (`graduation_ab.py`):

**1. Does the anchor win SURVIVE PROPAGATION + stay temporally stable? → YES.** Full quality
pipeline (`process_clip`, GRAIN OFF), deblock ON vs OFF, on a real CRF35-degraded 48-frame window,
scored over ALL frames (anchors + propagated) vs pseudo-HD GT: **LPIPS −0.7%, DISTS −0.9%, tOF −0.5%**
— all three the right direction, consistent, no flicker added. The −13%/−17% anchor-crop win dilutes
to a modest-but-real net gain over a MIXED clip (faces don't benefit, graphics/low-detail do); it is
larger on heavy-low-detail-throughout content. The lever is **safe and the win is real at clip scale.**

**2. Can the gate be made reliable for a silent default? → NO (without bitstream QP).**
- **Bitstream QP is unavailable** in this PyAV build (libavcodec 62 / PyAV 17 expose no `qscale_table`
  / qp frame attr — only MOTION_VECTORS side-data; the legacy QP export was removed from modern ffmpeg).
  So the precise "heavy-compression-only" gate cannot be built here.
- **The blockiness proxy OVER-FIRES on moderate content:** ordinary broadcast SD (`sample.mp4`,
  `short.mp4`) trips `block_min=1.30` on **9/25 frames** (mean 1.23, max 1.49) — it cannot separate
  the heavy regime (deblock helps, CRF35) from the moderate regime (deblock **loses both metrics**,
  CRF27, measured above). A silent default-ON would therefore apply detail-stripping deblock to ~36%
  of normal-content frames → a regression on exactly the general case.

**Decision: keep `deblock_pre` default-OFF — now upgraded from "anchor-only validated" to
"propagation-validated + tOF-safe".** The blocker to a default-flip is GATE RELIABILITY, not quality.
Ship as a documented opt-in: enable `MODE_CONFIG["quality"]["deblock_pre"] = {"model":
"scunet_color_real_psnr.pth", "gate":"blockiness", "block_min":<raise for your content>}` when you
KNOW the input is heavily compressed (low-scaled / re-encoded video). **What would unblock default-ON:**
(a) a libavcodec build exposing per-frame QP (gate precisely on QP≥~33), or (b) a calibrated
higher block_min + the dense-texture skip-guard validated across genres. Both are external/follow-up.
Artifact: `graduation_ab.py`.
