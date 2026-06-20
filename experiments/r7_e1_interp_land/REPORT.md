# R7-E1 — "Smooth 2×" MV interpolation: LANDED + verified end-to-end. PASS.

## What landed (2 shared files)
- **`server/interp_pass.py`** (new) — verbatim copy of the validated `experiments/r4_e2_interp_wire/interp_pass.py`
  (both ship-blockers baked in: intra-hole routing only + scene-cut guard).
- **`server/pipeline_api.py`** (+85/−10) — the R4-E2 default-OFF wiring on the real `process_clip` instant fast
  path: import `interp_pass`; `INSTANT_INTERP_2X=False`, `INTERP_CUT_THRESH=0.5`; `smooth = fast and
  INSTANT_INTERP_2X`; `out_fps = fps*2 if smooth else fps`; emit `interp_pass.midpoint_torch(left, recon_t,
  fx, fy, eff_scale, ...)` before each real frame (left = prev-chunk last recon for i==0 else
  `R[i-1]['recon']`; connecting field = frame i's codec 'past' MV via `connecting_flow`); carry the chunk's
  last recon (`.clone()` before `del R`); trailing-dup to close; mux with `n_emit`/`out_fps`. Additive
  `LAST_STATS` (`n_video_frames`/`out_fps` + `smooth_2x`/`n_interp`/`n_interp_dup` when ON).

Real frames stay raw (grain writes a new tensor — `GpuGrain.apply` non-mutating), so the midpoint is
structurally output-only — never enters `R[]`. `app.py` unaffected (flag default-OFF, LAST_STATS additive).

## Verification (through the REAL `process_clip`, sample.mp4, 40-frame window over the I-frame boundary @28)
| Check | Result |
|---|---|
| **OFF byte-identical to pre-change** (whole-mp4 file md5 + all 40 decoded-frame md5s) | **PASS** — identical `file_md5=ef8d4835405a`, 40/40 match; only additive LAST_STATS keys differ |
| **ON exactly 2×** (40 → 80) | **PASS** |
| **Valid in-sync 2×-fps mp4** | **PASS** — h264 1280×640, 80 frames, out_fps=50, video 1.6s ≈ audio 1.625s, sync_ok |
| **Midpoint output-only** (ON even/real frames == OFF frames) | **PASS** (40/40); OFF makes zero interp calls |
| **Scene-cut guard duplicates (not ghosts)** | **PASS** — I-frame boundary @28 fires (hole_frac>0.99); all 12 dups == their left recon exactly |

Lead-reconfirmed: imports clean, `INSTANT_INTERP_2X` default False, instant OFF n_sr=3 baseline, out_fps=source.

## Executive summary
Landed the validated R4-E2 "smooth 2×" MV-reuse interpolation as shipped default-OFF code in the real server
(`server/interp_pass.py` + a 2-flag wire in `process_clip`). OFF is byte-identical (entire output mp4 md5
unchanged), so instant/quality/layered cannot regress. ON doubles to exactly 2× frames at 2× fps with a
valid audio-synced mp4; the midpoint is output-only (never enters R[], never called when OFF); the cut-guard
duplicates the last real frame (no ghost) at the I-frame boundary. Both ship-blockers intact. Clean PASS; a
UI toggle (a `smooth_2x` param on `app.py`'s process_clip call) is the only optional follow-up.
