"""R4-E4 -- Auto-mode selector. A CHEAP probe that recommends instant | quality | layered
for a clip from signals computable in ONE light decode pass (codec MVs + a few decoded frames),
WITHOUT a full per-mode render.

Reuses the product/prototype READ-ONLY:
  * pipeline_api.stream_gops        -- the SAME GOP/scene chunking the product renders with.
  * anchor_sr.anchor_indices / _lr_fallback_fraction -- EXACT occlusion-fallback %, SR-independent.
  * derisk.build_lr_flow            -- LR motion-vector field -> motion magnitude.
  * scene_detect.find_cuts          -- detected scene-cut count (plate-safety).
  * background_plate.estimate_global_motion -- static-vs-moving camera verdict (plate-safety).
  * layered_api matte (RVM/seg)     -- human-coverage check, run ONLY on layered candidates.
  * cv2.Canny                       -- graphic/sharp-edge density.

Decision rule (derived from R3-E2's robustness sweep, thresholds re-measured here):
  1. fb_react_mean > 50%           -> quality   (low-light/noise: MVs unreliable -> instant
                                                 collapses to per-frame SR, real-time breaks;
                                                 the plate would denoise-corrupt too).
  2. mv_mag_mean > 8  OR  fb_react_mean > 15%  -> quality   (high motion / fast pan: instant
                                                 tOF 3-8x worse + breaks real-time).
  3. n_scenes > 1                  -> quality   (multi-cut: layered plate spans scenes = unsafe;
                                                 the safe default handles cuts via fresh anchors).
  4. layered candidate (static camera, single scene, no suspected hidden cut):
        run matte on K sampled frames -> human_coverage
        if HUMAN_LO <= coverage <= HUMAN_HI and plate residual is low+consistent -> layered.
  5. otherwise                     -> instant   (low-moderate motion, not fb-saturated, single
                                                 non-human scene: real-time AND acceptable).

The matte (the only neural step) runs ONLY when steps 1-3 pass AND the camera is static AND the
scene is single -> it never fires on the high-motion / noisy / multi-cut clips, keeping the probe
cheap. If the matte model can't load (offline), human coverage is UNKNOWN -> we conservatively
DROP layered and fall through to instant (real-time, no corruption risk) -- never crash.

CLI:  python3 experiments/r4_e4_automode/recommend_mode.py <clip> [--frames 24] [--start 0]
"""
from __future__ import annotations

import os
import sys
import gc
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, List

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "server"))
sys.path.insert(0, os.path.join(REPO, "prototype"))

import pipeline_api as P            # noqa: E402  product chunking + modes
import anchor_sr                    # noqa: E402  exact occlusion fallback
import derisk                       # noqa: E402  LR flow
import scene_detect                 # noqa: E402  scene-cut count
import background_plate as bp       # noqa: E402  static-camera verdict

try:
    import torch as _torch
except Exception:                   # pragma: no cover
    _torch = None


# --------------------------------------------------------------------------- #
# Thresholds (re-measured on R3-E2's authored clips; see REPORT validation table).
# --------------------------------------------------------------------------- #
FB_NOISE = 50.0      # % reactive fallback mean -> low-light/noise -> quality           (c3=94.9)
FB_HI = 15.0         # % reactive fallback mean -> high local occlusion -> quality       (c1=16.9,c6=15.3)
MOTION_HI = 10.0     # MEDIAN per-frame LR-MV magnitude (px) -> instant softens above. Median (not mean)
                     #   so a single cut-frame MV spike (the codec predicts huge MVs across a cut, e.g.
                     #   c5b max 207) does NOT inflate the motion signal: c1 med 30, c6 med 11.7 -> quality;
                     #   c5/c5b med 8.0, c2 med 2.3, c4/c7 med ~0 -> not high-motion.
SMEAR_MOTION = 3.0   # a MISSED cut only smears instant when there is motion to warp across it. Static
                     #   scenes (c7 med 0.11) survive a missed cut on instant (tOF 0.43); a moving missed
                     #   cut (c5 med 8) smears -> quality. Below this median, a missed cut is harmless.
STATIC_THRESH_PX = 0.6   # |median camera MV| below this = static (matches layered STATIC_THRESH_PX)
HUMAN_LO = 0.03      # matte coverage band for a plausible talking-head (3%..85%); a real
HUMAN_HI = 0.85      #   human @ sample.mp4#5000 measures ~0.25; synthetic non-humans -> 0.00
HIDDEN_CUT_CHROMA = 18.0  # mean |ΔRGB| (chroma-sensitive) between consecutive sampled frames that the
                          #   luma-only cut detector can MISS (c7 similar-luma + cool-tint cut) -> reject
                          #   layered (would paint the wrong plate). Belt-and-suspenders w/ the human gate.
PLATE_RESID_MAX = 12.0    # mean |frame - median-plate| over BACKGROUND pixels (0..255 luma levels). A
                          #   truly static bg -> small residual (just noise); a moving/parallax bg or a
                          #   wrong (hidden-cut) plate -> large residual -> the plate won't denoise -> reject.
MATTE_K = 3          # frames to matte for the human-coverage check (kept tiny -> cheap)


def _free_gpu():
    gc.collect()
    if _torch is not None:
        try:
            if _torch.backends.mps.is_available():
                _torch.mps.empty_cache()
        except Exception:
            pass


# Lazy, cached matte model (loaded only when a clip reaches the layered gate).
_MATTE = {"model": None, "tried": False, "err": None}


def _get_matte():
    if _MATTE["tried"]:
        return _MATTE["model"]
    _MATTE["tried"] = True
    try:
        import layered_api as L
        _MATTE["L"] = L
        _MATTE["model"] = L.load_matting_model()
    except Exception as e:                       # offline / no weights -> UNKNOWN human, never crash
        _MATTE["err"] = repr(e)
        _MATTE["model"] = None
    return _MATTE["model"]


@dataclass
class Signals:
    n_frames: int = 0
    n_chunks: int = 0
    n_scenes: int = 1
    mv_mag_mean: float = 0.0
    mv_mag_median: float = 0.0      # robust motion signal (cut-spike-immune); drives the routing
    mv_mag_max: float = 0.0
    edge_density_mean: float = 0.0
    fb_react_mean: float = 0.0
    fb_react_max: float = 0.0
    camera_verdict: str = "UNKNOWN"          # STATIC | MOVING | UNKNOWN
    global_vec_mag_px: float = float("nan")
    hidden_cut_suspected: bool = False
    chroma_diff_max: float = 0.0
    human_coverage: Optional[float] = None    # None = not probed / matte unavailable
    plate_resid: Optional[float] = None       # mean bg residual to median plate (luma levels)
    probe_s: float = 0.0


@dataclass
class Recommendation:
    mode: str
    reason: str
    signals: dict
    matte_unavailable: bool = False


# --------------------------------------------------------------------------- #
# ONE light decode pass: motion + edges + exact occlusion-fallback, over the product's chunking.
# --------------------------------------------------------------------------- #
def _scan(clip_path, n, stride):
    mags, edges, fbs = [], [], []
    decoded = []                  # flat (ptype,lr,mvs) for estimate_global_motion (reuses these MVs)
    samples = []                  # a few RGB frames (for chroma-cut check + matte)
    n_chunks = 0
    n_frames = 0
    for chunk in P.stream_gops(clip_path, max_frames=n):
        n_chunks += 1
        anchors, backbone = anchor_sr.anchor_indices(chunk)
        h_lr, w_lr = chunk[0][1].shape[:2]
        for i, (pt, lr, mvs, *_) in enumerate(chunk):   # R12: stream_gops yields 4-tuples (qp at [3])
            decoded.append((pt, lr, mvs))
            if n_frames % stride == 0:                  # sample subset for the cheap signals
                g = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY)
                edges.append(float((cv2.Canny(g, 80, 160) > 0).mean()))
                if mvs is not None and len(mvs):
                    fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want="all")
                    m = np.sqrt(fx * fx + fy * fy)
                    mags.append(float(np.nanmean(m)) if np.isfinite(m).any() else 0.0)
                else:
                    mags.append(0.0)
                if i in anchors:
                    fbs.append(0.0)
                else:
                    fbs.append(anchor_sr._lr_fallback_fraction(chunk, i, backbone, "reactive"))
                samples.append(lr)          # spread across the WHOLE window (stride-spaced) so the
                                            # chroma hidden-cut check + matte span any mid-window cut
            n_frames += 1
    return mags, edges, fbs, decoded, samples, n_chunks, n_frames


def _chroma_hidden_cut(samples):
    """Max mean |ΔRGB| between consecutive SAMPLED frames -- chroma-sensitive, so a similar-LUMA
    cut (the luma-only scene detector can miss, e.g. a cool-tint splice) still shows up here.
    Returns (max_chroma_diff, suspected)."""
    if len(samples) < 2:
        return 0.0, False
    diffs = []
    for a, b in zip(samples[:-1], samples[1:]):
        if a.shape != b.shape:
            diffs.append(255.0)
            continue
        diffs.append(float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32)))))
    mx = max(diffs) if diffs else 0.0
    return mx, (mx > HIDDEN_CUT_CHROMA)


def _human_coverage(samples):
    """Matte K sampled frames -> (coverage_fraction, plate_resid). coverage None if the matte model
    is unavailable. plate_resid = mean |frame - temporal-median-plate| over BACKGROUND pixels, in
    luma levels: small on a truly static bg, large on a moving/parallax/wrong-plate bg."""
    model = _get_matte()
    if model is None:
        return None, None
    L = _MATTE["L"]
    dev = L._device()
    pick = samples[:: max(1, len(samples) // MATTE_K)][:MATTE_K]
    covs, resid = [], []
    luma = [0.299 * s[..., 0] + 0.587 * s[..., 1] + 0.114 * s[..., 2] for s in pick]
    plate_ref = np.median(np.stack([l.astype(np.float32) for l in luma], 0), axis=0)
    rec = [None] * 4
    for s, l in zip(pick, luma):
        ratio = L.downsample_ratio(*s.shape[:2])
        pha, rec = L.matte_frame_np(model, s, rec, ratio, dev)
        covs.append(float((pha > 0.5).mean()))
        bg = (pha <= 0.5)                                   # background pixels only
        if bg.any():
            resid.append(float(np.mean(np.abs(l.astype(np.float32) - plate_ref)[bg])))
    _free_gpu()
    cov = float(np.median(covs)) if covs else 0.0
    pr = float(np.mean(resid)) if resid else None
    return cov, pr


def recommend_mode(input_path, max_frames: int = 24, stride: int = 1,
                   start: int = 0, verbose: bool = False) -> Recommendation:
    """Cheap probe -> (mode, reason, signals). `start` lets a caller probe a window of a long file
    by trimming first (handled by the CLI/validation harness via a temp clip; process_clip itself
    has no start arg, so for windows we probe a pre-trimmed clip)."""
    t0 = time.perf_counter()
    s = Signals()

    mags, edges, fbs, decoded, samples, n_chunks, n_frames = _scan(input_path, max_frames, stride)
    s.n_frames, s.n_chunks = n_frames, n_chunks
    s.mv_mag_mean = round(float(np.mean(mags)), 3) if mags else 0.0
    s.mv_mag_median = round(float(np.median(mags)), 3) if mags else 0.0
    s.mv_mag_max = round(float(np.max(mags)), 3) if mags else 0.0
    s.edge_density_mean = round(float(np.mean(edges)), 4) if edges else 0.0
    s.fb_react_mean = round(float(np.mean(fbs)) * 100, 2) if fbs else 0.0
    s.fb_react_max = round(float(np.max(fbs)) * 100, 2) if fbs else 0.0

    # scene-cut count (cheap luma pass) -> plate safety
    try:
        cuts, _total = scene_detect.find_cuts(input_path, max_frames=max_frames)
        s.n_scenes = len(cuts) + 1
    except Exception as e:
        s.n_scenes = 1
        if verbose:
            print("  [warn] find_cuts:", repr(e))

    # static-camera verdict (reuses the MVs already decoded) -> plate safety
    try:
        gm = bp.estimate_global_motion(decoded, static_thresh=STATIC_THRESH_PX)
        s.camera_verdict = gm["verdict"]
        s.global_vec_mag_px = round(float(gm["global_vec_mag_px"]), 3)
    except Exception as e:
        s.camera_verdict = "UNKNOWN"
        if verbose:
            print("  [warn] estimate_global_motion:", repr(e))

    s.chroma_diff_max, s.hidden_cut_suspected = _chroma_hidden_cut(samples)
    s.chroma_diff_max = round(s.chroma_diff_max, 2)

    matte_unavailable = False

    # a MISSED cut = a strong chroma discontinuity the luma cut-detector did NOT split on (n_scenes
    # stays 1). stream_gops splits at DETECTED cuts (instant then re-anchors cleanly across them), so
    # only a MISSED cut leaves a chunk spanning it -> instant warps the pre-cut anchor across = smear.
    missed_cut = s.hidden_cut_suspected and s.n_scenes == 1

    # ---------------- decision rule ----------------
    if s.fb_react_mean > FB_NOISE:
        mode = "quality"
        reason = (f"low-light/noisy: reactive fallback {s.fb_react_mean:.0f}% > {FB_NOISE:.0f}% "
                  f"-> unreliable MVs collapse instant to per-frame SR (real-time breaks); "
                  f"plate would denoise-corrupt. Safe default = quality.")
    elif s.mv_mag_median > MOTION_HI or s.fb_react_mean > FB_HI:
        mode = "quality"
        reason = (f"high motion / occlusion: median mvMag {s.mv_mag_median:.1f} (>{MOTION_HI:.0f}) "
                  f"or fb {s.fb_react_mean:.0f}% (>{FB_HI:.0f}%) -> instant softens/flickers "
                  f"(tOF 3-8x) & breaks real-time. Escalate to quality.")
    elif missed_cut and s.mv_mag_median > SMEAR_MOTION:
        mode = "quality"
        reason = (f"MISSED cut: chroma discontinuity {s.chroma_diff_max:.0f} (>{HIDDEN_CUT_CHROMA:.0f}) "
                  f"not split by the luma detector (n_scenes=1) + motion (median mvMag "
                  f"{s.mv_mag_median:.1f} > {SMEAR_MOTION:.0f}) -> instant warps across the un-split cut "
                  f"= smear. Escalate to quality.")
    else:
        # ---- layered candidate: static, single scene, low motion/fb. Confirm human + plate. ----
        static_ok = s.camera_verdict in ("STATIC", "UNKNOWN")
        if static_ok and not s.hidden_cut_suspected:
            cov, pr = _human_coverage(samples)
            s.human_coverage = None if cov is None else round(cov, 3)
            s.plate_resid = None if pr is None else round(pr, 2)
            human = (cov is not None and HUMAN_LO <= cov <= HUMAN_HI)
            if cov is None:
                matte_unavailable = True
                mode = "instant"
                reason = (f"static single non-multi-cut scene but matte unavailable "
                          f"({_MATTE['err']}) -> can't confirm human; instant is real-time & "
                          f"corruption-free (layered dropped for safety).")
            elif not human:                                  # non-human (incl. coverage 0)
                mode = "instant"
                reason = (f"static single scene, low motion (mvMag {s.mv_mag_mean:.1f}) & fb "
                          f"{s.fb_react_mean:.0f}%, but human coverage {cov:.2f} outside "
                          f"[{HUMAN_LO},{HUMAN_HI}] (non-human) -> instant: real-time & acceptable.")
            elif pr is not None and pr > PLATE_RESID_MAX:    # human but bg not static enough
                mode = "quality"
                reason = (f"static+human (coverage {cov:.2f}) but plate residual {pr:.1f} > "
                          f"{PLATE_RESID_MAX} levels -> bg not truly static / hidden cut; plate "
                          f"unsafe -> quality.")
            else:                                            # human + safe static plate
                mode = "layered"
                reason = (f"static camera ({s.camera_verdict}), single scene, human coverage "
                          f"{cov:.2f} in [{HUMAN_LO},{HUMAN_HI}], plate residual {pr:.1f} <= "
                          f"{PLATE_RESID_MAX} levels -> safe static-bg plate win.")
        else:
            mode = "instant"
            why = "moving camera" if not static_ok else "suspected hidden cut (chroma)"
            reason = (f"single scene, low motion (mvMag {s.mv_mag_mean:.1f}) & fb {s.fb_react_mean:.0f}% "
                      f"but {why} -> layered unsafe; instant is real-time & acceptable.")

    s.probe_s = round(time.perf_counter() - t0, 2)
    if verbose:
        print(f"  signals: {asdict(s)}")
    return Recommendation(mode=mode, reason=reason, signals=asdict(s),
                          matte_unavailable=matte_unavailable)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    rec = recommend_mode(a.clip, max_frames=a.frames, stride=a.stride, verbose=True)
    print(f"\nRECOMMEND: {rec.mode}\nREASON: {rec.reason}\nprobe_s={rec.signals['probe_s']}")
