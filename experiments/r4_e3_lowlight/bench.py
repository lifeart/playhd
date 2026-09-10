"""R4-E3: fix the instant-mode low-light/noise quality cliff.

The cliff (R3-E2): on noisy/low-light content sensor noise -> unreliable codec MVs ->
the occlusion fallback fires on ~95% of pixels -> EVERY non-anchor frame exceeds the
INSTANT_FALLBACK_THRESH (0.50) safeguard -> a full per-frame compact-SR runs on all
24/24 frames -> ms/frame 34 -> 124 (3.6x). The anchor-propagation premise gives ~zero
benefit on pure noise (warps are unreliable so almost everything is fallback anyway).

This harness drives the REAL product fast path (pipeline_api.process_clip, instant mode)
with surgical monkeypatches that emulate the two candidate fixes -- server/ stays
READ-ONLY, all new code is in this dir:

  baseline   : unchanged product (thresh 0.50, no cap, no denoise).
  cap<C>     : FALLBACK-SATURATION CAP. Via the EXISTING `thresh_fn` hook (the same one
               E2's motion-keyed feature uses): for any frame whose LR occlusion-fallback
               fraction exceeds C, return an impossible threshold (2.0) so it is NOT
               escalated to SR -> accept bicubic (the propagation/SR premise is void at
               >C fallback). Self-gating: a frame below C fallback is byte-identical to
               baseline, so clean content (fallback < a few %) is untouched.
  denoise_*  : CHEAP PRE-DENOISE at LR before the MV/occlusion stage (wrap stream_gops).
               Denoising drops the reactive residual |lr_cur - warp(lr_prev)| -> fallback
               drops -> propagation works -> fewer SR upgrades. NOT free (per-frame cost)
               and it SOFTENS clean content, so it is measured as a gated/auto-route option.

Honest metrics (ms/frame as in-session ratios under the shared GPU; never NR-sharpness
alone -- that's the noise-SR trap):
  * ms/frame + ratio vs baseline (same process, back-to-back)         -- the real-time test
  * n_sr_calls / sr_calls_per_frame / n_adaptive_upgrades             -- the cliff driver
  * effective fallback% (characterized under the config's LR frames)  -- the mechanism
  * tOF vs decoded-LR  AND  tOF vs the NOISE-FREE clean signal        -- temporal stability
  * PSNR/SSIM of output(down) vs the clean signal                     -- fidelity-to-intent
  * temporal-noise energy in a static patch (output luma std over t)  -- noise amplification
  * PSNR/SSIM output-vs-baseline-output (HD)                          -- how different from today

Run:  python3 experiments/r4_e3_lowlight/bench.py [--frames 24] [--clips c3_lowlight,c4_talkinghead]
"""
import os
import sys
import gc
import json
import time
import argparse

import numpy as np
import cv2
import av

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CLIPS = os.path.join(REPO, "experiments", "r3_e2_robustness", "clips")
OUT = os.path.join(HERE, "out")
SAMPLES = os.path.join(HERE, "samples")
RESULTS = os.path.join(HERE, "results.json")
os.makedirs(OUT, exist_ok=True)
os.makedirs(SAMPLES, exist_ok=True)

sys.path.insert(0, os.path.join(REPO, "server"))
sys.path.insert(0, os.path.join(REPO, "prototype"))
import pipeline_api as P        # noqa: E402
import anchor_sr                # noqa: E402
import derisk                   # noqa: E402

try:
    import torch as _torch
except Exception:
    _torch = None


def _free_gpu():
    gc.collect()
    if _torch is not None:
        try:
            if _torch.backends.mps.is_available():
                _torch.mps.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# FIX (a): fallback-saturation cap, via the existing thresh_fn hook.
# --------------------------------------------------------------------------- #
def make_cap_builder(cap, occ_mode="reactive", motion_gate=None):
    """Return a drop-in replacement for pipeline_api._motion_keyed_thresh_fn(chunk, base):
    a per-frame threshold fn that returns an IMPOSSIBLE threshold (2.0) for any frame whose
    LR occlusion-fallback fraction exceeds `cap` (so build_anchor_cache + patch_high_fallback
    both DECLINE to escalate it to SR), else the scalar base threshold (byte-identical).

    `motion_gate` (px/frame): if set, the cap fires ONLY on frames whose mean LR-MV magnitude
    is BELOW the gate -- i.e. high fallback + LOW motion = NOISE (cap it); high fallback + HIGH
    motion = genuine fast-motion disocclusion (do NOT cap, keep the SR). This makes the cap
    byte-identical on fast-pan content while still fixing the noise cliff."""
    def builder(chunk, base_thresh):
        anchors, backbone = anchor_sr.anchor_indices(chunk)
        h_lr, w_lr = chunk[0][1].shape[:2]
        memo, mmo = {}, {}

        def frac(i):
            if i not in memo:
                memo[i] = (0.0 if i in anchors
                           else anchor_sr._lr_fallback_fraction(chunk, i, backbone, occ_mode))
            return memo[i]

        def mvmag(i):
            if i not in mmo:
                _, _, mvs = chunk[i]
                if mvs is None or len(mvs) == 0:
                    mmo[i] = 0.0
                else:
                    fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want="all")
                    mg = np.sqrt(fx * fx + fy * fy)
                    mmo[i] = float(np.nanmean(mg)) if np.isfinite(mg).any() else 0.0
            return mmo[i]

        def thr(i):
            if frac(i) > cap and (motion_gate is None or mvmag(i) < motion_gate):
                return 2.0
            return base_thresh
        return thr
    return builder


# --------------------------------------------------------------------------- #
# FIX (b): cheap pre-denoise at LR, via wrapping stream_gops.
# --------------------------------------------------------------------------- #
def _denoise_gauss(lr):
    return cv2.GaussianBlur(lr, (3, 3), 0)


def _denoise_median(lr):
    return cv2.medianBlur(lr, 3)


def _denoise_bilateral(lr):
    return cv2.bilateralFilter(lr, 5, 40, 5)


def _denoise_nlm(lr):
    # fast non-local means (heavier; the "good denoise, watch the cost" reference)
    return cv2.fastNlMeansDenoisingColored(lr, None, 6, 6, 5, 11)


DENOISERS = {"gauss": _denoise_gauss, "median": _denoise_median,
             "bilat": _denoise_bilateral, "nlm": _denoise_nlm}


def make_denoise_stream(denoise_fn):
    orig = P.stream_gops

    def wrapped(*args, **kw):
        for chunk in orig(*args, **kw):
            yield [(pt, np.ascontiguousarray(denoise_fn(lr)), mvs, *rest) for (pt, lr, mvs, *rest) in chunk]  # R12: 4-tuple-safe, preserves qp
    return wrapped


# --------------------------------------------------------------------------- #
# Clean (noise-free) c3 signal -- the honest detail/fidelity reference. Mirrors
# r3_e2_robustness/make_clips.clip_lowlight WITHOUT the per-frame sensor noise.
# --------------------------------------------------------------------------- #
def clean_lowlight(n, w=480, h=272):
    base = np.full((h, w, 3), 14, np.uint8)
    cv2.circle(base, (w // 2, h // 2), 60, (40, 38, 30), -1, cv2.LINE_AA)
    cv2.rectangle(base, (60, 60), (120, 110), (28, 24, 20), -1)
    frames = []
    for i in range(n):
        f = base.copy()
        cx = w // 2 + int(6 * np.sin(i / 4.0))
        cv2.circle(f, (cx, h // 2), 30, (60, 55, 45), -1, cv2.LINE_AA)
        frames.append(f)
    return frames


# --------------------------------------------------------------------------- #
# decode / metrics
# --------------------------------------------------------------------------- #
def decode_rgb(path, max_frames=None):
    cont = av.open(path)
    vs = cont.streams.video[0]
    out = []
    for fr in cont.decode(vs):
        if max_frames is not None and len(out) >= max_frames:
            break
        out.append(fr.to_ndarray(format="rgb24"))
    cont.close()
    return out


def _down(frames, w, h):
    return [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]


def _ssim(a, b):
    return derisk.ssim(a, b)


def char_fallback(clip_path, n, denoise_fn=None, occ_mode="reactive"):
    """Effective mean/max LR occlusion-fallback % over the product's real chunking, optionally
    AFTER the config's denoise (so the denoise config reports the fallback it actually sees)."""
    fr = []
    for chunk in P.stream_gops(clip_path, max_frames=n):
        if denoise_fn is not None:
            chunk = [(pt, np.ascontiguousarray(denoise_fn(lr)), mvs, *rest) for (pt, lr, mvs, *rest) in chunk]  # R12: 4-tuple-safe, preserves qp
        anchors, backbone = anchor_sr.anchor_indices(chunk)
        for i in range(len(chunk)):
            if i in anchors:
                fr.append(0.0)
            else:
                fr.append(anchor_sr._lr_fallback_fraction(chunk, i, backbone, occ_mode))
    return round(float(np.mean(fr)) * 100, 2), round(float(np.max(fr)) * 100, 2)


def quality_metrics(out_path, clip_path, n, clean_frames, baseline_out):
    out_hd = decode_rgb(out_path)[:n]
    lr_in = decode_rgb(clip_path, max_frames=n)
    m = min(len(out_hd), len(lr_in), len(clean_frames))
    out_hd, lr_in, clean = out_hd[:m], lr_in[:m], clean_frames[:m]
    h_lr, w_lr = lr_in[0].shape[:2]
    out_lr = _down(out_hd, w_lr, h_lr)

    tof_lr = derisk.tof(out_lr, lr_in)
    tof_clean = derisk.tof(out_lr, clean)
    psnr_clean = float(np.mean([cv2.PSNR(out_lr[i], clean[i]) for i in range(m)]))
    ssim_clean = float(np.mean([_ssim(out_lr[i], clean[i]) for i in range(m)]))

    # temporal-noise energy: a static background patch far from the moving subject.
    yp, xp = slice(2, 40), slice(w_lr - 80, w_lr - 4)
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)[yp, xp]
                      for f in out_lr], axis=0)
    tnoise = float(np.mean(np.std(stack, axis=0)))

    res = {
        "tof_out_vs_lr": round(tof_lr, 4),
        "tof_out_vs_clean": round(tof_clean, 4),
        "psnr_out_vs_clean_lr": round(psnr_clean, 2),
        "ssim_out_vs_clean_lr": round(ssim_clean, 4),
        "temporal_noise_static": round(tnoise, 3),
        "n_compared": m,
    }
    if baseline_out is not None and os.path.exists(baseline_out):
        base_hd = decode_rgb(baseline_out)[:m]
        k = min(m, len(base_hd))
        res["psnr_vs_baseline_hd"] = round(
            float(np.mean([cv2.PSNR(out_hd[i], base_hd[i]) for i in range(k)])), 2)
        res["ssim_vs_baseline_hd"] = round(
            float(np.mean([_ssim(out_hd[i], base_hd[i]) for i in range(k)])), 4)
    return res


def save_triptych(out_path, clip_path, clean_frames, tag, idxs):
    out_hd = decode_rgb(out_path)
    lr_in = decode_rgb(clip_path)
    saved = []
    for j in idxs:
        if j >= len(out_hd):
            continue
        h, w = out_hd[j].shape[:2]
        noisy = cv2.resize(lr_in[j], (w, h), interpolation=cv2.INTER_NEAREST)
        clean = cv2.resize(clean_frames[j], (w, h), interpolation=cv2.INTER_NEAREST)
        combo = np.concatenate([noisy, out_hd[j], clean], axis=1)[:, :, ::-1]
        p = os.path.join(SAMPLES, f"{tag}_f{j:02d}.png")
        cv2.imwrite(p, combo)
        saved.append(os.path.basename(p))
    return saved


# --------------------------------------------------------------------------- #
# run one (clip, config) cell through the real product fast path
# --------------------------------------------------------------------------- #
def run_cell(clip, cfg_name, cfg, n):
    clip_path = os.path.join(CLIPS, clip + ".mp4")
    out_path = os.path.join(OUT, f"{clip}_{cfg_name}.mp4")

    # ---- install monkeypatches for this config ----
    orig_thresh = P.INSTANT_FALLBACK_THRESH
    orig_builder = P._motion_keyed_thresh_fn
    orig_stream = P.stream_gops
    denoise_fn = DENOISERS.get(cfg.get("denoise")) if cfg.get("denoise") else None
    try:
        if cfg.get("cap") is not None:
            P._motion_keyed_thresh_fn = make_cap_builder(
                cfg["cap"], occ_mode="reactive", motion_gate=cfg.get("motion_gate"))
        if denoise_fn is not None:
            P.stream_gops = make_denoise_stream(denoise_fn)

        t0 = time.perf_counter()
        P.process_clip(clip_path, "instant", max_frames=n, out_path=out_path)
        wall = time.perf_counter() - t0
        s = dict(P.LAST_STATS)
    finally:
        P._motion_keyed_thresh_fn = orig_builder
        P.stream_gops = orig_stream
        P.INSTANT_FALLBACK_THRESH = orig_thresh
        P.end_job()
        _free_gpu()

    fb_mean, fb_max = char_fallback(clip_path, n, denoise_fn=denoise_fn)
    cell = {
        "clip": clip, "config": cfg_name, "out_path": out_path,
        "ms_per_frame": s.get("ms_per_frame"),
        "wall_s": round(wall, 2),
        "n_sr_calls": s.get("n_sr_calls"),
        "sr_calls_per_frame": s.get("sr_calls_per_frame"),
        "n_adaptive_upgrades": s.get("n_adaptive_upgrades"),
        "t_sr_s": s.get("t_sr_s"), "t_recon_s": s.get("t_recon_s"),
        "fallback_mean_pct": fb_mean, "fallback_max_pct": fb_max,
        "resolution": s.get("out_resolution"), "encoder": s.get("video_encoder"),
    }
    return cell


CONFIGS = [
    ("baseline", {}),
    ("cap0.70", {"cap": 0.70}),
    ("cap0.60", {"cap": 0.60}),
    ("denoise_gauss", {"denoise": "gauss"}),
    ("denoise_bilat", {"denoise": "bilat"}),
    ("denoise_nlm", {"denoise": "nlm"}),
    # winning combo for the fast tier: cap (speed -> bicubic, no per-frame SR) + cheap
    # pre-denoise (quality -> the bicubic source is denoised, so noise doesn't flicker through).
    ("cap0.70+gauss", {"cap": 0.70, "denoise": "gauss"}),
    ("cap0.70+bilat", {"cap": 0.70, "denoise": "bilat"}),
    # motion-gated cap: cap ONLY low-motion (noise) frames -> byte-identical on genuine fast-motion.
    ("capmg0.70", {"cap": 0.70, "motion_gate": 8.0}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--clips", default="c3_lowlight,c4_talkinghead")
    ap.add_argument("--configs", default=",".join(c[0] for c in CONFIGS))
    a = ap.parse_args()
    clips = a.clips.split(",")
    want = set(a.configs.split(","))
    cfgs = [(nm, c) for nm, c in CONFIGS if nm in want]
    n = a.frames

    results = {}
    if os.path.exists(RESULTS):
        try:
            results = json.load(open(RESULTS))
        except Exception:
            results = {}

    clean = {"c3_lowlight": clean_lowlight(n)}     # only c3 has a known clean signal

    for clip in clips:
        clean_frames = clean.get(clip) or decode_rgb(os.path.join(CLIPS, clip + ".mp4"),
                                                     max_frames=n)
        baseline_out = os.path.join(OUT, f"{clip}_baseline.mp4")
        for cfg_name, cfg in cfgs:
            key = f"{clip}::{cfg_name}"
            print(f"\n=== {key} (frames={n}) ===", flush=True)
            t0 = time.perf_counter()
            cell = run_cell(clip, cfg_name, cfg, n)
            cell["quality"] = quality_metrics(
                cell["out_path"], os.path.join(CLIPS, clip + ".mp4"), n, clean_frames,
                None if cfg_name == "baseline" else baseline_out)
            cell["cell_wall_s"] = round(time.perf_counter() - t0, 2)
            results[key] = cell
            json.dump(results, open(RESULTS, "w"), indent=2)
            q = cell["quality"]
            print(f"  ms/frame={cell['ms_per_frame']}  n_sr={cell['n_sr_calls']}  "
                  f"fb={cell['fallback_mean_pct']}%  tOF(lr)={q['tof_out_vs_lr']}  "
                  f"tOF(clean)={q['tof_out_vs_clean']}  PSNR/clean={q['psnr_out_vs_clean_lr']}  "
                  f"tnoise={q['temporal_noise_static']}  "
                  f"PSNRvsBase={q.get('psnr_vs_baseline_hd')}", flush=True)
            _free_gpu()

    # visual triptychs (noisy-in | output | clean-signal) for c3 configs
    if "c3_lowlight" in clips:
        for cfg_name, _ in cfgs:
            op = os.path.join(OUT, f"c3_lowlight_{cfg_name}.mp4")
            if os.path.exists(op):
                save_triptych(op, os.path.join(CLIPS, "c3_lowlight.mp4"),
                              clean["c3_lowlight"], f"c3_{cfg_name}", [0, 8, 16])

    print(f"\nresults -> {RESULTS}")


if __name__ == "__main__":
    main()
