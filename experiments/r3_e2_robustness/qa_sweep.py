"""R3-E2 content-robustness QA sweep. For each authored clip x each product mode
(instant/quality/layered) it:

  1. CHARACTERIZES the content (no GPU): per-frame LR motion-vector magnitude, Canny
     edge density, and the EXACT occlusion-fallback fraction (anchor_sr._lr_fallback_*,
     which is SR-model-independent -> cheap & honest), under BOTH occ modes
     (reactive=instant, adaptive=quality), using the SAME stream_gops chunking the
     product uses. Done ONCE per clip.
  2. RUNS the product: pipeline_api.process_clip(clip, mode, max_frames=N). Any crash is
     caught WITH its traceback and recorded (no silent swallow).
  3. VERIFIES output: re-decode -> codec/frames/resolution/audio + sync.
  4. HONEST flicker: tOF of the OUTPUT (downscaled to input LR) vs the decoded input LR
     (ref=decoded LR = cleanest motion truth; lower=steadier/tracks true motion). Plus a
     supporting per-frame LR-consistency PSNR series (NOT used alone) to LOCALIZE where the
     reconstruction diverges (e.g. a scene-cut smear dip).

Results are written incrementally to results.json (resume-safe: a completed clip+mode cell
is skipped on re-run). GPU (MPS) is freed between every cell (shared with siblings).

Run:  python3 experiments/r3_e2_robustness/qa_sweep.py [--modes instant,quality,layered]
                                                       [--clips c1,c2,...] [--frames 24] [--force]
"""
import os
import sys
import gc
import json
import time
import argparse
import traceback

import numpy as np
import cv2
import av

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CLIPS = os.path.join(HERE, "clips")
OUT = os.path.join(HERE, "out")
RESULTS = os.path.join(HERE, "results.json")
SAMPLES = os.path.join(HERE, "samples")
os.makedirs(OUT, exist_ok=True)
os.makedirs(SAMPLES, exist_ok=True)

# import product + prototype READ-ONLY
sys.path.insert(0, os.path.join(REPO, "server"))
sys.path.insert(0, os.path.join(REPO, "prototype"))
import pipeline_api as P     # noqa: E402
import anchor_sr             # noqa: E402
import derisk                # noqa: E402

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
# decode helpers
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


# --------------------------------------------------------------------------- #
# CONTENT CHARACTERIZATION (no GPU, no neural SR)
# --------------------------------------------------------------------------- #
def characterize(clip_path, n):
    """Motion magnitude, edge density, and EXACT occlusion-fallback fraction (both occ modes),
    over the product's real GOP chunking. Returns a dict."""
    mags, edges = [], []
    fb_react, fb_adapt = [], []
    n_chunks = 0
    ptypes = []
    for chunk in P.stream_gops(clip_path, max_frames=n):
        n_chunks += 1
        anchors, backbone = anchor_sr.anchor_indices(chunk)
        h_lr, w_lr = chunk[0][1].shape[:2]
        for i, (pt, lr, mvs, *_) in enumerate(chunk):   # R12: stream_gops yields 4-tuples (qp at [3])
            ptypes.append(pt)
            # edge density (Canny on luma) -> graphic/sharp-edge signal
            g = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY)
            e = cv2.Canny(g, 80, 160)
            edges.append(float((e > 0).mean()))
            # MV magnitude (mean over covered pixels) -> motion signal
            if mvs is not None and len(mvs):
                fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want="all")
                m = np.sqrt(fx * fx + fy * fy)
                mags.append(float(np.nanmean(m)) if np.isfinite(m).any() else 0.0)
            else:
                mags.append(0.0)
            # exact occlusion-fallback fraction (SR-independent), both occ modes
            if i in anchors:
                fb_react.append(0.0)
                fb_adapt.append(0.0)
            else:
                fb_react.append(anchor_sr._lr_fallback_fraction(chunk, i, backbone, "reactive"))
                fb_adapt.append(anchor_sr._lr_fallback_fraction(chunk, i, backbone, "adaptive"))
    return {
        "n_frames": len(ptypes), "n_chunks": n_chunks,
        "ptypes": "".join(ptypes),
        "n_iframes": ptypes.count("I"),
        "mv_mag_mean": round(float(np.mean(mags)), 3),
        "mv_mag_max": round(float(np.max(mags)), 3),
        "edge_density_mean": round(float(np.mean(edges)), 4),
        "fallback_reactive_mean": round(float(np.mean(fb_react)) * 100, 2),
        "fallback_reactive_max": round(float(np.max(fb_react)) * 100, 2),
        "fallback_adaptive_mean": round(float(np.mean(fb_adapt)) * 100, 2),
        "fallback_adaptive_max": round(float(np.max(fb_adapt)) * 100, 2),
    }


# --------------------------------------------------------------------------- #
# OUTPUT VERIFICATION + honest flicker
# --------------------------------------------------------------------------- #
def verify_output(out_path, expect_n):
    cont = av.open(out_path)
    try:
        vs = cont.streams.video[0]
        codec = vs.codec_context.name
        w, h = vs.codec_context.width, vs.codec_context.height
        v_dur = float(vs.duration * vs.time_base) if vs.duration else None
        has_audio = len(cont.streams.audio) > 0
        a_codec = a_dur = None
        if has_audio:
            a = cont.streams.audio[0]
            a_codec = a.codec_context.name
            a_dur = float(a.duration * a.time_base) if a.duration else None
        n = sum(1 for _ in cont.decode(vs))
    finally:
        cont.close()
    sync_ok = (v_dur is not None and a_dur is not None and abs(v_dur - a_dur) <= 0.5)
    return {
        "valid_h264": codec == "h264",
        "frames": n, "frames_match": (n == expect_n),
        "resolution": f"{w}x{h}",
        "video_dur_s": round(v_dur, 3) if v_dur else None,
        "has_audio": has_audio, "audio_codec": a_codec,
        "audio_dur_s": round(a_dur, 3) if a_dur else None,
        "audio_sync_ok": sync_ok,
    }


def flicker_and_fidelity(out_path, clip_path, n):
    """Honest tOF (output vs true LR motion) + supporting per-frame LR-consistency PSNR series
    (for localizing divergence, e.g. a scene-cut smear). Both decoded at LR for speed."""
    out_hd = decode_rgb(out_path)
    lr_in = decode_rgb(clip_path, max_frames=n)
    m = min(len(out_hd), len(lr_in))
    out_hd, lr_in = out_hd[:m], lr_in[:m]
    h_lr, w_lr = lr_in[0].shape[:2]
    out_down = [cv2.resize(f, (w_lr, h_lr), interpolation=cv2.INTER_AREA) for f in out_hd]
    tof = derisk.tof(out_down, lr_in)
    # supporting per-frame LR-consistency (output-vs-true-LR PSNR); NOT a sole metric
    lrc = [round(derisk.psnr_lr_consistency(out_hd[i], lr_in[i]), 2) for i in range(m)]
    # output temporal-difference energy (mean |ΔF| of consecutive output luma, 0..255)
    df = []
    for i in range(1, m):
        a = cv2.cvtColor(out_hd[i - 1], cv2.COLOR_RGB2GRAY).astype(np.float32)
        b = cv2.cvtColor(out_hd[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        df.append(float(np.mean(np.abs(b - a))))
    return {
        "tof_out_vs_lr": round(tof, 4),
        "lr_consistency_mean": round(float(np.mean(lrc)), 2),
        "lr_consistency_min": round(float(np.min(lrc)), 2),
        "lr_consistency_series": lrc,
        "deltaF_mean": round(float(np.mean(df)), 3) if df else None,
        "deltaF_max": round(float(np.max(df)), 3) if df else None,
        "n_compared": m,
    }


def save_samples(out_path, clip_path, tag, idxs):
    """Save side-by-side (input-LR-upscaled | output) PNGs at chosen frame indices for
    visual artifact inspection."""
    out_hd = decode_rgb(out_path)
    lr_in = decode_rgb(clip_path)
    saved = []
    for j in idxs:
        if j >= len(out_hd) or j >= len(lr_in):
            continue
        h, w = out_hd[j].shape[:2]
        up = cv2.resize(lr_in[j], (w, h), interpolation=cv2.INTER_NEAREST)
        combo = np.concatenate([up, out_hd[j]], axis=1)[:, :, ::-1]  # RGB->BGR for cv2
        p = os.path.join(SAMPLES, f"{tag}_f{j:02d}.png")
        cv2.imwrite(p, combo)
        saved.append(os.path.basename(p))
    return saved


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #
ALL_CLIPS = ["c1_fastpan", "c2_graphics", "c3_lowlight",
             "c4_talkinghead", "c5_scenecut", "c5b_scenecut_strong", "c6_oddres"]
# frames at which to dump visual samples per clip (hard cut at frame 28 for c5/c5b)
SAMPLE_IDX = {"c5_scenecut": [0, 26, 28, 30, 38],
              "c5b_scenecut_strong": [0, 26, 28, 30, 38], "default": [0, 10, 20]}


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {"characterize": {}, "cells": {}}


def save_results(r):
    with open(RESULTS, "w") as f:
        json.dump(r, f, indent=2)


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="instant,quality,layered")
    ap.add_argument("--clips", default=",".join(ALL_CLIPS))
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    modes = a.modes.split(",")
    clips = a.clips.split(",")
    n = a.frames

    res = load_results()

    for clip in clips:
        clip_path = os.path.join(CLIPS, clip + ".mp4")
        if not os.path.exists(clip_path):
            print(f"[skip] missing {clip_path}")
            continue
        # ---- characterize once ----
        if a.force or clip not in res["characterize"]:
            t0 = time.perf_counter()
            try:
                res["characterize"][clip] = characterize(clip_path, n)
                res["characterize"][clip]["t_s"] = round(time.perf_counter() - t0, 2)
                print(f"[char] {clip}: {res['characterize'][clip]}")
            except Exception:
                res["characterize"][clip] = {"error": traceback.format_exc()}
                print(f"[char][ERROR] {clip}:\n{res['characterize'][clip]['error']}")
            save_results(res)

        for mode in modes:
            key = f"{clip}::{mode}"
            if not a.force and key in res["cells"]:
                print(f"[skip done] {key}")
                continue
            print(f"\n=== RUN {key} (max_frames={n}) ===")
            out_path = os.path.join(OUT, f"{clip}_{mode}.mp4")
            cell = {"clip": clip, "mode": mode, "max_frames": n}
            t0 = time.perf_counter()
            try:
                P.process_clip(clip_path, mode, max_frames=n, out_path=out_path)
                cell["wall_s"] = round(time.perf_counter() - t0, 2)
                cell["stats"] = dict(P.LAST_STATS)
                cell["verify"] = verify_output(out_path, cell["stats"].get("n_frames", n))
                cell["flicker"] = flicker_and_fidelity(out_path, clip_path, n)
                idxs = SAMPLE_IDX.get(clip, SAMPLE_IDX["default"])
                cell["samples"] = save_samples(out_path, clip_path, f"{clip}_{mode}", idxs)
                cell["ok"] = True
                print(f"[ok] {key} wall={cell['wall_s']}s "
                      f"ms/frame={cell['stats'].get('ms_per_frame')} "
                      f"tOF={cell['flicker']['tof_out_vs_lr']} "
                      f"valid={cell['verify']['valid_h264']} "
                      f"res={cell['verify']['resolution']} "
                      f"audio={cell['verify']['audio_codec']}")
            except Exception:
                cell["ok"] = False
                cell["error"] = traceback.format_exc()
                cell["wall_s"] = round(time.perf_counter() - t0, 2)
                print(f"[ERROR] {key}:\n{cell['error']}")
            finally:
                P.end_job()           # ensure single-job lock released even on crash
                _free_gpu()
            res["cells"][key] = cell
            save_results(res)

    print("\n=== DONE ===")
    print(f"results -> {RESULTS}")


if __name__ == "__main__":
    run()
