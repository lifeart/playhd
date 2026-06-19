"""R4-E4 validation -- HONEST, self-contained, no reliance on stale numbers.

For each clip we (a) run the cheap recommend_mode() probe (INPUT signals: MVs, fallback%, edges,
scenes, static-camera, human-coverage) and (b) DERIVE the confirmed-best mode from OUTCOME metrics
of real renders (tOF + ms/frame + LR-consistency-min) via a rule that is INDEPENDENT of the probe's
input signals -- so "probe agrees with best" is not circular.

CONFIRMED-BEST rule (outcome metrics only; escalate to quality only when it ACTUALLY helps):
  * layered is best  IF a layered render is SAFE (lrc_min >= LRC_SAFE -> no wrong-plate corruption)
                     AND beats instant on tOF (a genuine static-bg win).
  * else instant     IF instant is GOOD-ENOUGH: tOF <= INSTANT_TOF_OK AND real-time (ms <= RT_MS_CAP).
  * else quality     IF instant broke real-time (ms > RT_MS_CAP)  -- the real-time tier is unusable, OR
                        instant is soft (tOF > OK) AND quality is genuinely sharper (quality_tof < instant_tof).
  * else instant     (instant soft but quality is NO better -> stay on the cheaper real-time tier).

PROBE and CONFIRM use the SAME full clip so the cheap decision and the ground-truth render see the
SAME content (a late cut@28 is in view for both -> a fair test). Honest tOF = output downscaled to
LR vs decoded LR (cleanest motion truth; never NR-sharpness). GPU freed between renders. System
ffmpeg CLI is broken -> PyAV for the window trims.

Run:  python3 experiments/r4_e4_automode/validate.py
Writes -> experiments/r4_e4_automode/results.json
"""
from __future__ import annotations

import os
import sys
import json
import time

import numpy as np
import cv2
import av

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "server"))
sys.path.insert(0, os.path.join(REPO, "prototype"))

import recommend_mode as R          # noqa: E402
import pipeline_api as P            # noqa: E402
import derisk                       # noqa: E402

CLIPS = os.path.join(REPO, "experiments", "r3_e2_robustness", "clips")
SAMPLE = os.path.join(REPO, "sample.mp4")
OUT_JSON = os.path.join(HERE, "results.json")
TMP = os.path.join(HERE, "tmp")
os.makedirs(TMP, exist_ok=True)

AUTHORED = ["c1_fastpan", "c2_graphics", "c3_lowlight", "c4_talkinghead",
            "c5_scenecut", "c5b_scenecut_strong", "c6_oddres", "c7_staticcut"]

# Outcome-metric thresholds for the (probe-independent) confirmed-best rule.
INSTANT_TOF_OK = 1.0     # instant tOF at/under this = good-enough (not soft/flickery). c1=3.18,c6=1.14
                         #   fail; c2..c5b,c7 (0.36..0.90) pass.
RT_MS_CAP = 100.0        # instant ms/frame above this = real-time broken (c3 lowlight=124 via fb-collapse)
LRC_SAFE = 25.0          # layered LR-consistency-min below this = wrong-plate corruption (c7 layered=14.7)
AUTHORED_N = 40          # full authored clip (cut@28 in view for BOTH probe & confirm -> fair test)
WINDOW_N = 24            # sample.mp4 window length


def _decode_rgb(path, n=None):
    c = av.open(path)
    vs = c.streams.video[0]
    out = []
    for fr in c.decode(vs):
        if n is not None and len(out) >= n:
            break
        out.append(fr.to_ndarray(format="rgb24"))
    c.close()
    return out


def _trim_window(src, start, n, dst):
    """Copy n frames of `src` from display index `start` into a fresh libx264 clip (PyAV)."""
    cont = av.open(src)
    vs = cont.streams.video[0]
    fps = vs.average_rate
    frames = []
    for i, fr in enumerate(cont.decode(vs)):
        if i < start:
            continue
        if len(frames) >= n:
            break
        frames.append(fr.to_ndarray(format="rgb24"))
    cont.close()
    h, w = frames[0].shape[:2]
    h -= h % 2
    w -= w % 2
    out = av.open(dst, "w")
    st = out.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": "18"}
    for f in frames:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h, :w]), format="rgb24")
        for p in st.encode(vf):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
    out.close()
    return dst, len(frames)


def render_metrics(clip, mode, n, tag):
    """Render `mode` and return honest outcome metrics. Frees GPU after. Never silently swallows."""
    outp = os.path.join(TMP, f"_{tag}_{mode}.mp4")
    t0 = time.perf_counter()
    try:
        P.process_clip(clip, mode, max_frames=n, out_path=outp)
        ms = dict(P.LAST_STATS).get("ms_per_frame")
    finally:
        P.end_job()
        R._free_gpu()
    wall = time.perf_counter() - t0
    oh = _decode_rgb(outp)
    lr = _decode_rgb(clip, n)
    m = min(len(oh), len(lr))
    hl, wl = lr[0].shape[:2]
    down = [cv2.resize(f, (wl, hl), interpolation=cv2.INTER_AREA) for f in oh[:m]]
    tof = float(derisk.tof(down, lr[:m]))
    lrc = [derisk.psnr_lr_consistency(oh[i], lr[i]) for i in range(m)]
    return {"tof": round(tof, 4), "lrc_min": round(float(np.min(lrc)), 2),
            "ms_per_frame": ms, "wall_s": round(wall, 2)}


def confirmed_best(rec_mode, rmetrics):
    """Derive the best mode from OUTCOME metrics only (independent of the probe's input signals).
    Escalate to quality only when it ACTUALLY helps (else stay on the cheaper real-time tier)."""
    inst = rmetrics.get("instant", {})
    qual = rmetrics.get("quality", {})
    lay = rmetrics.get("layered")
    it = inst.get("tof", 9); ims = inst.get("ms_per_frame") or 1e9; qt = qual.get("tof", 9)
    if lay and lay.get("lrc_min", 0) >= LRC_SAFE and lay.get("tof", 9) < it:
        return "layered", (f"layered tOF {lay['tof']} < instant {it} & lrc_min {lay['lrc_min']}dB "
                           f">= {LRC_SAFE} (safe) -> static-bg win")
    if it <= INSTANT_TOF_OK and ims <= RT_MS_CAP:
        return "instant", f"instant tOF {it} <= {INSTANT_TOF_OK} & {ims}ms <= {RT_MS_CAP} -> good-enough & real-time"
    if ims > RT_MS_CAP:
        return "quality", f"instant broke real-time ({ims}ms > {RT_MS_CAP}) -> escalate to quality"
    if qt < it:
        return "quality", f"instant soft (tOF {it} > {INSTANT_TOF_OK}) & quality sharper (tOF {qt} < {it}) -> quality"
    return "instant", (f"instant soft (tOF {it}) but quality NO better (tOF {qt} >= {it}) -> stay on "
                       f"cheaper real-time instant")


def eval_clip(clip_path, tag, n, render_layered=False):
    """Probe and confirm on the SAME n frames (fair test). Always render instant+quality; render
    layered when the probe wants it OR to safety-check (render_layered)."""
    rec = R.recommend_mode(clip_path, max_frames=n, stride=1)
    sg = rec.signals
    modes = ["instant", "quality"]
    if rec.mode == "layered" or render_layered:
        modes.append("layered")
    rm = {}
    for m in modes:
        try:
            rm[m] = render_metrics(clip_path, m, n, tag)
        except Exception as e:
            rm[m] = {"error": repr(e)}
            P.end_job(); R._free_gpu()
    best, why_best = confirmed_best(rec.mode, rm)
    match = (rec.mode == best)
    return {"tag": tag, "recommended": rec.mode, "confirmed_best": best, "match": match,
            "probe_reason": rec.reason, "best_reason": why_best, "signals": sg, "render": rm}


def main():
    rows = []
    print("=== AUTHORED CLIPS (probe vs outcome-derived best) ===")
    for clip in AUTHORED:
        path = os.path.join(CLIPS, clip + ".mp4")
        if not os.path.exists(path):
            print(f"[skip] {clip}"); continue
        force_lay = (clip == "c7_staticcut")      # safety probe: does layered corrupt here?
        r = eval_clip(path, clip, AUTHORED_N, render_layered=force_lay)
        rows.append(r)
        sg = r["signals"]
        rj = {m: (v.get("tof"), v.get("lrc_min"), v.get("ms_per_frame")) for m, v in r["render"].items()}
        print(f"\n{clip}: probe={r['recommended']}  best={r['confirmed_best']}  "
              f"{'MATCH' if r['match'] else 'MISROUTE'}")
        print(f"   signals: mvMagMed={sg['mv_mag_median']}(mean {sg['mv_mag_mean']}) fb={sg['fb_react_mean']}% "
              f"edge={sg['edge_density_mean']} scenes={sg['n_scenes']} cam={sg['camera_verdict']} "
              f"chroma={sg['chroma_diff_max']} human={sg['human_coverage']} plateResid={sg['plate_resid']} "
              f"probe_s={sg['probe_s']}")
        print(f"   renders (tOF,lrc_min,ms/f): {rj}")
        print(f"   best: {r['best_reason']}")

    print("\n=== sample.mp4 WINDOWS ===")
    for tag, start in [("sample_talkinghead@5000", 5000), ("sample_window@0", 0)]:
        dst = os.path.join(TMP, tag.replace("@", "_") + ".mp4")
        clip, got = _trim_window(SAMPLE, start, WINDOW_N, dst)
        r = eval_clip(clip, tag, min(got, WINDOW_N))
        rows.append(r)
        sg = r["signals"]
        rj = {m: (v.get("tof"), v.get("lrc_min"), v.get("ms_per_frame")) for m, v in r["render"].items()}
        print(f"\n{tag} ({got}f): probe={r['recommended']}  best={r['confirmed_best']}  "
              f"{'MATCH' if r['match'] else 'MISROUTE'}")
        print(f"   signals: mvMagMed={sg['mv_mag_median']}(mean {sg['mv_mag_mean']}) fb={sg['fb_react_mean']}% "
              f"scenes={sg['n_scenes']} cam={sg['camera_verdict']} human={sg['human_coverage']} "
              f"chroma={sg['chroma_diff_max']}")
        print(f"   renders (tOF,lrc_min,ms/f): {rj}")
        print(f"   probe: {r['probe_reason']}")
        print(f"   best:  {r['best_reason']}")

    auth = [r for r in rows if r["tag"] in AUTHORED]
    nm = sum(1 for r in auth if r["match"])
    allm = sum(1 for r in rows if r["match"])
    print("\n" + "=" * 70)
    print(f"AGREEMENT authored: {nm}/{len(auth)} = {100*nm/max(1,len(auth)):.0f}%   "
          f"| all (incl. windows): {allm}/{len(rows)} = {100*allm/max(1,len(rows)):.0f}%")
    mis = [r for r in rows if not r["match"]]
    print("MISROUTES:", [f"{r['tag']}({r['recommended']}!={r['confirmed_best']})" for r in mis] or "none")
    json.dump({"rows": rows, "agreement_authored": f"{nm}/{len(auth)}",
               "agreement_all": f"{allm}/{len(rows)}"}, open(OUT_JSON, "w"), indent=2)
    print(f"results -> {OUT_JSON}")


if __name__ == "__main__":
    main()
