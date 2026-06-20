"""R6-E2 routing verification -- does the PORTED pipeline_api.recommend_mode reproduce R4-E4?

Three checks, all CHEAP (probe only, no per-mode render):
  1. EQUIVALENCE: pipeline_api.recommend_mode (the ported product probe) returns the SAME mode AND
     the same key signals as the standalone experiments/r4_e4_automode/recommend_mode.py on every
     clip -> the port did not change behaviour.
  2. ROUTING REPRODUCTION: the ported probe's mode == R4-E4's recorded `recommended` for each clip
     (results.json) -> the same routing the R4-E4 report validated.
  3. AGREEMENT-vs-BEST: R4-E4 already derived confirmed_best from HONEST renders (tOF/lrc/ms). Since
     the ported probe == recommended (check 2), it inherits R4-E4's 9/10 agreement. We print the
     per-clip best (from results.json, a real-render outcome) alongside so the agreement is explicit.

Probe frame budgets MATCH R4-E4 (authored n=40, windows n=24) so the routing is reproduced on the
SAME content the validation saw. PyAV for window trims (system ffmpeg is broken). GPU freed between.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import av

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "server"))
sys.path.insert(0, os.path.join(REPO, "prototype"))
sys.path.insert(0, os.path.join(REPO, "experiments", "r4_e4_automode"))

import pipeline_api as P                      # noqa: E402  ported product probe under test
import recommend_mode as R4                   # noqa: E402  standalone R4-E4 reference probe

CLIPS = os.path.join(REPO, "experiments", "r3_e2_robustness", "clips")
SAMPLE = os.path.join(REPO, "sample.mp4")
R4_RESULTS = os.path.join(REPO, "experiments", "r4_e4_automode", "results.json")
TMP = os.path.join(HERE, "tmp"); os.makedirs(TMP, exist_ok=True)

AUTHORED = ["c1_fastpan", "c2_graphics", "c3_lowlight", "c4_talkinghead",
            "c5_scenecut", "c5b_scenecut_strong", "c6_oddres", "c7_staticcut"]
AUTHORED_N, WINDOW_N = 40, 24
SIG_KEYS = ["mv_mag_median", "mv_mag_mean", "fb_react_mean", "n_scenes",
            "camera_verdict", "chroma_diff_max", "human_coverage", "plate_resid"]


def trim_window(src, start, n, dst):
    cont = av.open(src); vs = cont.streams.video[0]; fps = vs.average_rate
    frames = []
    for i, fr in enumerate(cont.decode(vs)):
        if i < start:
            continue
        if len(frames) >= n:
            break
        frames.append(fr.to_ndarray(format="rgb24"))
    cont.close()
    h, w = frames[0].shape[:2]; h -= h % 2; w -= w % 2
    out = av.open(dst, "w"); st = out.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"; st.options = {"crf": "18"}
    for f in frames:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h, :w]), format="rgb24")
        for p in st.encode(vf):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
    out.close()
    return dst, len(frames)


def _sig_close(a, b):
    """Compare the routing-relevant signals between the two probes (floats within 1e-6)."""
    for k in SIG_KEYS:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, float) and isinstance(vb, float):
            if not (np.isnan(va) and np.isnan(vb)) and abs(va - vb) > 1e-6:
                return False, k
        elif va != vb:
            return False, k
    return True, None


def probe_pair(path, n, tag):
    t0 = time.perf_counter()
    rec_new = P.recommend_mode(path, max_frames=n, stride=1)   # PORTED (product) probe
    t_new = time.perf_counter() - t0
    P._free_gpu()
    rec_old = R4.recommend_mode(path, max_frames=n, stride=1)  # standalone reference probe
    R4._free_gpu()
    same_mode = rec_new.mode == rec_old.mode
    sig_ok, sig_bad = _sig_close(rec_new.signals, rec_old.signals)
    return {"tag": tag, "ported_mode": rec_new.mode, "ref_mode": rec_old.mode,
            "mode_equal": same_mode, "sig_equal": sig_ok, "sig_mismatch_key": sig_bad,
            "probe_s_ported": round(t_new, 2),
            "signals": {k: rec_new.signals.get(k) for k in SIG_KEYS},
            "reason": rec_new.reason}


def main():
    r4 = json.load(open(R4_RESULTS))
    recorded = {row["tag"]: (row["recommended"], row["confirmed_best"], row["match"])
                for row in r4["rows"]}
    rows = []

    print("=== AUTHORED CLIPS ===")
    for clip in AUTHORED:
        path = os.path.join(CLIPS, clip + ".mp4")
        if not os.path.exists(path):
            print("[skip]", clip); continue
        rec, best, match = recorded.get(clip, ("?", "?", None))
        row = probe_pair(path, AUTHORED_N, clip)
        row["r4_recommended"], row["r4_best"], row["r4_match"] = rec, best, match
        row["routing_reproduced"] = (row["ported_mode"] == rec)
        rows.append(row)
        print(f"{clip:24s} ported={row['ported_mode']:8s} ref={row['ref_mode']:8s} "
              f"R4_rec={rec:8s} R4_best={best:8s} "
              f"eq={'Y' if row['mode_equal'] else 'N'} sig={'Y' if row['sig_equal'] else 'N'} "
              f"repro={'Y' if row['routing_reproduced'] else 'N'} probe={row['probe_s_ported']}s")

    print("\n=== sample.mp4 WINDOWS ===")
    for tag, start in [("sample_talkinghead@5000", 5000), ("sample_window@0", 0)]:
        dst = os.path.join(TMP, tag.replace("@", "_") + ".mp4")
        clip, got = trim_window(SAMPLE, start, WINDOW_N, dst)
        rec, best, match = recorded.get(tag, ("?", "?", None))
        row = probe_pair(clip, min(got, WINDOW_N), tag)
        row["r4_recommended"], row["r4_best"], row["r4_match"] = rec, best, match
        row["routing_reproduced"] = (row["ported_mode"] == rec)
        rows.append(row)
        print(f"{tag:24s} ported={row['ported_mode']:8s} ref={row['ref_mode']:8s} "
              f"R4_rec={rec:8s} R4_best={best:8s} "
              f"eq={'Y' if row['mode_equal'] else 'N'} sig={'Y' if row['sig_equal'] else 'N'} "
              f"repro={'Y' if row['routing_reproduced'] else 'N'} probe={row['probe_s_ported']}s")

    n = len(rows)
    eq = sum(1 for r in rows if r["mode_equal"])
    sigeq = sum(1 for r in rows if r["sig_equal"])
    repro = sum(1 for r in rows if r["routing_reproduced"])
    # agreement-vs-best: ported mode == R4 confirmed_best (a real-render outcome)
    agree = sum(1 for r in rows if r["ported_mode"] == r["r4_best"])
    print("\n" + "=" * 78)
    print(f"port-equivalence (mode): {eq}/{n}    (signals): {sigeq}/{n}")
    print(f"routing reproduced (ported==R4_recommended): {repro}/{n}")
    print(f"AGREEMENT ported-vs-honest-best (ported==R4_confirmed_best): {agree}/{n}")
    mis = [r['tag'] for r in rows if r['ported_mode'] != r['r4_best']]
    print("disagreements vs best:", mis or "none")
    json.dump({"rows": rows, "port_equiv_mode": f"{eq}/{n}", "port_equiv_sig": f"{sigeq}/{n}",
               "routing_reproduced": f"{repro}/{n}", "agreement_vs_best": f"{agree}/{n}"},
              open(os.path.join(HERE, "routing_results.json"), "w"), indent=2)
    print("-> routing_results.json")


if __name__ == "__main__":
    main()
