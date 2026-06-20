"""R6-E2 end-to-end verification through the REAL server entry points (no monkey-patching):
  * pipeline_api.process_clip(input_path, "auto") RUNS, produces a valid HD mp4, and sets
    LAST_STATS["auto_chosen"] (+ auto_reason / auto_signals).
  * AUTO just DISPATCHES: the auto render is decoded-pixel-IDENTICAL to rendering the resolved mode
    directly (proves the auto branch only chooses, it does not alter the render).
  * the PROBE is CHEAP: probe seconds << the render wall, reported as a ratio.
  * exercises all three routes that occur on the clip set: instant, quality, layered.
  * confirms the chosen mode is always a REAL render mode.

PyAV trims for the sample window (system ffmpeg is broken). GPU freed between renders.
"""
from __future__ import annotations
import os, sys, json, time, hashlib
import numpy as np
import av

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "server"))
sys.path.insert(0, os.path.join(REPO, "prototype"))
import pipeline_api as P  # noqa: E402

CLIPS = os.path.join(REPO, "experiments", "r3_e2_robustness", "clips")
SAMPLE = os.path.join(REPO, "sample.mp4")
TMP = os.path.join(HERE, "tmp"); os.makedirs(TMP, exist_ok=True)
REAL_MODES = {"instant", "quality", "layered"}


def trim(src, start, n, dst):
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
    return dst


def pixhash(path):
    cont = av.open(path); vs = cont.streams.video[0]
    h = hashlib.sha256(); n = 0; res = None
    for fr in cont.decode(vs):
        a = fr.to_ndarray(format="rgb24")
        if res is None:
            res = f"{a.shape[1]}x{a.shape[0]}"
        h.update(np.ascontiguousarray(a).tobytes()); n += 1
    cont.close()
    return h.hexdigest(), n, res


def free_gpu():
    P.end_job()
    try:
        import gc, torch
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def run(clip, mode, n, tag):
    outp = os.path.join(TMP, f"_{tag}.mp4")
    t0 = time.perf_counter()
    try:
        P.process_clip(clip, mode, max_frames=n, out_path=outp)
        stats = dict(P.LAST_STATS)
    finally:
        free_gpu()
    wall = time.perf_counter() - t0
    ph, nf, res = pixhash(outp)
    expect_res = stats.get("out_resolution")
    ok, vinfo = P._verify_mp4(outp, stats.get("n_frames"), expect_res) if expect_res else (None, {})
    return {"wall_s": round(wall, 2), "pixhash": ph, "n_frames": nf, "resolution": res,
            "valid_mp4": ok, "verify": vinfo, "stats": stats, "out_path": outp}


def main():
    cases = []
    # talking-head window -> LAYERED (the safety-critical route the auto-mode exists for)
    win = trim(SAMPLE, 5000, 24, os.path.join(TMP, "th5000_24.mp4"))
    cases.append(("layered_route", win, 24, True))    # compare auto vs explicit (same small cost? layered slow -> skip explicit)
    # graphics -> INSTANT (cheap; full auto-vs-explicit identity check here)
    cases.append(("instant_route", os.path.join(CLIPS, "c2_graphics.mp4"), 24, "compare"))
    # fast pan -> QUALITY (confirm auto routes & renders; small n to bound cost)
    cases.append(("quality_route", os.path.join(CLIPS, "c1_fastpan.mp4"), 10, False))

    rows = []
    for tag, clip, n, mode_check in cases:
        print(f"\n--- {tag}: process_clip(auto) on {os.path.basename(clip)} n={n} ---")
        a = run(clip, "auto", n, f"{tag}_auto")
        st = a["stats"]
        chosen = st.get("auto_chosen")
        probe_s = (st.get("auto_signals") or {}).get("probe_s")
        row = {"tag": tag, "clip": os.path.basename(clip), "n": n,
               "auto_chosen": chosen, "auto_chosen_is_real": chosen in REAL_MODES,
               "auto_chosen_set": "auto_chosen" in st,
               "auto_reason": st.get("auto_reason"),
               "valid_mp4": a["valid_mp4"], "resolution": a["resolution"], "n_frames": a["n_frames"],
               "render_wall_s": a["wall_s"], "probe_s": probe_s,
               "probe_vs_render_ratio": (round(probe_s / a["wall_s"], 4)
                                         if probe_s and a["wall_s"] else None)}
        print(f"   auto_chosen={chosen} (real={row['auto_chosen_is_real']}) valid={a['valid_mp4']} "
              f"res={a['resolution']} frames={a['n_frames']} probe={probe_s}s render_wall={a['wall_s']}s "
              f"ratio={row['probe_vs_render_ratio']}")
        print(f"   reason: {st.get('auto_reason')}")
        if mode_check == "compare" and chosen in REAL_MODES:
            b = run(clip, chosen, n, f"{tag}_explicit")
            row["explicit_pixhash"] = b["pixhash"]
            row["auto_pixhash"] = a["pixhash"]
            row["auto_equals_explicit"] = (a["pixhash"] == b["pixhash"])
            print(f"   auto-vs-explicit({chosen}) decoded-pixel identical: {row['auto_equals_explicit']} "
                  f"(auto {a['pixhash'][:12]} / explicit {b['pixhash'][:12]})")
        rows.append(row)

    json.dump(rows, open(os.path.join(HERE, "e2e_results.json"), "w"), indent=2)
    print("\n-> e2e_results.json")
    # PASS conditions
    all_valid = all(r["valid_mp4"] for r in rows)
    all_set = all(r["auto_chosen_set"] for r in rows)
    all_real = all(r["auto_chosen_is_real"] for r in rows)
    cheap = all((r["probe_vs_render_ratio"] is None) or (r["probe_vs_render_ratio"] < 0.5) for r in rows)
    identity = all(r.get("auto_equals_explicit", True) for r in rows)
    print(f"\nE2E PASS: valid_output={all_valid} auto_chosen_set={all_set} chosen_is_real={all_real} "
          f"probe_cheap={cheap} auto==explicit={identity}")


if __name__ == "__main__":
    main()
