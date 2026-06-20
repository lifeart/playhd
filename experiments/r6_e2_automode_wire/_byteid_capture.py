"""Capture decoded-pixel fingerprints of a non-auto render, BEFORE and AFTER the auto wiring,
to prove a non-"auto" mode is unchanged. Encoders (HW VideoToolbox for instant; libx264 for
quality) may not be byte-deterministic, so we fingerprint the DECODED pixels (the actual image
the user sees), not the container bytes. We also double-run each mode to report encoder
determinism so the before/after comparison is interpreted correctly.

Usage:
  python3 _byteid_capture.py before   # run against unmodified pipeline_api -> before.json
  python3 _byteid_capture.py after    # run against wired pipeline_api    -> after.json
"""
import os, sys, json, time, hashlib
import numpy as np
import av

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "server"))
sys.path.insert(0, os.path.join(REPO, "prototype"))
import pipeline_api as P  # noqa: E402

SAMPLE = os.path.join(REPO, "sample.mp4")
TMP = os.path.join(HERE, "tmp"); os.makedirs(TMP, exist_ok=True)
WIN_START, WIN_N = 5000, 12


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


def fingerprint(path):
    """SHA over every decoded frame's pixels (order-sensitive) + frame count + resolution."""
    cont = av.open(path); vs = cont.streams.video[0]
    h = hashlib.sha256(); n = 0; res = None
    for fr in cont.decode(vs):
        a = fr.to_ndarray(format="rgb24")
        if res is None:
            res = f"{a.shape[1]}x{a.shape[0]}"
        h.update(np.ascontiguousarray(a).tobytes()); n += 1
    cont.close()
    return {"pixhash": h.hexdigest(), "n_frames": n, "resolution": res}


def render(mode, clip, tag):
    outp = os.path.join(TMP, f"_{tag}.mp4")
    t0 = time.perf_counter()
    try:
        P.process_clip(clip, mode, max_frames=WIN_N, out_path=outp)
        ms = dict(P.LAST_STATS).get("ms_per_frame")
    finally:
        P.end_job()
        try:
            import gc, torch
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
    return fingerprint(outp) | {"ms_per_frame": ms, "wall_s": round(time.perf_counter() - t0, 2)}


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "before"
    clip = trim(SAMPLE, WIN_START, WIN_N, os.path.join(TMP, "byteid_win.mp4"))
    res = {}
    for mode in ("instant", "quality"):
        a = render(mode, clip, f"{phase}_{mode}_a")
        b = render(mode, clip, f"{phase}_{mode}_b")          # double-run -> encoder determinism
        res[mode] = {"runA": a, "runB": b,
                     "deterministic": a["pixhash"] == b["pixhash"]}
        print(f"[{phase}] {mode}: det={res[mode]['deterministic']} "
              f"hashA={a['pixhash'][:12]} hashB={b['pixhash'][:12]} "
              f"n={a['n_frames']} res={a['resolution']} ms={a['ms_per_frame']}")
    json.dump(res, open(os.path.join(HERE, f"byteid_{phase}.json"), "w"), indent=2)
    print(f"-> byteid_{phase}.json")


if __name__ == "__main__":
    main()
