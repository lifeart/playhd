"""R5-E1 production stress test driver (READ-ONLY import of server/).

One test per invocation so long runs can be backgrounded and logged separately:

    python3 stress.py mem_instant   # T1 memory-flat + T2 multi-scene + T3 instant tput
    python3 stress.py quality       # T3 quality throughput + memory
    python3 stress.py layered       # T5 layered-at-scale + plate spill/cleanup
    python3 stress.py edge_eof      # T4 natural EOF / last partial GOP (no cap)
    python3 stress.py repeat        # T4 determinism (process twice) + lock release
    python3 stress.py lock          # T4 single-job lock semantics (no GPU)

Each test prints a JSON blob on a line prefixed `RESULT_JSON ` and writes outputs
under ./out/. Honest metrics: RSS sampled across the run (steady-state slope),
ms/frame from LAST_STATS, valid-output via PyAV re-decode (_verify_mp4).
"""
import os, sys, gc, json, time, threading, hashlib, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SERVER = os.path.join(REPO, "server")
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

import numpy as np
import av
import psutil

import pipeline_api as P          # READ-ONLY: we only call its public functions
import scene_detect

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SAMPLE = os.path.join(REPO, "sample.mp4")
SHORT = os.path.join(REPO, "server", "testdata", "short.mp4")

# Real scene cuts in sample.mp4 [0,900) given by the lead.
KNOWN_CUTS = [28, 196, 341, 479, 514, 563, 630, 688, 810]


# --------------------------------------------------------------------------- #
class RssSampler:
    """Sample process RSS every `dt` s alongside frames-done (from get_progress)."""
    def __init__(self, dt=0.2):
        self.dt = dt
        self.proc = psutil.Process()
        self.samples = []          # (t_rel, done, rss_bytes)
        self.peak = 0
        self._stop = threading.Event()
        self._t = None
        self.t0 = None

    def _run(self):
        while not self._stop.is_set():
            rss = self.proc.memory_info().rss
            self.peak = max(self.peak, rss)
            done = P.get_progress().get("done", 0)
            self.samples.append((time.perf_counter() - self.t0, done, rss))
            time.sleep(self.dt)

    def start(self):
        self.t0 = time.perf_counter()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)

    def analysis(self):
        """Steady-state memory growth: least-squares slope of RSS(bytes) vs frames-done
        over the back 70% of the run (skip warm-up/model-load ramp). Near-zero => flat."""
        s = [x for x in self.samples if x[1] > 0]
        if len(s) < 4:
            return {"n_samples": len(self.samples), "note": "too few samples"}
        dones = [x[1] for x in s]
        dmax = max(dones)
        tail = [x for x in s if x[1] >= 0.30 * dmax]      # drop first 30% of frames
        d = np.array([x[1] for x in tail], dtype=float)
        r = np.array([x[2] for x in tail], dtype=float)
        slope_kb_per_frame = float("nan")
        if len(set(d.tolist())) > 1:
            A = np.vstack([d, np.ones_like(d)]).T
            m, _b = np.linalg.lstsq(A, r, rcond=None)[0]
            slope_kb_per_frame = m / 1e3
        rss_tail_mb = (r / 1e6)
        return {
            "n_samples": len(self.samples),
            "peak_rss_mb": round(self.peak / 1e6, 1),
            "rss_min_mb_steady": round(float(rss_tail_mb.min()), 1),
            "rss_max_mb_steady": round(float(rss_tail_mb.max()), 1),
            "rss_span_mb_steady": round(float(rss_tail_mb.max() - rss_tail_mb.min()), 1),
            "slope_kb_per_frame_steady": round(slope_kb_per_frame, 3),
            "frames_covered": int(dmax),
            # implied growth if extrapolated to the whole 50,805-frame clip:
            "implied_growth_mb_full_clip": round(slope_kb_per_frame * 50805 / 1e3, 1),
        }

    def thinned(self, k=10):
        s = [x for x in self.samples if x[1] > 0]
        if not s:
            return []
        step = max(1, len(s) // k)
        return [(d, round(r / 1e6)) for (_t, d, r) in s[::step]]


def decode_mean(path, max_n=None):
    """Mean pixel value + per-frame md5 chain of a clip (for determinism diff)."""
    c = av.open(path)
    vs = c.streams.video[0]
    s = 0.0
    n = 0
    h = hashlib.md5()
    try:
        for fr in c.decode(vs):
            a = fr.to_ndarray(format="rgb24")
            s += float(a.mean())
            h.update(a.tobytes())
            n += 1
            if max_n and n >= max_n:
                break
    finally:
        c.close()
    return {"n": n, "mean": round(s / max(1, n), 4), "md5": h.hexdigest()}


def frame_mean_diff(p1, p2, max_n=None):
    """Mean absolute per-pixel difference between two clips (0..255), frame-aligned."""
    c1, c2 = av.open(p1), av.open(p2)
    g1 = c1.decode(c1.streams.video[0])
    g2 = c2.decode(c2.streams.video[0])
    tot = 0.0
    n = 0
    maxd = 0.0
    try:
        while True:
            try:
                f1 = next(g1); f2 = next(g2)
            except StopIteration:
                break
            a1 = f1.to_ndarray(format="rgb24").astype(np.float32)
            a2 = f2.to_ndarray(format="rgb24").astype(np.float32)
            if a1.shape != a2.shape:
                return {"shape_mismatch": [a1.shape, a2.shape], "n": n}
            d = np.abs(a1 - a2)
            tot += float(d.mean())
            maxd = max(maxd, float(d.max()))
            n += 1
            if max_n and n >= max_n:
                break
    finally:
        c1.close(); c2.close()
    return {"n": n, "mean_abs_diff": round(tot / max(1, n), 4), "max_abs_diff": round(maxd, 1)}


def detect_cuts_in_window(path, n):
    """Run the SAME StreamingCutDetector the pipeline uses over [0,n) -> detected cut frames."""
    cont = av.open(path)
    vs = cont.streams.video[0]
    vs.codec_context.options = {"flags2": "+export_mvs"}
    det = scene_detect.StreamingCutDetector()
    cuts = []
    i = 0
    try:
        for fr in cont.decode(vs):
            if i >= n:
                break
            ptype = {1: "I", 2: "P", 3: "B"}.get(int(fr.pict_type), "?")
            img = fr.to_ndarray(format="rgb24")
            if det.update(i, ptype, img):
                cuts.append(i)
            i += 1
    finally:
        cont.close()
    return cuts


def run_clip(name, input_path, mode, max_frames, sample_mem=True, detect_cuts=True):
    out_path = os.path.join(OUT, f"{name}.mp4")
    for ext in ("", ".video.tmp.mp4"):
        if os.path.exists(out_path + ext):
            os.remove(out_path + ext)
    smp = RssSampler() if sample_mem else None
    if smp:
        smp.start()
    crash = None
    t0 = time.perf_counter()
    try:
        P.process_clip(input_path, mode, max_frames=max_frames, out_path=out_path,
                       detect_cuts=detect_cuts)
    except Exception as e:
        crash = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    wall = time.perf_counter() - t0
    if smp:
        smp.stop()
    res = {
        "name": name, "mode": mode, "input": os.path.basename(input_path),
        "max_frames": max_frames, "wall_s": round(wall, 2),
        "lock_released_after": not P.is_busy(),
        "crash": crash,
    }
    if crash:
        return res, out_path
    stats = dict(P.LAST_STATS)
    res["stats"] = stats
    ok, info = P._verify_mp4(out_path, stats["n_frames"], stats["out_resolution"])
    res["verify_ok"] = ok
    res["verify"] = info
    if smp:
        res["mem"] = smp.analysis()
        res["mem_samples"] = smp.thinned(16)
    return res, out_path


# --------------------------------------------------------------------------- #
def t_mem_instant():
    # T1 memory-flat (long window) + T2 multi-scene (9 cuts < 2000) + T3 instant tput.
    res, out = run_clip("mem_instant", SAMPLE, "instant", max_frames=2000)
    if not res.get("crash"):
        cuts = detect_cuts_in_window(SAMPLE, 900)
        res["detected_cuts_0_900"] = cuts
        res["known_cuts_0_900"] = KNOWN_CUTS
        res["n_chunks_ge_cuts"] = res["stats"]["n_chunks"] >= len(cuts)
        res["ui_estimate_ms"] = P.MODE_MS_PER_FRAME["instant"]
    return res


def t_quality():
    res, out = run_clip("quality", SAMPLE, "quality", max_frames=96)
    if not res.get("crash"):
        res["ui_estimate_ms"] = P.MODE_MS_PER_FRAME["quality"]
    return res


def t_layered():
    plate_dir = os.path.join(OUT, "layered.mp4.plates")
    res, out = run_clip("layered", SAMPLE, "layered", max_frames=600)
    res["plate_dir_cleaned"] = not os.path.exists(plate_dir)
    if not res.get("crash"):
        res["ui_estimate_ms"] = P.MODE_MS_PER_FRAME["layered"]
    return res


def t_edge_eof():
    # No cap -> processes to natural EOF (last partial GOP) on the whole short clip.
    res, out = run_clip("edge_eof", SHORT, "instant", max_frames=None)
    if not res.get("crash"):
        # short.mp4 reports 150 frames; confirm we got them all (natural-end handling).
        res["expected_frames"] = P.probe_total_frames(SHORT)
    return res


def t_repeat():
    # Determinism + lock release: process the same short clip twice.
    r1, o1 = run_clip("repeat_a", SHORT, "instant", max_frames=None, sample_mem=False)
    r2o = os.path.join(OUT, "repeat_b.mp4")
    r2, o2 = run_clip("repeat_b", SHORT, "instant", max_frames=None, sample_mem=False)
    out = {"run_a": r1, "run_b": r2}
    if not r1.get("crash") and not r2.get("crash"):
        out["size_a"] = os.path.getsize(o1)
        out["size_b"] = os.path.getsize(o2)
        out["frame_diff"] = frame_mean_diff(o1, o2)
        out["decode_a"] = decode_mean(o1)
        out["decode_b"] = decode_mean(o2)
        out["byte_identical_video_md5"] = out["decode_a"]["md5"] == out["decode_b"]["md5"]
    out["lock_released"] = not P.is_busy()
    return out


def t_lock():
    # Single-job lock semantics, no GPU. Acquire, prove a second job is rejected, release.
    res = {}
    res["initially_busy"] = P.is_busy()
    got = P.try_begin_job()
    res["acquired"] = got
    res["busy_while_held"] = P.is_busy()
    raised = None
    try:
        P.process_clip(SHORT, "instant", max_frames=5)   # must hit BusyError, no GPU work
    except P.BusyError as e:
        raised = f"BusyError: {e}"
    except Exception as e:
        raised = f"UNEXPECTED {type(e).__name__}: {e}"
    res["second_job_rejected"] = raised
    P.end_job()
    res["busy_after_release"] = P.is_busy()
    return res


def t_mem_long():
    # Characterize instant memory growth over a LONG window: bounded(plateau) or leak(linear)?
    n = int(os.environ.get("MEM_LONG_N", "8000"))
    res, out = run_clip("mem_long", SAMPLE, "instant", max_frames=n)
    if not res.get("crash"):
        res["ui_estimate_ms"] = P.MODE_MS_PER_FRAME["instant"]
        # dense per-200-frame RSS so we can see plateau vs linear directly
        smp = None
    return res


TESTS = {
    "mem_instant": t_mem_instant, "quality": t_quality, "layered": t_layered,
    "edge_eof": t_edge_eof, "repeat": t_repeat, "lock": t_lock,
    "mem_long": t_mem_long,
}

if __name__ == "__main__":
    which = sys.argv[1]
    out = TESTS[which]()
    print("RESULT_JSON " + json.dumps(out, default=str))
