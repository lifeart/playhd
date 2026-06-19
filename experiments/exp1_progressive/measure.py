"""E1 measurement + byte-validity harness.

Runs the progressive fragmented-MP4 producer offline (no browser needed) and proves:
  1. Byte validity   -- the whole emitted stream re-decodes with PyAV (frame count, HD res, audio).
  2. Play-before-EOF -- the FIRST init+fragment PREFIX (a tiny fraction of the bytes) already
                        decodes to playable frames. If PyAV can decode N frames from the first K
                        bytes (K << total), a browser's <video>/MSE can start there too.
  3. TTFF            -- progressive time-to-first-fragment vs the whole-clip baseline
                        (pipeline_api.process_clip: produce-all + mux + faststart, then download).
  4. Sustain         -- produce rate vs 25 fps playback drain; required lead buffer; stall point.

Usage:
  python3 experiments/exp1_progressive/measure.py [--input PATH] [--frames N] [--producer instant|bicubic|both]
"""
import os
import sys
import io
import time
import argparse

import av

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import progressive_pipe as pp   # noqa: E402
import pipeline_api as pipe      # noqa: E402  (already on path via progressive_pipe)


def _collect(producer, input_path, src_audio_path, fps, max_frames, soft_cap, codec):
    """Run the generator to completion, collecting every byte chunk + the timing dict."""
    timing = {}
    chunks = []
    first_fragment_prefix = None
    for data in pp.stream_fragmented(producer, input_path, src_audio_path, fps,
                                     max_frames=max_frames, soft_cap=soft_cap,
                                     codec=codec, timing=timing):
        chunks.append(data)
        if first_fragment_prefix is None and timing.get("t_first_fragment") is not None:
            first_fragment_prefix = b"".join(chunks)   # bytes up to & incl. first media fragment
    return b"".join(chunks), first_fragment_prefix, timing


def _decode_count(blob):
    """Re-decode an fMP4 blob with PyAV -> (n_video_frames, has_audio, vres, vdur, adur)."""
    cont = av.open(io.BytesIO(blob))
    try:
        vs = cont.streams.video[0]
        res = f"{vs.codec_context.width}x{vs.codec_context.height}"
        has_audio = len(cont.streams.audio) > 0
        n = sum(1 for _ in cont.decode(vs))
    finally:
        cont.close()
    # second pass for durations (decode consumed the demuxer)
    cont = av.open(io.BytesIO(blob))
    try:
        vs = cont.streams.video[0]
        vdur = float(vs.duration * vs.time_base) if vs.duration else None
        adur = None
        if cont.streams.audio:
            a = cont.streams.audio[0]
            adur = float(a.duration * a.time_base) if a.duration else None
    finally:
        cont.close()
    return n, has_audio, res, vdur, adur


def _decode_prefix_frames(prefix):
    """Decode as many frames as possible from a truncated init+fragment PREFIX (proves a partial
    download is already playable). PyAV may raise at the truncation point -- that is EXPECTED; we
    count the frames it yielded BEFORE the cut and surface the error class, never swallow it."""
    n = 0
    cont = av.open(io.BytesIO(prefix))
    try:
        vs = cont.streams.video[0]
        try:
            for _ in cont.decode(vs):
                n += 1
        except (av.error.EOFError, av.error.InvalidDataError) as e:
            return n, f"clean truncation ({type(e).__name__})"
    finally:
        cont.close()
    return n, "decoded prefix fully"


def _buffer_sim(per_frame, fps_drain, fps_src):
    """Given per-frame produce wall-times, find the minimal lead buffer L (frames) so that
    playback at `fps_drain` never underruns, plus produce-rate stats. produced_time[k] is when
    frame k is ready; playback shows frame k at produced_time[L-1] + k/fps_drain."""
    n = len(per_frame)
    prod = per_frame  # cumulative wall time per frame
    # produce rates
    cold_fps = n / prod[-1] if prod[-1] > 0 else float("inf")
    # warm: drop the first 24 frames (model load + first GOP warmup) if we have enough
    warm_lo = min(24, n // 2)
    warm_span = prod[-1] - prod[warm_lo]
    warm_fps = (n - warm_lo) / warm_span if warm_span > 0 else float("inf")

    def stalls(L):
        if L > n:
            return True, None
        t_play = prod[L - 1]
        for k in range(n):
            display = t_play + k / fps_drain
            if prod[k] > display + 1e-9:
                return True, k
        return False, None

    req_L = None
    for L in range(1, n + 1):
        bad, _ = stalls(L)
        if not bad:
            req_L = L
            break
    # margin at the chosen / a representative L
    return {
        "n": n, "cold_fps": cold_fps, "warm_fps": warm_fps,
        "fps_drain": fps_drain, "fps_src": fps_src,
        "required_lead_frames": req_L,
        "required_lead_s": (req_L / fps_src) if req_L else None,
        "sustains_warm": warm_fps >= fps_drain,
    }


def run_producer(name, producer, input_path, fps, max_frames, soft_cap, codec):
    print(f"\n=== producer: {name}  (frames<= {max_frames}, soft_cap={soft_cap}, codec={codec}) ===")
    t0 = time.perf_counter()
    blob, prefix, timing = _collect(producer, input_path, input_path, fps,
                                    max_frames, soft_cap, codec)
    wall = time.perf_counter() - t0

    n_dec, has_audio, res, vdur, adur = _decode_count(blob)
    print(f"[bytes]   total={len(blob)}  emitted_frames={timing['n_frames']}  wall={wall:.2f}s")
    print(f"[decode]  re-decoded frames={n_dec}  res={res}  audio={has_audio}  "
          f"vdur={vdur}  adur={adur}  audio_note={timing['audio_note']}")
    byte_ok = (n_dec == timing["n_frames"] and has_audio)

    pre_ok = pre_note = None
    if prefix is not None:
        pre_n, pre_note = _decode_prefix_frames(prefix)
        frac = 100.0 * len(prefix) / max(1, len(blob))
        pre_ok = pre_n > 0
        print(f"[prefix]  first init+fragment = {len(prefix)} bytes ({frac:.1f}% of total) "
              f"-> decodes {pre_n} frames  [{pre_note}]")

    print(f"[ttff]    t_first_bytes(init)={timing['t_first_bytes']:.3f}s  "
          f"t_first_fragment(playable)={timing['t_first_fragment']:.3f}s  "
          f"t_end={timing['t_end']:.3f}s")

    sim = _buffer_sim(timing["per_frame"], fps_drain=25.0, fps_src=fps)
    print(f"[rate]    produce cold={sim['cold_fps']:.1f} fps  warm={sim['warm_fps']:.1f} fps  "
          f"(drain {sim['fps_drain']:.0f} fps, src {sim['fps_src']:.2f} fps)")
    print(f"[buffer]  sustains_warm={sim['sustains_warm']}  "
          f"required_lead={sim['required_lead_frames']} frames "
          f"({sim['required_lead_s']:.2f}s)" if sim['required_lead_s'] is not None
          else f"[buffer]  sustains_warm={sim['sustains_warm']}  required_lead=NONE (cannot sustain whole window)")
    return {"timing": timing, "blob_len": len(blob), "prefix_len": (len(prefix) if prefix else None),
            "n_dec": n_dec, "byte_ok": byte_ok, "prefix_ok": pre_ok, "sim": sim, "res": res,
            "vdur": vdur, "adur": adur, "wall": wall}


def run_baseline(input_path, max_frames):
    """Whole-clip baseline: process_clip produces ALL frames + mux + faststart, THEN the browser
    downloads & plays. First-frame-playable ~= server t_total (download is extra). Honest TTFF
    floor for the current path."""
    print(f"\n=== baseline: pipeline_api.process_clip (whole-clip, {max_frames} frames) ===")
    t0 = time.perf_counter()
    out = pipe.process_clip(input_path, "instant", max_frames=max_frames)
    wall = time.perf_counter() - t0
    s = pipe.LAST_STATS
    print(f"[baseline] out={os.path.basename(out)}  t_total_s={s['t_total_s']}  "
          f"ms/frame={s['ms_per_frame']}  encoder={s.get('video_encoder')}  res={s['out_resolution']}")
    print(f"[baseline] first-frame-playable >= {s['t_total_s']}s (whole file must finish + faststart)")
    return s["t_total_s"], wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(pipe.TESTDATA_DIR, "short.mp4"))
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--soft-cap", type=int, default=24)
    ap.add_argument("--producer", choices=["instant", "bicubic", "both"], default="both")
    ap.add_argument("--codec", default=None, help="force 'libx264' or 'h264_videotoolbox'")
    ap.add_argument("--baseline", action="store_true", help="also run whole-clip baseline")
    a = ap.parse_args()

    fps = pp.probe_fps(a.input)
    print(f"input={a.input}  fps={fps:.3f}  frames<= {a.frames}")

    results = {}
    if a.producer in ("bicubic", "both"):
        results["bicubic"] = run_producer(
            "bicubic (GPU-free, delivery+buffer validation)", pp.BicubicProducer(),
            a.input, fps, a.frames, a.soft_cap, a.codec)
    if a.producer in ("instant", "both"):
        results["instant"] = run_producer(
            "instant (REAL fast path)", pp.InstantProducer(),
            a.input, fps, a.frames, a.soft_cap, a.codec)

    base_ttff = None
    if a.baseline:
        base_ttff, _ = run_baseline(a.input, a.frames)

    # ---- headline comparison ----
    print("\n================ HEADLINE ================")
    if "instant" in results:
        ti = results["instant"]["timing"]
        prog_ttff = ti["t_first_fragment"]
        print(f"progressive TTFF (instant, first playable fragment): {prog_ttff:.2f}s")
        if base_ttff is not None:
            print(f"baseline   TTFF (whole-clip, {a.frames}f): {base_ttff:.2f}s  "
                  f"-> {base_ttff / prog_ttff:.1f}x faster to first frame")
        sim = results["instant"]["sim"]
        print(f"instant produce warm={sim['warm_fps']:.1f} fps vs 25 fps drain -> "
              f"{'SUSTAINS' if sim['sustains_warm'] else 'STALLS (produce < drain)'}; "
              f"lead buffer {sim['required_lead_frames']} frames")
    all_ok = all(r["byte_ok"] for r in results.values())
    print(f"byte-validity all producers: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
