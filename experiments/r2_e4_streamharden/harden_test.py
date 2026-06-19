"""R2-E4: harden server/progressive.py's progressive-playback (fragmented-MP4) feature.

Imports server/ READ-ONLY (progressive.iter_fragments + pipeline_api job lock), drives the SAME
entrypoint the HTTP endpoint drives, and DECODES the produced fMP4 with PyAV to ASSERT correctness.
NO browser, NO system ffmpeg/ffprobe (both unavailable/broken) -- PyAV mux+decode only.

Cases:
  A  non-AAC transcode (mp3)  -> output has valid AAC track, dur ~= video dur, A/V sync start->end
  A2 non-AAC transcode (opus) -> same (bonus second non-AAC codec)
  B  long-clip bound          -> sample.mp4 capped at 600 frames bicubic: fragments flow + RSS flat
  C  video-only               -> audio stripped: streams clean, audio_note == none
  D  AAC copy (control)       -> baseline sync on the untouched copy path
"""
import os
import sys
import time
import gc

import av
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SERVER = os.path.join(REPO, "server")
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

import pipeline_api as pipe          # READ-ONLY: try_begin_job/end_job
import progressive as prog           # READ-ONLY: iter_fragments

PROC = psutil.Process(os.getpid())


def rss_mb():
    return PROC.memory_info().rss / 1e6


# --------------------------------------------------------------------------- #
# Drive iter_fragments EXACTLY like server/app.py: try_begin_job() before, then
# gen.close() + end_job() in a finally. Write fMP4 to disk; sample RSS + fragment timing.
# --------------------------------------------------------------------------- #
def run_stream(input_path, out_path, *, mode="bicubic", max_frames=None, rss_sample_every=25):
    timing = {}
    got_lock = pipe.try_begin_job()
    assert got_lock, "could not acquire single-job lock (something else holds it)"
    rss_samples = []          # (frame_idx_approx, rss_mb) sampled across the stream
    frag_walltimes = []       # wall time of each yielded chunk -> detect stalls
    n_chunks = 0
    bytes_total = 0
    gen = prog.iter_fragments(input_path, mode, max_frames=max_frames, timing=timing)
    t0 = time.perf_counter()
    last = t0
    try:
        with open(out_path, "wb") as f:
            for chunk in gen:
                now = time.perf_counter()
                frag_walltimes.append(now - last)
                last = now
                f.write(chunk)
                n_chunks += 1
                bytes_total += len(chunk)
                nf = timing.get("n_frames", 0)
                if nf and (nf % rss_sample_every == 0):
                    rss_samples.append((nf, rss_mb()))
    finally:
        try:
            gen.close()
        except Exception as e:
            print(f"[run] gen.close error: {type(e).__name__}: {e}")
        pipe.end_job()
    assert not pipe.is_busy(), "job lock NOT released after stream"
    return {
        "timing": timing,
        "rss_samples": rss_samples,
        "frag_walltimes": frag_walltimes,
        "n_chunks": n_chunks,
        "bytes_total": bytes_total,
        "wall_s": time.perf_counter() - t0,
        "out_path": out_path,
    }


# --------------------------------------------------------------------------- #
# Decode the produced fMP4 and extract per-stream timelines (frame PTS in seconds).
# --------------------------------------------------------------------------- #
def decode_timeline(path):
    c = av.open(path)
    vinfo = {"n": 0, "first": None, "last": None, "end": None, "codec": None, "fps": None}
    ainfo = {"n": 0, "first": None, "last": None, "end": None, "codec": None,
             "rate": None, "max_gap_ms": 0.0, "monotonic": True}
    vs = c.streams.video[0]
    vinfo["codec"] = vs.codec_context.name
    vfps = float(vs.average_rate or vs.guessed_rate or 25)
    vinfo["fps"] = vfps
    vinfo["decode_err"] = None
    vpts = []
    try:
        for fr in c.decode(vs):
            if fr.pts is None:
                continue
            vpts.append(float(fr.pts * fr.time_base))
    except av.error.InvalidDataError as e:   # surface (e.g. mid-GOP max_frames cap tail), don't crash
        vinfo["decode_err"] = f"{type(e).__name__}: {e}"
    vpts.sort()
    if vpts:
        vinfo["n"] = len(vpts)
        vinfo["first"] = vpts[0]
        vinfo["last"] = vpts[-1]
        vinfo["end"] = vpts[-1] + 1.0 / vfps

    if c.streams.audio:
        c.close()
        c = av.open(path)              # fresh demux for the audio stream
        as_ = c.streams.audio[0]
        ainfo["codec"] = as_.codec_context.name
        ainfo["rate"] = as_.codec_context.sample_rate
        ainfo["decode_err"] = None
        apts = []
        try:
            for fr in c.decode(as_):
                if fr.pts is None:
                    continue
                t = float(fr.pts * fr.time_base)
                dur = fr.samples / float(as_.codec_context.sample_rate)
                apts.append((t, dur))
        except av.error.InvalidDataError as e:
            ainfo["decode_err"] = f"{type(e).__name__}: {e}"
        apts.sort()
        if apts:
            ainfo["n"] = len(apts)
            ainfo["first"] = apts[0][0]
            ainfo["last"] = apts[-1][0]
            ainfo["end"] = apts[-1][0] + apts[-1][1]
            prev = None
            for t, _d in apts:
                if prev is not None:
                    if t < prev - 1e-6:
                        ainfo["monotonic"] = False
                    gap = (t - prev) * 1000.0
                    ainfo["max_gap_ms"] = max(ainfo["max_gap_ms"], gap)
                prev = t
    c.close()
    return vinfo, ainfo


def sync_report(vinfo, ainfo):
    """Return (head_drift_ms, tail_drift_ms, max_drift_ms) of audio vs video timeline."""
    if ainfo["n"] == 0 or vinfo["n"] == 0:
        return None
    head = abs((ainfo["first"] or 0.0) - (vinfo["first"] or 0.0)) * 1000.0
    tail = abs((ainfo["end"] or 0.0) - (vinfo["end"] or 0.0)) * 1000.0
    return head, tail, max(head, tail)


def fmt_ms(x):
    return f"{x:.1f}ms" if x is not None else "n/a"


# --------------------------------------------------------------------------- #
def case_transcode(label, clip, expect_src_codec):
    print(f"\n===== CASE {label}: NON-AAC TRANSCODE ({clip}) =====")
    out = os.path.join(HERE, f"out_{label}.mp4")
    r = run_stream(os.path.join(HERE, clip), out, mode="bicubic", max_frames=None)
    note = r["timing"].get("audio_note")
    vinfo, ainfo = decode_timeline(out)
    s = sync_report(vinfo, ainfo)
    print(f"  audio_note               : {note}")
    print(f"  produced video frames    : {r['timing'].get('n_frames')}")
    print(f"  OUT video  : codec={vinfo['codec']} n={vinfo['n']} end={vinfo['end']:.3f}s")
    print(f"  OUT audio  : codec={ainfo['codec']} n={ainfo['n']} "
          f"end={ainfo['end'] and round(ainfo['end'],3)}s rate={ainfo['rate']} "
          f"monotonic={ainfo['monotonic']} max_gap={fmt_ms(ainfo['max_gap_ms'])}")
    aac_ok = (ainfo["codec"] == "aac")
    dur_ok = ainfo["end"] is not None and abs(ainfo["end"] - vinfo["end"]) < 0.20
    head, tail, mx = s
    print(f"  AAC track present?       : {aac_ok}")
    print(f"  audio_end ~= video_end?  : {dur_ok}  "
          f"(|{ainfo['end']:.3f}-{vinfo['end']:.3f}|={abs(ainfo['end']-vinfo['end'])*1000:.0f}ms)")
    print(f"  A/V drift head/tail/max  : {fmt_ms(head)} / {fmt_ms(tail)} / {fmt_ms(mx)}")
    p = aac_ok and dur_ok and mx < 200.0 and ainfo["monotonic"]
    print(f"  --> {'PASS' if p else 'FAIL'}")
    return {"label": label, "note": note, "aac_ok": aac_ok, "dur_ok": dur_ok,
            "drift_ms": mx, "pass": p, "vinfo": vinfo, "ainfo": ainfo,
            "bytes": r["bytes_total"]}


def case_longclip(clip, max_frames=600):
    print(f"\n===== CASE B: LONG-CLIP BOUND ({clip}, max_frames={max_frames}) =====")
    out = os.path.join(HERE, "out_B_long.mp4")
    gc.collect()
    rss0 = rss_mb()
    r = run_stream(os.path.join(HERE, clip) if not os.path.isabs(clip) else clip,
                   out, mode="bicubic", max_frames=max_frames)
    nf = r["timing"].get("n_frames")
    samples = r["rss_samples"]
    walt = r["frag_walltimes"]
    max_gap = max(walt) if walt else 0.0
    rss_lo = min(v for _, v in samples) if samples else rss_mb()
    rss_hi = max(v for _, v in samples) if samples else rss_mb()
    # growth measured as last-quartile mean minus first-quartile mean
    if len(samples) >= 4:
        q = max(1, len(samples) // 4)
        first_q = sum(v for _, v in samples[:q]) / q
        last_q = sum(v for _, v in samples[-q:]) / q
        growth = last_q - first_q
    else:
        growth = rss_hi - rss_lo
    vinfo, ainfo = decode_timeline(out)
    s = sync_report(vinfo, ainfo)
    print(f"  frames produced          : {nf}  (requested cap {max_frames})")
    print(f"  fragment chunks yielded  : {r['n_chunks']}  bytes={r['bytes_total']/1e6:.1f}MB")
    print(f"  max inter-chunk gap      : {max_gap*1000:.0f}ms (stall check)")
    print(f"  RSS start={rss0:.0f}MB  lo={rss_lo:.0f}MB hi={rss_hi:.0f}MB  "
          f"q1->q4 growth={growth:+.0f}MB")
    print(f"  OUT video end            : {vinfo['end'] and round(vinfo['end'],2)}s "
          f"({vinfo['n']} frames)")
    print(f"  OUT audio end            : {ainfo['end'] and round(ainfo['end'],2)}s "
          f"({ainfo['n']} frames, codec={ainfo['codec']})")
    if s:
        head, tail, mx = s
        print(f"  A/V drift head/tail/max  : {fmt_ms(head)} / {fmt_ms(tail)} / {fmt_ms(mx)}")
    frames_ok = nf == max_frames
    flat_ok = growth < 80.0           # < 80MB drift across 600 frames == bounded
    flow_ok = max_gap < 5.0           # no multi-second stall between fragments
    print(f"  frames==cap? {frames_ok}  RSS flat(<80MB)? {flat_ok}  no-stall(<5s)? {flow_ok}")
    p = frames_ok and flat_ok and flow_ok
    print(f"  --> streaming/bound: {'PASS' if p else 'FAIL'}")
    return {"label": "B", "nf": nf, "growth": growth, "max_gap_ms": max_gap * 1000,
            "vinfo": vinfo, "ainfo": ainfo, "pass": p, "sync": s,
            "bytes": r["bytes_total"]}


def case_video_only(clip):
    print(f"\n===== CASE C: VIDEO-ONLY ({clip}) =====")
    out = os.path.join(HERE, "out_C_videoonly.mp4")
    crashed = None
    try:
        r = run_stream(os.path.join(HERE, clip), out, mode="bicubic", max_frames=None)
    except Exception as e:
        crashed = f"{type(e).__name__}: {e}"
        print(f"  CRASH: {crashed}")
        return {"label": "C", "pass": False, "crash": crashed}
    note = r["timing"].get("audio_note")
    vinfo, ainfo = decode_timeline(out)
    print(f"  audio_note               : {note}")
    print(f"  produced video frames    : {r['timing'].get('n_frames')}")
    print(f"  OUT video : codec={vinfo['codec']} n={vinfo['n']} end={vinfo['end']:.3f}s")
    print(f"  OUT audio streams        : {ainfo['n']} frames (expect 0)")
    note_ok = note is not None and note.lower().startswith("none")
    noaudio_ok = ainfo["n"] == 0
    vid_ok = vinfo["n"] == 150
    p = note_ok and noaudio_ok and vid_ok
    print(f"  audio_note==none? {note_ok}  no audio track? {noaudio_ok}  video intact? {vid_ok}")
    print(f"  --> {'PASS' if p else 'FAIL'}")
    return {"label": "C", "note": note, "pass": p}


if __name__ == "__main__":
    results = {}
    results["A"] = case_transcode("A", "short_mp3.mp4", "mp3")
    results["A2"] = case_transcode("A2", "short_opus.mp4", "opus")
    results["D"] = case_transcode("D", "../../server/testdata/short.mp4", "aac")  # control: copy path
    results["C"] = case_video_only("short_noaudio.mp4")
    results["B"] = case_longclip(os.path.join(REPO, "sample.mp4"), max_frames=600)

    print("\n\n================ SUMMARY ================")
    for k in ("A", "A2", "D", "C", "B"):
        r = results[k]
        print(f"  case {k:2s}: {'PASS' if r.get('pass') else 'FAIL'}   "
              f"{r.get('note','')}")
