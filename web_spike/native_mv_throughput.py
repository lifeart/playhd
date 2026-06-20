"""Web-only de-risk spike, part 1: native software-decode + export_mvs throughput at SD.

PyAV/libav uses SOFTWARE decode (no VideoToolbox), so its decode speed is the correct proxy
for a WASM-libav software-decode build. We measure, at SD, single-threaded (conservative, matches
common single-threaded ffmpeg.wasm) and multi-threaded:
  (a) decode-only fps
  (b) decode + export_mvs (read MOTION_VECTORS side-data) fps  <- the architecture's real cost
  (c) MV payload: records/frame + bytes/frame that would cross the WASM->JS boundary
Extrapolate to WASM via the documented ffmpeg.wasm penalty (~5-10x slower than native).
"""
import time
import numpy as np
import av
import av.sidedata

SRC = "sample.mp4"
N = 600   # frames to time (steady-state)


def run(threads, do_mvs):
    c = av.open(SRC)
    v = c.streams.video[0]
    v.thread_count = threads
    v.thread_type = "AUTO" if threads != 1 else "NONE"
    if do_mvs:
        v.codec_context.options = {"flags2": "+export_mvs"}
    w, h = v.width, v.height
    n = 0
    mv_records = []
    mv_bytes = []
    t0 = time.perf_counter()
    for frame in c.decode(video=0):
        if do_mvs:
            sd = frame.side_data.get(av.sidedata.sidedata.Type.MOTION_VECTORS)
            if sd is not None:
                arr = sd.to_ndarray()
                mv_records.append(len(arr))
                mv_bytes.append(arr.nbytes)
            else:
                mv_records.append(0); mv_bytes.append(0)
        n += 1
        if n >= N:
            break
    dt = time.perf_counter() - t0
    c.close()
    return w, h, n, dt, mv_records, mv_bytes


def main():
    # warm (page cache / lib init)
    run(1, False)
    print(f"clip={SRC}  timing {N} frames each\n")
    rows = []
    for threads in (1, 0):  # 1 = single-thread (conservative), 0 = auto (all cores)
        for do_mvs in (False, True):
            w, h, n, dt, mr, mb = run(threads, do_mvs)
            fps = n / dt
            tag = f"threads={'1(single)' if threads==1 else 'auto'}  {'decode+MVs' if do_mvs else 'decode-only'}"
            extra = ""
            if do_mvs and mr:
                extra = (f"  | MVs/frame: med={int(np.median(mr))} max={max(mr)} "
                         f"| bytes/frame: med={int(np.median(mb))} ({np.mean(mb)/1024:.1f}KB avg)")
            print(f"  {tag:34} {fps:7.1f} fps ({1000*dt/n:.2f} ms/frame){extra}")
            rows.append((threads, do_mvs, fps))
    print(f"\n  SD resolution: {w}x{h}")
    # the decision number: single-thread decode+MVs fps, extrapolated to WASM
    st_mv = [f for (t, m, f) in rows if t == 1 and m][0]
    print(f"\n  === EXTRAPOLATION TO WASM (single-thread, the conservative case) ===")
    print(f"  native single-thread decode+MVs: {st_mv:.0f} fps")
    for pen in (5, 8, 10):
        wasm = st_mv / pen
        verdict = "REAL-TIME (>=30fps)" if wasm >= 30 else ("MARGINAL" if wasm >= 20 else "TOO SLOW")
        print(f"    /{pen}x WASM penalty -> {wasm:5.0f} fps  [{verdict}]")
    print(f"\n  (auto-thread native is the optimistic ceiling if WASM threads / SharedArrayBuffer are enabled)")


if __name__ == "__main__":
    main()
