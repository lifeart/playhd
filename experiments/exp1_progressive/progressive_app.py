"""E1 prototype server -- progressive play-while-processing (standalone, port 8011).

Runs on a DIFFERENT port from the real app (8000) and shares NO files with it. It reuses
server/pipeline_api READ-ONLY for the instant SR frames and streams them as a fragmented MP4
that a browser can start playing after one lead buffer while processing continues.

Endpoints:
  GET /                       -> progressive.html (the demo page)
  GET /stream?source=&mode=&frames=
                              -> chunked fragmented-MP4 (video+audio interleaved), emitted live.
                                 mode=instant (real SR, GPU) | bicubic (GPU-free delivery demo).
  GET /baseline?source=&frames=
                              -> JSON: the whole-clip baseline time (process_clip), for comparison.

Run:
  cd /Users/lifeart/Repos/playhd
  python3 -m uvicorn experiments.exp1_progressive.progressive_app:app --host 127.0.0.1 --port 8011
  # open http://127.0.0.1:8011/
"""
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import progressive_pipe as pp     # noqa: E402  (puts server/ + prototype/ on path)
import pipeline_api as pipe        # noqa: E402  READ-ONLY

PAGE = os.path.join(_HERE, "progressive.html")

app = FastAPI(title="playhd E1 -- progressive play-while-processing")


@app.get("/")
def index():
    return FileResponse(PAGE, media_type="text/html")


def _resolve(source):
    try:
        return pipe.resolve_source(source)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/stream")
def stream(source: str = "short.mp4", mode: str = "instant", frames: int | None = None):
    """Stream the upscaled clip as a fragmented MP4, produced live. The response begins as soon
    as the first fragment is ready -> the browser starts playing before the clip is finished."""
    input_path = _resolve(source)
    fps = pp.probe_fps(input_path)

    if mode == "bicubic":
        producer = pp.BicubicProducer()
        codec = "libx264"                      # GPU-free path: deterministic software encode
    elif mode == "instant":
        if pipe._gpu_ops is None:
            raise HTTPException(503, "instant fast path needs gpu_ops (MPS) -- unavailable here")
        producer = pp.InstantProducer()
        codec = None                           # auto: videotoolbox if available, else libx264
    else:
        raise HTTPException(400, f"mode must be instant|bicubic, got {mode!r}")

    # Respect the shared single-job lock so we never collide with the main app's GPU job.
    if not pipe.try_begin_job():
        raise HTTPException(409, "a clip is already being processed; please wait")

    timing = {}

    def gen():
        try:
            yield from pp.stream_fragmented(
                producer, input_path, input_path, fps,
                max_frames=frames, soft_cap=24, codec=codec, timing=timing)
        finally:
            # Surface the measured TTFF server-side (the page also measures it client-side).
            tf = timing.get("t_first_fragment")
            print(f"[stream] mode={mode} source={os.path.basename(input_path)} "
                  f"frames={timing.get('n_frames')} TTFF={tf}s "
                  f"audio={timing.get('audio_note')} bytes={timing.get('total_bytes')}")
            pipe.end_job()

    headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), media_type="video/mp4", headers=headers)


@app.get("/baseline")
async def baseline(source: str = "short.mp4", frames: int | None = None):
    """Whole-clip baseline for the SAME window: process_clip produces ALL frames + mux +
    faststart before the browser can play. Returns the server time == TTFF floor for that path."""
    from starlette.concurrency import run_in_threadpool
    input_path = _resolve(source)
    if pipe.is_busy():
        raise HTTPException(409, "busy")
    t0 = time.perf_counter()
    try:
        await run_in_threadpool(pipe.process_clip, input_path, "instant", frames)
    except Exception as e:                      # surface, never swallow
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    s = dict(pipe.LAST_STATS)
    return JSONResponse({
        "whole_clip_server_s": s.get("t_total_s"),
        "ms_per_frame": s.get("ms_per_frame"),
        "n_frames": s.get("n_frames"),
        "note": "first frame is playable only AFTER this whole-clip time (then + download).",
        "wall_s": round(time.perf_counter() - t0, 2),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("progressive_app:app", host="127.0.0.1", port=8011, reload=False)
