"""playhd Stage-1 local server.

A minimal FastAPI app that wraps the validated streaming upscaling pipeline so a user
can, in a browser: pick a source mp4 (or upload one), choose Instant / Quality, click
"Process & Play", watch a live progress bar, then watch the upscaled result -- WITH SOUND.

Endpoints:
  GET  /                -> server/index.html (the demo console)
  GET  /api/sources     -> available source mp4s + the two modes
  POST /api/process     -> run the whole-clip streaming pipeline (source|upload + mode)
                           -> {url, source, stats}.  One job at a time (409 if busy).
  GET  /api/progress     -> live {state, done, total, elapsed_s, eta_s, ms_per_frame}
  GET  /outputs/<f>.mp4 -> the produced mp4 (StaticFiles: video/mp4 + HTTP Range -> seek)

Run:
  cd /Users/lifeart/Repos/playhd
  python3 -m uvicorn server.app:app --host 127.0.0.1 --port 8000
  # then open http://127.0.0.1:8000/
"""

import os
import sys
import time

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

_HERE = os.path.dirname(os.path.abspath(__file__))
# Make `import pipeline_api` work whether uvicorn loads this as `server.app`
# (from the repo root) or as `app:app` (from inside server/).
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pipeline_api as pipe   # noqa: E402
import progressive as prog    # noqa: E402  (play-while-process: fMP4 stream for instant mode)
INDEX_HTML = os.path.join(_HERE, "index.html")

app = FastAPI(title="playhd Stage-1 console")

# Output mp4s are served as static files. Starlette StaticFiles sets video/mp4 from the
# extension and honours HTTP Range requests (206) -> <video> can seek.
app.mount("/outputs", StaticFiles(directory=pipe.OUTPUTS_DIR), name="outputs")


@app.get("/")
def index():
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/api/sources")
def api_sources():
    return {"sources": pipe.list_sources(), "modes": [
        {"id": "instant", "label": "Instant",
         "desc": "Compact x4 anchor on the GPU, motion-propagated. Real-time (720p tier) — "
                 "plays as it processes (starts in ~1-2 s, no wait for the whole clip)."},
        {"id": "quality", "label": "Quality",
         "desc": "Heavy x4plus anchor + region-aware detail. Sharper, slower (~2.8 s/frame)."},
        {"id": "layered", "label": "Quality — Layered (talking-head, static camera)",
         "desc": "Stable, shimmer-free background: heavy-SR the static background plate ONCE "
                 "per scene, composite only the moving subject per frame. Needs a roughly "
                 "static camera + a human subject; non-commercial matte (RVM, CC BY-NC-SA). "
                 "Slowest path — buffered, two decode passes."},
    ]}


@app.get("/api/progress")
def api_progress():
    return pipe.get_progress()


@app.post("/api/process")
async def api_process(
    mode: str = Form("instant"),
    source: str = Form("sample.mp4"),
    file: UploadFile | None = File(None),
):
    if not pipe.is_valid_mode(mode):
        raise HTTPException(400, f"unknown mode {mode!r}")
    if pipe.is_busy():
        raise HTTPException(409, "a clip is already being processed; please wait")

    # An uploaded file takes precedence over the dropdown selection.
    if file is not None and file.filename:
        if not file.filename.lower().endswith(".mp4"):
            raise HTTPException(400, "please upload an .mp4 file")
        safe = os.path.basename(file.filename)
        dest = os.path.join(pipe.UPLOADS_DIR, safe)
        data = await file.read()
        with open(dest, "wb") as fh:
            fh.write(data)
        input_path = dest
        src_name = safe
    else:
        try:
            input_path = pipe.resolve_source(source)
        except ValueError as e:
            raise HTTPException(400, str(e))
        src_name = os.path.basename(input_path)

    try:
        # Run the (blocking, GPU-bound) WHOLE-CLIP streaming pipeline off the event loop.
        # /api/progress is served concurrently while this runs.
        out_path = await run_in_threadpool(pipe.process_clip, input_path, mode)
    except pipe.BusyError:
        raise HTTPException(409, "a clip is already being processed; please wait")
    except Exception as e:                      # surface real failures to the UI, never swallow
        raise HTTPException(500, f"pipeline failed: {type(e).__name__}: {e}")

    stats = dict(pipe.LAST_STATS)
    fname = os.path.basename(out_path)
    # cache-bust so the browser always loads the freshly produced clip
    url = f"/outputs/{fname}?t={int(time.time())}"
    return JSONResponse({"url": url, "source": src_name, "stats": stats})


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Save an uploaded mp4 to uploads/ and return its name, so the progressive GET /api/stream
    endpoint (which resolves by name) can stream an upload too. Used by instant mode."""
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "please upload an .mp4 file")
    safe = os.path.basename(file.filename)
    dest = os.path.join(pipe.UPLOADS_DIR, safe)
    data = await file.read()
    with open(dest, "wb") as fh:
        fh.write(data)
    return {"name": safe}


_STREAM_DONE = object()   # sentinel: the sync producer is exhausted (StopIteration can't cross the
                          # threadpool boundary cleanly, so map it to this and stop)


def _next_chunk(sync_gen):
    try:
        return next(sync_gen)
    except StopIteration:
        return _STREAM_DONE


@app.get("/api/stream")
async def api_stream(source: str = "sample.mp4", mode: str = "instant", frames: int | None = None):
    """Progressive play-while-process: stream the upscaled clip as a fragmented MP4 produced live,
    so the browser starts playing after one lead buffer (~1-2 s) instead of waiting for the whole
    clip. Instant mode only (it is the real-time tier); quality/layered stay on POST /api/process.

    The single-job lock is acquired HERE (so a busy collision is a clean 409 before any bytes are
    sent). The body is an ASYNC generator that pulls the sync producer one fragment at a time in a
    threadpool. On client disconnect Starlette cancels the stream task (StreamingResponse runs a
    `listen_for_disconnect` that cancels the task group), which raises CancelledError into the
    generator at its next await -> the `finally` runs SYNC cleanup (close the producer = flush+close
    the muxer + free GPU; release the lock). `run_in_threadpool` finishes the in-flight chunk before
    the cancel lands, so a disconnected client stops the GPU within ONE chunk instead of churning the
    whole clip. (Do NOT also poll request.is_disconnected() here -- that calls receive() concurrently
    with Starlette's own disconnect listener on the same ASGI channel and breaks the cancellation.)
    """
    if mode not in ("instant", "bicubic"):
        raise HTTPException(400, f"streaming supports instant|bicubic, not {mode!r}")
    try:
        input_path = pipe.resolve_source(source)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if mode == "instant" and pipe._gpu_ops is None:
        raise HTTPException(503, "instant fast path needs gpu_ops (MPS) — unavailable on this box")
    if not pipe.try_begin_job():
        raise HTTPException(409, "a clip is already being processed; please wait")

    timing = {}
    sync_gen = prog.iter_fragments(input_path, mode, max_frames=frames, timing=timing)

    async def agen():
        try:
            while True:
                chunk = await run_in_threadpool(_next_chunk, sync_gen)
                if chunk is _STREAM_DONE:
                    break
                yield chunk
        finally:
            # Runs on normal completion AND on cancellation (client disconnect). Cleanup is SYNC so
            # it completes even inside a cancelled scope: closing the sync producer runs its finally
            # (flush+close the muxer, free GPU); then release the single-job lock.
            try:
                sync_gen.close()
            except Exception as e:                  # surface, never silently swallow
                print(f"[stream] producer close error: {type(e).__name__}: {e}")
            n = timing.get("n_frames", 0)
            tf = timing.get("t_first_fragment")
            print(f"[stream] mode={mode} src={os.path.basename(input_path)} frames={n} "
                  f"TTFF={tf}s audio={timing.get('audio_note')} bytes={timing.get('total_bytes')}")
            pipe.end_job()

    # No Content-Length (length unknown until produced); X-Accel-Buffering off so a proxy does
    # not buffer the whole response before forwarding (which would defeat play-while-process).
    headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    return StreamingResponse(agen(), media_type="video/mp4", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
