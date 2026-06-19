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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

_HERE = os.path.dirname(os.path.abspath(__file__))
# Make `import pipeline_api` work whether uvicorn loads this as `server.app`
# (from the repo root) or as `app:app` (from inside server/).
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pipeline_api as pipe   # noqa: E402
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
         "desc": "Compact x4 anchor on the GPU, motion-propagated. Fast (~0.4 s/frame)."},
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
