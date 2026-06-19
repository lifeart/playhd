"""playhd Stage-1 local server.

A minimal FastAPI app that wraps the validated upscaling pipeline so a user can, in a
browser: pick a source mp4, choose a quality mode, click "Process & Play", and watch the
upscaled result.

Endpoints:
  GET  /                -> server/index.html (the demo console)
  GET  /api/sources     -> available source mp4s (repo sample + server/uploads/*.mp4)
  POST /api/process     -> run the pipeline (source|upload + mode + start/n) -> output mp4 URL
  GET  /outputs/<f>.mp4 -> the produced mp4, served with Content-Type video/mp4 + Range
                           support (StaticFiles) so <video> can play/seek.

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
        {"id": "instant", "label": "Instant (lower quality)",
         "desc": "Compact x4 anchor on the GPU, motion-propagated. Fast, real-time-style path."},
        {"id": "quality", "label": "Quality (buffered, slower)",
         "desc": "Heavy x4plus anchor + region-aware detail blend. Sharper, much slower (~2 s/frame)."},
    ]}


@app.post("/api/process")
async def api_process(
    mode: str = Form("instant"),
    source: str = Form("sample.mp4"),
    start: int = Form(5000),
    n: int = Form(48),
    file: UploadFile | None = File(None),
):
    if mode not in pipe.MODE_CONFIG:
        raise HTTPException(400, f"unknown mode {mode!r}")

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
        # Run the (blocking, GPU-bound) pipeline off the event loop.
        out_path = await run_in_threadpool(
            pipe.process_clip, input_path, mode, int(start), int(n)
        )
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
