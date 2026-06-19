# E1 — Progressive play-while-processing (instant mode) — REPORT

**Verdict: GO.** A fragmented-MP4 (fMP4) byte stream, produced live by the existing PyAV encode path and streamed over chunked HTTP, lets the browser start playing after one lead buffer (~1 fragment) while the server keeps upscaling. Proven byte-valid + play-before-EOF over real HTTP. The one real risk is **sustain margin**: progressive only holds up indefinitely if instant's steady produce-fps ≥ playback-fps, and instant currently sits right at that edge (~24 vs ~25).

All numbers are **measured** on this box unless marked *(inferred)*. Driven on `server/testdata/short.mp4` (150 f, 640×320 → 1280×640, 24.83 fps, AAC). GPU (MPS) was **shared with 3 sibling experiments**, so instant produce-rate figures are degraded and reported as a range with the contention noted; delivery/byte/audio results are GPU-free and contention-independent.

## 1. Delivery mechanism — decision

| option | starts before EOF? | new machinery | works with current PyAV path |
|---|---|---|---|
| **(c) chunked download of a `+faststart` mp4** | **NO** | none | ✗ — `+faststart` moves `moov` to the front *only by a final rewrite after the whole file is encoded* (needs every sample's size/offset). It's a post-process; a partial download has no playable moov. |
| **(a) fragmented MP4 + plain `<video src>`** | **YES** | tiny | ✓ |
| **(a′) fragmented MP4 + MSE `SourceBuffer`** | YES | medium (JS buffer loop) | ✓ (same bytes) |
| **(b) HLS (TS/fMP4 + playlist)** | YES | most (segmenter + live playlist + per-segment IO) | ✓ but heaviest |

**Chosen: fragmented MP4 over a single chunked HTTP response, consumed by a plain `<video src>`** (MSE is a drop-in upgrade — the *same byte stream* feeds it).

Decisive reasons: it's the **only** format both *producible incrementally by the current PyAV encoder* (`moof`+`mdat` fragments as each GOP closes — no final rewrite, no backward seek) **and** *playable before EOF* (a tiny `empty_moov` init box goes out front). **Least new machinery**: change `movflags` from `+faststart` to `empty_moov+frag_keyframe+default_base_moof`, set a small encoder GOP, interleave source audio into the *same* container. Verified leading boxes are `ftyp · moov · moof · mdat · moof …` — exactly both the progressive shape **and** the MSE init+media shape, so MSE works on identical bytes with zero server change. Confirmed the fMP4 muxer writes to a **non-seekable** Python file-like with **zero seeks**, so a FastAPI `StreamingResponse` generator is a valid sink.

## 2. Chosen design
`stream_gops` (existing, bounded memory) → `InstantProducer` (read-only mirror of `process_clip`'s fast path: anchor-only SR cache → GPU-resident reconstruct → adaptive B-leaf patch → GPU grain → download) yields HD frames → `FragmentMuxer` (one `av.open(sink, movflags=empty_moov+frag_keyframe+default_base_moof)`, fresh H.264 stream with `gop_size=12`, source audio in the SAME container kept ~1 s ahead of video PTS so each fragment flushes with its audio) → `ByteSink.drain()` → `StreamingResponse(media_type="video/mp4")`. Single-threaded (GPU stays single-threaded; consumer back-pressure bounds memory to one chunk, same invariant as `process_clip`).

## 3. Measured results

**3a. Byte validity + play-before-EOF (GPU-free, solid):** `bicubic`, 150 f over HTTP: HTTP TTFB **0.008 s**; whole stream wall 1.46 s (incremental); whole-stream re-decode **150 frames** 1280×640 + audio ✓; **first 64 KB (4.5 %) re-decodes 9 frames** ⇒ play-before-EOF proven over real HTTP. Real **instant** stream (48 f, GPU) re-decoded to 48 frames + audio; leading boxes `ftyp·moov·moof·mdat·moof`. **Single-job lock held**: two concurrent `/stream` → one `200`, one `409`.

**3b. TTFF** (progressive = constant in clip length; baseline = linear):

| | measured | note |
|---|---|---|
| progressive, bicubic | **0.29 s** | clean |
| progressive instant, 96 f, **heavy contention** | first fragment at **7.17 s** *while processing ran to 12.29 s* | play-before-whole-clip even with GPU saturated |
| progressive instant, **warm+uncontended** *(inferred)* | **~1.0–1.5 s** | ≈24 frames / ~24 fps + 2×GOP flush |
| baseline whole-clip, 96 f | **5.7 s** | first frame only after this |

Win scales with length: short 6 s ~5×; 1 min ~50×; 10 min ~500×; **sample.mp4 (34 min, 50 805 f): baseline ~35 min vs progressive ~1.5 s ≈ ~1700×.**

**3c. Sustain / buffer:** drain = source fps (24.83/25). Measured instant produce this session (3-way GPU contention): **5.7–16.1 fps** (best 16.1); uncontended ~24 fps *(inferred, handoff)*; GPU-free bicubic 64.7 fps sustained with a **1-frame** lead. Rule: sustains iff produce-fps ≥ drain-fps. If `r_p < r_d`, required lead = `N·(1/r_p − 1/r_d)` s (no finite buffer sustains an arbitrarily long clip):

| clip (N) | lead @ 24 vs 25 | @ 26 (margin) | @ 16 (contended) |
|---|---|---|---|
| 6 s (150) | 0.2 s | ~1 GOP | 3.4 s |
| 1 min (1 500) | 2.5 s | ~1 GOP | 33.8 s |
| 10 min (15 000) | 25 s | ~1 GOP | 5.6 min |
| 34 min (50 805) | 84.7 s | ~1 GOP | 19 min |

A **2–3 s lead** comfortably covers clips up to a few minutes at the 24-vs-25 edge; underrun is graceful (browser rebuffers). With a positive margin any length sustains on ~1 GOP.

## 4. Audio sync
Open source once; demux compressed audio in order; add to the **same** output container — **AAC copied** (no re-encode, like `_mux_av`'s fast path), non-AAC transcoded to AAC (mirrors `_transcode_audio_iter`, untested — clip is AAC). After each video frame, feed audio with `pts ≤ video_pts + 1.0 s` so audio stays slightly ahead and each video fragment flushes with its audio; `out.mux` (av_interleaved_write_frame) orders by dts. Verified: streamed output re-decodes with audio dur 6.12 s ≈ video 6.04 s, both starting at PTS 0 → in sync segment-by-segment.

## 5. Prototype — how to run
```bash
cd /Users/lifeart/Repos/playhd
python3 -m uvicorn experiments.exp1_progressive.progressive_app:app --host 127.0.0.1 --port 8011
# open http://127.0.0.1:8011/  (mode bicubic = GPU-free demo, instant = real SR; page logs TTFF; "Time the whole-clip baseline" compares)
python3 experiments/exp1_progressive/measure.py --producer bicubic --frames 150 --codec libx264
python3 experiments/exp1_progressive/measure.py --producer instant --frames 96 --baseline
```
Files (all NEW): `progressive_pipe.py` (ByteSink, FragmentMuxer, InstantProducer, BicubicProducer, stream_fragmented), `progressive_app.py` (:8011), `progressive.html`, `measure.py`. No shared file touched.

## 6. Integration plan (for SEAM-VERIFY) — all additive, `process_clip`/quality/layered untouched

**`server/pipeline_api.py`** — add (lift validated bodies from `progressive_pipe.py`): `class _FragmentMuxer(sink, fps, src_audio_path, w_hd, h_hd, codec=None, gop=12)` with `write_frame(rgb_uint8)`/`close()`; `_instant_hd_frames(input_path, max_frames=None, soft_cap=24)` generator (same body as the fast path); `stream_clip_fragments(input_path, mode="instant", max_frames=None, soft_cap=24, codec=None, gop=12)` → yields fMP4 byte chunks, wraps `try_begin_job()`/`end_job()` (release in generator `finally` — covers client disconnect via `GeneratorExit`), calls `_set_progress(...)` so `/api/progress` still works. Reuse `_HW_CODEC`/`_HW_BPP`/`_hw_encode_available`/`_probe_fps`/`INSTANT_*`/`MODE_CONFIG`/`stream_gops`/`_free_gpu`. Keep `_mux_av`/`+faststart`/`/outputs` as-is.

**`server/app.py`** — add `GET /api/stream?source=&mode=instant&frames=` → `StreamingResponse(pipe.stream_clip_fragments(input_path, mode, max_frames=frames), media_type="video/mp4", headers={"Cache-Control":"no-store","X-Accel-Buffering":"no"})`; 400 for non-instant; 409 if `pipe.is_busy()`; **no Content-Length**. Leave `POST /api/process` for quality/layered.

**`server/index.html`** — for `selectedMode==="instant"`: set `v.src = "/api/stream?source=...&mode=instant"`, hide progress on `loadeddata`, `v.play().catch(...)`; keep `/api/progress` polling. Quality/layered keep the existing POST→`/outputs` path.

**Seam checklist:** (1) caller params `{source,mode,frames}` ⟷ `api_stream` ⟷ `stream_clip_fragments(input_path, mode, max_frames=frames)` match; (2) generator yields `bytes`, media_type video/mp4, no Content-Length; (3) lock acquired in `stream_clip_fragments`, released in `finally` (client-abort frees lock; concurrent→409, verified); (4) encode rate == source fps; (5) one real browser end-to-end (loadeddata fires, audio synced, no stall on a short clip) before declaring done.

## 7. Honest: measured vs inferred
**Measured:** non-seekable fMP4 emission/zero seeks; whole-stream + truncated-prefix re-decode (play-before-EOF) over real HTTP; audio copy + duration match; leading-box shape (works for `<video src>` and MSE); 409 lock; bicubic 64.7 fps/1-frame lead; instant first-fragment **before** whole-clip even under contention (7.17 s vs 12.29 s); baseline 5.7 s @96 f; sample.mp4 = 50 805 f. **Inferred/modeled:** uncontended instant ~24 fps (handoff; not re-measurable under contention now); warm TTFF ~1–1.5 s; TTFF-scaling + lead-buffer tables (closed forms). **Not exercised in a real browser this run** (no display driven) — the page is provided; the byte stream is proven frame-for-frame decodable by PyAV (same demux/decode a browser does); browser end-to-end is the lead's seam test. **Not exercised:** non-AAC transcode path; HTTP range/seek (live stream not seekable until buffered — v1-acceptable).

## 8. Risks (ranked)
1. **Sustain margin** (the real risk): instant at the 24-vs-25 edge; under contention drops below source fps → rebuffer. Mitigate: positive produce margin (720p tier should target > source fps), 2–3 s lead buffer, graceful rebuffer. Confirm ~24 fps once siblings free the GPU.
2. **Chunk granularity taxes TTFF**: nothing yields until the first chunk's reconstruct finishes → use a **small first chunk** (~8–12 f), larger after.
3. **No seeking** on the live stream until buffered (v1-acceptable; future: tee fMP4 to disk + byte-range, or live HLS).
4. **VideoToolbox adds ~1 GOP of pipeline latency** vs libx264 (isolated GPU-free test: first fragment ~frame 22 either way, VT ~4 s slower under load). Prefer **libx264 + ThreadedEncoder** for the progressive path; keep VT for the buffered `/outputs` path.
