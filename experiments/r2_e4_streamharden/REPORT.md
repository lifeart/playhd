# R2-E4: Hardening `server/progressive.py` (progressive fMP4 playback)

All work server-side via PyAV (no browser, no system ffmpeg). Each clip streamed through
`progressive.iter_fragments(...)` with the single-job lock acquired/released exactly like
`server/app.py`. Output fMP4 decoded with PyAV and asserted.

## Results per case

| Case | What | Valid AAC + dur match? | Max A/V drift | Bound / clean? | Verdict |
|---|---|---|---|---|---|
| **A** non-AAC mp3→aac (bicubic, full) | `_transcode_audio_pairs` | YES — aac 44.1kHz, end 6.164s vs video 6.121s (44ms) | tail 44ms; head 80ms* | monotonic 23.2ms cadence | **PASS** |
| **A2** non-AAC opus→aac | second non-AAC path | YES — aac 48kHz, 6.139 vs 6.121s (19ms) | tail 19ms; head 80ms* | monotonic | **PASS** |
| **D** AAC copy control | `_copy_audio_pairs` | YES — copied, 6.118 vs 6.121s (3ms) | tail 3ms; head 80ms* | monotonic | **PASS** |
| **C** video-only | no-audio branch | n/a — 0 audio streams, `audio_note="none..."`, 150/150 video | n/a | clean, no crash | **PASS** |
| **B** long-clip (sample.mp4, max_frames=600, bicubic) | flow + memory | **NO — output broken (BUG-1)** | n/a | flow PASS (51 chunks, max gap 354ms); RSS PASS (q1→q4 −268MB) | **mechanics PASS / output FAIL → fixed** |

\* The 80ms "head" is a constant ~2-frame video-start offset (encoder-side), not accumulating drift.

## BUG-1 (REAL, ship-blocking for capped streams) — `close()` dumped the entire source audio track — FIXED

**Symptom.** `sample.mp4` with `max_frames=600` (=24s video) produced a 24s video track but a
**2032s audio track** (entire source audio muxed after video end); file corrupt (`InvalidDataError`
on decode ~24s mark), 19.6MB.

**Reachable in production.** `GET /api/stream?...&frames=N` passes `frames` → `max_frames`. Missed in
E1 because E1 used the uncapped `short.mp4` where video+audio reach EOF together (bug dormant).

**Root cause.** `FragmentMuxer.close()`: `self._feed_audio(float("inf"))` drains ALL remaining
source-audio packets regardless of where the video stopped; the audio is demuxed from a separate
`av.open` of the full file with no `max_frames` bound.

**Fix (landed in `server/progressive.py`).** `self._feed_audio(self._video_time())` — the streaming
loop already fed audio to ~`video_end + AUDIO_LOOKAHEAD_S`, so this adds nothing for an uncapped clip
and bounds a capped one to its real length.

**Verified by the lead after landing the fix:**
| | audio end | video dur | decodes cleanly? | bytes |
|---|---|---|---|---|
| BEFORE (`float("inf")`) | 2032.4s | 24.0s | NO (both streams InvalidDataError ~24s) | 19.6MB |
| AFTER (`_video_time()`), capped frames=600 | 25.06s | 24.0s | YES (600/600 video) | 7.2MB |
| AFTER, uncapped short.mp4 (regression) | 6.07s | 6.04s | YES (150/150) | 1.4MB |

## Secondary (minor, NOT this feature's regression) — ~80ms video-start offset, no edit list
In every case muxed video starts at PTS ~0.0805s (2 frames) while audio starts at 0.0, with no
`elst`/`edts` box — the classic libx264 + fragmented-MP4 B-frame DTS≥0 offset. Identical across
mp3/opus/aac-copy → it's the fresh video encode, not the audio path. Constant (no accumulating
drift). Optional polish: shift first video PTS to 0 or emit an edit list.

## PASS/FAIL
A/A2/D (transcode + copy + A/V sync) PASS; C (video-only) PASS; B (long-clip) streaming PASS,
output FAIL via BUG-1 → **fixed and verified**.

Artifacts: `make_clips.py`, `harden_test.py`, `fix_verify.py`.
