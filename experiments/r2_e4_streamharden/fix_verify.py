"""R2-E4 fix verification (server/ stays READ-ONLY -- we MONKEYPATCH in-process, no file edit).

PRIMARY BUG (found in case B): FragmentMuxer.close() drains audio with _feed_audio(float("inf")),
so when max_frames caps the video the muxer muxes the ENTIRE source audio track. Result for
sample.mp4 capped at 600 frames: 24.0s video + 2032.4s audio in one file.

The comment on that line says "drain remaining audio up to the real video end" -- the code does NOT.
Proposed fix in server/progressive.py: replace float("inf") with self._video_time() (the duration
of video actually produced). This caps audio to the produced video, AND -- because the transcode/
copy audio iterators are LAZY -- avoids decode/encode/remux of the entire leftover audio at close.

This script proves the fix: stream sample.mp4 capped at 600 with the patched close(), then decode.
"""
import os
import sys
import av

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SERVER = os.path.join(REPO, "server")
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

import pipeline_api as pipe          # noqa: E402
import progressive as prog           # noqa: E402
from harden_test import run_stream, decode_timeline   # noqa: E402

_ORIG_CLOSE = prog.FragmentMuxer.close


def _patched_close(self):
    """server/progressive.py FragmentMuxer.close() with the ONE-LINE fix applied."""
    for pkt in self.vst.encode():            # flush video encoder
        pkt.stream = self.vst
        self.out.mux(pkt)
    if self.aout is not None:                # FIX: cap audio to produced video, not float("inf")
        self._feed_audio(self._video_time())
    try:
        self.out.close()
    finally:
        self.scont.close()


def measure(out_path, label):
    vinfo, ainfo = decode_timeline(out_path)
    ve = vinfo["end"]
    ae = ainfo["end"]
    print(f"  [{label}] video_end={ve:.2f}s  audio_end={ae and round(ae,2)}s  "
          f"tail_drift={abs((ae or 0)-(ve or 0))*1000:.0f}ms  "
          f"file={os.path.getsize(out_path)/1e6:.1f}MB  v_n={vinfo['n']} a_n={ainfo['n']}")
    return ve, ae


if __name__ == "__main__":
    src = os.path.join(REPO, "sample.mp4")

    print("== BEFORE (current server/progressive.py close, float('inf')) ==")
    out_before = os.path.join(HERE, "fix_before.mp4")
    run_stream(src, out_before, mode="bicubic", max_frames=600)
    measure(out_before, "BEFORE")

    print("== AFTER (patched close, _feed_audio(self._video_time())) ==")
    prog.FragmentMuxer.close = _patched_close
    try:
        out_after = os.path.join(HERE, "fix_after.mp4")
        r = run_stream(src, out_after, mode="bicubic", max_frames=600)
        ve, ae = measure(out_after, "AFTER")
        ok = abs((ae or 0) - (ve or 0)) < 0.30
        print(f"\n  FIX RESULT: audio_end now tracks video_end? {ok}  "
              f"(was 2032s audio over 24s video; now {ae:.2f}s audio over {ve:.2f}s video)")
    finally:
        prog.FragmentMuxer.close = _ORIG_CLOSE
