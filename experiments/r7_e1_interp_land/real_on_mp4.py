#!/usr/bin/env python3
"""R7-E1 -- produce a REAL "smooth 2x" mp4 through the UNPATCHED process_clip (flag ON) and verify
it is a valid, in-sync, DOUBLED-fps H.264 file: exactly 2x the real frames, out_fps == 2*src_fps,
video duration ~= audio duration (synced). This exercises the true encode + audio mux end-to-end.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "server"), os.path.join(_REPO, "prototype")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_api as P  # noqa: E402

CLIP = os.path.join(_REPO, "sample.mp4")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    P.INSTANT_INTERP_2X = True
    out = os.path.join(_HERE, "smooth2x_real.mp4")
    P.process_clip(CLIP, "instant", max_frames=n, out_path=out)
    s = dict(P.LAST_STATS)
    nvf = s.get("n_video_frames", s["n_frames"])
    ok, info = P._verify_mp4(out, nvf, s["out_resolution"])
    result = {
        "out": out,
        "src_fps": s["fps"], "out_fps": s["out_fps"],
        "real_frames": s["n_frames"], "video_frames": nvf,
        "exact_2x": (nvf == 2 * s["n_frames"]),
        "out_fps_doubled": (abs(s["out_fps"] - 2 * s["fps"]) < 1e-6),
        "smooth_2x": s.get("smooth_2x"),
        "n_interp": s.get("n_interp"), "n_interp_dup": s.get("n_interp_dup"),
        "mp4_verify_ok": bool(ok), "mp4_info": info,
    }
    result["VERDICT"] = "PASS" if (ok and result["exact_2x"] and result["out_fps_doubled"]) else "FAIL"
    with open(os.path.join(_HERE, "real_on_mp4.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["VERDICT"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
