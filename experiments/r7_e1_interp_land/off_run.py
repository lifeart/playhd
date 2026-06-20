#!/usr/bin/env python3
"""R7-E1 -- run the REAL process_clip in instant mode (smooth OFF by flag default) and dump a
fingerprint of the produced mp4: frame count, per-frame md5 (decoded pixels), whole-file md5.

Used twice by the regression guard: once with the PRE-CHANGE pipeline_api (via `git stash`) ->
baseline.json, once with the landed change -> changed.json. Byte-identical OFF == identical jsons.
"""
import hashlib
import json
import os
import sys

import av

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "server"), os.path.join(_REPO, "prototype")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_api as P  # noqa: E402  (the REAL pipeline -- pre-change under stash, landed otherwise)

CLIP = os.path.join(_REPO, "sample.mp4")


def fingerprint_mp4(path):
    cont = av.open(path)
    hers = []
    try:
        vs = cont.streams.video[0]
        for fr in cont.decode(vs):
            arr = fr.to_ndarray(format="rgb24")
            hers.append(hashlib.md5(arr.tobytes()).hexdigest())
    finally:
        cont.close()
    with open(path, "rb") as f:
        file_md5 = hashlib.md5(f.read()).hexdigest()
    return {"n": len(hers), "frame_md5": hers, "file_md5": file_md5}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    out = os.path.join(_HERE, f"off_{tag}.mp4")
    has_flag = hasattr(P, "INSTANT_INTERP_2X")
    if has_flag:
        assert P.INSTANT_INTERP_2X is False, "flag must default OFF for the regression baseline"
    P.process_clip(CLIP, "instant", max_frames=n, out_path=out)
    fp = fingerprint_mp4(out)
    fp["tag"] = tag
    fp["has_interp_flag"] = has_flag
    fp["last_stats_keys"] = sorted(P.LAST_STATS.keys())
    with open(os.path.join(_HERE, f"off_{tag}.json"), "w") as f:
        json.dump(fp, f)
    print(f"[off_run:{tag}] n={fp['n']} file_md5={fp['file_md5'][:12]} "
          f"interp_flag_present={has_flag}")


if __name__ == "__main__":
    main()
