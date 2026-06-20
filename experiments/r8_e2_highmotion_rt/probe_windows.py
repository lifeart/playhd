#!/usr/bin/env python3
"""R8-E2 step 1: locate the strongest HIGH-MOTION windows of sample.mp4 by codec-MV magnitude.

Single sequential decode pass (MV side-data only, NO rgb24 conversion) over the first
SCAN_FRAMES frames; per frame compute mean |MV| (LR px/frame, block-area weighted over
MV-covered pixels) and the no-MV (intra/disocclusion) fraction. Then slide a 48-frame
window (stride 24) and rank by mean |MV|. Confirms exp2's "window A = start 0" is high
motion and surfaces any stronger fast-pan/action window for the clustering measurement.

READ-ONLY: pure PyAV decode + numpy; imports nothing shared except av.
"""
import json
import os
import sys

import av
import numpy as np
from av.sidedata.sidedata import Type as SDType

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
CLIP = os.path.join(_REPO, "sample.mp4")

SCAN_FRAMES = int(os.environ.get("SCAN_FRAMES", "9000"))
WIN = 48
STRIDE = 24


def frame_motion(mvs, h, w):
    """Block-area-weighted mean |MV| (LR px/frame) over MV-covered pixels + no-MV fraction.
    Mirrors region_quality.motion_mag_lr semantics without building the full dense field."""
    if mvs is None or len(mvs) == 0:
        return 0.0, 1.0
    covered = np.zeros((h, w), bool)
    magsum = 0.0
    area = 0.0
    for r in mvs:
        ms = float(r["motion_scale"]) or 1.0
        dx = float(r["motion_x"]) / ms
        dy = float(r["motion_y"]) / ms
        bw, bh = int(r["w"]), int(r["h"])
        cx, cy = int(r["dst_x"]), int(r["dst_y"])
        x0, x1 = max(cx - bw // 2, 0), min(cx + bw // 2, w)
        y0, y1 = max(cy - bh // 2, 0), min(cy + bh // 2, h)
        a = max(0, (x1 - x0)) * max(0, (y1 - y0))
        if a <= 0:
            continue
        covered[y0:y1, x0:x1] = True
        magsum += (dx * dx + dy * dy) ** 0.5 * a
        area += a
    mag = magsum / area if area > 0 else 0.0
    no_mv = 1.0 - covered.mean()
    return float(mag), float(no_mv)


def main():
    cont = av.open(CLIP)
    vs = cont.streams.video[0]
    vs.codec_context.options = {"flags2": "+export_mvs"}
    h = vs.codec_context.height
    w = vs.codec_context.width
    print(f"clip {w}x{h}; scanning first {SCAN_FRAMES} frames (MV-only)")
    per = []   # (ptype, mag, no_mv)
    for idx, frame in enumerate(cont.decode(vs)):
        if idx >= SCAN_FRAMES:
            break
        try:
            sd = frame.side_data.get(SDType.MOTION_VECTORS)
        except Exception:
            sd = None
        mvs = sd.to_ndarray() if sd is not None else None
        pt = {1: "I", 2: "P", 3: "B"}.get(int(frame.pict_type), "?")
        mag, no_mv = frame_motion(mvs, h, w)
        per.append((pt, mag, no_mv))
        if idx % 1000 == 0:
            print(f"  ..{idx}")
    cont.close()
    n = len(per)
    mags = np.array([p[1] for p in per])
    nomvs = np.array([p[2] for p in per])

    wins = []
    for s in range(0, n - WIN, STRIDE):
        seg_mag = mags[s:s + WIN]
        seg_nomv = nomvs[s:s + WIN]
        wins.append(dict(start=s, mean_mag=float(seg_mag.mean()), max_mag=float(seg_mag.max()),
                         mean_nomv=float(seg_nomv.mean())))
    wins.sort(key=lambda d: d["mean_mag"], reverse=True)

    print("\nTOP-10 highest mean|MV| 48-frame windows:")
    print(f"  {'start':>6}{'meanMag':>9}{'maxMag':>8}{'meanNoMV%':>11}")
    for d in wins[:10]:
        print(f"  {d['start']:>6}{d['mean_mag']:>9.2f}{d['max_mag']:>8.2f}{100*d['mean_nomv']:>11.2f}")
    print("\nLOWEST mean|MV| (talking-head candidates):")
    for d in sorted(wins, key=lambda d: d["mean_mag"])[:6]:
        print(f"  {d['start']:>6}{d['mean_mag']:>9.2f}{d['max_mag']:>8.2f}{100*d['mean_nomv']:>11.2f}")
    # reference: start 0 and start 5000 (exp2's A and C)
    def win_at(s):
        return next((d for d in wins if d["start"] == s), None)
    print("\nexp2 reference windows:")
    for s in (0, 5000):
        d = win_at(s) or (min(wins, key=lambda d: abs(d["start"] - s)))
        print(f"  ~start {s}: start={d['start']} meanMag={d['mean_mag']:.2f} maxMag={d['max_mag']:.2f} "
              f"noMV={100*d['mean_nomv']:.1f}%")

    out = dict(scan_frames=n, win=WIN, stride=STRIDE,
               top=wins[:15], low=sorted(wins, key=lambda d: d["mean_mag"])[:8])
    with open(os.path.join(_HERE, "windows.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(_HERE, 'windows.json')}")


if __name__ == "__main__":
    main()
