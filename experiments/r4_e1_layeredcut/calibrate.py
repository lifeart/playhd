"""R4-E1 calibration: per-frame luma / chroma / edge deltas for the cut-detector fix (b),
and locate a clean single-scene static window near sample.mp4 frame 5000 for the layered
no-regression test.

Read-only imports of server/prototype. Run:
  python3 experiments/r4_e1_layeredcut/calibrate.py
"""
import os
import sys

import numpy as np
import cv2
import av

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "server"))
sys.path.insert(0, os.path.join(REPO, "prototype"))
import scene_detect  # noqa: E402

CLIPS = os.path.join(REPO, "experiments", "r3_e2_robustness", "clips")
SAMPLE = os.path.join(REPO, "sample.mp4")


def _luma(img):
    f = img.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def per_frame_signals(path, lo=0, hi=None):
    """Yield (idx, ptype, dLuma, dChroma, dEdge) for frames [lo,hi)."""
    cont = av.open(path)
    vs = cont.streams.video[0]
    vs.codec_context.options = {"flags2": "+export_mvs"}
    prevY = prevU = prevV = prevE = None
    idx = 0
    rows = []
    for fr in cont.decode(vs):
        if hi is not None and idx >= hi:
            break
        if idx < lo:
            idx += 1
            continue
        pt = {1: "I", 2: "P", 3: "B"}.get(int(fr.pict_type), "?")
        img = fr.to_ndarray(format="rgb24")
        yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float32)
        Y, U, V = yuv[..., 0], yuv[..., 1], yuv[..., 2]
        E = cv2.Canny(Y.astype(np.uint8), 80, 160) > 0
        if prevY is not None:
            dL = float(np.abs(Y - prevY).mean())
            dC = float(np.abs(U - prevU).mean() + np.abs(V - prevV).mean())
            dE = float(np.abs(E.astype(np.float32) - prevE.astype(np.float32)).mean()) * 100.0
            rows.append((idx, pt, dL, dC, dE))
        prevY, prevU, prevV, prevE = Y, U, V, E
        idx += 1
    cont.close()
    return rows


def show_clip(name, cut_frames):
    path = os.path.join(CLIPS, name + ".mp4")
    rows = per_frame_signals(path)
    print(f"\n=== {name}  (expected missed/real cut near {cut_frames}) ===")
    print(" idx pt   dLuma  dChroma  dEdge")
    for (idx, pt, dL, dC, dE) in rows:
        mark = "  <== CUT" if idx in cut_frames else ""
        if idx in cut_frames or abs(dL) > 15 or dC > 12:
            print(f" {idx:4d} {pt}   {dL:5.1f}  {dC:6.2f}  {dE:5.2f}{mark}")


def find_static_window(lo, hi, win=44):
    """Scan sample.mp4 [lo,hi) for the longest run with small luma+chroma deltas (a static
    single scene), return the best (start, dluma_max, dchroma_max)."""
    rows = per_frame_signals(SAMPLE, lo=lo, hi=hi)
    print(f"\n=== sample.mp4 [{lo},{hi}) per-frame deltas (looking for a clean static run) ===")
    # mark frames whose delta is 'cut-like'
    cutish = [r for r in rows if r[2] > 25 or r[3] > 18]
    print("  cut-like frames (dLuma>25 or dChroma>18):")
    for (idx, pt, dL, dC, dE) in cutish:
        print(f"    {idx:5d} {pt}  dLuma={dL:5.1f} dChroma={dC:6.2f}")
    # find best static window of length `win`
    best = None
    arr = {r[0]: r for r in rows}
    idxs = sorted(arr)
    for s in idxs:
        seg = [arr[i] for i in range(s + 1, s + win) if i in arr]
        if len(seg) < win - 1:
            continue
        mxL = max(r[2] for r in seg)
        mxC = max(r[3] for r in seg)
        score = mxL + 0.5 * mxC
        if best is None or score < best[0]:
            best = (score, s, mxL, mxC)
    if best:
        print(f"  >> best static window start={best[1]} len={win}  "
              f"dLuma_max={best[2]:.1f} dChroma_max={best[3]:.1f}")
    return best


if __name__ == "__main__":
    # (b) cut-detector calibration on the R3-E2 missed/detected cuts
    show_clip("c7_staticcut", {28})
    show_clip("c5_scenecut", {28})
    show_clip("c5b_scenecut_strong", {28})

    # locate a clean static window near 5000 for the layered no-regression test
    find_static_window(5050, 5260, win=44)
