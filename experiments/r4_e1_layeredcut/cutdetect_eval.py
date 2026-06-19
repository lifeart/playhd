"""R4-E1 fix (b): evaluate a CHROMA-augmented cut detector.

The current scene_detect uses mean |Δluma| only. The missed cuts (c7 Δluma 26.4 but Δchroma
33.98; c5 Δluma 47 but Δchroma 109) are SIMILAR-LUMA, DIFFERENT-CHROMA cuts. We fold chroma
into the SAME per-frame diff used by the existing three tests:

    d = mean|ΔY| + CHROMA_W * (mean|ΔU| + mean|ΔV|)

and keep the EXISTING thresholds (CUT_THRESH 60 / IFRAME 45 / REL_FLOOR 40 / REL_MULT 8 /
MIN_SCENE_LEN 24 / EMA 0.30). CHROMA_W=0 reproduces the current luma-only detector byte-for-byte.
This is a strict ADD of detection power (chroma only raises d), so the requirement is: NO regression
on sample.mp4 [0,900) (the 9 real cuts stay, the periodic keyframes / flash stay non-cuts) while
catching c7 + c5.

Run:  python3 experiments/r4_e1_layeredcut/cutdetect_eval.py
"""
import os
import sys

import numpy as np
import cv2
import av

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "server"))
import scene_detect as SD  # noqa: E402

CLIPS = os.path.join(REPO, "experiments", "r3_e2_robustness", "clips")
SAMPLE = os.path.join(REPO, "sample.mp4")


def _yuv_means(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float32)
    return yuv[..., 0], yuv[..., 1], yuv[..., 2]


def detect_combined(path, max_frames=None, chroma_w=0.0,
                    cut_thresh=SD.CUT_THRESH, iframe_thresh=SD.IFRAME_THRESH,
                    rel_floor=SD.REL_FLOOR, rel_mult=SD.REL_MULT,
                    min_scene_len=SD.MIN_SCENE_LEN, ema_alpha=SD.EMA_ALPHA):
    """Re-implements StreamingCutDetector EXACTLY (same tests/thresholds) but with the diff
    d = mean|ΔY| + chroma_w*(mean|ΔU|+mean|ΔV|). chroma_w=0 == scene_detect (verified below)."""
    cont = av.open(path)
    vs = cont.streams.video[0]
    vs.codec_context.options = {"flags2": "+export_mvs"}
    prevY = prevU = prevV = None
    base = None
    last_cut = 0
    cuts = []
    idx = 0
    total = 0
    for fr in cont.decode(vs):
        if max_frames is not None and idx >= max_frames:
            break
        pt = {1: "I", 2: "P", 3: "B"}.get(int(fr.pict_type), "?")
        Y, U, V = _yuv_means(fr.to_ndarray(format="rgb24"))
        if prevY is None:
            prevY, prevU, prevV = Y, U, V
            last_cut = idx
            idx += 1
            continue
        dL = float(np.abs(Y - prevY).mean())
        dC = float(np.abs(U - prevU).mean() + np.abs(V - prevV).mean())
        d = dL + chroma_w * dC
        prevY, prevU, prevV = Y, U, V
        raw = (d > cut_thresh or (pt == "I" and d > iframe_thresh) or
               (base is not None and d > rel_floor and d > rel_mult * base))
        far = (idx - last_cut) >= min_scene_len
        is_cut = raw and far
        if is_cut:
            last_cut = idx
            base = None
            cuts.append(idx)
        else:
            base = d if base is None else (ema_alpha * d + (1.0 - ema_alpha) * base)
        idx += 1
    total = idx
    cont.close()
    return cuts, total


def eval_sample(chroma_w):
    expected = {28, 196, 341, 479, 514, 563, 630, 688, 810}
    cuts, _ = detect_combined(SAMPLE, max_frames=901, chroma_w=chroma_w)
    win = sorted(c for c in cuts if 0 <= c <= 900)
    tp = [c for c in win if c in expected]
    fp = [c for c in win if c not in expected]
    miss = [c for c in expected if c not in win]
    prec = len(tp) / max(1, len(win))
    rec = len(tp) / max(1, len(expected))
    return dict(cuts=win, tp=tp, fp=fp, miss=miss, prec=prec, rec=rec)


def eval_clip(name, chroma_w, expected_cut=28):
    cuts, total = detect_combined(os.path.join(CLIPS, name + ".mp4"), chroma_w=chroma_w)
    return dict(cuts=cuts, hit=expected_cut in cuts, total=total)


if __name__ == "__main__":
    # sanity: chroma_w=0 must match the shipped scene_detect on sample[0,900]
    base_cuts, _ = SD.find_cuts(SAMPLE, max_frames=901)
    mine0, _ = detect_combined(SAMPLE, max_frames=901, chroma_w=0.0)
    print(f"PARITY chroma_w=0 vs scene_detect.find_cuts on sample[0,900]: "
          f"{'IDENTICAL' if sorted(base_cuts)==sorted(mine0) else 'DIFFER'}")
    print(f"  shipped : {sorted(c for c in base_cuts if c<=900)}")
    print(f"  mine(w0): {sorted(c for c in mine0 if c<=900)}")

    print("\n W      sample[0,900) prec/rec  fp           miss          c7@28  c5@28  c5b@28")
    for w in [0.0, 0.3, 0.5, 0.7, 1.0]:
        s = eval_sample(w)
        c7 = eval_clip("c7_staticcut", w)
        c5 = eval_clip("c5_scenecut", w)
        c5b = eval_clip("c5b_scenecut_strong", w)
        print(f" {w:<4} prec={s['prec']:.2f} rec={s['rec']:.2f}  fp={s['fp']}  "
              f"miss={s['miss']}  {c7['hit']!s:5}  {c5['hit']!s:5}  {c5b['hit']!s:5}")
