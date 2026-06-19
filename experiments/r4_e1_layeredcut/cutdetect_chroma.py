"""R4-E1 fix (b), surgical version: a CHROMA-DOMINANT cut test ADDED on top of the existing
luma tests (the luma path is byte-identical -> sample.mp4's luma-driven cuts AND non-cuts are
unchanged). The new test fires only on a SIMILAR-LUMA, DIFFERENT-CHROMA change (dChroma > dLuma),
which is exactly the missed-cut signature (c7: dC 34.0 > dL 26.4; c5: dC 109 > dL 47) and is NOT
shared by sample.mp4's legitimate non-cuts, which are all LUMA-dominant brightness flashes
(125: dC 29.6 < dL 34.4; 303: dC 31 < dL 43; 857: dC 29.8 < dL 38.8).

Two chroma sub-tests mirror the luma absolute/relative pair, both gated by chroma-dominance:
  (D-abs) dC > CHROMA_ABS                         and dC > CHROMA_DOM*dL  -> big color change
  (D-rel) dC > CHROMA_REL_FLOOR and dC > CHROMA_REL_MULT*base_c and dC > CHROMA_DOM*dL
base_c is a SEPARATE chroma EMA baseline (so a panning scene's high chroma baseline, c5 ~29,
self-calibrates the relative test). MIN_SCENE_LEN gating is shared with the luma path.

Run:  python3 experiments/r4_e1_layeredcut/cutdetect_chroma.py
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

# proposed chroma-dominant test constants
CHROMA_ABS = 60.0
CHROMA_REL_FLOOR = 22.0
CHROMA_REL_MULT = 6.0
CHROMA_DOM = 1.1
CHROMA_EMA = 0.30


def detect(path, max_frames=None, with_chroma=True):
    cont = av.open(path)
    vs = cont.streams.video[0]
    vs.codec_context.options = {"flags2": "+export_mvs"}
    prevY = prevU = prevV = None
    base = None       # luma EMA baseline (identical to scene_detect)
    base_c = None     # chroma EMA baseline (new)
    last_cut = 0
    cuts = []
    fired_by = {}
    idx = 0
    for fr in cont.decode(vs):
        if max_frames is not None and idx >= max_frames:
            break
        pt = {1: "I", 2: "P", 3: "B"}.get(int(fr.pict_type), "?")
        yuv = cv2.cvtColor(fr.to_ndarray(format="rgb24"), cv2.COLOR_RGB2YUV).astype(np.float32)
        Y, U, V = yuv[..., 0], yuv[..., 1], yuv[..., 2]
        if prevY is None:
            prevY, prevU, prevV = Y, U, V
            last_cut = idx
            idx += 1
            continue
        dL = float(np.abs(Y - prevY).mean())
        dC = float(np.abs(U - prevU).mean() + np.abs(V - prevV).mean())
        prevY, prevU, prevV = Y, U, V

        # --- existing luma tests (byte-identical to scene_detect) ---
        luma_raw = (dL > SD.CUT_THRESH or
                    (pt == "I" and dL > SD.IFRAME_THRESH) or
                    (base is not None and dL > SD.REL_FLOOR and dL > SD.REL_MULT * base))
        # --- new chroma-dominant tests ---
        chroma_raw = False
        if with_chroma and dC > CHROMA_DOM * dL:
            if dC > CHROMA_ABS:
                chroma_raw = True
            elif (base_c is not None and dC > CHROMA_REL_FLOOR
                  and dC > CHROMA_REL_MULT * base_c):
                chroma_raw = True

        raw = luma_raw or chroma_raw
        far = (idx - last_cut) >= SD.MIN_SCENE_LEN
        if raw and far:
            cuts.append(idx)
            fired_by[idx] = "luma" if luma_raw else "chroma"
            last_cut = idx
            base = None
            base_c = None
        else:
            base = dL if base is None else (SD.EMA_ALPHA * dL + (1 - SD.EMA_ALPHA) * base)
            base_c = dC if base_c is None else (CHROMA_EMA * dC + (1 - CHROMA_EMA) * base_c)
        idx += 1
    cont.close()
    return cuts, fired_by, idx


def eval_sample(with_chroma):
    expected = {28, 196, 341, 479, 514, 563, 630, 688, 810}
    cuts, fired, _ = detect(SAMPLE, max_frames=901, with_chroma=with_chroma)
    win = sorted(c for c in cuts if 0 <= c <= 900)
    tp = [c for c in win if c in expected]
    fp = [(c, fired[c]) for c in win if c not in expected]
    miss = [c for c in expected if c not in win]
    return dict(win=win, tp=tp, fp=fp, miss=miss,
                prec=len(tp) / max(1, len(win)), rec=len(tp) / max(1, len(expected)))


if __name__ == "__main__":
    print("=== sample.mp4 [0,900): luma-only vs +chroma-dominant ===")
    for tag, wc in [("luma-only", False), ("+chroma", True)]:
        s = eval_sample(wc)
        print(f" {tag:10}  prec={s['prec']:.2f} rec={s['rec']:.2f}  "
              f"cuts={s['win']}  FP={s['fp']}  MISS={s['miss']}")

    print("\n=== missed-cut clips (cut @ frame 28) ===")
    for name in ["c7_staticcut", "c5_scenecut", "c5b_scenecut_strong"]:
        cuts, fired, _ = detect(os.path.join(CLIPS, name + ".mp4"), with_chroma=True)
        hit = 28 in cuts
        print(f" {name:24} cuts={cuts}  fired_by={ {c:fired[c] for c in cuts} }  "
              f"hit@28={hit}")
