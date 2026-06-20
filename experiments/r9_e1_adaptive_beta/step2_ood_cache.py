#!/usr/bin/env python3
"""R9-E1 step 2 (GPU, small): build the OUT-OF-DISTRIBUTION cells -- a REAL libx264
codec round-trip (PyAV) over a CRF range, on smooth + textured windows. These are the
falsifier for a per-clip beta selector: R8-E3 found that on a MILD real-H.264 smooth
clip x4plus is fine and beta<0.85 REGRESSES (opposite of the synthetic heavy operators
where the smooth face wants beta=0.5). A no-reference estimator must therefore push
beta DOWN on heavy-synthetic-smooth but KEEP it high on mild-H.264-smooth. We cache the
compact/x4plus/lr/gt arrays so step3 can sweep beta + compute the same signals offline.

GPU(MPS) shared with a sibling -> small windows, empty_cache between models.
"""
import os, sys, io, json, time, warnings
warnings.filterwarnings("ignore")
import av, cv2, numpy as np, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r6_e1_srdecision"))
import sr as SR                                   # noqa: E402
from run_matrix import decode_window, SAMPLE, free_gpu  # noqa: E402

CACHE = os.path.join(_HERE, "ood_cache")
N = 8
# (window, start_frame, crf) -- smooth face across a SEVERITY range (the flip), + textured controls
JOBS = [
    ("talkinghead", 5000, 18),   # near-lossless smooth  -> expect beta high (x4plus fine)
    ("talkinghead", 5000, 26),   # mild  (R8-E3 OOD)     -> beta high
    ("talkinghead", 5000, 32),   # mid   (R8-E3 OOD)     -> beta high-ish
    ("talkinghead", 5000, 40),   # heavy real codec      -> beta maybe lower
    ("highmotion",  0,    28),   # smooth low-detail control
    ("texture24k",  24000, 26),  # textured control (R8-E3 OOD)
    ("texture24k",  24000, 32),  # textured control (R8-E3 OOD)
    ("texture46k",  46000, 32),  # textured control
]


def h264_roundtrip(lr_frames, crf):
    h, w = lr_frames[0].shape[:2]
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="mp4")
    st = out.add_stream("libx264", rate=25)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": str(crf), "preset": "medium"}
    for f in lr_frames:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
        for pkt in st.encode(frame):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    buf.seek(0)
    dec = av.open(buf, mode="r")
    vs = dec.streams.video[0]
    res = [fr.to_ndarray(format="rgb24") for fr in dec.decode(vs)]
    dec.close()
    return res[:len(lr_frames)]


def restore(lr, w, h, model):
    name = "realesrgan" if model == "compact" else "realesrgan-x4plus"
    return SR.upscale_to(lr, w, h, model=name, half=False)


def main():
    os.makedirs(CACHE, exist_ok=True)
    # decode the unique windows once
    starts = {w: s for (w, s, _c) in JOBS}
    gts = {w: decode_window(SAMPLE, s, N) for w, s in starts.items()}
    # build LR per job (deterministic codec)
    lr_jobs = {}
    for (w, s, crf) in JOBS:
        gt = gts[w]
        h, wd = gt[0].shape[:2]
        lr_clean = [cv2.resize(g, (wd // 2, h // 2), interpolation=cv2.INTER_AREA) for g in gt]
        lr_jobs[(w, crf)] = h264_roundtrip(lr_clean, crf)
    # run model-by-model to bound MPS
    out = {}
    for model in ("compact", "x4plus"):
        t0 = time.perf_counter()
        for (w, s, crf) in JOBS:
            gt = gts[w]; h, wd = gt[0].shape[:2]
            lr = lr_jobs[(w, crf)]
            res = [restore(l, wd, h, model) for l in lr]
            out.setdefault((w, crf), {})[model] = np.stack(res)
        free_gpu({"compact": "realesrgan", "x4plus": "realesrgan-x4plus"}[model])
        print(f"  [{model:8s}] done {time.perf_counter()-t0:.1f}s")
    for (w, s, crf) in JOBS:
        gt = np.stack(gts[w]); lr = np.stack(lr_jobs[(w, crf)])
        d = out[(w, crf)]
        np.savez_compressed(os.path.join(CACHE, f"{w}_crf{crf}.npz"),
                            gt=gt, lr=lr, compact=d["compact"], x4plus=d["x4plus"])
        print(f"  cached {w}_crf{crf}")
    json.dump([(w, crf) for (w, s, crf) in JOBS], open(os.path.join(CACHE, "jobs.json"), "w"))
    print("[done] OOD cache ->", CACHE)


if __name__ == "__main__":
    main()
