#!/usr/bin/env python3
"""R12-E2 PROBE: validate exact bitstream QP extraction against KNOWN libx264 CRF.

Encodes a real sample.mp4 window at the SAME resolution the pipeline actually
deblocks (the LR: GT/2 = 320x160) across a CRF sweep, then extracts per-frame QP
via qp_extract.qp_per_frame (method a, venc_params). Reports:
  * mean/median per-frame QP vs CRF  -> is the signal monotonic & separating?
  * the DCT-histogram fallback (method c) on the same clips -> accuracy vs (a).
Also records exact QP=+CRF encodes (x264 `-qp`) as a ground-truth anchor: when you
encode with a FIXED qp, the decoder should report exactly that per-MB QP.
"""
import os, sys, io, json
import numpy as np
import cv2
import av

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import qp_extract as qe

SAMPLE = os.path.join(ROOT, "sample.mp4")
CLIPS = os.path.join(HERE, "clips")
os.makedirs(CLIPS, exist_ok=True)

N = int(os.environ.get("N_FRAMES", "40"))
START = int(os.environ.get("START", "5000"))     # talking-head window (demo content)
CRF_SWEEP = [18, 23, 28, 33, 38, 43]
QP_FIXED = [16, 22, 28, 34, 40]                   # exact -qp encodes = ground truth


def decode_window(path, start, n):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def encode(frames, path, w, h, opts, fps=25):
    """Encode RGB frames (downscaled to w x h) with the given libx264 options."""
    c = av.open(path, "w")
    st = c.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = dict(opts)
    for fr in frames:
        img = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        for p in st.encode(vf):
            c.mux(p)
    for p in st.encode():
        c.mux(p)
    c.close()


def main():
    gt = decode_window(SAMPLE, START, N)
    if not gt:
        print("NO FRAMES"); return
    H, W = gt[0].shape[:2]
    w, h = W // 2, H // 2      # the LR resolution the pipeline deblocks
    print(f"[probe] window start={START} n={len(gt)}  GT {W}x{H} -> LR {w}x{h}")
    print(f"[probe] ffmpeg via PyAV {av.__version__}  libavcodec {av.library_versions['libavcodec']}\n")

    results = {"env": {"pyav": av.__version__, "libavcodec": av.library_versions["libavcodec"]},
               "crf": [], "fixed_qp": []}

    # -------- exact fixed-QP encodes: decoder must report ~exactly that QP --------
    print("=== FIXED -qp encodes (ground truth: decoded QP should == encode QP) ===")
    print(f"{'set_qp':>6} | {'decQP mean':>10} {'decQP med':>9} {'base_qp':>8} {'dctQP':>7}")
    for q in QP_FIXED:
        p = os.path.join(CLIPS, f"fixedqp{q}.mp4")
        encode(gt, p, w, h, {"qp": str(q), "preset": "medium", "g": "24",
                             "x264-params": "aq-mode=0"})
        recs = qe.qp_per_frame(p, max_frames=N, want_rgb=True)
        m = np.mean([r["qp_mean"] for r in recs])
        md = np.median([r["qp_median"] for r in recs])
        bq = np.median([r["base_qp"] for r in recs])
        dct = np.mean([qe.dct_qp_estimate(r["rgb"]) for r in recs])
        print(f"{q:>6} | {m:>10.2f} {md:>9.1f} {bq:>8.0f} {dct:>7.1f}")
        results["fixed_qp"].append(dict(set_qp=q, dec_qp_mean=float(m), dec_qp_med=float(md),
                                        base_qp=float(bq), dct_qp=float(dct)))

    # -------- CRF sweep: the real distribution the gate must separate --------
    print(f"\n=== CRF sweep (real content @ {w}x{h}, aq-mode default ON) ===")
    print(f"{'CRF':>4} | {'QP mean':>8} {'QP med':>7} {'QP p90':>7} {'base':>5} "
          f"{'dctQP':>7} | {'kB':>6} {'I/P/B':>7}")
    for crf in CRF_SWEEP:
        p = os.path.join(CLIPS, f"crf{crf}.mp4")
        encode(gt, p, w, h, {"crf": str(crf), "preset": "medium", "g": "24"})
        nbytes = os.path.getsize(p)
        recs = qe.qp_per_frame(p, max_frames=N, want_rgb=True)
        qm = np.mean([r["qp_mean"] for r in recs])
        qmd = np.median([r["qp_median"] for r in recs])
        qp90 = np.mean([r["qp_p90"] for r in recs])
        bq = np.median([r["base_qp"] for r in recs])
        dct = np.mean([qe.dct_qp_estimate(r["rgb"]) for r in recs])
        ptypes = "".join(r["pict_type"] for r in recs)
        npb = f"{ptypes.count('I')}/{ptypes.count('P')}/{ptypes.count('B')}"
        print(f"{crf:>4} | {qm:>8.2f} {qmd:>7.1f} {qp90:>7.1f} {bq:>5.0f} "
              f"{dct:>7.1f} | {nbytes/1000:>6.1f} {npb:>7}")
        results["crf"].append(dict(crf=crf, qp_mean=float(qm), qp_med=float(qmd),
                                   qp_p90=float(qp90), base_qp=float(bq),
                                   dct_qp=float(dct), bytes=int(nbytes), ptypes=npb))

    with open(os.path.join(HERE, "probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n[probe] wrote probe_results.json")

    # correlation summary
    crf = np.array([r["crf"] for r in results["crf"]], float)
    qmean = np.array([r["qp_mean"] for r in results["crf"]], float)
    dctv = np.array([r["dct_qp"] for r in results["crf"]], float)
    print(f"\n[probe] Pearson r(CRF, bitstream-QP) = {np.corrcoef(crf, qmean)[0,1]:.4f}")
    print(f"[probe] Pearson r(CRF, DCT-QP)       = {np.corrcoef(crf, dctv)[0,1]:.4f}")
    # fixed-qp accuracy
    fq = results["fixed_qp"]
    err = np.mean([abs(r["dec_qp_mean"] - r["set_qp"]) for r in fq])
    print(f"[probe] fixed-qp: mean |decodedQP - setQP| = {err:.2f} (venc_params exactness)")


if __name__ == "__main__":
    main()
