#!/usr/bin/env python3
"""R8-E3 out-of-distribution operator check: does the fixed-beta blend's safety hold
under a REAL H.264 codec round-trip (not the synthetic blur+JPEG+noise chain the betas
were observed on)? This is the overfit falsifier for landing a fixed constant.

Degrade = 2x AREA downscale -> libx264 encode (PyAV, crf sweep) -> decode -> the LR.
A true H.264 transform (DCT + deblocking + chroma subsample) differs from JPEG, so if
beta=0.75 stays >= x4plus here too, the constant is not fit to the synthetic operators.
Two windows (smooth talkinghead + textured texture24k). LEAD = TRUE LPIPS.
"""
import os, sys, io, json, warnings
warnings.filterwarnings("ignore")
import av, cv2, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r5_e2_quality"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r6_e1_srdecision"))
import sr as SR              # noqa: E402
import metrics as M          # noqa: E402
from run_matrix import decode_window, SAMPLE  # noqa: E402

WINDOWS = {"talkinghead": 5000, "texture24k": 24000}
N = 8


def h264_roundtrip(lr_frames, crf):
    """Encode a list of HxWx3 RGB uint8 LR frames through libx264 (in-memory mp4) and
    decode back. Returns the decoded RGB frames (same count/size)."""
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
    for pkt in st.encode():           # flush
        out.mux(pkt)
    out.close()
    buf.seek(0)
    dec = av.open(buf, mode="r")
    vs = dec.streams.video[0]
    res = [fr.to_ndarray(format="rgb24") for fr in dec.decode(vs)]
    dec.close()
    return res[:len(lr_frames)]


def blend(c, x, b):
    return np.clip(np.round(c.astype(np.float32) + b * (x.astype(np.float32) - c.astype(np.float32))),
                   0, 255).astype(np.uint8)


def restore(lr, w, h, model):
    name = "realesrgan" if model == "compact" else "realesrgan-x4plus"
    return SR.upscale_to(lr, w, h, model=name, half=False)


def main():
    out = {}
    for wname, start in WINDOWS.items():
        gt = decode_window(SAMPLE, start, N)
        h, w = gt[0].shape[:2]
        lr_clean = [cv2.resize(g, (w // 2, h // 2), interpolation=cv2.INTER_AREA) for g in gt]
        for crf in (26, 32):
            lr = h264_roundtrip(lr_clean, crf)
            comp = [restore(l, w, h, "compact") for l in lr]
            x4 = [restore(l, w, h, "x4plus") for l in lr]
            xl = float(np.mean([M.lpips_dist(x, g) for x, g in zip(x4, gt)]))
            row = {"x4plus": xl, "compact": float(np.mean([M.lpips_dist(c, g) for c, g in zip(comp, gt)]))}
            for b in (0.50, 0.75, 0.85):
                bl = [blend(c, x, b) for c, x in zip(comp, x4)]
                row[f"fix{b:.2f}"] = float(np.mean([M.lpips_dist(r, g) for r, g in zip(bl, gt)]))
            out[f"{wname}|h264_crf{crf}"] = row
            tag = lambda k: f"{row[k]:.4f}{'*' if row[k] < xl - 1e-4 else ('=' if abs(row[k]-xl)<=1e-4 else '!')}"
            print(f"  {wname:12s} crf{crf}: x4={xl:.4f}  cmp={row['compact']:.4f}  "
                  f"fix.50={tag('fix0.50')}  fix.75={tag('fix0.75')}  fix.85={tag('fix0.85')}  "
                  f"(* beats / = ties / ! REGRESSES vs x4plus)")
    json.dump(out, open(os.path.join(_HERE, "ood_check.json"), "w"), indent=2)
    print("[done] -> ood_check.json")


if __name__ == "__main__":
    main()
