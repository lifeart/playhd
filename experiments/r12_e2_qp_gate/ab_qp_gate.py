#!/usr/bin/env python3
"""R12-E2 A/B: QP-GATED deblock_pre vs always-on vs off, on REAL H.264 clips.

Mirrors r10_e2/run_ab.py's anchor full-reference methodology (best/center crop,
2x-down -> REAL libx264 -> x4plus, arbiter = LPIPS & DISTS & PSNR vs GT), but:
  * encodes each content window as a MULTI-FRAME clip at a CRF sweep (real GOP,
    real per-frame QP), decodes it back, and
  * extracts the EXACT per-frame bitstream QP (qp_extract, method a) alongside the
    per-frame anchor A/B, so we can (1) find the QP where deblock flips from
    win->loss and (2) simulate the QP-gate and prove it fires on heavy / skips light.

Configs compared per frame:
    OFF    = x4plus(LR)                          (the validated baseline)
    ON     = x4plus(deblock_scunet(LR))          (always deblock)
    GATED  = ON if qp_mean >= THR else OFF       (fire only on heavy compression)

Run:  N_FRAMES=6 python experiments/r12_e2_qp_gate/ab_qp_gate.py
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import torch
import av

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROTO = os.path.join(ROOT, "prototype")
sys.path.insert(0, PROTO)
sys.path.insert(0, os.path.join(ROOT, "experiments", "r10_e2_deblock_pre"))
sys.path.insert(0, HERE)
import sr                                   # prototype anchor SR (read-only)
import qp_extract as qe
from deblock_pre import blockiness

DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SAMPLE = os.path.join(ROOT, "sample.mp4")
CLIPS = os.path.join(HERE, "clips")
os.makedirs(CLIPS, exist_ok=True)
SCUNET = os.path.join(ROOT, "experiments", "r10_e2_deblock_pre", "models",
                      "scunet_color_real_psnr.pth")

WINDOWS = {                # content type -> (start frame, smooth?)
    "talkinghead": (5000, True),    # smooth face  (deblock-favorable, demo content)
    "highmotion":  (0,    True),    # title card / low detail
    "texture18k":  (18000, False),  # news headline
    "texture24k":  (24000, False),  # chart + text
    "texture46k":  (46000, False),  # dense textured photo (deblock-UNfavorable)
}
CRF_SWEEP = [23, 27, 31, 35, 39]
N_FRAMES = int(os.environ.get("N_FRAMES", "6"))
CROP = 256


# --------------------------------------------------------------------------- #
def var_lap(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def best_crop_xy(rgb, c=CROP, stride=32):
    H, W = rgb.shape[:2]
    best, bxy = -1.0, (0, 0)
    for y in range(0, H - c + 1, stride):
        for x in range(0, W - c + 1, stride):
            v = var_lap(rgb[y:y + c, x:x + c])
            if v > best:
                best, bxy = v, (x, y)
    return bxy


def center_xy(rgb, c=CROP):
    H, W = rgb.shape[:2]
    return (W - c) // 2, max(0, (H - c) // 3)


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


def encode_clip(crops128, path, crf, fps=25):
    """Encode the 128px LR crop sequence as one real libx264 clip @ CRF."""
    h, w = crops128[0].shape[:2]
    c = av.open(path, "w")
    st = c.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": str(crf), "preset": "medium", "g": "24", "tune": "film"}
    for fr in crops128:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24")
        for p in st.encode(vf):
            c.mux(p)
    for p in st.encode():
        c.mux(p)
    c.close()


_SCU = {}
def scunet():
    if "n" not in _SCU:
        from spandrel import ModelLoader
        md = ModelLoader(device=DEV).load_from_file(SCUNET)
        _SCU["n"] = md.model.eval().to(DEV)
    return _SCU["n"]


@torch.no_grad()
def deblock(rgb):
    net = scunet()
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(DEV).permute(2, 0, 1).unsqueeze(0).float().div_(255.)
    o = net(t).clamp_(0, 1).mul_(255.).round_()
    return np.ascontiguousarray(o.squeeze(0).permute(1, 2, 0).to("cpu", torch.uint8).numpy())


def x4(rgb):
    return sr.upscale(rgb, model="realesrgan-x4plus")


def r256(rgb512):
    return cv2.resize(rgb512, (CROP, CROP), interpolation=cv2.INTER_AREA)


def to_t(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb)).to(DEV).permute(2, 0, 1).unsqueeze(0).float().div_(255.)


# --------------------------------------------------------------------------- #
def main():
    import pyiqa
    lpips = pyiqa.create_metric("lpips", device=DEV)
    dists = pyiqa.create_metric("dists", device=DEV)
    psnr = pyiqa.create_metric("psnr", device=DEV)
    sr.load_model("realesrgan-x4plus")
    warm = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    x4(warm); deblock(warm)
    print(f"[ab] dev={DEV} n_frames={N_FRAMES} windows={list(WINDOWS)} crf={CRF_SWEEP}")

    records = []
    for win, (start, smooth) in WINDOWS.items():
        frames = decode_window(SAMPLE, start, N_FRAMES)
        if not frames:
            print(f"[ab] {win}: no frames @ {start}"); continue
        x0, y0 = (center_xy(frames[0]) if smooth else best_crop_xy(frames[0]))
        gtc = [np.ascontiguousarray(f[y0:y0 + CROP, x0:x0 + CROP]) for f in frames]
        lrc = [np.ascontiguousarray(cv2.resize(g, (CROP // 2, CROP // 2),
                                               interpolation=cv2.INTER_AREA)) for g in gtc]
        gt_t = [to_t(g) for g in gtc]
        for crf in CRF_SWEEP:
            clip = os.path.join(CLIPS, f"ab_{win}_crf{crf}.mp4")
            encode_clip(lrc, clip, crf)
            recs = qe.qp_per_frame(clip, max_frames=len(lrc), want_rgb=True)
            for i, r in enumerate(recs):
                lr = np.ascontiguousarray(r["rgb"])          # decoded LR w/ real artifacts
                off512 = x4(lr)
                on512 = x4(deblock(lr))
                off, on = r256(off512), r256(on512)
                ot, nt = to_t(off), to_t(on)
                g = gt_t[i]
                rec = dict(window=win, smooth=smooth, crf=crf, frame=i,
                           qp_mean=r["qp_mean"], qp_base=r["base_qp"], qp_med=r["qp_median"],
                           pict=r["pict_type"], blockiness=blockiness(lr),
                           gt_varlap=var_lap(gtc[i]),
                           off_lpips=float(lpips(ot, g).item()), on_lpips=float(lpips(nt, g).item()),
                           off_dists=float(dists(ot, g).item()), on_dists=float(dists(nt, g).item()),
                           off_psnr=float(psnr(ot, g).item()), on_psnr=float(psnr(nt, g).item()))
                records.append(rec)
                del ot, nt
            torch.mps.empty_cache()
        print(f"[ab] {win} done ({len(frames)}f x {len(CRF_SWEEP)} crf)")
        del gt_t; torch.mps.empty_cache()

    with open(os.path.join(HERE, "ab_results.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(f"[ab] wrote {len(records)} records -> ab_results.json")


if __name__ == "__main__":
    main()
