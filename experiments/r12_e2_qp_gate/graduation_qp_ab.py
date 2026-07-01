#!/usr/bin/env python3
"""R12-E2 GRADUATION A/B (mirrors r10_e2/graduation_ab.py): does the EXACT-QP-GATED
deblock_pre, run through the REAL process_clip pipeline, beat both always-OFF and
always-ON, and does the gate correctly FIRE on a heavy clip / SKIP a light clip?

Protocol (degrade-restore):
  GT   = N-frame sample.mp4 window @ native 640x320 (pseudo-HD ground truth)
  LR   = GT/2 (320x160) -> REAL libx264 CRF{23 light, 40 heavy} -> the pipeline input
  run  = quality mode (x4plus 4x -> downscale 2x -> 640x320), GRAIN OFF, deblock in
         {OFF, ON(always), GATED(qp_mean>=THR)} -> score full clip vs GT (LPIPS/DISTS/tOF)
The QP is the EXACT per-frame bitstream QP (venc_params) plumbed via exp_qp_plumbing.

Run:  N=40 python experiments/r12_e2_qp_gate/graduation_qp_ab.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import av

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
sys.path.insert(0, os.path.join(HERE, "patch_src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "r10_e2_deblock_pre"))

import pipeline_api as pipe
import derisk
import exp_qp_plumbing as qpp
import deblock_pre

qpp.install()   # QP-carrying stream_gops + build_perframe_cache (exact bitstream QP)

N = int(os.environ.get("N", "40"))
START = int(os.environ.get("START", "5000"))     # talking-head (demo content)
THR = int(os.environ.get("THR", "40"))            # calibrated qp_mean gate
WORK = os.path.join(HERE, "out")
os.makedirs(WORK, exist_ok=True)
MODEL = "scunet_color_real_psnr.pth"

CONFIGS = {
    "off":   None,
    "on":    {"model": MODEL, "gate": "always"},
    "gated": {"model": MODEL, "gate": "qp", "qp_min": THR},
}

# ---- instrument the deblock to count fires + log QP seen ----
_FIRE = {"apply": 0, "fired": 0, "qps": []}
_orig_apply = deblock_pre.apply
def _counting_apply(rgb_lr, cfg, qp=None):
    if cfg:
        _FIRE["apply"] += 1
        _FIRE["qps"].append(qp)
    out = _orig_apply(rgb_lr, cfg, qp=qp)
    if cfg and out is not rgb_lr:
        _FIRE["fired"] += 1
    return out
deblock_pre.apply = _counting_apply


def decode(path, n=None):
    c = av.open(path); out = []
    for f in c.decode(video=0):
        out.append(f.to_ndarray(format="rgb24"))
        if n and len(out) >= n:
            break
    c.close()
    return out


def encode_crf(frames, path, w, h, crf, fps=25):
    c = av.open(path, "w")
    st = c.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": str(crf), "preset": "medium", "g": "24"}
    for fr in frames:
        img = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        for p in st.encode(vf):
            c.mux(p)
    for p in st.encode():
        c.mux(p)
    c.close()


def run_quality(deblock_cfg, lr_path, tag):
    cfg = pipe.MODE_CONFIG["quality"]
    g0, d0 = cfg["grain"], cfg.get("deblock_pre")
    cfg["grain"] = "off"
    cfg["deblock_pre"] = deblock_cfg
    out = os.path.join(WORK, f"_grad_{tag}.mp4")
    try:
        pipe.process_clip(lr_path, "quality", max_frames=N, out_path=out, detect_cuts=True)
    finally:
        cfg["grain"], cfg["deblock_pre"] = g0, d0
    return out


def to_t(rgb, dev):
    import torch
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float().div_(255.).to(dev)


def score(out_path, gt, W, H, dev, lpips, dists):
    fr = decode(out_path)
    n = min(len(fr), len(gt))
    sr = [cv2.resize(fr[i], (W, H), interpolation=cv2.INTER_AREA) for i in range(n)]
    gtt = [to_t(gt[i], dev) for i in range(n)]
    srt = [to_t(sr[i], dev) for i in range(n)]
    lp = float(np.mean([lpips(srt[i], gtt[i]).item() for i in range(n)]))
    ds = float(np.mean([dists(srt[i], gtt[i]).item() for i in range(n)]))
    tof = derisk.tof([cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in sr],
                     [cv2.cvtColor(gt[i], cv2.COLOR_RGB2BGR) for i in range(n)])
    return dict(lpips=lp, dists=ds, tof=tof, n=n)


def main():
    import torch, pyiqa
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    lpips = pyiqa.create_metric("lpips", device=dev)
    dists = pyiqa.create_metric("dists", device=dev)

    cap = cv2.VideoCapture(os.path.join(ROOT, "sample.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    gt = []
    for _ in range(N):
        ok, bgr = cap.read()
        if not ok:
            break
        gt.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    H, W = gt[0].shape[:2]
    print(f"[grad-qp] N={len(gt)}  GT {W}x{H}  THR(qp_mean)>={THR}  (quality, grain OFF)")

    allres = {}
    for clab, crf in [("LIGHT", 23), ("HEAVY", 40)]:
        lr = os.path.join(WORK, f"_grad_lr_crf{crf}.mp4")
        encode_crf(gt, lr, W // 2, H // 2, crf=crf)
        # measure the clip's actual QP for the header
        import qp_extract as qe
        clipqp = qe.mean_qp_stream(lr, max_frames=N)
        print(f"\n=== {clab} clip: CRF{crf}  measured clip QP(median-of-means)={clipqp:.1f} ===")
        res = {}
        for tag, cfg in CONFIGS.items():
            _FIRE["apply"] = _FIRE["fired"] = 0; _FIRE["qps"] = []
            out = run_quality(cfg, lr, f"{clab}_{tag}")
            r = score(out, gt, W, H, dev, lpips, dists)
            qps = [q for q in _FIRE["qps"] if q is not None]
            r["anchors_seen"] = _FIRE["apply"]
            r["deblocked"] = _FIRE["fired"]
            r["anchor_qp_mean"] = float(np.mean(qps)) if qps else float("nan")
            res[tag] = r
            print(f"  [{tag:5}] LPIPS={r['lpips']:.4f} DISTS={r['dists']:.4f} tOF={r['tof']:.4f} "
                  f"| anchors={r['anchors_seen']} deblocked={r['deblocked']} "
                  f"anchorQP~{r['anchor_qp_mean']:.1f}")
            torch.mps.empty_cache()
        allres[clab] = res
        o, on, g = res["off"], res["on"], res["gated"]
        print(f"  --> gated vs off:  LPIPS {100*(g['lpips']-o['lpips'])/o['lpips']:+.1f}%  "
              f"DISTS {100*(g['dists']-o['dists'])/o['dists']:+.1f}%")
        print(f"      on    vs off:  LPIPS {100*(on['lpips']-o['lpips'])/o['lpips']:+.1f}%  "
              f"DISTS {100*(on['dists']-o['dists'])/o['dists']:+.1f}%")

    print("\n=== VERDICT ===")
    L, H_ = allres["LIGHT"], allres["HEAVY"]
    light_skips = L["gated"]["deblocked"] == 0
    heavy_fires = H_["gated"]["deblocked"] > 0
    gated_light_ok = L["gated"]["lpips"] <= L["off"]["lpips"] * 1.005
    gated_heavy_win = (H_["gated"]["lpips"] <= H_["off"]["lpips"] and
                       H_["gated"]["dists"] <= H_["off"]["dists"])
    print(f"  gate SKIPS light (0 deblocks): {light_skips}   (light gated==off quality: {gated_light_ok})")
    print(f"  gate FIRES heavy ({H_['gated']['deblocked']} deblocks): {heavy_fires}   "
          f"(heavy gated beats/ties off on LPIPS&DISTS: {gated_heavy_win})")


if __name__ == "__main__":
    main()
