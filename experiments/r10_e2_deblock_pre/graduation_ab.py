"""R10-E2 GRADUATION A/B — does the deblock anchor win SURVIVE PROPAGATION + stay stable?

R10-E2 measured deblock on the ANCHOR (crops vs GT). To flip default-ON we must confirm the
win survives MV-propagation to the WHOLE clip (anchors are only ~2-12% of frames) and does not
add temporal flicker. This runs the REAL quality pipeline (process_clip) on a heavily-compressed
(CRF35) degraded clip, deblock ON vs OFF, GRAIN OFF (isolate), and scores the FULL propagated
output vs a pseudo-HD GT with LPIPS + DISTS (pyiqa) + tOF.

Protocol (degrade-restore, R6-E1/R8-E3 convention):
  GT       = N-frame sample.mp4 window at native 640x320 (pseudo-HD ground truth)
  LR       = GT downscaled 2x (320x160) -> REAL libx264 CRF35 encode -> the heavily-compressed input
  restore  = quality mode (x4plus 4x -> 1280x640) deblock {OFF, ON} -> downscale 2x -> 640x320 -> vs GT
Decision: deblock ON must beat OFF on LPIPS AND DISTS over ALL frames (win survives propagation)
AND not raise tOF (temporally safe) -> QP/proxy-gated default-ON candidate. Else keep default-OFF.

Run: python experiments/r10_e2_deblock_pre/graduation_ab.py [N]
"""
import os, sys
import numpy as np
import cv2
import av

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
import pipeline_api as pipe
import derisk

N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
WORK = os.path.join(ROOT, "server", "outputs")
LR_PATH = os.path.join(WORK, "_grad_lr_crf35.mp4")
DEBLOCK_CFG = {"model": "scunet_color_real_psnr.pth", "gate": "blockiness", "block_min": 1.30}


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


def run_quality(deblock_cfg, tag):
    cfg = pipe.MODE_CONFIG["quality"]
    g0, d0 = cfg["grain"], cfg.get("deblock_pre")
    cfg["grain"] = "off"
    cfg["deblock_pre"] = deblock_cfg
    out = os.path.join(WORK, f"_grad_quality_{tag}.mp4")
    try:
        pipe.process_clip(LR_PATH, "quality", max_frames=N, out_path=out, detect_cuts=True)
    finally:
        cfg["grain"], cfg["deblock_pre"] = g0, d0
    return out


def to_t(rgb, dev):
    import torch
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float().div_(255.).to(dev)


def main():
    import torch, pyiqa
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    lpips = pyiqa.create_metric("lpips", device=dev)
    dists = pyiqa.create_metric("dists", device=dev)

    print(f"[grad A/B] N={N}  (quality, GRAIN OFF, deblock blockiness-gate, CRF35 input)")
    gt = decode(os.path.join(ROOT, "sample.mp4"), N)
    H, W = gt[0].shape[:2]
    print(f"  GT {W}x{H} x{len(gt)}; degrading -> {W//2}x{H//2} @ CRF35 ...")
    encode_crf(gt, LR_PATH, W // 2, H // 2, crf=35)

    res = {}
    for tag, cfg in [("off", None), ("on", DEBLOCK_CFG)]:
        out = run_quality(cfg, tag)
        fr = decode(out)
        n = min(len(fr), len(gt))
        # restore output (4x of LR = 2x GT) -> downscale to GT res
        sr = [cv2.resize(fr[i], (W, H), interpolation=cv2.INTER_AREA) for i in range(n)]
        gtt = [to_t(gt[i], dev) for i in range(n)]
        srt = [to_t(sr[i], dev) for i in range(n)]
        lp = float(np.mean([lpips(srt[i], gtt[i]).item() for i in range(n)]))
        ds = float(np.mean([dists(srt[i], gtt[i]).item() for i in range(n)]))
        # tOF: Farneback EPE of output vs GT temporal flow (derisk.tof, RGB in)
        tof = derisk.tof([cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in sr],
                         [cv2.cvtColor(gt[i], cv2.COLOR_RGB2BGR) for i in range(n)])
        res[tag] = dict(lpips=lp, dists=ds, tof=tof, n=n)
        print(f"  [{tag:3}] n={n}  LPIPS={lp:.4f}  DISTS={ds:.4f}  tOF={tof:.4f}")
        del gtt, srt
        torch.mps.empty_cache()

    o, n_ = res["off"], res["on"]
    dl = 100 * (n_["lpips"] - o["lpips"]) / o["lpips"]
    dd = 100 * (n_["dists"] - o["dists"]) / o["dists"]
    dt = 100 * (n_["tof"] - o["tof"]) / o["tof"]
    print(f"\n  deblock ON vs OFF over the FULL propagated clip:")
    print(f"    LPIPS {dl:+.1f}%   DISTS {dd:+.1f}%   tOF {dt:+.1f}%")
    win = n_["lpips"] < o["lpips"] and n_["dists"] < o["dists"]
    safe = n_["tof"] <= o["tof"] * 1.02
    print(f"\n  VERDICT: ", end="")
    if win and safe:
        print("deblock win SURVIVES propagation (LPIPS+DISTS both down) AND tOF-safe -> default-flip candidate (proxy-gated)")
    elif win and not safe:
        print("win survives but tOF worsens -> gated opt-in, not a silent default")
    else:
        print("win does NOT survive propagation -> keep default-OFF")


if __name__ == "__main__":
    main()
