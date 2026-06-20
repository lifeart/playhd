"""P1 step 2: export a full GOP for the WebGPU PROPAGATION-CHAIN demo + the Python reference.

Demonstrates the NEMO thesis in-browser: SR only the anchors (I-frames), reconstruct every other
frame by warping the PREVIOUS recon with that frame's codec MVs, falling back to upscaled-LR at
occlusion holes (intra blocks). Defines a faithful instant-tier subset and computes it identically
in Python (the parity reference) and (next) in WebGPU.

Per-frame recon rule (both sides, bilinear so parity is exact):
  anchor (I):  recon[i] = SR_compact(LR[i])                         # the sparse heavy step
  else (P):    w = warp_bilinear(recon[i-1], flow[i])              # cheap propagation
               recon[i] = where(hole, bilinear_up(LR[i]), w)        # fallback at intra holes

Outputs under web_spike/gop_data/:
  lr_<i>.png     LR frame (for the hole fallback)            [every frame]
  flow_<i>.bin   rg32f [fx,fy], holes => fx=-1e9 sentinel    [P frames]
  anchor_<i>.png HD compact-SR                               [I frames only]
  ref_<i>.png    HD reference recon (the parity target)      [every frame]
  meta.json      {w_lr,h_lr,scale,w_hd,h_hd,fps,frames:[{i,type,anchor}]}
"""
import os, sys, json
import numpy as np
import cv2
import av, av.sidedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
import derisk, sr

SCALE = 2
NFRAMES = 16
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gop_data")
os.makedirs(OUT, exist_ok=True)
PT = {1: "I", 2: "P", 3: "B"}
SENT = -1e9


def save_png(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    c = av.open(os.path.join(ROOT, "sample.mp4"))
    v = c.streams.video[0]
    v.codec_context.options = {"flags2": "+export_mvs"}
    fps = float(v.average_rate)
    seq = []
    for f in c.decode(video=0):
        sd = f.side_data.get(av.sidedata.sidedata.Type.MOTION_VECTORS)
        mvs = sd.to_ndarray() if sd is not None else None
        seq.append((PT.get(int(f.pict_type), "?"), f.to_ndarray(format="rgb24"), mvs))
        if len(seq) >= NFRAMES:
            break
    c.close()
    h_lr, w_lr = seq[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    # treat B-frames as P for this backbone demo (use their past MVs); first frame is forced anchor
    recon_prev = None
    meta_frames = []
    for i, (pt, lr, mvs) in enumerate(seq):
        save_png(os.path.join(OUT, f"lr_{i}.png"), lr)
        is_anchor = (pt == "I") or (recon_prev is None)
        if is_anchor:
            recon = sr.upscale_to(lr, w_hd, h_hd, model="realesrgan")
            save_png(os.path.join(OUT, f"anchor_{i}.png"), recon)
        else:
            fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want="past")
            warped, hole = derisk.warp_hd(recon_prev, fx, fy, SCALE)         # bilinear cv2.remap
            fallback = cv2.resize(lr, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)  # bilinear @ hole
            recon = np.where(hole[..., None], fallback, warped).astype(np.uint8)
            # export flow rgba32f [fx, fy, 0, 0] with NaN->sentinel (rgba32f is the P1-proven format)
            flow = np.zeros((h_lr, w_lr, 4), np.float32)
            flow[:, :, 0] = np.where(np.isnan(fx), SENT, fx)
            flow[:, :, 1] = np.nan_to_num(fy)
            flow.tofile(os.path.join(OUT, f"flow_{i}.bin"))
        save_png(os.path.join(OUT, f"ref_{i}.png"), recon)
        meta_frames.append(dict(i=i, type=pt, anchor=bool(is_anchor)))
        recon_prev = recon
    json.dump(dict(w_lr=w_lr, h_lr=h_lr, scale=SCALE, w_hd=w_hd, h_hd=h_hd,
                   fps=round(fps, 3), n=len(seq), frames=meta_frames),
              open(os.path.join(OUT, "meta.json"), "w"))
    n_anchor = sum(1 for m in meta_frames if m["anchor"])
    print(f"GOP: {len(seq)} frames, {n_anchor} anchor(s) (SR'd), {len(seq)-n_anchor} propagated; "
          f"SD {w_lr}x{h_lr} -> HD {w_hd}x{h_hd} @ {fps:.1f} fps")
    print(f"types: {' '.join(m['type']+('*' if m['anchor'] else '') for m in meta_frames)}  (*=anchor)")


if __name__ == "__main__":
    main()
