"""P1 step 4: GOP data for the LIVE pipeline (on-GPU SR anchor + warp chain) — MVs the only offline input.

Same as extract_gop_data but: a 256x256 LR crop at the net's NATIVE x4 (so the on-GPU compact SR output
feeds the warp directly, no scale mismatch), and the anchor is NOT pre-SR'd into a PNG — the browser will
SR it on-GPU. We still export the LR crop (the browser's SR input + hole fallback) and the Python recon
reference (anchor via PyTorch SR, the parity target). Reuses compact_data/weights.bin + layers.json.

Outputs under web_spike/gop_live_data/:
  lr_<i>.png    256x256 LR crop                       [every frame: SR input on anchor, fallback elsewhere]
  flow_<i>.bin  rgba32f [fx,fy,0,0], holes fx=-1e9     [P frames] (cropped flow; offsets are crop-relative)
  ref_<i>.png   1024x1024 Python recon (parity target) [every frame]
  meta.json     {w_lr,h_lr,scale,w_hd,h_hd,fps,frames:[{i,type,anchor}]}
"""
import os, sys, json
import numpy as np
import cv2
import av, av.sidedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
import derisk, sr

SCALE = 4
NFRAMES = 12
CY, CX, CS = 40, 80, 256          # crop origin (y,x) + size (matches the sr.html crop region)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gop_live_data")
os.makedirs(OUT, exist_ok=True)
PT = {1: "I", 2: "P", 3: "B"}
SENT = -1e9


def savep(p, rgb): cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    c = av.open(os.path.join(ROOT, "sample.mp4"))
    v = c.streams.video[0]; v.codec_context.options = {"flags2": "+export_mvs"}
    fps = float(v.average_rate); seq = []
    for f in c.decode(video=0):
        sd = f.side_data.get(av.sidedata.sidedata.Type.MOTION_VECTORS)
        seq.append((PT.get(int(f.pict_type), "?"), f.to_ndarray(format="rgb24"),
                    sd.to_ndarray() if sd is not None else None))
        if len(seq) >= NFRAMES: break
    c.close()
    H = W = CS; w_hd, h_hd = W * SCALE, H * SCALE
    recon_prev = None; meta_frames = []
    for i, (pt, full, mvs) in enumerate(seq):
        lr = np.ascontiguousarray(full[CY:CY + H, CX:CX + W])
        savep(os.path.join(OUT, f"lr_{i}.png"), lr)
        is_anchor = (pt == "I") or (recon_prev is None)
        if is_anchor:
            recon = sr.upscale(lr, model="realesrgan")          # x4 -> 1024x1024 (the parity target anchor)
        else:
            fxf, fyf = derisk.build_lr_flow(mvs, full.shape[0], full.shape[1], want="past")
            fx = np.ascontiguousarray(fxf[CY:CY + H, CX:CX + W])  # crop the flow (offsets are crop-relative)
            fy = np.ascontiguousarray(fyf[CY:CY + H, CX:CX + W])
            warped, hole = derisk.warp_hd(recon_prev, fx, fy, SCALE)
            fallback = cv2.resize(lr, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
            recon = np.where(hole[..., None], fallback, warped).astype(np.uint8)
            flow = np.zeros((H, W, 4), np.float32)
            flow[:, :, 0] = np.where(np.isnan(fx), SENT, fx); flow[:, :, 1] = np.nan_to_num(fy)
            flow.tofile(os.path.join(OUT, f"flow_{i}.bin"))
        savep(os.path.join(OUT, f"ref_{i}.png"), recon)
        meta_frames.append(dict(i=i, type=pt, anchor=bool(is_anchor)))
        recon_prev = recon
    json.dump(dict(w_lr=W, h_lr=H, scale=SCALE, w_hd=w_hd, h_hd=h_hd, fps=round(fps, 3),
                   n=len(seq), frames=meta_frames), open(os.path.join(OUT, "meta.json"), "w"))
    na = sum(1 for m in meta_frames if m["anchor"])
    print(f"LIVE GOP: {len(seq)} frames, {na} anchor(s) (on-GPU SR), {len(seq)-na} propagated; "
          f"crop {W}x{H} -> {w_hd}x{h_hd} (x{SCALE}) @ {fps:.0f}fps; "
          f"types: {' '.join(m['type']+('*' if m['anchor'] else '') for m in meta_frames)}")


if __name__ == "__main__":
    main()
