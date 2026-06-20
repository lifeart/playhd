"""P1 step 1: export real data for the WebGPU MV-warp demo + the Python reference for parity.

The architecture's per-frame core is: warp an SR'd anchor by the NEXT frame's codec MVs to
reconstruct the next frame (cheap propagation instead of per-frame SR). This exports one real
propagation step from sample.mp4 so a browser WebGPU shader can reproduce the warp and be
parity-checked against the prototype's `warp_hd` (the validated cv2.remap path).

Outputs (under web_spike/demo_data/):
  anchor.png      HD SR'd reference frame (compact net), the texture the GPU warps
  flow.bin        LR fetch-flow, rgba32f [fx, fy, 0, 0] with NaN->0 (matches _remap's nan_to_num)
  ref_warp.png    prototype warp_hd(anchor, flow) -- the parity target
  meta.json       {w_lr, h_lr, scale, w_hd, h_hd}
"""
import os, sys, json
import numpy as np
import cv2
import av, av.sidedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
import derisk, sr

SCALE = 2                      # instant tier = 720p (x2). (The x4 path was separately verified =0.000.)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data")
os.makedirs(OUT, exist_ok=True)
PT = {1: "I", 2: "P", 3: "B"}


def main():
    c = av.open(os.path.join(ROOT, "sample.mp4"))
    v = c.streams.video[0]
    v.codec_context.options = {"flags2": "+export_mvs"}
    frames = []   # (ptype, rgb, mvs)
    for f in c.decode(video=0):
        sd = f.side_data.get(av.sidedata.sidedata.Type.MOTION_VECTORS)
        mvs = sd.to_ndarray() if sd is not None else None
        frames.append((PT.get(int(f.pict_type), "?"), f.to_ndarray(format="rgb24"), mvs))
        if len(frames) >= 60:
            break
    c.close()

    # pick a consecutive (anchor=k, target=k+1) where k+1 is a P-frame WITH past MVs
    k = None
    for i in range(1, len(frames)):
        pt, _, mvs = frames[i]
        if pt == "P" and mvs is not None and len(mvs) > 0:
            past = sum(1 for r in mvs if int(r["source"]) < 0)
            if past > 100:
                k = i - 1
                break
    if k is None:
        raise RuntimeError("no suitable P-frame-with-MVs found in first 60 frames")
    anchor_lr = frames[k][1]
    target_mvs = frames[k + 1][2]
    h_lr, w_lr = anchor_lr.shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    print(f"anchor=frame{k} ({frames[k][0]}), target=frame{k+1} (P, {len(target_mvs)} MVs); "
          f"SD {w_lr}x{h_lr} -> HD {w_hd}x{h_hd} (x{SCALE})")

    # SR the anchor (compact net) -> HD reference texture
    anchor_hd = sr.upscale_to(anchor_lr, w_hd, h_hd, model="realesrgan")
    cv2.imwrite(os.path.join(OUT, "anchor.png"), cv2.cvtColor(anchor_hd, cv2.COLOR_RGB2BGR))

    # LR fetch-flow from the target frame's PAST MVs (the prototype's exact build)
    fx, fy = derisk.build_lr_flow(target_mvs, h_lr, w_lr, want="past")
    flow = np.zeros((h_lr, w_lr, 4), np.float32)
    flow[:, :, 0] = np.nan_to_num(fx)      # nan->0 == _remap's nan_to_num (hole -> identity sample)
    flow[:, :, 1] = np.nan_to_num(fy)
    flow.tofile(os.path.join(OUT, "flow.bin"))

    # prototype reference warp (the parity target)
    ref_warp, hole = derisk.warp_hd(anchor_hd, fx, fy, SCALE)
    cv2.imwrite(os.path.join(OUT, "ref_warp.png"), cv2.cvtColor(ref_warp, cv2.COLOR_RGB2BGR))

    json.dump(dict(w_lr=w_lr, h_lr=h_lr, scale=SCALE, w_hd=w_hd, h_hd=h_hd,
                   n_mvs=int(len(target_mvs)), hole_frac=float(hole.mean())),
              open(os.path.join(OUT, "meta.json"), "w"))
    print(f"wrote demo_data/: anchor.png, flow.bin ({flow.nbytes} B), ref_warp.png, meta.json")
    print(f"hole fraction (intra, no past MV): {hole.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
