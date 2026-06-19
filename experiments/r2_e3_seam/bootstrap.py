"""bootstrap.py -- R2-E3 seam experiment data prep (runs RVM ONCE, then GPU-free).

Produces, into experiments/r2_e3_seam/cache/:
  frames_lr.npy   (N,h,w,3)  uint8  -- decoded LR talking-head scene[0]=[0,32)
  phas.npy        (N,h,w)    float32 -- RVM soft alpha mattes (LR, temporal order)
  gates.npy       (N,h,w)    float32 -- binary dilated FG gates (the L2 plate gate)
  plate_raw.npy   (h,w,3)    float32 -- temporal-median BG plate WITH NaN holes (pre-fill)
  fill_mask.npy   (h,w)      bool    -- pixels that build_plate would fill (hole|nan)
  coverage.npy    (h,w)      int32

Reuses the prototype's cached HD SR (out_layered/cache/sr_5000_0_32_*.npy) and the
HD plate PNG (out_layered/plate_hd.png) at analysis time -- this script only needs the
matte (RVM, ~0.6s for 32 frames) and the cheap numpy median. Import prototype READ-ONLY.
"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.abspath(os.path.join(HERE, "..", "..", "prototype"))
sys.path.insert(0, PROTO)
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

import derisk                  # READ-ONLY
import matting                 # READ-ONLY (L1)
import background_plate as bp  # READ-ONLY (L2)

SAMPLE = os.path.join(PROTO, "..", "sample.mp4")
START, NWIN = 5000, 48


def main():
    print(f"[decode] window {NWIN}f from start {START} ...")
    decoded = derisk.decode_lr_and_mvs(SAMPLE, start_frame=START, max_frames=NWIN)
    frames_all = [img for (_p, img, _m) in decoded]
    segs = bp.scene_segments(decoded, frames=frames_all)
    s0, s1 = segs[0]
    frames = frames_all[s0:s1]
    N = len(frames)
    h, w = frames[0].shape[:2]
    print(f"[scene] {w}x{h} LR, scene[0]=[{s0},{s1}) N={N}")

    print("[matte] RVM mobilenetv3 (recurrent, temporal order, human-only) ...")
    model = matting.load_rvm("mps")
    res = matting.matte_sequence(model, frames)
    phas = [p for (_f, p) in res]
    gates = [matting.fg_mask_lr(p, lr_hw=(h, w), soft=False, thresh=0.5, dilate=3) for p in phas]
    fg_frac = float(np.mean([(p >= 0.5).mean() for p in phas]))
    print(f"[matte] FG frac {fg_frac*100:.1f}%")

    # raw temporal-median plate WITH NaN holes (reproduce build_plate accumulation so the
    # ring-fill experiments have the pre-fill plate; we do NOT edit prototype).
    stack = np.stack([np.asarray(f, np.float32) for f in frames], axis=0)
    bg = np.stack([np.asarray(g, np.float32) < 0.5 for g in gates], axis=0)
    coverage = bg.sum(axis=0).astype(np.int32)
    plate_raw = np.empty((h, w, 3), np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for y0 in range(0, h, 64):
            y1 = min(y0 + 64, h)
            sub = stack[:, y0:y1]
            bm = bg[:, y0:y1, :, None]
            masked = np.where(bm, sub, np.nan)
            plate_raw[y0:y1] = np.nanmedian(masked, axis=0)
    hole_mask = coverage < 1
    nan_pix = np.isnan(plate_raw).any(axis=2)
    fill_mask = hole_mask | nan_pix
    print(f"[plate] holes {100*hole_mask.mean():.2f}%  fill {100*fill_mask.mean():.2f}%  "
          f"cov>=1 {100*(coverage>=1).mean():.1f}%")

    np.save(os.path.join(CACHE, "frames_lr.npy"), np.stack(frames))
    np.save(os.path.join(CACHE, "phas.npy"), np.stack(phas).astype(np.float32))
    np.save(os.path.join(CACHE, "gates.npy"), np.stack(gates).astype(np.float32))
    np.save(os.path.join(CACHE, "plate_raw.npy"), plate_raw)
    np.save(os.path.join(CACHE, "fill_mask.npy"), fill_mask)
    np.save(os.path.join(CACHE, "coverage.npy"), coverage)
    print(f"[done] cached to {CACHE}")

    # free GPU for siblings
    import gc, torch
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
