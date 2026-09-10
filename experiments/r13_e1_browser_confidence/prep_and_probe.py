#!/usr/bin/env python3
"""
R13-E1 step 1 -- extract the A/B windows for the BROWSER harness + probe the LR signals.

Two jobs:
 1) PREP: decode `web_spike/sd600.mp4` (the shipped demo clip, 640x320 H.264) with PyAV
    `+export_mvs` and dump, per frame of each window, the exact two inputs the browser
    player gets from its WASM decoder:
        rgb_<i>.bin   w*h*3 uint8   (== mvdec_rgb(),  RGB24, bit-exact vs PyAV per web_spike README)
        mv_<i>.bin    nmv*10 int32  (== mvdec_mvs(),  [source,w,h,src_x,src_y,dst_x,dst_y,mx,my,ms])
    written to a scratch dir (big, never into the repo) + a meta.json.
 2) PROBE: measure, at LR, the signals the soft gate would ride on, so the gate's
    parameters are chosen from data instead of copied blind from the server twin:
        - intra-hole fraction (the browser's `f.x < -1e8` sentinel branch)
        - the reactive residual distribution vs the shipped hard threshold tau=16
          (how many pixels actually sit in a soft ramp band -> is there anything to win?)
        - warp MAE conditioned on the FREE per-block codec signals: partition size and
          MV magnitude (the R13 brief's 8.6x / ~75x discriminators, re-measured here on
          the demo clip rather than on sample.mp4).

Browser semantics are reproduced exactly (they differ from the prototype on purpose):
    * flow is built with want="all" (the player calls buildLrFlow(mvs,h,w,"all")),
    * the reference frame is the PREVIOUS DISPLAY frame (the player chains prevHD in
      display order -- no I/P backbone; GOTCHA #12 does not apply to what ships in the
      browser, and this experiment measures what ships).

Run:  python3 prep_and_probe.py [--scratch DIR]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))

import av                       # noqa: E402
import av.sidedata              # noqa: E402
import derisk                   # noqa: E402  READ-ONLY (build_lr_flow / warp_lr)

CLIP = os.path.join(_ROOT, "web_spike", "sd600.mp4")
PT = {1: "I", 2: "P", 3: "B"}
# Same two windows R12-E1 used on this clip, so the browser numbers are comparable to the
# server-side twin's numbers (jelly = high motion, calm = talking-head).
WINDOWS = {"jelly": dict(start=582, n=40), "calm": dict(start=437, n=40)}
TAU_HARD = 16.0                  # the shipped browser threshold (player.html wU tau=16)


def decode_window(start, n):
    c = av.open(CLIP)
    v = c.streams.video[0]
    v.codec_context.options = {"flags2": "+export_mvs"}
    out = []
    for i, f in enumerate(c.decode(video=0)):
        if i < start:
            continue
        sd = f.side_data.get(av.sidedata.sidedata.Type.MOTION_VECTORS)
        mvs = sd.to_ndarray() if sd is not None else None
        out.append((PT.get(int(f.pict_type), "?"), f.to_ndarray(format="rgb24"), mvs))
        if len(out) >= n:
            break
    c.close()
    return out


def pack_mvs(mvs):
    """structured AVMotionVector array -> the packed int32 layout mv_decode.c emits."""
    if mvs is None or len(mvs) == 0:
        return np.zeros((0, 10), np.int32)
    cols = ["source", "w", "h", "src_x", "src_y", "dst_x", "dst_y",
            "motion_x", "motion_y", "motion_scale"]
    return np.stack([mvs[c].astype(np.int32) for c in cols], axis=1)


def block_stats(mvs, lr_cur, lr_prev, h, w):
    """warp MAE per block, bucketed by partition size and MV magnitude (free signals)."""
    if mvs is None or len(mvs) == 0:
        return []
    rows = []
    cur = lr_cur.astype(np.float32)
    prev = lr_prev.astype(np.float32)
    for r in mvs:
        bw, bh = int(r["w"]), int(r["h"])
        cx, cy = int(r["dst_x"]), int(r["dst_y"])
        ms = int(r["motion_scale"]) or 1
        dx, dy = float(r["motion_x"]) / ms, float(r["motion_y"]) / ms
        x0, y0 = max(cx - bw // 2, 0), max(cy - bh // 2, 0)
        x1, y1 = min(cx + bw // 2, w), min(cy + bh // 2, h)
        if x1 <= x0 or y1 <= y0:
            continue
        # bilinear fetch of the referenced block (same op as the shader's textureSampleLevel)
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        mapx, mapy = xs + dx, ys + dy
        pred = cv2.remap(prev, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mae = float(np.abs(cur[y0:y1, x0:x1] - pred).mean())
        rows.append((bw * bh, np.hypot(dx, dy), mae, int(r["source"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=os.environ.get("R13_SCRATCH", "/tmp/r13_e1_data"))
    args = ap.parse_args()

    summary = {}
    for name, wcfg in WINDOWS.items():
        seq = decode_window(wcfg["start"], wcfg["n"])
        h, w = seq[0][1].shape[:2]
        d = os.path.join(args.scratch, name)
        os.makedirs(d, exist_ok=True)
        frames = []
        holefr, band, hardfr, reactmeans = [], [], [], []
        blocks = []
        for i, (pt, rgb, mvs) in enumerate(seq):
            rgb.astype(np.uint8).tofile(os.path.join(d, f"rgb_{i}.bin"))
            packed = pack_mvs(mvs)
            packed.tofile(os.path.join(d, f"mv_{i}.bin"))
            frames.append(dict(i=i, type=pt, nmv=int(packed.shape[0])))
            if i == 0 or packed.shape[0] == 0:
                continue
            lr_prev, lr_cur = seq[i - 1][1], rgb
            fx, fy = derisk.build_lr_flow(mvs, h, w, want="all")
            hole = ~np.isfinite(fx)
            pred = derisk.warp_lr(lr_prev, fx, fy).astype(np.float32)
            react = np.abs(lr_cur.astype(np.float32) - pred).mean(axis=2)
            react[hole] = np.nan
            holefr.append(float(hole.mean()))
            hardfr.append(float((hole | (np.nan_to_num(react, nan=1e9) > TAU_HARD)).mean()))
            ok = ~hole
            reactmeans.append(float(np.nanmean(react)))
            # "ramp band": pixels a soft sigmoid (s=5) would treat as ambiguous, i.e. within +-2s
            band.append(float(((react > TAU_HARD - 10) & (react < TAU_HARD + 10) & ok).mean()))
            blocks += block_stats(mvs, lr_cur, lr_prev, h, w)

        json.dump(dict(w=w, h=h, n=len(seq), start=wcfg["start"], frames=frames),
                  open(os.path.join(d, "meta.json"), "w"))

        b = np.array([(a, m, e) for a, m, e, _ in blocks], np.float32)
        full = b[b[:, 0] == 256]
        sub = b[b[:, 0] < 256]
        mv_lo = b[b[:, 1] < 1.0]
        mv_mid = b[(b[:, 1] >= 1.0) & (b[:, 1] < 4.0)]
        mv_hi = b[b[:, 1] >= 4.0]
        s = dict(
            window=name, frames=len(seq), res=f"{w}x{h}",
            hole_frac_mean=round(float(np.mean(holefr)), 4),
            hard_fallback_frac_mean=round(float(np.mean(hardfr)), 4),
            react_mean_codes=round(float(np.mean(reactmeans)), 3),
            ramp_band_frac_mean=round(float(np.mean(band)), 4),
            blocks=len(b),
            mae_full16x16=round(float(full[:, 2].mean()), 3) if len(full) else None,
            mae_sub16=round(float(sub[:, 2].mean()), 3) if len(sub) else None,
            frac_sub16=round(float(len(sub) / max(1, len(b))), 4),
            mae_mv_lt1=round(float(mv_lo[:, 2].mean()), 3) if len(mv_lo) else None,
            mae_mv_1to4=round(float(mv_mid[:, 2].mean()), 3) if len(mv_mid) else None,
            mae_mv_ge4=round(float(mv_hi[:, 2].mean()), 3) if len(mv_hi) else None,
            frac_mv_ge4=round(float(len(mv_hi) / max(1, len(b))), 4),
        )
        summary[name] = s
        print(json.dumps(s, indent=2))
    json.dump(summary, open(os.path.join(_HERE, "probe_lr.json"), "w"), indent=2)
    print("wrote", os.path.join(_HERE, "probe_lr.json"), "| data ->", args.scratch)


if __name__ == "__main__":
    main()
