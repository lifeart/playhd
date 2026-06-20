#!/usr/bin/env python3
"""R8-E2 step 2 (THE CRUX): is the occlusion-fallback mask spatially CLUSTERED on HIGH-motion
frames (fast-moving object / disocclusion band) vs SCATTERED on general/talking-head footage?

The prior tile-SR NO-GO (server/pipeline_api.py INSTANT_TILE_SR comment + IMPROVEMENTS) measured
on general footage: single bbox ~97% of frame, 32x16 grid ~46% tiles touched -> no compute win.
This RE-EXAMINES that on the strongest high-motion windows (located by probe_windows.py).

For each window: decode 48 frames, build the EXACT LR occlusion-fallback mask per non-anchor frame
(anchor_sr._lr_fallback_mask, occ='reactive' = the instant default) and measure, per frame and
aggregated over the HIGH-fallback frames:
  * hole_frac (mask.mean) -- the weak-spot size
  * single padded bbox coverage = bbox_area / frame_area  (the prior "~97%" number)
  * connected components: count, largest-component fraction of fallback, fill = area/bbox_area
  * COARSE-GRID tile cost: for several grids, fraction of TILES touched (any fallback px) -> the
    area that tile-SR must super-resolve; and the densest-tile concentration (fallback px captured
    by the touched tiles). Lower touched-fraction = cheaper tile-SR = the clustering win.

READ-ONLY imports of prototype/ + server/. No shared file modified.
"""
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

import derisk as D          # noqa: E402
import anchor_sr as A       # noqa: E402

CLIP = os.path.join(_REPO, "sample.mp4")
N = 48
OCC = "reactive"
PAD = A._TILE_PAD_LR        # 12 LR px -- the deployed tile halo
GRIDS = [(4, 2), (8, 4), (16, 8), (20, 10), (32, 16)]   # cols x rows (LR-tile grids)

# windows: (label, start). Filled from probe_windows.py output.
WINDOWS = [
    ("H1_extreme(7392)", 7392),
    ("H2_fast(2352)", 2352),
    ("A_exp2(0)", 0),
    ("C_talkhead(4488)", 4488),
]


def bbox_cov(mask, pad):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0.0
    h, w = mask.shape
    x0 = max(int(xs.min()) - pad, 0); x1 = min(int(xs.max()) + 1 + pad, w)
    y0 = max(int(ys.min()) - pad, 0); y1 = min(int(ys.max()) + 1 + pad, h)
    return (x1 - x0) * (y1 - y0) / (w * h)


def cc_stats(mask):
    """Connected components of the fallback mask: count, largest-comp fraction-of-fallback,
    fill (fallback px / union of component bboxes)."""
    m = mask.astype(np.uint8)
    n_lab, lab = cv2.connectedComponents(m, connectivity=8)
    tot = int(m.sum())
    if tot == 0 or n_lab <= 1:
        return dict(n_cc=0, largest_frac=0.0, fill=0.0)
    sizes = np.bincount(lab.ravel())[1:]   # drop background label 0
    largest = int(sizes.max())
    # union of component bounding boxes (the tile-able footprint)
    bbox_area = 0
    for L in range(1, n_lab):
        ys, xs = np.where(lab == L)
        bbox_area += (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    fill = tot / max(bbox_area, 1)
    return dict(n_cc=int(n_lab - 1), largest_frac=round(largest / tot, 3), fill=round(fill, 3))


def grid_cost(mask, cols, rows):
    """Fraction of tiles TOUCHED (>=1 fallback px) -> the area tile-SR must super-resolve, and the
    fraction of fallback pixels captured by touched tiles (=1.0 by construction; we also report the
    'dense' cost: tiles needed to cover 90% of fallback px, the concentration measure)."""
    h, w = mask.shape
    th = int(np.ceil(h / rows)); tw = int(np.ceil(w / cols))
    counts = []
    touched = 0
    for ry in range(rows):
        for rx in range(cols):
            y0, y1 = ry * th, min((ry + 1) * th, h)
            x0, x1 = rx * tw, min((rx + 1) * tw, w)
            if y0 >= y1 or x0 >= x1:
                continue
            c = int(mask[y0:y1, x0:x1].sum())
            counts.append(c)
            if c > 0:
                touched += 1
    counts = np.array(sorted(counts, reverse=True))
    tot = counts.sum()
    n_tiles = len(counts)
    touched_frac = touched / n_tiles if n_tiles else 0.0
    # dense-90: how many of the densest tiles hold 90% of the fallback pixels (concentration)
    if tot > 0:
        cum = np.cumsum(counts) / tot
        n90 = int(np.searchsorted(cum, 0.90) + 1)
        dense90_area = n90 / n_tiles
    else:
        n90, dense90_area = 0, 0.0
    return dict(touched_frac=round(touched_frac, 3), touched_tiles=touched,
                n90_tiles=n90, dense90_area_frac=round(dense90_area, 3))


def main():
    out = {"config": dict(N=N, occ=OCC, pad=PAD, grids=GRIDS), "windows": {}}
    for label, start in WINDOWS:
        frames = D.decode_lr_and_mvs(CLIP, start, N)
        h_lr, w_lr = frames[0][1].shape[:2]
        anchors, backbone = A.anchor_indices(frames)
        types = "".join(f[0][0] for f in frames)

        rows = []
        for i in range(N):
            if i in anchors:
                continue
            m = A._lr_fallback_mask(frames, i, backbone, OCC)
            hf = float(m.mean())
            rec = dict(i=i, type=frames[i][0], hole_frac=round(hf, 4),
                       bbox_cov=round(bbox_cov(m, PAD), 3))
            rec.update({"cc_" + k: v for k, v in cc_stats(m).items()})
            for (c, r) in GRIDS:
                g = grid_cost(m, c, r)
                rec[f"g{c}x{r}_touched"] = g["touched_frac"]
                rec[f"g{c}x{r}_dense90"] = g["dense90_area_frac"]
            rows.append(rec)

        # aggregate over the HIGH-fallback frames (hole_frac > 0.08 -- the regime tile-SR targets)
        hi = [r for r in rows if r["hole_frac"] > 0.08]
        allnz = [r for r in rows if r["hole_frac"] > 0.0]

        def agg(rs, key):
            return round(float(np.mean([r[key] for r in rs])), 3) if rs else 0.0

        wrec = dict(start=start, types=types, anchors=sorted(anchors),
                    h_lr=h_lr, w_lr=w_lr,
                    mean_hole=agg(allnz, "hole_frac"),
                    max_hole=round(max([r["hole_frac"] for r in rows], default=0.0), 4),
                    n_hi=len(hi),
                    hi_bbox_cov=agg(hi, "bbox_cov"),
                    hi_cc_n=agg(hi, "cc_n_cc"),
                    hi_cc_largest=agg(hi, "cc_largest_frac"),
                    hi_cc_fill=agg(hi, "cc_fill"),
                    per_frame=rows)
        for (c, r) in GRIDS:
            wrec[f"hi_g{c}x{r}_touched"] = agg(hi, f"g{c}x{r}_touched")
            wrec[f"hi_g{c}x{r}_dense90"] = agg(hi, f"g{c}x{r}_dense90")
        out["windows"][label] = wrec

        print(f"\n=== {label}  types={types[:24]}.. anchors={sorted(anchors)} ===")
        print(f"   mean hole={wrec['mean_hole']*100:.1f}% max={wrec['max_hole']*100:.1f}% "
              f"#hi(>8%)={wrec['n_hi']}")
        if hi:
            print(f"   HI-frames: bboxCov={wrec['hi_bbox_cov']*100:.0f}%  "
                  f"#CC={wrec['hi_cc_n']:.0f} largestCC={wrec['hi_cc_largest']*100:.0f}% "
                  f"fill={wrec['hi_cc_fill']*100:.0f}%")
            print(f"   grid touched% / dense90-area%:")
            for (c, r) in GRIDS:
                print(f"     {c:>2}x{r:<2}: touched {wrec[f'hi_g{c}x{r}_touched']*100:>5.1f}%   "
                      f"dense90 {wrec[f'hi_g{c}x{r}_dense90']*100:>5.1f}%")

    with open(os.path.join(_HERE, "clustering.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(_HERE, 'clustering.json')}")


if __name__ == "__main__":
    main()
