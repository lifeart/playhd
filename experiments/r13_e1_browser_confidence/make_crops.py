#!/usr/bin/env python3
"""
R13-E1 visual evidence: crop the SAME propagated frame out of the reference (pure SR), the
shipped HARD switch and the SOFT gate, plus each arm's |error| vs the reference, into one PNG.
The hard switch's error map shows blocky patches (whole 16x16-ish blocks dumped to bilinear);
the soft gate's shows a feathered ramp.

Run: python3 make_crops.py --scratch DIR --window calm --frame 7 --variants hard4,soft_t16s3
"""
import argparse
import json
import os

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def load(d, i, wd, hd):
    return np.fromfile(os.path.join(d, f"hd_{i}.bin"), dtype=np.uint8).reshape(hd, wd, 4)[:, :, :3]


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--frame", type=int, default=7)
    ap.add_argument("--variants", default="hard4,soft_t16s3")
    ap.add_argument("--crop", default=None, help="x,y,w,h in HD px (default: auto = worst-error box)")
    args = ap.parse_args()

    refd = os.path.join(args.scratch, f"out_{args.window}_ref")
    run = json.load(open(os.path.join(refd, "run.json")))
    wd, hd = [int(x) for x in run["res"].split("->")[1].split("x")]
    ref = load(refd, args.frame, wd, hd)
    variants = args.variants.split(",")
    imgs = [load(os.path.join(args.scratch, f"out_{args.window}_{v}"), args.frame, wd, hd) for v in variants]

    if args.crop:
        x, y, cw, ch = [int(v) for v in args.crop.split(",")]
    else:                                   # auto: the 256x256 box where arm 0 deviates most
        err = np.abs(imgs[0].astype(np.float32) - ref.astype(np.float32)).mean(axis=2)
        k = 256
        integ = cv2.boxFilter(err, -1, (k, k), normalize=True)
        yy, xx = np.unravel_index(np.argmax(integ[k // 2:hd - k // 2, k // 2:wd - k // 2]), (hd - k, wd - k))
        x, y, cw, ch = xx, yy, k, k
    crop = lambda im: im[y:y + ch, x:x + cw]

    tiles = [label(crop(ref), "reference: SR every frame")]
    for v, im in zip(variants, imgs):
        tiles.append(label(crop(im), v))
    for v, im in zip(variants, imgs):
        e = np.abs(crop(im).astype(np.float32) - crop(ref).astype(np.float32)).mean(axis=2)
        emap = cv2.applyColorMap(np.clip(e * 6, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        tiles.append(label(cv2.cvtColor(emap, cv2.COLOR_BGR2RGB), f"|err| x6 {v[:14]} mean {e.mean():.2f}"))

    ncol = 1 + len(variants)
    while len(tiles) % ncol:                      # pad the error row so the sheet is rectangular
        tiles.insert(ncol + len(variants), np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + ncol]) for i in range(0, len(tiles), ncol)]
    sheet = np.vstack(rows)
    out = os.path.join(_HERE, f"crops_{args.window}_f{args.frame}.png")
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", out, sheet.shape, "crop box", (x, y, cw, ch))


if __name__ == "__main__":
    main()
