#!/usr/bin/env python3
"""
Step 8 / Task 1 -- visual A/B of the two anchor SR models on a real anchor frame.

Decodes one real frame, super-resolves it x4 with BOTH the compact (realesr-general-x4v3)
and the heavy (RealESRGAN_x4plus) anchor nets, and writes side-by-side full frames + 1:1
detail crops (bicubic | compact | x4plus) so the extra detail of x4plus is visible. Reports
a var-of-Laplacian sharpness number for each as a quantitative companion to the eyeball.

    python3 ab_anchor.py            # default: talking-head I-frame at abs frame 5031
"""
import os
import sys

import cv2
import numpy as np

import derisk as d
import sr

OUT = os.path.join(os.path.dirname(__file__), "out_quality")


def varlap(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 5031
    os.makedirs(OUT, exist_ok=True)
    frames = d.decode_lr_and_mvs("../sample.mp4", start, 1)
    pt, lr, _ = frames[0]
    h, w = lr.shape[:2]
    print(f"anchor frame: abs={start} type={pt} LR={w}x{h} -> HD {w*4}x{h*4}")

    bicubic = cv2.resize(lr, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    sr.load_model("realesrgan")
    sr.load_model("realesrgan-x4plus")
    compact = sr.upscale(lr, model="realesrgan")
    heavy = sr.upscale(lr, model="realesrgan-x4plus")

    print(f"sharpness (var-of-Laplacian, whole frame):")
    print(f"  bicubic           = {varlap(bicubic):8.1f}")
    print(f"  compact x4v3      = {varlap(compact):8.1f}")
    print(f"  x4plus (RRDBNet)  = {varlap(heavy):8.1f}  "
          f"(+{varlap(heavy)/max(varlap(compact),1e-6)-1:+.0%} vs compact)")

    # full-frame side-by-side
    full = np.concatenate([
        d._label(bicubic, "bicubic"),
        d._label(compact, "compact x4v3"),
        d._label(heavy, "x4plus RRDBNet")], axis=1)
    cv2.imwrite(os.path.join(OUT, f"ab_anchor_{start}_full.png"),
                cv2.cvtColor(full, cv2.COLOR_RGB2BGR))

    # 1:1 detail crops at three locations (so HF detail is visible at native HD pixels)
    H, W = heavy.shape[:2]
    cs = 360
    locs = {"center": (H // 2 - cs // 2, W // 2 - cs // 2),
            "upper":  (H // 4 - cs // 2, W // 2 - cs // 2),
            "left":   (H // 2 - cs // 2, W // 4 - cs // 2)}
    for tag, (y0, x0) in locs.items():
        y0 = int(np.clip(y0, 0, H - cs))
        x0 = int(np.clip(x0, 0, W - cs))
        crop = np.concatenate([
            d._label(bicubic[y0:y0 + cs, x0:x0 + cs], "bicubic"),
            d._label(compact[y0:y0 + cs, x0:x0 + cs], "compact x4v3"),
            d._label(heavy[y0:y0 + cs, x0:x0 + cs], "x4plus RRDBNet")], axis=1)
        cv2.imwrite(os.path.join(OUT, f"ab_anchor_{start}_crop_{tag}.png"),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print(f"wrote -> {OUT}/ab_anchor_{start}_*.png")


if __name__ == "__main__":
    main()
