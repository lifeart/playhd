#!/usr/bin/env python3
"""
Step 8 / Task 2 -- film-grain visual proof on real reconstructed HD frames.

Reconstructs a short real window (the talking-head all-P chain), then applies the per-frame
grain pass to 3 CONSECUTIVE output frames. Writes:
  * grain_ab.png         -- before/after (no-grain | grained) 1:1 crop, to confirm grain is
                            visible and filmic (not blocky).
  * grain_consec.png     -- the SAME crop of 3 consecutive frames, grained, side by side. The
                            CONTENT moves coherently while the GRAIN PATTERN changes each frame
                            (temporally independent, not frozen onto content).
  * grain_field_*.png    -- the isolated grain field (recon diff x6, +128) for each of the 3
                            frames, to make "the grain re-rolls every frame" obvious at a glance.
It also prints the frame-to-frame grain-field correlation (must be ~0 = independent) and the
per-frame self-correlation (must be 1 = deterministic).

    python3 grain_demo.py [strength=high]
"""
import os
import sys

import cv2
import numpy as np

import derisk as d
import grain as gr

OUT = os.path.join(os.path.dirname(__file__), "out_quality")
START, NF, SCALE = 5031, 6, 4   # talking-head I + 5 P (clean, low-fallback chain)


def main():
    strength = sys.argv[1] if len(sys.argv) > 1 else "high"
    os.makedirs(OUT, exist_ok=True)
    frames = d.decode_lr_and_mvs("../sample.mp4", START, NF)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    # reconstruct with the compact anchor (grain is content-agnostic; any clean recon works)
    cache = d.build_perframe_cache(frames, w_hd, h_hd, "realesrgan")
    _, R = d.reconstruct(frames, None, SCALE, True, "reactive", cache, set(),
                         backend="numpy", collect_metrics=False)
    # pick 3 consecutive backbone frames (content moves coherently between them)
    idxs = d.backbone_indices(frames)[1:4]   # skip the anchor; take 3 propagated P frames
    recons = [R[i]["recon"] for i in idxs]
    out_fld = [gr.apply_grain(r, i, strength, return_grain=True) for r, i in zip(recons, idxs)]
    grained = [o for o, _ in out_fld]
    raw_fields = [f for _, f in out_fld]   # exact additive luma grain (pre-clip/round-trip)

    # pick a BRIGHT, textured 320-crop (grain is luma-modulated => most visible in mid/high tones;
    # a dead-center crop of this clip lands on dark background where grain barely shows).
    cs = 320
    g0 = cv2.cvtColor(recons[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
    integ = cv2.boxFilter(g0, -1, (cs, cs), normalize=True)
    yc, xc = np.unravel_index(int(np.argmax(integ[cs // 2:h_hd - cs // 2,
                                                  cs // 2:w_hd - cs // 2])),
                              (h_hd - cs, w_hd - cs))
    y0, x0 = yc, xc
    crop = lambda im: im[y0:y0 + cs, x0:x0 + cs]

    # (1) before / after on the first frame
    ab = np.concatenate([d._label(crop(recons[0]), "no grain"),
                         d._label(crop(grained[0]), f"grain={strength}")], axis=1)
    cv2.imwrite(os.path.join(OUT, "grain_ab.png"), cv2.cvtColor(ab, cv2.COLOR_RGB2BGR))

    # (2) 3 consecutive grained frames (content moves, grain re-rolls)
    consec = np.concatenate([d._label(crop(g), f"frame {i} (grained)")
                             for g, i in zip(grained, idxs)], axis=1)
    cv2.imwrite(os.path.join(OUT, "grain_consec.png"), cv2.cvtColor(consec, cv2.COLOR_RGB2BGR))

    # (3) isolated grain fields (grained - ungrained), amplified, for the eyeball
    fields = []
    for g, r, i in zip(grained, recons, idxs):
        fld = (g.astype(np.int16) - r.astype(np.int16))
        vis = np.clip(128 + 6 * fld, 0, 255).astype(np.uint8)
        fields.append(d._label(crop(vis), f"grain field {i}"))
    cv2.imwrite(os.path.join(OUT, "grain_field.png"),
                cv2.cvtColor(np.concatenate(fields, axis=1), cv2.COLOR_RGB2BGR))

    # quantify temporal independence on the RAW additive grain field (artifact-free: the exact
    # noise added to Y, before clip/round-trip -- a measured-field diff would pick up the
    # content-dependent YCrCb round-trip error, which is ~identical between consecutive frames
    # and would spuriously inflate the correlation).
    f0, f1 = raw_fields[0].ravel(), raw_fields[1].ravel()
    corr01 = float(np.corrcoef(f0, f1)[0, 1])
    _, f0b = gr.apply_grain(recons[0], idxs[0], strength, return_grain=True)   # re-apply same idx
    corr_self = float(np.corrcoef(f0, f0b.ravel())[0, 1])
    print(f"raw grain-field corr  frame{idxs[0]} vs frame{idxs[1]} = {corr01:+.4f}  "
          f"(must be ~0 => temporally INDEPENDENT, not frozen)")
    print(f"raw grain-field corr  frame{idxs[0]} vs itself (re-applied) = {corr_self:+.4f}  "
          f"(must be 1.0 => deterministic per frame index)")
    print(f"grain std (luma code values) frame{idxs[0]} = {f0.std():.2f}")
    print(f"wrote -> {OUT}/grain_ab.png, grain_consec.png, grain_field.png")


if __name__ == "__main__":
    main()
