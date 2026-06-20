#!/usr/bin/env python3
"""R8-E1 REFUTATION: would the propagation>per-frame reg-dF gap COLLAPSE on a different window /
text / motion direction? (The project killed R5-E2 as a 2-window over-test; test generality.)

Falsifier: if prop/per-frame reg-dF ratio ~1.0 on these variants, the 'moving graphic shimmers'
claim is a single-setup artifact. Tests: (A) different bg window + different text (horizontal),
(B) VERTICAL credits-roll (different motion axis), (C) faster integer scroll."""
import os
import sys
import numpy as np
import cv2

import exp_common as E
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prototype"))
import derisk as d

SCALE = 4
N = 40


def overlay_vroll(rgb_frames, h, w, v_lr=2.0, col_left=None, col_w=200,
                  text="USACHEV  PRODUCTIONS  DIRECTOR  CAMERA  EDITOR  MUSIC  ",
                  fg=245, bg=8, tscale=0.8):
    """Vertical credits roll: a fixed column scrolls text UP at v_lr LR px/frame (sub-pixel)."""
    if col_left is None:
        col_left = w - col_w - 12
    n = len(rgb_frames)
    strip_h = h + int(np.ceil(v_lr * n)) + 8
    strip = np.full((strip_h, col_w, 3), bg, np.uint8)
    y = 20
    while y < strip_h:
        cv2.putText(strip, text[: (y % 30) + 12], (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    tscale, (fg, fg, fg), 2, cv2.LINE_AA)
        y += 34
    out = []
    for i in range(n):
        off = v_lr * i
        M = np.float32([[1, 0, 0], [0, 1, -off]])
        win = cv2.warpAffine(strip, M, (col_w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        fr = rgb_frames[i].copy()
        fr[:, col_left:col_left + col_w] = win
        out.append(fr)
    mask = np.zeros((h, w), bool)
    mask[:, col_left:col_left + col_w] = True
    return out, mask, v_lr


def reg_dframe_vertical(seq, mask_hd, v_hd, margin_extra=3):
    """Registered-dF for a VERTICAL (upward) scroll: shift recon_t DOWN by v_hd to align onto t-1."""
    h, w = mask_hd.shape
    cut = int(np.ceil(v_hd)) + margin_extra
    keep = mask_hd.copy(); keep[:cut, :] = False     # drop the top disocclusion band
    vals = []
    prev = E.luma(seq[0])
    for t in range(1, len(seq)):
        M = np.float32([[1, 0, 0], [0, 1, v_hd]])
        cur = cv2.warpAffine(E.luma(seq[t]), M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        vals.append(float(np.abs(cur - prev)[keep].mean()))
        prev = E.luma(seq[t])
    return float(np.mean(vals))


def run(tag, start, mod, mask_lr, v_lr, vertical=False):
    path = os.path.join(E.TMP, f"ref_{tag}.mp4")
    E.encode_h264(mod, path, crf=20, preset="medium", g=64, bf=2)
    frames = E.decode_mvs(path, len(mod))
    mask_hd = E.upscale_mask(mask_lr, SCALE)
    v_hd = v_lr * SCALE
    R, pf = E.build_recon(frames, SCALE, sr_mode="realesrgan", occ="reactive")
    recon = [R[i]["recon"] for i in range(len(frames))]
    perframe = [pf[i] for i in range(len(frames))]
    rfn = reg_dframe_vertical if vertical else (lambda s, m, v: E.registered_dframe(s, m, v))
    pr = rfn(recon, mask_hd, v_hd)
    pfv = rfn(perframe, mask_hd, v_hd)
    ff, _ = E.fallback_frac_on_mask(R, mask_hd, range(1, len(frames)))
    print(f"  {tag:22s} prop={pr:.3f} perframe={pfv:.3f}  ratio={pr/pfv:.2f}x  "
          f"reactive-fb={100*ff:.1f}%")
    E.free_gpu()
    return pr / pfv


def main():
    print("REFUTATION: prop/per-frame reg-dF ratio across windows/text/motion")
    # A: different window + different text, horizontal
    rgb, h, w = E.decode_clean_rgb(12000, N)
    modA, mA, vA = E.overlay_ticker(rgb, h, w, v_lr=2.0,
                                    text="MARKET WATCH  DOW +1.2%  NASDAQ  S&P 500  ")
    rA = run("A_win12k_horiz", 12000, modA, mA, vA)
    # B: vertical credits roll (different motion axis), same window
    rgb2, h2, w2 = E.decode_clean_rgb(12000, N)
    modB, mB, vB = overlay_vroll(rgb2, h2, w2, v_lr=2.0)
    rB = run("B_vertical_roll", 12000, modB, mB, vB, vertical=True)
    # C: faster integer scroll, yet another window
    rgb3, h3, w3 = E.decode_clean_rgb(8000, N)
    modC, mC, vC = E.overlay_ticker(rgb3, h3, w3, v_lr=4.0, text="LIVE  USACHEV TODAY  ALERT  ")
    rC = run("C_win8k_fast4", 8000, modC, mC, vC)
    print(f"\n  ratios: A={rA:.2f} B={rB:.2f} C={rC:.2f}  "
          f"-> {'ROBUST (all >1.3)' if min(rA,rB,rC)>1.3 else 'NOT robust / REFUTED'}")


if __name__ == "__main__":
    main()
