#!/usr/bin/env python3
"""R8-E1 pin validation:
  (1) MOVING ticker: detect->pin (to bicubic AND to compact SR) drops registered-dF toward floor.
  (2) STATIC USACHEV card (real sample.mp4, the R1-E3 NO-GO): the motion-gated detector fires on
      ZERO pixels -> pin is BYTE-IDENTICAL -> the static NO-GO is structurally preserved.
  (3) talking-head FACE: detector fires on ZERO pixels (bimodality FP guard) -> byte-identical."""
import os
import sys
import numpy as np
import cv2

import exp_common as E
import graphic_pin as gp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "prototype"))
import derisk as d

SCALE = 4
N = 48


def test_moving():
    print("=== (1) MOVING ticker int2: detect->pin to {bicubic, compact-SR} ===")
    rgb, h, w = E.decode_clean_rgb(5000, N)
    mod, mask_true, v = E.overlay_ticker(rgb, h, w, v_lr=2.0)
    path = os.path.join(E.TMP, "pin_int2.mp4")
    E.encode_h264(mod, path, crf=20, preset="medium", g=64, bf=2)
    frames = E.decode_mvs(path, N)
    h_lr, w_lr = frames[0][1].shape[:2]
    v_hd = v * SCALE
    R, pf = E.build_recon(frames, SCALE, sr_mode="realesrgan", occ="reactive")
    recon = [R[i]["recon"] for i in range(N)]
    compact = [pf[i] for i in range(N)]
    bicub = [cv2.resize(frames[i][1], (w_lr * SCALE, h_lr * SCALE), interpolation=cv2.INTER_CUBIC)
             for i in range(N)]
    # detect per-frame and measure detector recall vs the authored bar
    masks_lr = [gp.moving_graphic_mask_lr(frames[i][1], frames[i][2], h_lr, w_lr) for i in range(N)]
    masks_hd = [gp.upscale_mask(m, SCALE) for m in masks_lr]
    cov = np.mean([100 * m.mean() for m in masks_lr])
    recall = np.mean([(masks_lr[i] & mask_true).sum() / max(mask_true.sum(), 1) for i in range(N)])
    pinned_bic = [gp.apply_pin_np(recon[i], bicub[i], masks_hd[i]) for i in range(N)]
    pinned_cmp = [gp.apply_pin_np(recon[i], compact[i], masks_hd[i]) for i in range(N)]
    mask_eval = E.upscale_mask(mask_true, SCALE)
    for nm, seq in [("propagation (shipped)", recon), ("pin->bicubic (free)", pinned_bic),
                    ("pin->compact SR", pinned_cmp), ("per-frame compact SR (floor)", compact)]:
        print(f"  {nm:30s} reg-dF={E.registered_dframe(seq, mask_eval, v_hd):.3f}")
    print(f"  detector: mean coverage={cov:.1f}% of frame, recall on authored bar={100*recall:.0f}%")
    E.free_gpu()


def test_static_card():
    print("\n=== (2) STATIC USACHEV card (sample.mp4 f0-32): motion gate must EXCLUDE it ===")
    frames = d.decode_lr_and_mvs(E.SAMPLE, 0, 32)
    h_lr, w_lr = frames[0][1].shape[:2]
    R, pf = E.build_recon(frames, SCALE, sr_mode="realesrgan", occ="reactive")
    n_static = list(range(18, 28))            # the static-card run (R1-E3); f28 = scene cut
    fired = 0
    for i in n_static:
        m = gp.moving_graphic_mask_lr(frames[i][1], frames[i][2], h_lr, w_lr)
        fired += int(m.sum())
    print(f"  static-card frames 18-27: total pinned pixels = {fired}  "
          f"({'BYTE-IDENTICAL -> static NO-GO preserved' if fired == 0 else 'WARNING: fires!'})")
    # also confirm the byte-identical property end-to-end on one frame
    i = 22
    m_hd = gp.upscale_mask(gp.moving_graphic_mask_lr(frames[i][1], frames[i][2], h_lr, w_lr), SCALE)
    pinned = gp.apply_pin_np(R[i]["recon"], pf[i], m_hd)
    print(f"  f22 |pinned - recon| max = {int(np.abs(pinned.astype(int) - R[i]['recon'].astype(int)).max())} "
          f"(0 => exact copy)")
    E.free_gpu()


def test_face():
    print("\n=== (3) talking-head FACE (sample.mp4 f5000): bimodality FP guard ===")
    frames = d.decode_lr_and_mvs(E.SAMPLE, 5000, 28)
    h_lr, w_lr = frames[0][1].shape[:2]
    fired = sum(int(gp.moving_graphic_mask_lr(frames[i][1], frames[i][2], h_lr, w_lr).sum())
                for i in range(len(frames)))
    print(f"  face frames: total pinned pixels = {fired}  "
          f"({'no false positive' if fired == 0 else 'WARNING: fires on face!'})")


def main():
    test_moving()
    test_static_card()
    test_face()


if __name__ == "__main__":
    main()
