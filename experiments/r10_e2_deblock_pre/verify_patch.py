#!/usr/bin/env python3
"""Verify the R10-E2 deblock preprocessor is byte-identical when OFF, and that the gate
behaves (ON only on heavy/blocky LR). Does NOT modify shared code -- exercises deblock_pre
directly, mirroring exactly how build_perframe_cache.patch calls it."""
import os, sys, numpy as np, cv2, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "prototype"))
import deblock_pre as P
import run_ab as R

fr = R.decode_frames(R.SAMPLE, 24000, 1)[0]
gt, _ = R.best_crop(fr)
lr_heavy, _ = R.h264_degrade(gt, 35)
lr_light, _ = R.h264_degrade(gt, 27)

# 1) OFF path is byte-identical (cfg None and cfg empty)
for cfg in (None, {}, 0, ""):
    out = P.apply(lr_heavy, cfg)
    assert out is lr_heavy or np.array_equal(out, lr_heavy), f"OFF not identity for cfg={cfg!r}"
print("[1] OFF path (cfg None/empty) -> byte-identical: PASS")

# 2) gate=blockiness: ON for heavy, OFF (identity) for light at a mid threshold
b_h, b_l = P.blockiness(lr_heavy), P.blockiness(lr_light)
cfg = {"gate": "blockiness", "block_min": (b_h + b_l) / 2}
out_h = P.apply(lr_heavy, cfg); out_l = P.apply(lr_light, cfg)
changed_h = not np.array_equal(out_h, lr_heavy)
identity_l = np.array_equal(out_l, lr_light)
print(f"[2] blockiness gate (heavy {b_h:.2f} > thr {cfg['block_min']:.2f} > light {b_l:.2f}): "
      f"heavy deblocked={changed_h}  light identity={identity_l}: "
      f"{'PASS' if changed_h and identity_l else 'FAIL'}")

# 3) gate=qp: respects bitstream QP
cfg = {"gate": "qp", "qp_min": 30}
on  = P.apply(lr_heavy, cfg, qp=37)
off = P.apply(lr_heavy, cfg, qp=22)
print(f"[3] qp gate: qp37>=30 deblocked={not np.array_equal(on,lr_heavy)}  "
      f"qp22<30 identity={np.array_equal(off,lr_heavy)}: "
      f"{'PASS' if (not np.array_equal(on,lr_heavy)) and np.array_equal(off,lr_heavy) else 'FAIL'}")

# 4) texture guard: skip when LR var-Lap too high
vl = cv2.Laplacian(cv2.cvtColor(lr_heavy, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
cfg = {"gate": "always", "skip_texture_varlap": vl - 1}
print(f"[4] texture guard (LR varLap {vl:.0f} >= {vl-1:.0f}) -> skipped (identity): "
      f"{'PASS' if np.array_equal(P.apply(lr_heavy, cfg), lr_heavy) else 'FAIL'}")
