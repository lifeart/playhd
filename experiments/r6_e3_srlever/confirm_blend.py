#!/usr/bin/env python3
"""R6-E3 confirmation: finer gain sweep around the linear-blend optimum + a
visual crop strip (GT | compact | x4plus | blend g=0.5). Reuses run_srlever."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import cv2, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import run_srlever as R
import metrics as M

N = 6
windows = {"talkinghead": R.decode_window(R.SAMPLE, 5000, N),
           "detailed": R.decode_window(R.SAMPLE, 30000, N)}
fine = [0.30, 0.40, 0.50, 0.60, 0.70]
out = {}
crop_rows = {}
for wname, gt in windows.items():
    h, w = gt[0].shape[:2]
    lrs = [R.degrade_real(g) for g in gt]
    c_hr = [R.restore_plain(lr, w, h, "realesrgan", False) for lr in lrs]
    x_hr = [R.restore_plain(lr, w, h, "realesrgan-x4plus", True) for lr in lrs]
    base = M.mean_full_ref(c_hr, gt)
    print(f"[{wname}] compact baseline LPIPS={base['lpips']:.4f}")
    out.setdefault(wname, {})["compact"] = base["lpips"]
    for g in fine:
        bl = [R.blend_linear(c, x, g) for c, x in zip(c_hr, x_hr)]
        m = M.mean_full_ref(bl, gt)
        out[wname][f"blend_g{g:.2f}"] = m["lpips"]
        print(f"  blend g={g:.2f}  LPIPS={m['lpips']:.4f} PSNR={m['psnr']:.2f} "
              f"SSIM={m['ssim']:.4f} varLap={m['varlap']:.0f}")
    crop_rows[wname] = (gt[0], c_hr[0], x_hr[0], R.blend_linear(c_hr[0], x_hr[0], 0.5))
    R.free_gpu("realesrgan"); R.free_gpu("realesrgan-x4plus", half=True)

# crop strip: center 140x140 -> 280, rows = windows, cols = GT|compact|x4plus|blend
def crop(im):
    h, w = im.shape[:2]; cy, cx = h // 2, w // 2
    c = im[cy - 70:cy + 70, cx - 70:cx + 70]
    return cv2.resize(c, (280, 280), interpolation=cv2.INTER_NEAREST)
labels = ["GT", "compact", "x4plus", "blend g0.5"]
rows = []
for wname in windows:
    panels = []
    for lab, im in zip(labels, crop_rows[wname]):
        c = crop(im).copy()
        cv2.putText(c, f"{wname[:4]}:{lab}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(c)
    rows.append(np.hstack(panels))
strip = np.vstack(rows)
cv2.imwrite(os.path.join(_HERE, "crops_blend.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
json.dump(out, open(os.path.join(_HERE, "fine_gain.json"), "w"), indent=2)
print("[done] wrote crops_blend.png + fine_gain.json")
