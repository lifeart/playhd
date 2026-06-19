#!/usr/bin/env python3
"""
Q1 MAX-QUALITY A/B: REAL diffusion SR vs the GAN x4plus anchor, on REAL H.264 SD content.

Pulls a real talking-head anchor frame from ../sample.mp4 (frame ~5000) via
derisk.decode_lr_and_mvs (imported read-only), takes detailed LR crops (most-textured +
center/face), and upscales the SAME LR crop x4 with: bicubic, compact (realesr-general),
x4plus (heavy GAN), and diffusion (stable-diffusion-x4-upscaler). Saves side-by-sides +
100% zooms to out_diffusion_real/, reports var-of-Laplacian sharpness AND keeps the raw
crops so a human can judge TRUE detail vs hallucination (faces/text/texture).

Env knobs:
  DIFF_STEPS (default 50)      diffusion denoise steps
  DIFF_NOISE (default 20)      SD-upscaler conditioning noise_level
  DIFF_GUID  (default 0.0)     guidance scale (0 = faithful, no prompt CFG)
  START_FRAME (default 5000)   anchor frame index
  N_FRAMES   (default 3)       frames decoded in the window to pick the best crop
  SMOKE (default 0)            if 1, only time a few-step diffusion tile and exit
"""
import os
import json
import time

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_diffusion_real")
os.makedirs(OUT, exist_ok=True)
SAMPLE = os.path.join(HERE, "..", "sample.mp4")

CROP = 128
STEPS = int(os.environ.get("DIFF_STEPS", "50"))
NOISE = int(os.environ.get("DIFF_NOISE", "20"))
GUID = float(os.environ.get("DIFF_GUID", "0.0"))
START_FRAME = int(os.environ.get("START_FRAME", "5000"))
N_FRAMES = int(os.environ.get("N_FRAMES", "3"))


def var_lap(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def best_crop(rgb, c=CROP, stride=24):
    H, W = rgb.shape[:2]
    best, bxy = -1.0, (0, 0)
    for y in range(0, H - c + 1, stride):
        for x in range(0, W - c + 1, stride):
            v = var_lap(rgb[y:y + c, x:x + c])
            if v > best:
                best, bxy = v, (x, y)
    x, y = bxy
    return np.ascontiguousarray(rgb[y:y + c, x:x + c]), (x, y, c, c), best


def center_crop(rgb, c=CROP):
    H, W = rgb.shape[:2]
    x = (W - c) // 2; y = (H - c) // 3   # upper-center: where a talking head's face sits
    return np.ascontiguousarray(rgb[y:y + c, x:x + c]), (x, y, c, c)


def extract_anchor():
    """Decode the window [START_FRAME, START_FRAME+N_FRAMES) and return the sharpest frame."""
    import derisk
    frames = derisk.decode_lr_and_mvs(SAMPLE, start_frame=START_FRAME, max_frames=N_FRAMES)
    if not frames:
        raise RuntimeError("no frames decoded")
    # pick the frame with the most overall texture
    best = max(frames, key=lambda f: var_lap(f[1]))
    ptype, img, _ = best
    print(f"[ab] decoded {len(frames)} frames @start {START_FRAME}; picked {ptype}-frame "
          f"{img.shape} overall var-Lap={var_lap(img):.1f}")
    return img


def label(img_bgr, text):
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def zoom(rgb, box=(180, 180, 140, 140), to=420):
    """Crop a sub-box from a 512 output and nearest-upscale it for pixel-peeping."""
    x, y, w, h = box
    sub = rgb[y:y + h, x:x + w]
    return cv2.resize(sub, (to, to), interpolation=cv2.INTER_NEAREST)


def run_one_crop(name, lr_crop, results):
    import sr
    import sr_diffusion_real as sdr
    h, w = lr_crop.shape[:2]
    cols = []  # (label, rgb512)
    vl = {}

    nn = cv2.resize(lr_crop, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
    cols.append(("LR (nearest x4)", nn)); vl["lr_nearest"] = var_lap(nn)

    bic = cv2.resize(lr_crop, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    cols.append(("bicubic x4", bic)); vl["bicubic"] = var_lap(bic)

    cmp = sr.upscale(lr_crop, model="realesrgan")
    cols.append(("compact GAN", cmp)); vl["compact"] = var_lap(cmp)
    res_lat = {"compact_ms": float(sr.last_latency_ms("realesrgan"))}

    x4p = sr.upscale(lr_crop, model="realesrgan-x4plus")
    cols.append(("x4plus GAN", x4p)); vl["x4plus"] = var_lap(x4p)
    res_lat["x4plus_ms"] = float(sr.last_latency_ms("realesrgan-x4plus"))

    t0 = time.perf_counter()
    dif = sdr.upscale_diffusion_real(lr_crop, steps=STEPS, guidance=GUID, noise_level=NOISE)
    res_lat["diffusion_ms"] = float(sdr.last_tile_latency_ms())
    res_lat["diffusion_wall_s"] = time.perf_counter() - t0
    if dif.shape[:2] != (h * 4, w * 4):
        dif = cv2.resize(dif, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    cols.append((f"diffusion s{STEPS} n{NOISE}", dif)); vl["diffusion"] = var_lap(dif)

    # composite side-by-side
    labeled = [label(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), f"{t}  vL={vl[k]:.0f}")
               for (t, im), k in zip(cols, ["lr_nearest", "bicubic", "compact", "x4plus", "diffusion"])]
    comp = np.hstack(labeled)
    cv2.imwrite(os.path.join(OUT, f"sbs_{name}.png"), comp)

    # 100% pixel-peep zoom row (same sub-box across methods)
    zb = (180, 160, 150, 150)
    zlab = [label(cv2.cvtColor(zoom(im, zb), cv2.COLOR_RGB2BGR), t) for (t, im) in cols]
    cv2.imwrite(os.path.join(OUT, f"zoom_{name}.png"), np.hstack(zlab))

    # raw individual crops for honest inspection
    for (t, im), k in zip(cols, ["lr_nearest", "bicubic", "compact", "x4plus", "diffusion"]):
        cv2.imwrite(os.path.join(OUT, f"{name}_{k}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUT, f"{name}_LRsrc.png"), cv2.cvtColor(lr_crop, cv2.COLOR_RGB2BGR))

    results[name] = dict(var_lap=vl, latency=res_lat, crop_var_lap=var_lap(lr_crop))
    print(f"[ab] {name}: var-Lap " + "  ".join(f"{k}={v:.0f}" for k, v in vl.items()))
    print(f"[ab] {name}: latency " + "  ".join(f"{k}={v:.0f}" for k, v in res_lat.items()))


def smoke():
    """Fast timing probe: load pipeline, run a few-step tile, report per-step + footprint."""
    import sr_diffusion_real as sdr
    img = extract_anchor()
    crop, xywh, score = best_crop(img)
    print(f"[smoke] crop {xywh} var-Lap={score:.1f}")
    sdr.load_pipeline(use_taesd=False)
    for s in (5, 10):
        sdr.reset_latency()
        out = sdr.upscale_diffusion_real(crop, steps=s, guidance=GUID, noise_level=NOISE)
        print(f"[smoke] steps={s} -> {out.shape} tile={sdr.last_tile_latency_ms()/1000:.1f}s "
              f"mem={sdr._mem_mb(sdr._DEVICE)}")
    # TAESD compatibility check (reported, not assumed)
    import torch
    ok, note = sdr.try_swap_taesd(sdr._PIPE, sdr._DEVICE, torch.float16)
    print(f"[smoke] TAESD swap: ok={ok}  {note}")


def main():
    if os.environ.get("SMOKE", "0") == "1":
        smoke(); return
    results = dict(model="stabilityai/stable-diffusion-x4-upscaler",
                   steps=STEPS, noise_level=NOISE, guidance=GUID,
                   start_frame=START_FRAME)
    img = extract_anchor()
    cv2.imwrite(os.path.join(OUT, "anchor_frame.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    crops = {}
    tex, xy_t, s_t = best_crop(img); crops["textured"] = (tex, xy_t)
    cen, xy_c = center_crop(img); crops["face"] = (cen, xy_c)
    results["crops"] = {k: dict(xywh=list(xy)) for k, (c, xy) in crops.items()}
    print(f"[ab] textured crop {xy_t} vL={s_t:.0f} | face crop {xy_c} vL={var_lap(cen):.0f}")

    out = {}
    for name, (crop, xy) in crops.items():
        run_one_crop(name, crop, out)
    results["ab"] = out

    import sr_diffusion_real as sdr
    results["footprint"] = sdr._mem_mb(sdr._DEVICE)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== AB SUMMARY ===")
    print(json.dumps(results, indent=2, default=str))
    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()
