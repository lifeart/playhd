"""demo_matting.py -- Stage L1 feasibility demo for RVM foreground matting on MPS.

Decodes the talking-head window (sample.mp4 start 5000) via derisk.decode_lr_and_mvs
(read-only import), runs Robust Video Matting threading its recurrent state, saves
alpha + foreground visuals to out_matting/, benchmarks per-frame MPS latency at LR
and HD-ish input, and measures matte temporal stability (edge flicker).

Run:  python3 demo_matting.py
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derisk  # READ-ONLY: only decode_lr_and_mvs is used
import matting

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_matting")
os.makedirs(OUT, exist_ok=True)
SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sample.mp4")
START, N = 5000, 48


def rgb_save(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def upscale2x(rgb):
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


def alpha_gray(pha):
    return np.repeat((np.clip(pha, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)


def alpha_overlay(rgb, pha, color=(255, 0, 0)):
    """Tint foreground pixels (alpha-weighted) so subject separation is obvious."""
    a = np.clip(pha, 0, 1)[..., None]
    tint = np.array(color, np.float32)[None, None, :]
    return np.clip(rgb.astype(np.float32) * (1 - 0.45 * a) + tint * 0.45 * a, 0, 255).astype(np.uint8)


def edge_flicker(phas):
    """Temporal stability of the matte. Returns (overall_mad, edge_mad) averaged
    over consecutive frame pairs. edge_mad restricts to the semi-transparent band
    (0.05<alpha<0.95) where matte flicker actually shows. Lower = steadier."""
    overall, edge = [], []
    for i in range(1, len(phas)):
        d = np.abs(phas[i] - phas[i - 1])
        overall.append(float(d.mean()))
        band = ((phas[i] > 0.05) & (phas[i] < 0.95)) | ((phas[i - 1] > 0.05) & (phas[i - 1] < 0.95))
        edge.append(float(d[band].mean()) if band.any() else 0.0)
    return float(np.mean(overall)), float(np.mean(edge))


def main():
    print(f"[decode] {N} frames from sample.mp4 start {START} via derisk.decode_lr_and_mvs ...")
    t0 = time.time()
    decoded = derisk.decode_lr_and_mvs(SAMPLE, start_frame=START, max_frames=N)
    frames = [img for (_pt, img, _mv) in decoded]
    ptypes = "".join(pt for (pt, _i, _m) in decoded)
    h, w = frames[0].shape[:2]
    print(f"[decode] {len(frames)} frames {w}x{h} in {time.time()-t0:.1f}s; cadence {ptypes}")

    model = matting.load_rvm("mps")
    nparam = sum(p.numel() for p in model.parameters())
    # model footprint: param bytes (fp32) + the 14.5MB on-disk checkpoint
    fp_mb = nparam * 4 / 1e6
    print(f"[model] RVM mobilenetv3: {nparam/1e6:.2f}M params (~{fp_mb:.1f}MB fp32 in mem, 14.5MB ckpt)")

    # ---- LR pass (native 640x320) ----
    print("[LR ] matting native 640x320 ...")
    res_lr = matting.matte_sequence(model, frames)
    phas_lr = [p for (_f, p) in res_lr]

    # ---- HD-ish pass (bicubic 2x -> 1280x640) ----
    print("[HD ] matting bicubic-2x 1280x640 ...")
    frames_hd = [upscale2x(f) for f in frames]
    res_hd = matting.matte_sequence(model, frames_hd)
    phas_hd = [p for (_f, p) in res_hd]

    # ---- save visuals for several frames (LR) ----
    save_idx = [0, 8, 16, 24, 32, 40, 47]
    for i in save_idx:
        if i >= len(frames):
            continue
        fgr, pha = res_lr[i]
        panel = np.concatenate([
            frames[i],
            alpha_gray(pha),
            alpha_overlay(frames[i], pha),
            matting.composite(fgr, pha, bg=(0, 255, 0)),
        ], axis=1)
        rgb_save(os.path.join(OUT, f"panel_lr_f{i:02d}.png"), panel)
    # HD alpha for the same anchor frame to compare cleanliness
    for i in [16, 24]:
        fgr, pha = res_hd[i]
        panel = np.concatenate([
            frames_hd[i], alpha_gray(pha), alpha_overlay(frames_hd[i], pha),
            matting.composite(fgr, pha, bg=(0, 255, 0)),
        ], axis=1)
        rgb_save(os.path.join(OUT, f"panel_hd_f{i:02d}.png"), panel)

    # ---- 3 consecutive alphas (LR) for flicker eyeball ----
    c0 = 16
    consec = np.concatenate([alpha_gray(phas_lr[c0 + k]) for k in range(3)], axis=1)
    rgb_save(os.path.join(OUT, f"consec_alpha_lr_f{c0}-{c0+2}.png"), consec)
    consec_hd = np.concatenate([alpha_gray(phas_hd[c0 + k]) for k in range(3)], axis=1)
    rgb_save(os.path.join(OUT, f"consec_alpha_hd_f{c0}-{c0+2}.png"), consec_hd)

    # ---- the L2 layer gate (binary FG mask at LR, dilated) ----
    gate = matting.fg_mask_lr(phas_lr[c0], lr_hw=(h, w), soft=False, thresh=0.5, dilate=3)
    rgb_save(os.path.join(OUT, "fg_gate_lr_f16.png"), alpha_gray(gate))

    # ---- temporal stability ----
    ov_lr, ed_lr = edge_flicker(phas_lr)
    ov_hd, ed_hd = edge_flicker(phas_hd)
    # FG coverage (fraction of pixels with alpha>0.5) -- how much is "foreground"
    fg_frac_lr = float(np.mean([(p > 0.5).mean() for p in phas_lr]))
    fg_frac_hd = float(np.mean([(p > 0.5).mean() for p in phas_hd]))

    # ---- latency benchmarks (MPS-synced, recurrent, after warmup) ----
    print("[bench] timing ...")
    b_lr = matting.benchmark(model, frames, warmup=6)
    b_hd = matting.benchmark(model, frames_hd, warmup=6)
    # also time a true-HD (1920x1080-ish) single-res input to bound the HD cost
    frames_fhd = [cv2.resize(f, (1280, 720), interpolation=cv2.INTER_CUBIC) for f in frames]
    b_fhd = matting.benchmark(model, frames_fhd, warmup=6)

    # ---- report ----
    lines = []
    P = lines.append
    P("=== RVM matting on MPS -- talking-head window (sample.mp4 start 5000, 48f) ===")
    P(f"model: RVM mobilenetv3, {nparam/1e6:.2f}M params, ~{fp_mb:.1f}MB fp32, 14.5MB ckpt, CC BY-NC-SA 4.0")
    P(f"cadence: {ptypes}")
    P("")
    P("--- latency (median ms/frame, MPS-synced, recurrent, after warmup) ---")
    for tag, b in [("LR 640x320", b_lr), ("HD 1280x640 (2x)", b_hd), ("720p 1280x720", b_fhd)]:
        fps = 1000.0 / b["median_ms"]
        budget = "OK <40ms" if b["median_ms"] < 40 else "OVER 40ms"
        P(f"  {tag:18s} ratio={b['downsample_ratio']:.2f}  median {b['median_ms']:5.1f} ms  "
          f"(p90 {b['p90_ms']:5.1f})  {fps:5.1f} fps  [{budget}]")
    P("")
    P("--- matte coverage + temporal stability (consecutive-alpha MAD; lower=steadier) ---")
    P(f"  FG coverage (alpha>0.5):  LR {fg_frac_lr*100:.1f}%   HD {fg_frac_hd*100:.1f}%")
    P(f"  overall alpha flicker:    LR {ov_lr:.4f}   HD {ov_hd:.4f}")
    P(f"  edge-band alpha flicker:  LR {ed_lr:.4f}   HD {ed_hd:.4f}")
    P("")
    P(f"artifacts in {OUT}/:")
    P("  panel_{lr,hd}_fNN.png = [frame | alpha | overlay | FG-on-green]")
    P("  consec_alpha_{lr,hd}_f16-18.png = 3 consecutive alphas (flicker eyeball)")
    P("  fg_gate_lr_f16.png = the L2 binary FG gate (dilated)")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write(report + "\n")
    return dict(b_lr=b_lr, b_hd=b_hd, b_fhd=b_fhd, ov_lr=ov_lr, ed_lr=ed_lr,
               ov_hd=ov_hd, ed_hd=ed_hd, fg_frac_lr=fg_frac_lr)


if __name__ == "__main__":
    main()
