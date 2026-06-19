"""demo_background_plate.py -- Stage L2 demo: build + heavy-SR the background plate.

End-to-end on the talking-head window (sample.mp4 start 5000):
  1. Decode LR frames + MVs (derisk.decode_lr_and_mvs, read-only).
  2. STATIC-CAMERA VERDICT from the codec MVs (background_plate.estimate_global_motion).
  3. Matte the window (RVM, matting.py) -> per-frame FG gates (fg_mask_lr, dilate=3).
  4. BUILD the LR plate = per-pixel temporal median of background-only pixels; report
     coverage (>=1 / >=3 samples) and map the always-occluded holes.
  5. QUALITY: single decoded frame vs accumulated plate vs hole map -- confirm the
     subject is removed and the occluded background filled in.
  6. HEAVY-SR the plate ONCE (sr.upscale x4plus) -> HD plate; crop vs bicubic-upscaled
     plate to show the detail you get "for free" per frame.

Saves visuals + summary.txt under out_plate/.  Run:  python3 demo_background_plate.py
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derisk            # READ-ONLY: decode_lr_and_mvs
import matting           # READ-ONLY: Stage L1 matting
import sr                # READ-ONLY: heavy x4plus SR
import background_plate as bp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_plate")
os.makedirs(OUT, exist_ok=True)
SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sample.mp4")
START, N = 5000, 48
SCALE = 4
HEAVY = "realesrgan-x4plus"


def rgb_save(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def heat(mask01):
    """float [0,1] HxW -> uint8 HxWx3 heatmap (jet)."""
    u8 = np.clip(mask01 * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def main():
    P = []
    log = P.append

    # ---- 1. decode ----
    print(f"[decode] {N} frames from sample.mp4 start {START} ...")
    t0 = time.time()
    decoded = derisk.decode_lr_and_mvs(SAMPLE, start_frame=START, max_frames=N)
    frames_all = [img for (_pt, img, _mv) in decoded]
    ptypes = "".join(pt for (pt, _i, _m) in decoded)
    h, w = frames_all[0].shape[:2]
    print(f"[decode] {len(frames_all)} frames {w}x{h} in {time.time()-t0:.1f}s; cadence {ptypes}")

    # ---- 1b. SCENE SEGMENTATION: a plate is PER-SCENE. This window spans a hard cut at
    # the mid-stream I-frame (talking head -> title card), so we build the plate on the
    # FIRST scene only -- mixing scenes would contaminate the median. ----
    segs = bp.scene_segments(decoded, frames=frames_all)
    cuts = bp.find_scene_cuts(decoded, frames=frames_all)
    s0, s1 = segs[0]
    print(f"[scene] cuts at {cuts}; segments {segs}; building plate on scene[0] = frames [{s0},{s1})")
    decoded_seg = decoded[s0:s1]
    frames = frames_all[s0:s1]
    Nseg = len(frames)

    # ---- 3. matte FIRST (so the static-camera check can also use background MVs) ----
    print(f"[matte] RVM mobilenetv3 over scene[0] ({Nseg} frames) ...")
    model = matting.load_rvm("mps")
    res = matting.matte_sequence(model, frames)
    phas = [p for (_f, p) in res]
    # FG gate: binary, dilate=3 -> push hair/edge uncertainty into FG so it is EXCLUDED
    gates = [matting.fg_mask_lr(p, lr_hw=(h, w), soft=False, thresh=0.5, dilate=3) for p in phas]
    fg_frac = float(np.mean([g.mean() for g in gates]))
    print(f"[matte] mean FG coverage (dilated gate): {fg_frac*100:.1f}%")

    # ---- 2. static-camera verdict from MVs (overall + background cross-check) ----
    gm = bp.estimate_global_motion(decoded_seg, gates=gates)
    print(f"[motion] verdict={gm['verdict']}  global|MV vec|={gm['global_vec_mag_px']:.3f}px"
          f"  bg|MV|={gm['bg_block_median_mag_px']:.3f}px  all|MV|={gm['all_block_median_mag_px']:.3f}px")

    # ---- 4. build the LR plate (on the talking-head scene segment) ----
    print("[plate] accumulating temporal-median background plate ...")
    tp = time.time()
    plate_lr, coverage, hole_mask = bp.build_plate(frames, gates, min_samples=1)
    rep = bp.coverage_report(coverage, hole_mask)
    print(f"[plate] built in {time.time()-tp:.2f}s  cov>=1 {rep['pct_ge1']:.1f}%  "
          f"cov>=3 {rep['pct_ge3']:.1f}%  holes {rep['hole_pct']:.2f}%")

    # ---- 4b. CAUTIONARY CONTRAST: naive full-window plate (ignores the scene cut) ----
    # Matte + accumulate over ALL 48 frames including the title card. The 2nd scene's
    # frames dump full frames into the median -> the always-occluded hole gets falsely
    # "filled" with title-card content. This is WHY the plate must be per-scene.
    res_all = matting.matte_sequence(model, frames_all)
    gates_all = [matting.fg_mask_lr(p, lr_hw=(h, w), soft=False, thresh=0.5, dilate=3)
                 for (_f, p) in res_all]
    _, cov_all, hole_all = bp.build_plate(frames_all, gates_all, min_samples=1)
    rep_all = bp.coverage_report(cov_all, hole_all)
    print(f"[contam] naive full-{len(frames_all)}f plate (spans cut): holes {rep_all['hole_pct']:.2f}% "
          f"(vs scene plate {rep['hole_pct']:.2f}%) -- the hole is falsely filled by scene 2")

    # ---- 5. quality visuals (LR) ----
    ref_frame = frames[Nseg // 2]                                # a single decoded frame
    cov_norm = (coverage.astype(np.float32) / max(coverage.max(), 1))
    hole_vis = np.where(hole_mask[..., None], np.array([255, 0, 0], np.uint8),
                        plate_lr)                                # red = always-occluded hole
    panel_lr = np.concatenate([
        ref_frame,                                              # one frame (subject present)
        plate_lr,                                               # accumulated plate (subject gone)
        heat(cov_norm),                                         # coverage heatmap
        hole_vis,                                               # holes painted red on the plate
    ], axis=1)
    rgb_save(os.path.join(OUT, "panel_lr.png"), panel_lr)
    rgb_save(os.path.join(OUT, "plate_lr.png"), plate_lr)
    rgb_save(os.path.join(OUT, "single_frame.png"), ref_frame)
    rgb_save(os.path.join(OUT, "coverage_heat.png"), heat(cov_norm))
    rgb_save(os.path.join(OUT, "hole_map.png"),
             (hole_mask[..., None] * np.array([255, 255, 255], np.uint8)).astype(np.uint8))
    # subject-removal check: where the subject WAS in this frame but background was
    # RECOVERED elsewhere (mid_gate & not-hole), the plate should now show background ->
    # a LARGE diff from the subject-containing frame = subject removed + background filled.
    mid_gate = gates[Nseg // 2] > 0.5
    revealed = mid_gate & ~hole_mask                 # subject here, recovered as bg elsewhere
    diff_fg = float(np.abs(plate_lr[revealed].astype(int) - ref_frame[revealed].astype(int)).mean()) \
        if revealed.any() else 0.0
    diff_bg = float(np.abs(plate_lr[~mid_gate].astype(int) - ref_frame[~mid_gate].astype(int)).mean())
    print(f"[quality] mean|plate-frame| revealed-subject-region={diff_fg:.1f} (large=subject removed, bg filled) "
          f"static-bg-region={diff_bg:.1f} (small=background preserved)")

    # ---- 6. HEAVY-SR the plate ONCE ----
    print(f"[sr] heavy x4plus on the plate ONCE ({HEAVY}) ...")
    ts = time.time()
    plate_hd = bp.sr_plate(plate_lr, scale=SCALE, model=HEAVY)
    sr_ms = sr.last_latency_ms(HEAVY)
    print(f"[sr] plate {plate_lr.shape} -> HD {plate_hd.shape} in {time.time()-ts:.2f}s "
          f"(net {sr_ms:.0f}ms; amortized over {Nseg} frames = {sr_ms/Nseg:.1f}ms/frame)")
    rgb_save(os.path.join(OUT, "plate_hd.png"), plate_hd)

    # crop: SR'd plate vs bicubic-upscaled plate (the detail you get free per frame)
    bicubic_hd = cv2.resize(plate_lr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    # pick a textured BACKGROUND crop, away from the subject hole: most-textured 120-LR-px
    # window among the always-covered (non-hole) region, upper band (lamp/wall).
    cs_lr = 120
    best, bxy = -1.0, (int(h * 0.10), int(w * 0.62))
    gray_full = cv2.cvtColor(plate_lr, cv2.COLOR_RGB2GRAY)
    for yy in range(0, h - cs_lr, 24):
        for xx in range(0, w - cs_lr, 24):
            if hole_mask[yy:yy + cs_lr, xx:xx + cs_lr].any():
                continue
            v = float(cv2.Laplacian(gray_full[yy:yy + cs_lr, xx:xx + cs_lr], cv2.CV_64F).var())
            if v > best:
                best, bxy = v, (yy, xx)
    cy, cx, cs = bxy[0] * SCALE, bxy[1] * SCALE, cs_lr * SCALE
    crop_sr = plate_hd[cy:cy + cs, cx:cx + cs]
    crop_bi = bicubic_hd[cy:cy + cs, cx:cx + cs]
    crop_panel = np.concatenate([crop_bi, crop_sr], axis=1)
    rgb_save(os.path.join(OUT, "crop_bicubic_vs_sr.png"), crop_panel)
    # sharpness proxy (var of Laplacian) on the crops
    def vlap(rgb):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    vb, vs_ = vlap(crop_bi), vlap(crop_sr)
    print(f"[sr] crop sharpness var-Laplacian: bicubic {vb:.1f}  x4plus {vs_:.1f}  ({vs_/max(vb,1e-6):.2f}x)")

    # sample_plate sanity (static => identity)
    bg_for_f0 = bp.sample_plate(plate_hd, frame_idx=0, global_motion=None)
    identity_ok = bg_for_f0 is plate_hd or np.array_equal(bg_for_f0, plate_hd)

    # ---- report ----
    log("=== L2 BACKGROUND PLATE -- talking-head window (sample.mp4 start 5000, 48f) ===")
    log(f"cadence: {ptypes}")
    log("")
    log("--- 0. SCENE SEGMENTATION (a plate is PER-SCENE) ---")
    log(f"  scene cuts inside window at frame index: {cuts}  (mid-stream I-frame = codec scene cut)")
    log(f"  segments: {segs}")
    log(f"  plate built on scene[0] = frames [{s0},{s1}) ({Nseg} frames, the talking head)")
    log(f"  NAIVE full-{len(frames_all)}f plate (ignores cut) holes: {rep_all['hole_pct']:.2f}%  "
        f"<- FALSELY filled by scene 2 (title card); scene-correct holes below")
    log("")
    log("--- 1. STATIC-CAMERA VERDICT (from codec MVs, scene[0]) ---")
    log(f"  verdict: {gm['verdict']}   (threshold {gm['static_thresh_px']:.2f}px)")
    log(f"  global |median MV vector| = {gm['global_vec_mag_px']:.3f} px   <- camera translation proxy")
    log(f"  background-block median |MV| = {gm['bg_block_median_mag_px']:.3f} px   <- static if ~0")
    log(f"  all-block median |MV| = {gm['all_block_median_mag_px']:.3f} px   (subject motion included)")
    log(f"  inter frames used: {gm['n_inter_frames']}/{Nseg}")
    log("")
    log("--- 2. BACKGROUND COVERAGE (plate completeness) ---")
    log(f"  pixels with >=1 background sample: {rep['pct_ge1']:.2f}%")
    log(f"  pixels with >=3 background samples: {rep['pct_ge3']:.2f}%")
    log(f"  pixels with >=5 background samples: {rep['pct_ge5']:.2f}%")
    log(f"  always-occluded HOLE (0 samples): {rep['hole_pct']:.2f}%  (behind the subject; never displayed)")
    log(f"  max / median per-pixel coverage: {rep['max_coverage']} / {rep['median_coverage']:.0f} frames")
    log(f"  mean dilated FG gate per frame: {fg_frac*100:.1f}%")
    log("")
    log("--- 3. SUBJECT REMOVAL / BACKGROUND FILL ---")
    log(f"  mean|plate - frame| in REVEALED subject region (recovered as bg): {diff_fg:.1f}  (large => subject removed + bg filled)")
    log(f"  mean|plate - frame| in static background region: {diff_bg:.1f}  (small => background preserved)")
    log(f"  (the {rep['hole_pct']:.1f}%% always-occluded hole is inpainted; never displayed in the composite)")
    log("")
    log("--- 4. HEAVY-SR (the amortization) ---")
    log(f"  one x4plus call: {plate_lr.shape[1]}x{plate_lr.shape[0]} -> {plate_hd.shape[1]}x{plate_hd.shape[0]}"
        f"  in {sr_ms:.0f} ms")
    log(f"  amortized over the {Nseg}-frame scene: {sr_ms/Nseg:.1f} ms/frame (vs {sr_ms:.0f} ms if per-frame)")
    log(f"  background crop sharpness var-Laplacian: bicubic {vb:.1f} -> x4plus {vs_:.1f}  ({vs_/max(vb,1e-6):.2f}x)")
    log(f"  sample_plate(static) is identity: {identity_ok}")
    log("")
    log(f"artifacts in {OUT}/:")
    log("  panel_lr.png       = [single frame | plate | coverage-heat | holes-red]")
    log("  plate_lr.png       = the accumulated LR background plate (subject removed)")
    log("  plate_hd.png       = the heavy-SR'd HD background plate (ONE SR call/scene)")
    log("  coverage_heat.png  = per-pixel background sample count (jet)")
    log("  hole_map.png       = always-occluded region (white = behind subject every frame)")
    log("  crop_bicubic_vs_sr.png = [bicubic-up | x4plus] background crop (free per-frame detail)")

    report = "\n".join(P)
    print("\n" + report)
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write(report + "\n")
    return dict(gm=gm, coverage=rep, diff_fg=diff_fg, diff_bg=diff_bg, sr_ms=sr_ms)


if __name__ == "__main__":
    main()
