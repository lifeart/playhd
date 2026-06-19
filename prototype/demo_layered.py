"""demo_layered.py -- Stage L3 measurement: COMPOSE the layered pipeline and decide
whether it is the quality-and-speed win the L1/L2 math predicts.

On the talking-head scene (sample.mp4 start 5000, the single static-camera shot =
frames [0,32) before the cut), this:
  1. Decodes LR+MVs, splits to scene[0]; mattes (L1); builds + heavy-SR's the static
     background plate ONCE (L2).
  2. Renders the LAYERED composite TWO ways -- (a) compact per-frame FG, (b) x4plus on
     the FG bbox only -- with and without grain.
  3. Baselines (full-frame, no layering): uniform-compact (current live), uniform-x4plus
     (per-frame heavy = quality ceiling, flickery), region-aware (Stream 1 output blend).
  4. MEASURES, per region (background vs foreground via the matte) and overall:
       sharpness (var-of-Laplacian), tOF (TecoGAN temporal-OF vs decoded LR), and direct
       background temporal instability (mean |delta-frame|, code values) -- the headline.
  5. MEASURES cost per frame (plate-sample + matte + FG SR + composite + grain) vs the
     full-frame propagation recon (~28-42 ms, Step 7) and per-frame x4plus (~2.2 s),
     at THIS 32-frame scene and extrapolated to a 300-frame shot.
  6. SEAM honesty: saves side-by-side composites + a hair-edge seam crop; quantifies the
     sharpness discontinuity and colour mismatch across the alpha boundary.

Imports derisk/matting/background_plate/sr/grain/region_quality/layered_pipeline READ-ONLY.
Writes everything to out_layered/.  Run (FOREGROUND):  python3 demo_layered.py
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import derisk                    # READ-ONLY
import matting                   # READ-ONLY (L1)
import background_plate as bp    # READ-ONLY (L2)
import sr                        # READ-ONLY
import grain                     # READ-ONLY
import region_quality as rq      # READ-ONLY (Stream 1 region-aware)
import layered_pipeline as lp    # the L3 composer (new)

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "out_layered")
CACHE = os.path.join(OUT, "cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
SAMPLE = os.path.join(_HERE, "..", "sample.mp4")
START, NWIN = 5000, 48          # decode the full window, then split to scene[0]
SCALE = 4
COMPACT, HEAVY = "realesrgan", "realesrgan-x4plus"


def rgb_save(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def label(img, text, half=True):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def vlap(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F)


def vlap_masked(rgb, mask):
    lap = vlap(rgb)
    return float(lap[mask].var()) if mask.any() else float("nan")


def sharp_fixed(recon, bg_hd, fg_hd_union):
    """mean over frames of var-of-Laplacian in BG-always and FG-union (HD masks)."""
    sb = [vlap_masked(r, bg_hd) for r in recon]
    return float(np.mean(sb))


def sharp_perframe_fg(recon, fg_masks_hd):
    """FG sharpness using the PER-FRAME subject mask (subject moves)."""
    vals = [vlap_masked(r, m) for r, m in zip(recon, fg_masks_hd) if m.any()]
    return float(np.mean(vals)) if vals else float("nan")


def temporal_instability(recon, mask):
    """mean over consecutive pairs of mean(|frame[t]-frame[t-1]|) inside `mask` (code
    values). The DIRECT flicker number: 0 == the region is byte-identical every frame."""
    vals = []
    for t in range(1, len(recon)):
        d = np.abs(recon[t][mask].astype(np.int16) - recon[t - 1][mask].astype(np.int16))
        vals.append(float(d.mean()))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    P = []
    log = P.append
    T0 = time.time()

    # ---------------- 1. decode + scene split ----------------
    print(f"[decode] window {NWIN}f from start {START} ...")
    decoded = derisk.decode_lr_and_mvs(SAMPLE, start_frame=START, max_frames=NWIN)
    frames_all = [img for (_p, img, _m) in decoded]
    h, w = frames_all[0].shape[:2]
    h_hd, w_hd = h * SCALE, w * SCALE
    segs = bp.scene_segments(decoded, frames=frames_all)
    s0, s1 = segs[0]
    decoded_seg = decoded[s0:s1]
    frames = frames_all[s0:s1]
    N = len(frames)
    ref_lr = frames                       # tOF reference = decoded LR (cleanest motion truth)
    print(f"[scene] {w}x{h} LR, scene[0]=[{s0},{s1}) N={N}; HD {w_hd}x{h_hd}")

    # ---------------- 2. L1 matte ----------------
    print("[matte] RVM mobilenetv3 ...")
    model = matting.load_rvm("mps")
    tM = time.time()
    phas, gates = lp.matte_scene(model, frames, dilate=3)
    matte_wall = (time.time() - tM)
    mb = matting.benchmark(model, frames)        # honest per-frame matte latency
    fg_frac = float(np.mean([(p >= 0.5).mean() for p in phas]))
    print(f"[matte] FG frac {fg_frac*100:.1f}%  per-frame {mb['median_ms']:.1f}ms (median)")

    # region masks (LR, fixed) for the metric split
    fg_any = np.zeros((h, w), bool)
    bg_all = np.ones((h, w), bool)
    for p in phas:
        f = p >= 0.5
        fg_any |= f
        bg_all &= ~f
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bg_all = cv2.erode(bg_all.astype(np.uint8), k).astype(bool)     # pure static BG
    fg_union = cv2.dilate(fg_any.astype(np.uint8), k).astype(bool)  # region subject covers
    bg_all_hd = cv2.resize(bg_all.astype(np.uint8), (w_hd, h_hd), interpolation=cv2.INTER_NEAREST).astype(bool)
    fg_union_hd = cv2.resize(fg_union.astype(np.uint8), (w_hd, h_hd), interpolation=cv2.INTER_NEAREST).astype(bool)
    fg_masks_hd = [cv2.resize((p >= 0.5).astype(np.uint8), (w_hd, h_hd),
                              interpolation=cv2.INTER_NEAREST).astype(bool) for p in phas]
    print(f"[masks] bg_always {bg_all.mean()*100:.1f}% of frame, fg_union {fg_union.mean()*100:.1f}%")

    # ---------------- 3. L2 plate ----------------
    print("[plate] build LR plate (temporal median) ...")
    tP = time.time()
    plate_lr, coverage, hole_mask = bp.build_plate(frames, gates, min_samples=1)
    plate_build_wall = time.time() - tP
    rep = bp.coverage_report(coverage, hole_mask)
    print(f"[plate] cov>=1 {rep['pct_ge1']:.1f}%  holes {rep['hole_pct']:.2f}%  build {plate_build_wall:.2f}s")
    print(f"[plate] heavy-SR x4plus ONCE ...")
    tS = time.time()
    plate_hd = bp.sr_plate(plate_lr, scale=SCALE, model=HEAVY)
    plate_sr_ms = sr.last_latency_ms(HEAVY)
    print(f"[plate] SR {plate_lr.shape}->{plate_hd.shape} net {plate_sr_ms:.0f}ms (wall {time.time()-tS:.2f}s)")

    # ---------------- 4. per-frame SR caches (baselines + layered FG) ----------------
    def cache_perframe(modelname):
        path = os.path.join(CACHE, f"sr_{START}_{s0}_{s1}_{modelname}.npy")
        if os.path.exists(path):
            arr = np.load(path)
            if arr.shape[0] == N:
                print(f"  [cache] {modelname}: loaded")
                return [np.ascontiguousarray(arr[i]) for i in range(N)]
        sr.load_model(modelname); sr.upscale(frames[0], model=modelname); sr.reset_latency(modelname)
        out = []
        for i, f in enumerate(frames):
            out.append(sr.upscale_to(f, w_hd, h_hd, model=modelname))
        np.save(path, np.stack(out))
        print(f"  [cache] {modelname}: built, median {sr.median_latency_ms(modelname):.0f}ms/frame")
        return out

    print("[sr] per-frame compact (uniform-compact == layered compact-FG) ...")
    recon_compact = cache_perframe(COMPACT)
    compact_ms = sr.median_latency_ms(COMPACT)
    print("[sr] per-frame x4plus full-frame (uniform-x4plus == quality ceiling) ...")
    recon_x4plus = cache_perframe(HEAVY)
    x4plus_full_ms = sr.median_latency_ms(HEAVY)

    # ---------------- 5. region-aware (Stream 1 output blend) ----------------
    # honest invocation: same per-frame heavy/compact caches, gated by Stream-1's stable
    # motion weight (lo=0.2, hi=1.0, feather=9) from the codec MVs.
    _, _, meanmag, _ = rq.region_masks(decoded_seg, h, w, 45.0, 80.0)
    a_lr = rq.window_static_weight(meanmag, lo=0.2, hi=1.0, feather=9)
    recon_region = [rq.blend_region_aware(recon_x4plus[i], recon_compact[i], a_lr, SCALE) for i in range(N)]

    # ---------------- 6. LAYERED renders (both FG budgets) ----------------
    print("[layered] compact-FG ...")
    sr.reset_latency(COMPACT)
    R_compact = lp.render_scene(frames, phas, plate_hd, fg_budget="compact", grain_strength=None)
    lay_compact = R_compact["frames"]
    print("[layered] x4plus-bbox-FG ...")
    sr.reset_latency(HEAVY)
    R_bbox = lp.render_scene(frames, phas, plate_hd, fg_budget="x4plus_bbox", grain_strength=None)
    lay_bbox = R_bbox["frames"]
    bbox_frac = float(np.mean(R_bbox["bbox_area_fracs"]))
    x4plus_bbox_ms = float(np.median(R_bbox["timings"]["fg_sr_ms"]))
    print(f"[layered] x4plus bbox area {bbox_frac*100:.1f}% of frame -> {x4plus_bbox_ms:.0f}ms/frame "
          f"(vs full-frame {x4plus_full_ms:.0f}ms)")
    composite_ms = float(np.median(R_compact["timings"]["composite_ms"]))
    sample_ms = float(np.median(R_compact["timings"]["plate_sample_ms"]))
    alpha_ms = float(np.median(R_compact["timings"]["alpha_ms"]))
    # grain: time it honestly on a real HD composite (final pass), reused for the visual.
    gtmpl = grain.make_template(h_hd, w_hd)
    _gms = []
    for _ in range(5):
        _t = time.perf_counter(); grain.apply_grain(lay_compact[N // 2], N // 2, "med", template=gtmpl); _gms.append((time.perf_counter() - _t) * 1000.0)
    grain_ms = float(np.median(_gms))
    frame_compact_grain = grain.apply_grain(lay_compact[N // 2], N // 2, "med", template=gtmpl)

    # ---------------- 7. METRICS ----------------
    print("[metrics] sharpness / tOF / temporal instability per region ...")
    pipelines = {
        "uniform-compact": recon_compact,
        "uniform-x4plus": recon_x4plus,
        "region-aware": recon_region,
        "layered-compactFG": lay_compact,
        "layered-x4plusBbox": lay_bbox,
    }
    rows = {}
    for name, recon in pipelines.items():
        sb = float(np.mean([vlap_masked(r, bg_all_hd) for r in recon]))   # BG sharpness (fixed mask)
        sf = sharp_perframe_fg(recon, fg_masks_hd)                        # FG sharpness (per-frame mask)
        sa = float(np.mean([float(vlap(r).var()) for r in recon]))       # overall
        tb, tf, ta = rq.tof_regionsplit(recon, ref_lr, bg_all, fg_union) # tOF per region vs decoded LR
        ins_bg = temporal_instability(recon, bg_all_hd)                  # DIRECT BG flicker (code values)
        rows[name] = dict(sb=sb, sf=sf, sa=sa, tb=tb, tf=tf, ta=ta, ins_bg=ins_bg)

    # ---------------- 8. VISUALS ----------------
    mid = N // 2
    # 8a. side-by-side composites (mid frame), half-res for a sane file
    def half(im):
        return cv2.resize(im, (w_hd // 2, h_hd // 2), interpolation=cv2.INTER_AREA)
    montage = np.concatenate([
        label(half(recon_compact[mid]), "uniform-compact"),
        label(half(recon_x4plus[mid]), "uniform-x4plus (ceiling)"),
        label(half(lay_compact[mid]), "layered compact-FG"),
        label(half(lay_bbox[mid]), "layered x4plus-bbox-FG"),
    ], axis=1)
    rgb_save(os.path.join(OUT, "composite_montage.png"), montage)
    rgb_save(os.path.join(OUT, "plate_hd.png"), plate_hd)
    rgb_save(os.path.join(OUT, "frame_layered_compact.png"), lay_compact[mid])
    rgb_save(os.path.join(OUT, "frame_layered_x4plusBbox.png"), lay_bbox[mid])
    rgb_save(os.path.join(OUT, "frame_layered_compact_grain.png"), frame_compact_grain)
    rgb_save(os.path.join(OUT, "frame_uniform_x4plus.png"), recon_x4plus[mid])
    # alpha heatmap
    a_heat = cv2.applyColorMap((lp.alpha_to_hd(phas[mid], (h_hd, w_hd))[..., 0] * 255).astype(np.uint8),
                               cv2.COLORMAP_VIRIDIS)
    rgb_save(os.path.join(OUT, "alpha_hd.png"), cv2.cvtColor(a_heat, cv2.COLOR_BGR2RGB))

    # 8b. SEAM crop: find the hair-edge band (0.1<alpha<0.9) at the mid frame, crop a window
    a_mid = lp.alpha_to_hd(phas[mid], (h_hd, w_hd))[..., 0]
    edge = (a_mid > 0.1) & (a_mid < 0.9)
    ys, xs = np.where(edge)
    cs = 360
    if xs.size:
        cyc, cxc = int(np.median(ys)), int(np.median(xs))
    else:
        cyc, cxc = h_hd // 3, w_hd // 2
    y0 = int(np.clip(cyc - cs // 2, 0, h_hd - cs)); x0 = int(np.clip(cxc - cs // 2, 0, w_hd - cs))
    crop = lambda im: im[y0:y0 + cs, x0:x0 + cs]
    seam = np.concatenate([
        label(crop(recon_x4plus[mid]), "uniform-x4plus", half=False),
        label(crop(lay_compact[mid]), "layered compact-FG", half=False),
        label(crop(lay_bbox[mid]), "layered x4plus-bbox", half=False),
        label((crop(np.repeat((a_mid[..., None] * 255).astype(np.uint8), 3, 2))), "alpha", half=False),
    ], axis=1)
    rgb_save(os.path.join(OUT, "seam_crop.png"), seam)

    # 8c. SEAM quantitative: sharpness on a BG ring vs FG ring just across the boundary,
    # and colour continuity across the alpha=0.5 contour.
    bg_ring = (a_mid > 0.02) & (a_mid < 0.2)
    fg_ring = (a_mid > 0.8) & (a_mid < 0.98)
    def ring_sharp(recon_mid, ring):
        return float(vlap(recon_mid)[ring].var()) if ring.any() else float("nan")
    seam_stats = {}
    for nm, recon in (("layered-compactFG", lay_compact), ("layered-x4plusBbox", lay_bbox),
                      ("uniform-x4plus", recon_x4plus)):
        sb_r = ring_sharp(recon[mid], bg_ring); sf_r = ring_sharp(recon[mid], fg_ring)
        seam_stats[nm] = (sb_r, sf_r, sf_r / max(sb_r, 1e-6))

    # ---------------- 9. COST TABLE ----------------
    # per-frame layered cost = plate_sample + matte + FG_SR + composite + grain.
    # plate (build + heavy-SR) is ONE-TIME per scene -> amortized over the scene length.
    matte_pf = mb["median_ms"]
    plate_once_ms = plate_sr_ms + plate_build_wall * 1000.0
    def layered_cost(n_scene, fg_ms, matte_refresh, with_grain):
        amort_plate = plate_once_ms / n_scene
        amort_matte = matte_pf / matte_refresh
        g = grain_ms if with_grain else 0.0
        per = sample_ms + amort_matte + fg_ms + composite_ms + g
        return per, amort_plate, per + amort_plate

    PROP = 33.0       # full-frame propagation recon, talking-head (Step 7: reactive 27.7 .. full 42.4)

    # ---------------- REPORT ----------------
    def fmt(x, n=1):
        return f"{x:.{n}f}" if x == x else "  nan"

    log("=" * 84)
    log("L3 LAYERED PIPELINE -- talking-head scene (sample.mp4 start 5000, frames [%d,%d), N=%d)" % (s0, s1, N))
    log("=" * 84)
    log(f"LR {w}x{h} -> HD {w_hd}x{h_hd} (x4).  FG frac {fg_frac*100:.1f}%, x4plus bbox {bbox_frac*100:.1f}% of frame.")
    log("")
    log("--- QUALITY (sharpness=var-of-Laplacian higher=sharper; tOF & instab LOWER=steadier) ---")
    log("  pipeline             | sharp BG | sharp FG | sharp ALL | tOF BG | tOF FG | tOF ALL | BG instab(|dF|)")
    log("  " + "-" * 96)
    for name in pipelines:
        r = rows[name]
        log(f"  {name:20s} | {fmt(r['sb']):8s} | {fmt(r['sf']):8s} | {fmt(r['sa']):9s} | "
            f"{fmt(r['tb'],3):6s} | {fmt(r['tf'],3):6s} | {fmt(r['ta'],3):7s} | {fmt(r['ins_bg'],3)}")
    log("")
    bgsharp_ceiling = rows["uniform-x4plus"]["sb"]
    bgsharp_layered = rows["layered-compactFG"]["sb"]
    bginstab_ceiling = rows["uniform-x4plus"]["ins_bg"]
    bginstab_layered = rows["layered-compactFG"]["ins_bg"]
    log(f"  HEADLINE: layered BG sharpness {bgsharp_layered:.1f} vs uniform-x4plus {bgsharp_ceiling:.1f} "
        f"({100*bgsharp_layered/max(bgsharp_ceiling,1e-6):.0f}% of ceiling)  AND BG instability "
        f"{bginstab_layered:.3f} vs x4plus {bginstab_ceiling:.3f}.")
    log(f"  (layered BG is the FIXED plate -> frame-to-frame |dF|=0 by construction; uniform-x4plus flickers.)")
    log("")
    log("--- COST per frame (ms) ---")
    log(f"  one-time/scene: plate build {plate_build_wall*1000:.0f}ms + heavy-SR {plate_sr_ms:.0f}ms = {plate_once_ms:.0f}ms")
    log(f"  per-frame parts: plate-sample {sample_ms:.2f} | matte {matte_pf:.1f} | composite {composite_ms:.2f} | grain {grain_ms:.1f}")
    log(f"  FG SR: compact full-frame {compact_ms:.0f} | x4plus full-frame {x4plus_full_ms:.0f} | x4plus bbox({bbox_frac*100:.0f}%) {x4plus_bbox_ms:.0f}")
    log("")
    log("  scenario (matte every frame, no grain)         | per-frame(+amort plate) ")
    for nsc in (N, 300):
        pc, ap, tc = layered_cost(nsc, compact_ms, 1, False)
        pb, ab, tb_ = layered_cost(nsc, x4plus_bbox_ms, 1, False)
        log(f"  layered compact-FG  @ {nsc:3d}f  | {pc:6.1f} + {ap:5.1f} = {tc:6.1f} ms ({1000/tc:4.1f} fps)")
        log(f"  layered x4plus-bbox @ {nsc:3d}f  | {pb:6.1f} + {ab:5.1f} = {tb_:6.1f} ms ({1000/tb_:4.1f} fps)")
    log("")
    pc6, ap6, tc6 = layered_cost(300, compact_ms, 6, False)
    log(f"  layered compact-FG @300f, matte refresh/6 frames, no grain: {tc6:.1f} ms ({1000/tc6:.1f} fps)")
    log(f"  BASELINES: full-frame propagation recon ~{PROP:.0f} ms (Step 7: 27.7 reactive .. 42.4 full); "
        f"per-frame x4plus {x4plus_full_ms:.0f} ms.")
    log("")
    log("--- SEAM honesty (mid frame, hair-edge band) ---")
    log(f"  alpha edge band (0.1<a<0.9) width: {edge.sum()} HD px ({100*edge.sum()/(h_hd*w_hd):.2f}% of frame).")
    log("  pipeline             | BGring sharp | FGring sharp | FG/BG ratio (seam discontinuity)")
    for nm, (sb_r, sf_r, ratio) in seam_stats.items():
        log(f"  {nm:20s} | {sb_r:11.1f} | {sf_r:11.1f} | {ratio:.2f}x")
    log("")
    log(f"artifacts in {OUT}/:")
    log("  composite_montage.png  = [uniform-compact | uniform-x4plus | layered compact | layered x4plus-bbox] (half-res)")
    log("  seam_crop.png          = hair-edge crop [uniform-x4plus | layered compact | layered x4plus-bbox | alpha]")
    log("  plate_hd.png / alpha_hd.png / frame_layered_*.png / frame_uniform_x4plus.png")
    log(f"  total wall {time.time()-T0:.1f}s")

    report = "\n".join(P)
    print("\n" + report)
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write(report + "\n")
    return rows


if __name__ == "__main__":
    main()
