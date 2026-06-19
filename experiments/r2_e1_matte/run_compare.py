"""run_compare.py -- R2-E1: permissive matte vs RVM on the talking-head window.

Compares RVM (pseudo-GT, NON-COMMERCIAL) against permissive (BSD-3) torchvision
person-segmentation nets on sample.mp4 start-frame 5000, scene[0]. Measures:
  * matte quality vs RVM  : MAD, IoU
  * matte EDGE temporal stability: |dF| of the alpha edge map frame-to-frame (CRAWL),
    plus alpha temporal |d-alpha| overall and in the edge band.
  * FG % of frame
  * latency ms/frame on MPS, reported as a RATIO vs RVM (shared GPU caveat noted).
  * LAYERED SURVIVAL: build the background plate from EACH matte's gates and compare
    coverage / always-occluded hole% / plate sharpness / subject-bleed vs the RVM plate.

GPU (MPS) is SHARED with sibling experiments -> small window, gc+empty_cache between
configs, timings reported as ratios. prototype/ imported READ-ONLY.

Run:  python3 experiments/r2_e1_matte/run_compare.py
"""
import os
import sys
import gc
import time
import json

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "prototype")
sys.path.insert(0, _PROTO)          # prototype READ-ONLY
sys.path.insert(0, _HERE)

import torch                         # noqa: E402

# --- R2-E1 HARD RULE: NEVER block on a model download. ----------------------- #
# A prior run stalled for minutes pulling DeepLab weights over a slow link. We make
# ANY attempt to fetch weights fail FAST instead of hanging: torch.hub's downloader
# is replaced with a raiser. Already-cached weights still load (load_state_dict_from_url
# skips the download when the file exists), so RVM + LRASPP + DeepLab-mv3 (all cached)
# run; only an UNcached variant (DeepLab-r50, .partial) trips this and is skipped+inferred.
import torch.hub as _hub             # noqa: E402
def _blocked_download(*_a, **_k):
    raise RuntimeError("download blocked (R2-E1 no-download policy: weights not cached)")
_hub.download_url_to_file = _blocked_download

import derisk                        # noqa: E402  READ-ONLY
import matting                       # noqa: E402  READ-ONLY (RVM = pseudo-GT)
import background_plate as bp        # noqa: E402  READ-ONLY (L2)
import seg_matte                     # noqa: E402  the permissive adapter (this experiment)

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "sample.mp4")
OUT = os.path.join(_HERE, "out")
CACHE = os.path.join(OUT, "cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
START, NWIN = 5000, 48
DILATE = 3


def free_gpu():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


# --------------------------------------------------------------------------- #
# Decode the scene[0] window once, cache to disk (decode-from-0 is the slow part).
# --------------------------------------------------------------------------- #
def load_scene():
    fcache = os.path.join(CACHE, f"frames_{START}_{NWIN}.npy")
    mcache = os.path.join(CACHE, f"meta_{START}_{NWIN}.json")
    if os.path.exists(fcache) and os.path.exists(mcache):
        frames = np.load(fcache)
        meta = json.load(open(mcache))
        print(f"[decode] cache hit: scene {frames.shape} {meta}")
        return [np.ascontiguousarray(frames[i]) for i in range(frames.shape[0])], meta
    print(f"[decode] window {NWIN}f from start {START} (decodes from 0; one-time)...")
    t0 = time.time()
    decoded = derisk.decode_lr_and_mvs(SAMPLE, start_frame=START, max_frames=NWIN)
    frames_all = [img for (_p, img, _m) in decoded]
    segs = bp.scene_segments(decoded, frames=frames_all)
    s0, s1 = segs[0]
    frames = frames_all[s0:s1]
    h, w = frames[0].shape[:2]
    meta = dict(s0=int(s0), s1=int(s1), N=len(frames), h=int(h), w=int(w),
                decode_wall_s=round(time.time() - t0, 1))
    np.save(fcache, np.stack(frames))
    json.dump(meta, open(mcache, "w"))
    print(f"[decode] scene[0]=[{s0},{s1}) N={len(frames)} {w}x{h} in {meta['decode_wall_s']}s")
    return [np.ascontiguousarray(f) for f in frames], meta


# --------------------------------------------------------------------------- #
# Matte metrics
# --------------------------------------------------------------------------- #
def bin_mask(pha, thr=0.5):
    return (np.asarray(pha, np.float32) >= thr)


def edge_map(pha):
    """Soft alpha-edge map = Sobel gradient magnitude of the alpha (normalized [0,1])."""
    a = np.asarray(pha, np.float32)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def matte_metrics(phas, phas_gt):
    """MAD / IoU vs GT (per-frame mean), and SELF temporal-stability of the matte:
      edge_dF  : mean over consecutive frames of mean|edge_t - edge_{t-1}|  (EDGE CRAWL)
      alpha_dF : mean|alpha_t - alpha_{t-1}| over the whole frame
      band_dF  : mean|alpha_t - alpha_{t-1}| restricted to the union edge band (0.05..0.95)
    """
    mads, ious, fgfracs = [], [], []
    for p, g in zip(phas, phas_gt):
        mads.append(float(np.abs(p - g).mean()))
        a, b = bin_mask(p), bin_mask(g)
        inter = float((a & b).sum())
        union = float((a | b).sum())
        ious.append(inter / union if union > 0 else 1.0)
        fgfracs.append(float(a.mean()))
    edges = [edge_map(p) for p in phas]
    band = np.zeros_like(phas[0], dtype=bool)
    for p in phas:
        band |= (np.asarray(p, np.float32) > 0.05) & (np.asarray(p, np.float32) < 0.95)
    edge_dF, alpha_dF, band_dF = [], [], []
    for t in range(1, len(phas)):
        edge_dF.append(float(np.abs(edges[t] - edges[t - 1]).mean()))
        d = np.abs(phas[t] - phas[t - 1])
        alpha_dF.append(float(d.mean()))
        band_dF.append(float(d[band].mean()) if band.any() else float("nan"))
    return dict(
        mad=float(np.mean(mads)), iou=float(np.mean(ious)),
        fg_pct=100.0 * float(np.mean(fgfracs)),
        edge_dF=float(np.mean(edge_dF)),
        alpha_dF=float(np.mean(alpha_dF)),
        band_dF=float(np.nanmean(band_dF)),
    )


def vlap_var(rgb, mask=None):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    if mask is None:
        return float(lap.var())
    return float(lap[mask].var()) if mask.any() else float("nan")


# --------------------------------------------------------------------------- #
# Run one matte source -> (phas, gates, latency, build a plate)
# --------------------------------------------------------------------------- #
def run_source(name, loader, frames):
    print(f"\n[{name}] loading + matting ({len(frames)} frames)...")
    free_gpu()
    model = loader()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    res = model_matte = None
    # matte (recurrent state / EMA threaded in display order)
    res = (matting if name == "RVM" else seg_matte).matte_sequence(model, frames)
    phas = [p for (_f, p) in res]
    h, w = frames[0].shape[:2]
    gates = [(matting if name == "RVM" else seg_matte).fg_mask_lr(
        p, lr_hw=(h, w), soft=False, thresh=0.5, dilate=DILATE) for p in phas]
    # honest per-frame latency (MPS-synced)
    bench = (matting if name == "RVM" else seg_matte).benchmark(model, frames)
    print(f"[{name}] {nparams:.2f}M params, latency {bench['median_ms']:.1f}ms/frame (median, MPS-synced)")
    del model
    free_gpu()
    return dict(phas=phas, gates=gates, lat_ms=bench["median_ms"], nparams=nparams)


def build_and_score_plate(frames, gates):
    plate, cov, hole = bp.build_plate(frames, gates, min_samples=1)
    rep = bp.coverage_report(cov, hole)
    rep["sharp"] = vlap_var(plate)            # var-of-Laplacian (sharper / more detail = higher)
    return plate, rep


def main():
    T0 = time.time()
    frames, meta = load_scene()
    h, w, N = meta["h"], meta["w"], meta["N"]

    sources = {
        "RVM": lambda: matting.load_rvm("mps", "mobilenetv3"),
        "LRASPP-mv3": lambda: seg_matte.load_seg("mps", "lraspp_mobilenetv3", ema=0.0),
        "LRASPP-mv3+EMA": lambda: seg_matte.load_seg("mps", "lraspp_mobilenetv3", ema=0.5),
        "DeepLab-mv3": lambda: seg_matte.load_seg("mps", "deeplabv3_mobilenetv3", ema=0.0),
        "DeepLab-r50": lambda: seg_matte.load_seg("mps", "deeplabv3_resnet50", ema=0.0),
    }

    R = {}
    skipped = {}
    for name, loader in sources.items():
        try:
            R[name] = run_source(name, loader, frames)
        except Exception as e:
            msg = str(e).splitlines()[0][:140]
            skipped[name] = msg
            print(f"[{name}] SKIPPED (no-download policy / weights not cached): {msg}")
            free_gpu()
    ok = [n for n in sources if n in R]            # successfully-run sources, in order
    if "RVM" not in ok:
        raise RuntimeError("RVM (pseudo-GT) failed to run; cannot compare")

    # RVM is the pseudo-GT reference for matte quality + plate comparison.
    gt = R["RVM"]["phas"]

    # union FG region from RVM (where the subject ever is) -> for subject-bleed metric.
    rvm_unionfg = np.zeros((h, w), bool)
    for p in gt:
        rvm_unionfg |= bin_mask(p)

    # plates
    plates, plate_reps = {}, {}
    for name in ok:
        plates[name], plate_reps[name] = build_and_score_plate(frames, R[name]["gates"])
    rvm_plate = plates["RVM"]

    # matte metrics + plate-vs-RVM-plate metrics
    rows = {}
    for name in ok:
        mm = matte_metrics(R[name]["phas"], gt)
        # subject bleed: |plate - rvm_plate| restricted to RVM union-FG (where leak shows)
        diff = np.abs(plates[name].astype(np.int16) - rvm_plate.astype(np.int16)).mean(axis=2)
        bleed = float(diff[rvm_unionfg].mean()) if rvm_unionfg.any() else float("nan")
        plate_mad = float(diff.mean())
        rows[name] = dict(
            **mm, lat_ms=R[name]["lat_ms"], nparams=R[name]["nparams"],
            cov1=plate_reps[name]["pct_ge1"], cov3=plate_reps[name]["pct_ge3"],
            hole=plate_reps[name]["hole_pct"], sharp=plate_reps[name]["sharp"],
            plate_mad=plate_mad, bleed=bleed,
        )

    rvm_lat = rows["RVM"]["lat_ms"]
    rvm_sharp = rows["RVM"]["sharp"]

    # ----------------------------- VISUALS ----------------------------- #
    mid = N // 2
    def lab(im, t):
        out = im.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(out, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return out
    def amap(pha):
        return cv2.cvtColor((np.clip(pha, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    rows_imgs = []
    rows_imgs.append(np.concatenate([lab(amap(R[n]["phas"][mid]), f"{n} alpha") for n in ok], axis=1))
    rows_imgs.append(np.concatenate([lab(plates[n], f"{n} plate") for n in ok], axis=1))
    montage = np.concatenate(rows_imgs, axis=0)
    cv2.imwrite(os.path.join(OUT, "matte_plate_montage.png"), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

    # ----------------------------- REPORT ----------------------------- #
    L = []
    p = L.append
    p("=" * 110)
    p(f"R2-E1  PERMISSIVE MATTE vs RVM   sample.mp4 start {START}, scene[0] N={N}  LR {w}x{h}")
    p("  RVM = pseudo-GT (NON-COMMERCIAL).  others = BSD-3 torchvision person-seg (commercial-OK).")
    p("  GPU (MPS) SHARED with sibling experiments -> latency reported as RATIO vs RVM.")
    p("=" * 110)
    hdr = (f"{'candidate':16s}| {'MAD':>6s} {'IoU':>5s} | {'FG%':>5s} | "
           f"{'edge|dF|':>8s} {'a|dF|':>6s} {'band|dF|':>8s} | {'lat ms':>6s} {'xRVM':>5s} | "
           f"{'cov>=1':>6s} {'hole%':>6s} {'sharp':>6s} {'%RVM':>5s} | {'pMAD':>5s} {'bleed':>5s}")
    p(hdr)
    p("-" * len(hdr))
    for n in ok:
        r = rows[n]
        latr = r["lat_ms"] / rvm_lat
        shr = 100.0 * r["sharp"] / rvm_sharp
        p(f"{n:16s}| {r['mad']:6.3f} {r['iou']:5.2f} | {r['fg_pct']:5.1f} | "
          f"{r['edge_dF']:8.4f} {r['alpha_dF']:6.4f} {r['band_dF']:8.4f} | "
          f"{r['lat_ms']:6.1f} {latr:5.2f} | "
          f"{r['cov1']:6.1f} {r['hole']:6.2f} {r['sharp']:6.0f} {shr:5.0f} | "
          f"{r['plate_mad']:5.2f} {r['bleed']:5.2f}")
    p("-" * len(hdr))
    p("legend: MAD/IoU vs RVM (RVM row = self=0/1).  edge|dF|=alpha-edge crawl (LOWER=steadier).")
    p("  a|dF|=whole-frame alpha temporal diff. band|dF|=diff in soft edge band. lat xRVM=latency ratio.")
    p("  cov>=1/hole%/sharp = plate built from THIS matte's gates. pMAD=plate mean|.-RVMplate|.")
    p("  bleed=plate |.-RVMplate| INSIDE RVM union-FG (subject leak into plate; LOWER=cleaner).")
    p("")
    p(f"RVM latency baseline: {rvm_lat:.1f} ms/frame.  RVM plate sharpness: {rvm_sharp:.0f}.")
    if skipped:
        p("")
        for n, m in skipped.items():
            p(f"SKIPPED (download blocked -> INFERRED from model card, not measured): {n}  [{m}]")
    p(f"artifacts: {OUT}/matte_plate_montage.png  (row0=alphas, row1=plates; columns=candidates)")
    p(f"total wall {time.time()-T0:.1f}s")
    report = "\n".join(L)
    print("\n" + report)
    with open(os.path.join(OUT, "compare_summary.txt"), "w") as fh:
        fh.write(report + "\n")
    # machine-readable
    json.dump({"rows": {n: rows[n] for n in ok}, "skipped": skipped},
              open(os.path.join(OUT, "compare_rows.json"), "w"), indent=2)
    return rows


if __name__ == "__main__":
    main()
