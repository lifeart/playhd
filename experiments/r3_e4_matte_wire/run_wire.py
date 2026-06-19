"""run_wire.py -- R3-E4: verify the FULL layered swap (PASS A plate + PASS B per-frame
composite) end-to-end through the SERVER'S layered path with the permissive matte.

We import server/layered_api READ-ONLY and MONKEY-PATCH ONLY its `matting` module reference
(the swap the lead will land behind a flag) to point at our completed adapter, then drive the
EXACT server functions:

  PASS 0   layered_api.segment_scenes(clip)
  PASS A   layered_api.build_scene_plates(clip, segs, plate_dir, model)   # uses matting.matte_sequence
                                                                          # + matting.fg_mask_lr
  PASS B   layered_api.matte_frame_np(model, img, rec, ratio, device)     # uses matting.matte_frame (NEW)
           layered_api.composite_frame(img, pha, plate, w_hd, h_hd)       # alpha*fg_hd + (1-a)*plate_hd

Window: sample.mp4 start-frame 5000, scene[0] (N=32 real LR talking-head frames, 640x320), the
static-camera scene where the layered plate path activates. We encode that window to a temp H.264
clip with PyAV (system ffmpeg is broken) so the server's streaming decode + codec-MV static check
run for real. Configs: RVM (pseudo-GT, non-commercial) vs DeepLabV3-MobileNetV3-Large + alpha-EMA(0.5)
(R2-E1 recommended permissive pick). GPU (MPS) shared -> small window, free GPU between configs.

Run:  python3 experiments/r3_e4_matte_wire/run_wire.py
"""
import os
import sys
import gc
import json
import time

import numpy as np
import cv2
import av

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_PROTO = os.path.join(_REPO, "prototype")
_SERVER = os.path.join(_REPO, "server")
for p in (_PROTO, _SERVER, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

# --- HARD RULE: never block on a model download (R2-E1 policy). Cached weights still load. ---
import torch.hub as _hub  # noqa: E402
def _blocked(*_a, **_k):
    raise RuntimeError("download blocked (no-download policy: weights not cached)")
_hub.download_url_to_file = _blocked

import layered_api  # noqa: E402  READ-ONLY (server's layered path)
import matting  # noqa: E402  READ-ONLY (RVM = pseudo-GT, non-commercial)
import seg_matte_layered as seg  # noqa: E402  our completed permissive adapter (PASS A + PASS B)

OUT = os.path.join(_HERE, "out")
os.makedirs(OUT, exist_ok=True)
CACHE_FRAMES = os.path.join(_REPO, "experiments", "r2_e1_matte", "out", "cache", "frames_5000_48.npy")
FPS = 25


def free_gpu():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


def load_window():
    """The validated R2-E1 scene[0]: 32 real LR talking-head frames (HxWx3 RGB uint8)."""
    arr = np.load(CACHE_FRAMES)            # (48, 320, 640, 3); R2-E1 scene[0] = first 32
    frames = [np.ascontiguousarray(arr[i]) for i in range(min(32, arr.shape[0]))]
    return frames


def encode_clip(frames, path, fps=FPS):
    """Encode RGB frames -> H.264 mp4 with PyAV (ffmpeg CLI broken). yuv420p so MVs export on decode."""
    h, w = frames[0].shape[:2]
    cont = av.open(path, mode="w")
    st = cont.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": "18", "g": "1000"}   # high quality, no forced extra keyframes inside window
    for f in frames:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
        for pkt in st.encode(frame):
            cont.mux(pkt)
    for pkt in st.encode():
        cont.mux(pkt)
    cont.close()
    return path


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def vlap_var(rgb, mask=None):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    if mask is None:
        return float(lap.var())
    return float(lap[mask].var()) if mask.any() else float("nan")


def alpha_temporal(phas):
    """Whole-frame alpha temporal diff a|dF| and soft-edge-band diff band|dF| (GOTCHA #17)."""
    band = np.zeros_like(phas[0], bool)
    for p in phas:
        band |= (p > 0.05) & (p < 0.95)
    a_dF, band_dF = [], []
    for t in range(1, len(phas)):
        d = np.abs(phas[t] - phas[t - 1])
        a_dF.append(float(d.mean()))
        band_dF.append(float(d[band].mean()) if band.any() else float("nan"))
    return float(np.mean(a_dF)), float(np.nanmean(band_dF)), band


def bgplate_stability(comps, union_alpha_hd, thr=0.05):
    """|out_t - out_{t-1}| mean over the HD BACKGROUND region (union alpha < thr). The plate is
    fixed, so a correct composite has a near-zero, MATTE-INDEPENDENT background |dF|."""
    bg = (union_alpha_hd[..., 0] < thr)
    dl = []
    for t in range(1, len(comps)):
        d = np.abs(comps[t].astype(np.float32) - comps[t - 1].astype(np.float32)).mean(axis=2)
        dl.append(float(d[bg].mean()) if bg.any() else float("nan"))
    return float(np.mean(dl))


# --------------------------------------------------------------------------- #
# One config end-to-end through the REAL layered_api (matting reference patched)
# --------------------------------------------------------------------------- #
def run_config(name, matting_mod, model, clip, segs, plate_dir):
    print(f"\n===== [{name}] PASS A + PASS B through layered_api =====")
    free_gpu()
    layered_api.matting = matting_mod          # THE SWAP: patch the matting reference only
    os.makedirs(plate_dir, exist_ok=True)
    device = getattr(model, "_rvm_device", "cpu")

    # ---- PASS A: build + heavy-SR one plate per scene (real server function) ----
    tA = time.perf_counter()
    plates = layered_api.build_scene_plates(clip, segs, plate_dir, model)
    passA_s = time.perf_counter() - tA

    # ---- PASS B: stream the clip, composite per frame (mirror pipeline_api._run_layered) ----
    tB = time.perf_counter()
    h_lr, w_lr = None, None
    w_hd = h_hd = ratio = None
    cur_plate_sid, cur_plate = None, None
    rvm_sid, rec = None, [None] * 4
    comps, phas = [], []
    n_compose = 0
    for idx, ptype, img, mvs in layered_api.stream_frames(clip):
        if h_lr is None:
            h_lr, w_lr = img.shape[:2]
            w_hd, h_hd = w_lr * layered_api.SCALE, h_lr * layered_api.SCALE
            ratio = layered_api.downsample_ratio(h_lr, w_lr)
        sid = layered_api.scene_of(idx, segs)
        info = plates[sid]
        if info["fallback"]:
            rvm_sid = None                      # region-aware fallback path (not exercised here)
            comps.append(None); phas.append(None)
            continue
        if cur_plate_sid != sid:                # load ONE HD plate at a time (bounded)
            cur_plate = np.load(info["plate_path"]); cur_plate_sid = sid
        if rvm_sid != sid:                      # reset recurrent state at scene boundary
            rec, rvm_sid = [None] * 4, sid
        pha, rec = layered_api.matte_frame_np(model, img, rec, ratio, device)   # NEW matte_frame seam
        comp = layered_api.composite_frame(img, pha, cur_plate, w_hd, h_hd)
        comps.append(comp); phas.append(pha)
        n_compose += 1
    passB_s = time.perf_counter() - tB
    free_gpu()

    # focus on the first static (non-fallback) scene for the matte/plate metrics
    static_sids = [s for s in plates if not plates[s]["fallback"]]
    sid0 = static_sids[0] if static_sids else None
    info0 = plates[sid0] if sid0 is not None else None
    s0, s1 = (info0["seg"] if info0 else (0, len(comps)))
    sc_phas = [phas[i] for i in range(s0, s1) if phas[i] is not None]
    sc_comps = [comps[i] for i in range(s0, s1) if comps[i] is not None]

    # validity of QHD output
    ex = sc_comps[len(sc_comps) // 2]
    valid = (ex.dtype == np.uint8 and ex.shape == (h_hd, w_hd, 3) and np.isfinite(ex).all())

    # alpha stability + human-only fg fraction
    a_dF, band_dF, _band = alpha_temporal(sc_phas)
    fg_pct = 100.0 * float(np.mean([(p >= 0.5).mean() for p in sc_phas]))

    # HD union alpha (for the bg-plate stability region)
    union_lr = np.maximum.reduce([p.astype(np.float32) for p in sc_phas])
    union_hd = cv2.resize(union_lr, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)[..., None]
    bg_dF = bgplate_stability(sc_comps, union_hd)

    res = dict(
        name=name, n_scenes=len(plates), n_static=len(static_sids),
        verdict=info0["verdict"] if info0 else "n/a",
        coverage_pct=info0["coverage_pct"] if info0 else float("nan"),
        hole_pct=info0["hole_pct"] if info0 else float("nan"),
        plate_path=info0["plate_path"] if info0 else None,
        n_compose=n_compose, qhd_valid=bool(valid), qhd_shape=list(ex.shape),
        fg_pct=fg_pct, alpha_dF=a_dF, band_dF=band_dF, bgplate_dF=bg_dF,
        passA_s=round(passA_s, 2), passB_s=round(passB_s, 2),
    )
    # plate sharpness (var-Laplacian) from the spilled HD plate
    plate_hd = np.load(info0["plate_path"]) if info0 else None
    res["plate_sharp"] = vlap_var(plate_hd) if plate_hd is not None else float("nan")
    print(f"[{name}] scenes={res['n_scenes']} static={res['n_static']} verdict={res['verdict']} "
          f"cov={res['coverage_pct']}% hole={res['hole_pct']}% sharp={res['plate_sharp']:.0f}")
    print(f"[{name}] composed {n_compose} frames; QHD valid={valid} {ex.shape}; "
          f"fg%={fg_pct:.1f} a|dF|={a_dF:.4f} band|dF|={band_dF:.4f} bgplate|dF|={bg_dF:.3f}")
    return res, sc_phas, sc_comps, plate_hd


def main():
    T0 = time.time()
    frames = load_window()
    h, w = frames[0].shape[:2]
    print(f"window: {len(frames)} frames {w}x{h} (sample.mp4 start 5000, scene[0])")

    clip = os.path.join(OUT, "window_s5000.mp4")
    encode_clip(frames, clip)
    print(f"encoded temp clip -> {clip}")

    segs, total = layered_api.segment_scenes(clip)
    print(f"segment_scenes -> {len(segs)} scene(s) {segs}, total={total}")

    # ---- RVM baseline (pseudo-GT) ----
    rvm_model = matting.load_rvm("mps", "mobilenetv3")
    rvm = run_config("RVM", matting, rvm_model, clip,
                     segs, os.path.join(OUT, "plates_rvm"))
    del rvm_model
    free_gpu()
    rvm_res, rvm_phas, rvm_comps, rvm_plate = rvm

    # ---- permissive: DeepLabV3-MobileNetV3-Large + alpha-EMA(0.5) ----
    seg_model = seg.load_seg("mps", "deeplabv3_mobilenetv3", ema=0.5)
    sg = run_config("DeepLab-mv3+EMA", seg, seg_model, clip,
                    segs, os.path.join(OUT, "plates_seg"))
    del seg_model
    free_gpu()
    seg_res, seg_phas, seg_comps, seg_plate = sg

    layered_api.matting = matting              # restore (hygiene)

    # ---- cross metrics vs RVM ----
    n = min(len(rvm_phas), len(seg_phas))
    mad = float(np.mean([np.abs(seg_phas[i] - rvm_phas[i]).mean() for i in range(n)]))
    iou = float(np.mean([
        ((seg_phas[i] >= 0.5) & (rvm_phas[i] >= 0.5)).sum() /
        max(((seg_phas[i] >= 0.5) | (rvm_phas[i] >= 0.5)).sum(), 1)
        for i in range(n)]))
    # plate subject-bleed: |seg_plate - rvm_plate| inside the RVM union-FG (HD)
    rvm_union_lr = np.maximum.reduce([p.astype(np.float32) for p in rvm_phas])
    H_hd, W_hd = rvm_plate.shape[:2]
    rvm_union_hd = cv2.resize(rvm_union_lr, (W_hd, H_hd), interpolation=cv2.INTER_LINEAR) >= 0.5
    pdiff = np.abs(seg_plate.astype(np.int16) - rvm_plate.astype(np.int16)).mean(axis=2)
    bleed = float(pdiff[rvm_union_hd].mean()) if rvm_union_hd.any() else float("nan")
    plate_mad = float(pdiff.mean())

    # composite agreement in the BACKGROUND (plate-dominated) region -> should be ~equal
    bg = ~rvm_union_hd
    cdiff = float(np.mean([
        np.abs(seg_comps[i].astype(np.float32) - rvm_comps[i].astype(np.float32)).mean(axis=2)[bg].mean()
        for i in range(n)])) if bg.any() else float("nan")

    # ---- montage (RVM vs seg: alpha | plate-crop | composite-crop), middle frame ----
    mid = n // 2
    def amap(p):
        return cv2.cvtColor((np.clip(p, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    def lab(im, t):
        o = im.copy()
        cv2.rectangle(o, (0, 0), (o.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(o, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return o
    cw, ch = W_hd // 4, H_hd // 4                       # a quarter-crop so the montage stays small
    def crop(im):
        return im[:ch, W_hd // 2 - cw // 2: W_hd // 2 + cw // 2]
    row_rvm = np.concatenate([
        lab(cv2.resize(amap(rvm_phas[mid]), (cw, ch)), "RVM alpha"),
        lab(crop(rvm_plate), "RVM plate"),
        lab(crop(rvm_comps[mid]), "RVM composite")], axis=1)
    row_seg = np.concatenate([
        lab(cv2.resize(amap(seg_phas[mid]), (cw, ch)), "DeepLab+EMA alpha"),
        lab(crop(seg_plate), "DeepLab+EMA plate"),
        lab(crop(seg_comps[mid]), "DeepLab+EMA composite")], axis=1)
    montage = np.concatenate([row_rvm, row_seg], axis=0)
    mpath = os.path.join(OUT, "wire_montage.png")
    cv2.imwrite(mpath, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

    # ---- report ----
    L = []
    pr = L.append
    pr("=" * 96)
    pr("R3-E4  FULL LAYERED SWAP (PASS A plate + PASS B per-frame composite) through layered_api")
    pr(f"  window: sample.mp4 start 5000 scene[0]  N={len(frames)}  LR {w}x{h} -> QHD {W_hd}x{H_hd}")
    pr("  RVM = pseudo-GT (NON-COMMERCIAL).  DeepLab-mv3+EMA = BSD-3 permissive (commercial-OK).")
    pr("=" * 96)
    hdr = (f"{'config':18s}| {'scenes':>6s} {'verdict':>7s} | {'cov%':>5s} {'hole%':>5s} {'sharp':>6s} |"
           f" {'compose':>7s} {'QHD ok':>6s} | {'fg%':>5s} {'a|dF|':>6s} {'band|dF|':>8s} {'bgPlate|dF|':>10s}")
    pr(hdr); pr("-" * len(hdr))
    for r in (rvm_res, seg_res):
        pr(f"{r['name']:18s}| {r['n_scenes']:6d} {r['verdict']:>7s} | "
           f"{r['coverage_pct']:5.1f} {r['hole_pct']:5.2f} {r['plate_sharp']:6.0f} | "
           f"{r['n_compose']:7d} {str(r['qhd_valid']):>6s} | "
           f"{r['fg_pct']:5.1f} {r['alpha_dF']:6.4f} {r['band_dF']:8.4f} {r['bgplate_dF']:10.3f}")
    pr("-" * len(hdr))
    pr(f"seg vs RVM matte:  MAD={mad:.4f}  IoU={iou:.3f}")
    pr(f"seg plate vs RVM plate:  plate_MAD={plate_mad:.2f}  subject_bleed(inFG)={bleed:.2f}")
    pr(f"composite BG-region agreement |seg-RVM|={cdiff:.2f} code-values (plate-dominated; ~0 => same bg)")
    pr(f"plate sharpness %RVM = {100.0*seg_res['plate_sharp']/max(rvm_res['plate_sharp'],1e-6):.0f}%")
    pr(f"timing ratios (shared GPU): passA seg/RVM = {seg_res['passA_s']/max(rvm_res['passA_s'],1e-6):.2f}x, "
       f"passB seg/RVM = {seg_res['passB_s']/max(rvm_res['passB_s'],1e-6):.2f}x")
    pr(f"artifacts: {mpath}")
    pr(f"total wall {time.time()-T0:.1f}s")
    report = "\n".join(L)
    print("\n" + report)
    with open(os.path.join(OUT, "wire_summary.txt"), "w") as fh:
        fh.write(report + "\n")
    json.dump({"rvm": rvm_res, "seg": seg_res,
               "cross": {"mad": mad, "iou": iou, "plate_mad": plate_mad, "bleed": bleed,
                         "composite_bg_diff": cdiff}},
              open(os.path.join(OUT, "wire_rows.json"), "w"), indent=2, default=float)
    return rvm_res, seg_res


if __name__ == "__main__":
    main()
