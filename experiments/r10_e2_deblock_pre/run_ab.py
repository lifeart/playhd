#!/usr/bin/env python3
"""
R10-E2: Does removing H.264 codec artifacts BEFORE x4plus beat plain x4plus?

Novel angle (distinct from R9-E2's "replace the anchor", which NO-GO'd): keep the
validated x4plus ceiling, but feed it a CLEANER input -- run a codec-artifact-removal
(deblock/restoration) pass on the LR frame FIRST, then x4plus. If removing H.264's
blocking/ringing lets x4plus do its job, it integrates as a cheap PREPROCESSOR on the
sparse anchor (~2-12% of frames), not a model swap.

Protocol == R9-E2's REAL-H.264 harness (the methodology the task mandates I reuse):
  * GT  = decoded sample.mp4 256-crop (R6-E1 validated windows) -- pseudo-HD.
  * LR  = 2x INTER_AREA down (->128) -> REAL libx264 (PyAV) encode @CRF -> decode
          -> genuine H.264 artifacts (8x8 blocking / ringing / 4:2:0 chroma).
  * RESTORE pipelines (all net x2, 128->512->256):
        bicubic            -- distortion floor
        compact            -- realesr-general (1.2M) reference
        x4plus             -- RealESRGAN_x4plus (16.7M) THE BASELINE TO BEAT
        scunet_x4plus      -- SCUNet deblock(1x) -> x4plus            [PRIMARY hypothesis]
        scunet_x4plus_b85  -- (scunet->x4plus) blended 0.85 toward compact [stack on R8-E3]
        bilat_x4plus       -- classical bilateral deblock -> x4plus   [zero-dep baseline]
        h264db_x4plus      -- hand-written 8px-grid weak-deblock -> x4plus [on-thesis classical]
  * ARBITER = full-reference LPIPS(AlexNet) + DISTS + PSNR vs GT (pyiqa, MPS).
    var-Lap = FAKE/over-sharpen flag ONLY (GOTCHA #23), NEVER the verdict.
    DISTS = the OVER-SMOOTHING guard: a deblock that wins LPIPS by blurring must lose DISTS.
  * FABRICATION/SMOOTHING panel = each preprocessor on the NATIVE crop (no codec) -> var-Lap + visual.

READ-ONLY on prototype/ + server/. All artifacts under this dir.
"""
import os, sys, io, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.abspath(os.path.join(HERE, "..", "..", "prototype"))
SAMPLE = os.path.abspath(os.path.join(HERE, "..", "..", "sample.mp4"))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, PROTO)
import sr  # read-only import of the project's anchor loader (prototype/sr.py)

DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

WINDOWS = {
    "talkinghead": 5000,   # smooth face
    "highmotion":  0,      # title card / low detail
    "texture18k":  18000,  # news headline
    "texture24k":  24000,  # chart + text
    "texture46k":  46000,  # textured photo
}
N_FRAMES = int(os.environ.get("N_FRAMES", "4"))
CROP = 256
CRF_LEVELS = {"moderate": 27, "heavy": 35}
BETA = 0.85   # R8-E3 validated global anchor blend (x4plus toward compact)

SCUNET_PATH = os.path.join(HERE, "models", "scunet_color_real_psnr.pth")

# ----------------------------------------------------------------------------- #
def var_lap(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def best_crop(rgb, c=CROP, stride=32):
    H, W = rgb.shape[:2]
    best, bxy = -1.0, (0, 0)
    for y in range(0, H - c + 1, stride):
        for x in range(0, W - c + 1, stride):
            v = var_lap(rgb[y:y + c, x:x + c])
            if v > best:
                best, bxy = v, (x, y)
    x, y = bxy
    return np.ascontiguousarray(rgb[y:y + c, x:x + c]), (x, y)

def center_crop(rgb, c=CROP):
    H, W = rgb.shape[:2]
    x = (W - c) // 2; y = max(0, (H - c) // 3)
    return np.ascontiguousarray(rgb[y:y + c, x:x + c]), (x, y)

def decode_frames(path, start, n):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out

def h264_degrade(gt_rgb, crf):
    """GT(256) -> 2x down (128) -> REAL libx264 encode@crf -> decode -> LR(128) RGB uint8."""
    import av
    H, W = gt_rgb.shape[:2]
    lr = cv2.resize(gt_rgb, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    lr = np.ascontiguousarray(lr)
    buf = io.BytesIO()
    cont = av.open(buf, mode="w", format="mp4")
    st = cont.add_stream("libx264", rate=25)
    st.width, st.height, st.pix_fmt = lr.shape[1], lr.shape[0], "yuv420p"
    st.options = {"crf": str(crf), "preset": "medium", "tune": "film"}
    fr = av.VideoFrame.from_ndarray(lr, format="rgb24")
    for p in st.encode(fr):
        cont.mux(p)
    for p in st.encode():
        cont.mux(p)
    cont.close()
    data = buf.getvalue()
    c2 = av.open(io.BytesIO(data))
    dec = None
    for f in c2.decode(video=0):
        dec = f.to_ndarray(format="rgb24"); break
    c2.close()
    return np.ascontiguousarray(dec), len(data)

# ----------------------------------------------------------------------------- #
# DEBLOCK PREPROCESSORS -- all take uint8 RGB (128px LR), return uint8 RGB (128px, cleaned)
# ----------------------------------------------------------------------------- #
_SCUNET = {}
def _scunet():
    if "net" not in _SCUNET:
        from spandrel import ModelLoader
        md = ModelLoader(device=DEV).load_from_file(SCUNET_PATH)
        _SCUNET["net"] = md.model.eval().to(DEV)
        print(f"[deblock] SCUNet loaded scale={md.scale} "
              f"params={sum(p.numel() for p in md.model.parameters())/1e6:.1f}M")
    return _SCUNET["net"]

@torch.no_grad()
def deblock_scunet(rgb):
    net = _scunet()
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(DEV).permute(2,0,1).unsqueeze(0).float().div_(255.0)
    out = net(t).clamp_(0,1).mul_(255.0).round_()
    out = out.squeeze(0).permute(1,2,0).to("cpu", torch.uint8).numpy()
    return np.ascontiguousarray(out)

def deblock_bilateral(rgb):
    """Classical edge-preserving deblock: bilateral filter smooths flat block-edges,
    preserves true luminance edges. Zero-dependency baseline."""
    return cv2.bilateralFilter(np.ascontiguousarray(rgb), d=5, sigmaColor=45, sigmaSpace=5)

def deblock_h264grid(rgb, qstep=8, thr=14.0):
    """Hand-written H.264-style weak deblock on the 8px DCT block grid (on-thesis classical,
    zero-dep). For each block boundary, low-pass the 2 boundary pixels ONLY where the cross-
    boundary jump is small enough to be a blocking artifact (not a real edge) -- preserves edges.
    Operates in float on each channel; mirrors the spirit of the H.264 in-loop deblock filter."""
    f = np.ascontiguousarray(rgb).astype(np.float32)
    H, W = f.shape[:2]
    out = f.copy()
    # vertical boundaries at columns qstep, 2*qstep, ...
    cols = list(range(qstep, W, qstep))
    if cols:
        c = np.array(cols)
        p1 = f[:, c-2, :]; p0 = f[:, c-1, :]; q0 = f[:, c, :]; q1 = f[:, np.minimum(c+1, W-1), :]
        jump = np.abs(p0 - q0).mean(axis=2, keepdims=True)        # per-boundary-pixel luma-ish jump
        m = (jump < thr).astype(np.float32)                       # 1 where likely a block artifact
        np0 = (p1 + 2*p0 + q0) / 4.0
        nq0 = (p0 + 2*q0 + q1) / 4.0
        out[:, c-1, :] = m*np0 + (1-m)*p0
        out[:, c,   :] = m*nq0 + (1-m)*q0
    # horizontal boundaries at rows qstep, 2*qstep, ... (operate on the v-deblocked buffer)
    f2 = out.copy()
    rows = list(range(qstep, H, qstep))
    if rows:
        r = np.array(rows)
        p1 = f2[r-2, :, :]; p0 = f2[r-1, :, :]; q0 = f2[r, :, :]; q1 = f2[np.minimum(r+1, H-1), :, :]
        jump = np.abs(p0 - q0).mean(axis=2, keepdims=True)
        m = (jump < thr).astype(np.float32)
        np0 = (p1 + 2*p0 + q0) / 4.0
        nq0 = (p0 + 2*q0 + q1) / 4.0
        out[r-1, :, :] = m*np0 + (1-m)*p0
        out[r,   :, :] = m*nq0 + (1-m)*q0
    return np.clip(out, 0, 255).round().astype(np.uint8)

DEBLOCKS = {
    "scunet":  deblock_scunet,
    "bilat":   deblock_bilateral,
    "h264db":  deblock_h264grid,
}

# ----------------------------------------------------------------------------- #
# x4 SR backends
# ----------------------------------------------------------------------------- #
def sr_bicubic(rgb):
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (w*4, h*4), interpolation=cv2.INTER_CUBIC)

def sr_x4plus(rgb):
    return sr.upscale(rgb, model="realesrgan-x4plus")

def sr_compact(rgb):
    return sr.upscale(rgb, model="realesrgan")

def to_t(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb)).to(DEV).permute(2,0,1).unsqueeze(0).float().div_(255.0)

def resize256(rgb512):
    return cv2.resize(rgb512, (CROP, CROP), interpolation=cv2.INTER_AREA)

def blend_lerp(a512, b512, beta):
    """R8-E3 anchor blend: out = beta*a + (1-beta)*b (a=x4plus toward b=compact)."""
    out = beta*a512.astype(np.float32) + (1.0-beta)*b512.astype(np.float32)
    return np.clip(out, 0, 255).round().astype(np.uint8)

# ----------------------------------------------------------------------------- #
def main():
    import pyiqa
    lpips = pyiqa.create_metric('lpips', device=DEV)
    dists = pyiqa.create_metric('dists', device=DEV)
    psnr  = pyiqa.create_metric('psnr',  device=DEV)
    print(f"[run] device={DEV} n_frames={N_FRAMES} crf={CRF_LEVELS} beta={BETA}")

    # warmup nets so first-call MPS compile doesn't pollute latency
    warm = (np.random.rand(128,128,3)*255).astype(np.uint8)
    sr.load_model("realesrgan-x4plus"); sr.load_model("realesrgan")
    sr_x4plus(warm); sr_compact(warm); deblock_scunet(warm)
    sr.reset_latency("realesrgan-x4plus"); sr.reset_latency("realesrgan")
    lat = {"scunet_ms": [], "x4plus_ms": []}

    records = []
    fab_done = set()
    for win, start in WINDOWS.items():
        frames = decode_frames(SAMPLE, start, N_FRAMES)
        if not frames:
            print(f"[run] {win}: NO FRAMES @ {start}"); continue
        smooth = win in ("talkinghead", "highmotion")
        for fi, frame in enumerate(frames):
            gt, xy = (center_crop(frame) if smooth else best_crop(frame))
            gt_t = to_t(gt)

            # ---- fabrication/over-smoothing panel on NATIVE crop (no codec) ----
            if win not in fab_done:
                fab_done.add(win)
                nat_lr = cv2.resize(gt, (CROP//2, CROP//2), interpolation=cv2.INTER_AREA)
                cols = [("GT", gt),
                        ("nativeLR(nn)", cv2.resize(nat_lr,(CROP,CROP),interpolation=cv2.INTER_NEAREST))]
                fabvl = {"x4plus(raw)": var_lap(resize256(sr_x4plus(nat_lr)))}
                cols.append((f"x4plus vL={fabvl['x4plus(raw)']:.0f}",
                             resize256(sr_x4plus(nat_lr))))
                for dn, dfn in DEBLOCKS.items():
                    db = dfn(nat_lr)
                    o = resize256(sr_x4plus(db))
                    fabvl[f"{dn}->x4plus"] = var_lap(o)
                    cols.append((f"{dn} vL={fabvl[f'{dn}->x4plus']:.0f}", o))
                panel = np.hstack([_label(cv2.cvtColor(im,cv2.COLOR_RGB2BGR), t) for t,im in cols])
                cv2.imwrite(os.path.join(OUT, f"fab_{win}.png"), panel)
                # 512 full-res for pixel-peep: x4plus vs scunet->x4plus
                cv2.imwrite(os.path.join(OUT, f"peep_{win}_x4plus_512.png"),
                            cv2.cvtColor(sr_x4plus(nat_lr), cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(OUT, f"peep_{win}_scunet_x4plus_512.png"),
                            cv2.cvtColor(sr_x4plus(deblock_scunet(nat_lr)), cv2.COLOR_RGB2BGR))
                print(f"[fab] {win}: " + " ".join(f"{k}={v:.0f}" for k,v in fabvl.items()))

            # ---- full-reference A/B across CRF ----
            for clab, crf in CRF_LEVELS.items():
                lr, nbytes = h264_degrade(gt, crf)
                preds = {}

                preds["bicubic"] = resize256(sr_bicubic(lr))
                preds["compact"] = resize256(sr_compact(lr))

                torch.mps.synchronize(); t0 = time.perf_counter()
                x4_raw512 = sr_x4plus(lr)
                torch.mps.synchronize(); lat["x4plus_ms"].append((time.perf_counter()-t0)*1000)
                preds["x4plus"] = resize256(x4_raw512)

                # SCUNet deblock -> x4plus (+ blend stack)
                torch.mps.synchronize(); t0 = time.perf_counter()
                lr_scu = deblock_scunet(lr)
                torch.mps.synchronize(); lat["scunet_ms"].append((time.perf_counter()-t0)*1000)
                scu512 = sr_x4plus(lr_scu)
                preds["scunet_x4plus"] = resize256(scu512)
                comp512 = sr_compact(lr)
                preds["scunet_x4plus_b85"] = resize256(blend_lerp(scu512, comp512, BETA))

                # classical deblocks -> x4plus
                preds["bilat_x4plus"]  = resize256(sr_x4plus(deblock_bilateral(lr)))
                preds["h264db_x4plus"] = resize256(sr_x4plus(deblock_h264grid(lr)))

                for name, pred in preds.items():
                    pt = to_t(pred)
                    records.append(dict(
                        window=win, frame=fi, crf=clab, model=name,
                        lpips=float(lpips(pt, gt_t).item()),
                        dists=float(dists(pt, gt_t).item()),
                        psnr=float(psnr(pt, gt_t).item()),
                        varlap=var_lap(pred), gt_varlap=var_lap(gt),
                        lr_bytes=nbytes))
                    del pt
                torch.mps.empty_cache()
            del gt_t
        print(f"[run] {win} done ({len(frames)} frames)")
        torch.mps.empty_cache()

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(records, f, indent=2)
    lat_summary = {k: (float(np.median(v)) if v else float('nan')) for k,v in lat.items()}
    with open(os.path.join(HERE, "latency.json"), "w") as f:
        json.dump(lat_summary, f, indent=2)
    print(f"[run] wrote {len(records)} records -> results.json")
    print(f"[run] latency median ms: {lat_summary}")

def _label(img_bgr, text):
    out = img_bgr.copy()
    cv2.rectangle(out, (0,0), (out.shape[1], 22), (0,0,0), -1)
    cv2.putText(out, text, (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
    return out

if __name__ == "__main__":
    main()
