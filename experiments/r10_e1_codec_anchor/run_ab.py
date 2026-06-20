#!/usr/bin/env python3
"""
R10-E1: Can a CODEC/compression-trained anchor BEAT x4plus on REAL H.264 SD?

Direct reuse of R9-E2's validated harness (experiments/r9_e2_degaware_anchor/run_ab.py):
  * GT  = decoded sample.mp4 256x256 crop (R6-E1 var-Lap windows) = pseudo-HD ground truth.
  * LR  = 2x INTER_AREA down (->128) -> REAL libx264 (PyAV) encode @ CRF {27,35} -> decode
          -> genuine H.264 artifacts (blocking/ringing/4:2:0 chroma). The regime x4plus was
          NOT trained on and a codec-trained model SHOULD win, if the thesis holds.
  * RESTORE = model x4 (128->512) -> INTER_AREA 512->256. Net 2x SR, identical for all models.
  * ARBITER = full-reference LPIPS(AlexNet) + DISTS + PSNR vs GT (pyiqa, MPS).
              var-Lap = FAKE-detail flag ONLY (GOTCHA #23), NEVER the verdict.
  * FABRICATION CHECK = each model on the clean NATIVE crop (no codec) x4 -> 512, saved for
              visual pixel-peep + var-Lap. The diffusion/UltraSharp trap detector.

Candidates (all codec/compression-trained, architecturally NOVEL vs R9-E2's RRDB family;
loaded by spandrel auto-arch from HF safetensors):
  bicubic           -- distortion floor
  compact 1.2M      -- realesr-general-x4v3 (reference)
  x4plus 16.7M      -- RealESRGAN_x4plus (RRDBNet) THE CEILING/BASELINE
  realwebphoto_dat2 -- 4xRealWebPhoto_v4_dat2 (DAT2 transformer; degradation incl. VIDEO h264/h265 compression)
  nomos_atd_jpg     -- 4xNomos8k_atd_jpg (ATD transformer; explicit JPEG-compression degradation)
  nomos_hatl_otf    -- 4xNomos8kHAT-L_otf (HAT-L transformer; on-the-fly real degradation)
  nomos_sc          -- 4xNomos8kSC (Span; the R9-E2 download blocker, now via HF)

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
import sr  # read-only import of project's anchor loader

DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

WINDOWS = {
    "talkinghead": 5000,
    "highmotion":  0,
    "texture18k":  18000,
    "texture24k":  24000,
    "texture46k":  46000,
}
N_FRAMES = int(os.environ.get("N_FRAMES", "4"))
CROP = 256
CRF_LEVELS = {"moderate": 27, "heavy": 35}

MODELS_DIR = os.path.join(HERE, "models")
SPANDREL_CANDS = {
    "realwebphoto_dat2": "4xRealWebPhoto_v4_dat2.safetensors",
    "nomos_atd_jpg":     "4xNomos8k_atd_jpg.safetensors",
    "nomos_hatl_otf":    "4xNomos8kHAT-L_otf.safetensors",
    "nomos_sc":          "4xNomos8kSC.safetensors",
}

# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
_SPANDREL = {}
def _spandrel_model(path):
    if path not in _SPANDREL:
        from spandrel import ModelLoader
        md = ModelLoader(device=DEV).load_from_file(path)
        md.model.eval().to(DEV)
        _SPANDREL[path] = (md.model, type(md).__name__, md.architecture.name if hasattr(md,"architecture") else "?")
    return _SPANDREL[path]

_LAT = {}  # model -> list of per-call ms (anchor latency affordability check)
@torch.no_grad()
def sr_spandrel(rgb, path, name):
    net, _, _ = _spandrel_model(path)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(DEV).permute(2,0,1).unsqueeze(0).float().div_(255.0)
    if DEV.type == "mps": torch.mps.synchronize()
    t0 = time.time()
    out = net(t).clamp_(0,1).mul_(255.0).round_()
    if DEV.type == "mps": torch.mps.synchronize()
    _LAT.setdefault(name, []).append((time.time()-t0)*1000.0)
    out = out.squeeze(0).permute(1,2,0).to("cpu", torch.uint8).numpy()
    return np.ascontiguousarray(out)

def sr_bicubic(rgb):
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (w*4, h*4), interpolation=cv2.INTER_CUBIC)

@torch.no_grad()
def sr_proto(rgb, model, name):
    if DEV.type == "mps": torch.mps.synchronize()
    t0 = time.time()
    out = sr.upscale(rgb, model=model)
    if DEV.type == "mps": torch.mps.synchronize()
    _LAT.setdefault(name, []).append((time.time()-t0)*1000.0)
    return out

def build_backends():
    b = {}
    b["bicubic"] = ("floor", lambda x: sr_bicubic(x))
    b["compact"] = ("compact1.2M", lambda x: sr_proto(x, "realesrgan", "compact"))
    b["x4plus"]  = ("CEILING16.7M", lambda x: sr_proto(x, "realesrgan-x4plus", "x4plus"))
    archinfo = {}
    for name, fn in SPANDREL_CANDS.items():
        p = os.path.join(MODELS_DIR, fn)
        if not os.path.exists(p) or os.path.getsize(p) < 1e6:
            print(f"[backend] {name} MISSING ({fn}) -> skip"); continue
        try:
            _, cls, arch = _spandrel_model(p)
            archinfo[name] = arch
            b[name] = (arch, (lambda pp, nn: (lambda x: sr_spandrel(x, pp, nn)))(p, name))
            print(f"[backend] {name}: spandrel arch={arch} loaded")
        except Exception as e:
            print(f"[backend] {name} LOAD FAILED: {type(e).__name__}: {str(e)[:140]}")
    return b, archinfo

# --------------------------------------------------------------------------- #
def to_t(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb)).to(DEV).permute(2,0,1).unsqueeze(0).float().div_(255.0)

def _label(img_bgr, text):
    out = img_bgr.copy()
    cv2.rectangle(out, (0,0), (out.shape[1], 22), (0,0,0), -1)
    cv2.putText(out, text, (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
    return out

def main():
    import pyiqa
    lpips = pyiqa.create_metric('lpips', device=DEV)
    dists = pyiqa.create_metric('dists', device=DEV)
    psnr  = pyiqa.create_metric('psnr',  device=DEV)
    backends, archinfo = build_backends()
    print(f"[run] device={DEV} backends={list(backends.keys())} windows={list(WINDOWS)} "
          f"n_frames={N_FRAMES} crf={CRF_LEVELS}")

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
            if win not in fab_done:
                fab_done.add(win)
                nat_lr = cv2.resize(gt, (CROP//2, CROP//2), interpolation=cv2.INTER_AREA)
                cols = [("GT(256)", gt), ("nativeLR(128 nn)", cv2.resize(nat_lr,(CROP,CROP),interpolation=cv2.INTER_NEAREST))]
                fabvl = {}
                for name,(_,fn) in backends.items():
                    if name == "bicubic": continue
                    out512 = fn(nat_lr)
                    fabvl[name] = var_lap(out512)
                    cols.append((f"{name} vL={fabvl[name]:.0f}", cv2.resize(out512,(CROP,CROP),interpolation=cv2.INTER_AREA)))
                panel = np.hstack([_label(cv2.cvtColor(im,cv2.COLOR_RGB2BGR), t) for t,im in cols])
                cv2.imwrite(os.path.join(OUT, f"fab_{win}.png"), panel)
                for name in ["x4plus"] + list(SPANDREL_CANDS.keys()):
                    if name in backends:
                        o = backends[name][1](nat_lr)
                        cv2.imwrite(os.path.join(OUT, f"fab_{win}_{name}_512.png"), cv2.cvtColor(o,cv2.COLOR_RGB2BGR))
                print(f"[fab] {win}: native var-Lap " + " ".join(f"{k}={v:.0f}" for k,v in fabvl.items()))
                torch.mps.empty_cache()

            for clab, crf in CRF_LEVELS.items():
                lr, nbytes = h264_degrade(gt, crf)
                for name,(tag,fn) in backends.items():
                    out512 = fn(lr)
                    pred = cv2.resize(out512, (CROP, CROP), interpolation=cv2.INTER_AREA)
                    pt = to_t(pred)
                    rec = dict(window=win, frame=fi, crf=clab, model=name,
                               lpips=float(lpips(pt, gt_t).item()),
                               dists=float(dists(pt, gt_t).item()),
                               psnr=float(psnr(pt, gt_t).item()),
                               varlap=var_lap(pred), gt_varlap=var_lap(gt),
                               lr_bytes=nbytes)
                    records.append(rec)
                    del pt
                    torch.mps.empty_cache()
            del gt_t
            torch.mps.empty_cache()
        print(f"[run] {win} done ({len(frames)} frames)")
        torch.mps.empty_cache()

    lat = {k: dict(mean_ms=float(np.mean(v)), median_ms=float(np.median(v)), n=len(v))
           for k,v in _LAT.items()}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(dict(records=records, latency=lat, arch=archinfo), f, indent=2)
    print(f"[run] wrote {len(records)} records -> results.json")
    print(f"[lat] per-anchor latency (128->512 x4): " +
          " ".join(f"{k}={v['median_ms']:.0f}ms" for k,v in lat.items()))

if __name__ == "__main__":
    main()
