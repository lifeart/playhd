#!/usr/bin/env python3
"""
R11: Does 2xLiveActionV1_SPAN (the web_spike anchor) actually BEAT the compact on REAL H.264 SD?

The web_spike claim "SPAN beats compact on LPIPS+DISTS" came from the model card, NOT a measurement
on real codec output. R8-R10 repeatedly showed fancy models OVER-SMOOTH real H.264 (the fake-detail
trap, DISTS-caught). R10-E1 tested the Span ARCH family (4xNomos8kSC) but NOT this exact 2x model.
This settles it on real footage with the validated R10-E1 harness.

  GT      = sample.mp4 256 crop (pseudo-HD).
  LR      = 2x INTER_AREA down (->128) -> REAL libx264 (PyAV) CRF {27,35} -> decode (true h264 artifacts).
  RESTORE = net (SPAN native 2x: 128->256 ; 4x models: 128->512 -> INTER_AREA 256). Net 2x for all.
  ARBITER = full-ref LPIPS(Alex) + DISTS + PSNR vs GT (pyiqa, MPS). var-Lap = fake-detail flag only.
  FAB     = each model on the CLEAN native LR (no codec) -> fabrication/over-smooth pixel-peep.

Candidates: bicubic (floor) | compact realesr-general-x4v3 (what the player used BEFORE SPAN) |
            x4plus RRDBNet (the heavy ceiling/reference) | span 2xLiveActionV1 (what the player uses NOW).
READ-ONLY on prototype/. Artifacts under this dir.
"""
import os, sys, io, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.abspath(os.path.join(HERE, "..", "..", "prototype"))
SAMPLE = os.path.abspath(os.path.join(HERE, "..", "..", "sample.mp4"))
SPAN_PTH = os.path.join(PROTO, "models", "2xLiveActionV1_SPAN.pth")
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, PROTO)
import sr  # project's anchor loader (compact / x4plus)

DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
WINDOWS = {"talkinghead": 5000, "highmotion": 0, "texture24k": 24000}
N_FRAMES = int(os.environ.get("N_FRAMES", "3"))
CROP = 256
CRF_LEVELS = {"moderate": 27, "heavy": 35}

def var_lap(rgb):
    return float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

def best_crop(rgb, c=CROP, stride=32):
    H, W = rgb.shape[:2]; best, bxy = -1.0, (0, 0)
    for y in range(0, H - c + 1, stride):
        for x in range(0, W - c + 1, stride):
            v = var_lap(rgb[y:y+c, x:x+c])
            if v > best: best, bxy = v, (x, y)
    x, y = bxy; return np.ascontiguousarray(rgb[y:y+c, x:x+c]), (x, y)

def center_crop(rgb, c=CROP):
    H, W = rgb.shape[:2]; x = (W - c)//2; y = max(0, (H - c)//3)
    return np.ascontiguousarray(rgb[y:y+c, x:x+c]), (x, y)

def decode_frames(path, start, n):
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES, start); out = []
    for _ in range(n):
        ok, bgr = cap.read()
        if not ok: break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release(); return out

def h264_degrade(gt_rgb, crf):
    import av
    H, W = gt_rgb.shape[:2]
    lr = np.ascontiguousarray(cv2.resize(gt_rgb, (W//2, H//2), interpolation=cv2.INTER_AREA))
    buf = io.BytesIO(); cont = av.open(buf, mode="w", format="mp4")
    st = cont.add_stream("libx264", rate=25)
    st.width, st.height, st.pix_fmt = lr.shape[1], lr.shape[0], "yuv420p"
    st.options = {"crf": str(crf), "preset": "medium", "tune": "film"}
    for p in st.encode(av.VideoFrame.from_ndarray(lr, format="rgb24")): cont.mux(p)
    for p in st.encode(): cont.mux(p)
    cont.close(); data = buf.getvalue()
    c2 = av.open(io.BytesIO(data)); dec = None
    for f in c2.decode(video=0): dec = f.to_ndarray(format="rgb24"); break
    c2.close(); return np.ascontiguousarray(dec), len(data)

# --- backends ---
_LAT = {}
_SPAN = {}
def _span_net():
    if "net" not in _SPAN:
        from spandrel import ModelLoader
        md = ModelLoader(device=DEV).load_from_file(SPAN_PTH); md.model.eval().to(DEV)
        _SPAN["net"] = md.model
    return _SPAN["net"]

@torch.no_grad()
def sr_span(rgb):
    net = _span_net()
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(DEV).permute(2,0,1).unsqueeze(0).float().div_(255.0)
    if DEV.type == "mps": torch.mps.synchronize()
    t0 = time.time()
    out = net(t).clamp_(0,1).mul_(255.0).round_()
    if DEV.type == "mps": torch.mps.synchronize()
    _LAT.setdefault("span", []).append((time.time()-t0)*1000.0)
    return np.ascontiguousarray(out.squeeze(0).permute(1,2,0).to("cpu", torch.uint8).numpy())

def sr_bicubic(rgb):
    h, w = rgb.shape[:2]; return cv2.resize(rgb, (w*4, h*4), interpolation=cv2.INTER_CUBIC)

@torch.no_grad()
def sr_proto(rgb, model, name):
    if DEV.type == "mps": torch.mps.synchronize()
    t0 = time.time(); out = sr.upscale(rgb, model=model)
    if DEV.type == "mps": torch.mps.synchronize()
    _LAT.setdefault(name, []).append((time.time()-t0)*1000.0); return out

BACKENDS = {
    "bicubic": lambda x: sr_bicubic(x),
    "compact": lambda x: sr_proto(x, "realesrgan", "compact"),
    "x4plus":  lambda x: sr_proto(x, "realesrgan-x4plus", "x4plus"),
    "span":    lambda x: sr_span(x),
}

def to_t(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb)).to(DEV).permute(2,0,1).unsqueeze(0).float().div_(255.0)

def _label(img_bgr, text):
    out = img_bgr.copy(); cv2.rectangle(out, (0,0), (out.shape[1], 22), (0,0,0), -1)
    cv2.putText(out, text, (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA); return out

def main():
    import pyiqa
    lpips = pyiqa.create_metric('lpips', device=DEV)
    dists = pyiqa.create_metric('dists', device=DEV)
    psnr  = pyiqa.create_metric('psnr',  device=DEV)
    print(f"[run] device={DEV} backends={list(BACKENDS)} windows={list(WINDOWS)} n={N_FRAMES} crf={CRF_LEVELS}")
    records = []; fab_done = set()
    for win, start in WINDOWS.items():
        frames = decode_frames(SAMPLE, start, N_FRAMES)
        if not frames: print(f"[run] {win}: NO FRAMES @ {start}"); continue
        smooth = win in ("talkinghead", "highmotion")
        for fi, frame in enumerate(frames):
            gt, _ = (center_crop(frame) if smooth else best_crop(frame)); gt_t = to_t(gt)
            if win not in fab_done:  # fabrication check on CLEAN native LR
                fab_done.add(win)
                nat = cv2.resize(gt, (CROP//2, CROP//2), interpolation=cv2.INTER_AREA)
                cols = [("GT(256)", gt), ("nativeLR(128 nn)", cv2.resize(nat,(CROP,CROP),interpolation=cv2.INTER_NEAREST))]
                fv = {}
                for name, fn in BACKENDS.items():
                    if name == "bicubic": continue
                    o = fn(nat); fv[name] = var_lap(o)
                    cols.append((f"{name} vL={fv[name]:.0f}", cv2.resize(o,(CROP,CROP),interpolation=cv2.INTER_AREA)))
                panel = np.hstack([_label(cv2.cvtColor(im,cv2.COLOR_RGB2BGR), t) for t,im in cols])
                cv2.imwrite(os.path.join(OUT, f"fab_{win}.png"), panel)
                print(f"[fab] {win}: GT vL={var_lap(gt):.0f} | native " + " ".join(f"{k}={v:.0f}" for k,v in fv.items()))
                if DEV.type=="mps": torch.mps.empty_cache()
            for clab, crf in CRF_LEVELS.items():
                lr, nbytes = h264_degrade(gt, crf)
                for name, fn in BACKENDS.items():
                    out = fn(lr); pred = cv2.resize(out, (CROP, CROP), interpolation=cv2.INTER_AREA); pt = to_t(pred)
                    records.append(dict(window=win, frame=fi, crf=clab, model=name,
                        lpips=float(lpips(pt, gt_t).item()), dists=float(dists(pt, gt_t).item()),
                        psnr=float(psnr(pt, gt_t).item()), varlap=var_lap(pred), gt_varlap=var_lap(gt), lr_bytes=nbytes))
                    del pt
                    if DEV.type=="mps": torch.mps.empty_cache()
            del gt_t
            if DEV.type=="mps": torch.mps.empty_cache()
        print(f"[run] {win} done ({len(frames)} frames)")
    lat = {k: dict(median_ms=float(np.median(v)), n=len(v)) for k,v in _LAT.items()}
    json.dump(dict(records=records, latency=lat), open(os.path.join(HERE,"results.json"),"w"), indent=2)
    print(f"[run] wrote {len(records)} records. latency(128->out): " +
          " ".join(f"{k}={v['median_ms']:.0f}ms" for k,v in lat.items()))

if __name__ == "__main__":
    main()
