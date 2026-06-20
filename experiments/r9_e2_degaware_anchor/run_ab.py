#!/usr/bin/env python3
"""
R9-E2: Can a degradation-aware anchor BEAT x4plus on REAL H.264 SD?

Protocol (synthesis of the two prior methodologies in handoff.md):
  * GROUND TRUTH = decoded sample.mp4 frame crop as pseudo-HD GT  (R6-E1 convention:
    "the original decoded frame IS the pseudo-HD GT").
  * LR = 2x downscale (INTER_AREA) of GT, then a REAL libx264 (PyAV) encode at CRF c,
    then decode -> genuine H.264 codec artifacts (blocking/ringing/4:2:0 chroma).
    This is the REAL codec degradation regime (NOT R6-E1's SYNTHETIC blur+JPEG+noise,
    which the task forbids). Two CRF levels span moderate->heavy compression.
  * RESTORE = model x4 (128->512) then INTER_AREA resize 512->256 to GT scale
    (matches R6-E1 sr.upscale_to: x4 then resize-to-target). Net 2x SR, same for all.
  * ARBITER = full-reference LPIPS (AlexNet) + DISTS + PSNR vs GT. var-Lap = FAKE-detail
    flag ONLY (GOTCHA #23), NEVER the quality verdict.
  * FABRICATION CHECK = each model on the NATIVE real crop (the true deployment input,
    no GT) x4 -> 512, saved for visual inspection + var-Lap. The diffusion-trap detector.

Models A/B:
  bicubic            -- distortion floor reference
  compact            -- realesr-general-x4v3 (SRVGGNetCompact 1.21M) [prototype/sr.py]
  x4plus             -- RealESRGAN_x4plus (RRDBNet 16.7M) THE CEILING/BASELINE [sr.py]
  ultrasharp         -- 4x-UltraSharp (ESRGAN RRDB 16.7M, community real-world) [spandrel]
  nomos              -- 4xNomos8kSC (real-world degradation-trained) [spandrel] (if loads)
  wdn-dni-0.5/0.0    -- realesr-general DNI-blended w/ wdn (denoise-aware compact)

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

# R6-E1 validated windows (var-Lap scan of all 50,805 frames)
WINDOWS = {
    "talkinghead": 5000,   # smooth face (where denoise/compact historically helps)
    "highmotion":  0,      # title card / low detail
    "texture18k":  18000,  # news headline
    "texture24k":  24000,  # chart + text
    "texture46k":  46000,  # textured photo
}
N_FRAMES = int(os.environ.get("N_FRAMES", "4"))
CROP = 256                 # GT crop size (pseudo-HD)
CRF_LEVELS = {"moderate": 27, "heavy": 35}

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
# SR backends -- all take uint8 RGB HxWx3, return x4 uint8 RGB
# ----------------------------------------------------------------------------- #
_SPANDREL = {}
def _spandrel_model(path):
    if path not in _SPANDREL:
        from spandrel import ModelLoader
        md = ModelLoader(device=DEV).load_from_file(path)
        md.model.eval().to(DEV)
        _SPANDREL[path] = md.model
    return _SPANDREL[path]

@torch.no_grad()
def sr_spandrel(rgb, path):
    net = _spandrel_model(path)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(DEV).permute(2,0,1).unsqueeze(0).float().div_(255.0)
    out = net(t).clamp_(0,1).mul_(255.0).round_()
    out = out.squeeze(0).permute(1,2,0).to("cpu", torch.uint8).numpy()
    return np.ascontiguousarray(out)

_DNI = {}
def _dni_compact(w_general):
    """Blend realesr-general-x4v3 (general) with realesr-general-wdn-x4v3 (wdn):
       blend = w_general*general + (1-w_general)*wdn. w=1 -> pure general (==compact),
       w=0 -> pure wdn (max denoise). Loaded into sr.py's SRVGGNetCompact."""
    key = round(w_general, 3)
    if key in _DNI:
        return _DNI[key]
    gen_p = os.path.join(PROTO, "models", "realesr-general-x4v3.pth")
    wdn_p = os.path.join(HERE, "models", "realesr-general-wdn-x4v3.pth")
    gen = torch.load(gen_p, map_location="cpu"); gen = gen.get("params", gen)
    wdn = torch.load(wdn_p, map_location="cpu"); wdn = wdn.get("params", wdn)
    blend = {k: w_general * gen[k] + (1.0 - w_general) * wdn[k] for k in gen}
    net = sr._build_compact()
    net.load_state_dict(blend, strict=True)
    net.eval().to(DEV)
    _DNI[key] = net
    return net

@torch.no_grad()
def sr_dni(rgb, w_general):
    net = _dni_compact(w_general)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(DEV).permute(2,0,1).unsqueeze(0).float().div_(255.0)
    out = net(t).clamp_(0,1).mul_(255.0).round_()
    out = out.squeeze(0).permute(1,2,0).to("cpu", torch.uint8).numpy()
    return np.ascontiguousarray(out)

def sr_bicubic(rgb):
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (w*4, h*4), interpolation=cv2.INTER_CUBIC)

def build_backends():
    b = {}
    b["bicubic"]    = ("floor",      lambda x: sr_bicubic(x))
    b["compact"]    = ("compact1.2M",lambda x: sr.upscale(x, model="realesrgan"))
    b["x4plus"]     = ("CEILING16.7M",lambda x: sr.upscale(x, model="realesrgan-x4plus"))
    us = os.path.join(HERE, "models", "4x-UltraSharp.pth")
    if os.path.exists(us):
        b["ultrasharp"] = ("ESRGAN16.7M", lambda x: sr_spandrel(x, us))
    nm = os.path.join(HERE, "models", "4xNomos8kSC.pth")
    if os.path.exists(nm):
        try:
            _spandrel_model(nm)  # probe load
            b["nomos"] = ("Nomos8kSC", lambda x: sr_spandrel(x, nm))
        except Exception as e:
            print(f"[backend] nomos unavailable: {type(e).__name__}: {e}")
    b["wdn-dni0.5"] = ("denoise.5", lambda x: sr_dni(x, 0.5))
    b["wdn-dni0.0"] = ("wdnPure",   lambda x: sr_dni(x, 0.0))
    return b

# ----------------------------------------------------------------------------- #
def to_t(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb)).to(DEV).permute(2,0,1).unsqueeze(0).float().div_(255.0)

def main():
    import pyiqa
    lpips = pyiqa.create_metric('lpips', device=DEV)
    dists = pyiqa.create_metric('dists', device=DEV)
    psnr  = pyiqa.create_metric('psnr',  device=DEV)
    backends = build_backends()
    print(f"[run] device={DEV} backends={list(backends.keys())} windows={list(WINDOWS)} "
          f"n_frames={N_FRAMES} crf={CRF_LEVELS}")

    records = []          # one row per (window, frame_idx, crf, model)
    fab_done = set()      # native-crop fabrication saved per window
    for win, start in WINDOWS.items():
        frames = decode_frames(SAMPLE, start, N_FRAMES)
        if not frames:
            print(f"[run] {win}: NO FRAMES @ {start}"); continue
        smooth = win in ("talkinghead", "highmotion")
        for fi, frame in enumerate(frames):
            gt, xy = (center_crop(frame) if smooth else best_crop(frame))
            gt_t = to_t(gt)
            # --- native-crop fabrication panel (once per window, frame 0) ---
            if win not in fab_done:
                fab_done.add(win)
                nat_lr = cv2.resize(gt, (CROP//2, CROP//2), interpolation=cv2.INTER_AREA)  # real-ish SD scale crop
                cols = [("GT(256)", gt), ("nativeLR(128 nn)", cv2.resize(nat_lr,(CROP,CROP),interpolation=cv2.INTER_NEAREST))]
                fabvl = {}
                for name,(_,fn) in backends.items():
                    if name == "bicubic":
                        continue
                    out512 = fn(nat_lr)
                    fabvl[name] = var_lap(out512)
                    cols.append((f"{name} vL={fabvl[name]:.0f}", cv2.resize(out512,(CROP,CROP),interpolation=cv2.INTER_AREA)))
                panel = np.hstack([_label(cv2.cvtColor(im,cv2.COLOR_RGB2BGR), t) for t,im in cols])
                cv2.imwrite(os.path.join(OUT, f"fab_{win}.png"), panel)
                # full-res 512 crops for true pixel-peep (x4plus vs ultrasharp vs wdn)
                for name in ("x4plus","ultrasharp","wdn-dni0.0"):
                    if name in backends:
                        o = backends[name][1](nat_lr)
                        cv2.imwrite(os.path.join(OUT, f"fab_{win}_{name}_512.png"), cv2.cvtColor(o,cv2.COLOR_RGB2BGR))
                print(f"[fab] {win}: native var-Lap " + " ".join(f"{k}={v:.0f}" for k,v in fabvl.items()))

            # --- full-reference A/B across CRF ---
            for clab, crf in CRF_LEVELS.items():
                lr, nbytes = h264_degrade(gt, crf)
                for name,(tag,fn) in backends.items():
                    out512 = fn(lr)                                # 512x512 x4
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
        print(f"[run] {win} done ({len(frames)} frames)")
        torch.mps.empty_cache()

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(f"[run] wrote {len(records)} records -> results.json")

def _label(img_bgr, text):
    out = img_bgr.copy()
    cv2.rectangle(out, (0,0), (out.shape[1], 22), (0,0,0), -1)
    cv2.putText(out, text, (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return out

if __name__ == "__main__":
    main()
