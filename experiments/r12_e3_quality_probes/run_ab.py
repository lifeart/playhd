#!/usr/bin/env python3
"""
R12-E3 (research item #7): Two cheap quality probes.

(A) A/B a CODEC-TRAINED SISR model against the current x4plus anchor on REAL H.264.
    Core hypothesis: "an SR model that has actually seen H.264 artifacts beats the
    JPEG/OTF-trained x4plus on real compressed video." R8-R10 repeatedly found fancy
    models OVER-SMOOTH real H.264 (fake-detail trap, DISTS-caught). This measures it for
    the actual video-compression-trained models.

(B) VMAF-NEG guardrail column (anti-hallucination) alongside LPIPS+DISTS+PSNR. NEG clips the
    VMAF gain terms that reward artificial sharpening, so a "win" on LPIPS/DISTS that is really
    hallucination shows up as a LOW VMAF-NEG. Guardrail only -- never an optimisation target.

Harness = the VALIDATED R11/R10-E1 protocol (identical GT/degrade/arbiter), extended with the
codec-trained challengers and the VMAF-NEG column:
  GT      = sample.mp4 256 crop (pseudo-HD).  (face=talkinghead, texture=texture*, motion=highmotion)
  LR      = 2x INTER_AREA down (->128) -> REAL libx264 (PyAV) CRF {27,35} -> decode (true h264 artifacts).
  RESTORE = net (2x nets: 128->256 ; 4x nets: 128->512 -> INTER_AREA 256). NET 2x for ALL -> fair scale.
  ARBITER = full-ref LPIPS(Alex)+DISTS+PSNR (pyiqa,MPS) + VMAF-NEG (ffmpeg libvmaf, per window/crf).
  FAB     = each model on CLEAN native LR (no codec) -> var-Lap fake-detail pixel-peep panel.

Candidates:
  bicubic          -- distortion floor
  compact          -- realesr-general-x4v3 (SRVGGNetCompact, 1.2M) prior player anchor, 4x
  x4plus           -- RealESRGAN_x4plus (RRDBNet, 16.7M) THE CEILING / thing-to-beat, 4x
  span             -- 2xLiveActionV1_SPAN (2.2M) current player anchor, face specialist, 2x
  vcisr            -- 4x_VCISR_generator (GRL, 3.5M) OFFICIAL WACV'24, trained w/ real libx264
                      video-compression synthetic data. THE codec-trained challenger, 4x.
  avc_compact      -- 2xHFA2kAVCCompact (SRVGGNetCompact, 0.6M) trained on AVC/H.264 compression.
                      A second, cheap codec-trained data point, 2x.

READ-ONLY on prototype/. All artifacts under this dir.
"""
import os, sys, io, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.abspath(os.path.join(HERE, "..", "..", "prototype"))
SAMPLE = os.path.abspath(os.path.join(HERE, "..", "..", "sample.mp4"))
PROTO_MODELS = os.path.join(PROTO, "models")
LOCAL_MODELS = os.path.join(HERE, "models")
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, PROTO)
import sr                      # project's anchor loader (compact / x4plus)
import vmaf_neg as vneg        # our reusable NEG guardrail helper (this dir)

DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
WINDOWS = {"talkinghead": 5000, "highmotion": 0, "texture18k": 18000, "texture24k": 24000}
FACECLASS = {"talkinghead": "face", "highmotion": "motion",
             "texture18k": "texture", "texture24k": "texture"}
N_FRAMES = int(os.environ.get("N_FRAMES", "3"))
CROP = 256
CRF_LEVELS = {"moderate": 27, "heavy": 35}

SPAN_PTH  = os.path.join(PROTO_MODELS, "2xLiveActionV1_SPAN.pth")
AVC_PTH   = os.path.join(PROTO_MODELS, "2xHFA2kAVCCompact.safetensors")
VCISR_PTH = os.path.join(LOCAL_MODELS, "4x_VCISR_generator.pth")


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
_SPANDREL = {}
def _spandrel_net(path):
    if path not in _SPANDREL:
        from spandrel import ModelLoader
        md = ModelLoader(device=DEV).load_from_file(path); md.model.eval().to(DEV)
        arch = md.architecture.name if hasattr(md, "architecture") else "?"
        _SPANDREL[path] = (md.model, arch, int(getattr(md, "scale", 0) or 0))
    return _SPANDREL[path]

@torch.no_grad()
def sr_spandrel(rgb, path, name):
    net, _, _ = _spandrel_net(path)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(DEV).permute(2,0,1).unsqueeze(0).float().div_(255.0)
    if DEV.type == "mps": torch.mps.synchronize()
    t0 = time.time()
    out = net(t).clamp_(0,1).mul_(255.0).round_()
    if DEV.type == "mps": torch.mps.synchronize()
    _LAT.setdefault(name, []).append((time.time()-t0)*1000.0)
    return np.ascontiguousarray(out.squeeze(0).permute(1,2,0).to("cpu", torch.uint8).numpy())

def sr_bicubic(rgb):
    h, w = rgb.shape[:2]; return cv2.resize(rgb, (w*4, h*4), interpolation=cv2.INTER_CUBIC)

@torch.no_grad()
def sr_proto(rgb, model, name):
    if DEV.type == "mps": torch.mps.synchronize()
    t0 = time.time(); out = sr.upscale(rgb, model=model)
    if DEV.type == "mps": torch.mps.synchronize()
    _LAT.setdefault(name, []).append((time.time()-t0)*1000.0); return out

def build_backends():
    """name -> (upscale_fn, native_scale). native_scale used to normalise to net-2x (final 256)."""
    b = {}
    b["bicubic"]     = (lambda x: sr_bicubic(x), 4)
    b["compact"]     = (lambda x: sr_proto(x, "realesrgan", "compact"), 4)
    b["x4plus"]      = (lambda x: sr_proto(x, "realesrgan-x4plus", "x4plus"), 4)
    meta = {"compact": ("SRVGGNetCompact", 4), "x4plus": ("RRDBNet", 4)}
    for name, path in [("span", SPAN_PTH), ("vcisr", VCISR_PTH), ("avc_compact", AVC_PTH)]:
        if not os.path.exists(path):
            print(f"[backend] {name} MISSING ({path}) -> skip"); continue
        try:
            _, arch, scale = _spandrel_net(path)
            scale = scale or 2
            b[name] = ((lambda pp, nn: (lambda x: sr_spandrel(x, pp, nn)))(path, name), scale)
            meta[name] = (arch, scale)
            print(f"[backend] {name}: spandrel arch={arch} scale={scale} loaded")
        except Exception as e:
            print(f"[backend] {name} LOAD FAILED: {type(e).__name__}: {str(e)[:160]}")
    return b, meta

def to_256(out, native_scale):
    """Net-2x normaliser: 128->(native_scale*128) restore, resize down to 256 if native scale>2."""
    if out.shape[0] == CROP:            # already 256 (native 2x)
        return out
    return cv2.resize(out, (CROP, CROP), interpolation=cv2.INTER_AREA)

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
    backends, meta = build_backends()
    neg_ok = vneg.available()
    print(f"[run] device={DEV} backends={list(backends)} windows={list(WINDOWS)} "
          f"n={N_FRAMES} crf={CRF_LEVELS} vmaf_neg={'ON' if neg_ok else 'UNAVAILABLE'}")

    records = []
    # per (window,crf,model) accumulate frame sequences for a temporal VMAF-NEG
    seqs = {}   # key -> {"gt":[...256 rgb...], "pred":[...256 rgb...]}
    fab_done = set()

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
                for name, (fn, ns) in backends.items():
                    if name == "bicubic": continue
                    o = to_256(fn(nat), ns); fv[name] = var_lap(o)
                    cols.append((f"{name} vL={fv[name]:.0f}", o))
                panel = np.hstack([_label(cv2.cvtColor(im,cv2.COLOR_RGB2BGR), t) for t,im in cols])
                cv2.imwrite(os.path.join(OUT, f"fab_{win}.png"), panel)
                print(f"[fab] {win}: GT vL={var_lap(gt):.0f} | native " + " ".join(f"{k}={v:.0f}" for k,v in fv.items()))
                if DEV.type == "mps": torch.mps.empty_cache()
            for clab, crf in CRF_LEVELS.items():
                lr, nbytes = h264_degrade(gt, crf)
                for name, (fn, ns) in backends.items():
                    pred = to_256(fn(lr), ns); pt = to_t(pred)
                    records.append(dict(window=win, frame=fi, crf=clab, model=name,
                        cls=FACECLASS[win],
                        lpips=float(lpips(pt, gt_t).item()), dists=float(dists(pt, gt_t).item()),
                        psnr=float(psnr(pt, gt_t).item()), varlap=var_lap(pred),
                        gt_varlap=var_lap(gt), lr_bytes=nbytes))
                    key = (win, clab, name)
                    s = seqs.setdefault(key, {"gt": [], "pred": []})
                    s["gt"].append(gt); s["pred"].append(pred)
                    del pt
                    if DEV.type == "mps": torch.mps.empty_cache()
            del gt_t
            if DEV.type == "mps": torch.mps.empty_cache()
        print(f"[run] {win} done ({len(frames)} frames)")

    # --- VMAF-NEG guardrail: one temporal score per (window,crf,model) ---
    vmafneg = []
    if neg_ok:
        print("[vmaf-neg] computing guardrail column ...")
        for (win, clab, name), s in seqs.items():
            try:
                v = vneg.vmaf_neg(s["gt"], s["pred"])
            except Exception as e:
                print(f"[vmaf-neg] {win}/{clab}/{name} FAILED: {type(e).__name__}: {str(e)[:100]}")
                v = float("nan")
            vmafneg.append(dict(window=win, crf=clab, model=name, vmaf_neg=v))
    else:
        print("[vmaf-neg] backend unavailable -> A delivered with LPIPS/DISTS/PSNR only")

    lat = {k: dict(median_ms=float(np.median(v)), n=len(v)) for k, v in _LAT.items()}
    json.dump(dict(records=records, vmafneg=vmafneg, latency=lat, meta=meta,
                   n_frames=N_FRAMES, crf=CRF_LEVELS, faceclass=FACECLASS),
              open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print(f"[run] wrote {len(records)} records + {len(vmafneg)} vmaf-neg cells -> results.json")
    print("[lat] median(128->restore): " + " ".join(f"{k}={v['median_ms']:.0f}ms" for k,v in lat.items()))

if __name__ == "__main__":
    main()
