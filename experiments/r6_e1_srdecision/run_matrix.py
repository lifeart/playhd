#!/usr/bin/env python3
"""
R6-E1: stress-test the R5-E2 finding (compact realesr-general-x4v3 BEATS heavy
RealESRGAN_x4plus on perceptual quality) across CONTENT x DEGRADE, and give
x4plus its best shot with heavier/grittier degrade operators.

PROTOCOL (R5-E2 degrade-and-restore; no true HD GT exists for sample.mp4):
  SD frame (640x320) = pseudo-HD GT -> degrade 2x -> 320x160 LR -> restore 2x
  back to 640x320 through prototype.sr.upscale_to (x4 SR + INTER_CUBIC down to
  target) -> score restored vs GT with FULL-REFERENCE metrics LED BY TRUE LPIPS
  (AlexNet). var-Lap is NR/secondary only (GOTCHA #23).

CONTENT (5 windows): talkinghead@5000 (smooth face), highmotion@0 (low-detail
  intro), texture18000 / texture24000 / texture46000 (genuinely high-texture
  windows found by a var-Lap/HF scan -- where heavy SR *should* help).

DEGRADE (3 operators, increasingly x4plus's home turf):
  moderate = R5-E2 'real'  : blur .8 -> 2x AREA -> JPEG q40 -> noise s2
  heavy                    : blur 1.5 -> 2x AREA -> JPEG q25 -> noise s4
  gritty (2nd-order, RealESRGAN-style): blur 1.5 -> 2x AREA -> noise s4 ->
        JPEG q30 -> blur 1.0 -> JPEG q18  (double blur+compress = what x4plus
        was TRAINED to invert -> its best shot)

The SAME degraded LR is fed to every model (precomputed once per frame) so the
A/B is fair. Per-frame LPIPS kept so we can report x4plus win-RATE, not just
means. GPU(MPS) shared -> small windows, empty_cache between models, timing as
ratios only. READ-ONLY import of prototype/sr.py and r5_e2_quality/metrics.py.
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import av, cv2, numpy as np, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))                 # read-only
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r5_e2_quality"))  # read-only metrics
import sr as SR          # noqa: E402
import metrics as M      # noqa: E402

SAMPLE = os.path.join(_ROOT, "sample.mp4")

WINDOWS = {
    "talkinghead": 5000,   # R5-E2 smooth face (continuity)
    "highmotion":  0,      # R5-E2 low-detail intro (continuity)
    "texture18k":  18000,  # peak HF energy
    "texture24k":  24000,  # high var-Lap, lower motion (crisp texture)
    "texture46k":  46000,  # textured, low motion
}
DEGRADES = ["moderate", "heavy", "gritty"]
MODELS   = ["bicubic", "compact", "x4plus"]


def decode_window(path, start_frame, n):
    cont = av.open(path); vs = cont.streams.video[0]
    out, idx = [], 0
    for frame in cont.decode(vs):
        if idx < start_frame:
            idx += 1; continue
        if len(out) >= n:
            break
        out.append(frame.to_ndarray(format="rgb24")); idx += 1
    cont.close()
    return out


def degrade(gt, mode, seed):
    """SD GT (HxW) -> degraded LR at half size. Deterministic per (frame) seed."""
    h, w = gt.shape[:2]
    rng = np.random.default_rng(seed)
    def jpeg(x, q):
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if ok else x
    def noise(x, s):
        return np.clip(x.astype(np.float32) + rng.normal(0, s, x.shape), 0, 255).astype(np.uint8)
    if mode == "moderate":                                   # == R5-E2 'real'
        x = cv2.GaussianBlur(gt, (0, 0), 0.8)
        x = cv2.resize(x, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        x = jpeg(x, 40); x = noise(x, 2.0); return x
    if mode == "heavy":
        x = cv2.GaussianBlur(gt, (0, 0), 1.5)
        x = cv2.resize(x, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        x = jpeg(x, 25); x = noise(x, 4.0); return x
    if mode == "gritty":                                     # 2nd-order, RealESRGAN-style
        x = cv2.GaussianBlur(gt, (0, 0), 1.5)
        x = cv2.resize(x, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        x = noise(x, 4.0); x = jpeg(x, 30)
        x = cv2.GaussianBlur(x, (0, 0), 1.0); x = jpeg(x, 18); return x
    raise ValueError(mode)


def restore(lr, w, h, model):
    if model == "bicubic":
        return cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    if model == "compact":
        return SR.upscale_to(lr, w, h, model="realesrgan", half=False)
    if model == "x4plus":
        return SR.upscale_to(lr, w, h, model="realesrgan-x4plus", half=False)
    raise ValueError(model)


def free_gpu(model_name=None, half=None):
    if model_name is not None:
        for key in list(SR._MODELS.keys()):
            if key[0] == model_name and (half is None or key[1] == half):
                del SR._MODELS[key]
    if torch.backends.mps.is_available():
        torch.mps.synchronize(); torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    print(f"[setup] decoding {len(WINDOWS)} windows (n={args.n}) ...")
    windows = {k: decode_window(SAMPLE, s, args.n) for k, s in WINDOWS.items()}
    for k, v in windows.items():
        gu = cv2.cvtColor(v[0], cv2.COLOR_RGB2GRAY)
        print(f"  {k:11s} @{WINDOWS[k]:6d}: {len(v)}f  GT varLap={cv2.Laplacian(gu,cv2.CV_64F).var():.0f}")

    # Precompute the degraded LR ONCE per (window,degrade,frame) -> identical input to all models.
    lr_cache = {}   # (wname,dmode,i) -> lr
    for wname, gt in windows.items():
        for dmode in DEGRADES:
            for i, g in enumerate(gt):
                lr_cache[(wname, dmode, i)] = degrade(g, dmode, seed=1000 + i)

    # results[wname][dmode][model] = {metric: mean, ..., 'lpips_per': [...]}
    results, timing = {}, {}
    for model in MODELS:
        t_all = []
        for wname, gt in windows.items():
            h, w = gt[0].shape[:2]
            for dmode in DEGRADES:
                restored = []
                for i, g in enumerate(gt):
                    lr = lr_cache[(wname, dmode, i)]
                    t0 = time.perf_counter()
                    r = restore(lr, w, h, model)
                    t_all.append((time.perf_counter() - t0) * 1000.0)
                    restored.append(r)
                mt = M.mean_full_ref(restored, gt)
                mt["tof"] = M.tof(restored, gt)
                mt["lpips_per"] = [M.lpips_dist(r, g) for r, g in zip(restored, gt)]
                results.setdefault(wname, {}).setdefault(dmode, {})[model] = mt
                print(f"  [{model:8s}|{dmode:8s}] {wname:11s} "
                      f"LPIPS={mt['lpips']:.4f} PSNR={mt['psnr']:.2f} SSIM={mt['ssim']:.4f} "
                      f"MS-SSIM={mt['ms_ssim']:.4f} gradFid={mt['grad_fid']:.2f} varLap={mt['varlap']:.0f}")
        timing[model] = float(np.mean(t_all))
        free_gpu({"compact": "realesrgan", "x4plus": "realesrgan-x4plus"}.get(model))
        print(f"  -- {model} mean restore {timing[model]:.1f} ms/frame (ratio use only) --")

    out = dict(n=args.n, windows=WINDOWS, degrades=DEGRADES, models=MODELS,
               results=results, timing_ms_per_frame=timing)
    with open(os.path.join(_HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] timing ratios: compact={timing['compact']/timing['compact']:.1f}x "
          f"x4plus={timing['x4plus']/timing['compact']:.1f}x  -> results.json")


if __name__ == "__main__":
    main()
