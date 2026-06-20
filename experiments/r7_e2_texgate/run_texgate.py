#!/usr/bin/env python3
"""
R7-E2: TEXTURE-gated region-aware blend -- cut quality-mode heavy-SR compute at
~0 perceptual cost.

OPPORTUNITY (from R6-E1, TRUE LPIPS): heavy x4plus only BEATS compact on
DETAILED/TEXTURED content. On smooth content (still face, sky, flat fields) its
extra HF is misaligned -> ~0 LPIPS gain (sometimes worse). The CURRENT quality-mode
region gate (derisk._build_region_gate) keys ONLY on MOTION: a_motion ~ 1 on STATIC
regions -> heavy x4plus, a_motion ~ 0 on MOVING -> cheap compact. So a STATIC-but-
SMOOTH region (still face / sky) gets the 23x-compute heavy model for ~0 benefit.

THIS EXPERIMENT: gate the heavy model on TEXTURE x static:
    a' = a_motion * a_texture
where a_texture in [0,1] is a CHEAP, GT-FREE local-detail measure (local std of luma
on the bicubic upscale -- available essentially for free, NO heavy SR needed),
smoothed + thresholded + feathered. Heavy x4plus is applied only where the region is
BOTH static AND textured; compact (or bicubic) elsewhere.

MEASUREMENT = R6-E1 / R5-E2 degrade-restore protocol, LED BY TRUE AlexNet LPIPS
(var-Lap NR/secondary only -- GOTCHA #23):
  SD frame (640x320) = pseudo-HD GT -> degrade 2x -> 320x160 LR -> restore 2x back
  to 640x320 via prototype.sr.upscale_to. heavy=x4plus, compact=realesrgan, the
  CHEAP bicubic is the texture source. Identical degraded LR to every model.

REPORT per (window x gate): heavy-SR area % (mean of a'>eps -> the tile-skip compute
that would actually be spent), LPIPS vs GT, LPIPS delta vs the motion-only gate (must
be ~equal -> within noise), var-Lap (secondary), est compute saved.

GPU(MPS) SHARED with a sibling -> small windows (n frames), free GPU between models,
timing as ratios only. READ-ONLY import of prototype/{derisk,region_quality,sr} and
r5_e2_quality/metrics. New files only under this dir. System ffmpeg BROKEN -> PyAV.
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import av, cv2, numpy as np, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "prototype"))                     # read-only
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r5_e2_quality"))  # read-only metrics
import derisk as D            # noqa: E402  (read-only: decode + build_lr_flow)
import region_quality as RQ   # noqa: E402  (read-only: region_masks/window_static_weight/blend)
import sr as SR               # noqa: E402
import metrics as M           # noqa: E402

SAMPLE = os.path.join(_ROOT, "sample.mp4")
CACHE  = os.path.join(_HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

# talking-head (mostly SMOOTH -> should drop most heavy-SR area) + a detailed/graphics
# window (texture24k chart+text -> should KEEP heavy where it matters). Same windows R6-E1
# used so the LPIPS numbers are directly comparable.
WINDOWS  = {"talkinghead": 5000, "texture24k": 24000}
DEGRADES = ["moderate", "heavy"]
EPS_HEAVY = 0.05   # a' below this -> heavy contributes <5% -> a tile-skip impl would NOT run heavy there


# ----------------------------------------------------------------------------- #
# Decode all windows in ONE sequential MV-export pass (refs must be decoded in order)
# ----------------------------------------------------------------------------- #
def decode_windows(path, windows, n):
    from av.sidedata.sidedata import Type as SDType
    targets = sorted(windows.items(), key=lambda kv: kv[1])
    last_start = targets[-1][1]
    cont = av.open(path); vs = cont.streams.video[0]
    vs.codec_context.options = {"flags2": "+export_mvs"}
    out = {k: [] for k in windows}
    spans = {k: (s, s + n) for k, s in windows.items()}
    idx = 0
    for frame in cont.decode(vs):
        if idx > last_start + n:
            break
        in_any = any(s <= idx < e for (s, e) in spans.values())
        if in_any:
            img = frame.to_ndarray(format="rgb24")
            try:
                sd = frame.side_data.get(SDType.MOTION_VECTORS)
            except Exception:
                sd = None
            mvs = sd.to_ndarray() if sd is not None else None
            pt = {1: "I", 2: "P", 3: "B"}.get(int(frame.pict_type), "?")
            for k, (s, e) in spans.items():
                if s <= idx < e:
                    out[k].append((pt, img, mvs))
        idx += 1
    cont.close()
    return out


# ----------------------------------------------------------------------------- #
# Degrade (R6-E1, verbatim) + restore + cache
# ----------------------------------------------------------------------------- #
def degrade(gt, mode, seed):
    h, w = gt.shape[:2]
    rng = np.random.default_rng(seed)
    def jpeg(x, q):
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if ok else x
    def noise(x, s):
        return np.clip(x.astype(np.float32) + rng.normal(0, s, x.shape), 0, 255).astype(np.uint8)
    if mode == "moderate":
        x = cv2.GaussianBlur(gt, (0, 0), 0.8)
        x = cv2.resize(x, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        x = jpeg(x, 40); x = noise(x, 2.0); return x
    if mode == "heavy":
        x = cv2.GaussianBlur(gt, (0, 0), 1.5)
        x = cv2.resize(x, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        x = jpeg(x, 25); x = noise(x, 4.0); return x
    raise ValueError(mode)


def restore(lr, w, h, model):
    if model == "bicubic":
        return cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    if model == "compact":
        return SR.upscale_to(lr, w, h, model="realesrgan", half=False)
    if model == "x4plus":
        return SR.upscale_to(lr, w, h, model="realesrgan-x4plus", half=False)
    raise ValueError(model)


def free_gpu(model_name=None):
    if model_name is not None:
        for key in list(SR._MODELS.keys()):
            if key[0] == model_name:
                del SR._MODELS[key]
    if torch.backends.mps.is_available():
        torch.mps.synchronize(); torch.mps.empty_cache()


def build_restores(windows_frames, n):
    """Per (window,degrade): GT, LR, bicubic, compact, heavy. Cached to npy so gate
    re-tuning never re-runs SR. Heavy SR runs LAST and the model is freed after."""
    gt = {w: [f[1] for f in fr] for w, fr in windows_frames.items()}
    H, Wd = gt[next(iter(gt))][0].shape[:2]
    lr = {}
    for w, gts in gt.items():
        for dm in DEGRADES:
            lr[(w, dm)] = [degrade(g, dm, seed=1000 + i) for i, g in enumerate(gts)]

    cache = {}
    timing = {}
    for model in ["bicubic", "compact", "x4plus"]:
        t_all = []
        for w, gts in gt.items():
            h, wd = gts[0].shape[:2]
            for dm in DEGRADES:
                path = os.path.join(CACHE, f"{w}_{dm}_{model}_{n}.npy")
                if os.path.exists(path):
                    cache[(w, dm, model)] = list(np.load(path))
                    continue
                outs = []
                for i, g in enumerate(gts):
                    t0 = time.perf_counter()
                    r = restore(lr[(w, dm)][i], wd, h, model)
                    t_all.append((time.perf_counter() - t0) * 1000.0)
                    outs.append(r)
                np.save(path, np.stack(outs))
                cache[(w, dm, model)] = outs
        if t_all:
            timing[model] = float(np.mean(t_all))
        free_gpu({"compact": "realesrgan", "x4plus": "realesrgan-x4plus"}.get(model))
    return gt, lr, cache, timing


# ----------------------------------------------------------------------------- #
# Gates
# ----------------------------------------------------------------------------- #
def motion_gate(frames, h, w, lo=0.2, hi=1.0, feather=61):
    """EXACT replica of derisk._build_region_gate's a_lr (motion-only, temporally stable)."""
    _, _, meanmag, _ = RQ.region_masks(frames, h, w, 45.0, 80.0)
    return RQ.window_static_weight(meanmag, lo, hi, feather=feather)


def texture_map(restored_list, k=7):
    """CHEAP GT-FREE local-detail statistic: temporal-mean of per-frame local luma STD
    (box-filter sqrt(E[x^2]-E[x]^2), k x k). High on text/edges/texture, low on smooth
    skin/sky. Computed on the bicubic (or compact) restore -> NO heavy SR needed."""
    acc = None
    for r in restored_list:
        y = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mu = cv2.boxFilter(y, -1, (k, k))
        mu2 = cv2.boxFilter(y * y, -1, (k, k))
        std = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
        acc = std if acc is None else acc + std
    return acc / len(restored_list)


def texture_gate(std_map, t_lo, t_hi, feather=61):
    """std -> a_texture in [0,1]: 0 below t_lo (smooth -> NO heavy), 1 above t_hi
    (textured -> heavy OK), linear ramp; then a feather (Gaussian) to match the motion
    gate's wide soft seam (no per-frame seam jitter; map is already temporally pooled)."""
    a = np.clip((std_map - t_lo) / max(t_hi - t_lo, 1e-6), 0.0, 1.0).astype(np.float32)
    if feather and feather >= 3:
        kk = int(feather) | 1
        a = cv2.GaussianBlur(a, (kk, kk), 0)
    return a


def blend(heavy, compact, a, scale=1):
    return RQ.blend_region_aware(heavy, compact, a, scale)


def _tile_cov(a, tile):
    h, w = a.shape
    ny, nx = (h + tile - 1) // tile, (w + tile - 1) // tile
    fired = 0
    for ty in range(ny):
        for tx in range(nx):
            blk = a[ty*tile:(ty+1)*tile, tx*tile:(tx+1)*tile]
            if blk.size and blk.max() > EPS_HEAVY:
                fired += 1
    return float(fired / (ny * nx))


def heavy_area(a):
    """Compute proxies for a region-skip heavy-SR impl, from MOST to LEAST optimistic:
      mean_a   = mean(a') -- continuous/ideal 'effective heavy fraction' (only realizable if
                 compute could be weighted per-pixel; a lower bound on heavy area),
      px_eps   = fraction of pixels with a'>eps -- pixel-mask skip (NN can't really skip single
                 pixels, but this is the floor a fine sparse impl approaches),
      t16/t32/t64 = fraction of 16/32/64px HD tiles containing ANY a'>eps pixel -> the
                 REALIZABLE tiled heavy-SR compute at that granularity (run heavy on a tile iff
                 it overlaps textured-static content). THIS is the honest compute-saved headline;
                 it rises as tiles coarsen because scattered texture lights up more tiles."""
    return dict(mean=float(np.mean(a)), px=float(np.mean(a > EPS_HEAVY)),
                t16=_tile_cov(a, 16), t32=_tile_cov(a, 32), t64=_tile_cov(a, 64))


def lpips_seq(restored, gt):
    return float(np.mean([M.lpips_dist(r, g) for r, g in zip(restored, gt)]))


def varlap_seq(restored):
    return float(np.mean([M.var_laplacian(r) for r in restored]))


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--probe", action="store_true",
                    help="print texture-std distributions to calibrate t_lo/t_hi, then exit")
    ap.add_argument("--texsrc", default="bicubic", choices=["bicubic", "compact"])
    args = ap.parse_args()

    print(f"[decode] {list(WINDOWS)} n={args.n} (single MV-export pass) ...")
    t0 = time.perf_counter()
    wf = decode_windows(SAMPLE, WINDOWS, args.n)
    print(f"  decoded in {time.perf_counter()-t0:.1f}s: " +
          ", ".join(f"{k}={len(v)}f" for k, v in wf.items()))

    gt, lr, cache, timing = build_restores(wf, args.n)
    H, Wd = gt[next(iter(gt))][0].shape[:2]
    print(f"  frame size {Wd}x{H}; restore timing ms/frame {timing}")

    # texture distributions per (window,degrade)
    texsrc = args.texsrc
    tex = {}
    for w in WINDOWS:
        for dm in DEGRADES:
            tex[(w, dm)] = texture_map(cache[(w, dm, texsrc)])

    if args.probe:
        print(f"\n[probe] local-std (k=7) on {texsrc} restore -- percentiles per window x degrade:")
        for (w, dm), s in tex.items():
            ps = np.percentile(s, [10, 25, 50, 75, 90, 95, 99])
            gu = cv2.cvtColor(gt[w][0], cv2.COLOR_RGB2GRAY)
            print(f"  {w:11s} {dm:8s} GTvarLap={cv2.Laplacian(gu,cv2.CV_64F).var():6.0f}  "
                  f"std p10/25/50/75/90/95/99 = " + "/".join(f"{x:.1f}" for x in ps))
        return

    # motion gate (exact derisk replica) per window
    amot = {}
    for w in WINDOWS:
        h_lr, w_lr = gt[w][0].shape[:2]
        amot[w] = motion_gate(wf[w], h_lr, w_lr)

    # texture thresholds (ABSOLUTE local-std, so the gate is content-adaptive: smooth
    # windows drop heavy area, textured windows keep it -- a percentile threshold would
    # drop the SAME fraction everywhere and could not tell smooth from textured).
    TEX_CFG = {
        "tex_lo": dict(t_lo=4.0, t_hi=10.0, feather=61),   # gentle: only the smoothest pixels drop
        "tex_md": dict(t_lo=6.0, t_hi=14.0, feather=61),   # medium threshold, wide (motion-matched) feather
        "tex_hi": dict(t_lo=8.0, t_hi=18.0, feather=61),   # aggressive: keep only clearly-textured
        # feather sweep at the md threshold: the texture map is TEMPORALLY STABLE so it does
        # NOT need the motion gate's wide anti-tear feather. A tighter feather shrinks the soft
        # halo -> realizes much more of the tile-skip saving. Does LPIPS hold?
        "tex_md_f21": dict(t_lo=6.0, t_hi=14.0, feather=21),
        "tex_md_f9":  dict(t_lo=6.0, t_hi=14.0, feather=9),
    }

    results = {}
    for w in WINDOWS:
        for dm in DEGRADES:
            heavy = cache[(w, dm, "x4plus")]
            comp  = cache[(w, dm, "compact")]
            bic   = cache[(w, dm, "bicubic")]
            g = gt[w]
            full = dict(mean=1.0, px=1.0, t16=1.0, t32=1.0, t64=1.0)
            zero = dict(mean=0.0, px=0.0, t16=0.0, t32=0.0, t64=0.0)
            rec = {}
            # references
            rec["all_compact"] = dict(area=zero,
                                      lpips=lpips_seq(comp, g), varlap=varlap_seq(comp))
            rec["all_heavy"]   = dict(area=full,
                                      lpips=lpips_seq(heavy, g), varlap=varlap_seq(heavy))
            # current production gate: motion only
            am = amot[w]
            mot_blend = [blend(heavy[i], comp[i], am) for i in range(len(g))]
            rec["motion"] = dict(area=heavy_area(am),
                                 lpips=lpips_seq(mot_blend, g), varlap=varlap_seq(mot_blend))
            # texture-gated: a' = a_motion * a_texture, compact elsewhere
            for name, cfg in TEX_CFG.items():
                at = texture_gate(tex[(w, dm)], **cfg)
                ap_ = am * at
                tb = [blend(heavy[i], comp[i], ap_) for i in range(len(g))]
                rec[name] = dict(area=heavy_area(ap_),
                                 lpips=lpips_seq(tb, g), varlap=varlap_seq(tb), cfg=cfg)
            # aggressive variant: non-textured -> BICUBIC (even cheaper than compact)
            at = texture_gate(tex[(w, dm)], **TEX_CFG["tex_md"])
            ap_ = am * at
            tbB = [blend(heavy[i], bic[i], ap_) for i in range(len(g))]
            rec["tex_md_bicubic"] = dict(area=heavy_area(ap_),
                                         lpips=lpips_seq(tbB, g), varlap=varlap_seq(tbB))
            results[f"{w}|{dm}"] = rec
            print(f"\n=== {w} | {dm} ===")
            base = rec["motion"]["lpips"]
            for name, r in rec.items():
                ar = r["area"]
                dl = r["lpips"] - base
                print(f"  {name:16s} t16/t32/t64={ar['t16']*100:4.0f}/{ar['t32']*100:4.0f}/"
                      f"{ar['t64']*100:4.0f}%  px={ar['px']*100:4.0f}%  meanA={ar['mean']*100:4.0f}%  "
                      f"LPIPS={r['lpips']:.4f}  dMot={dl:+.4f}  vL={r['varlap']:5.0f}")

    out = dict(n=args.n, windows=WINDOWS, degrades=DEGRADES, texsrc=texsrc,
               eps_heavy=EPS_HEAVY, tex_cfg=TEX_CFG, timing_ms=timing, results=results)
    with open(os.path.join(_HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] -> results.json   (heavy/compact timing ratio "
          f"{timing.get('x4plus',float('nan'))/max(timing.get('compact',1),1e-6):.1f}x)")


if __name__ == "__main__":
    main()
