"""BEFORE/AFTER benchmark for the instant-mode speedup (Levers 1-4).

Runs the SAME talking-head window through two pipelines in one process for a clean,
contention-free per-component comparison:

  BEFORE = derisk.build_perframe_cache (SR every frame)  ->  reconstruct(download=True)
           ->  grain.apply_grain (CPU)  ->  libx264 (software) encode.
  AFTER  = anchor_sr.build_anchor_cache (SR anchors + adaptive)  ->  reconstruct(download=False,
           GPU-resident)  ->  fast_grain GpuGrain (MPS)  ->  one img_to_host  ->  VideoToolbox HW.

Timing is steady-state (SR warmup excluded via the sr-module latency accounting / the build's
internal reset; GPU sections are MPS-synchronised for honest wall time). Quality is verified
three ways: (1) anchor-only SR fallback delta -- grain-OFF recon PSNR before-vs-after per frame
+ the per-frame fallback fraction; (2) grain port parity vs apply_grain with a shared template;
(3) encode fidelity -- decoded output PSNR vs the pre-encode frames. tOF (temporal) is reported
for both. A before/after crop is saved for visual inspection.

    python3 server/bench_instant.py [--input F] [--max-frames N] [--thresh 0.08]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.join(os.path.dirname(_HERE), "prototype")
for p in (_HERE, _PROTO):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import derisk
import grain as _grain
import gpu_ops as G
import sr as _srmod

import pipeline_api as P
import anchor_sr
import fast_grain

SCALE = P.SCALE
STRENGTH = "med"
# Occlusion mode the AFTER (instant) path actually ships with -- read from the live instant config
# so the bench tracks what production runs (Lever 1 flipped this to 'reactive').
OCC_AFTER = P.MODE_CONFIG["instant"]["occ"]


def _psnr(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-9 else 10.0 * np.log10(255.0 ** 2 / mse)


def _sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _tof_seq(recons, lrs, w_lr, h_lr):
    """Prototype tOF of an HD recon sequence vs the LR sequence (lower = temporally smoother)."""
    sm = (w_lr, h_lr)
    seq = [cv2.resize(r, sm) for r in recons]
    ref = [f if f.shape[1::-1] == sm else cv2.resize(f, sm) for f in lrs]
    return derisk.tof(seq, ref)


def run_before(chunks, w_hd, h_hd, out_path):
    """Full-SR + CPU grain + libx264. Returns (timings_ms_total, preencode_frames, recons_nograin)."""
    _srmod.reset_latency("realesrgan")
    writer = P._VideoWriter(out_path, 25, codec="libx264")
    t = dict(sr=0.0, recon=0.0, grain=0.0, encode=0.0, n=0)
    preenc, recons_ng, lrs = [], [], []
    done = 0
    for chunk in chunks:
        ts = time.perf_counter()
        cache = derisk.build_perframe_cache(chunk, w_hd, h_hd, "realesrgan")
        # steady-state SR time = sum of the sr-module per-call latencies (warmup excluded by
        # build_perframe_cache's internal reset_latency).
        t["sr"] += sum(_srmod._LAT.get("realesrgan", [])) / 1000.0
        _srmod.reset_latency("realesrgan")
        _ = time.perf_counter() - ts

        _sync(); tr = time.perf_counter()
        _, R = derisk.reconstruct(chunk, None, SCALE, True, "adaptive", cache, set(),
                                  backend="torch", collect_metrics=False, download_output=True)
        _sync(); t["recon"] += time.perf_counter() - tr

        for i in range(len(chunk)):
            recon = R[i]["recon"]                       # numpy uint8 (downloaded)
            recons_ng.append(recon.copy())
            lrs.append(chunk[i][1])
            tg = time.perf_counter()
            grained = _grain.apply_grain(recon, done, STRENGTH)
            t["grain"] += time.perf_counter() - tg
            preenc.append(grained.copy())
            te = time.perf_counter()
            writer.write(grained)
            t["encode"] += time.perf_counter() - te
            done += 1
        t["n"] += len(chunk)
        del cache, R, chunk
        P._free_gpu()
    te = time.perf_counter()
    writer.close()
    t["encode"] += time.perf_counter() - te
    return t, preenc, recons_ng, lrs


def run_after(chunks, w_hd, h_hd, out_path, thresh, approach="patch"):
    """Anchor SR + GPU grain + VideoToolbox. `approach`: 'patch' (single reconstruct, the
    adaptive safeguard patches high-fallback frames' fallback pixels with real SR afterwards)
    or 'prescan' (a cheap LR occlusion scan SRs high-fallback frames in the cache BEFORE
    reconstruct, so upgraded backbone detail propagates). Returns timings + frames + info."""
    _srmod.reset_latency("realesrgan")
    dev = G.device()
    gg = fast_grain.GpuGrain(h_hd, w_hd, dev)
    writer = P._VideoWriter(out_path, 25)               # auto -> HW if available
    t = dict(sr=0.0, recon=0.0, grain=0.0, download=0.0, encode=0.0, n=0)
    preenc, recons_ng = [], []
    info_all = []
    done = 0
    for chunk in chunks:
        sr_set = None
        if approach == "prescan":
            cache, info = anchor_sr.build_anchor_cache_prescan(
                chunk, w_hd, h_hd, "realesrgan", occ_mode=OCC_AFTER, fallback_thresh=thresh)
            t["sr"] += info["t_scan_s"] + info["t_build_s"]   # full scan + (SR forwards + bicubic)
        else:                                                 # 'hybrid' (default)
            cache, info, sr_set = anchor_sr.build_anchor_cache(
                chunk, w_hd, h_hd, "realesrgan", occ_mode=OCC_AFTER, fallback_thresh=thresh,
                tile=P.INSTANT_TILE_SR, gpu_cache=P.INSTANT_GPU_CACHE)
            t["sr"] += info["t_scan_s"] + info["t_cache_s"]   # backbone scan + (SR + bicubic)
        _srmod.reset_latency("realesrgan")

        _sync(); tr = time.perf_counter()
        _, R = derisk.reconstruct(chunk, None, SCALE, True, OCC_AFTER, cache, set(),
                                  backend="torch", collect_metrics=False, download_output=False)
        _sync(); t["recon"] += time.perf_counter() - tr

        if approach == "hybrid":
            _sync(); tp = time.perf_counter()
            pinfo = anchor_sr.patch_high_fallback(chunk, R, w_hd, h_hd, "realesrgan",
                                                  fallback_thresh=thresh, skip=sr_set,
                                                  tile=P.INSTANT_TILE_SR)
            _sync(); t["sr"] += time.perf_counter() - tp      # B-leaf adaptive-upgrade SR forwards
            info = {**info, **pinfo}
        info_all.append(info)

        for i in range(len(chunk)):
            recon_t = R[i]["recon"]                      # GPU tensor [1,3,H,W] float, resident
            _sync(); tg = time.perf_counter()
            grained_t = gg.apply(recon_t, done, STRENGTH)
            _sync(); t["grain"] += time.perf_counter() - tg
            td = time.perf_counter()
            grained = fast_grain.download_rgb(grained_t)
            t["download"] += time.perf_counter() - td
            recons_ng.append(G.img_to_host(recon_t))     # grain-off, for the SR/recon quality check
            preenc.append(grained.copy())
            te = time.perf_counter()
            writer.write(grained)
            t["encode"] += time.perf_counter() - te
            done += 1
        t["n"] += len(chunk)
        del cache, R, chunk
        P._free_gpu()
    te = time.perf_counter()
    writer.close()
    t["encode"] += time.perf_counter() - te
    return t, preenc, recons_ng, info_all, writer.encoder


def _decode_frames(path, n):
    import av
    cont = av.open(path); vs = cont.streams.video[0]
    out = []
    for fr in cont.decode(vs):
        out.append(fr.to_ndarray(format="rgb24"))
        if len(out) >= n:
            break
    cont.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(P.TESTDATA_DIR, "short.mp4"))
    ap.add_argument("--max-frames", type=int, default=48)
    ap.add_argument("--thresh", type=float, default=0.08)
    ap.add_argument("--skip-chunks", type=int, default=0,
                    help="drop the first K GOP chunks (e.g. skip a high-motion intro)")
    ap.add_argument("--approach", choices=["hybrid", "prescan"], default="hybrid",
                    help="adaptive-safeguard strategy (default: hybrid -- backbone pre-scan in "
                         "cache + B-leaf post-hoc patch, single reconstruct)")
    a = ap.parse_args()

    print(f"[bench] input={a.input} max_frames={a.max_frames} thresh={a.thresh} "
          f"skip_chunks={a.skip_chunks}")
    # Decode enough to cover the skip + the window, then drop the leading (GOP-aligned) chunks.
    raw = list(P.stream_gops(a.input))
    raw = raw[a.skip_chunks:]
    chunks, tot = [], 0
    for ch in raw:
        chunks.append(ch); tot += len(ch)
        if tot >= a.max_frames:
            break
    h_lr, w_lr = chunks[0][0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    N = sum(len(c) for c in chunks)
    print(f"[bench] {len(chunks)} chunk(s), {N} frames, LR {w_lr}x{h_lr} -> HD {w_hd}x{h_hd}")

    # Pre-warm the SR net + MPS graph so the FIRST timed build is steady-state.
    _srmod.load_model("realesrgan")
    for _ in range(3):
        _srmod.upscale(chunks[0][0][1], model="realesrgan")
    _srmod.reset_latency("realesrgan")
    P._free_gpu()

    out_before = os.path.join(P.OUTPUTS_DIR, "_bench_before.mp4")
    out_after = os.path.join(P.OUTPUTS_DIR, "_bench_after.mp4")

    tb, pre_b, ng_b, lrs = run_before(chunks, w_hd, h_hd, out_before)
    P._free_gpu()
    ta, pre_a, ng_a, info_all, enc_used = run_after(chunks, w_hd, h_hd, out_after, a.thresh,
                                                    approach=a.approach)
    print(f"[bench] AFTER adaptive-safeguard approach: {a.approach}")

    n = tb["n"]
    def pf(d, k):  # per-frame ms
        return d.get(k, 0.0) * 1000.0 / n

    before_tot = pf(tb, "sr") + pf(tb, "recon") + pf(tb, "grain") + pf(tb, "encode")
    after_tot = (pf(ta, "sr") + pf(ta, "recon") + pf(ta, "grain")
                 + pf(ta, "download") + pf(ta, "encode"))

    print("\n=== PER-COMPONENT ms/frame (steady-state, single GPU, no contention) ===")
    print(f"{'component':<22}{'BEFORE':>10}{'AFTER':>10}{'speedup':>10}")
    rows = [
        ("SR", pf(tb, "sr"), pf(ta, "sr")),
        ("reconstruct", pf(tb, "recon"), pf(ta, "recon")),
        ("grain", pf(tb, "grain"), pf(ta, "grain")),
        ("download (GPU->host)", 0.0, pf(ta, "download")),
        ("encode", pf(tb, "encode"), pf(ta, "encode")),
    ]
    for name, b, aft in rows:
        sp = (b / aft) if aft > 1e-6 else float("nan")
        sps = f"{sp:.1f}x" if np.isfinite(sp) else "  -"
        print(f"{name:<22}{b:>10.1f}{aft:>10.1f}{sps:>10}")
    print("-" * 52)
    print(f"{'TOTAL':<22}{before_tot:>10.1f}{after_tot:>10.1f}{before_tot/after_tot:>9.1f}x")
    print(f"{'fps':<22}{1000.0/before_tot:>10.1f}{1000.0/after_tot:>10.1f}")
    print(f"[bench] AFTER video encoder actually used: {enc_used}")

    # ---- SR accounting ----
    tot_sr = sum(i["n_sr_calls"] for i in info_all)
    tot_up = sum(i["n_adaptive_upgrades"] for i in info_all)
    print("\n=== Lever 1: anchor-only SR ===")
    print(f"SR calls: BEFORE {n}/{n} (1.00/frame)  AFTER {tot_sr}/{n} "
          f"({tot_sr/n:.3f}/frame), of which {tot_up} adaptive upgrade(s)")
    for k, i in enumerate(info_all):
        bbu = i.get("backbone_upgrades", i.get("adaptive_upgrades", []))
        leaf = i.get("leaf_upgrades", [])
        print(f"  chunk {k}: anchors={i['anchors']} backbone_upgrades={bbu} leaf_upgrades={leaf} "
              f"max_fallback={i['max_fallback_frac']:.3f} thresh={i['fallback_thresh']}")

    # ---- quality: anchor-only SR fallback delta (grain-OFF recon PSNR) ----
    print("\n=== Quality A: anchor-only SR vs full-SR (grain OFF) -- propagated identical, "
          "fallback differs ===")
    fr = {}
    for i in info_all:
        fr.update(i["fallback_fracs"])
    psnrs = [_psnr(ng_b[i], ng_a[i]) for i in range(n)]
    perfect = sum(1 for p in psnrs if p >= 60.0)
    worst = sorted(range(n), key=lambda i: psnrs[i])[:6]
    print(f"recon PSNR(before,after): mean={np.mean(psnrs):.2f} dB  min={np.min(psnrs):.2f} dB  "
          f"frames>=60dB(propagated-identical)={perfect}/{n}")
    print("  lowest-PSNR frames (these carry the bicubic-vs-compactSR fallback delta):")
    for i in worst:
        print(f"    frame {i:3d}: PSNR={psnrs[i]:6.2f} dB  fallback_frac={fr.get(i,0):.4f}  "
              f"{'[SR-upgraded]' if psnrs[i] >= 99 else ''}")

    # ---- quality: grain port parity ----
    print("\n=== Quality B: GPU grain vs CPU apply_grain (same template) ===")
    dev = G.device()
    gg = fast_grain.GpuGrain(h_hd, w_hd, dev)
    tmpl = _grain.make_template(h_hd, w_hd, seed=0)
    gp = []
    for i in (0, n // 2, n - 1):
        cpu = _grain.apply_grain(ng_b[i], i, STRENGTH, template=tmpl)
        t = torch.from_numpy(ng_b[i]).to(dev).permute(2, 0, 1).unsqueeze(0).float()
        gpu = G.img_to_host(gg.apply(t, i, STRENGTH))
        gp.append(_psnr(cpu, gpu))
    print(f"  PSNR(cpu,gpu) on real recon frames: {[round(x,2) for x in gp]} dB (>=45 = within rounding)")

    # ---- quality: encode fidelity ----
    print("\n=== Quality C: encode fidelity (decoded vs pre-encode frames) ===")
    dec_b = _decode_frames(out_before, n)
    dec_a = _decode_frames(out_after, n)
    eb = float(np.mean([_psnr(pre_b[i], dec_b[i]) for i in range(min(n, len(dec_b)))]))
    ea = float(np.mean([_psnr(pre_a[i], dec_a[i]) for i in range(min(n, len(dec_a)))]))
    print(f"  libx264(BEFORE) decoded PSNR={eb:.2f} dB ({len(dec_b)} frames)")
    print(f"  VideoToolbox(AFTER) decoded PSNR={ea:.2f} dB ({len(dec_a)} frames)")

    # ---- tOF (temporal) ----
    print("\n=== Quality D: tOF (temporal consistency; lower=smoother) ===")
    tof_b = _tof_seq(ng_b, lrs, w_lr, h_lr)
    tof_a = _tof_seq(ng_a, lrs, w_lr, h_lr)
    print(f"  tOF(recon vs LR): BEFORE={tof_b:.4f}  AFTER={tof_a:.4f}  "
          f"({'AFTER not worse' if tof_a <= tof_b * 1.05 else 'CHECK: AFTER worse'})")

    # ---- save a before/after crop (mid frame, center) ----
    mid = n // 2
    cy, cx = h_hd // 2, w_hd // 2
    crop = lambda im: im[cy - 160:cy + 160, cx - 240:cx + 240]
    cb, ca = crop(pre_b[mid]), crop(pre_a[mid])
    pair = np.concatenate([cb, ca], axis=1)[:, :, ::-1]   # RGB->BGR for cv2
    cpath = os.path.join(P.OUTPUTS_DIR, "_bench_crop_before_after.png")
    cv2.imwrite(cpath, pair)
    cv2.imwrite(os.path.join(P.OUTPUTS_DIR, "_bench_crop_before.png"), cb[:, :, ::-1])
    cv2.imwrite(os.path.join(P.OUTPUTS_DIR, "_bench_crop_after.png"), ca[:, :, ::-1])
    print(f"\n[bench] saved before|after crop -> {cpath}  (frame {mid}, PSNR={_psnr(cb,ca):.2f} dB)")
    print(f"[bench] outputs: {out_before} | {out_after}")


if __name__ == "__main__":
    main()
