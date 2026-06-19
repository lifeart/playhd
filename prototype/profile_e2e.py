#!/usr/bin/env python3
"""
Step 6 end-to-end profiler: real per-frame cost of the propagation path, split by frame type,
numpy (before) vs torch/MPS (after), plus the reactive-only-mask ablation.

Run:
  python3 profile_e2e.py                       # default: start 5000, 48 frames, x4, realesrgan
  python3 profile_e2e.py --start-frame 0 --max-frames 48 --sr bicubic   # high-motion / fast

Writes out_profile/{summary.txt, components.csv, quality.csv}. SR runs ONCE (cache reused for
every backend/mask variant); reconstruct is warm-up'd then timed over --reps reps (median rep).
"""
import argparse
import csv
import os
import statistics as st

import cv2
import numpy as np

import derisk
from derisk import PROF

CORE = ["build_flow", "residual", "warp", "mask", "blend"]   # the recon math
XFER = ["upload_perframe", "download"]                        # torch host<->device transfers
ALL_COMPONENTS = ["decode", "sr"] + CORE + XFER


# --------------------------------------------------------------------------- #
def group_per_frame(events):
    """events -> {fidx: {component: summed_ms, '_type': ftype}}. A B-frame warps twice
    (past+future), so per-component ms are SUMMED within a frame to get its true per-frame cost."""
    per = {}
    for comp, ftype, fidx, ms in events:
        d = per.setdefault(fidx, {"_type": ftype})
        d[comp] = d.get(comp, 0.0) + ms
        d["_type"] = ftype
    return per


def table_by_type(per, components):
    """{component: {type: mean_per_frame_ms}} over I/P/B/all, from one rep's per-frame dict."""
    by = {c: {"I": [], "P": [], "B": [], "all": []} for c in components}
    for fidx, d in per.items():
        t = d["_type"]
        for c in components:
            v = d.get(c, 0.0)
            if t in by[c]:
                by[c][t].append(v)
            by[c]["all"].append(v)
    out = {}
    for c in components:
        out[c] = {t: (float(np.mean(vs)) if vs else 0.0) for t, vs in by[c].items()}
    return out


def avg_tables(tables, components):
    """Average several per-rep tables (mean of per-rep means) for stability."""
    out = {}
    for c in components:
        out[c] = {}
        for t in ("I", "P", "B", "all"):
            out[c][t] = float(np.mean([tb[c][t] for tb in tables]))
    return out


def run_timed(fn, reps, warmup=1):
    """Warm up (discarded), then time `reps` reps; return a list of per-frame-grouped dicts."""
    for _ in range(warmup):
        PROF.reset(enabled=False)
        fn()
    pers = []
    for _ in range(reps):
        PROF.reset(enabled=True)
        fn()
        pers.append(group_per_frame(PROF.events))
    return pers


def fmt_table(title, tab, components, counts):
    lines = [f"\n{title}", f"  {'component':<16}{'I':>10}{'P':>10}{'B':>10}{'all':>10}  (ms/frame)"]
    tot = {"I": 0.0, "P": 0.0, "B": 0.0, "all": 0.0}
    for c in components:
        row = tab[c]
        for t in tot:
            tot[t] += row[t]
        lines.append(f"  {c:<16}{row['I']:>10.2f}{row['P']:>10.2f}{row['B']:>10.2f}{row['all']:>10.2f}")
    lines.append(f"  {'-'*16}{'-'*40}")
    lines.append(f"  {'TOTAL':<16}{tot['I']:>10.2f}{tot['P']:>10.2f}{tot['B']:>10.2f}{tot['all']:>10.2f}")
    lines.append(f"  frame counts:   I={counts.get('I',0)} P={counts.get('P',0)} B={counts.get('B',0)}")
    return "\n".join(lines), tot


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../sample.mp4")
    ap.add_argument("--start-frame", type=int, default=5000)
    ap.add_argument("--max-frames", type=int, default=48)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--sr", choices=["bicubic", "realesrgan"], default="realesrgan")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="out_profile")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = []

    def emit(s=""):
        print(s)
        log.append(s)

    # ---------------- decode (timed) ----------------
    PROF.reset(enabled=True)
    frames = derisk.decode_lr_and_mvs(args.input, args.start_frame, args.max_frames)
    decode_per = group_per_frame(PROF.events)
    decode_ms = sorted(d.get("decode", 0.0) for d in decode_per.values())
    # median over the window excludes the first frame (it absorbs the seek-to-window cost)
    decode_median = st.median(decode_ms) if decode_ms else 0.0
    types = "".join(f[0] for f in frames)
    counts = {t: types.count(t) for t in ("I", "P", "B")}
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * args.scale, h_lr * args.scale
    emit(f"window: start={args.start_frame} n={len(frames)} ({w_lr}x{h_lr} -> {w_hd}x{h_hd}, x{args.scale})")
    emit(f"types: {types}   (I={counts['I']} P={counts['P']} B={counts['B']})")
    emit(f"decode+convert per frame: median={decode_median:.2f} ms "
         f"(first frame {decode_ms[-1] if decode_ms else 0:.0f} ms incl. seek, excluded)")

    # ---------------- SR cache (timed, built ONCE, reused everywhere) ----------------
    PROF.reset(enabled=True)
    perframe = derisk.build_perframe_cache(frames, w_hd, h_hd, args.sr)
    sr_ms = sorted(d.get("sr", 0.0) for d in group_per_frame(PROF.events).values())
    sr_median = st.median(sr_ms) if sr_ms else 0.0
    emit(f"SR ({args.sr}) per call: median={sr_median:.2f} ms over {len(sr_ms)} frames")

    # ---------------- per-component timing: numpy vs torch ----------------
    def make_fn(backend, occ="full"):
        return lambda: derisk.reconstruct(frames, None, args.scale, True, occ, perframe,
                                           set(), backend=backend)

    emit("\n=== timing reconstruct (warp/mask/blend), median of "
         f"{args.reps} reps after warm-up ===")
    np_pers = run_timed(make_fn("numpy"), args.reps)
    tab_np = avg_tables([table_by_type(p, ALL_COMPONENTS) for p in np_pers], ALL_COMPONENTS)
    torch_pers = run_timed(make_fn("torch"), args.reps)
    tab_t = avg_tables([table_by_type(p, ALL_COMPONENTS) for p in torch_pers], ALL_COMPONENTS)

    # inject decode + sr (measured once) into both tables for a full per-frame picture
    for tab in (tab_np, tab_t):
        tab["decode"] = {t: decode_median for t in ("I", "P", "B", "all")}

    s, tot_np = fmt_table("NUMPY backend (CPU) -- per-frame component cost", tab_np, ALL_COMPONENTS, counts)
    emit(s)
    s, tot_t = fmt_table("TORCH backend (MPS) -- per-frame component cost", tab_t, ALL_COMPONENTS, counts)
    emit(s)

    # ---------------- end-to-end fps: CLEAN deployable wall-clock (single sync) ----------------
    # The per-component table above is timed with per-op mps.synchronize() (honest breakdown,
    # but its torch TOTAL is inflated by sync overhead). The real per-frame cost is a single-sync
    # wall-clock of the deployable path (recon ops + the one HD output download, no metrics).
    import time as _time
    import gpu_ops as G
    pf_dev = {i: G.img_to_dev(perframe[i]) for i in range(len(frames))}  # models on-GPU SR output

    def clean(cache, backend, occ, reps=6, download_output=False):
        """Single-sync wall-clock ms/frame of the recon path. download_output=False keeps the HD
        recon RESIDENT on the GPU (deployable: rendered from a Metal texture); True reads it back
        to CPU each frame (the 'with-I/O' honesty number). Returns (best, median, fwdbwd-fires)."""
        kw = dict(backend=backend, collect_metrics=False, download_output=download_output)
        derisk.reconstruct(frames, None, args.scale, True, occ, cache, set(), **kw)  # warm-up
        if backend == "torch":
            G.sync()
        ts = []
        for _ in range(reps):
            t0 = _time.perf_counter()
            derisk.reconstruct(frames, None, args.scale, True, occ, cache, set(), **kw)
            if backend == "torch":
                G.sync()
            ts.append((_time.perf_counter() - t0) * 1000.0)
        fires = f"{derisk.MASK_FIRES[0]}/{derisk.MASK_FIRES[1]}"   # fwd-bwd fired / total mask calls
        return min(ts) / len(frames), st.median(ts) / len(frames), fires

    n_anchor = max(1, counts["I"])           # reanchor=none -> I-frames are the only anchors
    amort_sr = (n_anchor / len(frames)) * sr_median
    PROF.reset(enabled=False)
    emit(f"\n=== END-TO-END propagation path (single-sync wall-clock, no metrics) ===")
    emit(f"  anchors={n_anchor}/{len(frames)} ({100*n_anchor/len(frames):.1f}%), "
         f"amortized SR = {amort_sr:.2f} ms/frame (median SR {sr_median:.0f}ms x anchor-frac)")
    emit(f"  decode+convert {decode_median:.2f} ms/frame is additive on top of the recon path.")
    emit(f"  DEPLOYABLE = GPU-resident SR output (no perframe upload) + recon kept on GPU (no HD download).")
    emit(f"  with-I/O   = deployable + read the HD recon back to CPU each frame (the one real transfer).")
    emit(f"  {'config':<44}{'best':>8}{'median':>8}{'fps':>7}{'fwdbwd':>8}  ({'<=40ms?':>7})")

    def row(label, cache, backend, occ, dl):
        best, med, fires = clean(cache, backend, occ, download_output=dl)
        emit(f"  {label:<44}{best:>8.2f}{med:>8.2f}{1000/med:>7.1f}{fires:>8}  "
             f"({'YES' if med <= 40 else 'no':>7})")
        return med

    res = {}
    res["np_full"] = row("numpy full mask        (BEFORE, CPU)",    perframe, "numpy", "full", True)
    res["t_full_io_up"] = row("torch full  +upload +download (Step6)", perframe, "torch", "full", True)
    emit(f"  -- DEPLOYABLE (GPU-resident perframe, recon kept on GPU; no transfers) --")
    for occ in ("full", "reactive", "adaptive"):
        res[f"dep_{occ}"] = row(f"torch {occ:<8} mask  [deployable]", pf_dev, "torch", occ, False)
    emit(f"  -- with-I/O (deployable + HD recon download back to CPU) --")
    for occ in ("full", "reactive", "adaptive"):
        res[f"io_{occ}"] = row(f"torch {occ:<8} mask  [with-I/O]", pf_dev, "torch", occ, True)

    emit(f"\n  SPEEDUP torch-vs-numpy (full mask, deployable): "
         f"{res['np_full']/res['dep_full']:.2f}x  ({res['np_full']:.0f} -> {res['dep_full']:.0f} ms/frame)")
    emit(f"  transfers removed (full mask): with-upload+download {res['t_full_io_up']:.1f} -> "
         f"deployable {res['dep_full']:.1f} ms/frame  (saved {res['t_full_io_up']-res['dep_full']:.1f} ms)")
    emit(f"  HD download cost alone (full): deployable {res['dep_full']:.1f} -> with-I/O "
         f"{res['io_full']:.1f} ms/frame (+{res['io_full']-res['dep_full']:.1f} ms)")
    emit(f"  adaptive vs full (deployable): {res['dep_full']:.1f} -> {res['dep_adaptive']:.1f} ms/frame "
         f"({1000/res['dep_adaptive']:.0f} fps); reactive floor {res['dep_reactive']:.1f} ms/frame "
         f"({1000/res['dep_reactive']:.0f} fps)")

    # ---------------- torch-vs-numpy quality (correctness of the fast path) ----------------
    # Reference = numpy full mask. Compare torch full (backend faithfulness) AND torch adaptive
    # (does the Step-7 adaptive mask preserve full-mask quality?). PSNR & fallback delta vs numpy-full.
    emit("\n=== QUALITY vs numpy full mask (recon PSNR + fallback delta) ===")
    rows_np, R_np = derisk.reconstruct(frames, None, args.scale, True, "full", perframe, set(), backend="numpy")
    qrows = []
    for occ in ("full", "adaptive"):
        rows_t, R_t = derisk.reconstruct(frames, None, args.scale, True, occ, perframe, set(), backend="torch")
        psnrs, dhfs = [], []
        for rn, rt in zip(rows_np, rows_t):
            i = rn["frame"]
            p = cv2.PSNR(R_np[i]["recon"], R_t[i]["recon"])
            dhf = (rt["hole_frac"] - rn["hole_frac"]) * 100
            psnrs.append(p); dhfs.append(dhf)
            qrows.append({"variant": f"torch_{occ}", "frame": i, "type": rn["type"],
                          "psnr_vs_numpy_full": round(p, 3), "hole_frac_numpy_full": rn["hole_frac"],
                          "hole_frac_torch": rt["hole_frac"], "d_fallback_pts": round(dhf, 4)})
        emit(f"  torch {occ:<9} vs numpy full: PSNR min={min(psnrs):.2f} mean={np.mean(psnrs):.2f} dB; "
             f"fallback delta mean={np.mean(dhfs):+.3f} max|{max(abs(d) for d in dhfs):.3f}| pts")
    with open(os.path.join(args.out, "quality.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qrows[0].keys()))
        w.writeheader(); w.writerows(qrows)

    # ---------------- mask-variant quality ablation (Task 2/3): full vs reactive vs adaptive ----
    # Quality (fallback% + tOF) is backend-faithful, so use the torch fast path here (cheap). The
    # mask ms cost lives in the deployable wall-clock table above. fwdbwd = how often the adaptive
    # switch paid for the fwd-bwd splat (full always fires, reactive never).
    emit("\n=== MASK-VARIANT quality: full vs reactive vs adaptive (torch recon) ===")
    emit(f"  {'variant':<10}{'fallback% mean/max':>22}{'tOF(prop/LR)':>14}{'tOF(prop/PF)':>14}{'fwdbwd':>9}")
    abl = {}
    for occ in ("full", "reactive", "adaptive"):
        rows, R = derisk.reconstruct(frames, None, args.scale, True, occ, perframe, set(), backend="torch")
        fires = f"{derisk.MASK_FIRES[0]}/{derisk.MASK_FIRES[1]}"
        hf = [r["hole_frac"] for r in rows]
        tofr = derisk._tof_from_R(frames, None, R, w_lr, h_lr)
        abl[occ] = {"fallback_mean": 100*np.mean(hf), "fallback_max": 100*max(hf),
                    "tof_prop_vs_lr": tofr["prop_vs_lr"], "tof_prop_vs_pf": tofr["prop_vs_perframe"]}
        emit(f"  {occ:<10}{abl[occ]['fallback_mean']:>11.2f} /{abl[occ]['fallback_max']:>7.2f}"
             f"{abl[occ]['tof_prop_vs_lr']:>14.4f}{abl[occ]['tof_prop_vs_pf']:>14.4f}{fires:>9}")
    dfb = abl["reactive"]["fallback_mean"] - abl["full"]["fallback_mean"]
    dtof = abl["reactive"]["tof_prop_vs_lr"] - abl["full"]["tof_prop_vs_lr"]
    afb = abl["adaptive"]["fallback_mean"] - abl["full"]["fallback_mean"]
    atof = abl["adaptive"]["tof_prop_vs_lr"] - abl["full"]["tof_prop_vs_lr"]
    emit(f"  reactive vs full: fallback {dfb:+.2f} pts, tOF(prop/LR) {dtof:+.4f}")
    emit(f"  adaptive vs full: fallback {afb:+.2f} pts, tOF(prop/LR) {atof:+.4f}  "
         f"(adaptive should track full where reactive diverges)")

    # ---------------- write components.csv + summary ----------------
    with open(os.path.join(args.out, "components.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["backend", "component", "I_ms", "P_ms", "B_ms", "all_ms"])
        for backend, tab in (("numpy", tab_np), ("torch", tab_t)):
            for c in ALL_COMPONENTS:
                r = tab[c]
                w.writerow([backend, c, round(r["I"], 3), round(r["P"], 3),
                            round(r["B"], 3), round(r["all"], 3)])
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write("\n".join(log) + "\n")
    emit(f"\nartifacts -> {args.out}/ (summary.txt, components.csv, quality.csv)")


if __name__ == "__main__":
    main()
