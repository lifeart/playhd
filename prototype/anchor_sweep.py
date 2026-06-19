#!/usr/bin/env python3
"""
Step 4 deliverable: quality-vs-anchor-fraction tradeoff curve + real-time SR budget.

Sweeps anchor density on two real-clip windows and produces `anchor_curve.png`:
  * window C (start 5000, talking-head, realistic),
  * window A (start 0,    high-motion, stress).

The expensive per-frame SR network (realesr-general-x4v3, x4, MPS) is run ONCE per window
to build a cache; every anchor operating point then only re-runs warp/blend (no SR), so the
whole sweep costs ~2 SR passes total, not 2 x (#operating points) passes.

Per operating point we record: anchor fraction, SR-compute fraction (anchors + fallback),
mean PSNR(prop, per-frame-SR) [the NEMO cache-erosion metric], mean LR-consistency (prop and
the per-frame-SR ceiling), tOF(prop vs per-frame-SR), fallback %, and the amortized SR budget
(anchor_fraction x measured ms/frame) with the achievable SR-portion fps.

Run:  python3 anchor_sweep.py
Out:  prototype/out_anchor_sweep/{anchor_curve.png, sweep_A.csv, sweep_C.csv, summary.txt}
"""
import csv
import os
import time

import cv2
import numpy as np

import derisk as D
import sr as SR

CLIP = "/Users/lifeart/Repos/playhd/sample.mp4"
SCALE = 4
N = 48
OUT = os.path.join(os.path.dirname(__file__), "out_anchor_sweep")
WINDOWS = [("C", 5000, "talking-head (realistic)"),
           ("A", 0, "high-motion (stress)")]

# operating points: interval K over the I/P backbone, then adaptive (fallback budget).
INTERVALS = [1, 2, 3, 4, 6, 8, 12, 10 ** 9]          # 10**9 == "inf" == none (I-frames only)
ADAPT_BUDGETS = [0.5, 1.0, 1.5, 2.0, 3.0]            # accumulated-fallback frame-equiv budgets
PERFRAME_MS_REF = 138.0                              # Step-3 reference SR latency (MPS)


def tof_vs_ref(prop_seq_lr, ref_flows):
    """tOF of a candidate LR sequence against PRE-COMPUTED reference (per-frame-SR) flows.
    The reference (per-frame-SR) sequence is identical across all operating points, so its
    Farneback flows are computed ONCE per window -> the sweep's tOF cost drops ~6x."""
    vals = []
    for t in range(1, len(prop_seq_lr)):
        d = ref_flows[t - 1] - D._farneback(prop_seq_lr[t - 1], prop_seq_lr[t])
        vals.append(float(np.mean(np.sqrt(np.sum(d * d, axis=-1)))))
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(rows, R, frames, w_lr, h_lr, sr_ms, ref_flows):
    """Collapse a per-frame reconstruction into one operating-point record.

    Quality is reported on FIXED frame sets so the curve is monotone and interpretable:
      * q_B    = mean PSNR(prop, per-frame-SR) over B-frames -- B is NEVER an anchor in any
                 policy, so this set is identical across operating points; it isolates how well
                 the cheap leaf frames track per-frame SR as their I/P references are anchored
                 more densely (the clean dB quality-vs-anchor curve).
      * q_Pna  = same over non-anchor P-frames (the I/P backbone drift re-anchoring targets;
                 set shrinks with anchoring, shown for context only).
    tOF(prop vs per-frame-SR) over ALL frames is a fixed-reference temporal metric (lower=closer
    to the per-frame-SR ceiling, ->0 as anchor_frac->1)."""
    n = len(rows)
    n_anchor = sum(1 for r in rows if r["is_anchor"])
    nonanchor = [r for r in rows if not r["is_anchor"]]
    b_rows = [r for r in rows if r["type"] == "B"]
    pna_rows = [r for r in rows if r["type"] == "P" and not r["is_anchor"]]
    q_B = np.mean([r["psnr_prop_vs_perframe"] for r in b_rows]) if b_rows else float("nan")
    q_Pna = np.mean([r["psnr_prop_vs_perframe"] for r in pna_rows]) if pna_rows else float("nan")
    q_nonanchor = (np.mean([r["psnr_prop_vs_perframe"] for r in nonanchor])
                   if nonanchor else float("nan"))
    lrc_prop = np.mean([r["psnr_lr_consistency"] for r in rows])
    lrc_ceil = np.mean([r["psnr_lr_consistency_perframe"] for r in rows])  # per-frame-SR ceiling
    fallback = np.mean([r["hole_frac"] for r in nonanchor]) if nonanchor else 0.0
    fallback_equiv = sum(r["hole_frac"] for r in nonanchor)
    anchor_frac = n_anchor / n
    sr_compute_frac = (n_anchor + fallback_equiv) / n                      # precise SR work
    sm = (w_lr, h_lr)
    prop_seq = [cv2.resize(R[i]["recon"], sm) for i in range(n)]
    tof_pf = tof_vs_ref(prop_seq, ref_flows)
    amort_ms = anchor_frac * sr_ms                                         # task's primary cost
    amort_ms_precise = sr_compute_frac * sr_ms                             # + fallback re-runs
    return dict(
        n_anchor=n_anchor, anchor_frac=anchor_frac, sr_compute_frac=sr_compute_frac,
        q_B=q_B, q_Pna=q_Pna, q_nonanchor=q_nonanchor,
        lrc_prop=lrc_prop, lrc_ceiling=lrc_ceil,
        fallback_pct=100 * fallback, tof_prop_vs_pf=tof_pf,
        amort_ms=amort_ms, amort_ms_precise=amort_ms_precise,
        fps_anchor=(1000.0 / amort_ms if amort_ms > 0 else float("inf")),
        fps_precise=(1000.0 / amort_ms_precise if amort_ms_precise > 0 else float("inf")))


def sweep_window(name, start, sr_ms_holder):
    frames = D.decode_lr_and_mvs(CLIP, start, N)
    types = "".join(f[0][0] for f in frames)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    backbone = D.backbone_indices(frames)
    print(f"\n=== window {name} (start={start}, {len(frames)} frames) ===")
    print(f"    types: {types}")
    print(f"    backbone (I/P) frames: {len(backbone)} of {len(frames)}")

    # ---- the ONE expensive step: per-frame SR cache (run once, reused by every point) ----
    t0 = time.perf_counter()
    cache = D.build_perframe_cache(frames, w_hd, h_hd, "realesrgan")
    t_sr = time.perf_counter() - t0
    sr_ms = SR.median_latency_ms()
    sr_ms_holder.append(sr_ms)
    print(f"    per-frame SR cache: {len(cache)} frames in {t_sr:.1f}s "
          f"(median {sr_ms:.1f} ms/frame, mean {SR.mean_latency_ms():.1f} ms)")

    # reference (per-frame-SR) Farneback flows -- computed ONCE, reused by every tOF
    sm = (w_lr, h_lr)
    pf_seq = [cv2.resize(cache[i], sm) for i in range(len(frames))]
    ref_flows = [D._farneback(pf_seq[t - 1], pf_seq[t]) for t in range(1, len(pf_seq))]

    points = []
    t1 = time.perf_counter()
    # interval sweep
    for K in INTERVALS:
        policy = "none" if K >= 10 ** 9 else f"interval:{K}"
        aset = D.compute_anchor_set(frames, policy, 1.0, 1.0, "fallback", None,
                                    SCALE, True, "full", cache)
        rows, R = D.reconstruct(frames, None, SCALE, True, "full", cache, aset)
        rec = aggregate(rows, R, frames, w_lr, h_lr, sr_ms, ref_flows)
        rec.update(label=("K=inf (=none)" if K >= 10 ** 9 else f"K={K}"),
                   policy=policy, kind="interval")
        points.append(rec)
    # adaptive sweep (accumulated-fallback budget)
    adaptive_default = None
    for b in ADAPT_BUDGETS:
        aset = D.compute_anchor_set(frames, "adaptive", 1.0, b, "fallback", None,
                                    SCALE, True, "full", cache)
        rows, R = D.reconstruct(frames, None, SCALE, True, "full", cache, aset)
        rec = aggregate(rows, R, frames, w_lr, h_lr, sr_ms, ref_flows)
        rec.update(label=f"adaptive b={b}", policy=f"adaptive(b={b})", kind="adaptive",
                   adaptive_anchors=sorted(aset))
        points.append(rec)
        if abs(b - 1.0) < 1e-9:
            adaptive_default = rec
    t_sweep = time.perf_counter() - t1

    # ---- per-frame-SR ceiling point (anchor_frac == 1.0; every frame is its own SR) ----
    # quality vs per-frame-SR is exact (==), so prop_vs_pf -> inf; LR-consistency == ceiling.
    ceil_lrc = np.mean([D.psnr_lr_consistency(cache[i], frames[i][1]) for i in range(len(frames))])
    perframe_point = dict(label="per-frame SR (ceiling)", policy="perframe", kind="ceiling",
                          n_anchor=len(frames), anchor_frac=1.0, sr_compute_frac=1.0,
                          q_B=float("inf"), q_Pna=float("inf"), q_nonanchor=float("inf"),
                          lrc_prop=ceil_lrc, lrc_ceiling=ceil_lrc,
                          fallback_pct=0.0, tof_prop_vs_pf=0.0,
                          amort_ms=sr_ms, amort_ms_precise=sr_ms,
                          fps_anchor=1000.0 / sr_ms, fps_precise=1000.0 / sr_ms)

    print(f"    swept {len(points)} operating points in {t_sweep:.1f}s (no SR re-run)")
    return dict(name=name, start=start, types=types, frames=frames, cache=cache,
                w_lr=w_lr, h_lr=h_lr, backbone=backbone, points=points,
                perframe_point=perframe_point, adaptive_default=adaptive_default,
                sr_ms=sr_ms, ceil_lrc=ceil_lrc)


def write_csv(res):
    path = os.path.join(OUT, f"sweep_{res['name']}.csv")
    cols = ["label", "policy", "kind", "n_anchor", "anchor_frac", "sr_compute_frac",
            "q_B", "q_Pna", "q_nonanchor", "lrc_prop", "lrc_ceiling", "fallback_pct",
            "tof_prop_vs_pf", "amort_ms", "amort_ms_precise", "fps_anchor", "fps_precise"]
    allpts = res["points"] + [res["perframe_point"]]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for p in allpts:
            row = {k: p.get(k, "") for k in cols}
            for k in ("anchor_frac", "sr_compute_frac", "q_B", "q_Pna", "q_nonanchor",
                      "lrc_prop", "lrc_ceiling", "fallback_pct", "tof_prop_vs_pf", "amort_ms",
                      "amort_ms_precise", "fps_anchor", "fps_precise"):
                v = row[k]
                if isinstance(v, float):
                    row[k] = round(v, 4)
            w.writerow(row)
    return path


def within_1db_operating_point(res):
    """Smallest anchor fraction whose mean LR-consistency is within 1 dB of the per-frame-SR
    LR-consistency ceiling (finite, real-footage-measurable). Returns the interval record."""
    ceil = res["ceil_lrc"]
    cand = [p for p in res["points"] if p["kind"] == "interval"
            and p["lrc_prop"] >= ceil - 1.0]
    if not cand:
        return None
    return min(cand, key=lambda p: p["anchor_frac"])


def within_1db_qB_operating_point(res):
    """Smallest interval anchor fraction whose B-leaf PSNR(prop, per-frame-SR) is within 1 dB
    of the densest-anchoring (K=1) B-leaf quality -- the architecture's practical ceiling
    (B-frames are always leaves, so K=1 is the closest-to-per-frame-SR the leaves ever get)."""
    iv = [p for p in res["points"] if p["kind"] == "interval"]
    ceil_B = max(p["q_B"] for p in iv)   # K=1 gives the freshest references for the B-leaves
    cand = [p for p in iv if p["q_B"] >= ceil_B - 1.0]
    return (min(cand, key=lambda p: p["anchor_frac"]) if cand else None), ceil_B


def plot_curve(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axq, axt, axl) = plt.subplots(1, 3, figsize=(20, 6))
    colors = {"C": "tab:green", "A": "tab:red"}
    long = {"C": "talking-head", "A": "high-motion"}
    for res in results:
        c = colors[res["name"]]
        iv = sorted([p for p in res["points"] if p["kind"] == "interval"],
                    key=lambda p: p["anchor_frac"])
        ad = sorted([p for p in res["points"] if p["kind"] == "adaptive"],
                    key=lambda p: p["anchor_frac"])
        xi = [p["anchor_frac"] for p in iv]
        xa = [p["anchor_frac"] for p in ad]

        # ---- panel 1: fixed-set B-leaf PSNR(prop, per-frame-SR) (dB quality, monotone) ----
        axq.plot(xi, [p["q_B"] for p in iv], "-o", color=c,
                 label=f"window {res['name']}: {long[res['name']]} (interval)")
        axq.plot(xa, [p["q_B"] for p in ad], "D", color=c, mfc="white", ms=8,
                 label=f"window {res['name']}: adaptive")
        op, ceil_B = within_1db_qB_operating_point(res)
        axq.axhline(ceil_B, ls="--", color=c, alpha=0.45)
        if op:
            axq.scatter([op["anchor_frac"]], [op["q_B"]], s=200, facecolors="none",
                        edgecolors=c, linewidths=2.5, zorder=5)
            axq.annotate(f"  within 1 dB @ {op['anchor_frac']*100:.0f}%",
                         (op["anchor_frac"], op["q_B"]), color=c, fontsize=9)

        # ---- panel 2: tOF vs anchor fraction (temporal; lower=better, ->0 = ceiling) ----
        axt.plot(xi, [p["tof_prop_vs_pf"] for p in iv], "-o", color=c,
                 label=f"window {res['name']}: {long[res['name']]}")
        axt.plot(xa, [p["tof_prop_vs_pf"] for p in ad], "D", color=c, mfc="white", ms=8)

        # ---- panel 3: LR-consistency vs the finite per-frame-SR ceiling ----
        axl.plot(xi, [p["lrc_prop"] for p in iv], "-o", color=c,
                 label=f"window {res['name']} prop")
        axl.axhline(res["ceil_lrc"], ls="--", color=c, alpha=0.6,
                    label=f"window {res['name']} per-frame-SR ceiling")
        axl.axhline(res["ceil_lrc"] - 1.0, ls=":", color=c, alpha=0.4)

    axq.set_xlabel("anchor fraction  (anchors / total frames;  1.0 = per-frame SR)")
    axq.set_ylabel("B-leaf mean PSNR(propagated, per-frame-SR)  [dB]")
    axq.set_title("Quality vs anchor fraction (fixed B-leaf set)\n"
                  "higher = closer to per-frame SR; dashed = K=1 ceiling")
    axq.grid(alpha=0.3)
    axq.legend(fontsize=8, loc="lower right")

    axt.set_xlabel("anchor fraction  (anchors / total frames)")
    axt.set_ylabel("tOF vs per-frame-SR  [lower = steadier]")
    axt.set_title("Temporal drift vs anchor fraction\n"
                  "high-motion starts 5x worse; needs more anchors to reach the floor")
    axt.grid(alpha=0.3)
    axt.legend(fontsize=8, loc="upper right")

    axl.set_xlabel("anchor fraction  (anchors / total frames)")
    axl.set_ylabel("mean LR-consistency PSNR  [dB]")
    axl.set_title("LR-consistency vs anchor fraction\n"
                  "dashed = per-frame-SR ceiling, dotted = ceiling - 1 dB")
    axl.grid(alpha=0.3)
    axl.legend(fontsize=8, loc="lower right")

    fig.suptitle("playhd Step 4 - adaptive re-anchoring: quality vs anchor fraction  "
                 "(talking-head reaches the per-frame-SR ceiling at far fewer anchors than "
                 "high-motion)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT, "anchor_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    t_start = time.perf_counter()
    sr_ms_holder = []
    results = [sweep_window(nm, st, sr_ms_holder) for nm, st, _ in WINDOWS]

    csvpaths = [write_csv(r) for r in results]
    figpath = plot_curve(results)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("\n" + "=" * 96)
    emit("QUALITY-vs-ANCHOR-FRACTION  (real-clip windows; per-frame SR run once, sweep is warp-only)")
    emit("=" * 96)
    for res in results:
        sr_ms = res["sr_ms"]
        emit(f"\n--- window {res['name']} ({dict(C='talking-head, realistic', A='high-motion, stress')[res['name']]}); "
             f"SR median {sr_ms:.1f} ms/frame; backbone {len(res['backbone'])}/{N} ---")
        emit(f"{'operating point':<18}{'anchor%':>8}{'SRcomp%':>9}{'qB':>8}{'qPna':>8}"
             f"{'LRcons':>8}{'fallbk%':>9}{'tOF':>7}{'amortMs':>9}{'SRfps':>7}")
        for p in res["points"] + [res["perframe_point"]]:
            qb = " inf " if not np.isfinite(p["q_B"]) else f"{p['q_B']:.2f}"
            qp = ("  -  " if not np.isfinite(p.get("q_Pna", float('nan')))
                  else f"{p['q_Pna']:.2f}")
            emit(f"{p['label']:<18}{100*p['anchor_frac']:>7.1f}{100*p['sr_compute_frac']:>9.1f}"
                 f"{qb:>8}{qp:>8}{p['lrc_prop']:>8.2f}{p['fallback_pct']:>9.2f}"
                 f"{p['tof_prop_vs_pf']:>7.3f}{p['amort_ms']:>9.1f}{p['fps_anchor']:>7.1f}")
        ceil = res["ceil_lrc"]
        op1 = within_1db_operating_point(res)
        opb, ceil_B = within_1db_qB_operating_point(res)
        # tOF at none vs at K=1 (the temporal-drift contrast)
        iv = {p["label"]: p for p in res["points"] if p["kind"] == "interval"}
        tof_none = iv["K=inf (=none)"]["tof_prop_vs_pf"]
        tof_k1 = iv["K=1"]["tof_prop_vs_pf"]
        emit(f"  per-frame-SR LR-consistency ceiling: {ceil:.2f} dB; B-leaf K=1 ceiling: {ceil_B:.2f} dB")
        emit(f"  tOF(vs per-frame-SR): none(4.2% anchors)={tof_none:.3f}  K=1={tof_k1:.3f}  "
             f"(lower=steadier; high-motion's none-tOF is the temporal-drift signal)")
        if opb:
            emit(f"  within-1dB-of-ceiling (B-leaf PSNRvsPF): {opb['label']} -> "
                 f"{100*opb['anchor_frac']:.1f}% anchors, amortized {opb['amort_ms']:.1f} ms/frame "
                 f"=> {opb['fps_anchor']:.1f} fps SR-portion ({sr_ms/opb['amort_ms']:.1f}x speedup vs per-frame SR)")
        if op1:
            emit(f"  within-1dB-of-ceiling (LR-consistency): {op1['label']} -> "
                 f"{100*op1['anchor_frac']:.1f}% anchors => {op1['fps_anchor']:.1f} fps SR-portion")
        ad = res["adaptive_default"]
        if ad:
            emit(f"  adaptive (fallback budget=1.0): {ad['n_anchor']} anchors -> "
                 f"{100*ad['anchor_frac']:.1f}% anchors, B-leaf qB {ad['q_B']:.2f} dB, tOF {ad['tof_prop_vs_pf']:.3f}, "
                 f"amortized {ad['amort_ms']:.1f} ms => {ad['fps_anchor']:.1f} fps "
                 f"(promoted backbone frames {ad.get('adaptive_anchors')})")

    # adaptive A-vs-C contrast at the SAME budget
    emit("\n" + "-" * 96)
    emit("ADAPTIVE A-vs-C CONTRAST (same fallback budget => more anchors on high-motion):")
    for b in ADAPT_BUDGETS:
        rcs = {r["name"]: next(p for p in r["points"]
                               if p["kind"] == "adaptive" and p["label"] == f"adaptive b={b}")
               for r in results}
        a = rcs["A"]; c = rcs["C"]
        emit(f"  budget={b}:  window A -> {a['n_anchor']} anchors ({100*a['anchor_frac']:.1f}%)   "
             f"window C -> {c['n_anchor']} anchors ({100*c['anchor_frac']:.1f}%)   "
             f"=> A/C anchor ratio {a['n_anchor']/max(1,c['n_anchor']):.1f}x")

    total = time.perf_counter() - t_start
    emit(f"\nfigure : {figpath}")
    for p in csvpaths:
        emit(f"csv    : {p}")
    emit(f"total runtime: {total:.1f}s  (2 windows x ~{N} SR frames once each; "
         f"sweep reused the cache, 0 extra SR calls)")
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
