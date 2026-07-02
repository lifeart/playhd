#!/usr/bin/env python3
"""Aggregate R12-E3: codec-trained challengers (VCISR, avc_compact) vs x4plus on REAL H.264,
with the VMAF-NEG hallucination guardrail. GO bar (from R10/R11) = beat x4plus on LPIPS AND
DISTS on real crops WITHOUT fabricating. VMAF-NEG is a guardrail column ONLY: a LPIPS/DISTS
"win" paired with a LOW VMAF-NEG (vs x4plus) is a hallucination red flag, not a real win."""
import json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "results.json")))
recs = d["records"]; vneg = d.get("vmafneg", []); lat = d.get("latency", {}); meta = d.get("meta", {})

ORDER = ["bicubic", "compact", "x4plus", "span", "vcisr", "avc_compact"]
MODELS = [m for m in ORDER if any(r["model"] == m for r in recs)]
WINS = ["talkinghead", "highmotion", "texture18k", "texture24k"]
CRFS = ["moderate", "heavy"]
CLASSES = ["face", "motion", "texture"]

negmap = {(v["window"], v["crf"], v["model"]): v["vmaf_neg"] for v in vneg}

def mean(vals): return float(np.mean(vals)) if vals else float("nan")
def overall(model, metric): return mean([r[metric] for r in recs if r["model"] == model])
def by_class(model, cls, metric): return mean([r[metric] for r in recs if r["model"] == model and r["cls"] == cls])
def neg_overall(model): return mean([v for (w,c,m),v in negmap.items() if m == model and not np.isnan(v)])
def neg_class(model, cls):
    ws = [w for w in WINS if any(r["cls"]==cls and r["window"]==w for r in recs)]
    return mean([v for (w,c,m),v in negmap.items() if m==model and w in ws and not np.isnan(v)])

archname = lambda m: (meta.get(m, ["-"])[0] if isinstance(meta.get(m), (list, tuple)) else "-")

print("="*104)
print("R12-E3  CODEC-TRAINED vs x4plus on REAL H.264  (sample.mp4, 4 windows x 3f x CRF{27,35}, net-2x)")
print("GO bar = beat x4plus on LPIPS AND DISTS w/o fabrication.  VMAF-NEG = anti-hallucination GUARDRAIL only")
print("="*104)
print(f"{'model':<13}{'arch':<16}{'LPIPS(lo)':>10}{'DISTS(lo)':>10}{'PSNR(hi)':>9}{'VMAFneg(hi)':>12}{'varLap':>8}{'lat_ms':>8}")
for m in MODELS:
    lm = lat.get(m, {}).get("median_ms", float("nan"))
    print(f"{m:<13}{archname(m)[:15]:<16}{overall(m,'lpips'):>10.4f}{overall(m,'dists'):>10.4f}"
          f"{overall(m,'psnr'):>9.2f}{neg_overall(m):>12.2f}{overall(m,'varlap'):>8.0f}{lm:>8.0f}")
gtv = mean([r["gt_varlap"] for r in recs])
print(f"{'(GT varLap)':<13}{'':<16}{'':>10}{'':>10}{'':>9}{'':>12}{gtv:>8.0f}")

# paired win-rate vs x4plus
print("\n" + "="*104)
print("PAIRED vs x4plus  (per window/frame/crf sample; challengers only)  BOTH = the GO condition")
print("="*104)
keyf = lambda r: (r["window"], r["frame"], r["crf"])
x4 = {keyf(r): r for r in recs if r["model"] == "x4plus"}
for m in MODELS:
    if m in ("bicubic", "x4plus"): continue
    nl = nd = both = tot = 0
    for r in recs:
        if r["model"] != m: continue
        b = x4.get(keyf(r))
        if not b: continue
        tot += 1
        if r["lpips"] < b["lpips"]: nl += 1
        if r["dists"] < b["dists"]: nd += 1
        if r["lpips"] < b["lpips"] and r["dists"] < b["dists"]: both += 1
    print(f"{m:<13} LPIPS better {nl:>2}/{tot}   DISTS better {nd:>2}/{tot}   BOTH(GO) {both:>2}/{tot}")

# per-class (face / motion / texture) -- the team content is faces/live-action
for metric, hi in [("lpips", False), ("dists", False), ("vmaf_neg", True)]:
    print("\n" + "="*104)
    lbl = "VMAF-NEG (higher=better, GUARDRAIL)" if metric == "vmaf_neg" else f"{metric.upper()} (lower better)"
    print(f"### PER-CLASS {lbl}   * = beats x4plus")
    print("="*104)
    print(f"{'class':<12}" + "".join(f"{m[:11]:>12}" for m in MODELS))
    for cls in CLASSES:
        if not any(r["cls"] == cls for r in recs): continue
        getv = (lambda m: neg_class(m, cls)) if metric == "vmaf_neg" else (lambda m: by_class(m, cls, metric))
        x4v = getv("x4plus")
        cells = []
        for m in MODELS:
            v = getv(m); mark = ""
            if m not in ("bicubic", "x4plus") and not np.isnan(v):
                if (hi and v > x4v) or (not hi and v < x4v): mark = "*"
            cells.append(f"{v:>10.4f}{mark:<2}" if not np.isnan(v) else f"{'-':>12}")
        print(f"{cls:<12}" + "".join(cells))

# hallucination guardrail: LPIPS/DISTS "wins" that VMAF-NEG contradicts
print("\n" + "="*104)
print("HALLUCINATION GUARDRAIL: challenger beats x4plus on LPIPS or DISTS, but VMAF-NEG says WORSE")
print("(a LPIPS/DISTS 'win' with lower VMAF-NEG than x4plus = likely fabricated detail, not real gain)")
print("="*104)
flagged = 0
for (win, crf, m), nv in sorted(negmap.items()):
    if m in ("bicubic", "x4plus"): continue
    x4l = mean([r["lpips"] for r in recs if r["window"]==win and r["crf"]==crf and r["model"]=="x4plus"])
    x4d = mean([r["dists"] for r in recs if r["window"]==win and r["crf"]==crf and r["model"]=="x4plus"])
    ml  = mean([r["lpips"] for r in recs if r["window"]==win and r["crf"]==crf and r["model"]==m])
    md_ = mean([r["dists"] for r in recs if r["window"]==win and r["crf"]==crf and r["model"]==m])
    x4n = negmap.get((win, crf, "x4plus"), float("nan"))
    perc_win = (ml < x4l) or (md_ < x4d)
    if perc_win and not np.isnan(nv) and not np.isnan(x4n) and nv < x4n:
        flagged += 1
        which = []
        if ml < x4l: which.append(f"LPIPS {ml:.4f}<{x4l:.4f}")
        if md_ < x4d: which.append(f"DISTS {md_:.4f}<{x4d:.4f}")
        print(f"  {win}/{crf} {m}: {' & '.join(which)}  BUT VMAF-NEG {nv:.1f} < x4plus {x4n:.1f}  <- guardrail flag")
if not flagged:
    print("  none: no challenger's LPIPS/DISTS 'win' is contradicted by a lower VMAF-NEG.")

# GO cells (beat x4plus on BOTH lpips+dists)
print("\n" + "="*104)
print("GO CELLS: challenger beats x4plus on BOTH LPIPS+DISTS (with its VMAF-NEG vs x4plus for context)")
print("="*104)
found = False
def cell(win, crf, m, metric): return mean([r[metric] for r in recs if r["window"]==win and r["crf"]==crf and r["model"]==m])
for win in WINS:
    for crf in CRFS:
        x4l = cell(win, crf, "x4plus", "lpips"); x4d = cell(win, crf, "x4plus", "dists")
        x4n = negmap.get((win, crf, "x4plus"), float("nan"))
        for m in MODELS:
            if m in ("bicubic", "x4plus"): continue
            l = cell(win, crf, m, "lpips"); dd = cell(win, crf, m, "dists")
            if l < x4l and dd < x4d:
                found = True
                nv = negmap.get((win, crf, m), float("nan"))
                tag = "NEG-clean" if (not np.isnan(nv) and not np.isnan(x4n) and nv >= x4n) else "NEG-FLAG"
                print(f"  {win}/{crf}: {m}  LPIPS {l:.4f}<{x4l:.4f}  DISTS {dd:.4f}<{x4d:.4f}  "
                      f"VMAFneg {nv:.1f} vs x4 {x4n:.1f} [{tag}]")
if not found:
    print("  NONE. No codec-trained challenger beats x4plus on BOTH LPIPS and DISTS in any cell.")
