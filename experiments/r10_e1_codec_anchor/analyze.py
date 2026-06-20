#!/usr/bin/env python3
"""Aggregate R10-E1 results: model x window x crf cells, winners vs x4plus (the GO bar),
fabrication (var-Lap) flags, latency. GO = beats x4plus on LPIPS AND DISTS on real crops
without fabricating. var-Lap is the FAKE-detail flag ONLY, never the verdict (GOTCHA #23)."""
import json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "results.json")))
recs = data["records"]; lat = data.get("latency", {}); arch = data.get("arch", {})

ORDER = ["bicubic","compact","x4plus","realwebphoto_dat2","nomos_atd_jpg","nomos_hatl_otf","nomos_sc"]
MODELS = [m for m in ORDER if any(r["model"]==m for r in recs)]
WINS = ["talkinghead","highmotion","texture18k","texture24k","texture46k"]
CRFS = ["moderate","heavy"]

def cell(win, crf, model, metric):
    vals = [r[metric] for r in recs if r["window"]==win and r["crf"]==crf and r["model"]==model]
    return float(np.mean(vals)) if vals else float("nan")

def overall(model, metric):
    vals = [r[metric] for r in recs if r["model"]==model]
    return float(np.mean(vals)) if vals else float("nan")

print("="*112)
print("OVERALL MEAN (all windows x crf x frames)   GO bar = beat x4plus on LPIPS AND DISTS, no fabrication")
print("="*112)
print(f"{'model':<20}{'arch':<10}{'LPIPS(lo)':>11}{'DISTS(lo)':>11}{'PSNR(hi)':>10}{'varLap':>9}{'lat_ms':>9}")
for m in MODELS:
    a = arch.get(m,"-")[:9]
    lm = lat.get(m,{}).get("median_ms",float("nan"))
    print(f"{m:<20}{a:<10}{overall(m,'lpips'):>11.4f}{overall(m,'dists'):>11.4f}"
          f"{overall(m,'psnr'):>10.2f}{overall(m,'varlap'):>9.0f}{lm:>9.0f}")
gtv = np.mean([r["gt_varlap"] for r in recs])
print(f"{'(GT var-Lap)':<20}{'':<10}{'':>11}{'':>11}{'':>10}{gtv:>9.0f}")

# paired win-rate vs x4plus across all matched (window,frame,crf) cells
print("\n"+"="*112)
print("PAIRED vs x4plus (per window/frame/crf sample): n_better / n_total on each metric (challengers only)")
print("="*112)
keyf = lambda r:(r["window"],r["frame"],r["crf"])
x4 = {keyf(r):r for r in recs if r["model"]=="x4plus"}
for m in MODELS:
    if m in ("bicubic","x4plus"): continue
    nl=dl=nd=dd=tot=0
    for r in recs:
        if r["model"]!=m: continue
        b=x4.get(keyf(r));
        if not b: continue
        tot+=1
        if r["lpips"]<b["lpips"]: nl+=1
        if r["dists"]<b["dists"]: nd+=1
        if r["lpips"]<b["lpips"] and r["dists"]<b["dists"]: dl+=1
    print(f"{m:<20} LPIPS better {nl:>2}/{tot}   DISTS better {nd:>2}/{tot}   BOTH(GO) {dl:>2}/{tot}")

# per-cell tables
for metric, better in [("lpips","lo"),("dists","lo"),("psnr","hi")]:
    print("\n"+"="*112)
    print(f"### {metric.upper()} ({'lower' if better=='lo' else 'higher'} better)  * = beats x4plus, ! = best (excl bicubic)")
    print("="*112)
    print(f"{'win/crf':<20}" + "".join(f"{m[:11]:>12}" for m in MODELS))
    win_count=defaultdict(int); both_count=defaultdict(int)
    for win in WINS:
        for crf in CRFS:
            row={m:cell(win,crf,m,metric) for m in MODELS}
            x4v=row.get("x4plus",float("nan"))
            cand={m:v for m,v in row.items() if m!="bicubic" and not np.isnan(v)}
            best_m=(min if better=="lo" else max)(cand,key=lambda k:cand[k]) if cand else None
            cells=[]
            for m in MODELS:
                v=row[m]; mark=""
                if m not in("bicubic","x4plus") and not np.isnan(v):
                    if (better=="lo" and v<x4v) or (better=="hi" and v>x4v):
                        mark="*"; win_count[m]+=1
                if m==best_m: mark+="!"
                cells.append(f"{v:>10.4f}{mark:<2}" if not np.isnan(v) else f"{'-':>12}")
            print(f"{win[:8]+'/'+crf[:4]:<20}" + "".join(cells))
    if better=="lo" or True:
        print("  beats-x4plus count: " + " ".join(f"{m}={win_count[m]}" for m in MODELS if m not in("bicubic","x4plus")))

# cells where any challenger beats x4plus on BOTH lpips and dists
print("\n"+"="*112)
print("CELLS where a challenger beats x4plus on BOTH LPIPS+DISTS (the GO condition):")
print("="*112)
found=False
for win in WINS:
    for crf in CRFS:
        x4l=cell(win,crf,"x4plus","lpips"); x4d=cell(win,crf,"x4plus","dists")
        for m in MODELS:
            if m in("bicubic","x4plus"): continue
            l=cell(win,crf,m,"lpips"); d=cell(win,crf,m,"dists")
            if l<x4l and d<x4d:
                found=True
                print(f"  {win}/{crf}: {m}  LPIPS {l:.4f}<{x4l:.4f}  DISTS {d:.4f}<{x4d:.4f}  "
                      f"(PSNR {cell(win,crf,m,'psnr'):.2f} vs x4plus {cell(win,crf,'x4plus','psnr'):.2f})")
if not found:
    print("  NONE. No codec-trained challenger beats x4plus on BOTH LPIPS and DISTS in any cell.")
