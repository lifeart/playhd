#!/usr/bin/env python3
"""Aggregate R9-E2 results: model x window x crf cells, winners vs x4plus, fabrication flags."""
import json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
recs = json.load(open(os.path.join(HERE, "results.json")))

MODELS = ["bicubic","compact","x4plus","ultrasharp","wdn-dni0.5","wdn-dni0.0","nomos"]
MODELS = [m for m in MODELS if any(r["model"]==m for r in recs)]
WINS = ["talkinghead","highmotion","texture18k","texture24k","texture46k"]
CRFS = ["moderate","heavy"]

def cell(win, crf, model, metric):
    vals = [r[metric] for r in recs if r["window"]==win and r["crf"]==crf and r["model"]==model]
    return float(np.mean(vals)) if vals else float("nan")

print("="*100)
print("LPIPS (lower=better)  cells = mean over frames; bold-equivalent marked * if beats x4plus, ! if best overall")
print("="*100)
for metric, better in [("lpips","lo"),("dists","lo"),("psnr","hi")]:
    print(f"\n### {metric.upper()} ({'lower' if better=='lo' else 'higher'} better)")
    hdr = f"{'win/crf':<22}" + "".join(f"{m:>12}" for m in MODELS)
    print(hdr)
    win_count = defaultdict(int)   # model -> #cells beating x4plus
    best_count = defaultdict(int)
    for win in WINS:
        for crf in CRFS:
            row = {m: cell(win,crf,m,metric) for m in MODELS}
            x4 = row.get("x4plus", float("nan"))
            # best among real SR models (exclude bicubic floor)
            cand = {m:v for m,v in row.items() if m!="bicubic"}
            best_m = (min if better=="lo" else max)(cand, key=lambda k: cand[k])
            cells=[]
            for m in MODELS:
                v=row[m]; mark=""
                if m!="bicubic" and m!="x4plus":
                    if (better=="lo" and v<x4) or (better=="hi" and v>x4):
                        mark="*"; win_count[m]+=1
                if m==best_m: mark+="!"; best_count[m]+=1
                cells.append(f"{v:>10.4f}{mark:<2}" if metric!="psnr" else f"{v:>9.2f}{mark:<3}")
            print(f"{win+'/'+crf:<22}" + "".join(cells))
    print(f"  [{metric}] #cells BEATING x4plus: " + ", ".join(f"{m}={win_count[m]}" for m in MODELS if m not in('bicubic','x4plus')))
    print(f"  [{metric}] #cells BEST(non-bicubic): " + ", ".join(f"{m}={best_count[m]}" for m in MODELS if m!='bicubic'))

# Fabrication cross-check: per model per window, does var-Lap rank match LPIPS rank?
print("\n" + "="*100)
print("FABRICATION CROSS-CHECK  (var-Lap is NR/fake-flag; the test: high var-Lap WITHOUT LPIPS+DISTS win = fabrication)")
print("="*100)
print(f"{'window':<14}{'model':<13}{'varlap':>9}{'gt_vl':>9}{'LPIPS':>9}{'DISTS':>9}{'PSNR':>8}  verdict")
for win in WINS:
    gtvl = np.mean([r["gt_varlap"] for r in recs if r["window"]==win])
    x4_lp = cell(win,"moderate","x4plus","lpips")
    x4_di = cell(win,"moderate","x4plus","dists")
    for m in MODELS:
        if m=="bicubic": continue
        vl = np.mean([r["varlap"] for r in recs if r["window"]==win and r["model"]==m])
        lp = cell(win,"moderate",m,"lpips"); di=cell(win,"moderate",m,"dists"); ps=cell(win,"moderate",m,"psnr")
        verdict=""
        if m!="x4plus":
            sharper = vl > 1.15*np.mean([r["varlap"] for r in recs if r["window"]==win and r["model"]=="x4plus"])
            better_real = (lp < x4_lp) and (di < x4_di)
            if sharper and not better_real:
                verdict="FABRICATES (sharper, worse real)"
            elif sharper and better_real:
                verdict="real gain"
            elif better_real:
                verdict="real gain (not sharper)"
        print(f"{win:<14}{m:<13}{vl:>9.0f}{gtvl:>9.0f}{lp:>9.4f}{di:>9.4f}{ps:>8.2f}  {verdict}")

# Overall mean per model
print("\n" + "="*100)
print("OVERALL MEAN (all windows x crf x frames)")
print("="*100)
print(f"{'model':<13}{'LPIPS':>9}{'DISTS':>9}{'PSNR':>8}{'varlap':>9}")
for m in MODELS:
    lp=np.mean([r["lpips"] for r in recs if r["model"]==m])
    di=np.mean([r["dists"] for r in recs if r["model"]==m])
    ps=np.mean([r["psnr"]  for r in recs if r["model"]==m])
    vl=np.mean([r["varlap"]for r in recs if r["model"]==m])
    print(f"{m:<13}{lp:>9.4f}{di:>9.4f}{ps:>8.2f}{vl:>9.0f}")
