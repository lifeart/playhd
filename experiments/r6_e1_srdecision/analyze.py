#!/usr/bin/env python3
"""R6-E1 analysis: compact-vs-x4plus head-to-head on TRUE LPIPS (lead metric),
per-frame win rate, content-class aggregates, perception-distortion split, and a
visual crop on a textured window. Reads results.json (written by run_matrix.py)."""
import os, sys, json
import numpy as np, cv2, av

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
R = json.load(open(os.path.join(_HERE, "results.json")))
res, W, D = R["results"], R["windows"], R["degrades"]
SMOOTH = ["talkinghead", "highmotion"]          # low var-Lap content
TEXTURED = ["texture18k", "texture24k", "texture46k"]

def cell(w, d, m): return res[w][d][m]

print("="*92)
print("HEAD-TO-HEAD  compact vs x4plus  (TRUE AlexNet LPIPS, lower=better) | winner & margin")
print("="*92)
print(f"{'window':12s} {'degrade':8s} {'compact':>8s} {'x4plus':>8s} "
      f"{'winner':>7s} {'x4 Δ':>8s} {'x4 winrate':>11s}")
ncells = {"compact": 0, "x4plus": 0, "tie": 0}
for w in W:
    for d in D:
        c, x = cell(w, d, "compact"), cell(w, d, "x4plus")
        cl, xl = c["lpips"], x["lpips"]
        # per-frame x4plus win rate (frames where x4plus LPIPS strictly lower)
        x4wins = sum(1 for xp, cp in zip(x["lpips_per"], c["lpips_per"]) if xp < cp)
        n = len(x["lpips_per"]); wr = x4wins / n
        margin = cl - xl                      # >0 => x4plus better
        if abs(margin) < 0.003: win = "~tie"; ncells["tie"] += 1
        elif margin > 0:        win = "x4plus"; ncells["x4plus"] += 1
        else:                   win = "compact"; ncells["compact"] += 1
        print(f"{w:12s} {d:8s} {cl:8.4f} {xl:8.4f} {win:>7s} {margin:+8.4f} {wr*100:9.0f}%")
print(f"\ncell winners (15): x4plus={ncells['x4plus']}  compact={ncells['compact']}  tie={ncells['tie']}")

print("\n" + "="*92)
print("CONTENT-CLASS x DEGRADE  mean LPIPS (compact | x4plus | Δ=compact-x4plus, +=x4plus better)")
print("="*92)
for cls, names in [("SMOOTH(face/intro)", SMOOTH), ("TEXTURED(detail)", TEXTURED)]:
    for d in D:
        cs = np.mean([cell(w, d, "compact")["lpips"] for w in names])
        xs = np.mean([cell(w, d, "x4plus")["lpips"] for w in names])
        print(f"  {cls:20s} {d:8s}  compact={cs:.4f}  x4plus={xs:.4f}  Δ={cs-xs:+.4f}")

print("\n" + "="*92)
print("PERCEPTION-DISTORTION SPLIT (textured windows): x4plus often LOSES PSNR but WINS LPIPS")
print("="*92)
print(f"{'window':12s} {'degrade':8s} {'PSNR c/x':>14s} {'LPIPS c/x':>16s}  note")
for w in TEXTURED:
    for d in D:
        c, x = cell(w, d, "compact"), cell(w, d, "x4plus")
        note = ""
        if x["lpips"] < c["lpips"] and x["psnr"] < c["psnr"]:
            note = "x4plus: -PSNR +LPIPS (perception-distortion win)"
        print(f"{w:12s} {d:8s}  {c['psnr']:5.2f}/{x['psnr']:5.2f}   {c['lpips']:.4f}/{x['lpips']:.4f}   {note}")

print("\n" + "="*92)
print(f"COMPUTE: x4plus = {R['timing_ms_per_frame']['x4plus']/R['timing_ms_per_frame']['compact']:.0f}x compact "
      f"(ratio; MPS shared). Bicubic ~free.")
print("="*92)

# ---- visual crop: texture24k gritty (GT | bicubic | compact | x4plus) ----
def decode_one(path, start):
    cont = av.open(path); vs = cont.streams.video[0]; idx = 0
    for fr in cont.decode(vs):
        if idx == start:
            cont.close(); return fr.to_ndarray(format="rgb24")
        idx += 1
    cont.close(); return None

sys.path.insert(0, os.path.join(_ROOT, "prototype"))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r5_e2_quality"))
import sr as SR
from run_matrix import degrade, restore  # reuse, read-only
gt = decode_one(os.path.join(_ROOT, "sample.mp4"), W["texture24k"])
h, w = gt.shape[:2]
lr = degrade(gt, "gritty", seed=1000)
tiles = [("GT", gt), ("bicubic", restore(lr, w, h, "bicubic")),
         ("compact", restore(lr, w, h, "compact")), ("x4plus", restore(lr, w, h, "x4plus"))]
cy, cx = h // 2, w // 2
crop = lambda im: im[max(0, cy-80):cy+80, max(0, cx-80):cx+80]
panels = []
for label, im in tiles:
    cc = cv2.resize(crop(im), (320, 320), interpolation=cv2.INTER_NEAREST)
    cv2.putText(cc, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    panels.append(cc)
cv2.imwrite(os.path.join(_HERE, "crop_texture24k_gritty.png"),
            cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))
print("[viz] wrote crop_texture24k_gritty.png (GT|bicubic|compact|x4plus, center 160px)")
