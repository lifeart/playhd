#!/usr/bin/env python3
"""R9-E1 step 3 (CPU): sweep beta + compute the no-reference signal battery on the OOD
real-H.264 cells, and PRINT them side-by-side with the synthetic smooth cells. The
falsifier: can any no-reference signal place 'synthetic heavy talkinghead' (beta=0.5
wins) APART FROM 'mild H.264 talkinghead' (beta=0.85 needed) -- while keeping textured
cells at high beta? If the smooth cells overlap in every signal, a per-clip estimator
cannot avoid regressing on one family."""
import os, sys, json
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import signals as S  # noqa: E402

CACHE = os.path.join(_HERE, "ood_cache")
jobs = json.load(open(os.path.join(CACHE, "jobs.json")))

rows = {}
for w, crf in jobs:
    z = np.load(os.path.join(CACHE, f"{w}_crf{crf}.npz"))
    cell = {k: z[k] for k in z.files}
    sweep, x4_per = S.sweep_cell_lpips(cell)
    sig = S.cell_signals(cell)
    best_b = min(sweep, key=sweep.get)
    rows[f"{w}|crf{crf}"] = dict(sweep=sweep, x4_per=x4_per, sig=sig,
                                 best_beta=best_b, lpips_best=sweep[best_b],
                                 lpips_x4=sweep[1.0], lpips_085=sweep[0.85],
                                 headroom_vs_085=sweep[0.85] - sweep[best_b])
json.dump(dict(rows=rows), open(os.path.join(_HERE, "step3_ood.json"), "w"), indent=2)

print(f"{'cell':22s} {'bestb':>5s} {'L.50':>7s} {'L.85':>7s} {'Lx4':>7s} {'head085':>8s} | "
      f"{'edge':>6s} {'disag':>6s} {'noisMAD':>7s} {'immerk':>6s} {'hf':>5s}")
for k, r in rows.items():
    s = r["sig"]; sw = r["sweep"]
    print(f"{k:22s} {r['best_beta']:5.2f} {sw[0.5]:7.4f} {sw[0.85]:7.4f} {sw[1.0]:7.4f} "
          f"{r['headroom_vs_085']:+8.4f} | {s['edge']:6.4f} {s['disag_hr']:6.2f} "
          f"{s['noise_mad']:7.2f} {s['immerk']:6.2f} {s['hf_ratio']:5.3f}")

# ---- the decisive overlap table: smooth cells (synth vs OOD) in signal space ---- #
print("\n[FALSIFIER] smooth-face cells -- synthetic (want beta=0.5) vs H.264 (want beta>=0.85):")
synth = json.load(open(os.path.join(_HERE, "step1_synth.json")))["rows"]
def line(tag, r):
    s = r["sig"]
    print(f"  {tag:26s} bestb={r['best_beta']:.2f} head085={r['headroom_vs_085']:+.4f} | "
          f"edge={s['edge']:.4f} disag={s['disag_hr']:5.2f} noisMAD={s['noise_mad']:.2f} "
          f"immerk={s['immerk']:.2f} hf={s['hf_ratio']:.3f} lrhf={s['lrhf']:.2f}")
for d in ("moderate", "heavy", "gritty"):
    line(f"talkinghead|{d} (SYNTH)", synth[f"talkinghead|{d}"])
for w, crf in jobs:
    if w == "talkinghead":
        line(f"talkinghead|crf{crf} (H264)", rows[f"talkinghead|crf{crf}"])
print("[done] -> step3_ood.json")
