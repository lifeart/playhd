#!/usr/bin/env python3
"""
R13-E3 diagnostic #3: WHERE does a variant's win come from?

    python diag_sets.py jelly ema_adapt_50

Recomputes the motion-compensated luma flicker of a stabilised sequence over the SAME
three pixel sets diag_extra.py defines on the SHIPPING reconstruction (the sets are fixed
by the baseline's hard occlusion mask, so both sequences are scored on identical pixels):
    flip  -- the pixel's source switched between propagated warp and fresh per-frame SR
    prop  -- propagated in both frames
    fb    -- fresh per-frame SR in both frames
A temporal filter that merely blurs everything would lower all three equally; a fix that
targets the hard-switch oscillation lowers `flip` most.  Reported next to LPIPS-vs-x4plus
so a blur-win cannot hide.
"""
import os
import sys
import json

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from common import SCALE, OUT, decode, hd_size, sr_cache  # noqa: E402
import jelly_metrics as JM   # noqa: E402
import stabilizer as ST      # noqa: E402
import derisk                # noqa: E402  READ-ONLY
import metrics as M          # noqa: E402  READ-ONLY (var_laplacian)
from run_ab import _variants  # noqa: E402


def run(name, vname):
    frames = decode(name)
    N = len(frames)
    compact = sr_cache(name, "realesrgan")
    _, R = derisk.reconstruct(frames, None, SCALE, True, "full",
                              {i: compact[i] for i in range(N)}, set(), backend="numpy")
    base = {i: R[i]["recon"] for i in range(N)}
    w_hd, h_hd = hd_size(frames)
    masks = {i: (np.asarray(R[i]["mask"], bool) if R[i].get("mask") is not None
                 else np.zeros((h_hd, w_hd), bool)) for i in range(N)}
    seqs = {"baseline_hard": [base[i] for i in range(N)]}
    cfg = _variants()[vname]
    seqs[vname], st = ST.stabilize(frames, base, cfg, scale=SCALE, masks=masks, enabled=True)
    flows = JM.build_mc_flows(frames, SCALE)

    out = {}
    for sname, seq in seqs.items():
        acc = {k: [] for k in ("flip", "prop", "fb", "all")}
        for t in range(1, N):
            fl = flows[t]
            if fl is None:
                continue
            pm, _ = derisk.warp_hd(masks[t - 1].astype(np.uint8) * 255, fl["fx"], fl["fy"], SCALE)
            pm = pm > 127
            wprev, hole = derisk.warp_hd(seq[t - 1], fl["fx"], fl["fy"], SCALE)
            valid = fl["valid_hd"] & ~hole
            if valid.sum() < 1000:
                continue
            d = np.abs(JM.luma(seq[t]) - JM.luma(wprev))
            sets = dict(flip=(masks[t] != pm) & valid,
                        prop=(~masks[t]) & (~pm) & valid,
                        fb=masks[t] & pm & valid,
                        all=valid)
            for k, m in sets.items():
                if m.sum() > 500:
                    acc[k].append(float(d[m].mean()))
        out[sname] = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
        # GOTCHA #23 secondary guard: an NR sharpness number, reported NEVER as the headline
        # -- only to show a temporal filter is not winning the flicker metrics by blurring.
        out[sname]["varlap"] = float(np.mean([M.var_laplacian(f) for f in seq]))
    b, v = out["baseline_hard"], out[vname]
    print(f"\n[{name}] set-wise MC luma flicker (codes), {vname} vs baseline_hard "
          f"(alpha_mean={st['alpha_mean_all']:.3f})")
    print(f"{'set':6s} {'baseline':>9} {'variant':>9} {'delta':>8}")
    for k in ("all", "flip", "prop", "fb", "varlap"):
        print(f"{k:6s} {b[k]:9.3f} {v[k]:9.3f} {100*(v[k]-b[k])/b[k]:+7.1f}%")
    with open(os.path.join(OUT, f"sets_{name}_{vname}.json"), "w") as fh:
        json.dump({"window": name, "variant": vname, "sets": out,
                   "alpha_mean": st["alpha_mean_all"]}, fh, indent=2)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "jelly",
        sys.argv[2] if len(sys.argv) > 2 else "ema_adapt_50")
