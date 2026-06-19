"""run_sweep.py -- R2-E3: measure each OUTPUT-ONLY seam tweak vs the layered baseline.

Metrics (mean over the 32-frame talking-head scene; halo over 3 sampled frames):
  ratio       = seam discontinuity = FGring/BGring var-Laplacian (target ~3.45 = uniform-x4plus)
  FGring/BGring = the two ring sharpnesses (deep-BG/ceiling level ~16-17 = the match target)
  halo_w      = px (HD) the layered composite stays SOFTER than the uniform-x4plus ceiling
                near the edge (the soft 'moat'); lower=less halo, 0=ceiling-matched
  fg_bias     = (|out-plate|-|out-fg|) hair band; UP = hair looks more like SUBJECT (wisps kept)
  hairS       = hair-band |Laplacian| (wisp structure)
  coreS       = subject-core |Laplacian| (a>0.95) -- MUST hold (drop = smearing the face)
var-Laplacian is a RELATIVE seam-continuity metric here, NOT an SR-quality claim.
"""
import os
import numpy as np
import seam_lib as L

D = L.load_inputs()
N = D["x4"].shape[0]
H, W = D["plate_hd"].shape[:2]
HALO_FRAMES = [8, 16, 24]
PLATE0 = D["plate_hd"]


def alpha_base(i):
    return L.alpha_to_hd(D["phas"][i], (H, W))


def eval_config(name, fg_cache, alpha_tf=None, plate_tf=None, fg_tf=None):
    ratios, fgr, bgr, biases, hairs, cores, halos = [], [], [], [], [], [], []
    for i in range(N):
        a0 = alpha_base(i)
        a = alpha_tf(a0) if alpha_tf else a0
        plate = plate_tf(PLATE0, a0) if plate_tf else PLATE0
        fg = fg_tf(fg_cache[i], a0) if fg_tf else fg_cache[i]
        out = L.composite(fg, a, plate)
        sf, sb, r = L.seam_ratio(out, a)
        ratios.append(r); fgr.append(sf); bgr.append(sb)
        hd = L.hair_detail(out, fg_cache[i], plate, a)
        biases.append(hd["fg_pl_bias"]); hairs.append(hd["hair_struct"]); cores.append(hd["core_struct"])
        if i in HALO_FRAMES:
            halos.append(L.halo_deficit_width(out, a, D["x4"][i]))
    return dict(name=name, ratio=np.nanmean(ratios), fgr=np.nanmean(fgr), bgr=np.nanmean(bgr),
                halo=np.nanmean(halos), bias=np.nanmean(biases),
                hairS=np.nanmean(hairs), coreS=np.nanmean(cores))


def ref_uniform_x4plus():
    ratios, fgr, bgr, cores, hairs = [], [], [], [], []
    for i in range(N):
        a = alpha_base(i)
        sf, sb, r = L.seam_ratio(D["x4"][i], a)
        ratios.append(r); fgr.append(sf); bgr.append(sb)
        hd = L.hair_detail(D["x4"][i], D["x4"][i], PLATE0, a)
        cores.append(hd["core_struct"]); hairs.append(hd["hair_struct"])
    return dict(name="uniform-x4plus (REF)", ratio=np.nanmean(ratios), fgr=np.nanmean(fgr),
                bgr=np.nanmean(bgr), halo=0.0, bias=float("nan"),
                hairS=np.nanmean(hairs), coreS=np.nanmean(cores))


RING = lambda p, a: L.restore_plate_ring(p, a, strength=0.5)   # deep-BG-matched plate-ring restore


def budget_table(title, fg):
    rows = [
        eval_config("baseline", fg),
        eval_config("(a) feather", fg, alpha_tf=L.feather_alpha),
        eval_config("(b) ringRestore", fg, plate_tf=RING),
        eval_config("(b-alt) softenFG [rej]", fg, fg_tf=L.soften_fg_band),
        eval_config("(a)+(b) RECOMMENDED", fg, alpha_tf=L.feather_alpha, plate_tf=RING),
    ]
    print(f"\n=== {title} (32-frame talking-head, mean; halo over 3 frames) ===")
    hdr = f"{'config':22s} | {'ratio':>6s} | {'FGring':>7s} | {'BGring':>7s} | {'halo':>5s} | {'fg_bias':>7s} | {'hairS':>6s} | {'coreS':>6s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:22s} | {r['ratio']:6.2f} | {r['fgr']:7.1f} | {r['bgr']:7.1f} | "
              f"{r['halo']:5.1f} | {r['bias']:7.2f} | {r['hairS']:6.2f} | {r['coreS']:6.2f}")
    return rows


def main():
    rx = budget_table("x4plus-bbox budget (WORST seam)", D["x4"])
    rc = budget_table("compact-FG budget", D["cp"])
    ref = ref_uniform_x4plus()
    print(f"\nREF uniform-x4plus (no layering): ratio {ref['ratio']:.2f}  FGring {ref['fgr']:.1f}  "
          f"BGring {ref['bgr']:.1f}  halo 0.0  coreS {ref['coreS']:.2f}  (deep-BG plate var-Lap ~15.4)")
    print("target: BGring -> deep-BG ~15 (kill the soft moat); x4plus ratio -> ceiling ~3.45; "
          "coreS HOLD; halo down.")
    np.save(os.path.join(L.CACHE, "sweep_rows.npy"),
            np.array(dict(x4plus=rx, compact=rc, ref=ref), dtype=object))


if __name__ == "__main__":
    main()
