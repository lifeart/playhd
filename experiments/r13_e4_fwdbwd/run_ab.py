#!/usr/bin/env python3
"""
R13-E4 TIER 1 -- the perceptual A/B that actually decides the question.

Runs the SHIPPED reconstruction (derisk.reconstruct, quality-mode config: x4plus
fp16 anchors + anchor-blend beta=0.85 + region-aware output blend) over windows
A/B/C of the GT-bearing degrade-restore rig, swapping ONLY the occlusion mask:

  naive        (a) intra holes only                        -- floor
  reactive     (a)+(b)                                     -- the deletion candidate
  full         (a)+(b)+Ruder softmax-splat fwd-bwd         -- the incumbent
  adaptive     (a)+(b), Ruder gated by ADAPTIVE_TAU=0.06   -- what quality mode ships
  fsr2         (a)+(b)+FSR2 temporal MV divergence         -- the REPLACEMENT
  fsr2+ruder   (a)+(b)+Ruder+FSR2                          -- the ADDITION
  oracle*      per-pixel best-of(propagated, per-frame-SR) -- the ceiling (diagnostic)

Decision metrics: LPIPS (lead) + DISTS (triangulation, R8-E4) + PSNR + tOF +
fallback%. recall/precision are NOT used here (see run_diag.py for why).

Usage: python3 run_ab.py [--windows A,B,C] [--variants ...] [--backend torch|numpy]
"""
import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import harness as H          # noqa: E402
import fsr2_mask as F        # noqa: E402
import derisk as D           # noqa: E402  READ-ONLY

VARIANTS = ["naive", "reactive", "full", "adaptive", "fsr2", "fsr2+ruder"]
BETA = 0.85                  # R8-E3 shipped anchor blend (quality mode default-ON)


def build_window(name, backend):
    frames, gt = H.window(name)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * H.SCALE, h_lr * H.SCALE
    assert (h_hd, w_hd) == gt[0].shape[:2], (gt[0].shape, (h_hd, w_hd))
    heavy, compact = H.sr_caches(name, frames, w_hd, h_hd)
    types = "".join(f[0] for f in frames)
    print(f"[{name}] {len(frames)} frames  LR {w_lr}x{h_lr} -> HD {w_hd}x{h_hd}  {types}")
    # production quality mode: region-aware gate (motion) + anchor blend beta=0.85
    region_gate = D._build_region_gate(frames, w_hd, h_hd, H.SCALE, compact_cache=compact)
    heavy_b = D.blend_anchor_cache({k: v.copy() for k, v in heavy.items()}, compact, BETA)
    return dict(name=name, frames=frames, gt=gt, cache=heavy_b, compact=compact,
                region_gate=region_gate, w_hd=w_hd, h_hd=h_hd, backend=backend)


def run_variant(W, variant, **kw):
    """One reconstruction with the given mask variant. Returns (seq, stats)."""
    F.patch()
    F.set_variant(variant if variant in F._BASE_MODE else "reactive",
                  scale=H.SCALE, **kw)
    occ = "naive" if variant == "naive" else F.occ_mode_for(variant)
    D.MASK_FIRES[0] = D.MASK_FIRES[1] = 0
    t0 = time.perf_counter()
    rows, R = D.reconstruct(W["frames"], None, H.SCALE, True, occ, W["cache"], set(),
                            backend=W["backend"], collect_metrics=True,
                            region_gate=(None if W.get("no_region") else W["region_gate"]))
    dt = time.perf_counter() - t0
    seq = [R[i]["recon"] for i in range(len(W["frames"]))]
    holes = [float(np.mean(R[i]["mask"])) if R[i]["mask"] is not None else 0.0
             for i in range(len(W["frames"]))]
    extra = float(np.mean(F.STATS["fsr2_area"])) if F.STATS["fsr2_area"] else 0.0
    return seq, dict(variant=variant, wall_s=dt,
                     fallback_pct=100.0 * float(np.mean(holes)),
                     fallback_max_pct=100.0 * float(np.max(holes)),
                     splat_fires=f"{D.MASK_FIRES[0]}/{D.MASK_FIRES[1]}",
                     extra_area_pct=100.0 * extra,
                     hook_hits=F.REG.hits, hook_misses=F.REG.misses), R


def oracle_ceiling(W, R):
    """Per-pixel best-of(propagated, per-frame-SR) against GT -- the ceiling ANY
    per-pixel occlusion mask could reach on this propagation (diagnostic only:
    a real mask cannot see the GT, and the chain effect is not modelled)."""
    out = []
    for i in range(len(W["frames"])):
        rec = R[i]["recon"].astype(np.int16)
        pf = R[i]["perframe"].astype(np.int16)
        gt = W["gt"][i].astype(np.int16)
        take_pf = (np.abs(pf - gt).mean(axis=2) < np.abs(rec - gt).mean(axis=2))[:, :, None]
        out.append(np.where(take_pf, pf, rec).astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="A,B,C")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--backend", default="torch", choices=["torch", "numpy"])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--gate-px", type=float, default=20.0)
    ap.add_argument("--nan-prev", default="flag")
    ap.add_argument("--dilate", type=int, default=1,
                    help="FSR2 feeds the op DILATED MVs (3x3 max-magnitude here)")
    ap.add_argument("--normalize", type=int, default=1)
    ap.add_argument("--tag", default="ab")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--no-region", action="store_true",
                    help="drop the region-aware OUTPUT blend (isolates the mask's effect: "
                         "on high motion the motion gate blends the whole frame toward the "
                         "per-frame compact SR, which can mask any propagation difference)")
    args = ap.parse_args()

    import metrics_extra as ME
    sc = ME.selfcheck()
    assert sc["dists_self"] < 1e-3 and sc["lpips_self"] < 1e-3, sc
    print(f"[metric selfcheck] {sc}")

    results = {}
    for wname in args.windows.split(","):
        W = build_window(wname, args.backend)
        W["no_region"] = bool(args.no_region)
        if args.no_region:
            print(f"  (region-aware output blend DISABLED for this run)")
        results[wname] = {}
        for v in args.variants.split(","):
            seq, st, R = run_variant(W, v, thr=args.thr, gate_px=args.gate_px,
                                     nan_prev=args.nan_prev, dilate=bool(args.dilate),
                                     normalize=bool(args.normalize))
            s = H.score(seq, W["gt"], tag=v)
            s.update(st)
            results[wname][v] = s
            print(f"  [{wname}/{v:11}] LPIPS {s['lpips']:.4f}  DISTS {s['dists']:.4f}  "
                  f"PSNR {s['psnr']:.2f}  tOF {s['tof']:.4f}  fb {s['fallback_pct']:.2f}%"
                  f"  splat {st['splat_fires']}  extra {st['extra_area_pct']:.2f}%"
                  f"  {st['wall_s']:.1f}s")
            if v == "reactive" and not args.no_oracle:
                osq = oracle_ceiling(W, R)
                so = H.score(osq, W["gt"], tag="oracle")
                results[wname]["oracle"] = so
                print(f"  [{wname}/{'oracle':11}] LPIPS {so['lpips']:.4f}  "
                      f"DISTS {so['dists']:.4f}  PSNR {so['psnr']:.2f}  tOF {so['tof']:.4f}"
                      "   (per-pixel ceiling, diagnostic)")
            del seq, R
            H._free_gpu()
        del W
        H._free_gpu()
    H.save(dict(config=vars(args), results=results),
           os.path.join(H.OUT, f"{args.tag}.json"))


if __name__ == "__main__":
    main()
