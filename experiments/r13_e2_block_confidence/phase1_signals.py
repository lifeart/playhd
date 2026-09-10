#!/usr/bin/env python3
"""
R13-E2 phase 1 -- do the FREE per-block codec signals (partition size, MV magnitude) carry
information about propagation error that the LR RESIDUAL does not already carry?

This is the question the R13 brief's discrimination table cannot answer on its own: that table
measured warp error AT LR, and the residual term the gate already uses IS the LR warp error, so
any signal that correlates with it looks strong while adding nothing.

Here the error is measured at HD against a TRUE 2x ground truth (make_degraded.py), with a
PERFECT anchor (GT reference, GOTCHA #15's isolation rule) so the number is pure propagation
error and not re-injected fresh SR:

    err_hd = | warp_hd(GT_ref, MV) + upsample(LR_cur - warp_lr(LR_ref)) - GT_cur |

aggregated per MV RECORD (per codec partition), alongside that partition's
    react   = mean |LR_cur - warp_lr(LR_ref)|       (what the gate already sees)
    minside = min(partition_w, partition_h)          (free, unused)
    mvmag   = |MV| in LR px                          (free, unused)

Outputs out/phase1_blocks.npz (the raw table) and out/phase1_summary.json:
  * unconditional discrimination (the brief's table, recomputed on HD error)
  * CONDITIONAL discrimination: inside residual quintiles, do the block signals still separate?
  * a decision-relevant GAIN CURVE: ranking blocks by each predictor, what fraction of total HD
    error mass is captured by distrusting the top X% of frame area?

Run: python3 experiments/r13_e2_block_confidence/phase1_signals.py
"""
import os
import sys
import json

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "prototype"))

import derisk            # noqa: E402  READ-ONLY
import block_confidence as bc   # noqa: E402

OUT = os.path.join(HERE, "out")
PROBE = os.path.join(OUT, "probe")
SCALE = 2


def _blocks_for_frame(mvs, want, lr_cur, lr_ref, gt_cur, gt_ref):
    """Per-MV-record rows for one frame/direction.
    Returns list of (react, err_hd, minside, mvmag, npix)."""
    h_lr, w_lr = lr_cur.shape[:2]
    h_hd, w_hd = gt_cur.shape[:2]
    fx, fy = derisk.build_lr_flow(mvs, h_lr, w_lr, want=want)
    if not np.isfinite(fx).any():
        return [], None, None
    pred_lr = derisk.warp_lr(lr_ref, fx, fy)
    react = np.abs(lr_cur.astype(np.float32) - pred_lr.astype(np.float32)).mean(axis=2)
    res = lr_cur.astype(np.float32) - pred_lr.astype(np.float32)
    res_hd = cv2.resize(res, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
    warped, hole = derisk.warp_hd(gt_ref, fx, fy, SCALE)
    recon = np.clip(warped.astype(np.float32) + res_hd, 0, 255)
    err_hd = np.abs(recon - gt_cur.astype(np.float32)).mean(axis=2)

    rows = []
    for r in mvs:
        s = int(r["source"])
        if want == "past" and s >= 0:
            continue
        if want == "future" and s <= 0:
            continue
        ms = float(r["motion_scale"]) or 1.0
        dx = float(r["motion_x"]) / ms
        dy = float(r["motion_y"]) / ms
        bw, bh = int(r["w"]), int(r["h"])
        cx, cy = int(r["dst_x"]), int(r["dst_y"])
        x0, x1 = max(cx - bw // 2, 0), min(cx + bw // 2, w_lr)
        y0, y1 = max(cy - bh // 2, 0), min(cy + bh // 2, h_lr)
        if x1 <= x0 or y1 <= y0:
            continue
        rc = float(react[y0:y1, x0:x1].mean())
        eh = float(err_hd[y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE].mean())
        rows.append((rc, eh, float(min(bw, bh)), float(np.hypot(dx, dy)),
                     float((x1 - x0) * (y1 - y0))))
    # hole (intra) statistics for this frame/direction
    hole_lr = ~np.isfinite(fx)
    hole_err = float(err_hd[cv2.resize(hole_lr.astype(np.uint8), (w_hd, h_hd),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)].mean()) \
        if hole_lr.any() else np.nan
    cov_err = float(err_hd[cv2.resize((~hole_lr).astype(np.uint8), (w_hd, h_hd),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)].mean()) \
        if (~hole_lr).any() else np.nan
    return rows, hole_err, cov_err


def run_probe(name):
    lr_path = os.path.join(PROBE, f"{name}_lr.mp4")
    gt = np.load(os.path.join(PROBE, f"{name}_gt.npy"))
    frames = derisk.decode_lr_and_mvs(lr_path)
    n = min(len(frames), len(gt))
    bb = derisk.backbone_indices(frames[:n])
    rows = []
    holes, covs = [], []
    for i in range(n):
        pt = frames[i][0]
        if pt == "I":
            continue
        lr_cur, mvs = frames[i][1], frames[i][2]
        if pt == "P":
            p = max([b for b in bb if b < i], default=None)
            if p is None:
                continue
            out = _blocks_for_frame(mvs, "past", lr_cur, frames[p][1], gt[i], gt[p])
            rows += out[0]
            if out[1] is not None and np.isfinite(out[1]):
                holes.append(out[1]); covs.append(out[2])
        else:  # B: both directions
            p = max([b for b in bb if b < i], default=None)
            f = min([b for b in bb if b > i], default=None)
            for ref, want in ((p, "past"), (f, "future")):
                if ref is None:
                    continue
                out = _blocks_for_frame(mvs, want, lr_cur, frames[ref][1], gt[i], gt[ref])
                rows += out[0]
                if out[1] is not None and np.isfinite(out[1]):
                    holes.append(out[1]); covs.append(out[2])
    arr = np.array(rows, np.float32) if rows else np.zeros((0, 5), np.float32)
    print(f"[{name}] {n} frames, {len(arr)} MV records, "
          f"hole_err={np.mean(holes):.2f} covered_err={np.mean(covs):.2f}")
    return arr, float(np.mean(holes)) if holes else np.nan, float(np.mean(covs)) if covs else np.nan


# --------------------------------------------------------------------------- #
def wmean(v, w):
    return float(np.sum(v * w) / max(np.sum(w), 1e-9))


def gain_curve(score, err, npix, fracs=(0.05, 0.10, 0.20, 0.30, 0.50)):
    """Rank blocks by `score` DESC (score = predicted unreliability). Return the fraction of
    total HD error MASS (err*npix) captured by the top-X%-of-AREA blocks."""
    order = np.argsort(-score)
    a = npix[order]
    m = (err * npix)[order]
    ca = np.cumsum(a) / max(np.sum(a), 1e-9)
    cm = np.cumsum(m) / max(np.sum(m), 1e-9)
    return {f"top{int(f*100)}pct_area": float(np.interp(f, ca, cm)) for f in fracs}


def analyse(arr, tag):
    react, err, minside, mvmag, npix = [arr[:, k] for k in range(5)]
    res = {"tag": tag, "n_blocks": int(len(arr))}
    res["err_mean"] = wmean(err, npix)

    # --- unconditional discrimination (the brief's table, on HD error) ---
    m16 = minside >= 16
    res["partition"] = {
        "16x16_err": wmean(err[m16], npix[m16]) if m16.any() else None,
        "sub16_err": wmean(err[~m16], npix[~m16]) if (~m16).any() else None,
        "sub16_area_frac": float(np.sum(npix[~m16]) / max(np.sum(npix), 1e-9)),
    }
    if res["partition"]["16x16_err"]:
        res["partition"]["ratio"] = res["partition"]["sub16_err"] / res["partition"]["16x16_err"]
    mbins = [(0, 1), (1, 4), (4, 1e9)]
    res["mvmag"] = {}
    for lo, hi in mbins:
        sel = (mvmag >= lo) & (mvmag < hi)
        res["mvmag"][f"{lo}-{hi if hi < 1e9 else 'inf'}"] = dict(
            err=wmean(err[sel], npix[sel]) if sel.any() else None,
            area_frac=float(np.sum(npix[sel]) / max(np.sum(npix), 1e-9)))
    # the residual itself, for scale
    rq = np.quantile(react, [0.2, 0.4, 0.6, 0.8])
    res["residual_quintile_err"] = []
    edges = [-1e9] + list(rq) + [1e9]
    for k in range(5):
        sel = (react >= edges[k]) & (react < edges[k + 1])
        res["residual_quintile_err"].append(dict(
            q=k + 1, react_lo=float(edges[k] if k else react.min()),
            react_hi=float(edges[k + 1] if k < 4 else react.max()),
            err=wmean(err[sel], npix[sel]) if sel.any() else None,
            n=int(sel.sum())))

    # --- CONDITIONAL: inside each residual quintile, do block signals still separate? ---
    cond = []
    for k in range(5):
        sel = (react >= edges[k]) & (react < edges[k + 1])
        if sel.sum() < 50:
            continue
        e, mi, mv, np_ = err[sel], minside[sel], mvmag[sel], npix[sel]
        a16 = mi >= 16
        big = mv >= 4
        cond.append(dict(
            q=k + 1, n=int(sel.sum()),
            err_16x16=wmean(e[a16], np_[a16]) if a16.any() else None,
            err_sub16=wmean(e[~a16], np_[~a16]) if (~a16).any() else None,
            err_mv_lt4=wmean(e[~big], np_[~big]) if (~big).any() else None,
            err_mv_ge4=wmean(e[big], np_[big]) if big.any() else None,
        ))
        c = cond[-1]
        if c["err_16x16"]:
            c["part_ratio"] = (c["err_sub16"] or 0) / c["err_16x16"]
        if c["err_mv_lt4"]:
            c["mv_ratio"] = (c["err_mv_ge4"] or 0) / c["err_mv_lt4"]
    res["conditional"] = cond

    # --- decision-relevant gain curves ---
    conf = bc.confidence_lr(minside, mvmag)          # 1 = trusted
    res["gain"] = {
        "residual_only": gain_curve(react, err, npix),
        "block_only": gain_curve(1.0 - conf, err, npix),
        "residual_over_conf": gain_curve(react / np.maximum(conf, 1e-3), err, npix),
        "oracle": gain_curve(err, err, npix),
        "random": {k: float(v) for k, v in
                   zip(["top5pct_area", "top10pct_area", "top20pct_area",
                        "top30pct_area", "top50pct_area"], [.05, .10, .20, .30, .50])},
    }
    # --- THE decisive test: the RESIDUAL's BLIND SPOT --------------------------------- #
    # The shipping gate's trust is a = sigmoid((tau_res - react)/s_res), tau=10, s=5.
    # A block signal can only ADD value where the gate already trusts the warp (a high) but
    # the HD error is large. Quantify (a) how much total HD error mass lives under high trust,
    # and (b) whether the block prior can find it there better than chance.
    a_gate = 1.0 / (1.0 + np.exp(-(10.0 - react) / 5.0))
    mass = err * npix
    res["blindspot"] = dict(
        a_gate_max=float(a_gate.max()),          # NB: sigmoid(10/5)=0.881 -- never full trust
        mass_frac_trust_gt080=float(np.sum(mass[a_gate > 0.80]) / max(np.sum(mass), 1e-9)),
        mass_frac_trust_gt075=float(np.sum(mass[a_gate > 0.75]) / max(np.sum(mass), 1e-9)),
        area_frac_trust_gt080=float(np.sum(npix[a_gate > 0.80]) / max(np.sum(npix), 1e-9)),
        gate_weighted_err=float(np.sum(mass * a_gate) / max(np.sum(npix * a_gate), 1e-9)),
    )
    sel = a_gate > 0.80
    if sel.sum() > 200:
        res["blindspot"]["gain_inside_trusted"] = dict(
            block_only=gain_curve(1.0 - conf[sel], err[sel], npix[sel]),
            residual_only=gain_curve(react[sel], err[sel], npix[sel]),
            oracle=gain_curve(err[sel], err[sel], npix[sel]),
        )
        res["blindspot"]["spearman_conf_err_inside_trusted"] = float(
            np.corrcoef(np.argsort(np.argsort(1.0 - conf[sel])).astype(np.float64),
                        np.argsort(np.argsort(err[sel])).astype(np.float64))[0, 1])

    # --- BEST-CASE combination sweep (give the block signals every chance) ------------- #
    # score = react + w * (1 - conf) * K  with K = the residual scale, so w is dimensionless.
    K = float(np.quantile(react, 0.9)) or 1.0
    best = None
    sweep = {}
    for w in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        g = gain_curve(react + w * (1.0 - conf) * K, err, npix)["top10pct_area"]
        sweep[f"w={w}"] = float(g)
        if best is None or g > best[1]:
            best = (w, g)
    res["combo_sweep"] = dict(K=K, sweep=sweep, best_w=best[0], best_gain10=best[1])

    # --- fine-grained signal tables --------------------------------------------------- #
    res["partition_fine"] = {}
    for v in sorted(set(minside.tolist())):
        s2 = minside == v
        res["partition_fine"][str(int(v))] = dict(
            err=wmean(err[s2], npix[s2]), area_frac=float(np.sum(npix[s2]) / np.sum(npix)))
    res["mvmag_fine"] = {}
    for lo, hi in [(0, .25), (.25, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)]:
        s2 = (mvmag >= lo) & (mvmag < hi)
        if s2.any():
            res["mvmag_fine"][f"{lo}-{hi if hi < 1e9 else 'inf'}"] = dict(
                err=wmean(err[s2], npix[s2]),
                area_frac=float(np.sum(npix[s2]) / np.sum(npix)))

    # correlations (Spearman via rank Pearson)
    def sp(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])
    res["spearman"] = dict(
        react_err=sp(react, err), minside_err=sp(minside, err), mvmag_err=sp(mvmag, err),
        minside_react=sp(minside, react), mvmag_react=sp(mvmag, react))
    return res


def main():
    manifest = json.load(open(os.path.join(PROBE, "manifest.json")))
    all_arr = []
    per = []
    holes = {}
    for m in manifest:
        arr, he, ce = run_probe(m["name"])
        if len(arr) == 0:
            continue
        holes[m["name"]] = dict(hole_err=he, covered_err=ce,
                                ratio=(he / ce) if ce else None)
        per.append(analyse(arr, m["name"]))
        all_arr.append(arr)
    pooled = np.concatenate(all_arr, axis=0)
    summ = dict(per_clip=per, pooled=analyse(pooled, "POOLED"), intra_hole=holes)
    np.savez_compressed(os.path.join(OUT, "phase1_blocks.npz"), pooled=pooled)
    json.dump(summ, open(os.path.join(OUT, "phase1_summary.json"), "w"), indent=2)

    p = summ["pooled"]
    print("\n=== POOLED (HD propagation error per codec partition, perfect anchor) ===")
    print(f"blocks={p['n_blocks']}  mean err={p['err_mean']:.3f}")
    print(f"partition: 16x16 {p['partition']['16x16_err']:.3f} vs sub16 "
          f"{p['partition']['sub16_err']:.3f}  ratio {p['partition'].get('ratio', float('nan')):.2f}x "
          f"(sub16 = {100*p['partition']['sub16_area_frac']:.1f}% of MV area)")
    for k, v in p["mvmag"].items():
        print(f"mvmag {k:>8s}: err {v['err']:.3f}  area {100*v['area_frac']:.1f}%")
    print("conditional (inside residual quintiles):")
    for c in p["conditional"]:
        print(f"  q{c['q']} n={c['n']:6d}  16x16 {c['err_16x16']:.3f} / sub16 "
              f"{c['err_sub16']:.3f} = {c.get('part_ratio', float('nan')):.2f}x   |   "
              f"mv<4 {c['err_mv_lt4']:.3f} / mv>=4 {c['err_mv_ge4'] if c['err_mv_ge4'] else float('nan'):.3f}"
              f" = {c.get('mv_ratio', float('nan')):.2f}x")
    print("gain curve (fraction of HD error mass captured by top-X% of area):")
    for k, v in p["gain"].items():
        print(f"  {k:20s} " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in v.items()))
    print("spearman:", {k: round(v, 3) for k, v in p["spearman"].items()})
    b = p["blindspot"]
    print(f"BLIND SPOT: a_max={b['a_gate_max']:.3f}; {100*b['mass_frac_trust_gt080']:.1f}% of HD "
          f"error mass sits where gate trust a>0.80 ({100*b['area_frac_trust_gt080']:.1f}% of area)")
    print("combo sweep (gain@top10%):", p['combo_sweep']['sweep'],
          " best w=", p['combo_sweep']['best_w'])
    if "gain_inside_trusted" in b:
        for k, v in b["gain_inside_trusted"].items():
            print(f"   inside-trusted {k:14s} " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in v.items()))
        print(f"   spearman(1-conf, err | trusted) = {b['spearman_conf_err_inside_trusted']:.3f}")
    print("partition_fine:", {k: round(v["err"], 3) for k, v in p["partition_fine"].items()})
    print("            area:", {k: round(100 * v["area_frac"], 1) for k, v in p["partition_fine"].items()})
    print("mvmag_fine:", {k: round(v["err"], 3) for k, v in p["mvmag_fine"].items()})
    print("      area:", {k: round(100 * v["area_frac"], 1) for k, v in p["mvmag_fine"].items()})
    print("\nper-clip partition ratio / gain@top10%:")
    for c in summ["per_clip"]:
        print(f"  {c['tag']:16s} err {c['err_mean']:6.3f}  part_ratio "
              f"{c['partition'].get('ratio', float('nan')):5.2f}x  "
              f"gain10 res {c['gain']['residual_only']['top10pct_area']:.3f} / "
              f"blk {c['gain']['block_only']['top10pct_area']:.3f} / "
              f"res_conf {c['gain']['residual_over_conf']['top10pct_area']:.3f} / "
              f"oracle {c['gain']['oracle']['top10pct_area']:.3f}   "
              f"blind {100*c['blindspot']['mass_frac_trust_gt080']:.1f}%  "
              f"combo_best_w={c['combo_sweep']['best_w']}")
    print("intra-hole vs covered HD error:")
    for k, v in holes.items():
        print(f"  {k:16s} hole {v['hole_err']:.2f} vs covered {v['covered_err']:.2f} "
              f"= {v['ratio']:.1f}x")


if __name__ == "__main__":
    sys.exit(main())
