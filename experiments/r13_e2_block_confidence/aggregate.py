#!/usr/bin/env python3
"""R13-E2 -- aggregate the phase-2 (true-GT) and phase-2b (real-footage tOF) sweeps into the
per-clip tables the REPORT quotes. Prints markdown-ready rows; writes out/summary_tables.md."""
import os
import sys
import glob
import json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def load(sub):
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, sub, "*.json"))):
        rows.append(json.load(open(p)))
    return rows


def pct(new, old):
    if old in (None, 0):
        return float("nan")
    return 100.0 * (new - old) / old


def table_gt(rows):
    lines = []
    order = ["baseline_hard", "gate_res", "gate_res_block", "gate_res_bmul", "gate_block",
             "perframe_sr", "bicubic"]
    lines.append("| clip | variant | LPIPS_gt | DISTS_gt | tOF_gt | PSNR_gt | dF | distrust |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        for v in order:
            m = r["metrics"].get(v)
            if not m:
                continue
            d = r.get("distrust", {}).get(v)
            lines.append(f"| {r['name']} | {v} | {m['lpips_gt']:.4f} | {m['dists_gt']:.4f} | "
                         f"{m['tof_gt']:.4f} | {m['psnr_gt']:.2f} | {m['dF']:.3f} | "
                         + (f"{d:.3f} |" if d is not None else "- |"))
    return lines


def deltas_gt(rows):
    lines = ["| clip | hard fb% | dLPIPS_gt gate_res | dDISTS_gt | dtOF_gt | "
             "dLPIPS_gt gate_res_block | dDISTS_gt | dtOF_gt |", "|---|---|---|---|---|---|---|---|"]
    agg = {}
    for r in rows:
        b = r["metrics"]["baseline_hard"]
        cells = [f"{r['name']}", f"{100*r['hard_fallback_frac']:.2f}%"]
        for v in ("gate_res", "gate_res_block"):
            m = r["metrics"][v]
            for k in ("lpips_gt", "dists_gt", "tof_gt"):
                p = pct(m[k], b[k])
                cells.append(f"{p:+.1f}%")
                agg.setdefault((v, k), []).append(p)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("mean over clips: " + ", ".join(
        f"{v}.{k} {sum(a)/len(a):+.1f}%" for (v, k), a in agg.items()))
    # win counts
    lines.append("win counts (variant better than baseline_hard):")
    for v in ("gate_res", "gate_res_block", "gate_res_bmul", "gate_block"):
        cnt = {}
        for k in ("lpips_gt", "dists_gt", "tof_gt"):
            w = sum(1 for r in rows if v in r["metrics"]
                    and r["metrics"][v][k] < r["metrics"]["baseline_hard"][k])
            cnt[k] = f"{w}/{len(rows)}"
        lines.append(f"  {v}: " + "  ".join(f"{k} {c}" for k, c in cnt.items()))
    return lines


def table_real(rows):
    order = ["baseline_hard", "gate_res", "gate_res_norm", "gate_res_block", "gate_block",
             "perframe_sr", "bicubic"]
    lines = ["| window | variant | tOF_true | d vs hard | dF | LPIPS_pf* | DISTS_pf* |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        b = r["metrics"]["baseline_hard"]
        for v in order:
            m = r["metrics"].get(v)
            if not m:
                continue
            lines.append(f"| {r['name']} | {v} | {m['tof_truemotion']:.4f} | "
                         f"{pct(m['tof_truemotion'], b['tof_truemotion']):+.1f}% | {m['dF']:.3f} | "
                         f"{m['lpips_pf']:.4f} | {m['dists_pf']:.4f} |")
    lines.append("")
    lines.append("(*) LPIPS_pf/DISTS_pf are the CONFOUNDED R12-E1 arbiter: the gate's fallback "
                 "source is the per-frame SR cache that is also the reference, so they fall "
                 "monotonically with distrust by construction.")
    agg = {}
    for r in rows:
        b = r["metrics"]["baseline_hard"]
        for v in order[1:5]:
            if v in r["metrics"]:
                agg.setdefault(v, []).append(pct(r["metrics"][v]["tof_truemotion"],
                                                 b["tof_truemotion"]))
    lines.append("mean dtOF_true over windows: " + ", ".join(
        f"{v} {sum(a)/len(a):+.1f}%" for v, a in agg.items()))
    return lines


def table_compose(rows):
    """phase 4: gate x agent-C stabiliser composition (tOF is the shared currency)."""
    lines = ["| window | variant | tOF_true | d vs hard | LPIPS(x4) | DISTS(x4) | dF |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        b = r["metrics"]["baseline_hard"]
        for v, m in r["metrics"].items():
            lines.append(f"| {r['window']} | {v} | {m['tof_truemotion']:.4f} | "
                         f"{pct(m['tof_truemotion'], b['tof_truemotion']):+.1f}% | "
                         f"{m['lpips_x4']:.4f} | {m['dists_x4']:.4f} | {m['dF']:.3f} |")
    return lines


def table_hyst(rows):
    lines = ["| window | variant | tOF_true | d vs gate_res | LPIPS(x4) | d | DISTS(x4) | d |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r["metrics"]["gate_res"]
        for v, m in r["metrics"].items():
            lines.append(f"| {r['window']} | {v} | {m['tof_truemotion']:.4f} | "
                         f"{pct(m['tof_truemotion'], g['tof_truemotion']):+.1f}% | "
                         f"{m['lpips_x4']:.4f} | {pct(m['lpips_x4'], g['lpips_x4']):+.1f}% | "
                         f"{m['dists_x4']:.4f} | {pct(m['dists_x4'], g['dists_x4']):+.1f}% |")
    return lines


def main():
    gt = load("phase2")
    real = load("phase2b")
    md = ["# R13-E2 summary tables (generated by aggregate.py)", ""]
    if gt:
        md += ["## Phase 2 -- degrade-and-restore, TRUE-GT arbiter", ""] + table_gt(gt) + [""]
        md += ["### deltas vs baseline_hard (negative = better)", ""] + deltas_gt(gt) + [""]
    if real:
        md += ["## Phase 2b -- real footage, honest tOF arbiter", ""] + table_real(real) + [""]
    comp = load("phase4")
    if comp:
        md += ["## Phase 4 -- composition with R13-E3 (agent C) stabiliser M2", ""] + \
              table_compose(comp) + [""]
    hyst = load("phase5")
    if hyst:
        md += ["## Phase 5 -- temporal hysteresis on the reliability membership", ""] + \
              table_hyst(hyst) + [""]
    open(os.path.join(OUT, "summary_tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())
