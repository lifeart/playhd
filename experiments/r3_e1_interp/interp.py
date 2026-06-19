#!/usr/bin/env python3
"""R3-E1: MV-reuse motion-compensated frame interpolation, honestly measured.

Reuses prototype/derisk.py (READ-ONLY): build_lr_flow (codec MV -> dense LR fetch-flow),
warp_lr / warp_hd, occlusion_mask_lr, psnr/ssim/tof. No new optical flow is computed.

QUALITY (PSNR/SSIM/tOF vs EXACT ground truth at LR -- the decoded frame IS truth at LR):
  We synthesize a HELD-OUT REAL middle frame from its two real neighbours using only codec
  MVs, then score against the real frame. Two protocols (chosen by the window's GOP):

  * fwdbwd  (high-motion, all-P): reconstruct real frame o from o-1 and o+1.
      fwd = warp_lr(LR_{o-1}, +MV_o)        # o's own past MV: forward fetch  (codec-trusted)
      bwd = warp_lr(LR_{o+1}, -MV_{o+1})    # o+1's past MV negated: inverse fetch (the hard dir)
    Per-warp motion is ~ONE full codec step => CONSERVATIVE vs deployment's half-step (2x motion).

  * bidir   (talking-head, B-pyramid): reconstruct real B-frame from its past & future anchors
      fwd = warp_lr(LR_pastAnchor,  +MV_past)   # B's own source<0 MV
      bwd = warp_lr(LR_futureAnchor,+MV_future) # B's own source>0 MV
    This is exactly playhd PASS-2 bidirectional MC. Uses the frame's own MVs (optimistic) but
    anchors are several frames away (pessimistic baseline) -> the two biases offset.

  Methods scored: dup-prev, dup-next, linear (no-MV blend = the trivial baseline), MV-fwd,
  MV-bwd, MV-blend (occlusion-aware: intra-hole + fwd-bwd reuse), and MV-blend-occf (also
  uses the decoded-LR reactive mask -- valid where the real frame exists).

COST: ms/frame for the interpolation pass at HD (the real 1920x960 output), numpy-cv2 path,
reported as ratios vs a single per-frame upscale. (torch/MPS shared with siblings -> kept off
the timing hot loop; numpy-cv2 is the contention-free, reproducible number.)
"""
import argparse
import os
import sys
import time
import numpy as np
import cv2

PROTO = os.path.join(os.path.dirname(__file__), "..", "..", "prototype")
sys.path.insert(0, PROTO)
import derisk  # noqa: E402  READ-ONLY

CLIP = os.path.join(os.path.dirname(__file__), "..", "..", "sample.mp4")
OUT = os.path.dirname(__file__)


# --------------------------------------------------------------------------- #
# Occlusion-aware blend of a forward and a backward MC estimate (reuses derisk)
# --------------------------------------------------------------------------- #
def blend_occ(fwd, bwd, hole_f, hole_b, src_lin, ruder_f=None, ruder_b=None):
    """Combine two motion-compensated estimates into one midpoint frame.
    A direction is 'dead' where it has an intra hole (no MV) OR is fwd-bwd-flagged (Ruder).
    Rules: both alive -> average; one alive -> that one; both dead -> linear source blend.
    Returns uint8."""
    f = fwd.astype(np.float32)
    b = bwd.astype(np.float32)
    dead_f = hole_f.copy()
    dead_b = hole_b.copy()
    if ruder_f is not None:
        dead_f = dead_f | ruder_f
    if ruder_b is not None:
        dead_b = dead_b | ruder_b
    alive_f = ~dead_f
    alive_b = ~dead_b
    out = 0.5 * (f + b)
    both_dead = dead_f & dead_b
    out[alive_f & dead_b] = f[alive_f & dead_b]
    out[dead_f & alive_b] = b[dead_f & alive_b]
    out[both_dead] = src_lin.astype(np.float32)[both_dead]
    return np.clip(out, 0, 255).astype(np.uint8)


def ruder_mask(fx, fy, lr_cur, lr_ref):
    """Target-free-ish unreliability via the project's fwd-bwd splat (Ruder) + intra holes.
    We pass mode='full' (intra + reactive + fwd-bwd). For new-timestamp use the reactive term
    is unavailable; here lr_cur is the decoded frame (valid for the held-out real-frame test)."""
    occ, _ = derisk.occlusion_mask_lr(fx, fy, lr_cur, lr_ref, mode="full")
    return occ


# --------------------------------------------------------------------------- #
# Protocol A: high-motion all-P -> reconstruct real frame o from o-1, o+1
# --------------------------------------------------------------------------- #
def eval_fwdbwd(frames, use_occf):
    h, w = frames[0][1].shape[:2]
    rows = []
    seqs = {m: [] for m in ("gt", "dup", "lin", "fwd", "bwd", "blend")}
    for o in range(1, len(frames) - 1):
        pt, lr_o, mvs_o = frames[o]
        ptn, lr_n, mvs_n = frames[o + 1]
        lr_p = frames[o - 1][1]
        if pt != "P" or ptn != "P" or mvs_o is None or mvs_n is None:
            continue
        fx_o, fy_o = derisk.build_lr_flow(mvs_o, h, w, want="past")     # o <- o-1
        fx_n, fy_n = derisk.build_lr_flow(mvs_n, h, w, want="past")     # o+1 <- o
        fwd = derisk.warp_lr(lr_p, fx_o, fy_o)                          # forward fetch (trusted)
        bwd = derisk.warp_lr(lr_n, -fx_n, -fy_n)                        # inverse fetch (hard)
        hole_f = ~np.isfinite(fx_o)
        hole_b = ~np.isfinite(fx_n)
        lin = cv2.addWeighted(lr_p, 0.5, lr_n, 0.5, 0)
        rf = rb = None
        if use_occf:
            rf = ruder_mask(fx_o, fy_o, lr_o, lr_p)          # decoded LR available => valid
            rb = ruder_mask(fx_n, fy_n, lr_n, lr_o)
        blend = blend_occ(fwd, bwd, hole_f, hole_b, lin, rf, rb)
        cand = {"dup": lr_p, "lin": lin, "fwd": fwd, "bwd": bwd, "blend": blend, "gt": lr_o}
        for m in seqs:
            seqs[m].append(cand[m])
        row = {"frame": o, "type": pt,
               "mv_meanpx": round(_meanmv(mvs_o), 2),
               "hole_f%": round(100 * hole_f.mean(), 1)}
        for m in ("dup", "lin", "fwd", "bwd", "blend"):
            row[f"psnr_{m}"] = round(derisk.psnr(cand[m], lr_o), 2)
            row[f"ssim_{m}"] = round(derisk.ssim(cand[m], lr_o), 4)
        rows.append(row)
    return rows, seqs


# --------------------------------------------------------------------------- #
# Protocol B: talking-head B-pyramid -> reconstruct real B from its anchors
# --------------------------------------------------------------------------- #
def eval_bidir(frames, use_occf):
    h, w = frames[0][1].shape[:2]
    backbone = [i for i, (pt, _, _) in enumerate(frames) if pt in ("I", "P")]
    rows = []
    seqs = {m: [] for m in ("gt", "dup", "lin", "fwd", "bwd", "blend")}
    for i, (pt, lr_i, mvs) in enumerate(frames):
        if pt != "B" or mvs is None:
            continue
        pa = max([b for b in backbone if b < i], default=None)
        fa = min([b for b in backbone if b > i], default=None)
        if pa is None or fa is None:
            continue
        lr_pa, lr_fa = frames[pa][1], frames[fa][1]
        fxp, fyp = derisk.build_lr_flow(mvs, h, w, want="past")        # B <- past anchor
        fxf, fyf = derisk.build_lr_flow(mvs, h, w, want="future")      # B <- future anchor
        fwd = derisk.warp_lr(lr_pa, fxp, fyp)
        bwd = derisk.warp_lr(lr_fa, fxf, fyf)
        hole_f = ~np.isfinite(fxp)
        hole_b = ~np.isfinite(fxf)
        dp, df = (i - pa), (fa - i)
        a_p, a_f = df / (dp + df), dp / (dp + df)                       # temporal-distance weight
        lin = cv2.addWeighted(lr_pa, a_p, lr_fa, a_f, 0)
        rf = rb = None
        if use_occf:
            rf = ruder_mask(fxp, fyp, lr_i, lr_pa)
            rb = ruder_mask(fxf, fyf, lr_i, lr_fa)
        blend = blend_occ(fwd, bwd, hole_f, hole_b, lin, rf, rb)
        cand = {"dup": lr_pa, "lin": lin, "fwd": fwd, "bwd": bwd, "blend": blend, "gt": lr_i}
        for m in seqs:
            seqs[m].append(cand[m])
        row = {"frame": i, "type": pt, "mv_meanpx": round(_meanmv(mvs), 2),
               "hole_f%": round(100 * hole_f.mean(), 1)}
        for m in ("dup", "lin", "fwd", "bwd", "blend"):
            row[f"psnr_{m}"] = round(derisk.psnr(cand[m], lr_i), 2)
            row[f"ssim_{m}"] = round(derisk.ssim(cand[m], lr_i), 4)
        rows.append(row)
    return rows, seqs


def _meanmv(mvs):
    ms = mvs["motion_scale"].astype(np.float32); ms[ms == 0] = 1.0
    mx = mvs["motion_x"].astype(np.float32) / ms
    my = mvs["motion_y"].astype(np.float32) / ms
    return float(np.sqrt(mx * mx + my * my).mean())


def _agg(rows, methods):
    out = {}
    for m in methods:
        out[m] = (float(np.mean([r[f"psnr_{m}"] for r in rows])),
                  float(np.mean([r[f"ssim_{m}"] for r in rows])))
    return out


def _tofs(seqs):
    """tOF of each candidate doubled-sequence vs the real sequence (lower=smoother/truer motion)."""
    ref = seqs["gt"]
    out = {}
    for m in ("dup", "lin", "blend"):
        # build [real_prev? ] -- here seqs are the synthesized middle frames in order; tof needs
        # a temporal sequence. Use the sequence of synthesized frames vs real frames directly.
        out[m] = derisk.tof(seqs[m], ref)
    return out


# --------------------------------------------------------------------------- #
# HD half-step generation (deployment shape) + cost timing
# --------------------------------------------------------------------------- #
def hd_cost_and_samples(frames, scale, n_time=12):
    """Generate true HALF-STEP midpoints between consecutive real frames at HD (the deployment
    operation: 0.5x MV dual warp + blend), time it, and dump a couple of visual samples.
    Uses bicubic as the HD SR placeholder (isolates the interpolation cost/artifacts, matching
    derisk's convention). Returns dict of timings (ms)."""
    h, w = frames[0][1].shape[:2]
    w_hd, h_hd = w * scale, h * scale
    # find consecutive P pairs for clean half-step demo
    pairs = [o for o in range(1, len(frames))
             if frames[o][0] == "P" and frames[o - 1][0] in ("I", "P") and frames[o][2] is not None]
    t_warp = []
    t_total = []
    sample_done = 0
    for o in pairs[:n_time]:
        lr_prev, lr_cur, mvs = frames[o - 1][1], frames[o][1], frames[o][2]
        hd_prev = cv2.resize(lr_prev, (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
        hd_cur = cv2.resize(lr_cur, (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
        fx, fy = derisk.build_lr_flow(mvs, h, w, want="past")          # cur <- prev
        t0 = time.perf_counter()
        # half-step: midpoint pm. fwd from prev: pm -> prev at +0.5*fx ; bwd from cur: pm->cur at -0.5*fx
        wf, hole_f = derisk.warp_hd(hd_prev, 0.5 * fx, 0.5 * fy, scale)
        wb, hole_b = derisk.warp_hd(hd_cur, -0.5 * fx, -0.5 * fy, scale)
        t1 = time.perf_counter()
        lin = cv2.addWeighted(hd_prev, 0.5, hd_cur, 0.5, 0)
        mid = 0.5 * (wf.astype(np.float32) + wb.astype(np.float32))
        dead = hole_f & hole_b
        mid[dead] = lin.astype(np.float32)[dead]
        mid = np.clip(mid, 0, 255).astype(np.uint8)
        t2 = time.perf_counter()
        t_warp.append((t1 - t0) * 1000)
        t_total.append((t2 - t0) * 1000)
        if sample_done < 2:
            trip = np.concatenate([
                derisk._label(hd_prev, "real t"),
                derisk._label(mid, "MV-interp t+0.5"),
                derisk._label(hd_cur, "real t+1")], axis=1)
            cv2.imwrite(os.path.join(OUT, f"halfstep_{o:03d}.png"),
                        cv2.cvtColor(trip, cv2.COLOR_RGB2BGR))
            linmid = np.concatenate([
                derisk._label(lin, "linear-blend t+0.5 (no MV)"),
                derisk._label(mid, "MV-interp t+0.5")], axis=1)
            cv2.imwrite(os.path.join(OUT, f"halfstep_{o:03d}_vs_linear.png"),
                        cv2.cvtColor(linmid, cv2.COLOR_RGB2BGR))
            sample_done += 1
    return {"warp_ms": float(np.median(t_warp)), "total_ms": float(np.median(t_total)),
            "n": len(t_warp), "hd": (w_hd, h_hd)}


# --------------------------------------------------------------------------- #
def run_window(start, n, proto, tag, scale, use_occf):
    frames = derisk.decode_lr_and_mvs(CLIP, start, n)
    types = "".join(f[0] for f in frames)
    print(f"\n{'='*78}\n{tag}  start={start} n={len(frames)} proto={proto}\n  types={types}\n{'='*78}")
    if proto == "fwdbwd":
        rows, seqs = eval_fwdbwd(frames, use_occf)
    else:
        rows, seqs = eval_bidir(frames, use_occf)
    if not rows:
        print("  (no usable held-out frames)")
        return
    methods = ["dup", "lin", "fwd", "bwd", "blend"]
    agg = _agg(rows, methods)
    print(f"\n  held-out frames scored: {len(rows)}   mean codec |MV|: "
          f"{np.mean([r['mv_meanpx'] for r in rows]):.2f} px(LR)   "
          f"mean intra-hole: {np.mean([r['hole_f%'] for r in rows]):.1f}%")
    print(f"\n  {'method':<10}{'PSNR(dB)':>10}{'SSIM':>9}   note")
    notes = {"dup": "frame duplication (trivial)", "lin": "linear blend, NO MV (trivial)",
             "fwd": "MV forward warp only", "bwd": "MV backward warp only (inverse/future)",
             "blend": "MV occlusion-aware blend (intra-hole" + ("+Ruder)" if use_occf else ")")}
    for m in methods:
        p, s = agg[m]
        print(f"  {m:<10}{p:>10.2f}{s:>9.4f}   {notes[m]}")
    tofs = _tofs(seqs)
    print(f"\n  tOF (vs real GT, lower=truer motion):  dup={tofs['dup']:.3f}  "
          f"linear={tofs['lin']:.3f}  MV-blend={tofs['blend']:.3f}")
    # write csv
    import csv
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, f"metrics_{tag}.csv"), "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=keys); wcsv.writeheader(); wcsv.writerows(rows)
    # gain vs trivial baselines
    best_triv = max(agg["dup"][0], agg["lin"][0])
    print(f"\n  => MV-blend vs best trivial baseline: {agg['blend'][0]-best_triv:+.2f} dB PSNR, "
          f"{agg['blend'][1]-max(agg['dup'][1],agg['lin'][1]):+.4f} SSIM")
    return agg, tofs


def hd_cost_torch(frames, scale, n_time=12):
    """Deployment-realistic cost: the half-step pass on torch/MPS (gpu_ops.warp_hd). GPU is
    shared with siblings, so this is a short burst reported as a ratio; freed immediately."""
    try:
        import torch
        import gpu_ops as G
    except Exception as e:
        return {"err": str(e)}
    h, w = frames[0][1].shape[:2]
    w_hd, h_hd = w * scale, h * scale
    pairs = [o for o in range(1, len(frames)) if frames[o][2] is not None][:n_time]
    G.warp_hd(G.img_to_dev(np.zeros((h_hd, w_hd, 3), np.uint8)),
              *G.flow_to_dev(*derisk.build_lr_flow(frames[pairs[0]][2], h, w, "past")), scale)  # warm
    G.sync()
    t = []
    for o in pairs:
        hd_prev = G.img_to_dev(cv2.resize(frames[o - 1][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC))
        hd_cur = G.img_to_dev(cv2.resize(frames[o][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC))
        fx, fy = G.flow_to_dev(*derisk.build_lr_flow(frames[o][2], h, w, "past"))
        G.sync(); t0 = time.perf_counter()
        wf, hf = G.warp_hd(hd_prev, 0.5 * fx, 0.5 * fy, scale)
        wb, hb = G.warp_hd(hd_cur, -0.5 * fx, -0.5 * fy, scale)
        mid = 0.5 * (wf + wb)
        dead = (hf & hb)[None, None]
        mid = torch.where(dead, 0.5 * (hd_prev + hd_cur), mid).clamp(0, 255)
        G.sync(); t.append((time.perf_counter() - t0) * 1000)
    del wf, wb, mid
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {"ms": float(np.median(t)), "n": len(t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--occf", action="store_true", help="add decoded-LR reactive/Ruder mask to blend")
    args = ap.parse_args()
    # representative spread: pathological intro, true talking-head, moderate, coherent high-motion
    windows = [
        (0,     "fwdbwd", "intro-chaotic(|MV|10px,9dB)"),
        (5000,  "bidir",  "talkinghead(|MV|0.6px)"),
        (12000, "bidir",  "moderate(|MV|1px)"),
        (30000, "bidir",  "highmotion-coherent(|MV|7px)"),
    ]
    summary = {}
    for start, proto, tag in windows:
        r = run_window(start, args.n, proto, tag, args.scale, args.occf)
        if r:
            summary[tag] = r
    # cost+samples: use the coherent high-motion window (start 30000 has consecutive-P runs =>
    # clean single-step half-step demos). Timing is content-independent (fixed pixel count).
    frames = derisk.decode_lr_and_mvs(CLIP, 30000, max(24, args.n))
    cost = hd_cost_and_samples(frames, args.scale)
    tcost = hd_cost_torch(frames, args.scale)
    print(f"\n{'='*78}\nHD COST (deployment half-step interpolation, {cost['hd'][0]}x{cost['hd'][1]} output):")
    print(f"  numpy-cv2:  warp(2x warp_hd)={cost['warp_ms']:.1f} ms   warp+blend={cost['total_ms']:.1f} ms/frame (n={cost['n']})")
    if "ms" in tcost:
        print(f"  torch/MPS:  warp+blend={tcost['ms']:.2f} ms/inserted-frame (n={tcost['n']}, GPU shared)")
    else:
        print(f"  torch/MPS:  unavailable ({tcost['err']})")
    print(f"  => one inserted frame: NO new optical flow (codec MV reused); only 2 warps + 1 blend")


if __name__ == "__main__":
    main()
