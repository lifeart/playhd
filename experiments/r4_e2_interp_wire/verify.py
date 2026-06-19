#!/usr/bin/env python3
"""R4-E2 verification -- prove the SHIPPABLE interp_pass reproduces R3-E1's quality + both
ship-blockers, and measure cost on MPS.

  PART 1  QUALITY (held-out real frame, EXACT LR ground truth): reconstruct a held-out real
          frame from its real neighbours using interp_pass.blend_intra_hole_np (the wire's
          INTRA-HOLE-ONLY routing, ship-blocker #1 -- NO Ruder/reactive). Score MV-blend vs
          frame-dup and linear-blend (PSNR/SSIM/tOF). This is R3-E1's protocol driven THROUGH
          the shipped blend code, so the margins must match R3-E1.
  PART 2  SCENE-CUT GUARD (ship-blocker #2): show intra_hole_frac > 0.5 -> midpoint_* returns a
          DUPLICATE (no ghost), and a within-scene field interpolates.
  PART 3  COST on MPS: interp_pass.midpoint_torch ms/inserted-frame at the R3-E1 res (1920x960,
          scale 3 -> reproduce ~17 ms) AND the shipped instant res (1280x640, scale 2).

READ-ONLY imports of prototype/ + the wire. GPU used only in PART 3 (short burst, freed).
"""
import gc
import json
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "prototype"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import derisk as D          # noqa: E402  READ-ONLY prototype
import interp_pass as W     # noqa: E402  the shippable wire under test

CLIP = os.path.join(_REPO, "sample.mp4")


def _free_gpu():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        print(f"  [warn] mps empty_cache failed: {e}")


def _meanmv(mvs):
    ms = mvs["motion_scale"].astype(np.float32); ms[ms == 0] = 1.0
    mx = mvs["motion_x"].astype(np.float32) / ms
    my = mvs["motion_y"].astype(np.float32) / ms
    return float(np.sqrt(mx * mx + my * my).mean())


# --------------------------------------------------------------------------- #
# PART 1 -- held-out quality through the wire's intra-hole-only blend
# --------------------------------------------------------------------------- #
def eval_fwdbwd(frames):
    """All-P / consecutive-backbone: reconstruct real P-frame o from o-1 (+MV_o) and o+1 (-MV_o+1)."""
    h, w = frames[0][1].shape[:2]
    seqs = {m: [] for m in ("gt", "dup", "lin", "blend")}
    rows = []
    for o in range(1, len(frames) - 1):
        pt, lr_o, mvs_o = frames[o]
        ptn, lr_n, mvs_n = frames[o + 1]
        lr_p = frames[o - 1][1]
        if pt != "P" or ptn != "P" or mvs_o is None or mvs_n is None:
            continue
        fx_o, fy_o = D.build_lr_flow(mvs_o, h, w, want="past")
        fx_n, fy_n = D.build_lr_flow(mvs_n, h, w, want="past")
        fwd = D.warp_lr(lr_p, fx_o, fy_o)
        bwd = D.warp_lr(lr_n, -fx_n, -fy_n)
        hole_f = ~np.isfinite(fx_o)
        hole_b = ~np.isfinite(fx_n)
        lin = cv2.addWeighted(lr_p, 0.5, lr_n, 0.5, 0)
        blend = W.blend_intra_hole_np(fwd, bwd, hole_f, hole_b, lin)
        cand = {"dup": lr_p, "lin": lin, "blend": blend, "gt": lr_o}
        for m in seqs:
            seqs[m].append(cand[m])
        rows.append({"frame": o, "mv": _meanmv(mvs_o), "hole_f": float(hole_f.mean())})
    return rows, seqs


def eval_bidir(frames):
    """B-pyramid: reconstruct real B from its past & future ANCHORS via its own source<>0 MVs."""
    h, w = frames[0][1].shape[:2]
    backbone = [i for i, (pt, _, _) in enumerate(frames) if pt in ("I", "P")]
    seqs = {m: [] for m in ("gt", "dup", "lin", "blend")}
    rows = []
    for i, (pt, lr_i, mvs) in enumerate(frames):
        if pt != "B" or mvs is None:
            continue
        pa = max([b for b in backbone if b < i], default=None)
        fa = min([b for b in backbone if b > i], default=None)
        if pa is None or fa is None:
            continue
        lr_pa, lr_fa = frames[pa][1], frames[fa][1]
        fxp, fyp = D.build_lr_flow(mvs, h, w, want="past")
        fxf, fyf = D.build_lr_flow(mvs, h, w, want="future")
        fwd = D.warp_lr(lr_pa, fxp, fyp)
        bwd = D.warp_lr(lr_fa, fxf, fyf)
        hole_f = ~np.isfinite(fxp)
        hole_b = ~np.isfinite(fxf)
        dp, df = (i - pa), (fa - i)
        a_p, a_f = df / (dp + df), dp / (dp + df)
        lin = cv2.addWeighted(lr_pa, a_p, lr_fa, a_f, 0)
        blend = W.blend_intra_hole_np(fwd, bwd, hole_f, hole_b, lin)
        cand = {"dup": lr_pa, "lin": lin, "blend": blend, "gt": lr_i}
        for m in seqs:
            seqs[m].append(cand[m])
        rows.append({"frame": i, "mv": _meanmv(mvs), "hole_f": float(hole_f.mean())})
    return rows, seqs


def _score(seqs):
    out = {}
    for m in ("dup", "lin", "blend"):
        ps = [D.psnr(c, g) for c, g in zip(seqs[m], seqs["gt"])]
        ss = [D.ssim(c, g) for c, g in zip(seqs[m], seqs["gt"])]
        out[m] = (float(np.mean(ps)), float(np.mean(ss)), D.tof(seqs[m], seqs["gt"]))
    return out


def quality_window(start, n, proto, tag):
    frames = D.decode_lr_and_mvs(CLIP, start, n)
    rows, seqs = (eval_fwdbwd(frames) if proto == "fwdbwd" else eval_bidir(frames))
    if not rows:
        print(f"  {tag}: no usable held-out frames")
        return None
    sc = _score(seqs)
    mv = np.mean([r["mv"] for r in rows]); hole = np.mean([r["hole_f"] for r in rows])
    best_triv = max(sc["dup"][0], sc["lin"][0])
    gain = sc["blend"][0] - best_triv
    gain_ssim = sc["blend"][1] - max(sc["dup"][1], sc["lin"][1])
    print(f"\n  {tag}  (start={start} n={len(rows)} |MV|={mv:.2f}px hole={100*hole:.0f}%)")
    print(f"    {'method':<14}{'PSNR':>8}{'SSIM':>9}{'tOF':>9}")
    for m, name in (("dup", "frame-dup"), ("lin", "linear-blend"), ("blend", "MV-blend(wire)")):
        p, s, t = sc[m]
        print(f"    {name:<14}{p:>8.2f}{s:>9.4f}{t:>9.3f}")
    print(f"    => MV-blend vs best trivial: {gain:+.2f} dB PSNR, {gain_ssim:+.4f} SSIM, "
          f"tOF {sc['blend'][2]:.3f} vs dup {sc['dup'][2]:.3f}")
    return {"tag": tag, "start": start, "n": len(rows), "mv_px": round(float(mv), 2),
            "hole_pct": round(100 * float(hole), 1),
            "psnr": {m: round(sc[m][0], 2) for m in sc},
            "ssim": {m: round(sc[m][1], 4) for m in sc},
            "tof": {m: round(sc[m][2], 3) for m in sc},
            "gain_db": round(gain, 2), "gain_ssim": round(gain_ssim, 4)}


# --------------------------------------------------------------------------- #
# PART 2 -- scene-cut guard
# --------------------------------------------------------------------------- #
def guard_tests(scale=2):
    print(f"\n{'='*72}\nPART 2 -- SCENE-CUT GUARD (intra-hole frac > {W.CUT_THRESH} -> duplicate)")
    # (a) A real I-frame starts a scene/chunk and carries NO past MV -> 100% intra-hole ->
    #     the cross-cut midpoint MUST duplicate (this is the automatic chunk-boundary guard).
    frames = D.decode_lr_and_mvs(CLIP, 0, 2)                # frame 0 is the clip's I-frame
    h, w = frames[0][1].shape[:2]
    fx, fy = W.connecting_flow(frames, 1, h, w)             # connect frame0 -> frame1
    # Force the "first frame of a fresh scene is an I-frame" case: an I-frame's OWN past field.
    fxi, fyi = D.build_lr_flow(frames[0][2], h, w, want="past")
    hf_i = W.intra_hole_frac(fxi)
    left = np.zeros((h * scale, w * scale, 3), np.uint8) + 40
    right = np.zeros((h * scale, w * scale, 3), np.uint8) + 200   # very different -> ghost if blended
    mid_cut, info_cut = W.midpoint_numpy(left, right, fxi, fyi, scale)
    dup_ok = info_cut["duplicated"] and np.array_equal(mid_cut, left)
    print(f"  (a) cross-cut (I-frame, hole={hf_i*100:.0f}%): duplicated={info_cut['duplicated']} "
          f"==left={np.array_equal(mid_cut, left)}  -> {'PASS' if dup_ok else 'FAIL'} (no ghost)")

    # (b) A genuine high-motion frame whose past-hole > 0.5 also duplicates (saw 0.52-0.87 @30000).
    fr2 = D.decode_lr_and_mvs(CLIP, 30000, 12)
    h2, w2 = fr2[0][1].shape[:2]
    fired = []
    for t in range(1, len(fr2)):
        fxx, fyy = W.connecting_flow(fr2, t, h2, w2)
        hf = W.intra_hole_frac(fxx)
        L = np.zeros((h2 * scale, w2 * scale, 3), np.uint8) + 30
        Rr = np.zeros((h2 * scale, w2 * scale, 3), np.uint8) + 220
        _, inf = W.midpoint_numpy(L, Rr, fxx, fyy, scale)
        fired.append((round(hf, 2), inf["duplicated"]))
    consistent = all((hf > W.CUT_THRESH) == dup for hf, dup in fired)
    n_dup = sum(d for _, d in fired)
    print(f"  (b) high-motion pairs (hole,dup): {fired}")
    print(f"      guard fired on {n_dup}/{len(fired)} (hole>0.5); rule consistent: "
          f"{'PASS' if consistent else 'FAIL'}")

    # (c) within-scene low-motion field -> NOT duplicated, genuine interpolation.
    fr3 = D.decode_lr_and_mvs(CLIP, 5000, 12)
    h3, w3 = fr3[0][1].shape[:2]
    # pick a pair with low intra-hole
    chosen = None
    for t in range(1, len(fr3)):
        fxx, fyy = W.connecting_flow(fr3, t, h3, w3)
        if W.intra_hole_frac(fxx) <= W.CUT_THRESH:
            chosen = (t, fxx, fyy, W.intra_hole_frac(fxx)); break
    t, fxx, fyy, hf = chosen
    L = np.zeros((h3 * scale, w3 * scale, 3), np.uint8) + 30
    Rr = np.zeros((h3 * scale, w3 * scale, 3), np.uint8) + 220
    _, inf = W.midpoint_numpy(L, Rr, fxx, fyy, scale)
    ok_c = (not inf["duplicated"])
    print(f"  (c) within-scene pair t={t} (hole={hf*100:.0f}%): duplicated={inf['duplicated']} "
          f"-> {'PASS' if ok_c else 'FAIL'} (interpolates)")
    return {"cut_dup": dup_ok, "highmotion_rule_consistent": consistent,
            "highmotion_n_dup": n_dup, "highmotion_pairs": fired, "withinscene_interp": ok_c}


# --------------------------------------------------------------------------- #
# PART 3 -- cost on MPS (the shipped midpoint_torch)
# --------------------------------------------------------------------------- #
def cost_mps(scale, n_time=14):
    try:
        import torch
        import gpu_ops as G
    except Exception as e:
        return {"err": str(e)}
    frames = D.decode_lr_and_mvs(CLIP, 30000, max(28, n_time + 2))
    h, w = frames[0][1].shape[:2]
    w_hd, h_hd = w * scale, h * scale
    # consecutive pairs whose connecting 'past' field passes the guard (so we time real warps)
    pairs = []
    for o in range(1, len(frames)):
        fx, fy = W.connecting_flow(frames, o, h, w)
        if W.intra_hole_frac(fx) <= W.CUT_THRESH and frames[o][2] is not None:
            pairs.append((o, fx, fy))
        if len(pairs) >= n_time + 1:
            break
    # warm
    o0, fx0, fy0 = pairs[0]
    L0 = G.img_to_dev(cv2.resize(frames[o0 - 1][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC))
    R0 = G.img_to_dev(cv2.resize(frames[o0][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC))
    W.midpoint_torch(L0, R0, fx0, fy0, scale, _G=G)
    G.sync()
    ts = []
    for o, fx, fy in pairs:
        L = G.img_to_dev(cv2.resize(frames[o - 1][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC))
        R = G.img_to_dev(cv2.resize(frames[o][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC))
        G.sync(); t0 = time.perf_counter()
        mid, info = W.midpoint_torch(L, R, fx, fy, scale, _G=G)
        G.sync(); ts.append((time.perf_counter() - t0) * 1000)
    _free_gpu()
    return {"ms": float(np.median(ts)), "n": len(ts), "hd": f"{w_hd}x{h_hd}", "scale": scale}


def main():
    print(f"{'='*72}\nR4-E2 -- verify shippable interp_pass reproduces R3-E1 + ship-blockers")
    print(f"{'='*72}\nPART 1 -- QUALITY (held-out real frame, exact LR GT; wire's intra-hole blend)")
    windows = [
        (0,     40, "fwdbwd", "intro-chaotic"),
        (5000,  40, "bidir",  "talking-head"),
        (12000, 40, "bidir",  "moderate"),
        (30000, 40, "bidir",  "high-motion"),
    ]
    q = [r for r in (quality_window(*w) for w in windows) if r]

    g = guard_tests()

    print(f"\n{'='*72}\nPART 3 -- COST on MPS (interp_pass.midpoint_torch, ms/inserted-frame)")
    c3 = cost_mps(3)   # R3-E1 res 1920x960
    c2 = cost_mps(2)   # shipped instant res 1280x640
    for c in (c3, c2):
        if "ms" in c:
            print(f"  {c['hd']} (scale {c['scale']}): {c['ms']:.2f} ms/inserted-frame (n={c['n']})")
        else:
            print(f"  cost unavailable: {c['err']}")

    out = {"quality": q, "guard": g, "cost_scale3_1920x960": c3, "cost_scale2_1280x640": c2}
    with open(os.path.join(_HERE, "verify_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  wrote {os.path.join(_HERE, 'verify_results.json')}")


if __name__ == "__main__":
    main()
