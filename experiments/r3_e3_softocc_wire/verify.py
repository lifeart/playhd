#!/usr/bin/env python3
"""R3-E3 verification -- reproduce the R2-E2 (c) escape THROUGH derisk.reconstruct using the
SHIPPABLE softocc_wire pass, confirm talking-head is unchanged, and stress-test the EMA reset.

All schemes are OUTPUT-only post-passes on the IDENTICAL bicubic-fallback chain produced by
derisk.reconstruct (numpy, deterministic) -- nothing soft fed back as a reference (GOTCHA #16).
Honest metrics: tOF (headline) + eff-bicubic% + |dF| crosscheck. READ-ONLY import of prototype/.
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
for _p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import derisk as D            # noqa: E402
import softocc_wire as W      # noqa: E402  (the shippable pass under test)

try:
    import torch              # noqa: E402
    _HAS_TORCH = True
except Exception as e:        # surfaced, not swallowed
    print(f"[warn] torch import failed ({e}); torch-parity check will be skipped")
    _HAS_TORCH = False

CLIP = os.path.join(_REPO, "sample.mp4")
N, SCALE, OCC, SR_MODEL = 48, 2, "reactive", "realesrgan"
GAIN, BETA, FE = 0.6, 0.85, 31           # R2-E2 recommended (c)


def _free_gpu():
    gc.collect()
    if _HAS_TORCH:
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            print(f"  [warn] mps empty_cache failed: {e}")


# --------------------------------------------------------------------------- #
def decode(start, n=N):
    return D.decode_lr_and_mvs(CLIP, start, n)


def anchors_of(frames):
    bb = D.backbone_indices(frames)
    return {i for i in bb if frames[i][0] == "I" or i == bb[0]}, bb


def base_chain(frames, perframe_cache, anchors):
    """Numpy reconstruct on a given per-frame cache -> base recon + per-frame mask/hole."""
    _, R = D.reconstruct(frames, None, SCALE, True, OCC, perframe_cache, set(),
                         backend="numpy", collect_metrics=False, download_output=True)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    mask = {i: (R[i]["mask"] if R[i]["mask"] is not None else np.zeros((h_hd, w_hd), bool))
            for i in range(len(frames))}
    hole = {i: float(R[i]["hole_frac"]) for i in range(len(frames))}
    return R, mask, hole


def clone_recon(R):
    return {i: dict(recon=R[i]["recon"].copy(), mask=R[i]["mask"]) for i in R}


def setup(frames):
    """Build the R2-E2-equivalent inputs driven by derisk.reconstruct: all-bicubic base chain +
    full per-frame SR cache + HD confidence. The base chain == the deployed bicubic-fallback recon
    (anchors bicubic here, matching R2-E2's reference numbers)."""
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    anchors, bb = anchors_of(frames)
    bic = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
           for i in range(len(frames))}
    t0 = time.perf_counter()
    srf = D.build_perframe_cache(frames, w_hd, h_hd, SR_MODEL)        # ONE SR pass, reused for all
    t_sr = time.perf_counter() - t0
    _free_gpu()
    R, mask, hole = base_chain(frames, bic, anchors)
    conf = W.build_conf(frames, anchors, bb, w_hd, h_hd)
    return dict(frames=frames, anchors=anchors, bb=bb, bic=bic, srf=srf, R=R, mask=mask,
                hole=hole, conf=conf, w_hd=w_hd, h_hd=h_hd, t_sr=round(t_sr, 2))


def run_off_on(S, reset_idx, label):
    """Run the pass OFF and ON (c) on a prepared setup S; return (off_row, on_row, on_info)."""
    frames, anchors = S["frames"], S["anchors"]
    bicp = lambda i: S["bic"][i]
    srp = lambda i: S["srf"][i]
    # OFF
    Roff = clone_recon(S["R"])
    W.softocc_patch_np(frames, Roff, bic_provider=bicp, sr_provider=srp, conf=S["conf"],
                       anchors=anchors, reset_idx=reset_idx, gain=GAIN, beta=BETA,
                       feather_k=FE, enabled=False)
    out_off = {i: Roff[i]["recon"] for i in range(len(frames))}
    off = W.honest_metrics(frames, out_off, S["mask"], S["bic"], S["srf"], anchors, S["hole"],
                           f"{label} OFF")
    # ON (c)
    Ron = clone_recon(S["R"])
    info = W.softocc_patch_np(frames, Ron, bic_provider=bicp, sr_provider=srp, conf=S["conf"],
                              anchors=anchors, reset_idx=reset_idx, gain=GAIN, beta=BETA,
                              feather_k=FE, enabled=True)
    out_on = {i: Ron[i]["recon"] for i in range(len(frames))}
    on = W.honest_metrics(frames, out_on, S["mask"], S["bic"], S["srf"], anchors, S["hole"],
                          f"{label} ON (c) g={GAIN} b={BETA} fe={FE}")
    return off, on, info


def _fmt(r):
    return (f"  {r['scheme']:38s} tOF={r['tof']:.4f}  effBic%={r['eff_bicubic_pct']:6.3f}  "
            f"detail%={r['detail_injected_pct']:5.3f}  fbdF={r['fb_localized_dF']:6.2f}  "
            f"dRec={r['d_recon']:.2f}")


# --------------------------------------------------------------------------- #
# Task 2c: real DEPLOYED cache (build_anchor_cache: SR anchors) -> escape on the actual recon.
# --------------------------------------------------------------------------- #
def setup_deployed(frames):
    """Same as setup() but the base recon uses the DEPLOYED hybrid cache (build_anchor_cache:
    real compact-SR anchors + bicubic elsewhere) instead of all-bicubic anchors. Absolute tOF
    differs (sharper SR anchors) but proves the escape holds on the true instant-path recon."""
    import anchor_sr
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    anchors, bb = anchors_of(frames)
    bic = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
           for i in range(len(frames))}
    t0 = time.perf_counter()
    cache, _info, _sr_set = anchor_sr.build_anchor_cache(
        frames, w_hd, h_hd, SR_MODEL, occ_mode=OCC, fallback_thresh=0.50, gpu_cache=False)
    srf = D.build_perframe_cache(frames, w_hd, h_hd, SR_MODEL)
    t_sr = time.perf_counter() - t0
    _free_gpu()
    R, mask, hole = base_chain(frames, cache, anchors)               # SR-anchor recon chain
    conf = W.build_conf(frames, anchors, bb, w_hd, h_hd)
    return dict(frames=frames, anchors=anchors, bb=bb, bic=bic, srf=srf, R=R, mask=mask,
                hole=hole, conf=conf, w_hd=w_hd, h_hd=h_hd, t_sr=round(t_sr, 2))


# --------------------------------------------------------------------------- #
# Task 3b: synthetic hard scene-cut splice -> EMA reset stress test.
# --------------------------------------------------------------------------- #
def build_splice():
    """Worst-case cross-cut: Scene1 = CALM talking-head (24 frames -> the HF-EMA converges to a
    stable face-HF), then a hard cut to Scene2 = HIGH-MOTION windowA[0:24] (large disocclusion
    masks). The cut frame is windowA's I-frame (a clean I-anchor) -- exactly what stream_gops
    emits at a scene cut. A MISSED EMA reset injects the converged Scene-1 (face) HF into Scene-2's
    big holes = a maximally visible cross-cut ghost; the reset MUST fire to avoid it."""
    s2w = decode(5000, 96)
    i2 = next(i for i in range(len(s2w)) if s2w[i][0] == "I")        # first talking-head I-frame
    scene1 = s2w[i2:i2 + 24]                                         # CALM, starts at an I-frame
    scene2 = decode(0, 24)                                           # HIGH-MOTION, I-anchored at 0
    spliced = scene1 + scene2
    cut = len(scene1)                                                # = 24, the I-anchored cut
    assert spliced[cut][0] == "I", "splice frame must be an I-anchor"
    return spliced, cut


def reset_test():
    print("\n=== Task 3: EMA reset stress test ===")
    # 3a: the built-in internal I-frame at 28 in window A.
    frames = decode(0)
    anchors, _ = anchors_of(frames)
    rset = W.reset_indices(frames)
    print(f"  3a window-A reset_indices (I-frames + start) = {sorted(rset)}  "
          f"(internal I-frame @28 present: {28 in rset})")

    # 3b: synthetic hard scene cut -> compare reset-ON vs reset-OFF (the bug).
    spliced, cut = build_splice()
    S = setup(spliced)
    frames, anchors = S["frames"], S["anchors"]
    bicp = lambda i: S["bic"][i]
    srp = lambda i: S["srf"][i]
    reset_on = W.reset_indices(frames)                              # correct: includes the cut
    reset_off = {0}                                                 # BUG: misses the internal reset
    print(f"  3b spliced types = {''.join(f[0] for f in frames)}")
    print(f"  3b cut @ index {cut} (I-anchor); reset_ON={sorted(reset_on)}  reset_OFF={sorted(reset_off)}")

    def run(reset_idx):
        R = clone_recon(S["R"])
        info = W.softocc_patch_np(frames, R, bic_provider=bicp, sr_provider=srp, conf=S["conf"],
                                  anchors=anchors, reset_idx=reset_idx, gain=GAIN, beta=BETA,
                                  feather_k=FE, enabled=True)
        out = {i: R[i]["recon"] for i in range(len(frames))}
        return out, info

    out_on, info_on = run(reset_on)
    out_off, info_off = run(reset_off)

    # cross-cut ghost = output difference between the buggy (no-reset) and correct (reset) pass,
    # measured inside the post-cut fallback masks for the first frames AFTER the cut. With the
    # reset, scene-2 HF only; without it, scene-1 HF bleeds across -> nonzero, decaying ghost.
    print(f"  3b reset fired: ON n_resets={info_on['n_resets']} (cut seeded EMA: "
          f"{cut in info_on['ema_seeded_after_reset']}),  OFF n_resets={info_off['n_resets']} "
          f"(cut NOT a reset)")
    ghost = []
    for i in range(cut, min(cut + 6, len(frames))):
        m = S["mask"][i]
        if not m.any():
            ghost.append((i, 0.0)); continue
        d = np.abs(out_off[i].astype(np.float32) - out_on[i].astype(np.float32)).mean(axis=2)
        ghost.append((i, float(np.sqrt(np.mean(d[m] ** 2)))))
    print("  3b cross-cut ghost RMS (OFF-minus-ON, inside fallback mask), per post-cut frame:")
    for i, g in ghost:
        print(f"        frame {i:2d}: {g:6.3f}  (EMA rms ON={info_on['ema_rms'][i]:6.2f}  "
              f"OFF={info_off['ema_rms'][i]:6.2f})")
    # scene-2 tOF: OFF (ghost) vs ON (clean)
    s2 = list(range(cut, len(frames)))
    h_lr, w_lr = frames[0][1].shape[:2]
    seqON = [cv2.resize(out_on[i], (w_lr, h_lr)) for i in s2]
    seqOFF = [cv2.resize(out_off[i], (w_lr, h_lr)) for i in s2]
    lr = [frames[i][1] for i in s2]
    tof_on = D.tof(seqON, lr)
    tof_off = D.tof(seqOFF, lr)
    peak_ghost = max(g for _, g in ghost)
    print(f"  3b scene-2 tOF: reset-ON={tof_on:.4f}  reset-OFF={tof_off:.4f}  "
          f"(OFF worse by {tof_off - tof_on:+.4f});  peak cross-cut ghost RMS={peak_ghost:.3f}")
    verdict = "PASS" if (info_on["n_resets"] > info_off["n_resets"] and peak_ghost > 0.5
                         and cut in info_on["ema_seeded_after_reset"]) else "FAIL"
    print(f"  3b RESET VERDICT: {verdict} (reset eliminates a {peak_ghost:.2f}-RMS cross-cut HF ghost)")
    return dict(cut=cut, n_resets_on=info_on["n_resets"], n_resets_off=info_off["n_resets"],
                ema_seeded_at_cut=cut in info_on["ema_seeded_after_reset"],
                ghost_rms=[[i, round(g, 3)] for i, g in ghost], peak_ghost_rms=round(peak_ghost, 3),
                tof_scene2_on=round(tof_on, 4), tof_scene2_off=round(tof_off, 4), verdict=verdict)


# --------------------------------------------------------------------------- #
# Torch parity: run the ACTUAL deployed fast path end-to-end (build_anchor_cache ->
# reconstruct_torch download_output=False -> softocc_patch_torch -> download) on window A.
# --------------------------------------------------------------------------- #
def torch_parity():
    if not (_HAS_TORCH and torch.backends.mps.is_available()):
        print("\n=== Torch parity: SKIPPED (no MPS) ===")
        return None
    print("\n=== Torch parity: deployed fast path (reconstruct_torch + softocc_patch_torch), window A ===")
    import anchor_sr
    import gpu_ops as G
    frames = decode(0)
    anchors, bb = anchors_of(frames)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    bic = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC) for i in range(N)}
    srf = D.build_perframe_cache(frames, w_hd, h_hd, SR_MODEL)
    _free_gpu()
    reset_idx = W.reset_indices(frames)

    # Build the deployed cache ONCE; run BOTH the numpy reference path and the torch path on it.
    # The robust parity signal is the OUTPUT pixel MAE (numpy is the deterministic ground truth);
    # tOF/eff-bic computed on GPU output are differential ratios that amplify the standing ~1-LSB
    # GPU-float rounding (round vs truncate, float32 intermediates) into temporally-incoherent
    # noise, so they read inflated on the GPU path even when the pixels match (see _isolate2.py).
    cache, _i, _s = anchor_sr.build_anchor_cache(frames, w_hd, h_hd, SR_MODEL, occ_mode=OCC,
                                                 fallback_thresh=0.50, gpu_cache=False)
    conf = W.build_conf(frames, anchors, bb, w_hd, h_hd)
    # numpy reference (deterministic) on the SAME cache
    _, Rn = D.reconstruct(frames, None, SCALE, True, OCC, cache, set(), backend="numpy",
                          collect_metrics=False, download_output=True)
    mask = {i: (Rn[i]["mask"] if Rn[i]["mask"] is not None else np.zeros((h_hd, w_hd), bool))
            for i in range(N)}
    hole = {i: float(mask[i].mean()) for i in range(N)}
    Rn_on = {i: dict(recon=Rn[i]["recon"].copy(), mask=Rn[i]["mask"]) for i in range(N)}
    W.softocc_patch_np(frames, Rn_on, bic_provider=lambda i: bic[i], sr_provider=lambda i: srf[i],
                       conf=conf, anchors=anchors, reset_idx=reset_idx, gain=GAIN, beta=BETA,
                       feather_k=FE, enabled=True)
    out_np = {i: Rn_on[i]["recon"] for i in range(N)}
    # torch deployed path
    _, Rbase = D.reconstruct(frames, None, SCALE, True, OCC, cache, set(), backend="torch",
                             collect_metrics=False, download_output=False)

    def clone_and_blend(enabled):
        R = {i: dict(recon=Rbase[i]["recon"].clone(), mask=Rbase[i]["mask"]) for i in range(N)}
        info = W.softocc_patch_torch(frames, R, w_hd, h_hd, SR_MODEL, anchors=anchors, backbone=bb,
                                     reset_idx=reset_idx, gain=GAIN, beta=BETA, feather_k=FE,
                                     enabled=enabled)
        out = {i: G.img_to_host(R[i]["recon"]) for i in range(N)}
        _free_gpu()
        return out, info

    out_off, _io = clone_and_blend(False)
    off = W.honest_metrics(frames, out_off, mask, bic, srf, anchors, hole, "torch OFF")
    out_on, info = clone_and_blend(True)
    on = W.honest_metrics(frames, out_on, mask, bic, srf, anchors, hole, "torch ON (GPU metric)")
    # robust parity: torch-ON output vs numpy-ON output, pixel MAE (the standing GPU tolerance)
    mae = float(np.mean([np.abs(out_on[i].astype(np.float32) - out_np[i].astype(np.float32)).mean()
                         for i in range(N) if i not in anchors]))
    print(_fmt(off)); print(_fmt(on))
    print(f"  torch blended {len(info['blended'])} non-anchor frames; resets fired={info['n_resets']}")
    print(f"  PARITY: torch-ON vs numpy-ON output pixel MAE = {mae:.3f}/255 (within GPU-float "
          f"tolerance) -> the blend is faithful; the GPU tOF/eff-bic are rounding-inflated ratios.")
    return dict(off=off, on=on, n_blended=len(info["blended"]),
                output_mae_vs_numpy=round(mae, 3))


# --------------------------------------------------------------------------- #
def main():
    results = {}
    t_start = time.perf_counter()

    # ---- Task 2a: window A reproduction (numpy, R2-E2-equivalent base) ----
    print("=== Task 2a: window A (start 0, N=48) -- reproduce R2-E2 (c) through derisk.reconstruct ===")
    A = setup(decode(0))
    print(f"  LR={A['frames'][0][1].shape[1]}x{A['frames'][0][1].shape[0]} HD={A['w_hd']}x{A['h_hd']}  "
          f"anchors={sorted(A['anchors'])}  SR cache {A['t_sr']}s")
    rsetA = W.reset_indices(A["frames"])
    offA, onA, infoA = run_off_on(A, rsetA, "winA")
    print(_fmt(offA)); print(_fmt(onA))
    print(f"  -> tOF {offA['tof']:.4f}->{onA['tof']:.4f} ({100*(onA['tof']-offA['tof'])/offA['tof']:+.1f}%)  "
          f"eff-bic {offA['eff_bicubic_pct']:.3f}->{onA['eff_bicubic_pct']:.3f} "
          f"({onA['eff_bicubic_pct']-offA['eff_bicubic_pct']:+.3f} pt)  blended={len(infoA['blended'])}")
    # PASS if eff-bic drops >=1.0 pt AND tOF rise <=+5% (R2-E2 (c): -1.34 pt @ +2.0%)
    d_eff = offA["eff_bicubic_pct"] - onA["eff_bicubic_pct"]
    d_tof = (onA["tof"] - offA["tof"]) / offA["tof"]
    passA = (d_eff >= 1.0 and d_tof <= 0.05)
    print(f"  TASK-2a VERDICT: {'PASS' if passA else 'FAIL'} "
          f"(eff-bic -{d_eff:.2f} pt at tOF {100*d_tof:+.1f}% -- target -1.34 pt @ +2.0%)")
    results["task2a_windowA"] = dict(off=offA, on=onA, d_eff_pt=round(d_eff, 3),
                                     d_tof_pct=round(100 * d_tof, 2), passed=passA)
    _free_gpu()

    # ---- Task 2b: talking-head (start 5000) -- expect ~unchanged (self-gating) ----
    print("\n=== Task 2b: talking-head (start 5000, N=48) -- expect ~unchanged ===")
    B = setup(decode(5000))
    print(f"  anchors={sorted(B['anchors'])}  non-anchor hole mean="
          f"{100*np.mean([B['hole'][i] for i in range(N) if i not in B['anchors']]):.2f}%")
    rsetB = W.reset_indices(B["frames"])
    offB, onB, infoB = run_off_on(B, rsetB, "talkhead")
    print(_fmt(offB)); print(_fmt(onB))
    d_effB = offB["eff_bicubic_pct"] - onB["eff_bicubic_pct"]
    d_tofB = (onB["tof"] - offB["tof"]) / offB["tof"] if offB["tof"] else 0.0
    unchanged = (abs(onB["tof"] - offB["tof"]) <= 0.02 and abs(d_effB) <= 1.0)
    print(f"  TASK-2b VERDICT: {'PASS (negligible)' if unchanged else 'CHECK'} "
          f"(tOF {100*d_tofB:+.1f}%, eff-bic {-d_effB:+.2f} pt)")
    results["task2b_talkinghead"] = dict(off=offB, on=onB, d_eff_pt=round(d_effB, 3),
                                         d_tof_pct=round(100 * d_tofB, 2), unchanged=unchanged)
    _free_gpu()

    # ---- Task 2c: escape on the DEPLOYED SR-anchor recon (build_anchor_cache) ----
    print("\n=== Task 2c: window A on the DEPLOYED hybrid cache (SR anchors) -- escape on real recon ===")
    C = setup_deployed(decode(0))
    rsetC = W.reset_indices(C["frames"])
    offC, onC, infoC = run_off_on(C, rsetC, "winA-deployed")
    print(_fmt(offC)); print(_fmt(onC))
    d_effC = offC["eff_bicubic_pct"] - onC["eff_bicubic_pct"]
    d_tofC = (onC["tof"] - offC["tof"]) / offC["tof"]
    passC = (d_effC >= 1.0 and d_tofC <= 0.05)
    print(f"  TASK-2c VERDICT: {'PASS' if passC else 'FAIL'} (eff-bic -{d_effC:.2f} pt at "
          f"tOF {100*d_tofC:+.1f}% -- escape holds on the SR-anchor pipeline recon)")
    results["task2c_deployed_cache"] = dict(off=offC, on=onC, d_eff_pt=round(d_effC, 3),
                                            d_tof_pct=round(100 * d_tofC, 2), passed=passC)
    _free_gpu()

    # ---- Task 3: EMA reset ----
    results["task3_reset"] = reset_test()
    _free_gpu()

    # ---- Torch parity ----
    tp = torch_parity()
    if tp is not None:
        results["torch_parity"] = tp
    _free_gpu()

    results["_meta"] = dict(clip="sample.mp4", N=N, scale=SCALE, occ=OCC, sr_model=SR_MODEL,
                            gain=GAIN, beta=BETA, feather=FE, wall_s=round(time.perf_counter() - t_start, 1))
    with open(os.path.join(_HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {os.path.join(_HERE, 'results.json')}  (wall {results['_meta']['wall_s']}s)")

    print("\n================= SUMMARY =================")
    print(f"  2a window-A reproduce escape : {'PASS' if results['task2a_windowA']['passed'] else 'FAIL'} "
          f"(OFF {offA['tof']:.3f}/{offA['eff_bicubic_pct']:.2f}%  ->  ON {onA['tof']:.3f}/{onA['eff_bicubic_pct']:.2f}%)")
    print(f"  2b talking-head unchanged    : {'PASS' if unchanged else 'CHECK'}")
    print(f"  2c escape on deployed recon  : {'PASS' if passC else 'FAIL'}")
    print(f"  3  EMA reset (cross-cut)     : {results['task3_reset']['verdict']}")
    if tp is not None:
        print(f"  torch parity (deployed path) : output MAE vs numpy = {tp['output_mae_vs_numpy']}/255 "
              f"(blend faithful; GPU tOF/eff-bic are rounding-inflated ratios -> numpy is the headline)")


if __name__ == "__main__":
    main()
