#!/usr/bin/env python3
"""
R2-E2 -- break the high-motion tOF<->fallback% tension with SOFT-occlusion / temporally
         consistent fallback.  READ-ONLY imports of prototype/ (+server/); all new code here.

THE TENSION (from R1-E2): on high-motion window A the occlusion fallback is a HARD binary
switch (derisk.reconstruct: `recon[occ] = perframe[occ]`).  perframe=BICUBIC is temporally
smooth (tOF-optimal) but soft; perframe=fresh compact-SR lowers fallback% but the fresh
per-frame HF in disocclusion regions SHIMMERS -> tOF rises.  No HARD policy improved both.

THE QUESTION: is there a THIRD point OFF the (bicubic, hard-SR) frontier -- LOWER fallback%
at ~bicubic's tOF -- via a SOFT, feathered, and/or TEMPORALLY-CONSISTENT fallback?

DESIGN (output-only, GOTCHA #16): the propagation CHAIN is the deployed bicubic-fallback
reconstruction (derisk.reconstruct, numpy, occ='reactive') -- IDENTICAL for every scheme, so
nothing soft is ever fed back as a reference.  Each scheme is a per-frame OUTPUT post-pass on
the non-anchor frames:  out = (1-a)*base + a*T,  where `base` = the bicubic-chain recon
(== bicubic INSIDE the fallback mask, propagated-SR OUTSIDE it), `T` = a sharp/stabilized
injection target, and a(x,y) in [0,1] is the SR-injection weight (0 -> pure bicubic baseline,
=mask -> hard SR-escalate).  Anchors are byte-identical across all schemes.

HONEST METRICS ONLY (per the methodology): tOF (Farneback-EPE of recon vs decoded LR; the
headline temporal metric) + eff-bicubic% (generalized continuously below) + |dF| (raw flicker
cross-check), plus a fallback-localized |dF| to amplify the in-disocclusion shimmer signal.
NOT used: LR-consistency, NR-sharpness alone.

eff-bicubic% (scheme-agnostic generalization of R1's "% pixels still served by bicubic"):
inside the fallback region M, the realized-detail ratio r = ||out-bic|| / ||sr-bic|| in [0,1]
(0 = output is exactly bicubic -> still 'served by bicubic'; 1 = output reached fresh SR).
eff_bic_weight = 1-r.  eff-bic% = 100 * mean_nonanchor( sum_M eff_bic_weight / (H*W) ).
At a=0 -> eff-bic% == hole_frac (R1's 7.70%); at hard SR -> ~0.  This is the honest 'how much
soft bicubic is still being shown', valid for linear AND additive/EMA injections alike.
"""
import gc, json, os, sys, time
import cv2, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    if p not in sys.path:
        sys.path.insert(0, p)
import derisk as D            # noqa: E402  decode / reconstruct / warp_hd / warp_lr / tof
import sr as SR               # noqa: E402  compact SR cache
try:
    import torch              # noqa: E402  free MPS between heavy ops
    _HAS_TORCH = True
except Exception as e:        # surfaced, not swallowed
    print(f"[warn] torch import failed ({e}); SR cache will fail if MPS needed")
    _HAS_TORCH = False

CLIP = os.path.join(_REPO, "sample.mp4")
N = 48
SCALE = 2                     # 720p-tier instant
OCC = "reactive"             # MODE_CONFIG['instant'] ships occ='reactive'
SR_MODEL = "realesrgan"      # realesr-general-x4v3 compact net
EPS = 1e-3

# confidence ramp on the reactive residual (mean-abs LR diff, 0..255); tau_react=16 is derisk's
# binary threshold -> put it mid-ramp so the soft grade brackets the hard decision.
CONF_LO, CONF_HI = 6.0, 26.0


def _free_gpu():
    gc.collect()
    if _HAS_TORCH:
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            print(f"  [warn] mps empty_cache failed: {e}")


# --------------------------------------------------------------------------- #
# Setup: decode, caches, base bicubic chain, per-frame confidence + warp-pred
# --------------------------------------------------------------------------- #
def setup():
    frames = D.decode_lr_and_mvs(CLIP, 0, N)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    backbone = D.backbone_indices(frames)
    first = backbone[0]
    anchors = {i for i in backbone if frames[i][0] == "I" or i == first}
    iframes = {i for i in range(N) if frames[i][0] == "I"}

    bic = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
           for i in range(N)}
    SR.reset_latency(SR_MODEL)
    t0 = time.perf_counter()
    srf = D.build_perframe_cache(frames, w_hd, h_hd, SR_MODEL)        # ONE SR pass, reused for all
    t_sr = time.perf_counter() - t0
    _free_gpu()

    # base propagation chain = deployed bicubic fallback (numpy, deterministic). Provides recon
    # (== bicubic inside the occ mask) + the per-frame binary fallback mask + hole_frac.
    _, R = D.reconstruct(frames, None, SCALE, True, OCC, bic, set(), backend="numpy",
                         collect_metrics=False, download_output=True)
    base = {i: R[i]["recon"] for i in range(N)}
    mask = {i: (R[i]["mask"] if R[i]["mask"] is not None else
                np.zeros((h_hd, w_hd), bool)) for i in range(N)}
    hole = {i: float(R[i]["hole_frac"]) for i in range(N)}

    # per non-anchor frame: confidence-to-use-SR (HD float in [0,1]) + steady warp-prediction
    conf, warp_pred = {}, {}
    for i in range(N):
        if i in anchors:
            continue
        pt, lr_cur, mvs = frames[i]
        prev_bb = max([b for b in backbone if b < i], default=None)
        if pt == "P" and prev_bb is not None and mvs is not None and len(mvs):
            fx, fy = D.build_lr_flow(mvs, h_lr, w_lr, want="past")
            pred_lr = D.warp_lr(lr_cur if prev_bb is None else frames[prev_bb][1], fx, fy)
            react = np.abs(lr_cur.astype(np.float32) - pred_lr.astype(np.float32)).mean(axis=2)
            c_lr = np.clip((react - CONF_LO) / (CONF_HI - CONF_LO), 0.0, 1.0)
            c_lr[~np.isfinite(fx)] = 1.0                  # intra hole: no warp info -> full SR
            conf[i] = cv2.resize(c_lr, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
            wp, _ = D.warp_hd(base[prev_bb], fx, fy, SCALE)
            warp_pred[i] = wp
        else:
            # B-leaf or no-MV: fallback region is the small 'none' set -> treat as full-SR need
            conf[i] = np.ones((h_hd, w_hd), np.float32)
            warp_pred[i] = base[i]                          # no steady warp pred -> use base
    _free_gpu()
    return dict(frames=frames, h_lr=h_lr, w_lr=w_lr, w_hd=w_hd, h_hd=h_hd,
                anchors=anchors, iframes=iframes, backbone=backbone, bic=bic, srf=srf,
                base=base, mask=mask, hole=hole, conf=conf, warp_pred=warp_pred,
                t_sr=round(t_sr, 2), sr_ms=round(SR.median_latency_ms(SR_MODEL), 1))


def _feather(m_bool, k):
    if k < 3:
        return m_bool.astype(np.float32)
    k = int(k) | 1
    return cv2.GaussianBlur(m_bool.astype(np.float32), (k, k), 0)


# --------------------------------------------------------------------------- #
# Scheme runner: produce per-frame output via an injection (a, T) policy
# --------------------------------------------------------------------------- #
def run_scheme(W, policy):
    """policy(i, st) -> (a_hd[H,W] float[0,1], T_hd[H,W,3] float).  st = mutable temporal state
    dict (EMA buffers), reset at I-frames.  Returns dict of per-frame uint8 outputs + the
    leak fraction (SR injected OUTSIDE the fallback mask, via feather spill)."""
    frames, anchors, iframes = W["frames"], W["anchors"], W["iframes"]
    base, bic = W["base"], W["bic"]
    out, leak = {}, []
    st = {}
    for i in range(N):
        if i in iframes:
            st = {}                                        # scene cut -> reset temporal state
        if i in anchors:
            out[i] = base[i]                               # anchors identical across schemes
            # still advance any EMA on the SR stream so post-anchor frames have warmed state
            policy(i, st, advance_only=True)
            continue
        a, T = policy(i, st)
        a3 = a[..., None]
        o = (1.0 - a3) * base[i].astype(np.float32) + a3 * T
        out[i] = np.clip(o, 0, 255).astype(np.uint8)
        m = W["mask"][i]
        if m.any():
            leak.append(float((a[~m] > 0.02).mean()) if (~m).any() else 0.0)
    return out, (float(np.mean(leak)) if leak else 0.0)


# ---- policy factories (each returns a closure usable by run_scheme) -------- #
def pol_bicubic(W):
    def p(i, st, advance_only=False):
        if advance_only:
            return
        return np.zeros((W["h_hd"], W["w_hd"]), np.float32), W["bic"][i].astype(np.float32)
    return p


def pol_hard(W, feather=0):
    def p(i, st, advance_only=False):
        if advance_only:
            return
        a = _feather(W["mask"][i], feather)
        return a, W["srf"][i].astype(np.float32)
    return p


def pol_soft_feather(W, gain, feather, graded=False):
    """(a) spatial feathered injection of fresh SR; graded=True -> scale by residual confidence."""
    def p(i, st, advance_only=False):
        if advance_only:
            return
        a = gain * _feather(W["mask"][i], feather)
        if graded:
            a = a * W["conf"][i]
        return np.clip(a, 0, 1).astype(np.float32), W["srf"][i].astype(np.float32)
    return p


def pol_b1_srwarp(W, k, feather, graded=True):
    """(b1) inject SR blended with the STEADY warp-prediction: T=(1-k)*warp_pred + k*sr."""
    def p(i, st, advance_only=False):
        if advance_only:
            return
        a = _feather(W["mask"][i], feather)
        if graded:
            a = a * W["conf"][i]
        T = (1 - k) * W["warp_pred"][i].astype(np.float32) + k * W["srf"][i].astype(np.float32)
        return np.clip(a, 0, 1).astype(np.float32), T
    return p


def pol_b2_ema_sr(W, beta, feather, graded=True):
    """(b2) inject a screen-space temporal EMA of the SR source (low-pass the fresh HF)."""
    def p(i, st, advance_only=False):
        sr = W["srf"][i].astype(np.float32)
        e = st.get("ema")
        e = sr if e is None else (beta * e + (1 - beta) * sr)
        st["ema"] = e
        if advance_only:
            return
        a = _feather(W["mask"][i], feather)
        if graded:
            a = a * W["conf"][i]
        return np.clip(a, 0, 1).astype(np.float32), e
    return p


def pol_b3_ema_hf(W, beta, feather, graded=True):
    """(b3) LF from CURRENT bicubic (tracks motion -> tOF-safe), HF temporally smoothed:
    T = bic + EMA(sr - bic).  The flickery high-freq is low-passed; the motion-tracking low-freq
    is always fresh."""
    def p(i, st, advance_only=False):
        sr = W["srf"][i].astype(np.float32)
        bic = W["bic"][i].astype(np.float32)
        hf = sr - bic
        e = st.get("emahf")
        e = hf if e is None else (beta * e + (1 - beta) * hf)
        st["emahf"] = e
        if advance_only:
            return
        a = _feather(W["mask"][i], feather)
        if graded:
            a = a * W["conf"][i]
        return np.clip(a, 0, 1).astype(np.float32), bic + e
    return p


def pol_c_combo(W, gain, beta, feather):
    """(c) confidence-graded feathered injection (a) of the temporally-stabilized HF source (b3)."""
    def p(i, st, advance_only=False):
        sr = W["srf"][i].astype(np.float32)
        bic = W["bic"][i].astype(np.float32)
        hf = sr - bic
        e = st.get("emahf")
        e = hf if e is None else (beta * e + (1 - beta) * hf)
        st["emahf"] = e
        if advance_only:
            return
        a = gain * _feather(W["mask"][i], feather) * W["conf"][i]
        return np.clip(a, 0, 1).astype(np.float32), bic + e
    return p


# --------------------------------------------------------------------------- #
# Honest metrics
# --------------------------------------------------------------------------- #
def metrics(W, out, name):
    frames, anchors = W["frames"], W["anchors"]
    w_lr, h_lr, sm = W["w_lr"], W["h_lr"], (W["w_lr"], W["h_lr"])
    seq = [cv2.resize(out[i], sm) for i in range(N)]
    lr = [frames[i][1] for i in range(N)]
    tof = D.tof(seq, lr)
    # |dF| global (recon vs LR)
    s = [x.astype(np.float32) for x in seq]
    l = [x.astype(np.float32) for x in lr]
    d_recon = float(np.mean([np.abs(s[t] - s[t - 1]).mean() for t in range(1, N)]))
    d_lr = float(np.mean([np.abs(l[t] - l[t - 1]).mean() for t in range(1, N)]))
    # fallback-localized |dF| at HD: temporal change of the OUTPUT restricted to the union of
    # consecutive fallback masks -> isolates the in-disocclusion shimmer (the tension's locus).
    fb_d, fb_n = [], []
    for t in range(1, N):
        mu = W["mask"][t] | W["mask"][t - 1]
        if mu.any():
            diff = np.abs(out[t].astype(np.float32) - out[t - 1].astype(np.float32)).mean(axis=2)
            fb_d.append(float(diff[mu].mean()))
    fb_df = float(np.mean(fb_d)) if fb_d else 0.0
    # eff-bicubic% (continuous): inside M, realized-detail ratio r = ||out-bic||/||sr-bic||.
    nonanchor = [i for i in range(N) if i not in anchors]
    HW = W["h_hd"] * W["w_hd"]
    ebw, dtl, area = [], [], []
    for i in nonanchor:
        m = W["mask"][i]
        if not m.any():
            ebw.append(0.0); dtl.append(0.0); area.append(0.0); continue
        bic = W["bic"][i].astype(np.float32)
        sr = W["srf"][i].astype(np.float32)
        o = out[i].astype(np.float32)
        num = np.linalg.norm((o - bic)[m], axis=1)
        den = np.linalg.norm((sr - bic)[m], axis=1) + EPS
        r = np.clip(num / den, 0.0, 1.0)
        ebw.append(float((1.0 - r).sum()) / HW)            # bicubic-weighted pixel-equiv / frame
        dtl.append(float(r.sum()) / HW)
        area.append(float(m.mean()))
    eff_bic = 100.0 * float(np.mean(ebw))
    detail = 100.0 * float(np.mean(dtl))
    raw_fb = 100.0 * float(np.mean([W["hole"][i] for i in nonanchor]))
    return dict(scheme=name, tof=round(tof, 4), eff_bicubic_pct=round(eff_bic, 3),
                detail_injected_pct=round(detail, 3), raw_fallback_pct=round(raw_fb, 3),
                fb_localized_dF=round(fb_df, 3), d_recon=round(d_recon, 3), d_lr=round(d_lr, 3))


# --------------------------------------------------------------------------- #
def main():
    print("=== R2-E2 soft-occlusion / temporally-consistent fallback -- window A (start 0, N=48) ===")
    W = setup()
    print(f"LR={W['w_lr']}x{W['h_lr']} HD={W['w_hd']}x{W['h_hd']}  anchors={sorted(W['anchors'])}  "
          f"SR cache: {W['t_sr']}s, {W['sr_ms']} ms/f")
    na = [i for i in range(N) if i not in W["anchors"]]
    print(f"non-anchor hole mean={100*np.mean([W['hole'][i] for i in na]):.2f}%  "
          f"#>0.20={sum(W['hole'][i]>0.20 for i in na)}")

    FE = 21                                                 # default feather kernel (HD px)
    schemes = []
    schemes.append(("bicubic (R1 tOF-optimal)", pol_bicubic(W)))
    schemes.append(("HARD-SR all (binary)", pol_hard(W, feather=0)))
    schemes.append(("HARD-SR all (feather21)", pol_hard(W, feather=FE)))
    # (a) soft feather, gain sweep (no grade)
    for g in (0.25, 0.5, 0.75, 1.0):
        schemes.append((f"(a) feather g={g}", pol_soft_feather(W, g, FE, graded=False)))
    # (a') confidence-graded feather
    for g in (0.5, 0.75, 1.0):
        schemes.append((f"(a') conf-graded g={g}", pol_soft_feather(W, g, FE, graded=True)))
    # (b1) SR<->steady-warp blend
    for k in (0.3, 0.5, 0.7):
        schemes.append((f"(b1) sr+warp k={k}", pol_b1_srwarp(W, k, FE, graded=True)))
    # (b2) screen-space EMA of SR
    for b in (0.5, 0.7):
        schemes.append((f"(b2) ema-sr beta={b}", pol_b2_ema_sr(W, b, FE, graded=True)))
    # (b3) HF-EMA (LF from bicubic)
    for b in (0.5, 0.7, 0.85):
        schemes.append((f"(b3) ema-HF beta={b}", pol_b3_ema_hf(W, b, FE, graded=True)))
    # (c) combo
    for (g, b) in ((0.75, 0.7), (1.0, 0.7), (1.0, 0.85)):
        schemes.append((f"(c) combo g={g} beta={b}", pol_c_combo(W, g, b, FE)))

    results = []
    for name, pol in schemes:
        out, leak = run_scheme(W, pol)
        r = metrics(W, out, name)
        r["leak_pct"] = round(100 * leak, 2)
        results.append(r)
        print(f"  {name:32s} tOF={r['tof']:.4f}  effBic%={r['eff_bicubic_pct']:6.3f}  "
              f"detail%={r['detail_injected_pct']:6.3f}  fbdF={r['fb_localized_dF']:5.2f}  "
              f"dRec={r['d_recon']:.2f}  leak%={r['leak_pct']:.1f}")
        del out
        _free_gpu()

    payload = dict(config=dict(N=N, scale=SCALE, occ=OCC, sr_model=SR_MODEL, feather=FE,
                               conf_lo=CONF_LO, conf_hi=CONF_HI, clip=CLIP,
                               anchors=sorted(W["anchors"]), sr_ms=W["sr_ms"]),
                   baseline_hole_pct=round(100*np.mean([W['hole'][i] for i in na]), 3),
                   results=results)
    with open(os.path.join(_HERE, "results.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {os.path.join(_HERE,'results.json')}")
    _frontier_plot(results)


def _frontier_plot(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] no matplotlib ({e}); skipping plot")
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    bic = next(r for r in results if r["scheme"].startswith("bicubic"))
    for r in results:
        n = r["scheme"]
        if n.startswith("bicubic"):
            c, m = "k", "*"
        elif n.startswith("HARD"):
            c, m = "red", "X"
        elif n.startswith("(a'"):
            c, m = "tab:orange", "^"
        elif n.startswith("(a)"):
            c, m = "gold", "v"
        elif n.startswith("(b1"):
            c, m = "tab:green", "s"
        elif n.startswith("(b2"):
            c, m = "tab:cyan", "P"
        elif n.startswith("(b3"):
            c, m = "tab:blue", "o"
        else:
            c, m = "tab:purple", "D"
        ax.scatter(r["eff_bicubic_pct"], r["tof"], c=c, marker=m, s=90, zorder=3)
        ax.annotate(n, (r["eff_bicubic_pct"], r["tof"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(bic["tof"], color="k", ls="--", lw=0.8, alpha=0.6,
               label=f"bicubic tOF={bic['tof']:.3f} (escape = below+left)")
    ax.axvline(bic["eff_bicubic_pct"], color="k", ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel("eff-bicubic %  (lower = less soft fallback shown = better ->)")
    ax.set_ylabel("tOF  (lower = steadier = better)")
    ax.set_title("R2-E2 frontier: tOF vs eff-bicubic% (window A high-motion)\n"
                 "GOAL: a point BELOW the dashed line and LEFT of bicubic (off the frontier)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(_HERE, "frontier.png")
    fig.savefig(p, dpi=110)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
