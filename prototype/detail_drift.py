#!/usr/bin/env python3
"""
Step 8 / Task 3 -- THE KEY EXPERIMENT: does the heavy anchor's extra detail SURVIVE codec-MV
propagation, or does warp-blur erode it back toward the compact anchor within a few frames?

On a clean SINGLE-anchor all-P chain (one I anchor at dist 0, then consecutive P-frames warping
from it -- no re-anchoring, no mid-window I), measure, as a function of DISTANCE FROM THE ANCHOR,
for BOTH anchor models (compact realesr-general-x4v3 vs heavy RealESRGAN_x4plus):
  * sharpness/detail proxy: variance of Laplacian (+ high-freq FFT energy) of the recon, and
  * tOF (temporal stability) of the propagated chain vs the decoded LR.

Two reconstruction modes per model, because they answer different questions:
  * PURE PROPAGATION (primary, solid): SR the ANCHOR ONLY; occlusion holes are filled with
    BICUBIC, never with fresh per-frame SR. So the ONLY hallucinated detail in a propagated
    frame is what was WARPED from the anchor. The heavy-vs-compact gap then decays purely by
    warp erosion -> its half-life is the honest "how fast does warp blur the extra detail away".
  * DEPLOYABLE (dashed, context): the real system also re-runs SR at occluded pixels (the
    fallback). That re-INJECTS fresh heavy/compact SR every frame, so it partly restores the
    advantage -- but that is "re-running SR", not "propagation preserving detail". Shown so the
    difference between the two is explicit.

The heavy advantage = sharp_heavy(d) - sharp_compact(d) (content cancels: same frame, same warp,
same fallback mask). Half-life = distance at which the PURE-PROPAGATION advantage is halved --
which sets the anchor interval needed to actually deliver the extra detail to non-anchor frames.

Two real windows (genuine single-anchor all-P chains; both start ON an I-frame):
  * talkinghead : abs 5032, I + 11 P   * highmotion : abs 0, I + 15 P
    python3 detail_drift.py   -> out_quality/detail_drift.png (+ detail_drift.csv) + verdict
"""
import csv
import os

import cv2
import numpy as np

import derisk as d

OUT = os.path.join(os.path.dirname(__file__), "out_quality")
SCALE = 4
OCC = "full"                       # most accurate mask; SAME for both models => cancels in the gap
WINDOWS = [("talkinghead", 5032, 12), ("highmotion", 0, 16)]
MODELS = [("compact", "realesrgan"), ("x4plus", "realesrgan-x4plus")]


def varlap(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def hf_energy(rgb, cutoff=0.25):
    """Fraction of luma spectral energy beyond `cutoff` of Nyquist (high-frequency detail)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g))
    mag2 = F.real ** 2 + F.imag ** 2
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    tot = float(mag2.sum())
    return float(mag2[r > cutoff].sum() / tot) if tot > 0 else 0.0


def half_life(dists, gap):
    """Distance at which `gap` first falls to half its dist-0 value (linear-interp the crossing).
    Returns (half_life_frames, gap0). If never halved within the window, returns (max_dist, gap0)."""
    g0 = gap[0]
    if g0 <= 0:
        return float("nan"), g0
    thr = g0 / 2.0
    for k in range(1, len(dists)):
        if gap[k] <= thr:
            d0, d1 = dists[k - 1], dists[k]
            y0, y1 = gap[k - 1], gap[k]
            t = 0.0 if y0 == y1 else (y0 - thr) / (y0 - y1)
            return float(d0 + t * (d1 - d0)), g0
    return float(dists[-1]), g0          # not reached -> advantage persists past the window


def _bicubic(lr, w_hd, h_hd):
    return cv2.resize(lr, (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)


def curve(frames, bb, w_hd, h_hd, cache):
    """Reconstruct the single-anchor chain with `cache` and return {dist: dict(varlap,hf,fallback)}."""
    _, R = d.reconstruct(frames, None, SCALE, True, OCC, cache, set(),
                         backend="numpy", collect_metrics=True)
    out = {}
    for i in bb:
        dd = R[i]["dist"]
        out[dd] = dict(varlap=varlap(R[i]["recon"]), hf=hf_energy(R[i]["recon"]),
                       fallback=R[i]["hole_frac"], recon_lr=cv2.resize(R[i]["recon"], (w_hd // SCALE,
                                                                                       h_hd // SCALE)))
    return out


def measure_window(name, start, nf):
    frames = d.decode_lr_and_mvs("../sample.mp4", start, nf)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    bb = d.backbone_indices(frames)
    print(f"\n[{name}] abs {start}, {nf}f, types={''.join(f[0][0] for f in frames)}; "
          f"backbone={bb}; anchor=idx {bb[0]} (single anchor at dist 0)")
    res = {}
    for tag, sr_mode in MODELS:
        full = d.build_perframe_cache(frames, w_hd, h_hd, sr_mode)   # deployable: SR every frame
        a = bb[0]
        prop = {i: (full[a] if i == a else _bicubic(frames[i][1], w_hd, h_hd))
                for i in range(len(frames))}                          # pure prop: anchor SR only
        c_prop = curve(frames, bb, w_hd, h_hd, prop)
        c_full = curve(frames, bb, w_hd, h_hd, full)
        # tOF of the pure-propagation chain vs decoded LR
        seq = [c_prop[k]["recon_lr"] for k in sorted(c_prop)]
        lr = [frames[i][1] for i in bb]
        tof = d.tof(seq, lr)
        res[tag] = dict(prop=c_prop, full=c_full, tof=tof)
        print(f"  {tag:7s}: PURE-PROP anchor varlap={c_prop[0]['varlap']:.1f} "
              f"tail(d={max(c_prop)})={c_prop[max(c_prop)]['varlap']:.1f}  tOF={tof:.3f}  "
              f"(deployable max fallback {100*max(v['fallback'] for v in c_full.values()):.0f}%)")
    return bb, res


def main():
    os.makedirs(OUT, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results, csv_rows = {}, []
    for name, start, nf in WINDOWS:
        _, res = measure_window(name, start, nf)
        results[name] = res

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    summary = {}
    for col, (name, _, _) in enumerate(WINDOWS):
        res = results[name]
        dists = sorted(res["compact"]["prop"])
        vl_c = [res["compact"]["prop"][k]["varlap"] for k in dists]
        vl_h = [res["x4plus"]["prop"][k]["varlap"] for k in dists]
        vlf_c = [res["compact"]["full"][k]["varlap"] for k in dists]
        vlf_h = [res["x4plus"]["full"][k]["varlap"] for k in dists]
        fb = [100 * res["x4plus"]["full"][k]["fallback"] for k in dists]
        gap_prop = [h - c for h, c in zip(vl_h, vl_c)]
        gap_full = [h - c for h, c in zip(vlf_h, vlf_c)]
        hl, gap0 = half_life(dists, gap_prop)
        rel0 = gap0 / vl_c[0] if vl_c[0] else 0.0
        summary[name] = dict(hl=hl, gap0=gap0, rel0=rel0, maxd=dists[-1],
                             tof_c=res["compact"]["tof"], tof_h=res["x4plus"]["tof"],
                             fb_max=max(fb), gap_full0=gap_full[0], gap_full_tail=gap_full[-1])
        for j, k in enumerate(dists):
            csv_rows.append(dict(window=name, dist=k,
                                 varlap_compact_prop=round(vl_c[j], 2),
                                 varlap_x4plus_prop=round(vl_h[j], 2),
                                 gap_prop=round(gap_prop[j], 2),
                                 gap_deployable=round(gap_full[j], 2),
                                 fallback_pct=round(fb[j], 2)))

        ax = axes[0][col]
        ax.plot(dists, vl_h, "-^", color="tab:red", label="x4plus (heavy), pure propagation")
        ax.plot(dists, vl_c, "-o", color="tab:blue", label="compact, pure propagation")
        ax.set_title(f"{name}: sharpness vs distance from anchor (pure warp propagation)")
        ax.set_xlabel("distance from anchor (frames)")
        ax.set_ylabel("var-of-Laplacian (sharpness)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax2 = axes[1][col]
        ax2.plot(dists, gap_prop, "-s", color="purple", label="heavy advantage, PURE propagation")
        ax2.plot(dists, gap_full, "--d", color="darkorange", alpha=0.8,
                 label="heavy advantage, deployable (+fallback re-SR)")
        ax2.axhline(gap0 / 2, color="purple", ls=":", alpha=0.6, label="half of dist-0 advantage")
        if np.isfinite(hl) and hl < dists[-1]:
            ax2.axvline(hl, color="black", ls=":", label=f"half-life = {hl:.1f} frames")
        ax2.axhline(0, color="gray", lw=0.8)
        ax2.set_xlabel("distance from anchor (frames)")
        ax2.set_ylabel("heavy-anchor sharpness advantage")
        ax2.set_title(f"{name}: heavy-anchor advantage decay"
                      + (f" (half-life {hl:.1f}f)" if np.isfinite(hl) and hl < dists[-1]
                         else " (advantage persists)"))
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8, loc="upper right")
        axt = ax2.twinx()
        axt.bar(dists, fb, alpha=0.12, color="red", width=0.5)
        axt.set_ylabel("deployable fallback (fresh per-frame SR) %", color="red")

    fig.suptitle("Detail drift: does heavy-anchor (x4plus) detail survive codec-MV propagation?",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "detail_drift.png"), dpi=110)
    plt.close(fig)

    with open(os.path.join(OUT, "detail_drift.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    print("\n================ DETAIL-DRIFT VERDICT (pure warp propagation) ================")
    for name in summary:
        s = summary[name]
        hl = s["hl"]
        verdict = (f"half-life = {hl:.1f} frames" if hl < s["maxd"]
                   else f"advantage NOT halved within {s['maxd']} frames (persists)")
        print(f"[{name}] dist-0 advantage +{s['gap0']:.0f} var-Lap (+{100*s['rel0']:.0f}% sharper) "
              f"=> {verdict}")
        print(f"          deployable (with fallback re-SR): advantage {s['gap_full0']:.0f} -> "
              f"{s['gap_full_tail']:.0f} over the chain (max fallback {s['fb_max']:.0f}%); "
              f"tOF prop chain compact={s['tof_c']:.3f} x4plus={s['tof_h']:.3f}")
    print(f"\nwrote -> {OUT}/detail_drift.png, detail_drift.csv")


if __name__ == "__main__":
    main()
