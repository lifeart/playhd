#!/usr/bin/env python3
"""
exp3_visual / V2 -- motion-modulated grain on static regions. Talking-head window C.

Measures, on the GRAINED output, with vs without motion modulation:
  * background (STATIC-region) |Delta F|  -- the direct grain-flicker number,
  * DYNAMIC-region |Delta F|              -- must stay HIGH (fresh filmic grain kept on motion),
  * overall tOF                            -- temporal stability vs decoded LR,
  * RAW grain-field frame-to-frame correlation, split static/dynamic (independence proof).
Writes an amplified-diff artifact (static background patch: full grain vs modulated).
"""
import os

import cv2
import numpy as np

import common as C
import motion_grain as mg


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
START, NF, SCALE = 5000, 48, 4


def masked_corr_pairs(fields, mask):
    """Mean Pearson correlation of consecutive RAW grain fields, restricted to `mask`."""
    cs = []
    for t in range(1, len(fields)):
        a, b = fields[t - 1][mask], fields[t][mask]
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt(float((a * a).sum()) * float((b * b).sum()))
        if den > 1e-9:
            cs.append(float((a * b).sum()) / den)
    return float(np.mean(cs)) if cs else float("nan")


def main():
    os.makedirs(OUT, exist_ok=True)
    frames, h_lr, w_lr, types = C.decode_window(START, NF)
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    print(f"V2 window C: start {START}, {len(frames)}f  LR {w_lr}x{h_lr} -> HD {w_hd}x{h_hd}  types={types}")

    # ---- propagated recon (compact SR, the realistic instant-mode output) ----
    R, _ = C.reconstruct_window(frames, SCALE, sr_mode="realesrgan", occ="full")
    recon = [R[i]["recon"] for i in range(len(frames))]
    C.free_gpu()

    # ---- motion gate (free; codec-MV temporal-mean) + static/dynamic measurement masks ----
    static_lr, dynamic_lr, meanmag, info = C.rq.region_masks(frames, h_lr, w_lr, 45.0, 80.0)
    a_lr = C.rq.window_static_weight(meanmag, lo=0.2, hi=1.0, feather=61)     # 1=static
    a_hd = cv2.resize(a_lr, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
    static_hd = cv2.resize(static_lr.astype(np.uint8), (w_hd, h_hd), cv2.INTER_NEAREST).astype(bool)
    dynamic_hd = cv2.resize(dynamic_lr.astype(np.uint8), (w_hd, h_hd), cv2.INTER_NEAREST).astype(bool)
    print(f"motion (LR px/f): mean={info['mean_all']:.2f} p50={info['p50']:.2f} p90={info['p90']:.2f}  "
          f"static cov={100*info['static_cov']:.0f}% (mot {info['static_motion']:.2f})  "
          f"dynamic cov={100*info['dynamic_cov']:.0f}% (mot {info['dynamic_motion']:.2f})  "
          f"a_hd mean={a_hd.mean():.3f}")

    # ---- grain sequences (shared template = same grain recipe, fair comparison) ----
    tmpl = C._grain.make_template(h_hd, w_hd, seed=0)
    full, mod, red = [], [], []
    g_full, g_mod = [], []                       # RAW additive grain fields
    for i in range(len(frames)):
        f_img, gf = C._grain.apply_grain(recon[i], i, "med", template=tmpl, return_grain=True)
        m_img, gm = mg.apply_grain_motion(recon[i], i, a_hd, "med", template=tmpl,
                                          mode="frozen", return_grain=True)
        r_img = mg.apply_grain_motion(recon[i], i, a_hd, "med", template=tmpl,
                                      mode="reduced", static_floor=0.25)
        full.append(f_img); mod.append(m_img); red.append(r_img)
        g_full.append(gf); g_mod.append(gm)

    # ---- |Delta F| (luma) by region + overall tOF ----
    def row(seq):
        return (C.dframe_luma(seq, static_hd), C.dframe_luma(seq, dynamic_hd),
                C.dframe_luma(seq, None), C.tof_lr(seq, [frames[i][1] for i in range(len(frames))]))
    rows = {"recon (no grain)": row(recon),
            "full grain (current)": row(full),
            "motion grain frozen (V2)": row(mod),
            "motion grain reduced": row(red)}
    print("\n================ V2 |Delta F| (luma) + tOF ================")
    print(f"{'sequence':28s}  bg/STATIC|dF|  DYN|dF|   ALL|dF|   tOF(LR)")
    for k, (s, dy, al, tf) in rows.items():
        print(f"{k:28s}  {s:9.3f}    {dy:7.3f}  {al:7.3f}   {tf:6.3f}")

    # ---- grain temporal independence on the RAW field (the gotcha-safe check) ----
    print("\n---- RAW grain-field frame-to-frame correlation (independence; ~0 = filmic) ----")
    print(f"full grain : STATIC corr={masked_corr_pairs(g_full, static_hd):.4f}  "
          f"DYNAMIC corr={masked_corr_pairs(g_full, dynamic_hd):.4f}  "
          f"ALL corr={masked_corr_pairs(g_full, np.ones_like(static_hd)):.4f}")
    moving_hd = a_hd < 0.1                          # pixels the GATE calls genuinely moving
    print(f"V2 frozen  : STATIC corr={masked_corr_pairs(g_mod, static_hd):.4f} (->1 = frozen, no flicker)  "
          f"DYNAMIC-mask corr={masked_corr_pairs(g_mod, dynamic_hd):.4f}  "
          f"GATE-moving(a<0.1) corr={masked_corr_pairs(g_mod, moving_hd):.4f} (->0 = fresh filmic grain)")
    print(f"   context: mean gate a on DYNAMIC-mask={a_hd[dynamic_hd].mean():.3f}, "
          f"on STATIC-mask={a_hd[static_hd].mean():.3f}; gate-moving(a<0.1) cov={100*moving_hd.mean():.1f}%")

    # ---- artifacts: amplified diff on a STATIC background crop (full vs V2) ----
    ys, xs = np.where(static_hd)
    cy, cx = int(np.median(ys)), int(np.median(xs))      # a representative static pixel
    cs = 256
    y0 = int(np.clip(cy - cs // 2, 0, h_hd - cs)); x0 = int(np.clip(cx - cs // 2, 0, w_hd - cs))
    crop = lambda im: im[y0:y0 + cs, x0:x0 + cs]
    t = len(frames) // 2
    panels = [C.label(C.amplified_diff(crop(recon[t - 1]), crop(recon[t])), "recon dF"),
              C.label(C.amplified_diff(crop(full[t - 1]), crop(full[t])), "FULL grain dF"),
              C.label(C.amplified_diff(crop(mod[t - 1]), crop(mod[t])), "V2 frozen dF")]
    cv2.imwrite(os.path.join(OUT, "v2_static_ampdiff.png"), np.concatenate(panels, axis=1))
    # gate heatmap (red=static/frozen, blue=moving/fresh) over a recon frame
    heat = cv2.applyColorMap((a_hd * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(recon[t], cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUT, "v2_gate.png"),
                C.label(cv2.addWeighted(base, 0.5, heat, 0.5, 0), "motion gate a (red=static)"))
    print(f"\nwrote artifacts -> {OUT}/v2_static_ampdiff.png, v2_gate.png")
    C.free_gpu()
    return rows


if __name__ == "__main__":
    main()
