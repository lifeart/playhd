#!/usr/bin/env python3
"""
E4 Lever A -- FSR2 color-box clamping (UNTESTED transferable FSR2 idea).

FSR2's depth-based disocclusion does NOT transfer (decoded video has no depth buffer), but the
RGB-neighborhood color-box clamp might: clamp each warped-anchor HD pixel to a neighborhood
COLOR BOX of the CURRENT LR frame (channel-wise mean +/- gamma*std over a small LR window,
upscaled to HD) to suppress ghosting from rate-distortion MVs that point to "visually similar
but geometrically wrong" content.

The catch the FSR2 docs warn about: here the warped anchor is MORE detailed than the current
LR, so a TIGHT box (small gamma) clips exactly the SR detail you want to keep. So we sweep
LOOSE boxes gamma in {1,2,4,8,inf} (inf = no clamp = identity baseline) and ask:
  is there a gamma that REDUCES GHOSTING without CLIPPING SR DETAIL?

Honest metrics (ghosting is a temporal/structural defect -- never a single NR number):
  * GHOSTING proxy  = high-frequency divergence |HF(recon) - HF(perframe)| over the WARPED
    (non-occluded) region on the highest-motion frames. perframe = per-frame SR of the ACTUAL
    current LR (geometrically correct, less temporal detail), so HF divergence there = misplaced
    fine detail = ghosting. Also restricted to LOW-reactive warped pixels (the "visually similar
    but geometrically wrong" MVs the occlusion mask MISSES -- the residual ghosting this lever
    targets). Plus temporal |dF| of the recon sequence. Plus VISUAL crops (the real arbiter).
  * DETAIL preservation = var-Laplacian on a STATIC detailed crop (near-zero MV across the
    window) clamped vs unclamped, and PSNR(clamped, unclamped) on that crop.

The engine (derisk/gpu_ops) is imported READ-ONLY; the clamp is applied POST-HOC to the
recon that reconstruct() produced. Run with the COMPACT realtime anchor across the full 48-frame
window (fast, robust under MPS contention); the detail-clipping conclusion is model-agnostic (the
box clips ANY HD detail finer than the LR-upscaled std), and an additional HEAVY x4plus static
crop confirms the high-detail model loses MORE to a tight box.
"""
import os
import sys
import gc
import json

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", "..", "prototype"))
sys.path.insert(0, _PROTO)

import sr as srmod                                   # READ-ONLY
import derisk
from derisk import decode_lr_and_mvs, build_perframe_cache, reconstruct, warp_lr, build_lr_flow

SAMPLE = os.path.abspath(os.path.join(_HERE, "..", "..", "sample.mp4"))
ART = os.path.join(_HERE, "artifacts")
os.makedirs(ART, exist_ok=True)

SCALE = 4
WIN_LR = 3                                            # LR color-box window (3 -> 12px HD nbhd)
GAMMAS = [1.0, 2.0, 4.0, 8.0, float("inf")]          # inf == no clamp (baseline)


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:                            # never silently swallow
        print(f"  [warn] empty_cache failed: {e}")


def var_lap(img_u8):
    g = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def psnr(a_u8, b_u8):
    mse = np.mean((a_u8.astype(np.float64) - b_u8.astype(np.float64)) ** 2)
    return float("inf") if mse <= 1e-12 else float(10.0 * np.log10(255.0 ** 2 / mse))


def hf(img_u8, ksize=5):
    """High-frequency residual = image - gaussian-blur(image), float32."""
    f = img_u8.astype(np.float32)
    return f - cv2.GaussianBlur(f, (ksize, ksize), 0)


def color_box(lr_u8, win=WIN_LR, scale=SCALE):
    """Channel-wise (mean, std) over a win x win LR box, bilinear-upscaled to HD.
    Returns (mean_hd, std_hd) float32 HxWx3 at HD resolution."""
    f = lr_u8.astype(np.float32)
    k = (win, win)
    mean = cv2.boxFilter(f, -1, k, normalize=True, borderType=cv2.BORDER_REPLICATE)
    meansq = cv2.boxFilter(f * f, -1, k, normalize=True, borderType=cv2.BORDER_REPLICATE)
    var = np.clip(meansq - mean * mean, 0.0, None)
    std = np.sqrt(var)
    h, w = lr_u8.shape[:2]
    sz = (w * scale, h * scale)
    mean_hd = cv2.resize(mean, sz, interpolation=cv2.INTER_LINEAR)
    std_hd = cv2.resize(std, sz, interpolation=cv2.INTER_LINEAR)
    return mean_hd, std_hd


def clamp_recon(recon_hd_u8, mean_hd, std_hd, gamma):
    if gamma == float("inf"):
        return recon_hd_u8
    lo = mean_hd - gamma * std_hd
    hi = mean_hd + gamma * std_hd
    out = np.clip(recon_hd_u8.astype(np.float32), lo, hi)
    return out.round().astype(np.uint8)


def reactive_lr(frames, i, ref_i):
    """|LR_cur - warp(LR_prev by past MVs)| mean over channels, at LR. The bad-MV indicator
    occlusion_mask_lr uses internally (recomputed here so we can locate ghost-prone pixels)."""
    _, lr_cur, mvs = frames[i]
    _, lr_ref, _ = frames[ref_i]
    fx, fy = build_lr_flow(mvs, lr_cur.shape[0], lr_cur.shape[1], want="past")
    pred = warp_lr(lr_ref, fx, fy).astype(np.float32)
    react = np.abs(lr_cur.astype(np.float32) - pred).mean(axis=2)
    return react, np.isfinite(fx)


def main():
    print("decoding window A (start 0, 48 frames, high-motion) ...")
    frames = decode_lr_and_mvs(SAMPLE, 0, 48)
    N = len(frames)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    print(f"  {N} LR frames {frames[0][1].shape} -> HD {h_hd}x{w_hd}")

    print("building per-frame COMPACT SR cache (realtime anchor) ...")
    pf_cache = build_perframe_cache(frames, w_hd, h_hd, "realesrgan")
    free_gpu()

    print("reconstruct (numpy backend, occ=full, x4plus-style I/P chain) ...")
    rows, R = reconstruct(frames, frames if False else None, SCALE, True, "full",
                          pf_cache, set(), backend="numpy")
    # R[i]: recon (HD u8), perframe (HD u8), mask (occ bool HD), hole_frac, dist, type, is_anchor

    # --- locate the highest-motion non-anchor frames (most ghost-prone) ---
    motion = []
    backbone = derisk.backbone_indices(frames)
    for i in range(1, N):
        if R[i]["is_anchor"]:
            continue
        prev_bb = max([b for b in backbone if b < i], default=None)
        ref_i = prev_bb if (frames[i][0] != "B" and prev_bb is not None) else (i - 1)
        try:
            react, valid = reactive_lr(frames, i, ref_i)
        except Exception as e:
            print(f"  [warn] reactive_lr failed at {i}: {e}")
            continue
        motion.append((float(np.mean(react)), i, ref_i))
    motion.sort(reverse=True)
    hot = [m[1] for m in motion[:6]]                  # top-6 highest-motion frames
    print(f"  highest-motion frames: {[(i, round(s,1)) for s,i,_ in motion[:6]]}")

    # --- find a STATIC TEXTURED HD crop: low temporal LR change + high HF-energy (texture,
    #     not a single edge) + mid-range brightness (avoid black/white letterbox bars) ---
    lr_stack = np.stack([frames[i][1].astype(np.float32) for i in range(N)], 0)
    temporal = np.abs(np.diff(lr_stack, axis=0)).mean(axis=(0, 3))   # [h_lr,w_lr] motion energy
    cs_lr = 32                                         # LR crop size
    best = None
    detail_ref = R[0]["perframe"]                      # anchor per-frame SR (HD) for detail scoring
    hf_ref = np.abs(hf(detail_ref)).mean(axis=2)       # HD per-pixel texture energy
    for yy in range(0, h_lr - cs_lr, 16):
        for xx in range(0, w_lr - cs_lr, 16):
            mot = temporal[yy:yy + cs_lr, xx:xx + cs_lr].mean()
            crop = detail_ref[yy * SCALE:(yy + cs_lr) * SCALE, xx * SCALE:(xx + cs_lr) * SCALE]
            med = float(np.median(crop))
            if med < 50 or med > 205:                  # skip letterbox / blown-out flat bars
                continue
            tex = float(hf_ref[yy * SCALE:(yy + cs_lr) * SCALE,
                               xx * SCALE:(xx + cs_lr) * SCALE].mean())   # texture energy
            score = tex / (1.0 + mot)                  # textured AND static
            if best is None or score > best[0]:
                best = (score, yy, xx, mot, tex)
    _, sy, sx, smot, svl = best
    Sy, Sx, Scs = sy * SCALE, sx * SCALE, cs_lr * SCALE
    print(f"  static textured crop: LR ({sx},{sy}) motion={smot:.2f} HFenergy={svl:.2f} "
          f"varLap={var_lap(detail_ref[Sy:Sy+Scs, Sx:Sx+Scs]):.0f}")

    # ============ gamma sweep ============
    print(f"\ngamma sweep (WIN_LR={WIN_LR}); ghosting over warped region of hot frames, "
          f"detail on static crop\n")
    box_cache = {i: color_box(frames[i][1]) for i in set(hot)}
    # also need a static crop's box from a representative static frame (frame 0)
    box0 = color_box(frames[0][1])

    results = []
    # for temporal |dF| we clamp the whole sequence per gamma
    for gamma in GAMMAS:
        # ghosting proxy over hot frames
        gh_all, gh_low, n_low_tot, clipfrac = [], [], 0, []
        for i in hot:
            mean_hd, std_hd = box_cache[i]
            recon = R[i]["recon"]
            clamped = clamp_recon(recon, mean_hd, std_hd, gamma)
            occ = R[i]["mask"]
            warped = ~occ                                  # non-occluded = warped (ghost-prone)
            # reactive (HD) to split low- vs high-reactive warped pixels
            prev_bb = max([b for b in backbone if b < i], default=None)
            ref_i = prev_bb if (frames[i][0] != "B" and prev_bb is not None) else (i - 1)
            react, _ = reactive_lr(frames, i, ref_i)
            react_hd = cv2.resize(react, (w_hd, h_hd), interpolation=cv2.INTER_NEAREST)
            low = warped & (react_hd <= 8.0)               # warp accepted (mask-passed) but maybe ghosting
            hfc = hf(clamped)
            hfp = hf(R[i]["perframe"])
            div = np.abs(hfc - hfp).mean(axis=2)           # HF divergence map
            if warped.any():
                gh_all.append(float(div[warped].mean()))
            if low.any():
                gh_low.append(float(div[low].mean()))
                n_low_tot += int(low.sum())
            # how much did the clamp actually change (fraction of pixels clipped)
            if gamma != float("inf"):
                changed = (clamped != recon).any(axis=2)
                clipfrac.append(float(changed[warped].mean()) if warped.any() else 0.0)
        # temporal |dF| over the hot-frame recon (clamped) -- consecutive frames in window
        dF = []
        hot_sorted = sorted(set(hot))
        prev = None
        for i in range(min(hot_sorted), max(hot_sorted) + 1):
            mean_hd, std_hd = color_box(frames[i][1])
            c = clamp_recon(R[i]["recon"], mean_hd, std_hd, gamma)
            if prev is not None:
                dF.append(float(np.abs(c.astype(np.float32) - prev).mean()))
            prev = c.astype(np.float32)

        # DETAIL: static crop, clamp using frame-0 box, vs unclamped
        sc_un = R[0]["perframe"][Sy:Sy + Scs, Sx:Sx + Scs]
        mean0, std0 = box0
        sc_cl = clamp_recon(R[0]["perframe"], mean0, std0, gamma)[Sy:Sy + Scs, Sx:Sx + Scs]
        vl_un = var_lap(sc_un)
        vl_cl = var_lap(sc_cl)
        detail_psnr = psnr(sc_cl, sc_un)

        res = dict(
            gamma=("inf" if gamma == float("inf") else gamma),
            ghost_hf_warped=float(np.mean(gh_all)) if gh_all else float("nan"),
            ghost_hf_lowreact=float(np.mean(gh_low)) if gh_low else float("nan"),
            clip_frac_warped=float(np.mean(clipfrac)) if clipfrac else 0.0,
            dF=float(np.mean(dF)) if dF else float("nan"),
            static_varlap=vl_cl,
            static_varlap_ret=vl_cl / vl_un if vl_un > 0 else float("nan"),
            static_detail_psnr=detail_psnr,
        )
        results.append(res)
        print(f"  gamma={res['gamma']:>4}: ghost_HF(warped)={res['ghost_hf_warped']:.3f} "
              f"ghost_HF(lowReact)={res['ghost_hf_lowreact']:.3f} "
              f"clip%={res['clip_frac_warped']*100:5.1f} |dF|={res['dF']:.3f}  "
              f"static varLap={vl_cl:6.0f} ret={res['static_varlap_ret']*100:5.1f}% "
              f"PSNR(detail)={detail_psnr:5.1f}dB")

    # ============ heavy x4plus static-detail confirmation (1 SR call) ============
    print("\nheavy x4plus static-detail clipping confirmation (1 anchor SR) ...")
    heavy_hd = srmod.upscale(frames[0][1], model="realesrgan-x4plus")
    free_gpu()
    mean0, std0 = box0
    heavy_rows = []
    for gamma in GAMMAS:
        cl = clamp_recon(heavy_hd, mean0, std0, gamma)
        sc_un = heavy_hd[Sy:Sy + Scs, Sx:Sx + Scs]
        sc_cl = cl[Sy:Sy + Scs, Sx:Sx + Scs]
        vl_un, vl_cl = var_lap(sc_un), var_lap(sc_cl)
        heavy_rows.append(dict(gamma=("inf" if gamma == float("inf") else gamma),
                               static_varlap=vl_cl, ret=vl_cl / vl_un if vl_un > 0 else float("nan"),
                               psnr=psnr(sc_cl, sc_un)))
        print(f"  gamma={heavy_rows[-1]['gamma']:>4}: heavy static varLap={vl_cl:6.0f} "
              f"ret={heavy_rows[-1]['ret']*100:5.1f}%  PSNR(detail)={heavy_rows[-1]['psnr']:5.1f}dB")

    # ============ visual crops on the single hottest frame ============
    hi = hot[0]
    mean_hd, std_hd = box_cache[hi]
    recon = R[hi]["recon"]
    # a ghost-prone crop: pick the warped region with highest HF divergence vs perframe
    occ = R[hi]["mask"]; warped = ~occ
    div_full = np.abs(hf(recon) - hf(R[hi]["perframe"])).mean(axis=2)
    div_full[~warped] = -1
    bright = recon.astype(np.float32).mean(axis=2)     # avoid letterbox: require mid-brightness
    cs = 256
    by, bx, bestd = 0, 0, -1
    for yy in range(0, h_hd - cs, 64):
        for xx in range(0, w_hd - cs, 64):
            blk = div_full[yy:yy + cs, xx:xx + cs]
            med_b = float(np.median(bright[yy:yy + cs, xx:xx + cs]))
            if med_b < 50 or med_b > 205 or (blk < 0).mean() > 0.3:   # textured, mostly-warped
                continue
            d = blk[blk >= 0].mean()
            if d > bestd:
                bestd, by, bx = d, yy, xx
    panels = []
    for gamma in GAMMAS:
        c = clamp_recon(recon, mean_hd, std_hd, gamma)[by:by + cs, bx:bx + cs]
        panels.append(c)
    pf_crop = R[hi]["perframe"][by:by + cs, bx:bx + cs]
    strip = np.concatenate(panels + [pf_crop], axis=1)
    cv2.imwrite(os.path.join(ART, f"clamp_ghost_f{hi}_g1_g2_g4_g8_inf_perframe.png"),
                cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    # static detail strip (compact)
    sd = [clamp_recon(R[0]["perframe"], box0[0], box0[1], g)[Sy:Sy + Scs, Sx:Sx + Scs]
          for g in GAMMAS]
    cv2.imwrite(os.path.join(ART, "clamp_staticdetail_compact_g1_g2_g4_g8_inf.png"),
                cv2.cvtColor(np.concatenate(sd, axis=1), cv2.COLOR_RGB2BGR))
    sd_h = [clamp_recon(heavy_hd, box0[0], box0[1], g)[Sy:Sy + Scs, Sx:Sx + Scs] for g in GAMMAS]
    cv2.imwrite(os.path.join(ART, "clamp_staticdetail_heavy_g1_g2_g4_g8_inf.png"),
                cv2.cvtColor(np.concatenate(sd_h, axis=1), cv2.COLOR_RGB2BGR))

    with open(os.path.join(ART, "clamp_results.json"), "w") as f:
        json.dump(dict(win_lr=WIN_LR, hot_frames=[int(i) for i in hot],
                       static_crop_lr=[int(sx), int(sy)], compact=results, heavy=heavy_rows), f, indent=2)
    print(f"\nartifacts -> {ART}")
    print(f"visual: clamp_ghost_f{hi}_*.png (g1|g2|g4|g8|inf|perframe), "
          f"clamp_staticdetail_{{compact,heavy}}_*.png")


if __name__ == "__main__":
    main()
