#!/usr/bin/env python3
"""
R8-E1 / MOVING graphic-edge stabilization -- shared harness (READ-ONLY on prototype/+server/).

The open V3 item is the MOVING graphic (scrolling ticker / sliding lower-third) whose sharp
high-contrast edges carry NON-zero codec MVs. The STATIC card was already settled NO-GO (R1-E3):
zero-MV skip-coding makes propagation an identity copy that out-stabilizes per-frame SR.

To reproduce the RD-MV artifact faithfully we must NOT inject a synthetic flow. Instead:
  1. decode a clean window of real sample.mp4 LR frames,
  2. overlay a synthetic moving high-contrast caption (known per-frame sub-pixel velocity),
  3. RE-ENCODE to H.264 with PyAV (libx264) so the caption gets REAL rate-distortion codec MVs,
  4. re-decode with +export_mvs and run the SHIPPED reconstruction path.

Honest metrics:
  * registered-dF (rDF): consecutive |luma diff| AFTER compensating the KNOWN ticker velocity
    (shift recon_t by +v to align onto recon_{t-1}); isolates shimmer from legitimate motion.
    This is the right flicker metric for MOVING content (raw |dF| is dominated by the motion).
  * tof_lr: derisk Farneback tOF vs decoded-LR (motion-compensated temporal stability), x-check.
  * raw |dF| on the graphic mask, split STATIC-phase vs MOVING-phase.
  * fallback%: fraction of graphic-region pixels the occlusion mask routes to per-frame SR
    (R[i]['mask']) -- the ADVERSARIAL self-healing check.
"""
import os
import sys
import gc

import av
import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.join(_HERE, "..", "..", "prototype")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

import derisk as d          # READ-ONLY
import region_quality as rq  # READ-ONLY

SAMPLE = os.path.join(_HERE, "..", "..", "sample.mp4")
TMP = os.path.join(_HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        print(f"  [free_gpu] note: {e}")


def luma(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)


# --------------------------------------------------------------------------- #
# 1. clean background frames
# --------------------------------------------------------------------------- #
def decode_clean_rgb(start, n):
    """Decode n real LR rgb frames from sample.mp4 (the realistic background)."""
    frames = d.decode_lr_and_mvs(SAMPLE, start, n)
    rgb = [f[1].copy() for f in frames]
    h, w = rgb[0].shape[:2]
    return rgb, h, w


# --------------------------------------------------------------------------- #
# 2. synthetic moving captions  (high-contrast text, KNOWN velocity)
# --------------------------------------------------------------------------- #
def _text_strip(width, height, text, fg=255, bg=0, scale=1.0, thick=2):
    """Render repeated high-contrast text across a wide strip (for a scrolling ticker)."""
    strip = np.full((height, width, 3), bg, np.uint8)
    x = 5
    while x < width:
        cv2.putText(strip, text, (x, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (fg, fg, fg), thick, cv2.LINE_AA)
        x += int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]) + 40
    return strip


def overlay_ticker(rgb_frames, h, w, v_lr=1.7, bar_top=None, bar_h=42,
                   text="BREAKING NEWS  USACHEV TODAY  LIVE  ", fg=245, bg=8,
                   alpha=1.0, tscale=0.9):
    """Opaque/translucent ticker FIXED at screen rows [bar_top, bar_top+bar_h); text scrolls
    LEFT at v_lr LR px/frame (sub-pixel via warpAffine). Returns (mod_frames, mask_lr_bool, v_lr).
    The bar position is fixed so the graphic region mask is constant; only the text translates."""
    if bar_top is None:
        bar_top = h - bar_h - 8
    n = len(rgb_frames)
    strip_w = w + int(np.ceil(v_lr * n)) + 8
    strip = _text_strip(strip_w, bar_h, text, fg=fg, bg=bg, scale=tscale)
    out = []
    for i in range(n):
        off = v_lr * i
        M = np.float32([[1, 0, -off], [0, 1, 0]])
        win = cv2.warpAffine(strip, M, (w, bar_h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        fr = rgb_frames[i].copy()
        roi = fr[bar_top:bar_top + bar_h]
        if alpha >= 0.999:
            roi[:] = win
        else:
            roi[:] = np.clip(alpha * win + (1 - alpha) * roi, 0, 255).astype(np.uint8)
        out.append(fr)
    mask = np.zeros((h, w), bool)
    mask[bar_top:bar_top + bar_h] = True
    return out, mask, v_lr


def overlay_lowerthird(rgb_frames, h, w, v_lr=4.0, hold_from=14, bar_h=60,
                       text="USACHEV  TODAY", fg=245, bg=10, alpha=0.78, tscale=1.1):
    """A lower-third that SLIDES UP into place over frames [0,hold_from) then HOLDS. Returns
    (mod_frames, per-frame mask list, bar_top list). Gives a MOVING phase + a SETTLED phase
    in one clip (the settled phase is the already-NO-GO static case, kept for contrast)."""
    n = len(rgb_frames)
    final_top = h - bar_h - 6
    bar = np.full((bar_h, w, 3), bg, np.uint8)
    cv2.putText(bar, text, (24, int(bar_h * 0.66)), cv2.FONT_HERSHEY_SIMPLEX,
                tscale, (fg, fg, fg), 2, cv2.LINE_AA)
    cv2.rectangle(bar, (8, 8), (16, bar_h - 8), (fg, 60, 60), -1)  # accent block
    out, masks, tops = [], [], []
    for i in range(n):
        if i < hold_from:
            top = final_top + (hold_from - i) * v_lr      # slides up
        else:
            top = final_top
        top_i = int(round(top))
        fr = rgb_frames[i].copy()
        y0 = max(top_i, 0); y1 = min(top_i + bar_h, h)
        by0 = y0 - top_i; by1 = by0 + (y1 - y0)
        roi = fr[y0:y1]
        roi[:] = np.clip(alpha * bar[by0:by1] + (1 - alpha) * roi, 0, 255).astype(np.uint8)
        m = np.zeros((h, w), bool); m[y0:y1] = True
        out.append(fr); masks.append(m); tops.append(top_i)
    return out, masks, tops


# --------------------------------------------------------------------------- #
# 3. RE-ENCODE to H.264 (real codec MVs) + re-decode
# --------------------------------------------------------------------------- #
def encode_h264(rgb_frames, path, crf=20, preset="medium", g=64, bf=2, fps=30):
    """Encode rgb frames to H.264 via PyAV/libx264. g>=window => single I-frame (max prop drift)."""
    cont = av.open(path, "w")
    st = cont.add_stream("libx264", rate=fps)
    st.width = rgb_frames[0].shape[1]
    st.height = rgb_frames[0].shape[0]
    st.pix_fmt = "yuv420p"
    st.options = {"crf": str(crf), "preset": preset, "g": str(g), "bf": str(bf)}
    for img in rgb_frames:
        fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        for pkt in st.encode(fr):
            cont.mux(pkt)
    for pkt in st.encode():
        cont.mux(pkt)
    cont.close()
    return path


def decode_mvs(path, n):
    """Re-decode with +export_mvs in display order -> derisk frame tuples (ptype, lr_rgb, mvs)."""
    return d.decode_lr_and_mvs(path, 0, n)


# --------------------------------------------------------------------------- #
# 4. shipped reconstruction (selectable occ mode) + per-frame SR cache
# --------------------------------------------------------------------------- #
def build_recon(frames, scale, sr_mode="realesrgan", occ="reactive"):
    """Build the per-frame-SR cache + shipped propagation recon (numpy backend = deterministic).
    anchor_set=set() => I-frames are the only anchors (= shipped instant/quality backbone).
    Returns (R, perframe_cache). R[i]['mask'] = pixels routed to per-frame SR (fallback)."""
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * scale, h_lr * scale
    pf = d.build_perframe_cache(frames, w_hd, h_hd, sr_mode)
    _, R = d.reconstruct(frames, None, scale, True, occ, pf, set(),
                         backend="numpy", collect_metrics=False, download_output=True)
    return R, pf


# --------------------------------------------------------------------------- #
# 5. metrics
# --------------------------------------------------------------------------- #
def raw_dframe(seq, mask):
    """Mean |luma dF| over consecutive frames restricted to mask (per-frame mask list OR one mask)."""
    vals = []
    prev = luma(seq[0])
    for t in range(1, len(seq)):
        cur = luma(seq[t])
        m = mask[t] if isinstance(mask, list) else mask
        vals.append(float(np.abs(cur - prev)[m].mean()))
        prev = cur
    return float(np.mean(vals)) if vals else float("nan")


def registered_dframe(seq, mask_hd, v_hd, margin_extra=3):
    """Motion-compensated flicker for a constant-velocity ticker: shift recon_t RIGHT by v_hd to
    align its (left-scrolling) text onto recon_{t-1}, then |luma dF| on the bar EXCLUDING the
    disocclusion column at the entering (right) edge. Removes the legitimate motion -> leaves
    shimmer/resample-jitter only. v_hd is the KNOWN HD velocity (exact, authored)."""
    h, w = mask_hd.shape
    cut = int(np.ceil(v_hd)) + margin_extra
    keep = mask_hd.copy()
    keep[:, w - cut:] = False          # drop newly-entered pixels (true disocclusion, not shimmer)
    vals = []
    prev = luma(seq[0])
    for t in range(1, len(seq)):
        cur = luma(seq[t])
        M = np.float32([[1, 0, v_hd], [0, 1, 0]])
        cur_al = cv2.warpAffine(cur, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
        vals.append(float(np.abs(cur_al - prev)[keep].mean()))
        prev = cur
    return float(np.mean(vals)) if vals else float("nan")


def tof_lr(seq, ref_lr):
    h_lr, w_lr = ref_lr[0].shape[:2]
    seq_lr = [cv2.resize(r, (w_lr, h_lr)) for r in seq]
    return d.tof(seq_lr, ref_lr)


def fallback_frac_on_mask(R, mask_hd, frame_range):
    """Mean fraction of graphic-region (mask_hd) pixels routed to per-frame SR fallback
    (R[i]['mask']) over frame_range. The ADVERSARIAL self-healing measurement.
    mask_hd may be a single HD bool or a per-frame list of HD bools."""
    fr = []
    for i in frame_range:
        m = R[i].get("mask")
        gm = mask_hd[i] if isinstance(mask_hd, list) else mask_hd
        if m is None:                  # anchor frame: whole frame is fresh SR -> 100% "fallback"
            fr.append(1.0)
            continue
        denom = gm.sum()
        fr.append(float((m & gm).sum()) / denom if denom else 0.0)
    return float(np.mean(fr)) if fr else float("nan"), fr


def upscale_mask(mask_lr, scale):
    h, w = mask_lr.shape
    return cv2.resize(mask_lr.astype(np.uint8), (w * scale, h * scale),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
