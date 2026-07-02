#!/usr/bin/env python3
"""
r12_e4 -- REAL libx264 degradation pipeline for codec-matched SR training.

WHY: stock Real-ESRGAN/x4plus degradation is JPEG-only (`DiffJPEG`) -- it has NEVER
seen an H.264 artifact. That is the "fake-detail trap": a model trained on JPEG/bicubic
degradation hallucinates on real H.264 (measured in r12_e3). This module produces LR
frames via a REAL ffmpeg libx264 encode->decode round-trip, matched to playhd's
deployment CRF range (23..40), so a model trained on these pairs sees genuine H.264
blocking + mosquito/ringing + deblock-smoothing, not bicubic softness or JPEG 8x8 blocks.

Protocol (mirrors web_spike/eval_model_options.degrade_h264, generalised + randomised):
  HR frame (WxH, e.g. 640x320)
    -> INTER_AREA downscale to SD (W/scale, H/scale)     -- the "true SD"
    -> libx264 encode  (crf ~U[23,40], preset, GOP, yuv420p)   -- REAL H.264 artifacts
    -> decode back to RGB                                 -- the degraded LR

The SR model (scale x2 in the demo) maps this LR back up to the HR target.

Anchor realism: playhd only ever SR's ANCHOR frames, which are intra-coded, so the
default GOP=1 (all-intra) is the matched degradation for the anchor. `gop>1` with a
frame *window* introduces P-frame artifacts (a full-run robustness knob).

CLI:
  python degrade.py --validate      # artifact-validation crops + blockiness numbers -> out/
  python degrade.py --dump-pairs N  # write N (hr,lr) preview pairs -> out/pairs/
"""
import os
import sys
import argparse
import random

import numpy as np
import cv2
import av

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
OUT = os.path.join(_HERE, "out")

# Realistic streaming encode settings sampled per-frame (matched to deployment).
CRF_RANGE = (23, 40)                                   # playhd deployment CRF band
PRESETS = ["veryfast", "faster", "fast", "medium"]     # realistic live/VOD presets
DEFAULT_SCALE = 2                                       # SD->HD demo scale (SPAN x2)


# --------------------------------------------------------------------------- #
# Frame decode (reused convention from web_spike/eval_model_options.decode_frames)
# --------------------------------------------------------------------------- #
def decode_frames(path, idxs):
    """Decode requested display-order frame indices as HxWx3 uint8 RGB."""
    want = set(idxs)
    got = {}
    cont = av.open(path)
    vs = cont.streams.video[0]
    i = 0
    for frame in cont.decode(vs):
        if i in want:
            got[i] = frame.to_ndarray(format="rgb24")
        i += 1
        if len(got) == len(want):
            break
    cont.close()
    return [got[k] for k in idxs if k in got]


def decode_n(path, n, stride=1, start=0):
    """Decode the first `n` frames (every `stride`) as a list of HxWx3 uint8 RGB."""
    out = []
    cont = av.open(path)
    vs = cont.streams.video[0]
    i = 0
    for frame in cont.decode(vs):
        if i >= start and (i - start) % stride == 0:
            out.append(frame.to_ndarray(format="rgb24"))
            if len(out) >= n:
                break
        i += 1
    cont.close()
    return out


# --------------------------------------------------------------------------- #
# Core: REAL libx264 encode->decode round-trip
# --------------------------------------------------------------------------- #
def _libx264_roundtrip(sd_frames, crf, preset, gop, bf=0):
    """Encode a list of same-size SD RGB frames with libx264 and decode them back.
    Returns a list of decoded HxWx3 uint8 RGB frames (H.264-degraded)."""
    h, w = sd_frames[0].shape[:2]
    tmp = os.path.join(OUT, f"_rt_{os.getpid()}_{random.randint(0,1<<30)}.mp4")
    os.makedirs(OUT, exist_ok=True)
    cont = av.open(tmp, "w")
    st = cont.add_stream("libx264", rate=25)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": str(crf), "preset": preset, "g": str(gop), "bf": str(bf)}
    for f in sd_frames:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
        for p in st.encode(vf):
            cont.mux(p)
    for p in st.encode():
        cont.mux(p)
    cont.close()
    out = []
    c2 = av.open(tmp)
    for frame in c2.decode(c2.streams.video[0]):
        out.append(frame.to_ndarray(format="rgb24"))
    c2.close()
    os.remove(tmp)
    return out


def degrade_frame(hr_rgb, crf=None, preset=None, gop=1, scale=DEFAULT_SCALE, rng=None):
    """HR frame -> SD downscale -> REAL libx264 (all-intra) -> decode -> degraded LR.
    crf/preset default to a random draw from the deployment band if None."""
    rng = rng or random
    if crf is None:
        crf = rng.randint(*CRF_RANGE)
    if preset is None:
        preset = rng.choice(PRESETS)
    h, w = hr_rgb.shape[:2]
    sd = cv2.resize(hr_rgb, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    lr = _libx264_roundtrip([sd], crf, preset, gop)[0]
    return lr, dict(crf=crf, preset=preset, gop=gop)


def build_pairs(hr_frames, scale=DEFAULT_SCALE, seed=0, gop=1):
    """Return list of (hr_uint8, lr_uint8, meta) with per-frame randomised libx264 params.
    HR is the target (WxH); LR is (W/scale, H/scale). This is the trainer's pair source."""
    rng = random.Random(seed)
    pairs = []
    for hr in hr_frames:
        lr, meta = degrade_frame(hr, gop=gop, scale=scale, rng=rng)
        pairs.append((hr, lr, meta))
    return pairs


# --------------------------------------------------------------------------- #
# Artifact validation: prove the LR carries H.264 blocking (not bicubic / JPEG)
# --------------------------------------------------------------------------- #
def blockiness(gray, grid=8):
    """Mean |grad| ACROSS block-grid boundaries minus mean |grad| at interior columns/rows.
    Compression (JPEG/H.264) elevates energy on the transform grid -> positive score.
    A clean resample (bicubic) has ~0 (no grid). Cheap, standard blocking proxy."""
    g = gray.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1))       # horizontal gradient magnitude
    dy = np.abs(np.diff(g, axis=0))       # vertical gradient magnitude
    # boundary columns are at grid, grid*2, ... (diff index grid-1, 2*grid-1, ...)
    bcol = np.arange(grid - 1, dx.shape[1], grid)
    icol = np.setdiff1d(np.arange(dx.shape[1]), bcol)
    brow = np.arange(grid - 1, dy.shape[0], grid)
    irow = np.setdiff1d(np.arange(dy.shape[0]), brow)
    h_block = dx[:, bcol].mean() - dx[:, icol].mean()
    v_block = dy[brow, :].mean() - dy[irow, :].mean()
    return float((h_block + v_block) / 2.0)


def _jpeg_lr(hr, q, scale):
    """Comparison degradation: SD downscale + JPEG (DiffJPEG-equivalent, what x4plus saw)."""
    h, w = hr.shape[:2]
    sd = cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(sd, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def _bicubic_lr(hr, scale):
    """Comparison degradation: clean bicubic SD downscale (no compression)."""
    h, w = hr.shape[:2]
    return cv2.resize(hr, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


def _crop(img, cx, cy, s):
    return img[cy:cy + s, cx:cx + s]


def validate(src, idx=150, scale=DEFAULT_SCALE):
    """Save side-by-side crops + blockiness numbers proving H.264 artifacts."""
    os.makedirs(OUT, exist_ok=True)
    hr = decode_frames(src, [idx])[0]
    h, w = hr.shape[:2]
    # a deliberately HIGH crf makes blocking obvious for the validation figure
    lr_h264_hi, _ = degrade_frame(hr, crf=38, preset="medium", scale=scale)
    lr_h264_lo, _ = degrade_frame(hr, crf=24, preset="medium", scale=scale)
    lr_jpeg = _jpeg_lr(hr, q=18, scale=scale)     # matched-ish visual quality JPEG
    lr_bic = _bicubic_lr(hr, scale)

    def g(x):
        return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

    rows = []
    for name, im in [("bicubic(clean)", lr_bic), ("jpeg q18", lr_jpeg),
                     ("h264 crf24", lr_h264_lo), ("h264 crf38", lr_h264_hi)]:
        b8 = blockiness(g(im), 8)
        b4 = blockiness(g(im), 4)   # H.264 also uses 4x4 transform -> 4-grid energy too
        rows.append((name, b8, b4))
        print(f"  {name:16s} blockiness(grid8)={b8:+.3f}  blockiness(grid4)={b4:+.3f}")

    # crop a textured/edge region for the visual figure (upscaled x4 nearest for print)
    sh, sw = lr_bic.shape[:2]
    cs = 48
    cx, cy = sw // 2 - cs // 2, sh // 2 - cs // 2
    tiles = []
    labels = ["bicubic", "jpeg_q18", "h264_crf24", "h264_crf38"]
    for im in [lr_bic, lr_jpeg, lr_h264_lo, lr_h264_hi]:
        cr = _crop(im, cx, cy, cs)
        cr = cv2.resize(cr, (cs * 5, cs * 5), interpolation=cv2.INTER_NEAREST)
        tiles.append(cr)
    strip = np.concatenate(tiles, axis=1)
    fig_p = os.path.join(OUT, "artifact_validation.png")
    cv2.imwrite(fig_p, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    # also full LR frames for reference
    for nm, im in [("val_bicubic", lr_bic), ("val_jpeg", lr_jpeg),
                   ("val_h264_crf24", lr_h264_lo), ("val_h264_crf38", lr_h264_hi)]:
        cv2.imwrite(os.path.join(OUT, nm + ".png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    print(f"\n  crops (nearest x5, order {labels}) -> {fig_p}")
    print(f"  full LR frames -> out/val_*.png")
    return rows


def dump_pairs(src, n, scale=DEFAULT_SCALE):
    os.makedirs(os.path.join(OUT, "pairs"), exist_ok=True)
    frames = decode_n(src, n, stride=17)
    pairs = build_pairs(frames, scale=scale, seed=1)
    for i, (hr, lr, meta) in enumerate(pairs):
        cv2.imwrite(os.path.join(OUT, "pairs", f"{i:02d}_hr.png"),
                    cv2.cvtColor(hr, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUT, "pairs", f"{i:02d}_lr_{meta['crf']}_{meta['preset']}.png"),
                    cv2.cvtColor(lr, cv2.COLOR_RGB2BGR))
    print(f"wrote {len(pairs)} pairs -> out/pairs/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(_ROOT, "web_spike", "sd600.mp4"))
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dump-pairs", type=int, default=0)
    ap.add_argument("--idx", type=int, default=150)
    args = ap.parse_args()
    if args.validate:
        print(f"# artifact validation on {os.path.basename(args.src)} frame {args.idx}")
        validate(args.src, args.idx)
    if args.dump_pairs:
        dump_pairs(args.src, args.dump_pairs)
    if not args.validate and not args.dump_pairs:
        print("nothing to do; pass --validate or --dump-pairs N")
