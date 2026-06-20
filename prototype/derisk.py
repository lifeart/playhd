#!/usr/bin/env python3
"""
playhd de-risk experiment — codec-motion-vector detail propagation for real-time SR.

Tests the single riskiest seam of the NEMO-style architecture (MobiCom 2020):
    Are H.264 motion vectors clean enough to WARP a super-resolved anchor/keyframe
    onto later frames, instead of running the SR network every frame?

Per frame (display order):
  * I-frame (anchor): "super-resolve" the LR frame -> HD reference. v1 uses bicubic as
    an SR *placeholder* so we measure WARP/DRIFT error in isolation, not SR quality.
  * P-frame: block MVs -> dense per-pixel fetch-flow (zero-order hold), scaled to HD,
    warp the HD reference (cv2.remap, bilinear = sub-pixel). Optionally add a NEMO-style
    bilinear-upscaled residual (residual = LR_cur - motion_comp(LR_prev)). Chained, so
    drift ("cache erosion") accumulates over the GOP exactly like NEMO.
  * Pixels with no MV (intra blocks) = disocclusion holes -> filled from per-frame bicubic
    (the simplest "re-run SR locally" fallback). Hole fraction is reported per frame.

Outputs (into --out dir): metrics.csv, curves.png (PSNR + holes vs frame),
erosion.png (propagated-vs-per-frame quality, the NEMO metric), and sample frames.

Usage:
  python3 derisk.py                          # synthetic clip: pan + moving occluder
  python3 derisk.py --input clip.mp4 --scale 3 --no-residual
"""
import argparse
import csv
import os
import time
from contextlib import contextmanager

import av
import cv2
import numpy as np
from av.sidedata.sidedata import Type as SDType

import grain as _grain   # Step 8: per-frame film-grain final pass (numpy, cv2-only, no torch)

_GRAIN_STRENGTHS = list(_grain.STRENGTHS)


# --------------------------------------------------------------------------- #
# Lightweight per-component profiler (Step 6). DISABLED by default -> the context
# managers are pure pass-throughs, so the synthetic regression stays byte-identical and
# there is zero numeric effect; it only records wall-clock (optionally MPS-synced) timings.
# --------------------------------------------------------------------------- #
class _Prof:
    """Records (component, frame_type, frame_index, ms) events. `sync` is set to
    torch.mps.synchronize for the torch backend so GPU timings are honest (kernels are async).
    `ftype`/`fidx` are stamped per frame by the reconstruct/decode loops."""

    def __init__(self):
        self.enabled = False
        self.sync = None
        self.ftype = "?"
        self.fidx = -1
        self.events = []

    def reset(self, enabled=False, sync=None):
        self.enabled = enabled
        self.sync = sync
        self.ftype = "?"
        self.fidx = -1
        self.events = []

    @contextmanager
    def time(self, name):
        if not self.enabled:
            yield
            return
        if self.sync is not None:
            self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync is not None:
                self.sync()
            self.events.append((name, self.ftype, self.fidx, (time.perf_counter() - t0) * 1000.0))

    def add(self, name, ms, ftype=None, fidx=None):
        if self.enabled:
            self.events.append((name, ftype or self.ftype, self.fidx if fidx is None else fidx, ms))


PROF = _Prof()

# Step-7 adaptive-mask telemetry: [#mask calls that fired the fwd-bwd splat, #total mask calls].
# Reset at the start of every reconstruct(); read by profile_e2e to report the fwd-bwd fire rate.
MASK_FIRES = [0, 0]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def psnr(a, b):
    return float(cv2.PSNR(a, b))


def psnr_lr_consistency(recon_hd, lr_true):
    """Honest real-footage fidelity metric: there is no HD ground truth for a real clip,
    but the DECODED LR frame IS ground truth at LR resolution. Downscale the reconstructed
    HD (INTER_AREA, the area-averaging inverse of upscaling) and compare to the true LR.
    Higher = the HD reconstruction stays faithful to known data; catches gross warp errors
    that the bicubic-fallback HIDES from psnr_prop_vs_perframe. Does NOT measure HF/SR
    quality (any plausible HD that downscales correctly scores high) -- only consistency."""
    h, w = lr_true.shape[:2]
    down = cv2.resize(recon_hd, (w, h), interpolation=cv2.INTER_AREA)
    return float(cv2.PSNR(down, lr_true))


def ssim(a, b):
    """Single-scale SSIM on luma (Gaussian 11x11, sigma 1.5)."""
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k, s = (11, 11), 1.5
    mu_a = cv2.GaussianBlur(ga, k, s)
    mu_b = cv2.GaussianBlur(gb, k, s)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = cv2.GaussianBlur(ga * ga, k, s) - mu_a2
    vb = cv2.GaussianBlur(gb * gb, k, s) - mu_b2
    vab = cv2.GaussianBlur(ga * gb, k, s) - mu_ab
    smap = ((2 * mu_ab + C1) * (2 * vab + C2)) / ((mu_a2 + mu_b2 + C1) * (va + vb + C2))
    return float(smap.mean())


def _farneback(a, b):
    return cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0)


def tof(seq, ref):
    """TecoGAN tOF: mean end-point error between the Farneback flow of the candidate
    sequence and of the reference sequence. Lower = less flicker / more temporally stable."""
    vals = []
    for t in range(1, len(seq)):
        d = _farneback(ref[t - 1], ref[t]) - _farneback(seq[t - 1], seq[t])
        vals.append(float(np.mean(np.sqrt(np.sum(d * d, axis=-1)))))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
# Synthetic clip: HD ground truth in memory + LR H.264 on disk
# --------------------------------------------------------------------------- #
def make_synthetic(out_lr, w_lr=640, h_lr=360, scale=3, n=24, fps=30):
    w_hd, h_hd = w_lr * scale, h_lr * scale
    rng = np.random.default_rng(7)
    pan_px = 4 * scale * n + 16  # HD pan budget (camera pans => bg slides left)
    cw = w_hd + pan_px

    # structured background: smooth gradients (recoverable) ...
    yy, xx = np.mgrid[0:h_hd, 0:cw].astype(np.float32)
    canvas = np.stack([
        128 + 110 * np.sin(xx / 220.0),
        128 + 110 * np.sin(yy / 160.0 + 1.0),
        128 + 110 * np.sin((xx + yy) / 300.0 + 2.0),
    ], axis=-1)
    canvas = np.ascontiguousarray(np.clip(canvas, 0, 255).astype(np.uint8))
    # ... sharp-edged shapes (edges = where warp errors show) ...
    for _ in range(40):
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, cw)), int(rng.integers(0, h_hd))
        if rng.random() < 0.5:
            cv2.circle(canvas, (x, y), int(rng.integers(8, 40)), col, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (x, y), (x + int(rng.integers(10, 50)),
                          y + int(rng.integers(10, 50))), col, -1)
    # ... and a resolution-chart band of thin lines (HF detail SR must preserve)
    by = h_hd // 2
    for x in range(0, cw, 4):
        if (x % 160) // 4 < 8:
            cv2.line(canvas, (x, by), (x, by + h_hd // 6), (15, 15, 15), 1)

    obj = cv2.GaussianBlur(rng.integers(0, 256, (h_hd // 3, h_hd // 4, 3),
                                        dtype=np.uint8), (3, 3), 0)
    obj_w = obj.shape[1]

    hd_frames = []
    for i in range(n):
        bg_off = i * 4 * scale
        f = canvas[:, bg_off:bg_off + w_hd, :].copy()
        # foreground square moves right faster than bg -> occlusion + disocclusion
        ox = (i * 7 * scale) % max(1, (w_hd - obj_w))
        oy = h_hd // 3
        f[oy:oy + obj.shape[0], ox:ox + obj_w] = obj
        hd_frames.append(np.ascontiguousarray(f))

    cont = av.open(out_lr, "w")
    st = cont.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w_lr, h_lr, "yuv420p"
    # single GOP (one I-frame at 0), no B-frames for a clean v1 forward chain
    st.options = {"crf": "18", "g": str(n + 1), "bf": "0"}
    for f in hd_frames:
        lr = cv2.resize(f, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(lr), format="rgb24")
        for p in st.encode(vf):
            cont.mux(p)
    for p in st.encode():
        cont.mux(p)
    cont.close()
    return hd_frames


# --------------------------------------------------------------------------- #
# Decode LR frames + per-frame motion vectors
# --------------------------------------------------------------------------- #
def decode_lr_and_mvs(path, start_frame=0, max_frames=None):
    """Decode LR frames + per-frame MVs in DISPLAY order. Optionally restrict to the
    window [start_frame, start_frame+max_frames) of a long clip. Decoding still runs
    sequentially from frame 0 (the H.264 reference chain must be reconstructed), but
    frames before the window are decoded WITHOUT the expensive rgb24/MV conversion, and
    we stop as soon as the window is full -- so a window deep into a 50k-frame file is
    cheap. start_frame=0, max_frames=None reproduces the original whole-file behavior."""
    cont = av.open(path)
    vs = cont.streams.video[0]
    vs.codec_context.options = {"flags2": "+export_mvs"}
    out = []
    idx = 0
    last_t = time.perf_counter()
    for frame in cont.decode(vs):
        if idx < start_frame:          # decode-only (refs), skip conversion
            idx += 1
            continue
        if max_frames is not None and len(out) >= max_frames:
            break
        img = frame.to_ndarray(format="rgb24")
        try:
            sd = frame.side_data.get(SDType.MOTION_VECTORS)
        except Exception:
            sd = None
        mvs = sd.to_ndarray() if sd is not None else None
        # PyAV pict_type is an int enum (1=I, 2=P, 3=B) with no .name
        ptype = {1: "I", 2: "P", 3: "B"}.get(int(frame.pict_type), "?")
        # decode timing: wall time since the previous in-window frame = decode-step + rgb/MV
        # conversion for THIS frame. The first in-window frame also absorbs the seek-to-window
        # cost, so the profiler reports the median over the window (seek excluded).
        PROF.add("decode", (time.perf_counter() - last_t) * 1000.0, ftype=ptype, fidx=len(out))
        out.append((ptype, img, mvs))
        idx += 1
        last_t = time.perf_counter()
    cont.close()
    return out


# --------------------------------------------------------------------------- #
# Warp machinery
# --------------------------------------------------------------------------- #
def build_lr_flow(mvs, h, w, want="all"):
    """Dense per-pixel fetch-flow at LR. flow[y,x] = (dx,dy): the source pixel in the
    REFERENCED frame is (x+dx, y+dy). NaN where no MV of the requested direction covers
    the pixel (intra block / hole in that direction).

    `want` selects MV records by the sign of the codec `source` field (verified: <0 =
    past reference, >0 = future reference; |source|==1 for this stream so it is the
    nearest such reference picture):
      'past'   -> source<0  (the only kind P-frames carry; the legacy forward chain),
      'future' -> source>0  (forward refs; B-frames only),
      'all'    -> any source (default; preserves legacy callers -- the synthetic clip is
                  encoded bf=0 so every P MV is source<0 and 'all' == 'past' there).
    B-frames must be built one direction at a time (past/future separately) so the two
    reference fields don't overwrite each other."""
    fx = np.full((h, w), np.nan, np.float32)
    fy = np.full((h, w), np.nan, np.float32)
    if mvs is None or len(mvs) == 0:
        return fx, fy
    for r in mvs:
        s = int(r["source"])
        if want == "past" and s >= 0:    # keep past refs only
            continue
        if want == "future" and s <= 0:  # keep future refs only
            continue
        ms = float(r["motion_scale"]) or 1.0
        dx = float(r["motion_x"]) / ms
        dy = float(r["motion_y"]) / ms
        bw, bh = int(r["w"]), int(r["h"])
        cx, cy = int(r["dst_x"]), int(r["dst_y"])  # dst_x/y = block CENTER
        x0, x1 = max(cx - bw // 2, 0), min(cx + bw // 2, w)
        y0, y1 = max(cy - bh // 2, 0), min(cy + bh // 2, h)
        fx[y0:y1, x0:x1] = dx
        fy[y0:y1, x0:x1] = dy
    return fx, fy


def _remap(src, fx, fy):
    h, w = fx.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mapx = gx + np.nan_to_num(fx)
    mapy = gy + np.nan_to_num(fy)
    return cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def warp_hd(ref_hd, fx_lr, fy_lr, scale):
    """Warp an HD reference using an LR flow field. Returns (warped_hd, hole_mask_hd)."""
    h_lr, w_lr = fx_lr.shape
    w_hd, h_hd = w_lr * scale, h_lr * scale
    fx_hd = cv2.resize(fx_lr, (w_hd, h_hd), interpolation=cv2.INTER_NEAREST) * scale
    fy_hd = cv2.resize(fy_lr, (w_hd, h_hd), interpolation=cv2.INTER_NEAREST) * scale
    hole = np.isnan(fx_hd)
    return _remap(ref_hd, fx_hd, fy_hd), hole


def warp_lr(ref_lr, fx_lr, fy_lr):
    return _remap(ref_lr, fx_lr, fy_lr)


def _add_res(warped, res_hd):
    """warped (uint8 HD) + optional bilinear-upscaled residual (float HD) -> uint8."""
    if res_hd is None:
        return warped.copy()
    return np.clip(warped.astype(np.float32) + res_hd, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Softmax splatting + forward-backward occlusion detection (all at LR)
# --------------------------------------------------------------------------- #
def softmax_splat(values, fx, fy, weight):
    """Forward bilinear splat (Niklaus & Liu, CVPR'20 style): source pixel (x,y) is
    splatted to target (x+fx, y+fy), accumulating values weighted by softmax(weight).
    Returns (splatted_values[H,W,C], total_weight[H,W]). NaN where nothing landed."""
    h, w = fx.shape
    vals = values.reshape(h, w, -1).astype(np.float64)
    c = vals.shape[2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    tx, ty = gx + fx, gy + fy
    wsm = np.exp(weight - np.nanmax(weight[np.isfinite(weight)]))  # stabilized softmax weight
    valid0 = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(wsm)
    x0, y0 = np.floor(tx).astype(int), np.floor(ty).astype(int)
    num = np.zeros((h, w, c))
    den = np.zeros((h, w))
    for dx in (0, 1):
        for dy in (0, 1):
            xs, ys = x0 + dx, y0 + dy
            wb = np.clip(1 - np.abs(tx - xs), 0, 1) * np.clip(1 - np.abs(ty - ys), 0, 1) * wsm
            m = valid0 & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & (wb > 0)
            idx = (ys[m], xs[m])
            np.add.at(den, idx, wb[m])
            for k in range(c):
                np.add.at(num[:, :, k], idx, wb[m] * vals[m][:, k])
    out = np.full((h, w, c), np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz][:, None]
    return out, den


ADAPTIVE_TAU = 0.06   # Step-7: reactive-fallback fraction above which 'adaptive' fires fwd-bwd.
# Tuned (tune_adaptive.py): 0.06 keeps the high-motion window at FULL mask quality (tOF/fallback
# match full, fires ~75% of directions) while skipping the splat on genuinely-clean directions;
# >=0.08 starts under-flagging bad MVs on high-motion. Talking-head B-frames also trip the
# per-direction trigger (~60% fire), so adaptive there lands near full cost, not reactive's floor.


def occlusion_mask_lr(fb_x, fb_y, lr_cur, lr_prev, tau_react=16.0, mode="full",
                      adaptive_tau=None):
    """Combine three cheap occlusion signals at LR into one 'unreliable pixel' mask:
      (a) intra holes  -> no MV at all (NaN flow);
      (b) reactive     -> high prediction residual |LR_cur - warp(LR_prev)| (bad/occluded MV);
      (c) fwd-bwd       -> Ruder et al. 2016 forward-backward consistency: a forward flow,
                          built by softmax-splatting the backward MVs (collisions won by
                          lower-residual matches), must agree with the backward flow.
    `mode` selects which signals run:
      'full'     -> (a)+(b)+(c)  (the softmax-splat fwd-bwd is the single most expensive op, ~29ms
                    numpy / the dominant GPU recon op).
      'reactive' -> (a)+(b) only (drops the fwd-bwd splat -- Step-6 ablation; ~0 quality loss on
                    low-motion talking-head, loses more on high-motion).
      'adaptive' -> (a)+(b) always, and (c) ONLY when the reactive-fallback fraction exceeds
                    `adaptive_tau` (Step-7): a cheap per-direction switch that pays for the
                    fwd-bwd splat only on motion-stressed frames where it earns its cost,
                    mirroring adaptive anchoring. Returns (mask, used_fwdbwd)."""
    h, w = fb_x.shape
    # (b) reactive residual
    pred = warp_lr(lr_prev, fb_x, fb_y).astype(np.float32)
    react = np.abs(lr_cur.astype(np.float32) - pred).mean(axis=2)
    base = (~np.isfinite(fb_x)) | (react > tau_react)
    if mode == "adaptive":
        tau = ADAPTIVE_TAU if adaptive_tau is None else adaptive_tau
        use_fwdbwd = float(base.mean()) > tau
    else:
        use_fwdbwd = (mode == "full")
    ruder = False
    if use_fwdbwd:
        # (c) forward flow via softmax splat (collisions won by lower-residual sources)
        fwd = np.stack([-fb_x, -fb_y], axis=-1)
        ff, _ = softmax_splat(fwd, fb_x, fb_y, -react)
        ffx = np.nan_to_num(ff[:, :, 0], nan=1e6).astype(np.float32)
        ffy = np.nan_to_num(ff[:, :, 1], nan=1e6).astype(np.float32)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        sx, sy = gx + np.nan_to_num(fb_x), gy + np.nan_to_num(fb_y)
        wf_x = cv2.remap(ffx, sx, sy, cv2.INTER_LINEAR, borderValue=1e6)  # w~ (fwd flow @ mapped)
        wf_y = cv2.remap(ffy, sx, sy, cv2.INTER_LINEAR, borderValue=1e6)
        wb_x, wb_y = np.nan_to_num(fb_x), np.nan_to_num(fb_y)             # w^ (backward flow)
        # Ruder 2016: disocclusion where |w~ + w^|^2 > 0.01(|w~|^2 + |w^|^2) + 0.5
        lhs = (wf_x + wb_x) ** 2 + (wf_y + wb_y) ** 2
        rhs = 0.01 * (wf_x ** 2 + wf_y ** 2 + wb_x ** 2 + wb_y ** 2) + 0.5
        ruder = lhs > rhs
    occ = base | ruder
    return occ.astype(bool), use_fwdbwd


# --------------------------------------------------------------------------- #
# Reference resolution + single-direction warp
# --------------------------------------------------------------------------- #
def scan_source_magnitude(frames):
    """Largest |source| over all MV records, and how many exceed 1. |source|==1 means the
    codec MV references the nearest past/future reference picture, so display-order-neighbor
    resolution is exact; |source|>1 would mean multi-ref (needs real DPB/POC parsing)."""
    mx, nbad = 0, 0
    for _, _, mvs in frames:
        if mvs is None or len(mvs) == 0:
            continue
        a = np.abs(mvs["source"].astype(int))
        if len(a):
            mx = max(mx, int(a.max()))
            nbad += int((a > 1).sum())
    return mx, nbad


def _warp_one(ref_recon, ref_oracle, lr_cur, lr_ref, mvs, want, scale,
              use_residual, occ_mode, w_hd, h_hd):
    """Warp ONE reference (its recon, and optional oracle) by the requested MV direction,
    add the NEMO residual (LR_cur - motion_comp(LR_ref), bilinear-upscaled), and compute
    this direction's unreliability mask. Returns (recon_dir, oracle_dir|None, occ_bool_hd)
    where occ=True flags pixels to distrust: intra hole in this direction OR, in 'full'
    mode, reactive/fwd-bwd occlusion. The residual is per-direction (NEMO-style)."""
    h_lr, w_lr = lr_cur.shape[:2]
    with PROF.time("build_flow"):
        fx, fy = build_lr_flow(mvs, h_lr, w_lr, want=want)
    res_hd = None
    if use_residual:
        with PROF.time("residual"):
            pred_lr = warp_lr(lr_ref, fx, fy)
            res = lr_cur.astype(np.float32) - pred_lr.astype(np.float32)
            res_hd = cv2.resize(res, (w_hd, h_hd), interpolation=cv2.INTER_LINEAR)
    with PROF.time("warp"):
        warped, hole = warp_hd(ref_recon, fx, fy, scale)
    if occ_mode in ("full", "reactive", "adaptive"):
        with PROF.time("mask"):
            occ_lr, used_fb = occlusion_mask_lr(fx, fy, lr_cur, lr_ref, mode=occ_mode)
            occ = cv2.resize(occ_lr.astype(np.uint8), (w_hd, h_hd),
                             interpolation=cv2.INTER_NEAREST).astype(bool) | hole
        MASK_FIRES[0] += int(used_fb)
        MASK_FIRES[1] += 1
    else:
        occ = hole  # naive: intra blocks only
    recon_dir = _add_res(warped, res_hd)
    oracle_dir = None
    if ref_oracle is not None:
        o_warp, _ = warp_hd(ref_oracle, fx, fy, scale)
        oracle_dir = _add_res(o_warp, res_hd)
    return recon_dir, oracle_dir, occ


# --------------------------------------------------------------------------- #
# Experiment: reference-backbone reconstruction (I/P chain) + bidirectional B leaves
# --------------------------------------------------------------------------- #
def build_perframe_cache(frames, w_hd, h_hd, sr_mode, half=False):
    """Per-frame upscale (the SR placeholder) computed ONCE for every frame and cached.
    bicubic (default; byte-identical to all prior runs) OR a real lightweight SR network
    (realesr-general-x4v3, x4). The SAME perframe image is reused for (1) the anchor
    reconstruction, (2) the disocclusion/fallback source, and (3) the per-frame-SR baseline
    -- so with --sr realesrgan, `perframe` IS per-frame SR. Caching it once is what makes the
    anchor-placement SWEEP cheap: only warp/blend re-runs across operating points, SR runs 0x.

    half=True runs the SR net in fp16 on a GPU (experiment E4: ~1.24x faster on the x4plus anchor,
    visually identical). Default fp16=OFF keeps the byte-identical fp32 path."""
    N = len(frames)
    cache = {}
    if sr_mode in ("realesrgan", "realesrgan-x4plus"):
        import sr as _srmod
        _srmod.load_model(sr_mode, half=half)
        # warm up MPS graph compilation on the real frame size, then reset latency stats so
        # the reported per-frame latency is steady-state (not the one-off compile cost).
        _srmod.upscale(frames[0][1], model=sr_mode, half=half)
        _srmod.reset_latency(sr_mode)
        for i in range(N):
            PROF.ftype, PROF.fidx = frames[i][0], i
            with PROF.time("sr"):
                cache[i] = _srmod.upscale_to(frames[i][1], w_hd, h_hd, model=sr_mode, half=half)
        return cache
    for i in range(N):
        PROF.ftype, PROF.fidx = frames[i][0], i
        with PROF.time("sr"):
            cache[i] = cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
    return cache


def backbone_indices(frames):
    """Display-order indices of the I/P reference backbone (B-frames are leaves, never refs)."""
    return [i for i, (pt, _, _) in enumerate(frames) if pt in ("I", "P")]


def _apply_region_gate_np(R, N, region_gate, scale):
    """Stream-1 OUTPUT-ONLY region-aware blend (numpy): out = a_hd*recon_heavy + (1-a_hd)*compact.
    Applied to R[i]['recon'] AFTER both reconstruction passes are complete -- the propagation
    chain has already consumed the un-blended heavy recon as its reference, so this NEVER feeds
    back into R[]'s reference role (exactly like the grain final pass). a_hd is region_quality's
    temporally-stable, widely-feathered motion gate; compact[i] is the per-frame COMPACT SR (the
    same fallback-source family). Reuses region_quality.blend_region_aware -- no math duplicated."""
    import region_quality as _rq
    a_lr, compact = region_gate["a_lr"], region_gate["compact"]
    for i in range(N):
        R[i]["recon"] = _rq.blend_region_aware(R[i]["recon"], compact[i], a_lr, scale)


def reconstruct(frames, hd_frames, scale, use_residual, occ_mode, perframe_cache, anchor_set,
                backend="numpy", collect_metrics=True, download_output=True, region_gate=None):
    """Pure backbone+B reconstruction for an arbitrary ANCHOR SET (Step 4 re-anchoring).

    `backend`: 'numpy' (default, the regression guard) or 'torch' (the MPS fast path -- the
    warp/mask/blend run on-device and the recon chain stays resident on GPU; see
    reconstruct_torch). The numpy path below is left untouched so it stays byte-identical.

    `anchor_set` = set of backbone display-indices to PROMOTE to anchors. An anchor uses its
    cached per-frame-SR image directly (fresh, drift=0); a non-anchor P warps the PREVIOUS
    backbone frame's reconstruction (drift accumulates). Forced anchors (always, regardless of
    anchor_set): every I-frame, and any P whose reference is outside the window. anchor_set=set()
    => I-frames only => byte-identical to the Step-3 backbone.

      * PASS 1 (backbone): anchor -> perframe-SR; non-anchor P -> warp prev I/P recon by
        source<0 MVs (+residual, +occlusion fallback). Stored in R[]; later frames reference it.
      * PASS 2 (B leaves): each B warps the nearest PAST I/P recon (source<0) AND nearest FUTURE
        I/P recon (source>0), blends per pixel (temporal-distance weight); one-sided / per-frame
        bicubic fallback elsewhere. B-frames are NEVER used as a reference.

    NO SR is run here (uses perframe_cache) -> the anchor sweep is warp/blend only. Returns
    (rows, R). References resolved by display-order neighbor (exact because |source|==1)."""
    MASK_FIRES[0] = MASK_FIRES[1] = 0
    if backend == "torch":
        return reconstruct_torch(frames, hd_frames, scale, use_residual, occ_mode,
                                 perframe_cache, anchor_set, collect_metrics=collect_metrics,
                                 download_output=download_output, region_gate=region_gate)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * scale, h_lr * scale
    has_true = hd_frames is not None
    N = len(frames)
    anchor_set = set(anchor_set or ())
    backbone_idx = backbone_indices(frames)

    def prev_ip(i):
        return max([b for b in backbone_idx if b < i], default=None)

    def next_ip(i):
        return min([b for b in backbone_idx if b > i], default=None)

    R = {}  # i -> dict(recon, oracle, perframe, mask, hole_frac, dist, type, is_anchor)

    # ----------------- PASS 1: I/P reference backbone (forward chain) -----------------
    for i in backbone_idx:
        pt, lr, mvs = frames[i]
        PROF.ftype, PROF.fidx = pt, i
        perframe = perframe_cache[i]
        p = prev_ip(i)
        is_anchor = (pt == "I") or (p is None) or (i in anchor_set)
        if is_anchor:
            # anchor: per-frame SR (real) / perfect HD (oracle). NO propagation, drift reset.
            R[i] = dict(recon=perframe.copy(),
                        oracle=(hd_frames[i].copy() if has_true else None),
                        perframe=perframe, mask=None, hole_frac=0.0, dist=0, type=pt,
                        is_anchor=True)  # full-frame SR run here (cost = 1 SR call)
            continue
        # non-anchor P: warp the PREVIOUS I/P reconstruction by source<0 MVs
        recon, oracle, occ = _warp_one(
            R[p]["recon"], R[p]["oracle"], lr, frames[p][1], mvs, "past",
            scale, use_residual, occ_mode, w_hd, h_hd)
        with PROF.time("blend"):
            recon[occ] = perframe[occ]                   # fallback: re-run SR
            if has_true:
                oracle[occ] = hd_frames[i][occ]          # fallback: re-run SR (oracle=perfect)
        R[i] = dict(recon=recon, oracle=oracle, perframe=perframe, mask=occ,
                    hole_frac=float(occ.mean()), dist=R[p]["dist"] + 1, type=pt, is_anchor=False)

    # ----------------- PASS 2: B-frame bidirectional leaves -----------------
    zero = np.zeros((h_hd, w_hd), bool)
    for i in range(N):
        if frames[i][0] != "B":
            continue
        pt, lr, mvs = frames[i]
        PROF.ftype, PROF.fidx = "B", i
        perframe = perframe_cache[i]
        p, f = prev_ip(i), next_ip(i)
        recon_f32 = perframe.astype(np.float32)                       # bicubic where neither valid
        oracle_f32 = hd_frames[i].astype(np.float32) if has_true else None
        wp = wf = owp = owf = None
        valid_p = valid_f = zero
        if p is not None:                                             # past direction (source<0)
            wp, owp, occ_p = _warp_one(R[p]["recon"], R[p]["oracle"], lr, frames[p][1],
                                       mvs, "past", scale, use_residual, occ_mode, w_hd, h_hd)
            valid_p = ~occ_p
        if f is not None:                                            # future direction (source>0)
            wf, owf, occ_f = _warp_one(R[f]["recon"], R[f]["oracle"], lr, frames[f][1],
                                       mvs, "future", scale, use_residual, occ_mode, w_hd, h_hd)
            valid_f = ~occ_f
        # blend weight: temporal-distance (the closer reference is more reliable; reduces to
        # 0.5/0.5 when symmetric). One-sided when only one anchor exists / is in-window.
        if p is not None and f is not None:
            dp, df = (i - p), (f - i)
            a_p, a_f = df / (dp + df), dp / (dp + df)
        elif p is not None:
            a_p, a_f = 1.0, 0.0
        else:
            a_p, a_f = 0.0, 1.0
        both = valid_p & valid_f
        only_p = valid_p & ~valid_f
        only_f = valid_f & ~valid_p
        none = ~valid_p & ~valid_f                                    # true fallback region
        with PROF.time("blend"):
            if wp is not None:
                recon_f32[only_p] = wp.astype(np.float32)[only_p]
            if wf is not None:
                recon_f32[only_f] = wf.astype(np.float32)[only_f]
            if wp is not None and wf is not None:
                recon_f32[both] = (a_p * wp.astype(np.float32) + a_f * wf.astype(np.float32))[both]
            recon = np.clip(recon_f32, 0, 255).astype(np.uint8)
        oracle = None
        if has_true:
            if owp is not None:
                oracle_f32[only_p] = owp.astype(np.float32)[only_p]
            if owf is not None:
                oracle_f32[only_f] = owf.astype(np.float32)[only_f]
            if owp is not None and owf is not None:
                oracle_f32[both] = (a_p * owp.astype(np.float32)
                                    + a_f * owf.astype(np.float32))[both]
            oracle = np.clip(oracle_f32, 0, 255).astype(np.uint8)
        R[i] = dict(recon=recon, oracle=oracle, perframe=perframe,
                    mask=none, hole_frac=float(none.mean()), dist=i, type=pt, is_anchor=False)

    # ----------------- Stream-1 region-aware OUTPUT-ONLY final pass (never into R[] refs) -----
    if region_gate is not None:
        _apply_region_gate_np(R, N, region_gate, scale)

    # ----------------- assemble rows (display order) -----------------
    # collect_metrics=False (timing-only) skips the expensive per-frame SSIM/PSNR/LR-consistency
    # so a clean propagation-path wall-clock can be measured without metric overhead.
    rows = _assemble_rows(frames, hd_frames, R, N, has_true) if collect_metrics else []
    return rows, R


def _assemble_rows(frames, hd_frames, R, N, has_true):
    """Build the per-frame metrics rows from a reconstruction R (recon/oracle/perframe are
    numpy uint8). Shared by the numpy and torch backends so both report identical metrics."""
    rows = []
    for i in range(N):
        d = R[i]
        recon, oracle, perframe = d["recon"], d["oracle"], d["perframe"]
        lr_true = frames[i][1]
        row = {
            "frame": i, "type": d["type"], "dist_from_anchor": d["dist"],
            "is_anchor": int(d["is_anchor"]),      # 1 => full-frame SR run (cost accounting)
            "hole_frac": round(d["hole_frac"], 5),
            "psnr_lr_consistency": round(psnr_lr_consistency(recon, lr_true), 3),
            "psnr_lr_consistency_perframe": round(psnr_lr_consistency(perframe, lr_true), 3),
            "psnr_prop_vs_perframe": round(psnr(recon, perframe), 3),  # prop vs per-frame-SR
            "ssim_prop_vs_perframe": round(ssim(recon, perframe), 4),
        }
        if has_true:
            hd = hd_frames[i]
            row["psnr_perframe_vs_true"] = round(psnr(perframe, hd), 3)  # per-frame SR baseline
            row["psnr_prop_vs_true"] = round(psnr(recon, hd), 3)         # bicubic-anchor propagation
            row["psnr_oracle_vs_true"] = round(psnr(oracle, hd), 3)      # perfect-anchor propagation
            row["ssim_oracle_vs_true"] = round(ssim(oracle, hd), 4)
        rows.append(row)
    return rows


def reconstruct_torch(frames, hd_frames, scale, use_residual, occ_mode, perframe_cache, anchor_set,
                      collect_metrics=True, download_output=True, region_gate=None):
    """Torch/MPS fast-path twin of reconstruct(): identical structure & semantics, but the
    warp/mask/blend run on-device and the I/P recon chain stays RESIDENT on the GPU across
    frames (no per-frame host<->device round-trips for the HD references -- a warped recon is
    fed straight into the next frame's warp). Remaining transfers: per frame, upload the LR
    frame(s) + the per-frame-SR fallback image; once at the end, download the HD recon/oracle/
    perframe for the numpy metrics. Matches numpy within GPU-float tolerance, not byte-identical
    (see gpu_ops convention notes)."""
    import torch
    import gpu_ops as G
    MASK_FIRES[0] = MASK_FIRES[1] = 0
    if PROF.enabled:
        PROF.sync = G.sync           # honest GPU timing (kernels are async)
    dev = G.device()
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * scale, h_lr * scale
    has_true = hd_frames is not None
    N = len(frames)
    anchor_set = set(anchor_set or ())
    backbone_idx = backbone_indices(frames)

    def prev_ip(i):
        return max([b for b in backbone_idx if b < i], default=None)

    def next_ip(i):
        return min([b for b in backbone_idx if b > i], default=None)

    lr_dev = {i: G.img_to_dev(frames[i][1]) for i in range(N)}   # small LR uploads (once)
    R = {}

    def warp_one_t(ref_recon, ref_oracle, i, ref_i, want):
        """On-device twin of _warp_one: warp ONE reference by `want` MVs, +residual, +mask."""
        _, _, mvs = frames[i]
        with PROF.time("build_flow"):
            fx_np, fy_np = build_lr_flow(mvs, h_lr, w_lr, want=want)
            fx, fy = G.flow_to_dev(fx_np, fy_np)
        res_hd = None
        if use_residual:
            with PROF.time("residual"):
                res_hd = G.residual_hd(lr_dev[i], lr_dev[ref_i], fx, fy, scale)
        with PROF.time("warp"):
            warped, hole = G.warp_hd(ref_recon, fx, fy, scale)
        if occ_mode in ("full", "reactive", "adaptive"):
            with PROF.time("mask"):
                occ_lr, used_fb = G.occlusion_mask_lr(fx, fy, lr_dev[i], lr_dev[ref_i],
                                                       mode=occ_mode)
                occ = G.upsample_bool(occ_lr, scale) | hole
            MASK_FIRES[0] += int(used_fb)
            MASK_FIRES[1] += 1
        else:
            occ = hole
        recon_dir = G.add_res(warped, res_hd)
        oracle_dir = None
        if ref_oracle is not None:
            o_warp, _ = G.warp_hd(ref_oracle, fx, fy, scale)
            oracle_dir = G.add_res(o_warp, res_hd)
        return recon_dir, oracle_dir, occ

    # ----------------- PASS 1: I/P reference backbone (on GPU, resident chain) -----------------
    for i in backbone_idx:
        pt, lr, mvs = frames[i]
        PROF.ftype, PROF.fidx = pt, i
        p = prev_ip(i)
        is_anchor = (pt == "I") or (p is None) or (i in anchor_set)
        with PROF.time("upload_perframe"):
            pf = G.img_to_dev(perframe_cache[i])
        if is_anchor:
            R[i] = dict(recon=pf.clone(),
                        oracle=(G.img_to_dev(hd_frames[i]) if has_true else None),
                        perframe=pf, mask=None, hole_frac=0.0, dist=0, type=pt, is_anchor=True)
            continue
        recon, oracle, occ = warp_one_t(R[p]["recon"], R[p]["oracle"], i, p, "past")
        with PROF.time("blend"):
            occ3 = occ[None, None]
            recon = torch.where(occ3, pf, recon)
            if has_true:
                oracle = torch.where(occ3, G.img_to_dev(hd_frames[i]), oracle)
            hf = occ.float().mean()          # GPU scalar -- NOT .item()'d here (see note below)
        R[i] = dict(recon=recon, oracle=oracle, perframe=pf, mask=occ,
                    hole_frac=hf, dist=R[p]["dist"] + 1, type=pt, is_anchor=False)

    # ----------------- PASS 2: B-frame bidirectional leaves (on GPU) -----------------
    for i in range(N):
        if frames[i][0] != "B":
            continue
        pt, lr, mvs = frames[i]
        PROF.ftype, PROF.fidx = "B", i
        p, f = prev_ip(i), next_ip(i)
        with PROF.time("upload_perframe"):
            pf = G.img_to_dev(perframe_cache[i])
        recon = pf.clone()                                            # bicubic where neither valid
        oracle = G.img_to_dev(hd_frames[i]) if has_true else None
        wp = wf = owp = owf = None
        valid_p = valid_f = None
        if p is not None:
            wp, owp, occ_p = warp_one_t(R[p]["recon"], R[p]["oracle"], i, p, "past")
            valid_p = ~occ_p
        if f is not None:
            wf, owf, occ_f = warp_one_t(R[f]["recon"], R[f]["oracle"], i, f, "future")
            valid_f = ~occ_f
        if p is not None and f is not None:
            dp, df = (i - p), (f - i)
            a_p, a_f = df / (dp + df), dp / (dp + df)
        elif p is not None:
            a_p, a_f = 1.0, 0.0
        else:
            a_p, a_f = 0.0, 1.0
        zero = torch.zeros((h_hd, w_hd), dtype=torch.bool, device=dev)
        vp = valid_p if valid_p is not None else zero
        vf = valid_f if valid_f is not None else zero
        both = vp & vf
        only_p = vp & ~vf
        only_f = vf & ~vp
        none = ~vp & ~vf
        with PROF.time("blend"):
            if wp is not None:
                recon = torch.where(only_p[None, None], wp, recon)
            if wf is not None:
                recon = torch.where(only_f[None, None], wf, recon)
            if wp is not None and wf is not None:
                recon = torch.where(both[None, None], a_p * wp + a_f * wf, recon)
            recon = recon.clamp(0, 255)
            hf = none.float().mean()         # GPU scalar -- batched .item() after both passes
        if has_true:
            if owp is not None:
                oracle = torch.where(only_p[None, None], owp, oracle)
            if owf is not None:
                oracle = torch.where(only_f[None, None], owf, oracle)
            if owp is not None and owf is not None:
                oracle = torch.where(both[None, None], a_p * owp + a_f * owf, oracle)
            oracle = oracle.clamp(0, 255)
        R[i] = dict(recon=recon, oracle=oracle, perframe=pf, mask=none,
                    hole_frac=hf, dist=i, type=pt, is_anchor=False)

    # ----------------- batched hole_frac materialization (ONE device->host sync) -----------------
    # Each non-anchor frame's hole_frac was kept as a GPU SCALAR (occ.float().mean()) instead of
    # being .item()'d in its blend step: a per-frame .item() drains the MPS queue, serializing
    # consecutive frames' kernels (~12 ms/frame measured). The value is identical -- same op, same
    # order -- so this is a pure latency fix (recon stays numerically unchanged for ALL backends/
    # modes; the numpy regression path is untouched). One stacked .item() materializes them all.
    _ht = {i: R[i]["hole_frac"] for i in R if torch.is_tensor(R[i]["hole_frac"])}
    if _ht:
        _keys = list(_ht)
        _vals = torch.stack([_ht[i] for i in _keys]).tolist()    # single sync for every frame
        for i, v in zip(_keys, _vals):
            R[i]["hole_frac"] = float(v)

    # ----------------- Stream-1 region-aware gate (HD weight, built ONCE: temporally stable) ---
    # The gate a_lr is the SAME map for every frame (window_static_weight), so upsample it to HD
    # once and reuse it. Bilinear upsample == region_quality.blend_region_aware's cv2.INTER_LINEAR
    # (matches within GPU-float tolerance, the standing gpu_ops convention).
    gate_hd = None
    if region_gate is not None:
        import torch.nn.functional as _F
        g = torch.from_numpy(np.ascontiguousarray(region_gate["a_lr"])).to(dev)
        gate_hd = _F.interpolate(g[None, None], scale_factor=scale, mode="bilinear",
                                 align_corners=False)            # [1,1,Hhd,Whd], a=1->heavy, 0->compact

    # ----------------- download -> numpy, then assemble rows (numpy metrics) -----------------
    # Step-7: in the DEPLOYABLE path the reconstructed HD frame stays RESIDENT on the GPU (it is
    # handed to a Metal/texture display, not read back), so `download_output=False` skips the HD
    # download entirely -- that ~10 ms/frame round-trip is a removable deployment artifact, not a
    # recon cost. `download_output=True` (the "with-I/O" honesty number, and the default for the
    # metrics path) reads the HD recon back to CPU. The perframe/oracle/mask downloads exist
    # solely for this experiment's numpy metrics and are gated on collect_metrics.
    for i in range(N):
        d = R[i]
        PROF.ftype, PROF.fidx = d["type"], i
        # Stream-1 OUTPUT-ONLY region-aware blend (single torch.lerp, ~1-2 ms): applied AFTER the
        # whole resident chain is built, into a FRESH tensor -> the heavy recon stored in R[] (its
        # reference role already over) is never mutated. a=1 keeps heavy detail (static), a=0 the
        # temporally-stable compact (dynamic).  out = lerp(compact, heavy, a) = compact + a*(heavy-compact).
        out_t = d["recon"]
        if gate_hd is not None:
            with PROF.time("region_blend"):
                cpt = G.img_to_dev(region_gate["compact"][i])
                out_t = torch.lerp(cpt, d["recon"], gate_hd)
        if collect_metrics or download_output:
            with PROF.time("download"):
                d["recon"] = G.img_to_host(out_t)
        elif gate_hd is not None:
            d["recon"] = out_t                  # deployable: region-aware frame stays GPU-resident
        if not collect_metrics:
            continue
        d["perframe"] = G.img_to_host(d["perframe"])
        if d["oracle"] is not None:
            d["oracle"] = G.img_to_host(d["oracle"])
        if d["mask"] is not None:
            d["mask"] = d["mask"].to("cpu").numpy().astype(bool)
    rows = _assemble_rows(frames, hd_frames, R, N, has_true) if collect_metrics else []
    return rows, R


# --------------------------------------------------------------------------- #
# Re-anchoring policies (Step 4): map a policy -> the set of promoted P anchors
# --------------------------------------------------------------------------- #
def _adaptive_fallback(frames, R0, budget):
    """NEMO-style greedy anchor placement driven by ACCUMULATED FALLBACK FRACTION.
    Walk the I/P backbone; accumulate each non-anchor frame's fallback (unreliable-pixel)
    fraction since the last anchor; when the running sum reaches `budget` frame-equivalents,
    promote the current P to an anchor and reset. Forced reset at every I-frame / window start.

    Why fallback (not a dB margin on PSNR(prop,per-frame-SR)): the fallback fraction is
    bounded [0,1] and is the SAME physical quantity (re-SR'd pixels) across all content, so a
    fixed budget is content-FAIR. PSNR(prop,per-frame-SR) instead ranges ~35-65 dB across
    talking-head vs high-motion regions, so no fixed dB threshold/margin is comparable (it
    over-anchors static content, under-anchors motion). hole_frac is anchor-invariant (it
    depends only on frame i's MVs vs its reference), so the none-pass values are exact."""
    backbone = backbone_indices(frames)
    first = backbone[0] if backbone else None
    anchors, cum = set(), 0.0
    for i in backbone:
        pt = frames[i][0]
        if pt == "I" or i == first:        # forced anchor -> reset erosion budget
            cum = 0.0
            continue
        cum += R0[i]["hole_frac"]
        if cum >= budget:
            anchors.add(i)
            cum = 0.0
    return anchors


def _adaptive_psnr(frames, R0, psnr_floor):
    """Alternative (secondary) adaptive driver: promote a P to an anchor where the none-pass
    PSNR(prop, per-frame-SR) falls below an ABSOLUTE floor (dB). Kept for comparison; less
    robust than fallback-budget because the floor is content-dependent (a 38 dB floor that is
    'drifted' for high-motion is unreachably strict for a near-static talking head)."""
    backbone = backbone_indices(frames)
    first = backbone[0] if backbone else None
    anchors = set()
    for i in backbone:
        pt = frames[i][0]
        if pt == "I" or i == first:
            continue
        if R0[i]["psnr_prop_vs_perframe"] < psnr_floor:
            anchors.add(i)
    return anchors


def compute_anchor_set(frames, reanchor, quality_margin, fallback_budget, adapt_metric,
                       hd_frames, scale, use_residual, occ_mode, perframe_cache, backend="numpy"):
    """Resolve a --reanchor policy into the set of promoted backbone anchors.
      none        -> {} (I-frames only; the Step-3 default).
      interval:K  -> every K-th backbone frame (position in the I/P list).
      adaptive    -> greedy, driven by accumulated fallback (default) or PSNR floor.
    `quality_margin` is reserved for the dB-margin variant; the default adaptive driver is the
    content-fair accumulated-fallback budget (see _adaptive_fallback)."""
    backbone = backbone_indices(frames)
    if reanchor == "none":
        return set()
    if reanchor.startswith("interval:"):
        tok = reanchor.split(":", 1)[1]
        K = 10 ** 9 if tok in ("inf", "infinity", "0") else int(tok)
        return set(backbone[j] for j in range(0, len(backbone), max(1, K)))
    if reanchor == "adaptive":
        # one cheap 'none' reconstruction pass (SR already cached) to read the drift signal
        rows0, R0 = reconstruct(frames, hd_frames, scale, use_residual, occ_mode,
                                perframe_cache, set(), backend=backend)
        # attach the prop-vs-perframe psnr onto R0 for the psnr variant
        for r in rows0:
            R0[r["frame"]]["psnr_prop_vs_perframe"] = r["psnr_prop_vs_perframe"]
        if adapt_metric == "psnr":
            return _adaptive_psnr(frames, R0, quality_margin)
        return _adaptive_fallback(frames, R0, fallback_budget)
    raise ValueError(f"unknown --reanchor {reanchor!r}")


def _tof_from_R(frames, hd_frames, R, w_lr, h_lr):
    """Build the tOF metric(s) from a reconstruction R (at LR for speed)."""
    has_true = hd_frames is not None
    N = len(frames)
    sm = (w_lr, h_lr)
    if has_true:
        seq_oracle = [cv2.resize(R[i]["oracle"], sm) for i in range(N)]
        seq_perframe = [cv2.resize(R[i]["perframe"], sm) for i in range(N)]
        ref = [cv2.resize(f, sm) for f in hd_frames]
        return {"mode": "synthetic",
                "oracle_vs_true": tof(seq_oracle, ref),
                "perframe_vs_true": tof(seq_perframe, ref)}
    seq_prop = [cv2.resize(R[i]["recon"], sm) for i in range(N)]
    seq_pf = [cv2.resize(R[i]["perframe"], sm) for i in range(N)]
    seq_lr = [frames[i][1] if (w_lr, h_lr) == frames[i][1].shape[1::-1]
              else cv2.resize(frames[i][1], sm) for i in range(N)]
    return {"mode": "real",
            "prop_vs_perframe": tof(seq_prop, seq_pf),
            "prop_vs_lr": tof(seq_prop, seq_lr),
            "perframe_vs_lr": tof(seq_pf, seq_lr)}


def _build_region_gate(frames, w_hd, h_hd, scale, lo=0.2, hi=1.0, feather=61,
                       compact_model="realesrgan", compact_cache=None,
                       texture_aware=False, tex_lo=6.0, tex_hi=14.0, tex_feather=21, tex_k=7):
    """Build the Stream-1 region-aware gate + compact source for the OUTPUT-only blend.
    REUSES region_quality (no math duplicated):
      * region_masks() -> the (free) per-frame MV-magnitude temporal-mean map `meanmag`,
      * window_static_weight(meanmag, lo, hi, feather) -> the TEMPORALLY-STABLE, widely-feathered
        motion gate a_lr in [0,1] (1=static->heavy detail; 0=dynamic->stable compact). hi=1.0 LR
        px/frame is the physical 'can't carry heavy HF stably under warp' threshold.
    `compact` = the per-frame COMPACT SR (the same fallback-source family used for occlusion).
    Pass compact_cache to reuse a precomputed cache (the heavy chain stays single-model).

    R7-E2 (`texture_aware`, DEFAULT OFF -> a_lr byte-identical): the MOTION gate alone spends the
    heavy x4plus on STATIC-but-SMOOTH regions (still face / sky) where R6-E1 proved with TRUE LPIPS
    that x4plus's extra HF is MISALIGNED -> ~0 gain (sometimes worse). With texture_aware, a_lr is
    multiplied by a CHEAP, GT-FREE local-detail weight a_tex in [0,1] -- the temporal-mean local luma
    STD of the ALREADY-COMPUTED compact source (thresholded tex_lo..tex_hi, feathered) -- so
    a' = a_motion * a_tex: heavy ONLY where BOTH static AND textured. Measured (R7-E2 degrade-restore
    LPIPS): talking-head effective-heavy ~74%->~12% at LPIPS -0.02 (BETTER); detailed graphics
    ~66%->~28% at LPIPS +-0.001 (neutral). NB: as an OUTPUT-only blend this is quality-neutral-to-
    better but saves NO compute by itself -- the compute win needs the heavy anchor SR to actually
    SKIP tiles where a'~0 (a tiled x4plus pass); a_tex is the mask that enables it. Default OFF
    pending an integrated propagation+tOF A/B."""
    import region_quality as _rq
    h_lr, w_lr = frames[0][1].shape[:2]
    if compact_cache is None:
        compact_cache = build_perframe_cache(frames, w_hd, h_hd, compact_model)
    _, _, meanmag, _ = _rq.region_masks(frames, h_lr, w_lr, 45.0, 80.0)
    a_lr = _rq.window_static_weight(meanmag, lo, hi, feather=feather)
    if texture_aware:
        # temporal-mean local-std of the compact source (FREE: the cache is already built). High on
        # text/edges/texture, low on smooth skin/sky -> a_tex; heavy ONLY where static AND textured.
        acc = np.zeros((h_lr, w_lr), np.float32)
        for i in range(len(frames)):
            y = cv2.cvtColor(compact_cache[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
            if y.shape[:2] != (h_lr, w_lr):
                y = cv2.resize(y, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
            mu = cv2.boxFilter(y, -1, (tex_k, tex_k))
            acc += np.sqrt(np.maximum(cv2.boxFilter(y * y, -1, (tex_k, tex_k)) - mu * mu, 0.0))
        std_map = acc / max(len(frames), 1)
        a_tex = np.clip((std_map - tex_lo) / max(tex_hi - tex_lo, 1e-6), 0.0, 1.0).astype(np.float32)
        if tex_feather and tex_feather >= 3:
            kk = int(tex_feather) | 1
            a_tex = cv2.GaussianBlur(a_tex, (kk, kk), 0)
        a_lr = (a_lr * a_tex).astype(np.float32)
    return dict(a_lr=a_lr, compact=compact_cache)


def run(frames, hd_frames, scale, use_residual, out_dir, occ_mode="full", sr_mode="bicubic",
        reanchor="none", quality_margin=1.0, fallback_budget=1.0, adapt_metric="fallback",
        perframe_cache=None, anchor_set=None, backend="numpy", grain="off", region_aware=False):
    """Single-run entry point: build the per-frame-SR cache, resolve the re-anchoring policy
    into an anchor set, reconstruct, then write CSV/samples/plots and compute tOF. Defaults
    (reanchor='none', anchor_set=None, backend='numpy') reproduce the Step-3 backbone exactly."""
    os.makedirs(out_dir, exist_ok=True)
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * scale, h_lr * scale
    has_true = hd_frames is not None
    N = len(frames)

    smax, nbad = scan_source_magnitude(frames)
    if smax > 1:
        print(f"  WARNING: |source|={smax} on {nbad} MV record(s) -> reference is NOT the "
              f"display-order neighbor (multi-ref; needs real DPB/POC parsing, out of scope). "
              f"Reconstruction assumes |source|==1 and is approximate for those frames.")

    if perframe_cache is None:
        perframe_cache = build_perframe_cache(frames, w_hd, h_hd, sr_mode)
    if anchor_set is None:
        anchor_set = compute_anchor_set(frames, reanchor, quality_margin, fallback_budget,
                                        adapt_metric, hd_frames, scale, use_residual, occ_mode,
                                        perframe_cache, backend=backend)
    # Stream-1 region-aware (Step 9): OUTPUT-ONLY blend of the propagated HEAVY recon with a
    # per-frame COMPACT source via the temporally-stable motion gate. The propagation chain stays
    # single-model (heavy = perframe_cache); the gate/compact only re-paint the OUTPUT copy.
    region_gate = (_build_region_gate(frames, w_hd, h_hd, scale) if region_aware else None)
    rows, R = reconstruct(frames, hd_frames, scale, use_residual, occ_mode, perframe_cache,
                          anchor_set, backend=backend, region_gate=region_gate)

    # ----------------- samples for the visual spot check -----------------
    backbone_idx = backbone_indices(frames)

    def prev_ip(i):
        return max([b for b in backbone_idx if b < i], default=None)

    def next_ip(i):
        return min([b for b in backbone_idx if b > i], default=None)

    samples = {}
    b_sample = next((i for i in range(N) if frames[i][0] == "B"
                     and prev_ip(i) is not None and next_ip(i) is not None), None)
    sample_idx = {1, N // 2, N - 1}
    if b_sample is not None:
        sample_idx.add(b_sample)
    for i in sample_idx:
        d = R[i]
        recon, oracle, perframe = d["recon"], d["oracle"], d["perframe"]
        lr_true = frames[i][1]
        samples[i] = {"prop": recon, "perframe": perframe}
        if has_true:
            samples[i]["true"] = hd_frames[i]
            samples[i]["oracle"] = oracle
        else:
            samples[i]["bicubic"] = cv2.resize(lr_true, (w_hd, h_hd),
                                               interpolation=cv2.INTER_CUBIC)
            samples[i]["_is_anchor"] = bool(d["is_anchor"])
            if d["mask"] is not None:
                samples[i]["mask"] = np.broadcast_to(
                    (d["mask"][..., None].astype(np.uint8) * 255), recon.shape).copy()

    _write_csv(rows, out_dir)
    _dump_samples(samples, out_dir)
    # Step 8: per-frame film grain is the FINAL pass, applied to a COPY of the reconstructed
    # output ONLY (never to R[]'s recon used as a propagation reference). Re-seeded per frame
    # index => temporally independent. Emit a before/after crop for the visual spot check.
    if grain != "off":
        for i in sorted(samples):
            recon = samples[i].get("prop")
            if recon is None:
                continue
            grained = _grain.apply_grain(recon, i, grain)
            cv2.imwrite(os.path.join(out_dir, f"frame{i:03d}_grain.png"),
                        cv2.cvtColor(grained, cv2.COLOR_RGB2BGR))
            cs = 256
            h, w = recon.shape[:2]
            y0, x0 = max(0, h // 2 - cs // 2), max(0, w // 2 - cs // 2)
            ab = np.concatenate([
                _label(recon[y0:y0 + cs, x0:x0 + cs], "no grain"),
                _label(grained[y0:y0 + cs, x0:x0 + cs], f"grain={grain}")], axis=1)
            cv2.imwrite(os.path.join(out_dir, f"frame{i:03d}_grain_ab.png"),
                        cv2.cvtColor(ab, cv2.COLOR_RGB2BGR))
    _plots(rows, has_true, out_dir)
    tof_res = _tof_from_R(frames, hd_frames, R, w_lr, h_lr)
    return rows, tof_res


def _write_csv(rows, out_dir):
    keys = list(rows[0].keys())
    for r in rows:
        for k in keys:
            r.setdefault(k, "")
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _label(img, text):
    """RGB copy with a small banner so montage panels are self-identifying."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)
    return out


def _dump_samples(samples, out_dir):
    for i, imgs in samples.items():
        is_anchor = bool(imgs.pop("_is_anchor", False))  # metadata, not an image
        for name, img in imgs.items():
            cv2.imwrite(os.path.join(out_dir, f"frame{i:03d}_{name}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if "true" in imgs:
            d = cv2.absdiff(imgs["prop"], imgs["true"]).max(axis=2)
            hm = cv2.applyColorMap((np.clip(d * 3, 0, 255)).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(out_dir, f"frame{i:03d}_err.png"), hm)
        # real-footage 3-way side-by-side: bicubic | propagated-SR | per-frame-SR
        if "bicubic" in imgs:
            tag = "ANCHOR (full SR)" if is_anchor else "PROPAGATED-SR"
            panels = [_label(imgs["bicubic"], "bicubic"),
                      _label(imgs["prop"], tag),
                      _label(imgs["perframe"], "per-frame SR")]
            montage = np.concatenate(panels, axis=1)
            cv2.imwrite(os.path.join(out_dir, f"frame{i:03d}_sidebyside.png"),
                        cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
            # center crop (256x256 region of each) to make HF detail visible at 1:1
            h, w = imgs["prop"].shape[:2]
            cs = 256
            y0, x0 = max(0, h // 2 - cs // 2), max(0, w // 2 - cs // 2)
            crops = [_label(im[y0:y0 + cs, x0:x0 + cs], lab) for im, lab in (
                (imgs["bicubic"], "bicubic"), (imgs["prop"], tag),
                (imgs["perframe"], "per-frame SR"))]
            cv2.imwrite(os.path.join(out_dir, f"frame{i:03d}_crop.png"),
                        cv2.cvtColor(np.concatenate(crops, axis=1), cv2.COLOR_RGB2BGR))


def _plots(rows, has_true, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = [r["frame"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    if has_true:
        ax1.plot(fr, [r["psnr_oracle_vs_true"] for r in rows], "-^", color="green",
                 label="propagated from PERFECT anchor vs true HD")
        ax1.plot(fr, [r["psnr_perframe_vs_true"] for r in rows], "-s", color="gray",
                 label="per-frame bicubic vs true HD (baseline)")
        ax1.plot(fr, [r["psnr_prop_vs_true"] for r in rows], "-o", color="orange",
                 label="propagated from bicubic anchor vs true HD")
    else:  # real footage: no HD ground truth -> show the LR-consistency metric instead
        ax1.plot(fr, [r["psnr_lr_consistency"] for r in rows], "-o", color="teal",
                 label="LR-consistency PSNR (downscaled recon vs decoded LR)")
        bcol = {"I": "blue", "P": "green", "B": "red"}
        for t, c in bcol.items():
            xs = [r["frame"] for r in rows if r["type"] == t]
            ys = [r["psnr_lr_consistency"] for r in rows if r["type"] == t]
            ax1.scatter(xs, ys, s=22, color=c, zorder=3, label=f"{t}-frame")
    ax1.set_xlabel("frame (distance from anchor)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.grid(alpha=0.3)
    if has_true:  # frame 0 oracle==true => ~inf PSNR; focus on the meaningful range
        vals = [r["psnr_oracle_vs_true"] for r in rows if r["dist_from_anchor"] > 0]
        vals += [r["psnr_perframe_vs_true"] for r in rows]
        ax1.set_ylim(min(vals) - 2, max(vals) + 3)
    ax1.legend(loc="lower right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.bar(fr, [r["hole_frac"] * 100 for r in rows], alpha=0.15, color="red", width=0.6)
    ax2.set_ylabel("fallback / disocclusion (%)", color="red")
    plt.title("Quality & fallback vs frame (B-frames now bidirectionally reconstructed)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(fr, [r["psnr_prop_vs_perframe"] for r in rows], "-o", color="purple")
    ax.set_xlabel("frame (distance from anchor)")
    ax.set_ylabel("PSNR propagated vs per-frame (dB)")
    ax.set_title("Cache-erosion curve (NEMO metric: warp+drift loss vs per-frame SR)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "erosion.png"), dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="H.264 video; if omitted, a synthetic clip is generated")
    ap.add_argument("--scale", type=int, default=3, help="upscale factor (default 3 => 360p->1080p)")
    ap.add_argument("--frames", type=int, default=24, help="synthetic clip length")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="(real --input only) first display-order frame of the window")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="(real --input only) decode only this many frames from --start-frame")
    ap.add_argument("--no-residual", action="store_true", help="warp only, skip NEMO residual")
    ap.add_argument("--occ", choices=["naive", "full", "reactive", "adaptive"], default="full",
                    help="occlusion mask: naive=intra blocks only; full=+softmax-splat fwd-bwd+reactive; "
                         "reactive=intra+reactive only (drops the ~29ms fwd-bwd splat -- Step-6 ablation); "
                         "adaptive=reactive + fwd-bwd only on motion-stressed frames (Step-7)")
    ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy",
                    help="warp/mask/blend backend: numpy (default, the regression guard) or torch "
                         "(MPS fast path -- recon chain stays resident on GPU; see reconstruct_torch)")
    ap.add_argument("--sr", choices=["bicubic", "realesrgan", "realesrgan-x4plus"],
                    default="bicubic",
                    help="anchor/fallback upscaler: bicubic (default, byte-identical to prior runs); "
                         "realesrgan (realesr-general-x4v3, compact, x4 -- use with --scale 4); "
                         "realesrgan-x4plus (RRDBNet x23, heavy perceptual anchor, x4 -- Step 8)")
    ap.add_argument("--grain", choices=list(_GRAIN_STRENGTHS), default="off",
                    help="per-frame film-grain final pass (Step 8): off (default) | low | med | "
                         "high. Regenerated per output frame (temporally independent), luma-only, "
                         "applied AFTER reconstruction; never warped/propagated/fed to references")
    ap.add_argument("--region-aware", action="store_true",
                    help="Stream-1 region-aware detail gating (Step 9): OUTPUT-ONLY final pass that "
                         "blends the propagated HEAVY recon with a per-frame COMPACT SR by a "
                         "temporally-stable, widely-feathered motion gate (static->heavy detail, "
                         "dynamic->stable compact). Requires --sr realesrgan-x4plus (heavy chain). "
                         "Never fed into the propagation reference chain. Default OFF => "
                         "byte-identical regression.")
    ap.add_argument("--reanchor", default="none",
                    help="anchor policy: none (I-frames only; default, Step-3 behaviour) | "
                         "interval:K (anchor every K-th I/P backbone frame; K=inf => none) | "
                         "adaptive (greedy NEMO-style re-anchoring)")
    ap.add_argument("--quality-margin", type=float, default=1.0,
                    help="(adaptive --adapt-metric psnr) dB floor on PSNR(prop, per-frame-SR)")
    ap.add_argument("--fallback-budget", type=float, default=1.0,
                    help="(adaptive, default driver) re-anchor once accumulated fallback-pixel "
                         "fraction since the last anchor reaches this many frame-equivalents")
    ap.add_argument("--adapt-metric", choices=["fallback", "psnr"], default="fallback",
                    help="adaptive driver: fallback (accumulated unreliable-pixel budget; default, "
                         "content-fair) or psnr (absolute PSNR(prop,per-frame-SR) floor)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    args = ap.parse_args()
    if args.sr in ("realesrgan", "realesrgan-x4plus") and args.scale != 4:
        print(f"  NOTE: {args.sr} is an x4 model; running at --scale {args.scale} "
              f"will resize SR output to the target (use --scale 4 for native x4).")
    if args.region_aware and args.sr != "realesrgan-x4plus":
        raise SystemExit("--region-aware requires --sr realesrgan-x4plus (the heavy propagation "
                         "chain + a compact source); see prototype/README.md Step 9.")

    if args.input:
        lr_path, hd_frames = args.input, None
    else:
        lr_path = os.path.join(args.out, "synthetic_lr.mp4")
        os.makedirs(args.out, exist_ok=True)
        hd_frames = make_synthetic(lr_path, scale=args.scale, n=args.frames)

    if args.input:
        frames = decode_lr_and_mvs(lr_path, args.start_frame, args.max_frames)
    else:
        frames = decode_lr_and_mvs(lr_path)
    print(f"decoded {len(frames)} frames "
          f"(window start={args.start_frame if args.input else 0}), "
          f"types: {''.join(f[0][0] for f in frames)}")
    rows, tof_res = run(frames, hd_frames, args.scale, not args.no_residual, args.out, args.occ,
                        args.sr, reanchor=args.reanchor, quality_margin=args.quality_margin,
                        fallback_budget=args.fallback_budget, adapt_metric=args.adapt_metric,
                        backend=args.backend, grain=args.grain, region_aware=args.region_aware)
    n_anch = sum(1 for r in rows if r.get("is_anchor"))
    print(f"reanchor: {args.reanchor}  anchors: {n_anch}/{len(rows)} "
          f"(anchor-fraction {100*n_anch/len(rows):.1f}%)")

    # summary
    prop_loss = [r["psnr_prop_vs_perframe"] for r in rows if r["dist_from_anchor"] > 0]
    holes = [r["hole_frac"] for r in rows if r["dist_from_anchor"] > 0]
    print(f"\nresidual: {'OFF' if args.no_residual else 'ON'}  scale: x{args.scale}  "
          f"occ: {args.occ}  sr: {args.sr}")
    print(f"warp+drift error (propagated-vs-per-frame PSNR): first={prop_loss[0]:.2f}  "
          f"last={prop_loss[-1]:.2f}  min={min(prop_loss):.2f} dB")
    if hd_frames is not None:
        # crossover: how many frames does propagating a PERFECT anchor beat per-frame bicubic?
        win = [r for r in rows if r["dist_from_anchor"] > 0
               and r["psnr_oracle_vs_true"] >= r["psnr_perframe_vs_true"]]
        streak = 0
        for r in rows[1:]:
            if r["psnr_oracle_vs_true"] >= r["psnr_perframe_vs_true"]:
                streak += 1
            else:
                break
        last = rows[-1]
        print(f"perfect-anchor propagation vs true HD: first={rows[1]['psnr_oracle_vs_true']:.2f}  "
              f"last={last['psnr_oracle_vs_true']:.2f} dB")
        print(f"per-frame bicubic vs true HD:          first={rows[1]['psnr_perframe_vs_true']:.2f}  "
              f"last={last['psnr_perframe_vs_true']:.2f} dB")
        print(f"=> propagating a perfect anchor BEATS per-frame bicubic for the first "
              f"{streak} frame(s); wins {len(win)}/{len(rows)-1} total")
    print(f"fallback pixels (SR re-run): mean={100*np.mean(holes):.2f}%  max={100*max(holes):.2f}%")
    if tof_res is not None and tof_res["mode"] == "synthetic":
        print(f"tOF flicker (lower=steadier): perfect-anchor propagation={tof_res['oracle_vs_true']:.3f}  "
              f"per-frame SR={tof_res['perframe_vs_true']:.3f}")
    elif tof_res is not None and tof_res["mode"] == "real":
        print(f"tOF (ref=per-frame-SR): propagated-SR vs per-frame-SR = {tof_res['prop_vs_perframe']:.3f} "
              f"(temporal drift of propagation from the SR ceiling; lower=closer)")
        print(f"tOF (ref=decoded LR, cleanest motion truth; lower=steadier/tracks true motion): "
              f"propagated-SR={tof_res['prop_vs_lr']:.3f}  per-frame-SR={tof_res['perframe_vs_lr']:.3f}"
              f"  => propagation {'ADDS' if tof_res['prop_vs_lr'] < tof_res['perframe_vs_lr'] else 'does NOT add'} "
              f"temporal stability vs per-frame SR")

    # ----- COST ARGUMENT (the economic point): SR network calls amortized over the window ---
    N = len(rows)
    n_anchor = sum(1 for r in rows if r.get("is_anchor"))
    nonanchor = [r for r in rows if not r.get("is_anchor")]
    fallback_pix = sum(r["hole_frac"] for r in nonanchor)  # pixel-fraction SR re-runs
    prop_calls = n_anchor + fallback_pix                   # full-SR-frame-equivalents
    ratio = prop_calls / N if N else float("nan")
    print(f"\nCOST (SR-network-frame-equivalents over {N} frames):")
    print(f"  per-frame-SR  = {N} full-frame SR calls (100%)")
    print(f"  propagated-SR = {n_anchor} anchor(s) + {fallback_pix:.2f} frame-equiv of fallback "
          f"pixels ({100*np.mean([r['hole_frac'] for r in nonanchor]):.2f}% of {len(nonanchor)} "
          f"non-anchor frames) = {prop_calls:.2f} SR-frame-equiv")
    print(f"  => propagated-SR uses ~{100*ratio:.1f}% of the per-frame-SR compute "
          f"({1/ratio:.1f}x fewer SR-frame-equivalent invocations)")
    if args.sr in ("realesrgan", "realesrgan-x4plus"):
        try:
            import sr as _srmod
            if _srmod.n_calls(args.sr) > 0:
                med = _srmod.median_latency_ms(args.sr)
                h0, w0 = frames[0][1].shape[:2]
                print(f"SR latency on MPS ({args.sr}, {w0}x{h0}->{w0*args.scale}x"
                      f"{h0*args.scale}, x4): median={med:.1f}ms  "
                      f"mean={_srmod.mean_latency_ms(args.sr):.1f}ms over "
                      f"{_srmod.n_calls(args.sr)} calls  => amortized at 1 anchor/48 frames = "
                      f"{med/48:.2f} ms/frame")
        except Exception as e:
            print(f"  (SR latency unavailable: {e})")

    # honest per-frame-type breakdown (fallback% + LR-consistency PSNR)
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    print("per frame-type:  type    n    fallback% (mean / max)    LR-consistency dB (mean / min)")
    for t in ("I", "P", "B"):
        rs = by_type.get(t)
        if not rs:
            continue
        hf = [x["hole_frac"] for x in rs]
        lc = [x["psnr_lr_consistency"] for x in rs]
        print(f"                 {t:>3}  {len(rs):>3}      {100*np.mean(hf):6.2f} / {100*max(hf):6.2f}"
              f"            {np.mean(lc):6.2f} / {min(lc):6.2f}")
    print("  (LR-consistency = PSNR(INTER_AREA-downscale(recon) vs decoded LR); fidelity to "
          "known data, NOT HF/SR quality)")
    print(f"outputs -> {args.out}")


if __name__ == "__main__":
    main()
