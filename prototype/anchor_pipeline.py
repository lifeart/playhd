#!/usr/bin/env python3
"""anchor_pipeline.py — Stream 2: lookahead / pipelining scheduler for the HEAVY anchor.

PROBLEM (Step 7/8 finding)
--------------------------
The compact anchor SR (~0.13 s) amortizes inline: live recon holds ~38-40 ms/frame (25 fps).
The HEAVY `realesrgan-x4plus` anchor is ~2.2 s/frame — that EXCEEDS a whole GOP's playback
time (~1.9 s for 48 frames @ 25 fps). So x4plus cannot be computed inline, even amortized:
its compute must run AHEAD of the playhead, hidden behind a bounded lookahead buffer.

This module is the scheduler/buffer MODEL + a working producer/consumer demo. It answers:
  (1) the minimum LOOKAHEAD buffer so the playhead never stalls on an anchor,
  (2) the THROUGHPUT ceiling — when a single GPU (shared by anchor-SR and per-frame recon)
      saturates, i.e. the max sustainable anchor rate, and
  (3) the added STARTUP latency.

MODEL OF THE GPU (honest about Python + MPS)
--------------------------------------------
Python threads do NOT give GPU parallelism (GIL), and MPS serializes submitted work +
we `torch.mps.synchronize()` around every SR call. So on ONE Apple-Silicon GPU the anchor-SR
and the per-frame recon are a SINGLE SERIAL RESOURCE. We model the GPU as one FIFO server
that must do BOTH:
  * per-frame recon (warp+mask+blend): r seconds / displayed frame, F frames/sec,
  * anchor SR: L seconds / anchor, one anchor every K frames.
A "dedicated anchor accelerator" variant (separate GPU / a second machine / the ANE) is also
modelled (`dedicated_anchor_gpu=True`): then recon and anchor-SR run on two parallel servers.

NOTATION
--------
  L  : heavy anchor SR latency (s)            (x4plus ~2.2 s on MPS; measured 2.46 s contended)
  F  : target playback fps                    -> frame period  tau = 1/F
  r  : per-frame recon GPU cost (s)           (Step 7: full ~0.042, adaptive ~0.039, reactive ~0.028)
  K  : anchor interval (frames between anchors)
  N  : clip length (frames)

THE TWO CONDITIONS (derived; see _analysis below)
-------------------------------------------------
(A) THROUGHPUT / SATURATION  (steady state, buffer-INDEPENDENT):
        per displayed frame the GPU spends r (recon) + L/K (amortized anchor); this must
        fit the frame budget tau:
                 r + L/K  <=  tau = 1/F
        <=>  K  >=  K_min = L / (tau - r) = L*F / (1 - F*r)        [needs F*r < 1]
        Max sustainable anchor RATE:  a_max = (1 - F*r) / L   anchors/sec.
        If K < K_min the worker can NEVER keep up — the buffer drains without bound and
        playback stalls regardless of how large the lookahead is. (Single GPU.)
        DEDICATED anchor GPU: the two servers decouple -> K_min = L*F (anchor must finish
        within its own period K*tau) and recon only needs r <= tau.

(B) LOOKAHEAD BUFFER / STARTUP  (for a throughput-feasible K >= K_min):
        while one anchor SR runs (L s) the recon GPU is blocked, so the playhead drains the
        output buffer by L*F frames with nothing refilling it. Therefore the buffer must hold
                 B_min  =  ceil(L * F)  frames  ( = L seconds of pre-rendered output )
        before each anchor SR is launched. Pre-rolling that cushion (plus the first anchor's
        own SR) gives an added STARTUP latency of
                 startup ~= L * (1 + F*r)      (single GPU; first-anchor SR + cushion fill)
                 startup ~= L                  (dedicated anchor GPU)

INTEGRATION HOOK (how derisk's real pipeline adopts this — described, NOT wired here)
-------------------------------------------------------------------------------------
derisk already splits SR from warp: `build_perframe_cache()` produces the SR images and
`reconstruct(frames, ..., perframe_cache, anchor_set, backend)` is a PURE warp/blend consumer
that only READS the cache. That seam is exactly where this scheduler drops in:
  * DECODE-AHEAD producer  : `decode_lr_and_mvs(path, start, max)` already yields (ptype,img,mvs)
    in display order; run it `B_min = ceil(L*F)` frames ahead of the playhead into a ring buffer.
  * ANCHOR-SR worker       : a thread pulling anchor frames (positions from `backbone_indices()`
    + `compute_anchor_set()`/the I-frame map, known from the bitstream lookahead) and calling
    `sr.upscale(img, model="realesrgan-x4plus")` to fill `perframe_cache[anchor]` BEFORE the
    recon frontier reaches that anchor. It shares the one GPU, so it must respect condition (A).
  * RECON consumer         : `reconstruct(..., perframe_cache, anchor_set, backend="torch",
    download_output=False)` pulls decoded frames + ready anchors and emits HD frames at F.
The scheduler here tells the real pipeline: how big the decode-ahead ring + anchor prefetch
lead must be (B_min), what startup latency to expect, and — the load-bearing result — whether
the chosen (L,F,r,K) is even sustainable on one GPU.

API (importable)
----------------
  PipelineParams(L,F,r,K,dedicated_anchor_gpu=False)
  frame_period(F) / gpu_utilization(p) / is_sustainable(p)
  min_anchor_interval(L,F,r,dedicated) -> K_min (frames)
  max_anchor_rate(L,F,r,dedicated)     -> anchors/sec
  min_lookahead_frames(L,F)            -> ceil(L*F)
  startup_latency_s(p)
  simulate(p, n_frames, buffer_frames, anchor_lead_frames=0) -> SimResult   (discrete-event)
  min_feasible_buffer(p, n_frames)     -> smallest B with zero stalls (or None)
  threaded_demo(...)                   -> real threads+queue+gpu_lock producer/consumer
  sweep_and_report(out=OUT)            -> writes CSV + plots + summary under out_pipeline/

Run:  python3 anchor_pipeline.py            # analytic + sim sweeps + threaded demos + artifacts
      python3 anchor_pipeline.py --measure-L   # one real x4plus call to ground L (needs MPS)
      python3 anchor_pipeline.py --demo        # just the real-thread hold-vs-stall demo
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_pipeline")

# Grounding constants (handoff Step 7/8 + this stream's one real measurement).
L_X4PLUS = 2.2        # nominal heavy-anchor SR latency (s); measured 2.46 s on contended MPS
L_COMPACT = 0.13      # compact anchor SR (for contrast)
R_FULL = 0.042        # per-frame recon GPU cost, full fwd-bwd mask  (Step 7: 42.4 / 40.1 ms)
R_ADAPT = 0.039       # adaptive mask                                (Step 7: 39.5 / 38.3 ms)
R_REACT = 0.0277      # reactive mask (cheapest live path)           (Step 7: 27.7 / 32.6 ms)


# --------------------------------------------------------------------------- #
# Parameters + closed-form analysis
# --------------------------------------------------------------------------- #
@dataclass
class PipelineParams:
    L: float                       # heavy anchor SR latency (s)
    F: float                       # target fps
    r: float                       # per-frame recon GPU cost (s)
    K: int                         # anchor interval (frames between anchors)
    dedicated_anchor_gpu: bool = False   # True -> anchor SR on a separate parallel server
    decode: float = 0.004          # per-frame decode cost (s, CPU/PyAV); assumed overlapped


def frame_period(F: float) -> float:
    return 1.0 / F


def gpu_utilization(p: PipelineParams) -> float:
    """Utilization of the busiest GPU server (>1 => unsustainable). Single GPU: one server does
    both, F*r (recon) + (F/K)*L (anchor SR). Dedicated: two parallel servers; the BINDING one is
    max(F*r recon, (F/K)*L anchor) -- either can be the bottleneck."""
    recon_u = p.F * p.r
    anchor_u = (p.F / p.K) * p.L
    if p.dedicated_anchor_gpu:
        return max(recon_u, anchor_u)
    return recon_u + anchor_u


def anchor_gpu_utilization(p: PipelineParams) -> float:
    """Anchor-SR server utilization L/(K*tau) = (F/K)*L. Only meaningful for dedicated GPU."""
    return (p.F / p.K) * p.L


def is_sustainable(p: PipelineParams) -> bool:
    if p.dedicated_anchor_gpu:
        return (p.F * p.r <= 1.0 + 1e-9) and (anchor_gpu_utilization(p) <= 1.0 + 1e-9)
    return gpu_utilization(p) <= 1.0 + 1e-9


def min_anchor_interval(L: float, F: float, r: float, dedicated: bool = False) -> float:
    """K_min: smallest anchor interval (in frames) that is throughput-sustainable.
    Single GPU:  K_min = L / (tau - r) = L*F / (1 - F*r)   (inf if recon alone >= budget).
    Dedicated:   K_min = L * F        (anchor must finish within its own period)."""
    tau = 1.0 / F
    if dedicated:
        return L * F
    if r >= tau:                   # recon alone already saturates the GPU -> no room at all
        return math.inf
    return L / (tau - r)


def max_anchor_rate(L: float, F: float, r: float, dedicated: bool = False) -> float:
    """Max sustainable anchors/sec. Single GPU: (1 - F*r)/L. Dedicated: 1/L."""
    if dedicated:
        return 1.0 / L
    free = 1.0 - F * r
    return free / L if free > 0 else 0.0


def min_lookahead_frames(L: float, F: float) -> int:
    """B_min = ceil(L*F): output frames the buffer must hold to ride out one L-second SR burst.
    (Epsilon guards the float case L*F = 55.0000001 from rounding up to 56.)"""
    return int(math.ceil(L * F - 1e-9))


def startup_latency_s(p: PipelineParams) -> float:
    """Added startup latency to pre-roll the cushion + compute the first anchor.
    Single GPU:  L + B_min*r = L*(1 + F*r).   Dedicated: ~L (cushion fills in parallel)."""
    B = min_lookahead_frames(p.L, p.F)
    if p.dedicated_anchor_gpu:
        return max(p.L, B * p.r)
    return p.L + B * p.r


# --------------------------------------------------------------------------- #
# Discrete-event simulator: single serial GPU (or two parallel servers)
# --------------------------------------------------------------------------- #
@dataclass
class SimResult:
    n: int
    buffer_frames: int
    prod: list            # prod[i] = wall time (s) frame i becomes display-ready
    anchors: list         # anchor display positions
    startup_s: float      # T0: when playback begins (cushion pre-rolled)
    stalls: int           # number of frames that underran the buffer
    stall_time_s: float   # total rebuffering time added
    achieved_fps: float
    sustainable: bool
    util: float
    display: list         # actual display times after rebuffering

    @property
    def buffer_sec(self) -> float:
        return self.buffer_frames / max(self.achieved_fps, 1e-9)


def _gpu_timeline(p: PipelineParams, n: int, anchor_lead_frames: int, buffer_frames: int):
    """COUPLED bounded-buffer discrete-event sim.
    Returns (prod[], display[], anchors, T0, stalls).

    The lookahead buffer of size B is the CAP on how far recon may run ahead of the playhead
    (you only decode/recon B frames ahead in a live system). So the producer (the serial GPU)
    can start frame i only when BOTH the GPU is free (frame i-1 done) AND a buffer slot is free
    (frame i-B has been displayed). Anchors at {0} U {kK}; an anchor's L-second SR is charged to
    frame max(0, anchor - lead) (lead=0 = just-in-time = worst case for the buffer).
        prod[i]    = max(prod[i-1], display[i-B]) + cost[i]            (GPU serial + buffer cap)
        display[i] = max(display[i-1] + tau, prod[i])                 (cadence, rebuffer on late)
    Dedicated anchor GPU: SR runs on a parallel server (back-to-back from t=0), so cost[i]=r and
    prod[i] additionally waits for that anchor's sr_done.
    Playback starts (display[0]) once the first B frames are buffered: T0 = prod[B-1]."""
    B = max(1, min(buffer_frames, n))
    tau = 1.0 / p.F
    anchors = [0] + [k for k in range(1, n) if k % p.K == 0]
    aset = set(anchors)

    # per-frame GPU cost (single GPU folds the anchor SR into the frame it's scheduled on)
    cost = [p.r] * n
    sr_done = None
    if p.dedicated_anchor_gpu:
        sr_done = {}
        a_free = 0.0
        for a in anchors:                       # parallel server, processed back-to-back
            a_free += p.L
            sr_done[a] = a_free
    else:
        for a in anchors:
            slot = max(0, a - anchor_lead_frames)
            cost[slot] += p.L                   # SR blocks the one GPU at this slot

    prod = [0.0] * n
    display = [0.0] * n
    # preroll: first B frames produced back-to-back (buffer not yet full -> no slot constraint)
    t = 0.0
    for i in range(B):
        start = prod[i - 1] if i > 0 else 0.0
        if sr_done is not None and i in aset:
            start = max(start, sr_done[i])
        prod[i] = start + cost[i]
    T0 = prod[B - 1]                            # playback begins when the cushion is full
    for i in range(B):
        display[i] = T0 + i * tau               # all produced by T0 -> clean cadence
    # steady state: bounded buffer couples producer to the playhead
    stalls = 0
    for i in range(B, n):
        slot_free = display[i - B]              # frame i-B must be displayed to free a slot
        start = max(prod[i - 1], slot_free)
        if sr_done is not None and i in aset:
            start = max(start, sr_done[i])
        prod[i] = start + cost[i]
        earliest = display[i - 1] + tau
        if prod[i] > earliest + 1e-9:           # underrun -> rebuffer
            display[i] = prod[i]
            stalls += 1
        else:
            display[i] = earliest
    return prod, display, anchors, T0, stalls


def simulate(p: PipelineParams, n_frames: int = 240, buffer_frames: int = None,
             anchor_lead_frames: int = 0) -> SimResult:
    """Full coupled bounded-buffer sim. Default buffer = the analytic B_min = ceil(L*F)."""
    if buffer_frames is None:
        buffer_frames = min_lookahead_frames(p.L, p.F)
    prod, display, anchors, T0, stalls = _gpu_timeline(p, n_frames, anchor_lead_frames,
                                                       buffer_frames)
    ideal_end = T0 + (n_frames - 1) * (1.0 / p.F)
    stall_time = max(0.0, display[-1] - ideal_end)
    span = display[-1] - display[0]
    fps = (n_frames - 1) / span if span > 1e-9 else p.F
    return SimResult(n=n_frames, buffer_frames=buffer_frames, prod=prod, anchors=anchors,
                     startup_s=T0, stalls=stalls, stall_time_s=stall_time, achieved_fps=fps,
                     sustainable=is_sustainable(p), util=gpu_utilization(p), display=display)


def min_feasible_buffer(p: PipelineParams, n_frames: int = 240,
                        anchor_lead_frames: int = 0):
    """Smallest LIVE lookahead buffer (frames) giving ZERO stalls. None if the config is not
    throughput-sustainable (util > 1): there is then no BOUNDED live buffer -- the requirement
    grows without bound with stream length (you could only pre-buffer an ever-larger FRACTION of
    a FINITE clip, i.e. VOD "download-ahead", not live). For sustainable configs this converges
    to ~ceil((L+r)*F) independent of n. Stalls are monotone in B, so binary search finds it."""
    if not is_sustainable(p):
        return None                            # unsustainable -> no bounded live buffer
    def stalls_at(B):
        return _gpu_timeline(p, n_frames, anchor_lead_frames, max(1, B))[4]
    hi = n_frames - 1
    if stalls_at(hi) > 0:
        return None
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if stalls_at(mid) == 0:
            hi = mid
        else:
            lo = mid + 1
    return lo


def buffer_occupancy(p: PipelineParams, res: SimResult, dt: float = None):
    """Sample buffer depth (frames produced - frames displayed) over time, for plotting."""
    if dt is None:
        dt = 1.0 / p.F / 2.0
    t_end = max(res.prod[-1], res.display[-1])
    ts = np.arange(0.0, t_end + dt, dt)
    prod = np.asarray(res.prod)
    disp = np.asarray(res.display)
    occ = np.array([(prod <= t).sum() - (disp <= t).sum() for t in ts])
    return ts, occ


# --------------------------------------------------------------------------- #
# Real threaded producer/consumer demo (mechanics) — single GPU lock = serial GPU
# --------------------------------------------------------------------------- #
def threaded_demo(L=L_X4PLUS, F=25.0, r=R_REACT, K=200, n_frames=460,
                  buffer_frames=None, time_scale=0.1, real_sr=False, verbose=True):
    """A genuine producer/consumer with `threading` + a BOUNDED `queue.Queue` + a GPU LOCK.

    The single serial GPU is modelled two ways at once:
      * a `gpu` Lock  -> anchor-SR and recon never overlap (what MPS+GIL enforce for real), and
      * a BOUNDED queue of size `buffer_frames` -> the recon-ahead is CAPPED at the lookahead
        buffer (the producer blocks on `put` when the buffer is full). This is the crux: the
        producer stays exactly `buffer_frames` ahead, so a stall happens precisely when the
        L-second anchor drains more than the buffer holds, i.e. when buffer_frames < L*F. That
        makes the demo robust to ms-jitter (it compares L vs buffer_frames*tau, both >> jitter).

    Producer thread: decode (cheap) + GPU anchor-SR (cost L at anchors) + GPU recon (cost r),
    `put`ting finished frame indices. Consumer thread: pull one frame every tau (playback); an
    empty buffer at the pull instant = a STALL (rebuffer: block for the late frame, reset cadence).
    Times are scaled by `time_scale` (default 0.1 = 10x) so a 2.2 s anchor demos in 0.22 s; the
    stall/no-stall verdict and fps RATIO are scale-invariant. `real_sr=True` grounds L with ONE
    real x4plus call, then uses scaled sleeps (keeps the demo bounded on contended MPS).

    Returns dict(stalls, played, startup_s, achieved_fps, buffer_frames, ...)."""
    if buffer_frames is None:
        buffer_frames = min_lookahead_frames(L, F)
    buffer_frames = max(1, min(buffer_frames, n_frames - 1))
    tau = 1.0 / F
    anchors = set([0] + [k for k in range(1, n_frames) if k % K == 0])

    real_L = None
    if real_sr:
        try:
            import sr as _sr
            x = np.random.default_rng(0).integers(0, 256, (320, 640, 3)).astype(np.uint8)
            _sr.upscale(x, model="realesrgan-x4plus")           # warm
            _sr.upscale(x, model="realesrgan-x4plus")
            real_L = _sr.last_latency_ms("realesrgan-x4plus") / 1000.0
            if verbose:
                print(f"[threaded_demo] grounded L from one real x4plus call: {real_L:.2f}s")
        except Exception as e:                                   # never swallow silently
            print(f"[threaded_demo] real SR grounding failed ({e!r}); using L={L}s parameter")
            real_L = None
    L_use = real_L if real_L else L

    gpu = threading.Lock()                       # the single serial GPU
    q: "queue.Queue[int]" = queue.Queue(maxsize=buffer_frames)   # recon-ahead capped at buffer
    state = dict(stalls=0, played=0, t_play_start=None)

    def gpu_work(seconds):
        with gpu:                                # serialize: anchor-SR and recon cannot overlap
            time.sleep(seconds * time_scale)

    def producer():
        for i in range(n_frames):
            if i in anchors:
                gpu_work(L_use)                  # heavy anchor SR holds the GPU
            gpu_work(r)                          # per-frame recon
            q.put(i)                             # BLOCKS when the buffer is full (the cap)

    slot = tau * time_scale
    # 1-frame jitter tolerance: a real rebuffer (anchor underrun) arrives TENS of slots late;
    # OS sleep-granularity jitter is <1 slot. grace ~1.2*slot rejects jitter, keeps true stalls.
    grace = 1.2 * slot
    show = []                                    # wall time each frame is actually displayed

    def consumer():
        # pre-roll: wait until the lookahead buffer is full, THEN start the playback clock.
        while q.qsize() < buffer_frames:
            time.sleep(0.0005)
        base = time.perf_counter()               # playback baseline (display deadline origin)
        state["t_play_start"] = base
        for idx in range(n_frames):
            deadline = base + idx * slot
            q.get()                              # in-order delivery; blocks if producer behind
            avail = time.perf_counter()
            if avail > deadline + grace:          # frame arrived after its cadence slot -> STALL
                state["stalls"] += 1
                base = avail - idx * slot         # rebuffer: re-baseline so we don't cascade-count
                show.append(avail)
            else:                                 # on time: hold until the cadence deadline
                s = deadline - time.perf_counter()
                if s > 0:
                    time.sleep(s)
                show.append(time.perf_counter())
            state["played"] += 1

    t0 = time.perf_counter()
    tp = threading.Thread(target=producer, daemon=True)
    tc = threading.Thread(target=consumer, daemon=True)
    tp.start()
    tc.start()
    tc.join()
    wall = time.perf_counter() - t0
    startup = ((state["t_play_start"] or t0) - t0) / time_scale
    played_n = state["played"]
    span = (show[-1] - show[0]) if len(show) > 1 else 0.0
    achieved_fps = ((played_n - 1) * time_scale / span) if span > 1e-9 else 0.0   # real-world equiv
    res = dict(stalls=state["stalls"], played=played_n, startup_s=startup,
               achieved_fps=achieved_fps, buffer_frames=buffer_frames, B_min=min_lookahead_frames(L, F),
               wall_s=wall, time_scale=time_scale, L=L_use, F=F, r=r, K=K)
    if verbose:
        tag = "HOLDS" if res["stalls"] == 0 else f"STALLS x{res['stalls']}"
        print(f"  [threaded] L={L_use:.2f}s F={F:.0f} r={r*1000:.0f}ms K={K} buffer={buffer_frames}f "
              f"(B_min={res['B_min']}f) -> {tag}; played {played_n}/{n_frames}, "
              f"~{achieved_fps:.1f}fps, startup~{startup:.1f}s (wall {wall:.2f}s @ {1/time_scale:.0f}x)")
    return res


# --------------------------------------------------------------------------- #
# Sweeps + artifacts
# --------------------------------------------------------------------------- #
def _ensure_out(out):
    os.makedirs(out, exist_ok=True)


def sweep_and_report(out=OUT, n_frames=300):
    """Run the analytic + discrete-event sweeps, write CSV + plots + a summary to out_pipeline/."""
    _ensure_out(out)
    rows = []

    Fs = [25.0, 30.0, 15.0]
    rs = [("full", R_FULL), ("adaptive", R_ADAPT), ("reactive", R_REACT)]
    Ls = [("x4plus", L_X4PLUS), ("x4plus-contended", 2.46), ("compact", L_COMPACT)]
    Ks = [12, 24, 48, 96, 192, 384]
    GOP = 48

    for Lname, L in Ls:
        for F in Fs:
            for rname, r in rs:
                for ded in (False, True):
                    kmin = min_anchor_interval(L, F, r, dedicated=ded)
                    amax = max_anchor_rate(L, F, r, dedicated=ded)
                    for K in Ks:
                        p = PipelineParams(L=L, F=F, r=r, K=K, dedicated_anchor_gpu=ded)
                        # size the window so each K sees >=3 anchors (the buffer gets exercised)
                        nf = max(n_frames, 3 * K + 20)
                        bmin = min_feasible_buffer(p, n_frames=nf)
                        # one sim at a generous buffer to confirm sustainable->holds / unsust->stalls
                        res = simulate(p, n_frames=nf,
                                       buffer_frames=min_lookahead_frames(L, F) + 3)
                        rows.append(dict(
                            L_name=Lname, L=L, F=F, r_name=rname, r=r,
                            mode=("dedicated" if ded else "single"),
                            K=K, K_min=(round(kmin, 1) if math.isfinite(kmin) else "inf"),
                            K_min_le_GOP=("yes" if math.isfinite(kmin) and kmin <= GOP else "no"),
                            a_max_per_s=round(amax, 4),
                            util=round(gpu_utilization(p), 3),
                            sustainable=("yes" if is_sustainable(p) else "no"),
                            B_min_frames=(bmin if bmin is not None else "none"),
                            B_min_sec=(round(bmin / F, 3) if bmin is not None else "inf"),
                            B_analytic_LF=min_lookahead_frames(L, F),
                            B_analytic_tight=int(math.ceil((L + r) * F - 1e-9)),
                            startup_s=round(startup_latency_s(p), 3),
                            stalls_at_Bmin3=res.stalls,
                            achieved_fps=round(res.achieved_fps, 2),
                        ))

    csv_path = os.path.join(out, "sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _plots(out, n_frames)
    summary = _summary(out, rows)
    return csv_path, summary


def _plots(out, n_frames):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] matplotlib unavailable ({e!r}); skipping plots")
        return

    # (1) Throughput ceiling: K_min vs F, single GPU, for the three recon costs, L=x4plus.
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    Fgrid = np.linspace(10, 35, 200)
    for rname, r in [("full 42ms", R_FULL), ("adaptive 39ms", R_ADAPT), ("reactive 28ms", R_REACT)]:
        kmin = [min_anchor_interval(L_X4PLUS, F, r) for F in Fgrid]
        kmin = [k if math.isfinite(k) else np.nan for k in kmin]
        ax[0].plot(Fgrid, kmin, label=f"single GPU, {rname}")
    ax[0].plot(Fgrid, [min_anchor_interval(L_X4PLUS, F, 0, dedicated=True) for F in Fgrid],
               "k--", label="dedicated anchor GPU (L*F)")
    ax[0].axhline(48, color="gray", ls=":", label="GOP = 48 frames")
    ax[0].set_xlabel("target fps F"); ax[0].set_ylabel("min anchor interval K_min (frames)")
    ax[0].set_title("Throughput ceiling for x4plus (L=2.2s)\nK below the curve -> stalls forever")
    ax[0].set_ylim(0, 600); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # (2) Buffer occupancy: adequate buffer (no stall) vs too-small buffer (stall), single GPU.
    # K=200 > K_min(~179) is throughput-sustainable; the window spans anchors at 0/200/400.
    nplot = 480
    B_ok = int(math.ceil((L_X4PLUS + R_REACT) * 25 - 1e-9)) + 1   # tight bound + 1 frame margin
    p_ok = PipelineParams(L=L_X4PLUS, F=25, r=R_REACT, K=200)
    res_ok = simulate(p_ok, n_frames=nplot, buffer_frames=B_ok)
    res_small = simulate(p_ok, n_frames=nplot, buffer_frames=20)      # < L*F -> drains to a stall
    for res, lab, c in [(res_ok, f"buffer={res_ok.buffer_frames}f (>=B_min) -> "
                                  f"{res_ok.stalls} stalls, {res_ok.achieved_fps:.1f}fps", "tab:green"),
                        (res_small, f"buffer=20f (<B_min={B_ok}) -> {res_small.stalls} stalls, "
                                    f"{res_small.achieved_fps:.1f}fps", "tab:red")]:
        ts, occ = buffer_occupancy(p_ok, res)
        ax[1].plot(ts, occ, label=lab, color=c)
    ax[1].axhline(0, color="k", lw=0.8)
    for a in [x for x in res_ok.anchors if x > 0][:3]:
        ax[1].axvline(res_ok.display[a], color="tab:blue", ls=":", alpha=0.4)
    ax[1].set_xlabel("wall time (s)"); ax[1].set_ylabel("buffer depth (frames ready, undisplayed)")
    ax[1].set_title("Buffer occupancy: x4plus, F=25, reactive recon, K=200\n"
                    "each anchor SR (blue) drains ~L*F=55 frames; hitting 0 = stall")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(out, "pipeline_overview.png")
    fig.savefig(p1, dpi=110); plt.close(fig)
    print(f"[plots] wrote {p1}")


def _summary(out, rows):
    lines = []
    P = lines.append
    P("=" * 78)
    P("ANCHOR PIPELINING — buffer + throughput model (Stream 2)")
    P("=" * 78)
    P("")
    P("Grounding: x4plus anchor SR measured 2.14-2.74 s/call (median 2.46 s) on CONTENDED")
    P("MPS, 640x320 -> 2560x1280; handoff nominal ~2.2 s. Per-frame recon r from Step 7:")
    P(f"  full {R_FULL*1000:.0f}ms / adaptive {R_ADAPT*1000:.0f}ms / reactive {R_REACT*1000:.0f}ms.")
    P("")
    P("BUFFERING CONDITION (the formula):")
    P("  (A) throughput:  r + L/K <= 1/F   <=>   K >= K_min = L*F/(1 - F*r)   [single GPU]")
    P("      max sustainable anchor rate a_max = (1 - F*r)/L  anchors/sec")
    P("      dedicated anchor GPU: K_min = L*F,  recon needs r <= 1/F")
    P("  (B) lookahead:   B_min = ceil(L*F) frames = L seconds of pre-rendered output")
    P("      (tight, incl. the anchor's own recon: ceil((L+r)*F); sim confirms this +1 frame)")
    P("      startup ~= L*(1 + F*r) [single]   |   ~= L [dedicated]")
    P("")
    P("KEY OPERATING POINTS (x4plus, L=2.2s):")

    def line_for(L, F, r, rname, ded):
        kmin = min_anchor_interval(L, F, r, dedicated=ded)
        amax = max_anchor_rate(L, F, r, dedicated=ded)
        B = min_lookahead_frames(L, F)
        p = PipelineParams(L=L, F=F, r=r, K=max(1, int(math.ceil(kmin)) if math.isfinite(kmin) else 999999),
                           dedicated_anchor_gpu=ded)
        startup = startup_latency_s(p)
        kshow = f"{kmin:7.0f}f" if math.isfinite(kmin) else "    inf"
        secs = (f"{kmin/F:5.1f}s" if math.isfinite(kmin) else "  inf")
        mode = "dedicated" if ded else "single  "
        feas = "<=GOP OK" if (math.isfinite(kmin) and kmin <= 48) else "x4plus INFEASIBLE at GOP"
        P(f"  {mode} F={F:>2.0f} {rname:>8} | K_min={kshow}={secs} | a_max={amax:6.3f}/s "
          f"| B_min={B:3d}f={B/F:4.1f}s startup~{startup:4.1f}s | {feas}")

    for ded in (False, True):
        for F in (25, 30, 15):
            for rname, r in [("full", R_FULL), ("reactive", R_REACT)]:
                line_for(L_X4PLUS, F, r, rname, ded)
        P("")

    P("VERDICT (single Apple-Silicon GPU):")
    kmin_full_25 = min_anchor_interval(L_X4PLUS, 25, R_FULL)
    kmin_react_25 = min_anchor_interval(L_X4PLUS, 25, R_REACT)
    P(f"  At F=25, full recon (r=42ms): F*r={25*R_FULL:.2f} -> K_min={kmin_full_25:.0f} frames"
      f" ({'INFEASIBLE - recon alone ~saturates the GPU' if not math.isfinite(kmin_full_25) else ''}).")
    P(f"  At F=25, reactive recon (r=28ms): K_min={kmin_react_25:.0f} frames "
      f"(~{kmin_react_25/25:.1f}s) >> GOP=48 -> drift destroys quality long before re-anchor.")
    P("  => Live x4plus on ONE GPU at 25fps is NOT feasible at a useful anchor interval:")
    P("     per-frame recon already consumes most of the 40ms/frame budget, leaving almost")
    P("     no GPU time for the 2.2s anchor. You can buffer the LATENCY but not beat the")
    P("     THROUGHPUT ceiling. Feasible regimes:")
    P("       * drop to F=15fps with reactive recon  -> K_min ~= "
      f"{min_anchor_interval(L_X4PLUS,15,R_REACT):.0f} frames (~GOP, borderline OK);")
    P("       * a DEDICATED anchor accelerator (2nd GPU / machine / ANE) -> K_min = L*F = "
      f"{min_anchor_interval(L_X4PLUS,25,0,dedicated=True):.0f} frames (~GOP) at 25fps, OK;")
    P(f"       * or a CHEAPER heavy anchor (L down to ~{ (1/25 - R_REACT)*48:.2f}s would put K_min<=GOP).")
    P("  Lookahead cost when feasible: B_min = ceil(L*F) = "
      f"{min_lookahead_frames(L_X4PLUS,25)} frames = {L_X4PLUS:.1f}s buffer, "
      f"startup ~{startup_latency_s(PipelineParams(L_X4PLUS,25,R_REACT,200)):.1f}s.")
    P("")
    P("Artifacts: out_pipeline/sweep.csv, out_pipeline/pipeline_overview.png, this summary.")
    P("=" * 78)
    text = "\n".join(lines)
    path = os.path.join(out, "summary.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    return text


def measure_L():
    """One real x4plus call to ground L (needs torch+MPS+weights). Returns seconds or None."""
    try:
        import sr as _sr
        x = np.random.default_rng(0).integers(0, 256, (320, 640, 3)).astype(np.uint8)
        _sr.load_model("realesrgan-x4plus")
        for _ in range(3):
            _sr.upscale(x, model="realesrgan-x4plus")
        med = _sr.median_latency_ms("realesrgan-x4plus") / 1000.0
        print(f"[measure_L] x4plus median = {med:.2f}s "
              f"(calls: {[round(v) for v in _sr._lat('realesrgan-x4plus')]} ms)")
        return med
    except Exception as e:
        print(f"[measure_L] failed ({e!r})")
        return None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Anchor-SR lookahead pipelining model + demo")
    ap.add_argument("--measure-L", action="store_true", help="one real x4plus call to ground L")
    ap.add_argument("--demo", action="store_true", help="only the real-thread hold-vs-stall demo")
    ap.add_argument("--sweep", action="store_true", help="only the analytic+sim sweep + artifacts")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--real-sr", action="store_true", help="ground the threaded demo with one real SR call")
    args = ap.parse_args()

    if args.measure_L:
        measure_L()
        return

    if args.demo:
        run_threaded_demos(real_sr=args.real_sr)
        return

    if args.sweep:
        _, summary = sweep_and_report(out=args.out)
        print(summary)
        return

    # default: analytic headline + sweep artifacts + threaded hold-vs-stall demos
    print(">> Analytic headline")
    for F in (25, 15):
        for rname, r in [("full", R_FULL), ("reactive", R_REACT)]:
            kmin = min_anchor_interval(L_X4PLUS, F, r)
            ded = min_anchor_interval(L_X4PLUS, F, 0, dedicated=True)
            print(f"  x4plus L=2.2 F={F} r={rname}: K_min(single)="
                  f"{kmin if math.isfinite(kmin) else 'inf'}  K_min(dedicated)={ded:.0f}  "
                  f"B_min={min_lookahead_frames(L_X4PLUS,F)}f")
    print("\n>> Sweep + artifacts")
    csv_path, summary = sweep_and_report(out=args.out)
    print(summary)
    print("\n>> Threaded producer/consumer demos (real threads, scaled time)")
    run_threaded_demos(real_sr=args.real_sr, out=args.out)


def run_threaded_demos(real_sr=False, time_scale=0.25, n_frames=440, out=None):
    # Three regimes (real threads, GPU lock = serial GPU, bounded queue = the lookahead cap).
    # K=200 > K_min(25,reactive)~179 is throughput-sustainable; anchors at 0/200/400 fall inside
    # the window so the buffer is exercised mid-playback. The DETERMINISTIC sim gives the exact
    # B_min (~56); here we use a comfortable margin so the wall-clock verdict is jitter-robust.
    bmin = min_lookahead_frames(L_X4PLUS, 25)        # 55
    cases = [
        ("case 1: sustainable K=200, buffer=100f (>> B_min=%d)  -> expect HOLDS" % bmin,
         dict(K=200, buffer_frames=100, real_sr=real_sr)),
        ("case 2: sustainable K=200, buffer=20f (< B_min=%d)    -> expect STALLS" % bmin,
         dict(K=200, buffer_frames=20)),
        ("case 3: anchors too frequent K=100 < K_min~179, buf=100f -> expect STALLS",
         dict(K=100, buffer_frames=100)),
    ]
    results = []
    for label, kw in cases:
        print("  " + label)
        res = threaded_demo(L=L_X4PLUS, F=25, r=R_REACT, n_frames=n_frames,
                            time_scale=time_scale, **kw)
        results.append((label, res))
    if out:
        _ensure_out(out)
        with open(os.path.join(out, "threaded_demo.txt"), "w") as f:
            f.write("Real-thread producer/consumer demo (GPU lock = serial GPU; bounded queue =\n"
                    "lookahead cap). Times scaled %gx; fps/startup reported as real-world equiv.\n"
                    "Verdict is jitter-robust (1-frame grace); exact B_min is in the sim/CSV.\n\n"
                    % (1 / time_scale))
            for label, res in results:
                tag = "HOLDS" if res["stalls"] == 0 else f"STALLS x{res['stalls']}"
                f.write(f"{label}\n  -> {tag}; played {res['played']}/{n_frames}, "
                        f"~{res['achieved_fps']:.1f}fps, startup~{res['startup_s']:.1f}s "
                        f"(buffer={res['buffer_frames']}f, B_min={res['B_min']}f)\n\n")
    return results


if __name__ == "__main__":
    main()
