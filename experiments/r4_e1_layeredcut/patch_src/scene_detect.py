"""scene_detect.py -- robust scene-CUT detector (one source of truth for the server).

WHY THIS EXISTS
---------------
The propagation pipeline reconstructs every non-anchor frame by WARPING a super-resolved
reference (an I/P backbone recon) with the codec motion vectors. That is only valid WITHIN
one scene. When a scene CUT happens that is NOT a clean codec I-frame -- e.g. two segments
spliced + re-encoded so the cut lands mid-GOP, or B-frame leaves that straddle the cut --
`derisk.reconstruct` warps the PRE-cut anchor across the cut and smears the old scene into
the new one (a visible cross-cut artifact).

The fix is to force a FRESH ANCHOR / chunk boundary at every detected cut so a reconstruction
chunk never spans a cut. `stream_gops` already cuts a chunk at every codec I-frame; this module
adds the missing signal: detect content cuts that the codec did NOT mark with an I-frame, and
cut a chunk there too (its first frame becomes a forced fresh anchor).

THE SIGNAL (tuned on sample.mp4; see module test `__main__`)
------------------------------------------------------------
Per-frame mean |Δluma| between consecutive DISPLAY frames, combined with the codec pict_type
(I/P/B) and an adaptive motion baseline (hysteresis), then a minimum-scene-length greedy
filter. A frame `i` STARTS a new scene (is a cut) when ANY of:

  (A) ABSOLUTE   d > CUT_THRESH                      -- near-total content change, any frame
                                                        type (catches a mid-GOP P/B cut with a
                                                        large residual that the codec did not
                                                        promote to an I-frame).
  (B) I-FRAME    ptype == "I" and d > IFRAME_THRESH  -- the encoder re-anchored AND the content
                                                        changed: a real cut. A PERIODIC keyframe
                                                        (small d) is NOT a cut (the whole reason
                                                        we cannot anchor on I-frames alone).
  (C) RELATIVE   d > REL_FLOOR and d > REL_MULT*base -- a spike far above the LOCAL motion floor
                                                        (`base` = EMA of recent non-cut diffs).
                                                        The adaptive baseline is the hysteresis:
                                                        during sustained fast motion `base` rises
                                                        so a motion BURST does not fragment into
                                                        tiny scenes; a quiet scene keeps `base`
                                                        low so a moderate real cut still fires.

A greedy MIN_SCENE_LEN filter then drops any cut closer than MIN_SCENE_LEN frames to the
previous accepted cut, so a busy run never spawns one-frame scenes.

USAGE
-----
  * STREAMING (stream_gops): `det = StreamingCutDetector(); det.update(idx, ptype, img)` per
    frame in display order -> True when this frame starts a new scene. Holds ONE prev-luma
    frame; constant memory.
  * BATCH (layered segment_scenes): `find_cuts(path, max_frames)` -> (cut_indices, total) using
    the SAME StreamingCutDetector -> ONE source of truth for both call sites.

The detector consumes uint8 HxWx3 RGB frames (the format `stream_gops`/`stream_frames` already
decode), so no extra decode pass and no GPU is needed.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Tuned thresholds (sample.mp4). The absolute + I-frame pair reproduces the values the
# prototype/layered path already validated (CUT_THRESH 60 / IFRAME 45 / MIN_SCENE 24); the
# relative/hysteresis pair is the new robustness layer for mid-GOP moderate cuts. All are
# overridable per-instance so a caller can retune without editing the module.
# --------------------------------------------------------------------------- #
CUT_THRESH = 60.0        # mean |Δluma| above this => cut on ANY frame type
IFRAME_THRESH = 45.0     # at a codec I-frame, a smaller jump still counts as a cut
REL_FLOOR = 40.0         # the relative test never fires below this absolute diff (kills it in
                         #   static scenes where base~0, on small periodic-keyframe diffs, AND on
                         #   1-frame transients/flashes whose diff returns to baseline next frame
                         #   -- those moderate-diff anomalies are NOT scene cuts; tuned on sample)
REL_MULT = 8.0           # ...and only when the diff is this many x the local motion baseline
MIN_SCENE_LEN = 24       # frames; cuts closer than this to the last accepted cut are dropped
EMA_ALPHA = 0.30         # motion-baseline EMA weight (higher = faster to follow motion changes)

# --------------------------------------------------------------------------- #
# CHROMA-DOMINANT cut test (R4-E1). A SIMILAR-LUMA scene cut (two shots of similar brightness,
# e.g. a warm->cool re-grade boundary) is INVISIBLE to the luma tests above: |Δluma| stays below
# REL_FLOOR even though the content changed (verified: c7 cut |Δluma|=26 < 40; c5 |Δluma|=47 < 8*base).
# Such cuts are betrayed by CHROMA -- the colour changes far more than the brightness. We ADD a
# chroma test (the luma path is byte-identical) that fires ONLY when chroma moves MORE than luma
# (dChroma > CHROMA_DOM*dLuma) -- the missed-cut signature. That dominance gate is what separates a
# real similar-luma cut (c7 dC 34 > dL 26; c5 dC 109 > dL 47) from sample.mp4's legitimate NON-cuts,
# which are all LUMA-dominant brightness flashes (frame 125 dC 30 < dL 34; 303 dC 31 < dL 43;
# 857 dC 30 < dL 39) and so never trip it. R4-E1 verified: sample.mp4 [0,5060] cut set is UNCHANGED
# (zero new false positives, 1.00/1.00 preserved) while the previously-missed c5/c7 cuts are caught.
CHROMA_ABS = 60.0        # |ΔU|+|ΔV| above this (and chroma-dominant) => cut (catches c5 dC~109)
CHROMA_REL_FLOOR = 22.0  # the relative chroma test never fires below this (catches c7 dC~34)
CHROMA_REL_MULT = 6.0    # ...and only at this many x the local CHROMA baseline (self-calibrates to a
                         #   panning scene's high chroma baseline, e.g. c5/c5b ~29)
CHROMA_DOM = 1.1         # gate: chroma must exceed luma by this factor (the similar-luma signature)


def luma(img: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 RGB -> float32 HxW Rec.601 luma. (Luma is enough for cut detection and is
    cheaper + less chroma-noise-sensitive than full RGB; the prototype used mean|ΔRGB|, which
    tracks |Δluma| to within a few percent on this content -- verified on sample.mp4.)"""
    f = img.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def frame_diff(prev_luma: np.ndarray, cur_luma: np.ndarray) -> float:
    """Mean absolute luma difference between two consecutive display frames."""
    return float(np.abs(cur_luma - prev_luma).mean())


def chroma_uv(img: np.ndarray) -> "Tuple[np.ndarray, np.ndarray]":
    """uint8 HxWx3 RGB -> (U, V) chroma planes (float32) via BT.601 YUV. Used only by the
    chroma-dominant cut test; cheap (one cv2 colour convert per frame, no GPU)."""
    import cv2
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV).astype(np.float32)
    return yuv[..., 1], yuv[..., 2]


def chroma_diff(prev_u, prev_v, cur_u, cur_v) -> float:
    """mean|ΔU| + mean|ΔV| between two consecutive display frames (the chroma-change signal)."""
    return float(np.abs(cur_u - prev_u).mean() + np.abs(cur_v - prev_v).mean())


class StreamingCutDetector:
    """Online scene-cut detector. Feed frames in DISPLAY order; holds one previous luma frame.

    update(idx, ptype, img) -> True iff `idx` STARTS a new scene (a cut between idx-1 and idx).
    The first frame (idx with no predecessor) is never reported as a cut (it is the start of the
    first scene, already a chunk boundary)."""

    def __init__(self, cut_thresh=CUT_THRESH, iframe_thresh=IFRAME_THRESH,
                 rel_floor=REL_FLOOR, rel_mult=REL_MULT, min_scene_len=MIN_SCENE_LEN,
                 ema_alpha=EMA_ALPHA, chroma_abs=CHROMA_ABS, chroma_rel_floor=CHROMA_REL_FLOOR,
                 chroma_rel_mult=CHROMA_REL_MULT, chroma_dom=CHROMA_DOM):
        self.cut_thresh = cut_thresh
        self.iframe_thresh = iframe_thresh
        self.rel_floor = rel_floor
        self.rel_mult = rel_mult
        self.min_scene_len = min_scene_len
        self.ema_alpha = ema_alpha
        self.chroma_abs = chroma_abs
        self.chroma_rel_floor = chroma_rel_floor
        self.chroma_rel_mult = chroma_rel_mult
        self.chroma_dom = chroma_dom                # 0 disables the chroma test (luma-only legacy)
        self._prev_luma: Optional[np.ndarray] = None
        self._prev_u: Optional[np.ndarray] = None   # previous chroma planes (for the chroma test)
        self._prev_v: Optional[np.ndarray] = None
        self._base: Optional[float] = None     # EMA motion baseline (None until re-seeded)
        self._base_c: Optional[float] = None   # EMA CHROMA baseline (None until re-seeded)
        self._last_cut_idx: int = 0            # idx of the last accepted cut (or scene start)
        self.last_diff: float = 0.0            # most recent diff (exposed for diagnostics)
        self.last_chroma_diff: float = 0.0     # most recent chroma diff (diagnostics)

    def _raw_cut(self, d: float, ptype: str) -> bool:
        if d > self.cut_thresh:                                  # (A) absolute
            return True
        if ptype == "I" and d > self.iframe_thresh:             # (B) I-frame corroborated
            return True
        if (self._base is not None and d > self.rel_floor       # (C) relative / hysteresis
                and d > self.rel_mult * self._base):
            return True
        return False

    def _chroma_cut(self, d: float, dc: float) -> bool:
        # (D) CHROMA-DOMINANT: a similar-luma colour cut. Only when chroma moves MORE than luma
        # (the missed-cut signature; rejects luma-dominant brightness flashes). Mirrors the luma
        # absolute/relative pair on the SEPARATE chroma baseline.
        if self.chroma_dom <= 0 or dc <= self.chroma_dom * d:
            return False
        if dc > self.chroma_abs:                                          # (D-abs) big colour change
            return True
        if (self._base_c is not None and dc > self.chroma_rel_floor       # (D-rel) colour spike
                and dc > self.chroma_rel_mult * self._base_c):
            return True
        return False

    def update(self, idx: int, ptype: str, img: np.ndarray) -> bool:
        cur = luma(img)
        cu, cv = (chroma_uv(img) if self.chroma_dom > 0 else (None, None))
        if self._prev_luma is None:                 # first frame: scene start, not a cut
            self._prev_luma = cur
            self._prev_u, self._prev_v = cu, cv
            self._last_cut_idx = idx
            return False
        d = frame_diff(self._prev_luma, cur)
        dc = (chroma_diff(self._prev_u, self._prev_v, cu, cv)
              if self.chroma_dom > 0 else 0.0)
        self.last_diff = d
        self.last_chroma_diff = dc
        self._prev_luma = cur
        self._prev_u, self._prev_v = cu, cv

        raw = self._raw_cut(d, ptype) or self._chroma_cut(d, dc)
        far_enough = (idx - self._last_cut_idx) >= self.min_scene_len
        is_cut = raw and far_enough

        if is_cut:
            self._last_cut_idx = idx
            self._base = None                        # re-seed baselines from the new scene
            self._base_c = None
        else:
            # update the motion baselines ONLY on non-cut frames so a cut spike never poisons them.
            # Suppressed-by-min-scene-length spikes ARE folded in (they are part of the burst the
            # baseline is meant to track).
            self._base = d if self._base is None else (
                self.ema_alpha * d + (1.0 - self.ema_alpha) * self._base)
            self._base_c = dc if self._base_c is None else (
                self.ema_alpha * dc + (1.0 - self.ema_alpha) * self._base_c)
        return is_cut


# --------------------------------------------------------------------------- #
# Batch helpers (share the SAME StreamingCutDetector -> one source of truth).
# --------------------------------------------------------------------------- #
def detect_cut_indices(frames, **kw) -> List[int]:
    """Cut indices for an in-memory list of (ptype, img, mvs) frames (display order)."""
    det = StreamingCutDetector(**kw)
    cuts = []
    for i, (ptype, img, _mvs) in enumerate(frames):
        if det.update(i, ptype, img):
            cuts.append(i)
    return cuts


def find_cuts(path, max_frames=None, **kw) -> Tuple[List[int], int]:
    """Stream a file ONCE and return (accepted_cut_indices, total_frames). Bounded memory
    (holds one previous luma frame). Used by layered_api.segment_scenes."""
    import av
    try:
        from av.sidedata.sidedata import Type as _SDType  # noqa: F401  (parity w/ decode setup)
    except Exception:
        _SDType = None
    det = StreamingCutDetector(**kw)
    cuts: List[int] = []
    total = 0
    cont = av.open(path)
    try:
        vs = cont.streams.video[0]
        # export_mvs is not needed for the diff signal, but keeping the same decode options as
        # the rest of the pipeline avoids any pict_type/timestamp surprises across call sites.
        vs.codec_context.options = {"flags2": "+export_mvs"}
        idx = 0
        for frame in cont.decode(vs):
            if max_frames is not None and idx >= max_frames:
                break
            ptype = {1: "I", 2: "P", 3: "B"}.get(int(frame.pict_type), "?")
            img = frame.to_ndarray(format="rgb24")
            if det.update(idx, ptype, img):
                cuts.append(idx)
            idx += 1
        total = idx
    finally:
        cont.close()
    return cuts, total


# --------------------------------------------------------------------------- #
# Module self-test / tuning report on sample.mp4 windows (CPU-only, no GPU).
#   python3 server/scene_detect.py [path]
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample.mp4")

    # Ground truth on sample.mp4 (from manual probing; see the task notes):
    #   real cuts in [0,900): 28,196,341,479,514,563,630,688,810
    #   periodic keyframes (NOT cuts): 0,146,291,437,582,728,873  (must NOT fire)
    #   transient flash (NOT a cut): 125 (d~34 for one frame, returns to baseline -> must NOT fire)
    #   borderline I-cut: 857 (I, d~38.8 < IFRAME_THRESH 45 -> not emitted; it is an I-frame so
    #                          stream_gops boundaries it anyway -> no missed-cut artifact)
    #   real cuts near 5000: 5032 (talking-head -> "USACHEV TODAY" title), 5051 (title -> next).
    #     BOTH are codec I-frames here, so the artifact problem does NOT occur on the raw sample;
    #     5051 is 19 frames after 5032 (< MIN_SCENE_LEN) -> merged as a short title card, and it is
    #     an I-frame so stream_gops cuts a chunk there regardless. The detector's job is the cuts
    #     the codec did NOT mark with an I-frame (see the BEFORE/AFTER mid-GOP splice test).
    print(f"scene_detect self-test on {path}")
    for lo, hi, expected in [(0, 900, {28, 196, 341, 479, 514, 563, 630, 688, 810}),
                             (5000, 5060, {5032})]:   # 5051 merged (short scene) + I-frame-covered
        cuts, total = find_cuts(path, max_frames=hi + 1)
        win = sorted(c for c in cuts if lo <= c <= hi)
        tp = sorted(c for c in win if c in expected)
        fp = sorted(c for c in win if c not in expected)
        miss = sorted(c for c in expected if c not in win)
        prec = len(tp) / max(1, len(win))
        rec = len(tp) / max(1, len(expected))
        print(f"\n[{lo},{hi}] emitted cuts={win}")
        print(f"   true-positives={tp}")
        print(f"   false-positives={fp}   missed={miss}")
        print(f"   precision={prec:.2f}  recall={rec:.2f}")
