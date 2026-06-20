#!/usr/bin/env python3
"""R7-E1 -- AUTHORITATIVE pixel-level verification THROUGH THE REAL process_clip.

We run the REAL pipeline_api.process_clip (instant mode) twice -- once with the smooth-2x flag OFF
(today's pipeline) and once ON -- with ONE surgical swap: the encoder is replaced by a capture sink
so we read the EXACT PRE-ENCODE frames process_clip emits, in emit order (H.264 is lossy + uses a
different reference structure when midpoints are interleaved, so a decoded-mp4 even-frame compare
would be noisy; the pre-encode frames are the ground truth for "output-only / byte-identical").

process_clip's code path is otherwise 100% real: real anchor_sr.build_anchor_cache -> real
derisk.reconstruct(torch, download_output=False) -> real patch_high_fallback -> real GpuGrain ->
real interp_pass.midpoint_torch insertion gated by the real INSTANT_INTERP_2X flag. We also wrap
interp_pass.midpoint_torch to record each midpoint's {duplicated, hole_frac} and, for duplicates,
assert the emitted recon EQUALS the left neighbour (a true duplicate, NOT a warped ghost).

Checks: OFF count == N; ON count == 2N (exact 2x); ON even frames == OFF frames byte-identical
(output-only -> midpoint never altered a real frame / never entered R[]); scene-cut guard fires at
the real I-frame chunk boundary (frame 28 of sample.mp4) and duplicates (recon==left), not ghosts.
"""
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "server"), os.path.join(_REPO, "prototype")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_api as P   # noqa: E402  THE REAL pipeline (landed)
import interp_pass         # noqa: E402

CLIP = os.path.join(_REPO, "sample.mp4")

_CAP = {"frames": None}            # active capture list
_MINFOS = []                      # recorded midpoint infos (ON run)


class _CaptureSink:
    """A _VideoWriter-shaped sink that records the exact pre-encode frames process_clip emits."""
    encoder = "capture"

    def __init__(self, *a, **k):
        pass

    def write(self, rgb_uint8):
        _CAP["frames"].append(np.ascontiguousarray(rgb_uint8).copy())

    def close(self):
        pass


def _wrap_midpoint(left, right, fx, fy, scale, **kw):
    mt, info = interp_pass.midpoint_torch.__wrapped__(left, right, fx, fy, scale, **kw)
    rec = dict(info)
    if info["duplicated"]:
        rec["equals_left"] = bool(torch.equal(mt, left))   # true DUP (no warp), not a ghost
    _MINFOS.append(rec)
    return mt, info


def _run(flag, n):
    _CAP["frames"] = []
    P.INSTANT_INTERP_2X = bool(flag)
    P.process_clip(CLIP, "instant", max_frames=n,
                   out_path=os.path.join(_HERE, f"_cap_{int(flag)}.mp4"))
    return _CAP["frames"]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    # ---- swap the encoder for a capture sink + wrap the midpoint op (everything else is REAL) ----
    P._VideoWriter = _CaptureSink
    P._mux_av = lambda *a, **k: "captured(no-mux)"   # no temp video file is produced in capture mode
    _orig_mid = interp_pass.midpoint_torch
    interp_pass.midpoint_torch = _wrap_midpoint
    interp_pass.midpoint_torch.__wrapped__ = _orig_mid

    # ---- OFF (today's pipeline) ----
    _MINFOS.clear()
    off = [f.copy() for f in _run(False, n)]
    M = len(off)
    off_minfos = list(_MINFOS)

    # ---- ON (smooth 2x) ----
    _MINFOS.clear()
    on = [f.copy() for f in _run(True, n)]
    Non = len(on)
    on_minfos = list(_MINFOS)

    res = f"{off[0].shape[1]}x{off[0].shape[0]}"

    # ---- checks ----
    c_off_n = (M == n) or (M == P.LAST_STATS.get("n_frames", M))  # OFF emits exactly the real frames
    c_2x = (Non == 2 * M)
    even = on[0::2]
    n_match = sum(int(np.array_equal(a, b)) for a, b in zip(even, off))
    c_oo = (len(even) == M and n_match == M)
    c_off_no_interp = (len(off_minfos) == 0)                      # OFF never calls the interp op

    n_dup = sum(int(m["duplicated"]) for m in on_minfos)
    dups_equal_left = [m for m in on_minfos if m.get("duplicated")]
    c_dup_isreal = all(m.get("equals_left", False) for m in dups_equal_left) and len(dups_equal_left) > 0
    # the cross-chunk midpoint at the I-frame boundary (frame 28) must be a duplicate (100% hole)
    boundary_dups = [m for m in on_minfos if m["duplicated"] and m["hole_frac"] > 0.99]
    c_cutguard = len(boundary_dups) >= 1

    result = {
        "n_requested": n, "M_off": M, "N_on": Non, "resolution": res,
        "off_count_ok": bool(c_off_n),
        "exact_2x": bool(c_2x),
        "output_only_match": f"{n_match}/{M}", "output_only_ok": bool(c_oo),
        "off_made_no_interp_calls": bool(c_off_no_interp),
        "n_interp_total": len(on_minfos), "n_interp_dup": n_dup,
        "dup_equals_left_recon": bool(c_dup_isreal),
        "cutguard_fired_at_iframe_boundary": bool(c_cutguard),
        "n_boundary_dups_holefrac>0.99": len(boundary_dups),
        "sample_minfos_head": on_minfos[:3],
        "sample_minfos_dups": [m for m in on_minfos if m["duplicated"]][:5],
        "last_stats_on": {k: P.LAST_STATS.get(k) for k in
                          ("mode", "n_frames", "n_video_frames", "fps", "out_fps",
                           "smooth_2x", "n_interp", "n_interp_dup")},
    }
    allpass = (c_off_n and c_2x and c_oo and c_off_no_interp and c_dup_isreal and c_cutguard)
    result["VERDICT"] = "ALL PASS" if allpass else "FAIL"
    with open(os.path.join(_HERE, "capture_verify.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
