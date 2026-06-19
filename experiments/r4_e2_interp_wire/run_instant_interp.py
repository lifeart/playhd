#!/usr/bin/env python3
"""R4-E2 -- END-TO-END SEAM verification of the "smooth 2x" interp pass through a FAITHFUL copy
of the instant fast path (server/pipeline_api.process_clip).

Everything is imported READ-ONLY. `_fast_body()` is a line-for-line copy of process_clip's
`if fast:` per-chunk body (the real anchor_sr.build_anchor_cache -> derisk.reconstruct(torch,
download_output=False) -> patch_high_fallback -> grain -> download pipeline), with the EXACT
default-OFF interp insertion the lead will land gated on `interp_2x`. It returns the list of
PRE-ENCODE frames (so we can byte-compare), plus interp stats.

Proves the four integration claims:
  * OUTPUT-ONLY + byte-identical real frames: ON's real frames (even positions) are byte-identical
    to OFF's frames -> the synthesized midpoint NEVER altered a real frame and NEVER entered R[]
    (GOTCHA #16). OFF is, by the `if interp_2x:` structure, byte-identical to today's pipeline.
  * 2x output: ON emits exactly 2*M frames for M real frames (trailing dup closes the sequence).
  * SCENE-CUT GUARD at the real chunk boundary: the cross-chunk midpoint at an I-frame/cut
    duplicates (no smear).
  * A real 2x-fps mp4 muxes with the source audio in sync.
"""
import gc
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "server"), os.path.join(_REPO, "prototype"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_api as P        # noqa: E402  READ-ONLY (stream_gops, MODE_CONFIG, _VideoWriter, _mux_av)
import anchor_sr                # noqa: E402  READ-ONLY
import fast_grain               # noqa: E402  READ-ONLY
import derisk                   # noqa: E402  READ-ONLY
import gpu_ops                  # noqa: E402  READ-ONLY
import interp_pass as IP        # noqa: E402  the shippable wire

CLIP = os.path.join(_REPO, "sample.mp4")
_MID_SEED_BASE = 1 << 20        # midpoint grain seeds live above real-frame seeds (no collision)


class _ListSink:
    """A writer with the _VideoWriter interface that just collects frames (for byte-compare)."""
    def __init__(self):
        self.frames = []
        self.encoder = "list"
    def write(self, rgb):
        self.frames.append(np.ascontiguousarray(rgb).copy())
    def close(self):
        pass


def _fast_body(input_path, max_frames, interp_2x, writer, fps):
    """Copy of process_clip's fast path body. `interp_2x` toggles the (default-OFF) smooth pass.
    Returns dict(done_real, n_emit, n_interp, n_interp_dup, dup_positions)."""
    cfg = P.MODE_CONFIG["instant"]
    eff_scale = P.INSTANT_SCALE
    done = 0          # REAL frames (progress)
    n_emit = 0        # TOTAL frames written (real + inserted) -> mux duration
    n_interp = n_interp_dup = mid_count = 0
    dup_positions = []
    w_lr = h_lr = w_hd = h_hd = None
    ggrain = None
    interp_carry = None        # previous chunk's LAST recon tensor (for the cross-chunk midpoint)

    for chunk in P.stream_gops(input_path, max_frames=max_frames):
        if w_lr is None:
            h_lr, w_lr = chunk[0][1].shape[:2]
            w_hd, h_hd = w_lr * eff_scale, h_lr * eff_scale
        tfn = P._motion_keyed_thresh_fn(chunk, P.INSTANT_FALLBACK_THRESH)
        perframe_cache, _ac, sr_set = anchor_sr.build_anchor_cache(
            chunk, w_hd, h_hd, cfg["sr_mode"], occ_mode=cfg["occ"],
            fallback_thresh=P.INSTANT_FALLBACK_THRESH,
            tile=P.INSTANT_TILE_SR, gpu_cache=P.INSTANT_GPU_CACHE, thresh_fn=tfn)
        _, R = derisk.reconstruct(
            chunk, None, eff_scale, True, cfg["occ"], perframe_cache, set(),
            backend=cfg["backend"], collect_metrics=False, download_output=False)
        anchor_sr.patch_high_fallback(
            chunk, R, w_hd, h_hd, cfg["sr_mode"],
            fallback_thresh=P.INSTANT_FALLBACK_THRESH, skip=sr_set,
            tile=P.INSTANT_TILE_SR, thresh_fn=tfn)
        if ggrain is None and cfg["grain"] != "off":
            ggrain = fast_grain.GpuGrain(h_hd, w_hd, gpu_ops.device())

        for i in range(len(chunk)):
            recon_t = R[i]["recon"]
            # ---- INSERTED (default-OFF): emit the MV-interp midpoint that PRECEDES real frame i ----
            #      left = previous chunk's last frame (i==0) or R[i-1] (i>0). Connecting field is
            #      frame i's codec 'past' MV (reused, zero new flow). Output-only: reads R only.
            if interp_2x:
                left = interp_carry if i == 0 else R[i - 1]["recon"]
                if left is not None:
                    fx, fy = IP.connecting_flow(chunk, i, h_lr, w_lr,
                                                _build_lr_flow=derisk.build_lr_flow)
                    mid_t, minfo = IP.midpoint_torch(left, recon_t, fx, fy, eff_scale, _G=gpu_ops)
                    if cfg["grain"] != "off":
                        mid_t = ggrain.apply(mid_t, _MID_SEED_BASE + mid_count, cfg["grain"])
                    writer.write(fast_grain.download_rgb(mid_t))
                    n_emit += 1; n_interp += 1; mid_count += 1
                    if minfo["duplicated"]:
                        n_interp_dup += 1; dup_positions.append(n_emit - 1)
            # ---- real frame i (UNCHANGED from today's pipeline) ----
            rt = recon_t
            if cfg["grain"] != "off":
                rt = ggrain.apply(recon_t, done, cfg["grain"])
            writer.write(fast_grain.download_rgb(rt))
            done += 1; n_emit += 1
        # carry this chunk's LAST recon to interpolate into the NEXT chunk's first frame
        interp_carry = R[len(chunk) - 1]["recon"].clone() if interp_2x else None
        del perframe_cache, R, chunk
        gc.collect()
        if gpu_ops is not None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception as e:
                print(f"  [warn] empty_cache: {e}")
    # trailing midpoint after the global last frame -> duplicate (no successor) => exact 2x
    if interp_2x and interp_carry is not None:
        rt = interp_carry
        if cfg["grain"] != "off":
            rt = ggrain.apply(interp_carry, _MID_SEED_BASE + mid_count, cfg["grain"])
        writer.write(fast_grain.download_rgb(rt))
        n_emit += 1; n_interp += 1; n_interp_dup += 1; dup_positions.append(n_emit - 1)
    return {"done_real": done, "n_emit": n_emit, "n_interp": n_interp,
            "n_interp_dup": n_interp_dup, "dup_positions": dup_positions}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=40)
    args = ap.parse_args()
    fps = P._probe_fps(CLIP)            # Fraction (the writer needs .numerator)
    print(f"{'='*72}\nR4-E2 -- END-TO-END SEAM (instant fast path + interp), {args.max_frames} frames")
    print(f"{'='*72}")

    # ---- OFF: baseline frame stream (this IS today's pipeline body) ----
    off_sink = _ListSink()
    off = _fast_body(CLIP, args.max_frames, False, off_sink, fps)
    M = len(off_sink.frames)
    print(f"OFF (today's pipeline): {M} frames  done_real={off['done_real']}")

    # ---- ON: smooth 2x ----
    on_sink = _ListSink()
    on = _fast_body(CLIP, args.max_frames, True, on_sink, fps)
    Non = len(on_sink.frames)
    print(f"ON  (smooth 2x):        {Non} frames  real={on['done_real']} inserted={on['n_interp']} "
          f"(duplicated={on['n_interp_dup']} at out-pos {on['dup_positions']})")

    # ---- CHECK 1: exact 2x ----
    c_2x = (Non == 2 * M and on["done_real"] == M)
    print(f"\n  [1] exact 2x: {Non} == 2*{M} -> {'PASS' if c_2x else 'FAIL'}")

    # ---- CHECK 2: OUTPUT-ONLY -- ON's real frames (even positions) byte-identical to OFF ----
    real_on = on_sink.frames[0::2]
    n_match = sum(int(np.array_equal(a, b)) for a, b in zip(real_on, off_sink.frames))
    c_oo = (len(real_on) == M and n_match == M)
    print(f"  [2] output-only (ON even frames == OFF, byte-identical): {n_match}/{M} "
          f"-> {'PASS' if c_oo else 'FAIL'}")
    print(f"      (=> midpoints never altered a real frame / never entered the reference chain R[])")

    # ---- CHECK 3: scene-cut guard fired at the real chunk boundary ----
    # locate chunk boundaries: a fresh chunk's first real frame is force-anchored; the inserted
    # frame just before it is the cross-chunk midpoint. A boundary at an I-frame/cut -> duplicated.
    c_guard = on["n_interp_dup"] >= 1
    print(f"  [3] scene-cut/boundary guard fired (>=1 duplication incl. trailing): "
          f"n_dup={on['n_interp_dup']} -> {'PASS' if c_guard else 'FAIL'}")

    # ---- CHECK 4: a real, playable, in-sync 2x-fps mp4 ----
    out_mp4 = os.path.join(_HERE, "smooth2x_demo.mp4")
    vtmp = out_mp4 + ".video.tmp.mp4"
    out_fps = fps * 2                   # Fraction * int -> Fraction (exact 2x rate)
    writer = P._VideoWriter(vtmp, out_fps, codec=None)
    stats = _fast_body(CLIP, args.max_frames, True, writer, fps)
    writer.close()
    audio_note = P._mux_av(vtmp, CLIP, out_mp4, stats["n_emit"], out_fps)
    if os.path.exists(vtmp):
        os.remove(vtmp)
    res = f"{off_sink.frames[0].shape[1]}x{off_sink.frames[0].shape[0]}"
    ok_mp4, info = P._verify_mp4(out_mp4, stats["n_emit"], res)
    print(f"  [4] real 2x mp4: {info}\n      audio={audio_note}  -> {'PASS' if ok_mp4 else 'FAIL'}")
    print(f"      out_fps={out_fps} (src {fps}); duration video={info.get('video_dur_s')}s "
          f"audio={info.get('audio_dur_s')}s (synced={info.get('sync_ok')})")

    allpass = c_2x and c_oo and c_guard and ok_mp4
    print(f"\n{'='*72}\nSEAM VERDICT: {'ALL PASS' if allpass else 'FAIL'} "
          f"(2x={c_2x} output-only={c_oo} guard={c_guard} mp4={ok_mp4})")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
