"""R4-E1 fix (a) END-TO-END verification through the REAL layered pipeline.

Monkeypatches layered_api.composite_frame so PASS B applies the plate-validity guard, runs the
ACTUAL pipeline_api.process_clip('layered') on:
  * c7_staticcut.mp4         -- the repro (missed similar-luma cut @ frame 28 -> corrupt bg)
  * sample_static_5093.mp4   -- a normal single-scene static window (no-regression control)
and measures per-frame LR-consistency (output downscaled-to-LR vs decoded LR -- the HONEST metric
for THIS bug; tOF is blind). Two passes per clip:
  OBSERVE  guard records bg-PSNR but never acts -> reproduces the BROKEN baseline.
  ACTIVE   guard falls back (full-frame compact SR) when bg-PSNR < threshold -> the FIX.

GPU (MPS) freed between runs. Run:
  python3 experiments/r4_e1_layeredcut/verify_guard.py
"""
import os
import sys
import gc
import json

import numpy as np
import cv2
import av

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "server"))
sys.path.insert(0, os.path.join(REPO, "prototype"))

import pipeline_api as P          # noqa: E402
import layered_api as _layered    # noqa: E402
import derisk                     # noqa: E402
import guard as G                 # noqa: E402

try:
    import torch as _torch
except Exception:
    _torch = None

CLIPS = os.path.join(REPO, "experiments", "r3_e2_robustness", "clips")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

_orig_composite_frame = _layered.composite_frame
_LOG = []   # per-frame (bg_psnr, tripped) for the current run


def _free_gpu():
    gc.collect()
    if _torch is not None:
        try:
            if _torch.backends.mps.is_available():
                _torch.mps.empty_cache()
        except Exception:
            pass


_BASE = {"v": None}   # per-scene bg-PSNR EMA baseline (the harness runs ~one scene per clip)


def make_guarded(active, thresh):
    """Return a composite_frame replacement. Mirrors layered_api.composite_frame but inserts the
    guard: compute fg_hd ONCE; if active and the plate fails the bg-PSNR check (absolute floor OR
    relative cliff vs the per-scene baseline), output fg_hd (full-frame compact SR -- faithful to
    the real frame) instead of the plate composite."""
    lp = _layered.lp

    def guarded(img, pha, plate_hd, w_hd, h_hd):
        fg_hd, _ms = lp.foreground_compact(img)                  # computed anyway
        bg_psnr = G.plate_bg_psnr(img, pha, plate_hd)
        bad = G.plate_is_bad(bg_psnr, _BASE["v"])
        tripped = active and bad
        _LOG.append((round(float(bg_psnr), 2) if np.isfinite(bg_psnr) else 999.0, bool(tripped)))
        if not bad and np.isfinite(bg_psnr):                     # update baseline on PASSED frames
            _BASE["v"] = bg_psnr if _BASE["v"] is None else (
                G.PLATE_GUARD_EMA * bg_psnr + (1 - G.PLATE_GUARD_EMA) * _BASE["v"])
        if tripped:
            return fg_hd                                         # FALL BACK: real content, no plate
        alpha_hd = lp.alpha_to_hd(pha, (h_hd, w_hd))
        out, _c = lp.composite(fg_hd, alpha_hd, plate_hd,
                               feather=_layered.LAYERED_SEAM_FIX)
        return out

    return guarded


def decode_rgb(path, max_frames=None):
    cont = av.open(path)
    vs = cont.streams.video[0]
    out = []
    for fr in cont.decode(vs):
        if max_frames is not None and len(out) >= max_frames:
            break
        out.append(fr.to_ndarray(format="rgb24"))
    cont.close()
    return out


def lr_consistency_series(out_path, clip_path, n):
    out_hd = decode_rgb(out_path)
    lr_in = decode_rgb(clip_path, max_frames=n)
    m = min(len(out_hd), len(lr_in))
    return [round(derisk.psnr_lr_consistency(out_hd[i], lr_in[i]), 2) for i in range(m)]


def run_one(clip_path, tag, n, active, thresh):
    global _LOG
    _LOG = []
    _BASE["v"] = None
    _layered.composite_frame = make_guarded(active, thresh)
    out_path = os.path.join(OUT, f"{tag}_{'active' if active else 'observe'}.mp4")
    try:
        P.process_clip(clip_path, "layered", max_frames=n, out_path=out_path)
        stats = dict(P.LAST_STATS)
    finally:
        P.end_job()
        _layered.composite_frame = _orig_composite_frame
        _free_gpu()
    lrc = lr_consistency_series(out_path, clip_path, n)
    return {
        "tag": tag, "active": active, "thresh": thresh,
        "n_scenes": stats.get("n_scenes"), "fallback_scenes": stats.get("fallback_scenes"),
        "scene_verdicts": stats.get("scene_verdicts"),
        "lrc": lrc,
        "lrc_min": round(min(lrc), 2), "lrc_mean": round(float(np.mean(lrc)), 2),
        "bg_psnr_log": list(_LOG),
        "n_tripped": sum(1 for (_p, t) in _LOG if t),
    }


def main():
    N = 40
    jobs = [
        (os.path.join(CLIPS, "c7_staticcut.mp4"), "c7", N, 28),       # cut @ 28
        (os.path.join(HERE, "sample_static_5093.mp4"), "sampstatic", 44, None),
    ]
    results = {}
    for clip_path, tag, n, cut in jobs:
        print(f"\n##### {tag}  (clip={os.path.basename(clip_path)}, n={n}, cut@{cut}) #####")
        obs = run_one(clip_path, tag, n, active=False, thresh=G.PLATE_GUARD_PSNR_DB)
        print(f"[OBSERVE] n_scenes={obs['n_scenes']} verdicts={obs['scene_verdicts']} "
              f"fallback={obs['fallback_scenes']}")
        print(f"  LRC mean={obs['lrc_mean']} min={obs['lrc_min']}")
        print(f"  LRC series={obs['lrc']}")
        print(f"  bg_psnr series={[p for (p,_t) in obs['bg_psnr_log']]}")
        act = run_one(clip_path, tag, n, active=True, thresh=G.PLATE_GUARD_PSNR_DB)
        print(f"[ACTIVE ] tripped {act['n_tripped']}/{len(act['bg_psnr_log'])} frames "
              f"(thresh={act['thresh']} dB)")
        print(f"  LRC mean={act['lrc_mean']} min={act['lrc_min']}")
        print(f"  LRC series={act['lrc']}")
        results[tag] = {"observe": obs, "active": act, "cut": cut}
    with open(os.path.join(HERE, "verify_guard_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote verify_guard_results.json")

    # ---- PASS/FAIL summary ----
    print("\n===== PASS/FAIL =====")
    c7 = results["c7"]
    post = lambda s: s[c7["cut"]:]    # noqa: E731  post-cut frames
    c7_obs_post_min = min(post(c7["observe"]["lrc"]))
    c7_act_post_min = min(post(c7["active"]["lrc"]))
    print(f"c7 post-cut LRC min: OBSERVE(broken)={c7_obs_post_min}  ACTIVE(fixed)={c7_act_post_min}")
    ss = results["sampstatic"]
    print(f"sampstatic guard trips: {ss['active']['n_tripped']} (must be 0); "
          f"LRC observe_mean={ss['observe']['lrc_mean']} active_mean={ss['active']['lrc_mean']} "
          f"(must be ~equal); verdicts={ss['observe']['scene_verdicts']}")


if __name__ == "__main__":
    main()
