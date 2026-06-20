#!/usr/bin/env python3
"""R8-E2 step 7: seam + byte-identical verification of the unsharp patch.

(1) OFF (unsharp=0.0): the PATCHED build_anchor_cache must produce a cache BYTE-IDENTICAL to the
    SHIPPED build_anchor_cache (today's call) -- every entry torch.equal -- on both gpu_cache paths.
(2) ON (unsharp=0.5): anchors UNCHANGED (full SR, not base_hd); non-anchor entries EQUAL the
    expected _gpu_unsharp(bicubic) and DIFFER from bicubic; the occlusion MASK / hole_frac are
    UNCHANGED (the fill is read only at fallback pixels, changes no threshold) -> eff-fallback equal.
(3) patched pipeline_api / progressive import and expose the new flags (syntax/seam).
"""
import importlib.util
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

import derisk as D            # noqa: E402
import anchor_sr as REAL      # noqa: E402  the shipped server module


def load_patched(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PATCHED = load_patched("anchor_sr_patched", os.path.join(_HERE, "patch_src", "anchor_sr.py"))

CLIP = os.path.join(_REPO, "sample.mp4")
N, SCALE, OCC, SR_MODE = 14, 2, "reactive", "realesrgan"


def eq(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.equal(a if torch.is_tensor(a) else torch.as_tensor(a),
                           b if torch.is_tensor(b) else torch.as_tensor(b))
    return np.array_equal(a, b)


def main():
    frames = D.decode_lr_and_mvs(CLIP, 2352, N)   # high-motion window
    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
    anchors, _ = REAL.anchor_indices(frames)
    ok = True

    for gpu_cache in (True, False):
        # SHIPPED call (exactly today's instant fast-path args; no unsharp param)
        c_real, _, srset_r = REAL.build_anchor_cache(
            frames, w_hd, h_hd, SR_MODE, occ_mode=OCC, fallback_thresh=0.50,
            tile=False, gpu_cache=gpu_cache, thresh_fn=None)
        # PATCHED OFF
        c_off, _, srset_o = PATCHED.build_anchor_cache(
            frames, w_hd, h_hd, SR_MODE, occ_mode=OCC, fallback_thresh=0.50,
            tile=False, gpu_cache=gpu_cache, thresh_fn=None, unsharp=0.0)
        identical = (srset_r == srset_o) and all(eq(c_real[i], c_off[i]) for i in range(N))
        print(f"[OFF gpu_cache={gpu_cache}] byte-identical to shipped: {identical}")
        ok = ok and identical

        # PATCHED ON (amount 0.5)
        c_on, _, srset_n = PATCHED.build_anchor_cache(
            frames, w_hd, h_hd, SR_MODE, occ_mode=OCC, fallback_thresh=0.50,
            tile=False, gpu_cache=gpu_cache, thresh_fn=None, unsharp=0.5)
        anchors_same = all(eq(c_real[i], c_on[i]) for i in anchors)
        nonanchor = [i for i in range(N) if i not in anchors]
        n_diff = sum(0 if eq(c_real[i], c_on[i]) else 1 for i in nonanchor)
        # expected unsharp of the bicubic entry
        if gpu_cache:
            exp = PATCHED._gpu_unsharp(c_real[nonanchor[0]].clone(), 0.5, 1.0)
        else:
            exp = PATCHED._cpu_unsharp(c_real[nonanchor[0]], 0.5, 1.0)
        matches_expected = eq(exp, c_on[nonanchor[0]])
        print(f"[ON  gpu_cache={gpu_cache}] anchors unchanged: {anchors_same}; "
              f"non-anchors changed: {n_diff}/{len(nonanchor)}; matches _unsharp(bicubic): {matches_expected}")
        ok = ok and anchors_same and (n_diff == len(nonanchor)) and matches_expected

    # (2b) mask / hole_frac unchanged ON vs OFF (fill read only at fallback pixels)
    c_off, _, _ = PATCHED.build_anchor_cache(frames, w_hd, h_hd, SR_MODE, occ_mode=OCC,
                                             fallback_thresh=0.50, gpu_cache=True, unsharp=0.0)
    c_on, _, _ = PATCHED.build_anchor_cache(frames, w_hd, h_hd, SR_MODE, occ_mode=OCC,
                                            fallback_thresh=0.50, gpu_cache=True, unsharp=0.5)
    _, Roff = D.reconstruct(frames, None, SCALE, True, OCC, c_off, set(),
                            backend="torch", collect_metrics=False, download_output=True)
    _, Ron = D.reconstruct(frames, None, SCALE, True, OCC, c_on, set(),
                           backend="torch", collect_metrics=False, download_output=True)
    hf_off = [round(float(Roff[i]["hole_frac"]), 6) for i in range(N)]
    hf_on = [round(float(Ron[i]["hole_frac"]), 6) for i in range(N)]
    hf_same = hf_off == hf_on
    print(f"[seam] hole_frac identical ON vs OFF (no threshold/mask change): {hf_same}")
    ok = ok and hf_same

    # (3) patched pipeline_api + progressive import & expose flags
    pa = load_patched("pipeline_api_patched", os.path.join(_HERE, "patch_src", "pipeline_api.py"))
    has_flags = hasattr(pa, "INSTANT_FALLBACK_UNSHARP") and hasattr(pa, "INSTANT_FALLBACK_UNSHARP_SIGMA")
    off_default = (pa.INSTANT_FALLBACK_UNSHARP == 0.0)
    print(f"[flags] pipeline_api exposes unsharp flags: {has_flags}; default OFF (==0.0): {off_default}")
    ok = ok and has_flags and off_default

    print(f"\nALL CHECKS PASSED: {ok}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
