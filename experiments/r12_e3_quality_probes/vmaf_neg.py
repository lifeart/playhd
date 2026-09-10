#!/usr/bin/env python3
"""
Reusable VMAF-NEG helper for playhd A/B eval harnesses (research item #7).

WHY: LPIPS/DISTS rank OVERALL perceptual similarity well on compressed content but are
largely BLIND to hallucination -- a sharpening/detail-inventing model can "win" LPIPS/DISTS
while fabricating structure that was never in the source. VMAF-NEG (No Enhancement Gain) is
the standard anti-cheat: it clips the VMAF gain terms that reward artificial sharpening, so a
model that hallucinates crisp-but-wrong detail scores LOWER on NEG than its plain VMAF.

USE IT ONLY AS A GUARDRAIL COLUMN alongside LPIPS+DISTS+PSNR. Do NOT optimise against it --
VMAF (and NEG) is itself gameable (~+22% reported), so it is a sanity check, never the target.

Backend: ffmpeg's built-in `libvmaf` filter with `model=version=vmaf_v0.6.1neg` (the NEG model
ships inside recent ffmpeg/libvmaf -- no extra download). Ref and dist frames are written as
LOSSLESS FFV1 so the metric sees ONLY the model's output, not an extra codec pass.

API:
    vmaf_neg(ref_frames, dist_frames) -> float          # list[HxWx3 uint8 RGB] each; pooled mean
    vmaf_neg_single(ref_rgb, dist_rgb, reps=3) -> float # one frame (duplicated for the filter)
    available() -> bool                                 # ffmpeg + neg model present?

Returns VMAF-NEG in [0,100] (higher = closer to reference). NaN if the backend is unavailable
so callers can degrade gracefully to LPIPS/DISTS/PSNR-only.
"""
import os
import json
import shutil
import subprocess
import tempfile

import numpy as np
import cv2

_FFMPEG = shutil.which("ffmpeg")
_MODEL = "version=vmaf_v0.6.1neg"
_AVAIL = None


def _write_ffv1(frames, path):
    """Write a list of HxWx3 uint8 RGB frames to a LOSSLESS FFV1 .mkv (no added codec noise)."""
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(
        [_FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "25", "-i", "-",
         "-c:v", "ffv1", "-pix_fmt", "yuv444p", path],
        stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2BGR).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffv1 encode failed")


def available():
    """True iff ffmpeg exists and the NEG model runs (probed once, cached)."""
    global _AVAIL
    if _AVAIL is not None:
        return _AVAIL
    if _FFMPEG is None:
        _AVAIL = False
        return _AVAIL
    try:
        a = (np.random.default_rng(0).integers(0, 256, (64, 64, 3))).astype(np.uint8)
        v = vmaf_neg([a, a, a], [a, a, a])
        _AVAIL = not np.isnan(v)
    except Exception:
        _AVAIL = False
    return _AVAIL


def vmaf_neg(ref_frames, dist_frames):
    """VMAF-NEG (pooled mean) of dist vs ref. Each arg is a list of HxWx3 uint8 RGB frames of
    equal, matching size. Returns float in [0,100], or NaN if the backend is unavailable."""
    if _FFMPEG is None:
        return float("nan")
    if not ref_frames or len(ref_frames) != len(dist_frames):
        raise ValueError("ref/dist frame lists must be non-empty and equal length")
    d = tempfile.mkdtemp(prefix="vmafneg_")
    try:
        ref_p = os.path.join(d, "ref.mkv")
        dis_p = os.path.join(d, "dis.mkv")
        log_p = os.path.join(d, "neg.json")
        _write_ffv1(ref_frames, ref_p)
        _write_ffv1(dist_frames, dis_p)
        # dist is FIRST input, ref is SECOND (libvmaf convention: [dist][ref]).
        lav = (f"[0:v]setpts=PTS-STARTPTS[d];[1:v]setpts=PTS-STARTPTS[r];"
               f"[d][r]libvmaf=model={_MODEL}:log_fmt=json:log_path={log_p}:shortest=1")
        r = subprocess.run(
            [_FFMPEG, "-hide_banner", "-loglevel", "error",
             "-i", dis_p, "-i", ref_p, "-lavfi", lav, "-f", "null", "-"],
            capture_output=True, text=True)
        if not os.path.exists(log_p):
            raise RuntimeError(f"libvmaf produced no log: {r.stderr.strip()[:300]}")
        js = json.load(open(log_p))
        return float(js["pooled_metrics"]["vmaf"]["mean"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def vmaf_neg_single(ref_rgb, dist_rgb, reps=3):
    """VMAF-NEG for a single frame pair (duplicated `reps` times so the filter has frames to
    pool; temporal/motion terms are inert but the spatial NEG comparison is what we want)."""
    return vmaf_neg([ref_rgb] * reps, [dist_rgb] * reps)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ref = (rng.integers(0, 256, (128, 128, 3))).astype(np.uint8)
    print("available:", available())
    # identical -> ~100 ; blurred (detail lost) -> lower ; note NEG does not reward re-sharpening
    blur = cv2.GaussianBlur(ref, (0, 0), 1.5)
    print("VMAF-NEG identical:", round(vmaf_neg_single(ref, ref), 3))
    print("VMAF-NEG blurred  :", round(vmaf_neg_single(ref, blur), 3))
