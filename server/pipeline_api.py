"""playhd Stage-1 pipeline wrapper.

A thin, read-only wrapper around the validated prototype (`prototype/derisk.py` +
`sr.py` / `region_quality.py` / `grain.py`). It does NOT reimplement any of the
NEMO warp/mask/blend math -- it mirrors exactly what `derisk.run()` wires up,
pulls the reconstructed HD frames out of `R[]`, applies the mode-dependent output
passes (region-aware blend / film grain), and muxes the result to an H.264 mp4 via
PyAV at the source fps.

Two modes:
  * "instant" -- compact anchor (realesr-general-x4v3), backend=torch, occ=adaptive,
                 + per-frame film grain. The fast / real-time-style path.
  * "quality" -- heavy x4plus anchor (RealESRGAN_x4plus), region-aware detail blend,
                 + per-frame film grain. The slow / buffered path (~2.2 s/frame SR).

Public API:
    process_clip(input_path, mode, start_frame=5000, n_frames=48, out_path=...) -> out_path
"""

import os
import sys
import time
from fractions import Fraction

import numpy as np
import av

# --------------------------------------------------------------------------- #
# Import the prototype READ-ONLY. The prototype modules import each other by bare
# name (`import sr`, `import grain`, `import region_quality`), so the prototype dir
# must be on sys.path. `sr.py` resolves its weights via its own __file__, so cwd
# does not matter (weights already live in prototype/models/).
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_PROTO = os.path.join(_REPO, "prototype")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

import derisk            # noqa: E402  (the validated prototype)
import grain as _grain   # noqa: E402  (per-frame film-grain final pass)

SAMPLE_MP4 = os.path.join(_REPO, "sample.mp4")
OUTPUTS_DIR = os.path.join(_HERE, "outputs")
UPLOADS_DIR = os.path.join(_HERE, "uploads")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Every real SR net in sr.py is an x4 model -> the pipeline always runs at scale 4
# (matches the README: "--sr realesrgan ... it is an x4 net => use --scale 4").
SCALE = 4

# Mode -> exactly the flag combination the handoff recommends for each regime.
MODE_CONFIG = {
    "instant": dict(sr_mode="realesrgan",          # compact realesr-general-x4v3 (~0.13 s/frame)
                    backend="torch",               # MPS fast path (recon stays GPU-resident)
                    occ="adaptive",                # full quality both regimes, cheaper than full
                    region_aware=False,
                    grain="med",
                    label="Instant (compact anchor, torch/MPS, adaptive occ, grain)"),
    "quality": dict(sr_mode="realesrgan-x4plus",   # heavy RRDBNet x4plus (~2.2 s/frame, +61% sharper)
                    backend="torch",               # region-aware integration is tested on torch (out_region_e2e)
                    occ="adaptive",
                    region_aware=True,             # OUTPUT-only motion-gated heavy/compact blend
                    grain="med",
                    label="Quality (x4plus anchor, region-aware blend, grain)"),
}

# Stats from the most recent process_clip call (the server reads this to report timing).
LAST_STATS = {}


def list_sources():
    """Available source mp4s the UI can offer without an upload: the repo sample +
    anything dropped into server/uploads/. Returns a list of {name, path, size_mb}."""
    items = []
    if os.path.exists(SAMPLE_MP4):
        items.append({"name": "sample.mp4", "path": SAMPLE_MP4,
                      "size_mb": round(os.path.getsize(SAMPLE_MP4) / 1e6, 1)})
    for fn in sorted(os.listdir(UPLOADS_DIR)):
        if fn.lower().endswith(".mp4"):
            p = os.path.join(UPLOADS_DIR, fn)
            items.append({"name": fn, "path": p,
                          "size_mb": round(os.path.getsize(p) / 1e6, 1)})
    return items


def resolve_source(name):
    """Map a UI-supplied source name to a server-side path (sample or an upload).
    Raises ValueError for anything outside the two known dirs (no path traversal)."""
    if name in (None, "", "sample.mp4"):
        return SAMPLE_MP4
    base = os.path.basename(name)               # strip any path components
    cand = os.path.join(UPLOADS_DIR, base)
    if os.path.exists(cand):
        return cand
    if os.path.basename(SAMPLE_MP4) == base and os.path.exists(SAMPLE_MP4):
        return SAMPLE_MP4
    raise ValueError(f"unknown source {name!r}")


def _probe_fps(path):
    cont = av.open(path)
    try:
        vs = cont.streams.video[0]
        r = vs.average_rate or vs.base_rate or vs.guessed_rate
        return Fraction(r) if r else Fraction(25, 1)
    finally:
        cont.close()


def _encode_mp4(frames_rgb, out_path, fps):
    """Mux a list of HxWx3 uint8 RGB HD frames to an H.264 mp4 (PyAV / libx264,
    yuv420p) at `fps`. Mirrors make_synthetic()'s encode in derisk.py."""
    h_hd, w_hd = frames_rgb[0].shape[:2]
    cont = av.open(out_path, "w")
    st = cont.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w_hd, h_hd, "yuv420p"
    st.options = {"crf": "18"}
    for f in frames_rgb:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f, dtype=np.uint8), format="rgb24")
        for pkt in st.encode(vf):
            cont.mux(pkt)
    for pkt in st.encode():                      # flush
        cont.mux(pkt)
    cont.close()
    return w_hd, h_hd


def process_clip(input_path, mode, start_frame=5000, n_frames=48, out_path=None):
    """Run the existing pipeline on a [start_frame, start_frame+n_frames) window of
    `input_path` and ENCODE the upscaled HD frames to an mp4 at the source fps.

    Returns out_path. Timing/metadata for the run is also stashed in LAST_STATS.
    """
    if mode not in MODE_CONFIG:
        raise ValueError(f"unknown mode {mode!r}; choices: {list(MODE_CONFIG)}")
    cfg = MODE_CONFIG[mode]
    if out_path is None:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(OUTPUTS_DIR, f"{stem}_{mode}_{start_frame}_{n_frames}.mp4")

    t0 = time.perf_counter()
    fps = _probe_fps(input_path)

    # 1) Decode LR frames + per-frame motion vectors (display order) for the window.
    frames = derisk.decode_lr_and_mvs(input_path, start_frame=start_frame, max_frames=n_frames)
    if not frames:
        raise RuntimeError(f"no frames decoded from {input_path} at window "
                           f"[{start_frame}, {start_frame + n_frames})")
    t_decode = time.perf_counter() - t0

    h_lr, w_lr = frames[0][1].shape[:2]
    w_hd, h_hd = w_lr * SCALE, h_lr * SCALE

    # 2) Per-frame SR cache (the anchor / fallback / baseline source). For "quality"
    #    this is the heavy x4plus pass; for "instant" the compact net.
    t_sr0 = time.perf_counter()
    perframe_cache = derisk.build_perframe_cache(frames, w_hd, h_hd, cfg["sr_mode"])

    # 3) Region-aware gate (quality only) -- builds the temporally-stable motion gate
    #    + the per-frame COMPACT source for the OUTPUT-only blend. Reuses derisk's
    #    _build_region_gate (which reuses region_quality.py). None for instant.
    region_gate = (derisk._build_region_gate(frames, w_hd, h_hd, SCALE)
                   if cfg["region_aware"] else None)
    t_sr = time.perf_counter() - t_sr0

    # 4) Reconstruct. anchor_set=set() => I-frames-only backbone (the real-footage path
    #    used throughout the prototype). collect_metrics=False skips the SSIM/PSNR overhead
    #    (no ground truth here); download_output=True brings the HD recon back to numpy.
    #    The region-aware blend (if any) is applied INSIDE reconstruct to R[i]["recon"]
    #    as an OUTPUT-only pass -- never into the propagation reference chain.
    t_rec0 = time.perf_counter()
    _, R = derisk.reconstruct(
        frames, None, SCALE, True, cfg["occ"], perframe_cache, set(),
        backend=cfg["backend"], collect_metrics=False, download_output=True,
        region_gate=region_gate,
    )
    t_recon = time.perf_counter() - t_rec0

    # 5) Pull the HD recon out of R[] and apply the per-frame film-grain final pass
    #    (regenerated per frame index => temporally independent; never propagated).
    out_frames = []
    for i in range(len(frames)):
        recon = R[i]["recon"]
        if cfg["grain"] != "off":
            recon = _grain.apply_grain(recon, i, cfg["grain"])
        out_frames.append(np.ascontiguousarray(recon, dtype=np.uint8))

    # 6) Mux to mp4 at the source fps.
    t_enc0 = time.perf_counter()
    w_out, h_out = _encode_mp4(out_frames, out_path, fps)
    t_encode = time.perf_counter() - t_enc0

    total = time.perf_counter() - t0
    LAST_STATS.clear()
    LAST_STATS.update({
        "mode": mode, "label": cfg["label"], "input": input_path, "out_path": out_path,
        "start_frame": start_frame, "n_frames": len(frames),
        "fps": float(fps), "scale": SCALE,
        "src_resolution": f"{w_lr}x{h_lr}", "out_resolution": f"{w_out}x{h_out}",
        "t_decode_s": round(t_decode, 2), "t_sr_s": round(t_sr, 2),
        "t_recon_s": round(t_recon, 2), "t_encode_s": round(t_encode, 2),
        "t_total_s": round(total, 2),
        "ms_per_frame": round(total * 1000.0 / max(1, len(frames)), 1),
    })
    return out_path


# --------------------------------------------------------------------------- #
# CLI: `python3 server/pipeline_api.py instant|quality [--start N] [--n N]`
# Foreground verification helper (also re-decodes the output to confirm validity).
# --------------------------------------------------------------------------- #
def _verify_mp4(path, expect_n, expect_res):
    """Re-decode the produced mp4 with PyAV: confirm it is valid H.264 with the
    expected frame count + HD resolution. Returns (ok, info)."""
    cont = av.open(path)
    try:
        vs = cont.streams.video[0]
        codec = vs.codec_context.name
        w, h = vs.codec_context.width, vs.codec_context.height
        n = sum(1 for _ in cont.decode(vs))
    finally:
        cont.close()
    ok = (codec == "h264" and n == expect_n and f"{w}x{h}" == expect_res)
    return ok, {"codec": codec, "frames": n, "resolution": f"{w}x{h}"}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=list(MODE_CONFIG))
    ap.add_argument("--input", default=SAMPLE_MP4)
    ap.add_argument("--start", type=int, default=5000)
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()

    print(f"[pipeline] mode={a.mode}  input={a.input}  window=[{a.start},{a.start + a.n})")
    out = process_clip(a.input, a.mode, start_frame=a.start, n_frames=a.n)
    s = LAST_STATS
    print(f"[pipeline] wrote {out}")
    print(f"[pipeline] stats: {s}")
    ok, info = _verify_mp4(out, s["n_frames"], s["out_resolution"])
    print(f"[pipeline] re-decode check: ok={ok}  {info}")
    sys.exit(0 if ok else 2)
