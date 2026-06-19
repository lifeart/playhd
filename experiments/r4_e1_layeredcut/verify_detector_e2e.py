"""R4-E1 fix (b) END-TO-END: prove the chroma-dominant cut detector kills the layered corruption
at the ROOT (per-scene plates) on a clip where the post-cut scene is long enough to NOT be merged
by segment_scenes' short-trailing-scene rule (the 40-frame c7's 12-frame tail IS merged, which is
why c7 needs the guard).

Builds a 72-frame TWO-STATIC-SCENE clip: scene A [0,32) warm texture, hard cut to scene B [32,72)
that is LUMA-MATCHED to A (small dLuma) but strongly RE-HUED (large dChroma) -- the similar-luma
missed-cut signature -- both static (small moving subject). Encoded sc_threshold=0/bf=0 so the cut
lands on a P-frame (no codec I-frame to save us).

Then runs the REAL pipeline_api.process_clip('layered') twice:
  BASELINE  shipped scene_detect (luma-only)         -> expect 1 scene -> corruption in scene B.
  FIXED(b)  monkeypatch find_cuts w/ chroma-dominant -> expect 2 scenes -> per-scene plate, no corruption.

Run:  python3 experiments/r4_e1_layeredcut/verify_detector_e2e.py
"""
import os
import sys
import gc

import numpy as np
import cv2
import av

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "server"))
sys.path.insert(0, os.path.join(REPO, "prototype"))

import pipeline_api as P          # noqa: E402
import scene_detect as SD         # noqa: E402
import derisk                     # noqa: E402
import cutdetect_chroma as CC     # noqa: E402  (the proposed chroma-dominant detector)

try:
    import torch as _torch
except Exception:
    _torch = None

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
CLIP = os.path.join(HERE, "ext_staticcut.mp4")
CUT = 32
N = 72


def _free_gpu():
    gc.collect()
    if _torch is not None:
        try:
            if _torch.backends.mps.is_available():
                _torch.mps.empty_cache()
        except Exception:
            pass


def _texture(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c = np.stack([128 + 90 * np.sin(xx / 19.0 + seed),
                  128 + 90 * np.sin(yy / 13.0 + 1.0 + seed),
                  128 + 90 * np.sin((xx + yy) / 27.0 + 2.0 + seed)], -1)
    c = np.clip(c, 0, 255).astype(np.uint8)
    for _ in range(50):
        col = tuple(int(v) for v in rng.integers(40, 220, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(c, (x, y), int(rng.integers(8, 24)), col, -1, cv2.LINE_AA)
    return c


def _luma_match(img, target_mean):
    """Scale RGB so luma mean matches target (keeps hue, shifts brightness) -> small dLuma."""
    f = img.astype(np.float32)
    Y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    g = target_mean / max(1.0, float(Y.mean()))
    return np.clip(f * g, 0, 255).astype(np.uint8)


def build_clip():
    h, w = 272, 480
    A = _texture(h, w, seed=3)                                   # warm-ish texture
    B = _texture(h, w, seed=40)
    B = B[..., [2, 0, 1]].copy()                                 # rotate channels -> different hue
    # SIMILAR-LUMA, DIFFERENT-CHROMA cut: give B EXACTLY A's luma plane (so per-pixel dLuma ~ 0 at
    # the cut) but keep B's chroma -> the missed-cut signature the luma detector cannot see.
    Ayuv = cv2.cvtColor(A, cv2.COLOR_RGB2YUV)
    Byuv = cv2.cvtColor(B, cv2.COLOR_RGB2YUV)
    Byuv[..., 0] = Ayuv[..., 0]
    B = cv2.cvtColor(Byuv, cv2.COLOR_YUV2RGB)
    frames = []
    for i in range(N):
        if i < CUT:
            f = A.copy()
            cv2.circle(f, (60 + (i * 5) % (w - 120), 80), 20, (235, 70, 70), -1, cv2.LINE_AA)
        else:
            f = B.copy()
            j = i - CUT
            cv2.circle(f, (90 + (j * 5) % (w - 160), 150), 20, (70, 235, 90), -1, cv2.LINE_AA)
        frames.append(f)
    cont = av.open(CLIP, "w")
    st = cont.add_stream("libx264", rate=25)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": "18", "sc_threshold": "0", "g": str(N + 10), "bf": "0"}
    for f in frames:
        for p in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            cont.mux(p)
    for p in st.encode():
        cont.mux(p)
    cont.close()


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


def lrc_series(out_path, clip_path, n):
    out_hd = decode_rgb(out_path)
    lr_in = decode_rgb(clip_path, max_frames=n)
    m = min(len(out_hd), len(lr_in))
    return [round(derisk.psnr_lr_consistency(out_hd[i], lr_in[i]), 2) for i in range(m)]


def run_layered(tag, chroma_detector):
    orig = SD.find_cuts
    if chroma_detector:
        def patched(path, max_frames=None, **kw):
            cuts, _f, total = CC.detect(path, max_frames=max_frames, with_chroma=True)
            return cuts, total
        SD.find_cuts = patched
    out_path = os.path.join(OUT, f"ext_{tag}.mp4")
    try:
        P.process_clip(CLIP, "layered", max_frames=N, out_path=out_path)
        stats = dict(P.LAST_STATS)
    finally:
        P.end_job()
        SD.find_cuts = orig
        _free_gpu()
    lrc = lrc_series(out_path, CLIP, N)
    return stats, lrc


if __name__ == "__main__":
    build_clip()
    # characterize the authored cut
    import calibrate
    rows = {r[0]: r for r in calibrate.per_frame_signals(CLIP)}
    idx, pt, dL, dC, dE = rows[CUT]
    print(f"authored cut @ {CUT}: ptype={pt} dLuma={dL:.1f} dChroma={dC:.1f} "
          f"(chroma-dominant={dC > dL})")
    base_cuts, _ = SD.find_cuts(CLIP, max_frames=N)
    chr_cuts, _f, _t = CC.detect(CLIP, max_frames=N, with_chroma=True)
    print(f"shipped luma detector cuts={base_cuts}   chroma detector cuts={chr_cuts}")

    print("\n--- BASELINE (shipped luma-only detector) ---")
    s0, l0 = run_layered("baseline", chroma_detector=False)
    print(f"  n_scenes={s0.get('n_scenes')} verdicts={s0.get('scene_verdicts')} "
          f"fallback={s0.get('fallback_scenes')}")
    print(f"  LRC scene-A[0,{CUT}) mean={np.mean(l0[:CUT]):.2f}  "
          f"scene-B[{CUT},{N}) mean={np.mean(l0[CUT:]):.2f} min={min(l0[CUT:]):.2f}")

    print("\n--- FIXED (b) (chroma-dominant detector) ---")
    s1, l1 = run_layered("fixedb", chroma_detector=True)
    print(f"  n_scenes={s1.get('n_scenes')} verdicts={s1.get('scene_verdicts')} "
          f"fallback={s1.get('fallback_scenes')}")
    print(f"  LRC scene-A[0,{CUT}) mean={np.mean(l1[:CUT]):.2f}  "
          f"scene-B[{CUT},{N}) mean={np.mean(l1[CUT:]):.2f} min={min(l1[CUT:]):.2f}")

    print("\n===== fix (b) end-to-end =====")
    print(f"scene-B LRC min: BASELINE={min(l0[CUT:]):.2f}  FIXED(b)={min(l1[CUT:]):.2f}")
