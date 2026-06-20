"""R8-E3 propagation+tOF A/B (the one gap the experiment could not close).

E3 measured the beta=0.85 blend on the ANCHOR only (single-frame degrade-restore LPIPS).
This closes it: run the REAL quality pipeline (process_clip) twice -- beta=None (today's
full x4plus) vs beta=0.85 -- with GRAIN OFF (to isolate propagation flicker from grain),
decode both outputs, and compare temporal stability of the PROPAGATED result.

Decision: if beta=0.85 has tOF and |dF| <= beta=None, pre-blending the anchor does NOT add
temporal flicker (V6 predicts it REDUCES it: x4plus's hallucinated HF flickers more under
warp than compact, tOF 0.66 vs 0.33) -> beta=0.85 is safe to make the quality DEFAULT.
Else keep default-OFF.

Run:  python experiments/r8_e3_degrade_blend/propagation_ab.py <input.mp4> <n_frames>
"""
import os, sys, hashlib
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
import pipeline_api as pipe
import derisk

INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "server/testdata/short.mp4")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 32


def decode_mp4(path):
    import av
    frames = []
    c = av.open(path)
    for f in c.decode(video=0):
        frames.append(f.to_ndarray(format="rgb24"))
    c.close()
    return frames


def luma(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def mean_absdF(frames):
    """Mean consecutive |luma dF| over the whole frame. Grain OFF -> this is propagation flicker."""
    d = [float(np.mean(np.abs(luma(frames[t]) - luma(frames[t - 1])))) for t in range(1, len(frames))]
    return float(np.mean(d)), d


def run(beta, tag):
    # Isolate propagation: grain OFF + set/clear the blend beta. Restore after.
    cfg = pipe.MODE_CONFIG["quality"]
    g0, b0 = cfg["grain"], cfg.get("anchor_blend_beta")
    cfg["grain"] = "off"
    cfg["anchor_blend_beta"] = beta
    out = os.path.join(ROOT, "server/outputs", f"_ab_quality_{tag}.mp4")
    try:
        pipe.process_clip(INP, "quality", max_frames=N, out_path=out, detect_cuts=True)
    finally:
        cfg["grain"], cfg["anchor_blend_beta"] = g0, b0
    return out


def main():
    print(f"[propagation A/B] input={INP} N={N}  (quality, GRAIN OFF)")
    out_none = run(None, "betaNone")
    out_085 = run(0.85, "beta085")

    fr_none = decode_mp4(out_none)
    fr_085 = decode_mp4(out_085)
    n = min(len(fr_none), len(fr_085))
    fr_none, fr_085 = fr_none[:n], fr_085[:n]
    print(f"  decoded {n} HD frames each ({fr_none[0].shape[1]}x{fr_none[0].shape[0]})")

    # how much do the two outputs differ at all (sanity: the blend must actually change pixels)
    diff = np.mean([float(np.mean(np.abs(fr_none[i].astype(np.float32) - fr_085[i].astype(np.float32))))
                    for i in range(n)])
    print(f"  mean |beta085 - betaNone| over frames = {diff:.3f} codes  (0 => blend inert here)")

    # (1) whole-frame propagation flicker (grain off)
    df_none, _ = mean_absdF(fr_none)
    df_085, _ = mean_absdF(fr_085)
    print(f"\n  [|dF| whole-frame, grain off]   betaNone={df_none:.4f}   beta0.85={df_085:.4f}   "
          f"delta={df_085 - df_none:+.4f} ({100*(df_085-df_none)/df_none:+.1f}%)")

    # (2) flow-based tOF vs a bicubic-of-LR reference (TecoGAN EPE), at half-res for speed
    import av
    lr = []
    c = av.open(INP)
    for i, f in enumerate(c.decode(video=0)):
        if i >= n:
            break
        lr.append(f.to_ndarray(format="rgb24"))
    c.close()
    H, W = fr_none[0].shape[:2]
    hs, ws = H // 2, W // 2
    ref = [cv2.resize(x, (ws, hs), interpolation=cv2.INTER_CUBIC) for x in lr]
    s_none = [cv2.resize(x, (ws, hs), interpolation=cv2.INTER_AREA) for x in fr_none]
    s_085 = [cv2.resize(x, (ws, hs), interpolation=cv2.INTER_AREA) for x in fr_085]
    g = lambda L: [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in L]
    tof_none = derisk.tof(g(s_none), g(ref))
    tof_085 = derisk.tof(g(s_085), g(ref))
    print(f"  [tOF vs bicubic-LR ref, half-res]  betaNone={tof_none:.4f}   beta0.85={tof_085:.4f}   "
          f"delta={tof_085 - tof_none:+.4f} ({100*(tof_085-tof_none)/tof_none:+.1f}%)")

    print("\n  VERDICT:", "beta0.85 <= betaNone on BOTH -> propagation-safe, default-FLIP candidate"
          if (df_085 <= df_none + 1e-3 and tof_085 <= tof_none + 1e-3)
          else "beta0.85 adds flicker on >=1 metric -> keep default-OFF")


if __name__ == "__main__":
    main()
