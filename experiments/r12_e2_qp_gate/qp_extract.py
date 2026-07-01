#!/usr/bin/env python3
"""R12-E2: EXACT per-frame QP extraction from the H.264 bitstream.

PROBE RESULT (this env: ffmpeg 8.0.1 / PyAV 17.0.1 / libavcodec 62.11):
  Method (a) IN-PROCESS PyAV WORKS NATIVELY. Set the decoder option
  `export_side_data=venc_params`, then read frame.side_data[VIDEO_ENC_PARAMS].
  PyAV 17 exposes a full `VideoEncParams` object with:
      .qp                 -> frame/slice base QP (int)
      .qp_map()           -> numpy (mb_h, mb_w) int32 of ABSOLUTE per-macroblock QP
      .block_params(i)    -> per-block .delta_qp (fallback if qp_map absent)
  This is the exact per-MB QP the H.264 decoder computed (base_qp + adaptive
  delta), i.e. the true compression signal. No raw-struct unpack (PyAV #779
  workaround) is needed on this version.

Fallback (method c, pixels only): `dct_qp_estimate()` estimates QP from the
first-peak of the DCT-coefficient histogram (Qstep doubles every +6 QP). Used
only when venc_params is unavailable (e.g. a container/codec with no side data).

The public entry point is `qp_per_frame(path, max_frames)` -> list of dicts.
`mean_qp_stream(path)` -> a single scalar clip-level QP (median of per-frame means).
"""
import os
import numpy as np

try:
    import av
    from av.sidedata.sidedata import Type as _SDT
    _VENC = getattr(_SDT, "VIDEO_ENC_PARAMS", None)
except Exception:  # pragma: no cover
    av = None
    _VENC = None


# --------------------------------------------------------------------------- #
# Method (a): exact bitstream QP via venc_params side data
# --------------------------------------------------------------------------- #
def _frame_qp_stats(ep):
    """Reduce a VideoEncParams side-data object to per-frame QP stats.

    Prefers .qp_map() (a numpy (mb_h, mb_w) of ABSOLUTE per-MB QP); falls back to
    base .qp + per-block .delta_qp iteration; finally to the scalar base .qp."""
    # 1) qp_map(): absolute per-MB QP, fastest + most accurate
    try:
        qm = np.asarray(ep.qp_map())
        if qm.size:
            q = qm.astype(np.float64).ravel()
            return dict(base_qp=int(ep.qp), qp_mean=float(q.mean()),
                        qp_median=float(np.median(q)), qp_p90=float(np.percentile(q, 90)),
                        qp_min=float(q.min()), qp_max=float(q.max()),
                        nb_blocks=int(ep.nb_blocks), src="qp_map")
    except Exception:
        pass
    # 2) base_qp + per-block delta_qp
    try:
        nb = int(ep.nb_blocks)
        if nb:
            dq = np.fromiter((ep.block_params(i).delta_qp for i in range(nb)),
                             dtype=np.float64, count=nb)
            q = float(ep.qp) + dq
            return dict(base_qp=int(ep.qp), qp_mean=float(q.mean()),
                        qp_median=float(np.median(q)), qp_p90=float(np.percentile(q, 90)),
                        qp_min=float(q.min()), qp_max=float(q.max()),
                        nb_blocks=nb, src="block_params")
    except Exception:
        pass
    # 3) scalar base qp only
    bq = float(ep.qp)
    return dict(base_qp=int(ep.qp), qp_mean=bq, qp_median=bq, qp_p90=bq,
                qp_min=bq, qp_max=bq, nb_blocks=int(getattr(ep, "nb_blocks", 0)), src="base_qp")


def qp_per_frame(path, max_frames=None, want_rgb=False):
    """Yield-list of per-frame QP stats via bitstream venc_params (method a).

    Returns list of dicts: {frame, pict_type, base_qp, qp_mean, qp_median, qp_p90,
    qp_min, qp_max, nb_blocks, src, (optional) rgb}. Empty list if venc_params is
    unavailable (caller should fall back to dct_qp_estimate on the pixels)."""
    if av is None or _VENC is None:
        return []
    out = []
    cont = av.open(path)
    try:
        vs = cont.streams.video[0]
        vs.codec_context.options = {"export_side_data": "venc_params"}
        i = 0
        for frame in cont.decode(vs):
            if max_frames is not None and i >= max_frames:
                break
            ep = frame.side_data.get(_VENC)
            if ep is None:
                rec = dict(base_qp=None, qp_mean=None, qp_median=None, qp_p90=None,
                           qp_min=None, qp_max=None, nb_blocks=0, src="missing")
            else:
                rec = _frame_qp_stats(ep)
            rec["frame"] = i
            rec["pict_type"] = {1: "I", 2: "P", 3: "B"}.get(int(frame.pict_type), "?")
            if want_rgb:
                rec["rgb"] = frame.to_ndarray(format="rgb24")
            out.append(rec)
            i += 1
    finally:
        cont.close()
    return out


def mean_qp_stream(path, max_frames=None):
    """Clip-level scalar QP = median over frames of the per-frame mean per-MB QP.
    None if venc_params unavailable."""
    recs = qp_per_frame(path, max_frames=max_frames)
    vals = [r["qp_mean"] for r in recs if r.get("qp_mean") is not None]
    return float(np.median(vals)) if vals else None


# --------------------------------------------------------------------------- #
# Method (c): pixel-only DCT-histogram QP estimate (fallback / cross-check)
# --------------------------------------------------------------------------- #
def dct_qp_estimate(rgb):
    """Estimate H.264 QP from the luma 8x8-block DCT-coefficient quantization step.

    H.264's Qstep doubles every +6 in QP (Qstep(QP) = 0.625 * 2^(QP/6) for the AC
    scale used here). We measure the effective quantization step of the AC
    coefficients as the smallest non-zero magnitude they cluster onto, then invert.
    This is a coarse proxy (no bitstream), good to a few QP; used only when
    venc_params is unavailable."""
    import cv2
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = g.shape
    H8, W8 = (H // 8) * 8, (W // 8) * 8
    if H8 < 8 or W8 < 8:
        return None
    g = g[:H8, :W8]
    # gather AC coefficients over all 8x8 blocks
    acs = []
    for by in range(0, H8, 8):
        for bx in range(0, W8, 8):
            blk = g[by:by + 8, bx:bx + 8]
            d = cv2.dct(blk)
            a = np.abs(d.ravel()[1:])          # drop DC
            acs.append(a[a > 0.5])             # non-zero AC magnitudes
    if not acs:
        return None
    a = np.concatenate(acs)
    if a.size < 32:
        return None
    # effective Qstep ~ the modal spacing of small non-zero coeffs. Robust proxy:
    # the 20th percentile of the small-coeff magnitudes approximates one Qstep.
    small = a[a < np.percentile(a, 60)]
    if small.size < 16:
        small = a
    qstep = float(np.median(small))
    qstep = max(qstep, 0.5)
    # invert Qstep = 0.625 * 2^(QP/6)  ->  QP = 6*log2(qstep/0.625)
    qp = 6.0 * np.log2(qstep / 0.625)
    return float(np.clip(qp, 0, 51))


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..", "sample.mp4")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    recs = qp_per_frame(p, max_frames=n)
    if not recs:
        print("venc_params UNAVAILABLE -> would fall back to dct_qp_estimate")
    for r in recs:
        print(f"f{r['frame']:>3} {r['pict_type']} base={r['base_qp']} "
              f"mean={r['qp_mean']:.2f} med={r['qp_median']:.1f} "
              f"p90={r['qp_p90']:.1f} [{r['qp_min']:.0f}-{r['qp_max']:.0f}] "
              f"nb={r['nb_blocks']} src={r['src']}")
    print("clip median-of-means QP:", mean_qp_stream(p, max_frames=n))
