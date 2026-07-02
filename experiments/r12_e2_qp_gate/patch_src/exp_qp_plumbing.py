#!/usr/bin/env python3
"""R12-E2 EXPERIMENT SHIM (not the production patch — see stream_gops_qp.py + INTEGRATION.md
for the proposed server change). Lets the graduation A/B feed EXACT bitstream QP into the
UNMODIFIED derisk.reconstruct path.

Why a shim: R10 anticipated carrying QP as frame-tuple index [3] (build_perframe_cache reads
`frames[i][3]`), but the reconstruct path still does rigid 3-tuple unpacks
(prototype/derisk.py:409 `for _, _, mvs in frames`), so a real 4-tuple crashes reconstruct.
The clean production fix (star-unpack those 2 sites) is documented in INTEGRATION.md; for the
EXPERIMENT we keep 3-tuples and pass QP OUT-OF-BAND via an id(img)->qp map, so we test the GATE
BEHAVIOUR through the real pipeline without editing prototype/ or server/.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import av
from av.sidedata.sidedata import Type as _SDT

_VENC = getattr(_SDT, "VIDEO_ENC_PARAMS", None)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
import pipeline_api as pipe
import derisk

# id(img_ndarray) -> exact per-frame mean per-MB QP (populated by the patched stream_gops,
# consumed by the patched build_perframe_cache in the same run).
QP_BY_IMG_ID = {}


def _frame_qp_mean(frame):
    """Exact per-frame QP = mean of the absolute per-MB QP map (venc_params). None if absent."""
    if _VENC is None:
        return None
    ep = frame.side_data.get(_VENC)
    if ep is None:
        return None
    try:
        return float(np.asarray(ep.qp_map()).mean())
    except Exception:
        try:
            return float(ep.qp)
        except Exception:
            return None


def patched_stream_gops(path, max_frames=None, soft_cap=pipe.SOFT_CAP_FRAMES, detect_cuts=True):
    """Copy of pipeline_api.stream_gops + venc_params QP extraction. Still yields 3-tuples
    (ptype, img, mvs) so reconstruct is untouched; the exact QP is stashed in QP_BY_IMG_ID."""
    cont = av.open(path)
    try:
        vs = cont.streams.video[0]
        vs.codec_context.options = {"flags2": "+export_mvs", "export_side_data": "venc_params"}
        det = pipe.scene_detect.StreamingCutDetector() if detect_cuts else None
        chunk, produced = [], 0
        for frame in cont.decode(vs):
            if max_frames is not None and produced >= max_frames:
                break
            ptype = {1: "I", 2: "P", 3: "B"}.get(int(frame.pict_type), "?")
            img = frame.to_ndarray(format="rgb24")
            QP_BY_IMG_ID[id(img)] = _frame_qp_mean(frame)     # exact bitstream QP for this frame
            is_cut = det.update(produced, ptype, img) if det is not None else False
            if chunk and (ptype == "I" or is_cut
                          or (len(chunk) >= soft_cap and ptype == "P")):
                yield chunk
                chunk = []
            try:
                sd = frame.side_data.get(derisk.SDType.MOTION_VECTORS)
            except Exception:
                sd = None
            mvs = sd.to_ndarray() if sd is not None else None
            chunk.append((ptype, img, mvs))
            produced += 1
        if chunk:
            yield chunk
    finally:
        cont.close()


def patched_build_perframe_cache(frames, w_hd, h_hd, sr_mode, half=False, deblock_cfg=None):
    """Copy of derisk.build_perframe_cache but sourcing the deblock gate's QP from the exact
    bitstream QP map (QP_BY_IMG_ID[id(lr)]) instead of the never-wired tuple index [3]."""
    _pre = None
    if deblock_cfg:
        _dbdir = os.path.join(ROOT, "experiments", "r10_e2_deblock_pre")
        if _dbdir not in sys.path:
            sys.path.insert(0, _dbdir)
        import deblock_pre as _pre
    N = len(frames)
    cache = {}
    if sr_mode in ("realesrgan", "realesrgan-x4plus"):
        import sr as _srmod
        _srmod.load_model(sr_mode, half=half)
        _srmod.upscale(frames[0][1], model=sr_mode, half=half)
        _srmod.reset_latency(sr_mode)
        for i in range(N):
            derisk.PROF.ftype, derisk.PROF.fidx = frames[i][0], i
            with derisk.PROF.time("sr"):
                lr_i = frames[i][1]
                if _pre is not None:
                    qp_i = QP_BY_IMG_ID.get(id(lr_i))          # EXACT bitstream QP (method a)
                    lr_i = _pre.apply(lr_i, deblock_cfg, qp=qp_i)
                cache[i] = _srmod.upscale_to(lr_i, w_hd, h_hd, model=sr_mode, half=half)
        return cache
    for i in range(N):
        derisk.PROF.ftype, derisk.PROF.fidx = frames[i][0], i
        with derisk.PROF.time("sr"):
            cache[i] = cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
    return cache


def install():
    """Monkeypatch the pipeline to use the QP-carrying stream_gops + build_perframe_cache."""
    pipe.stream_gops = patched_stream_gops
    derisk.build_perframe_cache = patched_build_perframe_cache


def deblock_fire_count():
    """Count how many cached frames actually deblocked (for reporting). Not used by A/B score."""
    return None
