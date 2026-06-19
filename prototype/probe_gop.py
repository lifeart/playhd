#!/usr/bin/env python3
"""Probe real-clip GOP structure, MV source signs, |source| range, duplicate frames."""
import sys
import av
import numpy as np
from av.sidedata.sidedata import Type as SDType


def probe(path, start, n):
    cont = av.open(path)
    vs = cont.streams.video[0]
    vs.codec_context.options = {"flags2": "+export_mvs"}
    rows = []
    idx = 0
    prev_img = None
    for frame in cont.decode(vs):
        if idx < start:
            idx += 1
            continue
        if len(rows) >= n:
            break
        img = frame.to_ndarray(format="rgb24")
        try:
            sd = frame.side_data.get(SDType.MOTION_VECTORS)
        except Exception:
            sd = None
        mvs = sd.to_ndarray() if sd is not None else None
        ptype = {1: "I", 2: "P", 3: "B"}.get(int(frame.pict_type), "?")
        if mvs is None or len(mvs) == 0:
            npast = nfut = maxabs = 0
            nmv = 0
        else:
            src = mvs["source"].astype(int)
            npast = int((src < 0).sum())
            nfut = int((src > 0).sum())
            maxabs = int(np.abs(src).max())
            nmv = len(src)
        dup = (prev_img is not None and np.array_equal(prev_img, img))
        rows.append((idx, ptype, nmv, npast, nfut, maxabs, dup))
        prev_img = img
        idx += 1
    cont.close()
    return rows


if __name__ == "__main__":
    path = "/Users/lifeart/Repos/playhd/sample.mp4"
    for name, start in [("A", 0), ("B", 30), ("C", 5000)]:
        rows = probe(path, start, 48)
        types = "".join(r[1] for r in rows)
        print(f"\n=== window {name} (start={start}, n={len(rows)}) ===")
        print(f"types: {types}")
        bmaxabs = max([r[5] for r in rows if r[1] == "B"], default=0)
        pmaxabs = max([r[5] for r in rows if r[1] == "P"], default=0)
        print(f"max|source|  B={bmaxabs}  P={pmaxabs}")
        ndup = sum(1 for r in rows if r[6])
        print(f"duplicate-of-prev frames: {ndup}")
        print(f"{'idx':>5} {'t':>2} {'nmv':>6} {'past':>6} {'fut':>6} {'|s|max':>6} {'dup':>4}")
        for r in rows:
            mark = ""
            if r[1] == "B":
                tot = r[3] + r[4]
                fdom = (r[4] / tot * 100) if tot else 0
                mark = f"  fut%={fdom:4.0f}"
            print(f"{r[0]:>5} {r[1]:>2} {r[2]:>6} {r[3]:>6} {r[4]:>6} {r[5]:>6} {str(r[6]):>5}{mark}")
