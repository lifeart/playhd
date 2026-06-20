#!/usr/bin/env python3
"""R8-E2 step 5: unsharp-AMOUNT frontier -- joint full-reference (synthetic) + composite tOF
(codec-LR-through-warp), to find the knee and a recommended amount.

Per window, per amount a in [0, .25, .5, .75, 1.0] (a=0 == bicubic baseline):
  FULL-REF (synthetic /2 downscale, isolated fill): PSNR/SSIM/LPIPS vs HD_truth, dF.
  COMPOSITE (codec LR, through warp @ scale 2): tOF vs decoded LR, band-dF.
A near-free quality win = full-ref UP at tOF rise ~0; the knee tells us the best amount.

READ-ONLY imports of prototype/ + server/.
"""
import gc
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_REPO, "prototype"), os.path.join(_REPO, "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

import derisk as D    # noqa: E402
import sr as SR       # noqa: E402
import anchor_sr as A # noqa: E402
import torch          # noqa: E402
import lpips          # noqa: E402

CLIP = os.path.join(_REPO, "sample.mp4")
N = 40
OCC, SCALE = "reactive", 2
AMOUNTS = [0.0, 0.25, 0.5, 0.75, 1.0]
WINDOWS = [("A(0)", 0), ("H2(2352)", 2352)]
_LP = None


def lp():
    global _LP
    if _LP is None:
        _LP = lpips.LPIPS(net="alex", verbose=False)
    return _LP


def _free():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def unsharp(bic, a, sigma=1.0):
    if a <= 0:
        return bic
    blur = cv2.GaussianBlur(bic, (0, 0), sigma)
    return cv2.addWeighted(bic, 1.0 + a, blur, -a, 0)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k, s = (11, 11), 1.5
    ma, mb = cv2.GaussianBlur(a, k, s), cv2.GaussianBlur(b, k, s)
    ma2, mb2, mab = ma * ma, mb * mb, ma * mb
    sa = cv2.GaussianBlur(a * a, k, s) - ma2
    sb = cv2.GaussianBlur(b * b, k, s) - mb2
    sab = cv2.GaussianBlur(a * b, k, s) - mab
    return float((((2 * mab + C1) * (2 * sab + C2)) / ((ma2 + mb2 + C1) * (sa + sb + C2))).mean())


def lpd(a, b):
    def t(x):
        return torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        return float(lp()(t(a), t(b)).item())


def tof_vs_lr(recon, frames, w_lr, h_lr):
    sm = (w_lr, h_lr)
    seq = [cv2.resize(recon[i], sm) for i in range(len(frames))]
    lr = [frames[i][1] if frames[i][1].shape[1::-1] == sm else cv2.resize(frames[i][1], sm)
          for i in range(len(frames))]
    return D.tof(seq, lr)


def band_dF(recon, R, frames, w_lr, h_lr):
    sm = (w_lr, h_lr)
    seq = [cv2.resize(recon[i], sm).astype(np.float32) for i in range(len(frames))]
    masks = []
    for i in range(len(frames)):
        m = R[i].get("mask")
        if m is None:
            masks.append(np.zeros((h_lr, w_lr), bool))
        else:
            mm = m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
            masks.append(cv2.resize(mm.astype(np.uint8), sm, interpolation=cv2.INTER_NEAREST) > 0)
    vals = [float(np.abs(seq[t] - seq[t - 1])[(masks[t] | masks[t - 1])].mean())
            for t in range(1, len(frames)) if (masks[t] | masks[t - 1]).any()]
    return float(np.mean(vals)) if vals else 0.0


def main():
    SR.load_model("realesrgan")
    out = {"config": dict(N=N, amounts=AMOUNTS), "windows": {}}
    for label, start in WINDOWS:
        frames = D.decode_lr_and_mvs(CLIP, start, N)
        h_lr, w_lr = frames[0][1].shape[:2]
        w_hd, h_hd = w_lr * SCALE, h_lr * SCALE
        anchors, _ = A.anchor_indices(frames)

        # ---- FULL-REF (synthetic: codec frame = HD truth, /2 = LR) ----
        hd_truth = [f[1] for f in frames]
        syn_lr = [cv2.resize(t, (w_lr // 2, h_lr // 2), interpolation=cv2.INTER_AREA) for t in hd_truth]
        syn_bic = [cv2.resize(l, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC) for l in syn_lr]

        # ---- COMPOSITE (codec LR through warp) ----
        compact = D.build_perframe_cache(frames, w_hd, h_hd, "realesrgan")
        cod_bic = {i: cv2.resize(frames[i][1], (w_hd, h_hd), interpolation=cv2.INTER_CUBIC) for i in range(N)}

        rows = []
        for a in AMOUNTS:
            # full-ref
            ps = float(np.mean([psnr(unsharp(syn_bic[i], a), hd_truth[i]) for i in range(N)]))
            ss = float(np.mean([ssim(unsharp(syn_bic[i], a), hd_truth[i]) for i in range(N)]))
            lpv = float(np.mean([lpd(unsharp(syn_bic[i], a), hd_truth[i]) for i in range(N)]))
            # composite
            cache = {i: (compact[i] if i in anchors else unsharp(cod_bic[i], a)) for i in range(N)}
            _, R = D.reconstruct(frames, None, SCALE, True, OCC, cache, set(),
                                 backend="torch", collect_metrics=False, download_output=True)
            recon = {i: R[i]["recon"] for i in range(N)}
            tof = tof_vs_lr(recon, frames, w_lr, h_lr)
            bdf = band_dF(recon, R, frames, w_lr, h_lr)
            rows.append(dict(amount=a, psnr=round(ps, 3), ssim=round(ss, 4), lpips=round(lpv, 4),
                             tof=round(tof, 4), band_dF=round(bdf, 3)))
            del R, recon, cache
            _free()
        out["windows"][label] = rows

        b = rows[0]   # a=0 bicubic baseline
        print(f"\n=== {label}  anchors={sorted(anchors)} (deltas vs bicubic a=0) ===")
        print(f"   {'amt':>5}{'PSNR':>8}{'dPSNR':>7}{'SSIM':>8}{'LPIPS':>8}{'dLPIPS':>8}"
              f"{'tOF':>8}{'dtOF%':>7}{'bandDF':>8}")
        for r in rows:
            print(f"   {r['amount']:>5.2f}{r['psnr']:>8.2f}{r['psnr']-b['psnr']:>+7.2f}"
                  f"{r['ssim']:>8.4f}{r['lpips']:>8.4f}{r['lpips']-b['lpips']:>+8.4f}"
                  f"{r['tof']:>8.4f}{100*(r['tof']-b['tof'])/b['tof']:>+7.1f}{r['band_dF']:>8.2f}")
        del compact, cod_bic
        _free()

    with open(os.path.join(_HERE, "sweep.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(_HERE, 'sweep.json')}")


if __name__ == "__main__":
    main()
