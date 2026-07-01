#!/usr/bin/env python3
"""
r12_e4 -- load a trained SPAN-codec checkpoint, run it on a HELD-OUT real-libx264 LR
frame, and report perceptual metrics vs the HR target + a bicubic baseline.

Proves: (a) the checkpoint LOADS (both via the raw SPAN class AND via spandrel's
ModelLoader -> so it is a drop-in for web_spike/export_span_weights.py), (b) it RUNS,
(c) output is sane (before/after crop + LPIPS/DISTS/VMAF-NEG vs HR, and vs bicubic).

NOTE: with a smoke-train checkpoint these numbers prove the HARNESS, not quality.
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "r8_e4_metric_triangulation"))
import degrade as DG
import metrics_extra as MX  # dists + lpips_pyiqa + vmaf_neg (reused arbiters)

from spandrel.architectures.SPAN.__arch.span import SPAN
from spandrel import ModelLoader

import lpips as _lpips_pkg


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_raw(ckpt, scale, feat, dev):
    net = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=feat, upscale=scale, norm=False)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "params" in sd:
        sd = sd["params"]
    missing, unexpected = net.load_state_dict(sd, strict=True)
    return net.eval().to(dev)


def run(net, lr_u8, dev):
    t = torch.from_numpy(lr_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        out = net(t).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (out * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_HERE, "out", "span_codec.pth"))
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--feat", type=int, default=48)
    ap.add_argument("--holdout", default=os.path.join(_HERE, "out", "holdout"))
    ap.add_argument("--src", default=os.path.join(_ROOT, "web_spike", "sd600.mp4"))
    ap.add_argument("--idx", type=int, default=311, help="fallback held-out frame if no holdout dir")
    ap.add_argument("--out-json", default=os.path.join(_HERE, "out", "eval.json"))
    args = ap.parse_args()
    dev = pick_device()

    # ---- (a) load via raw SPAN class ----
    net = load_raw(args.ckpt, args.scale, args.feat, dev)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"[load] raw SPAN class: OK  ({nparams/1e6:.3f}M params) on {dev}")

    # ---- (a') load via spandrel ModelLoader (drop-in proof) ----
    try:
        desc = ModelLoader().load_from_file(args.ckpt)
        print(f"[load] spandrel ModelLoader: OK  -> {type(desc.model).__name__} "
              f"scale={getattr(desc,'scale','?')}")
    except Exception as e:
        print(f"[load] spandrel ModelLoader: FAILED -> {e!r}  (raw-class load still valid)")

    # ---- held-out pair (real libx264 LR the trainer never saw) ----
    hpath = os.path.join(args.holdout, "0_hr.png")
    if os.path.exists(hpath):
        hr = cv2.cvtColor(cv2.imread(hpath), cv2.COLOR_BGR2RGB)
        lr = cv2.cvtColor(cv2.imread(os.path.join(args.holdout, "0_lr.png")), cv2.COLOR_BGR2RGB)
        print(f"[data] holdout pair from {args.holdout}")
    else:
        hr = DG.decode_frames(args.src, [args.idx])[0]
        lr, meta = DG.degrade_frame(hr, crf=32, preset="medium", scale=args.scale)
        print(f"[data] fresh held-out frame {args.idx} crf32")
    H, W = hr.shape[:2]

    sr = run(net, lr, dev)                                   # SPAN-codec output
    bic = cv2.resize(lr, (W, H), interpolation=cv2.INTER_CUBIC)  # baseline
    print(f"[run] LR {lr.shape[:2]} -> SR {sr.shape[:2]} (target HR {hr.shape[:2]})")

    # ---- metrics: SR vs HR, bicubic vs HR (lower LPIPS/DISTS better; higher VMAF-NEG better) ----
    lp = _lpips_pkg.LPIPS(net="alex", verbose=False)

    def lpips_a(a, b):
        ta = torch.from_numpy(a.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        tb = torch.from_numpy(b.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        with torch.no_grad():
            return float(lp(ta, tb).item())

    vmaf_ok = MX.vmaf_available()
    res = {}
    for nm, img in [("span_codec", sr), ("bicubic", bic)]:
        row = dict(lpips=lpips_a(img, hr), dists=MX.dists(img, hr),
                   psnr=float(cv2.PSNR(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                       cv2.cvtColor(hr, cv2.COLOR_RGB2BGR))))
        if vmaf_ok:
            row["vmaf_neg"] = MX.vmaf_neg_single(hr, img)
        res[nm] = row
        vn = f" VMAF-NEG={row.get('vmaf_neg', float('nan')):.2f}" if vmaf_ok else ""
        print(f"[metric] {nm:11s} LPIPS={row['lpips']:.4f}  DISTS={row['dists']:.4f}  "
              f"PSNR={row['psnr']:.2f}dB{vn}")

    dl = res["span_codec"]["lpips"] - res["bicubic"]["lpips"]
    dd = res["span_codec"]["dists"] - res["bicubic"]["dists"]
    print(f"[delta] span_codec vs bicubic: dLPIPS={dl:+.4f} dDISTS={dd:+.4f} "
          f"({'SR better' if dl < 0 else 'bicubic better'} on LPIPS)")

    # ---- before/after crop figure ----
    cs = 80
    cx, cy = W // 2 - cs, H // 2 - cs
    tiles = []
    for img in [bic, sr, hr]:
        cr = img[cy:cy + 2 * cs, cx:cx + 2 * cs]
        tiles.append(cr)
    strip = np.concatenate(tiles, axis=1)  # bicubic | span_codec | HR-target
    fig = os.path.join(_HERE, "out", "eval_before_after.png")
    cv2.imwrite(fig, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"[fig] crop [bicubic | span_codec | HR] -> {fig}")

    json.dump(dict(metrics=res, dlpips=dl, ddists=dd, vmaf=vmaf_ok, params=nparams),
              open(args.out_json, "w"), indent=1)
    print(f"[json] -> {args.out_json}")


if __name__ == "__main__":
    main()
