#!/usr/bin/env python3
"""
r12_e4 -- minimal but REAL trainer for a codec-matched SR model (SPAN-arch, Apache-2.0).

Trains the SPAN *architecture* (spandrel's `SPAN`, the same arch as the non-commercial
`2xLiveActionV1_SPAN` weights, but a FRESH commercially-clean init) on HR<->LR pairs whose
LR is a REAL libx264 encode->decode round-trip (see degrade.py). Output is a drop-in
replacement for the current non-commercial demo weights: same arch, same scale (x2),
same norm=False -> loads unchanged in web_spike/export_span_weights.py.

Loss = L1 + lambda * LPIPS(alex)   (pixel fidelity + a perceptual term; LPIPS is one of the
project's arbiters, r8_e4/r12_e3). NO GAN term (GAN is what hallucinates -> the fake-detail
trap; a codec-matched L1+LPIPS net is the honest StreamSR-style recipe).

MPS-aware. Conv3XC reparam: the model is kept in .train() so the multi-branch conv path is
used; at inference .eval() collapses each Conv3XC to a single 3x3 (eval_conv) automatically.

Config is fully CLI-driven. The SAME command scales from the smoke-train to the full run
(just raise --frames/--iters). See --help.

Full-run launch command (documented in the report) is just this script with production args.
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import degrade as DG  # decode + REAL libx264 degradation

from spandrel.architectures.SPAN.__arch.span import SPAN  # Apache-2.0 architecture

try:
    import lpips as _lpips_pkg
except Exception as e:  # loud: the perceptual term is not optional in this recipe
    raise RuntimeError(f"lpips package required for the perceptual loss term: {e!r}")


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_span(scale=2, feat=48, norm=False):
    """Fresh SPAN, matched to 2xLiveActionV1_SPAN (feat=48, norm=False) but random init."""
    return SPAN(num_in_ch=3, num_out_ch=3, feature_channels=feat, upscale=scale, norm=norm)


def frames_to_tensor(frames):
    """list HxWx3 uint8 -> (N,3,H,W) float32 [0,1] on CPU."""
    arr = np.stack([f for f in frames]).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def sample_patch_batch(hr, lr, patch_hr, scale, batch, device, rng):
    """Random aligned HR/LR patch batch + shared flip/rot augmentation.
    hr:(N,3,H,W) lr:(N,3,h,w) with H=h*scale. patch_hr divisible by scale."""
    n = hr.shape[0]
    p_lr = patch_hr // scale
    _, _, h, w = lr.shape
    hb, lb = [], []
    for _ in range(batch):
        i = rng.randrange(n)
        lx = rng.randrange(0, w - p_lr + 1)
        ly = rng.randrange(0, h - p_lr + 1)
        lp = lr[i, :, ly:ly + p_lr, lx:lx + p_lr]
        hp = hr[i, :, ly * scale:ly * scale + patch_hr, lx * scale:lx * scale + patch_hr]
        # shared augmentation
        if rng.random() < 0.5:
            lp = torch.flip(lp, [2]); hp = torch.flip(hp, [2])
        if rng.random() < 0.5:
            lp = torch.flip(lp, [1]); hp = torch.flip(hp, [1])
        if rng.random() < 0.5:
            lp = torch.rot90(lp, 1, [1, 2]); hp = torch.rot90(hp, 1, [1, 2])
        lb.append(lp); hb.append(hp)
    return (torch.stack(lb).to(device), torch.stack(hb).to(device))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(_ROOT, "web_spike", "sd600.mp4"),
                    help="HR video source (talking-head). frames are the HR target.")
    ap.add_argument("--frames", type=int, default=24, help="# HR frames to sample")
    ap.add_argument("--stride", type=int, default=23, help="frame stride when sampling")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--feat", type=int, default=48)
    ap.add_argument("--patch", type=int, default=96, help="HR patch size (div by scale)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr", type=float, default=5e-5)
    ap.add_argument("--lpips-weight", type=float, default=0.5)
    ap.add_argument("--gop", type=int, default=1, help="1=all-intra (anchor-matched)")
    ap.add_argument("--holdout", type=int, default=2, help="# frames held out of training")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_HERE, "out", "span_codec.pth"))
    ap.add_argument("--loss-json", default=os.path.join(_HERE, "out", "loss.json"))
    args = ap.parse_args()

    assert args.patch % args.scale == 0, "patch must be divisible by scale"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = pick_device()
    print(f"# device={dev} src={os.path.basename(args.src)} frames={args.frames} "
          f"scale=x{args.scale} feat={args.feat} patch={args.patch} batch={args.batch} "
          f"iters={args.iters} lr={args.lr} lpips_w={args.lpips_weight}")

    # ---- data: decode HR frames, build REAL-libx264 LR pairs (once) ----
    t0 = time.perf_counter()
    hr_frames = DG.decode_n(args.src, args.frames, stride=args.stride, start=args.start)
    pairs = DG.build_pairs(hr_frames, scale=args.scale, seed=args.seed, gop=args.gop)
    n = len(pairs)
    hold = min(args.holdout, max(0, n - 1))
    tr = pairs[:n - hold] if hold else pairs
    hr = frames_to_tensor([p[0] for p in tr])
    lr = frames_to_tensor([p[1] for p in tr])
    crfs = [p[2]["crf"] for p in tr]
    print(f"# built {n} pairs ({len(tr)} train / {hold} holdout) in {time.perf_counter()-t0:.1f}s; "
          f"HR {tuple(hr.shape[2:])} LR {tuple(lr.shape[2:])} crf∈[{min(crfs)},{max(crfs)}]")
    # persist holdout for the eval script (byte-exact same LR the trainer never saw)
    if hold:
        hd = os.path.join(_HERE, "out", "holdout")
        os.makedirs(hd, exist_ok=True)
        import cv2
        for j, p in enumerate(pairs[n - hold:]):
            cv2.imwrite(os.path.join(hd, f"{j}_hr.png"), cv2.cvtColor(p[0], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(hd, f"{j}_lr.png"), cv2.cvtColor(p[1], cv2.COLOR_RGB2BGR))

    # ---- model + loss + opt ----
    net = build_span(args.scale, args.feat, norm=False).to(dev).train()
    nparams = sum(p.numel() for p in net.parameters())
    print(f"# SPAN(feat={args.feat}, x{args.scale}, norm=False): {nparams/1e6:.3f}M params")
    lpips_fn = _lpips_pkg.LPIPS(net="alex", verbose=False).to(dev)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=args.min_lr)

    traj = []
    ema = None
    t0 = time.perf_counter()
    for it in range(1, args.iters + 1):
        lp_b, hp_b = sample_patch_batch(hr, lr, args.patch, args.scale, args.batch, dev, rng)
        sr = net(lp_b).clamp(0.0, 1.0)
        l1 = F.l1_loss(sr, hp_b)
        # LPIPS wants [-1,1]
        perc = lpips_fn(sr * 2 - 1, hp_b * 2 - 1).mean()
        loss = l1 + args.lpips_weight * perc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        lv = float(loss.detach().cpu())
        ema = lv if ema is None else 0.9 * ema + 0.1 * lv
        if it % args.log_every == 0 or it == 1:
            print(f"  it {it:4d}/{args.iters}  loss={lv:.4f} (ema {ema:.4f})  "
                  f"l1={float(l1):.4f} lpips={float(perc):.4f}  lr={sched.get_last_lr()[0]:.2e}")
            traj.append(dict(it=it, loss=lv, ema=ema, l1=float(l1), lpips=float(perc)))
    dt = time.perf_counter() - t0
    print(f"# trained {args.iters} iters in {dt:.1f}s ({1000*dt/args.iters:.1f} ms/iter)")

    # ---- save: bare SPAN state_dict (drop-in for spandrel / export_span_weights) ----
    net.eval()  # collapse Conv3XC so eval_conv in the saved dict is consistent too
    with torch.no_grad():
        for m in net.modules():
            if hasattr(m, "update_params"):
                m.update_params()
    torch.save(net.state_dict(), args.out)
    meta = dict(arch="SPAN", feat=args.feat, scale=args.scale, norm=False,
                params=nparams, iters=args.iters, lr=args.lr, lpips_weight=args.lpips_weight,
                patch=args.patch, batch=args.batch, frames=len(tr), holdout=hold,
                crf_range=[min(crfs), max(crfs)], sec=dt, ms_per_iter=1000 * dt / args.iters,
                loss_start=traj[0]["loss"] if traj else None,
                loss_end=traj[-1]["ema"] if traj else None, device=str(dev))
    json.dump(dict(meta=meta, traj=traj), open(args.loss_json, "w"), indent=1)
    print(f"# saved checkpoint -> {args.out}")
    print(f"# saved loss trajectory -> {args.loss_json}")
    print(f"# loss(start ema~{traj[0]['loss']:.4f}) -> (end ema {traj[-1]['ema']:.4f})")


if __name__ == "__main__":
    main()
