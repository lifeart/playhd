#!/usr/bin/env python3
"""
Real SR networks for the playhd anchor (replaces the bicubic placeholder).

Two anchor models, selectable by name (Step 8 added the heavier perceptual one):

  * "realesrgan"         -- Real-ESRGAN `realesr-general-x4v3`
                            == SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_conv=32, upscale=4, act='prelu'). 1.21M params, ~5 MB. A
                            VGG-style body (first conv+PReLU, 32x (conv+PReLU)), a final
                            conv to 3*upscale^2 channels + PixelShuffle(4), plus a
                            nearest-upsampled input residual. The COMPACT real-time anchor.

  * "realesrgan-x4plus"  -- Real-ESRGAN `RealESRGAN_x4plus`
                            == RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4). ~16.7M params, ~64 MB.
                            23 Residual-in-Residual Dense Blocks. The HEAVY perceptual
                            anchor (Step 8): much more hallucinated high-frequency detail.
                            Affordable because SR runs only on sparse anchors (~2-12% of
                            frames), so its higher per-frame latency is amortized away.

Both are x4 spatial SR nets. The architectures are hand-written here (NOT basicsr/realesrgan
-- those are torchvision dependency hell on this box) and the published state_dicts are loaded
directly with strict=True (catches any arch mismatch). Checkpoint key layouts verified against
the downloaded weights:
    realesr-general-x4v3.pth : key 'params'      -> body.0.. / body.66 (48,64,3,3) -> PixelShuffle(4)
    RealESRGAN_x4plus.pth    : key 'params_ema'  -> conv_first / body.0..22.rdb{1,2,3}.conv{1..5}
                               / conv_body / conv_up1 / conv_up2 / conv_hr / conv_last

Public API (model name optional, defaults to the compact "realesrgan" for back-compat):
    upscale(rgb_uint8_HxWx3, model="realesrgan")  -> rgb_uint8_(4H)x(4W)x3   (x4, cached, MPS)
    upscale_to(rgb_uint8, w_hd, h_hd, model=...)  -> x4 SR then resize-if-needed to target
    load_model(name="realesrgan")                                            (build+cache)
    last/mean/median_latency_ms(model=None) / n_calls(model=None) / reset_latency(model=None)
Each model is loaded once and cached; weights download to models/ on first use. Latency is
tracked per-model (MPS-synchronized per call); accessors default to the most recently used.
"""
import os
import time
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
UPSCALE = 4


# --------------------------------------------------------------------------- #
# Architecture 1: SRVGGNetCompact (realesr-general-x4v3, act_type='prelu')
# --------------------------------------------------------------------------- #
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        # first conv + activation
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        # body: num_conv x (conv + activation)
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        # last conv -> num_out_ch * upscale^2 channels for pixel-shuffle
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # residual: network learns HF on top of a nearest-neighbour upsample of the input
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


# --------------------------------------------------------------------------- #
# Architecture 2: RRDBNet (RealESRGAN_x4plus) -- the heavy perceptual anchor (Step 8)
# Matches BasicSR's rrdbnet_arch exactly so the published state_dict loads strict=True.
# --------------------------------------------------------------------------- #
class ResidualDenseBlock(nn.Module):
    """5-conv dense block with 0.2 residual scaling (BasicSR)."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 dense blocks with 0.2 residual scaling."""

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23,
                 num_grow_ch=32):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsampling: two nearest x2 stages (scale=4)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
def _build_compact():
    return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=UPSCALE)


def _build_x4plus():
    return RRDBNet(num_in_ch=3, num_out_ch=3, scale=UPSCALE, num_feat=64, num_block=23,
                   num_grow_ch=32)


MODELS = {
    "realesrgan": dict(
        build=_build_compact,
        url=("https://github.com/xinntao/Real-ESRGAN/releases/download/"
             "v0.2.5.0/realesr-general-x4v3.pth"),
        fname="realesr-general-x4v3.pth",
        label="realesr-general-x4v3 (SRVGGNetCompact, compact)",
    ),
    "realesrgan-x4plus": dict(
        build=_build_x4plus,
        url=("https://github.com/xinntao/Real-ESRGAN/releases/download/"
             "v0.1.0/RealESRGAN_x4plus.pth"),
        fname="RealESRGAN_x4plus.pth",
        label="RealESRGAN_x4plus (RRDBNet x23, heavy)",
    ),
}
SR_NAMES = list(MODELS.keys())


# --------------------------------------------------------------------------- #
# Cached loader + inference (per-model caches; latency tracked per-model)
# --------------------------------------------------------------------------- #
_MODELS = {}          # name -> nn.Module (loaded, on device)
_DEVICE = None
_LAT = {}             # name -> [per-call ms, ...]
_LAST_MODEL = None    # most recently invoked model (accessor default)


def _ensure_weights(name):
    spec = MODELS[name]
    path = os.path.join(_HERE, "models", spec["fname"])
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[sr] downloading {name} weights -> {path}")
    urllib.request.urlretrieve(spec["url"], path)
    return path


def _pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name="realesrgan", device=None):
    """Build the named arch, load its published state_dict (strict=True), cache on device."""
    global _DEVICE
    if name not in MODELS:
        raise ValueError(f"unknown SR model {name!r}; choices: {SR_NAMES}")
    if name in _MODELS:
        return _MODELS[name]
    path = _ensure_weights(name)
    _DEVICE = device or _DEVICE or _pick_device()
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "params_ema" in sd:     # x4plus ckpt wraps weights under params_ema
        sd = sd["params_ema"]
    elif isinstance(sd, dict) and "params" in sd:       # compact realesr ckpt -> 'params'
        sd = sd["params"]
    model = MODELS[name]["build"]()
    model.load_state_dict(sd, strict=True)              # strict: catch arch mismatch
    model.eval().to(_DEVICE)
    _MODELS[name] = model
    _LAT.setdefault(name, [])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[sr] loaded {MODELS[name]['label']} (x{UPSCALE}, {n_params/1e6:.2f}M params) on {_DEVICE}")
    return model


@torch.no_grad()
def upscale(rgb_uint8, model="realesrgan"):
    """Super-resolve an HxWx3 uint8 RGB image by x4 with `model`. Returns (4H)x(4W)x3 uint8."""
    global _LAST_MODEL
    net = load_model(model)
    _LAST_MODEL = model
    t = torch.from_numpy(np.ascontiguousarray(rgb_uint8)).to(_DEVICE)
    t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)  # 1,3,H,W in [0,1]
    if _DEVICE.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    out = net(t)
    if _DEVICE.type == "mps":
        torch.mps.synchronize()
    _LAT.setdefault(model, []).append((time.perf_counter() - t0) * 1000.0)
    out = out.clamp_(0.0, 1.0).mul_(255.0).round_()
    out = out.squeeze(0).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return np.ascontiguousarray(out)


def upscale_to(rgb_uint8, w_hd, h_hd, model="realesrgan"):
    """x4 SR then (only if needed) resize to an explicit target. At scale 4 this is identity;
    the safety resize lets the warp pipeline stay scale-agnostic without forcing the x4 model
    into a non-4 scale internally."""
    out = upscale(rgb_uint8, model=model)
    if out.shape[1] != w_hd or out.shape[0] != h_hd:
        out = cv2.resize(out, (w_hd, h_hd), interpolation=cv2.INTER_CUBIC)
    return out


# --- latency accounting (MPS-synchronized per call; per-model) --- #
def _lat(model):
    return _LAT.get(model or _LAST_MODEL, [])


def last_latency_ms(model=None):
    lat = _lat(model)
    return lat[-1] if lat else float("nan")


def mean_latency_ms(model=None):
    lat = _lat(model)
    return float(np.mean(lat)) if lat else float("nan")


def median_latency_ms(model=None):
    lat = _lat(model)
    return float(np.median(lat)) if lat else float("nan")


def n_calls(model=None):
    return len(_lat(model))


def reset_latency(model=None):
    if model is None:
        for v in _LAT.values():
            v.clear()
    else:
        _LAT.setdefault(model, []).clear()


if __name__ == "__main__":
    # smoke test: load + upscale a small tile with BOTH models, report shape + latency
    x = (np.random.default_rng(0).integers(0, 256, (320, 640, 3))).astype(np.uint8)
    for name in SR_NAMES:
        load_model(name)
        y = upscale(x, model=name)
        for _ in range(3):
            upscale(x, model=name)
        print(f"  {name}: in {x.shape} -> out {y.shape}  "
              f"latency last={last_latency_ms(name):.1f}ms median={median_latency_ms(name):.1f}ms")
