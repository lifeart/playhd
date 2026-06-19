#!/usr/bin/env python3
"""
OPTIONAL diffusion-SR anchor for playhd  (Stream-3 feasibility spike, 2026-06).

WHY THIS FILE EXISTS
--------------------
The verified quality research flagged one-step real-world diffusion SR (OSEDiff/SD2,
TSD-SR/SD3, ResShift/SinSR) as the strongest perceptual SR family -- a candidate for the
HEAVY anchor (amortized over sparse anchors). But every published latency number is on an
A100; whether the SD-style UNet + VAE-decode even *runs* on Apple MPS, which ops fall back
to CPU, the per-tile latency and the unified-memory footprint were ALL unverified. This
module answers that, and is written so it NEVER breaks the main prototype: it is import-safe
(degrades gracefully if `diffusers` is absent) and is only loaded behind a `--sr diffusion`
option (see `integration_hint()` at the bottom).

TWO things live here:
  1. `mps_feasibility_probe()` -- the core measurement. Builds the EXACT compute graph an
     SD2-based one-step real-world SR (OSEDiff) executes per 512x512 output tile -- the SD2.1
     UNet2DConditionModel (865M params, cross-attention dim 1024) run for ONE denoise step on
     a 64x64x4 latent, plus the SD AutoencoderKL decode 64x64 -> 512x512 -- using diffusers'
     real module configs with RANDOM weights. Random weights give garbage pixels but VALID
     timing / operator-coverage / memory numbers, which is exactly the unverified risk. No
     multi-GB download required.
  2. `upscale_diffusion()` -- the real-anchor hook. Runs IF a real one-step real-world SR is
     actually importable+loadable on this box (e.g. a wired-up SinSR/ResShift or an OSEDiff
     diffusers pipeline). Mirrors sr.py's `upscale(rgb_uint8) -> rgb_uint8` so it can drop in
     as a `--sr diffusion` anchor. If no real model is wired, it raises a clear error -- it
     never silently no-ops.

Mirrors sr.py's MPS conventions: pick mps if available, torch.mps.synchronize() around timed
regions, latency in ms.
"""
import os
import time
import contextlib

import numpy as np

try:
    import torch
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False

try:
    import diffusers
    from diffusers import UNet2DConditionModel, AutoencoderKL
    DIFFUSERS_AVAILABLE = True
    DIFFUSERS_VERSION = diffusers.__version__
except Exception:  # pragma: no cover
    DIFFUSERS_AVAILABLE = False
    DIFFUSERS_VERSION = None


def pick_device():
    if _TORCH and torch.backends.mps.is_available():
        return torch.device("mps")
    if _TORCH and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# SD2.1-base module configs (the OSEDiff backbone). Hardcoded so NO download is
# needed -- these are the published stable-diffusion-2-1-base configs.
# --------------------------------------------------------------------------- #
SD21_UNET_CONFIG = dict(
    sample_size=64,
    in_channels=4,
    out_channels=4,
    down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                      "CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
    block_out_channels=(320, 640, 1280, 1280),
    layers_per_block=2,
    cross_attention_dim=1024,
    attention_head_dim=(5, 10, 20, 20),
    use_linear_projection=True,
    norm_num_groups=32,
)

SD_VAE_CONFIG = dict(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D",
                      "DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D",
                    "UpDecoderBlock2D", "UpDecoderBlock2D"),
    block_out_channels=(128, 256, 512, 512),
    layers_per_block=2,
    latent_channels=4,
    sample_size=768,
    norm_num_groups=32,
)


def build_sd2_unet(dtype, device):
    net = UNet2DConditionModel(**SD21_UNET_CONFIG)
    net = net.eval().to(device=device, dtype=dtype)
    return net


def build_sd_vae(dtype, device):
    vae = AutoencoderKL(**SD_VAE_CONFIG)
    vae = vae.eval().to(device=device, dtype=dtype)
    return vae


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _mem_mb(device):
    if device.type == "mps":
        return dict(
            current_alloc_mb=torch.mps.current_allocated_memory() / 1e6,
            driver_alloc_mb=torch.mps.driver_allocated_memory() / 1e6,
            recommended_max_mb=torch.mps.recommended_max_memory() / 1e6,
        )
    return {}


@contextlib.contextmanager
def _no_grad():
    if _TORCH:
        with torch.no_grad():
            yield
    else:
        yield


def mps_feasibility_probe(tile_out=512, n_warmup=1, n_iter=3, dtype=None, verbose=True):
    """Measure the SD2/OSEDiff one-step compute pattern on the local device.

    For one `tile_out`x`tile_out` OUTPUT tile (the chop unit real-world SR diffusion uses):
      * UNet2DConditionModel (SD2.1, 865M) ONE forward on a (tile_out/8)^2 x4 latent
      * AutoencoderKL.decode  (tile_out/8)^2 -> tile_out^2

    Returns a dict of latencies (ms), param counts, memory (MB), and the dtype/device used.
    Honors PYTORCH_ENABLE_MPS_FALLBACK: if it is *unset/0* and an op is unsupported on MPS,
    torch raises NotImplementedError naming the op -- which this function lets propagate so the
    caller can record exactly what falls back. With it =1, unsupported ops silently run on CPU
    and this just times the (possibly slower) end-to-end.
    """
    assert _TORCH and DIFFUSERS_AVAILABLE, "needs torch + diffusers"
    device = pick_device()
    if dtype is None:
        dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
    lat = (tile_out // 8)

    if verbose:
        print(f"[diff-probe] device={device} dtype={dtype} tile_out={tile_out} "
              f"latent={lat}x{lat} fallback_env={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','unset')}")

    out = dict(device=str(device), dtype=str(dtype).replace("torch.", ""),
               tile_out=tile_out, latent=lat,
               diffusers_version=DIFFUSERS_VERSION)

    # ---- build ----
    t0 = time.perf_counter()
    unet = build_sd2_unet(dtype, device)
    vae = build_sd_vae(dtype, device)
    _sync(device)
    out["build_s"] = time.perf_counter() - t0
    out["unet_params_m"] = sum(p.numel() for p in unet.parameters()) / 1e6
    out["vae_params_m"] = sum(p.numel() for p in vae.parameters()) / 1e6
    out["mem_after_build"] = _mem_mb(device)

    # ---- inputs (one-step real-world SR: a single timestep, an LR-derived latent,
    #      and a text/DAPE embedding context of 77 tokens x 1024 dim) ----
    lat_in = torch.randn(1, 4, lat, lat, device=device, dtype=dtype)
    ctx = torch.randn(1, 77, 1024, device=device, dtype=dtype)
    t = torch.tensor([249], device=device)  # one-step models use a fixed large-ish t

    with _no_grad():
        # ---- UNet one-step timing ----
        for _ in range(n_warmup):
            unet(lat_in, t, encoder_hidden_states=ctx).sample
        _sync(device)
        ts = []
        for _ in range(n_iter):
            _sync(device); a = time.perf_counter()
            unet(lat_in, t, encoder_hidden_states=ctx).sample
            _sync(device); ts.append((time.perf_counter() - a) * 1000.0)
        out["unet_step_ms"] = float(np.median(ts))
        out["unet_step_ms_all"] = [round(x, 1) for x in ts]
        out["mem_after_unet"] = _mem_mb(device)

        # ---- VAE decode timing ----
        dec_in = torch.randn(1, 4, lat, lat, device=device, dtype=dtype)
        for _ in range(n_warmup):
            vae.decode(dec_in).sample
        _sync(device)
        ts = []
        for _ in range(n_iter):
            _sync(device); a = time.perf_counter()
            img = vae.decode(dec_in).sample
            _sync(device); ts.append((time.perf_counter() - a) * 1000.0)
        out["vae_decode_ms"] = float(np.median(ts))
        out["vae_decode_ms_all"] = [round(x, 1) for x in ts]
        out["decoded_shape"] = list(img.shape)
        out["mem_peak"] = _mem_mb(device)

    out["tile_total_ms"] = out["unet_step_ms"] + out["vae_decode_ms"]
    # full 640x320 LR -> 2560x1280 HD == ceil(2560/512)*ceil(1280/512)=5*3=15 tiles of 512^2
    out["tiles_per_frame_640x320_x4"] = 15
    out["frame_est_ms"] = out["tile_total_ms"] * 15
    if verbose:
        print(f"[diff-probe] unet 1-step {out['unet_step_ms']:.1f} ms | "
              f"vae decode {out['vae_decode_ms']:.1f} ms | tile {out['tile_total_ms']:.1f} ms | "
              f"~frame(15 tiles) {out['frame_est_ms']/1000:.2f} s")
    return out


# --------------------------------------------------------------------------- #
# Real-anchor hook (only does something if a real one-step real-world SR is wired up).
# Kept import-safe + loud-on-failure (no silent no-op) per project rules.
# --------------------------------------------------------------------------- #
_REAL_RUNNER = None


def real_model_available():
    """True only if a real one-step real-world SR is importable+loadable here.

    On this box it is NOT (SinSR/ResShift need `basicsr` + a ~750MB weight pull at ~1.6MB/s;
    OSEDiff needs the gated SD2.1-base ~5GB base on a 98%-full disk). Left as the single wiring
    point: set `_REAL_RUNNER` to a callable(rgb_uint8)->rgb_uint8 to enable `--sr diffusion`.
    """
    return _REAL_RUNNER is not None


def upscale_diffusion(rgb_uint8, scale=4):
    """x4 SR via a real diffusion anchor. Raises (never silently no-ops) if none is wired."""
    if _REAL_RUNNER is None:
        raise RuntimeError(
            "diffusion anchor not available on this box: no real one-step real-world SR is "
            "wired up. See sr_diffusion.real_model_available() docstring for why "
            "(basicsr dep + multi-GB download + disk). Use --sr realesrgan-x4plus instead.")
    return _REAL_RUNNER(rgb_uint8)


def integration_hint():
    return (
        "To expose this as `--sr diffusion` in derisk.py (DO NOT silently swallow import errors):\n"
        "    try:\n"
        "        import sr_diffusion\n"
        "        HAVE_DIFFUSION = sr_diffusion.real_model_available()\n"
        "    except Exception:\n"
        "        HAVE_DIFFUSION = False\n"
        "  then in the anchor SR branch: if args.sr == 'diffusion': hd = sr_diffusion.upscale_diffusion(lr)\n"
        "  (gate the choice on HAVE_DIFFUSION; fall back to realesrgan-x4plus otherwise).")


if __name__ == "__main__":
    print("diffusers available:", DIFFUSERS_AVAILABLE, DIFFUSERS_VERSION)
    if DIFFUSERS_AVAILABLE:
        import json
        r = mps_feasibility_probe()
        print(json.dumps({k: v for k, v in r.items() if not k.endswith("_all")}, indent=2))
