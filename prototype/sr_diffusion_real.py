#!/usr/bin/env python3
"""
REAL diffusion super-resolution anchor for playhd  (Q1 MAX-QUALITY track, 2026-06).

WHY THIS FILE EXISTS
--------------------
Stream-3's `sr_diffusion.py` proved the SD2/OSEDiff one-step *compute pattern* runs on MPS
with zero CPU fallback, but never downloaded real weights, so we had no idea whether a
diffusion SR actually recovers MORE TRUE detail than the heavy GAN anchor (RealESRGAN_x4plus)
on OUR content: real, H.264-compressed SD. This module loads a REAL, ungated, diffusers-native
diffusion SR -- `stabilityai/stable-diffusion-x4-upscaler` via StableDiffusionUpscalePipeline --
runs it on MPS, and is the single wiring point for an offline `--sr diffusion-real` anchor.

It is import-safe (degrades gracefully if `diffusers`/weights are absent), never silently
no-ops (raises loudly if the model can't load), and -- per the Q1 guardrails -- only CREATES a
new file; it imports `sr` read-only and touches nothing shared.

Mechanism note (matters for TAESD): the x4-upscaler runs latent diffusion in a 4x-downsample
latent space (output/4), NOT the standard SD 8x space (output/8), and its UNet takes 7 in-
channels (4 latent + 3 for the low-res image concatenated in *pixel* space at latent res).
TAESD (`madebyollin/taesd`) is tuned to the standard SD 8x/4-channel VAE latent, so it is NOT a
drop-in decoder for this pipeline -- `try_swap_taesd()` checks and reports this empirically
rather than assuming.

Public API (mirrors sr.py conventions so it can drop in as an anchor):
    upscale_diffusion_real(rgb_uint8_HxWx3, **opts) -> rgb_uint8_(4H)x(4W)x3   (x4, MPS)
    load_pipeline(use_taesd=False)                  -> cached pipeline on device
    last/mean/median_latency_ms() / n_calls() / reset_latency()
    real_model_available()                          -> bool (can we load it here?)
    integration_hint()
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
    from diffusers import StableDiffusionUpscalePipeline
    DIFFUSERS_AVAILABLE = True
    DIFFUSERS_VERSION = diffusers.__version__
except Exception:  # pragma: no cover
    DIFFUSERS_AVAILABLE = False
    DIFFUSERS_VERSION = None

MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"
UPSCALE = 4
TILE_LR = 128          # natural LR tile -> 512 HD output tile (the pipeline's comfortable unit)

_PIPE = None           # cached pipeline
_PIPE_USES_TAESD = None
_DEVICE = None
_LAT = []              # per-call ms (MPS-synchronized, whole-image incl. tiling)
_TILE_LAT = []         # per-tile ms (one pipe() invocation)


def pick_device():
    if _TORCH and torch.backends.mps.is_available():
        return torch.device("mps")
    if _TORCH and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _mem_mb(device):
    if device.type == "mps":
        return dict(
            current_alloc_mb=round(torch.mps.current_allocated_memory() / 1e6, 1),
            driver_alloc_mb=round(torch.mps.driver_allocated_memory() / 1e6, 1),
            recommended_max_mb=round(torch.mps.recommended_max_memory() / 1e6, 1),
        )
    return {}


def try_swap_taesd(pipe, device, dtype):
    """Attempt to replace the x4-upscaler VAE with TAESD (madebyollin/taesd).

    Returns (ok: bool, note: str). Never raises -- on any incompatibility it leaves the
    original VAE in place and explains why. TAESD targets the standard SD 8x/4-ch latent,
    while the x4-upscaler decodes a 4x/4-ch latent, so this is expected to be reported
    INCOMPATIBLE; we verify empirically with a real decode of the right-shaped latent.
    """
    try:
        from diffusers import AutoencoderTiny
    except Exception as e:
        return False, f"AutoencoderTiny import failed: {type(e).__name__}: {e}"
    orig_vae = pipe.vae
    # latent shape the x4-upscaler actually produces for a 512 output tile
    try:
        sf = pipe.vae_scale_factor
    except Exception:
        sf = 4
    lat_hw = 512 // sf
    try:
        taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype)
        taesd = taesd.to(device)
        with torch.no_grad():
            z = torch.randn(1, orig_vae.config.latent_channels, lat_hw, lat_hw,
                            device=device, dtype=dtype)
            img = taesd.decode(z).sample
        out_hw = img.shape[-1]
        if out_hw != 512:
            taesd.to("cpu"); del taesd
            return False, (f"TAESD decodes {orig_vae.config.latent_channels}x{lat_hw}^2 latent "
                           f"-> {out_hw}^2, but the x4-upscaler needs ->512 (it uses a {sf}x VAE, "
                           f"TAESD is 8x). Incompatible; keeping the native VAE.")
        pipe.vae = taesd
        return True, f"TAESD swapped in (decodes ->{out_hw}). NOTE: latent stats differ; expect artifacts."
    except Exception as e:
        return False, f"TAESD decode test failed ({type(e).__name__}: {e}); keeping native VAE."


def load_pipeline(use_taesd=False, dtype=None):
    """Load + cache StableDiffusionUpscalePipeline on the local device.

    Downloads ~3.5GB to the HF cache on first call. Raises loudly if diffusers/weights are
    unavailable (never silently no-ops). use_taesd attempts the tiny-VAE decode swap (see
    try_swap_taesd -- expected incompatible for this pipeline; reported, not silently ignored).
    """
    global _PIPE, _PIPE_USES_TAESD, _DEVICE
    if not (_TORCH and DIFFUSERS_AVAILABLE):
        raise RuntimeError("sr_diffusion_real needs torch + diffusers installed.")
    if _PIPE is not None and _PIPE_USES_TAESD == use_taesd:
        return _PIPE
    _DEVICE = pick_device()
    if dtype is None:
        dtype = torch.float16 if _DEVICE.type in ("mps", "cuda") else torch.float32
    print(f"[diff-real] loading {MODEL_ID} dtype={dtype} device={_DEVICE} "
          f"(downloads ~3.5GB on first run)...")
    t0 = time.perf_counter()
    pipe = StableDiffusionUpscalePipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to(_DEVICE)
    pipe.set_progress_bar_config(disable=True)
    taesd_note = "native VAE"
    if use_taesd:
        ok, note = try_swap_taesd(pipe, _DEVICE, dtype)
        taesd_note = note
        print(f"[diff-real] TAESD: {note}")
    _sync(_DEVICE)
    print(f"[diff-real] loaded in {time.perf_counter()-t0:.1f}s | {taesd_note} | mem={_mem_mb(_DEVICE)}")
    _PIPE = pipe
    _PIPE_USES_TAESD = use_taesd
    return pipe


def _to_pil(rgb_uint8):
    from PIL import Image
    return Image.fromarray(np.ascontiguousarray(rgb_uint8), mode="RGB")


@contextlib.contextmanager
def _no_grad():
    if _TORCH:
        with torch.no_grad():
            yield
    else:
        yield


def _upscale_tile(pipe, rgb_tile, prompt, steps, guidance, noise_level, generator):
    """One pipeline invocation on a single LR tile -> x4 uint8. Times it (per-tile)."""
    pil = _to_pil(rgb_tile)
    _sync(_DEVICE)
    t0 = time.perf_counter()
    with _no_grad():
        out = pipe(prompt=prompt, image=pil, num_inference_steps=steps,
                   guidance_scale=guidance, noise_level=noise_level,
                   generator=generator, output_type="np").images[0]
    _sync(_DEVICE)
    _TILE_LAT.append((time.perf_counter() - t0) * 1000.0)
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def upscale_diffusion_real(rgb_uint8, prompt="", steps=50, guidance=0.0,
                           noise_level=20, use_taesd=False, seed=0, tile=TILE_LR,
                           overlap=16, verbose=False):
    """x4 super-resolve an HxWx3 uint8 RGB image with stable-diffusion-x4-upscaler on MPS.

    Returns (4H)x(4W)x3 uint8. Images larger than `tile` are processed in overlapping `tile`-px
    LR tiles and feather-blended in the x4 output canvas (the pipeline is happiest on small
    inputs). guidance=0.0 keeps it faithful (no prompt-driven CFG hallucination); noise_level is
    the SD-upscaler conditioning-noise knob (higher = more invented HF, default 20).
    """
    pipe = load_pipeline(use_taesd=use_taesd)
    H, W = rgb_uint8.shape[:2]
    gen = torch.Generator(device="cpu").manual_seed(seed)  # cpu generator: MPS-safe determinism
    _sync(_DEVICE)
    t_all = time.perf_counter()

    if H <= tile and W <= tile:
        out = _upscale_tile(pipe, rgb_uint8, prompt, steps, guidance, noise_level, gen)
        _LAT.append((time.perf_counter() - t_all) * 1000.0)
        return out

    # ---- overlapping-tile path for full frames ----
    s = UPSCALE
    canvas = np.zeros((H * s, W * s, 3), np.float32)
    wsum = np.zeros((H * s, W * s, 1), np.float32)
    step = tile - overlap
    ys = list(range(0, max(1, H - tile + 1), step)) or [0]
    xs = list(range(0, max(1, W - tile + 1), step)) or [0]
    if ys[-1] != H - tile and H > tile:
        ys.append(H - tile)
    if xs[-1] != W - tile and W > tile:
        xs.append(W - tile)
    # feather window (raised-cosine) to hide seams
    def _feather(n):
        w = np.ones(n, np.float32)
        f = max(1, overlap * s)
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, f)))
        w[:f] = ramp; w[-f:] = ramp[::-1]
        return w
    for y in ys:
        for x in xs:
            th = min(tile, H - y); tw = min(tile, W - x)
            sub = rgb_uint8[y:y + th, x:x + tw]
            up = _upscale_tile(pipe, sub, prompt, steps, guidance, noise_level, gen)
            oy, ox = y * s, x * s
            fh, fw = up.shape[:2]
            win = (_feather(fh)[:, None] * _feather(fw)[None, :])[:, :, None]
            canvas[oy:oy + fh, ox:ox + fw] += up.astype(np.float32) * win
            wsum[oy:oy + fh, ox:ox + fw] += win
            if verbose:
                print(f"[diff-real] tile ({x},{y}) {sub.shape}->{up.shape} "
                      f"{_TILE_LAT[-1]/1000:.1f}s")
    out = (canvas / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8)
    _LAT.append((time.perf_counter() - t_all) * 1000.0)
    return out


# --- latency accounting (MPS-synchronized) --- #
def last_latency_ms():
    return _LAT[-1] if _LAT else float("nan")


def mean_latency_ms():
    return float(np.mean(_LAT)) if _LAT else float("nan")


def median_latency_ms():
    return float(np.median(_LAT)) if _LAT else float("nan")


def last_tile_latency_ms():
    return _TILE_LAT[-1] if _TILE_LAT else float("nan")


def median_tile_latency_ms():
    return float(np.median(_TILE_LAT)) if _TILE_LAT else float("nan")


def n_calls():
    return len(_LAT)


def n_tiles():
    return len(_TILE_LAT)


def reset_latency():
    _LAT.clear(); _TILE_LAT.clear()


def real_model_available():
    """True if the diffusion anchor can actually be used here (deps present). Weight download
    happens lazily in load_pipeline; this just gates the `--sr diffusion-real` option."""
    return bool(_TORCH and DIFFUSERS_AVAILABLE)


def integration_hint():
    return (
        "Offline MAX-QUALITY anchor wiring (DO NOT silently swallow import errors):\n"
        "    try:\n"
        "        import sr_diffusion_real\n"
        "        HAVE_DIFF_REAL = sr_diffusion_real.real_model_available()\n"
        "    except Exception:\n"
        "        HAVE_DIFF_REAL = False\n"
        "  Add '--sr diffusion-real' to the SR-mode choices; in the anchor SR branch:\n"
        "    if args.sr == 'diffusion-real':\n"
        "        hd = sr_diffusion_real.upscale_diffusion_real(lr, steps=50, guidance=0.0)\n"
        "  OFFLINE/BUFFERED ONLY: ~tens of seconds per 512 tile on MPS; gate on HAVE_DIFF_REAL\n"
        "  and fall back to realesrgan-x4plus for the realtime path.")


if __name__ == "__main__":
    print("diffusers:", DIFFUSERS_AVAILABLE, DIFFUSERS_VERSION, "| device:", pick_device())
    print(integration_hint())
