"""GPU (torch/MPS) film-grain pass -- a fast twin of prototype/grain.apply_grain.

Lever 2 of the instant-mode speedup. The prototype grain (prototype/grain.py) is CPU numpy:
a YCrCb round-trip + a per-frame cv2.GaussianBlur + a per-frame noise roll, ~73 ms/frame at
2560x1280. This module ports the SAME recipe to torch/MPS so it runs on the HD recon tensor
that is ALREADY resident on the GPU after derisk.reconstruct_torch(download_output=False) --
no host round-trip, ~a few ms/frame. The prototype grain is left UNTOUCHED (it is the
regression guard and still used by the quality / layered paths).

Recipe parity with grain.apply_grain (verified in __main__ + server/bench_instant.py):
  * Per-frame seed == frame_idx  -> temporally INDEPENDENT, deterministic, never frozen.
  * Spatially-correlated unit template (grain.make_template, Gaussian-blurred white noise),
    re-rolled per frame by the EXACT same per-frame RNG (sy, sx, sign) as grain._frame_grain.
  * Local-luma amplitude modulation: Y = 0.299R+0.587G+0.114B, blurred (sigma=2.0, the same
    auto ksize=17 cv2 picks for a 32F image), through the filmic LUT
    amp = 0.45 + 0.55*sin(pi*Y_local/255) (grain._luma_lut, evaluated continuously).
  * Luma-only, gamma-space, FINAL pass.

Subtlety -- why we add the grain to all three RGB channels instead of round-tripping YCrCb:
adding delta to Y and converting YCrCb->RGB adds exactly delta to each of R,G,B (the chroma
terms are unchanged and Y enters R,G,B with unit weight). So `rgb + grain` is the YCrCb-luma
result WITHOUT the prototype's extra uint8 YCrCb quantization -- it matches apply_grain to
~48 dB (within grain rounding) while skipping two colour-space conversions. Verified.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# The prototype imports modules by bare name; ensure it is importable when this module is
# loaded standalone (when loaded via pipeline_api the dir is already on sys.path).
_PROTO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prototype")
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

import grain as _grain          # prototype (read-only): STRENGTHS, make_template, _luma_lut

# BT.601 luma weights, RGB order -- matches cv2.COLOR_RGB2YCrCb's Y row.
_LUMA_W = (0.299, 0.587, 0.114)
_LUMA_BLUR_SIGMA = 2.0          # == grain._LUMA_BLUR_SIGMA


def _gaussian_kernel1d(sigma, device):
    """1-D Gaussian matching cv2.getGaussianKernel with cv2's auto ksize for a 32F image.

    cv2.GaussianBlur(src32f, (0,0), sigma) picks ksize = round(sigma*4*2+1)|1, then builds
    getGaussianKernel(ksize, sigma) = exp(-(i-c)^2 / (2 sigma^2)) normalised to sum 1."""
    ksize = int(round(sigma * 4 * 2 + 1)) | 1
    c = (ksize - 1) / 2.0
    x = torch.arange(ksize, dtype=torch.float32, device=device) - c
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k, ksize


def download_rgb(rgb_1c3hw):
    """[1,3,H,W] float (0..255) on device -> HxWx3 uint8 RGB numpy. Byte-identical to
    gpu_ops.img_to_host but ~5x faster: it makes the HWC layout CONTIGUOUS on-device first, so
    the device->host copy is a single contiguous transfer and the numpy result needs no CPU-side
    re-pack (img_to_host's permute leaves a strided host array that ascontiguousarray must recopy)."""
    t = rgb_1c3hw.clamp(0, 255).round_().to(torch.uint8)[0].permute(1, 2, 0).contiguous()
    return t.cpu().numpy()


class GpuGrain:
    """Reusable GPU grain applier for a fixed HD frame size on a fixed device.

    Build once per (H, W, device); call apply() per frame. Holds the unit grain template and
    the separable luma-blur kernel on-device so each frame is pure GPU work (no host transfer
    of the frame). frame_idx seeds the per-frame roll so the grain stays temporally independent.
    """

    def __init__(self, h, w, device, template_seed=0):
        self.h, self.w = h, w
        self.device = device
        # Unit-variance, spatially-correlated template -- built ONCE with the prototype's exact
        # numpy recipe (content-independent), then resident on device. Re-rolled per frame below.
        tmpl = _grain.make_template(h, w, seed=template_seed)            # numpy HxW float32
        self.template = torch.from_numpy(np.ascontiguousarray(tmpl)).to(device)
        k1d, self.ksize = _gaussian_kernel1d(_LUMA_BLUR_SIGMA, device)
        self.kx = k1d.view(1, 1, 1, self.ksize)                          # separable: horiz
        self.ky = k1d.view(1, 1, self.ksize, 1)                          # separable: vert
        self.pad = self.ksize // 2
        self.lw = torch.tensor(_LUMA_W, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def _frame_unit_grain(self, frame_idx):
        """Unit grain field for this frame index -- EXACT same per-frame RNG sequence as
        grain._frame_grain (sy=integers(0,H), sx=integers(0,W), sign from random()<0.5),
        but the roll runs on-device (torch.roll == np.roll)."""
        rng = np.random.default_rng(frame_idx * 2654435761 % (2 ** 32))
        sy = int(rng.integers(0, self.h))
        sx = int(rng.integers(0, self.w))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        g = torch.roll(self.template, shifts=(sy, sx), dims=(0, 1))
        return g * sign

    def _luma_blur(self, y):
        """Gaussian-blur a [1,1,H,W] luma map (separable, reflect pad == cv2 BORDER_REFLECT_101)."""
        y = F.conv2d(F.pad(y, (self.pad, self.pad, 0, 0), mode="reflect"), self.kx)
        y = F.conv2d(F.pad(y, (0, 0, self.pad, self.pad), mode="reflect"), self.ky)
        return y

    def apply(self, rgb_1c3hw, frame_idx, strength="med"):
        """Add per-frame film grain to an HD recon tensor [1,3,H,W] float (0..255) on device.
        Returns a NEW tensor (the input -- a propagation reference -- is never mutated)."""
        sigma = (_grain.STRENGTHS.get(strength, 0.0) if isinstance(strength, str)
                 else float(strength))
        if sigma <= 0.0:
            return rgb_1c3hw.clone()
        y = (rgb_1c3hw * self.lw).sum(dim=1, keepdim=True)               # [1,1,H,W] luma
        y_local = self._luma_blur(y).clamp(0.0, 255.0)
        amp = 0.45 + 0.55 * torch.sin(torch.pi * y_local / 255.0)        # filmic LUT (continuous)
        unit = self._frame_unit_grain(frame_idx)[None, None]            # [1,1,H,W]
        grain = unit * (sigma * amp)                                     # broadcast amp
        return (rgb_1c3hw + grain).clamp(0.0, 255.0)                     # grain adds to all RGB


if __name__ == "__main__":
    # Parity check vs the CPU prototype grain on a real-ish frame.
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    rng = np.random.default_rng(0)
    H, W = 1280, 2560
    frame = rng.integers(0, 256, (H, W, 3)).astype(np.uint8)
    gg = GpuGrain(H, W, dev)
    tmpl = _grain.make_template(H, W, seed=0)          # the SAME fixed template the GPU path uses
    print("port correctness -- GPU vs apply_grain with the SAME template (isolates the port):")
    for idx in (0, 7, 42):
        cpu = _grain.apply_grain(frame, idx, "med", template=tmpl).astype(np.int16)
        t = torch.from_numpy(frame).to(dev).permute(2, 0, 1).unsqueeze(0).float()
        out_t = gg.apply(t, idx, "med")
        gpu = out_t.clamp(0, 255).round_().squeeze(0).permute(1, 2, 0).to("cpu", torch.uint8).numpy().astype(np.int16)
        mse = float(np.mean((cpu - gpu) ** 2))
        psnr = 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)
        mad = float(np.abs(cpu - gpu).mean())
        print(f"  idx={idx}: PSNR={psnr:.2f} dB  mean|delta|={mad:.3f} codes")
    # statistical equivalence: rolled-fixed-template vs fresh-per-frame (apply_grain default).
    print("statistical equivalence -- grain std should match (both unit*sigma*amp):")
    res_cpu = (_grain.apply_grain(frame, 3, "med").astype(np.float32) - frame.astype(np.float32))
    t = torch.from_numpy(frame).to(dev).permute(2, 0, 1).unsqueeze(0).float()
    res_gpu = (gg.apply(t, 3, "med").clamp(0, 255).round_().squeeze(0).permute(1, 2, 0)
               .to("cpu").numpy() - frame.astype(np.float32))
    print(f"  grain std: cpu(fresh)={res_cpu.std():.3f}  gpu(rolled)={res_gpu.std():.3f} codes")
