#!/usr/bin/env bash
# r12_e4 -- FULL codec-matched SPAN retrain launch command (NOT run automatically).
#
# Produces a commercially-clean (Apache-2.0 arch, fresh weights) SPAN x2 that has SEEN real
# H.264 -> a drop-in replacement for the non-commercial 2xLiveActionV1_SPAN demo weights.
# Output: out/span_codec_full.pth  (loads via spandrel + web_spike/export_span_weights.py).
#
# BEFORE RUNNING (see report "Full-run plan"):
#   1. Replace --src with a HIGH-BITRATE / near-pristine talking-head HR source (the current
#      sd600.mp4 is itself ~400 kbps H.264 -> a SOFT HR target = the model learns softness).
#      Fallback permissive corpora: REDS (720p sharp), BVI-DVC (codec-research UHD), Vimeo-90K.
#   2. For degradation diversity across the full CRF band, prefer on-the-fly re-encode
#      (degrade.py already draws random crf∈[23,40]/preset/gop per call); raise --frames so the
#      precompute path still covers the distribution, or extend train.py to re-degrade per epoch.
#
# Wall-clock on THIS box (Apple MPS, measured 170 ms/iter @ batch12/patch96; ~2x at batch32/patch128):
#   150k iters  ~=  10-12 h on MPS.   On a rented CUDA GPU (A10/4090): ~30-90 min (10-20x faster).
set -e
cd "$(dirname "$0")"

PYTORCH_ENABLE_MPS_FALLBACK=1 python3 train.py \
  --src   /path/to/pristine_talkinghead_HR.mp4 \
  --frames 3000 \
  --stride 1 \
  --scale 2 \
  --feat  48 \
  --patch 128 \
  --batch 32 \
  --iters 150000 \
  --lr    5e-4 \
  --min-lr 1e-5 \
  --lpips-weight 0.5 \
  --gop   1 \
  --holdout 32 \
  --log-every 500 \
  --out out/span_codec_full.pth \
  --loss-json out/loss_full.json

# then evaluate the trained checkpoint on the held-out real-libx264 frames:
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 eval_ckpt.py \
  --ckpt out/span_codec_full.pth --out-json out/eval_full.json
