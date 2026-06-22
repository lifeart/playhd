"""Export 2xLiveActionV1_SPAN for an on-GPU WGSL port (replaces the compact anchor SR).

SPAN (spandrel 0.4.2), scale=2, 48 feature ch, in=3 out=3.
  is_norm = FALSE (state_dict has 'no_norm')  -> NO input normalization (x straight into conv_1)
  act1    = SiLU (x*sigmoid(x))               -> the activation between SPAB convs
  Conv3XC has_relu = FALSE everywhere         -> every conv is a pure linear 3x3 (no leaky_relu)
  Conv3XC @ inference collapses to a single 3x3 (eval_conv via update_params()).

forward(x):
  F   = conv_1(x)                         # Conv3XC 3->48
  B1,_,_      = block_1(F)                # SPAB  (out_b1)
  B2 = block_2(B1); B3=block_3(B2); B4=block_4(B3); B5=block_5(B4)
  B6, B5_2, _ = block_6(B5)               # SPAB returns (out, out1=raw c1_r output, att)
  B6  = conv_2(B6)                        # Conv3XC 48->48
  out = conv_cat( cat([F, B6, B1, B5_2], 1) )   # Conv2d 1x1 192->48
  output = upsampler(out)                 # Conv2d(48->12,3x3) then PixelShuffle(2)   (NO LR residual)

SPAB(x):
  o1 = c1_r(x); a = SiLU(o1); o2 = c2_r(a); a2 = SiLU(o2); o3 = c3_r(a2)
  sim = sigmoid(o3) - 0.5; out = (o3 + x) * sim;  return out, o1, sim

Outputs under web_spike/span_data/:
  weights.bin    all f32; per conv: [w (oc*ic*kk), b (oc)]   (kk=9 for 3x3, 1 for 1x1)
  spec.json      {H,W,scale,feat,n_floats, weights:{name:{in_c,out_c,k,w_off,b_off}}, graph:[...]}
  lr.png         the SD LR input (320x160, real H.264-degraded frame)
  lr_planar.bin  lr as f32 [0,1] PLANAR RGB (3*H*W)  -- the exact WGSL input
  sr_ref.bin     PyTorch forward output, UNCLAMPED, f32 PLANAR (3*2H*2W)  -- exact parity target
  sr_ref.png     clamp[0,1]*255 of sr_ref           -- visual
  inter/*.bin    intermediate tensors (f32 planar) for piecewise WGSL validation
"""
import os, sys, json
import numpy as np
import cv2, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "web_spike"))
from eval_model_options import decode_frames, degrade_h264  # reuse decode + H.264 degrade

from spandrel import ModelLoader
from spandrel.architectures.SPAN.__arch.span import Conv3XC, SPAB

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "span_data")
INTER = os.path.join(OUT, "inter")
os.makedirs(INTER, exist_ok=True)

MODEL = os.path.join(ROOT, "prototype", "models", "2xLiveActionV1_SPAN.pth")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd600.mp4")
FRAME_IDX = 150        # a representative textured frame
CRF = 28               # match eval_x2_candidates production protocol


def planar(t):  # (1,C,H,W) torch -> (C*H*W,) f32 numpy, C-order
    return np.ascontiguousarray(t.detach().cpu().numpy()[0]).reshape(-1).astype(np.float32)


def main():
    net = ModelLoader().load_from_file(MODEL).model.eval()
    assert net.is_norm is False, "expected no_norm model"

    # ---- collapse every Conv3XC to its single eval 3x3 ----
    for _, m in net.named_modules():
        if isinstance(m, Conv3XC):
            m.update_params()

    blob, floats, wtab = [], 0, {}

    def add(name, w, b, k):
        nonlocal floats
        w = w.detach().cpu().numpy().astype(np.float32)   # (oc,ic,kh,kw)
        b = b.detach().cpu().numpy().astype(np.float32)   # (oc,)
        oc, ic = w.shape[0], w.shape[1]
        w_off = floats; blob.append(w.reshape(-1)); floats += w.size
        b_off = floats; blob.append(b.reshape(-1)); floats += b.size
        wtab[name] = dict(in_c=int(ic), out_c=int(oc), k=int(k), w_off=int(w_off), b_off=int(b_off))

    add("conv_1", net.conv_1.eval_conv.weight, net.conv_1.eval_conv.bias, 3)
    for bi in range(1, 7):
        blk = getattr(net, f"block_{bi}")
        for cn in ("c1_r", "c2_r", "c3_r"):
            c = getattr(blk, cn)
            add(f"block_{bi}.{cn}", c.eval_conv.weight, c.eval_conv.bias, 3)
    add("conv_2", net.conv_2.eval_conv.weight, net.conv_2.eval_conv.bias, 3)
    add("conv_cat", net.conv_cat.weight, net.conv_cat.bias, 1)             # 1x1 192->48
    add("upsampler", net.upsampler[0].weight, net.upsampler[0].bias, 3)    # 3x3 48->12

    weights = np.concatenate(blob).astype(np.float32)
    weights.tofile(os.path.join(OUT, "weights.bin"))

    # ---- real SD LR input: decode 640x320 frame -> AREA x2 -> 320x160 -> libx264 crf28 ----
    refs = decode_frames(SRC, [FRAME_IDX])
    H0, W0 = refs[0].shape[:2]                  # 320 x 640
    sd_w, sd_h = W0 // 2, H0 // 2               # 320 x 160
    lr_u8 = degrade_h264(refs, CRF, sd_w, sd_h)[0]   # (160,320,3) uint8 RGB
    H, W = lr_u8.shape[:2]
    cv2.imwrite(os.path.join(OUT, "lr.png"), cv2.cvtColor(lr_u8, cv2.COLOR_RGB2BGR))
    lr_t = torch.from_numpy(lr_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None]  # (1,3,H,W) [0,1]
    planar(lr_t).tofile(os.path.join(OUT, "lr_planar.bin"))

    # ---- forward, capturing intermediates exactly as SPAB/SPAN do ----
    sig = torch.sigmoid
    silu = torch.nn.functional.silu
    with torch.no_grad():
        x = lr_t
        F = net.conv_1(x)

        def spab(blk, xin):
            # act1 is SiLU(inplace=True): the returned out1 is the ACTIVATED c1_r output
            # (the same tensor that feeds c2_r), NOT the raw conv. out_b5_2 uses this.
            o1a = silu(blk.c1_r(xin))
            o2a = silu(blk.c2_r(o1a))
            o3 = blk.c3_r(o2a)
            sim = sig(o3) - 0.5
            return (o3 + xin) * sim, o1a, sim

        B1, _, _ = spab(net.block_1, F)
        B2, _, _ = spab(net.block_2, B1)
        B3, _, _ = spab(net.block_3, B2)
        B4, _, _ = spab(net.block_4, B3)
        B5, _, _ = spab(net.block_5, B4)
        B6, B5_2, _ = spab(net.block_6, B5)
        B6c = net.conv_2(B6)
        cat = torch.cat([F, B6c, B1, B5_2], 1)
        catout = net.conv_cat(cat)
        up0 = net.upsampler[0](catout)      # (1,12,H,W)
        out = net.upsampler[1](up0)         # PixelShuffle(2) -> (1,3,2H,2W)

        # sanity: manual SPAB == real module
        rB1, _, _ = net.block_1(F)
        assert torch.allclose(rB1, B1, atol=1e-5), "manual SPAB mismatch"
        ref_out = net(x)
        assert torch.allclose(ref_out, out, atol=1e-5), "manual forward mismatch"

    # save intermediates for piecewise validation
    for nm, t in [("F", F), ("B1", B1), ("B5_2", B5_2), ("B6c", B6c),
                  ("catout", catout), ("up0", up0)]:
        planar(t).tofile(os.path.join(INTER, nm + ".bin"))
    # also block_1 internals to debug SiLU / gate
    with torch.no_grad():
        o1 = net.block_1.c1_r(F); a = silu(o1)
        o2 = net.block_1.c2_r(a); a2 = silu(o2); o3 = net.block_1.c3_r(a2)
    for nm, t in [("blk1_o1", o1), ("blk1_a", a), ("blk1_o2", o2),
                  ("blk1_a2", a2), ("blk1_o3", o3)]:
        planar(t).tofile(os.path.join(INTER, nm + ".bin"))

    planar(out).tofile(os.path.join(OUT, "sr_ref.bin"))    # UNCLAMPED parity target
    sr_u8 = (out[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, "sr_ref.png"), cv2.cvtColor(sr_u8, cv2.COLOR_RGB2BGR))

    spec = dict(H=int(H), W=int(W), scale=2, feat=48, n_floats=int(weights.size),
                out_h=int(2 * H), out_w=int(2 * W), weights=wtab,
                graph="conv_1->[6x SPAB]->conv_2->conv_cat(F,B6,B1,B5_2)->upsampler(conv+PS2); "
                      "act1=SiLU, no_norm, has_relu=False, no LR residual")
    json.dump(spec, open(os.path.join(OUT, "spec.json"), "w"), indent=1)

    print(f"OK: {weights.size/1e6:.2f}M floats ({weights.nbytes/1e6:.1f} MB); "
          f"LR {W}x{H} -> SR {2*W}x{2*H}; out range [{float(out.min()):.3f},{float(out.max()):.3f}]")
    print("weights:", " ".join(f"{k}({v['in_c']}->{v['out_c']},k{v['k']})" for k, v in wtab.items()))


if __name__ == "__main__":
    main()
