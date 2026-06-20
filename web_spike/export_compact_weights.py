"""P1 step 3: export the COMPACT SR net (realesr-general-x4v3 / SRVGGNetCompact) for on-GPU WGSL.

Removes the last offline dependency except the MVs: the anchor SR currently runs in PyTorch; this
exports the net so a WebGPU shader chain can run it in-browser. SRVGGNetCompact =
  conv(3->64)+PReLU, 32x [conv(64->64)+PReLU], conv(64->48), PixelShuffle(4), + nearest(x4) residual.
=> 34 conv passes (33 with a fused PReLU), then a pixelshuffle+residual pass.

Outputs under web_spike/compact_data/:
  weights.bin   all floats, per pass: [conv_w (oc*ic*9), conv_b (oc), prelu_slopes (oc) if any]
  layers.json   [{in_c,out_c,w_off,b_off,prelu_off(or -1)}...] (float offsets) + meta {H,W,upscale}
  lr.png        the LR crop (input)            sr_ref.png  the PyTorch SR output (parity target)
"""
import os, sys, json
import numpy as np
import cv2, torch
import av, av.sidedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "prototype"))
import sr

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compact_data")
os.makedirs(OUT, exist_ok=True)
CROP = (256, 256)   # LR crop (keeps the parity test snappy; x4 -> 1024x1024)


def main():
    # a SEPARATE CPU net for weight extraction (don't disturb the cached MPS net used for the ref)
    path = sr._ensure_weights("realesrgan")
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("params", sd) if isinstance(sd, dict) else sd
    net = sr._build_compact(); net.load_state_dict(sd, strict=True); net.eval()
    up = net.upscale
    body = list(net.body)
    # group into passes: (conv, optional following prelu)
    passes, i = [], 0
    while i < len(body):
        conv = body[i]; prelu = None
        if i + 1 < len(body) and isinstance(body[i + 1], torch.nn.PReLU):
            prelu = body[i + 1]; i += 2
        else:
            i += 1
        passes.append((conv, prelu))
    blob = []
    floats = 0
    meta_layers = []
    for (conv, prelu) in passes:
        w = conv.weight.detach().numpy().astype(np.float32)   # (oc, ic, 3, 3)
        b = conv.bias.detach().numpy().astype(np.float32)      # (oc,)
        oc, ic = w.shape[0], w.shape[1]
        w_off = floats; blob.append(w.reshape(-1)); floats += w.size
        b_off = floats; blob.append(b.reshape(-1)); floats += b.size
        p_off = -1
        if prelu is not None:
            s = prelu.weight.detach().numpy().astype(np.float32).reshape(-1)  # (oc,)
            p_off = floats; blob.append(s); floats += s.size
        meta_layers.append(dict(in_c=int(ic), out_c=int(oc), w_off=int(w_off),
                                b_off=int(b_off), prelu_off=int(p_off)))
    weights = np.concatenate(blob).astype(np.float32)
    weights.tofile(os.path.join(OUT, "weights.bin"))

    # real LR crop from sample.mp4 frame 0 (the anchor source)
    c = av.open(os.path.join(ROOT, "sample.mp4"))
    f0 = next(c.decode(video=0)).to_ndarray(format="rgb24"); c.close()
    H, W = CROP
    lr = np.ascontiguousarray(f0[40:40 + H, 80:80 + W])   # a textured region (title/face area)
    cv2.imwrite(os.path.join(OUT, "lr.png"), cv2.cvtColor(lr, cv2.COLOR_RGB2BGR))

    # PyTorch reference SR (the exact net we are porting) -- the parity target
    ref = sr.upscale(lr, model="realesrgan")
    cv2.imwrite(os.path.join(OUT, "sr_ref.png"), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))

    json.dump(dict(layers=meta_layers, upscale=int(up), H=int(H), W=int(W),
                   n_passes=len(passes), n_floats=int(weights.size)),
              open(os.path.join(OUT, "layers.json"), "w"))
    print(f"compact net: {len(passes)} conv passes, {weights.size/1e6:.2f}M floats "
          f"({weights.nbytes/1e6:.1f} MB); LR {W}x{H} -> SR {ref.shape[1]}x{ref.shape[0]} (x{up})")
    print(f"pass shapes: in_c/out_c = " + " ".join(f"{m['in_c']}->{m['out_c']}" for m in meta_layers[:3])
          + " ... " + " ".join(f"{m['in_c']}->{m['out_c']}" for m in meta_layers[-2:]))


if __name__ == "__main__":
    main()
