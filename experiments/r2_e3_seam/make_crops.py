"""make_crops.py -- before/after SEAM crops for R2-E3 (saved to crops/). GPU-free.

Two seam locations on frame 16: the upper HAIRLINE (hair vs plain wall -> wisp recovery)
and the LEFT-FACE edge vs the textured room (the soft-moat halo + its restore). Each:
a labelled RGB montage [baseline | feather | ringRestore | (a)+(b) | softenFG | ceiling]
plus a local-|Laplacian| heatmap row (shows the soft moat and its repair), and a 2x
before/after/ceiling zoom. Recommended = feather + restore_plate_ring(strength=0.5).
"""
import os
import numpy as np
import cv2
import seam_lib as L

D = L.load_inputs()
H, W = D["plate_hd"].shape[:2]
P0 = D["plate_hd"]
OUT = os.path.join(L.HERE, "crops")
os.makedirs(OUT, exist_ok=True)
I = 16


def lab(im, t):
    o = im.copy()
    cv2.rectangle(o, (0, 0), (o.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(o, t, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return o


def heat(rgb):
    s = np.clip(L._local_sharp(rgb) / 8.0 * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(s, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def variants(i):
    a0 = L.alpha_to_hd(D["phas"][i], (H, W))
    af = L.feather_alpha(a0)
    ring = L.restore_plate_ring(P0, a0, strength=0.5)
    fg = D["x4"][i]
    return [
        ("baseline", L.composite(fg, a0, P0)),
        ("feather(a)", L.composite(fg, af, P0)),
        ("ringRestore(b)", L.composite(fg, a0, ring)),
        ("(a)+(b) RECOMMENDED", L.composite(fg, af, ring)),
        ("softenFG[rej]", L.composite(L.soften_fg_band(fg, a0), a0, P0)),
        ("ceiling x4plus", fg),
    ]


def windows(i):
    a = L.alpha_to_hd(D["phas"][i], (H, W))[..., 0]
    edge = (a > 0.3) & (a < 0.7)
    ys, xs = np.where(edge)
    up = ys < 350
    cy1, cx1 = int(np.median(ys[up])), int(np.median(xs[up]))            # upper hairline
    sel = (ys > 300) & (ys < 650)
    xthr = np.percentile(xs[sel], 15); m = sel & (xs < xthr)
    cy2, cx2 = int(np.median(ys[m])), int(np.median(xs[m]))             # left-face vs room
    return {"hairline": (cy1, cx1), "leftface": (cy2, cx2)}


def render(tag, cy, cx, cs=430):
    y0 = int(np.clip(cy - cs // 2, 0, H - cs)); x0 = int(np.clip(cx - cs // 2, 0, W - cs))
    cr = lambda im: im[y0:y0 + cs, x0:x0 + cs]
    V = variants(I)
    rgb = [lab(cr(o), n) for n, o in V]
    ht = [lab(heat(cr(o)), n) for n, o in V]
    montage = np.concatenate([np.concatenate(rgb, 1), np.concatenate(ht, 1)], 0)
    cv2.imwrite(os.path.join(OUT, f"{tag}_montage.png"), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    z = lambda im: cv2.resize(im, (cs * 2, cs * 2), interpolation=cv2.INTER_NEAREST)
    pair = np.concatenate([lab(z(cr(V[0][1])), "BEFORE baseline"),
                           lab(z(cr(V[3][1])), "AFTER (a)+(b)"),
                           lab(z(cr(V[5][1])), "ceiling x4plus")], 1)
    cv2.imwrite(os.path.join(OUT, f"{tag}_before_after.png"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
    print(f"saved {tag}_* at y{y0} x{x0} cs{cs}")


def main():
    wins = windows(I)
    for tag, (cy, cx) in wins.items():
        render(tag, cy, cx)


if __name__ == "__main__":
    main()
