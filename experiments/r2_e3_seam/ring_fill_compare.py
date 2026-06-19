"""ring_fill_compare.py -- lever (c): is a better disocclusion-ring FILL worth it?

GPU-free LR comparison. Rebuilds the raw NaN temporal-median plate (from bootstrap cache)
and applies fill variants, measuring (1) sharpness of the always-occluded HOLE region and
(2) the share of the visible BG-ring that is hole vs recovered-median. The seam metric is
dominated by the RECOVERED-median ring (soft from low coverage + matte-edge contamination),
NOT the inpaint -- so a better inpaint can only touch the small hole share.
"""
import os
import numpy as np
import cv2
import seam_lib as L

plate_raw = np.load(os.path.join(L.CACHE, "plate_raw.npy"))   # (h,w,3) float32 + NaN
fill_mask = np.load(os.path.join(L.CACHE, "fill_mask.npy"))   # (h,w) bool (always-occluded)
frames = np.load(os.path.join(L.CACHE, "frames_lr.npy"))      # (N,h,w,3)
phas = np.load(os.path.join(L.CACHE, "phas.npy"))
h, w = plate_raw.shape[:2]
allmed = np.median(frames.astype(np.float32), axis=0)


def base_for_inpaint():
    p = plate_raw.copy()
    nanpix = np.isnan(p).any(2)
    p[nanpix] = allmed[nanpix]
    return np.clip(p, 0, 255).astype(np.uint8)


def fill(method):
    base = base_for_inpaint()
    m8 = (fill_mask.astype(np.uint8)) * 255
    bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    if method == "telea5":
        out = cv2.inpaint(bgr, m8, 5, cv2.INPAINT_TELEA)
    elif method == "telea15":
        out = cv2.inpaint(bgr, m8, 15, cv2.INPAINT_TELEA)
    elif method == "ns15":
        out = cv2.inpaint(bgr, m8, 15, cv2.INPAINT_NS)
    elif method == "allmedian":
        p = base.copy(); p[fill_mask] = allmed[fill_mask].astype(np.uint8)
        return p
    elif method == "diffuse":
        # coarse iterative blur-fill: keep known px, blur, re-paste known -> seamless smooth
        f = base.astype(np.float32)
        known = (~fill_mask)[..., None].astype(np.float32)
        cur = f * known
        for _ in range(60):
            cur = cv2.GaussianBlur(cur, (0, 0), 3.0)
            cur = cur * (1 - known) + f * known
        return np.clip(cur, 0, 255).astype(np.uint8)
    else:
        raise ValueError(method)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def vlap_var(img, m):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    return float(lap[m].var()) if m.any() else float("nan")


def main():
    # boundary band around the always-occluded region (the visible disocclusion ring at LR)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    holed = cv2.dilate(fill_mask.astype(np.uint8), k).astype(bool)
    ring_band = holed & ~fill_mask          # recovered pixels adjacent to holes
    print("=== (c) ring-fill comparison (LR, GPU-free) ===")
    print(f"always-occluded holes: {100*fill_mask.mean():.2f}% of frame")
    # share of the mid-frame visible BG-ring that is hole vs recovered
    aH = cv2.resize(phas[16], (w, h)) if phas[16].shape != (h, w) else phas[16]
    bgring = (aH > 0.02) & (aH < 0.2)
    print(f"mid-frame BG-ring: {100*(bgring & fill_mask).sum()/max(bgring.sum(),1):.1f}% hole, "
          f"{100*(bgring & ~fill_mask).sum()/max(bgring.sum(),1):.1f}% recovered-median")
    print(f"sharpness var-Lap: recovered-ring {vlap_var(base_for_inpaint(), ring_band):.2f} (the soft moat)")
    print(f"{'fill':10s} | hole-region var-Lap | seam interpretation")
    for mth in ("telea5", "telea15", "ns15", "diffuse", "allmedian"):
        img = fill(mth)
        s = vlap_var(img, fill_mask)
        print(f"{mth:10s} | {s:18.2f}  | {'(baseline)' if mth=='telea5' else ''}")
    print("\nNote: holes are ALWAYS behind the subject (gotcha #19) -> only the ~8% hole share of "
          "the\nvisible ring is ever affected; the soft RECOVERED-median moat (92%) dominates the "
          "seam,\nso the composite-time plate-band restore (lever b) is the effective fix, not a "
          "better inpaint.")


if __name__ == "__main__":
    main()
