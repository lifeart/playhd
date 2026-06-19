"""R4-E2 -- shippable MV-reuse frame-interpolation pass ("smooth 2x"), OUTPUT-ONLY.

THE LEAD-LANDABLE MODULE. It reuses prototype/derisk.build_lr_flow + gpu_ops.warp_hd
(READ-ONLY) to synthesize a motion-compensated MIDPOINT between two consecutive HD recon
frames the pipeline ALREADY produced -- doubling the output frame rate. It NEVER feeds the
recon reference chain (GOTCHA #16): it only READS R[t]['recon'], R[t+1]['recon'] AFTER
reconstruct() returns, and emits an EXTRA frame between them. The midpoint is never appended to
`frames`, never stored back into R[], so it is structurally incapable of becoming a reference.

R3-E1 (experiments/r3_e1_interp) validated this op (+3.6..+8.9 dB PSNR / tOF cut 2-4x vs
frame-dup & linear-blend, ~17 ms/inserted-frame at 1920x960 on MPS) and surfaced TWO
ship-blockers, both baked in here and NON-optional:

  (1) INTRA-HOLE routing ONLY. A warp is distrusted ONLY where the codec MV field has a true
      hole (no MV -> NaN). The project's full Ruder/reactive occlusion mask OVER-flags
      large-motion blocks and routes them back to the ghosting linear blend (R3-E1 'Where it
      breaks' #3: 29.4 -> 27.8 dB on high motion). So that mask is deliberately NOT used here.

  (2) SCENE-CUT GUARD. If the connecting MV field's intra-hole fraction exceeds CUT_THRESH
      (0.5) there is no reliable motion (scene cut / chaotic intro / cross-anchor mispair), so
      we FALL BACK TO FRAME DUPLICATION rather than synthesize a ghosting midpoint.

Connecting field between display-adjacent output frames (t, t+1): frames[t+1]'s codec 'past' MV
field, build_lr_flow(frames[t+1][2], want='past') -- the field already built while reconstructing
t+1, reused here at ZERO new flow cost (no new optical flow). The midpoint is the deployment
HALF-STEP:
    mid = blend( warp_hd(R[t], +0.5*MV), warp_hd(R[t+1], -0.5*MV) )
Both warps share ONE field => identical hole pattern => the routing reduces to
"intra-hole pixel -> linear blend of the two neighbours, else average of the two warps".
(== R3-E1 interp.hd_cost_torch + the two ship-blockers.)

A NOTE on the B-pyramid (honest, GOTCHA #12): `source` is a ref-list index (+-1), NOT a display
distance, so for a display pair whose t+1 is a P/B whose nearest past reference is NOT the
display-adjacent t, the 'past' field is an APPROXIMATE connector. It degrades safely: small
motion (talking-head) => the half-step warp is a small correction that still beats linear blend;
large motion (high-motion) => its intra-hole fraction is high => the scene-cut guard fires =>
duplication. The guard is what makes the approximate connector safe everywhere.
"""
import numpy as np

CUT_THRESH = 0.5          # intra-hole fraction above which we DUPLICATE instead of interpolate


# --------------------------------------------------------------------------- #
# Connecting field + guard signal (cheap, LR)
# --------------------------------------------------------------------------- #
def connecting_flow(frames, t1, h_lr, w_lr, _build_lr_flow=None):
    """LR 'past' MV field of frame t1 = the motion that connects display frame (t1-1) -> t1.
    Reuses derisk.build_lr_flow READ-ONLY. NaN where t1 carries no past MV (intra/hole)."""
    if _build_lr_flow is None:
        import derisk
        _build_lr_flow = derisk.build_lr_flow
    return _build_lr_flow(frames[t1][2], h_lr, w_lr, want="past")


def intra_hole_frac(fx):
    """Fraction of LR pixels with NO past MV (NaN). This is the scene-cut guard signal."""
    return float(np.isnan(fx).mean())


# --------------------------------------------------------------------------- #
# Intra-hole-ONLY blend core (ship-blocker #1). Shared by the deployment half-step
# (one field => hole_f==hole_b) and the held-out two-field quality harness.
# --------------------------------------------------------------------------- #
def blend_intra_hole_np(fwd, bwd, hole_f, hole_b, src_lin):
    """uint8 fwd/bwd warps + their INTRA-hole masks + a linear-blend source -> blended uint8.
    Routing (intra-hole ONLY -- no Ruder/reactive): both alive -> average; exactly one alive ->
    that one; both dead -> linear blend of the neighbours."""
    f = fwd.astype(np.float32)
    b = bwd.astype(np.float32)
    alive_f = ~hole_f
    alive_b = ~hole_b
    out = 0.5 * (f + b)
    out[alive_f & hole_b] = f[alive_f & hole_b]
    out[hole_f & alive_b] = b[hole_f & alive_b]
    both_dead = hole_f & hole_b
    out[both_dead] = src_lin.astype(np.float32)[both_dead]
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Deployment half-step midpoint -- TORCH/MPS (the shipped instant fast path)
# --------------------------------------------------------------------------- #
def midpoint_torch(left_recon, right_recon, fx, fy, scale, cut_thresh=CUT_THRESH,
                   _G=None):
    """Synthesize the HALF-STEP midpoint between two GPU-resident HD recon tensors
    ([1,3,Hhd,Whd] float, 0..255) using the LR 'past' field (fx,fy) that connects them.

    Returns (mid_tensor, info) where info = {'duplicated': bool, 'hole_frac': float}.
    SCENE-CUT GUARD: hole_frac > cut_thresh -> return left_recon.clone() (frame duplication),
    NO warp computed (also saves the warp cost on cuts). OUTPUT-ONLY: never mutates either input.
    """
    if _G is None:
        import gpu_ops as _G
    import torch
    hole_frac = intra_hole_frac(fx)
    if hole_frac > cut_thresh:                      # ship-blocker #2: scene-cut -> duplicate
        return left_recon.clone(), {"duplicated": True, "hole_frac": hole_frac}
    fxf, fyf = _G.flow_to_dev(0.5 * fx, 0.5 * fy)
    fxb, fyb = _G.flow_to_dev(-0.5 * fx, -0.5 * fy)
    wf, hole_hf = _G.warp_hd(left_recon, fxf, fyf, scale)    # left warped +0.5*MV -> midpoint
    wb, hole_hb = _G.warp_hd(right_recon, fxb, fyb, scale)   # right warped -0.5*MV -> midpoint
    # both warps share ONE field => hole_hf == hole_hb; intra-hole ONLY (no Ruder) -> linear there
    dead = (hole_hf & hole_hb)[None, None]
    src_lin = 0.5 * (left_recon + right_recon)
    mid = torch.where(dead, src_lin, 0.5 * (wf + wb)).clamp(0, 255)
    return mid, {"duplicated": False, "hole_frac": hole_frac}


# --------------------------------------------------------------------------- #
# Deployment half-step midpoint -- NUMPY (quality / layered path, and standalone tests)
# --------------------------------------------------------------------------- #
def midpoint_numpy(left_recon, right_recon, fx, fy, scale, cut_thresh=CUT_THRESH,
                   _warp_hd=None):
    """Numpy twin of midpoint_torch on uint8 HD frames. Returns (mid_uint8, info)."""
    if _warp_hd is None:
        import derisk
        _warp_hd = derisk.warp_hd
    hole_frac = intra_hole_frac(fx)
    if hole_frac > cut_thresh:
        return left_recon.copy(), {"duplicated": True, "hole_frac": hole_frac}
    wf, hole_f = _warp_hd(left_recon, 0.5 * fx, 0.5 * fy, scale)
    wb, hole_b = _warp_hd(right_recon, -0.5 * fx, -0.5 * fy, scale)
    src_lin = (0.5 * (left_recon.astype(np.float32) + right_recon.astype(np.float32)))
    mid = blend_intra_hole_np(wf, wb, hole_f, hole_b, src_lin)
    return mid, {"duplicated": False, "hole_frac": hole_frac}
