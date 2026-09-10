"""Independently verify B's two claimed invariants + the 4-way formula equivalence."""
import os, sys
import numpy as np
ROOT="/Users/lifeart/Repos/playhd"
sys.path.insert(0, os.path.join(ROOT,"prototype"))
import av, av.sidedata
import derisk, residual_gate, block_confidence as BC

CLIP=os.path.join(ROOT,"web_spike","sd600.mp4")
PT={1:"I",2:"P",3:"B"}
def decode(start,n):
    c=av.open(CLIP); v=c.streams.video[0]
    v.codec_context.options={"flags2":"+export_mvs"}
    out=[]
    for i,f in enumerate(c.decode(video=0)):
        if i<start: continue
        if len(out)>=n: break
        rgb=f.to_ndarray(format="rgb24")
        sd=f.side_data.get(av.sidedata.sidedata.Type.MOTION_VECTORS)
        mvs=sd.to_ndarray() if sd is not None else None
        out.append((PT.get(int(f.pict_type),"?"),rgb,mvs))
    c.close(); return out

frames=decode(582,12)
h,w=frames[0][1].shape[:2]
print("frames",len(frames),"shape",h,w)

# --- invariant 1: c_floor=1.0, mode="tau" == residual_gate.reliability_lr, bit-identical
maxd=0.0; n=0
holes_ok=True; hole_checked=0
for i in range(1,len(frames)):
    for want in ("past","future","all"):
        _,lr_cur,mvs=frames[i]
        _,lr_ref,_=frames[i-1]
        fx,fy=derisk.build_lr_flow(mvs,h,w,want=want)
        a_ref=residual_gate.reliability_lr(fx,fy,lr_cur,lr_ref,dict(tau_res=10.0,s_res=5.0))
        ms,mm,ar=BC.block_signals(mvs,h,w,want=want)
        a_bc=BC.reliability_lr_bc(fx,fy,lr_cur,lr_ref,ms,mm,dict(tau_res=10.0,s_res=5.0,c_floor=1.0,mode="tau"),area=ar)
        d=float(np.max(np.abs(a_ref-a_bc))); maxd=max(maxd,d); n+=1
        # invariant 2: hole sets identical
        hf=~np.isfinite(fx); hb=~(np.isfinite(ms)&np.isfinite(mm))
        if not np.array_equal(hf,hb):
            holes_ok=False
            print("  HOLE MISMATCH frame",i,want,"diff px",int((hf!=hb).sum()))
        hole_checked+=1
print(f"INV1 max|delta| over {n} (frame,direction) pairs = {maxd!r}  -> {'PASS' if maxd==0.0 else 'FAIL'}")
print(f"INV2 isfinite(block_signals)==isfinite(build_lr_flow) on {hole_checked} pairs -> {'PASS' if holes_ok else 'FAIL'}")

# --- 4-way formula equivalence: A's WGSL form vs block_confidence penalty mode
# reproduce the WGSL arithmetic in numpy from the same inputs
def wgsl_a(fx,fy,lr_cur,lr_ref,area,mvmag,tau,s,kp,km,mvref):
    pred=derisk.warp_lr(lr_ref,fx,fy).astype(np.float32)
    react=np.abs(lr_cur.astype(np.float32)-pred).mean(axis=2)
    zc=np.clip(np.nan_to_num(area,nan=0.0)/256.0,0.0,1.0)
    mc=np.clip(np.nan_to_num(mvmag,nan=np.inf)/max(mvref,1e-3),0.0,1.0)
    t=np.maximum(tau-kp*(1.0-zc)-km*mc,0.0)
    x=np.clip((react-t)/max(s,1e-6),-30.0,30.0)
    a=1.0/(1.0+np.exp(x))
    a[~np.isfinite(fx)]=0.0
    return a.astype(np.float32)

md=0.0
for i in range(1,len(frames)):
    _,lr_cur,mvs=frames[i]; _,lr_ref,_=frames[i-1]
    fx,fy=derisk.build_lr_flow(mvs,h,w,want="all")
    ms,mm,ar=BC.block_signals(mvs,h,w,want="all")
    a_w=wgsl_a(fx,fy,lr_cur,lr_ref,ar,mm,16.0,3.0,6.0,6.0,8.0)
    a_p=BC.reliability_lr_bc(fx,fy,lr_cur,lr_ref,ms,mm,dict(tau_res=16.0,s_res=3.0,kp=6.0,km=6.0,mvref=8.0,mode="penalty"),area=ar)
    md=max(md,float(np.max(np.abs(a_w-a_p))))
print(f"CROSS: A's WGSL arithmetic (numpy transcription) vs block_confidence mode='penalty' max|delta| = {md:.3e}")
