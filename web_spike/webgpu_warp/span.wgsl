// SPAN WGSL (f32 reference). Generated from genSpanWGSL("f32") in conv_opt/span_driver.ts.
// Entry points: conv3x3, conv1x1, silu, gate, pshuffle. PLANAR layout c*H*W+y*W+x.

struct P { H:u32, W:u32, in_c:u32, out_c:u32, w_off:u32, b_off:u32, p6:u32, p7:u32 };
@group(0) @binding(0) var<storage,read>       fin  : array<f32>;
@group(0) @binding(1) var<storage,read_write> fout : array<f32>;
@group(0) @binding(2) var<storage,read>       Wt   : array<f32>;
@group(0) @binding(3) var<uniform>            u    : P;
@group(0) @binding(4) var<storage,read>       fx   : array<f32>;   // gate's x input (dummy otherwise)

// generic 3x3 conv, zero-pad 1 (== PyTorch padding=1 cross-correlation). no activation.
@compute @workgroup_size(8,8,1) fn conv3x3(@builtin(global_invocation_id) g:vec3u){
  let x=i32(g.x); let y=i32(g.y); let oc=i32(g.z);
  if(x>=i32(u.W)||y>=i32(u.H)||oc>=i32(u.out_c)){ return; }
  var acc = f32(Wt[u.b_off+u32(oc)]);
  let bw = u.w_off + u32(oc)*u.in_c*9u;
  for(var ic=0u; ic<u.in_c; ic++){
    let pl=ic*u.H*u.W; let wic=bw+ic*9u;
    for(var ky=0; ky<3; ky++){ let yy=y+ky-1; if(yy<0||yy>=i32(u.H)){ continue; }
      for(var kx=0; kx<3; kx++){ let xx=x+kx-1; if(xx<0||xx>=i32(u.W)){ continue; }
        acc += f32(Wt[wic+u32(ky*3+kx)]) * f32(fin[pl+u32(yy)*u.W+u32(xx)]); } } }
  fout[u32(oc)*u.H*u.W + u32(y)*u.W + u32(x)] = f32(acc);
}

// generic 1x1 conv (conv_cat 192->48). weight layout (oc, ic).
@compute @workgroup_size(8,8,1) fn conv1x1(@builtin(global_invocation_id) g:vec3u){
  let x=g.x; let y=g.y; let oc=g.z;
  if(x>=u.W||y>=u.H||oc>=u.out_c){ return; }
  let pl=u.H*u.W; let p=y*u.W+x;
  var acc = f32(Wt[u.b_off+oc]);
  let bw = u.w_off + oc*u.in_c;
  for(var ic=0u; ic<u.in_c; ic++){ acc += f32(Wt[bw+ic]) * f32(fin[ic*pl+p]); }
  fout[oc*pl+p] = f32(acc);
}

// SiLU: x*sigmoid(x). n = in_c*H*W (in_c carries the channel count).
@compute @workgroup_size(64,1,1) fn silu(@builtin(global_invocation_id) g:vec3u){
  let i=g.x; let n=u.in_c*u.H*u.W; if(i>=n){ return; }
  let v=f32(fin[i]); fout[i]=f32(v*(1.0/(1.0+exp(-v))));
}

// SPAB gate: (o3 + x) * (sigmoid(o3) - 0.5). fin=o3, fx=x.
@compute @workgroup_size(64,1,1) fn gate(@builtin(global_invocation_id) g:vec3u){
  let i=g.x; let n=u.in_c*u.H*u.W; if(i>=n){ return; }
  let o3=f32(fin[i]); let xv=f32(fx[i]);
  fout[i]=f32((o3+xv)*(1.0/(1.0+exp(-o3))-0.5));
}

// PixelShuffle r=2: in (12ch, H,W) -> out (3ch, 2H,2W).
// out[oc, oy, ox] = in[oc*4 + (oy%2)*2 + (ox%2), oy/2, ox/2].
@compute @workgroup_size(8,8,1) fn pshuffle(@builtin(global_invocation_id) g:vec3u){
  let ox=g.x; let oy=g.y; let oc=g.z;
  let OW=u.W*2u; let OH=u.H*2u;
  if(ox>=OW||oy>=OH||oc>=3u){ return; }
  let xx=ox/2u; let yy=oy/2u;
  let ic = oc*4u + (oy%2u)*2u + (ox%2u);
  let lrpl=u.H*u.W;
  fout[oc*(OW*OH) + oy*OW + ox] = f32(f32(fin[ic*lrpl + yy*u.W + xx]));
}
