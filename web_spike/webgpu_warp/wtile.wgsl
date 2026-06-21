
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f32>;
@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
@group(0) @binding(2) var<storage,read> Wt:array<f32>;
@group(0) @binding(3) var<uniform> u:P;

// double-buffered shared: [2] halos + [2] weight tiles (one being computed, one prefetched)
var<workgroup> sIn:array<f32,648>;
var<workgroup> sW:array<vec4<f32>,144>;

fn loadTile(ic:u32, sBase:u32, wBase:u32, gx0:u32, gy0:u32, ocbase:u32, plane:u32, inc9:u32, lidx:u32){
  for(var t=lidx; t<324u; t+=256u){
    let hx = t % 18u;
    let hy = t / 18u;
    let xx = i32(gx0)+i32(hx)-1;
    let yy = i32(gy0)+i32(hy)-1;
    var v = 0.0;
    if(xx>=0 && xx<i32(u.W) && yy>=0 && yy<i32(u.H)){ v = fin[ic*plane + u32(yy)*u.W + u32(xx)]; }
    sIn[sBase + t] = v;
  }
  for(var t=lidx; t<72u; t+=256u){
    let g = t / 9u;
    let k = t % 9u;
    let wb = u.w_off + ic*9u + k;
    let oc0 = ocbase + g*4u;
    var w = vec4<f32>(0.0);
    if(oc0+0u < u.out_c){ w.x = Wt[wb + (oc0+0u)*inc9]; }
    if(oc0+1u < u.out_c){ w.y = Wt[wb + (oc0+1u)*inc9]; }
    if(oc0+2u < u.out_c){ w.z = Wt[wb + (oc0+2u)*inc9]; }
    if(oc0+3u < u.out_c){ w.w = Wt[wb + (oc0+3u)*inc9]; }
    sW[wBase + t] = w;
  }
}

@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) wid:vec3u, @builtin(local_invocation_index) lidx:u32){
  let lx = lidx % 16u;
  let ly = lidx / 16u;
  let gx0 = wid.x * 16u;
  let gy0 = wid.y * 16u;
  let x = gx0 + lx;
  let y = gy0 + ly;
  let ocbase = wid.z * 32u;
  let plane = u.H * u.W;
  let inc9 = u.in_c*9u;

  var a0 = vec4<f32>(0.0);
  var a1 = vec4<f32>(0.0);
  var a2 = vec4<f32>(0.0);
  var a3 = vec4<f32>(0.0);
  var a4 = vec4<f32>(0.0);
  var a5 = vec4<f32>(0.0);
  var a6 = vec4<f32>(0.0);
  var a7 = vec4<f32>(0.0);

  loadTile(0u, 0u, 0u, gx0, gy0, ocbase, plane, inc9, lidx);
  workgroupBarrier();
  for(var ic=0u; ic<u.in_c; ic++){
    let c = ic & 1u;
    let sBase = c*324u;
    let wBase = c*72u;
    // prefetch next input channel into the alternate buffer (overlaps the FMAs below)
    if(ic+1u < u.in_c){ loadTile(ic+1u, (1u-c)*324u, (1u-c)*72u, gx0, gy0, ocbase, plane, inc9, lidx); }
    for(var ky=0u; ky<3u; ky++){
      for(var kx=0u; kx<3u; kx++){
        let inv = sIn[sBase + (ly+ky)*18u + (lx+kx)];
        let wk = ky*3u+kx;
        a0 += sW[wBase + 0u + wk] * inv;
        a1 += sW[wBase + 9u + wk] * inv;
        a2 += sW[wBase + 18u + wk] * inv;
        a3 += sW[wBase + 27u + wk] * inv;
        a4 += sW[wBase + 36u + wk] * inv;
        a5 += sW[wBase + 45u + wk] * inv;
        a6 += sW[wBase + 54u + wk] * inv;
        a7 += sW[wBase + 63u + wk] * inv;
      }
    }
    workgroupBarrier();
  }

  if(x<u.W && y<u.H){
    { let oc = ocbase + 0u; if(oc < u.out_c){ var v = a0.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 1u; if(oc < u.out_c){ var v = a0.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 2u; if(oc < u.out_c){ var v = a0.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 3u; if(oc < u.out_c){ var v = a0.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 4u; if(oc < u.out_c){ var v = a1.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 5u; if(oc < u.out_c){ var v = a1.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 6u; if(oc < u.out_c){ var v = a1.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 7u; if(oc < u.out_c){ var v = a1.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 8u; if(oc < u.out_c){ var v = a2.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 9u; if(oc < u.out_c){ var v = a2.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 10u; if(oc < u.out_c){ var v = a2.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 11u; if(oc < u.out_c){ var v = a2.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 12u; if(oc < u.out_c){ var v = a3.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 13u; if(oc < u.out_c){ var v = a3.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 14u; if(oc < u.out_c){ var v = a3.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 15u; if(oc < u.out_c){ var v = a3.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 16u; if(oc < u.out_c){ var v = a4.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 17u; if(oc < u.out_c){ var v = a4.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 18u; if(oc < u.out_c){ var v = a4.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 19u; if(oc < u.out_c){ var v = a4.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 20u; if(oc < u.out_c){ var v = a5.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 21u; if(oc < u.out_c){ var v = a5.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 22u; if(oc < u.out_c){ var v = a5.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 23u; if(oc < u.out_c){ var v = a5.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 24u; if(oc < u.out_c){ var v = a6.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 25u; if(oc < u.out_c){ var v = a6.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 26u; if(oc < u.out_c){ var v = a6.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 27u; if(oc < u.out_c){ var v = a6.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 28u; if(oc < u.out_c){ var v = a7.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 29u; if(oc < u.out_c){ var v = a7.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 30u; if(oc < u.out_c){ var v = a7.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 31u; if(oc < u.out_c){ var v = a7.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
  }
}