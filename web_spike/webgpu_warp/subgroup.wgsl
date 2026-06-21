
enable subgroups;
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f32>;
@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
@group(0) @binding(2) var<storage,read> Wt:array<f32>;
@group(0) @binding(3) var<uniform> u:P;

// ONLY the input halo is in shared memory (double-buffered); weights ride in regs.
var<workgroup> sIn:array<f32,648>;

fn loadHalo(ic:u32, sBase:u32, gx0:u32, gy0:u32, plane:u32, lidx:u32){
  for(var t=lidx; t<324u; t+=256u){
    let hx = t % 18u; let hy = t / 18u;
    let xx = i32(gx0)+i32(hx)-1; let yy = i32(gy0)+i32(hy)-1;
    var v = 0.0;
    if(xx>=0 && xx<i32(u.W) && yy>=0 && yy<i32(u.H)){ v = fin[ic*plane + u32(yy)*u.W + u32(xx)]; }
    sIn[sBase + t] = v;
  }
}

@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) wid:vec3u,
        @builtin(local_invocation_index) lidx:u32,
        @builtin(subgroup_invocation_id) sid:u32){
  let lx = lidx % 16u; let ly = lidx / 16u;
  let gx0 = wid.x * 16u; let gy0 = wid.y * 16u;
  let x = gx0 + lx; let y = gy0 + ly;
  let ocbase = wid.z * 64u;
  let plane = u.H * u.W; let inc9 = u.in_c*9u;

  var a0 = vec4<f32>(0.0);
  var a1 = vec4<f32>(0.0);
  var a2 = vec4<f32>(0.0);
  var a3 = vec4<f32>(0.0);
  var a4 = vec4<f32>(0.0);
  var a5 = vec4<f32>(0.0);
  var a6 = vec4<f32>(0.0);
  var a7 = vec4<f32>(0.0);
  var a8 = vec4<f32>(0.0);
  var a9 = vec4<f32>(0.0);
  var a10 = vec4<f32>(0.0);
  var a11 = vec4<f32>(0.0);
  var a12 = vec4<f32>(0.0);
  var a13 = vec4<f32>(0.0);
  var a14 = vec4<f32>(0.0);
  var a15 = vec4<f32>(0.0);

  loadHalo(0u, 0u, gx0, gy0, plane, lidx);
  workgroupBarrier();
  for(var ic=0u; ic<u.in_c; ic++){
    let c = ic & 1u; let sBase = c*324u;
    var w0 = vec4<f32>(0.0);
    var w1 = vec4<f32>(0.0);
    var w2 = vec4<f32>(0.0);
    var w3 = vec4<f32>(0.0);
    var w4 = vec4<f32>(0.0);

    { let gi = 0u*32u + sid;
      if(gi < 144u){
        let g = gi / 9u; let k = gi % 9u; let wb = u.w_off + ic*9u + k; let oc0 = ocbase + g*4u;
        if(oc0+0u < u.out_c){ w0.x = Wt[wb + (oc0+0u)*inc9]; }
        if(oc0+1u < u.out_c){ w0.y = Wt[wb + (oc0+1u)*inc9]; }
        if(oc0+2u < u.out_c){ w0.z = Wt[wb + (oc0+2u)*inc9]; }
        if(oc0+3u < u.out_c){ w0.w = Wt[wb + (oc0+3u)*inc9]; }
      } }
    { let gi = 1u*32u + sid;
      if(gi < 144u){
        let g = gi / 9u; let k = gi % 9u; let wb = u.w_off + ic*9u + k; let oc0 = ocbase + g*4u;
        if(oc0+0u < u.out_c){ w1.x = Wt[wb + (oc0+0u)*inc9]; }
        if(oc0+1u < u.out_c){ w1.y = Wt[wb + (oc0+1u)*inc9]; }
        if(oc0+2u < u.out_c){ w1.z = Wt[wb + (oc0+2u)*inc9]; }
        if(oc0+3u < u.out_c){ w1.w = Wt[wb + (oc0+3u)*inc9]; }
      } }
    { let gi = 2u*32u + sid;
      if(gi < 144u){
        let g = gi / 9u; let k = gi % 9u; let wb = u.w_off + ic*9u + k; let oc0 = ocbase + g*4u;
        if(oc0+0u < u.out_c){ w2.x = Wt[wb + (oc0+0u)*inc9]; }
        if(oc0+1u < u.out_c){ w2.y = Wt[wb + (oc0+1u)*inc9]; }
        if(oc0+2u < u.out_c){ w2.z = Wt[wb + (oc0+2u)*inc9]; }
        if(oc0+3u < u.out_c){ w2.w = Wt[wb + (oc0+3u)*inc9]; }
      } }
    { let gi = 3u*32u + sid;
      if(gi < 144u){
        let g = gi / 9u; let k = gi % 9u; let wb = u.w_off + ic*9u + k; let oc0 = ocbase + g*4u;
        if(oc0+0u < u.out_c){ w3.x = Wt[wb + (oc0+0u)*inc9]; }
        if(oc0+1u < u.out_c){ w3.y = Wt[wb + (oc0+1u)*inc9]; }
        if(oc0+2u < u.out_c){ w3.z = Wt[wb + (oc0+2u)*inc9]; }
        if(oc0+3u < u.out_c){ w3.w = Wt[wb + (oc0+3u)*inc9]; }
      } }
    { let gi = 4u*32u + sid;
      if(gi < 144u){
        let g = gi / 9u; let k = gi % 9u; let wb = u.w_off + ic*9u + k; let oc0 = ocbase + g*4u;
        if(oc0+0u < u.out_c){ w4.x = Wt[wb + (oc0+0u)*inc9]; }
        if(oc0+1u < u.out_c){ w4.y = Wt[wb + (oc0+1u)*inc9]; }
        if(oc0+2u < u.out_c){ w4.z = Wt[wb + (oc0+2u)*inc9]; }
        if(oc0+3u < u.out_c){ w4.w = Wt[wb + (oc0+3u)*inc9]; }
      } }
    if(ic+1u < u.in_c){ loadHalo(ic+1u, (1u-c)*324u, gx0, gy0, plane, lidx); }
        a0 += subgroupBroadcast(w0, 0u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a0 += subgroupBroadcast(w0, 1u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a0 += subgroupBroadcast(w0, 2u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a0 += subgroupBroadcast(w0, 3u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a0 += subgroupBroadcast(w0, 4u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a0 += subgroupBroadcast(w0, 5u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a0 += subgroupBroadcast(w0, 6u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a0 += subgroupBroadcast(w0, 7u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a0 += subgroupBroadcast(w0, 8u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a1 += subgroupBroadcast(w0, 9u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a1 += subgroupBroadcast(w0, 10u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a1 += subgroupBroadcast(w0, 11u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a1 += subgroupBroadcast(w0, 12u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a1 += subgroupBroadcast(w0, 13u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a1 += subgroupBroadcast(w0, 14u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a1 += subgroupBroadcast(w0, 15u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a1 += subgroupBroadcast(w0, 16u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a1 += subgroupBroadcast(w0, 17u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a2 += subgroupBroadcast(w0, 18u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a2 += subgroupBroadcast(w0, 19u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a2 += subgroupBroadcast(w0, 20u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a2 += subgroupBroadcast(w0, 21u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a2 += subgroupBroadcast(w0, 22u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a2 += subgroupBroadcast(w0, 23u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a2 += subgroupBroadcast(w0, 24u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a2 += subgroupBroadcast(w0, 25u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a2 += subgroupBroadcast(w0, 26u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a3 += subgroupBroadcast(w0, 27u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a3 += subgroupBroadcast(w0, 28u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a3 += subgroupBroadcast(w0, 29u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a3 += subgroupBroadcast(w0, 30u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a3 += subgroupBroadcast(w0, 31u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a3 += subgroupBroadcast(w1, 0u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a3 += subgroupBroadcast(w1, 1u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a3 += subgroupBroadcast(w1, 2u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a3 += subgroupBroadcast(w1, 3u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a4 += subgroupBroadcast(w1, 4u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a4 += subgroupBroadcast(w1, 5u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a4 += subgroupBroadcast(w1, 6u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a4 += subgroupBroadcast(w1, 7u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a4 += subgroupBroadcast(w1, 8u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a4 += subgroupBroadcast(w1, 9u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a4 += subgroupBroadcast(w1, 10u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a4 += subgroupBroadcast(w1, 11u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a4 += subgroupBroadcast(w1, 12u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a5 += subgroupBroadcast(w1, 13u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a5 += subgroupBroadcast(w1, 14u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a5 += subgroupBroadcast(w1, 15u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a5 += subgroupBroadcast(w1, 16u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a5 += subgroupBroadcast(w1, 17u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a5 += subgroupBroadcast(w1, 18u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a5 += subgroupBroadcast(w1, 19u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a5 += subgroupBroadcast(w1, 20u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a5 += subgroupBroadcast(w1, 21u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a6 += subgroupBroadcast(w1, 22u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a6 += subgroupBroadcast(w1, 23u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a6 += subgroupBroadcast(w1, 24u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a6 += subgroupBroadcast(w1, 25u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a6 += subgroupBroadcast(w1, 26u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a6 += subgroupBroadcast(w1, 27u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a6 += subgroupBroadcast(w1, 28u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a6 += subgroupBroadcast(w1, 29u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a6 += subgroupBroadcast(w1, 30u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a7 += subgroupBroadcast(w1, 31u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a7 += subgroupBroadcast(w2, 0u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a7 += subgroupBroadcast(w2, 1u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a7 += subgroupBroadcast(w2, 2u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a7 += subgroupBroadcast(w2, 3u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a7 += subgroupBroadcast(w2, 4u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a7 += subgroupBroadcast(w2, 5u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a7 += subgroupBroadcast(w2, 6u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a7 += subgroupBroadcast(w2, 7u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a8 += subgroupBroadcast(w2, 8u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a8 += subgroupBroadcast(w2, 9u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a8 += subgroupBroadcast(w2, 10u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a8 += subgroupBroadcast(w2, 11u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a8 += subgroupBroadcast(w2, 12u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a8 += subgroupBroadcast(w2, 13u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a8 += subgroupBroadcast(w2, 14u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a8 += subgroupBroadcast(w2, 15u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a8 += subgroupBroadcast(w2, 16u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a9 += subgroupBroadcast(w2, 17u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a9 += subgroupBroadcast(w2, 18u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a9 += subgroupBroadcast(w2, 19u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a9 += subgroupBroadcast(w2, 20u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a9 += subgroupBroadcast(w2, 21u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a9 += subgroupBroadcast(w2, 22u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a9 += subgroupBroadcast(w2, 23u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a9 += subgroupBroadcast(w2, 24u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a9 += subgroupBroadcast(w2, 25u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a10 += subgroupBroadcast(w2, 26u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a10 += subgroupBroadcast(w2, 27u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a10 += subgroupBroadcast(w2, 28u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a10 += subgroupBroadcast(w2, 29u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a10 += subgroupBroadcast(w2, 30u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a10 += subgroupBroadcast(w2, 31u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a10 += subgroupBroadcast(w3, 0u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a10 += subgroupBroadcast(w3, 1u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a10 += subgroupBroadcast(w3, 2u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a11 += subgroupBroadcast(w3, 3u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a11 += subgroupBroadcast(w3, 4u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a11 += subgroupBroadcast(w3, 5u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a11 += subgroupBroadcast(w3, 6u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a11 += subgroupBroadcast(w3, 7u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a11 += subgroupBroadcast(w3, 8u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a11 += subgroupBroadcast(w3, 9u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a11 += subgroupBroadcast(w3, 10u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a11 += subgroupBroadcast(w3, 11u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a12 += subgroupBroadcast(w3, 12u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a12 += subgroupBroadcast(w3, 13u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a12 += subgroupBroadcast(w3, 14u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a12 += subgroupBroadcast(w3, 15u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a12 += subgroupBroadcast(w3, 16u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a12 += subgroupBroadcast(w3, 17u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a12 += subgroupBroadcast(w3, 18u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a12 += subgroupBroadcast(w3, 19u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a12 += subgroupBroadcast(w3, 20u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a13 += subgroupBroadcast(w3, 21u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a13 += subgroupBroadcast(w3, 22u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a13 += subgroupBroadcast(w3, 23u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a13 += subgroupBroadcast(w3, 24u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a13 += subgroupBroadcast(w3, 25u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a13 += subgroupBroadcast(w3, 26u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a13 += subgroupBroadcast(w3, 27u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a13 += subgroupBroadcast(w3, 28u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a13 += subgroupBroadcast(w3, 29u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a14 += subgroupBroadcast(w3, 30u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a14 += subgroupBroadcast(w3, 31u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a14 += subgroupBroadcast(w4, 0u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a14 += subgroupBroadcast(w4, 1u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a14 += subgroupBroadcast(w4, 2u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a14 += subgroupBroadcast(w4, 3u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a14 += subgroupBroadcast(w4, 4u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a14 += subgroupBroadcast(w4, 5u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a14 += subgroupBroadcast(w4, 6u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
        a15 += subgroupBroadcast(w4, 7u) * sIn[sBase + (ly+0u)*18u + (lx+0u)];
        a15 += subgroupBroadcast(w4, 8u) * sIn[sBase + (ly+0u)*18u + (lx+1u)];
        a15 += subgroupBroadcast(w4, 9u) * sIn[sBase + (ly+0u)*18u + (lx+2u)];
        a15 += subgroupBroadcast(w4, 10u) * sIn[sBase + (ly+1u)*18u + (lx+0u)];
        a15 += subgroupBroadcast(w4, 11u) * sIn[sBase + (ly+1u)*18u + (lx+1u)];
        a15 += subgroupBroadcast(w4, 12u) * sIn[sBase + (ly+1u)*18u + (lx+2u)];
        a15 += subgroupBroadcast(w4, 13u) * sIn[sBase + (ly+2u)*18u + (lx+0u)];
        a15 += subgroupBroadcast(w4, 14u) * sIn[sBase + (ly+2u)*18u + (lx+1u)];
        a15 += subgroupBroadcast(w4, 15u) * sIn[sBase + (ly+2u)*18u + (lx+2u)];
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
    { let oc = ocbase + 32u; if(oc < u.out_c){ var v = a8.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 33u; if(oc < u.out_c){ var v = a8.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 34u; if(oc < u.out_c){ var v = a8.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 35u; if(oc < u.out_c){ var v = a8.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 36u; if(oc < u.out_c){ var v = a9.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 37u; if(oc < u.out_c){ var v = a9.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 38u; if(oc < u.out_c){ var v = a9.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 39u; if(oc < u.out_c){ var v = a9.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 40u; if(oc < u.out_c){ var v = a10.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 41u; if(oc < u.out_c){ var v = a10.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 42u; if(oc < u.out_c){ var v = a10.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 43u; if(oc < u.out_c){ var v = a10.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 44u; if(oc < u.out_c){ var v = a11.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 45u; if(oc < u.out_c){ var v = a11.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 46u; if(oc < u.out_c){ var v = a11.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 47u; if(oc < u.out_c){ var v = a11.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 48u; if(oc < u.out_c){ var v = a12.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 49u; if(oc < u.out_c){ var v = a12.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 50u; if(oc < u.out_c){ var v = a12.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 51u; if(oc < u.out_c){ var v = a12.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 52u; if(oc < u.out_c){ var v = a13.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 53u; if(oc < u.out_c){ var v = a13.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 54u; if(oc < u.out_c){ var v = a13.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 55u; if(oc < u.out_c){ var v = a13.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 56u; if(oc < u.out_c){ var v = a14.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 57u; if(oc < u.out_c){ var v = a14.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 58u; if(oc < u.out_c){ var v = a14.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 59u; if(oc < u.out_c){ var v = a14.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 60u; if(oc < u.out_c){ var v = a15.x + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 61u; if(oc < u.out_c){ var v = a15.y + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 62u; if(oc < u.out_c){ var v = a15.z + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
    { let oc = ocbase + 63u; if(oc < u.out_c){ var v = a15.w + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }
  }
}