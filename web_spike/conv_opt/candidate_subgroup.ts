// ============================================================================
// SUBGROUP strategy: WEIGHT-BROADCAST conv (f32).  *** UNMEASURED IN DENO ***
//
// !!! THE DENO HARNESS CANNOT COMPILE THIS. !!!
//   Both available Deno builds reject `enable subgroups;`:
//     - system deno 2.6.3 : naga "unknown enable-extension `subgroups`"
//                           (the `subgroups` adapter feature is a PHANTOM -- the
//                            device capability is advertised but the WGSL front-end
//                            never implemented the language extension)
//     - latest deno 2.8.3 : naga "the `subgroups` enable-extension is not yet
//                            implemented in Naga" (wgpu issue #5555)
//   The `subgroups` *device* feature can be force-requested in 2.8.3
//   (requestDevice({requiredFeatures:["subgroups"]}) succeeds) but the WGSL
//   compiler still won't parse subgroup builtins/ops. => no subgroup kernel can
//   be timed or parity-checked in the harness, the designated arbiter.
//
//   This file is provided for the MANAGER to compile + GPU-time in CHROME
//   (Dawn/Tint DOES implement the subgroups WGSL extension; request the
//   "subgroups" GPUFeatureName, may need chrome://flags WebGPU experimental).
//   The math MIRRORS candidate_wtile.ts exactly (f32, bit-exact target), so the
//   manager can parity-check vs naive in Chrome. Correctness of the lane-indexing
//   is validated in Deno by the structural twin candidate_subgroup_emu.ts
//   (subgroupBroadcast replaced by an equivalent shared-mem bounce) -> PARITY-OK.
//
// STRATEGY: the 3x3 weights are spatially UNIFORM (every pixel in the tile uses
//   the same weights). Instead of staging them in workgroup shared memory, each
//   32-lane subgroup loads the WSZ=G*9 weight vec4s cooperatively into REGISTERS
//   (lane L holds slots s=0..4 -> global weight index gi = s*32 + L), then
//   distributes each weight to all lanes with subgroupBroadcast(w{slot}, gi%32).
//   The input halo stays in shared memory (the 2D 3x3 stencil's VERTICAL reuse
//   crosses the 2-row subgroup boundary -> can't be expressed with size-32
//   subgroup shuffles; only horizontal reuse could, which is marginal).
//
// EXPECTATION (honest, Apple M-series, 32-lane): TIE or SLIGHT LOSS vs combo.
//   - Apple threadgroup-memory uniform reads (all lanes same address) are already
//     a ~free broadcast with no bank conflict; subgroupBroadcast can't beat that
//     for the READ -- it only FREES the ~4.6KB weight shared tile.
//   - But OCB=64 makes the kernel REGISTER-bound (16 vec4 acc = 32 regs), so
//     freeing shared memory does NOT raise the occupancy ceiling -> no win.
//   - It also ADDS 144 broadcasts/channel on top of the 144 MACs and makes the
//     weight load 8x redundant (per-subgroup) vs 1x (shared).
//   Could help MORE on NVIDIA/AMD (cheaper shuffle, costlier shared bank
//   conflicts) but the bigger lever there is dp4a/tensor, not subgroups.
// ============================================================================
const TW = 16, TH = 16, OCB = 64;
const HW = TW + 2, HSZ = HW * (TH + 2);
const NT = TW * TH;
const G = OCB / 4;            // 16 vec4 accumulators
const WSZ = G * 9;            // 144 weight vec4 per input channel
const SUB = 32;              // Apple subgroup size
const SLOTS = Math.ceil(WSZ / SUB);   // 5 weight registers per lane

const accDecl = Array.from({ length: G }, (_, g) => `  var a${g} = vec4<f32>(0.0);`).join("\n");
// per-channel: load this lane's SLOTS weight registers (gi = s*32 + sid)
const wregDecl = Array.from({ length: SLOTS }, (_, s) => `    var w${s} = vec4<f32>(0.0);`).join("\n");
const wregLoad = Array.from({ length: SLOTS }, (_, s) => `
    { let gi = ${s}u*${SUB}u + sid;
      if(gi < ${WSZ}u){
        let g = gi / 9u; let k = gi % 9u; let wb = u.w_off + ic*9u + k; let oc0 = ocbase + g*4u;
        if(oc0+0u < u.out_c){ w${s}.x = Wt[wb + (oc0+0u)*inc9]; }
        if(oc0+1u < u.out_c){ w${s}.y = Wt[wb + (oc0+1u)*inc9]; }
        if(oc0+2u < u.out_c){ w${s}.z = Wt[wb + (oc0+2u)*inc9]; }
        if(oc0+3u < u.out_c){ w${s}.w = Wt[wb + (oc0+3u)*inc9]; }
      } }`).join("");
// compute body: fully unrolled g x ky x kx ; weight via subgroupBroadcast
const accBody = Array.from({ length: G }, (_, g) =>
  [0, 1, 2].flatMap((ky) => [0, 1, 2].map((kx) => {
    const wk = ky * 3 + kx;
    const i = g * 9 + wk;                    // constant global weight index
    const slot = Math.floor(i / SUB), lane = i % SUB;
    return `        a${g} += subgroupBroadcast(w${slot}, ${lane}u) * sIn[sBase + (ly+${ky}u)*${HW}u + (lx+${kx}u)];`;
  })).join("\n")
).join("\n");
const writeBody = Array.from({ length: G }, (_, g) =>
  [0, 1, 2, 3].map((j) => {
    const ch = g * 4 + j, comp = ["x", "y", "z", "w"][j];
    return `    { let oc = ocbase + ${ch}u; if(oc < u.out_c){ var v = a${g}.${comp} + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }`;
  }).join("\n")
).join("\n");

const code = `
enable subgroups;
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f32>;
@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
@group(0) @binding(2) var<storage,read> Wt:array<f32>;
@group(0) @binding(3) var<uniform> u:P;

// ONLY the input halo is in shared memory (double-buffered); weights ride in regs.
var<workgroup> sIn:array<f32,${2 * HSZ}>;

fn loadHalo(ic:u32, sBase:u32, gx0:u32, gy0:u32, plane:u32, lidx:u32){
  for(var t=lidx; t<${HSZ}u; t+=${NT}u){
    let hx = t % ${HW}u; let hy = t / ${HW}u;
    let xx = i32(gx0)+i32(hx)-1; let yy = i32(gy0)+i32(hy)-1;
    var v = 0.0;
    if(xx>=0 && xx<i32(u.W) && yy>=0 && yy<i32(u.H)){ v = fin[ic*plane + u32(yy)*u.W + u32(xx)]; }
    sIn[sBase + t] = v;
  }
}

@compute @workgroup_size(${TW},${TH},1)
fn main(@builtin(workgroup_id) wid:vec3u,
        @builtin(local_invocation_index) lidx:u32,
        @builtin(subgroup_invocation_id) sid:u32){
  let lx = lidx % ${TW}u; let ly = lidx / ${TW}u;
  let gx0 = wid.x * ${TW}u; let gy0 = wid.y * ${TH}u;
  let x = gx0 + lx; let y = gy0 + ly;
  let ocbase = wid.z * ${OCB}u;
  let plane = u.H * u.W; let inc9 = u.in_c*9u;

${accDecl}

  loadHalo(0u, 0u, gx0, gy0, plane, lidx);
  workgroupBarrier();
  for(var ic=0u; ic<u.in_c; ic++){
    let c = ic & 1u; let sBase = c*${HSZ}u;
${wregDecl}
${wregLoad}
    if(ic+1u < u.in_c){ loadHalo(ic+1u, (1u-c)*${HSZ}u, gx0, gy0, plane, lidx); }
${accBody}
    workgroupBarrier();
  }

  if(x<u.W && y<u.H){
${writeBody}
  }
}`;

export default {
  code,
  dispatch: (ly: any, H: number, W: number) => [Math.ceil(W / TW), Math.ceil(H / TH), Math.ceil(ly.out_c / OCB)],
};
