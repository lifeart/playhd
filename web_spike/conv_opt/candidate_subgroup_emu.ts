// ============================================================================
// EMULATION TWIN of candidate_subgroup.ts -- runs in Deno to VALIDATE CORRECTNESS
// of the subgroup weight-broadcast indexing (subgroupBroadcast can't compile in
// the Deno naga front-end; see candidate_subgroup.ts header).
//
// Faithful structural mirror: each "subgroup" of 32 lanes (lane = lidx%32,
// subgroup = lidx/32) loads the SAME per-lane weight registers (gi = s*32+lane)
// as the real kernel, then -- instead of subgroupBroadcast(w{slot}, srcLane) --
// bounces them through a per-subgroup shared region and reads back at the
// IDENTICAL index sWreg[base + srcLane*SLOTS + slot]. If this passes parity vs
// naive, the lane<->global-weight-index mapping (load gi=s*32+lane; consume
// lane=i%32, slot=i/32) is proven correct, so the real subgroupBroadcast version
// (mechanical substitution) is correct too. f32, mirrors candidate_wtile math.
//
// NOTE: this is NOT a perf proxy for subgroups -- it ADDS a shared bounce + extra
// barriers + 8x redundant per-subgroup weight loads, so it is EXPECTED to be
// SLOWER than wtile. Its only job is correctness validation of the indexing.
// ============================================================================
const TW = 16, TH = 16, OCB = 64;
const HW = TW + 2, HSZ = HW * (TH + 2);
const NT = TW * TH;
const G = OCB / 4;            // 16
const WSZ = G * 9;            // 144
const SUB = 32, SLOTS = Math.ceil(WSZ / SUB); // 5
const SGROUPS = NT / SUB;     // 8
const REGION = SUB * SLOTS;   // 160 vec4 per subgroup

const accDecl = Array.from({ length: G }, (_, g) => `  var a${g} = vec4<f32>(0.0);`).join("\n");
const wregDecl = Array.from({ length: SLOTS }, (_, s) => `    var w${s} = vec4<f32>(0.0);`).join("\n");
const wregLoad = Array.from({ length: SLOTS }, (_, s) => `
    { let gi = ${s}u*${SUB}u + lane;
      if(gi < ${WSZ}u){
        let g = gi / 9u; let k = gi % 9u; let wb = u.w_off + ic*9u + k; let oc0 = ocbase + g*4u;
        if(oc0+0u < u.out_c){ w${s}.x = Wt[wb + (oc0+0u)*inc9]; }
        if(oc0+1u < u.out_c){ w${s}.y = Wt[wb + (oc0+1u)*inc9]; }
        if(oc0+2u < u.out_c){ w${s}.z = Wt[wb + (oc0+2u)*inc9]; }
        if(oc0+3u < u.out_c){ w${s}.w = Wt[wb + (oc0+3u)*inc9]; }
      } }`).join("");
const wregStore = Array.from({ length: SLOTS }, (_, s) =>
  `    sWreg[wbase + lane*${SLOTS}u + ${s}u] = w${s};`).join("\n");
// consume via shared bounce at IDENTICAL index the subgroupBroadcast would address
const accBody = Array.from({ length: G }, (_, g) =>
  [0, 1, 2].flatMap((ky) => [0, 1, 2].map((kx) => {
    const wk = ky * 3 + kx;
    const i = g * 9 + wk;
    const slot = Math.floor(i / SUB), lane = i % SUB;
    return `        a${g} += sWreg[wbase + ${lane}u*${SLOTS}u + ${slot}u] * sIn[(ly+${ky}u)*${HW}u + (lx+${kx}u)];`;
  })).join("\n")
).join("\n");
const writeBody = Array.from({ length: G }, (_, g) =>
  [0, 1, 2, 3].map((j) => {
    const ch = g * 4 + j, comp = ["x", "y", "z", "w"][j];
    return `    { let oc = ocbase + ${ch}u; if(oc < u.out_c){ var v = a${g}.${comp} + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }`;
  }).join("\n")
).join("\n");

const code = `
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f32>;
@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
@group(0) @binding(2) var<storage,read> Wt:array<f32>;
@group(0) @binding(3) var<uniform> u:P;

var<workgroup> sIn:array<f32,${HSZ}>;                 // single-buffer halo
var<workgroup> sWreg:array<vec4<f32>,${SGROUPS * REGION}>; // per-subgroup weight bounce

fn loadHalo(ic:u32, gx0:u32, gy0:u32, plane:u32, lidx:u32){
  for(var t=lidx; t<${HSZ}u; t+=${NT}u){
    let hx = t % ${HW}u; let hy = t / ${HW}u;
    let xx = i32(gx0)+i32(hx)-1; let yy = i32(gy0)+i32(hy)-1;
    var v = 0.0;
    if(xx>=0 && xx<i32(u.W) && yy>=0 && yy<i32(u.H)){ v = fin[ic*plane + u32(yy)*u.W + u32(xx)]; }
    sIn[t] = v;
  }
}

@compute @workgroup_size(${TW},${TH},1)
fn main(@builtin(workgroup_id) wid:vec3u, @builtin(local_invocation_index) lidx:u32){
  let lx = lidx % ${TW}u; let ly = lidx / ${TW}u;
  let lane = lidx % ${SUB}u; let wbase = (lidx / ${SUB}u) * ${REGION}u;
  let gx0 = wid.x * ${TW}u; let gy0 = wid.y * ${TH}u;
  let x = gx0 + lx; let y = gy0 + ly;
  let ocbase = wid.z * ${OCB}u;
  let plane = u.H * u.W; let inc9 = u.in_c*9u;

${accDecl}

  for(var ic=0u; ic<u.in_c; ic++){
    loadHalo(ic, gx0, gy0, plane, lidx);
${wregDecl}
${wregLoad}
${wregStore}
    workgroupBarrier();
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
