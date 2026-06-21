// ============================================================================
// WEIGHT shared-memory tiled 3x3 conv (f32, bit-exact parity).
//
// Strategy (the winning combination found by sweeping in the Deno harness):
//   * Workgroup = TW x TH pixel tile (16x16 = 256 threads). Each thread owns ONE
//     output pixel and computes OCB output channels for it, held as G=OCB/4
//     vec4<f32> accumulators that are FULLY UNROLLED on the JS side so they live
//     in registers (dynamic-indexed accumulator arrays spill -> ~0.6x; unrolled
//     vec4 -> 10x+). gz = ceil(out_c/OCB) covers the channel chunks.
//   * The weights for the current input channel are loaded ONCE into workgroup
//     shared memory (as vec4 over 4 oc) and reused by every pixel in the tile;
//     the input halo (TW+2)x(TH+2) is likewise cached and reused across the OCB
//     channels. This is the core "load weights once, reuse across a spatial tile"
//     idea -- the naive kernel reloads all weights for every output pixel.
//   * OCB=32 (gz=2) beat OCB=64 (gz=1) and OCB=16 (gz=4): 64 channels/thread is
//     16 vec4 accumulators = heavy register pressure that throttles occupancy;
//     16 channels needs gz=4 input reloads. OCB=32 (8 vec4 acc) is the sweet spot
//     -- enough occupancy, only 2x input reload.
//   * DOUBLE BUFFERING: while computing input-channel ic from shared buffer A we
//     prefetch ic+1 into buffer B -> 1 barrier/ic and global loads overlap the
//     FMA compute (~4% faster at 256, tie at 128).
//
// Bindings/layout/math per the harness spec; PLANAR features, zero-pad borders,
// optional PReLU. Parity vs naive: mean|Δ|=0, max=0 (bit-identical) at 128 & 256.
// Harness (Deno wgpu, GPU contended): ~10.5x @128, ~12.9x @256 vs naive.
// ============================================================================
const TW = 16, TH = 16;      // pixel tile -> 256 threads (= max invocations)
const OCB = 32;              // output channels per workgroup (multiple of 4); gz = out_c/OCB
const HW = TW + 2, HSZ = HW * (TH + 2);
const NT = TW * TH;
const G = OCB / 4;           // vec4 accumulators per thread
const WSZ = G * 9;           // vec4 weights in shared per input channel

const accDecl = Array.from({ length: G }, (_, g) => `  var a${g} = vec4<f32>(0.0);`).join("\n");
const accBody = Array.from({ length: G }, (_, g) => `        a${g} += sW[wBase + ${g * 9}u + wk] * inv;`).join("\n");
const writeBody = Array.from({ length: G }, (_, g) =>
  [0, 1, 2, 3].map((j) => {
    const ch = g * 4 + j;
    const comp = ["x", "y", "z", "w"][j];
    return `    { let oc = ocbase + ${ch}u; if(oc < u.out_c){ var v = a${g}.${comp} + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v = v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x] = v; } }`;
  }).join("\n")
).join("\n");

const code = `
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f32>;
@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
@group(0) @binding(2) var<storage,read> Wt:array<f32>;
@group(0) @binding(3) var<uniform> u:P;

// double-buffered shared: [2] halos + [2] weight tiles (one being computed, one prefetched)
var<workgroup> sIn:array<f32,${2 * HSZ}>;
var<workgroup> sW:array<vec4<f32>,${2 * WSZ}>;

fn loadTile(ic:u32, sBase:u32, wBase:u32, gx0:u32, gy0:u32, ocbase:u32, plane:u32, inc9:u32, lidx:u32){
  for(var t=lidx; t<${HSZ}u; t+=${NT}u){
    let hx = t % ${HW}u;
    let hy = t / ${HW}u;
    let xx = i32(gx0)+i32(hx)-1;
    let yy = i32(gy0)+i32(hy)-1;
    var v = 0.0;
    if(xx>=0 && xx<i32(u.W) && yy>=0 && yy<i32(u.H)){ v = fin[ic*plane + u32(yy)*u.W + u32(xx)]; }
    sIn[sBase + t] = v;
  }
  for(var t=lidx; t<${WSZ}u; t+=${NT}u){
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

@compute @workgroup_size(${TW},${TH},1)
fn main(@builtin(workgroup_id) wid:vec3u, @builtin(local_invocation_index) lidx:u32){
  let lx = lidx % ${TW}u;
  let ly = lidx / ${TW}u;
  let gx0 = wid.x * ${TW}u;
  let gy0 = wid.y * ${TH}u;
  let x = gx0 + lx;
  let y = gy0 + ly;
  let ocbase = wid.z * ${OCB}u;
  let plane = u.H * u.W;
  let inc9 = u.in_c*9u;

${accDecl}

  loadTile(0u, 0u, 0u, gx0, gy0, ocbase, plane, inc9, lidx);
  workgroupBarrier();
  for(var ic=0u; ic<u.in_c; ic++){
    let c = ic & 1u;
    let sBase = c*${HSZ}u;
    let wBase = c*${WSZ}u;
    // prefetch next input channel into the alternate buffer (overlaps the FMAs below)
    if(ic+1u < u.in_c){ loadTile(ic+1u, (1u-c)*${HSZ}u, (1u-c)*${WSZ}u, gx0, gy0, ocbase, plane, inc9, lidx); }
    for(var ky=0u; ky<3u; ky++){
      for(var kx=0u; kx<3u; kx++){
        let inv = sIn[sBase + (ly+ky)*${HW}u + (lx+kx)];
        let wk = ky*3u+kx;
${accBody}
      }
    }
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
