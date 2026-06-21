// ============================================================================
// COMBO = wtile structure (16x16 tile, weight+input shared cache, register-
// blocked UNROLLED vec4 accumulators, double-buffered) + fp16 storage/shared/MAC.
//
// vs wtile (f32): identical tiling/blocking, but fin/fout/Wt are array<f16>, the
// shared input halo is array<f16>, the shared weight tile is array<vec4<f16>>, and
// the MAC runs in f16 (2x Apple ALU + half the global+shared bandwidth). The UNROLLED
// vec4 accumulators stay in REGISTERS (not a spilling dyn-indexed array like the prior
// candidate_fp16*.ts, which got 0 win: 289 vs wtile 293 ms). That register-blocking is
// the whole point -- it's what lets fp16 actually pay off on top of the tiling.
//
// MEASURED at SIZE=256 (Deno wgpu->Metal, GPU contended), back-to-back vs wtile=288 ms:
//   OCB=64 ACC=f16 PROD=f16  -> 150.6 ms  (1.91x faster than wtile)  parity 2.69e-3  <- THIS
//   OCB=64 ACC=f32 PROD=f16  -> 212   ms  (1.37x)                    parity 3.98e-4  (tightest)
//   OCB=32 hybrid (f16 tap-sum -> f32 flush/ic) -> 259 ms            parity 7.46e-4  (mid)
//   OCB=64 dominates: the net is 64-ch so OCB=64 -> gz=1, each input channel loaded ONCE
//   (OCB=32 -> gz=2, 2x input reloads = 232 ms; OCB=48 -> ragged gz=2 = 242 ms).
//   f16 accumulate wins big because BOTH mul AND add are 2x on Apple + 16 vec4<f16> acc
//   = 32 regs (same as 8 vec4<f32>) so OCB=64 fits without throttling occupancy.
//   Double-buffer prefetch still helps in f16 (150.6 vs 156.3 single-buffer).
//   Parity 2.69e-3 mean / 2.04e-2 max passes the mean<1e-2 gate with 3.7x margin.
//
// To trade speed for the tightest parity, flip ACC below: "f32" -> 212 ms / 3.98e-4
// (squarely in the project's validated visually-identical band), "hybrid" -> 7.46e-4.
//
// Sweepable: OCB (16/32/64), ACC ('f32'|'f16'|'hybrid'), PROD ('f16'|'f32').
// ============================================================================
const TW = 16, TH = 16;      // pixel tile -> 256 threads
const OCB = 64;              // output channels per workgroup (multiple of 4); gz = ceil(out_c/OCB)
const ACC: "f32" | "f16" | "hybrid" = "f16";   // accumulator precision (see table above)
const PROD: "f16" | "f32" = "f16";  // multiply precision

const HW = TW + 2, HSZ = HW * (TH + 2);
const NT = TW * TH;
const G = OCB / 4;           // vec4 accumulators per thread
const WSZ = G * 9;           // vec4 weights in shared per input channel

// ----- accumulator declaration -----
// f32   : master vec4<f32> a_g, MAC straight into it (tightest parity, f32 adds = 1x ALU)
// f16   : master vec4<f16> a_g, MAC straight into it (fastest: f16 mul+add both 2x)
// hybrid: per-ic f16 tap-sum t_g (9 f16 MACs) flushed into f32 master a_g once/ic
//         -> mostly-f16 ALU but f32 cross-channel accumulation (parity between the two)
const accDecl = (() => {
  if (ACC === "hybrid")
    return Array.from({ length: G }, (_, g) => `  var a${g} = vec4<f32>(0.0);`).join("\n");
  const z = ACC === "f32" ? "vec4<f32>(0.0)" : "vec4<f16>(0.0)";
  return Array.from({ length: G }, (_, g) => `  var a${g} = ${z};`).join("\n");
})();

// MAC: sW is vec4<f16>, inv is f16. PROD f16 -> product in f16 then promote to acc type.
//      PROD f32 -> promote both to f32 first (more precise, loses f16 ALU).
const macInto = (dst: string, accIsF32: boolean, g: number) => {
  const w = `sW[wBase + ${g * 9}u + wk]`;
  if (PROD === "f16") {
    const prod = `(${w} * inv)`;              // vec4<f16>
    return `${dst} += ${accIsF32 ? `vec4<f32>(${prod})` : prod};`;
  } else {
    const prod = `(vec4<f32>(${w}) * f32(inv))`; // vec4<f32>
    return `${dst} += ${accIsF32 ? prod : `vec4<f16>(${prod})`};`;
  }
};
const accBody = ACC === "hybrid"
  ? Array.from({ length: G }, (_, g) => `        ${macInto(`t${g}`, false, g)}`).join("\n")
  : Array.from({ length: G }, (_, g) => `        ${macInto(`a${g}`, ACC === "f32", g)}`).join("\n");
// hybrid: declare f16 tap-sums (zeroed each ic) + flush into f32 masters after the 9 taps
const tapDecl = ACC === "hybrid"
  ? Array.from({ length: G }, (_, g) => `    var t${g} = vec4<f16>(0.0);`).join("\n") : "";
const tapFlush = ACC === "hybrid"
  ? Array.from({ length: G }, (_, g) => `    a${g} += vec4<f32>(t${g});`).join("\n") : "";

const writeBody = Array.from({ length: G }, (_, g) =>
  [0, 1, 2, 3].map((j) => {
    const ch = g * 4 + j;
    const comp = ["x", "y", "z", "w"][j];
    return `    { let oc = ocbase + ${ch}u; if(oc < u.out_c){ var v = f32(a${g}.${comp}) + f32(Wt[u.b_off+oc]); if(u.has_prelu==1u && v<0.0){ v = v*f32(Wt[u.prelu_off+oc]); } fout[oc*plane + y*u.W + x] = f16(v); } }`;
  }).join("\n")
).join("\n");

const code = `
enable f16;
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f16>;
@group(0) @binding(1) var<storage,read_write> fout:array<f16>;
@group(0) @binding(2) var<storage,read> Wt:array<f16>;
@group(0) @binding(3) var<uniform> u:P;

// double-buffered shared: [2] halos + [2] weight tiles (one being computed, one prefetched)
var<workgroup> sIn:array<f16,${2 * HSZ}>;
var<workgroup> sW:array<vec4<f16>,${2 * WSZ}>;

fn loadTile(ic:u32, sBase:u32, wBase:u32, gx0:u32, gy0:u32, ocbase:u32, plane:u32, inc9:u32, lidx:u32){
  for(var t=lidx; t<${HSZ}u; t+=${NT}u){
    let hx = t % ${HW}u;
    let hy = t / ${HW}u;
    let xx = i32(gx0)+i32(hx)-1;
    let yy = i32(gy0)+i32(hy)-1;
    var v:f16 = f16(0.0);
    if(xx>=0 && xx<i32(u.W) && yy>=0 && yy<i32(u.H)){ v = fin[ic*plane + u32(yy)*u.W + u32(xx)]; }
    sIn[sBase + t] = v;
  }
  for(var t=lidx; t<${WSZ}u; t+=${NT}u){
    let g = t / 9u;
    let k = t % 9u;
    let wb = u.w_off + ic*9u + k;
    let oc0 = ocbase + g*4u;
    var w = vec4<f16>(0.0);
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
    if(ic+1u < u.in_c){ loadTile(ic+1u, (1u-c)*${HSZ}u, (1u-c)*${WSZ}u, gx0, gy0, ocbase, plane, inc9, lidx); }
${tapDecl}
    for(var ky=0u; ky<3u; ky++){
      for(var kx=0u; kx<3u; kx++){
        let inv = sIn[sBase + (ly+ky)*${HW}u + (lx+kx)];
        let wk = ky*3u+kx;
${accBody}
      }
    }
${tapFlush}
    workgroupBarrier();
  }

  if(x<u.W && y<u.H){
${writeBody}
  }
}`;

export default {
  f16: true,
  code,
  dispatch: (ly: any, H: number, W: number) => [Math.ceil(W / TW), Math.ceil(H / TH), Math.ceil(ly.out_c / OCB)],
};
