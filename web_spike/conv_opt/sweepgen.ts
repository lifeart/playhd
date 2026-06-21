// ============================================================================
// PARAMETRIC conv generator for the portability sweep. Reproduces wtile (f32) and
// combo (f16) and everything in between, driven by env vars so one file covers the
// whole grid through the existing bench.ts harness:
//   TW,TH      pixel tile / workgroup dims (threads = TW*TH)
//   OCB        output channels per workgroup (mult of 4); gz=ceil(out_c/OCB)
//   DB         1=double-buffered prefetch, 0=single shared buffer + 2 barriers/ic
//   F16        1=f16 storage/shared path, 0=pure f32 (bit-exact wtile)
//   ACC        f16 path accumulator: f32|f16|hybrid (ignored when F16=0 -> f32)
//   PROD       f16 path multiply precision: f16|f32 (ignored when F16=0)
//
//   OCB=32 TW=16 TH=16 DB=1 F16=0 deno run --allow-read --allow-env bench.ts sweepgen.ts 256
// ============================================================================
const env = (k: string, d: string) => (Deno.env.get(k) ?? d);
const TW = parseInt(env("TW", "16")), TH = parseInt(env("TH", "16"));
const OCB = parseInt(env("OCB", "32"));
const DB = env("DB", "1") === "1";
const F16 = env("F16", "0") === "1";
const ACC = (F16 ? env("ACC", "f16") : "f32") as "f32" | "f16" | "hybrid";
const PROD = (F16 ? env("PROD", "f16") : "f32") as "f16" | "f32";

const HW = TW + 2, HSZ = HW * (TH + 2);
const NT = TW * TH;
const G = OCB / 4;          // vec4 accumulators per thread
const WSZ = G * 9;          // vec4 weights in shared per input channel
const ST = F16 ? "f16" : "f32";         // storage/shared scalar type
const NB = DB ? 2 : 1;                  // shared buffers

// ---- accumulator declaration ----
const accDecl = (() => {
  if (ACC === "hybrid") return Array.from({ length: G }, (_, g) => `  var a${g} = vec4<f32>(0.0);`).join("\n");
  const z = ACC === "f16" ? "vec4<f16>(0.0)" : "vec4<f32>(0.0)";
  return Array.from({ length: G }, (_, g) => `  var a${g} = ${z};`).join("\n");
})();

// ---- MAC ----
const macInto = (dst: string, accIsF32: boolean, g: number) => {
  const w = `sW[wBase + ${g * 9}u + wk]`;
  if (!F16) return `${dst} += ${w} * inv;`;          // pure f32: sW vec4<f32>, inv f32
  if (PROD === "f16") { const p = `(${w} * inv)`; return `${dst} += ${accIsF32 ? `vec4<f32>(${p})` : p};`; }
  const p = `(vec4<f32>(${w}) * f32(inv))`; return `${dst} += ${accIsF32 ? p : `vec4<f16>(${p})`};`;
};
const accBody = ACC === "hybrid"
  ? Array.from({ length: G }, (_, g) => `        ${macInto(`t${g}`, false, g)}`).join("\n")
  : Array.from({ length: G }, (_, g) => `        ${macInto(`a${g}`, ACC === "f32", g)}`).join("\n");
const tapDecl = ACC === "hybrid" ? Array.from({ length: G }, (_, g) => `    var t${g} = vec4<f16>(0.0);`).join("\n") : "";
const tapFlush = ACC === "hybrid" ? Array.from({ length: G }, (_, g) => `    a${g} += vec4<f32>(t${g});`).join("\n") : "";

const writeBody = Array.from({ length: G }, (_, g) =>
  [0, 1, 2, 3].map((j) => {
    const ch = g * 4 + j, comp = ["x", "y", "z", "w"][j];
    const av = F16 ? `f32(a${g}.${comp})` : `a${g}.${comp}`;
    const bw = F16 ? `f32(Wt[u.b_off+oc])` : `Wt[u.b_off+oc]`;
    const ps = F16 ? `f32(Wt[u.prelu_off+oc])` : `Wt[u.prelu_off+oc]`;
    const st = F16 ? `f16(v)` : `v`;
    return `    { let oc = ocbase + ${ch}u; if(oc < u.out_c){ var v = ${av} + ${bw}; if(u.has_prelu==1u && v<0.0){ v = v*${ps}; } fout[oc*plane + y*u.W + x] = ${st}; } }`;
  }).join("\n")
).join("\n");

const zeroS = F16 ? "f16(0.0)" : "0.0";

// ---- compute body: double-buffered vs single ----
const computeLoop = DB ? `
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
  }` : `
  for(var ic=0u; ic<u.in_c; ic++){
    loadTile(ic, 0u, 0u, gx0, gy0, ocbase, plane, inc9, lidx);
    workgroupBarrier();
    let sBase = 0u;
    let wBase = 0u;
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
  }`;

const code = `${F16 ? "enable f16;\n" : ""}
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<${ST}>;
@group(0) @binding(1) var<storage,read_write> fout:array<${ST}>;
@group(0) @binding(2) var<storage,read> Wt:array<${ST}>;
@group(0) @binding(3) var<uniform> u:P;

var<workgroup> sIn:array<${ST},${NB * HSZ}>;
var<workgroup> sW:array<vec4<${ST}>,${NB * WSZ}>;

fn loadTile(ic:u32, sBase:u32, wBase:u32, gx0:u32, gy0:u32, ocbase:u32, plane:u32, inc9:u32, lidx:u32){
  for(var t=lidx; t<${HSZ}u; t+=${NT}u){
    let hx = t % ${HW}u;
    let hy = t / ${HW}u;
    let xx = i32(gx0)+i32(hx)-1;
    let yy = i32(gy0)+i32(hy)-1;
    var v:${ST} = ${zeroS};
    if(xx>=0 && xx<i32(u.W) && yy>=0 && yy<i32(u.H)){ v = fin[ic*plane + u32(yy)*u.W + u32(xx)]; }
    sIn[sBase + t] = v;
  }
  for(var t=lidx; t<${WSZ}u; t+=${NT}u){
    let g = t / 9u;
    let k = t % 9u;
    let wb = u.w_off + ic*9u + k;
    let oc0 = ocbase + g*4u;
    var w = vec4<${ST}>(${zeroS});
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
${computeLoop}

  if(x<u.W && y<u.H){
${writeBody}
  }
}`;

// shared-mem byte budget (for the report): sIn scalar + sW vec4
const esz = F16 ? 2 : 4;
const smemBytes = NB * HSZ * esz + NB * WSZ * 4 * esz;
if (env("SMEM", "0") === "1") console.error(`[cfg] TW=${TW} TH=${TH} threads=${NT} OCB=${OCB} G=${G} DB=${DB} F16=${F16} ACC=${ACC} PROD=${PROD} smem=${smemBytes}B`);

export default {
  f16: F16,
  code,
  dispatch: (ly: any, H: number, W: number) => [Math.ceil(W / TW), Math.ceil(H / TH), Math.ceil(ly.out_c / OCB)],
};
