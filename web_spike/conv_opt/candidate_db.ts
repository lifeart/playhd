// Double-buffered weight-tiled conv: prefetch next ic's halo+weights into the alternate
// shared buffer while computing the current ic -> 1 barrier/ic + load/compute overlap.
const TW = 16, TH = 16;
const OCB = 32;
const HW = TW + 2, HH = TH + 2, HSZ = HW * HH;
const NT = TW * TH;
const G = OCB / 4;
const WSZ = G * 9;

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
