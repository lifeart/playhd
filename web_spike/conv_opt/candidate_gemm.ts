// GEMM-style register-blocked conv. Workgroup computes a BX x BY pixel tile for OCB
// output channels. Threads = TX x TY; each thread computes PX x PY pixels (PX=BX/TX,
// PY=BY/TY) for G=OCB/4 vec4 channel-groups -> PX*PY*G vec4 accumulators (unrolled).
// Shared holds the (BX+2)x(BY+2) input halo + OCB weights (vec4/4oc) per ic; gz=out_c/OCB.
const BX = 16, BY = 16;      // pixel tile
const TX = 8, TY = 8;        // threads
const OCB = 16;              // output channels per workgroup (mult of 4)
const PX = BX / TX, PY = BY / TY;
const HW = BX + 2, HSZ = HW * (BY + 2);
const NT = TX * TY;
const G = OCB / 4;
const WSZ = G * 9;

const accDecl = [];
for (let py = 0; py < PY; py++) for (let px = 0; px < PX; px++) for (let g = 0; g < G; g++)
  accDecl.push(`  var a_${py}_${px}_${g} = vec4<f32>(0.0);`);

const innerK = () => {
  const L: string[] = [];
  // load PX*PY input taps for this (ky,kx)
  for (let py = 0; py < PY; py++) for (let px = 0; px < PX; px++)
    L.push(`        let in_${py}_${px} = sIn[(py0+${py}u+ky)*${HW}u + (px0+${px}u+kx)];`);
  // load G weights for this k
  for (let g = 0; g < G; g++) L.push(`        let w_${g} = sW[${g * 9}u + wk];`);
  for (let py = 0; py < PY; py++) for (let px = 0; px < PX; px++) for (let g = 0; g < G; g++)
    L.push(`        a_${py}_${px}_${g} += w_${g} * in_${py}_${px};`);
  return L.join("\n");
};

const writeBody = () => {
  const L: string[] = [];
  for (let py = 0; py < PY; py++) for (let px = 0; px < PX; px++) {
    L.push(`  { let x = px0+${px}u; let y = py0+${py}u; if(x<u.W && y<u.H){`);
    for (let g = 0; g < G; g++) for (let j = 0; j < 4; j++) {
      const ch = g * 4 + j, comp = ["x", "y", "z", "w"][j];
      L.push(`    { let oc = ocbase + ${ch}u; if(oc<u.out_c){ var v = a_${py}_${px}_${g}.${comp} + Wt[u.b_off+oc]; if(u.has_prelu==1u && v<0.0){ v=v*Wt[u.prelu_off+oc]; } fout[oc*plane + y*u.W + x]=v; } }`);
    }
    L.push(`  } }`);
  }
  return L.join("\n");
};

const code = `
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f32>;
@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
@group(0) @binding(2) var<storage,read> Wt:array<f32>;
@group(0) @binding(3) var<uniform> u:P;

var<workgroup> sIn:array<f32,${HSZ}>;
var<workgroup> sW:array<vec4<f32>,${WSZ}>;

@compute @workgroup_size(${TX},${TY},1)
fn main(@builtin(workgroup_id) wid:vec3u, @builtin(local_invocation_index) lidx:u32){
  let tlx = lidx % ${TX}u;
  let tly = lidx / ${TX}u;
  let gx0 = wid.x * ${BX}u;
  let gy0 = wid.y * ${BY}u;
  let px0 = gx0 + tlx*${PX}u;
  let py0 = gy0 + tly*${PY}u;
  let ocbase = wid.z * ${OCB}u;
  let plane = u.H * u.W;
  let inc9 = u.in_c*9u;

${accDecl.join("\n")}

  for(var ic=0u; ic<u.in_c; ic++){
    let inpl = ic*plane;
    for(var t=lidx; t<${HSZ}u; t+=${NT}u){
      let hx = t % ${HW}u;
      let hy = t / ${HW}u;
      let xx = i32(gx0)+i32(hx)-1;
      let yy = i32(gy0)+i32(hy)-1;
      var v = 0.0;
      if(xx>=0 && xx<i32(u.W) && yy>=0 && yy<i32(u.H)){ v = fin[inpl + u32(yy)*u.W + u32(xx)]; }
      sIn[t] = v;
    }
    let wbase = u.w_off + ic*9u;
    for(var t=lidx; t<${WSZ}u; t+=${NT}u){
      let g = t / 9u;
      let k = t % 9u;
      let oc0 = ocbase + g*4u;
      var w = vec4<f32>(0.0);
      if(oc0+0u<u.out_c){ w.x = Wt[wbase + (oc0+0u)*inc9 + k]; }
      if(oc0+1u<u.out_c){ w.y = Wt[wbase + (oc0+1u)*inc9 + k]; }
      if(oc0+2u<u.out_c){ w.z = Wt[wbase + (oc0+2u)*inc9 + k]; }
      if(oc0+3u<u.out_c){ w.w = Wt[wbase + (oc0+3u)*inc9 + k]; }
      sW[t] = w;
    }
    workgroupBarrier();
    for(var ky=0u; ky<3u; ky++){
      for(var kx=0u; kx<3u; kx++){
        let wk = ky*3u+kx;
${innerK()}
      }
    }
    workgroupBarrier();
  }

${writeBody()}
}`;

export default {
  code,
  dispatch: (ly: any, H: number, W: number) => [Math.ceil(W / BX), Math.ceil(H / BY), Math.ceil(ly.out_c / OCB)],
};
