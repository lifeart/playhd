// v2: spatial (PX x PY) + channel (K) register blocking, shared-memory input tile.
// Workgroup = TX x TY threads, each thread computes PX x PY output pixels for K channels.
//   accumulators = K*PX*PY ; input window in regs = (PX+2)*(PY+2) per ic.
//   weights staged in smem (K*9), each weight reused across PX*PY pixels.
const env = (k: string, d: number) => { try { const v = Deno.env.get(k); return v ? parseInt(v) : d; } catch { return d; } };
const TX = env("TX", 16), TY = env("TY", 16), K = env("K", 8), PX = env("PX", 2), PY = env("PY", 1);
const OWX = TX * PX, OWY = TY * PY;       // output region per workgroup
const TWW = OWX + 2, TWH = OWY + 2, TILE = TWW * TWH, NT = TX * TY;

// build register-blocked inner: load input window regs, then 9 taps x K x (PX*PY) FMA
let loadWin = "";
for (let wy = 0; wy < PY + 2; wy++)
  for (let wx = 0; wx < PX + 2; wx++)
    loadWin += `    let in_${wx}_${wy} = tile[(oy+${wy}u)*${TWW}u + (ox+${wx}u)];\n`;
let inner = "";
for (let ky = 0; ky < 3; ky++)
  for (let kx = 0; kx < 3; kx++) {
    const tap = ky * 3 + kx;
    inner += `    {\n`;
    for (let k = 0; k < K; k++) inner += `      let w${k}=ws[${k * 9 + tap}u];\n`;
    for (let py = 0; py < PY; py++)
      for (let px = 0; px < PX; px++) {
        const a = (k: number) => `acc_${k}_${px}_${py}`;
        for (let k = 0; k < K; k++)
          inner += `      ${a(k)} += w${k}*in_${px + kx}_${py + ky};\n`;
      }
    inner += `    }\n`;
  }
let declAcc = "", initAcc = "", writeOut = "";
for (let py = 0; py < PY; py++)
  for (let px = 0; px < PX; px++)
    for (let k = 0; k < K; k++) {
      declAcc += `  var acc_${k}_${px}_${py}:f32;\n`;
      initAcc += `  acc_${k}_${px}_${py} = Wt[u.b_off + ocb + ${k}u];\n`;
    }
for (let py = 0; py < PY; py++)
  for (let px = 0; px < PX; px++) {
    writeOut += `  { let xx=x+${px}u; let yy=y+${py}u; if(xx<W && yy<H){ let po=yy*W+xx;\n`;
    for (let k = 0; k < K; k++)
      writeOut += `    { let oc=ocb+${k}u; if(oc<u.out_c){ var a=acc_${k}_${px}_${py}; if(u.has_prelu==1u){let s=Wt[u.prelu_off+oc]; if(a<0.0){a=a*s;}} fout[oc*plane+po]=a; } }\n`;
    writeOut += `  } }\n`;
  }

const code = `struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f32>;
@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
@group(0) @binding(2) var<storage,read> Wt:array<f32>;
@group(0) @binding(3) var<uniform> u:P;
var<workgroup> tile: array<f32, ${TILE}u>;
var<workgroup> ws: array<f32, ${K * 9}u>;
@compute @workgroup_size(${TX},${TY},1)
fn main(@builtin(workgroup_id) wg:vec3u, @builtin(local_invocation_id) lid:vec3u, @builtin(local_invocation_index) tid:u32){
  let W=u.W; let H=u.H; let plane=H*W;
  let bx = wg.x*${OWX}u; let by = wg.y*${OWY}u;
  let ox = lid.x*${PX}u; let oy = lid.y*${PY}u;
  let ocb = wg.z*${K}u;
  let x = bx+ox; let y = by+oy;
${declAcc}${initAcc}  let icN = u.in_c;
  for(var ic=0u; ic<icN; ic++){
    let baseI = ic*plane;
    for(var i=tid; i<${TILE}u; i+=${NT}u){
      let lx = i % ${TWW}u; let ly = i / ${TWW}u;
      let gx = i32(bx + lx) - 1; let gy = i32(by + ly) - 1;
      var v = 0.0;
      if(gx>=0 && gx<i32(W) && gy>=0 && gy<i32(H)){ v = fin[baseI + u32(gy)*W + u32(gx)]; }
      tile[i] = v;
    }
    let wbase = u.w_off + ic*9u; let i9 = u.in_c*9u;
    for(var j=tid; j<${K * 9}u; j+=${NT}u){ let kk=j/9u; let r=j%9u; ws[j]=Wt[wbase+(ocb+kk)*i9+r]; }
    workgroupBarrier();
${loadWin}${inner}    workgroupBarrier();
  }
${writeOut}}`;

export default {
  code,
  dispatch: (ly: any, H: number, W: number): [number, number, number] =>
    [Math.ceil(W / OWX), Math.ceil(H / OWY), Math.ceil(ly.out_c / K)],
};
