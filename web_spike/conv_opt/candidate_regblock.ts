// Register-blocking conv: spatial (PX x PY) + output-channel (K) blocking with a shared-memory input tile.
//
// Each workgroup (TX x TY threads) computes a (TX*PX) x (TY*PY) output region for K output channels:
//   * The (TX*PX+2) x (TY*PY+2) input window for the current ic is staged ONCE in workgroup memory and
//     reused by every thread (3x3 windows overlap heavily across the tile -> ~ (TX*PX*TY*PY)/area fewer
//     global input reads).
//   * Each THREAD computes PX*PY pixels x K channels = PX*PY*K outputs held in registers. The (PX+2)x(PY+2)
//     input values it needs are loaded from the shared tile into registers ONCE and reused across all K
//     channels; the K*9 weights (staged in workgroup memory) are reused across all PX*PY pixels.
// Net: every loaded input feeds K register accumulators, every loaded weight feeds PX*PY accumulators ->
// high arithmetic intensity, ~10x over the zero-reuse naive at peak GPU clock, exact f32 parity.
//
// Best params found empirically (Deno/wgpu->Metal, Apple GPU): PX=2, PY=2, K=8, 16x16 workgroup (256
// threads = Apple cap; 32 accumulators/thread, no spill). K must divide 64 and 48 -> K in {1,2,4,8,16}.
// (Env vars TX/TY/K/PX/PY override for tuning; defaults are the winner and need no --allow-env.)
const env = (k: string, d: number) => { try { const v = Deno.env.get(k); return v ? parseInt(v) : d; } catch { return d; } };
const TX = env("TX", 16), TY = env("TY", 16), K = env("K", 8), PX = env("PX", 2), PY = env("PY", 2);
const OWX = TX * PX, OWY = TY * PY;
const TWW = OWX + 2, TILE = TWW * (OWY + 2), NT = TX * TY;

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
      for (let px = 0; px < PX; px++)
        for (let k = 0; k < K; k++)
          inner += `      acc_${k}_${px}_${py} = fma(w${k}, in_${px + kx}_${py + ky}, acc_${k}_${px}_${py});\n`;
    inner += `    }\n`;
  }
let declAcc = "", initAcc = "", writeOut = "";
for (let py = 0; py < PY; py++)
  for (let px = 0; px < PX; px++)
    for (let k = 0; k < K; k++) {
      declAcc += `  var acc_${k}_${px}_${py}:f32;\n`;
      initAcc += `  acc_${k}_${px}_${py} = Wt[bb + ${k}u];\n`;
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
  let ocb = wg.z*${K}u; let bb = u.b_off + ocb;
  let x = bx+ox; let y = by+oy;
${declAcc}${initAcc}  let icN = u.in_c; let i9 = u.in_c*9u;
  for(var ic=0u; ic<icN; ic++){
    let baseI = ic*plane;
    for(var i=tid; i<${TILE}u; i+=${NT}u){
      let lx = i % ${TWW}u; let ly = i / ${TWW}u;
      let gx = i32(bx + lx) - 1; let gy = i32(by + ly) - 1;
      var v = 0.0;
      if(gx>=0 && gx<i32(W) && gy>=0 && gy<i32(H)){ v = fin[baseI + u32(gy)*W + u32(gx)]; }
      tile[i] = v;
    }
    let wbase = u.w_off + ic*9u;
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
