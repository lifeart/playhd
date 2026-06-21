// v3: vec4-vectorized channel blocking + spatial blocking + shared input tile.
// K output channels handled as KG=K/4 vec4 groups. Each input value broadcasts into a
// vec4 FMA across 4 channels at once (4 MACs/instruction). Spatial PX x PY pixels per thread.
const env = (k: string, d: number) => { try { const v = Deno.env.get(k); return v ? parseInt(v) : d; } catch { return d; } };
const TX = env("TX", 16), TY = env("TY", 16), K = env("K", 4), PX = env("PX", 2), PY = env("PY", 2);
const KG = K / 4;
const OWX = TX * PX, OWY = TY * PY;
const TWW = OWX + 2, TWH = OWY + 2, TILE = TWW * TWH, NT = TX * TY;

const comp = ["x", "y", "z", "w"];
// load input window into registers
let loadWin = "";
for (let wy = 0; wy < PY + 2; wy++)
  for (let wx = 0; wx < PX + 2; wx++)
    loadWin += `    let in_${wx}_${wy} = tile[(oy+${wy}u)*${TWW}u + (ox+${wx}u)];\n`;
// inner: 9 taps x KG groups x (PX*PY) vec4 FMA
let inner = "";
for (let ky = 0; ky < 3; ky++)
  for (let kx = 0; kx < 3; kx++) {
    const tap = ky * 3 + kx;
    inner += `    {\n`;
    for (let g = 0; g < KG; g++) inner += `      let w${g}=wsv[${g * 9 + tap}u];\n`;
    for (let py = 0; py < PY; py++)
      for (let px = 0; px < PX; px++)
        for (let g = 0; g < KG; g++)
          inner += `      acc_${g}_${px}_${py} = fma(w${g}, vec4<f32>(in_${px + kx}_${py + ky}), acc_${g}_${px}_${py});\n`;
    inner += `    }\n`;
  }
let declAcc = "", initAcc = "", writeOut = "";
for (let py = 0; py < PY; py++)
  for (let px = 0; px < PX; px++)
    for (let g = 0; g < KG; g++) {
      declAcc += `  var acc_${g}_${px}_${py}:vec4<f32>;\n`;
      initAcc += `  acc_${g}_${px}_${py} = vec4<f32>(Wt[bb+${g * 4}u],Wt[bb+${g * 4 + 1}u],Wt[bb+${g * 4 + 2}u],Wt[bb+${g * 4 + 3}u]);\n`;
    }
for (let py = 0; py < PY; py++)
  for (let px = 0; px < PX; px++) {
    writeOut += `  { let xx=x+${px}u; let yy=y+${py}u; if(xx<W && yy<H){ let po=yy*W+xx;\n`;
    for (let g = 0; g < KG; g++)
      for (let c = 0; c < 4; c++)
        writeOut += `    { let oc=ocb+${g * 4 + c}u; if(oc<u.out_c){ var a=acc_${g}_${px}_${py}.${comp[c]}; if(u.has_prelu==1u){let s=Wt[u.prelu_off+oc]; if(a<0.0){a=a*s;}} fout[oc*plane+po]=a; } }\n`;
    writeOut += `  } }\n`;
  }

const code = `struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f32>;
@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
@group(0) @binding(2) var<storage,read> Wt:array<f32>;
@group(0) @binding(3) var<uniform> u:P;
var<workgroup> tile: array<f32, ${TILE}u>;
var<workgroup> wsv: array<vec4<f32>, ${KG * 9}u>;
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
    for(var j=tid; j<${KG * 9}u; j+=${NT}u){
      let g=j/9u; let tap=j%9u; let b=wbase+(ocb+g*4u)*i9+tap;
      wsv[j]=vec4<f32>(Wt[b], Wt[b+i9], Wt[b+2u*i9], Wt[b+3u*i9]);
    }
    workgroupBarrier();
${loadWin}${inner}    workgroupBarrier();
  }
${writeOut}}`;

export default {
  code,
  dispatch: (ly: any, H: number, W: number): [number, number, number] =>
    [Math.ceil(W / OWX), Math.ceil(H / OWY), Math.ceil(ly.out_c / K)],
};
