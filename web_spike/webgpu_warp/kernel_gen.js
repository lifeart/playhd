// Parametric compact-SR conv generator (browser ESM port of conv_opt/sweepgen.ts).
// genConv(opts) -> { code, dispatch, f16, threads, smemBytes, label }.
// opts: { TW=16, TH=16, OCB=32, DB=true, F16=false, ACC='f16'|'f32'|'hybrid', PROD='f16'|'f32' }
// Bindings/layout match the pipeline: 0 fin, 1 fout, 2 Wt, 3 uniform P; PLANAR c*H*W+y*W+x.
export function genConv(opts = {}) {
  const TW = opts.TW ?? 16, TH = opts.TH ?? 16, OCB = opts.OCB ?? 32;
  const DB = opts.DB ?? true, F16 = opts.F16 ?? false;
  const ACC = F16 ? (opts.ACC ?? 'f16') : 'f32';
  const PROD = F16 ? (opts.PROD ?? 'f16') : 'f32';
  const HW = TW + 2, HSZ = HW * (TH + 2), NT = TW * TH, G = OCB / 4, WSZ = G * 9;
  const ST = F16 ? 'f16' : 'f32', NB = DB ? 2 : 1, zeroS = F16 ? 'f16(0.0)' : '0.0';
  const rep = (f) => Array.from({ length: G }, (_, g) => f(g)).join('\n');

  const accDecl = ACC === 'hybrid'
    ? rep((g) => `  var a${g} = vec4<f32>(0.0);`)
    : rep((g) => `  var a${g} = ${ACC === 'f16' ? 'vec4<f16>(0.0)' : 'vec4<f32>(0.0)'};`);
  const macInto = (dst, accIsF32, g) => {
    const w = `sW[wBase + ${g * 9}u + wk]`;
    if (!F16) return `${dst} += ${w} * inv;`;
    if (PROD === 'f16') { const p = `(${w} * inv)`; return `${dst} += ${accIsF32 ? `vec4<f32>(${p})` : p};`; }
    const p = `(vec4<f32>(${w}) * f32(inv))`; return `${dst} += ${accIsF32 ? p : `vec4<f16>(${p})`};`;
  };
  const accBody = ACC === 'hybrid'
    ? rep((g) => `        ${macInto(`t${g}`, false, g)}`)
    : rep((g) => `        ${macInto(`a${g}`, ACC === 'f32', g)}`);
  const tapDecl = ACC === 'hybrid' ? rep((g) => `    var t${g} = vec4<f16>(0.0);`) : '';
  const tapFlush = ACC === 'hybrid' ? rep((g) => `    a${g} += vec4<f32>(t${g});`) : '';
  const writeBody = rep((g) => [0, 1, 2, 3].map((j) => {
    const ch = g * 4 + j, comp = ['x', 'y', 'z', 'w'][j];
    const av = F16 ? `f32(a${g}.${comp})` : `a${g}.${comp}`;
    const bw = F16 ? `f32(Wt[u.b_off+oc])` : `Wt[u.b_off+oc]`;
    const ps = F16 ? `f32(Wt[u.prelu_off+oc])` : `Wt[u.prelu_off+oc]`;
    const st = F16 ? `f16(v)` : `v`;
    return `    { let oc = ocbase + ${ch}u; if(oc < u.out_c){ var v = ${av} + ${bw}; if(u.has_prelu==1u && v<0.0){ v = v*${ps}; } fout[oc*plane + y*u.W + x] = ${st}; } }`;
  }).join('\n'));

  const loop = DB ? `
  loadTile(0u, 0u, 0u, gx0, gy0, ocbase, plane, inc9, lidx);
  workgroupBarrier();
  for(var ic=0u; ic<u.in_c; ic++){
    let c = ic & 1u; let sBase = c*${HSZ}u; let wBase = c*${WSZ}u;
    if(ic+1u < u.in_c){ loadTile(ic+1u, (1u-c)*${HSZ}u, (1u-c)*${WSZ}u, gx0, gy0, ocbase, plane, inc9, lidx); }
${tapDecl}
    for(var ky=0u; ky<3u; ky++){ for(var kx=0u; kx<3u; kx++){
      let inv = sIn[sBase + (ly+ky)*${HW}u + (lx+kx)]; let wk = ky*3u+kx;
${accBody}
    }}
${tapFlush}
    workgroupBarrier();
  }` : `
  for(var ic=0u; ic<u.in_c; ic++){
    loadTile(ic, 0u, 0u, gx0, gy0, ocbase, plane, inc9, lidx);
    workgroupBarrier();
    let sBase = 0u; let wBase = 0u;
${tapDecl}
    for(var ky=0u; ky<3u; ky++){ for(var kx=0u; kx<3u; kx++){
      let inv = sIn[sBase + (ly+ky)*${HW}u + (lx+kx)]; let wk = ky*3u+kx;
${accBody}
    }}
${tapFlush}
    workgroupBarrier();
  }`;

  const code = `${F16 ? 'enable f16;\n' : ''}
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<${ST}>;
@group(0) @binding(1) var<storage,read_write> fout:array<${ST}>;
@group(0) @binding(2) var<storage,read> Wt:array<${ST}>;
@group(0) @binding(3) var<uniform> u:P;
var<workgroup> sIn:array<${ST},${NB * HSZ}>;
var<workgroup> sW:array<vec4<${ST}>,${NB * WSZ}>;
fn loadTile(ic:u32, sBase:u32, wBase:u32, gx0:u32, gy0:u32, ocbase:u32, plane:u32, inc9:u32, lidx:u32){
  for(var t=lidx; t<${HSZ}u; t+=${NT}u){
    let hx = t % ${HW}u; let hy = t / ${HW}u;
    let xx = i32(gx0)+i32(hx)-1; let yy = i32(gy0)+i32(hy)-1;
    var v:${ST} = ${zeroS};
    if(xx>=0 && xx<i32(u.W) && yy>=0 && yy<i32(u.H)){ v = fin[ic*plane + u32(yy)*u.W + u32(xx)]; }
    sIn[sBase + t] = v;
  }
  for(var t=lidx; t<${WSZ}u; t+=${NT}u){
    let g = t / 9u; let k = t % 9u; let wb = u.w_off + ic*9u + k; let oc0 = ocbase + g*4u;
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
  let lx = lidx % ${TW}u; let ly = lidx / ${TW}u;
  let gx0 = wid.x * ${TW}u; let gy0 = wid.y * ${TH}u;
  let x = gx0 + lx; let y = gy0 + ly;
  let ocbase = wid.z * ${OCB}u; let plane = u.H * u.W; let inc9 = u.in_c*9u;
${accDecl}
${loop}
  if(x<u.W && y<u.H){
${writeBody}
  }
}`;
  const esz = F16 ? 2 : 4;
  return {
    code, f16: F16, threads: NT, OCB, TW, TH, DB, ACC,
    smemBytes: NB * HSZ * esz + NB * WSZ * 4 * esz,
    label: `${F16 ? 'f16' : 'f32'} OCB${OCB} ${TW}x${TH}${DB ? ' DB' : ''}${F16 && ACC !== 'f16' ? ' ACC=' + ACC : ''}`,
    dispatch: (ly, H, W) => [Math.ceil(W / TW), Math.ceil(H / TH), Math.ceil(ly.out_c / OCB)],
  };
}
