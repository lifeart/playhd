// span_driver.ts
function genSpanWGSL(T) {
  const head = T === "f16" ? "enable f16;\n" : "";
  return `${head}
struct P { H:u32, W:u32, in_c:u32, out_c:u32, w_off:u32, b_off:u32, p6:u32, p7:u32 };
@group(0) @binding(0) var<storage,read>       fin  : array<${T}>;
@group(0) @binding(1) var<storage,read_write> fout : array<${T}>;
@group(0) @binding(2) var<storage,read>       Wt   : array<${T}>;
@group(0) @binding(3) var<uniform>            u    : P;
@group(0) @binding(4) var<storage,read>       fx   : array<${T}>;   // gate's x input (dummy otherwise)

// generic 3x3 conv, zero-pad 1 (== PyTorch padding=1 cross-correlation). no activation.
@compute @workgroup_size(8,8,1) fn conv3x3(@builtin(global_invocation_id) g:vec3u){
  let x=i32(g.x); let y=i32(g.y); let oc=i32(g.z);
  if(x>=i32(u.W)||y>=i32(u.H)||oc>=i32(u.out_c)){ return; }
  var acc = f32(Wt[u.b_off+u32(oc)]);
  let bw = u.w_off + u32(oc)*u.in_c*9u;
  for(var ic=0u; ic<u.in_c; ic++){
    let pl=ic*u.H*u.W; let wic=bw+ic*9u;
    for(var ky=0; ky<3; ky++){ let yy=y+ky-1; if(yy<0||yy>=i32(u.H)){ continue; }
      for(var kx=0; kx<3; kx++){ let xx=x+kx-1; if(xx<0||xx>=i32(u.W)){ continue; }
        acc += f32(Wt[wic+u32(ky*3+kx)]) * f32(fin[pl+u32(yy)*u.W+u32(xx)]); } } }
  fout[u32(oc)*u.H*u.W + u32(y)*u.W + u32(x)] = ${T}(acc);
}

// generic 1x1 conv (conv_cat 192->48). weight layout (oc, ic).
@compute @workgroup_size(8,8,1) fn conv1x1(@builtin(global_invocation_id) g:vec3u){
  let x=g.x; let y=g.y; let oc=g.z;
  if(x>=u.W||y>=u.H||oc>=u.out_c){ return; }
  let pl=u.H*u.W; let p=y*u.W+x;
  var acc = f32(Wt[u.b_off+oc]);
  let bw = u.w_off + oc*u.in_c;
  for(var ic=0u; ic<u.in_c; ic++){ acc += f32(Wt[bw+ic]) * f32(fin[ic*pl+p]); }
  fout[oc*pl+p] = ${T}(acc);
}

// SiLU: x*sigmoid(x). n = in_c*H*W (in_c carries the channel count).
@compute @workgroup_size(64,1,1) fn silu(@builtin(global_invocation_id) g:vec3u){
  let i=g.x; let n=u.in_c*u.H*u.W; if(i>=n){ return; }
  let v=f32(fin[i]); fout[i]=${T}(v*(1.0/(1.0+exp(-v))));
}

// SPAB gate: (o3 + x) * (sigmoid(o3) - 0.5). fin=o3, fx=x.
@compute @workgroup_size(64,1,1) fn gate(@builtin(global_invocation_id) g:vec3u){
  let i=g.x; let n=u.in_c*u.H*u.W; if(i>=n){ return; }
  let o3=f32(fin[i]); let xv=f32(fx[i]);
  fout[i]=${T}((o3+xv)*(1.0/(1.0+exp(-o3))-0.5));
}

// PixelShuffle r=2: in (12ch, H,W) -> out (3ch, 2H,2W).
// out[oc, oy, ox] = in[oc*4 + (oy%2)*2 + (ox%2), oy/2, ox/2].
@compute @workgroup_size(8,8,1) fn pshuffle(@builtin(global_invocation_id) g:vec3u){
  let ox=g.x; let oy=g.y; let oc=g.z;
  let OW=u.W*2u; let OH=u.H*2u;
  if(ox>=OW||oy>=OH||oc>=3u){ return; }
  let xx=ox/2u; let yy=oy/2u;
  let ic = oc*4u + (oy%2u)*2u + (ox%2u);
  let lrpl=u.H*u.W;
  fout[oc*(OW*OH) + oy*OW + ox] = ${T}(f32(fin[ic*lrpl + yy*u.W + xx]));
}
`;
}
var F16 = globalThis.Float16Array;
var toF16 = (f) => new Uint16Array(new F16(f).buffer);

// ../webgpu_warp/kernel_gen.js
function genConv(opts = {}) {
  const TW = opts.TW ?? 16, TH = opts.TH ?? 16, OCB = opts.OCB ?? 32;
  const DB = opts.DB ?? true, F162 = opts.F16 ?? false;
  const ACC = F162 ? opts.ACC ?? "f16" : "f32";
  const PROD = F162 ? opts.PROD ?? "f16" : "f32";
  const HW = TW + 2, HSZ = HW * (TH + 2), NT = TW * TH, G = OCB / 4, WSZ = G * 9;
  const ST = F162 ? "f16" : "f32", NB = DB ? 2 : 1, zeroS = F162 ? "f16(0.0)" : "0.0";
  const rep = (f) => Array.from({ length: G }, (_, g) => f(g)).join("\n");
  const accDecl = ACC === "hybrid" ? rep((g) => `  var a${g} = vec4<f32>(0.0);`) : rep((g) => `  var a${g} = ${ACC === "f16" ? "vec4<f16>(0.0)" : "vec4<f32>(0.0)"};`);
  const macInto = (dst, accIsF32, g) => {
    const w = `sW[wBase + ${g * 9}u + wk]`;
    if (!F162) return `${dst} += ${w} * inv;`;
    if (PROD === "f16") {
      const p2 = `(${w} * inv)`;
      return `${dst} += ${accIsF32 ? `vec4<f32>(${p2})` : p2};`;
    }
    const p = `(vec4<f32>(${w}) * f32(inv))`;
    return `${dst} += ${accIsF32 ? p : `vec4<f16>(${p})`};`;
  };
  const accBody = ACC === "hybrid" ? rep((g) => `        ${macInto(`t${g}`, false, g)}`) : rep((g) => `        ${macInto(`a${g}`, ACC === "f32", g)}`);
  const tapDecl = ACC === "hybrid" ? rep((g) => `    var t${g} = vec4<f16>(0.0);`) : "";
  const tapFlush = ACC === "hybrid" ? rep((g) => `    a${g} += vec4<f32>(t${g});`) : "";
  const writeBody = rep((g) => [0, 1, 2, 3].map((j) => {
    const ch = g * 4 + j, comp = ["x", "y", "z", "w"][j];
    const av = F162 ? `f32(a${g}.${comp})` : `a${g}.${comp}`;
    const bw = F162 ? `f32(Wt[u.b_off+oc])` : `Wt[u.b_off+oc]`;
    const ps = F162 ? `f32(Wt[u.prelu_off+oc])` : `Wt[u.prelu_off+oc]`;
    const st = F162 ? `f16(v)` : `v`;
    return `    { let oc = ocbase + ${ch}u; if(oc < u.out_c){ var v = ${av} + ${bw}; if(u.has_prelu==1u && v<0.0){ v = v*${ps}; } fout[oc*plane + y*u.W + x] = ${st}; } }`;
  }).join("\n"));
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
  const code = `${F162 ? "enable f16;\n" : ""}
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
  const esz = F162 ? 2 : 4;
  return {
    code,
    f16: F162,
    threads: NT,
    OCB,
    TW,
    TH,
    DB,
    ACC,
    smemBytes: NB * HSZ * esz + NB * WSZ * 4 * esz,
    label: `${F162 ? "f16" : "f32"} OCB${OCB} ${TW}x${TH}${DB ? " DB" : ""}${F162 && ACC !== "f16" ? " ACC=" + ACC : ""}`,
    dispatch: (ly, H, W) => [Math.ceil(W / TW), Math.ceil(H / TH), Math.ceil(ly.out_c / OCB)]
  };
}

// span_driver_fast.ts
var ELEM = ["conv1x1", "silu", "gate", "pshuffle"];
var F16C = globalThis.Float16Array;
async function initSpanGPUFast(device, weightsF32, spec, f16, convOpts = {}) {
  const T = f16 ? "f16" : "f32";
  device.pushErrorScope("validation");
  const elemCode = genSpanWGSL(T);
  const elemMod = device.createShaderModule({ code: elemCode });
  const ci1 = await elemMod.getCompilationInfo?.();
  if (ci1) {
    for (const m of ci1.messages) if (m.type === "error") throw new Error("elem WGSL: " + m.message + " @line " + m.lineNum);
  }
  const elemBgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
  ] });
  const elemPl = device.createPipelineLayout({ bindGroupLayouts: [elemBgl] });
  const elemPipes = {};
  for (const e of ELEM) elemPipes[e] = device.createComputePipeline({ layout: elemPl, compute: { module: elemMod, entryPoint: e } });
  const OCB = convOpts.OCB ?? 48, TW = convOpts.TW ?? 16, TH = convOpts.TH ?? 16;
  const ACC = f16 ? convOpts.ACC ?? "f16" : "f32";
  const conv = genConv({ OCB, TW, TH, F16: f16, ACC, PROD: f16 ? "f16" : "f32", DB: true });
  const convMod = device.createShaderModule({ code: conv.code });
  const ci2 = await convMod.getCompilationInfo?.();
  if (ci2) {
    for (const m of ci2.messages) if (m.type === "error") throw new Error("conv WGSL: " + m.message + " @line " + m.lineNum);
  }
  const convBgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
  ] });
  const convPl = device.createPipelineLayout({ bindGroupLayouts: [convBgl] });
  const convPipe = device.createComputePipeline({ layout: convPl, compute: { module: convMod, entryPoint: "main" } });
  const eb = f16 ? 2 : 4;
  const wbytes = f16 ? toF16(weightsF32) : weightsF32;
  const Wbuf = device.createBuffer({ size: wbytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(Wbuf, 0, wbytes.buffer, wbytes.byteOffset, wbytes.byteLength);
  const err = await device.popErrorScope();
  if (err) throw new Error("initFast validation: " + err.message);
  return {
    device,
    f16,
    eb,
    spec,
    elemPipes,
    elemBgl,
    convPipe,
    convBgl,
    convTW: TW,
    convTH: TH,
    convOCB: OCB,
    convLabel: conv.label,
    Wbuf,
    weightsF32
  };
}
async function runSpanFast(g, lrPlanarF32, trials = 5, warmup = 2) {
  const { device, eb, elemPipes, elemBgl, convPipe, convBgl, Wbuf, spec, convTW, convTH, convOCB } = g;
  const { H, W } = spec, plane = H * W, feat = 48;
  const mkbuf = (ch) => device.createBuffer({ size: ch * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const IN = mkbuf(3);
  const F = mkbuf(feat), B1 = mkbuf(feat), B5_2 = mkbuf(feat), B6 = mkbuf(feat);
  const Pb = mkbuf(feat), Qb = mkbuf(feat), sA = mkbuf(feat), sB = mkbuf(feat), sC = mkbuf(feat);
  const CAT = mkbuf(192), COUT = mkbuf(feat), U12 = mkbuf(12);
  const OUT = device.createBuffer({ size: 3 * (2 * H) * (2 * W) * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const inBytes = g.f16 ? toF16(lrPlanarF32) : lrPlanarF32;
  device.queue.writeBuffer(IN, 0, inBytes.buffer, inBytes.byteOffset, inBytes.byteLength);
  const mkU = (Hh, Ww, in_c, out_c, w_off, b_off) => {
    const u = new Uint32Array([Hh, Ww, in_c, out_c, w_off >>> 0, b_off >>> 0, 0, 0]);
    const ub = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ub, 0, u);
    return ub;
  };
  const bgConv = (fin, fout, ub) => device.createBindGroup({ layout: convBgl, entries: [
    { binding: 0, resource: { buffer: fin } },
    { binding: 1, resource: { buffer: fout } },
    { binding: 2, resource: { buffer: Wbuf } },
    { binding: 3, resource: { buffer: ub } }
  ] });
  const bgElem = (pipe, fin, fout, fx, ub) => device.createBindGroup({ layout: elemBgl, entries: [
    { binding: 0, resource: { buffer: fin } },
    { binding: 1, resource: { buffer: fout } },
    { binding: 2, resource: { buffer: Wbuf } },
    { binding: 3, resource: { buffer: ub } },
    { binding: 4, resource: { buffer: fx } }
  ] });
  const ops = [];
  const W3 = (name, fin, fout, Hh = H, Ww = W) => {
    const w = spec.weights[name];
    const ub = mkU(Hh, Ww, w.in_c, w.out_c, w.w_off, w.b_off);
    ops.push({
      pipe: convPipe,
      bg: bgConv(fin, fout, ub),
      gx: Math.ceil(Ww / convTW),
      gy: Math.ceil(Hh / convTH),
      gz: Math.ceil(w.out_c / convOCB)
    });
  };
  const W1 = (name, fin, fout) => {
    const w = spec.weights[name];
    const ub = mkU(H, W, w.in_c, w.out_c, w.w_off, w.b_off);
    ops.push({ pipe: elemPipes.conv1x1, bg: bgElem(elemPipes.conv1x1, fin, fout, Wbuf, ub), gx: Math.ceil(W / 8), gy: Math.ceil(H / 8), gz: w.out_c });
  };
  const SiLU = (fin, fout, ch = feat) => {
    const ub = mkU(H, W, ch, ch, 0, 0);
    const n = ch * plane;
    ops.push({ pipe: elemPipes.silu, bg: bgElem(elemPipes.silu, fin, fout, Wbuf, ub), gx: Math.ceil(n / 64), gy: 1, gz: 1 });
  };
  const Gate = (o3, x, fout, ch = feat) => {
    const ub = mkU(H, W, ch, ch, 0, 0);
    const n = ch * plane;
    ops.push({ pipe: elemPipes.gate, bg: bgElem(elemPipes.gate, o3, fout, x, ub), gx: Math.ceil(n / 64), gy: 1, gz: 1 });
  };
  const PShuf = (fin, fout) => {
    const ub = mkU(H, W, 12, 3, 0, 0);
    ops.push({ pipe: elemPipes.pshuffle, bg: bgElem(elemPipes.pshuffle, fin, fout, Wbuf, ub), gx: Math.ceil(2 * W / 8), gy: Math.ceil(2 * H / 8), gz: 3 });
  };
  const SPAB = (blk, x, out2, o1dst) => {
    W3(`${blk}.c1_r`, x, sA);
    SiLU(sA, o1dst);
    W3(`${blk}.c2_r`, o1dst, sA);
    SiLU(sA, sB);
    W3(`${blk}.c3_r`, sB, sC);
    Gate(sC, x, out2);
  };
  W3("conv_1", IN, F);
  SPAB("block_1", F, B1, sB);
  SPAB("block_2", B1, Pb, sB);
  SPAB("block_3", Pb, Qb, sB);
  SPAB("block_4", Qb, Pb, sB);
  SPAB("block_5", Pb, Qb, sB);
  SPAB("block_6", Qb, Pb, B5_2);
  W3("conv_2", Pb, B6);
  const cb = feat * plane * eb;
  ops.push({ src: F, so: 0, dst: CAT, do: 0 * cb, sz: cb });
  ops.push({ src: B6, so: 0, dst: CAT, do: 1 * cb, sz: cb });
  ops.push({ src: B1, so: 0, dst: CAT, do: 2 * cb, sz: cb });
  ops.push({ src: B5_2, so: 0, dst: CAT, do: 3 * cb, sz: cb });
  W1("conv_cat", CAT, COUT);
  W3("upsampler", COUT, U12);
  PShuf(U12, OUT);
  const canTS = device.features.has("timestamp-query");
  const qs = canTS ? device.createQuerySet({ type: "timestamp", count: 2 }) : null;
  const qResolve = canTS ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const qRead = canTS ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
  const isCopy = (o) => o.src !== void 0;
  let best = 1e9;
  for (let t = 0; t < trials + warmup; t++) {
    device.queue.writeBuffer(IN, 0, inBytes.buffer, inBytes.byteOffset, inBytes.byteLength);
    const enc = device.createCommandEncoder();
    const computeOps = ops.filter((o) => !isCopy(o));
    let ci = 0;
    for (const o of ops) {
      if (isCopy(o)) {
        enc.copyBufferToBuffer(o.src, o.so, o.dst, o.do, o.sz);
        continue;
      }
      const first = ci === 0, last = ci === computeOps.length - 1;
      ci++;
      const tsw = canTS ? first ? { querySet: qs, beginningOfPassWriteIndex: 0 } : last ? { querySet: qs, endOfPassWriteIndex: 1 } : void 0 : void 0;
      const cp = enc.beginComputePass(tsw ? { timestampWrites: tsw } : void 0);
      cp.setPipeline(o.pipe);
      cp.setBindGroup(0, o.bg);
      cp.dispatchWorkgroups(o.gx, o.gy, o.gz);
      cp.end();
    }
    if (canTS) {
      enc.resolveQuerySet(qs, 0, 2, qResolve, 0);
      enc.copyBufferToBuffer(qResolve, 0, qRead, 0, 16);
    }
    device.queue.submit([enc.finish()]);
    if (canTS) {
      await qRead.mapAsync(GPUMapMode.READ);
      const ts = new BigUint64Array(qRead.getMappedRange().slice(0));
      qRead.unmap();
      const dt = Number(ts[1] - ts[0]) / 1e6;
      if (t >= warmup && dt > 0) best = Math.min(best, dt);
    } else {
      await device.queue.onSubmittedWorkDone();
    }
  }
  const outN = 3 * (2 * H) * (2 * W);
  const rb = device.createBuffer({ size: outN * eb, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e2 = device.createCommandEncoder();
  e2.copyBufferToBuffer(OUT, 0, rb, 0, outN * eb);
  device.queue.submit([e2.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = g.f16 ? new Float32Array(new F16C(rb.getMappedRange().slice(0))) : new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  return { out, ms: best };
}
export {
  initSpanGPUFast,
  runSpanFast
};
