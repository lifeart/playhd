// ============================================================================
// Single-process portability sweep. Generates each conv config inline, times all
// of them BACK-TO-BACK in one Deno process (shared contention env -> comparable),
// two passes min to filter transient GPU contention spikes, normalizes every ms to
// an in-process anchor (wtile f32 OCB=32 16x16) so "rel" is contention-robust.
// Captures compile/pipeline failures (try/catch) to surface real failure modes.
//   deno run --allow-read --unstable-webgpu sweep.ts [SIZE]   (use /tmp deno for f16)
// ============================================================================
const log = (s: string) => console.error(s);
const DIR = new URL("../compact_data/", import.meta.url);
const wf32 = new Float32Array((Deno.readFileSync(new URL("weights.bin", DIR))).buffer);
const layers = JSON.parse(Deno.readTextFileSync(new URL("layers.json", DIR))).layers;
const SIZE = parseInt(Deno.args[0] || "256");
const H = SIZE, W = SIZE, FC = 64, plane = H * W;
const TRIALS = 5, WARMUP = 2, PASSES = 2;

const seedF32 = new Float32Array(FC * plane);
for (let c = 0; c < 3; c++) for (let i = 0; i < plane; i++) seedF32[c * plane + i] = (((i * 7 + c * 31) % 101) / 101);

const adapter = await navigator.gpu.requestAdapter();
const hasF16 = adapter.features.has("shader-f16");
const feats: GPUFeatureName[] = ["timestamp-query"];
if (hasF16) feats.push("shader-f16");
const dev = await adapter.requestDevice({
  requiredFeatures: feats,
  requiredLimits: { maxComputeWorkgroupStorageSize: Math.min(32768, adapter.limits.maxComputeWorkgroupStorageSize) },
});
const F16A = (globalThis as any).Float16Array;
const f16bytes = (f: Float32Array) => new Uint16Array(new F16A(f).buffer);
const Wf32 = dev.createBuffer({ size: wf32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(Wf32, 0, wf32);
let Wf16: GPUBuffer | null = null;
const weights = (f16: boolean) => { if (!f16) return Wf32; if (!Wf16) { const b = f16bytes(wf32); Wf16 = dev.createBuffer({ size: b.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(Wf16, 0, b); } return Wf16; };

type Cfg = { TW: number; TH: number; OCB: number; DB: boolean; F16: boolean; ACC: "f32" | "f16" | "hybrid"; PROD: "f16" | "f32" };
function genCode(c: Cfg) {
  const { TW, TH, OCB, DB, F16 } = c;
  const ACC = F16 ? c.ACC : "f32", PROD = F16 ? c.PROD : "f32";
  const HW = TW + 2, HSZ = HW * (TH + 2), NT = TW * TH, G = OCB / 4, WSZ = G * 9;
  const ST = F16 ? "f16" : "f32", NB = DB ? 2 : 1, zeroS = F16 ? "f16(0.0)" : "0.0";
  const accDecl = (() => {
    if (ACC === "hybrid") return Array.from({ length: G }, (_, g) => `  var a${g} = vec4<f32>(0.0);`).join("\n");
    const z = ACC === "f16" ? "vec4<f16>(0.0)" : "vec4<f32>(0.0)";
    return Array.from({ length: G }, (_, g) => `  var a${g} = ${z};`).join("\n");
  })();
  const macInto = (dst: string, accIsF32: boolean, g: number) => {
    const w = `sW[wBase + ${g * 9}u + wk]`;
    if (!F16) return `${dst} += ${w} * inv;`;
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
    }).join("\n")).join("\n");
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
    workgroupBarrier(); let sBase = 0u; let wBase = 0u;
${tapDecl}
    for(var ky=0u; ky<3u; ky++){ for(var kx=0u; kx<3u; kx++){
        let inv = sIn[sBase + (ly+ky)*${HW}u + (lx+kx)]; let wk = ky*3u+kx;
${accBody}
    }}
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
    let hx=t%${HW}u; let hy=t/${HW}u; let xx=i32(gx0)+i32(hx)-1; let yy=i32(gy0)+i32(hy)-1;
    var v:${ST}=${zeroS};
    if(xx>=0 && xx<i32(u.W) && yy>=0 && yy<i32(u.H)){ v = fin[ic*plane + u32(yy)*u.W + u32(xx)]; }
    sIn[sBase + t] = v;
  }
  for(var t=lidx; t<${WSZ}u; t+=${NT}u){
    let g=t/9u; let k=t%9u; let wb=u.w_off+ic*9u+k; let oc0=ocbase+g*4u;
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
  let lx=lidx%${TW}u; let ly=lidx/${TW}u; let gx0=wid.x*${TW}u; let gy0=wid.y*${TH}u;
  let x=gx0+lx; let y=gy0+ly; let ocbase=wid.z*${OCB}u; let plane=u.H*u.W; let inc9=u.in_c*9u;
${accDecl}
${loop}
  if(x<u.W && y<u.H){
${writeBody}
  }
}`;
  const esz = F16 ? 2 : 4;
  const smem = NB * HSZ * esz + NB * WSZ * 4 * esz;
  return { code, f16: F16, smem, threads: NT, regAcc: G * 4 * (ACC === "f16" ? 1 : 2), dispatch: (ly: any) => [Math.ceil(W / TW), Math.ceil(H / TH), Math.ceil(ly.out_c / OCB)] };
}

async function runOne(gen: ReturnType<typeof genCode>): Promise<{ ms: number; feat: Float32Array | null; err?: string }> {
  const f16 = gen.f16, eb = f16 ? 2 : 4;
  try {
    dev.pushErrorScope("validation");
    const mod = dev.createShaderModule({ code: gen.code });
    const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
    const pe = await dev.popErrorScope(); if (pe) return { ms: -1, feat: null, err: pe.message.split("\n")[0].slice(0, 90) };
    const A = dev.createBuffer({ size: FC * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const B = dev.createBuffer({ size: FC * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const seedBytes: any = f16 ? f16bytes(seedF32) : seedF32;
    let inb = A, outb = B; const passes: any[] = [];
    for (const ly of layers) {
      const u = new Uint32Array([H, W, ly.in_c, ly.out_c, ly.w_off, ly.b_off, ly.prelu_off < 0 ? 0 : ly.prelu_off, ly.prelu_off < 0 ? 0 : 1]);
      const ub = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(ub, 0, u);
      const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: inb } }, { binding: 1, resource: { buffer: outb } }, { binding: 2, resource: { buffer: weights(f16) } }, { binding: 3, resource: { buffer: ub } }] });
      const [gx, gy, gz] = gen.dispatch(ly); passes.push({ bg, gx, gy, gz }); [inb, outb] = [outb, inb];
    }
    const finalBuf = inb;
    const qs = dev.createQuerySet({ type: "timestamp", count: 2 });
    const qResolve = dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const qRead = dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let best = 1e9;
    for (let t = 0; t < TRIALS + WARMUP; t++) {
      dev.queue.writeBuffer(A, 0, seedBytes.buffer);
      const enc = dev.createCommandEncoder();
      passes.forEach((p, i) => {
        const tsw = i === 0 ? { querySet: qs, beginningOfPassWriteIndex: 0 } : i === passes.length - 1 ? { querySet: qs, endOfPassWriteIndex: 1 } : undefined;
        const cp = enc.beginComputePass(tsw ? { timestampWrites: tsw } : undefined);
        cp.setPipeline(pipe); cp.setBindGroup(0, p.bg); cp.dispatchWorkgroups(p.gx, p.gy, p.gz); cp.end();
      });
      enc.resolveQuerySet(qs, 0, 2, qResolve, 0); enc.copyBufferToBuffer(qResolve, 0, qRead, 0, 16);
      dev.queue.submit([enc.finish()]);
      await qRead.mapAsync(GPUMapMode.READ);
      const ts = new BigUint64Array(qRead.getMappedRange().slice(0)); qRead.unmap();
      const dt = Number(ts[1] - ts[0]) / 1e6; if (t >= WARMUP && dt > 0) best = Math.min(best, dt);
    }
    const rb = dev.createBuffer({ size: FC * plane * eb, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e2 = dev.createCommandEncoder(); e2.copyBufferToBuffer(finalBuf, 0, rb, 0, FC * plane * eb); dev.queue.submit([e2.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const feat = f16 ? new Float32Array(new F16A(rb.getMappedRange().slice(0))) : new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    return { ms: best, feat };
  } catch (e) { return { ms: -1, feat: null, err: String((e as Error).message || e).split("\n")[0].slice(0, 90) }; }
}

// reference (naive f32) for parity
const NAIVE = {
  code: `struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
    @group(0) @binding(0) var<storage,read> fin:array<f32>;@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
    @group(0) @binding(2) var<storage,read> Wt:array<f32>;@group(0) @binding(3) var<uniform> u:P;
    @compute @workgroup_size(8,8,1) fn main(@builtin(global_invocation_id) g:vec3u){
      let x=i32(g.x);let y=i32(g.y);let oc=i32(g.z);if(x>=i32(u.W)||y>=i32(u.H)||oc>=i32(u.out_c)){return;}
      var acc=Wt[u.b_off+u32(oc)];let bw=u.w_off+u32(oc)*u.in_c*9u;
      for(var ic=0u;ic<u.in_c;ic=ic+1u){let pl=ic*u.H*u.W;let wic=bw+ic*9u;
        for(var ky=0;ky<3;ky=ky+1){let yy=y+ky-1;if(yy<0||yy>=i32(u.H)){continue;}
          for(var kx=0;kx<3;kx=kx+1){let xx=x+kx-1;if(xx<0||xx>=i32(u.W)){continue;}
            acc=acc+Wt[wic+u32(ky*3+kx)]*fin[pl+u32(yy)*u.W+u32(xx)];}}}
      if(u.has_prelu==1u){let s=Wt[u.prelu_off+u32(oc)];if(acc<0.0){acc=acc*s;}}
      fout[u32(oc)*u.H*u.W+u32(y)*u.W+u32(x)]=acc;}`,
  f16: false, smem: 0, threads: 64, regAcc: 0, dispatch: (ly: any) => [Math.ceil(W / 8), Math.ceil(H / 8), ly.out_c],
};

const CFGS: { name: string; cfg: Cfg }[] = [];
const add = (name: string, cfg: Cfg) => CFGS.push({ name, cfg });
// Block A: f32 (bit-exact), OCB x tile, DB on
for (const OCB of [16, 32, 64]) for (const [TW, TH] of [[8, 8], [16, 16], [8, 32], [32, 8]] as [number, number][])
  add(`f32 OCB${OCB} ${TW}x${TH} DB`, { TW, TH, OCB, DB: true, F16: false, ACC: "f32", PROD: "f32" });
// Block B: f16 (combo family), OCB x tile, DB on, ACC=f16
for (const OCB of [16, 32, 64]) for (const [TW, TH] of [[8, 8], [16, 16], [8, 32], [32, 8]] as [number, number][])
  add(`f16 OCB${OCB} ${TW}x${TH} DB`, { TW, TH, OCB, DB: true, F16: true, ACC: "f16", PROD: "f16" });
// Block C: ACC precision at f16 sweet spot
for (const ACC of ["f32", "hybrid"] as const)
  add(`f16 OCB64 16x16 DB ACC=${ACC}`, { TW: 16, TH: 16, OCB: 64, DB: true, F16: true, ACC, PROD: "f16" });
// Block D: double-buffer OFF at the winners
add(`f32 OCB32 16x16 noDB`, { TW: 16, TH: 16, OCB: 32, DB: false, F16: false, ACC: "f32", PROD: "f32" });
add(`f16 OCB64 16x16 noDB`, { TW: 16, TH: 16, OCB: 64, DB: false, F16: true, ACC: "f16", PROD: "f16" });
// Block E: oversize workgroups (probe >256-thread failure mode)
add(`f32 OCB32 32x16=512 DB`, { TW: 32, TH: 16, OCB: 32, DB: true, F16: false, ACC: "f32", PROD: "f32" });
add(`f32 OCB32 32x32=1024 DB`, { TW: 32, TH: 32, OCB: 32, DB: true, F16: false, ACC: "f32", PROD: "f32" });
// anchor (re-timed for drift)
const anchorCfg: Cfg = { TW: 16, TH: 16, OCB: 32, DB: true, F16: false, ACC: "f32", PROD: "f32" };

const refRun = await runOne(NAIVE as any);
const ref = refRun.feat!;
log(`[sweep] SIZE=${SIZE} NAIVE=${refRun.ms.toFixed(0)}ms f16cap=${hasF16} PASSES=${PASSES}`);

const parity = (feat: Float32Array | null) => {
  if (!feat) return { mad: NaN, mx: NaN };
  let s = 0, mx = 0; const n = 48 * plane;
  for (let i = 0; i < n; i++) { const d = Math.abs(feat[i] - ref[i]); s += d; if (d > mx) mx = d; }
  return { mad: s / n, mx };
};

const best: Record<string, { ms: number; par: { mad: number; mx: number }; err?: string; gen: any }> = {};
for (let pass = 0; pass < PASSES; pass++) {
  // re-time anchor each pass for drift
  const ag = genCode(anchorCfg); const ar = await runOne(ag);
  log(`  pass${pass} anchor(f32 OCB32 16x16)=${ar.ms.toFixed(0)}ms`);
  for (const { name, cfg } of CFGS) {
    const gen = genCode(cfg); const r = await runOne(gen);
    const par = parity(r.feat);
    if (!best[name] || (r.ms > 0 && r.ms < best[name].ms)) best[name] = { ms: r.ms, par, err: r.err, gen };
    if (r.err && best[name] && best[name].ms < 0) best[name] = { ms: -1, par: { mad: NaN, mx: NaN }, err: r.err, gen };
  }
}

// print table
log(`\n${"config".padEnd(30)} ${"ms".padStart(8)} ${"rel".padStart(6)} ${"thr".padStart(4)} ${"smem".padStart(6)} ${"regA".padStart(5)} ${"mad".padStart(9)}  status`);
const anchorMs = best[`f32 OCB32 16x16 DB`]?.ms || 1;
const rows: string[] = [];
for (const { name } of CFGS) {
  const b = best[name]; if (!b) continue;
  const g = b.gen;
  if (b.ms < 0 || b.err) { rows.push(`${name.padEnd(30)} ${"FAIL".padStart(8)} ${"".padStart(6)} ${String(g.threads).padStart(4)} ${String(g.smem).padStart(6)} ${String(g.regAcc).padStart(5)} ${"".padStart(9)}  ${b.err || "fail"}`); continue; }
  const rel = (b.ms / anchorMs).toFixed(2);
  const status = isNaN(b.par.mad) ? "?" : b.par.mad < 1e-2 ? "OK" : "PARITY-FAIL";
  rows.push(`${name.padEnd(30)} ${b.ms.toFixed(1).padStart(8)} ${rel.padStart(6)} ${String(g.threads).padStart(4)} ${String(g.smem).padStart(6)} ${String(g.regAcc).padStart(5)} ${(isNaN(b.par.mad) ? "" : b.par.mad.toExponential(1)).padStart(9)}  ${status}`);
}
log(rows.join("\n"));
log(`\n[anchor f32 OCB32 16x16 = ${anchorMs.toFixed(1)}ms = rel 1.00; lower rel = faster]`);
