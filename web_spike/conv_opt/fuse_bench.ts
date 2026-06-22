// ============================================================================
// Fusion harness. Builds three full 34-layer pipelines and times each via GPU
// timestamps, parity vs the all-naive reference (mean|delta| of final 48-ch buf):
//   * NAIVE       (f32, the reference)
//   * ALL-COMBO   (f16 per-layer combo kernel, the apples-to-apples baseline)
//   * FUSED(K)    (combo for layer 0 & 33; the 32 middle 64->64 layers as
//                  ceil(32/K) FUSED dispatches of <=K layers via candidate_fuse)
//
//   /tmp/deno_latest/deno run --allow-read fuse_bench.ts [K|all] [SIZE]
//   K default 2, SIZE default 256. K=1 == per-layer (codegen/halo VALIDATION).
//
// f16 needs /tmp/deno_latest/deno (system deno can't compile `enable f16`).
// ============================================================================
import makeFused from "./candidate_fuse.ts";
import comboCand from "./candidate_combo.ts";

const log = (s: string) => { console.error(s); };
const DIR = new URL("../compact_data/", import.meta.url);
const wf32 = new Float32Array((Deno.readFileSync(new URL("weights.bin", DIR))).buffer);
const layers = JSON.parse(Deno.readTextFileSync(new URL("layers.json", DIR))).layers;
const argK = Deno.args[0] ?? "2";
const SIZE = parseInt(Deno.args[1] || "256");
const H = SIZE, W = SIZE, FC = 64, plane = H * W;
const TRIALS = 6, WARMUP = 2;

const seedF32 = new Float32Array(FC * plane);
for (let c = 0; c < 3; c++) for (let i = 0; i < plane; i++) seedF32[c * plane + i] = (((i * 7 + c * 31) % 101) / 101);

const adapter = await navigator.gpu.requestAdapter();
const hasF16 = adapter!.features.has("shader-f16");
if (!hasF16) { log("FATAL: shader-f16 not available -> need /tmp/deno_latest/deno"); Deno.exit(1); }
const feats: GPUFeatureName[] = ["timestamp-query", "shader-f16"];
if (adapter!.features.has("subgroups")) feats.push("subgroups" as GPUFeatureName);
const dev = await adapter!.requestDevice({
  requiredFeatures: feats,
  requiredLimits: { maxComputeWorkgroupStorageSize: Math.min(32768, adapter!.limits.maxComputeWorkgroupStorageSize) },
});
const F16 = (globalThis as any).Float16Array;
const f16bytes = (f: Float32Array) => new Uint16Array(new F16(f).buffer);

const Wf32 = dev.createBuffer({ size: wf32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
dev.queue.writeBuffer(Wf32, 0, wf32);
const Wf16b = f16bytes(wf32);
const Wf16 = dev.createBuffer({ size: Wf16b.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
dev.queue.writeBuffer(Wf16, 0, Wf16b);

// ---- pipelines ----
function compile(code: string): GPUComputePipeline {
  dev.pushErrorScope("validation");
  const mod = dev.createShaderModule({ code });
  const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
  return pipe;
}
async function checkCompile(label: string) {
  const e = await dev.popErrorScope();
  if (e) throw new Error(`PIPELINE ${label}: ${e.message}`);
}

const NAIVE_CODE = `struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
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
    fout[u32(oc)*u.H*u.W+u32(y)*u.W+u32(x)]=acc;}`;

dev.pushErrorScope("validation"); const naivePipe = compile(NAIVE_CODE); await checkCompile("naive");
dev.pushErrorScope("validation"); const comboPipe = compile(comboCand.code); await checkCompile("combo");

// fused pipelines, cached by block size kb
const fusedCache = new Map<number, ReturnType<typeof makeFused> & { pipe: GPUComputePipeline }>();
async function fusedFor(kb: number) {
  if (fusedCache.has(kb)) return fusedCache.get(kb)!;
  const T = 12 - 2 * kb;                 // keep S=T+2K=12 (max within 32 KB shared) for all kb
  const m = makeFused({ K: kb, T });
  dev.pushErrorScope("validation");
  const pipe = compile(m.code);
  await checkCompile(`fused K=${kb} T=${T}`);
  const rec = { ...m, pipe }; fusedCache.set(kb, rec); return rec;
}

// ---- plan builders ----  pass = { pipe, u:Uint32Array, ubBytes, dims }
type Pass = { pipe: GPUComputePipeline; u: Uint32Array; ubBytes: number; dims: [number, number, number] };

function comboPass(ly: any): Pass {
  const u = new Uint32Array([H, W, ly.in_c, ly.out_c, ly.w_off, ly.b_off, ly.prelu_off < 0 ? 0 : ly.prelu_off, ly.prelu_off < 0 ? 0 : 1]);
  return { pipe: comboPipe, u, ubBytes: 32, dims: comboCand.dispatch(ly, H, W) };
}
function naivePass(ly: any): Pass {
  const u = new Uint32Array([H, W, ly.in_c, ly.out_c, ly.w_off, ly.b_off, ly.prelu_off < 0 ? 0 : ly.prelu_off, ly.prelu_off < 0 ? 0 : 1]);
  return { pipe: naivePipe, u, ubBytes: 32, dims: [Math.ceil(W / 8), Math.ceil(H / 8), ly.out_c] };
}
const naivePlan = () => layers.map(naivePass);
const comboPlan = () => layers.map(comboPass);

async function fusedPlan(K: number): Promise<Pass[]> {
  const passes: Pass[] = [comboPass(layers[0])];
  // middle layers are indices 1..32 (32 layers). chunk into blocks of <=K.
  for (let i = 1; i <= 32; i += K) {
    const chunk: number[] = [];
    for (let j = i; j < i + K && j <= 32; j++) chunk.push(j);
    const kb = chunk.length;
    const rec = await fusedFor(kb);
    const u = new Uint32Array(rec.ubBytes / 4);
    u[0] = H; u[1] = W; // [2],[3] padding
    chunk.forEach((li, idx) => {
      const ly = layers[li];
      u[4 + idx * 4 + 0] = ly.w_off;
      u[4 + idx * 4 + 1] = ly.b_off;
      u[4 + idx * 4 + 2] = ly.prelu_off;
      u[4 + idx * 4 + 3] = 1; // all middle layers have PReLU
    });
    passes.push({ pipe: rec.pipe, u, ubBytes: rec.ubBytes, dims: rec.dispatch(H, W) });
  }
  passes.push(comboPass(layers[33]));
  return passes;
}

// ---- runner: ping-pong A/B global feature buffers, timestamp first->last pass ----
async function runPlan(plan: Pass[], eb: number, weightsBuf: GPUBuffer): Promise<{ feat: Float32Array; ms: number }> {
  const A = dev.createBuffer({ size: FC * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const B = dev.createBuffer({ size: FC * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const seedBytes: any = eb === 2 ? f16bytes(seedF32) : seedF32;
  let inb = A, outb = B; const built: any[] = [];
  for (const p of plan) {
    const ub = dev.createBuffer({ size: p.ubBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(ub, 0, p.u);
    const bg = dev.createBindGroup({
      layout: p.pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: inb } }, { binding: 1, resource: { buffer: outb } },
        { binding: 2, resource: { buffer: weightsBuf } }, { binding: 3, resource: { buffer: ub } }],
    });
    built.push({ pipe: p.pipe, bg, dims: p.dims }); [inb, outb] = [outb, inb];
  }
  const finalBuf = inb;
  const qs = dev.createQuerySet({ type: "timestamp", count: 2 });
  const qResolve = dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const qRead = dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  let best = 1e9;
  for (let t = 0; t < TRIALS + WARMUP; t++) {
    dev.queue.writeBuffer(A, 0, seedBytes.buffer);
    const enc = dev.createCommandEncoder();
    built.forEach((p, i) => {
      const tsw = i === 0 ? { querySet: qs, beginningOfPassWriteIndex: 0 }
        : i === built.length - 1 ? { querySet: qs, endOfPassWriteIndex: 1 } : undefined;
      const cp = enc.beginComputePass(tsw ? { timestampWrites: tsw } : undefined);
      cp.setPipeline(p.pipe); cp.setBindGroup(0, p.bg); cp.dispatchWorkgroups(p.dims[0], p.dims[1], p.dims[2]); cp.end();
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
  const feat = eb === 2 ? new Float32Array(new F16(rb.getMappedRange().slice(0))) : new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  return { feat, ms: best };
}

function parity(a: Float32Array, ref: Float32Array): { mad: number; mx: number } {
  let s = 0, mx = 0; const n = 48 * plane;
  for (let i = 0; i < n; i++) { const d = Math.abs(a[i] - ref[i]); s += d; if (d > mx) mx = d; }
  return { mad: s / n, mx };
}

// ---- run ----
log(`[fuse_bench] SIZE=${SIZE} f16=${hasF16} subgroups=${adapter!.features.has("subgroups")}`);
const ref = await runPlan(naivePlan(), 4, Wf32);
log(`NAIVE(ref): ${ref.ms.toFixed(1)} ms`);

const combo = await runPlan(comboPlan(), 2, Wf16);
const cp = parity(combo.feat, ref.feat);
log(`ALL-COMBO : ${combo.ms.toFixed(1)} ms | parity mean|d|=${cp.mad.toExponential(2)} max=${cp.mx.toExponential(2)} ${cp.mad < 1e-2 ? "OK" : "FAIL"}`);

const Ks = argK === "all" ? [1, 2, 3, 4] : [parseInt(argK)];
for (const K of Ks) {
  const plan = await fusedPlan(K);
  const fz = await runPlan(plan, 2, Wf16);
  const fp = parity(fz.feat, ref.feat);
  const Ts = [...new Set([...Array(Math.ceil(32 / K)).keys()].map((b) => {
    const kb = Math.min(K, 32 - b * K); return `K${kb}/T${12 - 2 * kb}`;
  }))].join(",");
  log(`FUSED K=${K} : ${fz.ms.toFixed(1)} ms | ${(combo.ms / fz.ms).toFixed(2)}x vs combo | ${(ref.ms / fz.ms).toFixed(2)}x vs naive | parity mean|d|=${fp.mad.toExponential(2)} max=${fp.mx.toExponential(2)} ${fp.mad < 1e-2 ? "OK" : "FAIL"} | blocks=${Ts}`);
}
