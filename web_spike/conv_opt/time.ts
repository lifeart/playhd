// fast candidate-only timer (no naive). best-of-N GPU-timestamp time for the 34-pass pipeline.
//   deno run --allow-read time.ts <candidate.ts> <SIZE> [TRIALS]
const DIR = new URL("../compact_data/", import.meta.url);
const wf32 = new Float32Array((Deno.readFileSync(new URL("weights.bin", DIR))).buffer);
const layers = JSON.parse(Deno.readTextFileSync(new URL("layers.json", DIR))).layers;
const SIZE = parseInt(Deno.args[1] || "128");
const TRIALS = parseInt(Deno.args[2] || "8"), WARMUP = 3;
const H = SIZE, W = SIZE, FC = 64, plane = H * W;
const seedF32 = new Float32Array(FC * plane);
for (let c = 0; c < 3; c++) for (let i = 0; i < plane; i++) seedF32[c * plane + i] = (((i * 7 + c * 31) % 101) / 101);
const adapter = await navigator.gpu.requestAdapter();
const dev = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", "shader-f16"] });
const F16 = (globalThis as any).Float16Array;
const f16bytes = (f: Float32Array) => new Uint16Array(new F16(f).buffer);
const Wf32 = dev.createBuffer({ size: wf32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(Wf32, 0, wf32);
let Wf16: any = null;
const weights = (f16: boolean) => { if (!f16) return Wf32; if (!Wf16) { const b = f16bytes(wf32); Wf16 = dev.createBuffer({ size: b.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(Wf16, 0, b); } return Wf16; };
const cand = (await import(new URL(Deno.args[0], `file://${Deno.cwd()}/`).href)).default;
const f16 = !!cand.f16, eb = f16 ? 2 : 4;
const A = dev.createBuffer({ size: FC * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
const B = dev.createBuffer({ size: FC * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
const seedBytes: any = f16 ? f16bytes(seedF32) : seedF32;
dev.pushErrorScope("validation");
const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: cand.code }), entryPoint: "main" } });
const pe = await dev.popErrorScope(); if (pe) { console.log("PIPELINE-ERR:", pe.message); Deno.exit(1); }
let inb = A, outb = B; const passes: any[] = [];
for (const ly of layers) {
  const u = new Uint32Array([H, W, ly.in_c, ly.out_c, ly.w_off, ly.b_off, ly.prelu_off < 0 ? 0 : ly.prelu_off, ly.prelu_off < 0 ? 0 : 1]);
  const ub = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(ub, 0, u);
  const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: inb } }, { binding: 1, resource: { buffer: outb } }, { binding: 2, resource: { buffer: weights(f16) } }, { binding: 3, resource: { buffer: ub } }] });
  const [gx, gy, gz] = cand.dispatch(ly, H, W);
  passes.push({ bg, gx, gy, gz }); [inb, outb] = [outb, inb];
}
// Saturating timer that NEVER stalls the GPU between trials: each trial is one encoder running REPS
// full pipelines with a timestamp pair into its own slot; all encoders are submitted back-to-back
// (no await between), then a single map reads every slot. Continuous feed -> sustained boost clock.
const REPS = 8;
const nP = passes.length;
const NT = TRIALS + WARMUP;
const qs = dev.createQuerySet({ type: "timestamp", count: 2 * NT });
const qResolve = dev.createBuffer({ size: 16 * NT, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const qRead = dev.createBuffer({ size: 16 * NT, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
dev.queue.writeBuffer(A, 0, seedBytes.buffer);
for (let t = 0; t < NT; t++) {
  const enc = dev.createCommandEncoder();
  for (let r = 0; r < REPS; r++) {
    passes.forEach((p, i) => {
      const first = r === 0 && i === 0, last = r === REPS - 1 && i === nP - 1;
      const tsw = first ? { querySet: qs, beginningOfPassWriteIndex: 2 * t } : last ? { querySet: qs, endOfPassWriteIndex: 2 * t + 1 } : undefined;
      const cp = enc.beginComputePass(tsw ? { timestampWrites: tsw } : undefined);
      cp.setPipeline(pipe); cp.setBindGroup(0, p.bg); cp.dispatchWorkgroups(p.gx, p.gy, p.gz); cp.end();
    });
  }
  dev.queue.submit([enc.finish()]); // no await -> GPU stays fed
}
const encR = dev.createCommandEncoder();
encR.resolveQuerySet(qs, 0, 2 * NT, qResolve, 0); encR.copyBufferToBuffer(qResolve, 0, qRead, 0, 16 * NT);
dev.queue.submit([encR.finish()]);
await qRead.mapAsync(GPUMapMode.READ);
const ts = new BigUint64Array(qRead.getMappedRange().slice(0)); qRead.unmap();
let best = 1e9;
for (let t = WARMUP; t < NT; t++) { const dt = Number(ts[2 * t + 1] - ts[2 * t]) / 1e6 / REPS; if (dt > 0) best = Math.min(best, dt); }
console.log(`TIME ${Deno.args[0]} SIZE=${SIZE}: ${best.toFixed(2)} ms`);
