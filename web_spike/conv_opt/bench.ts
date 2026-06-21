// Deno headless WebGPU benchmark for the compact-SR conv (SRVGGNetCompact, 34 conv layers).
// Runs a CANDIDATE conv through the full pipeline, times it best-of-N, checks parity (mean|Δ| of the
// final 48-ch feature buffer) vs the built-in NAIVE reference. wgpu->Metal; RELATIVE speedups transfer
// to Chrome (confirm the winner in the browser at full size). Synthetic deterministic input (the conv
// math, not the image, is what we benchmark; parity is candidate-vs-naive on the same input).
//
//   deno run --allow-read bench.ts [candidate.ts] [SIZE]      (SIZE default 128 for fast iteration; use 256 for final)
//
// Candidate default-exports { code, dispatch:(ly,H,W)=>[gx,gy,gz], f16?:bool }.
// Conv shader bindings: 0 fin(read), 1 fout(read_write), 2 Wt(read),
//   3 uniform P{H,W,in_c,out_c,w_off,b_off,prelu_off,has_prelu} (8x u32); entry @compute fn main.
//   f16:true -> fin/fout/Wt are array<f16> (weights+features stored f16). PReLU+bias as usual.
const log = (s: string) => { console.error(s); };  // stderr flushes immediately

const DIR = new URL("../compact_data/", import.meta.url);
const wf32 = new Float32Array((Deno.readFileSync(new URL("weights.bin", DIR))).buffer);
const layers = JSON.parse(Deno.readTextFileSync(new URL("layers.json", DIR))).layers;
const SIZE = parseInt(Deno.args[1] || "128");
const H = SIZE, W = SIZE, FC = 64, plane = H * W;
const TRIALS = 5, WARMUP = 2;
// synthetic deterministic input (3 ch active)
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
const F16 = (globalThis as any).Float16Array;
const f16bytes = (f: Float32Array) => new Uint16Array(new F16(f).buffer);
const Wf32 = dev.createBuffer({ size: wf32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(Wf32, 0, wf32);
let Wf16: GPUBuffer | null = null;
const weights = (f16: boolean) => { if (!f16) return Wf32; if (!Wf16) { const b = f16bytes(wf32); Wf16 = dev.createBuffer({ size: b.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(Wf16, 0, b); } return Wf16; };

async function run(cfg: any): Promise<{ feat: Float32Array; ms: number }> {
  const f16 = !!cfg.f16, eb = f16 ? 2 : 4;
  // BOTH buffers need COPY_SRC: the 34-layer (even) pipeline ends on A, so the parity readback copies
  // from A (earlier bug: A lacked COPY_SRC -> readback failed silently -> vacuous all-zeros parity).
  const A = dev.createBuffer({ size: FC * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const B = dev.createBuffer({ size: FC * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const seedBytes: any = f16 ? f16bytes(seedF32) : seedF32;
  dev.pushErrorScope("validation");
  const mod = dev.createShaderModule({ code: cfg.code });
  const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
  const pe = await dev.popErrorScope(); if (pe) throw new Error("PIPELINE: " + pe.message);
  let inb = A, outb = B; const passes: any[] = [];
  for (const ly of layers) {
    const u = new Uint32Array([H, W, ly.in_c, ly.out_c, ly.w_off, ly.b_off, ly.prelu_off < 0 ? 0 : ly.prelu_off, ly.prelu_off < 0 ? 0 : 1]);
    const ub = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); dev.queue.writeBuffer(ub, 0, u);
    const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: inb } }, { binding: 1, resource: { buffer: outb } }, { binding: 2, resource: { buffer: weights(f16) } }, { binding: 3, resource: { buffer: ub } }] });
    const [gx, gy, gz] = cfg.dispatch(ly, H, W);
    passes.push({ bg, gx, gy, gz }); [inb, outb] = [outb, inb];
  }
  const finalBuf = inb;
  // TIMING via GPU timestamp queries (accurate GPU time; immune to Deno's CPU-sync quirk): write a
  // timestamp at the start of pass 0 and the end of pass 33 -> diff = total GPU time for the 34 passes.
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
  const feat = f16 ? new Float32Array(new F16(rb.getMappedRange().slice(0))) : new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  return { feat, ms: best };
}

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
  dispatch: (ly: any, H: number, W: number) => [Math.ceil(W / 8), Math.ceil(H / 8), ly.out_c],
};

log(`[bench] SIZE=${SIZE} 34 layers, f16-cap=${hasF16}`);
const ref = await run(NAIVE);
log(`NAIVE: ${ref.ms.toFixed(1)} ms`);
const candPath = Deno.args[0];
if (candPath && candPath !== "-") {
  const cand = (await import(new URL(candPath, `file://${Deno.cwd()}/`).href)).default;
  const out = await run(cand);
  let s = 0, mx = 0; const n = 48 * plane;
  for (let i = 0; i < n; i++) { const d = Math.abs(out.feat[i] - ref.feat[i]); s += d; if (d > mx) mx = d; }
  const mad = s / n;
  log(`CANDIDATE ${candPath}: ${out.ms.toFixed(1)} ms | ${(ref.ms / out.ms).toFixed(2)}x | parity mean|Δ|=${mad.toExponential(2)} max=${mx.toExponential(2)} ${mad < 1e-2 ? "PARITY-OK" : "PARITY-FAIL"}`);
}
