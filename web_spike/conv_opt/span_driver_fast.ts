// SPAN WGSL driver — FAST variant. Same bit-faithful graph as span_driver.ts, but every
// 3x3 conv pass runs the optimized shared-mem-tiled + register-blocked kernel from
// webgpu_warp/kernel_gen.js (genConv) instead of the naive one-thread-per-output conv3x3.
//
// The 1x1 conv_cat, silu, gate and pixelshuffle passes are UNCHANGED — they reuse the
// element-wise WGSL module (genSpanWGSL) on the original 5-binding layout (binding 4 = fx).
// The genConv conv has only 4 bindings (0 fin, 1 fout, 2 Wt, 3 uniform P) so it gets its OWN
// bind-group + pipeline layout. Weight layout / bias offsets / PLANAR layout are identical to
// the naive conv, so SPAN's exported weights work unchanged. has_prelu=0 => bias-only, no act.
//
// initSpanGPUFast(device, weights, spec, f16, convOpts) ; runSpanFast(g, lr, ...) -> {out, ms}.

import { genSpanWGSL, Spec, toF16 } from "./span_driver.ts";
// @ts-ignore  JS ESM, untyped — genConv({OCB,TW,TH,F16,ACC,PROD,DB}) -> {code, dispatch, ...}
import { genConv } from "../webgpu_warp/kernel_gen.js";

export type { Spec };

const ELEM = ["conv1x1", "silu", "gate", "pshuffle"] as const;
type Elem = typeof ELEM[number];

export type ConvOpts = { OCB?: number; TW?: number; TH?: number; ACC?: "f16" | "f32" | "hybrid" };

export type SpanGPUFast = {
  device: GPUDevice; f16: boolean; eb: number; spec: Spec;
  // element-wise (5-binding) module
  elemPipes: Record<Elem, GPUComputePipeline>; elemBgl: GPUBindGroupLayout;
  // optimized conv (4-binding) module
  convPipe: GPUComputePipeline; convBgl: GPUBindGroupLayout;
  convTW: number; convTH: number; convOCB: number; convLabel: string;
  Wbuf: GPUBuffer; weightsF32: Float32Array;
};

const F16C = (globalThis as any).Float16Array;

export async function initSpanGPUFast(
  device: GPUDevice, weightsF32: Float32Array, spec: Spec, f16: boolean, convOpts: ConvOpts = {},
): Promise<SpanGPUFast> {
  const T: "f16" | "f32" = f16 ? "f16" : "f32";
  device.pushErrorScope("validation");

  // --- element-wise module (reuses the verified span WGSL; we only build the non-3x3 entries) ---
  const elemCode = genSpanWGSL(T);
  const elemMod = device.createShaderModule({ code: elemCode });
  const ci1 = await elemMod.getCompilationInfo?.();
  if (ci1) for (const m of ci1.messages) if (m.type === "error") throw new Error("elem WGSL: " + m.message + " @line " + m.lineNum);
  const elemBgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
  ] });
  const elemPl = device.createPipelineLayout({ bindGroupLayouts: [elemBgl] });
  const elemPipes = {} as Record<Elem, GPUComputePipeline>;
  for (const e of ELEM) elemPipes[e] = device.createComputePipeline({ layout: elemPl, compute: { module: elemMod, entryPoint: e } });

  // --- optimized 3x3 conv module (genConv) — 4 bindings, own layout ---
  const OCB = convOpts.OCB ?? 48, TW = convOpts.TW ?? 16, TH = convOpts.TH ?? 16;
  const ACC = f16 ? (convOpts.ACC ?? "f16") : "f32";
  const conv = genConv({ OCB, TW, TH, F16: f16, ACC, PROD: f16 ? "f16" : "f32", DB: true });
  const convMod = device.createShaderModule({ code: conv.code });
  const ci2 = await convMod.getCompilationInfo?.();
  if (ci2) for (const m of ci2.messages) if (m.type === "error") throw new Error("conv WGSL: " + m.message + " @line " + m.lineNum);
  const convBgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
  ] });
  const convPl = device.createPipelineLayout({ bindGroupLayouts: [convBgl] });
  const convPipe = device.createComputePipeline({ layout: convPl, compute: { module: convMod, entryPoint: "main" } });

  const eb = f16 ? 2 : 4;
  const wbytes: ArrayBufferView = f16 ? toF16(weightsF32) : weightsF32;
  const Wbuf = device.createBuffer({ size: wbytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(Wbuf, 0, wbytes.buffer, wbytes.byteOffset, wbytes.byteLength);

  const err = await device.popErrorScope();
  if (err) throw new Error("initFast validation: " + err.message);

  return {
    device, f16, eb, spec, elemPipes, elemBgl, convPipe, convBgl,
    convTW: TW, convTH: TH, convOCB: OCB, convLabel: conv.label, Wbuf, weightsF32,
  };
}

// Run the full SPAN graph (optimized 3x3 conv). lrPlanarF32 = 3*H*W in [0,1].
export async function runSpanFast(
  g: SpanGPUFast, lrPlanarF32: Float32Array, trials = 5, warmup = 2,
): Promise<{ out: Float32Array; ms: number }> {
  const { device, eb, elemPipes, elemBgl, convPipe, convBgl, Wbuf, spec, convTW, convTH, convOCB } = g;
  const { H, W } = spec, plane = H * W, feat = 48;
  const mkbuf = (ch: number) => device.createBuffer({ size: ch * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const IN = mkbuf(3);
  const F = mkbuf(feat), B1 = mkbuf(feat), B5_2 = mkbuf(feat), B6 = mkbuf(feat);
  const Pb = mkbuf(feat), Qb = mkbuf(feat), sA = mkbuf(feat), sB = mkbuf(feat), sC = mkbuf(feat);
  const CAT = mkbuf(192), COUT = mkbuf(feat), U12 = mkbuf(12);
  const OUT = device.createBuffer({ size: 3 * (2 * H) * (2 * W) * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  const inBytes: ArrayBufferView = g.f16 ? toF16(lrPlanarF32) : lrPlanarF32;
  device.queue.writeBuffer(IN, 0, inBytes.buffer, inBytes.byteOffset, inBytes.byteLength);

  const mkU = (Hh: number, Ww: number, in_c: number, out_c: number, w_off: number, b_off: number) => {
    // [H, W, in_c, out_c, w_off, b_off, prelu_off=0, has_prelu=0]
    const u = new Uint32Array([Hh, Ww, in_c, out_c, w_off >>> 0, b_off >>> 0, 0, 0]);
    const ub = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ub, 0, u); return ub;
  };
  // conv uses 4-binding layout (no fx); element-wise uses 5-binding layout (fx at 4).
  const bgConv = (fin: GPUBuffer, fout: GPUBuffer, ub: GPUBuffer) =>
    device.createBindGroup({ layout: convBgl, entries: [
      { binding: 0, resource: { buffer: fin } }, { binding: 1, resource: { buffer: fout } },
      { binding: 2, resource: { buffer: Wbuf } }, { binding: 3, resource: { buffer: ub } }] });
  const bgElem = (pipe: GPUComputePipeline, fin: GPUBuffer, fout: GPUBuffer, fx: GPUBuffer, ub: GPUBuffer) =>
    device.createBindGroup({ layout: elemBgl, entries: [
      { binding: 0, resource: { buffer: fin } }, { binding: 1, resource: { buffer: fout } },
      { binding: 2, resource: { buffer: Wbuf } }, { binding: 3, resource: { buffer: ub } },
      { binding: 4, resource: { buffer: fx } }] });

  type Op = { pipe: GPUComputePipeline; bg: GPUBindGroup; gx: number; gy: number; gz: number };
  type Copy = { src: GPUBuffer; so: number; dst: GPUBuffer; do: number; sz: number };
  const ops: (Op | Copy)[] = [];

  // optimized 3x3 conv: dispatch (ceil(W/TW), ceil(H/TH), ceil(out_c/OCB))
  const W3 = (name: string, fin: GPUBuffer, fout: GPUBuffer, Hh = H, Ww = W) => {
    const w = spec.weights[name]; const ub = mkU(Hh, Ww, w.in_c, w.out_c, w.w_off, w.b_off);
    ops.push({ pipe: convPipe, bg: bgConv(fin, fout, ub),
      gx: Math.ceil(Ww / convTW), gy: Math.ceil(Hh / convTH), gz: Math.ceil(w.out_c / convOCB) });
  };
  const W1 = (name: string, fin: GPUBuffer, fout: GPUBuffer) => {
    const w = spec.weights[name]; const ub = mkU(H, W, w.in_c, w.out_c, w.w_off, w.b_off);
    ops.push({ pipe: elemPipes.conv1x1, bg: bgElem(elemPipes.conv1x1, fin, fout, Wbuf, ub), gx: Math.ceil(W / 8), gy: Math.ceil(H / 8), gz: w.out_c });
  };
  const SiLU = (fin: GPUBuffer, fout: GPUBuffer, ch = feat) => {
    const ub = mkU(H, W, ch, ch, 0, 0); const n = ch * plane;
    ops.push({ pipe: elemPipes.silu, bg: bgElem(elemPipes.silu, fin, fout, Wbuf, ub), gx: Math.ceil(W / 8), gy: Math.ceil(H / 8), gz: feat });
  };
  const Gate = (o3: GPUBuffer, x: GPUBuffer, fout: GPUBuffer, ch = feat) => {
    const ub = mkU(H, W, ch, ch, 0, 0); const n = ch * plane;
    ops.push({ pipe: elemPipes.gate, bg: bgElem(elemPipes.gate, o3, fout, x, ub), gx: Math.ceil(W / 8), gy: Math.ceil(H / 8), gz: feat });
  };
  const PShuf = (fin: GPUBuffer, fout: GPUBuffer) => {
    const ub = mkU(H, W, 12, 3, 0, 0);
    ops.push({ pipe: elemPipes.pshuffle, bg: bgElem(elemPipes.pshuffle, fin, fout, Wbuf, ub), gx: Math.ceil((2 * W) / 8), gy: Math.ceil((2 * H) / 8), gz: 3 });
  };
  const SPAB = (blk: string, x: GPUBuffer, out: GPUBuffer, o1dst: GPUBuffer) => {
    W3(`${blk}.c1_r`, x, sA);
    SiLU(sA, o1dst);
    W3(`${blk}.c2_r`, o1dst, sA);
    SiLU(sA, sB);
    W3(`${blk}.c3_r`, sB, sC);
    Gate(sC, x, out);
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
  ops.push({ src: F, so: 0, dst: CAT, do: 0 * cb, sz: cb } as Copy);
  ops.push({ src: B6, so: 0, dst: CAT, do: 1 * cb, sz: cb } as Copy);
  ops.push({ src: B1, so: 0, dst: CAT, do: 2 * cb, sz: cb } as Copy);
  ops.push({ src: B5_2, so: 0, dst: CAT, do: 3 * cb, sz: cb } as Copy);
  W1("conv_cat", CAT, COUT);
  W3("upsampler", COUT, U12);
  PShuf(U12, OUT);

  const canTS = device.features.has("timestamp-query");
  const qs = canTS ? device.createQuerySet({ type: "timestamp", count: 2 }) : null;
  const qResolve = canTS ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const qRead = canTS ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
  const isCopy = (o: Op | Copy): o is Copy => (o as Copy).src !== undefined;

  let best = 1e9;
  for (let t = 0; t < trials + warmup; t++) {
    device.queue.writeBuffer(IN, 0, inBytes.buffer, inBytes.byteOffset, inBytes.byteLength);
    const enc = device.createCommandEncoder();
    const computeOps = ops.filter(o => !isCopy(o)) as Op[];
    let ci = 0;
    for (const o of ops) {
      if (isCopy(o)) { enc.copyBufferToBuffer(o.src, o.so, o.dst, o.do, o.sz); continue; }
      const first = ci === 0, last = ci === computeOps.length - 1; ci++;
      const tsw = canTS ? (first ? { querySet: qs!, beginningOfPassWriteIndex: 0 } : last ? { querySet: qs!, endOfPassWriteIndex: 1 } : undefined) : undefined;
      const cp = enc.beginComputePass(tsw ? { timestampWrites: tsw } : undefined);
      cp.setPipeline(o.pipe); cp.setBindGroup(0, o.bg); cp.dispatchWorkgroups(o.gx, o.gy, o.gz); cp.end();
    }
    if (canTS) { enc.resolveQuerySet(qs!, 0, 2, qResolve!, 0); enc.copyBufferToBuffer(qResolve!, 0, qRead!, 0, 16); }
    device.queue.submit([enc.finish()]);
    if (canTS) {
      await qRead!.mapAsync(GPUMapMode.READ);
      const ts = new BigUint64Array(qRead!.getMappedRange().slice(0)); qRead!.unmap();
      const dt = Number(ts[1] - ts[0]) / 1e6; if (t >= warmup && dt > 0) best = Math.min(best, dt);
    } else { await device.queue.onSubmittedWorkDone(); }
  }

  const outN = 3 * (2 * H) * (2 * W);
  const rb = device.createBuffer({ size: outN * eb, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e2 = device.createCommandEncoder(); e2.copyBufferToBuffer(OUT, 0, rb, 0, outN * eb); device.queue.submit([e2.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = g.f16 ? new Float32Array(new F16C(rb.getMappedRange().slice(0))) : new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  return { out, ms: best };
}

// PERSISTENT runner for per-frame use: allocates buffers + builds the op list ONCE; per frame only the
// IN contents change. recordInto(encoder) replays the SPAN graph (result left in OUT, planar 3×2H×2W).
// No per-call allocation -> no leak (runSpanFast was a one-shot benchmark fn and leaked over many calls).
export function makeSpanRunner(g: any) {
  const { device, eb, elemPipes, convPipe, convBgl, elemBgl, Wbuf, spec, convTW, convTH, convOCB } = g;
  const { H, W } = spec, plane = H * W, feat = 48;
  const mk = (ch: number) => device.createBuffer({ size: ch * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const IN = mk(3), F = mk(feat), B1 = mk(feat), B5_2 = mk(feat), B6 = mk(feat), Pb = mk(feat), Qb = mk(feat), sA = mk(feat), sB = mk(feat), sC = mk(feat), CAT = mk(192), COUT = mk(feat), U12 = mk(12);
  const OUT = device.createBuffer({ size: 3 * (2 * H) * (2 * W) * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const mkU = (Hh: number, Ww: number, in_c: number, out_c: number, w_off: number, b_off: number) => {
    const ub = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ub, 0, new Uint32Array([Hh, Ww, in_c, out_c, w_off >>> 0, b_off >>> 0, 0, 0])); return ub;
  };
  const bgC = (fin: any, fout: any, ub: any) => device.createBindGroup({ layout: convBgl, entries: [{ binding: 0, resource: { buffer: fin } }, { binding: 1, resource: { buffer: fout } }, { binding: 2, resource: { buffer: Wbuf } }, { binding: 3, resource: { buffer: ub } }] });
  const bgE = (fin: any, fout: any, fx: any, ub: any) => device.createBindGroup({ layout: elemBgl, entries: [{ binding: 0, resource: { buffer: fin } }, { binding: 1, resource: { buffer: fout } }, { binding: 2, resource: { buffer: Wbuf } }, { binding: 3, resource: { buffer: ub } }, { binding: 4, resource: { buffer: fx } }] });
  const ops: any[] = [];
  const W3 = (name: string, fin: any, fout: any) => { const w = spec.weights[name]; ops.push({ pipe: convPipe, bg: bgC(fin, fout, mkU(H, W, w.in_c, w.out_c, w.w_off, w.b_off)), gx: Math.ceil(W / convTW), gy: Math.ceil(H / convTH), gz: Math.ceil(w.out_c / convOCB) }); };
  const W1 = (name: string, fin: any, fout: any) => { const w = spec.weights[name]; ops.push({ pipe: elemPipes.conv1x1, bg: bgE(fin, fout, Wbuf, mkU(H, W, w.in_c, w.out_c, w.w_off, w.b_off)), gx: Math.ceil(W / 8), gy: Math.ceil(H / 8), gz: w.out_c }); };
  const SiLU = (fin: any, fout: any) => { const n = feat * plane; ops.push({ pipe: elemPipes.silu, bg: bgE(fin, fout, Wbuf, mkU(H, W, feat, feat, 0, 0)), gx: Math.ceil(W / 8), gy: Math.ceil(H / 8), gz: feat }); };
  const Gate = (o3: any, x: any, fout: any) => { const n = feat * plane; ops.push({ pipe: elemPipes.gate, bg: bgE(o3, fout, x, mkU(H, W, feat, feat, 0, 0)), gx: Math.ceil(W / 8), gy: Math.ceil(H / 8), gz: feat }); };
  const SPAB = (blk: string, x: any, out: any, o1: any) => { W3(`${blk}.c1_r`, x, sA); SiLU(sA, o1); W3(`${blk}.c2_r`, o1, sA); SiLU(sA, sB); W3(`${blk}.c3_r`, sB, sC); Gate(sC, x, out); };
  W3("conv_1", IN, F); SPAB("block_1", F, B1, sB); SPAB("block_2", B1, Pb, sB); SPAB("block_3", Pb, Qb, sB); SPAB("block_4", Qb, Pb, sB); SPAB("block_5", Pb, Qb, sB); SPAB("block_6", Qb, Pb, B5_2);
  W3("conv_2", Pb, B6);
  const cb = feat * plane * eb;
  ops.push({ src: F, dst: CAT, do: 0 * cb, sz: cb }); ops.push({ src: B6, dst: CAT, do: 1 * cb, sz: cb }); ops.push({ src: B1, dst: CAT, do: 2 * cb, sz: cb }); ops.push({ src: B5_2, dst: CAT, do: 3 * cb, sz: cb });
  W1("conv_cat", CAT, COUT); W3("upsampler", COUT, U12);
  ops.push({ pipe: elemPipes.pshuffle, bg: bgE(U12, OUT, Wbuf, mkU(H, W, 12, 3, 0, 0)), gx: Math.ceil((2 * W) / 8), gy: Math.ceil((2 * H) / 8), gz: 3 });
  function recordInto(enc: any) {
    for (const o of ops) {
      if (o.src !== undefined) { enc.copyBufferToBuffer(o.src, 0, o.dst, o.do, o.sz); continue; }
      const cp = enc.beginComputePass(); cp.setPipeline(o.pipe); cp.setBindGroup(0, o.bg); cp.dispatchWorkgroups(o.gx, o.gy, o.gz); cp.end();
    }
  }
  return { IN, OUT, recordInto, OW: 2 * W, OH: 2 * H, eb, f16: g.f16 };
}
