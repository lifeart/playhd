// SPAN (2xLiveActionV1_SPAN) WGSL driver — bit-faithful port of the PyTorch SPAN graph.
// PLANAR feature layout: index = c*H*W + y*W + x. f32 (parity) or f16 (speed) via genSpanWGSL(T).
//
// Graph (see export_span_weights.py / span_data/spec.json):
//   F   = conv_1(IN)                          Conv3XC 3->48  (collapsed 3x3)
//   for block 1..6:  SPAB(x):
//       o1 = c1_r(x); a1 = SiLU(o1);  o2 = c2_r(a1); a2 = SiLU(o2);  o3 = c3_r(a2)
//       out = (o3 + x) * (sigmoid(o3) - 0.5)
//       returned "out1" = a1 (SiLU-activated, because act1 is inplace) -> used as out_b5_2
//   B1=block_1(F); B2=block_2(B1); ...; B5=block_5(B4); (B6raw,B5_2)=block_6(B5)
//   B6  = conv_2(B6raw)                        Conv3XC 48->48
//   CAT = concat([F, B6, B1, B5_2])            (192ch; planar => byte concat)
//   COUT= conv_cat(CAT)                        Conv2d 1x1 192->48
//   U12 = upsampler.0(COUT)                    Conv2d 3x3 48->12
//   OUT = PixelShuffle2(U12)                   -> 3ch @ 2H x 2W   (no LR residual, no denorm)
// act1=SiLU, no input normalization (no_norm), no leaky_relu (has_relu=False everywhere).

export type Spec = {
  H: number; W: number; scale: number; feat: number; out_h: number; out_w: number;
  weights: Record<string, { in_c: number; out_c: number; k: number; w_off: number; b_off: number }>;
};

export function genSpanWGSL(T: "f32" | "f16"): string {
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

// SiLU: x*sigmoid(x). 3D grid (x=W, y=H, z=channel) — avoids the 65535 1D-dispatch limit at large H*W.
@compute @workgroup_size(8,8,1) fn silu(@builtin(global_invocation_id) g:vec3u){
  if(g.x>=u.W||g.y>=u.H||g.z>=u.in_c){ return; } let i=g.z*u.H*u.W + g.y*u.W + g.x;
  let v=f32(fin[i]); fout[i]=${T}(v*(1.0/(1.0+exp(-v))));
}

// SPAB gate: (o3 + x) * (sigmoid(o3) - 0.5). fin=o3, fx=x. 3D grid (same reason).
@compute @workgroup_size(8,8,1) fn gate(@builtin(global_invocation_id) g:vec3u){
  if(g.x>=u.W||g.y>=u.H||g.z>=u.in_c){ return; } let i=g.z*u.H*u.W + g.y*u.W + g.x;
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

const ELEMS = ["conv3x3", "conv1x1", "silu", "gate", "pshuffle"] as const;
type Entry = typeof ELEMS[number];

export type SpanGPU = {
  device: GPUDevice; f16: boolean; eb: number;
  pipes: Record<Entry, GPUComputePipeline>;
  bgl: GPUBindGroupLayout; Wbuf: GPUBuffer; weightsF32: Float32Array; spec: Spec;
};

const F16 = (globalThis as any).Float16Array;
export const toF16 = (f: Float32Array) => new Uint16Array(new F16(f).buffer);

export async function initSpanGPU(device: GPUDevice, weightsF32: Float32Array, spec: Spec, f16: boolean): Promise<SpanGPU> {
  const code = genSpanWGSL(f16 ? "f16" : "f32");
  device.pushErrorScope("validation");
  const mod = device.createShaderModule({ code });
  const cinfo = await mod.getCompilationInfo?.();
  if (cinfo) for (const m of cinfo.messages) if (m.type === "error") throw new Error("WGSL compile: " + m.message + " @line " + m.lineNum);
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });
  const pl = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const pipes = {} as Record<Entry, GPUComputePipeline>;
  for (const e of ELEMS) {
    pipes[e] = device.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint: e } });
  }
  const eb = f16 ? 2 : 4;
  const wbytes: ArrayBufferView = f16 ? toF16(weightsF32) : weightsF32;
  const Wbuf = device.createBuffer({ size: wbytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(Wbuf, 0, wbytes.buffer, wbytes.byteOffset, wbytes.byteLength);
  const err = await device.popErrorScope();
  if (err) throw new Error("init validation: " + err.message);
  return { device, f16, eb, pipes, bgl, Wbuf, weightsF32, spec };
}

// Run the full SPAN graph. lrPlanarF32 = 3*H*W in [0,1]. Returns {out: Float32Array(3*2H*2W planar), ms}.
export async function runSpan(g: SpanGPU, lrPlanarF32: Float32Array, trials = 5, warmup = 2, wantInter: string[] = []): Promise<{ out: Float32Array; ms: number; inter: Record<string, Float32Array> }> {
  const { device, eb, pipes, bgl, Wbuf, spec } = g;
  const { H, W } = spec, plane = H * W, feat = 48;
  const mkbuf = (ch: number) => device.createBuffer({ size: ch * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const IN = mkbuf(3);
  const F = mkbuf(feat), B1 = mkbuf(feat), B5_2 = mkbuf(feat), B6 = mkbuf(feat);
  const Pb = mkbuf(feat), Qb = mkbuf(feat), sA = mkbuf(feat), sB = mkbuf(feat), sC = mkbuf(feat);
  const CAT = mkbuf(192), COUT = mkbuf(feat), U12 = mkbuf(12);
  const OUT = device.createBuffer({ size: 3 * (2 * H) * (2 * W) * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  // upload input
  const inBytes: ArrayBufferView = g.f16 ? toF16(lrPlanarF32) : lrPlanarF32;
  device.queue.writeBuffer(IN, 0, inBytes.buffer, inBytes.byteOffset, inBytes.byteLength);

  const ubufs: GPUBuffer[] = [];
  const mkU = (Hh: number, Ww: number, in_c: number, out_c: number, w_off: number, b_off: number) => {
    const u = new Uint32Array([Hh, Ww, in_c, out_c, w_off >>> 0, b_off >>> 0, 0, 0]);
    const ub = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ub, 0, u); ubufs.push(ub); return ub;
  };
  const bg = (pipe: GPUComputePipeline, fin: GPUBuffer, fout: GPUBuffer, fx: GPUBuffer, ub: GPUBuffer) =>
    device.createBindGroup({ layout: bgl, entries: [
      { binding: 0, resource: { buffer: fin } }, { binding: 1, resource: { buffer: fout } },
      { binding: 2, resource: { buffer: Wbuf } }, { binding: 3, resource: { buffer: ub } },
      { binding: 4, resource: { buffer: fx } }] });

  type Op = { pipe: GPUComputePipeline; bg: GPUBindGroup; gx: number; gy: number; gz: number };
  type Copy = { src: GPUBuffer; so: number; dst: GPUBuffer; do: number; sz: number };
  const ops: (Op | Copy)[] = [];
  const W3 = (name: string, fin: GPUBuffer, fout: GPUBuffer, Hh = H, Ww = W) => {
    const w = spec.weights[name]; const ub = mkU(Hh, Ww, w.in_c, w.out_c, w.w_off, w.b_off);
    ops.push({ pipe: pipes.conv3x3, bg: bg(pipes.conv3x3, fin, fout, Wbuf, ub), gx: Math.ceil(Ww / 8), gy: Math.ceil(Hh / 8), gz: w.out_c });
  };
  const W1 = (name: string, fin: GPUBuffer, fout: GPUBuffer) => {
    const w = spec.weights[name]; const ub = mkU(H, W, w.in_c, w.out_c, w.w_off, w.b_off);
    ops.push({ pipe: pipes.conv1x1, bg: bg(pipes.conv1x1, fin, fout, Wbuf, ub), gx: Math.ceil(W / 8), gy: Math.ceil(H / 8), gz: w.out_c });
  };
  const SiLU = (fin: GPUBuffer, fout: GPUBuffer, ch = feat) => {
    const ub = mkU(H, W, ch, ch, 0, 0); const n = ch * plane;
    ops.push({ pipe: pipes.silu, bg: bg(pipes.silu, fin, fout, Wbuf, ub), gx: Math.ceil(W / 8), gy: Math.ceil(H / 8), gz: ch });
  };
  const Gate = (o3: GPUBuffer, x: GPUBuffer, fout: GPUBuffer, ch = feat) => {
    const ub = mkU(H, W, ch, ch, 0, 0); const n = ch * plane;
    ops.push({ pipe: pipes.gate, bg: bg(pipes.gate, o3, fout, x, ub), gx: Math.ceil(W / 8), gy: Math.ceil(H / 8), gz: ch });
  };
  const PShuf = (fin: GPUBuffer, fout: GPUBuffer) => {
    const ub = mkU(H, W, 12, 3, 0, 0);
    ops.push({ pipe: pipes.pshuffle, bg: bg(pipes.pshuffle, fin, fout, Wbuf, ub), gx: Math.ceil((2 * W) / 8), gy: Math.ceil((2 * H) / 8), gz: 3 });
  };
  // SPAB: x -> out; o1dst receives SiLU(c1_r(x)) (== returned out1).
  const SPAB = (blk: string, x: GPUBuffer, out: GPUBuffer, o1dst: GPUBuffer) => {
    W3(`${blk}.c1_r`, x, sA);   // sA = raw o1
    SiLU(sA, o1dst);            // o1dst = SiLU(o1)  (= out1_act = returned out1)
    W3(`${blk}.c2_r`, o1dst, sA);
    SiLU(sA, sB);
    W3(`${blk}.c3_r`, sB, sC);  // sC = o3
    Gate(sC, x, out);           // out = (o3+x)*(sigmoid(o3)-0.5)
  };

  W3("conv_1", IN, F);
  SPAB("block_1", F, B1, sB);
  SPAB("block_2", B1, Pb, sB);
  SPAB("block_3", Pb, Qb, sB);
  SPAB("block_4", Qb, Pb, sB);
  SPAB("block_5", Pb, Qb, sB);   // Qb = out_b5
  SPAB("block_6", Qb, Pb, B5_2); // Pb = out_b6(raw), B5_2 = SiLU(c1_r) saved
  W3("conv_2", Pb, B6);
  // concat [F, B6, B1, B5_2] -> CAT (planar => byte concat)
  const cb = feat * plane * eb;
  ops.push({ src: F, so: 0, dst: CAT, do: 0 * cb, sz: cb } as Copy);
  ops.push({ src: B6, so: 0, dst: CAT, do: 1 * cb, sz: cb } as Copy);
  ops.push({ src: B1, so: 0, dst: CAT, do: 2 * cb, sz: cb } as Copy);
  ops.push({ src: B5_2, so: 0, dst: CAT, do: 3 * cb, sz: cb } as Copy);
  W1("conv_cat", CAT, COUT);
  W3("upsampler", COUT, U12);
  PShuf(U12, OUT);

  // registry of intermediates for piecewise validation (channel counts as in PyTorch dumps)
  const reg: Record<string, { buf: GPUBuffer; ch: number }> = {
    F: { buf: F, ch: feat }, B1: { buf: B1, ch: feat }, B5_2: { buf: B5_2, ch: feat },
    B6c: { buf: B6, ch: feat }, catout: { buf: COUT, ch: feat }, up0: { buf: U12, ch: 12 },
  };

  // timestamp timing over the whole graph
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

  // readback OUT
  const outN = 3 * (2 * H) * (2 * W);
  const rb = device.createBuffer({ size: outN * eb, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e2 = device.createCommandEncoder(); e2.copyBufferToBuffer(OUT, 0, rb, 0, outN * eb); device.queue.submit([e2.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = g.f16 ? new Float32Array(new F16(rb.getMappedRange().slice(0))) : new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();

  const inter: Record<string, Float32Array> = {};
  for (const name of wantInter) {
    const r = reg[name]; if (!r) continue;
    const n = r.ch * plane;
    const ib = device.createBuffer({ size: n * eb, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e3 = device.createCommandEncoder(); e3.copyBufferToBuffer(r.buf, 0, ib, 0, n * eb); device.queue.submit([e3.finish()]);
    await ib.mapAsync(GPUMapMode.READ);
    inter[name] = g.f16 ? new Float32Array(new F16(ib.getMappedRange().slice(0))) : new Float32Array(ib.getMappedRange().slice(0));
    ib.unmap();
  }
  return { out, ms: best, inter };
}
