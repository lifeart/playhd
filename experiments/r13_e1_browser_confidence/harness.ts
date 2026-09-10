// R13-E1 headless A/B harness — runs the SHIPPED browser pipeline (player.html's Player()) on
// real decoded frames + real codec MVs, with the occlusion stage swapped between the shipped
// HARD binary switch and the new SOFT confidence gate, and dumps the HD output frames so the
// Python side can score them.
//
// It is deliberately a transcription of player.html (same weights, same combo/wtile conv kernel,
// same pixelshuffle, same textures, same anchor cadence rule) with three things removed —
// canvas present, audio, rAF clock — and one thing added: raw RGBA frame dumps + timings.
// The warp/gate WGSL is IMPORTED from web_spike/wasm_mv/warp_shader.js, the same module the
// player imports, so there is no second copy of the shader to drift.
//
// Input data comes from prep_and_probe.py (rgb_<i>.bin + mv_<i>.bin + meta.json).
//
// Run (needs Deno with WebGPU):
//   deno run --unstable-webgpu --allow-read --allow-write --allow-env harness.ts \
//     --data <scratch>/calm --mode hard --reanchor 4 --out <scratch>/out_calm_hard
//
// Modes: ref (SR every frame = the ceiling) | hard | soft | softblk
import { buildLrFlow, packLrFlowRGBA } from "../../web_spike/wasm_mv/flow.js";
import { warpShaderWGSL, reliabilityShaderWGSL } from "../../web_spike/wasm_mv/warp_shader.js";

function arg(name: string, def: string | null = null): string | null {
  const i = Deno.args.indexOf("--" + name);
  return i >= 0 && i + 1 < Deno.args.length ? Deno.args[i + 1] : def;
}
const DATA = arg("data")!;
const MODE = arg("mode", "hard")!;
const REANCHOR = MODE === "ref" ? 1 : parseInt(arg("reanchor", "4")!);
const OUT = arg("out");
const TAU = parseFloat(arg("tau", "16")!);
const SGM = parseFloat(arg("s", "3")!);
const KP = parseFloat(arg("kp", "0")!);
const KM = parseFloat(arg("km", "0")!);
const MVREF = parseFloat(arg("mvref", "8")!);
const TIMING_REPS = parseInt(arg("reps", "0")!);          // >0: also time the warp stage N times
const SPIKE = arg("spike", "");                            // "" | "limits"
const HERE = new URL(".", import.meta.url).pathname;
const WS = HERE + "../../web_spike/";

type Meta = { w: number; h: number; n: number; start: number; frames: { i: number; type: string; nmv: number }[] };
const meta: Meta = JSON.parse(await Deno.readTextFile(`${DATA}/meta.json`));
const w = meta.w, h = meta.h;

// ---------------- GPU setup (transcribed from player.html) ----------------
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("no WebGPU adapter");
// f16: Deno's wgpu/Naga does NOT implement the `enable f16` extension yet (it fails to parse
// combo.wgsl even though the adapter advertises shader-f16), so the harness defaults to the f32
// wtile kernel. Chrome runs the f16 combo kernel; the SR net is the same, only the accumulator
// precision differs, and BOTH A/B arms share whichever path is used -> the comparison is unaffected.
const F16ARG = arg("f16", "0")!;
const f16 = F16ARG === "auto" ? adapter.features.has("shader-f16") : F16ARG === "1";
// NOTE (R13 side-question): player.html requests NO requiredLimits, so it gets the WebGPU
// defaults (16KiB workgroup storage / 256 invocations). --spike limits asks for the adapter
// maximum instead, to show what is being left on the table.
const reqLimits: Record<string, number> = {};
if (SPIKE === "limits") {
  reqLimits.maxComputeWorkgroupStorageSize = adapter.limits.maxComputeWorkgroupStorageSize;
  reqLimits.maxComputeInvocationsPerWorkgroup = adapter.limits.maxComputeInvocationsPerWorkgroup;
}
const TS = adapter.features.has("timestamp-query");
const feats: GPUFeatureName[] = [];
if (f16) feats.push("shader-f16");
if (TS) feats.push("timestamp-query");
const dev = await adapter.requestDevice({
  requiredFeatures: feats,
  requiredLimits: SPIKE === "limits" ? reqLimits : undefined,
});
dev.addEventListener("uncapturederror", (e: any) => { console.error("WEBGPU ERROR:", e.error?.message ?? e); });

const F16 = (globalThis as any).Float16Array;
const packW = (f32: Float32Array) => f16 ? new Uint16Array(new F16(f32).buffer) : f32;
const EB = f16 ? 2 : 4;
const MS = 4, scale = 2;
const srW = Math.max(8, Math.round(w * scale / MS)), srH = Math.max(8, Math.round(h * scale / MS));
const WD = srW * MS, HD = srH * MS, plane_sr = srW * srH, FC = 64;

const wf = new Float32Array((await Deno.readFile(`${WS}compact_data/weights.bin`)).buffer);
const layers = JSON.parse(await Deno.readTextFile(`${WS}compact_data/layers.json`)).layers;
const Wg = dev.createBuffer({ size: packW(wf).byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
dev.queue.writeBuffer(Wg, 0, packW(wf));
const convCode = await Deno.readTextFile(`${WS}webgpu_warp/${f16 ? "combo" : "wtile"}.wgsl`);
const OCB = f16 ? 64 : 32, TILE = 16;
const cpipe = dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code: convCode }), entryPoint: "main" } });
const A = dev.createBuffer({ size: FC * plane_sr * EB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
const B = dev.createBuffer({ size: FC * plane_sr * EB, usage: GPUBufferUsage.STORAGE });
const LRb = dev.createBuffer({ size: 3 * plane_sr * EB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
let inb = A, outb = B;
const convBg: { bg: GPUBindGroup; gz: number }[] = [];
for (const ly of layers) {
  const u = new Uint32Array([srH, srW, ly.in_c, ly.out_c, ly.w_off, ly.b_off, ly.prelu_off < 0 ? 0 : ly.prelu_off, ly.prelu_off < 0 ? 0 : 1]);
  const ub = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(ub, 0, u);
  convBg.push({
    bg: dev.createBindGroup({ layout: cpipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: inb } }, { binding: 1, resource: { buffer: outb } },
      { binding: 2, resource: { buffer: Wg } }, { binding: 3, resource: { buffer: ub } }] }),
    gz: Math.ceil(ly.out_c / OCB),
  });
  [inb, outb] = [outb, inb];
}
const featLast = inb;
const fT = f16 ? "f16" : "f32";
const psh = dev.createShaderModule({ code: `${f16 ? "enable f16;" : ""}struct P{H:u32,W:u32,up:u32,_p:u32};@group(0)@binding(0) var<storage,read> feat:array<${fT}>;@group(0)@binding(1) var<storage,read> lr:array<${fT}>;@group(0)@binding(2) var<uniform> u:P;
    @vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return vec4f(p[i],0,1);}
    @fragment fn fs(@builtin(position) pos:vec4f)->@location(0) vec4f{let X=i32(floor(pos.x));let Y=i32(floor(pos.y));let up=i32(u.up);let x=X/up;let y=Y/up;let i=Y%up;let j=X%up;let pl=u.H*u.W;
      var rgb=vec3f(0.0);for(var c=0;c<3;c=c+1){let ch=u32(c*up*up+i*up+j);rgb[c]=clamp(f32(feat[ch*pl+u32(y)*u.W+u32(x)])+f32(lr[u32(c)*pl+u32(y)*u.W+u32(x)]),0.0,1.0);}return vec4f(rgb,1.0);}` });
const pshp = dev.createRenderPipeline({ layout: "auto", vertex: { module: psh, entryPoint: "vs" }, fragment: { module: psh, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] }, primitive: { topology: "triangle-list" } });
const pshU = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
dev.queue.writeBuffer(pshU, 0, new Uint32Array([srH, srW, MS, 0]));

const hdTex = () => dev.createTexture({ size: [WD, HD], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
let hdA = hdTex(), hdB = hdTex();
const lrTex = () => dev.createTexture({ size: [w, h], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC });
let lrCur = lrTex(), lrPrev = lrTex();
const flowTex = dev.createTexture({ size: [w, h], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
// GPU timing: timestamp queries (Deno's CPU-side queue sync -- onSubmittedWorkDone() -- hangs, and
// Chrome quantizes CPU timestamps anyway; the repo convention, web_spike/conv_opt/bench.ts, is
// timestamp queries + mapAsync as the sync point).
const qs = TS ? dev.createQuerySet({ type: "timestamp", count: 2 }) : null;
const qResolve = TS ? dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
const qRead = TS ? dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
async function readTimestamp(): Promise<number> {
  if (!TS) return 0;
  await qRead!.mapAsync(GPUMapMode.READ);
  const ts = new BigUint64Array(qRead!.getMappedRange().slice(0)); qRead!.unmap();
  return Number(ts[1] - ts[0]) / 1e6;
}
const samp = dev.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" });

const SOFT = MODE === "soft" || MODE === "softblk";
const BLOCKS = MODE === "softblk";
const wmod = dev.createShaderModule({ code: warpShaderWGSL({ gate: SOFT ? "soft" : "hard" }) });
const wpipe = dev.createRenderPipeline({ layout: "auto", vertex: { module: wmod, entryPoint: "vs" }, fragment: { module: wmod, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] }, primitive: { topology: "triangle-list" } });
const wU = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
dev.queue.writeBuffer(wU, 0, new Uint32Array([WD, HD, scale, w, h, TAU, 0, 0]));

// soft gate: LR reliability pass (r16float, filterable -> the HD pass gets the bilinear upsample)
let reliTex: GPUTexture | null = null, rpipe: GPURenderPipeline | null = null, rU: GPUBuffer | null = null, rbg: GPUBindGroup | null = null;
if (SOFT) {
  reliTex = dev.createTexture({ size: [w, h], format: "r16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
  const rmod = dev.createShaderModule({ code: reliabilityShaderWGSL({ blocks: BLOCKS }) });
  rpipe = dev.createRenderPipeline({ layout: "auto", vertex: { module: rmod, entryPoint: "vs" }, fragment: { module: rmod, entryPoint: "fs", targets: [{ format: "r16float" }] }, primitive: { topology: "triangle-list" } });
  rU = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const gb = new ArrayBuffer(32); new Uint32Array(gb, 0, 2).set([w, h]);
  new Float32Array(gb, 8, 6).set([TAU, SGM, KP, KM, MVREF, 0]);
  dev.queue.writeBuffer(rU, 0, gb);
}

// ---------------- per-frame ops (same shapes as player.html) ----------------
function uploadLR(rgb: Uint8Array) {
  const n = w * h; const rgba = new Uint8Array(n * 4);
  for (let i = 0, j = 0, k = 0; i < n; i++) { rgba[j++] = rgb[k++]; rgba[j++] = rgb[k++]; rgba[j++] = rgb[k++]; rgba[j++] = 255; }
  dev.queue.writeTexture({ texture: lrCur }, rgba, { bytesPerRow: w * 4, rowsPerImage: h }, [w, h]);
}
const FAC = Math.max(1, Math.round(w / srW));
function srAnchor(rgb: Uint8Array, dst: GPUTexture) {
  const lp = new Float32Array(3 * plane_sr); const inv = 1 / (FAC * FAC * 255);
  for (let sy = 0; sy < srH; sy++) for (let sx = 0; sx < srW; sx++) {
    let r = 0, g = 0, b = 0;
    for (let dy = 0; dy < FAC; dy++) { const yy = Math.min(sy * FAC + dy, h - 1); for (let dx = 0; dx < FAC; dx++) { const xx = Math.min(sx * FAC + dx, w - 1); const o = (yy * w + xx) * 3; r += rgb[o]; g += rgb[o + 1]; b += rgb[o + 2]; } }
    const bi = sy * srW + sx; lp[bi] = r * inv; lp[plane_sr + bi] = g * inv; lp[2 * plane_sr + bi] = b * inv;
  }
  const lpb = packW(lp); dev.queue.writeBuffer(A, 0, lpb); dev.queue.writeBuffer(LRb, 0, lpb);
  const e = dev.createCommandEncoder();
  convBg.forEach((b, i) => {
    const cp = e.beginComputePass(TS && i === 0 ? { timestampWrites: { querySet: qs!, beginningOfPassWriteIndex: 0 } } : undefined);
    cp.setPipeline(cpipe); cp.setBindGroup(0, b.bg); cp.dispatchWorkgroups(Math.ceil(srW / TILE), Math.ceil(srH / TILE), b.gz); cp.end();
  });
  const fbg = dev.createBindGroup({ layout: pshp.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: featLast } }, { binding: 1, resource: { buffer: LRb } }, { binding: 2, resource: { buffer: pshU } }] });
  const rp = e.beginRenderPass({ colorAttachments: [{ view: dst.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }], ...(TS ? { timestampWrites: { querySet: qs!, endOfPassWriteIndex: 1 } } : {}) });
  rp.setPipeline(pshp); rp.setBindGroup(0, fbg); rp.draw(3); rp.end();
  if (TS) { e.resolveQuerySet(qs!, 0, 2, qResolve!, 0); e.copyBufferToBuffer(qResolve!, 0, qRead!, 0, 16); }
  dev.queue.submit([e.finish()]);
}
let flowStage: Float32Array | null = null;
function uploadFlow(mvs: Int32Array) {
  const ft = BLOCKS ? (flowStage ??= new Float32Array(w * h * 4)) : new Float32Array(w * h * 4);
  if (BLOCKS) {
    packLrFlowRGBA(mvs, h, w, ft, "all", true);          // direct rgba32float pack (see flow.js)
  } else {
    const { fx, fy } = buildLrFlow(mvs, h, w, "all");
    for (let i = 0; i < w * h; i++) { const hole = Number.isNaN(fx[i]); ft[i * 4] = hole ? -1e9 : fx[i]; ft[i * 4 + 1] = hole ? 0 : fy[i]; }
  }
  dev.queue.writeTexture({ texture: flowTex }, ft, { bytesPerRow: w * 16, rowsPerImage: h }, [w, h]);
}
function warpPass(prev: GPUTexture, dst: GPUTexture) {
  const e = dev.createCommandEncoder();
  if (SOFT) {
    rbg = dev.createBindGroup({ layout: rpipe!.getBindGroupLayout(0), entries: [
      { binding: 0, resource: flowTex.createView() }, { binding: 1, resource: lrCur.createView() },
      { binding: 2, resource: lrPrev.createView() }, { binding: 3, resource: samp }, { binding: 4, resource: { buffer: rU! } }] });
    const rp0 = e.beginRenderPass({ colorAttachments: [{ view: reliTex!.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }], ...(TS ? { timestampWrites: { querySet: qs!, beginningOfPassWriteIndex: 0 } } : {}) });
    rp0.setPipeline(rpipe!); rp0.setBindGroup(0, rbg); rp0.draw(3); rp0.end();
  }
  // NB: with layout:"auto" the derived layout contains only the bindings the shader USES. The soft
  // HD pass does not read lrPrev (the residual moved to the reliability pass), so binding 5 must be
  // omitted there or the bind group is rejected ("7 bindings vs 6 in layout").
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: prev.createView() }, { binding: 1, resource: samp }, { binding: 2, resource: flowTex.createView() },
    { binding: 3, resource: lrCur.createView() }, { binding: 4, resource: { buffer: wU } }];
  if (SOFT) entries.push({ binding: 6, resource: reliTex!.createView() });
  else entries.push({ binding: 5, resource: lrPrev.createView() });
  const bg = dev.createBindGroup({ layout: wpipe.getBindGroupLayout(0), entries });
  const rp = e.beginRenderPass({ colorAttachments: [{ view: dst.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }], ...(TS ? { timestampWrites: { querySet: qs!, ...(SOFT ? {} : { beginningOfPassWriteIndex: 0 }), endOfPassWriteIndex: 1 } } : {}) });
  rp.setPipeline(wpipe); rp.setBindGroup(0, bg); rp.draw(3); rp.end();
  if (TS) { e.resolveQuerySet(qs!, 0, 2, qResolve!, 0); e.copyBufferToBuffer(qResolve!, 0, qRead!, 0, 16); }
  dev.queue.submit([e.finish()]);
}
async function readTex(tex: GPUTexture, W2: number, H2: number, bpp = 4): Promise<Uint8Array> {
  const bpr = Math.ceil(W2 * bpp / 256) * 256;
  const rb = dev.createBuffer({ size: bpr * H2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = dev.createCommandEncoder();
  e.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: bpr, rowsPerImage: H2 }, [W2, H2]);
  dev.queue.submit([e.finish()]);
  await rb.mapAsync(GPUMapMode.READ);          // mapAsync IS the sync point (Deno)
  const src = new Uint8Array(rb.getMappedRange());
  const out = new Uint8Array(W2 * H2 * bpp);
  for (let y = 0; y < H2; y++) out.set(src.subarray(y * bpr, y * bpr + W2 * bpp), y * W2 * bpp);
  rb.unmap(); rb.destroy();
  return out;
}

// ---------------- the frame loop ----------------
if (OUT) await Deno.mkdir(OUT, { recursive: true });
let prevHD = hdA, freeHD = hdB;
const tSr: number[] = [], tWarp: number[] = [];
const gateStats: { i: number; mean_a: number; frac_a_lt_05: number; frac_a_gt_95: number }[] = [];
const kinds: string[] = [];
for (let fi = 0; fi < meta.n; fi++) {
  const rgb = await Deno.readFile(`${DATA}/rgb_${fi}.bin`);
  const mvbuf = await Deno.readFile(`${DATA}/mv_${fi}.bin`);
  const mvs = new Int32Array(mvbuf.buffer, mvbuf.byteOffset, mvbuf.byteLength / 4);
  const tmp = lrPrev; lrPrev = lrCur; lrCur = tmp;              // [lrPrev,lrCur]=[lrCur,lrPrev]
  uploadLR(rgb);
  const isAnchor = mvs.length === 0 || (fi % REANCHOR === 0);
  const t0 = performance.now();
  if (isAnchor) {
    srAnchor(rgb, freeHD); [prevHD, freeHD] = [freeHD, prevHD];
    tSr.push(TS ? await readTimestamp() : performance.now() - t0);
  } else {
    uploadFlow(mvs);
    warpPass(prevHD, freeHD); [prevHD, freeHD] = [freeHD, prevHD];
    tWarp.push(TS ? await readTimestamp() : performance.now() - t0);
    if (SOFT && reliTex) {                                        // gate distribution (diagnostics)
      const raw = await readTex(reliTex, w, h, 2);
      if (OUT) await Deno.writeFile(`${OUT}/reli_${fi}.f16.bin`, raw);   // r16float LR reliability map
      const dv = new DataView(raw.buffer); let s = 0, lo = 0, hi = 0;
      for (let i = 0; i < w * h; i++) { const a = f16ToF32(dv.getUint16(i * 2, true)); s += a; if (a < 0.5) lo++; if (a > 0.95) hi++; }
      gateStats.push({ i: fi, mean_a: s / (w * h), frac_a_lt_05: lo / (w * h), frac_a_gt_95: hi / (w * h) });
    }
  }
  kinds.push(isAnchor ? "A" : "W");
  if (OUT) await Deno.writeFile(`${OUT}/hd_${fi}.bin`, await readTex(prevHD, WD, HD));
}

function f16ToF32(u: number): number {
  const s = (u >> 15) & 1, e = (u >> 10) & 0x1f, m = u & 0x3ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
}
const med = (a: number[]) => { if (!a.length) return 0; const b = [...a].sort((x, y) => x - y); return b[b.length >> 1]; };
const pctl = (a: number[], q: number) => { if (!a.length) return 0; const b = [...a].sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(q * b.length))]; };

// optional: repeat-timed warp stage (Chrome quantizes timestamp queries; time N repeats instead)
let warpRepMs = 0, warpRepP90 = 0, warpRepMin = 0;
if (TIMING_REPS > 0 && tWarp.length) {
  const reps: number[] = [];
  for (let r = 0; r < TIMING_REPS; r++) { warpPass(prevHD, freeHD); reps.push(TS ? await readTimestamp() : 0); }
  const kept = reps.slice(Math.floor(reps.length / 4));          // drop the warm-up quarter
  warpRepMs = med(kept); warpRepP90 = pctl(kept, 0.9); warpRepMin = Math.min(...kept);
}
const res = {
  mode: MODE, reanchor: REANCHOR, data: DATA, frames: meta.n, res: `${w}x${h}->${WD}x${HD}`,
  f16, anchors: kinds.filter((k) => k === "A").length, warps: kinds.filter((k) => k === "W").length,
  gate: SOFT ? { tau: TAU, s: SGM, kp: KP, km: KM, mvref: MVREF, blocks: BLOCKS } : { tau: TAU, kind: "hard" },
  ms_sr_median: +med(tSr).toFixed(3), ms_warp_median: +med(tWarp).toFixed(3), ms_warp_repeat: +warpRepMs.toFixed(3), ms_warp_repeat_p90: +warpRepP90.toFixed(3), ms_warp_repeat_min: +warpRepMin.toFixed(3),
  ms_warp_p90: +pctl(tWarp, 0.9).toFixed(3), ms_sr_p90: +pctl(tSr, 0.9).toFixed(3),
  gate_mean_a: gateStats.length ? +(gateStats.reduce((s, g) => s + g.mean_a, 0) / gateStats.length).toFixed(4) : null,
  gate_frac_a_lt_05: gateStats.length ? +(gateStats.reduce((s, g) => s + g.frac_a_lt_05, 0) / gateStats.length).toFixed(4) : null,
  gate_frac_a_gt_95: gateStats.length ? +(gateStats.reduce((s, g) => s + g.frac_a_gt_95, 0) / gateStats.length).toFixed(4) : null,
  limits: { workgroupStorage: dev.limits.maxComputeWorkgroupStorageSize, invocations: dev.limits.maxComputeInvocationsPerWorkgroup,
            adapterWorkgroupStorage: adapter.limits.maxComputeWorkgroupStorageSize, adapterInvocations: adapter.limits.maxComputeInvocationsPerWorkgroup },
  kinds: kinds.join(""),
};
console.log(JSON.stringify(res));
if (OUT) await Deno.writeTextFile(`${OUT}/run.json`, JSON.stringify(res, null, 2));
