// Adapter-aware narrow-then-micro-bench conv kernel selector (the PORTABILITY.md recommendation, shippable).
// Queries the adapter -> shortlists 2-3 configs from the per-family table -> validates limits -> micro-benchmarks
// each on this GPU (first load) -> caches the winner by adapter signature -> returns the chosen kernel.
// The candidate configs come from the VALIDATED parametric generator (kernel_gen.js -> same math as the
// wtile/combo kernels proven at parity 3e-7 / 2.7e-3), so the selector only chooses SPEED; correctness is given.
import { genConv } from "./kernel_gen.js";

const SAFE_THREADS = 256;                 // the only universally safe ceiling (>256 hard-fails despite adapter claims)
const f16pack = (f32) => new Uint16Array(new self.Float16Array(f32).buffer);

function familyOf(info) {                  // coarse hint only; never load-bearing (browsers may mask)
  const v = (info?.vendor || "").toLowerCase(), a = (info?.architecture || "").toLowerCase();
  if (a.includes("apple") || v.includes("apple")) return "apple";
  if (v.includes("nvidia") || v.includes("10de")) return "nvidia";
  if (v.includes("amd") || v.includes("1002") || a.includes("rdna")) return "amd";
  if (a.includes("mali")) return "mali";
  if (a.includes("adreno")) return "adreno";
  if (v.includes("intel") || v.includes("8086")) return "intel";
  return "unknown";
}

// per-family shortlist (from PORTABILITY.md §2b). Each entry -> genConv opts. f16 primary when available.
function shortlistFor(family, hasF16) {
  if (!hasF16) {                           // f32 fallback path (no shader-f16)
    const base = [{ OCB: 32, TW: 16, TH: 16, DB: true }];
    if (family !== "mali" && family !== "adreno") base.push({ OCB: 64, TW: 8, TH: 8, DB: true });  // sweep's fastest f32
    base.push({ OCB: 16, TW: 16, TH: 16, DB: false });   // most spill-proof
    return base.map((o) => ({ ...o, F16: false }));
  }
  const F = (o) => ({ ...o, F16: true, ACC: "f16", PROD: "f16" });
  switch (family) {
    case "apple": case "nvidia": case "amd":
      return [F({ OCB: 64, TW: 16, TH: 16, DB: true }), F({ OCB: 32, TW: 16, TH: 16, DB: true })];
    case "intel": case "adreno":
      return [F({ OCB: 32, TW: 16, TH: 16, DB: true }), F({ OCB: 16, TW: 16, TH: 16, DB: false }), F({ OCB: 64, TW: 16, TH: 16, DB: true })];
    case "mali":
      return [F({ OCB: 16, TW: 8, TH: 8, DB: false }), F({ OCB: 32, TW: 8, TH: 8, DB: false })];
    default:                               // unknown: broad sweep, micro-bench decides
      return [F({ OCB: 32, TW: 16, TH: 16, DB: true }), F({ OCB: 64, TW: 16, TH: 16, DB: true }), F({ OCB: 16, TW: 16, TH: 16, DB: false })];
  }
}

// micro-bench one config: time a handful of 64->64 layers at a small size (timestamp if available, else wall-clock).
async function timeCfg(device, hasTS, cfg, weightsBuf, layers, H, W, iters) {
  const FC = 64, plane = H * W, eb = cfg.f16 ? 2 : 4;
  device.pushErrorScope("validation");
  const mod = device.createShaderModule({ code: cfg.code });
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
  const perr = await device.popErrorScope();
  if (perr) return { ok: false, reason: "pipeline: " + perr.message.split("\n")[0] };
  const A = device.createBuffer({ size: FC * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const B = device.createBuffer({ size: FC * plane * eb, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(A, 0, new Uint8Array(FC * plane * eb));   // zeros — timing is data-independent
  // representative 64->64 layers (the body of the net); fall back to all if few
  const body = layers.filter((l) => l.in_c === 64 && l.out_c === 64).slice(0, 8);
  const use = body.length ? body : layers;
  let inb = A, outb = B; const passes = [];
  for (const ly of use) {
    const u = new Uint32Array([H, W, ly.in_c, ly.out_c, ly.w_off, ly.b_off, ly.prelu_off < 0 ? 0 : ly.prelu_off, ly.prelu_off < 0 ? 0 : 1]);
    const ub = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(ub, 0, u);
    const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: inb } }, { binding: 1, resource: { buffer: outb } }, { binding: 2, resource: { buffer: weightsBuf } }, { binding: 3, resource: { buffer: ub } }] });
    const [gx, gy, gz] = cfg.dispatch(ly, H, W); passes.push({ bg, gx, gy, gz }); [inb, outb] = [outb, inb];
  }
  const qs = hasTS ? device.createQuerySet({ type: "timestamp", count: 2 }) : null;
  const qR = hasTS ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const qRd = hasTS ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
  let best = 1e9;
  for (let t = 0; t < iters + 1; t++) {
    const enc = device.createCommandEncoder();
    passes.forEach((p, i) => {
      const tsw = hasTS ? (i === 0 ? { querySet: qs, beginningOfPassWriteIndex: 0 } : i === passes.length - 1 ? { querySet: qs, endOfPassWriteIndex: 1 } : undefined) : undefined;
      const cp = enc.beginComputePass(tsw ? { timestampWrites: tsw } : undefined);
      cp.setPipeline(pipe); cp.setBindGroup(0, p.bg); cp.dispatchWorkgroups(p.gx, p.gy, p.gz); cp.end();
    });
    if (hasTS) { enc.resolveQuerySet(qs, 0, 2, qR, 0); enc.copyBufferToBuffer(qR, 0, qRd, 0, 16); }
    const w0 = performance.now(); device.queue.submit([enc.finish()]);
    if (hasTS) { await qRd.mapAsync(GPUMapMode.READ); const ts = new BigUint64Array(qRd.getMappedRange().slice(0)); qRd.unmap(); if (t > 0) best = Math.min(best, Number(ts[1] - ts[0]) / 1e6); }
    else { await device.queue.onSubmittedWorkDone(); if (t > 0) best = Math.min(best, performance.now() - w0); }
  }
  return { ok: true, ms: best };
}

// Main entry. weightsF32 = the conv weights (Float32Array). Returns the chosen kernel + the shortlist results.
export async function selectKernel(adapter, device, { weightsF32, layers, H = 128, W = 128, iters = 3, useCache = true } = {}) {
  const hasF16 = adapter.features.has("shader-f16");
  const hasTS = adapter.features.has("timestamp-query");
  const family = familyOf(adapter.info);
  const smemLimit = adapter.limits.maxComputeWorkgroupStorageSize;
  const sig = `playhd.kern|${adapter.info?.vendor || "?"}|${adapter.info?.architecture || "?"}|f16=${hasF16}`;

  if (useCache && globalThis.localStorage) {
    const hit = localStorage.getItem(sig);
    if (hit) { try { const o = JSON.parse(hit); const cfg = genConv(o); return { chosen: cfg, results: [], cached: true, family, signature: sig }; } catch {} }
  }
  // weights buffers (pack f16 if any candidate needs it)
  const wF32 = device.createBuffer({ size: weightsF32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(wF32, 0, weightsF32);
  let wF16 = null; const getW = (f16) => { if (!f16) return wF32; if (!wF16) { const b = f16pack(weightsF32); wF16 = device.createBuffer({ size: b.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(wF16, 0, b); } return wF16; };

  const shortlist = shortlistFor(family, hasF16).map((o) => ({ opts: o, cfg: genConv(o) }));
  const results = [];
  for (const { opts, cfg } of shortlist) {
    if (cfg.threads > SAFE_THREADS) { results.push({ label: cfg.label, ok: false, reason: `${cfg.threads} threads > ${SAFE_THREADS}` }); continue; }
    if (cfg.smemBytes > smemLimit) { results.push({ label: cfg.label, ok: false, reason: `smem ${cfg.smemBytes} > ${smemLimit}` }); continue; }
    const r = await timeCfg(device, hasTS, cfg, getW(cfg.f16), layers, H, W, iters);
    results.push({ label: cfg.label, opts, cfg, ...r });
  }
  const valid = results.filter((r) => r.ok).sort((a, b) => a.ms - b.ms);
  const winner = valid[0];
  if (!winner) throw new Error("kernel-select: no valid candidate (all exceeded limits) — ship the fallback wtile");
  if (useCache && globalThis.localStorage) { try { localStorage.setItem(sig, JSON.stringify(winner.opts)); } catch {} }
  return { chosen: winner.cfg, winnerMs: winner.ms, results, cached: false, family, signature: sig, smemLimit, hasF16, hasTS };
}
