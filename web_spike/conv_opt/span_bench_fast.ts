// Parity + timing harness for the FAST SPAN driver (optimized tiled 3x3 conv via genConv).
//   /tmp/deno_latest/deno run --allow-read --unstable-webgpu span_bench_fast.ts            (f16: naive vs fast, OCB sweep)
//   deno run --allow-read --unstable-webgpu span_bench_fast.ts --f32                       (f32 wiring sanity, system deno ok)
// Compares full WGSL SPAN output to span_data/sr_ref.bin (PyTorch, UNCLAMPED). Same device session
// for naive-vs-fast so the speedup is apples-to-apples. Gate: fast f16 mean|Δ| < 1e-2 (PARITY-OK).
import { initSpanGPU, runSpan } from "./span_driver.ts";
import { initSpanGPUFast, runSpanFast, Spec } from "./span_driver_fast.ts";

const log = (s: string) => console.error(s);
const DIR = new URL("../span_data/", import.meta.url);
const rdF32 = (p: string) => new Float32Array(Deno.readFileSync(new URL(p, DIR)).buffer);

const spec: Spec = JSON.parse(Deno.readTextFileSync(new URL("spec.json", DIR)));
const weights = rdF32("weights.bin");
const lr = rdF32("lr_planar.bin");
const ref = rdF32("sr_ref.bin");
const wantF32 = Deno.args.includes("--f32");
const ocbArg = Deno.args.find(a => a.startsWith("--ocb="));
const OCBS = ocbArg ? ocbArg.slice(6).split(",").map(Number) : [16, 24, 48];

const stats = (a: Float32Array, b: Float32Array, n = Math.min(a.length, b.length)) => {
  let s = 0, mx = 0; for (let i = 0; i < n; i++) { const d = Math.abs(a[i] - b[i]); s += d; if (d > mx) mx = d; }
  return { mad: s / n, max: mx };
};
const range = (a: Float32Array) => { let lo = Infinity, hi = -Infinity; for (let i = 0; i < a.length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; } return [lo, hi]; };

const adapter = await navigator.gpu.requestAdapter();
const hasF16 = adapter!.features.has("shader-f16");
const f16 = !wantF32 && hasF16;
const feats: GPUFeatureName[] = [];
if (adapter!.features.has("timestamp-query")) feats.push("timestamp-query");
if (hasF16) feats.push("shader-f16" as GPUFeatureName);
const device = await adapter!.requestDevice({ requiredFeatures: feats });

log(`[span_bench_fast] LR ${spec.W}x${spec.H} -> SR ${spec.out_w}x${spec.out_h}; precision=${f16 ? "f16" : "f32"}; f16-cap=${hasF16}`);
log(`PyTorch sr_ref range ${JSON.stringify(range(ref).map(v => +v.toFixed(3)))}`);
const npix = spec.W * spec.H;

// --- NAIVE baseline (same session) ---
const gN = await initSpanGPU(device, weights, spec, f16);
const rN = await runSpan(gN, lr, 5, 2);
const sN = stats(rN.out, ref);
log(`\n=== NAIVE ${f16 ? "f16" : "f32"} ===  time=${rN.ms.toFixed(2)} ms  (${(rN.ms * 1e6 / npix).toFixed(1)} ns/px)`);
log(`  PARITY vs PyTorch: mean|Δ|=${sN.mad.toExponential(3)} max|Δ|=${sN.max.toExponential(3)} ${sN.mad < 1e-2 ? "PARITY-OK" : "PARITY-FAIL"}`);

// --- FAST sweep (same session) ---
type Row = { OCB: number; ms: number; mad: number; max: number; label: string };
const rows: Row[] = [];
for (const OCB of OCBS) {
  try {
    const gF = await initSpanGPUFast(device, weights, spec, f16, { OCB });
    const rF = await runSpanFast(gF, lr, 5, 2);
    const sF = stats(rF.out, ref);
    rows.push({ OCB, ms: rF.ms, mad: sF.mad, max: sF.max, label: gF.convLabel });
    log(`\n=== FAST ${gF.convLabel} ===  time=${rF.ms.toFixed(2)} ms  (${(rF.ms * 1e6 / npix).toFixed(1)} ns/px)  speedup=${(rN.ms / rF.ms).toFixed(2)}x`);
    log(`  OUT range ${JSON.stringify(range(rF.out).map(v => +v.toFixed(3)))}`);
    log(`  PARITY vs PyTorch: mean|Δ|=${sF.mad.toExponential(3)} max|Δ|=${sF.max.toExponential(3)} ${sF.mad < 1e-2 ? "PARITY-OK" : "PARITY-FAIL"}`);
  } catch (e) {
    const msg = String(e);
    if (msg.includes("f16` enable") || msg.includes("not yet supported")) { log(`\nFAST OCB${OCB}: f16 N/A in this Deno build (needs /tmp/deno_latest/deno)`); break; }
    log(`\nFAST OCB${OCB} FAILED: ${msg}`);
  }
}

if (rows.length) {
  const best = rows.filter(r => r.mad < 1e-2).sort((a, b) => a.ms - b.ms)[0] ?? rows.sort((a, b) => a.ms - b.ms)[0];
  log(`\n[summary] ${f16 ? "f16" : "f32"}  naive=${rN.ms.toFixed(2)}ms (parity ${sN.mad.toExponential(2)})`);
  for (const r of rows) log(`  fast OCB${String(r.OCB).padStart(2)}  ${r.ms.toFixed(2)}ms  ${(rN.ms / r.ms).toFixed(2)}x  parity ${r.mad.toExponential(2)} ${r.mad < 1e-2 ? "OK" : "FAIL"}`);
  log(`\n[BEST] OCB${best.OCB}: ${best.ms.toFixed(2)}ms  ${(rN.ms / best.ms).toFixed(2)}x vs naive  parity ${best.mad.toExponential(2)}  (${(best.ms * 1e6 / npix).toFixed(1)} ns/px @ ${spec.W}x${spec.H})`);
}
