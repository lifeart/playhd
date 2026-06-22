// Deno WebGPU parity + timing harness for the WGSL SPAN port.
//   deno run --allow-read span_bench.ts            (f32 + f16 parity vs PyTorch sr_ref.bin)
//   deno run --allow-read span_bench.ts --inter     (also dump piecewise intermediate parity)
// Compares the full WGSL SPAN graph output to span_data/sr_ref.bin (PyTorch forward, UNCLAMPED).
// Parity gate: f32 mean|Δ| < 1e-2 (near-exact), f16 < ~1e-2.
import { initSpanGPU, runSpan, Spec } from "./span_driver.ts";

const log = (s: string) => console.error(s);
const DIR = new URL("../span_data/", import.meta.url);
const rdF32 = (p: string) => new Float32Array(Deno.readFileSync(new URL(p, DIR)).buffer);

const spec: Spec = JSON.parse(Deno.readTextFileSync(new URL("spec.json", DIR)));
const weights = rdF32("weights.bin");
const lr = rdF32("lr_planar.bin");
const ref = rdF32("sr_ref.bin");
const wantInter = Deno.args.includes("--inter");
const interNames = ["F", "B1", "B5_2", "B6c", "catout", "up0"];

const stats = (a: Float32Array, b: Float32Array, n = Math.min(a.length, b.length)) => {
  let s = 0, mx = 0; for (let i = 0; i < n; i++) { const d = Math.abs(a[i] - b[i]); s += d; if (d > mx) mx = d; }
  return { mad: s / n, max: mx };
};
const range = (a: Float32Array) => { let lo = Infinity, hi = -Infinity; for (let i = 0; i < a.length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; } return [lo, hi]; };

const adapter = await navigator.gpu.requestAdapter();
const hasF16 = adapter!.features.has("shader-f16");
const feats: GPUFeatureName[] = [];
if (adapter!.features.has("timestamp-query")) feats.push("timestamp-query");
if (hasF16) feats.push("shader-f16" as GPUFeatureName);
const device = await adapter!.requestDevice({ requiredFeatures: feats });

log(`[span_bench] LR ${spec.W}x${spec.H} -> SR ${spec.out_w}x${spec.out_h}; ref=${ref.length} floats; f16-cap=${hasF16}`);
log(`PyTorch sr_ref range ${JSON.stringify(range(ref).map(v=>+v.toFixed(3)))}`);

async function one(f16: boolean) {
  const g = await initSpanGPU(device, weights, spec, f16);
  const { out, ms, inter } = await runSpan(g, lr, 5, 2, wantInter ? interNames : []);
  const { mad, max } = stats(out, ref);
  log(`\n=== SPAN WGSL ${f16 ? "f16" : "f32"} ===`);
  log(`OUT range ${JSON.stringify(range(out).map(v=>+v.toFixed(3)))}  time=${ms.toFixed(2)} ms`);
  log(`PARITY vs PyTorch: mean|Δ|=${mad.toExponential(3)}  max|Δ|=${max.toExponential(3)}  ${mad < 1e-2 ? "PARITY-OK" : "PARITY-FAIL"}`);
  if (wantInter) {
    log("  piecewise (WGSL vs PyTorch inter/*.bin):");
    for (const nm of interNames) {
      try {
        const pt = rdF32(`inter/${nm}.bin`);
        const st = stats(inter[nm], pt);
        log(`    ${nm.padEnd(8)} mean|Δ|=${st.mad.toExponential(3)} max=${st.max.toExponential(3)} ${st.mad < 1e-2 ? "ok" : "FAIL"}`);
      } catch (e) { log(`    ${nm}: ${e}`); }
    }
  }
  return { mad, ms };
}

const r32 = await one(false);
let r16: { mad: number; ms: number } | null = null;
let f16note = "f16 not attempted";
if (hasF16) {
  try { r16 = await one(true); }
  catch (e) {
    const msg = String(e);
    if (msg.includes("f16` enable") || msg.includes("not yet supported"))
      f16note = "f16 N/A in Deno (wgpu/Naga lacks `enable f16;`); generated WGSL is valid -> confirm in Chrome/Tint";
    else throw e;
  }
}

log(`\n[summary] f32 parity=${r32.mad.toExponential(2)} @ ${r32.ms.toFixed(2)}ms` +
  (r16 ? ` | f16 parity=${r16.mad.toExponential(2)} @ ${r16.ms.toFixed(2)}ms` : ` | ${f16note}`));
