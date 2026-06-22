import { initSpanGPUFast, makeSpanRunner } from "./span_driver_fast.ts";
const DIR=new URL("../span_data/",import.meta.url);
const spec=JSON.parse(Deno.readTextFileSync(new URL("spec.json",DIR)));
const weights=new Float32Array(Deno.readFileSync(new URL("weights.bin",DIR)).buffer);
const lr=new Float32Array(Deno.readFileSync(new URL("lr_planar.bin",DIR)).buffer);
const ref=new Float32Array(Deno.readFileSync(new URL("sr_ref.bin",DIR)).buffer);
const adapter=await navigator.gpu.requestAdapter();
const dev=await adapter.requestDevice({requiredFeatures:adapter.features.has("shader-f16")?["shader-f16"]:[]});
const g=await initSpanGPUFast(dev,weights,spec,adapter.features.has("shader-f16"),{OCB:48});
const R=makeSpanRunner(g);
const F16=globalThis.Float16Array; const pack=f=>g.f16?new Uint16Array(new F16(f).buffer):f;
dev.queue.writeBuffer(R.IN,0,pack(lr));
const e=dev.createCommandEncoder(); R.recordInto(e); dev.queue.submit([e.finish()]);
// readback OUT
const outN=3*R.OH*R.OW; const rb=dev.createBuffer({size:outN*g.eb,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
const e2=dev.createCommandEncoder(); e2.copyBufferToBuffer(R.OUT,0,rb,0,outN*g.eb); dev.queue.submit([e2.finish()]); await dev.queue.onSubmittedWorkDone();
await rb.mapAsync(GPUMapMode.READ); const out=g.f16?new Float32Array(new F16(rb.getMappedRange().slice(0))):new Float32Array(rb.getMappedRange().slice(0)); rb.unmap();
let s=0,mx=0; for(let i=0;i<outN;i++){const d=Math.abs(out[i]-ref[i]); s+=d; if(d>mx)mx=d;}
console.log(`makeSpanRunner: OH×OW=${R.OH}×${R.OW}, outMean=${(out.reduce((a,b)=>a+b,0)/outN).toFixed(3)}, parity vs PyTorch mean|Δ|=${(s/outN).toExponential(2)} max=${mx.toExponential(2)} ${s/outN<1e-2?"OK":"FAIL"}`);
