import { initSpanGPUFast, makeSpanRunner } from "./span_driver_fast.ts";
const DIR=new URL("../span_data/",import.meta.url);
const spec=JSON.parse(Deno.readTextFileSync(new URL("spec.json",DIR)));
const weights=new Float32Array(Deno.readFileSync(new URL("weights.bin",DIR)).buffer);
const adapter=await navigator.gpu.requestAdapter();
const f16=adapter.features.has("shader-f16");
const dev=await adapter.requestDevice({requiredFeatures:f16?["shader-f16"]:[]});
for(const [H,W] of [[160,320],[320,640]]){   // 160×320 (known-good) vs 320×640 (player's override)
  const g=await initSpanGPUFast(dev,weights,{...spec,H,W},f16,{OCB:48});
  const R=makeSpanRunner(g);
  const lr=new Float32Array(3*H*W); for(let i=0;i<3*H*W;i++) lr[i]=((i*7)%101)/101;  // synthetic non-zero
  const F16=globalThis.Float16Array; const pack=(x)=>f16?new Uint16Array(new F16(x).buffer):x;
  dev.queue.writeBuffer(R.IN,0,pack(lr));
  const e=dev.createCommandEncoder(); R.recordInto(e); dev.queue.submit([e.finish()]);
  const outN=3*R.OH*R.OW; const rb=dev.createBuffer({size:outN*g.eb,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const e2=dev.createCommandEncoder(); e2.copyBufferToBuffer(R.OUT,0,rb,0,outN*g.eb); dev.queue.submit([e2.finish()]); await dev.queue.onSubmittedWorkDone();
  await rb.mapAsync(GPUMapMode.READ); const out=f16?new Float32Array(new F16(rb.getMappedRange().slice(0))):new Float32Array(rb.getMappedRange().slice(0)); rb.unmap();
  let s=0; for(let i=0;i<outN;i++)s+=out[i];
  console.log(`${H}×${W} -> OH×OW=${R.OH}×${R.OW}, outMean=${(s/outN).toFixed(4)}`);
}
