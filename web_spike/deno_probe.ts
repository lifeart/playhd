const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) { console.log("NO ADAPTER"); Deno.exit(1); }
const dev = await adapter.requestDevice();
// trivial compute: out[i] = i*2
const N = 256;
const out = dev.createBuffer({ size: N*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC });
const mod = dev.createShaderModule({ code:`
  @group(0) @binding(0) var<storage,read_write> o:array<f32>;
  @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3u){ if(g.x<${N}u){o[g.x]=f32(g.x)*2.0;} }`});
const pipe = dev.createComputePipeline({ layout:"auto", compute:{module:mod,entryPoint:"main"} });
const bg = dev.createBindGroup({ layout:pipe.getBindGroupLayout(0), entries:[{binding:0,resource:{buffer:out}}] });
const rb = dev.createBuffer({ size:N*4, usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ });
const e = dev.createCommandEncoder(); const p=e.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0,bg); p.dispatchWorkgroups(Math.ceil(N/64)); p.end();
e.copyBufferToBuffer(out,0,rb,0,N*4); dev.queue.submit([e.finish()]);
await rb.mapAsync(GPUMapMode.READ); const a=new Float32Array(rb.getMappedRange());
console.log("adapter:", adapter.info?.device || adapter.info?.vendor || "ok", "| shader-f16:", adapter.features.has("shader-f16"), "| wgStorage:", adapter.limits.maxComputeWorkgroupStorageSize);
console.log("compute check out[10]=", a[10], "(expect 20)", a[10]===20 ? "OK" : "FAIL");
