const a = await navigator.gpu.requestAdapter();
console.log("timestamp-query:", a.features.has("timestamp-query"), "| f16:", a.features.has("shader-f16"));
if (a.features.has("timestamp-query")) {
  const dev = await a.requestDevice({ requiredFeatures:["timestamp-query"] });
  const qs = dev.createQuerySet({ type:"timestamp", count:2 });
  const resolve = dev.createBuffer({ size:16, usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC });
  const read = dev.createBuffer({ size:16, usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ });
  const buf = dev.createBuffer({ size:4096*4, usage:GPUBufferUsage.STORAGE });
  const mod = dev.createShaderModule({ code:`@group(0) @binding(0) var<storage,read_write> o:array<f32>;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3u){ var s=0.0; for(var i=0u;i<10000u;i++){s+=sin(f32(i)+f32(g.x));} o[g.x%4096u]=s; }`});
  const pipe = dev.createComputePipeline({ layout:"auto", compute:{module:mod,entryPoint:"main"} });
  const bg = dev.createBindGroup({ layout:pipe.getBindGroupLayout(0), entries:[{binding:0,resource:{buffer:buf}}] });
  const enc = dev.createCommandEncoder();
  const p = enc.beginComputePass({ timestampWrites:{ querySet:qs, beginningOfPassWriteIndex:0, endOfPassWriteIndex:1 } });
  p.setPipeline(pipe); p.setBindGroup(0,bg); p.dispatchWorkgroups(64); p.end();
  enc.resolveQuerySet(qs,0,2,resolve,0); enc.copyBufferToBuffer(resolve,0,read,0,16);
  dev.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const t = new BigUint64Array(read.getMappedRange());
  console.log("GPU pass time:", Number(t[1]-t[0])/1e6, "ms (timestamps work!)");
} 
