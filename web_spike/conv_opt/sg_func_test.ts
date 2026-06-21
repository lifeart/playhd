// Functional test: do subgroup OPERATIONS actually work in this deno/wgpu?
// Test subgroupAdd (reduction), subgroupBroadcast, subgroupShuffle independently.
const adapter = await navigator.gpu.requestAdapter();
console.log("subgroups feature:", adapter.features.has("subgroups"));
const dev = await adapter.requestDevice({ requiredFeatures: ["subgroups" as GPUFeatureName] });
const N = 32;
const code = `
enable subgroups;
@group(0) @binding(0) var<storage,read_write> out:array<u32>;
@compute @workgroup_size(${N},1,1)
fn main(@builtin(local_invocation_index) lidx:u32,
        @builtin(subgroup_invocation_id) sid:u32,
        @builtin(subgroup_size) ssz:u32){
  let v = f32(lidx) + 1.0;            // 1,2,3,...,32
  let s = subgroupAdd(v);            // sum within subgroup
  let b = subgroupBroadcast(v, 5u);  // value from lane 5 (=6.0)
  let sh = subgroupShuffle(v, lidx ^ 1u); // swap neighbor pairs
  // pack results: out[0..]=sum*1000 ; we store per-lane to inspect
  out[lidx*4u+0u] = u32(s);          // expect 528 if full 32-lane subgroup
  out[lidx*4u+1u] = u32(b);          // expect 6 everywhere
  out[lidx*4u+2u] = u32(sh);         // expect neighbor-swapped
  out[lidx*4u+3u] = ssz;             // expect 32
}`;
const buf = dev.createBuffer({ size: N * 4 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const rb = dev.createBuffer({ size: N * 4 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
dev.pushErrorScope("validation");
const mod = dev.createShaderModule({ code });
const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
const pe = await dev.popErrorScope(); if (pe) { console.log("PIPELINE ERROR:", pe.message); Deno.exit(1); }
const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
const enc = dev.createCommandEncoder();
const cp = enc.beginComputePass(); cp.setPipeline(pipe); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(1, 1, 1); cp.end();
enc.copyBufferToBuffer(buf, 0, rb, 0, N * 4 * 4); dev.queue.submit([enc.finish()]);
await rb.mapAsync(GPUMapMode.READ);
const d = new Uint32Array(rb.getMappedRange().slice(0)); rb.unmap();
console.log("lane0: sum=%d (expect 528) bcast=%d (expect 6) shuffle=%d (expect 2) ssz=%d (expect 32)", d[0], d[1], d[2], d[3]);
console.log("lane1: sum=%d bcast=%d shuffle=%d (expect 1) ssz=%d", d[4], d[5], d[6], d[7]);
console.log("lane7: sum=%d bcast=%d (expect 6) shuffle=%d (expect 7) ssz=%d", d[28], d[29], d[30], d[31]);
const ok = d[0] === 528 && d[1] === 6 && d[2] === 2 && d[3] === 32;
console.log(ok ? "SUBGROUP OPS WORK" : "SUBGROUP OPS BROKEN/NOOP");
