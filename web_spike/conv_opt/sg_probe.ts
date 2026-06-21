// Probe the subgroup_invocation_id <-> local_invocation_index mapping for a 16x16 workgroup
// on this wgpu->Metal device. Essential for correct subgroupShuffle indexing in a 2D stencil.
const adapter = await navigator.gpu.requestAdapter();
const dev = await adapter.requestDevice({ requiredFeatures: ["subgroups" as GPUFeatureName] });
const TW = 16, TH = 16, N = TW * TH;
const code = `
enable subgroups;
@group(0) @binding(0) var<storage,read_write> out:array<u32>;
@compute @workgroup_size(${TW},${TH},1)
fn main(@builtin(local_invocation_index) lidx:u32,
        @builtin(subgroup_invocation_id) sid:u32,
        @builtin(subgroup_size) ssz:u32){
  // pack: sid in low 16, ssz in high 16
  out[lidx] = sid | (ssz << 16u);
}`;
const buf = dev.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const rb = dev.createBuffer({ size: N * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const mod = dev.createShaderModule({ code });
const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
const enc = dev.createCommandEncoder();
const cp = enc.beginComputePass(); cp.setPipeline(pipe); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(1, 1, 1); cp.end();
enc.copyBufferToBuffer(buf, 0, rb, 0, N * 4); dev.queue.submit([enc.finish()]);
await rb.mapAsync(GPUMapMode.READ);
const d = new Uint32Array(rb.getMappedRange().slice(0)); rb.unmap();
const ssz = d[0] >>> 16;
console.log("subgroup_size =", ssz);
// print sid grid (lx across, ly down) to see geometry
let linear = true;
for (let lidx = 0; lidx < N; lidx++) { if ((d[lidx] & 0xffff) !== lidx % ssz) { linear = false; } }
console.log("sid == lidx % ssz everywhere? ", linear);
console.log("first 3 rows of sid (16 wide):");
for (let ly = 0; ly < 3; ly++) {
  let row = "";
  for (let lx = 0; lx < TW; lx++) row += String(d[ly * TW + lx] & 0xffff).padStart(3, " ");
  console.log(`ly=${ly}: ${row}`);
}
