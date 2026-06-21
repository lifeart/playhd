const a = await navigator.gpu.requestAdapter();
const F = [...a.features].sort();
console.log("=== adapter features ===");
console.log(F.join("\n"));
console.log("\n=== platform-relevant limits ===");
const L = a.limits;
for (const k of ["maxComputeWorkgroupSizeX","maxComputeInvocationsPerWorkgroup","maxComputeWorkgroupStorageSize","minSubgroupSize","maxSubgroupSize","maxComputeWorkgroupsPerDimension","maxStorageBufferBindingSize"]) {
  console.log(`${k}: ${L[k]}`);
}
console.log("\nsubgroups feature:", a.features.has("subgroups"), "| subgroups-f16:", a.features.has("subgroups-f16"));
