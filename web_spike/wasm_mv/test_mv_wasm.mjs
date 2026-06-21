// Seam verification: the WASM libav build must surface the SAME motion vectors PyAV extracts natively.
// Loads mv_wasm (WASM FFmpeg), mounts sd600.mp4 into MEMFS, runs the extract_mvs main(), captures the CSV
// from stdout, counts MV rows per frame, and diffs against mv_reference.json (the PyAV reference).
import { readFileSync } from "node:fs";
import createModule from "./mv_wasm.mjs";

const clip = readFileSync(new URL("../sd600.mp4", import.meta.url));
const ref = JSON.parse(readFileSync(new URL("./mv_reference.json", import.meta.url), "utf8"));

const out = [];
const Module = await createModule({
  print: (s) => out.push(s),
  printErr: () => {},               // swallow av_dump_format
});
Module.FS.writeFile("/in.mp4", clip);
Module.callMain(["/in.mp4"]);

// parse CSV: framenum,source,blockw,blockh,srcx,srcy,dstx,dsty,flags,motion_x,motion_y,motion_scale
// The example's framenum is 1-based and in decoder OUTPUT order (= display order, post-reorder), so it
// aligns 1:1 with PyAV's display-order frames: WASM frame (i+1) == PyAV frame i.
const perFrame = new Map();
let total = 0, sample = null;
for (const line of out) {
  if (!line || line.startsWith("framenum")) continue;
  const f = line.split(","); if (f.length < 12) continue;
  const fn = parseInt(f[0]);
  perFrame.set(fn, (perFrame.get(fn) || 0) + 1);
  total++;
  if (!sample && +f[1] === -1) sample = { frame: fn, dst_x: +f[6], dst_y: +f[7], motion_x: +f[9], motion_y: +f[10], scale: +f[11] };
}

const window = ref.slice(0, 30);                       // the frames the PyAV reference covered
let matched = 0;
const rows = window.map((r) => { const w = perFrame.get(r.frame + 1) || 0; if (w === r.nmv) matched++; return { f: r.frame, ptype: r.ptype, ref: r.nmv, wasm: w }; });
const refTot = window.reduce((a, r) => a + r.nmv, 0);
const wasmTot = window.reduce((a, r) => a + (perFrame.get(r.frame + 1) || 0), 0);

console.log(`WASM libav: ${total} MV rows across ${perFrame.size} frames (whole clip); sample MV frame ${sample?.frame} dst=(${sample?.dst_x},${sample?.dst_y}) motion=(${sample?.motion_x},${sample?.motion_y})/${sample?.scale}`);
console.log(`per-frame match (WASM frameN+1 vs PyAV frameN), first ${window.length} frames:`);
for (const r of rows.slice(0, 8)) console.log(`  frame ${String(r.f).padStart(2)} ${r.ptype}: ref=${String(r.ref).padStart(4)} wasm=${String(r.wasm).padStart(4)} ${r.ref === r.wasm ? "✓" : "✗"}`);
console.log(`  ... ${matched}/${window.length} frames match exactly`);
const ok = total > 0 && matched === window.length && wasmTot === refTot;
console.log(ok
  ? `\n✅ SEAM VERIFIED: WASM libav surfaces motion vectors IDENTICAL to the native PyAV pipeline (${wasmTot} MVs over ${window.length} frames, exact).`
  : (total > 0 ? `\n⚠️  MVs surfaced (${wasmTot} vs ref ${refTot}) but per-frame counts differ — investigate.` : `\n❌ No MVs surfaced — export_mvs path broken in WASM.`));
process.exit(ok ? 0 : 1);
