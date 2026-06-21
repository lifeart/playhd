// Test the clean {frame, mvs} JS API: decode sd600.mp4 in WASM, pull RGB24 pixels + packed MVs per frame
// from the heap, and verify per-frame MV counts match the PyAV reference exactly. Writes frame-0 RGB stats
// for the separate pixel cross-check.
import { readFileSync, writeFileSync } from "node:fs";
import createModule from "./mv_decode.mjs";

const clip = readFileSync(new URL("../sd600.mp4", import.meta.url));
const ref = JSON.parse(readFileSync(new URL("./mv_reference.json", import.meta.url), "utf8"));

const M = await createModule({ printErr: () => {} });
M.FS.writeFile("/in.mp4", clip);
const open = M.cwrap("mvdec_open", "number", ["string"]);
const next = M.cwrap("mvdec_next", "number", []);
const W = M.cwrap("mvdec_width", "number", []);
const H = M.cwrap("mvdec_height", "number", []);
const rgbPtr = M.cwrap("mvdec_rgb", "number", []);
const nmvF = M.cwrap("mvdec_nmv", "number", []);
const mvsPtr = M.cwrap("mvdec_mvs", "number", []);

if (open("/in.mp4") !== 0) { console.log("❌ mvdec_open failed"); process.exit(1); }
const w = W(), h = H();
let fi = 0, matched = 0, frame0mean = null, sampleMV = null;
while (fi < 30) {
  const r = next();
  if (r !== 0) break;
  const n = nmvF();
  const expect = ref[fi]?.nmv ?? -1;
  if (n === expect) matched++;
  if (fi === 1 && n > 0) {                       // first P-frame: dump a sample MV
    const mv = M.HEAP32.subarray(mvsPtr() >> 2, (mvsPtr() >> 2) + 10);
    sampleMV = { source: mv[0], w: mv[1], h: mv[2], src_x: mv[3], src_y: mv[4], dst_x: mv[5], dst_y: mv[6], motion_x: mv[7], motion_y: mv[8], scale: mv[9] };
  }
  if (fi === 0) {                                 // RGB sanity + stats for the pixel cross-check
    const rgb = M.HEAPU8.subarray(rgbPtr(), rgbPtr() + w * h * 3);
    let s = 0; for (let i = 0; i < rgb.length; i++) s += rgb[i];
    frame0mean = s / rgb.length;
    writeFileSync(new URL("./wasm_frame0_rgb.bin", import.meta.url), Buffer.from(rgb));   // for pixel diff vs PyAV
  }
  fi++;
}
console.log(`clean API: decoded ${fi} frames at ${w}x${h} (RGB24 + packed MVs from heap)`);
console.log(`frame-0 RGB mean code: ${frame0mean?.toFixed(1)} (non-trivial: ${frame0mean > 5 && frame0mean < 250})`);
console.log(`sample MV (frame 1, first P): source=${sampleMV?.source} dst=(${sampleMV?.dst_x},${sampleMV?.dst_y}) motion=(${sampleMV?.motion_x},${sampleMV?.motion_y})/${sampleMV?.scale} blk=${sampleMV?.w}x${sampleMV?.h}`);
console.log(`per-frame MV count match vs PyAV reference: ${matched}/${fi}`);
M.cwrap("mvdec_close", null, [])();
const ok = fi >= 30 && matched === fi && frame0mean > 5;
console.log(ok ? "\n✅ Clean {frame, mvs} API works: RGB pixels + motion vectors per frame, MV counts identical to native." :
  "\n❌ API mismatch (see counts above).");
process.exit(ok ? 0 : 1);
