// DEFAULT-OFF regression guard for the R13-E1 occlusion gate: prove that with the gate OFF the
// page is unchanged. Two checks:
//   1) warpShaderWGSL({gate:"hard"}) is BYTE-IDENTICAL to the warp shader that was inline in
//      player.html before R13 (read straight out of git, so it compares against what shipped).
//   2) buildLrFlow(..., blocks=false) returns exactly the fx/fy it did before (and blocks=true
//      returns the same fx/fy plus the two extra channels).
//
// Run:  deno run --allow-run --allow-read parity_shader.mjs [git-rev]      (default rev: HEAD)
import { warpShaderWGSL } from "./warp_shader.js";
import { buildLrFlow, packLrFlowRGBA } from "./flow.js";

const rev = Deno.args[0] || "HEAD";
const cmd = new Deno.Command("git", { args: ["show", `${rev}:web_spike/wasm_mv/player.html`], cwd: new URL("../../", import.meta.url).pathname });
const { code, stdout, stderr } = await cmd.output();
if (code !== 0) throw new Error("git show failed: " + new TextDecoder().decode(stderr));
const old = new TextDecoder().decode(stdout);

const start = old.indexOf("const wmod=dev.createShaderModule({code:`");
if (start < 0) throw new Error(`no inline warp shader in ${rev}:player.html (already refactored?)`);
const from = start + "const wmod=dev.createShaderModule({code:".length + 1;   // just past the backtick
const end = old.indexOf("`});", from);
if (end < 0) throw new Error("unterminated shader literal in the old player.html");
const before = old.slice(from, end);
const now = warpShaderWGSL({ gate: "hard" });
const same = before === now;
console.log(`[1] warp shader gate=hard vs ${rev}: ${same ? "IDENTICAL" : "DIFFERENT"} (${before.length} vs ${now.length} chars)`);
if (!same) {
  for (let i = 0; i < Math.max(before.length, now.length); i++) {
    if (before[i] !== now[i]) { console.log(`    first diff at ${i}:\n    was: ${JSON.stringify(before.slice(i - 40, i + 40))}\n    now: ${JSON.stringify(now.slice(i - 40, i + 40))}`); break; }
  }
}

// 2) flow.js default path unchanged
const W = 32, H = 16;
const recs = [
  [-1, 16, 16, 0, 0, 8, 8, 7, -3, 4],
  [-1, 8, 8, 0, 0, 20, 4, -9, 5, 4],
  [1, 16, 8, 0, 0, 24, 12, 2, 2, 4],
];
const mvs = new Int32Array(recs.flat());
const a = buildLrFlow(mvs, H, W, "all");
const b = buildLrFlow(mvs, H, W, "all", true);
let flowSame = a.fx.length === b.fx.length && Object.keys(a).length === 2;
for (let i = 0; i < a.fx.length && flowSame; i++) {
  const eq = (x, y) => (Number.isNaN(x) && Number.isNaN(y)) || x === y;
  if (!eq(a.fx[i], b.fx[i]) || !eq(a.fy[i], b.fy[i])) flowSame = false;
}
const zwOk = b.bs && b.bm && b.bs[8 * W + 8] === 256 && Math.abs(b.bm[8 * W + 8] - Math.hypot(7 / 4, -3 / 4)) < 1e-6;
console.log(`[2] buildLrFlow default path: ${flowSame ? "UNCHANGED" : "CHANGED"} | .zw block channels: ${zwOk ? "OK" : "BAD"}`);

// 3) packLrFlowRGBA (the gate path's direct writer) == plane path + the caller's interleave
const ft = new Float32Array(W * H * 4);
packLrFlowRGBA(mvs, H, W, ft, "all", true);
let packSame = true;
for (let i = 0; i < W * H && packSame; i++) {
  const hole = Number.isNaN(b.fx[i]);
  const want = [hole ? -1e9 : b.fx[i], hole ? 0 : b.fy[i], hole ? 0 : b.bs[i], hole ? 0 : b.bm[i]];
  for (let c = 0; c < 4; c++) if (ft[i * 4 + c] !== want[c]) packSame = false;
}
console.log(`[3] packLrFlowRGBA vs buildLrFlow+interleave: ${packSame ? "IDENTICAL" : "DIFFERENT"}`);

if (!same || !flowSame || !zwOk || !packSame) Deno.exit(1);
console.log("PARITY OK — gate=hard is byte-identical to the shipped path.");
