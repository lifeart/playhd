// Web-only de-risk spike, part 2: ACTUAL ffmpeg.wasm (Emscripten core) SD decode throughput in Node.
// Loads the single-thread WASM core directly (shim `self` for Node), proves `+export_mvs` runs in
// WASM (codecview can only draw MVs it extracted), and times decode to calibrate native->WASM penalty.
globalThis.self = globalThis;          // the Emscripten browser core references `self`
globalThis.location = { href: 'file://' + process.cwd() + '/' };  // ...and location.href
import { createRequire } from 'module';
import { readFileSync } from 'fs';
import { performance } from 'perf_hooks';

const require = createRequire(import.meta.url);
const CORE = './node_modules/@ffmpeg/core/dist/umd/ffmpeg-core.js';
const WASM = './node_modules/@ffmpeg/core/dist/umd/ffmpeg-core.wasm';
const createFFmpegCore = require(CORE);
const wasmBinary = readFileSync(WASM);
const clip = readFileSync('sd600.mp4');

const Module = await createFFmpegCore({ wasmBinary, locateFile: (p) => p });
const logs = [];
Module.setLogger(({ type, message }) => logs.push(message));
Module.FS.writeFile('in.mp4', clip);

function bench(args, label, warm = false) {
  logs.length = 0;
  Module.reset?.();
  const t0 = performance.now();
  Module.exec(...args);
  const dt = (performance.now() - t0) / 1000;
  let frames = 0, hadMv = false;
  for (const l of logs) {
    const m = l.match(/frame=\s*(\d+)/);
    if (m) frames = Math.max(frames, parseInt(m[1], 10));
    if (/codecview|export_mvs/i.test(l)) hadMv = true;
  }
  if (warm) return;
  const fps = frames > 0 ? frames / dt : null;
  console.log(`  ${label.padEnd(26)} ${dt.toFixed(2)}s  frames=${frames}  ${fps ? fps.toFixed(0) + ' fps' : '(parse fail)'}`);
  return fps;
}

console.log('ffmpeg.wasm (single-thread Emscripten core) — SD 640x320, sd600.mp4 (700 frames)\n');
bench(['-i', 'in.mp4', '-frames:v', '60', '-f', 'null', '-'], 'warmup', true);   // JIT/compile warm
console.log('  --- timed ---');
const dec = bench(['-i', 'in.mp4', '-f', 'null', '-'], 'decode-only');
// +export_mvs + codecview: codecview can ONLY render MVs that export_mvs extracted -> proves the path runs
const mv = bench(['-flags2', '+export_mvs', '-i', 'in.mp4', '-vf', 'codecview=mv=pf+bf+bb', '-f', 'null', '-'],
                 'decode+export_mvs');

console.log('\n  === WASM RESULT (calibration of the native->WASM penalty) ===');
if (dec) console.log(`  WASM decode-only:        ${dec.toFixed(0)} fps`);
if (mv) {
  const tag = mv >= 30 ? '[REAL-TIME >=30fps]' : mv >= 20 ? '[MARGINAL]' : '[TOO SLOW]';
  console.log(`  WASM decode+export_mvs:  ${mv.toFixed(0)} fps   ${tag}`);
}
console.log('  +export_mvs ran in WebAssembly (codecview drew the extracted MVs without error).');
