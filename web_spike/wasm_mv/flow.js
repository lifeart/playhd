// JS port of prototype/derisk.py build_lr_flow: dense per-pixel LR fetch-flow from codec motion vectors.
// flow[y,x] = (dx,dy): the source pixel in the referenced frame is (x+dx, y+dy). NaN where no MV of the
// requested direction covers the pixel (intra/hole). `want`: 'past' (source<0), 'future' (source>0), 'all'.
// mvs = packed Int32Array, 10 ints/MV: [source, bw, bh, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale].
//
// `blocks` (R13-E1, default OFF => byte-identical output): also emit the two FREE per-block codec
// signals the MV records already carry, for the flow texture's unused .zw channels —
//   bs[y,x] = partition area in px (16..256; sub-16 partitions predict much larger warp error)
//   bm[y,x] = |MV| in LR px (long MVs predict larger warp error)
// Same last-writer-wins overwrite order as fx/fy, so all four channels always describe the SAME
// MV record for a pixel. 0 where no MV covers the pixel (an intra hole is flagged by fx=NaN).
export function buildLrFlow(mvs, h, w, want = "all", blocks = false) {
  const n = mvs.length / 10;
  const fx = new Float32Array(h * w).fill(NaN);
  const fy = new Float32Array(h * w).fill(NaN);
  const bs = blocks ? new Float32Array(h * w) : null;
  const bm = blocks ? new Float32Array(h * w) : null;
  for (let i = 0; i < n; i++) {
    const o = i * 10;
    const s = mvs[o];
    if (want === "past" && s >= 0) continue;       // keep past refs only
    if (want === "future" && s <= 0) continue;     // keep future refs only
    const ms = mvs[o + 9] || 1;                     // motion_scale (0 -> 1)
    const dx = mvs[o + 7] / ms;                     // motion_x / scale
    const dy = mvs[o + 8] / ms;
    const bw = mvs[o + 1], bh = mvs[o + 2];         // block size; dst is the block CENTER
    const cx = mvs[o + 5], cy = mvs[o + 6];
    const x0 = Math.max(cx - (bw >> 1), 0), x1 = Math.min(cx + (bw >> 1), w);
    const y0 = Math.max(cy - (bh >> 1), 0), y1 = Math.min(cy + (bh >> 1), h);
    for (let y = y0; y < y1; y++) { const row = y * w; for (let x = x0; x < x1; x++) { fx[row + x] = dx; fy[row + x] = dy; } }
    if (blocks) {
      const area = bw * bh, mag = Math.hypot(dx, dy);
      for (let y = y0; y < y1; y++) { const row = y * w; for (let x = x0; x < x1; x++) { bs[row + x] = area; bm[row + x] = mag; } }
    }
  }
  return blocks ? { fx, fy, bs, bm } : { fx, fy };
}

// Direct rgba32float packer: writes the flow texture's interleaved [fx,fy,area,|mv|] straight from
// the MV records, skipping the two intermediate planes AND the full-image interleave pass that
// buildLrFlow + the caller's loop do. Holes keep the sentinel fx=-1e9 (same encoding the shader
// tests). Measured on 640x320, main-thread JS, on a CONTENDED box (R13-E1; load avg 25-34, so read
// the ratios, not the absolutes) over two sessions: plane path .xy only 2.0 / 3.4 ms per frame,
// plane path + .zw 6.6 / 4.5 ms, THIS packer 1.1 / 1.6 ms (.xy) and 1.2 / 1.4 ms (.xy+.zw).
// Consistent conclusion: the per-block channels are ~free ONLY if packed directly, and packing
// directly is ~2x faster than the shipped plane path even without them.
// `ft` must be a Float32Array of w*h*4 (reused across frames).
let _sentinel = null;
export function packLrFlowRGBA(mvs, h, w, ft, want = "all", blocks = true) {
  const n = mvs.length / 10, np = w * h * 4;
  if (!_sentinel || _sentinel.length !== np) {          // hole template: fx=-1e9, fy=z=w=0
    _sentinel = new Float32Array(np);
    for (let i = 0; i < np; i += 4) _sentinel[i] = -1e9;
  }
  ft.set(_sentinel);
  for (let i = 0; i < n; i++) {
    const o = i * 10;
    const s = mvs[o];
    if (want === "past" && s >= 0) continue;
    if (want === "future" && s <= 0) continue;
    const ms = mvs[o + 9] || 1;
    const dx = mvs[o + 7] / ms, dy = mvs[o + 8] / ms;
    const bw = mvs[o + 1], bh = mvs[o + 2];
    const cx = mvs[o + 5], cy = mvs[o + 6];
    const x0 = Math.max(cx - (bw >> 1), 0), x1 = Math.min(cx + (bw >> 1), w);
    const y0 = Math.max(cy - (bh >> 1), 0), y1 = Math.min(cy + (bh >> 1), h);
    const area = blocks ? bw * bh : 0, mag = blocks ? Math.hypot(dx, dy) : 0;
    for (let y = y0; y < y1; y++) {
      let p = (y * w + x0) * 4;
      for (let x = x0; x < x1; x++) { ft[p] = dx; ft[p + 1] = dy; ft[p + 2] = area; ft[p + 3] = mag; p += 4; }
    }
  }
  return ft;
}
