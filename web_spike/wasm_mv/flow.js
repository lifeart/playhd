// JS port of prototype/derisk.py build_lr_flow: dense per-pixel LR fetch-flow from codec motion vectors.
// flow[y,x] = (dx,dy): the source pixel in the referenced frame is (x+dx, y+dy). NaN where no MV of the
// requested direction covers the pixel (intra/hole). `want`: 'past' (source<0), 'future' (source>0), 'all'.
// mvs = packed Int32Array, 10 ints/MV: [source, bw, bh, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale].
export function buildLrFlow(mvs, h, w, want = "all") {
  const n = mvs.length / 10;
  const fx = new Float32Array(h * w).fill(NaN);
  const fy = new Float32Array(h * w).fill(NaN);
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
  }
  return { fx, fy };
}
