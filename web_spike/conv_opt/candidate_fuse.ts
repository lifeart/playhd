// ============================================================================
// CROSS-LAYER FUSION kernel for the 64->64 middle convs (f16).
//
// THESIS: the per-layer combo kernel is latency/occupancy-bound, not ALU- or
// bandwidth-bound. Every layer round-trips its full 64-ch feature map through
// GLOBAL memory and the next layer reads it back -> 34 high-latency global
// round-trips. FUSION computes K consecutive 64->64 layers in ONE dispatch,
// keeping the K-1 intermediate feature maps ON-CHIP (workgroup shared), so they
// never touch global. That directly attacks the round-trip latency.
//
// One workgroup -> one T x T output tile, all 64 oc (OCB=64 -> gz=1, so each
// intermediate channel lives in shared exactly once). To produce a valid T x T
// tile after K stacked 3x3 convs you need an (T+2K) x (T+2K) input halo (each
// 3x3 layer shrinks the valid region by 1 px/side). Plan:
//   * load the 64-ch input halo (S=T+2K) into buf0 (workgroup), zero-pad outside.
//   * layer l: each actv thread owns one output pixel, holds 16 vec4<f16> acc
//     (64 oc), loops ic 0..64 reading the previous feature tile from shared and
//     the current ic's weights (cached in sW), MAC, bias+PReLU, writes the next
//     (smaller) tile back to shared. workgroupBarrier() between layers.
//   * final layer writes the T x T x 64 result to GLOBAL fout.
//
// SHARED BUDGET (32768 B limit). Feature tiles MUST be on chip (the whole point):
//   buf0 = S*S*64 f16, buf1 = (S-2)*(S-2)*64 f16 (ping-pong; dims only shrink so
//   these two physical buffers cover any K). For S=12 (T=8,K=2): buf0=18432 +
//   buf1=12800 = 31232 B, leaving 1536 B -> a SINGLE ic weight buffer sW (1152 B,
//   64 oc x 9 taps as 16 vec4<f16>). No room to double-buffer weights -> 2
//   barriers/ic. This is fusion's structural cost: features on chip crowd out the
//   weight double-buffer that combo relies on.
//
// Weights for the K layers are read from global per-ic into sW and reused across
// the whole tile (combo's amortization); caching is mandatory -- reading weights
// straight from global per actv pixel is ~12 GB of redundant traffic per pass.
//
// Tunable: K (fused depth), T (output tile). S = T+2K is capped at ~12 by the
// 32 KB shared limit, so deeper K forces smaller T -> more halo recompute. The
// honest risk: small T -> few output pixels/workgroup -> low occupancy on an
// already occupancy-bound kernel.
// ============================================================================
export function makeFused(opts: { K: number; T: number; ACC?: "f16" | "f32" }) {
  const K = opts.K, T = opts.T, ACC = opts.ACC ?? "f16";
  const S = T + 2 * K;          // input-halo side for the whole fused block
  const P = S - 2;              // layer-0 output side = max -> threads = P*P
  const NT = P * P;             // one thread per layer-0 output pixel
  const G = 16;                 // 64/4 vec4 accumulators per thread (OCB=64)
  const buf0Sz = S * S * 64;
  const buf1Sz = (S - 2) * (S - 2) * 64;
  const accZ = ACC === "f16" ? "vec4<f16>(0.0)" : "vec4<f32>(0.0)";

  const macTap = Array.from({ length: G }, (_, g) =>
    ACC === "f16"
      ? `        a${g} += sW[${g * 9}u + wk] * inv;`
      : `        a${g} += vec4<f32>(sW[${g * 9}u + wk]) * f32(inv);`
  ).join("\n");

  function layerCode(l: number): string {
    const din = S - 2 * l;
    const dout = din - 2;
    const src = (l % 2 === 0) ? "buf0" : "buf1";
    const isFinal = (l === K - 1);
    const dstBuf = (l % 2 === 0) ? "buf1" : "buf0";
    const accDecl = Array.from({ length: G }, (_, g) => `    var a${g} = ${accZ};`).join("\n");
    const writeBody = Array.from({ length: G }, (_, g) =>
      [0, 1, 2, 3].map((j) => {
        const ch = g * 4 + j;
        const comp = ["x", "y", "z", "w"][j];
        const vexpr = `f32(a${g}.${comp}) + f32(Wt[boff+${ch}u])`;
        const prelu = `if(hasp==1u && v<0.0){ v = v*f32(Wt[poff+${ch}u]); }`;
        if (isFinal) {
          return `      { var v = ${vexpr}; ${prelu} if(gx<u.W && gy<u.H){ fout[${ch}u*plane + gy*u.W + gx] = f16(v); } }`;
        }
        return `      { var v = ${vexpr}; ${prelu} ${dstBuf}[${ch}u*${dout * dout}u + oy*${dout}u + ox] = f16(v); }`;
      }).join("\n")
    ).join("\n");
    const finalCoords = isFinal ? `    let gx = outX0 + ox; let gy = outY0 + oy;` : "";
    return `
  // ---- fused layer ${l}: din=${din} dout=${dout} src=${src} -> ${isFinal ? "GLOBAL" : dstBuf} ----
  {
    let woff = u.lp[${l}].x; let boff = u.lp[${l}].y; let poff = u.lp[${l}].z; let hasp = u.lp[${l}].w;
    let actv = lidx < ${dout * dout}u;
    var ox = 0u; var oy = 0u;
    if(actv){ ox = lidx % ${dout}u; oy = lidx / ${dout}u; }
${finalCoords}
${accDecl}
    for(var ic=0u; ic<64u; ic++){
      for(var t=lidx; t<144u; t+=${NT}u){
        let g = t / 9u; let k = t % 9u; let oc0 = g*4u; let wb = woff + ic*9u + k;
        sW[t] = vec4<f16>(Wt[wb+(oc0+0u)*576u], Wt[wb+(oc0+1u)*576u], Wt[wb+(oc0+2u)*576u], Wt[wb+(oc0+3u)*576u]);
      }
      workgroupBarrier();
      if(actv){
        let inbase = ic*${din * din}u;
        for(var ky=0u; ky<3u; ky++){
          for(var kx=0u; kx<3u; kx++){
            let inv = ${src}[inbase + (oy+ky)*${din}u + (ox+kx)];
            let wk = ky*3u+kx;
${macTap}
          }
        }
      }
      workgroupBarrier();
    }
    if(actv){
${writeBody}
    }
${isFinal ? "" : "    workgroupBarrier();"}
  }`;
  }

  const layers = Array.from({ length: K }, (_, l) => layerCode(l)).join("\n");
  const buf1Decl = K >= 2 ? `var<workgroup> buf1: array<f16, ${buf1Sz}>;` : "";

  const code = `
enable f16;
struct FP { H:u32, W:u32, _0:u32, _1:u32, lp: array<vec4<u32>, ${K}> };
@group(0) @binding(0) var<storage,read> fin:array<f16>;
@group(0) @binding(1) var<storage,read_write> fout:array<f16>;
@group(0) @binding(2) var<storage,read> Wt:array<f16>;
@group(0) @binding(3) var<uniform> u:FP;

var<workgroup> buf0: array<f16, ${buf0Sz}>;
${buf1Decl}
var<workgroup> sW: array<vec4<f16>, 144>;

@compute @workgroup_size(${NT})
fn main(@builtin(workgroup_id) wid:vec3u, @builtin(local_invocation_index) lidx:u32){
  let plane = u.H * u.W;
  let outX0 = wid.x * ${T}u;
  let outY0 = wid.y * ${T}u;
  let halX0 = i32(outX0) - ${K};
  let halY0 = i32(outY0) - ${K};
  // load 64-ch halo (S x S) into buf0, zero-pad outside the image
  for(var t=lidx; t<${buf0Sz}u; t+=${NT}u){
    let c = t / ${S * S}u;
    let r = t % ${S * S}u;
    let hy = r / ${S}u;
    let hx = r % ${S}u;
    let gx = halX0 + i32(hx);
    let gy = halY0 + i32(hy);
    var v = f16(0.0);
    if(gx>=0 && gx<i32(u.W) && gy>=0 && gy<i32(u.H)){ v = fin[c*plane + u32(gy)*u.W + u32(gx)]; }
    buf0[t] = v;
  }
  workgroupBarrier();
${layers}
}`;

  return {
    f16: true,
    code,
    K, T, S, NT,
    ubBytes: 16 + K * 16,
    dispatch: (H: number, W: number): [number, number, number] => [Math.ceil(W / T), Math.ceil(H / T), 1],
  };
}
export default makeFused;
