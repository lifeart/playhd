// ============================================================================
// Winograd F(2x2, 3x3) fused conv (f32). One workgroup processes a block of
// TBW x TBH output TILES (each tile = 2x2 output px), for OCB output channels.
//
// Per input channel ic:
//   * load the (2*TBW+2)x(2*TBH+2) input halo into shared (zero-pad borders)
//   * cooperatively compute the weight transform U = G g G^T (16 vals/oc) for the
//     OCB output channels into shared sU  <-- THE AMORTIZATION (transform once per
//     workgroup, reused across all TBW*TBH tiles), mirroring wtile's weight cache
//   * each thread transforms its own 4x4 input tile V = B^T d B (adds only) and
//     accumulates M[oc] += U[oc] (elementwise, 16 mults) over the transform domain
// After the ic loop: each thread inverse-transforms M[oc] = A^T M A -> 2x2 output,
// adds bias, optional PReLU, writes 4 px/oc.
//
// Accumulators: 16 transform-domain vals per oc = 4 vec4 per oc -> OCB*4 vec4/thread.
// This is the Winograd register-pressure problem: matching wtile's OCB=32 would need
// 128 vec4/thread (vs wtile's 8). So OCB must stay small, which forces gz reloads.
// ============================================================================
const E = (k: string, d: number) => { try { const v = (globalThis as any).Deno?.env?.get(k); return v ? parseInt(v) : d; } catch { return d; } };
// Best config found by sweeping in the harness: 16x16 tiles (256 threads, large
// halo => max spatial reuse of each loaded input pixel) + OCB=4 (only 4 oc/thread
// => 16 vec4 accumulators, low register pressure). Higher OCB spills the 16-per-oc
// transform accumulators and tanks occupancy (OCB=8@256 -> 1007ms; OCB=16 -> worse).
const TBW = E("WG_TBW", 16), TBH = E("WG_TBH", 16);  // output tiles per workgroup (x,y) -> TBW*TBH threads
const OCB = E("WG_OCB", 4);    // output channels per workgroup; gz = ceil(out_c/OCB)
const NT = TBW * TBH;          // threads per workgroup
const HW = 2 * TBW + 2, HH = 2 * TBH + 2, HSZ = HW * HH;  // input halo (shared)
const OW = 2 * TBW, OH = 2 * TBH;                          // output px region

// per-oc accumulators: 4 vec4 (rows of the 4x4 transform-domain matrix M)
const accDecl = Array.from({ length: OCB }, (_, o) =>
  `  var mA${o}=vec4<f32>(0.0); var mB${o}=vec4<f32>(0.0); var mC${o}=vec4<f32>(0.0); var mD${o}=vec4<f32>(0.0);`
).join("\n");

// channel-sum: M[oc] += U[oc] (.) V   (16 element products as 4 vec4 FMAs)
const accBody = Array.from({ length: OCB }, (_, o) =>
  `      mA${o} += sU[${o}u*4u+0u]*vr0; mB${o} += sU[${o}u*4u+1u]*vr1; mC${o} += sU[${o}u*4u+2u]*vr2; mD${o} += sU[${o}u*4u+3u]*vr3;`
).join("\n");

// inverse transform + write, per oc
const writeBody = Array.from({ length: OCB }, (_, o) => `
    { let oc = ocbase + ${o}u;
      if(oc < u.out_c){
        let t0 = mA${o} + mB${o} + mC${o};
        let t1 = mB${o} - mC${o} - mD${o};
        let b = Wt[u.b_off+oc];
        var o00 = t0.x+t0.y+t0.z + b;
        var o01 = t0.y-t0.z-t0.w + b;
        var o10 = t1.x+t1.y+t1.z + b;
        var o11 = t1.y-t1.z-t1.w + b;
        if(u.has_prelu==1u){ let s = Wt[u.prelu_off+oc];
          if(o00<0.0){o00*=s;} if(o01<0.0){o01*=s;} if(o10<0.0){o10*=s;} if(o11<0.0){o11*=s;} }
        let base = oc*plane;
        if(ox<u.W   && oy<u.H)  { fout[base + oy*u.W + ox] = o00; }
        if(ox+1u<u.W&& oy<u.H)  { fout[base + oy*u.W + ox+1u] = o01; }
        if(ox<u.W   && oy+1u<u.H){ fout[base + (oy+1u)*u.W + ox] = o10; }
        if(ox+1u<u.W&& oy+1u<u.H){ fout[base + (oy+1u)*u.W + ox+1u] = o11; }
      } }`).join("\n");

const code = `
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f32>;
@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
@group(0) @binding(2) var<storage,read> Wt:array<f32>;
@group(0) @binding(3) var<uniform> u:P;

var<workgroup> sIn:array<f32,${HSZ}>;          // input halo for current ic
var<workgroup> sU:array<vec4<f32>,${OCB * 4}>; // transformed weights (4 vec4 rows / oc)

@compute @workgroup_size(${NT},1,1)
fn main(@builtin(workgroup_id) wid:vec3u, @builtin(local_invocation_index) lidx:u32){
  let tx = lidx % ${TBW}u;
  let ty = lidx / ${TBW}u;
  let gx0 = wid.x * ${OW}u;        // output px origin
  let gy0 = wid.y * ${OH}u;
  let ox = gx0 + 2u*tx;
  let oy = gy0 + 2u*ty;
  let ocbase = wid.z * ${OCB}u;
  let plane = u.H * u.W;
  let inc9 = u.in_c*9u;

${accDecl}

  for(var ic=0u; ic<u.in_c; ic++){
    // --- load input halo (top-left global = gx0-1, gy0-1), zero-pad ---
    for(var t=lidx; t<${HSZ}u; t+=${NT}u){
      let hx = t % ${HW}u;
      let hy = t / ${HW}u;
      let xx = i32(gx0)+i32(hx)-1;
      let yy = i32(gy0)+i32(hy)-1;
      var v = 0.0;
      if(xx>=0 && xx<i32(u.W) && yy>=0 && yy<i32(u.H)){ v = fin[ic*plane + u32(yy)*u.W + u32(xx)]; }
      sIn[t] = v;
    }
    // --- weight transform U = G g G^T for OCB channels (amortized) ---
    for(var t=lidx; t<${OCB}u; t+=${NT}u){
      let oc = ocbase + t;
      var g0=0.0; var g1=0.0; var g2=0.0; var g3=0.0; var g4=0.0; var g5=0.0; var g6=0.0; var g7=0.0; var g8=0.0;
      if(oc < u.out_c){
        let wb = u.w_off + oc*inc9 + ic*9u;
        g0=Wt[wb+0u]; g1=Wt[wb+1u]; g2=Wt[wb+2u];
        g3=Wt[wb+3u]; g4=Wt[wb+4u]; g5=Wt[wb+5u];
        g6=Wt[wb+6u]; g7=Wt[wb+7u]; g8=Wt[wb+8u];
      }
      // U_temp = G g  (4x3): rows over output, cols = kernel col j (g col0=g0,g3,g6 etc)
      // col j of g = (g_j, g_{3+j}, g_{6+j})
      let ut0_0=g0; let ut0_1=g1; let ut0_2=g2;                       // row0 = g row0
      let ut1_0=0.5*(g0+g3+g6); let ut1_1=0.5*(g1+g4+g7); let ut1_2=0.5*(g2+g5+g8);
      let ut2_0=0.5*(g0-g3+g6); let ut2_1=0.5*(g1-g4+g7); let ut2_2=0.5*(g2-g5+g8);
      let ut3_0=g6; let ut3_1=g7; let ut3_2=g8;                       // row3 = g row2
      // U = U_temp G^T  (4x4): per row, cols j'=0..3
      sU[t*4u+0u] = vec4<f32>( ut0_0, 0.5*(ut0_0+ut0_1+ut0_2), 0.5*(ut0_0-ut0_1+ut0_2), ut0_2 );
      sU[t*4u+1u] = vec4<f32>( ut1_0, 0.5*(ut1_0+ut1_1+ut1_2), 0.5*(ut1_0-ut1_1+ut1_2), ut1_2 );
      sU[t*4u+2u] = vec4<f32>( ut2_0, 0.5*(ut2_0+ut2_1+ut2_2), 0.5*(ut2_0-ut2_1+ut2_2), ut2_2 );
      sU[t*4u+3u] = vec4<f32>( ut3_0, 0.5*(ut3_0+ut3_1+ut3_2), 0.5*(ut3_0-ut3_1+ut3_2), ut3_2 );
    }
    workgroupBarrier();

    // --- per-thread input transform V = B^T d B (adds only) ---
    let hx0 = 2u*tx; let hy0 = 2u*ty;            // tile top-left in halo coords
    // load 4x4 d
    let d00=sIn[(hy0+0u)*${HW}u + hx0+0u]; let d01=sIn[(hy0+0u)*${HW}u + hx0+1u]; let d02=sIn[(hy0+0u)*${HW}u + hx0+2u]; let d03=sIn[(hy0+0u)*${HW}u + hx0+3u];
    let d10=sIn[(hy0+1u)*${HW}u + hx0+0u]; let d11=sIn[(hy0+1u)*${HW}u + hx0+1u]; let d12=sIn[(hy0+1u)*${HW}u + hx0+2u]; let d13=sIn[(hy0+1u)*${HW}u + hx0+3u];
    let d20=sIn[(hy0+2u)*${HW}u + hx0+0u]; let d21=sIn[(hy0+2u)*${HW}u + hx0+1u]; let d22=sIn[(hy0+2u)*${HW}u + hx0+2u]; let d23=sIn[(hy0+2u)*${HW}u + hx0+3u];
    let d30=sIn[(hy0+3u)*${HW}u + hx0+0u]; let d31=sIn[(hy0+3u)*${HW}u + hx0+1u]; let d32=sIn[(hy0+3u)*${HW}u + hx0+2u]; let d33=sIn[(hy0+3u)*${HW}u + hx0+3u];
    // V_temp = B^T d  (rows)
    let vt0=vec4<f32>(d00-d20, d01-d21, d02-d22, d03-d23);
    let vt1=vec4<f32>(d10+d20, d11+d21, d12+d22, d13+d23);
    let vt2=vec4<f32>(d20-d10, d21-d11, d22-d12, d23-d13);
    let vt3=vec4<f32>(d10-d30, d11-d31, d12-d32, d13-d33);
    // V = V_temp B  (cols): per row r, cols 0..3
    let vr0=vec4<f32>(vt0.x-vt0.z, vt0.y+vt0.z, vt0.z-vt0.y, vt0.y-vt0.w);
    let vr1=vec4<f32>(vt1.x-vt1.z, vt1.y+vt1.z, vt1.z-vt1.y, vt1.y-vt1.w);
    let vr2=vec4<f32>(vt2.x-vt2.z, vt2.y+vt2.z, vt2.z-vt2.y, vt2.y-vt2.w);
    let vr3=vec4<f32>(vt3.x-vt3.z, vt3.y+vt3.z, vt3.z-vt3.y, vt3.y-vt3.w);

${accBody}
    workgroupBarrier();
  }

${writeBody}
}`;

export default {
  code,
  dispatch: (ly: any, _H: number, _W: number) => [
    Math.ceil(_W / OW), Math.ceil(_H / OH), Math.ceil(ly.out_c / OCB),
  ],
};
