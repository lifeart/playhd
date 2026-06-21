// double-buffered tiled conv: prefetch next ic's tile+weights into the alternate shared buffer while
// computing the current ic -> 1 barrier/ic instead of 2. OC-blocked. usage:
//   deno run --allow-write gen3.ts <OCB> <TILE> <mode> <out.ts>   mode: f16d | f16a | f32
const OCB = parseInt(Deno.args[0] || "64");
const TILE = parseInt(Deno.args[1] || "16");
const mode = Deno.args[2] || "f16d";
const out = Deno.args[3] || "cand_var.ts";
const HW = TILE + 2;
const tileN = HW * HW;
const wN = OCB * 9;
const WG = TILE * TILE;
const acc = mode === "f16a" ? "f16" : "f32";
const C = (s: string) => mode === "f16a" ? s : `f32(${s})`;
const zero = mode === "f16a" ? "f16(0.0)" : "0.0";

let mac = "";
if (mode === "f32") {
  mac = `
      let i00=f32(tileS[buf][r0]); let i01=f32(tileS[buf][r0+1u]); let i02=f32(tileS[buf][r0+2u]);
      let i10=f32(tileS[buf][r1]); let i11=f32(tileS[buf][r1+1u]); let i12=f32(tileS[buf][r1+2u]);
      let i20=f32(tileS[buf][r2]); let i21=f32(tileS[buf][r2+1u]); let i22=f32(tileS[buf][r2+2u]);
      for(var j=0u;j<OCB;j++){ let b=j*9u;
        acc[j]+= f32(wS[buf][b])*i00+f32(wS[buf][b+1u])*i01+f32(wS[buf][b+2u])*i02
               + f32(wS[buf][b+3u])*i10+f32(wS[buf][b+4u])*i11+f32(wS[buf][b+5u])*i12
               + f32(wS[buf][b+6u])*i20+f32(wS[buf][b+7u])*i21+f32(wS[buf][b+8u])*i22; }`;
} else {
  mac = `
      let a0=vec3<f16>(tileS[buf][r0],tileS[buf][r0+1u],tileS[buf][r0+2u]);
      let a1=vec3<f16>(tileS[buf][r1],tileS[buf][r1+1u],tileS[buf][r1+2u]);
      let a2=vec3<f16>(tileS[buf][r2],tileS[buf][r2+1u],tileS[buf][r2+2u]);
      for(var j=0u;j<OCB;j++){ let b=j*9u;
        acc[j]+= ${C(`dot(vec3<f16>(wS[buf][b],wS[buf][b+1u],wS[buf][b+2u]),a0)
               + dot(vec3<f16>(wS[buf][b+3u],wS[buf][b+4u],wS[buf][b+5u]),a1)
               + dot(vec3<f16>(wS[buf][b+6u],wS[buf][b+7u],wS[buf][b+8u]),a2)`)}; }`;
}

// loader snippet: fills buffer `dst` for input channel `cc`
const loader = (dst: string, cc: string) => `{
    let plB=${cc}*H*W;
    for(var idx=tid; idx<${tileN}u; idx+=${WG}u){
      let tlx=idx%HW; let tly=idx/HW;
      let gx=i32(ox)+i32(tlx)-1; let gy=i32(oy)+i32(tly)-1;
      var v:f16=f16(0.0);
      if(gx>=0 && gx<i32(W) && gy>=0 && gy<i32(H)){ v=fin[plB+u32(gy)*W+u32(gx)]; }
      tileS[${dst}][idx]=v;
    }
    let wbo=u.w_off+${cc}*9u;
    for(var idx=tid; idx<${wN}u; idx+=${WG}u){
      let j=idx/9u; let t=idx%9u;
      wS[${dst}][idx]=Wt[wbo+(ocBase+j)*in_c*9u+t];
    }
  }`;

const code = `enable f16;
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f16>;
@group(0) @binding(1) var<storage,read_write> fout:array<f16>;
@group(0) @binding(2) var<storage,read> Wt:array<f16>;
@group(0) @binding(3) var<uniform> u:P;
const TILE=${TILE}u; const HW=${HW}u; const OCB=${OCB}u;
var<workgroup> tileS:array<array<f16,${tileN}>,2>;
var<workgroup> wS:array<array<f16,${wN}>,2>;
@compute @workgroup_size(${TILE},${TILE},1)
fn main(@builtin(workgroup_id) wg:vec3u,@builtin(local_invocation_id) lid:vec3u){
  let H=u.H; let W=u.W; let in_c=u.in_c;
  let ox=wg.x*TILE; let oy=wg.y*TILE;
  let lx=lid.x; let ly=lid.y;
  let px=ox+lx; let py=oy+ly;
  let ocBase=wg.z*OCB;
  let tid=ly*TILE+lx;
  let r0=ly*HW+lx; let r1=(ly+1u)*HW+lx; let r2=(ly+2u)*HW+lx;
  var acc:array<${acc},${OCB}>;
  for(var j=0u;j<OCB;j++){ acc[j]=${C("Wt[u.b_off+ocBase+j]")}; }
  ${loader("0", "0u")}
  workgroupBarrier();
  for(var ic=0u; ic<in_c; ic++){
    let buf=ic&1u;
    let nxt=(ic+1u)&1u;
    if(ic+1u<in_c){ let cc=ic+1u; ${loader("nxt", "cc")} }
    if(px<W && py<H){${mac}
    }
    workgroupBarrier();
  }
  if(px<W && py<H){
    let opl=H*W;
    for(var j=0u;j<OCB;j++){
      let oc=ocBase+j;
      if(oc<u.out_c){
        var v=acc[j];
        if(u.has_prelu==1u){ let s=${C("Wt[u.prelu_off+oc]")}; if(v<${zero}){ v=v*s; } }
        fout[oc*opl+py*W+px]=f16(v);
      }
    }
  }
}`;
const ts = `export default {\n  f16: true,\n  code: ${JSON.stringify(code)},\n  dispatch: (ly:any,H:number,W:number)=>[Math.ceil(W/${TILE}),Math.ceil(H/${TILE}),Math.ceil(ly.out_c/${OCB})],\n};\n`;
await Deno.writeTextFile(out, ts);
console.log("wrote", out, "OCB", OCB, "TILE", TILE, "mode", mode, "(double-buffered)");
