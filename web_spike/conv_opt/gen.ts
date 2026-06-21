// generate a candidate variant. usage: deno run --allow-write gen.ts <OCB> <TILE> <mode> <out.ts>
// mode: f32 (f32 mac), f16d (f16 vec3 dot, f32 accum), f16a (full f16 accum)
const OCB = parseInt(Deno.args[0] || "16");
const TILE = parseInt(Deno.args[1] || "16");
const mode = Deno.args[2] || "f32";
const out = Deno.args[3] || "cand_var.ts";
const HW = TILE + 2;
const tileN = HW * HW;
const wN = OCB * 9;
const WG = TILE * TILE;

let macBody = "";
if (mode === "f32") {
  macBody = `
      let i00=f32(tileS[r0]); let i01=f32(tileS[r0+1u]); let i02=f32(tileS[r0+2u]);
      let i10=f32(tileS[r1]); let i11=f32(tileS[r1+1u]); let i12=f32(tileS[r1+2u]);
      let i20=f32(tileS[r2]); let i21=f32(tileS[r2+1u]); let i22=f32(tileS[r2+2u]);
      for(var j=0u;j<OCB;j++){ let b=j*9u;
        acc[j]+= f32(wS[b])*i00+f32(wS[b+1u])*i01+f32(wS[b+2u])*i02
               + f32(wS[b+3u])*i10+f32(wS[b+4u])*i11+f32(wS[b+5u])*i12
               + f32(wS[b+6u])*i20+f32(wS[b+7u])*i21+f32(wS[b+8u])*i22; }`;
} else if (mode === "f16d") {
  macBody = `
      let a0=vec3<f16>(tileS[r0],tileS[r0+1u],tileS[r0+2u]);
      let a1=vec3<f16>(tileS[r1],tileS[r1+1u],tileS[r1+2u]);
      let a2=vec3<f16>(tileS[r2],tileS[r2+1u],tileS[r2+2u]);
      for(var j=0u;j<OCB;j++){ let b=j*9u;
        acc[j]+= f32(dot(vec3<f16>(wS[b],wS[b+1u],wS[b+2u]),a0)
                   + dot(vec3<f16>(wS[b+3u],wS[b+4u],wS[b+5u]),a1)
                   + dot(vec3<f16>(wS[b+6u],wS[b+7u],wS[b+8u]),a2)); }`;
} else { // f16a: accumulate in f16
  macBody = `
      let a0=vec3<f16>(tileS[r0],tileS[r0+1u],tileS[r0+2u]);
      let a1=vec3<f16>(tileS[r1],tileS[r1+1u],tileS[r1+2u]);
      let a2=vec3<f16>(tileS[r2],tileS[r2+1u],tileS[r2+2u]);
      for(var j=0u;j<OCB;j++){ let b=j*9u;
        acc[j]+= dot(vec3<f16>(wS[b],wS[b+1u],wS[b+2u]),a0)
               + dot(vec3<f16>(wS[b+3u],wS[b+4u],wS[b+5u]),a1)
               + dot(vec3<f16>(wS[b+6u],wS[b+7u],wS[b+8u]),a2); }`;
}
const accType = mode === "f16a" ? "f16" : "f32";
const accInit = mode === "f16a" ? "Wt[u.b_off+ocBase+j]" : "f32(Wt[u.b_off+ocBase+j])";
const finalV = mode === "f16a" ? "var v=acc[j];" : "var v=acc[j];";
const sExpr = mode === "f16a" ? "Wt[u.prelu_off+oc]" : "f32(Wt[u.prelu_off+oc])";
const zero = mode === "f16a" ? "f16(0.0)" : "0.0";

const code = `enable f16;
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f16>;
@group(0) @binding(1) var<storage,read_write> fout:array<f16>;
@group(0) @binding(2) var<storage,read> Wt:array<f16>;
@group(0) @binding(3) var<uniform> u:P;
const TILE=${TILE}u;
const HW=${HW}u;
const OCB=${OCB}u;
var<workgroup> tileS:array<f16,${tileN}>;
var<workgroup> wS:array<f16,${wN}>;
@compute @workgroup_size(${TILE},${TILE},1)
fn main(@builtin(workgroup_id) wg:vec3u,@builtin(local_invocation_id) lid:vec3u){
  let H=u.H; let W=u.W; let in_c=u.in_c;
  let ox=wg.x*TILE; let oy=wg.y*TILE;
  let lx=lid.x; let ly=lid.y;
  let px=ox+lx; let py=oy+ly;
  let ocBase=wg.z*OCB;
  let tid=ly*TILE+lx;
  var acc:array<${accType},${OCB}>;
  for(var j=0u;j<OCB;j++){ acc[j]=${accInit}; }
  for(var ic=0u; ic<in_c; ic++){
    let plBase=ic*H*W;
    for(var idx=tid; idx<${tileN}u; idx+=${WG}u){
      let tlx=idx%HW; let tly=idx/HW;
      let gx=i32(ox)+i32(tlx)-1; let gy=i32(oy)+i32(tly)-1;
      var v:f16=f16(0.0);
      if(gx>=0 && gx<i32(W) && gy>=0 && gy<i32(H)){ v=fin[plBase+u32(gy)*W+u32(gx)]; }
      tileS[idx]=v;
    }
    let wb=u.w_off+ic*9u;
    for(var idx=tid; idx<${wN}u; idx+=${WG}u){
      let j=idx/9u; let t=idx%9u;
      wS[idx]=Wt[wb+(ocBase+j)*in_c*9u+t];
    }
    workgroupBarrier();
    if(px<W && py<H){
      let r0=ly*HW+lx; let r1=(ly+1u)*HW+lx; let r2=(ly+2u)*HW+lx;${macBody}
    }
    workgroupBarrier();
  }
  if(px<W && py<H){
    let opl=H*W;
    for(var j=0u;j<OCB;j++){
      let oc=ocBase+j;
      if(oc<u.out_c){
        ${finalV}
        if(u.has_prelu==1u){ let s=${sExpr}; if(v<${zero}){ v=v*s; } }
        fout[oc*opl+py*W+px]=f16(v);
      }
    }
  }
}`;
const ts = `export default {\n  f16: true,\n  code: ${JSON.stringify(code)},\n  dispatch: (ly:any,H:number,W:number)=>[Math.ceil(W/${TILE}),Math.ceil(H/${TILE}),Math.ceil(ly.out_c/${OCB})],\n};\n`;
await Deno.writeTextFile(out, ts);
console.log("wrote", out, "OCB", OCB, "TILE", TILE, "mode", mode);
