// horizontal-pixel-blocked tiled conv. each thread computes PX adjacent x-pixels for OCB channels.
// usage: deno run --allow-write gen2.ts <OCB> <TILE> <PX> <mode> <out.ts>
// mode: f32 | f16d (f16 dot,f32 acc) | f16a (f16 acc)
const OCB = parseInt(Deno.args[0] || "32");
const TILE = parseInt(Deno.args[1] || "16");   // output tile side
const PX = parseInt(Deno.args[2] || "2");      // x-pixels per thread
const mode = Deno.args[3] || "f32";
const out = Deno.args[4] || "cand_var.ts";
const WGX = TILE / PX, WGY = TILE;
const HW = TILE + 2;
const tileN = HW * HW;
const wN = OCB * 9;
const WG = WGX * WGY;
const acc = mode === "f16a" ? "f16" : "f32";
const cv = mode === "f16a" ? "" : "f32";   // convert wrapper
const C = (s: string) => mode === "f16a" ? s : `f32(${s})`;
const zero = mode === "f16a" ? "f16(0.0)" : "0.0";

// build inner MAC: for p in 0..PX, for j in 0..OCB
let inner = "";
if (mode === "f32") {
  // load window values per p reused; compute scalar f32
  inner = `
      for(var j=0u;j<OCB;j++){ let b=j*9u;
        let w0=f32(wS[b]); let w1=f32(wS[b+1u]); let w2=f32(wS[b+2u]);
        let w3=f32(wS[b+3u]); let w4=f32(wS[b+4u]); let w5=f32(wS[b+5u]);
        let w6=f32(wS[b+6u]); let w7=f32(wS[b+7u]); let w8=f32(wS[b+8u]);
        for(var p=0u;p<PX;p++){ let c=base+p;
          acc[j*PX+p]+= w0*f32(tileS[r0+c])+w1*f32(tileS[r0+c+1u])+w2*f32(tileS[r0+c+2u])
                      + w3*f32(tileS[r1+c])+w4*f32(tileS[r1+c+1u])+w5*f32(tileS[r1+c+2u])
                      + w6*f32(tileS[r2+c])+w7*f32(tileS[r2+c+1u])+w8*f32(tileS[r2+c+2u]); } }`;
} else {
  inner = `
      for(var j=0u;j<OCB;j++){ let b=j*9u;
        let wa=vec3<f16>(wS[b],wS[b+1u],wS[b+2u]);
        let wb2=vec3<f16>(wS[b+3u],wS[b+4u],wS[b+5u]);
        let wc=vec3<f16>(wS[b+6u],wS[b+7u],wS[b+8u]);
        for(var p=0u;p<PX;p++){ let c=base+p;
          let s=dot(wa,vec3<f16>(tileS[r0+c],tileS[r0+c+1u],tileS[r0+c+2u]))
               +dot(wb2,vec3<f16>(tileS[r1+c],tileS[r1+c+1u],tileS[r1+c+2u]))
               +dot(wc,vec3<f16>(tileS[r2+c],tileS[r2+c+1u],tileS[r2+c+2u]));
          acc[j*PX+p]+= ${C("s")}; } }`;
}

const code = `enable f16;
struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<f16>;
@group(0) @binding(1) var<storage,read_write> fout:array<f16>;
@group(0) @binding(2) var<storage,read> Wt:array<f16>;
@group(0) @binding(3) var<uniform> u:P;
const TILE=${TILE}u; const HW=${HW}u; const OCB=${OCB}u; const PX=${PX}u;
var<workgroup> tileS:array<f16,${tileN}>;
var<workgroup> wS:array<f16,${wN}>;
@compute @workgroup_size(${WGX},${WGY},1)
fn main(@builtin(workgroup_id) wg:vec3u,@builtin(local_invocation_id) lid:vec3u){
  let H=u.H; let W=u.W; let in_c=u.in_c;
  let ox=wg.x*TILE; let oy=wg.y*TILE;
  let lx=lid.x; let ly=lid.y;
  let px0=ox+lx*PX; let py=oy+ly;
  let ocBase=wg.z*OCB;
  let tid=ly*${WGX}u+lx;
  var acc:array<${acc},${OCB * PX}>;
  for(var j=0u;j<OCB*PX;j++){ acc[j]=${C("Wt[u.b_off+ocBase+j/PX]")}; }
  for(var ic=0u; ic<in_c; ic++){
    let plBase=ic*H*W;
    for(var idx=tid; idx<${tileN}u; idx+=${WG}u){
      let tlx=idx%HW; let tly=idx/HW;
      let gx=i32(ox)+i32(tlx)-1; let gy=i32(oy)+i32(tly)-1;
      var v:f16=f16(0.0);
      if(gx>=0 && gx<i32(W) && gy>=0 && gy<i32(H)){ v=fin[plBase+u32(gy)*W+u32(gx)]; }
      tileS[idx]=v;
    }
    let wbo=u.w_off+ic*9u;
    for(var idx=tid; idx<${wN}u; idx+=${WG}u){
      let j=idx/9u; let t=idx%9u;
      wS[idx]=Wt[wbo+(ocBase+j)*in_c*9u+t];
    }
    workgroupBarrier();
    let base=lx*PX;
    let r0=ly*HW; let r1=(ly+1u)*HW; let r2=(ly+2u)*HW;${inner}
    workgroupBarrier();
  }
  let opl=H*W;
  for(var j=0u;j<OCB;j++){
    let oc=ocBase+j;
    if(oc<u.out_c){
      let s=${C("Wt[u.prelu_off+oc]")};
      for(var p=0u;p<PX;p++){ let xx=px0+p;
        if(xx<W && py<H){
          var v=acc[j*PX+p];
          if(u.has_prelu==1u){ if(v<${zero}){ v=v*s; } }
          fout[oc*opl+py*W+xx]=f16(v);
        } } }
  }
}`;
const ts = `export default {\n  f16: true,\n  code: ${JSON.stringify(code)},\n  dispatch: (ly:any,H:number,W:number)=>[Math.ceil(W/${TILE}),Math.ceil(H/${TILE}),Math.ceil(ly.out_c/${OCB})],\n};\n`;
await Deno.writeTextFile(out, ts);
console.log("wrote", out, "OCB", OCB, "TILE", TILE, "PX", PX, "mode", mode);
