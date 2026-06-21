// Generator for 2D register-tiled interleaved+vec4 conv (my strategy).
// Each thread computes PX horizontal pixels x (OCG*4) output channels.
// Interleaved [y][x][c] layout; vec4 over input channels (dot). Input vec4 reused across OCG
// groups, weight vec4 reused across PX pixels (register-blocked GEMM-on-the-fly). f32, parity-exact.
export function gen(PX: number, OCG: number, WGX = 8, WGY = 8) {
  const px = [...Array(PX).keys()];
  const og = [...Array(OCG).keys()];
  const biasDecl = og.map(t =>
    `let bias${t}=vec4<f32>(Wt[u.b_off+oc0+${t * 4}u],Wt[u.b_off+oc0+${t * 4 + 1}u],Wt[u.b_off+oc0+${t * 4 + 2}u],Wt[u.b_off+oc0+${t * 4 + 3}u]);`
  ).join("\n  ");
  const wbDecl = og.map(t => [0, 1, 2, 3].map(j =>
    `let wb${t}_${j}=u.w_off+(oc0+${t * 4 + j}u)*s9;`).join("")).join("\n  ");
  const accDecl = px.map(p => og.map(t => `var a${p}_${t}=bias${t};`).join("")).join("\n  ");

  // interleaved inner
  const inLoads = px.map(p => `let i${p}=fin[b${p}+gi];`).join(" ");
  const ocgBlock = og.map(t => {
    const w = [0, 1, 2, 3].map(j =>
      `let w${t}_${j}=vec4<f32>(Wt[wb${t}_${j}+o],Wt[wb${t}_${j}+o+9u],Wt[wb${t}_${j}+o+18u],Wt[wb${t}_${j}+o+27u]);`).join("");
    const acc = px.map(p =>
      `if(ok${p}){a${p}_${t}.x+=dot(w${t}_0,i${p});a${p}_${t}.y+=dot(w${t}_1,i${p});a${p}_${t}.z+=dot(w${t}_2,i${p});a${p}_${t}.w+=dot(w${t}_3,i${p});}`).join("");
    return w + acc;
  }).join("\n          ");
  const okDecl = px.map(p => `let xx${p}=x0+kx-1+${p}; let ok${p}=xx${p}>=0&&xx${p}<Wi; let b${p}=select(0u,(rb+u32(xx${p}))*16u,ok${p});`).join("\n        ");
  const interBranch = `
    for(var ky=0;ky<3;ky++){ let yy=y+ky-1; if(yy<0||yy>=Hi){continue;} let rb=u32(yy)*W;
      for(var kx=0;kx<3;kx++){ let tap=u32(ky*3+kx);
        ${okDecl}
        for(var gi=0u;gi<16u;gi++){ let o=gi*36u+tap;
          ${inLoads}
          ${ocgBlock}
        }
      }
    }`;

  // planar inner (in_c==3)
  const okDeclP = px.map(p => `let xx${p}=x0+kx-1+${p}; let ok${p}=xx${p}>=0&&xx${p}<Wi; let pp${p}=rb+u32(xx${p});`).join("\n        ");
  const ocgBlockP = og.map(t => {
    const wv = `let w${t}=vec4<f32>(Wt[wb${t}_0+o],Wt[wb${t}_1+o],Wt[wb${t}_2+o],Wt[wb${t}_3+o]);`;
    const acc = px.map(p => `if(ok${p}){a${p}_${t}+=rdP(ic,pp${p})*w${t};}`).join("");
    return wv + acc;
  }).join("\n          ");
  const planarBranch = `
    for(var ky=0;ky<3;ky++){ let yy=y+ky-1; if(yy<0||yy>=Hi){continue;} let rb=u32(yy)*W;
      for(var kx=0;kx<3;kx++){ let tap=u32(ky*3+kx);
        ${okDeclP}
        for(var ic=0u;ic<3u;ic++){ let o=ic*9u+tap;
          ${ocgBlockP}
        }
      }
    }`;

  // prelu
  const spDecl = og.map(t =>
    `let sp${t}=vec4<f32>(Wt[u.prelu_off+oc0+${t * 4}u],Wt[u.prelu_off+oc0+${t * 4 + 1}u],Wt[u.prelu_off+oc0+${t * 4 + 2}u],Wt[u.prelu_off+oc0+${t * 4 + 3}u]);`).join("\n    ");
  const prelu = px.map(p => og.map(t =>
    `a${p}_${t}=select(a${p}_${t},a${p}_${t}*sp${t},a${p}_${t}<z);`).join("")).join("\n    ");

  // write
  const xoDecl = px.map(p => `let xo${p}=x0+${p};`).join(" ");
  const writePlanar = px.map(p => og.map(t => {
    const oc = `(oc0+${t * 4}u)`;
    return `if(xo${p}<Wi){let pp=rowo+u32(xo${p}); fout[${oc}*pl+pp]=a${p}_${t}.x; fout[(${oc}+1u)*pl+pp]=a${p}_${t}.y; fout[(${oc}+2u)*pl+pp]=a${p}_${t}.z; fout[(${oc}+3u)*pl+pp]=a${p}_${t}.w;}`;
  }).join("")).join("\n    ");
  const writeInter = px.map(p => og.map(t =>
    `if(xo${p}<Wi){let b=(rowo+u32(xo${p}))*64u+oc0+${t * 4}u; fout[b]=a${p}_${t}.x; fout[b+1u]=a${p}_${t}.y; fout[b+2u]=a${p}_${t}.z; fout[b+3u]=a${p}_${t}.w;}`).join("")).join("\n    ");

  const code = `struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
@group(0) @binding(2) var<storage,read> Wt:array<f32>;
@group(0) @binding(3) var<uniform> u:P;
fn rdP(c:u32,p:u32)->f32{ let k=c*u.H*u.W+p; return fin[k>>2u][k&3u]; }
@compute @workgroup_size(${WGX},${WGY},1) fn main(@builtin(global_invocation_id) g:vec3u){
  let ocg=g.z*${OCG}u; let oc0=ocg*4u;
  let Wi=i32(u.W); let Hi=i32(u.H); let W=u.W; let H=u.H;
  let x0=i32(g.x)*${PX}; let y=i32(g.y);
  if(x0>=Wi||y>=Hi||oc0>=u.out_c){ return; }
  let s9=u.in_c*9u;
  ${biasDecl}
  ${wbDecl}
  ${accDecl}
  if(u.in_c==3u){${planarBranch}
  } else {${interBranch}
  }
  if(u.has_prelu==1u){
    let z=vec4<f32>(0.0);
    ${spDecl}
    ${prelu}
  }
  let pl=H*W; let rowo=u32(y)*W; ${xoDecl}
  if(u.out_c==48u){
    ${writePlanar}
  } else {
    ${writeInter}
  }
}`;
  return {
    code,
    dispatch: (ly: any, H: number, W: number): [number, number, number] =>
      [Math.ceil(W / (WGX * PX)), Math.ceil(H / WGY), Math.ceil(ly.out_c / (4 * OCG))],
  };
}

// CLI: write a candidate file. usage: deno run --allow-write genvec.ts <PX> <OCG> <WGX> <WGY> <out.ts>
if (import.meta.main) {
  const PX = parseInt(Deno.args[0] || "4");
  const OCG = parseInt(Deno.args[1] || "2");
  const WGX = parseInt(Deno.args[2] || "8");
  const WGY = parseInt(Deno.args[3] || "8");
  const out = Deno.args[4] || "cand_var.ts";
  const g = gen(PX, OCG, WGX, WGY);
  const ts = `export default {\n  code: ${JSON.stringify(g.code)},\n  dispatch: (ly,H,W)=>[${"Math.ceil(W/" + (WGX * PX) + ")"},${"Math.ceil(H/" + WGY + ")"},${"Math.ceil(ly.out_c/" + (4 * OCG) + ")"}],\n};\n`;
  await Deno.writeTextFile(out, ts);
  console.log("wrote", out, "PX", PX, "OCG", OCG, "WG", WGX, WGY);
}
