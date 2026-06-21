// Stage 1: interleaved channel layout + vec4 loads, 4 output channels per thread.
// Layout: inter-layer buffers are INTERLEAVED [y][x][c] (stride 64 channels).
//   channel c of pixel p=(y*W+x): vec4 index p*16 + c/4, component c%4.
// First layer (in_c==3) reads the PLANAR seed; last layer (out_c==48) writes PLANAR.
export default {
  code: `struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
@group(0) @binding(2) var<storage,read> Wt:array<f32>;
@group(0) @binding(3) var<uniform> u:P;

fn rdP(c:u32, p:u32)->f32{ let k=c*u.H*u.W+p; return fin[k>>2u][k&3u]; }

@compute @workgroup_size(8,8,1) fn main(@builtin(global_invocation_id) g:vec3u){
  let x=i32(g.x); let y=i32(g.y); let ocg=g.z;
  if(x>=i32(u.W)||y>=i32(u.H)||ocg*4u>=u.out_c){ return; }
  let oc0=ocg*4u; let W=u.W; let H=u.H;
  var acc=vec4<f32>(Wt[u.b_off+oc0],Wt[u.b_off+oc0+1u],Wt[u.b_off+oc0+2u],Wt[u.b_off+oc0+3u]);
  let wb0=u.w_off+oc0*u.in_c*9u;
  let wb1=wb0+u.in_c*9u;
  let wb2=wb1+u.in_c*9u;
  let wb3=wb2+u.in_c*9u;

  if(u.in_c==3u){
    // planar 3-channel input
    for(var ky=0;ky<3;ky++){ let yy=y+ky-1; if(yy<0||yy>=i32(H)){continue;}
      for(var kx=0;kx<3;kx++){ let xx=x+kx-1; if(xx<0||xx>=i32(W)){continue;}
        let pp=u32(yy)*W+u32(xx); let tap=u32(ky*3+kx);
        for(var ic=0u;ic<3u;ic++){
          let v=rdP(ic,pp); let o=ic*9u+tap;
          acc+=v*vec4<f32>(Wt[wb0+o],Wt[wb1+o],Wt[wb2+o],Wt[wb3+o]);
        }
      }
    }
  } else {
    // interleaved 64-channel input, vec4 over input channels
    for(var ky=0;ky<3;ky++){ let yy=y+ky-1; if(yy<0||yy>=i32(H)){continue;}
      for(var kx=0;kx<3;kx++){ let xx=x+kx-1; if(xx<0||xx>=i32(W)){continue;}
        let base=(u32(yy)*W+u32(xx))*16u; let tap=u32(ky*3+kx);
        for(var gi=0u;gi<16u;gi++){
          let in4=fin[base+gi]; let o=gi*36u+tap; // (4*gi)*9 = gi*36
          acc.x+=dot(vec4<f32>(Wt[wb0+o],Wt[wb0+o+9u],Wt[wb0+o+18u],Wt[wb0+o+27u]),in4);
          acc.y+=dot(vec4<f32>(Wt[wb1+o],Wt[wb1+o+9u],Wt[wb1+o+18u],Wt[wb1+o+27u]),in4);
          acc.z+=dot(vec4<f32>(Wt[wb2+o],Wt[wb2+o+9u],Wt[wb2+o+18u],Wt[wb2+o+27u]),in4);
          acc.w+=dot(vec4<f32>(Wt[wb3+o],Wt[wb3+o+9u],Wt[wb3+o+18u],Wt[wb3+o+27u]),in4);
        }
      }
    }
  }

  if(u.has_prelu==1u){
    let s=vec4<f32>(Wt[u.prelu_off+oc0],Wt[u.prelu_off+oc0+1u],Wt[u.prelu_off+oc0+2u],Wt[u.prelu_off+oc0+3u]);
    acc=select(acc,acc*s,acc<vec4<f32>(0.0));
  }

  let pout=u32(y)*W+u32(x);
  if(u.out_c==48u){
    let pl=H*W;
    fout[oc0*pl+pout]=acc.x; fout[(oc0+1u)*pl+pout]=acc.y; fout[(oc0+2u)*pl+pout]=acc.z; fout[(oc0+3u)*pl+pout]=acc.w;
  } else {
    let b=pout*64u+oc0;
    fout[b]=acc.x; fout[b+1u]=acc.y; fout[b+2u]=acc.z; fout[b+3u]=acc.w;
  }
}`,
  dispatch: (ly: any, H: number, W: number) => [Math.ceil(W / 8), Math.ceil(H / 8), Math.ceil(ly.out_c / 4)],
};
