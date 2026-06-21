// Stage 3: interleaved+vec4 (Stage 1) + 4-wide horizontal PIXEL blocking.
// Each thread computes 4 output channels for 4 horizontally-adjacent pixels.
// A weight vec4 is loaded once per (ky,kx,ic-group) and reused across the 4 pixels.
export default {
  code: `struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
@group(0) @binding(0) var<storage,read> fin:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
@group(0) @binding(2) var<storage,read> Wt:array<f32>;
@group(0) @binding(3) var<uniform> u:P;

fn rdP(c:u32, p:u32)->f32{ let k=c*u.H*u.W+p; return fin[k>>2u][k&3u]; }

@compute @workgroup_size(8,8,1) fn main(@builtin(global_invocation_id) g:vec3u){
  let ocg=g.z; let oc0=ocg*4u;
  let Wi=i32(u.W); let Hi=i32(u.H); let W=u.W; let H=u.H;
  let x0=i32(g.x)*4; let y=i32(g.y);
  if(x0>=Wi||y>=Hi||oc0>=u.out_c){ return; }
  let wb0=u.w_off+oc0*u.in_c*9u; let s1=u.in_c*9u; let wb1=wb0+s1; let wb2=wb1+s1; let wb3=wb2+s1;
  let bias=vec4<f32>(Wt[u.b_off+oc0],Wt[u.b_off+oc0+1u],Wt[u.b_off+oc0+2u],Wt[u.b_off+oc0+3u]);
  var a0=bias; var a1=bias; var a2=bias; var a3=bias;

  if(u.in_c==3u){
    for(var ky=0;ky<3;ky++){ let yy=y+ky-1; if(yy<0||yy>=Hi){continue;} let rb=u32(yy)*W;
      for(var kx=0;kx<3;kx++){ let tap=u32(ky*3+kx);
        let x1=x0+kx-1;
        let v0=x1; let v1=x1+1; let v2=x1+2; let v3=x1+3;
        for(var ic=0u;ic<3u;ic++){ let o=ic*9u+tap;
          let w=vec4<f32>(Wt[wb0+o],Wt[wb1+o],Wt[wb2+o],Wt[wb3+o]);
          if(v0>=0&&v0<Wi){ a0+=rdP(ic,rb+u32(v0))*w; }
          if(v1>=0&&v1<Wi){ a1+=rdP(ic,rb+u32(v1))*w; }
          if(v2>=0&&v2<Wi){ a2+=rdP(ic,rb+u32(v2))*w; }
          if(v3>=0&&v3<Wi){ a3+=rdP(ic,rb+u32(v3))*w; }
        }
      }
    }
  } else {
    for(var ky=0;ky<3;ky++){ let yy=y+ky-1; if(yy<0||yy>=Hi){continue;} let rb=u32(yy)*W;
      for(var kx=0;kx<3;kx++){ let tap=u32(ky*3+kx);
        let x1=x0+kx-1;
        let v0=x1; let v1=x1+1; let v2=x1+2; let v3=x1+3;
        let ok0=v0>=0&&v0<Wi; let ok1=v1>=0&&v1<Wi; let ok2=v2>=0&&v2<Wi; let ok3=v3>=0&&v3<Wi;
        let b0=select(0u,(rb+u32(v0))*16u,ok0);
        let b1=select(0u,(rb+u32(v1))*16u,ok1);
        let b2=select(0u,(rb+u32(v2))*16u,ok2);
        let b3=select(0u,(rb+u32(v3))*16u,ok3);
        for(var gi=0u;gi<16u;gi++){ let o=gi*36u+tap;
          let w0=vec4<f32>(Wt[wb0+o],Wt[wb0+o+9u],Wt[wb0+o+18u],Wt[wb0+o+27u]);
          let w1=vec4<f32>(Wt[wb1+o],Wt[wb1+o+9u],Wt[wb1+o+18u],Wt[wb1+o+27u]);
          let w2=vec4<f32>(Wt[wb2+o],Wt[wb2+o+9u],Wt[wb2+o+18u],Wt[wb2+o+27u]);
          let w3=vec4<f32>(Wt[wb3+o],Wt[wb3+o+9u],Wt[wb3+o+18u],Wt[wb3+o+27u]);
          if(ok0){ let i4=fin[b0+gi]; a0.x+=dot(w0,i4); a0.y+=dot(w1,i4); a0.z+=dot(w2,i4); a0.w+=dot(w3,i4); }
          if(ok1){ let i4=fin[b1+gi]; a1.x+=dot(w0,i4); a1.y+=dot(w1,i4); a1.z+=dot(w2,i4); a1.w+=dot(w3,i4); }
          if(ok2){ let i4=fin[b2+gi]; a2.x+=dot(w0,i4); a2.y+=dot(w1,i4); a2.z+=dot(w2,i4); a2.w+=dot(w3,i4); }
          if(ok3){ let i4=fin[b3+gi]; a3.x+=dot(w0,i4); a3.y+=dot(w1,i4); a3.z+=dot(w2,i4); a3.w+=dot(w3,i4); }
        }
      }
    }
  }

  if(u.has_prelu==1u){
    let sp=vec4<f32>(Wt[u.prelu_off+oc0],Wt[u.prelu_off+oc0+1u],Wt[u.prelu_off+oc0+2u],Wt[u.prelu_off+oc0+3u]);
    let z=vec4<f32>(0.0);
    a0=select(a0,a0*sp,a0<z); a1=select(a1,a1*sp,a1<z); a2=select(a2,a2*sp,a2<z); a3=select(a3,a3*sp,a3<z);
  }

  let pl=H*W; let rowo=u32(y)*W;
  if(u.out_c==48u){
    if(x0<Wi){ let p=rowo+u32(x0); fout[oc0*pl+p]=a0.x; fout[(oc0+1u)*pl+p]=a0.y; fout[(oc0+2u)*pl+p]=a0.z; fout[(oc0+3u)*pl+p]=a0.w; }
    if(x0+1<Wi){ let p=rowo+u32(x0+1); fout[oc0*pl+p]=a1.x; fout[(oc0+1u)*pl+p]=a1.y; fout[(oc0+2u)*pl+p]=a1.z; fout[(oc0+3u)*pl+p]=a1.w; }
    if(x0+2<Wi){ let p=rowo+u32(x0+2); fout[oc0*pl+p]=a2.x; fout[(oc0+1u)*pl+p]=a2.y; fout[(oc0+2u)*pl+p]=a2.z; fout[(oc0+3u)*pl+p]=a2.w; }
    if(x0+3<Wi){ let p=rowo+u32(x0+3); fout[oc0*pl+p]=a3.x; fout[(oc0+1u)*pl+p]=a3.y; fout[(oc0+2u)*pl+p]=a3.z; fout[(oc0+3u)*pl+p]=a3.w; }
  } else {
    if(x0<Wi){ let b=(rowo+u32(x0))*64u+oc0; fout[b]=a0.x; fout[b+1u]=a0.y; fout[b+2u]=a0.z; fout[b+3u]=a0.w; }
    if(x0+1<Wi){ let b=(rowo+u32(x0+1))*64u+oc0; fout[b]=a1.x; fout[b+1u]=a1.y; fout[b+2u]=a1.z; fout[b+3u]=a1.w; }
    if(x0+2<Wi){ let b=(rowo+u32(x0+2))*64u+oc0; fout[b]=a2.x; fout[b+1u]=a2.y; fout[b+2u]=a2.z; fout[b+3u]=a2.w; }
    if(x0+3<Wi){ let b=(rowo+u32(x0+3))*64u+oc0; fout[b]=a3.x; fout[b+1u]=a3.y; fout[b+2u]=a3.z; fout[b+3u]=a3.w; }
  }
}`,
  dispatch: (ly: any, H: number, W: number) => [Math.ceil(W / 32), Math.ceil(H / 8), Math.ceil(ly.out_c / 4)],
};
