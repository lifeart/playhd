// trivial candidate == naive, to validate the harness candidate path (expect ~1x, parity 0)
export default {
  code: `struct P{H:u32,W:u32,in_c:u32,out_c:u32,w_off:u32,b_off:u32,prelu_off:u32,has_prelu:u32};
    @group(0) @binding(0) var<storage,read> fin:array<f32>;@group(0) @binding(1) var<storage,read_write> fout:array<f32>;
    @group(0) @binding(2) var<storage,read> Wt:array<f32>;@group(0) @binding(3) var<uniform> u:P;
    @compute @workgroup_size(8,8,1) fn main(@builtin(global_invocation_id) g:vec3u){
      let x=i32(g.x);let y=i32(g.y);let oc=i32(g.z);if(x>=i32(u.W)||y>=i32(u.H)||oc>=i32(u.out_c)){return;}
      var acc=Wt[u.b_off+u32(oc)];let bw=u.w_off+u32(oc)*u.in_c*9u;
      for(var ic=0u;ic<u.in_c;ic=ic+1u){let pl=ic*u.H*u.W;let wic=bw+ic*9u;
        for(var ky=0;ky<3;ky=ky+1){let yy=y+ky-1;if(yy<0||yy>=i32(u.H)){continue;}
          for(var kx=0;kx<3;kx=kx+1){let xx=x+kx-1;if(xx<0||xx>=i32(u.W)){continue;}
            acc=acc+Wt[wic+u32(ky*3+kx)]*fin[pl+u32(yy)*u.W+u32(xx)];}}}
      if(u.has_prelu==1u){let s=Wt[u.prelu_off+u32(oc)];if(acc<0.0){acc=acc*s;}}
      fout[u32(oc)*u.H*u.W+u32(y)*u.W+u32(x)]=acc;}`,
  dispatch: (ly:any,H:number,W:number)=>[Math.ceil(W/8),Math.ceil(H/8),ly.out_c],
};
