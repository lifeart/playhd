// HD warp + occlusion handling — the ONE source of truth for the shader text, shared by the
// shipped player (player.html) and the R13-E1 headless A/B harness
// (experiments/r13_e1_browser_confidence/harness.ts). Keeping a single copy is deliberate:
// two hand-synced copies of a shader is exactly the seam-mismatch failure this repo keeps
// paying for.
//
// gate:"hard"  = the SHIPPED binary switch. warpShaderWGSL({gate:"hard"}) returns the shader
//                text byte-identical to the string that was inline in player.html before R13,
//                so the default page is unchanged. (Checked by parity_shader.mjs.)
// gate:"soft"  = per-pixel confidence lerp, a port of prototype/residual_gate.py's
//                reliability_lr()/reliability_hd(): a = sigmoid((tau - react)/s), a = 0 at
//                intra holes, the LR reliability map BILINEARLY upsampled to HD and then
//                hard-zeroed at HD holes (== reliability_hd). Needs the extra LR reliability
//                pass below (reliabilityShaderWGSL) rendered into a filterable r16float
//                texture bound at @binding(6).
//
// The soft path's ONE deliberate semantic deviation from residual_gate.py: the (1-a) source.
// The server twin blends FRESH PER-FRAME SR there; the browser has no per-frame SR on a
// propagated frame (that is the point of MV propagation), so it blends the bilinear upsample
// of the current LR frame — i.e. the same pixels the hard switch already falls back to.
// Consequence: a < 1 costs sharpness in the browser but is free on the server, so the browser
// wants a gate that saturates to 1 on well-predicted pixels (see REPORT.md).

/** The shipped HD warp pass. opts: {gate:"hard"|"soft", blocks:boolean} */
export function warpShaderWGSL(opts = {}) {
  const gate = opts.gate || "hard";
  if (gate === "hard") {
    // ---- byte-identical to the pre-R13 inline shader; do not reformat ----
    return `
    struct U{w:u32,h:u32,scale:u32,lr_w:u32,lr_h:u32,tau:u32,_a:u32,_b:u32};
    @group(0)@binding(0) var prevRecon:texture_2d<f32>;@group(0)@binding(1) var s:sampler;@group(0)@binding(2) var flowT:texture_2d<f32>;@group(0)@binding(3) var lrCur:texture_2d<f32>;@group(0)@binding(4) var<uniform> u:U;@group(0)@binding(5) var lrPrev:texture_2d<f32>;
    @vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return vec4f(p[i],0,1);}
    @fragment fn fs(@builtin(position) pos:vec4f)->@location(0) vec4f{
      let dst=vec2i(floor(pos.xy));let lr=dst/i32(u.scale);let f=textureLoad(flowT,lr,0).xy;
      var bad = f.x < -1.0e8;
      if(!bad){ let predUV=(vec2f(lr)+f+vec2f(0.5))/vec2f(f32(u.lr_w),f32(u.lr_h));
        let pv=textureSampleLevel(lrPrev,s,predUV,0.0).rgb; let cv=textureLoad(lrCur,lr,0).rgb;
        bad = (abs(cv.r-pv.r)+abs(cv.g-pv.g)+abs(cv.b-pv.b))/3.0*255.0 > f32(u.tau); }
      if(bad){ return vec4f(textureSampleLevel(lrCur,s,pos.xy/vec2f(f32(u.w),f32(u.h)),0.0).rgb,1.0); }
      let src=vec2f(dst)+f*f32(u.scale); return vec4f(textureSampleLevel(prevRecon,s,(src+vec2f(0.5))/vec2f(f32(u.w),f32(u.h)),0.0).rgb,1.0); }`;
  }
  // ---- soft confidence lerp ----
  // Same bindings 0..5 as the hard path (so the JS bind-group code is shared) + binding 6 =
  // the LR reliability map produced by reliabilityShaderWGSL. u.tau is unused here (the gate
  // lives entirely in the reliability pass) but the uniform layout is kept identical.
  return `
    struct U{w:u32,h:u32,scale:u32,lr_w:u32,lr_h:u32,tau:u32,_a:u32,_b:u32};
    @group(0)@binding(0) var prevRecon:texture_2d<f32>;@group(0)@binding(1) var s:sampler;@group(0)@binding(2) var flowT:texture_2d<f32>;@group(0)@binding(3) var lrCur:texture_2d<f32>;@group(0)@binding(4) var<uniform> u:U;@group(0)@binding(5) var lrPrev:texture_2d<f32>;@group(0)@binding(6) var reliT:texture_2d<f32>;
    @vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return vec4f(p[i],0,1);}
    @fragment fn fs(@builtin(position) pos:vec4f)->@location(0) vec4f{
      let dst=vec2i(floor(pos.xy));let lr=dst/i32(u.scale);let f=textureLoad(flowT,lr,0).xy;
      let uvHD=pos.xy/vec2f(f32(u.w),f32(u.h));
      let fresh=textureSampleLevel(lrCur,s,uvHD,0.0).rgb;          // (1-a) source: bilinear LR
      // intra hole -> a=0, exactly like reliability_hd()'s HD hard-zero (NEAREST hole mask).
      if(f.x < -1.0e8){ return vec4f(fresh,1.0); }
      // a_hd = BILINEAR upsample of the LR reliability map (== cv2.resize INTER_LINEAR in
      // reliability_hd): sampling at uvHD is the same mapping cv2 uses (dst center -> src).
      let a=clamp(textureSampleLevel(reliT,s,uvHD,0.0).r,0.0,1.0);
      if(a<=0.00390625){ return vec4f(fresh,1.0); }                 // <1/256: skip the warp fetch
      let src=vec2f(dst)+f*f32(u.scale);
      let warped=textureSampleLevel(prevRecon,s,(src+vec2f(0.5))/vec2f(f32(u.w),f32(u.h)),0.0).rgb;
      return vec4f(mix(fresh,warped,a),1.0); }`;
}

/**
 * The LR-resolution reliability pass for gate:"soft" (renders into an r16float LR texture).
 * a = sigmoid((tau_eff - react)/s), react = mean_c|LR_cur - MV_warp(LR_prev)|*255 (codes),
 * a = 0 at intra holes — i.e. prototype/residual_gate.py::reliability_lr, per-pixel.
 * blocks:true additionally lowers tau per block from the FREE per-block codec signals that
 * ride in the flow texture's otherwise-unused .zw: .z = partition area in px (16..256),
 * .w = |MV| in LR px. Both are priors on MV trustworthiness measured on this clip
 * (sub-16 partitions and long MVs predict much larger warp error) — they cost zero extra
 * bandwidth because the flow texture is already rgba32float.
 */
export function reliabilityShaderWGSL(opts = {}) {
  const blocks = !!opts.blocks;
  return `
    struct G{lr_w:u32,lr_h:u32,tau:f32,s:f32,kp:f32,km:f32,mvref:f32,_p:f32};
    @group(0)@binding(0) var flowT:texture_2d<f32>;@group(0)@binding(1) var lrCur:texture_2d<f32>;@group(0)@binding(2) var lrPrev:texture_2d<f32>;@group(0)@binding(3) var s:sampler;@group(0)@binding(4) var<uniform> g:G;
    @vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4f{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return vec4f(p[i],0,1);}
    @fragment fn fs(@builtin(position) pos:vec4f)->@location(0) vec4f{
      let p=vec2i(floor(pos.xy)); let f=textureLoad(flowT,p,0);
      if(f.x < -1.0e8){ return vec4f(0.0,0.0,0.0,1.0); }            // intra hole -> fully fresh
      let predUV=(vec2f(p)+f.xy+vec2f(0.5))/vec2f(f32(g.lr_w),f32(g.lr_h));
      let pv=textureSampleLevel(lrPrev,s,predUV,0.0).rgb; let cv=textureLoad(lrCur,p,0).rgb;
      let react=(abs(cv.r-pv.r)+abs(cv.g-pv.g)+abs(cv.b-pv.b))/3.0*255.0;
      var tau=g.tau;
${blocks ? `      let zc=clamp(f.z/256.0,0.0,1.0);                              // 1.0 = full 16x16 partition
      let mc=clamp(f.w/max(g.mvref,1.0e-3),0.0,1.0);                 // 1.0 = long MV (>= mvref px)
      tau=max(tau-g.kp*(1.0-zc)-g.km*mc,0.0);` : ""}
      let x=clamp((react-tau)/max(g.s,1.0e-6),-30.0,30.0);
      return vec4f(1.0/(1.0+exp(x)),0.0,0.0,1.0); }`;
}
