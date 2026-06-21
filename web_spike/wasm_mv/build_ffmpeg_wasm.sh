#!/usr/bin/env bash
# Build a MINIMAL FFmpeg (libav*) to WASM via Emscripten, then compile the MV-extraction wrapper.
# Goal: surface av_frame_get_side_data(AV_FRAME_DATA_MOTION_VECTORS) in the browser/Node — the make-or-break
# seam for the web-only port (WebCodecs has no MV API). H.264 decode + MOV/h264 demux only; no asm, no x264/x265.
set -e
FF=/tmp/ffmpeg-src
EMSDK=/tmp/emsdk
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$EMSDK/emsdk_env.sh" >/dev/null 2>&1

echo "=== [1/3] configure FFmpeg for WASM (if not already) ==="
cd "$FF"
if [ ! -f config.h ]; then
  emconfigure ./configure \
    --cc=emcc --cxx=em++ --ar=emar --ranlib=emranlib --nm=llvm-nm --objcc=emcc --dep-cc=emcc \
    --enable-cross-compile --target-os=none --arch=x86_32 --cpu=generic \
    --disable-asm --disable-x86asm --disable-inline-asm \
    --disable-everything \
    --enable-decoder=h264 \
    --enable-decoder=mpeg2video `# NEEDED: pulls in mpegvideodec -> CONFIG_MPEGVIDEODEC=1, which gates the h264 MV-export call (ff_print_debug_info2) in h264dec.c. Without it the decoder runs but emits ZERO motion vectors.` \
    --enable-demuxer=mov --enable-demuxer=h264 --enable-demuxer=matroska \
    --enable-parser=h264 \
    --enable-protocol=file \
    --disable-programs --disable-doc --disable-network --disable-autodetect \
    --disable-pthreads --disable-w32threads --disable-os2threads \
    --disable-shared --enable-static --disable-debug \
    --extra-cflags="-O2" 2>&1 | tail -8
fi

echo "=== [2/3] build libav*.a (emmake make) ==="
emmake make -j"$(sysctl -n hw.ncpu)" 2>&1 | tail -5
ls -la libavcodec/libavcodec.a libavformat/libavformat.a libavutil/libavutil.a

echo "=== [3/3] compile the MV-extraction wrapper to WASM ==="
cd "$HERE"
emcc mv_wasm.c -I"$FF" \
  "$FF/libavformat/libavformat.a" "$FF/libavcodec/libavcodec.a" "$FF/libavutil/libavutil.a" \
  -O2 -sWASM=1 -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node,web \
  -sEXPORTED_RUNTIME_METHODS='["FS","callMain"]' -sINVOKE_RUN=0 -sEXIT_RUNTIME=0 \
  -sINITIAL_MEMORY=67108864 -sALLOW_MEMORY_GROWTH=1 -sFORCE_FILESYSTEM=1 \
  -o mv_wasm.mjs
echo "=== DONE: $(ls -la mv_wasm.wasm 2>/dev/null | awk '{print $5}') byte wasm ==="
