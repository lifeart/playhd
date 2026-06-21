#!/usr/bin/env bash
# Compile the clean {frame, mvs, fps} decode API to WASM (needs the libav*.a from build_ffmpeg_wasm.sh).
set -e; FF=/tmp/ffmpeg-src; source /tmp/emsdk/emsdk_env.sh >/dev/null 2>&1
emcc mv_decode.c -I"$FF" "$FF/libavformat/libavformat.a" "$FF/libavcodec/libavcodec.a" "$FF/libswscale/libswscale.a" "$FF/libavutil/libavutil.a" \
  -O2 -sWASM=1 -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node,web \
  -sEXPORTED_FUNCTIONS='["_mvdec_open","_mvdec_next","_mvdec_width","_mvdec_height","_mvdec_fps","_mvdec_rgb","_mvdec_nmv","_mvdec_mvs","_mvdec_close","_malloc","_free"]' \
  -sEXPORTED_RUNTIME_METHODS='["FS","ccall","cwrap","HEAPU8","HEAP32"]' \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=67108864 -sFORCE_FILESYSTEM=1 -o mv_decode.mjs
