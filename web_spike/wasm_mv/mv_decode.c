// Clean JS-callable decode API over the WASM libav build: per decoded frame, yields BOTH the RGB24 pixels
// (libswscale) AND the packed motion vectors (av_frame_get_side_data(MOTION_VECTORS)). Replaces the offline
// lr_*.png + flow_*.bin: one in-browser software decode gives the pipeline {frame, mvs} live.
//   mvdec_open("/in.mp4") -> 0 ; loop mvdec_next() (0=frame,1=eof,<0=err) reading mvdec_rgb()/mvdec_mvs().
#include <stdlib.h>
#include <string.h>
#include <libavutil/motion_vector.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <emscripten.h>

static AVFormatContext *fmt;
static AVCodecContext *dec;
static int vstream = -1, eof_sent = 0;
static AVFrame *frame;
static AVPacket *pkt;
static struct SwsContext *sws;
static uint8_t *rgb; static int rgb_w, rgb_h;
static int *mvbuf; static int mvcap, nmv;   // packed: 10 ints/MV [source,w,h,src_x,src_y,dst_x,dst_y,motion_x,motion_y,motion_scale]

EMSCRIPTEN_KEEPALIVE int mvdec_open(const char *path) {
  if (avformat_open_input(&fmt, path, NULL, NULL) < 0) return -1;
  if (avformat_find_stream_info(fmt, NULL) < 0) return -2;
  const AVCodec *codec = NULL;
  int ret = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  if (ret < 0) return -3;
  vstream = ret;
  dec = avcodec_alloc_context3(codec);
  if (!dec) return -4;
  avcodec_parameters_to_context(dec, fmt->streams[vstream]->codecpar);
  AVDictionary *opts = NULL;
  av_dict_set(&opts, "flags2", "+export_mvs", 0);   // THE motion-vector switch
  ret = avcodec_open2(dec, codec, &opts);
  av_dict_free(&opts);
  if (ret < 0) return -5;
  frame = av_frame_alloc();
  pkt = av_packet_alloc();
  rgb_w = dec->width; rgb_h = dec->height;
  rgb = malloc((size_t)rgb_w * rgb_h * 3);
  sws = sws_getContext(rgb_w, rgb_h, dec->pix_fmt, rgb_w, rgb_h, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
  return (rgb && sws) ? 0 : -6;
}

// pull one decoded frame from the decoder; 0=got it, 2=need more packets, 1=eof, <0=err
static int receive_one(void) {
  int ret = avcodec_receive_frame(dec, frame);
  if (ret == AVERROR(EAGAIN)) return 2;
  if (ret == AVERROR_EOF) return 1;
  if (ret < 0) return -10;
  uint8_t *dst[4] = { rgb, NULL, NULL, NULL };
  int dstst[4] = { rgb_w * 3, 0, 0, 0 };
  sws_scale(sws, (const uint8_t * const *)frame->data, frame->linesize, 0, rgb_h, dst, dstst);
  nmv = 0;
  AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
  if (sd) {
    const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
    int n = sd->size / sizeof(AVMotionVector);
    if (n > mvcap) { mvcap = n; mvbuf = realloc(mvbuf, (size_t)mvcap * 10 * sizeof(int)); }
    for (int i = 0; i < n; i++) {
      const AVMotionVector *m = &mvs[i]; int *o = &mvbuf[i * 10];
      o[0] = m->source; o[1] = m->w; o[2] = m->h; o[3] = m->src_x; o[4] = m->src_y;
      o[5] = m->dst_x; o[6] = m->dst_y; o[7] = m->motion_x; o[8] = m->motion_y; o[9] = m->motion_scale;
    }
    nmv = n;
  }
  av_frame_unref(frame);
  return 0;
}

EMSCRIPTEN_KEEPALIVE int mvdec_next(void) {
  for (;;) {
    int r = receive_one();
    if (r != 2) return r;                 // 0=frame / 1=eof / <0=err
    int got = 0;                          // EAGAIN -> feed a packet
    while (av_read_frame(fmt, pkt) >= 0) {
      if (pkt->stream_index == vstream) { avcodec_send_packet(dec, pkt); av_packet_unref(pkt); got = 1; break; }
      av_packet_unref(pkt);
    }
    if (!got) {
      if (!eof_sent) { avcodec_send_packet(dec, NULL); eof_sent = 1; continue; }  // flush
      return 1;
    }
  }
}

EMSCRIPTEN_KEEPALIVE int mvdec_width(void)  { return rgb_w; }
EMSCRIPTEN_KEEPALIVE int mvdec_height(void) { return rgb_h; }
EMSCRIPTEN_KEEPALIVE uint8_t *mvdec_rgb(void) { return rgb; }     // rgb_w*rgb_h*3 bytes
EMSCRIPTEN_KEEPALIVE int mvdec_nmv(void)    { return nmv; }
EMSCRIPTEN_KEEPALIVE int *mvdec_mvs(void)   { return mvbuf; }     // nmv*10 ints
EMSCRIPTEN_KEEPALIVE void mvdec_close(void) {
  sws_freeContext(sws); avcodec_free_context(&dec); avformat_close_input(&fmt);
  av_frame_free(&frame); av_packet_free(&pkt); free(rgb); free(mvbuf);
  rgb = NULL; mvbuf = NULL; mvcap = nmv = 0; vstream = -1; eof_sent = 0;
}
