"""R2-E4 setup: build test clips with PyAV ONLY (system ffmpeg/ffprobe is broken).

Creates, under experiments/r2_e4_streamharden/:
  * short_mp3.mp4      -- short.mp4 video COPIED, audio transcoded aac -> mp3   (non-AAC src)
  * short_opus.mp4     -- short.mp4 video COPIED, audio transcoded aac -> opus  (non-AAC src, bonus)
  * short_noaudio.mp4  -- short.mp4 video COPIED, audio dropped                 (video-only src)

These EXERCISE progressive.FragmentMuxer's non-AAC `_transcode_audio_pairs` and the
audio-less `audio_note="none"` branch respectively.
"""
import os
import av

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "server", "testdata", "short.mp4")


def _remux_with_audio_codec(src, dst, audio_codec, rate=None):
    """Copy the video stream verbatim; transcode the source audio to `audio_codec`.
    Returns the produced (src_audio_codec -> dst_audio_codec) note."""
    sin = av.open(src)
    vin = sin.streams.video[0]
    ain = sin.streams.audio[0]
    out = av.open(dst, "w")
    vout = out.add_stream_from_template(vin)            # COPY video (no re-encode)
    aout = out.add_stream(audio_codec, rate=rate or ain.codec_context.sample_rate or 44100)
    cc = aout.codec_context
    resampler = av.AudioResampler(format=cc.format, layout=cc.layout, rate=cc.sample_rate)
    fifo = av.AudioFifo()

    # Interleave by decoding audio fully first into packets, then write both streams ordered.
    apkts = []
    for frame in sin.decode(ain):
        frame.pts = None
        for rf in resampler.resample(frame):
            fifo.write(rf)
        fs = cc.frame_size or 1024
        while fifo.samples >= fs:
            for pkt in aout.encode(fifo.read(fs)):
                pkt.stream = aout
                apkts.append(pkt)
    rem = fifo.read()
    if rem is not None:
        for pkt in aout.encode(rem):
            pkt.stream = aout
            apkts.append(pkt)
    for pkt in aout.encode():
        pkt.stream = aout
        apkts.append(pkt)

    # Re-demux video packets (copy) and mux both streams; PyAV's interleaver orders by dts.
    sin.seek(0)
    for pkt in sin.demux(vin):
        if pkt.dts is None:
            continue
        pkt.stream = vout
        out.mux(pkt)
    for pkt in apkts:
        out.mux(pkt)
    out.close()
    sin.close()
    return f"{ain.codec_context.name} -> {audio_codec}"


def _strip_audio(src, dst):
    sin = av.open(src)
    vin = sin.streams.video[0]
    out = av.open(dst, "w")
    vout = out.add_stream_from_template(vin)
    for pkt in sin.demux(vin):
        if pkt.dts is None:
            continue
        pkt.stream = vout
        out.mux(pkt)
    out.close()
    sin.close()


def _verify(path):
    c = av.open(path)
    rows = []
    for s in c.streams:
        dur = float(s.duration * s.time_base) if s.duration else None
        rows.append((s.type, getattr(s.codec_context, "name", None), dur, s.frames))
    c.close()
    return rows


if __name__ == "__main__":
    jobs = [
        ("short_mp3.mp4", lambda d: _remux_with_audio_codec(SRC, d, "mp3")),
        ("short_opus.mp4", lambda d: _remux_with_audio_codec(SRC, d, "libopus", rate=48000)),
        ("short_noaudio.mp4", lambda d: _strip_audio(SRC, d)),
    ]
    for name, fn in jobs:
        dst = os.path.join(HERE, name)
        try:
            note = fn(dst)
            print(f"[make] {name}: OK  {note or 'video-only'}")
            for r in _verify(dst):
                print(f"        stream {r}")
        except Exception as e:
            print(f"[make] {name}: FAILED {type(e).__name__}: {e}")
