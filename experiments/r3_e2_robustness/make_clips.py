"""R3-E2 robustness QA -- author short, diverse H.264 test clips (PyAV; system ffmpeg CLI is
broken). Each clip targets ONE content regime the product was NOT validated on. All clips are
libx264 yuv420p, LR ~480x272 (x2=960x544 instant, x4=1920x1088 quality -- both even), short
(<=48 frames). One clip carries an MP3 audio stream to exercise the audio TRANSCODE mux path.

Run:  python3 experiments/r3_e2_robustness/make_clips.py
Writes clips into experiments/r3_e2_robustness/clips/ and prints a manifest.
"""
import os
import numpy as np
import cv2
import av
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(HERE, "clips")
os.makedirs(CLIPS, exist_ok=True)

N = 40  # author 40 frames; sweep caps at <=24-32


def _encode(path, frames_rgb, fps=25, sc_zero=False, audio_mp3=False, fps_num=None, fps_den=None):
    """Encode a list of HxWx3 uint8 RGB frames as libx264 yuv420p. sc_zero disables x264's
    own scene-cut I-frame insertion (so a hard visual cut lands on a P-frame -> exercises
    the pipeline's RGB-diff scene detector). audio_mp3 adds a sine-tone MP3 stream (transcode
    path). fps_num/den allow odd frame rates (e.g. 24000/1001 = 23.976)."""
    cont = av.open(path, "w")
    rate = Fraction(fps_num, fps_den) if fps_num else fps
    vst = cont.add_stream("libx264", rate=rate)
    h, w = frames_rgb[0].shape[:2]
    vst.width, vst.height, vst.pix_fmt = w, h, "yuv420p"
    opts = {"crf": "18"}
    if sc_zero:
        opts.update({"sc_threshold": "0", "g": str(len(frames_rgb) + 10), "bf": "0"})
    vst.options = opts

    ast = None
    if audio_mp3:
        ast = cont.add_stream("mp3", rate=44100)

    for f in frames_rgb:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")
        for p in vst.encode(vf):
            cont.mux(p)
    for p in vst.encode():
        cont.mux(p)

    if ast is not None:
        sr = 44100
        dur = len(frames_rgb) / float(fps)
        t = np.arange(int(sr * dur), dtype=np.float32) / sr
        tone = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        # frame the audio into 1152-sample MP3 frames (mono->fltp planar)
        fs = ast.codec_context.frame_size or 1152
        pts = 0
        for s in range(0, len(tone), fs):
            chunk = tone[s:s + fs]
            if len(chunk) < fs:
                chunk = np.pad(chunk, (0, fs - len(chunk)))
            af = av.AudioFrame.from_ndarray(chunk[None, :], format="fltp", layout="mono")
            af.sample_rate = sr
            af.pts = pts
            af.time_base = Fraction(1, sr)
            pts += fs
            for p in ast.encode(af):
                cont.mux(p)
        for p in ast.encode():
            cont.mux(p)
    cont.close()


def _textured_canvas(h, w, seed=7):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    canvas = np.stack([
        128 + 110 * np.sin(xx / 23.0),
        128 + 110 * np.sin(yy / 17.0 + 1.0),
        128 + 110 * np.sin((xx + yy) / 31.0 + 2.0),
    ], axis=-1)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    for _ in range(60):
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.5:
            cv2.circle(canvas, (x, y), int(rng.integers(6, 26)), col, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (x, y), (x + int(rng.integers(8, 34)),
                          y + int(rng.integers(8, 34))), col, -1)
    return np.ascontiguousarray(canvas)


# --------------------------------------------------------------------------- #
# C1 fastpan -- high global motion / fast horizontal pan + a fast object.
# --------------------------------------------------------------------------- #
def clip_fastpan(w=480, h=272):
    pan = 26                      # LR px/frame pan -> very large MVs
    canvas = _textured_canvas(h, w + pan * N + 80, seed=3)
    obj = cv2.GaussianBlur(np.random.default_rng(1).integers(0, 256, (60, 70, 3),
                           dtype=np.uint8), (3, 3), 0)
    frames = []
    for i in range(N):
        off = i * pan
        f = canvas[:, off:off + w].copy()
        ox = (i * 17) % max(1, w - obj.shape[1])
        f[80:80 + obj.shape[0], ox:ox + obj.shape[1]] = obj
        frames.append(f)
    return frames


# --------------------------------------------------------------------------- #
# C2 graphics -- sharp text / UI overlay on FLAT fields + a scrolling ticker.
# --------------------------------------------------------------------------- #
def clip_graphics(w=480, h=272):
    frames = []
    for i in range(N):
        f = np.full((h, w, 3), (14, 22, 40), np.uint8)              # flat dark-blue field
        cv2.rectangle(f, (0, 0), (w, 40), (200, 40, 40), -1)        # flat header bar
        cv2.putText(f, "BREAKING NEWS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(f, (20, h - 70), (w - 20, h - 30), (240, 240, 240), -1)  # lower third
        cv2.putText(f, "Q3 REVENUE +12.4%%", (30, h - 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (10, 10, 10), 2, cv2.LINE_AA)
        # scrolling ticker (sharp moving text on a flat band)
        tick = "  LIVE  ::  MARKETS UP  ::  INDEX 4823.55  ::  +0.8%  ::  "
        scr = (i * 13) % 400
        cv2.rectangle(f, (0, h - 26), (w, h), (0, 0, 0), -1)
        cv2.putText(f, tick * 3, (-scr, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 120), 1, cv2.LINE_AA)
        # thin resolution lines (HF flat-field detail)
        for x in range(60, 180, 4):
            cv2.line(f, (x, 60), (x, 120), (255, 255, 255), 1)
        frames.append(f)
    return frames


# --------------------------------------------------------------------------- #
# C3 lowlight -- very dark, low-contrast, heavy per-frame noise; small motion.
# --------------------------------------------------------------------------- #
def clip_lowlight(w=480, h=272):
    rng = np.random.default_rng(11)
    base = np.full((h, w, 3), 14, np.uint8)
    cv2.circle(base, (w // 2, h // 2), 60, (40, 38, 30), -1, cv2.LINE_AA)  # dim subject
    cv2.rectangle(base, (60, 60), (120, 110), (28, 24, 20), -1)
    frames = []
    for i in range(N):
        f = base.copy()
        cx = w // 2 + int(6 * np.sin(i / 4.0))                # tiny subject motion
        cv2.circle(f, (cx, h // 2), 30, (60, 55, 45), -1, cv2.LINE_AA)
        noise = rng.normal(0, 14, (h, w, 3))                 # heavy sensor noise
        f = np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        frames.append(f)
    return frames


# --------------------------------------------------------------------------- #
# C4 talkinghead -- near-static (validated regime, control) + AUDIO (mp3 transcode path).
# --------------------------------------------------------------------------- #
def clip_talkinghead(w=480, h=272):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.stack([90 + 30 * np.sin(xx / 90.0), 80 + 20 * np.sin(yy / 70.0),
                   110 + 25 * np.sin((xx + yy) / 120.0)], -1)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    frames = []
    for i in range(N):
        f = bg.copy()
        cy = h // 2 + int(3 * np.sin(i / 5.0))               # slight head bob
        cv2.ellipse(f, (w // 2, cy), (55, 70), 0, 0, 360, (200, 175, 150), -1, cv2.LINE_AA)
        mh = 6 + int(5 * abs(np.sin(i / 2.0)))               # moving mouth
        cv2.ellipse(f, (w // 2, cy + 30), (16, mh), 0, 0, 360, (90, 40, 40), -1, cv2.LINE_AA)
        cv2.circle(f, (w // 2 - 20, cy - 12), 6, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.circle(f, (w // 2 + 20, cy - 12), 6, (40, 40, 40), -1, cv2.LINE_AA)
        frames.append(f)
    return frames


# --------------------------------------------------------------------------- #
# C5 scenecut -- hard cut at frame 20; encoded WITHOUT x264 scenecut so the cut
# lands on a P-frame -> the pipeline's RGB-diff scene detector must catch it.
# --------------------------------------------------------------------------- #
def clip_scenecut(w=480, h=272):
    a = _textured_canvas(h, w + 8 * N + 40, seed=5)          # scene A: warm pan
    b = _textured_canvas(h, w + 8 * N + 40, seed=99)         # scene B: different texture
    b = (b.astype(np.float32) * np.array([0.5, 0.7, 1.4])).clip(0, 255).astype(np.uint8)  # cool tint
    frames = []
    for i in range(N):
        if i < 20:
            off = i * 8
            f = a[:, off:off + w].copy()
            cv2.circle(f, (60 + i * 6, 80), 22, (255, 60, 60), -1, cv2.LINE_AA)
        else:
            off = (i - 20) * 10
            f = b[:, off:off + w].copy()
            cv2.rectangle(f, (200, 120), (260, 180), (40, 255, 90), -1)
        frames.append(f)
    return frames


# --------------------------------------------------------------------------- #
# C6 oddres -- non-multiple-of-16 resolution (642x362) + odd fps (23.976);
# moderate pan content. Tests chunk/encoder/mux on awkward dims & timebase.
# --------------------------------------------------------------------------- #
def clip_oddres(w=642, h=362):
    canvas = _textured_canvas(h, w + 10 * N + 40, seed=21)
    frames = []
    for i in range(N):
        off = i * 10
        f = canvas[:, off:off + w].copy()
        cv2.circle(f, (100 + i * 5, h // 2), 26, (255, 255, 0), -1, cv2.LINE_AA)
        frames.append(f)
    return frames


def main():
    manifest = []
    jobs = [
        ("c1_fastpan.mp4", clip_fastpan(), dict(fps=25)),
        ("c2_graphics.mp4", clip_graphics(), dict(fps=25)),
        ("c3_lowlight.mp4", clip_lowlight(), dict(fps=25)),
        ("c4_talkinghead.mp4", clip_talkinghead(), dict(fps=25, audio_mp3=True)),
        ("c5_scenecut.mp4", clip_scenecut(), dict(fps=25, sc_zero=True)),
        ("c6_oddres.mp4", clip_oddres(), dict(fps_num=24000, fps_den=1001, fps=23.976)),
    ]
    for name, frames, kw in jobs:
        path = os.path.join(CLIPS, name)
        _encode(path, frames, **kw)
        # re-decode to confirm validity + report what the codec produced
        c = av.open(path)
        vs = c.streams.video[0]
        ptypes = []
        for fr in c.decode(vs):
            ptypes.append({1: "I", 2: "P", 3: "B"}.get(int(fr.pict_type), "?"))
        has_a = len(c.streams.audio) > 0
        a_codec = c.streams.audio[0].codec_context.name if has_a else None
        c.close()
        manifest.append(dict(name=name, res=f"{vs.width}x{vs.height}",
                             fps=round(float(vs.average_rate), 3), n=len(ptypes),
                             iframes=ptypes.count("I"), pframes=ptypes.count("P"),
                             bframes=ptypes.count("B"), audio=a_codec,
                             ptypes="".join(ptypes)))
    print("MANIFEST:")
    for m in manifest:
        print(" ", m)


if __name__ == "__main__":
    main()
