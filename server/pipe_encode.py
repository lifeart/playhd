"""Lever 2 (pipeline parallelism) helpers for the instant fast path.

The instant per-frame tail is GPU work (SR -> recon -> grain -> GPU->host download) followed by
a VideoToolbox H.264 encode. The encode runs on Apple silicon's dedicated MEDIA ENGINE, which is
a separate unit from the GPU -- so it can run on frame i while the GPU is already producing frame
i+1. Serially it costs ~12 ms/frame on the critical path; overlapped it is hidden behind the GPU
stage (which is the floor). Likewise the next GOP chunk can be DECODED (CPU) while the GPU is busy
on the current chunk.

Two small, bounded-memory primitives:
  * ThreadedEncoder -- a one-worker queue in front of _VideoWriter. write() hands off an
    already-downloaded CPU frame and returns immediately; the worker encodes+muxes. A bounded
    queue back-pressures the producer so at most `maxsize` finished frames are in flight.
  * prefetch_chunks -- a one-worker prefetch of a chunk generator (decode the next GOP while the
    GPU processes the current one). Bounded to `maxsize` chunks in flight.

Both are GPU-free: the GPU stays single-threaded on the main thread (one GPU job at a time, as
required), and only the CPU-side encode/decode is moved off the critical path.
"""
import queue
import threading


class ThreadedEncoder:
    """Background-thread wrapper around a writer with a .write(rgb_uint8)/.close() interface.

    The caller (main GPU thread) downloads the finished HD frame to CPU and calls write(); the
    frame is queued and a single worker thread encodes it. Encoding (VideoToolbox media engine +
    CPU mux) thus overlaps the GPU producing the next frame. Bounded queue => bounded memory and
    natural back-pressure if the encoder ever falls behind. Exceptions in the worker are captured
    and re-raised on the next write()/close() so a failed encode never hangs silently."""

    def __init__(self, writer, maxsize=8):
        self.writer = writer
        self.q = queue.Queue(maxsize=maxsize)
        self.err = None
        self._t = threading.Thread(target=self._run, name="instant-encoder", daemon=True)
        self._t.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            try:
                self.writer.write(item)
            except Exception as e:               # surface, never swallow (CLAUDE.md rule)
                self.err = e
                self.q.task_done()
                # drain remaining items so producers don't block on a full queue after a failure
                continue
            self.q.task_done()

    def _check(self):
        if self.err is not None:
            raise self.err

    def write(self, rgb_uint8):
        self._check()
        self.q.put(rgb_uint8)                    # blocks if the encoder is >maxsize frames behind

    @property
    def encoder(self):
        return getattr(self.writer, "encoder", None)

    def close(self):
        self.q.put(None)
        self._t.join()
        self._check()
        self.writer.close()


def prefetch_chunks(gen, maxsize=2):
    """Yield from a (decode-heavy) chunk generator while a worker thread pulls the NEXT chunk(s)
    ahead -- so the next GOP decodes on the CPU while the GPU is busy on the current one. Bounded
    to `maxsize` chunks in flight. Exceptions from the generator are propagated in order."""
    q = queue.Queue(maxsize=maxsize)
    _SENTINEL = object()

    def _run():
        try:
            for item in gen:
                q.put(("item", item))
        except Exception as e:
            q.put(("err", e))
            return
        q.put(("done", _SENTINEL))

    t = threading.Thread(target=_run, name="instant-decoder", daemon=True)
    t.start()
    while True:
        kind, payload = q.get()
        if kind == "item":
            yield payload
        elif kind == "err":
            t.join()
            raise payload
        else:
            t.join()
            return
