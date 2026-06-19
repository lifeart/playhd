"""Honest ms/frame: warm ONCE, then time every config BACK-TO-BACK in one process so they
share the warmed SR graph and the same shared-GPU contention -> the RATIO vs baseline is the
trustworthy number (absolute ms/frame drifts run-to-run under the shared GPU). Two passes; the
second is the steady-state read. Reuses bench.run_cell (the real product fast path)."""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402

CLIP = "c3_lowlight"
CFGS = [("baseline", {}), ("cap0.70", {"cap": 0.70}),
        ("cap0.70+bilat", {"cap": 0.70, "denoise": "bilat"}),
        ("cap0.70+gauss", {"cap": 0.70, "denoise": "gauss"}),
        ("denoise_gauss", {"denoise": "gauss"})]
N = 24


def time_cfg(name, cfg):
    t0 = time.perf_counter()
    bench.run_cell(CLIP, "t_" + name, cfg, N)         # writes/overwrites out/<clip>_t_<name>.mp4
    s = dict(bench.P.LAST_STATS)
    return {"ms_per_frame": s.get("ms_per_frame"), "n_sr_calls": s.get("n_sr_calls"),
            "t_sr_s": s.get("t_sr_s"), "t_recon_s": s.get("t_recon_s"),
            "t_grain_io_s": s.get("t_grain_io_s"), "t_encode_s": s.get("t_encode_s"),
            "wall_s": round(time.perf_counter() - t0, 2)}


def main():
    print("warmup ...", flush=True)
    bench.run_cell(CLIP, "warmup", {}, N)             # warm the MPS SR graph (one-off per process)
    bench._free_gpu()
    out = {}
    for pass_i in (1, 2):
        for name, cfg in CFGS:
            r = time_cfg(name, cfg)
            out.setdefault(name, {})[f"pass{pass_i}"] = r
            bench._free_gpu()
            print(f"pass{pass_i} {name:16s} ms/frame={r['ms_per_frame']:6} "
                  f"n_sr={r['n_sr_calls']:3} t_sr={r['t_sr_s']} t_recon={r['t_recon_s']}", flush=True)
    base = out["baseline"]["pass2"]["ms_per_frame"]
    print("\n-- steady-state (pass2) ms/frame and ratio vs baseline --")
    for name, _ in CFGS:
        m = out[name]["pass2"]["ms_per_frame"]
        print(f"  {name:16s} {m:7} ms/frame   {m / base:4.2f}x baseline   real-time(<=45ms): {m <= 45.0}")
    json.dump(out, open(os.path.join(HERE, "timing.json"), "w"), indent=2)
    print("\ntiming -> timing.json")


if __name__ == "__main__":
    main()
