"""HONEST re-validation of the clips where the CURRENT probe disagrees with R4-E4's STALE
results.json (c5_scenecut, c5b_scenecut_strong, sample_window@0). Shared deps evolved since R4-E4
was recorded, so we re-derive confirmed_best from CURRENT honest renders (tOF / lrc_min / ms) using
the EXACT R4-E4 rule, and check whether the ported probe agrees with the *current* honest best.

Reuses experiments/r4_e4_automode/validate.py wholesale (render_metrics + confirmed_best + eval_clip
+ _trim_window) so the methodology is identical to the validated R4-E4 study. One render at a time
(shared GPU); GPU freed between. PyAV trims (system ffmpeg broken)."""
from __future__ import annotations
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "server"))
sys.path.insert(0, os.path.join(REPO, "prototype"))
sys.path.insert(0, os.path.join(REPO, "experiments", "r4_e4_automode"))

import validate as V   # noqa: E402  (R4-E4 honest harness: eval_clip/confirmed_best/render_metrics)

CLIPS = V.CLIPS
SAMPLE = V.SAMPLE


def main():
    rows = []
    # authored disagreements (n=40, exactly as R4-E4)
    for clip in ["c5_scenecut", "c5b_scenecut_strong"]:
        path = os.path.join(CLIPS, clip + ".mp4")
        r = V.eval_clip(path, clip, V.AUTHORED_N)
        rows.append(r)
        print(f"\n{clip}: probe={r['recommended']}  honest_best={r['confirmed_best']}  "
              f"{'MATCH' if r['match'] else 'MISROUTE'}")
        print("   renders:", {m: (v.get('tof'), v.get('lrc_min'), v.get('ms_per_frame'))
                              for m, v in r['render'].items()})
        print("   best:", r['best_reason'])

    # sample window @0 (n=24)
    dst = os.path.join(V.TMP, "sample_window_0.mp4")
    clip, got = V._trim_window(SAMPLE, 0, V.WINDOW_N, dst)
    r = V.eval_clip(clip, "sample_window@0", min(got, V.WINDOW_N))
    rows.append(r)
    print(f"\nsample_window@0: probe={r['recommended']}  honest_best={r['confirmed_best']}  "
          f"{'MATCH' if r['match'] else 'MISROUTE'}")
    print("   renders:", {m: (v.get('tof'), v.get('lrc_min'), v.get('ms_per_frame'))
                          for m, v in r['render'].items()})
    print("   best:", r['best_reason'])

    json.dump(rows, open(os.path.join(HERE, "honest3_results.json"), "w"), indent=2)
    matches = sum(1 for r in rows if r["match"])
    print(f"\nHONEST agreement on the 3 disagreeing clips: {matches}/3")
    print("-> honest3_results.json")


if __name__ == "__main__":
    main()
