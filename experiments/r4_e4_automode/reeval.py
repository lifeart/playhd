"""Re-evaluate the (refined) probe against the CACHED renders in results.json -- no re-rendering.
For each clip/window it re-runs the cheap recommend_mode() probe and re-derives confirmed-best from
the already-measured render metrics (validate.confirmed_best), then prints the agreement table.

Run AFTER validate.py has produced results.json. Cheap (probes only; renders are reused)."""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "server")); sys.path.insert(0, os.path.join(REPO, "prototype"))
import recommend_mode as R          # noqa: E402
import validate as V                # noqa: E402  (reuse confirmed_best + paths)

OUT = os.path.join(HERE, "results.json")


def clip_path(tag):
    if tag in V.AUTHORED:
        return os.path.join(V.CLIPS, tag + ".mp4"), V.AUTHORED_N
    return os.path.join(V.TMP, tag.replace("@", "_") + ".mp4"), V.WINDOW_N


def main():
    data = json.load(open(OUT))
    rows = data["rows"]
    out = []
    print(f"{'clip':24s} {'mvMagMed':>8s} {'fb%':>6s} {'scenes':>6s} {'chroma':>6s} {'human':>6s} "
          f"{'probe':>8s} {'best':>8s}  match")
    for row in rows:
        tag = row["tag"]
        path, n = clip_path(tag)
        if not os.path.exists(path):
            print(f"{tag}: missing {path}"); continue
        rec = R.recommend_mode(path, max_frames=n, stride=1)
        best, why = V.confirmed_best(rec.mode, row["render"])
        match = (rec.mode == best)
        sg = rec.signals
        print(f"{tag:24s} {sg['mv_mag_median']:8.2f} {sg['fb_react_mean']:6.1f} {sg['n_scenes']:6d} "
              f"{sg['chroma_diff_max']:6.1f} {str(sg['human_coverage']):>6s} {rec.mode:>8s} {best:>8s}  "
              f"{'OK' if match else 'MISROUTE'}")
        out.append({"tag": tag, "recommended": rec.mode, "confirmed_best": best, "match": match,
                    "probe_reason": rec.reason, "best_reason": why, "signals": sg,
                    "render": row["render"]})
    auth = [r for r in out if r["tag"] in V.AUTHORED]
    na = sum(1 for r in auth if r["match"]); nall = sum(1 for r in out if r["match"])
    print("\n" + "=" * 70)
    print(f"AGREEMENT authored {na}/{len(auth)} = {100*na/max(1,len(auth)):.0f}%   |   "
          f"all {nall}/{len(out)} = {100*nall/max(1,len(out)):.0f}%")
    mis = [r for r in out if not r["match"]]
    print("MISROUTES:", [f"{r['tag']}({r['recommended']}!={r['confirmed_best']})" for r in mis] or "none")
    json.dump({"rows": out, "agreement_authored": f"{na}/{len(auth)}",
               "agreement_all": f"{nall}/{len(out)}"}, open(os.path.join(HERE, "reeval.json"), "w"), indent=2)
    print("written -> reeval.json")


if __name__ == "__main__":
    main()
