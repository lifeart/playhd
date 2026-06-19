#!/usr/bin/env python3
"""Final R2-E2 frontier plot: combine the coarse sweep (results.json: incl. ghosting b1/b2)
and the refined escape grid (refine_results.json). One figure, the deliverable."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
_HERE = os.path.dirname(os.path.abspath(__file__))

coarse = json.load(open(os.path.join(_HERE, "results.json")))["results"]
refine = json.load(open(os.path.join(_HERE, "refine_results.json")))["results"]
bic = next(r for r in coarse if r["scheme"].startswith("bicubic"))
hard = next(r for r in coarse if r["scheme"].startswith("HARD-SR all (binary)"))

def style(n):
    if n.startswith("bicubic"):      return "k", "*", 260, "bicubic (tOF-optimal baseline)"
    if n.startswith("HARD"):         return "red", "X", 130, "hard SR-escalate (R1 point)"
    if n.startswith("(a')"):         return "darkorange", "^", 70, "(a') conf-graded feather"
    if n.startswith("(a)"):          return "gold", "v", 70, "(a) spatial feather"
    if n.startswith("(b1"):          return "tab:brown", "s", 70, "(b1) SR+warp blend (GHOST)"
    if n.startswith("(b2"):          return "magenta", "P", 70, "(b2) screen-EMA (GHOST)"
    if n.startswith("(b3"):          return "tab:blue", "o", 70, "(b3) HF-only EMA"
    if n.startswith("(c)"):          return "tab:green", "D", 70, "(c) combo (conf-feather x HF-EMA)"
    return "gray", ".", 40, n

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.5))
seen = set()
allpts = coarse + [r for r in refine if r["scheme"] not in ("bicubic", "HARD-SR all")]
for ax, zoom in ((axL, False), (axR, True)):
    for r in allpts:
        c, m, s, lab = style(r["scheme"])
        ax.scatter(r["eff_bicubic_pct"], r["tof"], c=c, marker=m, s=s, zorder=3,
                   edgecolors="k", linewidths=0.3,
                   label=(lab if lab not in seen else None))
        seen.add(lab)
    # the R1 frontier line (bicubic <-> hard): points below+left of it ESCAPE
    ax.plot([bic["eff_bicubic_pct"], hard["eff_bicubic_pct"]],
            [bic["tof"], hard["tof"]], "r--", lw=1.2, alpha=0.7,
            label="R1 frontier (bicubic<->hard)")
    ax.axhline(bic["tof"], color="k", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("eff-bicubic %  (lower = less soft fallback shown ->)")
    ax.set_ylabel("tOF  (lower = steadier)")
    ax.grid(alpha=0.3)
    if zoom:
        ax.set_xlim(5.5, 8.0); ax.set_ylim(0.72, 1.0)
        ax.set_title("ZOOM: the escape region (near bicubic's tOF)")
    else:
        ax.set_title("FULL: all schemes (b1/b2 screen-EMA & warp-blend GHOST -> tOF blows up)")
axL.legend(fontsize=7, loc="upper right")
fig.suptitle("R2-E2 frontier -- window A (high-motion): tOF vs eff-bicubic%.  "
             "ESCAPE = below the red dashed line.  (b3)/(c) lower fallback% AT bicubic's tOF.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
p = os.path.join(_HERE, "frontier.png")
fig.savefig(p, dpi=120)
print("wrote", p)
