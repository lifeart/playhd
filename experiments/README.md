# playhd — research experiments (round R1)

Acting as research lead, after the improvement loop took **instant → real-time (~24 fps @ 720p)**.
Four parallel Opus experiments, each importing `prototype/` + `server/` **READ-ONLY**, producing a
`REPORT.md` with measured tables, a GO/NO-GO verdict, and a concrete integration proposal that the
lead **seam-verifies** before integrating.

| exp | dir | thread | why now |
|---|---|---|---|
| **E1** | `exp1_progressive/` | Progressive play-while-processing (fMP4/HLS, audio interleave, TTFF) | #1 UX lever — instant now keeps up with playback |
| **E2** | `exp2_highmotion/` | Content-adaptive fallback/anchoring for the high-motion instant weak spot | standing weak spot; 720p safeguard is off |
| **E3** | `exp3_visual/` | V2 motion-modulated grain on static regions + V3 graphic/text-edge pinning | two cheap, high-value visual wins (output-only passes) |
| **E4** | `exp4_untested_levers/` | Color-box clamp (FSR2) + fp16 SR net | the two levers the docs flagged "test, don't assume" |

Methodology (all experiments): honest metrics only (tOF + fallback% + direct |ΔF|; never
LR-consistency or NR-sharpness alone); regression byte-identical (no shared-code edits); timing as
ratios/best-of-N under shared-GPU contention; free MPS between configs; no empty catch blocks.
