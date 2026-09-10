set -e
SCR=/private/tmp/claude-501/-Users-lifeart-Repos-playhd/31ea3233-a1d0-4284-81d1-54f556b10b5b/scratchpad/e5
cd /Users/lifeart/Repos/playhd/experiments/r13_e1_browser_confidence
D="deno run --unstable-webgpu --allow-read --allow-write --allow-env harness.ts"
for W in calm jelly; do
  $D --data $SCR/$W --mode ref --out $SCR/out_${W}_ref
  $D --data $SCR/$W --mode hard --reanchor 4 --out $SCR/out_${W}_hard4
  $D --data $SCR/$W --mode soft --reanchor 4 --tau 16 --s 3 --out $SCR/out_${W}_soft_t16s3
  $D --data $SCR/$W --mode softblk --reanchor 4 --tau 16 --s 3 --kp 6 --km 6 --out $SCR/out_${W}_softblk_kp6km6
  $D --data $SCR/$W --mode softblk --reanchor 4 --tau 16 --s 3 --kp 6 --km 0 --out $SCR/out_${W}_blk_kp6km0
  $D --data $SCR/$W --mode softblk --reanchor 4 --tau 16 --s 3 --kp 0 --km 6 --out $SCR/out_${W}_blk_kp0km6
done
echo ALLDONE
