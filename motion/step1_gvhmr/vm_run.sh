#!/usr/bin/env bash
# Step 1 on the GPU VM: run GVHMR (static camera) on GVHMR's own example clip (smoke test),
# then on our candidate clips. Writes ~/step1_run.log and ~/step1_run.done (or .failed).
# Results: ~/step1_results/<clip>/ (hmr4d_results.pt + rendered videos).
set -euo pipefail
exec > >(tee -a ~/step1_run.log) 2>&1
trap 'echo "FAILED at line $LINENO"; touch ~/step1_run.failed' ERR

cd ~/GVHMR && source .venv/bin/activate
OUT=~/step1_results
mkdir -p $OUT

echo "== $(date) smoke test on GVHMR example (tennis.mp4)"
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s --output_root=$OUT

for clip in IMG_0532 IMG_0531; do
  echo "== $(date) $clip"
  python tools/demo/demo.py --video=$HOME/videos/$clip.mp4 -s --output_root=$OUT
done

ls -la $OUT/*
echo "== $(date) run done"
touch ~/step1_run.done
