#!/usr/bin/env bash
# Runs on the VM: stop the running step 4 run (and its shutdown watchdog), start a new one from the final step 3
# checkpoint. Usage: swap_step4_run.sh [tag]   (v3+: the checkpoint is padded for the extra observation first)
set -u
for W in $(pgrep -f "watchdog.sh"); do kill "$W"; done; echo "watchdogs stopped"
OLD=$(cat ~/train_bag.pid)
kill "$OLD"; for i in $(seq 1 30); do kill -0 "$OLD" 2>/dev/null || break; sleep 1; done
kill -0 "$OLD" 2>/dev/null && { echo "old run still alive"; exit 1; } || echo "old run ($OLD) stopped"
TAG=${1:-v2}
mv ~/train_bag.log ~/train_bag_s4_before_$TAG.log
CK=~/roboparty_train/logs/rsl_rl/rpo_bagmimic_walk/2026-11-17_19-14-00/model_14999.pt
cd ~/roboparty_train && source .venv/bin/activate
if [ "$TAG" != v2 ]; then python ~/hcr/sim/step4_bag/pad_checkpoint.py "$CK" ~/step3_final_padded.pt || exit 1; CK=~/step3_final_padded.pt; fi
PYTHONPATH=~/hcr/sim/step3_train:~/hcr/sim/step4_bag nohup python ~/hcr/sim/step4_bag/train_bagcarry.py \
  --task=RPO-BagCarry --viz=none --logger=tensorboard --num_envs=4096 --max_iterations=5000 --checkpoint="$CK" \
  > ~/train_bag.log 2>&1 &
NEW=$!; echo "$NEW" > ~/train_bag.pid; echo "$TAG pid $NEW"
nohup ~/watchdog.sh "$NEW" >/dev/null 2>&1 &
echo "watchdog armed on $NEW"
