#!/usr/bin/env bash
# Runs on the VM: stop step 4 v1 (and its shutdown watchdog), start step 4 v2 from the final step 3 checkpoint.
set -u
for W in $(pgrep -f "watchdog.sh"); do kill "$W"; done; echo "watchdogs stopped"
OLD=$(cat ~/train_bag.pid)
kill "$OLD"; for i in $(seq 1 30); do kill -0 "$OLD" 2>/dev/null || break; sleep 1; done
kill -0 "$OLD" 2>/dev/null && { echo "v1 still alive"; exit 1; } || echo "v1 ($OLD) stopped"
mv ~/train_bag.log ~/train_bag_s4v1.log
CK=~/roboparty_train/logs/rsl_rl/rpo_bagmimic_walk/2026-11-17_19-14-00/model_14999.pt
cd ~/roboparty_train && source .venv/bin/activate
PYTHONPATH=~/hcr/sim/step3_train:~/hcr/sim/step4_bag nohup python ~/hcr/sim/step4_bag/train_bagcarry.py \
  --task=RPO-BagCarry --viz=none --logger=tensorboard --num_envs=4096 --max_iterations=5000 --checkpoint="$CK" \
  > ~/train_bag.log 2>&1 &
NEW=$!; echo "$NEW" > ~/train_bag.pid; echo "v2 pid $NEW"
nohup ~/watchdog.sh "$NEW" >/dev/null 2>&1 &
echo "watchdog armed on $NEW"
