#!/usr/bin/env bash
set -u
OLD=$(cat ~/train_bag.pid)
kill "$OLD"; for i in $(seq 1 30); do kill -0 "$OLD" 2>/dev/null || break; sleep 1; done
kill -0 "$OLD" 2>/dev/null && { echo "run1 still alive"; exit 1; } || echo "run1 ($OLD) stopped"
echo "run1 checkpoints: $(ls ~/roboparty_train/logs/rsl_rl/rpo_bagmimic/*/model_*.pt | wc -l)"
mv ~/train_bag.log ~/train_bag_run1.log
cd ~/roboparty_train && source .venv/bin/activate
PYTHONPATH=~/hcr/sim/step3_train nohup python ~/hcr/sim/step3_train/train_bagmimic.py --task=RPO-BagMimic \
  --viz=none --logger=tensorboard --num_envs=4096 --max_iterations=15000 > ~/train_bag.log 2>&1 &
NEW=$!; echo "$NEW" > ~/train_bag.pid; echo "run2 pid $NEW"
nohup ~/watchdog.sh "$NEW" >/dev/null 2>&1 &
echo "watchdog armed on $NEW"
sudo shutdown -c; sudo shutdown -h +480 2>&1 | tail -1
