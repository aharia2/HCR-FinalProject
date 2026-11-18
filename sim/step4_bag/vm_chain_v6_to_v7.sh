#!/usr/bin/env bash
# Runs on the VM: wait for v6 to exit, then resume from its last checkpoint with the v7 task (4000 more iterations).
set -u
exec >> ~/chain_v7.log 2>&1
P=$(cat ~/train_bag.pid); echo "$(date -u) waiting for v6 pid $P"
while kill -0 "$P" 2>/dev/null; do sleep 30; done
CK=$(ls ~/roboparty_train/logs/rsl_rl/rpo_bagcarry/2026-11-18_06-38-38/model_*.pt | sort -V | tail -1)
echo "$(date -u) v6 exited, last checkpoint $CK"
mkdir -p ~/hcr/sim/step4_bag/bagcarry && cp ~/v7_bagcarry_init.py ~/hcr/sim/step4_bag/bagcarry/__init__.py
echo 0 > ~/train_bag.pid
RESUME="$CK" ITERS=4000 bash ~/swap_step4_run.sh v7
sudo shutdown -c; sudo shutdown -h +270
