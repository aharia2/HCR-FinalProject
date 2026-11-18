#!/usr/bin/env bash
# Runs on the VM: wait for the step 4 training PID to exit, stop the shutdown watchdog, then run the gate-4 test
# matrix on the last checkpoint (masses x clean/randomized x start zero/lift) and shut down 15 min later.
# Log: ~/eval_final.log, results: ~/eval_final/
set -u
exec >> ~/eval_final.log 2>&1
P=$(cat ~/train_bag.pid); echo "$(date -u) waiting for training pid $P"
while kill -0 "$P" 2>/dev/null; do sleep 30; done
for W in $(pgrep -f watchdog.sh); do kill "$W"; done
CK=$(ls -t ~/roboparty_train/logs/rsl_rl/rpo_bagcarry/*/model_*.pt | head -1); echo "$(date -u) checkpoint $CK"
cd ~/roboparty_train && source .venv/bin/activate
for S in zero lift; do for M in clean randomized; do for K in 0.5 1 1.25 1.5 2; do
  PYTHONPATH=~/hcr/sim/step3_train:~/hcr/sim/step4_bag timeout 900 python ~/hcr/sim/step4_bag/eval_bagcarry.py \
    --task=RPO-BagCarry --num_envs=64 --viz=none --checkpoint="$CK" --mass="$K" --mode="$M" --start="$S" \
    --out=$HOME/eval_final > ~/eval_final_run.log 2>&1
  echo "$S $M $K: $(grep RESULT ~/eval_final_run.log | cut -c1-60)"
done; done; done
echo "ALL DONE $(date -u)"
sudo shutdown -c; sudo shutdown -h +15
