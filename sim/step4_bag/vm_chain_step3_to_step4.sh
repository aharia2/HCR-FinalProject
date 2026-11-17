#!/usr/bin/env bash
# Runs on the VM. Waits for step 3 training to exit, evaluates its final checkpoint (clean + randomized),
# then starts step 4 (RPO-BagCarry) resuming from it, and re-arms the shutdown watchdog on the step 4 process.
# Log: ~/chain.log
set -u
exec >> ~/chain.log 2>&1
S3=$(cat ~/train_bag.pid)
echo "$(date -u) waiting for step 3 pid $S3"
while kill -0 "$S3" 2>/dev/null; do sleep 60; done
echo "$(date -u) step 3 exited"
cd ~/roboparty_train && source .venv/bin/activate
CK=$(ls ~/roboparty_train/logs/rsl_rl/rpo_bagmimic_walk/*/model_*.pt | sort -V | tail -1)
echo "final step 3 checkpoint: $CK"
for M in clean randomized; do
  PYTHONPATH=~/hcr/sim/step3_train timeout 1500 python ~/hcr/sim/step3_train/eval_bagmimic.py --task=RPO-BagMimic \
    --num_envs=64 --viz=none --checkpoint="$CK" --mode=$M --out=$HOME/eval_run2_final > ~/eval_run2_final_$M.log 2>&1
  echo "step 3 final eval $M exit $?"
done
mv ~/train_bag.log ~/train_bag_run2.log
PYTHONPATH=~/hcr/sim/step3_train:~/hcr/sim/step4_bag nohup python ~/hcr/sim/step4_bag/train_bagcarry.py \
  --task=RPO-BagCarry --viz=none --logger=tensorboard --num_envs=4096 --max_iterations=5000 --checkpoint="$CK" \
  > ~/train_bag.log 2>&1 &
S4=$!; echo "$S4" > ~/train_bag.pid
echo "$(date -u) step 4 started, pid $S4, resumed from $CK"
nohup ~/watchdog.sh "$S4" >/dev/null 2>&1 &
echo "watchdog armed on $S4"
