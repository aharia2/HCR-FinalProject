#!/usr/bin/env bash
# Runs on the VM: stop step 4 training + watchdog, then carry-only evals (bag already on the hook at the lift frame)
# at every mass, clean and randomized. Log: ~/eval_carry.log. Shuts the VM down 20 min after finishing.
set -u
exec >> ~/eval_carry.log 2>&1
for W in $(pgrep -f watchdog.sh); do kill "$W"; done
P=$(cat ~/train_bag.pid); kill "$P"; for i in $(seq 1 30); do kill -0 "$P" 2>/dev/null || break; sleep 1; done
echo "training stopped at: $(grep 'Learning iteration' ~/train_bag.log | tail -1)"
cd ~/roboparty_train && source .venv/bin/activate
CK=$(ls ~/roboparty_train/logs/rsl_rl/rpo_bagcarry/2026-11-18_02-47-13/model_*.pt | sort -V | tail -1); echo "checkpoint $CK"
for spec in clean:0.5 clean:1 clean:1.25 clean:1.5 clean:2 randomized:0.5 randomized:1.25 randomized:2; do
  M=${spec%%:*}; K=${spec##*:}
  PYTHONPATH=~/hcr/sim/step3_train:~/hcr/sim/step4_bag timeout 900 python ~/hcr/sim/step4_bag/eval_bagcarry.py \
    --task=RPO-BagCarry --num_envs=64 --viz=none --checkpoint="$CK" --mass="$K" --mode="$M" --start=lift \
    --out=$HOME/eval_carry > ~/eval_carry_$M$K.log 2>&1
  grep RESULT ~/eval_carry_$M$K.log
done
echo "ALL DONE $(date -u)"
sudo shutdown -c; sudo shutdown -h +20
