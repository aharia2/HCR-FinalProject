#!/usr/bin/env bash
# Runs on the VM: how much does localization error cost? Evaluate one checkpoint from frame 0 (clean, 1 kg) with
# the per-episode odometry error (HCR_ODO_DRIFT, uniform ±d in x and y) at 0 / 3 / 6 cm. Log ~/drift_test.log.
# Usage: CK=<model.pt> bash ~/vm_drift_test.sh
set -u
exec >> ~/drift_test.log 2>&1
cd ~/roboparty_train && source .venv/bin/activate
for D in 0 0.03 0.06; do
  HCR_ODO_DRIFT=$D PYTHONPATH=~/hcr/sim/step3_train:~/hcr/sim/step4_bag timeout 1200 python ~/hcr/sim/step4_bag/eval_bagcarry.py \
    --task=RPO-BagCarry --num_envs=128 --viz=none --checkpoint="$CK" --mass=1 --mode=clean --start=zero \
    --out=$HOME/drift_test/d$D > ~/drift_run.log 2>&1
  echo "drift $D: $(grep RESULT ~/drift_run.log)"
done
echo "ALL DONE $(date -u)"
sudo shutdown -h +10
