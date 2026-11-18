#!/usr/bin/env bash
# Runs on the VM (after the files below are uploaded): fine-tune step 4 for the user's paper bag (HCR_BAG=wf:
# 30 x 17 x 35 cm, 9 x 9 cm handle) from a step 4 checkpoint, then run the test matrix and shut down.
#   Upload first: sim/step4_bag/{bag_profiles.py,bag_placement_wf.json,assets/bag_wf.urdf,bagcarry/__init__.py}
#   -> ~/hcr/sim/step4_bag/..., bag_pickup_IMG_0531_hook_wf.npz -> ~/roboparty_train/robolab/data/motions/rpo_bm/
# Usage: RESUME=<model.pt> ITERS=3000 bash ~/vm_start_wf.sh
set -u
export HCR_BAG=wf
: "${RESUME:?set RESUME to the starting checkpoint}"
test -f ~/roboparty_train/robolab/data/motions/rpo_bm/bag_pickup_IMG_0531_hook_wf.npz || { echo "motion missing"; exit 1; }
[ -d ~/eval_final ] && mv ~/eval_final ~/eval_final_prev_$(date +%s) && mv ~/eval_final.log ~/eval_final_prev.log
RESUME=$RESUME ITERS=${ITERS:-3000} bash ~/swap_step4_run.sh wf
sleep 2; nohup ~/vm_eval_final.sh >/dev/null 2>&1 &
sudo shutdown -h +$(( ${ITERS:-3000} * 2 / 60 + 60 ))
