#!/usr/bin/env bash
# Step 3 setup on the GPU VM: Isaac Lab 3.0.0 + roboparty_train (pinned) + our task.
# Expects ~/hcr/ uploaded from this repo: robot/, sim/, motion/step2_gmr/results/IMG_0531/bag_pickup_IMG_0531.npz
# Writes ~/step3_setup.log and ~/step3_setup.done (or .failed).
set -euo pipefail
exec > >(tee -a ~/step3_setup.log) 2>&1
trap 'echo "FAILED at line $LINENO"; touch ~/step3_setup.failed' ERR

TRAIN_COMMIT=53b411a0790fe172f717df773122bbc4ce190757
ROBOLAB_COMMIT=45cdc13a3465449315552af3dae09fa23f7083f5
RSLRL_COMMIT=9fb0cc653bc306c5c19288d0f72a8bfbfe459a74
RPODESC_COMMIT=8e0f8f78903dc147a5c094c0fdc14b4a78f1b667

echo "== $(date) system packages"
sudo apt-get -qq update && sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y git git-lfs build-essential tmux ffmpeg >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== Isaac Lab v3.0.0"
[ -d ~/IsaacLab-3.0.0 ] || git clone -q --branch v3.0.0 --recursive https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab-3.0.0

echo "== roboparty_train @ ${TRAIN_COMMIT:0:7}"
[ -d ~/roboparty_train ] || git clone -q https://github.com/Roboparty/roboparty_train.git ~/roboparty_train
cd ~/roboparty_train
git fetch -q
git checkout -q $TRAIN_COMMIT
git submodule update -q --init --recursive
( cd robolab && git checkout -q $ROBOLAB_COMMIT && git submodule update -q --init --recursive )
( cd rsl_rl && git checkout -q $RSLRL_COMMIT )
( cd robolab/data/robots/roboparty/rpo && git checkout -q $RPODESC_COMMIT )
echo "pins: train $(git rev-parse --short HEAD) robolab $(git -C robolab rev-parse --short HEAD) rsl_rl $(git -C rsl_rl rev-parse --short HEAD) rpo_description $(git -C robolab/data/robots/roboparty/rpo rev-parse --short HEAD)"

echo "== Python env (MJWarp backend, default group) + video extra"
uv sync --locked --extra video
source .venv/bin/activate

echo "== hooked robot into robolab's own rpo folders (so ../meshes paths resolve like the stock URDF)"
RPO=robolab/data/robots/roboparty/rpo
cp ~/hcr/robot/urdf/rpo_hook.urdf $RPO/urdf/
cp ~/hcr/robot/meshes/right_hook_link.STL ~/hcr/robot/meshes/right_hook_collision_*.STL $RPO/meshes/
cmp $RPO/urdf/rpo.urdf ~/hcr/robot/urdf/rpo.urdf && echo "stock rpo.urdf identical to ours"

echo "== our motion"
mkdir -p robolab/data/motions/rpo_bm
cp ~/hcr/motion/bag_pickup_IMG_0531.npz robolab/data/motions/rpo_bm/

echo "== env list"
cd ~/roboparty_train && PYTHONPATH=~/hcr/sim/step3_train python -c "
import robolab.tasks, bagmimic, gymnasium as gym
print([e for e in gym.registry if e.startswith('RPO-')])"

echo "== $(date) setup done"
touch ~/step3_setup.done
