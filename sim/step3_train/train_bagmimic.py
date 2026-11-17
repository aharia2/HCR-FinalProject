"""robolab's scripts/rsl_rl/train.py plus our task registration (robolab is not edited).

    cd ~/roboparty_train && source .venv/bin/activate
    PYTHONPATH=~/hcr/sim/step3_train python ~/hcr/sim/step3_train/train_bagmimic.py \
        --task=RPO-BagMimic --viz=none --logger=tensorboard --num_envs=4096
All arguments are passed through to Isaac Lab's RSL-RL entry point, exactly as robolab's train.py does.
"""
import sys

import robolab.tasks  # noqa: F401  (registers the stock RPO-* tasks)
import bagmimic  # noqa: F401  (registers RPO-BagMimic)
from isaaclab_rl.entrypoints.backends.train_rsl_rl import run

if __name__ == "__main__":
    run(sys.argv[1:])
