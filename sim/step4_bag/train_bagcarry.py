"""robolab's train.py + our tasks (robolab not edited). Step 4 resumes from the final step 3 checkpoint:

    cd ~/roboparty_train && source .venv/bin/activate
    PYTHONPATH=~/hcr/sim/step3_train:~/hcr/sim/step4_bag python ~/hcr/sim/step4_bag/train_bagcarry.py \
        --task=RPO-BagCarry --viz=none --logger=tensorboard --num_envs=4096 --max_iterations=N --checkpoint=<step3 model.pt>
"""
import sys

import robolab.tasks  # noqa: F401
import bagmimic  # noqa: F401
import bagcarry  # noqa: F401
from isaaclab_rl.entrypoints.backends.train_rsl_rl import run

if __name__ == "__main__":
    run(sys.argv[1:])
