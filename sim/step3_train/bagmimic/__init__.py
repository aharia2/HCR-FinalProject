"""RPO-BagMimic: Roboparty's RPO-BeyondMimic task with the hooked robot and our bag-pickup motion.

Robolab itself is not edited. Importing this module registers the gym ids:
  RPO-BagMimic       step 3: motion tracking only (no bag yet)
Usage (on the VM, from ~/roboparty_train with its venv):
  PYTHONPATH=~/hcr/sim/step3_train python ~/hcr/sim/step3_train/train_bagmimic.py --task=RPO-BagMimic ...
"""
import os

import gymnasium as gym
from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots import RPO_CFG
from robolab.tasks.manager_based.beyondmimic.agents.rpo_beyondmimic_agent_cfg import RPOBeyondMimicPPORunnerCfg
from robolab.tasks.manager_based.beyondmimic.rpo_beyondmimic_env_cfg import RPOBeyondMimicEnvCfg

HOOK_URDF = os.path.join(ROBOLAB_ROOT_DIR, "data", "robots", "roboparty", "rpo", "urdf", "rpo_hook.urdf")
MOTION = os.path.join(ROBOLAB_ROOT_DIR, "data", "motions", "rpo_bm", "bag_pickup_IMG_0531.npz")
CLIP_S = 17.0  # 850 frames @ 50 Hz


@configclass
class RPOBagMimicEnvCfg(RPOBeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # same robot config (actuators, gains, limits, soft factor 0.9, self-collisions) but the hooked URDF
        self.scene.robot = RPO_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot", spawn=RPO_CFG.spawn.replace(asset_path=HOOK_URDF))
        self.commands.motion.motion_file = MOTION
        # stock task uses 20 s episodes for its clips; keep >= clip length so a full run fits in one episode
        self.episode_length_s = max(self.episode_length_s, CLIP_S + 1.0)


@configclass
class RPOBagMimicPPORunnerCfg(RPOBeyondMimicPPORunnerCfg):
    experiment_name = "rpo_bagmimic"
    wandb_project = "rpo_bagmimic"


def register():
    if "RPO-BagMimic" in gym.registry:
        return
    gym.register(
        id="RPO-BagMimic",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:RPOBagMimicEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}:RPOBagMimicPPORunnerCfg",
        },
    )


register()
