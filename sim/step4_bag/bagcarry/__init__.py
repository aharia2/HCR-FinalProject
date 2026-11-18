"""RPO-BagCarry (step 4): RPO-BagMimic (step 3) + a bag with a handle + bag-mass randomization + task rewards.

Policy and critic observations are UNCHANGED from step 3 (proprioception + motion command only, no bag state),
so training resumes from the step 3 checkpoint and the policy stays deployable with encoders + IMU only.
The policy must hook the bag by tracking the motion precisely and feel the bag through its joints.

Bag phases follow the reference time step (50 Hz, frames from bag_placement.json):
  t <  lift     bag stands at the pickup spot          t in [lift, release)  bag hangs from the reference hook
  t >= release  bag stands at the set-down spot
Each episode reset (BeyondMimic starts episodes at sampled points of the clip) puts the bag in the matching state.
"""
import json
import math
import os

import gymnasium as gym
import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
import isaaclab.sim as sim_utils

from bagmimic import RPOBagMimicEnvCfg, RPOBagMimicPPORunnerCfg
from robolab.tasks.manager_based.beyondmimic import mdp
from robolab.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand, MotionCommandCfg

HERE = os.path.dirname(os.path.abspath(__file__))
PLACE = json.load(open(os.path.join(HERE, "..", "bag_placement.json")))
BAG_URDF = os.path.join(HERE, "..", "assets", "bag.urdf")
F = PLACE["frames"]
BAR_Z = PLACE["geometry"]["bar_z"]
THROAT_IN_FOREARM = PLACE["hook"]["throat_in_forearm"]
MASS_RANGE = (0.5, 2.0)


def yaw_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:          # Isaac Lab 3.0 quaternions are (x, y, z, w)
    z = torch.zeros_like(yaw)
    return torch.stack([z, z, torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)


class BagMotionCommand(MotionCommand):
    """MotionCommand that also places the bag to match the sampled start time, and exposes bag/hook quantities."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.bag = env.scene["bag"]
        self.forearm_ref = self.cfg.body_names.index("right_elbow_yaw_link")
        self.forearm_robot = self.robot.body_names.index("right_elbow_yaw_link")
        dev = self.device
        self.throat_local = torch.tensor(THROAT_IN_FOREARM, device=dev)
        self.start_pos = torch.tensor(PLACE["start"]["pos"], device=dev)
        self.start_yaw = torch.tensor(PLACE["start"]["yaw"], device=dev)
        self.drop_pos = torch.tensor(PLACE["drop"]["pos"], device=dev)
        self.drop_yaw = torch.tensor(PLACE["drop"]["yaw"], device=dev)
        self.bar_local = torch.tensor([0.0, 0.0, BAR_Z], device=dev)
        self.was_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)   # bag carried this episode

    # --- quantities used by rewards / terminations / eval ---
    @property
    def robot_throat_w(self):
        p = self.robot.data.body_pos_w.torch[:, self.forearm_robot]
        q = self.robot.data.body_quat_w.torch[:, self.forearm_robot]
        return p + quat_apply(q, self.throat_local.expand(self.num_envs, 3))

    @property
    def ref_throat_w(self):
        p, q = self.body_pos_w[:, self.forearm_ref], self.body_quat_w[:, self.forearm_ref]
        return p + quat_apply(q, self.throat_local.expand(self.num_envs, 3))

    @property
    def bag_pos_w(self):
        return self.bag.data.root_pos_w.torch

    @property
    def bag_quat_w(self):
        return self.bag.data.root_quat_w.torch

    @property
    def bar_w(self):
        return self.bag_pos_w + quat_apply(self.bag_quat_w, self.bar_local.expand(self.num_envs, 3))

    @property
    def bag_up_z(self):
        up = torch.zeros(self.num_envs, 3, device=self.device); up[:, 2] = 1
        return quat_apply(self.bag_quat_w, up)[:, 2]

    @property
    def bag_height(self):            # bag origin = bottom centre; 0 when standing on the floor
        return self.bag_pos_w[:, 2] - self._env.scene.env_origins[:, 2]

    # --- reset ---
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids) == 0:
            return
        t = self.time_steps[env_ids]
        org = self._env.scene.env_origins[env_ids]
        n = len(env_ids)
        pos = torch.zeros(n, 3, device=self.device); yaw = torch.zeros(n, device=self.device)
        before, after = t < F["lift"], t >= F["release"]
        hang = ~before & ~after
        pos[before] = self.start_pos + org[before]; yaw[before] = self.start_yaw
        pos[after] = self.drop_pos + org[after]; yaw[after] = self.drop_yaw
        self.was_lifted[env_ids] = hang
        if hang.any():
            # Hang the bag on the ROBOT's hook. The robot was just reset to the reference pose PLUS start noise
            # (root ±5 cm / small rotations, joints ±0.1 rad); super() refreshed forward kinematics, so the
            # robot's body poses are current. (v1 used the reference hook here, so most hanging bags spawned
            # a few cm off the real hook and fell at once.)
            ids = env_ids[hang]
            p = self.robot.data.body_pos_w.torch[ids, self.forearm_robot]
            q = self.robot.data.body_quat_w.torch[ids, self.forearm_robot]
            throat = p + quat_apply(q, self.throat_local.expand(len(ids), 3))
            hy = quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(len(ids), 3))
            yaw[hang] = torch.atan2(hy[:, 1], hy[:, 0])
            pos[hang] = throat - torch.tensor([0.0, 0.0, BAR_Z], device=self.device)
        pos[:, 2] += 0.002                                     # spawn just above the floor
        self.bag.write_root_pose_to_sim_index(root_pose=torch.cat([pos, yaw_quat_xyzw(yaw)], -1), env_ids=env_ids)
        self.bag.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids)


@configclass
class BagMotionCommandCfg(MotionCommandCfg):
    class_type: type = BagMotionCommand


# --- rewards / terminations (all read the command term) ---
def _cmd(env) -> BagMotionCommand:
    return env.command_manager.get_term("motion")


def _window(cmd, a, b):
    return ((cmd.time_steps >= a) & (cmd.time_steps < b)).float()


def hook_to_handle(env, std: float, start: int = F["lift"] - 60, end: int = F["release"]):
    c = _cmd(env)
    d2 = ((c.robot_throat_w - c.bar_w) ** 2).sum(-1)
    return torch.exp(-d2 / std ** 2) * _window(c, start, end)


def bag_lifted(env, height: float):
    c = _cmd(env)
    return (c.bag_height / height).clamp(0, 1) * _window(c, F["lift"] + 20, F["place_min"] - 20)


def bag_placed(env, std: float):
    c = _cmd(env)
    xy = c.bag_pos_w[:, :2] - env.scene.env_origins[:, :2] - c.drop_pos[:2]
    ok = (c.bag_up_z > 0.9) & (c.bag_height < 0.03)
    return torch.exp(-(xy ** 2).sum(-1) / std ** 2) * ok.float() * _window(c, F["release"] + 25, 10 ** 9)


def bag_knocked_over(env):
    c = _cmd(env)
    return (c.bag_up_z < 0.7).float() * _window(c, 0, F["lift"])


def bag_dropped(env):
    """Bag back on the floor mid-carry AFTER having been carried this episode. Episodes that never hooked the
    bag keep running (tracking rewards, no lift reward) instead of all ending at lift + 1 s."""
    c = _cmd(env)
    c.was_lifted |= c.bag_height > 0.05
    return c.was_lifted & (c.bag_height < 0.03) & (_window(c, F["lift"] + 50, F["place_min"] - 25) > 0)


@configclass
class RPOBagCarryEnvCfg(RPOBagMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Isaac Lab 3.0's URDF importer always builds an articulation, so the bag is a 0-joint articulation
        # (RigidObjectCfg with a URDF spawn fails: "No articulations matching .../Bag/Geometry/bag").
        self.scene.bag = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Bag",
            spawn=sim_utils.UrdfFileCfg(asset_path=BAG_URDF, fix_base=False, activate_contact_sensors=False,
                                        joint_drive=None),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), joint_pos={}, joint_vel={}),
            actuators={},
        )
        old = self.commands.motion
        self.commands.motion = BagMotionCommandCfg(**{k: getattr(old, k) for k in old.__dataclass_fields__
                                                     if k != "class_type"})
        self.events.bag_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("bag"), "mass_distribution_params": MASS_RANGE, "operation": "abs"})
        self.rewards.hook_to_handle = RewTerm(func=hook_to_handle, weight=1.0, params={"std": 0.05})
        # coarse guidance on the approach: the 5 cm kernel gives no gradient from 25+ cm away
        self.rewards.hook_to_handle_coarse = RewTerm(func=hook_to_handle, weight=1.0, params={
            "std": 0.25, "start": F["lift"] - 100, "end": F["lift"] + 20})
        self.rewards.bag_lifted = RewTerm(func=bag_lifted, weight=2.0, params={"height": 0.15})
        self.rewards.bag_placed = RewTerm(func=bag_placed, weight=1.0, params={"std": 0.2})
        self.rewards.bag_knocked_over = RewTerm(func=bag_knocked_over, weight=-1.0)
        self.terminations.bag_dropped = DoneTerm(func=bag_dropped)


@configclass
class RPOBagCarryPPORunnerCfg(RPOBagMimicPPORunnerCfg):
    experiment_name = "rpo_bagcarry"


if "RPO-BagCarry" not in gym.registry:
    gym.register(
        id="RPO-BagCarry",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}:RPOBagCarryEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}:RPOBagCarryPPORunnerCfg"},
    )
