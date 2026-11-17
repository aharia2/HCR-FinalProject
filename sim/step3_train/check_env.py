"""Spawn RPO-BagMimic and print what the simulator actually built (bodies, joints, masses, motion mapping).

    PYTHONPATH=~/hcr/sim/step3_train python ~/hcr/sim/step3_train/check_env.py --task=RPO-BagMimic --num_envs=2 --viz=none
"""
from __future__ import annotations

import argparse
import contextlib
import sys

import torch

import robolab.tasks  # noqa: F401
import bagmimic  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_common_play_args, apply_env_overrides, close_env, create_isaaclab_env, set_hydra_args
from isaaclab.utils.string import list_intersection


def T(x):
    if isinstance(x, torch.Tensor):
        return x
    import warp as wp
    return wp.to_torch(x)


def main(argv):
    p = argparse.ArgumentParser()
    add_common_play_args(p, agent_default="rsl_rl_cfg_entry_point", agent_help="agent cfg entry point")
    cli_args.add_rsl_rl_args(p)
    add_launcher_args(p)
    reg_args = cli_args.register_external_tasks(argv)          # same sequence as robolab play.py
    args, remaining = setup_preset_cli(p, argv)
    set_hydra_args(list_intersection(remaining, reg_args))
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
    # nominal masses: turn off the startup mass randomization for this check only
    env_cfg.events.scale_link_mass = None
    env_cfg.events.add_base_mass = None
    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        apply_env_overrides(args, env_cfg)
        env = create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        U = env.unwrapped
        robot = U.scene["robot"]
        cmd = U.command_manager.get_term("motion")
        print("ROBOT body_names (%d):" % len(robot.body_names), robot.body_names)
        print("ROBOT joint_names (%d):" % len(robot.joint_names), robot.joint_names)
        try:
            m = T(robot.data.default_mass)[0] if hasattr(robot.data, "default_mass") else T(robot.root_view.get_masses())[0]
            print("MASSES:", {b: round(float(v), 4) for b, v in zip(robot.body_names, m)})
            print("TOTAL MASS:", round(float(m.sum()), 3))
        except Exception as e:  # API differences across backends
            print("mass query failed:", repr(e))
        print("MOTION frames:", int(cmd.motion.time_step_total), "| anchor:", cmd.cfg.anchor_body_name,
              "| tracked bodies:", len(cmd.cfg.body_names), "| body_indexes:", cmd.body_indexes.tolist())
        print("soft joint limits (first 3):", T(robot.data.soft_joint_pos_limits)[0, :3].tolist())
        print("CHECK_ENV_OK")


if __name__ == "__main__":
    main(sys.argv[1:])
