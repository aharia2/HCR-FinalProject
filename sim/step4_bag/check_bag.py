"""Spawn RPO-BagCarry with 4 envs, put them at clip frames 0 / lift+40 / 600 / 820, step 25 times
(robot holding the reference joint targets, no policy) and print bag state, to check spawning, phases and physics.

    PYTHONPATH=~/hcr/sim/step3_train:~/hcr/sim/step4_bag python ~/hcr/sim/step4_bag/check_bag.py --task=RPO-BagCarry --num_envs=4 --viz=none
"""
from __future__ import annotations

import argparse
import contextlib
import sys

import torch

import robolab.tasks  # noqa: F401
import bagmimic  # noqa: F401
import bagcarry
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.string import list_intersection
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_common_play_args, apply_env_overrides, close_env, create_isaaclab_env, set_hydra_args


def main(argv):
    p = argparse.ArgumentParser()
    add_common_play_args(p, agent_default="rsl_rl_cfg_entry_point", agent_help="agent cfg entry point")
    cli_args.add_rsl_rl_args(p)
    add_launcher_args(p)
    reg = cli_args.register_external_tasks(argv)
    args, rem = setup_preset_cli(p, argv)
    set_hydra_args(list_intersection(rem, reg))
    env_cfg, _ = resolve_task_config(args.task, args.agent, play_mode=False)
    env_cfg.events.randomize_push_robot = None
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        apply_env_overrides(args, env_cfg)
        env = create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        U = env.unwrapped
        c = U.command_manager.get_term("motion")
        starts = torch.tensor([0, bagcarry.F["lift"] + 40, 600, 820], device=U.device)[: U.num_envs]

        def fixed(ids):
            c.time_steps[ids] = starts[ids]
        c._adaptive_sampling = fixed
        U.reset()
        print("bag masses:", [round(float(m), 3) for m in c.bag.data.default_mass.torch[:, 0]])
        names = ["before lift (floor, start)", "hanging (carry)", "hanging (carry)", "after release (floor, drop)"]
        for k in range(26):
            if k % 5 == 0:
                for i in range(U.num_envs):
                    print(f"step {k:2d} env{i} [{names[i]}] t={int(c.time_steps[i])} bag_h={float(c.bag_height[i]):+.3f} "
                          f"up_z={float(c.bag_up_z[i]):.2f} |throat_robot-bar|={float((c.robot_throat_w[i]-c.bar_w[i]).norm()):.3f} "
                          f"|throat_ref-bar|={float((c.ref_throat_w[i]-c.bar_w[i]).norm()):.3f}")
            act = torch.zeros(U.num_envs, U.action_manager.total_action_dim, device=U.device)
            # hold reference joints: action = (ref - default) / scale
            jp = c.joint_pos
            act[:] = (jp - U.scene["robot"].data.default_joint_pos.torch) / 0.25
            U.step(act)
        print("CHECK_BAG_OK")


if __name__ == "__main__":
    main(sys.argv[1:])
