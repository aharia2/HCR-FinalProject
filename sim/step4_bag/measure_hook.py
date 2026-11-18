"""Where does the policy's hook actually go? Frame-0 rollouts with the bag moved out of the way.

    PYTHONPATH=~/hcr/sim/step3_train:~/hcr/sim/step4_bag python ~/hcr/sim/step4_bag/measure_hook.py \
        --task=RPO-BagCarry --num_envs=64 --viz=none --checkpoint=<model.pt> --mode=clean|randomized --out=<json>

For every env, records the hook throat (in the motion frame, env origin removed) at the lift frame and at the
release frame, and the reference throat at the same frames. Prints the mean / spread of policy-minus-reference.
Used to place the bag where the trained robot really puts its hook (both are fixed in the real setup).
"""
from __future__ import annotations

import argparse
import contextlib
import json
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

F = bagcarry.F


def main(argv):
    p = argparse.ArgumentParser()
    add_common_play_args(p, agent_default="rsl_rl_cfg_entry_point", agent_help="agent cfg entry point")
    cli_args.add_rsl_rl_args(p)
    add_launcher_args(p)
    p.add_argument("--mode", choices=["clean", "randomized"], default="clean")
    p.add_argument("--out", required=True)
    reg = cli_args.register_external_tasks(argv)
    args, rem = setup_preset_cli(p, argv)
    set_hydra_args(list_intersection(rem, reg))
    ver = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.terminations.bag_dropped = None
    if args.mode == "clean":
        env_cfg.events.randomize_push_robot = None
        env_cfg.observations.policy.enable_corruption = False
    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args), ver)
        apply_env_overrides(args, env_cfg)
        env = create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        U = env.unwrapped
        c = U.command_manager.get_term("motion")

        def start_at_zero(ids):
            c.time_steps[ids] = 0
        c._adaptive_sampling = start_at_zero
        U.reset()
        # move the bag out of the way (the target observation still points at the real pickup spot)
        pose = torch.cat([c.bag_pos_w + torch.tensor([0.0, 30.0, 0.0], device=U.device), c.bag_quat_w], -1)
        c.bag.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.arange(U.num_envs, device=U.device))
        obs = env.get_observations()
        org = U.scene.env_origins
        alive = torch.ones(U.num_envs, dtype=torch.bool, device=U.device)
        rec = {}
        for k in range(F["release"] + 1):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            alive &= ~dones.bool()
            if k + 1 in (F["pickup_min"], F["lift"], F["release"]):   # time step after this env.step
                rec[k + 1] = ((c.robot_throat_w - org).clone(), (c.ref_throat_w - org).clone())
        out = {"mode": args.mode, "checkpoint": args.checkpoint, "alive": int(alive.sum()), "num_envs": U.num_envs}
        for fr, (pol, ref) in rec.items():
            dlt = (pol - ref)[alive]
            out[str(fr)] = {"policy_mean": pol[alive].mean(0).tolist(), "ref_mean": ref[alive].mean(0).tolist(),
                            "offset_mean": dlt.mean(0).tolist(), "offset_std": dlt.std(0).tolist(),
                            "spread_around_mean_p90_m": float((pol[alive] - pol[alive].mean(0)).norm(dim=-1).quantile(0.9))}
        json.dump(out, open(args.out, "w"), indent=2)
        print("RESULT", json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:])
