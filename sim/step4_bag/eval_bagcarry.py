"""Gate-4 evaluation: full clip from frame 0 with a FIXED bag mass.

    PYTHONPATH=~/hcr/sim/step3_train:~/hcr/sim/step4_bag python ~/hcr/sim/step4_bag/eval_bagcarry.py \
        --task=RPO-BagCarry --num_envs=64 --viz=none --checkpoint=<model.pt> --mass=1.0 --mode=clean|randomized --out=<dir>

Per env: fell (non-timeout termination other than bag_dropped is still counted as a failure), max bag height in
the carry window, bag hooked (handle within 3 cm of the throat at mid-carry), final bag state (upright, on the
floor, distance to the set-down spot). Success = no fall AND lifted >= 10 cm AND placed upright within a 0.4 m
square (|dx|,|dy| <= 0.2 m) AND robot standing at the end. Saves env 0's robot + bag trajectory for rendering.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import numpy as np
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
    p.add_argument("--mass", type=float, required=True)
    p.add_argument("--out", required=True)
    reg = cli_args.register_external_tasks(argv)
    args, rem = setup_preset_cli(p, argv)
    set_hydra_args(list_intersection(rem, reg))
    ver = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.events.bag_mass.params["mass_distribution_params"] = (args.mass, args.mass)
    env_cfg.terminations.bag_dropped = None          # keep running after a drop so the failure is visible
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
        robot = U.scene["robot"]

        def start_at_zero(ids):
            c.time_steps[ids] = 0
        c._adaptive_sampling = start_at_zero
        U.reset()
        obs = env.get_observations()
        N, T = U.num_envs, int(c.motion.time_step_total)
        org = U.scene.env_origins
        alive = torch.ones(N, dtype=torch.bool, device=U.device)
        fell_at = torch.full((N,), -1, dtype=torch.long, device=U.device)
        max_h = torch.zeros(N, device=U.device)
        hooked_mid = torch.zeros(N, dtype=torch.bool, device=U.device)
        mid = (F["lift"] + F["place_min"]) // 2
        traj = {k: [] for k in ("t", "root_pos", "root_quat_wxyz", "joint_pos", "bag_pos", "bag_quat_wxyz",
                                "bag_height", "hook_bar_dist")}
        final = None
        for k in range(T - 2):                       # stop before the clip end resets the episode
            with torch.inference_mode():
                obs, _, dones, extras = env.step(policy(obs))
            to = extras.get("time_outs", torch.zeros_like(dones)).bool()
            fell = dones.bool() & ~to & alive
            fell_at[fell] = k
            alive &= ~dones.bool()
            h = c.bag_height
            if F["lift"] + 20 <= k < F["place_min"] - 20:
                h_up = torch.where(c.bag_up_z > 0.8, h, torch.zeros_like(h))     # tipped bags do not count
                max_h = torch.where(alive, torch.maximum(max_h, h_up), max_h)
            if k == mid:
                hooked_mid = alive & ((c.robot_throat_w - c.bar_w).norm(dim=-1) < 0.03)
            q = robot.data.root_quat_w.torch[0].cpu().numpy(); bq = c.bag_quat_w[0].cpu().numpy()
            traj["t"].append(k * U.step_dt)
            traj["root_pos"].append((robot.data.root_pos_w.torch[0] - org[0]).cpu().numpy())
            traj["root_quat_wxyz"].append(q[[3, 0, 1, 2]])
            traj["joint_pos"].append(robot.data.joint_pos.torch[0].cpu().numpy())
            traj["bag_pos"].append((c.bag_pos_w[0] - org[0]).cpu().numpy())
            traj["bag_quat_wxyz"].append(bq[[3, 0, 1, 2]])
            traj["bag_height"].append(float(h[0]))
            traj["hook_bar_dist"].append(float((c.robot_throat_w[0] - c.bar_w[0]).norm()))
        dxy = c.bag_pos_w[:, :2] - org[:, :2] - c.drop_pos[:2]
        placed = (c.bag_up_z > 0.9) & (c.bag_height < 0.03) & (dxy.abs() <= 0.2).all(-1)
        standing = robot.data.root_pos_w.torch[:, 2] - org[:, 2] > 0.5
        lifted = max_h >= 0.10
        success = alive & lifted & placed & standing
        f = lambda x: f"{int(x.sum())}/{N}"
        res = dict(mode=args.mode, mass_kg=args.mass, checkpoint=args.checkpoint, num_envs=N,
                   success=f(success), no_fall=f(alive), hooked_mid_carry=f(hooked_mid), lifted_10cm=f(lifted),
                   placed_upright_in_square=f(placed), standing_at_end=f(standing),
                   max_bag_height_m=dict(median=round(float(max_h.median()), 3), p10=round(float(max_h.quantile(0.1)), 3)),
                   drop_dist_m_median=round(float(dxy.norm(dim=-1).median()), 3),
                   fall_times_s=sorted(round(float(x) * U.step_dt, 2) for x in fell_at[fell_at >= 0].tolist())[:30])
        os.makedirs(args.out, exist_ok=True)
        tag = f"{args.mode}_{args.mass:g}kg"
        json.dump(res, open(os.path.join(args.out, f"eval_{tag}.json"), "w"), indent=2)
        np.savez_compressed(os.path.join(args.out, f"traj_env0_{tag}.npz"), joint_names=np.array(robot.joint_names),
                            **{k: np.array(v) for k, v in traj.items()})
        print("RESULT", json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1:])
