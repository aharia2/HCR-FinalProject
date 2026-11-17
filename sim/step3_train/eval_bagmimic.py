"""Gate-3 evaluation: full-clip rollouts from frame 0 (not training statistics).

    cd ~/roboparty_train && source .venv/bin/activate
    PYTHONPATH=~/hcr/sim/step3_train python ~/hcr/sim/step3_train/eval_bagmimic.py \
        --task=RPO-BagMimic --num_envs=256 --viz=none --checkpoint=<model_XXXX.pt> --mode=clean|randomized --out=<dir>

Same environment/policy loading as robolab's play.py. Differences:
  * every episode starts at motion frame 0 (adaptive start sampling replaced), no initial pose/velocity/joint noise
  * clean mode: interval pushes off. randomized mode: training randomization (startup + pushes) left on
  * runs one full clip (850 steps) + 1 s hold; records falls (non-timeout terminations), per-body position error,
    hook-throat error at pickup/set-down, and the full root/joint trajectory of env 0 for local rendering.
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
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.common import add_common_play_args, apply_env_overrides, close_env, create_isaaclab_env, set_hydra_args
from isaaclab.utils.string import list_intersection
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

THROAT_IN_FOREARM = (0.13275 + 0.060, 0.0, -0.020)   # hook mount offset + handle seat, in right_elbow_yaw_link
PICK_T, PLACE_T = 5.08, 13.30                          # from Gate 2 (hook throat lowest points)


def T(x):
    """Isaac Lab 3.0 data may be warp arrays; return a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x
    import warp as wp
    return wp.to_torch(x)


def quat_apply_wxyz(q, v):
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def main(argv):
    p = argparse.ArgumentParser()
    add_common_play_args(p, agent_default="rsl_rl_cfg_entry_point", agent_help="agent cfg entry point")
    cli_args.add_rsl_rl_args(p)
    add_launcher_args(p)
    p.add_argument("--mode", choices=["clean", "randomized"], default="clean")
    p.add_argument("--out", required=True)
    reg_args = cli_args.register_external_tasks(argv)          # same sequence as robolab play.py
    args, remaining = setup_preset_cli(p, argv)
    set_hydra_args(list_intersection(remaining, reg_args))
    ver = check_rsl_rl_version()
    env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
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
        cmd = U.command_manager.get_term("motion")
        robot = U.scene["robot"]

        def start_at_zero(env_ids):           # replaces adaptive start sampling
            cmd.time_steps[env_ids] = 0
        cmd._adaptive_sampling = start_at_zero
        U.reset()
        obs = env.get_observations()

        N, steps = U.num_envs, int(cmd.motion.time_step_total) + 50
        fa = cmd.cfg.body_names.index("right_elbow_yaw_link")
        alive = torch.ones(N, dtype=torch.bool, device=U.device)
        fell_at = torch.full((N,), -1, dtype=torch.long, device=U.device)
        body_err_sum = torch.zeros(len(cmd.cfg.body_names), device=U.device)
        count = 0
        hook_err = {"pickup": [], "place": []}
        traj = dict(t=[], root_pos=[], root_quat_wxyz=[], joint_pos=[], ref_root_pos=[], hook_err=[], body_err=[])
        throat = torch.tensor(THROAT_IN_FOREARM, device=U.device)
        for k in range(steps):
            with torch.inference_mode():
                obs, _, dones, extras = env.step(policy(obs))
            ts = cmd.time_steps.clone()
            rb_pos = T(robot.data.body_pos_w)[:, cmd.body_indexes]
            rb_quat = T(robot.data.body_quat_w)[:, cmd.body_indexes]
            ref_pos, ref_quat = cmd.body_pos_w, cmd.body_quat_w
            err = (rb_pos - ref_pos).norm(dim=-1)                      # global position error per body
            r_th = rb_pos[:, fa] + quat_apply_wxyz(rb_quat[:, fa], throat.expand(N, 3))
            f_th = ref_pos[:, fa] + quat_apply_wxyz(ref_quat[:, fa], throat.expand(N, 3))
            th_err = (r_th - f_th).norm(dim=-1)
            timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
            fell = dones.bool() & ~timeouts & alive
            fell_at[fell] = k
            alive &= ~dones.bool() | timeouts
            if alive.any():
                body_err_sum += err[alive].sum(0)
                count += int(alive.sum())
            for name, t_ref in (("pickup", PICK_T), ("place", PLACE_T)):
                if k == int(t_ref * 50):
                    hook_err[name] = th_err[alive].tolist()
            traj["t"].append(k * U.step_dt)
            traj["root_pos"].append(T(robot.data.root_pos_w)[0].cpu().numpy() - U.scene.env_origins[0].cpu().numpy())
            traj["root_quat_wxyz"].append(T(robot.data.root_quat_w)[0].cpu().numpy())
            traj["joint_pos"].append(T(robot.data.joint_pos)[0].cpu().numpy())
            traj["ref_root_pos"].append((cmd.body_pos_w[0, 0] - U.scene.env_origins[0]).cpu().numpy())
            traj["hook_err"].append(float(th_err[0]))
            traj["body_err"].append(err[0].cpu().numpy())

        n_fell = int((fell_at >= 0).sum())
        res = dict(
            mode=args.mode, checkpoint=args.checkpoint, num_envs=N, steps=steps,
            completed_without_fall=f"{N - n_fell}/{N}", completion_rate=round((N - n_fell) / N, 4),
            fall_times_s=sorted(round(float(x) * U.step_dt, 2) for x in fell_at[fell_at >= 0].tolist())[:50],
            mean_body_pos_err_m={b: round(float(e), 4) for b, e in zip(cmd.cfg.body_names, body_err_sum / max(count, 1))},
            hook_throat_err_m={k: dict(median=round(float(np.median(v)), 4) if v else None,
                                       p90=round(float(np.percentile(v, 90)), 4) if v else None) for k, v in hook_err.items()},
        )
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"eval_{args.mode}.json"), "w") as f:
            json.dump(res, f, indent=2)
        np.savez_compressed(os.path.join(args.out, f"traj_env0_{args.mode}.npz"),
                            joint_names=np.array(robot.joint_names), **{k: np.array(v) for k, v in traj.items()})
        print(json.dumps({k: v for k, v in res.items() if k != "mean_body_pos_err_m"}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
