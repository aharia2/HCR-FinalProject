# Step 3: Imitation policy (robolab BeyondMimic, Isaac Lab 3.0 / MJWarp): in progress

## Setup (verified on the VM)

- Isaac Lab v3.0.0, roboparty_train `53b411a` (robolab `45cdc13`, rsl_rl `9fb0cc6`, rpo_description `8e0f8f7`),
  MuJoCo-Warp backend, 1× L4. `sim/step3_train/vm_setup.sh`.
- Task `RPO-BagMimic` (`sim/step3_train/bagmimic`): Roboparty's `RPO-BeyondMimic` (PPO, 4096 envs, same
  observations/actions/rewards/domain randomization/actuators) with the **hooked URDF** and **our motion**.
  robolab is not edited.
- `check_env.py`: the importer merges the hook into `right_elbow_yaw_link` → 24 bodies / 23 joints, identical
  to the motion file. Nominal forearm mass 0.7417 kg = 0.6815 + hook 0.0603. Motion (850 frames) mapped
  by name onto 23 tracked bodies. The URDF importer 3.0 ignores `replace_cylinders_with_capsules`.
- Speed ≈ 1.45 s / iteration (4096 envs) → 15,000 iterations ≈ 6 h ≈ $5.

## Run 1: stock BeyondMimic terminations/rewards → marched in place (stopped at ≈ iteration 1600)

- Training statistics looked good: by iteration 1400, 95 % of episodes reached the time limit and the
  relative body error was 9 cm.
- The frame-0 evaluation (64 envs, no pushes, checkpoint 1400) showed the failure: **64/64 finished without
  falling, but the base travelled 1.3 m vs 5.6 m in the reference** and ended **2 m** from it. Hook-throat error
  was 0.75 m at pickup and 2.1 m at set-down. The policy copied the leg motion on the spot.
- Cause: the stock termination checks torso **height** only, and the global torso-position reward
  (std 0.3 m) has almost no gradient once the robot is > 0.6 m off. The actor observes no position or linear
  velocity. Roboparty's own sample motions are in-place (dances), so the stock settings were never tested on
  a walking clip.
- Baseline kept: `logs/rsl_rl/rpo_bagmimic/2026-11-17_18-34-01` on the VM,
  `sim/step3_train/results/early_1400/` locally.

## Run 2: deviation from stock (`experiment_name = rpo_bagmimic_walk`)

| Change | Value | Why |
|---|---|---|
| New termination `anchor_pos_3d` | torso 3-D distance to reference > **0.8 m** | forces travelling; loose enough for push drift (pushes ≤ 0.5 m/s every 1–3 s, which the policy can't observe) |
| New reward `motion_global_anchor_pos_wide` | exp(−err²/1.0²), weight 1.0 (stock 0.3 m term kept) | gives a pull back toward the path at long range |

Everything else is unchanged. Launched 19:15 UTC.

## Evaluation (`sim/step3_train/eval_bagmimic.py`)

Full-clip rollouts starting at frame 0 with no start offsets, in two modes: clean (no pushes, no observation
noise) and randomized (training randomization on). Reports falls, **torso xy offset at 4 s / pickup / 12 s /
set-down**, **torso path-length ratio**, hook-throat error (total and with the torso offset removed), and
per-body error. Saves env 0's trajectory for `render_rollout.py`. Fixes made after the first use: the
reference body is now the torso (was the left hip), and quaternion maths uses Isaac Lab's own functions
(Isaac Lab 3.0 is x-first).

## Run 2 progress (frame-0 evaluation, clean, 64 envs)

| Checkpoint | Finished clip | Torso path ratio | Torso xy offset at pickup | Hook error at pickup (total / arm only) |
|---|---|---|---|---|
| run 1 @ 1400 (baseline) | 64/64, in place | 0.23 | 0.66 m | 0.75 m / – |
| run 2 @ 1000 | 0/64 (all hit the 0.8 m limit at ≈ 2.3 s: did not start walking from a standstill) | – | – | – |
| run 2 @ 2000 | 44/64 | 0.74 | 0.45 m (p90 0.53) | 0.40 m / **0.08 m** |

The arm/hook placement relative to the body is good (8–9 cm). The remaining error is the body lagging the
reference by ≈ 0.45 m, accumulated in the first ≈ 4 s after a slow start from standing.
| run 2 @ 4000, clean | **64/64** | 0.78 | 0.27 m | 0.25 m / **0.03 m** (set-down 0.41 / 0.17 m) |
| run 2 @ 4000, randomized | 43/64 | 0.90 | 0.25 m | 0.24 m / 0.05 m (set-down 0.53 / 0.13 m) |
| run 2 @ 8000, clean | **64/64** | 0.88 | **0.14 m** | **0.13 m / 0.04 m** (set-down 0.26 / 0.07 m) |
| run 2 @ 8000, randomized | **56/64** | 0.96 | 0.23 m | 0.20 m / 0.08 m (set-down 0.37 / 0.09 m) |

| run 2 @ 14000, clean | 63/64 (one fall at 11.2 s) | 0.90 | 0.11 m | **0.09 m** / 0.05 m (set-down 0.26 / 0.11 m) |
| **run 2 @ 15000 (final), clean** | **64/64** | 0.87 | 0.15 m | 0.11 m / 0.05 m (set-down 0.29 / 0.15 m) |
| **run 2 @ 15000 (final), randomized** | **60/64** (falls at 14.0–16.5 s) | 0.96 | 0.17 m | 0.16 m / 0.07 m (set-down 0.49 / 0.16 m) |

Final training stats (iteration 14999): 94% of episodes reach the time limit, mean body-position error 4.1 cm.
The final policy (`model_14999.pt`, 5.6 h on the L4) is the starting point for step 4. The remaining hook error
at pickup/set-down is mostly whole-body drift, which step 4's hook/lift/place rewards target directly.

Videos (policy right, reference left): `results/run2_final_15000/policy_vs_ref_{clean,randomized}.mp4`
(also 14000, 8000, 4000).
