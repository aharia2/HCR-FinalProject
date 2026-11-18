# Step 4: Bag + bag-mass randomization + domain randomization (design, in progress)

Approved to start right after step 3 finishes (user, 2026-11-17).

## Why step 4 needs task rewards, not just a bag

Your video shows a hand grasping a handle, not a hook. Replaying the reference motion with a handle placed in
front of the robot:
- The hook comes down almost vertically (27 cm in the last 1.2 s before pickup), so a handle where the hook
  bottoms out is hit from above.
- A search over 1,485 handle placements (position ±5 cm, height +0–4 cm, yaw ±30°) found only **6** that the
  reference hook path threads (clear on the way down, seated on the lift), with **4 mm** clearance.
- The step 3 policy places the hook ≈ 13 cm from the reference at pickup (mostly whole-body lag), so tracking
  alone won't hook reliably. The policy has to learn to hook, which is what the task rewards below are for.

## Bag model (`sim/step4_bag/make_bag.py` → `assets/bag.urdf`, `bag_placement.json`)

- Box 0.30 × 0.12 × 0.24 m, two handle posts and a crossbar (radius 6 mm, 12 cm wide) with the bar at
  **0.362 m**, matching the step 2 agreement (handle ≈ 0.35 m).
- Pickup placement = best of the 6 candidates (yaw +30° from the hook's bar axis, 3.5 cm above the throat's
  lowest point). Robot-vs-bag clearance along the reference approach: hook 4 mm, knees/feet ≥ 3 cm.
- Set-down spot = under the hook at the reference set-down (1.6 m from pickup). The release lets the bag drop ≈ 7 cm.
- Phase frames (50 Hz): lift **272** (5.44 s), set-down **665** (13.30 s), release **676** (13.52 s).
- Isaac Lab 3.0's URDF importer always builds an articulation, so the bag is a **0-joint articulation**.

## Task `RPO-BagCarry` (`sim/step4_bag/bagcarry`)

- Everything from step 3 (`RPO-BagMimic`: tracking rewards, walking fix, robolab's domain randomization)
  plus:
- **Bag at every reset in the state matching the sampled clip time:** on the floor (start), hanging from
  the reference hook (carry), or on the floor (after release).
- **Bag mass uniform 0.5–2.0 kg**, sampled per environment at startup (4,096 envs cover the range).
- **Domain randomization already in robolab's task, kept:** ground friction 0.3–1.6, link mass ×0.8–1.2,
  torso mass ±1 kg, torso/base COM ±3–5.5 cm, motor stiffness/damping ×0.8–1.2, armature ×0.8–1.2,
  default joint offsets ±0.01 rad, random pushes (≤ 0.5 m/s every 1–3 s), observation noise, actuator delay
  (DelayedPD).
- **New rewards:** hook-to-handle distance (std 5 cm, from 1.2 s before lift to release); bag lifted
  (height/15 cm, during the carry); bag placed upright near the set-down spot (after release); penalty for
  knocking the bag over before the lift.
- **New termination:** bag back on the floor in mid-carry (dropped).
- **Observations unchanged** (proprioception + motion command), so training resumes from the step 3
  checkpoint and the policy only needs encoders + IMU on the robot.

## Verified on the VM (`check_bag.py`, 4 envs, robot holding reference joint targets, no policy)

Bag spawns in all three phases. A hanging bag stays seated on the hook (handle 1–14 mm from the throat) while
the arm moves and the robot falls. Random masses: 0.68 / 0.56 / 1.45 / 0.72 kg.

## Evaluation plan

Full clip from frame 0 at **0.5, 1.0, 1.5, 2.0 kg** and **1.25 kg** (inside the training range, so an
interpolation test as in the contract), clean and with training randomization. Success = hooked and lifted
≥ 10 cm during the carry, bag set down upright within 0.4 m of the spot, robot standing at the end, no fall.

## Run v1 (00:54–01:30 UTC, stopped at iteration ≈ 15550)

Resumed from the final step 3 policy (`model_14999.pt`). After ≈ 550 iterations:
- 84–94% of episodes ended in `bag_dropped`; mean episode 115 → 200 steps (step 3: ≈ 990).
- Body tracking error rose from 4.1 to 7.5 cm, and the fine hook reward fell (0.021 → 0.008).
- Frame-0 test at 1 kg (`results/early_15400/`): 0/64 hooked. At pickup the torso was 45 cm to the side of the
  reference, and the hook was never closer than 45 cm to the handle (step 3 alone: 25–30 cm).

Causes found:
1. **Spawn bug:** a hanging bag was placed on the *reference* hook, but the robot starts at the reference
   pose plus training noise (root ±5 cm, joints ±0.1 rad, initial velocity), so most hanging bags spawned off
   the real hook and fell immediately.
2. **Termination too strict:** any episode that hadn't hooked the bag ended 1 s after the lift, which cut
   off the rest of the clip and its tracking reward, so walking quality dropped.
3. **No gradient toward the handle:** with a 5 cm kernel, the hook reward is ≈ 0 at 25 cm.

## Run v2 (from 01:33 UTC, again resumed from `model_14999.pt`)

- Hanging bags spawn on the robot's actual hook; forward kinematics is refreshed after the reset.
- `bag_dropped` fires only if the bag was carried (≥ 5 cm) earlier in the episode.
- New `hook_to_handle_coarse` reward: std 25 cm, weight 1, from 2 s before the lift to 0.4 s after.
- Everything else unchanged.

## Run v2 → v3: the policy gets the fixed bag position (user decision, 02:00 UTC)

v2 after ≈ 400 iterations: carrying improved a lot (mean episode 520 steps, hook and lift rewards ×10 vs v1), but
pickup from the floor didn't. Frame-0 test at 1 kg (`results/v2_15400/`): 0/64 hooked, closest hook–handle
approach 35 cm, and early anchor terminations at 2–4 s.

Root problem: the policy was **blind** (joints + IMU only). Even the best step 3 policy is 10–15 cm from the
reference at pickup, and the hook must land within a few cm of the handle.

**User decision:** the bag and the robot always start at fixed positions (bag ≈ 1 m in front), so give the robot
the bag's position.

v3 implementation:
- New observation `target_handle_b` (policy + critic): the handle position, in the robot torso frame, for the
  current target. Before the lift that is the pickup handle; afterwards it is the set-down spot (handle height).
  Both are fixed points relative to the start, so on hardware this needs **odometry (leg kinematics + IMU) + the
  known placement**, with no camera or bag sensing.
- Odometry error in training: a per-episode constant offset of ±3 cm in x and y, plus ±1 cm per-step noise.
- The new term is appended last. `pad_checkpoint.py` adds zero input weights (actor 121 → 124, critic
  340 → 343, plus Adam buffers), so v3 starts exactly from the step 3 policy.
- All v2 fixes kept. Started at ≈ 02:05 UTC with 5000 iterations (`swap_step4_run.sh v3`).

## Run v3 → v4: knocked-over bags were being rewarded (03:10 UTC)

v3 frame-0 tests at 1 kg:

| Checkpoint | No fall | Hooked | Closest hook–handle | Bag "max height" (median) |
|---|---|---|---|---|
| v3 @ 15400 | 61/64 | 0/64 | 20 cm | 6.9 cm |
| v3 @ 16200 | **64/64** | 0/64 | – | 6.6 cm |

Video at 15400 (`results/v3_15600/`): the robot squats about 20 cm short of the bag. The "lifted" ≈ 6 cm was the
bag **knocked onto its side** (its origin is then ≈ 6 cm up). That earned `bag_lifted` reward (6/15) and set the
`was_lifted` flag. A reward loophole.

v4 changes (resumed from v3 @ 16200, which has the same observations; 3000 more iterations):
- `bag_lifted` counts only while the bag is upright (`up_z > 0.8`); so does `was_lifted` for the drop
  termination.
- `bag_knocked_over` (tipped, `up_z < 0.7`) is penalised during the whole clip, weight −1 → **−3**.
- `eval_bagcarry.py`: max bag height counts only an upright bag.
