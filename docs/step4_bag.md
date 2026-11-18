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

## Run v1 (00:54–01:14 UTC, stopped at iteration ≈ 15550)

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

## Run v2 (from 01:14 UTC, again resumed from `model_14999.pt`)

- Hanging bags spawn on the robot's actual hook; forward kinematics is refreshed after the reset.
- `bag_dropped` fires only if the bag was carried (≥ 5 cm) earlier in the episode.
- New `hook_to_handle_coarse` reward: std 25 cm, weight 1, from 2 s before the lift to 0.4 s after.
- Everything else unchanged.

## Run v2 → v3: the policy gets the fixed bag position (user decision; v3 started 01:37 UTC)

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
- All v2 fixes kept. Started at 01:37 UTC with 5000 iterations (`swap_step4_run.sh v3`).

## Run v3 → v4: knocked-over bags were being rewarded (v4 started 02:20 UTC)

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

## Run v4 → v5: hook under the handle, not onto it (v5 started 02:47 UTC)

v4 @ 16800, frame-0 test at 1 kg: 62/64 no fall, **0/64 hooked**; upright bag max height 3.5 cm (median), and
the knocked-over penalty was still large in training. Diagnosis: before the lift, the hook reward pulled the
throat onto the bar itself. An upward-opening hook that goes for the bar from above presses the handle down and
tips the bag.

v5 (resumed from v4 @ 17000, 2200 more iterations):
- New `hook_under_handle`: target 4 cm **below** the bar (inside the handle loop; the box top is 8 cm lower).
  std 5 cm, weight 2, from 1.2 s before the lift until just after it.
- The coarse approach term (std 25 cm) now targets the same under-bar point.
- `hook_to_handle` (seated on the bar) now applies only from the lift to the release.

## v5 result and where the hook actually goes (stopped 03:17 UTC at iteration 17934)

v5 @ 17600, frame-0 test at 1 kg: **64/64 no fall**, 0/64 hooked; the `hook_under_handle` reward was 0.0000
throughout training.

`measure_hook.py` (v5 @ 17800, frame 0, bag moved out of the way, all 64 envs). Policy hook minus reference
hook:

| Frame | Clean: mean offset (x, y, z) | Clean: spread p90 | Randomized: spread p90 |
|---|---|---|---|
| pickup_min 254 | (0.40, 0.33, 0.12) m | 0.20 m | 0.29 m |
| lift 272 | (0.34, 0.31, 0.08) m | 0.22 m | 0.32 m |
| release 676 | (0.34, −0.01, −0.09) m | 0.12 m | 0.22 m |

Two conclusions:
1. **The bag-carry policy moved away from the bag.** Its hook is ≈ 45 cm from the reference at pickup (step 3
   alone: 11 cm). The tipping penalty taught it to avoid the bag, because a successful hook was never found.
2. **Moving the bag to where the policy goes would not fix it.** Even in clean mode (only the per-env physics
   randomization: masses, friction, motor gains), the hook position varies by 20 cm (p90) between envs. A
   rigid handle needs a few cm.

Hooking a handle from the floor with this reward design does not emerge. Carrying a bag that is already on the
hook does work in training (hook_to_handle and bag_lifted rewards are high), so that part was measured on its
own: `eval_bagcarry.py --start=lift` starts at the lift frame with the bag already on the robot's hook.

## Carry-only results (v5 @ 17800, `--start=lift`, bag on the robot's hook at 5.44 s, 64 envs per row)

| Mode | Mass | No fall | Lifted ≥ 10 cm (upright) | Hooked at mid-carry (≤ 3 cm) | Max bag height (median) | Set down in the 0.4 m square |
|---|---|---|---|---|---|---|
| clean | 0.5 kg | 64/64 | **64/64** | 52/64 | 21.5 cm | 0/64 |
| clean | 1.0 kg | 64/64 | **64/64** | 54/64 | 21.4 cm | 0/64 |
| clean | 1.25 kg (held out) | 64/64 | **64/64** | 49/64 | 21.4 cm | 0/64 |
| clean | 1.5 kg | 64/64 | **64/64** | 43/64 | 21.4 cm | 0/64 |
| clean | 2.0 kg | 64/64 | **64/64** | 44/64 | 20.3 cm | 0/64 |
| randomized | 0.5 kg | 64/64 | **64/64** | 47/64 | 23.4 cm | 1/64 |
| randomized | 1.25 kg | 63/64 | **64/64** | 49/64 | 23.3 cm | 2/64 |
| randomized | 2.0 kg | 62/64 | **64/64** | 39/64 | 22.0 cm | 0/64 |

- **Carrying works across 0.5–2 kg, with domain randomization:** every bag is carried, ≥ 97% of runs don't fall,
  and the bag stays on the hook. ("Hooked" < 64 means the handle slid along the hook by more than 3 cm, not
  that it fell off.) The policy carries with the arm lowered at its side instead of raised as in the reference.
- **Set-down fails:** the robot lowers the bag to ≈ 4 cm above the floor about 23 cm from the spot, but the
  upward-opening hook never comes out of the handle. It then walks back still carrying the bag (the
  reference walks back after the set-down).
- Videos: `results/carry_v5_17800/carry_clean_{1kg,2kg}.mp4`.

## Step 4 status

| Part of the task | Status |
|---|---|
| Walk the clip without falling, with domain randomization | ✅ (step 3 policy; step 4 keeps it: 64/64 from frame 0) |
| Carry a 0.5–2 kg bag, including the held-out 1.25 kg | ✅ (64/64 lifted, ≤ 2/64 falls) |
| Pick the bag up from the floor | ❌ 0/64 (v1–v5) |
| Release the bag at the set-down spot | ❌ the hook does not let go |

Both failures are about **getting the hook into and out of the handle**. The reference motion comes from a
human hand (grasp → lift, set down → open the hand), and a hook can't copy the "close/open hand" part. The
reference hook comes straight down onto the handle and never moves out of it.

## v6: hook-edited reference motion (user chose option 1, 2026-11-18)

Both failures (pickup 0/64, release never happens) come from the reference: a human hand grabs the handle and
opens to let go, and the retargeted hook copies neither. New script `motion/step2_gmr/hook_edit.py` edits the step
2 motion (no new video, no re-retargeting):

**Hook geometry used:** a "C" that opens up, in the forearm x-z plane. The tip is ≈ 2.3 cm above the seat and
3 cm in front of it. The handle bar lies along the forearm's y axis. To go under the bar, the throat must be
≥ 3.5 cm below it; the plan uses 5 cm.

**Edit (IK with mink on a throat site; only the waist and right arm move; base, legs and left arm keep the
reference):**

| Phase | Frames (50 Hz) | Hook throat |
|---|---|---|
| pickup approach | 204 → 238 | from the reference to 10 cm **behind** the bar, 5 cm **below** it |
| slide under | 238 → 258 | forward along the forearm until under the bar |
| seat / lift | 258 → 272 | up until the bar rests in the seat (2.1 cm above the throat) |
| rejoin | 272 → 310 | decaying offset back onto the reference lift |
| lower | 615 → 672 | down until the bag stands on the floor |
| drop out of the seat | 672 → 692 | 3 cm further down (5 cm under the bar) |
| pull back | 692 → 715 | 9 cm back along the forearm, so the tip clears the bar |
| rise behind | 715 → 735 | 10 cm above the bar, still behind it |
| rejoin | 735 → 770 | back onto the reference |

Supporting changes:
- **Time warp:** in the reference the person stands straight up after the set-down (hips +28 cm in 0.5 s), so the
  hook couldn't stay down to unhook. Frames 655–675 (the bottom of the squat) play 4× slower: +60 frames, so the
  clip is now 910 frames (18.2 s); episodes are still 20 s.
- **Set-down spot:** 33 cm in front of the right shoulder (horizontally), so the arm can reach 5 cm under the bar
  and the bag clears the thighs.
- **Pickup spot:** under the reference throat at its lowest point. The bar axis is the forearm's y axis (v1–v5
  used a +30° yaw from the 1485-placement search).
- **IK details:** continuity regularization (each frame stays near the previous one: no arm-branch flips). At the
  end of each edit, a 30-frame joint crossfade back to the reference.

**Checks (`results/IMG_0531_hook/hook_edit_report.json`):**
- Throat on every waypoint within 0.2 mm; worst frame 2.9 cm, inside the end crossfade (bag hanging or
  already released).
- Max joint speed 6.05 rad/s, the same as the original clip. Inside the 0.9 soft limits.
- Robot–bag clearance: pickup ≥ **7 mm** (hook vs bar while sliding under); set-down ≥ **2.2 cm**.
- Converted to `bag_pickup_IMG_0531_hook.npz` with `to_robolab.py`: FK error 1 mm.
- Video: `motion/step2_gmr/results/IMG_0531_hook/hook_edit_reference.mp4` (two close-up views, kinematic bag).

`bag_placement.json` v2 (v1 kept as `bag_placement_v1.json`): pickup_min 258, lift 272, place_min 672
(touch), release 715 (hook out), total 911 (training clip 910).

**Task v6 (`RPO-BagCarry`):**
- Uses the edited motion.
- The bag stands at the set-down spot from `place_min` (touch) on.
- `hook_to_handle` (seated) applies from lift to touch.
- The v5 under-bar terms are replaced by `hook_to_ref`: hook throat to the **reference** throat, absolute
  (std 4 cm, weight 2, plus a coarse std 20 cm, weight 1), in the engage window (lift − 70 … lift + 10) and the
  release window (touch − 10 … release + 20). The bag sits exactly where the edited reference engages it, so
  following the reference hook path there threads and releases the handle.
- Knocked-over penalty back to −1.
- Training restarts from the **step 3 policy** (padded), not v5, which had learned to keep away from the bag.
- v6 started 06:38 UTC from `step3_final_padded.pt` (5000 iterations). The env config confirms
  `bag_pickup_IMG_0531_hook.npz`. Fixed in `swap_step4_run.sh`: with no previous run the saved PID was 0, and
  `kill 0` killed the script's own process group (and the SSH session).

### v6 progress (frame-0 and carry-only tests, 1 kg, clean, 64 envs)

| Checkpoint | From frame 0: hooked / success | From the lift: success (carry + set down + unhook) | Hook at the lift vs the bar (along the forearm) |
|---|---|---|---|
| 15800 | 0 / 0 | – | 37 cm short |
| 17200 | 0 / 0 | **12/64** (v5: 0/64), 64/64 no fall | **8.5 cm short**, correct height, 1.5 cm sideways |

- Release works: from the lift, the policy sets the bag down upright in the square and unhooks in 12/64 runs.
- Pickup: the hook does the behind → under → up manoeuvre at the right height but stops short. The body is
  ≈ 20 cm behind the reference, and the arm makes up most of that.
| 18600 | 0 / 0 (first lift ≥ 10 cm: 1/64) | **21/64**, 64/64 no fall | – |

**v7** (automatic continuation, `vm_chain_v6_to_v7.sh`): when v6 ends at iteration 19999, resume from its last
checkpoint for 4000 more iterations, with extra engage-window weight (`hook_to_ref_engage` std 4 cm, weight 4;
`hook_to_ref_engage_coarse` std 12 cm, weight 2). Everything else as v6.
| **19998 (v6 final)** | 1/64 hooked / **1/64 full success** (pick up → carry → set down upright → unhook) | **28/64**, 64/64 no fall | – |

v7 started automatically at 09:20 UTC (resumed from 19998, 4000 iterations). `eval_bagcarry.py` now records all
envs and saves the first *successful* env's trajectory for rendering (key `saved_env`).
