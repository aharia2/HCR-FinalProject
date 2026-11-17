# Step 2: Retarget to Origin (Roboparty GMR): Gate 2 report

**Status: waiting for approval.** Input: `motion/step1_gvhmr/results/IMG_0531/hmr4d_results.pt` (Gate 1 approved).

## Pipeline (all on the local machine, no GPU)

| Stage | Script | What it does |
|---|---|---|
| 2a Retarget | `motion/step2_gmr/retarget.py` | Roboparty GMR `f7bfaea`, headless, 50 Hz. Robot = `robot/mjcf/rpo_hook.xml` with joint ranges shrunk to **0.9 × URDF** (robolab's soft limit). IK config `configs/smplx_to_rpo_hook.json` |
| 2b Clean | `motion/step2_gmr/postprocess.py` | Zero-phase low-pass (6 Hz; arms 3 Hz), **ground the feet**, re-clamp to the soft limits |
| 2c Check | `motion/step2_gmr/analyze_retarget.py` | Limits, speeds, feet, hook height, side-by-side video |
| 2c' Extra checks | `motion/step2_gmr/extra_checks.py` | Hook opening direction, balance (COM vs support polygon, foot tilt), self-collision (added after review) |
| 2d Convert | `motion/step2_gmr/to_robolab.py` | Writes the CSV robolab's converter expects and runs robolab's own `convert_motion.py` (unchanged) → BeyondMimic `.npz` |

GMR runs from `~/HCR/third_party/GMR` (venv `~/HCR/third_party/.venv-gmr`, CPU torch). The robot is registered
with GMR at runtime, so no GMR files were edited.

## Changes from Roboparty's tuned `smplx_to_rpo.json` (one entry)

Roboparty mapped the human **right wrist** to `right_elbow_yaw_link`, which is the *start* of the forearm. That
put the hook about 16 cm beyond the hand along the forearm. It went low when the forearm pointed down (pickup)
but stayed high when the forearm pointed forward (set-down): **20 cm gap**.

New mapping: human right wrist → **`right_hook_link`** (forearm end / hook mount), +2 cm offset so the
handle seat sits about 6 cm past the wrist (the palm), with position weight **50** and rotation weight **2**
(was 10/10). The left arm and legs are unchanged.

| | Roboparty 10/10 | hook link 10/10 | **hook link 50/2 (used)** |
|---|---|---|---|
| Hook throat at pickup | 0.33 m | 0.34 m | **0.32 m** |
| Hook throat at set-down | 0.58 m | 0.54 m | **0.43 m** |
| Hook ↔ scaled human wrist (median) | 0.16 m | 0.16 m | **0.07 m** (≈ the 6 cm palm offset) |

The remaining 11 cm set-down gap (the human's own was ≈ 6 cm) comes from **ankle pitch saturating at the soft
limit (−0.54 rad)** during both squats. The robot can't bring its shins forward like the human, so it can't
lower its hips as far. IK can't fix that.

**Self-collision avoidance (second change, IK only).** The first version passed the right forearm **2.4 cm into
the right thigh** during the pickup squat (4.8–6.5 s). robolab trains with self-collisions enabled, so that
reference could not be tracked. `retarget.py` now adds a mink `CollisionAvoidanceLimit` that keeps ≥ 1 cm
between the right elbow/forearm/hook and the pelvis, torso, right leg and left thigh/knee. In GMR's private
copy of the model only, robot geoms are made mutually collidable so mink considers the pairs. Result: 0
penetrating frames (1.1 mm closest after filtering). Hook heights changed by < 1 cm.

"Hook throat" = the handle seat, `(0.060, 0, −0.020)` m in `right_hook_link`, measured from a cross-section
of the hook mesh (top of the bottom bar is at z = −0.026 m for x = 0.045–0.068 m).

## Results (`motion/step2_gmr/results/IMG_0531/`)

| Check | Result |
|---|---|
| Duration / rate | 17.0 s, 851 frames @ 50 Hz |
| Joints beyond robolab soft limit (0.9 × URDF) | **none** (several joints touch it during the squats; IK clamps there) |
| Joint speed vs robolab sim limits (legs 25, rest 8 rad/s) | **all within** |
| Joint speed vs V2 hardware URDF (legs 10.5, arms/ankles 3.77 rad/s) | all within **except left elbow pitch, 4.6 rad/s peak** (free left-arm swing at ≈ 9.4 s; not used by the task) |
| Feet height (lowest foot point) | −0.7 … +1.0 cm (was +9 … +21 cm floating before grounding) |
| Foot sliding while planted | median 2.6 / 3.2 cm/s (L/R) |
| Base (pelvis) height | 0.40 m (squat) … 0.78 m (standing) |
| Hook throat: pickup / carry / set-down | **0.33 m** at 5.1 s / ≈ 0.8 m / **0.43 m** at 13.3 s |
| **Hook opening direction** (hook +z · world up; < 0 = handle can fall out) | ✅ ≥ 0.63 throughout the carry (median 0.87, i.e. tilted ≤ 51°, typically 29°); 0.85 at pickup, 0.96 at set-down |
| **Self-collision** (right arm/hook vs pelvis, torso, legs) | ✅ 0 penetrating frames; closest 1.1 mm at 6.3 s (lift). Was −2.4 cm before the fix |
| **Balance in the squats** (COM ground projection vs support polygon of feet) | ⚠️ borderline: COM up to **2.2 cm outside** (median 0.6 cm outside); with a 2 kg bag at the hook, 0.5 cm outside. Feet tilt 3–12° in the pickup squat (heels slightly up). The large negative margins in `gate2_extra.png` are single-support phases of walking, which are normal for dynamic gait |
| Converted `.npz` vs our FK | max body-position error **1.0 mm**, joint error 6e-7 rad (also confirms root quaternion order) |
| robolab loader compatibility | same arrays as Roboparty's `rpo_bm/yundong0.npz`; adds `quaternion_order: xyzw` and joint/body names. **Verified by reading robolab's loader source, not by running it** (it needs Isaac Lab). Loading this file through robolab's motion loader is the first thing to do on the VM in step 3 |

Files:
- `gate2_side_by_side.mp4`: video (left) next to Origin (right). Green dot = hook throat, magenta = scaled human right wrist.
- `gate2_keyframes.jpg`: frames at walk, approach, pickup (5.1 s), carry, set-down approach, set-down (13.3 s).
- `gate2_plots.png`: joint positions vs limits, joint speeds vs hardware limits, foot/hook heights, top-down paths.
- `gate2_extra.png`: hook opening direction, COM margin (robot and robot + 2 kg bag), foot tilt, arm-to-body clearance.
- `bag_pickup_IMG_0531.npz`: **the step 3 training input** (robolab BeyondMimic format).
- `robot_motion.npz` (raw GMR) / `robot_motion_clean.npz` (cleaned) / `robot_motion_hhtools.csv` (converter input).

## Implications for steps 3–4 and the hardware course

- **Bag/handle geometry must match the motion, not the other way round.** The hook throat bottoms out at
  0.32 m, so the bag's handle top should be ≈ 0.35 m above the floor when the bag stands (hook passes ≈ 3 cm
  under it, then lifts). The bag in the video is a full-size tote. For hardware, use a bag (or stand height)
  that gives that handle height.
- **Set-down will drop the bag ≈ 10 cm** (0.43 vs 0.32 m). Options for step 4: accept it (placement test
  still passes if the bag lands upright), or add a task reward that lowers the hook at set-down.
- **Distances:** walk to bag ≈ 0.6 m, carry ≈ 1.9 m (robot scale), as noted at Gate 1.
- Joints spend time at the soft limit during squats (ankles especially), and the squat reference is only just
  balanced (COM ≈ 1–2 cm outside the feet). BeyondMimic tracks relative body poses, so the policy can shift
  its weight slightly. If tracking fails in the squats, the next fix is a centre-of-mass task in the IK
  (mink `ComTask`) to pull the COM over the feet.

## Addendum (2026-11-17, before step 3): checks against the *training* model

1. **Self-collision with the training URDF's collision shapes** (boxes for thighs/knees/torso, cylinders for
   the arms, no collision on the forearm link) with robolab's `replace_cylinders_with_capsules=True` (worst
   case: caps added to the full length). Of all 269 non-adjacent body pairs, the closest is 1.6 cm (pelvis vs
   left thigh), and the right arm/hook vs leg pairs are ≥ 2.4 cm. ✅
2. **Hook vs elbow capsule: fixed in the robot model.** In that check, the forearm's collision capsule (on
   `right_elbow_pitch_link`) overlapped the hook's mounting flange by 3.1 cm **in every pose**. That would be a
   permanent self-collision if the simulator keeps the hook as a separate body. The hook's collision
   geometry now covers only x ≥ 4 cm of the hook (the U and the throat's back wall). Visual mesh, mass and
   inertia are unchanged. Result: 3 mm clearance to the capsule, and the handle seat still has ≥ 0.8 cm of
   free space. 6 convex pieces instead of 10 (`robot/scripts/build_hook_model.py`, `COLLISION_MIN_X`).
3. **Left forearm roll.** GMR pinned `left_elbow_yaw_joint` at its soft limit (+1.41 rad) for the whole
   clip. That's a left-wrist orientation artifact with no task role, and the training would penalise it.
   `postprocess.py` now holds it at 0 rad (the standing default).
4. **Entering and leaving the motion on the robot.** `roboparty_deploy` switches from the standing/walking
   policy to a motion policy with LB, starting at frame 0 with no blending, and holds the last frame at the
   end. Our first and last frames are now 0.65 rad max / 0.3 rad rms from the deploy default pose, the same
   range as Roboparty's own sample motions (0.4–0.6 rad max). The person is standing still at both ends,
   so no extra blend segment is needed.

Step 2 numbers after these changes are the same (hook 0.33 / 0.43 m, opening up, no self-collision, limits OK).
`bag_pickup_IMG_0531.npz` regenerated.
