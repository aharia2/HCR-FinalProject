# PLAN: Variable-weight bag transfer on Roboto Origin

**Goal:** a working video of Origin (hook on the right wrist) picking up a shopping bag, carrying it and
setting it down, robust to bag mass 0.5–2.0 kg, with a policy ready to run on the real robot.

All steps are done by Claude, and each step **stops for approval** before the next one starts.
Progress and decisions are logged in [`docs/WORKLOG.md`](docs/WORKLOG.md).

## Decisions (2026-11-16)

| # | Decision |
|---|---|
| D1 | Hook on the **right wrist only** |
| D2 | Hook printed in **PLA**, attached with 6× M3 (model mass 60.3 g, assuming 100 % infill) |
| D3 | **No time limit** in the success test |
| D4 | One cloud GPU VM, `hcr-gpu` (GCP us-central1-b, 1× L4 24 GB; see WORKLOG). Training runs on Isaac Lab 3.0 with the **MuJoCo Warp (Newton)** backend (robolab's default), so an RTX card is not required |
| D5 | **URDF joint limits are the source of truth** (`rpo_description/urdf/rpo.urdf`). robolab trains on them at 0.9 × range. `robot/mjcf/rpo_hook.xml` and the GMR robot model use the same limits |
| D6 | **One continuous clip** (walk → hook → carry → place) |
| D7 | Source video: **`Videos/IMG_0531.mp4`** (approved at Gate 1, 2026-11-17): side view, static camera, right-hand one-handed carry, no vertical drift in GVHMR. IMG_0532 was the first choice but floated ≈ 10 cm |
| D8 | Bag mass randomized over **0.5–2.0 kg** |
| D9 | Git: commit on `main` only, **push later** in one batch |

## Pinned upstream versions

| Repo | Commit | Used for |
|---|---|---|
| [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) | pin at install | Step 1: video → SMPL-X |
| [Roboparty/GMR](https://github.com/Roboparty/GMR) | `f7bfaea` | Step 2: SMPL-X → Origin (has `smplx_to_rpo.json`, `gvhmr_to_robot.py`) |
| [Roboparty/roboparty_train](https://github.com/Roboparty/roboparty_train) | `53b411a` | Steps 3–5: Isaac Lab 3.0 + RSL-RL |
| ├ robolab | `45cdc13` | `RPO-BeyondMimic` task, `convert_motion.py`, Origin robot config |
| ├ rsl_rl | `9fb0cc6` | PPO |
| └ rpo_description (in robolab) | `8e0f8f7` | Training URDF, byte-identical to `robot/urdf/rpo.urdf` |
| [Roboparty/roboto_origin](https://github.com/Roboparty/roboto_origin) `roboparty_deploy` | `d94afba` | Step 5: on-robot ONNX inference (`beyondmimic.yaml`) |

## Pipeline and approval gates

### Step 1: Video → human motion (GVHMR) · needs GPU
1. Install GVHMR on the VM; body models + checkpoints in place (see *Inputs needed* below).
2. Run GVHMR on **0532 and 0531**. Pick the clip on output quality: global root path, feet planted when
   standing, right arm clearly carrying.
3. **Gate 1, show:** overlay video (SMPL mesh on the footage), a top-down root-trajectory plot, and notes on
   any artefacts (foot sliding, jitter). Then stop.

### Step 2: Retarget to Origin (Roboparty GMR) · CPU
1. Point GMR's `rpo` robot at our hooked model with D5 limits. Add `right_hook_link` as the IK target for the
   human right hand, so the **hook**, not the wrist, follows the hand.
2. `gvhmr_to_robot.py --robot rpo` → Origin motion `.pkl`.
3. Adapter: GMR `.pkl` → robolab CSV (`time`, `root_x..root_qw` in xyzw, `dof_<joint>` columns,
   `# sample_rate:` header) → `convert_motion.py --robot rpo` → BeyondMimic `.npz` at 50 Hz.
4. Checks: joint limits vs **0.9 × URDF**, joint velocities vs actuator limits, feet not penetrating or
   skating, and hook height at the moment the handle is lifted.
5. **Gate 2, show:** retarget video (human and Origin side by side), joint-limit plot, `.npz` replayed
   kinematically in MuJoCo. Then stop.

### Step 3: Imitation policy (robolab BeyondMimic) · needs GPU
1. Install Isaac Lab 3.0 + roboparty_train on the VM. **Smoke test:** stock `RPO-BeyondMimic` trains for a
   few iterations.
2. New task `RPO-BagMimic`: BeyondMimic + hooked URDF + our `.npz`, no bag yet. Train until it tracks the
   whole clip without falling.
3. **Gate 3, show:** play video, tracking-error curves, fall rate. Then stop.

### Step 4: Post-train with bag + domain randomization · needs GPU
1. Add the bag (rigid body + handle the hook can catch) at the clip's pickup spot. Start at a fixed 1.0 kg.
2. Randomize **bag mass 0.5–2.0 kg**. Keep and extend robolab's domain randomization (friction, link mass
   and COM, motor strength/gains, action delay, pushes, observation noise).
3. Add task terms if tracking alone doesn't hold the bag: hook-to-handle distance, bag height, bag stays on
   the hook, place in the drop zone.
4. Evaluate at 0.5 / 1.0 / 1.5 / 2.0 kg plus a held-out 1.25 kg: success rate, bag lift ≥ 10 cm, no fall,
   bag placed. Also run a MuJoCo sim-to-sim check.
5. **Gate 4, show:** videos per mass, success table. Then stop.

### Step 5: Prepare for hardware · CPU
1. Export ONNX (robolab play/export).
2. Make a `roboparty_deploy` config: `obs_layouts`, joint order (`usd2urdf`), default angles, action scale,
   motion file. Already confirmed: BeyondMimic's policy observations match deploy's layout
   (`motion_pos, motion_vel, ang_vel, gravity_b, dof_pos, dof_vel, last_action`).
3. Sim-to-sim dry run through the deploy stack if possible. Write a safety checklist (tether, torque
   limits, e-stop) and a hardware test procedure.
4. **Gate 5, show:** ONNX, config, dry-run video, checklist.

## Inputs needed from Arya

1. ~~GPU VM~~ done: `hcr-gpu` (see WORKLOG).
2. **Body models** (free registration, needs your own account):
   - SMPL-X (https://smpl-x.is.tue.mpg.de): `SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, `SMPLX_FEMALE.npz`
     **and** the `.pkl` versions (`SMPLX_NEUTRAL.pkl`, …). GVHMR uses the `.npz` files, GMR the `.pkl` files.
   - SMPL (https://smpl.is.tue.mpg.de): `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl`, `SMPL_FEMALE.pkl`
     (GVHMR uses them for rendering and evaluation).
   - GVHMR's other checkpoints (GVHMR, HMR2, ViTPose, YOLO) download from its Google Drive link; I can fetch these.

## Risks

- **Deep squat vs ankle limits.** The human squats with heels down. Origin's ankle pitch is ±0.6 rad
  (±0.54 rad after training's 0.9 factor), so the pickup posture may have to be changed by the IK. Gate 2
  will show this. Fallback: re-record with the bag on a box.
- **2 kg vs hook strength.** The PLA hook should be load-tested at ≥ 2× the top mass (4 kg) before any
  hardware run. The model doesn't check structural strength.
- **Hooking contact in sim** is the hardest physics part (thin handle, contact stability). It's handled in
  Step 4 with a simple rigid handle first.
