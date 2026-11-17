# Work log

Chronological record of what was done, why, and what was found. Newest at the bottom.

## 2026-11-16: Robot model with hook

- Copied the Origin model from `Roboparty/roboto_origin` `modules/rpo_description` (commit `d94afba`) into
  `robot/`. This is the same URDF robolab trains on (byte-identical to robolab's submodule `8e0f8f7`).
- Added the team hook (`roboto-origin-v2-right-square-hook.stl`) to the right wrist as `right_hook_link`
  (fixed joint on `right_elbow_yaw_link` at x = 0.13275 m). Mounting was measured from both meshes: the
  hook's 36 mm boss sits in the forearm's 37.2 mm bore, and the 6× M3 bolt circle lines up at a 90° roll,
  which makes the hook open upward at the zero pose.
- Mass 60.3 g (solid PLA + screws). Collision is 10 CoACD convex pieces so the hook opening stays open.
- Details: `robot/README.md`. Generator: `robot/scripts/build_hook_model.py`. Render: `robot/hook_preview.png`.

## 2026-11-16: Route decisions

- Training framework: **Isaac Lab via Roboparty's robolab** instead of a custom MuJoCo trainer. robolab
  already has a BeyondMimic motion-tracking task for Origin, a motion converter, PPO, and a deploy path.
- robolab (Isaac Lab 3.0) defaults to the **MuJoCo Warp** physics backend, so an RTX GPU isn't required.
- Retargeting: **Roboparty's GMR fork** already supports Origin (`smplx_to_rpo.json`, `gvhmr_to_robot.py`),
  so "add Origin to GMR" is mostly done. Roboparty's own docs now point to HHTools for retargeting; we
  stay on GMR as requested and convert its output with a small adapter.
- Decisions D1–D9 recorded in `PLAN.md`.

## 2026-11-16: Joint limits (D5)

- Compared 7 Roboparty model files + the deploy config. 16 of 23 joints disagree somewhere; Roboparty
  doesn't document which is correct.
- robolab trains on the URDF limits (soft limit factor 0.9); Roboparty GMR's MJCF uses a third mix; deploy
  sets ±3.14 for all joints.
- **Decision:** URDF limits everywhere. `build_hook_model.py` now overwrites every joint range in
  `robot/mjcf/rpo_hook.xml` with the URDF value (verified: 23/23 match; e.g. right elbow pitch is now
  −0.6..1.57 instead of −1.0..3.57).

## 2026-11-16: Video selection (D7)

- 7 clips in `Videos/`: 0530–0533 are landscape side views, 0534–0536 portrait front views (the person walks
  toward the camera and leaves the frame, so not used).
- **0531 and 0532**: static camera, whole body visible throughout, walk → squat → lift → walk → squat →
  set down, ~17 s, 1280×720 @ 30 fps.
- Checked the carrying hand by zooming into the lift and carry frames. The person faces screen-right, so the
  camera sees their right side; the carrying arm is the near arm and the bag hangs on the near side of the
  legs → **right hand** in both clips, matching the hook side.
- **0532 preferred**: the hand comes palm-up under the handle and lifts (5.8–6.8 s), like a hook would.
  0531 grasps from above. Final pick after GVHMR (Gate 1).

## 2026-11-16: Trainer input format verified

- robolab's `scripts/tools/hhtools/convert_motion.py` reads a CSV with `time`, `root_x, root_y, root_z,
  root_qx, root_qy, root_qz, root_qw` (xyzw) and one `dof_<joint_name>` column per joint, plus an optional
  `# sample_rate: <fps>` comment. It runs forward kinematics on the training URDF and writes the BeyondMimic
  `.npz` (50 Hz default).
- GMR (`gvhmr_to_robot.py --save_path`) saves `fps`, `root_pos`, `root_rot`, `dof_names`, `dof_pos`,
  `body_names`, `body_positions` in a `.pkl` → a small adapter to that CSV is enough.
- **Caution:** GMR reorders `root_rot` twice with contradictory comments ("xyzw to wxyz" / "wxyz to xyzw"),
  so the saved quaternion order is not certain from reading the code. The adapter will determine it
  empirically: run FK under both interpretations and keep the one that reproduces GMR's saved
  `body_positions`.
- BeyondMimic policy observations (`command`, `base_ang_vel`, `projected_gravity`, `joint_pos`,
  `joint_vel`, `actions`) match `roboparty_deploy`'s `obs_layouts`, so Step 5 needs no observation changes.

## 2026-11-16: GPU VM

- Google Cloud project `project-bc8c6e21-0f04-4e24-8d6` (billing account `01D95C-F2BA4A-0CEF0B`, $300 credits).
  Account upgraded from free trial; global GPU quota raised 0 → 1.
- VM **`hcr-gpu`** in **`us-central1-b`** (`us-central1-a` was out of L4s): `g2-standard-8` (8 vCPU, 32 GB),
  1× **NVIDIA L4 24 GB**, image `common-cu129-ubuntu-2204-nvidia-580` (Ubuntu 22.04.5, driver 580.178,
  CUDA 12.9), 200 GB pd-balanced disk. Verified with `nvidia-smi`.
- Cost: ~$0.85/h while running, ~$20/month disk. **Policy: start only when a step needs the GPU, stop
  right after.** Stopped after verification.
- Budget **"HCR GPU budget"**: $300, email alerts at ~$50 / $150 / $250 / $300.
- Commands (from this machine, `gcloud` at `~/google-cloud-sdk/bin/gcloud`):
  `gcloud compute instances start|stop hcr-gpu --zone=us-central1-b`,
  `gcloud compute ssh hcr-gpu --zone=us-central1-b`.

## 2026-11-17: Step 1 (GVHMR) run, awaiting Gate 1 approval

- Body models received (SMPL-X v1.1 npz+pkl, SMPL v1.1.0 python); unpacked to `body_models/` (git-ignored,
  license forbids redistribution). Wrong first download (SMPL v1.0.0, no neutral model) deleted.
- GVHMR checkpoints: Google Drive rate-limited → Hugging Face mirror `camenduru/GVHMR`, SHA-256 matched
  across two independent mirrors.
- Ran GVHMR on the VM (≈ 25 min of VM time, then stopped). Install fixes recorded in `vm_setup.sh`.
- Results + full report: `docs/step1_gvhmr.md`. **Recommendation: switch to IMG_0531** (no vertical
  drift, cleaner right-hand signal, longer carry). One-handed, right-hand carry confirmed.

## 2026-11-17: Gate 1 approved (IMG_0531); Step 2 (GMR) run, awaiting Gate 2 approval

- Pushed all commits to GitHub at the user's request (`491215a..94c0e77`).
- GMR installed locally (`~/HCR/third_party/GMR` @ f7bfaea, own venv, CPU). Robot `rpo_hook` registered at
  runtime with 0.9 × URDF joint ranges.
- First run with Roboparty's IK config: robot floated 9–21 cm (GVHMR ground), joint-speed spikes, hook 20 cm
  higher at set-down than at pickup.
- Fixes: post-processing (low-pass, per-frame grounding, re-clamp); IK change: human right wrist →
  `right_hook_link`, weights 50/2 (set-down gap 20 → 11 cm; remaining gap from ankle-pitch soft limit).
- Converted with robolab's own `convert_motion.py` → `bag_pickup_IMG_0531.npz`; verified to 1 mm vs our FK.
- Full report: `docs/step2_gmr.md`.
- Review before Gate 2 added three checks (`extra_checks.py`): hook opening direction ✅ (opening points up
  throughout the carry), balance ⚠️ (squat COM ≤ 2.2 cm outside the feet; flagged, not forced), self-collision
  ❌ → fixed (forearm was 2.4 cm inside the right thigh at pickup; added mink collision avoidance, now 0
  penetrating frames). Regenerated `bag_pickup_IMG_0531.npz` (still 1 mm vs FK).

## 2026-11-17: Gate 2 approved (bag handle ≈ 0.35 m; accept ≈ 10 cm set-down drop for now)

- Pre-training review → checked the motion against the training URDF's collision shapes (capsule arms):
  OK, but found the forearm capsule overlapping the hook flange by 3.1 cm in every pose → hook collision
  geometry clipped to x ≥ 4 cm (6 pieces). Left forearm roll artifact held at 0. Deploy entry/exit poses
  checked (same range as Roboparty's motions). Details: addendum in `docs/step2_gmr.md`.
