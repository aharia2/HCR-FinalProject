# Step 1: Video → human motion (GVHMR): Gate 1 report

**Status: waiting for approval.**

## What was run

- GVHMR `ee960bb` (zju3dv) on the GCP L4 VM, static-camera mode (`-s`, no visual odometry).
  torch 2.3.0+cu121, smplx 0.1.28, numpy 1.23.5, lightning 2.3.0, ultralytics 8.2.42.
- Checkpoints from the Hugging Face mirror `camenduru/GVHMR@21b32d5` (Google Drive was rate-limited).
  SHA-256 verified; the identical hashes are in a second, independent mirror (`ryanrudes/gvhmr`).
- Smoke test on GVHMR's own `tennis.mp4` passed, then **IMG_0532** and **IMG_0531** (~3 min each).
- Scripts: `motion/step1_gvhmr/vm_setup.sh`, `vm_run.sh`, `analyze.py`. Logs: `results/step1_*.log`.
- Install needed four fixes for the stock Ubuntu image (chumpy build isolation, gcc, Python headers,
  `python3-tk` for a stray `from turtle import forward` in GVHMR). All are in `vm_setup.sh`.
- VM time ≈ 25 min (≈ $0.35); stopped afterwards.

## Outputs (`motion/step1_gvhmr/results/<clip>/`)

| File | What |
|---|---|
| `hmr4d_results.pt` | GVHMR output (SMPL-X params, global + in-camera). **This is the input to step 2 (GMR).** |
| `<clip>_3_incam_global_horiz.mp4` | Left: mesh overlaid on the video. Right: motion in 3D world view |
| `joints_global.npz` | SMPL-24 joints in the world frame (Y up, start at origin, initial facing +Z) |
| `gate1_plots.png` | Top-down path, joint heights, wrist heights, foot speeds |
| `gate1_summary.json` | Numbers below |
| `../gate1_keyframes.jpg` | Overlay + world view at pickup, carry and place for both clips |

## Results

| Check | IMG_0531 | IMG_0532 |
|---|---|---|
| Mesh overlay fits the person | ✅ all key frames | ✅ all key frames |
| Duration | 17.0 s | 16.8 s |
| Pelvis path (length / net) | 4.11 m / 1.67 m | 3.90 m / 2.20 m |
| Pickup squat (pelvis min, standing ≈ 1.0 m) | 0.75 m at 5.8 s | 0.73 m at 6.0 s |
| Place squat | 0.76 m at 12.7 s | 0.78 m at 13.4 s |
| **Right wrist at pickup vs left** | **0.55 m vs 0.69 m** (right clearly lower) | 0.57 vs 0.59 m (both low, ambiguous) |
| **Right wrist at place vs left** | **0.63 vs 0.77 m** | 0.65 vs 0.76 m |
| Carry: right vs left wrist height | **1.04 vs 0.87 m** | 1.14 vs 0.93 m |
| Walking distance while carrying | 1.9 m | 0.9 m |
| Feet planted between steps (foot speed → 0) | ✅ | ✅ |
| Vertical drift (start → end standing height) | none (head 1.58 → 1.54 m) | **≈ +10 cm** (pelvis 1.0 → 1.1 m, head 1.55 → 1.65 m) |
| Ground penetration | none | none |

**Right hand confirmed** (the advisor's concern about left/right swaps in side views): in both clips the
right wrist is the one that drops to the bag at set-down and is held higher during the carry. In 0531 it is
also clearly the lower wrist at pickup. The left hand is not on the handle (wrists 0.5–0.65 m apart while
carrying), so it's a **one-handed carry**.

## Recommendation: use **IMG_0531** (changes my earlier preference for 0532)

1. **No vertical drift.** 0532 floats up by about 10 cm over the clip, which would show up as the robot's base
   reference creeping upward.
2. **Cleaner right-hand signal.** 0531's right wrist is unambiguously the low hand at pickup and place.
   0532 has both hands equally low at pickup.
3. **Longer carry** (1.9 m vs 0.9 m), which is closer to the task (carry to a drop zone).
4. 0532's advantage, the palm-up hook-like grasp, matters less than I thought. GVHMR/GMR track the wrist
   position and orientation, not the fingers, and the hook is rigid.

## Known artefacts to handle in step 2

- **Ankle joints sit ≈ 15–20 cm above the ground** (expected ≈ 8–10 cm). The global ground estimate is a
  little low, so the body floats slightly. GMR re-grounds the motion. I'll check foot height on the robot at Gate 2.
- **Distances:** the clip's walk to the bag is ≈ 0.6 m and the carry ≈ 1.9 m. The policy reproduces the
  clip, so the sim/hardware course should use these distances (the contract's 1.5 m / 2 m came from the
  original plan, before the video was recorded).
- **Squat depth:** pelvis drops ≈ 25 cm with heels down. Origin's ankle-pitch limit (±0.54 rad usable) is the
  risk flagged in `PLAN.md`; step 2's joint-limit check will show whether the IK can reach the handle.
