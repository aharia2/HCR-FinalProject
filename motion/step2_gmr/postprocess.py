"""Step 2c: clean the raw GMR output (robot_motion.npz -> robot_motion_clean.npz).

    python motion/step2_gmr/postprocess.py motion/step2_gmr/results/IMG_0531

1. Low-pass filter (zero-phase Butterworth, 6 Hz; arm joints 3 Hz) joint angles, root position and root quaternion.
   Removes IK/GVHMR jitter that produced joint-speed spikes; walking/lifting content is < 3 Hz.
2. Ground the motion: GVHMR's ground estimate is low, so the robot floated 9-21 cm. Per frame, take the
   lowest foot-mesh point; smooth that signal (1.5 Hz) and subtract it from the root height, so the lowest
   foot touches z = 0 throughout (walking always has a foot on the ground; there is no flight phase).
3. Clamp joints to the robolab soft range (0.9 x URDF) again after filtering.
4. Hold left_elbow_yaw_joint at 0 (GMR artifact: pinned at its soft limit all clip; no task role).
hook_pos is recomputed from the cleaned motion.
"""
import pathlib
import sys

import mujoco
import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_retarget import foot_points, load_model  # noqa: E402

CUT_HZ, ARM_CUT_HZ, GROUND_HZ, SOFT = 6.0, 3.0, 1.5, 0.9
THROAT_LOCAL = np.array([0.060, 0.0, -0.020])  # handle seat on the hook (mesh section at y=0)


def lowpass(x, fps, cut):
    b, a = butter(4, cut / (fps / 2))
    return filtfilt(b, a, x, axis=0)


def main(d):
    d = pathlib.Path(d)
    M = dict(np.load(d / "robot_motion.npz"))
    fps, Q = float(M["fps"]), M["qpos"].copy()
    names = [str(n) for n in M["joint_names"]]
    model = load_model()
    data = mujoco.MjData(model)

    # quaternion sign continuity before filtering
    quat = Q[:, 3:7].copy()
    for k in range(1, len(quat)):
        if np.dot(quat[k], quat[k - 1]) < 0:
            quat[k] *= -1
    Q[:, 0:3] = lowpass(Q[:, 0:3], fps, CUT_HZ)
    quat = lowpass(quat, fps, CUT_HZ)
    Q[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    Q[:, 7:] = lowpass(Q[:, 7:], fps, CUT_HZ)
    # arm joints: gentler 3 Hz cut, otherwise the free left-arm swing peaks at ~7 rad/s
    # (V2 hardware arm limit 3.77 rad/s); lifting motions are well below 3 Hz.
    arm = [i for i, n in enumerate(names) if "_arm_" in n or "_elbow_" in n]
    Q[:, 7 + np.array(arm)] = lowpass(Q[:, 7 + np.array(arm)], fps, ARM_CUT_HZ)

    # left forearm roll: GMR pins it at the soft limit (+1.41 rad) for the whole clip (left-wrist orientation
    # artifact). It moves no other body and has no task role (hook is on the right), and a reference sitting on
    # the limit is penalised in training, so hold it at the deploy/standing default (0.0 rad).
    Q[:, 7 + names.index("left_elbow_yaw_joint")] = 0.0

    rng = model.jnt_range[[model.joint(n).id for n in names]]
    mid, half = rng.mean(1), (rng[:, 1] - rng[:, 0]) / 2 * SOFT
    Q[:, 7:] = np.clip(Q[:, 7:], mid - half, mid + half)

    low = []
    for k in range(len(Q)):
        data.qpos[:] = Q[k]
        mujoco.mj_kinematics(model, data)
        low.append(min(foot_points(model, data, "left_ankle_roll_link")[:, 2].min(),
                       foot_points(model, data, "right_ankle_roll_link")[:, 2].min()))
    low = np.array(low)
    shift = lowpass(low, fps, GROUND_HZ)
    Q[:, 2] -= shift

    hook = []
    hid = model.body("right_hook_link").id
    for k in range(len(Q)):
        data.qpos[:] = Q[k]
        mujoco.mj_kinematics(model, data)
        hook.append(data.xpos[hid] + data.xmat[hid].reshape(3, 3) @ THROAT_LOCAL)
    M["qpos"], M["hook_pos"] = Q, np.array(hook)
    M["human_rwrist"] = M["human_rwrist"] - np.c_[np.zeros((len(Q), 2)), shift]
    M["ground_shift"] = shift
    np.savez_compressed(d / "robot_motion_clean.npz", **M)
    print(f"ground shift: min {shift.min():.3f} m, median {np.median(shift):.3f} m, max {shift.max():.3f} m")


if __name__ == "__main__":
    main(sys.argv[1])
