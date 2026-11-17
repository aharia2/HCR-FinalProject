"""Render a policy rollout (traj_env0_*.npz from eval_bagmimic.py) next to the reference motion, locally (MuJoCo, EGL).

    MUJOCO_GL=egl ~/HCR/third_party/.venv-gmr/bin/python sim/step3_train/render_rollout.py \
        <traj_env0_clean.npz> --out <video.mp4>

Left panel: reference (motion/step2_gmr/results/IMG_0531/robot_motion_clean.npz). Right panel: policy rollout.
Both use robot/mjcf/rpo_hook.xml; Isaac Lab joint order is mapped to MJCF order by name.
"""
import argparse
import pathlib
import sys

import cv2
import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "motion" / "step2_gmr"))
from analyze_retarget import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=str(REPO / "motion/step2_gmr/results/IMG_0531/robot_motion_clean.npz"))
    a = ap.parse_args()
    R = np.load(a.traj, allow_pickle=True)
    M = np.load(a.ref)
    model = load_model()
    d_ref, d_pol = mujoco.MjData(model), mujoco.MjData(model)
    mj_names = [str(n) for n in M["joint_names"]]
    isaac_names = [str(n) for n in R["joint_names"]]
    perm = [isaac_names.index(n) for n in mj_names]           # MJCF order <- Isaac order
    Qref = M["qpos"]
    T = len(R["t"])
    W, H = 640, 480
    rr = mujoco.Renderer(model, H, W)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 50, (2 * W, H))
    walk = Qref[-1, :2] - Qref[0, :2]
    az = float(np.degrees(np.arctan2(walk[1], walk[0]))) + 90
    for k in range(T):
        kr = min(k, len(Qref) - 1)
        d_ref.qpos[:] = Qref[kr]
        d_pol.qpos[:3] = R["root_pos"][k]
        d_pol.qpos[3:7] = R["root_quat_wxyz"][k]
        d_pol.qpos[7:] = R["joint_pos"][k][perm]
        imgs = []
        for d, label in ((d_ref, "reference"), (d_pol, "policy")):
            mujoco.mj_forward(model, d)
            cam = mujoco.MjvCamera()
            cam.distance, cam.elevation, cam.azimuth = 2.2, -10, az
            cam.lookat[:] = [d_ref.qpos[0], d_ref.qpos[1], 0.55]       # same camera for both panels
            rr.update_scene(d, cam)
            img = cv2.cvtColor(rr.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(img, f"{label}  t={k * 0.02:5.2f}s", (10, 25), 0, 0.7, (0, 0, 0), 2)
            if label == "policy" and "hook_err" in R.files:
                cv2.putText(img, f"hook err {R['hook_err'][k] * 100:4.1f} cm", (10, 50), 0, 0.6, (0, 0, 0), 2)
            imgs.append(img)
        vw.write(np.hstack(imgs))
    vw.release()
    print("video:", a.out)


if __name__ == "__main__":
    main()
