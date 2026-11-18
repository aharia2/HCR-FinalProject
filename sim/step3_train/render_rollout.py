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
    ap.add_argument("--ref", default=str(REPO / "motion/step2_gmr/results/IMG_0531/robot_motion_clean.npz"))  # step 4 v6+: pass IMG_0531_hook
    a = ap.parse_args()
    R = np.load(a.traj, allow_pickle=True)
    M = np.load(a.ref)
    model = load_model()
    has_bag = "bag_pos" in R.files
    if has_bag:   # add the bag (same geometry as sim/step4_bag/assets/bag.urdf) as a free body
        import json as _json, os as _os
        G = _json.load(open(REPO / f"sim/step4_bag/bag_placement{'_wf' if _os.environ.get('HCR_BAG') == 'wf' else ''}.json"))["geometry"]
        xml = (REPO / "robot/mjcf/rpo_hook.xml").read_text().replace('meshdir="../meshes/"', f'meshdir="{REPO / "robot/meshes"}/"')
        spec = mujoco.MjSpec.from_string(xml)
        b = spec.worldbody.add_body(name="bag"); b.add_freejoint()
        bx, by, bz = G["box"]
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[bx / 2, by / 2, bz / 2], pos=[0, 0, bz / 2], rgba=[0.3, 0.6, 0.3, 1], contype=0, conaffinity=0)
        for sx in (-1, 1):
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[sx * G["post_x"], 0, bz, sx * G["post_x"], 0, G["bar_z"]], size=[G["bar_r"], 0, 0], rgba=[0.9, 0.9, 0.2, 1], contype=0, conaffinity=0)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[-G["post_x"], 0, G["bar_z"], G["post_x"], 0, G["bar_z"]], size=[G["bar_r"], 0, 0], rgba=[0.9, 0.9, 0.2, 1], contype=0, conaffinity=0)
        model = spec.compile()
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
    k0 = int(round(float(R["t"][0]) / 0.02))                  # clip frame of the first sample (carry-only evals start at the lift)
    for k in range(T):
        kr = min(k + k0, len(Qref) - 1)
        d_ref.qpos[:Qref.shape[1]] = Qref[kr]
        d_pol.qpos[:3] = R["root_pos"][k]
        d_pol.qpos[3:7] = R["root_quat_wxyz"][k]
        d_pol.qpos[7:7 + len(perm)] = R["joint_pos"][k][perm]
        if has_bag:
            d_pol.qpos[-7:-4] = R["bag_pos"][k]; d_pol.qpos[-4:] = R["bag_quat_wxyz"][k]
            d_ref.qpos[-7:] = d_pol.qpos[-7:]; d_ref.qpos[-7] += 100.0       # hide the bag in the reference panel
        imgs = []
        for d, label in ((d_ref, "reference"), (d_pol, "policy")):
            mujoco.mj_forward(model, d)
            cam = mujoco.MjvCamera()
            cam.distance, cam.elevation, cam.azimuth = 2.2, -10, az
            cam.lookat[:] = [d_ref.qpos[0], d_ref.qpos[1], 0.55]       # same camera for both panels
            rr.update_scene(d, cam)
            img = cv2.cvtColor(rr.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(img, f"{label}  t={(k + k0) * 0.02:5.2f}s", (10, 25), 0, 0.7, (0, 0, 0), 2)
            if label == "policy" and has_bag:
                cv2.putText(img, f"bag height {R['bag_height'][k] * 100:4.1f} cm", (10, 50), 0, 0.6, (0, 0, 0), 2)
            elif label == "policy" and "hook_err" in R.files:
                cv2.putText(img, f"hook err {R['hook_err'][k] * 100:4.1f} cm", (10, 50), 0, 0.6, (0, 0, 0), 2)
            imgs.append(img)
        vw.write(np.hstack(imgs))
    vw.release()
    print("video:", a.out)


if __name__ == "__main__":
    main()
