"""Step 2d: cleaned GMR motion -> robolab BeyondMimic .npz (50 Hz).

    python motion/step2_gmr/to_robolab.py motion/step2_gmr/results/IMG_0531

Uses robolab's own converter (robolab/scripts/tools/hhtools/convert_motion.py @ 45cdc13) unchanged:
  1. write the HHTools-style CSV it expects (time, root_x..root_qw in xyzw, dof_<joint> columns, # sample_rate)
  2. load_motion() -> resample_motion() -> convert_motion()
Only load_robot() is replaced, because the original imports robolab/Isaac Lab just to read three constants
(asset path, joint order, body order). They are copied below from robolab/assets/robots/roboparty.py @ 45cdc13.
Then verifies the .npz against our own forward kinematics.
"""
import importlib.util
import pathlib
import sys

import mujoco
import numpy as np

HOME = pathlib.Path.home()
ROBOLAB = HOME / "HCR" / "third_party" / "roboparty_train" / "robolab"
URDF = ROBOLAB / "data/robots/roboparty/rpo/urdf/rpo.urdf"          # training asset (rpo_description 8e0f8f7)
MESHES = ROBOLAB / "data/robots/roboparty/rpo/meshes"

RPO_JOINT_ORDER = (  # robolab roboparty.py @ 45cdc13
    "left_thigh_yaw_joint", "right_thigh_yaw_joint", "torso_joint",
    "left_thigh_roll_joint", "right_thigh_roll_joint",
    "left_arm_pitch_joint", "right_arm_pitch_joint",
    "left_thigh_pitch_joint", "right_thigh_pitch_joint",
    "left_arm_roll_joint", "right_arm_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_arm_yaw_joint", "right_arm_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_yaw_joint", "right_elbow_yaw_joint",
)
RPO_BODY_ORDER = ("base_link", *(n.removesuffix("_joint") + "_link" for n in RPO_JOINT_ORDER))


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_motion", ROBOLAB / "scripts/tools/hhtools/convert_motion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_robot():
    """Same as convert_motion.load_robot('rpo') without the robolab import."""
    spec = mujoco.MjSpec.from_file(str(URDF))
    spec.compiler.fusestatic = False
    spec.meshdir = str(MESHES)
    spec.body(RPO_BODY_ORDER[0]).add_freejoint(name="root")
    return spec.compile(), list(RPO_JOINT_ORDER), list(RPO_BODY_ORDER)


def main(d):
    d = pathlib.Path(d)
    M = np.load(d / "robot_motion_clean.npz")
    fps, Q, names = float(M["fps"]), M["qpos"], [str(n) for n in M["joint_names"]]

    csv = d / "robot_motion_hhtools.csv"
    t = np.arange(len(Q)) / fps
    quat_xyzw = Q[:, [4, 5, 6, 3]]                       # MuJoCo qpos quaternion is wxyz
    cols = ["time", "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"] + [f"dof_{n}" for n in names]
    with csv.open("w") as f:
        f.write(f"# sample_rate: {fps}\n# source: GVHMR -> Roboparty GMR (rpo_hook) -> postprocess.py\n")
        f.write(",".join(cols) + "\n")
        for k in range(len(Q)):
            f.write(",".join(f"{v:.6f}" for v in [t[k], *Q[k, :3], *quat_xyzw[k], *Q[k, 7:]]) + "\n")

    cm = load_converter()
    times, joint_q, src_names, src_fps = cm.load_motion(csv)
    joint_q = cm.resample_motion(times, joint_q, 50.0, keep_last=False)
    model, jn, bn = load_robot()
    motion = cm.convert_motion(model, jn, bn, joint_q, src_names, 50.0)
    out = d / "bag_pickup_IMG_0531.npz"
    with out.open("wb") as f:
        np.savez_compressed(f, **motion)

    # --- verification against our own FK (hooked MJCF, same kinematics) ---
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from analyze_retarget import load_model
    ours = load_model(); data = mujoco.MjData(ours)
    n = min(len(Q), len(motion["joint_pos"]))
    errs = []
    for k in range(0, n, 10):
        data.qpos[:] = Q[k]; mujoco.mj_kinematics(ours, data)
        for j, b in enumerate(bn):
            errs.append(np.linalg.norm(data.xpos[ours.body(b).id] - motion["body_pos_w"][k, j]))
    jp_err = np.abs(motion["joint_pos"][:n] - Q[:n, 7:][:, [names.index(x) for x in jn]]).max()
    print(f"saved {out}: {len(motion['joint_pos'])} frames @ {float(motion['fps'][0])} Hz, "
          f"{len(jn)} joints, {len(bn)} bodies")
    print(f"check: max body-position error vs our FK = {max(errs) * 1000:.3f} mm, max joint error = {jp_err:.2e} rad")
    print("keys:", {k: getattr(v, 'shape', v) for k, v in motion.items()})


if __name__ == "__main__":
    main(sys.argv[1])
