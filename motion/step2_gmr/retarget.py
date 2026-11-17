"""Step 2a: retarget GVHMR output to Origin-with-hook using Roboparty's GMR fork (headless).

    ~/HCR/third_party/.venv-gmr/bin/python motion/step2_gmr/retarget.py \
        --gvhmr motion/step1_gvhmr/results/IMG_0531/hmr4d_results.pt --out motion/step2_gmr/results/IMG_0531

What it does
  * Registers a GMR robot "rpo_hook" at runtime (GMR's own files are not edited):
      model     = robot/mjcf/rpo_hook.xml with every joint range shrunk to 0.9 x the URDF range
                  (= robolab's soft_joint_pos_limit_factor, so the reference never leaves the training limits)
      IK config = configs/smplx_to_rpo_hook.json (Roboparty's tuned smplx_to_rpo.json + our hook entry, if any)
  * Runs GMR's IK frame by frame at 50 Hz (BeyondMimic's rate).
  * Saves qpos straight from MuJoCo (root quaternion is MuJoCo's w-first; no GMR re-ordering involved).

Outputs in --out:
  robot_motion.npz   fps, qpos (T, 7+23: xyz, quat wxyz, joints in MJCF order), joint_names,
                     hook_pos (T,3), human_rwrist (T,3) and human_rhand (T,3) after GMR scaling, in robot world
"""
import argparse
import json
import pathlib
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
GMR_ROOT = pathlib.Path.home() / "HCR" / "third_party" / "GMR"
SOFT = 0.9  # robolab RPO_CFG.soft_joint_pos_limit_factor


def build_gmr_model(tmpdir: pathlib.Path) -> pathlib.Path:
    """rpo_hook.xml with soft (0.9x) joint ranges and an absolute meshdir."""
    src = (REPO / "robot" / "mjcf" / "rpo_hook.xml").read_text()
    src = src.replace('meshdir="../meshes/"', f'meshdir="{REPO / "robot" / "meshes"}/"')

    def shrink(m):
        lo, hi = map(float, m.group(2).split())
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT
        return f'{m.group(1)}{mid - half:.6f} {mid + half:.6f}"'

    src, n = re.subn(r'(<joint name="[^"]+"[^>]*?range=")([^"]+)"', shrink, src)
    assert n == 23, n
    out = tmpdir / "rpo_hook_soft.xml"
    out.write_text(src)
    return out


def register_robot(xml_path: pathlib.Path, ik_config: pathlib.Path):
    from general_motion_retargeting import params
    params.ROBOT_XML_DICT["rpo_hook"] = xml_path
    params.IK_CONFIG_DICT["smplx"]["rpo_hook"] = ik_config
    params.ROBOT_BASE_DICT["rpo_hook"] = "base_link"
    params.VIEWER_CAM_DISTANCE_DICT["rpo_hook"] = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvhmr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ik_config", default=str(HERE / "configs" / "smplx_to_rpo_hook.json"))
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(GMR_ROOT))
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast, load_gvhmr_pred_file

    tmp = pathlib.Path(tempfile.mkdtemp())
    xml = build_gmr_model(tmp)
    register_robot(xml, pathlib.Path(args.ik_config))

    smplx_data, body_model, smplx_output, human_h = load_gvhmr_pred_file(args.gvhmr, GMR_ROOT / "assets" / "body_models")
    frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=args.fps)
    print(f"human height estimate {human_h:.3f} m, {len(frames)} frames @ {fps} Hz")

    gmr = GMR(actual_human_height=human_h, src_human="smplx", tgt_robot="rpo_hook")
    model, data = gmr.model, mujoco.MjData(gmr.model)
    hook_id = model.body("right_hook_link").id
    # hook throat = where a handle rests: top of the bottom bar is z=-0.026 for x=0.045..0.068 (mesh section)
    throat_local = np.array([0.060, 0.0, -0.020])

    qpos, hook, rwrist, rhand = [], [], [], []
    for f in frames:
        q = gmr.retarget(f).copy()
        qpos.append(q)
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        hook.append(data.xpos[hook_id] + data.xmat[hook_id].reshape(3, 3) @ throat_local)
        sd = gmr.scaled_human_data
        rwrist.append(np.asarray(sd["right_wrist"][0]))
        rhand.append(np.asarray(sd.get("right_index1", sd["right_wrist"])[0]))

    joint_names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    np.savez_compressed(out / "robot_motion.npz", fps=fps, qpos=np.array(qpos), joint_names=np.array(joint_names),
                        hook_pos=np.array(hook), human_rwrist=np.array(rwrist), human_rhand=np.array(rhand),
                        human_height=human_h)
    (out / "retarget_info.json").write_text(json.dumps(dict(
        gvhmr=str(args.gvhmr), ik_config=str(args.ik_config), fps=fps, frames=len(qpos),
        human_height_m=float(human_h), soft_limit_factor=SOFT, gmr_commit="f7bfaea"), indent=2))
    print("saved", out / "robot_motion.npz")


if __name__ == "__main__":
    main()
