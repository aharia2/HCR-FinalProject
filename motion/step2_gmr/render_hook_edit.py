"""Render the hook-edited reference with the bag (kinematic: bag on the floor, then hanging from the throat during
the carry, then on the floor at the set-down spot). Two close-up views per frame.

    MUJOCO_GL=egl ~/HCR/third_party/.venv-gmr/bin/python motion/step2_gmr/render_hook_edit.py [--out video.mp4]
"""
import argparse
import json
import pathlib
import sys

import cv2
import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "motion/step2_gmr"))
import hook_edit as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(H.DST / "hook_edit_reference.mp4"))
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    E = np.load(H.DST / "robot_motion_clean.npz"); Q = E["qpos"]
    P = json.load(open(H.PLACEMENT)); F = P["frames"]
    xml = (REPO / "robot/mjcf/rpo_hook.xml").read_text().replace('meshdir="../meshes/"', f'meshdir="{REPO / "robot/meshes"}/"')
    spec = mujoco.MjSpec.from_string(xml)
    spec.body("right_elbow_yaw_link").add_site(name="throat", pos=H.THROAT.tolist())
    b = spec.worldbody.add_body(name="bag"); b.add_freejoint()
    bx, by, bz = H.BOX
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[bx / 2, by / 2, bz / 2], pos=[0, 0, bz / 2], rgba=[0.3, 0.6, 0.3, 0.6], contype=0, conaffinity=0)
    for sx in (-1, 1):
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[sx * H.POST_X, 0, bz, sx * H.POST_X, 0, H.BAR_Z], size=[H.BAR_R, 0, 0], rgba=[0.9, 0.9, 0.2, 1], contype=0, conaffinity=0)
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[-H.POST_X, 0, H.BAR_Z, H.POST_X, 0, H.BAR_Z], size=[H.BAR_R, 0, 0], rgba=[0.9, 0.9, 0.2, 1], contype=0, conaffinity=0)
    m = spec.compile(); d = mujoco.MjData(m)
    site, fa = m.site("throat").id, m.body("right_elbow_yaw_link").id

    def yq(yaw):
        return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    W, Hh = 640, 480
    rr = mujoco.Renderer(m, Hh, W)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 50 // a.every, (2 * W, Hh))
    for k in range(0, len(Q), a.every):
        d.qpos[:Q.shape[1]] = Q[k]
        mujoco.mj_kinematics(m, d)
        if k < F["lift"]:
            pos, q = P["start"]["pos"], yq(P["start"]["yaw"])
        elif k < F["place_min"]:
            th = d.site_xpos[site]; ax = H.horiz(d.xmat[fa].reshape(3, 3)[:, 1])
            pos = th - np.array([0, 0, H.BAR_Z + H.SEAT]); q = yq(np.arctan2(ax[1], ax[0]))
            pos[2] = max(pos[2], 0.0)
        else:
            pos, q = P["drop"]["pos"], yq(P["drop"]["yaw"])
        d.qpos[-7:-4] = pos; d.qpos[-4:] = q
        mujoco.mj_forward(m, d)
        focus = np.array(P["start"]["pos"] if k < (F["lift"] + F["place_min"]) // 2 else P["drop"]["pos"]) + [0, 0, 0.35]
        imgs = []
        for az in (0.0, 90.0):
            cam = mujoco.MjvCamera(); cam.lookat[:] = focus; cam.distance, cam.elevation = 1.1, -12
            ax = H.horiz(d.xmat[fa].reshape(3, 3)[:, 1]); cam.azimuth = float(np.degrees(np.arctan2(ax[1], ax[0]))) + az
            rr.update_scene(d, cam); img = cv2.cvtColor(rr.render(), cv2.COLOR_RGB2BGR)
            th = d.site_xpos[site]
            cv2.putText(img, f"t={k / 50:5.2f}s  frame {k}  {'along bar' if az == 0 else 'side'}", (10, 25), 0, 0.6, (0, 0, 0), 2)
            cv2.putText(img, f"throat z {th[2] * 100:4.1f} cm", (10, 50), 0, 0.6, (0, 0, 0), 2)
            imgs.append(img)
        vw.write(np.hstack(imgs))
    vw.release()
    print("video:", a.out)


if __name__ == "__main__":
    main()
