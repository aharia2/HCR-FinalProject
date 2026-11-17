"""Step 2b: Gate-2 checks + video for a retargeted motion.

    MUJOCO_GL=egl python motion/step2_gmr/analyze_retarget.py motion/step2_gmr/results/IMG_0531 \
        --video Videos/IMG_0531.mp4

Checks (hard model = robot/mjcf/rpo_hook.xml, i.e. URDF limits):
  * joint positions vs URDF range and vs robolab soft range (0.9 x)
  * joint speeds vs robolab sim limits (legs/torso 25 rad/s, rest 8 rad/s) and V2 hardware URDF limits
  * feet: lowest foot point height (floating / penetration), skating while in contact
  * hook throat height over time (pickup / place) and distance to the scaled human right hand
Writes gate2_summary.json, gate2_plots.png, gate2_side_by_side.mp4 (+ optional shift of root height, see --ground).
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SIM_VEL = {"legs": 25.0, "other": 8.0}
LEGS = ("thigh_yaw", "thigh_roll", "thigh_pitch", "knee", "torso")
HW_VEL = {"ankle": 3.7692, "arm": 3.7692, "elbow": 3.7692}  # everything else 10.47 (V2 hardware URDF)


def hw_vel(name):
    for k, v in HW_VEL.items():
        if k in name:
            return v
    return 10.47


def load_model():
    xml = (REPO / "robot/mjcf/rpo_hook.xml").read_text().replace(
        'meshdir="../meshes/"', f'meshdir="{REPO / "robot/meshes"}/"')
    return mujoco.MjModel.from_xml_string(xml)


def foot_points(model, data, body):
    """World positions of all mesh vertices of the given body's geoms."""
    pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != model.body(body).id or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        pts.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    return np.vstack(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--video", help="source video for the side-by-side render")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--motion", default="robot_motion.npz", help="file inside dir (raw or *_clean.npz)")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    M = np.load(d / args.motion)
    tag = "" if args.motion == "robot_motion.npz" else "_clean"
    fps, Q, names = float(M["fps"]), M["qpos"], [str(n) for n in M["joint_names"]]
    hook, hand = M["hook_pos"], M["human_rwrist"]
    T = len(Q)
    t = np.arange(T) / fps

    model = load_model()
    data = mujoco.MjData(model)
    jid = [model.joint(n).id for n in names]
    rng = model.jnt_range[jid]
    mid, half = rng.mean(1), (rng[:, 1] - rng[:, 0]) / 2
    q = Q[:, 7:]
    norm = (q - mid) / half                    # -1..1 = URDF range
    dq = np.gradient(q, 1 / fps, axis=0)

    lfoot, rfoot, base_z = [], [], []
    for k in range(T):
        data.qpos[:] = Q[k]
        mujoco.mj_kinematics(model, data)
        lfoot.append(foot_points(model, data, "left_ankle_roll_link"))
        rfoot.append(foot_points(model, data, "right_ankle_roll_link"))
        base_z.append(Q[k, 2])
    lmin = np.array([p[:, 2].min() for p in lfoot]); rmin = np.array([p[:, 2].min() for p in rfoot])
    lc = np.array([p.mean(0) for p in lfoot]); rc = np.array([p.mean(0) for p in rfoot])
    foot_min = np.minimum(lmin, rmin)

    def skate(cpos, zmin):
        v = np.linalg.norm(np.gradient(cpos[:, :2], 1 / fps, axis=0), axis=1)
        contact = zmin < foot_min.min() + 0.02
        return dict(contact_frac=round(float(contact.mean()), 3),
                    median_speed_in_contact_mps=round(float(np.median(v[contact])), 4) if contact.any() else None,
                    p95_speed_in_contact_mps=round(float(np.percentile(v[contact], 95)), 4) if contact.any() else None)

    worst = []
    for i, n in enumerate(names):
        sim_lim = SIM_VEL["legs"] if any(k in n for k in LEGS) else SIM_VEL["other"]
        worst.append(dict(joint=n, min_norm=round(float(norm[:, i].min()), 3), max_norm=round(float(norm[:, i].max()), 3),
                          max_abs_vel=round(float(np.abs(dq[:, i]).max()), 2), sim_vel_limit=sim_lim,
                          hw_vel_limit=hw_vel(n)))
    over_soft = [w["joint"] for w in worst if max(abs(w["min_norm"]), abs(w["max_norm"])) > SOFT_CHECK + 1e-3]
    over_hw_vel = [(w["joint"], w["max_abs_vel"]) for w in worst if w["max_abs_vel"] > w["hw_vel_limit"]]
    over_sim_vel = [(w["joint"], w["max_abs_vel"]) for w in worst if w["max_abs_vel"] > w["sim_vel_limit"]]

    hz = hook[:, 2]
    k_pick = int(np.argmin(hz[: T // 2])); k_place = T // 2 + int(np.argmin(hz[T // 2:]))
    summary = dict(
        frames=T, fps=fps, duration_s=round(T / fps, 2),
        joints_beyond_soft_limit=over_soft,
        joints_over_sim_vel_limit=over_sim_vel, joints_over_hw_vel_limit=over_hw_vel,
        joints=worst,
        feet=dict(lowest_point_min_m=round(float(foot_min.min()), 4),
                  lowest_point_median_m=round(float(np.median(foot_min)), 4),
                  lowest_point_max_m=round(float(foot_min.max()), 4),
                  left=skate(lc, lmin), right=skate(rc, rmin)),
        base_height_m=dict(min=round(float(min(base_z)), 3), max=round(float(max(base_z)), 3)),
        hook=dict(pickup=dict(t=round(k_pick / fps, 2), throat_height_m=round(float(hz[k_pick]), 3)),
                  place=dict(t=round(k_place / fps, 2), throat_height_m=round(float(hz[k_place]), 3)),
                  max_height_m=round(float(hz.max()), 3),
                  dist_to_scaled_human_rwrist_median_m=round(float(np.median(np.linalg.norm(hook - hand, axis=1))), 3)),
    )
    (d / f"gate2_summary{tag}.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    a = ax[0, 0]
    for i, n in enumerate(names):
        a.plot(t, norm[:, i], lw=0.8, label=n.replace("_joint", ""))
    for s in (SOFT_CHECK, -SOFT_CHECK):
        a.axhline(s, color="orange", ls="--");
    for s in (1, -1):
        a.axhline(s, color="r")
    a.set_ylim(-1.1, 1.1); a.set_title("Joint position / URDF range (red = URDF limit, orange = 0.9 soft)")
    a.set_xlabel("time [s]"); a.legend(fontsize=5, ncol=3)
    a = ax[0, 1]
    ratio = np.abs(dq) / np.array([hw_vel(n) for n in names])
    for i, n in enumerate(names):
        a.plot(t, ratio[:, i], lw=0.8)
    a.axhline(1, color="r"); a.set_title("Joint speed / V2-hardware velocity limit (red = limit)")
    a.set_xlabel("time [s]")
    a = ax[1, 0]
    a.plot(t, lmin, "b", label="left foot lowest point"); a.plot(t, rmin, "r", label="right foot lowest point")
    a.plot(t, hz, "g", lw=2, label="hook throat height")
    a.axhline(0, color="k"); a.set_title("Heights [m]"); a.set_xlabel("time [s]"); a.legend(fontsize=8)
    a = ax[1, 1]
    a.plot(Q[:, 0], Q[:, 1], "k", label="base"); a.plot(hook[:, 0], hook[:, 1], "g", label="hook throat")
    a.plot(hand[:, 0], hand[:, 1], "m:", label="scaled human R wrist")
    a.set_aspect("equal"); a.set_title("Top-down [m]"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(d / f"gate2_plots{tag}.png", dpi=90)
    print(json.dumps({k: v for k, v in summary.items() if k != "joints"}, indent=2))

    if args.no_render:
        return
    import cv2
    W, H = 640, 480
    r = mujoco.Renderer(model, H, W)
    cap = cv2.VideoCapture(args.video) if args.video else None
    vfps = cap.get(cv2.CAP_PROP_FPS) if cap else None
    out = cv2.VideoWriter(str(d / f"gate2_side_by_side{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (W * (2 if cap else 1), H))
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 2.2, -10
    for k in range(T):
        data.qpos[:] = Q[k]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [Q[k, 0], Q[k, 1], 0.55]
        # side view, perpendicular to the overall walking direction
        walk = Q[-1, :2] - Q[0, :2]
        cam.azimuth = float(np.degrees(np.arctan2(walk[1], walk[0]))) + 90  # view from the robot's right side, like the video
        r.update_scene(data, cam)
        for p, rgba in ((hook[k], [0, 1, 0, 1]), (hand[k], [1, 0, 1, 0.8])):
            if r.scene.ngeom < r.scene.maxgeom:
                mujoco.mjv_initGeom(r.scene.geoms[r.scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    np.array([0.02, 0, 0]), p, np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
                r.scene.ngeom += 1
        img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(img, f"t={k / fps:5.2f}s  hook z={hz[k]:.2f}m", (10, 25), 0, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "green = hook throat, magenta = human R wrist (scaled)", (10, H - 12), 0, 0.5, (0, 0, 0), 1)
        if cap:
            cap.set(cv2.CAP_PROP_POS_MSEC, k / fps * 1000)
            ok, fr = cap.read()
            fr = cv2.resize(fr, (W, int(fr.shape[0] * W / fr.shape[1]))) if ok else np.zeros((H, W, 3), np.uint8)
            canvas = np.zeros((H, W, 3), np.uint8); y0 = (H - fr.shape[0]) // 2
            canvas[max(0, y0):max(0, y0) + min(H, fr.shape[0])] = fr[:min(H, fr.shape[0])]
            img = np.hstack([canvas, img])
        out.write(img)
    out.release()
    print("video:", d / f"gate2_side_by_side{tag}.mp4")


SOFT_CHECK = 0.9

if __name__ == "__main__":
    main()
