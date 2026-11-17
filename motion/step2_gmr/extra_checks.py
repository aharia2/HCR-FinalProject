"""Step 2e: extra Gate-2 checks on robot_motion_clean.npz (added after review).

    python motion/step2_gmr/extra_checks.py motion/step2_gmr/results/IMG_0531

1. Hook orientation: the U is open toward the hook's +z. A handle can only fall out if gravity points out of
   the opening, i.e. if the hook's +z axis points DOWN in the world. Reports up = (hook z-axis) . world_up and
   the forearm (hook x-axis) elevation, from lift-off to set-down.
2. Balance: centre of mass (robot alone, and robot + 2 kg bag hanging at the hook throat) projected on the
   ground vs the support polygon of the foot-mesh points within 2 cm of the floor. Signed margin, + = inside.
   Plus foot tilt (ankle_roll_link z axis vs world z) while in contact.
3. Self-penetration: minimum distance between the right forearm/hook geoms and the right leg, pelvis and
   torso geoms (mj_geomDistance on convex hulls; < 0 means penetration).
Appends results to gate2_summary.json and writes gate2_extra.png.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_retarget import foot_points, load_model  # noqa: E402

THROAT = np.array([0.060, 0.0, -0.020])
BAG_KG = 2.0
ARM = ["right_elbow_pitch_link", "right_elbow_yaw_link", "right_hook_link"]
BODY = ["base_link", "torso_link", "right_thigh_yaw_link", "right_thigh_roll_link", "right_thigh_pitch_link",
        "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link", "left_thigh_pitch_link", "left_knee_link"]


def hull_margin(points_xy, p):
    """Signed distance from p to the convex hull boundary (+ inside)."""
    if len(points_xy) < 3:
        return -np.inf
    h = ConvexHull(points_xy)
    # each facet: n.x + b <= 0 inside (n is outward unit normal)
    return float(-np.max(h.equations[:, :2] @ p + h.equations[:, 2]))


def collision_geoms(model, bodies, hook_collision_only=True):
    out = []
    for g in range(model.ngeom):
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        if b not in bodies:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if b == "right_hook_link" and not name.startswith("right_hook_collision"):
            continue  # skip the visual-only hook mesh
        out.append(g)
    return out


def main(d):
    d = pathlib.Path(d)
    M = np.load(d / "robot_motion_clean.npz")
    fps, Q = float(M["fps"]), M["qpos"]
    T = len(Q); t = np.arange(T) / fps
    model = load_model(); data = mujoco.MjData(model)
    hid = model.body("right_hook_link").id
    robot_mass = float(model.body_subtreemass[0])
    arm_g, body_g = collision_geoms(model, ARM), collision_geoms(model, BODY)

    S = json.loads((d / "gate2_summary.json").read_text())
    k_pick, k_place = int(S["hook"]["pickup"]["t"] * fps), int(S["hook"]["place"]["t"] * fps)

    up, x_elev, com_m, com_bag_m, tilt_l, tilt_r, dmin = [], [], [], [], [], [], []
    for k in range(T):
        data.qpos[:] = Q[k]
        mujoco.mj_forward(model, data)
        R = data.xmat[hid].reshape(3, 3)
        up.append(R[2, 2]); x_elev.append(np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1))))
        throat = data.xpos[hid] + R @ THROAT
        com = data.subtree_com[0]
        com_bag = (com * robot_mass + throat * BAG_KG) / (robot_mass + BAG_KG)
        pts = np.vstack([foot_points(model, data, "left_ankle_roll_link"), foot_points(model, data, "right_ankle_roll_link")])
        contact = pts[pts[:, 2] < 0.02][:, :2]
        com_m.append(hull_margin(contact, com[:2]) if len(contact) >= 3 else np.nan)
        com_bag_m.append(hull_margin(contact, com_bag[:2]) if len(contact) >= 3 else np.nan)
        for side, store in (("left", tilt_l), ("right", tilt_r)):
            b = model.body(f"{side}_ankle_roll_link").id
            fp = foot_points(model, data, f"{side}_ankle_roll_link")
            store.append(np.degrees(np.arccos(np.clip(data.xmat[b].reshape(3, 3)[2, 2], -1, 1)))
                         if fp[:, 2].min() < 0.01 else np.nan)
        if k % 2 == 0:
            dd = [mujoco.mj_geomDistance(model, data, a, b, 0.2, None) for a in arm_g for b in body_g]
            dmin.append(min(dd))
        else:
            dmin.append(dmin[-1])
    up, x_elev = np.array(up), np.array(x_elev)
    com_m, com_bag_m, dmin = np.array(com_m), np.array(com_bag_m), np.array(dmin)
    tilt_l, tilt_r = np.array(tilt_l), np.array(tilt_r)

    carry = slice(k_pick, k_place + 1)
    squat = Q[:, 2] < 0.55                       # pelvis low = squats (standing ~0.75 m)
    stance2 = np.isfinite(tilt_l) & np.isfinite(tilt_r)
    extra = dict(
        hook_orientation=dict(
            note="up = hook +z . world up; handle can fall out only if up < 0",
            carry_window_s=[round(k_pick / fps, 2), round(k_place / fps, 2)],
            up_min_carry=round(float(up[carry].min()), 3), up_median_carry=round(float(np.median(up[carry])), 3),
            forearm_elevation_deg_carry=[round(float(x_elev[carry].min()), 1), round(float(x_elev[carry].max()), 1)],
            up_at_pickup=round(float(up[k_pick]), 3), up_at_place=round(float(up[k_place]), 3)),
        balance=dict(
            robot_mass_kg=round(robot_mass, 2), bag_kg=BAG_KG,
            note="margin = distance of COM ground projection inside support polygon (+ inside), metres",
            squat_frames=int(squat.sum()),
            com_margin_squat_min_m=round(float(np.nanmin(com_m[squat])), 3),
            com_margin_squat_median_m=round(float(np.nanmedian(com_m[squat])), 3),
            com_with_bag_margin_squat_min_m=round(float(np.nanmin(com_bag_m[squat])), 3),
            com_with_bag_margin_carry_double_support_min_m=round(float(np.nanmin(com_bag_m[carry][stance2[carry]])), 3),
            frac_frames_com_outside_double_support=round(float(np.mean(com_m[stance2] < 0)), 3),
            foot_tilt_in_contact_deg_max=dict(left=round(float(np.nanmax(tilt_l)), 1), right=round(float(np.nanmax(tilt_r)), 1)),
            foot_tilt_in_contact_deg_p95=dict(left=round(float(np.nanpercentile(tilt_l, 95)), 1), right=round(float(np.nanpercentile(tilt_r, 95)), 1))),
        self_collision=dict(
            pairs="right elbow/forearm/hook collision geoms vs pelvis, torso, right leg, left thigh/knee",
            min_distance_m=round(float(dmin.min()), 4), at_t_s=round(float(t[int(np.argmin(dmin))]), 2),
            min_distance_squats_m=round(float(dmin[squat].min()), 4),
            frames_penetrating=int((dmin < 0).sum())),
    )
    S.update(extra)
    (d / "gate2_summary.json").write_text(json.dumps(S, indent=2))

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    for a in ax.flat:
        a.axvspan(k_pick / fps, k_place / fps, color="g", alpha=0.08)
    a = ax[0, 0]; a.plot(t, up, "g"); a.axhline(0, color="r")
    a.set_title("Hook opening: hook +z . world up  (<0 = handle can fall out; green band = carrying)")
    a2 = a.twinx(); a2.plot(t, x_elev, "k:", lw=0.8); a2.set_ylabel("forearm elevation [deg] (dotted)")
    a = ax[0, 1]; a.plot(t, com_m, "b", lw=0.8, label="robot COM"); a.plot(t, com_bag_m, "m", lw=0.8, label=f"robot + {BAG_KG:g} kg bag")
    a.axhline(0, color="r"); a.set_ylim(-0.15, 0.15); a.legend(fontsize=8)
    a.set_title("COM margin inside support polygon [m] (<0 outside)")
    a = ax[1, 0]; a.plot(t, tilt_l, "b.", ms=2, label="left"); a.plot(t, tilt_r, "r.", ms=2, label="right")
    a.set_title("Foot tilt while in contact [deg]"); a.legend(fontsize=8)
    a = ax[1, 1]; a.plot(t, dmin * 100, "k"); a.axhline(0, color="r")
    a.set_title("Min distance right arm/hook to body/legs [cm] (<0 = penetration)")
    for a in ax.flat:
        a.set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(d / "gate2_extra.png", dpi=90)
    print(json.dumps(extra, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
