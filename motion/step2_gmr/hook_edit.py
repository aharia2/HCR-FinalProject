"""Step 2e (added during step 4): edit the reference motion so the HOOK can engage and release the handle.

    MUJOCO_GL=egl ~/HCR/third_party/.venv-gmr/bin/python motion/step2_gmr/hook_edit.py

Why: the human in the video grasps the handle and opens the hand to let go. The hook (a "C" opening up, tip ≈ 2.3 cm
above the seat, profile in the forearm x-z plane) can do neither: in the retargeted motion it comes straight down
onto the handle and never leaves it. Step 4 v1–v5 (reward shaping only) got 0 pickups and 0 releases.

Edit (only the waist + right arm move; base, legs and left arm keep the reference): IK on the hook throat in two
windows, everything else unchanged.
  pickup : reference -> BEHIND the handle, 5 cm below the bar (tip passes under it) -> slide forward under the bar
           -> rise until the bar sits in the throat (lift) -> blend back into the reference lift
  release: lower until the bag stands on the floor -> 3 cm further (bar leaves the seat) -> pull back along the
           forearm until the tip clears the bar -> blend back into the reference
The bag placement (pickup handle, set-down spot) is chosen from the same plan, so the edited reference threads the
handle by construction, and is written to sim/step4_bag/bag_placement.json (frames updated).

Outputs: motion/step2_gmr/results/IMG_0531_hook/robot_motion_clean.npz (same format as step 2; convert with
to_robolab.py), sim/step4_bag/bag_placement.json, and a check report (IK residual, joint limits, robot-vs-bag
clearance along the edited motion).
"""
import json
import pathlib
import shutil
import sys

import mink
import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "motion/step2_gmr/results/IMG_0531"
sys.path.insert(0, str(REPO / "sim/step4_bag"))
import bag_profiles  # noqa: E402
BAG = bag_profiles.get()                       # HCR_BAG=sim12 (default) | wf
DST = REPO / f"motion/step2_gmr/results/IMG_0531_hook{BAG['suffix']}"
PLACEMENT = REPO / f"sim/step4_bag/bag_placement{BAG['suffix']}.json"
BASE_PLACEMENT = REPO / "sim/step4_bag/bag_placement.json"     # hook / fps fields; geometry replaced per profile

THROAT = np.array([0.19275, 0.0, -0.020])     # handle seat in right_elbow_yaw_link
BOX = tuple(BAG["box"])
POST_X, BAR_R, BAR_Z = BAG["post_x"], BAG["bar_r"], BAG["bar_z"]
SEAT = -0.021                                  # throat z relative to the bar when the bar rests in the seat
UNDER = -0.05                                  # throat 5 cm under the bar: hook tip (+2.3 cm) clears the bar (r 6 mm)
BACK_PICK, BACK_REL = 0.10, 0.09               # how far behind the bar the hook starts / ends (along the forearm)
XFADE = 30                                     # frames of joint crossfade at the end of each edit
SOFT = 0.9                                     # robolab soft joint-limit factor (reference stays inside it)

# Time warp: in the reference the body stands up right after the set-down squat (hips +28 cm in 0.5 s), so the hook
# could not stay down to unhook. Frames [WARP_A, WARP_B) (the bottom of the squat) are played WARP_X times slower.
WARP_A, WARP_B, WARP_X = 655, 675, 4
WARP_ADD = (WARP_B - WARP_A) * (WARP_X - 1)    # +60 frames -> 911 frames, 18.2 s

# phase frames (50 Hz, warped timeline); reference throat minimum at pickup 254 (before the warp, unchanged)
PICK = dict(start=204, behind=238, under=258, lift=272, rejoin=310)
REL = dict(start=615, touch=672, drop=692, out=715, up=735, rejoin=770)
DROP_REACH = BAG.get("drop_reach", 0.33)       # set-down handle: this far in front of the shoulder (horizontal)
CARRY_OUT = BAG.get("carry_out", 0.0)          # carry: hook this far further OUT to the robot's right (bag clears thigh)
CARRY_RAMP = 30
PICK_PUSH = BAG.get("pick_push", 0.0)          # pickup handle moved this far further out along the forearm (thigh room)

FREE = ["torso_joint", "right_arm_pitch_joint", "right_arm_roll_joint", "right_arm_yaw_joint",
        "right_elbow_pitch_joint", "right_elbow_yaw_joint"]


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def build_model(with_bag=None):
    xml = (REPO / "robot/mjcf/rpo_hook.xml").read_text().replace('meshdir="../meshes/"', f'meshdir="{REPO / "robot/meshes"}/"')
    spec = mujoco.MjSpec.from_string(xml)
    spec.body("right_elbow_yaw_link").add_site(name="throat", pos=THROAT.tolist())
    if with_bag is not None:                   # static bag geoms for the clearance check
        pos, yaw = with_bag
        b = spec.worldbody.add_body(name="bag", pos=pos, quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        bx, by, bz = BOX
        b.add_geom(name="bag_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[bx / 2, by / 2, bz / 2], pos=[0, 0, bz / 2])
        for sx in (-1, 1):
            b.add_geom(name=f"bag_post{sx:+d}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       fromto=[sx * POST_X, 0, bz, sx * POST_X, 0, BAR_Z], size=[BAR_R, 0, 0])
        b.add_geom(name="bag_bar", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                   fromto=[-POST_X, 0, BAR_Z, POST_X, 0, BAR_Z], size=[BAR_R, 0, 0])
    return spec.compile()


def horiz(v):
    v = np.array(v, dtype=float); v[2] = 0
    return v / np.linalg.norm(v)


def warp(M):
    """Slow down [WARP_A, WARP_B) by WARP_X: linear interpolation (root quaternion: normalized lerp)."""
    T0 = len(M["qpos"])
    t_new = np.concatenate([np.arange(WARP_A), WARP_A + np.arange((WARP_B - WARP_A) * WARP_X) / WARP_X,
                            np.arange(WARP_B, T0)])
    out = {}
    for k in M.files:
        a = M[k]
        if not (isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[0] == T0 and a.dtype.kind == "f"):
            out[k] = a; continue
        i0 = np.floor(t_new).astype(int); i1 = np.minimum(i0 + 1, T0 - 1); w = (t_new - i0).reshape(-1, *[1] * (a.ndim - 1))
        b = a[i0] * (1 - w) + a[i1] * w
        if k == "qpos":
            q0, q1 = a[i0, 3:7], a[i1, 3:7]
            q1 = np.where((np.sum(q0 * q1, -1) < 0)[:, None], -q1, q1)
            qq = q0 * (1 - w) + q1 * w; b[:, 3:7] = qq / np.linalg.norm(qq, axis=-1, keepdims=True)
        out[k] = b
    return out


def main():
    M = warp(np.load(SRC / "robot_motion_clean.npz"))
    Q = M["qpos"].copy()
    names = [str(n) for n in M["joint_names"]]
    m = build_model(); d = mujoco.MjData(m)
    site, fa = m.site("throat").id, m.body("right_elbow_yaw_link").id

    def fk(q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        return d.site_xpos[site].copy(), d.xmat[fa].reshape(3, 3).copy()

    P, R = zip(*(fk(q) for q in Q)); P, R = np.array(P), np.array(R)

    # --- handle placement from the plan: bar axis = forearm y (the hook's C lies in the forearm x-z plane) ---
    def anchor(k):
        bar = np.array([P[k, 0], P[k, 1], BAR_Z]); u = horiz(R[k][:, 0]); ax = horiz(R[k][:, 1])
        return bar, u, ax
    bar_p, u_p, ax_p = anchor(254)
    bar_p = bar_p + PICK_PUSH * u_p
    _, u_d, ax_d = anchor(REL["touch"])
    d.qpos[:] = Q[REL["touch"]]; mujoco.mj_kinematics(m, d)
    sh = d.xpos[m.body("right_arm_pitch_link").id].copy()
    bar_d = np.array([sh[0], sh[1], BAR_Z]) + DROP_REACH * u_d     # reachable: in front of the shoulder

    # --- throat targets ---
    T = len(Q); tgt = P.copy()
    z = np.array([0, 0, 1.0])
    wp_pick = {PICK["behind"]: bar_p + UNDER * z - BACK_PICK * u_p,
               PICK["under"]: bar_p + UNDER * z + 0.005 * u_p,
               PICK["lift"]: bar_p + SEAT * z}
    wp_rel = {REL["touch"]: bar_d + SEAT * z,
              REL["drop"]: bar_d + UNDER * z,
              REL["out"]: bar_d + UNDER * z - BACK_REL * u_d,
              REL["up"]: bar_d + 0.10 * z - BACK_REL * u_d}          # rise BEHIND the handle before rejoining

    def fill(start, wps, rejoin):
        ks = sorted(wps)
        for k in range(start, ks[0] + 1):                       # blend from the reference into the first waypoint
            s = minjerk((k - start) / (ks[0] - start)); tgt[k] = (1 - s) * P[k] + s * wps[ks[0]]
        for a, b in zip(ks[:-1], ks[1:]):                        # waypoint to waypoint
            for k in range(a, b + 1):
                s = minjerk((k - a) / (b - a)); tgt[k] = (1 - s) * wps[a] + s * wps[b]
        off = wps[ks[-1]] - P[ks[-1]]                            # blend back: decaying offset on the reference
        for k in range(ks[-1], rejoin + 1):
            s = minjerk((k - ks[-1]) / (rejoin - ks[-1])); tgt[k] = P[k] + (1 - s) * off
    fill(PICK["start"], wp_pick, PICK["rejoin"])
    fill(REL["start"], wp_rel, REL["rejoin"])
    edit = np.zeros(T, bool); edit[PICK["start"]:PICK["rejoin"] + 1] = True; edit[REL["start"]:REL["rejoin"] + 1] = True
    if CARRY_OUT != 0:   # taller bag profiles: from the lift to the set-down, shift the hook sideways (+: base +y side)
        a, b = PICK["lift"] + 1, REL["start"]                    # starts AT the lift (thigh is close while standing up)
        base_id = m.body("base_link").id
        for k in range(a, b):
            d.qpos[:] = Q[k]; mujoco.mj_kinematics(m, d)
            side = horiz(d.xmat[base_id].reshape(3, 3)[:, 1])
            s_ = minjerk(min(k - a, b - 1 - k) / CARRY_RAMP)
            tgt[k] = tgt[k] + s_ * CARRY_OUT * side              # on top of the pickup blend-back (tgt = P after it)
        edit[a:b] = True

    # --- IK: only waist + right arm move; joint limits shrunk to the soft limits ---
    lo, hi = m.jnt_range[:, 0].copy(), m.jnt_range[:, 1].copy()
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    m.jnt_range[1:, 0] = (mid - SOFT * half)[1:]; m.jnt_range[1:, 1] = (mid + SOFT * half)[1:]
    cfg = mink.Configuration(m)
    free_dofs = [m.jnt_dofadr[m.joint(n).id] for n in FREE]
    cost = np.full(m.nv, 1e4); cost[:6] = 1e4
    cost[free_dofs] = 0.5; cost[m.jnt_dofadr[m.joint("torso_joint").id]] = 2.0   # free joints: stay near the PREVIOUS frame
    posture = mink.PostureTask(m, cost=cost)
    base = mink.FrameTask("base_link", "body", position_cost=1e4, orientation_cost=1e4)
    hook = mink.FrameTask("throat", "site", position_cost=50.0, orientation_cost=0.5)
    limits = [mink.ConfigurationLimit(m)]
    Qe = Q.copy(); res = np.zeros(T); q_prev = None
    fixed_q = [int(m.jnt_qposadr[j]) for j in range(1, m.njnt)
               if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) not in FREE]   # hinge joints that must not move
    for k in np.flatnonzero(edit):
        q0 = Q[k].copy()
        q = (q_prev if q_prev is not None and edit[k - 1] else q0).copy()
        q[:7] = q0[:7]; q[fixed_q] = q0[fixed_q]
        cfg.update(q)
        pt = q0.copy()
        if q_prev is not None and edit[k - 1]:
            fq = [int(m.jnt_qposadr[m.joint(n).id]) for n in FREE]; pt[fq] = q_prev[fq]   # continuity (no IK branch flips)
        posture.set_target(pt)
        base.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(wxyz=np.array(q0[3:7])), q0[:3]))
        hook.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(R[k]), tgt[k]))
        for _ in range(60):
            v = mink.solve_ik(cfg, [hook, posture, base], 0.02, "daqp", damping=1e-3, limits=limits)
            cfg.integrate_inplace(v, 0.02)
        q = cfg.q.copy(); q[:7] = q0[:7]; q[fixed_q] = q0[fixed_q]
        Qe[k] = q; q_prev = Qe[k]
        res[k] = np.linalg.norm(fk(Qe[k])[0] - tgt[k])
    # joint-space crossfade back to the reference at the end of each edit (the IK may sit on another arm branch
    # with the same hook position, which would otherwise snap at the rejoin frame)
    for end in ((REL["rejoin"],) if CARRY_OUT != 0 else (PICK["rejoin"], REL["rejoin"])):   # no pickup rejoin with a carry edit
        for k in range(end - XFADE, end + 1):
            s_ = minjerk((k - (end - XFADE)) / XFADE); Qe[k, 7:] = (1 - s_) * Qe[k, 7:] + s_ * Q[k, 7:]
            res[k] = np.linalg.norm(fk(Qe[k])[0] - tgt[k])
    moved = np.abs(Qe - Q).max(0)
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    moved_names = sorted({jn[np.searchsorted(m.jnt_qposadr, i, side="right") - 1] for i in np.flatnonzero(moved > 1e-6)})

    # --- checks ---
    rep = dict(ik_residual_m=dict(max=float(res.max()), p95=float(np.quantile(res[edit], 0.95)),
                                  at={str(k): float(res[k]) for k in sorted({**wp_pick, **wp_rel})}),
               joints_changed=moved_names)
    vel = np.abs(np.diff(Qe[:, 7:], axis=0)) * 50
    ev = edit[1:]
    rep["max_joint_speed_rad_s"] = dict(pickup_edit=float(vel[PICK["start"]:PICK["rejoin"]].max()),
                                        release_edit=float(vel[REL["start"]:REL["rejoin"]].max()),
                                        whole_edited_clip=float(vel.max()),
                                        original_whole_clip=float(np.abs(np.diff(Q[:, 7:], axis=0)).max() * 50))
    rep["frames_total"] = int(T)
    rep["ik_residual_trace_release"] = {str(k): round(float(res[k]), 4) for k in range(REL["start"], REL["rejoin"] + 1, 5)}
    rep["within_soft_limits"] = bool(((Qe[:, 7:] >= m.jnt_range[1:, 0] - 1e-6) & (Qe[:, 7:] <= m.jnt_range[1:, 1] + 1e-6)).all())

    def clearance(bag_pose, frames):
        mb = build_model(bag_pose); db = mujoco.MjData(mb)
        bag_g = [g for g in range(mb.ngeom) if mujoco.mj_id2name(mb, mujoco.mjtObj.mjOBJ_GEOM, g) and
                 mujoco.mj_id2name(mb, mujoco.mjtObj.mjOBJ_GEOM, g).startswith("bag_")]
        rob_g = [g for g in range(mb.ngeom) if mb.geom_bodyid[g] != mb.body("bag").id and mb.geom_bodyid[g] != 0
                 and mb.geom_contype[g] + mb.geom_conaffinity[g] > 0]
        out = {}
        for k in frames:
            db.qpos[:len(Qe[k])] = Qe[k]; mujoco.mj_kinematics(mb, db)
            best = (9.0, "", "")
            for bg in bag_g:
                for rg in rob_g:
                    dist = mujoco.mj_geomDistance(mb, db, bg, rg, 0.3, None)
                    if dist < best[0]:
                        best = (dist, mujoco.mj_id2name(mb, mujoco.mjtObj.mjOBJ_GEOM, bg),
                                mujoco.mj_id2name(mb, mujoco.mjtObj.mjOBJ_BODY, mb.geom_bodyid[rg]))
            out[k] = best
        return out
    yaw_p = float(np.arctan2(ax_p[1], ax_p[0])); yaw_d = float(np.arctan2(ax_d[1], ax_d[0]))
    start_pos = [float(bar_p[0]), float(bar_p[1]), 0.0]; drop_pos = [float(bar_d[0]), float(bar_d[1]), 0.0]
    cp = clearance((start_pos, yaw_p), range(150, PICK["lift"] + 1, 2))
    cd = clearance((drop_pos, yaw_d), range(REL["drop"], REL["rejoin"] + 30, 2))
    rep["pickup_min_clearance"] = {str(k): [round(v[0], 4), v[1], v[2]] for k, v in cp.items() if k % 10 == 0 or v[0] < 0.005}
    rep["release_min_clearance"] = {str(k): [round(v[0], 4), v[1], v[2]] for k, v in cd.items() if k % 10 == 0 or v[0] < 0.005}
    rep["pickup_worst"] = min(((v[0], k, v[1], v[2]) for k, v in cp.items() if k < PICK["lift"] - 4))
    rep["release_worst"] = min(((v[0], k, v[1], v[2]) for k, v in cd.items() if k > REL["drop"] + 4))

    # --- write ---
    DST.mkdir(exist_ok=True)
    np.savez_compressed(DST / "robot_motion_clean.npz", **{**M, "qpos": Qe})
    if BAG["name"] == "sim12" and PLACEMENT.exists() and not (PLACEMENT.parent / "bag_placement_v1.json").exists():
        shutil.copy(PLACEMENT, PLACEMENT.parent / "bag_placement_v1.json")
    old = json.loads(BASE_PLACEMENT.read_text())
    old["geometry"] = dict(box=list(BOX), post_x=POST_X, bar_r=BAR_R, bar_z=BAR_Z, profile=BAG["name"])
    frames = dict(pickup_min=PICK["under"], lift=PICK["lift"], place_min=REL["touch"], release=REL["out"], total=T)
    new = dict(old, note="v2 (hook-edited reference, motion/step2_gmr/hook_edit.py): motion frame (add env origin); "
                          "yaw about +z of the bag x axis (= handle bar = forearm y at pickup/set-down)",
               frames=frames, times_s={k: round(v / 50, 2) for k, v in frames.items() if k != "total"},
               start=dict(pos=start_pos, yaw=yaw_p), drop=dict(pos=drop_pos, yaw=yaw_d),
               reference_throat_z=dict(pickup=float(BAR_Z + SEAT), place=float(BAR_Z + SEAT)),
               hook_edit=dict(PICK=PICK, REL=REL, UNDER=UNDER, SEAT=SEAT, BACK_PICK=BACK_PICK, BACK_REL=BACK_REL))
    PLACEMENT.write_text(json.dumps(new, indent=2))
    (DST / "hook_edit_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
