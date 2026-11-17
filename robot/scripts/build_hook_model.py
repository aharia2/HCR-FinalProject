#!/usr/bin/env python3
"""Build Roboto Origin (RPO) URDF + MJCF with the 3D-printed hook on the RIGHT wrist.

Inputs  (relative to robot/):
  urdf/rpo.urdf, mjcf/rpo.xml          unmodified upstream Roboparty files
  ../roboto-origin-v2-right-square-hook.stl   team hook CAD (mm, origin at a corner)
Outputs:
  meshes/right_hook_link.STL           hook visual mesh, metres, in right_hook_link frame
  meshes/right_hook_collision_*.STL    convex pieces (CoACD) for collision
  urdf/rpo_hook.urdf, mjcf/rpo_hook.xml

Requires: numpy, trimesh, coacd   (mujoco optional, for the load check at the end)
Run from anywhere:  python robot/scripts/build_hook_model.py
"""
from pathlib import Path
import re

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]          # robot/
HOOK_STL = ROOT.parent / "roboto-origin-v2-right-square-hook.stl"
MESH_DIR = ROOT / "meshes"

# ---------------------------------------------------------------- geometry ---
# Mounting (measured from the two meshes):
#  * Hook STL: mount axis is +X, centred at (y, z) = (40.0, 19.75) mm. A 36 mm boss
#    spans x = 0..7 mm; its x = 0 face is the mating face.
#  * Forearm (right_elbow_yaw_link): Ø37.2 mm bore, floor at x = 132.75 mm, axis = link X.
#  * 6x M3 on a Ø27.5 mm circle. Hook holes are at 30, 90, ... deg; forearm holes at
#    0, 60, ... deg, so a +90 deg roll lines the holes up AND makes the hook open
#    toward +Z (up) at the zero pose.
MOUNT_XYZ = (0.13275, 0.0, 0.0)                     # joint origin in right_elbow_yaw_link
HOOK_AXIS_CENTER_MM = np.array([0.0, 40.0, 19.75])  # point on hook STL that maps to the joint origin
ROLL_DEG = 90.0                                     # rotation about the forearm X axis

# ------------------------------------------------------------------- mass ----
PLA_DENSITY = 1240.0        # kg/m^3. Assumes a SOLID (100% infill) print. Lower infill means less mass.
N_SCREWS = 6
SCREW_MASS = 0.00106        # kg, M3x12 steel socket-head cap screw (approx.)
SCREW_PCD_R = 0.01375       # m, bolt-circle radius
SCREW_COM_X = 0.0043        # m, along link X (head sits at x = 7..10 mm, shank runs back into the forearm)
SCREW_ANGLES_DEG = [0, 60, 120, 180, 240, 300]      # in the link frame, measured from +Y toward +Z

# CoACD convex decomposition. Fewer pieces = faster sim. 10 pieces fill the screw holes
# (~10% extra volume) but keep the hook throat open. One convex hull would be 2.6x the volume.
COACD_THRESHOLD = 0.08
COACD_MAX_HULLS = 10
# Collision geometry only covers the hook for x >= this (m, link frame). Reason (2026-11-17): the training URDF
# gives the forearm its collision capsule on right_elbow_pitch_link (robolab turns the cylinder into a capsule),
# and that capsule's end reaches 3.1 cm into the hook's flange in EVERY pose. If the simulator keeps the hook as
# its own body, that would be a permanent self-collision. The U (x = 0.045-0.09) and the throat's back wall
# (diagonal) are what touch a bag handle; the flange and the straight bar from the mount are not. 0.036 still
# overlapped by 1 mm; 0.040 gives 3 mm clearance and keeps >= 0.8 cm free space at the handle seat.
COLLISION_MIN_X = 0.040


def rot_x(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def build_meshes():
    m = trimesh.load(HOOK_STL, force="mesh")
    assert m.is_watertight, "hook STL not watertight; mass properties would be wrong"
    T = np.eye(4)
    T[:3, :3] = rot_x(ROLL_DEG)
    T[:3, 3] = -rot_x(ROLL_DEG) @ (HOOK_AXIS_CENTER_MM * 1e-3)
    m.apply_scale(1e-3)
    m.apply_transform(T)
    m.export(MESH_DIR / "right_hook_link.STL")

    # --- inertial: printed body + screws (point masses) ---
    m.density = PLA_DENSITY
    mb, cb, Ib = m.mass, m.center_mass, m.moment_inertia   # Ib about cb, in link axes
    pts = [np.array([SCREW_COM_X, SCREW_PCD_R * np.cos(np.radians(a)), SCREW_PCD_R * np.sin(np.radians(a))])
           for a in SCREW_ANGLES_DEG[:N_SCREWS]]
    masses = [mb] + [SCREW_MASS] * N_SCREWS
    coms = [cb] + pts
    M = sum(masses)
    com = sum(mi * ci for mi, ci in zip(masses, coms)) / M

    def parallel(I, mi, d):
        return I + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    I = parallel(Ib, mb, cb - com)
    for p in pts:
        I = parallel(np.zeros((3, 3)), SCREW_MASS, p - com) + I

    # --- collision: convex decomposition so the hook throat stays open ---
    import coacd
    coacd.set_log_level("error")
    cm = m.slice_plane([COLLISION_MIN_X, 0, 0], [1, 0, 0], cap=True)   # needs shapely + mapbox_earcut
    assert cm.is_watertight
    parts = coacd.run_coacd(coacd.Mesh(cm.vertices, cm.faces), threshold=COACD_THRESHOLD,
                             max_convex_hull=COACD_MAX_HULLS, seed=0)
    for old in MESH_DIR.glob("right_hook_collision_*.STL"):
        old.unlink()
    names = []
    for i, (v, f) in enumerate(parts):
        n = f"right_hook_collision_{i}"
        trimesh.Trimesh(v, f).export(MESH_DIR / f"{n}.STL")
        names.append(n)
    return dict(mass_body=mb, mass=M, com=com, I=I, collision=names, bounds=m.bounds)


def f(x):
    return f"{x:.8g}"


def write_urdf(p):
    src = (ROOT / "urdf" / "rpo.urdf").read_text()
    I, c = p["I"], p["com"]
    cols = "\n".join(
        f"""    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../meshes/{n}.STL"/>
      </geometry>
    </collision>""" for n in p["collision"])
    block = f"""
  <!-- ===== Team hook (3D-printed PLA, 6x M3) on the RIGHT wrist; generated by scripts/build_hook_model.py ===== -->
  <link name="right_hook_link">
    <inertial>
      <origin xyz="{f(c[0])} {f(c[1])} {f(c[2])}" rpy="0 0 0"/>
      <mass value="{f(p['mass'])}"/>
      <inertia ixx="{f(I[0,0])}" ixy="{f(I[0,1])}" ixz="{f(I[0,2])}" iyy="{f(I[1,1])}" iyz="{f(I[1,2])}" izz="{f(I[2,2])}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="../meshes/right_hook_link.STL"/>
      </geometry>
      <material name="hook_pla">
        <color rgba="0.95 0.55 0.15 1"/>
      </material>
    </visual>
{cols}
  </link>
  <joint name="right_hook_joint" type="fixed">
    <origin xyz="{MOUNT_XYZ[0]} {MOUNT_XYZ[1]} {MOUNT_XYZ[2]}" rpy="0 0 0"/>
    <parent link="right_elbow_yaw_link"/>
    <child link="right_hook_link"/>
  </joint>
"""
    out = re.sub(r'<robot name="([^"]+)"', r'<robot name="\1_hook"', src, count=1)
    i = out.rstrip().rfind("</robot>")
    out = out[:i] + block + "</robot>\n"
    (ROOT / "urdf" / "rpo_hook.urdf").write_text(out)


def sync_joint_limits_from_urdf(mjcf_text):
    """Upstream's MJCF and URDF disagree on 16 of 23 joint ranges (e.g. elbow pitch 3.57 vs 1.57 rad).
    Decision (2026-11-16): the URDF is the source of truth, because robolab trains on it
    (soft limit = 0.9 x URDF range). Overwrite every MJCF joint range with the URDF value."""
    import xml.etree.ElementTree as ET
    limits = {j.get("name"): (j.find("limit").get("lower"), j.find("limit").get("upper"))
              for j in ET.parse(ROOT / "urdf" / "rpo.urdf").getroot().findall("joint")
              if j.find("limit") is not None and j.get("type") == "revolute"}
    changed = 0
    for name, (lo, hi) in limits.items():
        pattern = rf'(<joint name="{re.escape(name)}"[^>]*?range=")[^"]*(")'
        mjcf_text, n = re.subn(pattern, rf"\g<1>{float(lo):g} {float(hi):g}\g<2>", mjcf_text)
        assert n == 1, f"joint {name} not found exactly once in MJCF"
        changed += n
    assert changed == 23, changed
    return mjcf_text


def write_mjcf(p):
    src = (ROOT / "mjcf" / "rpo.xml").read_text()
    I, c = p["I"], p["com"]
    # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz
    full = " ".join(f(v) for v in (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))
    assets = "\n".join([f'    <mesh name="right_hook_link" file="right_hook_link.STL"/>'] +
                       [f'    <mesh name="{n}" file="{n}.STL"/>' for n in p["collision"]])
    geoms = "\n".join(
        f'                    <geom name="{n}" type="mesh" mesh="{n}" group="3" rgba="0.95 0.55 0.15 0.4"/>'
        for n in p["collision"])
    body = f"""
                  <!-- Team hook on the RIGHT wrist; generated by scripts/build_hook_model.py -->
                  <body name="right_hook_link" pos="{MOUNT_XYZ[0]} {MOUNT_XYZ[1]} {MOUNT_XYZ[2]}">
                    <inertial pos="{f(c[0])} {f(c[1])} {f(c[2])}" mass="{f(p['mass'])}" fullinertia="{full}"/>
                    <geom name="right_hook_visual" type="mesh" mesh="right_hook_link" contype="0" conaffinity="0" group="1" rgba="0.95 0.55 0.15 1"/>
{geoms}
                  </body>"""
    out = src.replace('<mujoco model="rpo">', '<mujoco model="rpo_hook">', 1)
    out = sync_joint_limits_from_urdf(out)
    anchor = '<mesh name="right_elbow_yaw_link" file="right_elbow_yaw_link.STL"/>'
    assert out.count(anchor) == 1
    out = out.replace(anchor, anchor + "\n" + assets)
    geom_anchor = '<geom type="mesh" rgba="0.752941 0.752941 0.752941 1" mesh="right_elbow_yaw_link"/>'
    assert out.count(geom_anchor) == 1
    out = out.replace(geom_anchor, geom_anchor + body)
    (ROOT / "mjcf" / "rpo_hook.xml").write_text(out)


def check_mujoco():
    try:
        import mujoco
    except ImportError:
        print("mujoco not installed; skipped load check")
        return
    for path in [ROOT / "mjcf" / "rpo_hook.xml", ROOT / "urdf" / "rpo_hook.urdf"]:
        if path.suffix == ".urdf":   # MuJoCo resolves URDF mesh names by basename via meshdir
            xml = re.sub(r'filename="\.\./meshes/', 'filename="', path.read_text())
            xml = re.sub(r'(<robot [^>]*>)', rf'\1<mujoco><compiler meshdir="{MESH_DIR}" discardvisual="false" fusestatic="false"/></mujoco>',
                         xml, count=1)
            model = mujoco.MjModel.from_xml_string(xml)
        else:
            model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        b = model.body("right_hook_link").id
        print(f"{path.name}: loads OK, nbody={model.nbody}, hook world pos={data.xpos[b].round(4)}, "
              f"hook subtree mass={model.body_mass[b]:.4f} kg")


if __name__ == "__main__":
    p = build_meshes()
    write_urdf(p)
    write_mjcf(p)
    print(f"hook mass: printed body {p['mass_body']*1000:.1f} g + screws "
          f"{N_SCREWS*SCREW_MASS*1000:.1f} g = {p['mass']*1000:.1f} g")
    print("COM (link frame, m):", p["com"].round(5))
    print("inertia (kg m^2):\n", p["I"])
    print("bounds (link frame, m):\n", p["bounds"].round(4))
    print("collision pieces:", len(p["collision"]))
    check_mujoco()
