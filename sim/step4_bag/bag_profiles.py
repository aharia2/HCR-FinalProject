"""Bag geometry profiles shared by make_bag.py, hook_edit.py, the RPO-BagCarry task and the render scripts.

Select with the environment variable HCR_BAG (default "sim12"). Each profile has its own files, so profiles can
coexist: assets/bag{suffix}.urdf, bag_placement{suffix}.json, motion results IMG_0531_hook{suffix}/ and the robolab
motion bag_pickup_IMG_0531_hook{suffix}.npz.

Bag frame: origin on the floor under the box centre, x = handle bar axis (along the box width), z up.
  box      (width x, depth y, height z) [m]
  post_x   handle posts at x = ±post_x (centre lines)
  bar_r    radius of the posts and the bar
  bar_z    height of the bar centre above the floor
"""
import os

PROFILES = {
    # steps 4 v1–v8: stand-in bag, rigid 12 cm handle
    "sim12": dict(box=(0.30, 0.12, 0.24), post_x=0.06, bar_r=0.006, bar_z=0.362, suffix=""),
    # user's Whole Foods paper bag (2026-11-18): 35 cm tall x 30 cm wide, handle 9 cm wide x 9 cm high.
    # Depth not measured: 17 cm assumed (standard grocery bag). Handle modelled rigid (the real one is paper).
    "wf": dict(box=(0.30, 0.17, 0.35), post_x=0.045, bar_r=0.004, bar_z=0.35 + 0.09 - 0.004, suffix="_wf",
               pick_push=0.05, drop_reach=0.37, carry_out=0.035),     # bag placed further out: the taller/deeper box hit the thigh
}


def get(name=None):
    name = name or os.environ.get("HCR_BAG", "sim12")
    return dict(PROFILES[name], name=name)


def urdf(g, mass=1.0):
    bx, by, bz = g["box"]; px, r, zb = g["post_x"], g["bar_r"], g["bar_z"]
    ixx = mass / 12 * (by ** 2 + bz ** 2); iyy = mass / 12 * (bx ** 2 + bz ** 2); izz = mass / 12 * (bx ** 2 + by ** 2)
    pl = zb - bz
    cyl = lambda x, z, l, rpy="0 0 0": f'<origin xyz="{x} 0 {z}" rpy="{rpy}"/><geometry><cylinder radius="{r}" length="{l}"/></geometry>'
    parts = [cyl(-px, bz + pl / 2, pl), cyl(px, bz + pl / 2, pl), cyl(0, zb, 2 * px + 2 * r, "0 1.5707963 0")]
    geo = "".join(f"\n    <collision>{p}</collision>\n    <visual>{p}</visual>" for p in parts)
    return f"""<?xml version="1.0"?>
<!-- Shopping-bag model ({g['name']}) for the HCR bag-transfer task. Generated from sim/step4_bag/bag_profiles.py -->
<robot name="bag">
  <link name="bag">
    <inertial>
      <origin xyz="0 0 {bz / 2}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx:.6g}" ixy="0" ixz="0" iyy="{iyy:.6g}" iyz="0" izz="{izz:.6g}"/>
    </inertial>
    <visual><origin xyz="0 0 {bz / 2}"/><geometry><box size="{bx} {by} {bz}"/></geometry></visual>
    <collision><origin xyz="0 0 {bz / 2}"/><geometry><box size="{bx} {by} {bz}"/></geometry></collision>{geo}
  </link>
</robot>
"""


if __name__ == "__main__":
    import pathlib, sys
    g = get(sys.argv[1] if len(sys.argv) > 1 else None)
    out = pathlib.Path(__file__).resolve().parent / "assets" / f"bag{g['suffix']}.urdf"
    out.write_text(urdf(g)); print(out)
