# Roboto Origin + right-wrist hook

The Roboto Origin (RPO) robot model with the team's 3D-printed hook
(`../roboto-origin-v2-right-square-hook.stl`) fixed to the **right** wrist. The left arm is unchanged.

| File | What |
|---|---|
| `urdf/rpo_hook.urdf` | URDF with the hook (**use this**) |
| `mjcf/rpo_hook.xml` | MuJoCo model with the hook (**use this**) |
| `urdf/rpo.urdf`, `mjcf/rpo.xml` | Unmodified upstream files, kept for diffing and regenerating |
| `meshes/` | Upstream meshes, plus `right_hook_link.STL` (visual) and `right_hook_collision_*.STL` (collision) |
| `scripts/build_hook_model.py` | Regenerates the hook meshes and both model files |

Upstream source: [Roboparty/roboto_origin](https://github.com/Roboparty/roboto_origin) `modules/rpo_description`
at commit `d94afba45ef029c5f77a268751ee638cc64c84d0`. Licensed CERN-OHL-W-2.0 (see `LICENSE`).

## How the hook is attached

- New link `right_hook_link` on a **fixed** joint `right_hook_joint`. The parent is `right_elbow_yaw_link`
  (the forearm) and the joint origin is `xyz = 0.13275 0 0`. The link axes match the forearm axes.
- The mount was measured from both meshes. The hook's Ø36 mm boss sits in the forearm's Ø37.2 mm bore,
  and the hook's back face rests on the bore floor at x = 132.75 mm. 6× M3 screws on a Ø27.5 mm bolt circle
  hold it on.
- **Orientation:** the hook is rolled 90° about the forearm axis, so at the zero pose (forearm level and
  pointing forward) the hook **opens upward (+Z)**. This roll also lines up the hook's holes with the
  forearm's 60° hole pattern.

## Mass and collision

- Mass is **60.3 g**: 53.9 g of PLA (43.5 cm³ × 1.24 g/cm³, **assuming 100 % infill**) plus 6.4 g for
  6× M3×12 steel socket-head screws. Inertia is computed from the mesh, with the screws added as point masses.
  If you print at lower infill or weigh the real part, change `PLA_DENSITY` / `SCREW_MASS` and re-run the script.
- **Collision:** the hook mesh is split into 10 convex pieces with CoACD. This matters because MuJoCo and
  most simulators turn a single collision mesh into a convex hull, which would fill in the hook's opening
  and make it impossible to catch a bag handle. In MuJoCo the collision pieces are in geom group 3
  (hidden by default) and the visual mesh is in group 1.
- Collision filtering follows upstream: robot geoms use `contype=1 conaffinity=0`, so a bag only touches
  the hook if the bag's geoms have `conaffinity` bit 1 set.

## Regenerating

```bash
pip install numpy trimesh coacd mujoco
python scripts/build_hook_model.py
```

## Notes

- **Joint limits:** upstream's URDF and MJCF disagree on 16 of 23 joints. For example,
  `right_elbow_pitch_joint` is `-0.6..1.57` in the URDF and `-1.0..3.57` in the MJCF. We decided
  (2026-11-16) that the **URDF limits are the source of truth**, because robolab trains on that URDF and keeps
  the policy within 90 % of its limits. `rpo_hook.xml` therefore uses the URDF ranges for every joint;
  the build script overwrites them. The unmodified upstream `mjcf/rpo.xml` keeps its original ranges.
- When MuJoCo loads the URDF directly, it merges the fixed hook link into `right_elbow_yaw_link` unless
  you set `<compiler fusestatic="false"/>`. The MJCF keeps it as its own body.
