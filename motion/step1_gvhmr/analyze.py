"""Gate-1 analysis of GVHMR output. Run inside the GVHMR env on the VM (from ~/GVHMR):

    python ~/analyze.py ~/step1_results/IMG_0532 [~/step1_results/IMG_0531 ...]

For each clip dir (containing hmr4d_results.pt) writes:
  joints_global.npz   SMPL-24 joints, world frame: Y up, start at origin, initial facing +Z (same as GVHMR's render)
  gate1_plots.png     top-down root path, joint heights vs time, wrist heights, foot sliding
  gate1_summary.json  numbers quoted in the Gate-1 report
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import einsum

from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.smplx_utils import make_smplx

FPS = 30.0
J = dict(pelvis=0, l_ankle=7, r_ankle=8, head=15, l_wrist=20, r_wrist=21, l_foot=10, r_foot=11)


def global_joints(pt_path):
    pred = torch.load(pt_path, map_location="cpu")
    smplx = make_smplx("supermotion")
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")
    J_reg = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt")
    with torch.no_grad():
        out = smplx(**pred["smpl_params_global"])
        verts = torch.stack([torch.matmul(smplx2smpl, v) for v in out.vertices])
        # identical to GVHMR demo.move_to_start_point_face_z
        offset = einsum(J_reg, verts[0], "j v, v i -> j i")[0].clone()
        offset[1] = verts[:, :, 1].min()
        verts = verts - offset
        T = compute_T_ayfz2ay(einsum(J_reg, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T)
        joints = einsum(J_reg, verts, "j v, l v i -> l j i")
    return joints.numpy(), verts[:, :, 1].min(dim=1).values.numpy()


def analyze(clip_dir):
    clip_dir = Path(clip_dir)
    joints, lowest_vertex = global_joints(clip_dir / "hmr4d_results.pt")
    n = len(joints)
    t = np.arange(n) / FPS
    pel = joints[:, J["pelvis"]]
    h = {k: joints[:, i, 1] for k, i in J.items()}

    step = np.linalg.norm(np.diff(pel[:, [0, 2]], axis=0), axis=1)
    path_len = float(step.sum())
    disp = float(np.linalg.norm(pel[-1, [0, 2]] - pel[0, [0, 2]]))

    # foot sliding: horizontal speed of a foot joint while it is within 2 cm of its lowest height (= in contact)
    slide = {}
    for side in ("l_foot", "r_foot"):
        p = joints[:, J[side]]
        v = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1) * FPS
        contact = (p[1:, 1] - p[:, 1].min()) < 0.02
        slide[side] = dict(contact_frac=float(contact.mean()),
                           median_speed_in_contact_mps=float(np.median(v[contact])) if contact.any() else None)

    # squats = pelvis height minima well below standing height
    stand = float(np.percentile(h["pelvis"], 90))
    low = h["pelvis"] < stand - 0.15
    squats = []
    i = 0
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            k = i + int(np.argmin(h["pelvis"][i:j]))
            squats.append(dict(t_start=round(t[i], 2), t_end=round(t[j - 1], 2), t_min=round(t[k], 2),
                               pelvis_min_m=round(float(h["pelvis"][k]), 3),
                               r_wrist_height_m=round(float(h["r_wrist"][k]), 3),
                               l_wrist_height_m=round(float(h["l_wrist"][k]), 3)))
            i = j
        else:
            i += 1

    # carry phase = between end of first squat and start of second squat
    carry = None
    if len(squats) >= 2:
        a, b = int(squats[0]["t_end"] * FPS), int(squats[1]["t_start"] * FPS)
        if b > a:
            sl = slice(a, b)
            carry = dict(t=[round(a / FPS, 2), round(b / FPS, 2)],
                         r_wrist_mean_height_m=round(float(h["r_wrist"][sl].mean()), 3),
                         l_wrist_mean_height_m=round(float(h["l_wrist"][sl].mean()), 3),
                         wrist_separation_mean_m=round(float(np.linalg.norm(
                             joints[sl, J["r_wrist"]] - joints[sl, J["l_wrist"]], axis=1).mean()), 3),
                         pelvis_distance_m=round(float(step[a:b].sum()), 3))

    summary = dict(clip=clip_dir.name, frames=n, duration_s=round(n / FPS, 2),
                   standing_height_head_m=round(float(np.percentile(h["head"], 90)), 3),
                   pelvis_path_length_m=round(path_len, 3), pelvis_net_displacement_m=round(disp, 3),
                   squats=squats, carry=carry, foot_sliding=slide,
                   ground_penetration_min_m=round(float(lowest_vertex.min()), 3))

    np.savez_compressed(clip_dir / "joints_global.npz", joints=joints, fps=FPS, joint_ids=json.dumps(J))
    (clip_dir / "gate1_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    a.plot(pel[:, 0], pel[:, 2], "k-", lw=1.5, label="pelvis")
    a.plot(joints[:, J["l_foot"], 0], joints[:, J["l_foot"], 2], "b-", lw=0.6, alpha=0.6, label="L foot")
    a.plot(joints[:, J["r_foot"], 0], joints[:, J["r_foot"], 2], "r-", lw=0.6, alpha=0.6, label="R foot")
    a.plot(*pel[0, [0, 2]], "go", label="start"); a.plot(*pel[-1, [0, 2]], "rs", label="end")
    a.set_aspect("equal"); a.set_xlabel("x [m]"); a.set_ylabel("z [m] (initial facing)"); a.grid(alpha=0.3)
    a.set_title(f"Top-down path: length {path_len:.2f} m, net {disp:.2f} m"); a.legend(fontsize=8)
    a = ax[0, 1]
    for k, c in [("head", "k"), ("pelvis", "g"), ("l_ankle", "b"), ("r_ankle", "r")]:
        a.plot(t, h[k], c, label=k)
    for s in squats:
        a.axvspan(s["t_start"], s["t_end"], color="orange", alpha=0.2)
    a.set_xlabel("time [s]"); a.set_ylabel("height [m]"); a.set_title("Heights (orange = squat)"); a.grid(alpha=0.3); a.legend(fontsize=8)
    a = ax[1, 0]
    a.plot(t, h["r_wrist"], "r", label="RIGHT wrist"); a.plot(t, h["l_wrist"], "b", label="left wrist")
    for s in squats:
        a.axvspan(s["t_start"], s["t_end"], color="orange", alpha=0.2)
    a.set_xlabel("time [s]"); a.set_ylabel("height [m]"); a.set_title("Wrist heights: hook hand should go low at pickup & place"); a.grid(alpha=0.3); a.legend(fontsize=8)
    a = ax[1, 1]
    for side, c in [("l_foot", "b"), ("r_foot", "r")]:
        p = joints[:, J[side]]
        v = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1) * FPS
        a.plot(t[1:], v, c, lw=0.8, label=f"{side} horiz. speed")
    a.set_ylim(0, 3); a.set_xlabel("time [s]"); a.set_ylabel("m/s"); a.set_title("Foot speed (should drop to ~0 when planted)"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(f"GVHMR, {clip_dir.name}", fontsize=14)
    fig.tight_layout(); fig.savefig(clip_dir / "gate1_plots.png", dpi=90)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        analyze(d)
