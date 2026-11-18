"""Pad a step 3 checkpoint for step 4 v3's extra observation (fixed task target, 3 values, appended last).

    python pad_checkpoint.py <step3 model.pt> <out.pt> [--extra 3]

The first Linear layer of the actor (121 -> 124 inputs) and the critic (340 -> 343) gets zero columns for the new
inputs, so the padded policy behaves exactly like the step 3 policy at the start of training. Adam's moment
buffers for those weights are padded the same way; the iteration counter is kept.
"""
import argparse

import torch


def pad_cols(t, extra):
    return torch.cat([t, torch.zeros(t.shape[0], extra, dtype=t.dtype)], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--extra", type=int, default=3)
    a = ap.parse_args()
    d = torch.load(a.src, map_location="cpu", weights_only=False)
    old_shapes = set()
    for key in ("actor_state_dict", "critic_state_dict"):
        w = d[key]["mlp.0.weight"]
        old_shapes.add(tuple(w.shape))
        d[key]["mlp.0.weight"] = pad_cols(w, a.extra)
        print(key, tuple(w.shape), "->", tuple(d[key]["mlp.0.weight"].shape))
    n = 0
    for st in d["optimizer_state_dict"]["state"].values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st and tuple(st[k].shape) in old_shapes:
                st[k] = pad_cols(st[k], a.extra); n += 1
    print("padded optimizer buffers:", n)
    assert n == 4, "expected exp_avg + exp_avg_sq for the actor and critic input layers"
    torch.save(d, a.dst)
    print("saved", a.dst)


if __name__ == "__main__":
    main()
