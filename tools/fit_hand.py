#!/usr/bin/env python3
"""
The gripper's shape for the new safety check: the CAD hand (hand_1, the one the old map used: a box-like envelope of
the gripper, camera and fingers), placed in the world model's gripper frame G by register_cad.py (the gripper tags and
camera sit on its surface to 5-10 mm there), with its front end sliding with the fingers: points past a split along the
gripper ride with the left or the right finger (finger model), the rest stay put. Your table touches pin down the two
things the CAD cannot say: where that split is, and the gripper value the fingers were drawn at (at every touch the
lowest point of the hand must be on the table).

    python fit_hand.py world.json cad_in_world.json touch_folder [--out hand_model.json]

hand_model.json: points (mm) in G at the reference gripper value, per part: "G" (turns with rotate), "L" / "R" (slide
by world finger_offset_side(side, grip) - (same at ref_grip)).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
from armlab.world import World, PARTS                    # noqa: E402
from armlab.modes.touch import camera_dq, parse_label   # noqa: E402



FINGERS_FROM_MM = 150.0     # in G: the CAD hand's front end from here on is the fingers (the finger tags sit at ~190,
                            # the arc radius is ~35 mm, so the fingers start about 150 mm along)


def split_hand(H1, w, x_split=FINGERS_FROM_MM):
    """The CAD hand's points, kept whole (G: does not slide, a cautious stand-in for wherever the fingers were drawn),
    plus its front end again as the two fingers (L, R: by which finger tag each point is nearer sideways), which slide
    with the gripper value. The union is never smaller than the real hand at the drawn value, and covers the fingers
    at every other value."""
    front = H1[:, 0] >= x_split
    mid = 0.5 * (w.tags[31]["t"][1] + w.tags[32]["t"][1])
    left = (H1[:, 1] > mid) == (w.tags[31]["t"][1] > mid)
    return {"G": H1, "L": H1[front & left], "R": H1[front & ~left]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world")
    ap.add_argument("cad")
    ap.add_argument("touches")
    ap.add_argument("--out", default="hand_model.json")
    a = ap.parse_args()
    w = World(a.world)
    H1 = np.array(json.load(open(a.cad))["parts"]["hand"]["points_mm"])
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.touches, "touch_*.jsonl"))) for l in open(f) if l.strip()]
    use = []
    for k, r in enumerate(rows):
        if r["label"].strip().lower() == "table" or parse_label(r["label"]):
            dq, px, n = camera_dq(w, r)
            if np.abs(dq[:3]).max() > 5:                   # a pressing touch: the arm bent, the hand pushed in
                print(f"   touch {k + 1} ({r['label']}): pressing (a joint bent {np.abs(dq[:3]).max():.0f} steps), left out")
                continue
            use.append((k + 1, r, dq))
    print(f"{len(use)} table touches")

    pc = w.table_height(0.0, 0.0)                          # the table plane z = pa x + pb y + pc
    pa, pb = w.table_height(1.0, 0.0) - pc, w.table_height(0.0, 1.0) - pc

    pts = split_hand(H1, w)

    def heights(x, r, dq):
        g0 = x[0]
        grip = r["joints"][4]
        R, o = w.parts(r["joints"][:4], r["base"], base_offset=r["base_offset"], dq=dq)[PARTS.index("G")]
        P = np.vstack([pts["G"]] + [pts[k] + w.finger_offset_side(k, grip) - w.finger_offset_side(k, g0) for k in "LR"])
        Y = P @ R.T + o
        return Y[:, 2] - (pa * Y[:, 0] + pb * Y[:, 1] + pc)

    def cost(x):
        lows = np.array([heights(x, r, dq).min() for _, r, dq in use])
        return float(np.sum(np.minimum(lows ** 2, 100.0))), lows       # robust: a misjudged touch counts up to 10 mm

    best = min(((cost((g0,))[0], g0) for g0 in np.arange(40, 161, 2.5)), key=lambda t: t[0])
    x = (best[1],)
    lows = cost(x)[1]
    print(f"fingers drawn at gripper {x[0]:.0f} ({len(pts['L'])} + {len(pts['R'])} points ride with the fingers)")
    print("lowest hand point above the table at each touch (mm; 0 = on it): "
          + "  ".join(f"#{n} {v:+.1f}" for (n, _, _), v in zip(use, lows)))
    print(f"   median {np.median(lows):+.1f}, spread (1 sigma) {lows.std():.1f}, worst {lows.min():+.1f} / {lows.max():+.1f}")
    print("   (+ = the model's hand stops short of the table where the real one touched it: the model is too small there)")
    model = {"frame": "G", "ref_grip": float(x[0]), "fingers_from_x_mm": FINGERS_FROM_MM,
             "doc": "points (mm) in G at ref_grip; L/R slide by finger_offset_side(side, grip) - finger_offset_side(side, ref_grip)",
             "fit": {"touches": len(use), "median_mm": float(np.median(lows)), "sigma_mm": float(lows.std()),
                     "worst_mm": [float(lows.min()), float(lows.max())]},
             "parts": {k: np.round(v, 2).tolist() for k, v in pts.items()}}
    json.dump(model, open(a.out, "w"))
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
