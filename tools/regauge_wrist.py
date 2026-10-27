#!/usr/bin/env python3
"""
Turn a world model's hand frames (H, and G, which hangs off H through the wrist rotate) about the wrist joint by
DELTA degrees and add DELTA to th3. Every prediction the model makes stays exactly the same -- every tag corner,
every camera, the fingertip, the fingers at every gripper value -- but G's axes, which the planner reads "pitch" and
"square" off, turn by DELTA relative to the real hand.

    python tools/regauge_wrist.py world.json --like reference_world.json --out world_regauged.json
    python tools/regauge_wrist.py world.json --delta 5.7 --out world_regauged.json

Why: the big solve is free to turn the hand's frame about the wrist axis and move th3 the other way; the
tags on the hand turn with it and the residuals do not change by a pixel. The 623-pose solve landed 5-6 deg away
from the previous one. Put the same CAD gripper into both and it points the same way within ~1 deg, so the hand
itself agrees; only G's x-axis moved. But the planner's PITCH_RANGE / PITCH_PREFER were tuned on G's x-axis in the
old convention, so the 623 world was being asked for a hand 6 deg off what those numbers meant, and close blocks
stopped planning ("fingers square to the face: out of reach" at 260 mm). This puts the 623 world back in the
convention the planner was tuned in, without touching what it predicts.

--like picks DELTA so G's x-axis points, at the same slider values, as it does in the reference world (least
squares over a spread of poses). The finger model measures its angle from G's x-axis, so its phi0 moves too.

Anything else expressed in H or G (the CAD placement, the hand model, table_model.json, the safety map) has to be
rebuilt from the regauged world afterwards: register_cad.py, fit_hand.py, export_table_model.py,
build_world_map.py.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
from armlab.world import World, PARTS, rot_xz        # noqa: E402

HAND_PARTS = ("H", "G")


def regauge(d, delta_deg):
    """A copy of world.json dict d with the hand frames turned by delta_deg about the wrist joint."""
    d = copy.deepcopy(d)
    Q = rot_xz(np.radians(delta_deg))                   # new H = old H @ Q: points in H or G go to Q^T p
    QT = Q.T
    a = d["arm"]
    a["th3"] = float(a["th3"]) + delta_deg
    n = QT @ np.array([1.0, a["ax_u"], a["ax_v"]])      # the rotate axis, back in its [1, u, v] form
    a["ax_u"], a["ax_v"] = float(n[1] / n[0]), float(n[2] / n[0])
    pa = QT @ np.array([0.0, a["ax_y"], a["ax_z"]])     # a point on it, slid along it back to x = 0
    pa = pa - (pa[0] / n[0]) * n
    a["ax_y"], a["ax_z"] = float(pa[1]), float(pa[2])
    for t in d["tags"].values():
        if t["on"] in HAND_PARTS:
            R = Rotation.from_rotvec(t["rvec"]).as_matrix()
            t["rvec"] = list(Rotation.from_matrix(QT @ R).as_rotvec())
            t["t_mm"] = list(QT @ np.asarray(t["t_mm"], float))
    for c in d["cameras"].values():
        if c["parent"] in HAND_PARTS:                    # p_cam = R p_part + t, p_part = Q p_new
            R = Rotation.from_rotvec(c["rvec"]).as_matrix()
            c["rvec"] = list(Rotation.from_matrix(R @ Q).as_rotvec())
    if "tip_in_G" in d:
        d["tip_in_G"] = list(QT @ np.asarray(d["tip_in_G"], float))
    f = d.get("fingers")
    if f and "axis" in f:
        # The arc's hinge axis turns like any vector in G. Its angle is measured from u = G's x made perpendicular
        # to the axis, and G's x has turned, so phi0 shifts by the angle between the old u and the new one.
        ax, ay = f["axis"]
        n_old = np.array([ax, ay, 1.0]); n_old /= np.linalg.norm(n_old)
        u_old = np.array([1.0, 0.0, 0.0]) - n_old[0] * n_old; u_old /= np.linalg.norm(u_old)
        x_new = Q @ np.array([1.0, 0.0, 0.0])           # the new G x-axis, in old G coordinates
        u_new = x_new - (x_new @ n_old) * n_old; u_new /= np.linalg.norm(u_new)
        theta = np.degrees(np.arctan2(np.cross(u_old, u_new) @ n_old, u_old @ u_new))
        nn = QT @ n_old
        f["axis"] = [float(nn[0] / nn[2]), float(nn[1] / nn[2])]
        for side in ("L", "R"):
            if side in f:
                f[side]["phi0_deg"] = float(f[side]["phi0_deg"]) - theta
    d.setdefault("regauged", []).append({"wrist_frame_deg": delta_deg,
                                         "doc": "hand frames turned about the wrist joint; predictions unchanged"})
    return d


def sample_poses(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return [(list(rng.uniform([5, 40, 20, 14], [70, 120, 140, 70])), float(rng.uniform(-60, 60)),
             float(rng.uniform(40, 150))) for _ in range(n)]


def g_pitch_dirs(w, poses):
    return np.array([w.parts(q, b)[PARTS.index("G")][0][:, 0] for q, b, _ in poses])


def pick_delta(d, ref_path, poses):
    """DELTA that makes G's x-axis agree with the reference world's at the same slider values."""
    ref = World(ref_path)
    want = g_pitch_dirs(ref, poses)
    best = None
    for delta in np.arange(-15.0, 15.001, 0.05):
        w = _world(regauge(d, delta))
        err = float(np.degrees(np.arccos(np.clip(np.sum(g_pitch_dirs(w, poses) * want, 1), -1, 1))).mean())
        if best is None or err < best[1]:
            best = (float(delta), err)
    return best


def _world(d):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(d, fh)
    try:
        return World(fh.name)
    finally:
        os.remove(fh.name)


def check_same(d0, d1, poses):
    """Largest change in any prediction: tag corners (mm), camera poses (mm / deg), fingertip (mm)."""
    w0, w1 = _world(d0), _world(d1)
    worst = {"tag mm": 0.0, "camera mm": 0.0, "camera deg": 0.0, "tip mm": 0.0}
    for q, b, g in poses:
        for t in w0.tags:
            if w0.tags[t]["part"] in (PARTS.index("H"), PARTS.index("G")):
                worst["tag mm"] = max(worst["tag mm"], float(np.abs(w0.tag_corners_table(t, q, b, grip=g)
                                                                    - w1.tag_corners_table(t, q, b, grip=g)).max()))
        for c in w0.cams:
            R0, t0 = w0.camera_pose(c, q, b); R1, t1 = w1.camera_pose(c, q, b)
            worst["camera mm"] = max(worst["camera mm"], float(np.abs(t0 - t1).max()))
            worst["camera deg"] = max(worst["camera deg"],             # not arccos(trace): that reads 3e-6 deg of
                                      float(np.degrees(Rotation.from_matrix(R0.T @ R1).magnitude())))   # pure noise
        worst["tip mm"] = max(worst["tip mm"], float(np.abs(w0.tip(q, b, grip=g) - w1.tip(q, b, grip=g)).max()))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--like", help="reference world.json whose G convention to match")
    g.add_argument("--delta", type=float, help="degrees to turn the hand frames by")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d0 = json.load(open(args.world))
    poses = sample_poses()
    if args.like:
        delta, err = pick_delta(d0, args.like, poses[:120])
        print(f"G x-axis vs {os.path.basename(args.like)}: {err:.2f} deg apart on average after turning by {delta:+.2f} deg")
    else:
        delta = args.delta
    d1 = regauge(d0, delta)
    worst = check_same(d0, d1, poses)
    print(f"th3 {d0['arm']['th3']:+.3f} -> {d1['arm']['th3']:+.3f}. Largest change in any prediction over "
          f"{len(poses)} poses: " + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()))
    if max(worst["tag mm"], worst["camera mm"], worst["tip mm"]) > 1e-6 or worst["camera deg"] > 1e-6:
        sys.exit("NOT saved: the regauged world predicts something different. That is a bug in this tool.")
    json.dump(d1, open(args.out, "w"), indent=1)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
