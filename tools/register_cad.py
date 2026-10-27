#!/usr/bin/env python3
"""
Place the CAD parts (Fusion export, hardware/cad/rigid_parts) in the world model's frames, for the new safety check.

The old safety map moves the CAD parts with its own tape-measure kinematics (build_arm_map.py). The world model
(world.json, from the big solve) has the real kinematics but no shapes. For many survey poses this finds the one
fixed placement of each CAD part in its world frame (upper arm in U, forearm in F, wrist block in H, gripper in G)
that makes the two agree best, plus where the CAD tower sits in the world's tower frame T.

    pip install mujoco numpy scipy
    python register_cad.py world.json [--cad ../hardware/cad/rigid_parts] [--out cad_in_world.json]

Prints how well each part agrees (mm). The result (cad_in_world.json) has, per part, the transform and the CAD
mesh's points in the world frame; refine the gripper with the touch test afterwards (fit_hand.py).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
from build_arm_map import ArmModel                      # noqa: E402
from armlab.world import World, PARTS                    # noqa: E402

BODY_FRAME = {"upper_arm": "U", "forearm": "F", "wrist": "H", "hand": "G"}
ROLL_LEVEL = 42.0                                        # the drawn hand is the level one (the old map's assumption)


def world_angles(w, s, e, wr):
    """The joint angles (deg) the world model gives these slider values, in ArmModel's terms: upper arm above level,
    elbow bend, wrist bend.

    ArmModel's own conversion (build_arm_map.py) is the tape-measure one: shoulder zero at slider 117.5 and
    0.455-0.526 deg a step. Reseating the shoulder horn moved it ~90 steps, and the world now says ~0.40 deg
    a step from a different zero, so at the same slider the two disagreed by 41-43 deg. The fit took that up by
    tilting the CAD tower 39 deg about the shoulder axis, which tipped the vertical base axis over: the "base axis:
    CAD vs world 105 mm" this used to print. Registering into a world means moving the CAD the way that
    world moves.
    """
    a = w.arm
    d = np.array([s, e, wr], float) - w.ref[:3]
    a1 = a["th1"] + a["k1"] * d[0]
    a2 = a1 + a["th2"] + a["k2"] * d[1]
    a3 = a2 + a["th3"] + a["k3"] * d[2]
    return {"shoulder": a1, "elbow": a2 - a1, "wrist": a3 - a2}


def tf(x):
    return Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6]


def world_in_T(w, joints):
    """Each world part's (R, o) in the TOWER frame (base cancels)."""
    P = w.parts(joints[:4], 0.0)
    RT, oT = P[PARTS.index("T")]
    return {k: (RT.T @ P[PARTS.index(k)][0], RT.T @ (P[PARTS.index(k)][1] - oT)) for k in BODY_FRAME.values()}


def cad_in_tower(A, F, name, pts):
    """CAD points of a body, in the CAD tower body's frame."""
    return A.local(F, "tower", A.world(F, name, pts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world")
    ap.add_argument("--cad", default=os.path.join(HERE, "..", "hardware", "cad", "rigid_parts"))
    ap.add_argument("--poses", type=int, default=150)
    ap.add_argument("--out", default="cad_in_world.json")
    a = ap.parse_args()
    w = World(a.world)
    d = json.load(open(a.world))
    q = np.array(d["poses"]["joints"], float)
    q = q[np.abs(q[:, 3] - ROLL_LEVEL) < 0.6]                 # hand level: the CAD's drawn roll
    rng = np.random.default_rng(0)
    q = q[rng.choice(len(q), min(a.poses, len(q)), replace=False)]
    A = ArmModel(a.cad, spacing=12.0, hand_rolls=(0,))
    A.real_angles = lambda s_, e_, w_: world_angles(w, s_, e_, w_)     # pose the CAD the way THIS world moves
    pts = {b: A.P[b][rng.choice(len(A.P[b]), min(40, len(A.P[b])), replace=False)] for b in BODY_FRAME}
    cad = []                                                 # per pose: body -> CAD points in the CAD tower frame
    wor = []
    for s, e, wr, rot, _ in q:
        F = A.fk(A.q(s, e, wr))
        cad.append({b: cad_in_tower(A, F, b, pts[b]) for b in BODY_FRAME})
        wor.append(world_in_T(w, [s, e, wr, rot]))
    names = list(BODY_FRAME)
    F0 = A.fk(np.zeros(A.m.nq))
    base_axis = A.local(F0, "tower", np.array([A.anch0["base"], A.anch0["base"] + [0.0, 0.0, 200.0]]))
    axis_xy = np.asarray(w.axis[:2], float)              # the base axis in the world tower frame

    def resid(x):
        RY, tY = tf(x[:6])                                   # CAD tower frame -> world tower frame
        out = [10.0 * ((base_axis @ RY.T + tY)[:, :2] - axis_xy).ravel()]   # the CAD base axis on the world's:
        for k, b in enumerate(names):                        # fixes the sideways shift nothing else sees
            RX, tX = tf(x[6 + 6 * k: 12 + 6 * k])            # CAD body points (local) -> world frame f
            for cp, wp in zip(cad, wor):
                Rf, of = wp[BODY_FRAME[b]]
                pw = (pts[b] @ RX.T + tX) @ Rf.T + of        # through the world kinematics
                pc = cp[b] @ RY.T + tY                       # through the CAD kinematics
                out.append((pw - pc).ravel())
        return np.concatenate(out)

    # first guess: tower frames lined up by the shoulder hinge (both origins there), bodies from the first pose
    x0 = np.zeros(6 + 6 * len(names))
    F0 = A.fk(A.q(*q[0, :3]))
    sh = A.local(F0, "tower", A.anch0["shoulder"][None])[0]
    x0[3:6] = -sh
    for k, b in enumerate(names):                            # body local -> world f, at pose 0
        Rf, of = wor[0][BODY_FRAME[b]]
        p_w = cad[0][b] - sh                                 # CAD tower frame, shoulder at origin (rotation guessed 0)
        loc = (p_w - of) @ Rf                                # into f
        R, _ = Rotation.align_vectors(loc - loc.mean(0), pts[b] - pts[b].mean(0))
        x0[6 + 6 * k: 9 + 6 * k] = R.as_rotvec()
        x0[9 + 6 * k: 12 + 6 * k] = loc.mean(0) - pts[b].mean(0) @ R.as_matrix().T
    s = least_squares(resid, x0, loss="soft_l1", f_scale=5.0, max_nfev=200)
    print(f"   base axis: CAD vs world {np.abs(s.fun[:4]).max() / 10:.2f} mm")
    r = np.linalg.norm(s.fun[4:].reshape(-1, 3), axis=1).reshape(len(names), len(q), -1)
    print(f"{len(q)} survey poses with the hand level; agreement between the CAD kinematics and the world model:")
    for k, b in enumerate(names):
        print(f"   {b:10s} -> {BODY_FRAME[b]}: median {np.median(r[k]):5.1f} mm, 90% {np.percentile(r[k], 90):5.1f}, worst {r[k].max():5.1f}")
    RY, tY = tf(s.x[:6])
    out = {"made_from": os.path.abspath(a.world), "cad": os.path.abspath(a.cad),
           "tower": {"R": RY.tolist(), "t_mm": tY.tolist(), "doc": "CAD tower frame -> world tower frame T"},
           "parts": {}}
    for k, b in enumerate(names):
        RX, tX = tf(s.x[6 + 6 * k: 12 + 6 * k])
        allp = A.P[b] @ RX.T + tX
        out["parts"][b] = {"frame": BODY_FRAME[b], "R": RX.tolist(), "t_mm": tX.tolist(),
                           "median_mm": float(np.median(r[k])), "points_mm": np.round(allp, 1).tolist()}
    json.dump(out, open(a.out, "w"))
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
