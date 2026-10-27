#!/usr/bin/env python3
"""
Safety map v2: the same arm_map.npz the Pi already reads (shoulder x elbow x wrist grid, clear space in mm), but the
gap to the table, plywood and base comes from the world model (world.json: the real kinematics from the big solve),
the CAD arm links placed in it (register_cad.py) and the touch-checked hand (fit_hand.py), instead of the old
tape-measure kinematics and a hand box 35 mm too deep. The gaps to the tower and to the arm itself are copied from the
old map unchanged.

Cautious where nothing is measured: the hand counts at every gripper value 40-150 and wrist rotate in ROTATE (the
survey's range near the table), the plywood and base are spun round the base axis (the tallest thing at each distance,
as before), and the table's tilt counts at the worst base angle. Clearances are what the model predicts minus
PLANNER_BIAS_MM (at the touches the real hand sat 2.7 mm lower than the commanded joints said).

    python build_world_map.py world.json cad_in_world.json hand_model.json old_arm_map.npz [--out arm_map_v2.npz]
"""
import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
from armlab.world import World, PARTS                    # noqa: E402

GRIPS = (40.0, 80.0, 115.0, 150.0)
ROTATE = (14.0, 28.0, 42.0, 56.0, 70.0)
BASE_ANGLES = np.arange(-60.0, 91.0, 15.0)                 # the table's tilt is checked at these base counts
PLANNER_BIAS_MM = 3.0
JOINT_NAMES_BASE = "Revolute-4"   # build_arm_map.JOINT_NAMES["base"], kept here so the CAD can be read without mujoco
HAND_POINTS = 700
CAD = os.path.join(HERE, "..", "hardware", "cad", "rigid_parts", "meshes", "plywood_and_base_1.stl")
CAD_TABLE_Z, BIN = -19.0, 5.0                              # the CAD's table top (mm), radius bins for the spun profile

_W = None


def load_stl(path):
    import struct
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b, offset=84, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
    return a["v"].reshape(-1, 3).astype(float)


def setup(args):
    """Everything a worker needs, in the world tower frame T (base-independent)."""
    global _W
    world, cad_path, hand_path = args
    w = World(world)
    cad = json.load(open(cad_path))
    hm = json.load(open(hand_path))
    rng = np.random.default_rng(0)
    links = {p["frame"]: np.array(p["points_mm"]) for b, p in cad["parts"].items() if b != "hand"}
    hand = {k: np.array(v) for k, v in hm["parts"].items()}
    if len(hand["G"]) > HAND_POINTS:
        hand["G"] = hand["G"][rng.choice(len(hand["G"]), HAND_POINTS, replace=False)]
    ref = hm["ref_grip"]
    fing = {k: [hand[k] + w.finger_offset_side(k, g) - w.finger_offset_side(k, ref) for g in GRIPS] for k in "LR"}
    hand_all = np.vstack([hand["G"]] + fing["L"] + fing["R"])
    # plywood + base: the CAD mesh's height above the table at each distance from the base axis (spun round it)
    P = load_stl(CAD)
    cxy = base_axis_xy(os.path.dirname(os.path.dirname(CAD)))   # the base axis in the CAD frame, for the spun profile
    r = np.linalg.norm(P[:, :2] - cxy, axis=1)
    h = P[:, 2] - CAD_TABLE_Z
    nb = int(np.ceil(r.max() / BIN)) + 1
    H = np.zeros(nb)
    np.maximum.at(H, (r / BIN).astype(int), h)
    # the table plane in T at each checked base count: height above table = z - (a x + b y + c) in W
    c0 = w.table_height(0.0, 0.0)
    pa, pb = w.table_height(1.0, 0.0) - c0, w.table_height(0.0, 1.0) - c0
    planes = []
    for bc in BASE_ANGLES:
        RT, oT = w.parts([123, 54, 68, 42], bc)[PARTS.index("T")]   # T in W at this base count
        n = np.array([-pa, -pb, 1.0])                        # W: n . p = c0 (unnormalised: height = n.p - c0 for small tilt)
        planes.append((RT.T @ n, n @ oT - c0))               # in T: height(pT) = nT . pT + d
    _W = dict(w=w, links=links, hand=hand_all, H=H, axis=w.axis[:2], planes=planes)


def base_axis_xy(folder):
    """The base joint's x,y in the CAD frame (mm), which is all this map needs the CAD model for.

    MuJoCo is the authority when it will load. On the dev laptop it will not: Smart App Control is enforced
    and blocks mujoco's unsigned _structs DLL ("An Application Control policy has blocked this file"), and
    turning Smart App Control off cannot be undone without reinstalling Windows -- too much to give up for two
    numbers. So when the import fails, the same two numbers are read straight out of the MJCF, which is XML:
    with every joint at zero a joint's world anchor is just the static pos and rotation of each body composed
    down the tree, which is arithmetic, not physics.

    When both work the two are compared and a disagreement is shouted about, so the fallback cannot drift away
    from what MuJoCo would have said without anyone noticing.
    """
    import glob as _glob
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation as _R

    xml = _glob.glob(os.path.join(folder, "*.xml"))
    if not xml:
        sys.exit("No .xml in %s: point this at the exported CAD folder (the one with meshes/ in it)." % folder)

    def vec(t):
        return np.array([float(v) for v in t.split()]) if t else np.zeros(3)

    def rot(e):                                    # compiler angle="radian" in this export
        if e.get("quat") is not None:
            q = vec(e.get("quat"))
            return _R.from_quat([q[1], q[2], q[3], q[0]])       # MuJoCo orders them w x y z
        if e.get("euler") is not None:
            return _R.from_euler("xyz", vec(e.get("euler")))
        return _R.identity()

    root = ET.parse(xml[0]).getroot()
    comp = root.find("compiler")
    if comp is not None and (comp.get("angle") or "degree") != "radian":
        sys.exit("This export is in degrees; base_axis_xy assumes radians. Check the compiler tag.")
    found = {}

    def walk(e, Rw, tw):
        for b in e.findall("body"):
            Rb, tb = Rw * rot(b), tw + Rw.apply(vec(b.get("pos")))
            for j in b.findall("joint"):
                found[j.get("name")] = tb + Rb.apply(vec(j.get("pos")))
            walk(b, Rb, tb)

    walk(root.find("worldbody"), _R.identity(), np.zeros(3))
    name = JOINT_NAMES_BASE
    if name not in found:
        sys.exit("No joint %s in %s: the export's joint names must match build_arm_map.JOINT_NAMES." % (name, xml[0]))
    xy = found[name][:2] * 1000.0

    try:
        import mujoco                                                     # noqa: F401
        sys.path.insert(0, HERE)
        from build_arm_map import ArmModel
        true_xy = ArmModel(folder, spacing=20.0, hand_rolls=(0,)).anch0["base"][:2]
        if np.linalg.norm(true_xy - xy) > 0.5:
            sys.exit("base_axis_xy disagrees with MuJoCo: %s vs %s mm. Fix it before trusting this map."
                     % (np.round(xy, 3), np.round(true_xy, 3)))
        return true_xy
    except ImportError:
        print("   (mujoco will not load here, so the base axis came from the MJCF directly: "
              "x %.3f y %.3f mm)" % (xy[0], xy[1]))
        return xy


def to_T(w, joints):
    P = w.parts(joints, 0.0)
    RT, oT = P[PARTS.index("T")]
    return {k: (RT.T @ P[PARTS.index(k)][0], RT.T @ (P[PARTS.index(k)][1] - oT)) for k in ("U", "F", "H", "G")}


def gaps(s, e, wr):
    """(lowest point above the table, gap to table/plywood/base) at a pose, over every rotate in ROTATE."""
    W = _W
    pts = []
    for rot in ROTATE:
        F = to_T(W["w"], [s, e, wr, rot])
        R, o = F["G"]
        pts.append(W["hand"] @ R.T + o)
    for k, p in W["links"].items():
        R, o = to_T(W["w"], [s, e, wr, 42.0])[k]
        pts.append(p @ R.T + o)
    X = np.vstack(pts)
    z = np.min([X @ n + d for n, d in W["planes"]], axis=0) - PLANNER_BIAS_MM   # worst base angle
    low = float(z.min())
    r = np.linalg.norm(X[:, :2] - W["axis"], axis=1)
    near = z < W["H"].max() + 150
    base = low
    if near.any():
        r_lo = np.arange(len(W["H"])) * BIN
        dr = np.maximum(np.maximum(r_lo[None] - r[near][:, None], r[near][:, None] - (r_lo + BIN)[None]), 0)
        dz = np.maximum(z[near][:, None] - W["H"][None], 0)
        base = min(float(np.hypot(dr, dz).min()), low)
    return low, base


def row(job):
    i, s, E, Wv = job
    out = np.zeros((len(E), len(Wv), 2), np.float32)
    for j, e in enumerate(E):
        for k, wr in enumerate(Wv):
            out[j, k] = gaps(s, e, wr)
    return i, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world")
    ap.add_argument("cad")
    ap.add_argument("hand")
    ap.add_argument("old_map")
    ap.add_argument("--out", default="arm_map_v2.npz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    old = np.load(a.old_map)
    S, E, Wv = old["s"].astype(float), old["e"].astype(float), old["w"].astype(float)
    t0 = time.time()
    jobs = [(i, s, E, Wv) for i, s in enumerate(S)]
    G = np.zeros((len(S), len(E), len(Wv), 2), np.float32)
    with multiprocessing.Pool(a.workers, initializer=setup, initargs=((a.world, a.cad, a.hand),)) as pool:
        for n, (i, out) in enumerate(pool.imap_unordered(row, jobs)):
            G[i] = out
            print(f"\r   {n + 1}/{len(S)} shoulder rows ({time.time() - t0:.0f} s)", end="", flush=True)
    print()
    meta = json.loads(str(old["meta"]))
    meta.update({"made": time.strftime("%Y-%m-%d %H:%M"), "v2": {"world": os.path.abspath(a.world), "grips": GRIPS,
                 "rotate": ROTATE, "planner_bias_mm": PLANNER_BIAS_MM, "tower_self_from": os.path.abspath(a.old_map)}})
    np.savez_compressed(a.out, s=old["s"], e=old["e"], w=old["w"],
                        low=np.clip(G[..., 0], -500, 2000).astype(np.int16), base=np.clip(G[..., 1], 0, 255).astype(np.uint8),
                        tower=old["tower"], self=old["self"], meta=json.dumps(meta))
    for name, g in (("old", old["base"].astype(float)), ("new", np.clip(G[..., 1], 0, 255))):
        allowed = lambda m: (np.minimum(np.minimum(g, old["tower"]), old["self"]) >= m).mean() * 100
        print(f"   {name} map: table/plywood gap >= 40 mm at {(g >= 40).mean() * 100:.0f}% of poses, >= 20 at "
              f"{(g >= 20).mean() * 100:.0f}%;  allowed overall at margin 40: {allowed(40):.0f}%")
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
