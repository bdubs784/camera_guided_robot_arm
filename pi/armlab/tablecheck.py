"""
The exact table check (safety v2). The old safety map works out the gap to the table from tape-measure kinematics and
one fixed hand, for the shoulder/elbow/wrist only, and touch tests caught it claiming 20-57 mm of room where the hand
was on the table (a closed gripper reaches ~20 mm further, rotate far from level, low reaches). This works the
gap out for the exact pose instead: the world model (world.json) puts the hand (touch-checked to about +-3 mm, fingers
at the real gripper value), the wrist block and the arm links at the real rotate and base angle, against the table
plane (from the mat tags) and the plywood + base (spun round the base axis, the tallest thing at each distance).

    python3 arm.py <mode> --exact-table --table-margin 20      replaces the map's table gap (tower/self stay the map's)

Needs ~/arm_data/world.json and table_model.json (tools/export_table_model.py).
"""
import json
import os

import numpy as np

from .world import PARTS

TABLE_MODEL_FILE = "table_model.json"
FAST = True                                 # gaps() through fastik.gaps (numba) when it can: same answer, ~20x
BASE_CHECK = np.arange(-60.0, 91.0, 15.0)   # base counts checked when the base angle is not known (table tilt)


class TableCheck:
    def __init__(self, world, path):
        d = json.load(open(path))
        self.w = world
        self.ref_grip = float(d["hand"]["ref_grip"])
        self.hand = {k: np.asarray(v, float) for k, v in d["hand"]["parts"].items()}
        self.links = {k: np.asarray(v, float) for k, v in d["links"].items()}
        self.bin = float(d["plywood"]["bin_mm"])
        self.ply = np.asarray(d["plywood"]["height_mm"], float)
        # The plywood and base by bearing as well as distance, when the model has it. The spun profile takes the
        # tallest thing at each distance and assumes it is there at every angle, which turns the stepper (69 mm,
        # bearing +30..+90) into a wall all the way round, over plywood that is really 19 mm. The base and plywood
        # are fixed in the table frame, so a bearing means the same thing at every base angle.
        self.ply2 = np.asarray(d["plywood"].get("height_by_angle_mm") or [], float)
        self.angle_bin = float(d["plywood"].get("angle_bin_deg") or 0.0)
        if self.ply2.size == 0 or not self.angle_bin:
            self.ply2, self.angle_bin = None, None
        # The cells of the bearing map that hold something, worked out once: gaps() runs thousands of times per
        # plan, and re-deriving this per call made a plan take 115 s.
        self._cells = None
        if self.ply2 is not None:
            ri, ti = np.nonzero(self.ply2 > 0.5)
            self._cells = (ri * self.bin, (ri + 1) * self.bin, ti * self.angle_bin + self.angle_bin / 2.0,
                           self.ply2[ri, ti]) if len(ri) else None
        self.bias = float(d.get("planner_bias_mm", 3.0))
        c0 = world.table_height(0.0, 0.0)
        self.plane = (world.table_height(1.0, 0.0) - c0, world.table_height(0.0, 1.0) - c0, c0)
        self.axis = np.asarray(world.axis[:2], float)        # the base axis (table frame)
        self.made = d.get("made", "?")

    def points(self, joints, base):
        """Every checked point of the arm in table mm. joints: shoulder, elbow, wrist, rotate, gripper (slider)."""
        P = self.w.parts(joints[:4], base)
        R, o = P[PARTS.index("G")]
        g = float(joints[4])
        out = [self.hand["G"] @ R.T + o]
        for s in ("L", "R"):
            shift = self.w.finger_offset_side(s, g) - self.w.finger_offset_side(s, self.ref_grip)
            out.append((self.hand[s] + shift) @ R.T + o)
        for f, p in self.links.items():
            Rf, of = P[PARTS.index(f)]
            out.append(p @ Rf.T + of)
        return np.vstack(out)

    def gaps(self, joints, base=None):
        """(gap to the table/plywood/base, lowest point above the table), mm, with the planner bias taken off.
        base None: the worst of BASE_CHECK (the table's tilt)."""
        if base is None:
            return min((self.gaps(joints, b) for b in BASE_CHECK), key=lambda g: g[0])
        global FAST
        if FAST:                                             # the same arithmetic compiled (fastik.points/gaps)
            try:
                from . import fastik
                X = fastik.points(self, joints, float(base))
                got = None if X is None else fastik.gaps(self, X)
            except Exception as e:
                print(f"   (compiled table check unavailable, using the Python one: {type(e).__name__}: {e})")
                FAST, got = False, None
            if got is not None:
                return float(got[0]), float(got[1])
        X = self.points(joints, float(base))
        a, b, c = self.plane
        z = X[:, 2] - (a * X[:, 0] + b * X[:, 1] + c) - self.bias
        low = float(z.min())
        r = np.linalg.norm(X[:, :2] - self.axis, axis=1)
        near = z < self.ply.max() + 150
        gap = low
        if near.any():
            r_lo = np.arange(len(self.ply)) * self.bin
            rn, zn = r[near][:, None], z[near][:, None]
            dr = np.maximum(np.maximum(r_lo[None] - rn, rn - (r_lo + self.bin)[None]), 0)
            if self.ply2 is None:
                dz = np.maximum(zn - self.ply[None], 0)
                gap = min(float(np.hypot(dr, dz).min()), low)
                return gap, low
            # Real distance to each piece of structure, around as well as out. Checking the point's own bearing bin
            # and its neighbours was too blunt: a bin is 50 mm of arc at 190 mm out, so a hand at bearing -17 got
            # refused by the frame at -45. Each cell is a box in (radius, bearing): the gap is how far the point is
            # from it outwards, around, and up.
            if self._cells is None:
                return low, low
            c_rlo, c_rhi, c_mid, c_h = self._cells
            # Same answer as measuring every arm point against every cell (2000 x 727 distances, 134 ms a call),
            # without doing it. The spun distance of a point is never more than its real one, so it is a cheap
            # lower bound: go through the points in order of that bound and stop once it passes the best real
            # distance found, since no later point can beat it.
            r_lo = np.arange(len(self.ply)) * self.bin
            dr_s = np.maximum(np.maximum(r_lo[None] - rn, rn - (r_lo + self.bin)[None]), 0.0)
            lb = np.hypot(dr_s, np.maximum(zn - self.ply[None], 0.0)).min(axis=1)
            th = np.degrees(np.arctan2(X[near, 1] - self.axis[1], X[near, 0] - self.axis[0])) % 360.0
            gap = low
            for i in np.argsort(lb):
                if lb[i] >= gap:
                    break
                d_out = np.maximum(np.maximum(c_rlo - rn[i, 0], rn[i, 0] - c_rhi), 0.0)
                dth = np.maximum(np.abs((th[i] - c_mid + 180.0) % 360.0 - 180.0) - self.angle_bin / 2.0, 0.0)
                d_round = np.radians(dth) * rn[i, 0]
                d_up = np.maximum(zn[i, 0] - c_h, 0.0)
                gap = min(gap, float(np.sqrt(d_out ** 2 + d_round ** 2 + d_up ** 2).min()))
        return gap, low


def load(world=None, data_dir=None):
    """The exact table check from ~/arm_data, or an error saying what is missing."""
    from .errors import ArmError
    from .state import S
    from .world import World
    data_dir = data_dir or S.data_dir
    tm, wp = os.path.join(data_dir, TABLE_MODEL_FILE), os.path.join(data_dir, "world.json")
    for p, how in ((wp, "the big solve's world.json"), (tm, "tools/export_table_model.py")):
        if not os.path.exists(p):
            raise ArmError(f"--exact-table needs {p} ({how}).")
    return TableCheck(world or World(wp), tm)
