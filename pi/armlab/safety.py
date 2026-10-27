"""
The safety map: clear space (mm) to the table, the tower and the arm itself for any shoulder/elbow/wrist pose,
and the checks every automatic move goes through before and while it runs.
"""
import json
import math
import os

import numpy as np

from .config import HOME_FILE, HOME_POSE, JOINT_NAMES, MAP_FILE, MARGIN_MM, MOVING, ROLL_SLIDER
from .errors import ArmError
from .state import S


def pose_text(pose):
    return "  ".join(f"{JOINT_NAMES[j]} {pose[i]:.0f}" for i, j in enumerate(MOVING))


class Zone:
    """
    The safety map. For a pose (shoulder, elbow, wrist slider values) it gives the clear space in mm
    to the nearest thing the arm could hit. Between the points the map was worked out at (every 4
    degrees) it blends the surrounding points, which is good to a millimetre or two.
    """
    PARTS = (("base", "the table or the base"), ("tower", "the tower"), ("self", "the arm itself"))
    TRAVEL_GRIP = 80.0                                      # the gripper value the arm travels at (survey, grab)

    def __init__(self, margin=MARGIN_MM):
        path = os.path.join(S.data_dir, MAP_FILE)
        if not os.path.exists(path):
            raise ArmError(f"No safety map at {path}.\nMake arm_map.npz on the laptop with build_arm_map.py "
                           "and copy it there.")
        d = np.load(path)
        self.margin = float(margin)
        self.table_margin = float(S.table_margin_mm if S.table_margin_mm is not None else margin)
        self.table_extra = self.margin - self.table_margin      # added to the table/base clearance before comparing with margin
        self.axes = [d[k].astype(float) for k in ("s", "e", "w")]
        if any(len(a) < 2 for a in self.axes):
            raise ArmError("The safety map is too small to use. Build it again.")
        self.step = [a[1] - a[0] for a in self.axes]
        self.node = {k: d[k].astype(float) for k in ("base", "tower", "self", "low")}
        self.lo = [a[0] for a in self.axes]                 # range the script may use on each joint
        self.hi = [a[-1] for a in self.axes]
        try:
            self.made = json.loads(str(d["meta"])).get("made", "?")
        except (KeyError, ValueError):
            self.made = "?"
        self.home = self._load_home()
        self.exact = None                                   # --exact-table: the table gap from tablecheck.py, not the map
        self.hand = (ROLL_SLIDER, self.TRAVEL_GRIP)         # rotate and gripper assumed for a pose given as 3 joints
        if S.exact_table:
            from .tablecheck import load
            self.exact = load()

    def _state(self, q):
        """(5 joints, base) for the exact check: a pose's own rotate/gripper/base when it has them
        (arm_angles() gives all six), else the travel hand and the base as last commanded (None: the worst angle)."""
        rot, grip = (q[3], q[4]) if len(q) >= 5 else self.hand
        base = q[5] if len(q) >= 6 else S.base_cmd
        return [q[0], q[1], q[2], rot, grip], base

    def _exact(self, points, what):
        """The exact table gap ("base") or lowest point ("low") at each pose."""
        out = []
        for q in np.atleast_2d(np.asarray(points, dtype=float)):
            g, low = self.exact.gaps(*self._state(q))
            out.append(max(g, 0.0) if what == "base" else low)
        return np.array(out)

    def _load_home(self):
        path = os.path.join(S.data_dir, HOME_FILE)
        if os.path.exists(path):
            return [float(v) for v in json.load(open(path))["pose"]]
        return list(HOME_POSE)

    def keep_within(self, joints):
        """Never ask for an angle the ESP32 would refuse (its own limits for each joint)."""
        for i, j in enumerate(MOVING):
            self.lo[i] = max(self.lo[i], float(joints[j]["min"]))
            self.hi[i] = min(self.hi[i], float(joints[j]["max"]))

    def _lookup(self, points, what):
        """Value of the map at each pose, blended smoothly between the 8 map points around it."""
        if what == "gap":                                   # clear space to the nearest thing of all
            return np.minimum(np.minimum(self._lookup(points, "base") + self.table_extra, self._lookup(points, "tower")),
                              self._lookup(points, "self"))
        full = np.atleast_2d(np.asarray(points, dtype=float))
        if self.exact is not None and what in ("base", "low"):
            inside = np.all([(full[:, k] >= self.lo[k] - 1e-6) & (full[:, k] <= self.hi[k] + 1e-6) for k in range(3)], axis=0)
            return np.where(inside, self._exact(full, what), 0.0 if what != "low" else -999.0)
        p = full[:, :3]
        inside = np.ones(len(p), dtype=bool)
        idx, frac = [], []
        for k in range(3):
            inside &= (p[:, k] >= self.lo[k] - 1e-6) & (p[:, k] <= self.hi[k] + 1e-6)
            n = len(self.axes[k])
            t = np.clip((p[:, k] - self.axes[k][0]) / self.step[k], 0, n - 1)
            i = np.clip(np.floor(t).astype(int), 0, n - 2)
            idx.append(i)
            frac.append(t - i)
        out = np.zeros(len(p))
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    weight = ((frac[0] if a else 1 - frac[0]) * (frac[1] if b else 1 - frac[1])
                              * (frac[2] if c else 1 - frac[2]))
                    out += weight * self.node[what][idx[0] + a, idx[1] + b, idx[2] + c]
        return np.where(inside, out, 0.0 if what != "low" else -999.0)

    def gap(self, pose):
        """Clear space in mm at this pose. 0 if the pose is off the map."""
        return float(self._lookup(pose, "gap")[0])

    def gaps(self, poses):
        """Clear space in mm at each of these poses (as gap; 6 numbers each: the hand and base the exact check uses)."""
        return self._lookup(poses, "gap")

    def detail(self, pose):
        """Clear space to each thing separately, and how high the lowest point of the arm is."""
        return {k: float(self._lookup(pose, k)[0]) for k in ("base", "tower", "self", "low")}

    def on_map(self, pose):
        return all(self.lo[i] - 1e-6 <= pose[i] <= self.hi[i] + 1e-6 for i in range(3))

    def contains(self, pose):
        return self.gap(pose) >= self.margin

    def path_gap(self, a, b):
        """Smallest clear space anywhere along the straight move from pose a to pose b (shoulder, elbow, wrist; the
        hand travels as it is, see _state)."""
        a, b = np.asarray(a[:3], dtype=float), np.asarray(b[:3], dtype=float)
        n = int(np.abs(b - a).max()) + 2                       # a point at least every degree
        return float(self._lookup(a + (b - a) * np.linspace(0.0, 1.0, n)[:, None], "gap").min())

    def path_ok(self, a, b):
        return self.path_gap(a, b) >= self.margin

    def route(self, a, b):
        """How to get from a to b: [b] if the straight move is clear, [home, b] if not, None if neither."""
        if self.path_ok(a, b):
            return [list(b)]
        if self.path_ok(a, self.home) and self.path_ok(self.home, b):
            return [list(self.home), list(b)]
        return None

    def why_not(self, pose):
        if not self.on_map(pose):
            return "it is outside the range the map covers (" + self.describe_range() + ")"
        d = self.detail(pose)
        key, name = min(self.PARTS, key=lambda kn: d[kn[0]] + (self.table_extra if kn[0] == "base" else 0.0))
        limit = self.table_margin if key == "base" else self.margin
        return f"that pose has only {d[key]:.0f} mm of clear space to {name} (the margin there is {limit:.0f} mm)"

    def random_pose(self, rng, spare=0.0):
        """A random allowed pose that can be reached in a straight line from home."""
        for _ in range(20000):
            p = [round(rng.uniform(self.lo[i], self.hi[i]) * 2) / 2 for i in range(3)]    # 0.5 degree grid
            if self.gap(p) >= self.margin + spare and self.path_ok(self.home, p):
                return p
        raise ArmError("Could not find any allowed pose. Try a smaller --margin.")

    def approach_pose(self, target, rng):
        """A pose a few degrees away from target on every joint, to arrive from a random direction."""
        for _ in range(200):
            p = [round(target[k] + rng.choice([-1, 1]) * rng.uniform(6, 20), 1) for k in range(3)]
            if self.contains(p) and self.path_ok(p, target) and self.path_ok(self.home, p):
                return p
        return None

    def describe_range(self):
        return "   ".join(f"{JOINT_NAMES[j]} {self.lo[i]:.0f} to {self.hi[i]:.0f}" for i, j in enumerate(MOVING))

    def describe(self):
        g = np.minimum(np.minimum(self.node["base"] + self.table_extra, self.node["tower"]), self.node["self"])
        table = "" if self.table_extra == 0 else f" ({self.table_margin:.0f} mm to the table and base)"
        if self.exact is not None:
            return (f"made {self.made}, margin {self.margin:.0f} mm{table}; table and plywood from the exact check "
                    f"(world model + touch-checked hand, {self.exact.made})")
        return (f"made {self.made}, margin {self.margin:.0f} mm{table}, "
                f"{(g >= self.margin).mean() * 100:.0f}% of all poses allowed")


def gap_text(zone, pose):
    d = zone.detail(pose)
    if not zone.on_map(pose):
        return "off the map"
    return (f"table/base {d['base']:3.0f}   tower {d['tower']:3.0f}   itself {d['self']:3.0f} mm   "
            f"(lowest point {d['low']:.0f} mm above the table)")


def off_path(p, a, b):
    """How far pose p is from the straight line a -> b, in degrees."""
    ab = [b[i] - a[i] for i in range(3)]
    ap = [p[i] - a[i] for i in range(3)]
    length2 = sum(v * v for v in ab)
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, sum(ap[i] * ab[i] for i in range(3)) / length2))
    return math.sqrt(sum((ap[i] - t * ab[i]) ** 2 for i in range(3)))


