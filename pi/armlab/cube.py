"""
The cube in the world model: where its top tag (TARGET_TAG, 7) is in table millimetres, which way the cube is turned,
and how high it sits, from every camera that sees the tag at once (tower and webcam turn with the base; the gripper
camera rides on the hand). The tag lies flat on the cube's top, face up, so the fit has four unknowns (x, y, height,
turn), with the height kept near "on the table, CUBE_MM up" unless the cameras clearly say otherwise (on the plywood).

    c = locate(world, sightings, joints, base, base_offset=off)      sightings: [(camera, tag, corners 4x2)]
    c["xyz"], c["yaw_deg"], c["above_table_mm"], c["px"], c["cams"]
"""
import math

import numpy as np

from .config import TAG_MM, TARGET_TAG
from .world import rotvec, tag_corners

CUBE_MM = 50.8                  # the block is 2 in (50.8 mm) tall with a 39.7 mm square top, where the tag is.
                                # Square, so either pair of faces can be gripped
HEIGHT_SIGMA_MM = 8.0           # how firmly the fit holds the height at table + CUBE_MM
CORNER_PX = 2.0                 # how well tag corners fit (1 sigma, px), the same as the arm filter's


def _corners(x):
    """Table-frame corners of the cube's top tag at (x, y, z, yaw)."""
    return tag_corners(TAG_MM.get(TARGET_TAG, 30.0)) @ rotvec([0.0, 0.0, x[3]]).T + x[:3]


# Each camera reads the block's top height with its own error, measured on 31 blocks 194-445 mm out:
#   tower +4.2 mm and webcam +9.0 mm, roughly constant;
#   gripper camera: a slope, not an offset (-9 mm close in, +11 mm at 450 mm; fitting the line cut its scatter from
#   7.2 to 1.3 mm). Close in, where the grab relies on it, it reads low, which puts the fingers deeper: the safer way
#   to be wrong.
# The grasp depth is measured from the top, so this matters. The fit lives in world.json ("cam_top_bias"); a world
# without it just gets no correction.
BIAS_MAX_PX = 5.0                 # a tag fit worse than this is a bad read, not something to correct for


def top_bias(w, cam, out_mm):
    """How much too high this camera reads the block's top, at this distance from the base axis (mm)."""
    got = (w.d.get("cam_top_bias") or {}).get(cam)
    if not got:
        return None
    return float(got[0]) * float(out_mm) + float(got[1])


def locate(w, sightings, joints, base, base_offset=0.0, grip=None, dq=None, tag=TARGET_TAG, iters=15):
    """The cube's top tag from every camera that sees it, or None. Returns xyz (tag centre, table mm), yaw_deg (the
    cube's turn, folded into -45..45 since it is square), above_table_mm, px (median corner error), cams {name: px}."""
    kw = {"base_offset": base_offset, "dq": dq}
    obs = [(c, np.asarray(px, float)) for c, t, px in sightings if t == tag and c in w.cams]
    if not obs:
        return None
    # first guess: the camera that sees the tag biggest
    c0, px0 = max(obs, key=lambda o: np.linalg.norm(o[1][0] - o[1][2]))
    got = w.locate_tag(c0, px0, TAG_MM.get(tag, 30.0), joints, base, **kw)
    if got is None:
        return None
    R0, t0 = got
    x = np.array([t0[0], t0[1], t0[2], math.atan2(R0[1, 0], R0[0, 0])])

    def resid(x):
        X = _corners(x)
        r = [((w.project(c, X, joints, base, **kw) - px) / CORNER_PX).ravel() for c, px in obs]
        z0 = w.table_height(x[0], x[1]) + CUBE_MM
        return np.concatenate(r + [[(x[2] - z0) / HEIGHT_SIGMA_MM]])
    for _ in range(iters):
        r = np.nan_to_num(resid(x), nan=50.0)
        J = np.column_stack([(np.nan_to_num(resid(x + d), nan=50.0) - r) / d.max() for d in np.diag([0.01, 0.01, 0.01, 1e-4])])
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        x = x + step
        if np.abs(step[:3]).max() < 1e-3 and abs(step[3]) < 1e-5:
            break
    X = _corners(x)
    cams, tops = {}, {}
    for c, px in obs:
        cams[c] = float(np.linalg.norm(w.project(c, X, joints, base, **kw) - px, axis=1).mean())
        got_c = w.locate_tag(c, px, TAG_MM.get(tag, 30.0), joints, base, **kw)      # that camera on its own
        if got_c is not None:
            _, t_c = got_c
            tops[c] = float(t_c[2] - w.table_height(t_c[0], t_c[1]))
    yaw = (math.degrees(x[3]) + 45.0) % 90.0 - 45.0
    # Take each camera's own height, take its known bias off, and weigh them by how well the tag fitted. The
    # cameras disagree about the top by up to 15 mm and the solve above cannot tell them apart -- it fits all
    # their corners at once and leans on the block's nominal height. x and y they agree on to about 2.5 mm, so
    # only the height is replaced.
    out_mm = float(np.hypot(x[0] - w.axis[0], x[1] - w.axis[1]))
    num = den = 0.0
    for c, h in tops.items():
        px = cams.get(c, 99.0)
        bias = top_bias(w, c, out_mm)
        if px > BIAS_MAX_PX or bias is None:
            continue
        wgt = 1.0 / max(px, 0.5) ** 2
        num += wgt * (h - bias)
        den += wgt
    if den > 0:
        x = x.copy()
        x[2] = w.table_height(x[0], x[1]) + num / den
    return {"xyz": x[:3].copy(), "yaw_deg": yaw, "above_table_mm": float(x[2] - w.table_height(x[0], x[1])),
            "px": float(np.median(list(cams.values()))), "cams": cams, "tops": tops,
            "top_corrected": bool(den > 0)}


LOOK_SIGMA_MM = {"gripper": 1.0}   # uncertainty of one look (1 sigma, mm sideways, per px of corner fit): the gripper
BASE_LOOK_SIGMA_MM = 2.0           # camera up close, and the tower camera/webcam from further away
FAR_GRIPPER_SIGMA_MM = 4.0         # the gripper camera from far away without the slack measured (it was off by 10 mm)
GATE_MM = 15.0                     # a look this far from the estimate means the block was moved: start again from it
CARRY_SIGMA_MM = 3.0               # carrying it adds this much doubt about where it ended up


def look_sigma(c):
    """How far off (1 sigma, mm sideways) one look from cube.locate may be: the best camera that saw it."""
    def one(cam):
        if cam == "gripper" and not c.get("close"):
            return FAR_GRIPPER_SIGMA_MM
        return LOOK_SIGMA_MM.get(cam, BASE_LOOK_SIGMA_MM)
    return min((one(cam) * max(1.0, px) for cam, px in c["cams"].items()),
               default=BASE_LOOK_SIGMA_MM * max(1.0, c.get("px", 1.0)))


class BlockFilter:
    """Where the block is (a Kalman filter on x, y, height, turn): every look updates it, a carry moves it, and when
    the cameras cannot see it (the arm in the way) its estimate and sigma say whether going on without a look is safe.
    The block stands still unless the arm moves it, so between looks nothing changes."""

    def __init__(self):
        self.x, self.P, self.top, self.looks = None, None, None, 0
        self.notes = {}

    def see(self, c, still=False):
        """Add a look (from cube.locate). Returns how far (mm sideways) it was from the estimate before it.

        still: nothing has touched the block since the last look, so a reading far from the estimate can't mean it
        moved; fuse it instead of restarting from it. (The same webcam from two base angles 12 deg apart once put the
        same block 19 mm apart, over GATE_MM, so without this a second look just replaced one error with another.)
        """
        s = look_sigma(c)
        z = np.array(list(c["xyz"]) + [math.radians(c["yaw_deg"])], float)
        R = np.diag([s, s, 1.5 * s, math.radians(3.0 * max(1.0, s))]) ** 2
        self.top, self.looks = c.get("above_table_mm", self.top), self.looks + 1
        if self.x is None:
            self.x, self.P = z, R
            return 0.0
        v = z - self.x
        v[3] = (v[3] + math.pi / 4) % (math.pi / 2) - math.pi / 4      # square: a turn is only known modulo 90 deg
        d = float(np.linalg.norm(v[:2]))
        if not still and d > min(GATE_MM, 3.0 * math.hypot(self.sigma_mm(), s)):   # moved, or the estimate was bad
            self.x, self.P = z, R
            return d
        K = self.P @ np.linalg.inv(self.P + R)
        self.x = self.x + K @ v
        self.P = (np.eye(4) - K) @ self.P
        return d

    def moved(self, d, sigma=CARRY_SIGMA_MM):
        """The arm carried it by d (mm, xyz)."""
        self.x[:3] = self.x[:3] + np.asarray(d, float)
        self.P[:3, :3] = self.P[:3, :3] + np.eye(3) * sigma ** 2

    def sigma_mm(self):
        return float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))

    def cube(self):
        """The estimate in cube.locate's form."""
        yaw = (math.degrees(self.x[3]) + 45.0) % 90.0 - 45.0
        return {"xyz": self.x[:3].copy(), "yaw_deg": yaw, "above_table_mm": self.top, "px": float("nan"),
                "cams": {}, "filter_mm": self.sigma_mm(), "looks": self.looks}
