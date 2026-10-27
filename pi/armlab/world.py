"""
The world model from the big solve (tools/big_solve.py -> world.json): the arm's real geometry, where every camera
sits, and where every tag is. Given the joint values and the base angle it says where every part of the arm is in
TABLE millimetres, where any tag should appear in any camera, and where a tag a camera sees is on the table.

    w = World("~/arm_data/world.json")
    w.tip(joints, base, grip=80)             fingertip point, table mm (x, y, z)
    w.predict_tag(9, "webcam", joints, base) where tag 9's corners should be in the webcam picture
                                             (finger tags 31/32: pass grip=, they slide with the gripper)
    w.locate_tag("gripper", corners, 25.0, joints, base)   a tag a camera sees -> its pose on the table
    w.fit_base_offset(sightings, joints, base)              how far the base really is from its count (deg)

Table frame (W): the tower frame at base 0. x along the arm, z up the base axis, origin at the shoulder hinge.
The table surface is about 250 mm below that origin; table_height() says exactly where under any point.

Same math as the solver (World.frames / predict in big_solve.py): tests/test_world.py checks they agree.
"""
import json
import os

import numpy as np

PARTS = ("W", "U", "F", "H", "T", "G")        # table, upper arm, forearm, wrist block, tower, gripper (turns with rotate)
AFTER = 100000                                 # a tag that moved gets AFTER + id for its position after the move
ROLL_REF = 42.0
GRIP_REF = 80.0                                # the gripper value finger tag positions are given at (the finger model)
MAT_TAGS = range(100, 136)                     # the tag mat: defines the table surface


_I3 = np.eye(3)                       # read-only on purpose: these are only ever read, never written through
_I3.flags.writeable = False
_Z5 = np.zeros(5)
_Z5.flags.writeable = False
_Z3 = np.zeros(3)
_Z3.flags.writeable = False
_DEG = np.pi / 180.0                  # np.radians on one float costs more than the multiply it does

# The three below are the innermost loop of a grasp plan -- parts() calls them a few hundred thousand times per
# plan -- so they are written for speed rather than for looks. Filling an empty 3x3 beats np.array of lists, and
# the identity in rotvec's sum is a constant because the sum makes a new array anyway. Bit-identical either way.


def rotvec(r):
    r = np.asarray(r, float)
    th = float(np.sqrt(r @ r))                        # what np.linalg.norm does, without its dispatch
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.zeros((3, 3))
    K[0, 1], K[0, 2] = -k[2], k[1]
    K[1, 0], K[1, 2] = k[2], -k[0]
    K[2, 0], K[2, 1] = -k[1], k[0]
    return _I3 + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def rot_xz(a):
    c, s = np.cos(a), np.sin(a)
    m = np.zeros((3, 3))
    m[0, 0], m[0, 2] = c, -s
    m[1, 1] = 1.0
    m[2, 0], m[2, 2] = s, c
    return m


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    m = np.zeros((3, 3))
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    m[2, 2] = 1.0
    return m


def _robust_fit(res, x, f_scale=3.0, iters=30):
    """A small damped Gauss-Newton for a few numbers, with soft_l1 weighting (a residual of f_scale px counts fully,
    a 10x bigger one about a third as much) -- the same fit scipy's least_squares(loss="soft_l1") does, written out
    because the Pi's scipy cannot load: scipy 1.17 on numpy 1.24, and that numpy belongs to the system Python the
    Pi's other projects share, so upgrading it is not an option."""
    lam, h = 1e-3, 1e-4
    r = res(x)

    def cost(r):
        return float(np.sum(2.0 * f_scale ** 2 * (np.sqrt(1.0 + (r / f_scale) ** 2) - 1.0)))
    c = cost(r)
    for _ in range(iters):
        J = np.column_stack([(res(x + h * e) - r) / h for e in np.eye(len(x))])
        wgt = 1.0 / np.sqrt(np.sqrt(1.0 + (r / f_scale) ** 2))      # IRLS: sqrt of soft_l1's weight on each row
        Jw, rw = J * wgt[:, None], r * wgt
        A = Jw.T @ Jw
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -Jw.T @ rw)
        xn = x + step
        rn = res(xn)
        cn = cost(rn)
        if cn < c:
            x, r, c, lam = xn, rn, cn, max(lam / 3.0, 1e-7)
            if np.abs(step).max() < 1e-6:
                break
        else:
            lam *= 10.0
            if lam > 1e6:
                break
    return x


def tag_corners(size):
    h = size / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def finger_offset(fing, side, grip):
    """Where a finger tag is (mm, in G) relative to where it is at the reference gripper value: the finger model of
    big_solve.py (each finger slides along an arc without turning; see world.json "fingers" "doc")."""
    ax, ay = fing["axis"]
    n = np.array([ax, ay, 1.0])
    n /= np.linalg.norm(n)
    u = np.array([1.0, 0.0, 0.0]) - n[0] * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    f = fing[side]
    d = float(grip) - float(fing.get("ref_grip", GRIP_REF))
    phi0 = np.radians(f["phi0_deg"])
    a = np.radians(f["k_deg_per_step"] * d + f["k2_deg_per_step2"] * d * d)
    return f["r_mm"] * ((np.cos(phi0 + a) - np.cos(phi0)) * u + (np.sin(phi0 + a) - np.sin(phi0)) * v)


def project(p, K, dist):
    """Camera-frame points (N, 3) -> pixels (N, 2), OpenCV's 5-number lens model."""
    p = np.atleast_2d(p)
    z = np.maximum(p[:, 2], 1e-6)
    x, y = p[:, 0] / z, p[:, 1] / z
    k1, k2, p1, p2, k3 = list(dist[:5]) + [0.0] * (5 - len(dist[:5]))
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=-1)


class World:
    def __init__(self, path, calib=None):
        """path: world.json from the big solve. calib: {camera: (K, dist)}; default: the world file's own copy, else
        ~/calib_<camera>.json (the lens calibrations the surveys used)."""
        d = json.load(open(os.path.expanduser(path)))
        self.d = d
        a = d["arm"]
        self.arm = {k: float(v) for k, v in a.items()}
        self._axis = {}                       # wrist-rotate axis, kept per arm parameters: see _rotate_axis
        for k, v in (("th4", 0.0), ("k4", 0.0), ("ax_u", 0.0), ("ax_v", 0.0), ("ax_y", 0.0), ("ax_z", 0.0), ("k4b", 0.0)):
            self.arm.setdefault(k, v)
        self.ref = np.array(d["ref_slider"] + [ROLL_REF] * (4 - len(d["ref_slider"])), float)[:4]
        self.gain = float(d["base"]["gain"])
        self.axis = np.array(list(d["base"]["axis_xy_mm"]) + [0.0])
        self.base_zero = float(d.get("base_zero_deg", 0.0))       # newest survey's base zero vs the count
        self.cams = {}
        for name, c in d["cameras"].items():
            K, dist = self._lens(name, c, calib)
            if K is None:
                continue
            K = K.copy()
            K[0, 0] *= c.get("focal_scale", 1.0)
            K[1, 1] *= c.get("focal_scale", 1.0)
            self.cams[name] = {"parent": PARTS.index(c["parent"]), "R": rotvec(c["rvec"]), "t": np.array(c["t_mm"], float),
                               "K": K, "dist": np.asarray(dist, float)}
        if "gripper2" in self.cams:                 # the gripper camera as it sits now (it was moved): that one counts
            self.cams["gripper_before"] = self.cams.get("gripper")
            self.cams["gripper"] = self.cams["gripper2"]
        self.tags = {}
        for tid, t in d["tags"].items():
            self.tags[int(tid)] = {"part": PARTS.index(t["on"]), "R": rotvec(t["rvec"]), "t": np.array(t["t_mm"], float),
                                   "size": float(t["size_mm"])}
        for tid in [t for t in self.tags if t >= AFTER]:          # a tag that moved: its newest position counts
            self.tags[tid - AFTER] = self.tags[tid]
        self.fingers = d.get("fingers")                            # the finger model, or None (an older solve)
        self.finger_side = {int(t) % AFTER: s for t, s in (self.fingers or {}).get("tags", {}).items()}
        self.tip_in_G = np.array(d["tip_in_G"], float) if "tip_in_G" in d else self._guess_tip()
        self.points = d.get("points", {})                         # touch-measured points on the hand (fit_touch)
        self._plane = None
        self._tip_shift = {}
        # The tower leans under the arm. Fitted per pose over the 623-pose surveys, the base cameras saw the table
        # tilt along the arm linearly in the shoulder: 0.032 deg per step, -1.3 deg at the bottom of its range to
        # +0.9 at the top. The shoulder alone explains 94% of it; base angle, reach, elbow and approach direction add
        # nothing on a held-out survey. The tower carries the cameras and the shoulder hinge, so they lean together;
        # it is the table that moves relative to both. Optional: a world without "tower_lean" ignores it.
        lean = d.get("tower_lean")
        self.lean = None if not lean else (float(lean["deg_per_shoulder_step"]), float(lean["deg_at_shoulder_0"]))

    def finger_shift(self, tag, grip):
        """How far (mm, in G) a finger tag has slid from where world.json has it, at this gripper value.
        Zero for any other tag, without a finger model, or without a gripper value."""
        side = self.finger_side.get(tag)
        if side is None or grip is None or not self.fingers:
            return np.zeros(3)
        return finger_offset(self.fingers, side, grip)

    def finger_offset_side(self, side, grip):
        """How far (mm, in G) finger L or R has slid from where it is at the reference gripper value."""
        if grip is None or not self.fingers or side not in self.fingers:
            return np.zeros(3)
        return finger_offset(self.fingers, side, grip)


    def tip_shift(self, grip):
        """How far (mm, in G) the fingertip point has moved from where it is at the reference gripper value:
        halfway between the two fingers' shifts (closing pushes the tips forward, ~20 mm).

        Kept per gripper value: the IK solver asks for this on every residual evaluation and the gripper does not
        move during a plan, so without the cache it was the same two finger_offset solves 275,000 times a plan.
        The returned array is read-only so a caller cannot write through the cache.
        """
        if grip is None or not self.fingers or len(set(self.finger_side.values())) < 2:
            return np.zeros(3)
        key = round(float(grip), 4)
        got = self._tip_shift.get(key)
        if got is None:
            got = 0.5 * (finger_offset(self.fingers, "L", key) + finger_offset(self.fingers, "R", key))
            got.flags.writeable = False
            self._tip_shift[key] = got
        return got

    @staticmethod
    def _lens(name, c, calib):
        if "camera_matrix" in c:
            return np.array(c["camera_matrix"], float), np.array(c["dist_coeffs"], float)
        if calib and name in calib:
            return np.asarray(calib[name][0], float), np.asarray(calib[name][1], float)
        base = name[:-1] if name.endswith("2") else name
        try:
            f = json.load(open(os.path.expanduser(f"~/calib_{base}.json")))
            return np.array(f["camera_matrix"], float), np.array(f["dist_coeffs"], float)
        except (OSError, ValueError, KeyError):
            return None, None

    def _guess_tip(self):
        """Until a touch test measures it: halfway between the two finger tags, a little further out."""
        f = [self.tags[t]["t"] for t in (31, 32) if t in self.tags and self.tags[t]["part"] == PARTS.index("G")]
        if len(f) == 2:
            p = (f[0] + f[1]) / 2.0
            return p + np.array([15.0, 0.0, 0.0])
        return np.array([200.0, 0.0, 0.0])

    # ---------------------------------------------------------------- the arm's parts
    def parts(self, joints, base, base_offset=0.0, dq=None):
        """Rotation and origin of every part (W, U, F, H, T, G) in the TABLE frame: list of (R, o).
        joints: slider values [shoulder, elbow, wrist, rotate, ...] (rotate defaults to level). base: base count (deg).
        base_offset: real base minus count (deg), e.g. from fit_base_offset. dq: optional per-joint corrections
        (shoulder, elbow, wrist, rotate, base), in the same units as the solve's."""
        RT, oT, Rw = self._frames(joints, base, base_offset, dq)
        # tower frame -> table frame: p_W = Rw^T (p_T - o_W)
        RwT, o0 = Rw.T, oT[0]
        out = [(RwT @ R, RwT @ (o - o0)) for R, o in zip(RT, oT)]
        if self.lean is not None:
            L, piv = self._lean(joints, Rw, dq)
            out = [out[0]] + [(L @ R, L @ (o - piv) + piv) for R, o in out[1:]]
        return out

    def frame(self, part, joints, base, base_offset=0.0, dq=None):
        """One part's (R, o), for callers that want a single frame. Same arithmetic as parts()[i] -- it is only
        the other five transforms that are skipped. The grasp solver asks for the gripper alone a few hundred
        thousand times a plan, and the six-way transform was most of what parts() cost."""
        i = PARTS.index(part) if isinstance(part, str) else part
        RT, oT, Rw = self._frames(joints, base, base_offset, dq)
        RwT = Rw.T
        R, o = RwT @ RT[i], RwT @ (oT[i] - oT[0])
        if self.lean is not None and i != 0:
            L, piv = self._lean(joints, Rw, dq)
            R, o = L @ R, L @ (o - piv) + piv
        return R, o

    def _lean(self, joints, Rw, dq=None):
        """(rotation, pivot) that leans everything on the tower -- the arm and the base cameras, not the table -- by
        the tower_lean for this shoulder value: about the horizontal line across the arm through the base axis at
        table height. Signed as fitted: the table was seen tilted by k s + b about z x (arm direction), so the tower
        turns by minus that."""
        k, b = self.lean
        s = float(joints[0]) + (float(dq[0]) if dq is not None else 0.0)
        u = Rw[0]                                             # T's x (along the arm) in the table frame
        perp = np.array([-u[1], u[0], 0.0])
        piv = np.array([self.axis[0], self.axis[1], self.table_height(self.axis[0], self.axis[1])])
        return rotvec(perp * np.radians(-(k * s + b))), piv

    def _frames(self, joints, base, base_offset=0.0, dq=None):
        """(RT, oT, Rw): every part in the TOWER frame, and the base rotation that takes it to the table."""
        a = self.arm
        nj = len(joints[:4])
        q = np.array(joints[:4], float) if nj == 4 else np.array(list(joints[:4]) + [self.ref[3]] * (4 - nj), float)
        if dq is None:                                        # the common case: no slack correction asked for
            d = q - self.ref                                  # adding a zero is exact, so this matches q + 0 - ref
            db = 0.0
        else:
            dq = np.asarray(dq, float)
            d = q + dq[:4] - self.ref
            db = dq[4]
        a1 = (a["th1"] + a["k1"] * d[0]) * _DEG
        a2 = a1 + (a["th2"] + a["k2"] * d[1]) * _DEG
        a3 = a2 + (a["th3"] + a["k3"] * d[2]) * _DEG
        a4 = (a["th4"] + a["k4"] * d[3] + a["k4b"] * d[3] ** 2) * _DEG
        RT, oT = [None] * 6, [None] * 6
        Rw = rot_z(-(self.gain * (float(base) + base_offset + self.base_zero + db)) * _DEG)
        RT[0], oT[0] = Rw, self.axis - Rw @ self.axis
        RT[1], oT[1] = rot_xz(a1), _Z3
        c1, s1 = np.cos(a1), np.sin(a1)
        elbow = np.array([a["L1"] * c1, 0.0, a["L1"] * s1])
        RT[2], oT[2] = rot_xz(a2), elbow
        L2 = a["L2"]
        RT[3] = rot_xz(a3)
        oT[3] = elbow + np.array([L2 * np.cos(a2), 0.0, L2 * np.sin(a2)])
        RT[4], oT[4] = _I3, _Z3
        n, pa = self._rotate_axis()
        Rr = rotvec(a4 * n)
        RT[5], oT[5] = RT[3] @ Rr, oT[3] + RT[3] @ (pa - Rr @ pa)
        return RT, oT, Rw

    def _rotate_axis(self):
        """The wrist-rotate axis direction and the point it passes through, in H. Fixed by the arm parameters, so
        it is kept rather than rebuilt on every parts() call; the key is the parameters themselves, so a solver
        that writes into w.arm still gets a fresh one."""
        a = self.arm
        key = (a["ax_u"], a["ax_v"], a["ax_y"], a["ax_z"])
        got = self._axis.get(key)
        if got is None:
            n = np.array([1.0, key[0], key[1]])
            n /= np.linalg.norm(n)
            pa = np.array([0.0, key[2], key[3]])
            n.flags.writeable = pa.flags.writeable = False
            got = self._axis[key] = (n, pa)
        return got

    def point(self, part, p_local, joints, base, **kw):
        R, o = self.frame(part, joints, base, **kw)
        return R @ np.asarray(p_local, float) + o

    def tip(self, joints, base, grip=None, **kw):
        """The fingertip point in table mm (grip: the gripper value; with a finger model the tips move with it)."""
        return self.point("G", self.tip_in_G + self.tip_shift(self._grip_now(grip, kw.get("dq"))), joints, base, **kw)

    # ---------------------------------------------------------------- cameras and tags
    def camera_pose(self, cam, joints, base, **kw):
        """(R, t) taking TABLE points into this camera's frame: p_cam = R p + t."""
        c = self.cams[cam]
        Rp, op = self.frame(c["parent"], joints, base, **kw)
        R = c["R"] @ Rp.T
        return R, c["t"] - R @ op

    def project(self, cam, pts_table, joints, base, **kw):
        R, t = self.camera_pose(cam, joints, base, **kw)
        c = self.cams[cam]
        return project(np.atleast_2d(pts_table) @ R.T + t, c["K"], c["dist"])

    @staticmethod
    def _grip_now(grip, dq):
        """The gripper value with its slack (dq[5], from the arm filter or the solve) added, or None."""
        if grip is None:
            return None
        return float(grip) + (float(dq[5]) if dq is not None and len(dq) > 5 else 0.0)

    def tag_corners_table(self, tag, joints, base, grip=None, **kw):
        """A tag's corners in table mm. grip: the gripper value (moves the finger tags with a finger model)."""
        g = self.tags[tag]
        Rp, op = self.parts(joints, base, **kw)[g["part"]]
        X = tag_corners(g["size"]) @ g["R"].T + g["t"] + self.finger_shift(tag, self._grip_now(grip, kw.get("dq")))
        return X @ Rp.T + op

    def predict_tag(self, tag, cam, joints, base, grip=None, **kw):
        return self.project(cam, self.tag_corners_table(tag, joints, base, grip=grip, **kw), joints, base, **kw)

    def locate_tag(self, cam, corners, size, joints, base, **kw):
        """A tag this camera sees -> (R, t) of the tag in table coordinates (t = its centre, mm), or None."""
        import cv2
        c = self.cams[cam]
        obj = tag_corners(size).astype(np.float32)
        img = np.asarray(corners, np.float32).reshape(4, 1, 2)
        ok, r, t = cv2.solvePnP(obj, img, c["K"], c["dist"], flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        Rtc, ttc = cv2.Rodrigues(r)[0], t.ravel()
        R, tc = self.camera_pose(cam, joints, base, **kw)
        return R.T @ Rtc, R.T @ (ttc - tc)

    # ---------------------------------------------------------------- base zero from what the cameras see
    def fit_base_offset(self, sightings, joints, base, span=3.0, tilt=False):
        """sightings: [(camera, tag, corners 4x2), ...] of TABLE tags seen by cameras that turn with the base (tower,
        webcam). Returns (offset deg, median px error at it): how far the base really is from its count.

        tilt: also let the table tilt about the base axis at table height (two angles), and keep only the turn. The
        tower leans under the arm (~1.8 deg at home) and a turn alone cannot match that, so the fit
        came out at 27 px and the base angle was dragged 0.26 deg off. Tilt and turn are different motions; with
        the tilt free the same reading fitted at 4.8 px and gave the same base angle (-1.34..-1.39) from any half
        of the tags. The tilt it found is left in self.last_tilt_deg (about the table's x and y)."""
        use = [(c, t, np.asarray(px, float)) for c, t, px in sightings
               if c in self.cams and t in self.tags and self.tags[t]["part"] == 0 and self.cams[c]["parent"] == 4]
        if not use:
            return 0.0, float("nan")

        # One projection per camera per trial angle, all its tags at once, instead of the whole arm recomputed for
        # every tag: ~110 trial angles x ~60 tags made this the slow part of every look, and of any replay that
        # refits it per pose (30+ min over 600 survey poses). The same numbers: each tag's error is still
        # the mean over its four corners and the score the median over tags. Table tags hang off W, so their
        # corners only go through W's own frame, which is taken per trial angle exactly as predict_tag did.
        groups = {}
        for c, t, px in use:
            g = self.tags[t]
            X = tag_corners(g["size"]) @ g["R"].T + g["t"] + self.finger_shift(t, None)
            groups.setdefault(c, ([], []))
            groups[c][0].append(X)
            groups[c][1].append(px)
        groups = {c: (np.vstack(X), np.vstack(P), len(X)) for c, (X, P) in groups.items()}

        def err(off):
            R0, o0 = self.frame(0, joints, base, base_offset=off)
            e = [np.linalg.norm(self.project(c, X @ R0.T + o0, joints, base, base_offset=off) - P, axis=1)
                 .reshape(n, 4).mean(1) for c, (X, P, n) in groups.items()]
            return float(np.median(np.concatenate(e)))
        best = min(np.linspace(-span, span, 61), key=err)
        lo, hi = best - span / 30, best + span / 30
        for _ in range(25):                                         # golden-section refine
            m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
            if err(m1) < err(m2):
                hi = m2
            else:
                lo = m1
        off = (lo + hi) / 2
        if not tilt:
            return off, err(off)
        R0, o0 = self.frame(0, joints, base, base_offset=off)
        piv = np.array([self.axis[0], self.axis[1], self.table_height(self.axis[0], self.axis[1])])
        tab = {c: (X @ R0.T + o0, P, n) for c, (X, P, n) in groups.items()}

        def moved(p):
            Rt = rotvec(np.radians([p[1], p[2], 0.0]))
            return {c: ((X - piv) @ Rt.T + piv, P, n) for c, (X, P, n) in tab.items()}

        def res(p):
            return np.concatenate([(self.project(c, X, joints, base, base_offset=p[0]) - P).ravel()
                                   for c, (X, P, n) in moved(p).items()])
        x = _robust_fit(res, np.array([off, 0.0, 0.0]))
        self.last_tilt_deg = (float(x[1]), float(x[2]))
        e = [np.linalg.norm(self.project(c, X, joints, base, base_offset=x[0]) - P, axis=1).reshape(n, 4).mean(1)
             for c, (X, P, n) in moved(x).items()]
        return float(x[0]), float(np.median(np.concatenate(e)))

    # ---------------------------------------------------------------- the table
    def table_height(self, x, y):
        """Height (mm, table frame z) of the table surface under (x, y): the plane through the mat tags."""
        if self._plane is None:
            P = np.array([self.tags[t]["t"] for t in MAT_TAGS if t in self.tags and self.tags[t]["part"] == 0])
            A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
            self._plane = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
        a, b, c = self._plane
        return a * x + b * y + c
