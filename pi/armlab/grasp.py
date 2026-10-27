"""
Grasp planning with the world model.

For a block located by cube.locate: solve the joints (shoulder, elbow, wrist, rotate) and base angle that put the
calibrated fingertip at the grasp point, with the hand pitched down and the fingers square to one of the block's
faces; then a hover above it and the path in between. Every pose on the way passes the exact table check
(tablecheck.py) and the joint limits.

The fingertip point (world.json tip_in_G) moves with the gripper value (finger model), so plans are made with the
gripper at the value it will actually have on the way down (GRIP_OPEN).

    p = plan(world, check, cube, margin)          p["hover"], p["grasp"], p["path"] (joint lists), p["why"] if none

Joints are [shoulder, elbow, wrist, rotate, gripper, base] as arm_angles() returns them (slider values; base in deg).
"""
import math
import os

import numpy as np

from .config import ROLL_RANGE
from .world import PARTS

GRIP_OPEN = 55.0              # ~67 mm between the pads (the block is 39.7 mm). Opening further mostly slides the
                              # fingers back rather than apart
GRIP_CLOSE = 114.0            # closed on the block: ~38.5 mm between the pads, a firm grip
DEPTH_MM = 20.0               # default grasp depth: fingertips this far below the block's top (show uses 10)
HOVER_MM = 60.0               # the hover is this far back along the way in
GRASP_MARGIN_MM = 12.0        # table margin for the last part of the way in only (the fingers have to get down beside
                              # the block); everything else keeps the normal margin
STEP_MM = 5.0                 # the way in is checked every this many mm
PITCH_PREFER = 40.0           # hand pitch (below level) the planner leans toward
PITCH_RANGE = (25.0, 90.0)    # straight down only reaches close to the base; across the mat 30-45 deg works best
GRASP_PULL_MM = (0.0, 5.0, 10.0, 15.0, 20.0)
# Aim at the block's middle first; if that's out of reach, at a point up to 20 mm back toward the tower. At arm's
# length the near side is reachable when the middle isn't, and it's a fine place to hold a 39.7 mm block (grabbing
# by hand at 380-417 mm worked with the fingertip ~20 mm off centre). 20 is half the block: past that the fingers
# would close on nothing.
PITCH_WEIGHT = 0.0            # don't pull the grasp solve toward PITCH_PREFER: it dragged poses into the joint limits
ELBOW_MAX = 120.0             # the elbow's spring hits the arm above ~125
BASE_RANGE = (-57.0, 90.0)    # the base's working range (limited by the cables)
MAP_LO, MAP_HI = (0.0, 0.0, 0.0), (150.0, 160.0, 148.0)        # placeholders: set_limits() takes these from the map
SLACK_SIGMA = (1.11, 1.56, 0.96)   # measured gear slack (1 sigma, slider steps): shoulder, elbow, wrist
SLACK_K = 2.5                      # plans stay this many sigma inside each joint's end (per joint: one number for
                                   # all three made the wrist, with the least slack, pay the most margin)
SLACK_ROOM = tuple(SLACK_K * v for v in SLACK_SIGMA)
JOINT_LO = tuple(v + r for v, r in zip(MAP_LO, SLACK_ROOM))
JOINT_HI = (MAP_HI[0] - SLACK_ROOM[0], ELBOW_MAX, MAP_HI[2] - SLACK_ROOM[2])


def set_limits(zone):
    """Take the joint range from the safety map rather than constants in this file. The map is what actually refuses
    moves, and its range changes when the arm is rebuilt (after one servo-horn reseat the shoulder ran 1-72 while
    this file still said 0-150, and the planner kept producing hovers the map refused)."""
    global MAP_LO, MAP_HI, JOINT_LO, JOINT_HI, SLACK_ROOM, _LIMIT_GEN
    SLACK_ROOM = tuple(SLACK_K * v for v in SLACK_SIGMA)
    MAP_LO = tuple(float(v) for v in zone.lo[:3])
    MAP_HI = tuple(float(v) for v in zone.hi[:3])
    JOINT_LO = tuple(v + r for v, r in zip(MAP_LO, SLACK_ROOM))
    JOINT_HI = (MAP_HI[0] - SLACK_ROOM[0], min(MAP_HI[1] - SLACK_ROOM[1], ELBOW_MAX),
                MAP_HI[2] - SLACK_ROOM[2])
    _GRID.clear()                             # the start grid is built between the limits, so it's stale now
    _LIMIT_GEN += 1                           # and so are the poses reach_all found
    return JOINT_LO, JOINT_HI
FAST_IK = True                # use the numba-compiled solve (fastik.py) when it works: ~50x faster, same maths
TOL_MM, TOL_DEG = 1.0, 2.0    # a solved pose must hit the point within this and the angles within this
SQUARE_TOL_DEG = 12.0         # the fingers may meet the block this far off square (the squeeze squares it up).
                              # At 20 deg the fingers closed on the corners and both grabs failed
SQUARE_TOL_LOOSE_DEG = 17.0   # fallback when nothing is within SQUARE_TOL_DEG (the wrist rotate can't reach square
                              # for some block angles); open fingers still clear a block turned 17 deg
SQUARE_WEIGHT = 0.3           # in the grasp solve, reaching the point matters more than being square


class square_tol:
    """with square_tol(deg): plan with SQUARE_TOL_DEG set to deg, then put it back."""

    def __init__(self, deg):
        self.deg = float(deg)

    def __enter__(self):
        global SQUARE_TOL_DEG
        self.was, SQUARE_TOL_DEG = SQUARE_TOL_DEG, self.deg

    def __exit__(self, *exc):
        global SQUARE_TOL_DEG
        SQUARE_TOL_DEG = self.was


_G_PART = PARTS.index("G")


def hand(w, q):
    """(fingertip xyz, hand rotation) in table mm for q = [s, e, w, rot, grip, base]."""
    R, o = w.frame(_G_PART, q[:4], q[5])
    tip = R @ (w.tip_in_G + w.tip_shift(q[4])) + o
    return tip, R


def errors(w, q, target, pitch_deg, face, wp=1.0, ws=1.0):
    """Residuals: fingertip minus target (mm, 3), pitch error (deg, times wp), and how far the line between the
    fingers is off square to the block face (deg; the fingers are symmetric, so either way round counts)."""
    tip, R = hand(w, q)
    x, y = R[:, 0], R[:, 1]                       # x: where the fingers point, y: the way they close
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -x[2]))))
    y2 = y[:2]
    yh = y2 / max(1e-9, float(np.sqrt(y2 @ y2)))
    sq = math.degrees(math.asin(max(-1.0, min(1.0, yh[0] * math.sin(face) - yh[1] * math.cos(face)))))
    return np.concatenate([tip - target, [wp * (pitch - pitch_deg), ws * sq]])


def pitch_of(w, q):
    _, R = hand(w, q)
    return math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))


def clamp(q):
    q = list(q)
    for i in range(3):
        q[i] = min(JOINT_HI[i], max(JOINT_LO[i], q[i]))
    q[3] = min(ROLL_RANGE[1], max(ROLL_RANGE[0], q[3]))
    q[5] = min(BASE_RANGE[1], max(BASE_RANGE[0], q[5]))
    return q


def solve(w, target, pitch_deg, face, grip, start, iters=40, wp=1.0, ws=1.0):
    """Damped Gauss-Newton (Levenberg-Marquardt style) on (shoulder, elbow, wrist, rotate, base) from `start`.
    Returns (q, residual vector)."""
    global FAST_IK
    if FAST_IK:                                   # the compiled version; it returns None for worlds it can't handle
        try:
            from . import fastik
            lim = np.array(list(JOINT_LO) + list(JOINT_HI) + list(ROLL_RANGE) + list(BASE_RANGE), float)
            got = fastik.solve(w, target, pitch_deg, face, grip, start, iters, wp, ws, lim)
        except Exception as e:                    # e.g. numba can't compile here: use Python from now on
            print(f"   (compiled solve unavailable, using the Python one: {type(e).__name__}: {e})")
            FAST_IK, got = False, None
        if got is not None:
            return got
    q = clamp(list(start[:4]) + [grip, start[5]])
    idx = (0, 1, 2, 3, 5)
    lam = 1e-2
    r = errors(w, q, target, pitch_deg, face, wp, ws)
    for _ in range(iters):
        J = np.zeros((5, 5))
        for k, i in enumerate(idx):
            dq = list(q)
            dq[i] += 0.05
            J[:, k] = (errors(w, dq, target, pitch_deg, face, wp, ws) - r) / 0.05
        step = np.linalg.solve(J.T @ J + lam * np.eye(5), -J.T @ r)
        cand = list(q)
        for k, i in enumerate(idx):
            cand[i] += float(np.clip(step[k], -15.0, 15.0))
        cand = clamp(cand)
        rc = errors(w, cand, target, pitch_deg, face, wp, ws)
        if np.sqrt(rc @ rc) < np.sqrt(r @ r) - 1e-9:
            q, r, lam = cand, rc, max(1e-4, lam / 3)
        else:
            lam *= 5
            if lam > 1e4:
                break
    return q, r


def solved(r, wp=1.0, ws=1.0):
    """Did the solve succeed? It must hit the point, be square to the face (strictly when ws=1, within
    SQUARE_TOL_DEG when squareness is only weighted), and hit the pitch when that's required (wp=1)."""
    square = abs(r[4]) <= TOL_DEG if ws >= 1.0 else abs(r[4]) / ws <= SQUARE_TOL_DEG
    return np.abs(r[:3]).max() <= TOL_MM and square and (wp < 1.0 or abs(r[3]) <= TOL_DEG)


def base_toward(w, target):
    """The base angle that points the arm's plane at the target."""
    ax = w.axis[:2]

    def bearing(b):
        tip, _ = hand(w, [PROBE_POSE[0], PROBE_POSE[1], PROBE_POSE[2], 42.0, 80.0, b])
        return math.atan2(tip[1] - ax[1], tip[0] - ax[0])
    b0, b1 = bearing(0.0), bearing(10.0)
    per_deg = math.atan2(math.sin(b1 - b0), math.cos(b1 - b0)) / 10.0
    want = math.atan2(target[1] - ax[1], target[0] - ax[0])
    return math.degrees(0.0) + math.atan2(math.sin(want - b0), math.cos(want - b0)) / per_deg


PATH_PITCH_WEIGHT = 0.3       # further back along the way in the hand may tilt by up to PATH_PITCH_DRIFT (holding
PATH_PITCH_DRIFT = 35.0       # the grasp's angle all the way back pushed the shoulder past its limit); the angle
                              # only has to be exact where the fingers close (--pitch-drift)
STRAIGHT_MM = 20.0            # the last part of the way in keeps the grasp's angle exactly


def reach(w, target, pitch_deg, face, grip, near, ws=1.0, wp=1.0):
    """The pose for this target, solved starting from the nearby pose `near`. wp=1: exactly this pitch; less: within
    PATH_PITCH_DRIFT of it. None if it can't be reached."""
    q, r = solve(w, target, pitch_deg, face, grip, near, ws=ws, wp=wp)
    if wp >= 1.0:
        return q if solved(r, 1.0, ws) else None
    return q if solved(r, wp, ws) and abs(pitch_of(w, q) - pitch_deg) <= PATH_PITCH_DRIFT else None


PROBE_POSE = (36.0, 60.0, 60.0)   # any reachable mid-range pose, only used to find which way the arm points
GRID_STEP = 5.0                # the starting-pose grid for the solver: shoulder/elbow/wrist every this many steps.
                               # Too coarse and plans come and go between neighbouring block positions
_GRID = {}


def tip_grid(w, grip):
    """Fingertip position (tower frame, so independent of the base) and hand pitch for a grid of shoulder, elbow
    and wrist values with the rotate level. The solver's starting poses come from it. Built once per world and
    gripper value, and cached on disk."""
    key = (id(w), round(grip))
    if key not in _GRID:
        cached = _grid_from_disk(w, grip)
        if cached is not None:
            _GRID[key] = cached
            return _GRID[key]
        Q, T, P = [], [], []
        for s_ in np.arange(JOINT_LO[0], JOINT_HI[0] + 0.1, GRID_STEP):
            for e_ in np.arange(JOINT_LO[1], JOINT_HI[1] + 0.1, GRID_STEP):
                for w_ in np.arange(JOINT_LO[2], JOINT_HI[2] + 0.1, GRID_STEP):
                    q = [s_, e_, w_, 42.0, grip, 0.0]
                    parts = w.parts(q[:4], 0.0)
                    RT, oT = parts[PARTS.index("T")]
                    tip, R = hand(w, q)
                    Q.append((s_, e_, w_))
                    T.append(RT.T @ (tip - oT))
                    P.append(math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))))
        _GRID[key] = (np.array(Q), np.array(T), np.array(P))
        _grid_to_disk(w, grip, _GRID[key])
    return _GRID[key]


def _grid_name(w, grip):
    """A cache file name that changes whenever the grid would: world model, fingertip, gripper value, joint range."""
    import hashlib
    # The fingertip has to be in the key: refitting it doesn't change the world's "made" date, and a stale grid
    # quietly aims every plan at the old fingertip.
    made = str(w.d.get("made", "")) + str(w.arm) + str(list(w.ref)) + str(list(np.round(w.tip_in_G, 4))) \
        + ("" if getattr(w, "lean", None) is None else str(w.lean))
    key = f"{made}|{round(grip)}|{JOINT_LO}|{JOINT_HI}|{GRID_STEP}"
    return "tipgrid_" + hashlib.md5(key.encode()).hexdigest()[:16] + ".npz"


def _grid_dir():
    try:
        from .state import S
        return S.data_dir
    except Exception:
        return None


def _grid_from_disk(w, grip):
    """The saved grid, if one matches. Building it (~9000 forward-kinematics solves) was most of the planning time
    on the Pi, and it never changes between runs."""
    d = _grid_dir()
    if not d:
        return None
    try:
        z = np.load(os.path.join(d, _grid_name(w, grip)))
        return z["Q"], z["T"], z["P"]
    except Exception:
        return None


def _grid_to_disk(w, grip, g):
    d = _grid_dir()
    if not d:
        return
    try:
        np.savez_compressed(os.path.join(d, _grid_name(w, grip)), Q=g[0], T=g[1], P=g[2])
    except Exception:
        pass


def seeds_for(w, target, grip, b, n=6, within=20.0):
    """Starting poses for the solve: grid poses whose fingertip lands near the target, preferring mid-range joints
    and a pitch near PITCH_PREFER."""
    Q, T, P = tip_grid(w, grip)
    RT, oT = w.parts([123.0, 54.0, 68.0, 42.0], b)[PARTS.index("T")]
    tT = RT.T @ (np.asarray(target, float) - oT)
    d = np.linalg.norm(T - tT, axis=1)
    near = np.where(d < within)[0]
    if len(near) == 0:
        near = np.argsort(d)[:n]
    room = np.min(np.minimum(Q - np.array(JOINT_LO), np.array(JOINT_HI) - Q), axis=1)
    score = d[near] + 0.5 * np.abs(P[near] - PITCH_PREFER) - 0.5 * np.minimum(room[near], 30.0)
    return [list(Q[i]) + [42.0, grip, b] for i in near[np.argsort(score)][:n]]


ROT_SEEDS = (0.0, 112.0, 75.0)   # wrist rotate starts to try if the mid-range one can't get square. Starting at mid
                                 # range only, a block presenting a corner to the arm came out "out of reach" at every
                                 # distance; a square grasp existed, with the rotate near the end of its range


_LIMIT_GEN = 0                       # bumped by set_limits, which invalidates the cached poses below


def reach_all(w, target, face, grip):
    """Every distinct grasp pose for this target (the arm can often reach a point several ways) with a pitch in
    PITCH_RANGE, most joint room first.

    Cached per target: plan_best calls plan() once per hover height, and the grasp doesn't depend on the hover, so
    without the cache it repeated the same IK searches six times (one plan was 1514 solves)."""
    keep = w.__dict__.setdefault("_reach_cache", {})       # kept on the world object, so it can't outlive it
    key = (_LIMIT_GEN, float(target[0]), float(target[1]), float(target[2]), float(face), round(float(grip), 3),
           SQUARE_TOL_DEG)
    got = keep.get(key)
    if got is not None:
        return [list(q) for q in got]                      # copies: callers put these poses in their paths
    b = base_toward(w, target)
    seeds = seeds_for(w, target, grip, b)
    out = []

    def collect(sts):
        for st in sts:
            q, r = solve(w, target, PITCH_PREFER, face, grip, st, wp=PITCH_WEIGHT, ws=SQUARE_WEIGHT)
            if solved(r, PITCH_WEIGHT, SQUARE_WEIGHT) and PITCH_RANGE[0] <= pitch_of(w, q) <= PITCH_RANGE[1] \
                    and all(max(abs(q[i] - o[i]) for i in range(3)) > 3.0 for o in out):
                out.append(q)

    collect(seeds)
    if not out:                                  # nothing got square from mid-rotate: try turning the wrist
        for rot in ROT_SEEDS:
            collect([st[:3] + [rot] + st[4:] for st in seeds])
            if out:
                break
    out = sorted(out, key=lambda q: limit_room(q), reverse=True)
    if len(keep) > 4000:                                   # don't let a long replay grow it without bound
        keep.clear()
    keep[key] = [list(q) for q in out]
    return out


def room_text(q, check=None, base=None):
    """Each joint's room at this pose, both ways, plus the table gap. "shoulder 52 [+48 -17]" means 48 steps above
    its lower limit and 17 below its upper one; the joint with a small number is the one limiting the pose."""
    names = ("shoulder", "elbow", "wrist")
    txt = "  ".join(f"{names[i]} {q[i]:.0f} [+{q[i] - JOINT_LO[i]:.0f} -{JOINT_HI[i] - q[i]:.0f}]"
                    for i in range(3))
    if check is not None:
        try:
            g, _ = check.gaps(list(q[:5]), float(base if base is not None else q[5]))
            txt += f"   table {g:.0f} mm"
        except Exception:
            pass
    return txt


def limit_room(q):
    """How far (slider steps) the nearest joint is from its limit."""
    return min(min(q[i] - JOINT_LO[i], JOINT_HI[i] - q[i]) for i in range(3))


# Ways in: how the hand backs off from the grasp to the hover, as (name, share along the fingers, share radially
# out from the base axis); the rest is straight up. At the steep pitches close blocks need, "along the fingers" and
# "straight up" are almost the same direction, so every hover ended up straight above the grasp and needed vertical
# room the folded arm doesn't have close in.
APPROACHES = (("along the fingers", 1.0, 0.0), ("halfway", 0.5, 0.0), ("from above", 0.0, 0.0))


HOVER_SHORTER = (1.0,)              # fractions of the hover distance way_in tries (plan_best already steps the hover
                                    # down itself, so trying shorter ones here just repeated that search)
ROOM_PLENTY = 4.0                   # a way in with this much joint room is taken immediately
ROOM_ENOUGH = 2.0                   # enough room to keep without looking for a shorter way in (slack is ~1.1 steps,
                                    # 1 sigma)


VIEW_EDGE_PX = 60.0  # the block's top tag must be this far inside the gripper camera's picture at the hover: the
                     # shoulder's slack (~3 deg) shifts the picture ~60 px


def top_tag(target, yaw_deg, depth=DEPTH_MM):
    """The block's top tag corners (table mm), given the grasp point (depth below the top) and the block's turn."""
    from .config import TAG_MM, TARGET_TAG
    from .world import rotvec, tag_corners
    top = np.asarray(target, float) + [0.0, 0.0, depth]
    return tag_corners(TAG_MM.get(TARGET_TAG, 30.0)) @ rotvec([0.0, 0.0, math.radians(yaw_deg)]).T + top


CAM_SIZE_PX = {"gripper": (1152, 864), "torre": (1280, 960), "webcam": (1920, 1080)}
                     # each camera's real picture size. Estimating it as twice the lens centre was off by up to
                     # 140 px and said the webcam couldn't see blocks it was seeing


GRIPPER_USABLE_Y_PX = 540.0  # the gripper camera can only read the block's tag above this line; the hand's own
                             # hardware covers the bottom of its picture. From every look on record: tags with their
                             # lowest corner under 500 px were seen 48/49 times, below 650 px only 5/117


def in_view(w, X, q, cam="gripper", edge=VIEW_EDGE_PX):
    """Would this camera see all the points X (table mm), at least `edge` px inside its picture, with the arm at q?
    True if the world has no such camera. For the gripper camera the picture effectively ends at
    GRIPPER_USABLE_Y_PX."""
    if cam not in w.cams:
        return True
    K = w.cams[cam]["K"]
    wide, high = CAM_SIZE_PX.get(cam, (2 * K[0, 2], 2 * K[1, 2]))
    if cam == "gripper" and GRIPPER_USABLE_Y_PX is not None:
        high = min(high, GRIPPER_USABLE_Y_PX + edge)
    R, t = w.camera_pose(cam, q[:4], q[5])
    if ((np.asarray(X) @ R.T + t)[:, 2] <= 0).any():
        return False
    px = w.project(cam, X, q[:4], q[5])
    return bool(np.isfinite(px).all() and (px[:, 0] >= edge).all() and (px[:, 0] <= wide - edge).all()
                and (px[:, 1] >= edge).all() and (px[:, 1] <= high - edge).all())


def way_in(w, check, target, g, face, grip, hover_mm, margin, grasp_margin, approaches=None, view=None, joint=True):
    """From grasp pose g: the hover and the checked path in (hover first) as a plan dict, or a string saying why
    not. view: the block's top tag corners; a hover the gripper camera can see them from is preferred
    (plan["sees"]), and a blind one is the fallback."""
    approaches = APPROACHES if approaches is None else approaches
    why, blind, best = None, None, None
    for k in HOVER_SHORTER:
        got = _way_in(w, check, target, g, face, grip, hover_mm * k, margin, grasp_margin, approaches, view, joint)
        if isinstance(got, dict):
            got["hover_mm"] = hover_mm * k
            # Take the longest hover that has enough room, not the roomiest: shortening always adds room, so
            # scoring by room alone walks to the shortest hover, which ends up inside the block.
            if got["sees"]:
                if got["room"] >= ROOM_ENOUGH:
                    return got
                best = got if best is None or got["room"] > best["room"] else best
            elif blind is None or got["room"] > blind["room"]:
                blind = got
        else:
            why = why or got
    return best or blind or why


def _way_in(w, check, target, g, face, grip, hover_mm, margin, grasp_margin, approaches, view=None, joint=True):
    n = int(math.ceil(hover_mm / STEP_MM))
    pitch = pitch_of(w, g)
    _, R = hand(w, g)
    why, blind, best = None, None, None
    for app in approaches:
        how, share, out = app if len(app) == 3 else (app[0], app[1], 0.0)
        d = share * -R[:, 0] + (1.0 - share - out) * np.array([0.0, 0.0, 1.0])
        if out:
            radial = target[:2] - w.axis[:2]                  # horizontally away from the base axis
            n_r = np.linalg.norm(radial)
            if n_r < 1.0:
                continue                                      # right over the axis "out" is undefined
            d = d + out * np.array([radial[0] / n_r, radial[1] / n_r, 0.0])
        back = d / np.linalg.norm(d) * hover_mm               # the hover is this far back along the way in
        for path in ways_in(w, target, back, g, pitch, face, grip, n):
            if path is None:
                why = why or (f"face {math.degrees(face):.0f}, pitch {pitch:.0f}: no way in (shoulder {g[0]:.0f} "
                              f"elbow {g[1]:.0f} wrist {g[2]:.0f})")
                continue
            hop = max_hop(path)
            if hop > MAX_HOP_DEG:
                why = why or (f"face {math.degrees(face):.0f}, pitch {pitch:.0f}: the way in refolds the arm "
                              f"({hop:.0f} deg between two poses 5 mm apart)")
                continue
            gaps = [check.gaps(p[:5], p[5])[0] for p in path]
            if gaps[0] < margin:
                why = why or (f"face {math.degrees(face):.0f}, pitch {pitch:.0f}: the hover is {gaps[0]:.0f} mm from "
                              f"the table/plywood (margin {margin:.0f})")
                continue
            if min(gaps) < grasp_margin:
                why = why or (f"face {math.degrees(face):.0f}, pitch {pitch:.0f}: {min(gaps):.0f} mm to the "
                              f"table/plywood on the way in (grasp margin {grasp_margin:.0f})")
                continue
            got = {"pitch": pitch, "face_deg": math.degrees(face),
                   "off_square_deg": float(errors(w, path[-1], target, pitch, face)[4]), "approach": how,
                   "share": share, "hover": path[0], "grasp": path[-1], "path": path,
                   "min_gap_mm": float(min(gaps)), "hover_gap_mm": float(gaps[0]), "target": target, "grip": grip,
                   "sees": view is None or in_view(w, view, path[0]), "back": back / hover_mm}
            got["room"] = float(min(limit_room(q) for q in path))
            if got["sees"]:                                  # keep the roomiest way in the camera can see from
                best = got if best is None or got["room"] > best["room"] else best
            else:
                blind = got if blind is None or got["room"] > blind["room"] else blind

    if best is None and blind is None and joint:             # no straight line works: back out on the joints
        best, blind, why = _joint_way_in(w, check, target, g, face, grip, hover_mm, margin, grasp_margin, view, pitch,
                                         why)
    return best or blind or why


def _joint_way_in(w, check, target, g, face, grip, hover_mm, margin, grasp_margin, view, pitch, why):
    """_way_in's joint-space fallback: (best seen, best blind, why). Tries a retreat ending in a straight vertical
    tail first (joint_ways_in_tail), and the plain retreat only if no tail fits."""
    best, blind = None, None
    for name, paths in ((JOINT_TAIL_APPROACH, joint_ways_in_tail(w, target, g, grip, hover_mm)),
                        (JOINT_APPROACH, joint_ways_in(w, target, g, grip, hover_mm))):
        for path in paths:
            room_here = float(min(limit_room(q) for q in path))
            if room_here <= 0.0 or max_hop(path) > MAX_HOP_DEG:   # a joint past its limit, or the arm refolds
                continue
            gaps = [check.gaps(p[:5], p[5])[0] for p in path]
            if gaps[0] < margin or min(gaps) < grasp_margin:
                why = why or (f"face {math.degrees(face):.0f}, pitch {pitch:.0f}: backing out on the joints comes "
                              f"{min(gaps):.0f} mm from the table/plywood (margin {margin:.0f}, "
                              f"on the way in {grasp_margin:.0f})")
                continue
            tip_back = hand(w, path[0])[0] - target
            got = {"pitch": pitch, "face_deg": math.degrees(face),
                   "off_square_deg": float(errors(w, path[-1], target, pitch, face)[4]),
                   "approach": name, "share": 0.0, "hover": path[0], "grasp": path[-1], "path": path,
                   "min_gap_mm": float(min(gaps)), "hover_gap_mm": float(gaps[0]), "target": target, "grip": grip,
                   "sees": view is None or in_view(w, view, path[0]),
                   "back": tip_back / max(1e-9, np.linalg.norm(tip_back))}
            got["room"] = room_here
            if got["sees"]:
                best = got if best is None or got["room"] > best["room"] else best
                if best["room"] >= ROOM_PLENTY:              # clearly good enough; the other retreats cost time
                    break
            else:
                blind = got if blind is None or got["room"] > blind["room"] else blind
        if best is not None or blind is not None:
            break                                            # found one with a tail: no need for the plain retreat
    return best, blind, why


MAX_HOP_DEG = 15.0            # no joint may move more than this between neighbouring poses of a way in (normal hops
                              # are 5-8). Some plans switched arm configuration between the hover and the rest of the
                              # path, so the first hop refolded the whole arm above the block, unchecked


def max_hop(path):
    """The biggest single-joint change (shoulder/elbow/wrist) between neighbouring poses of a path."""
    return max((max(abs(b[i] - a[i]) for i in range(3)) for a, b in zip(path[:-1], path[1:])), default=0.0)


JOINT_APPROACH = "elbow then wrist"   # the "approach" name for the joint-space retreat
JOINT_TAIL_APPROACH = "elbow then wrist, straight down"   # ...and for the one ending in a vertical tail
JOINT_APPROACHES = (JOINT_APPROACH, JOINT_TAIL_APPROACH)
TAIL_MM = (25.0, 20.0, 15.0, 12.0, 10.0)   # vertical tail lengths to try, longest first. It has to be clearly longer
                                           # than the grasp depth: with 15 mm at depth 10, the retreat started ~5 mm
                                           # above the block's top and its sideways steps dragged blocks off


def straight_up(w, q0, grip, mm, step=5.0, sideways=1.0):
    """True poses that lift the fingertip straight up from q0 in `step` mm steps to `mm`, moving only shoulder,
    elbow and wrist and staying inside the joint limits; or None. Gauss-Newton on the three joints.

    Three joints moving the tip in one vertical plane leaves one degree of freedom spare. Plain least squares spent
    it on whichever joint moved least and ran the shoulder or wrist past its limit at most close grasps (9 of 9 in
    one sweep), even when another split stayed inside. So each joint's step is damped by how close it is to its
    limit, which pushes the motion onto the joints that have room."""
    q = [float(v) for v in q0[:6]]
    q[4] = grip
    tip0 = hand(w, q)[0]
    out = []
    for k in range(1, int(np.ceil(mm / step - 1e-9)) + 1):
        target = tip0 + [0.0, 0.0, min(mm, k * step)]
        for _ in range(40):
            e = hand(w, q)[0] - target
            if np.linalg.norm(e) < 0.2:
                break
            J = np.column_stack([(hand(w, q[:i] + [q[i] + 0.05] + q[i + 1:])[0] - hand(w, q)[0]) / 0.05
                                 for i in range(3)])
            room = [max(0.25, min(q[i] - JOINT_LO[i], JOINT_HI[i] - q[i])) for i in range(3)]
            Wd = np.diag([1e-3 + 4.0 / r ** 2 for r in room])         # near its limit, a joint barely moves
            dq = np.linalg.solve(J.T @ J + Wd, -J.T @ e)
            for i in range(3):
                q[i] = min(JOINT_HI[i], max(JOINT_LO[i], q[i] + float(np.clip(dq[i], -5.0, 5.0))))
        tip = hand(w, q)[0]
        if np.linalg.norm(tip - target) > 0.5 or np.linalg.norm((tip - tip0)[:2]) > sideways \
                or any(q[i] < JOINT_LO[i] or q[i] > JOINT_HI[i] for i in range(3)):
            return None
        out.append(list(q))
    return out


def joint_ways_in_tail(w, target, g, grip, hover_mm):
    """A joint-space retreat from the top of a straight vertical tail above grasp g, then the tail down to g
    (hover first).

    The plain joint-space way in moves the fingertips ~15 mm sideways in its last 10 mm, along the face the fingers
    grip, and none of 45 such grabs on record held (the pads caught the block's far corner). With the last TAIL_MM
    straight down, the fingers arrive beside the block like a straight way in, and the retreat above still gives
    the extra reach close in."""
    for tail in TAIL_MM:
        up = straight_up(w, g, grip, tail)
        if up is None:
            continue
        down = up[-2::-1] + [list(g)]                        # from just under the tail's top down to the grasp
        for path in joint_ways_in(w, target, up[-1], grip, max(0.0, hover_mm - tail)):
            yield path + down
        return
JOINT_RETREAT = ((1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75))  # how the retreat splits between elbow and wrist
JOINT_STEP_DEG = 2.0          # slider steps per sample along a joint retreat (the table is checked at each)
JOINT_RETREAT_MAX = 80.0      # give up after this much total joint travel without getting far enough back


def joint_ways_in(w, target, g, grip, hover_mm):
    """Ways in that retreat in JOINT space instead of along a straight line. Yields paths, hover first.

    Close to the base (inside ~280 mm) a straight-line way in forces the shoulder into its upper limit or the wrist
    into its lower one, so nothing planned there, even though the arm can clearly make the move. Driving it by hand
    at 227 mm showed how: from the grasp the elbow came back 108 -> 61, then the wrist went 7 -> 73, lifting the
    fingertip 290 mm without the shoulder moving at all, and the joint room went up, not down.

    So hold the shoulder, rotate and base where the grasp put them and back out on the elbow and wrist only. No IK,
    so nothing can pin a joint; the pose only moves away from the limits. The hand tilts as it goes."""
    tip0 = hand(w, g)[0]
    for ke, kw in JOINT_RETREAT:
        path, t = [list(g)], 0.0
        while t < JOINT_RETREAT_MAX:
            t += JOINT_STEP_DEG
            e, wr = g[1] - ke * t, g[2] + kw * t
            if not (JOINT_LO[1] <= e <= JOINT_HI[1] and JOINT_LO[2] <= wr <= JOINT_HI[2]):
                break
            q = [g[0], e, wr, g[3], grip, g[5]]
            path.append(q)
            if np.linalg.norm(hand(w, q)[0] - tip0) >= hover_mm:
                yield path[::-1]                             # hover first, like every other way in
                break


def ways_in(w, target, back, g, pitch, face, grip, n):
    """Straight-line ways in from the hover to the grasp (hover first). Yields None when one can't be solved.

    Which end you solve from decides which family of arm poses you get. Solving from the grasp outward keeps the
    elbow where the grasp put it, and at arm's length that walks the shoulder into its limit (every hover at 320 mm
    came out pinned). Solving from the hover inward picks a different family. Lazy: the grasp-first one is tried
    first, and if it already has ROOM_ENOUGH the costly alternative isn't computed."""
    hover_pt = target + back
    path, q = [g], g                                          # from the grasp, outward
    for k in range(n - 1, -1, -1):
        step = hover_mm_frac = 1.0 - k / n
        q = reach(w, target + back * hover_mm_frac, pitch, face, grip, near=q, ws=SQUARE_WEIGHT,
                  wp=1.0 if np.linalg.norm(back) * step <= STRAIGHT_MM else PATH_PITCH_WEIGHT)
        if q is None:
            break
        path.append(q)
    first = path[::-1] if q is not None else None
    yield first
    if first is not None and min(limit_room(p) for p in first) >= ROOM_ENOUGH:
        return
    for h0 in reach_all(w, hover_pt, face, grip)[:2]:         # from the hover, inward
        path, q = [h0], h0
        for k in range(n - 1, -1, -1):
            frac = k / n
            q = reach(w, target + back * frac, pitch, face, grip, near=q, ws=SQUARE_WEIGHT,
                      wp=1.0 if np.linalg.norm(back) * frac <= STRAIGHT_MM else PATH_PITCH_WEIGHT)
            if q is None:
                break
            path.append(q)
        yield path if q is not None else None


# Hover heights to try, longest first. A long hover is worth keeping (the close look happens there, and a stopped
# grab backs out to it), but close in only short hovers plan. The usable window can be narrow (at 322 mm only a
# 35 mm hover worked: 40 pinned the shoulder and 30 came too close to the block), hence 5 mm steps in the middle.
HOVER_LADDER = (60.0, 45.0, 40.0, 35.0, 25.0, 15.0)


def plan_best(w, check, cube, margin, **kw):
    """_plan_best; if it finds nothing, try again with the fingers allowed SQUARE_TOL_LOOSE_DEG off square (the
    squeeze squares the block up). The result is marked with plan["square_tol"] so re-plans use the same tolerance."""
    got = _plan_best(w, check, cube, margin, **kw)
    if "why" not in got or SQUARE_TOL_DEG >= SQUARE_TOL_LOOSE_DEG:
        return got
    with square_tol(SQUARE_TOL_LOOSE_DEG):
        loose = _plan_best(w, check, cube, margin, **kw)
    if "why" in loose:
        return got
    loose["square_tol"] = SQUARE_TOL_LOOSE_DEG
    loose["tried"] = got.get("tried", []) + [f"nothing square within {SQUARE_TOL_DEG:.0f} deg at any hover: "
                                             f"{abs(loose.get('off_square_deg', 0.0)):.0f} deg off, the squeeze squares it"]
    return loose


def _plan_best(w, check, cube, margin, ladder=None, room_min=0.5, accept=None, hover_floor=12.0,
               **kw):
    """plan() down the hover ladder, longest hover first, then with the aim pulled in toward the tower.

    accept(plan) -> bool lets the caller add its own test. The grab needs one: a shorter hover sits closer to the
    block, and without the test the ladder shortens the hover until it plans and the grab then refuses it.

    Returns the first plan that the gripper camera can see from, keeps room_min of joint room and the caller
    accepts. Fallbacks, in order: a blind straight plan, then a joint-space one. Otherwise {"why": [...]}."""
    ladder = HOVER_LADDER if ladder is None else ladder
    tried, blind_best, joint_best = [], None, None
    # A hover shorter than the grasp depth plus the clearance starts below the block's top, so skip those rungs.
    floor = kw.get("depth", DEPTH_MM) + hover_floor
    # Aim at the middle first and only pull the aim in if the whole ladder fails. This has to live here, not in
    # plan(): plan() can't tell whether the caller will accept what it found, and stopping the pull ladder at a
    # plan that then got rejected lost blocks that would have planned with a few mm of pull.
    for pull in GRASP_PULL_MM:
        if pull:
            tried.append(f"nothing at the block's middle: aiming {pull:.0f} mm nearer the tower")
        for hover in ladder:
            if hover < floor:
                tried.append(f"{hover:.0f} mm: below the block's top before it starts (the grasp is "
                             f"{kw.get('depth', DEPTH_MM):.0f} mm down, so a hover needs at least {floor:.0f} mm)")
                continue
            # Once there's a joint-space plan in hand, only look for straight ones (computing the joint-space
            # retreat on every rung of every pull made one plan take 2 minutes on the Pi).
            got = plan(w, check, cube, margin, hover_mm=hover, pull=pull, joint=joint_best is None, **kw)
            if "why" in got:
                # Keep every distinct reason, not just the first: plan() tries both faces, and the first reason
                # often hid the real failure.
                seen_r, uniq = set(), []
                for r_ in (got["why"] or ["no plan"]):
                    k_ = r_.split(":")[0]
                    if k_ in seen_r:
                        continue
                    seen_r.add(k_)
                    uniq.append(r_)
                tried.append(f"{hover:.0f} mm: " + "  ||  ".join(uniq[:4]))
                continue
            room = min(limit_room(q) for q in got["path"])
            if got["approach"] in JOINT_APPROACHES:
                # Joint-space plans are kept for last, after every hover's straight ways in. From their hovers the
                # gripper camera sees the top tag edge-on and read the block 0 times in 4 (all 4 grabs missed),
                # against 10 of 10 from "from above" hovers. Prefer one with the vertical tail.
                rank = lambda p: (p["approach"] == JOINT_TAIL_APPROACH, bool(p["sees"]))
                if room >= room_min and (accept is None or accept(got)) and (
                        joint_best is None or rank(got) > rank(joint_best)):
                    joint_best = got
                tried.append(f"{got['hover_mm']:.0f} mm (asked {hover:.0f}): only the joint-space way in -- kept "
                             f"in case no straight one works")
                continue
            if not got["sees"]:
                # Keep the best blind plan instead of discarding it: the grab can cope with not seeing the block from
                # the hover (it backs off to look, hunts, or trusts the block filter). Only keep one the grab would
                # actually take, since keeping one stops the pull ladder.
                if room >= room_min and (accept is None or accept(got)):
                    blind_best = got if blind_best is None or got["room"] > blind_best["room"] else blind_best
                tried.append(f"{got['hover_mm']:.0f} mm (asked {hover:.0f}): planned, but the gripper camera would not see the block from it")
                continue
            if room < room_min:
                names = ("shoulder", "elbow", "wrist")           # say which joint ran out of room, and where
                worst = min(((limit_room(q), k) for k, q in enumerate(got["path"])), key=lambda t: t[0])[1]
                q = got["path"][worst]
                j = min(range(3), key=lambda i: min(q[i] - JOINT_LO[i], JOINT_HI[i] - q[i]))
                where = "the grasp" if worst == len(got["path"]) - 1 else ("the hover" if worst == 0 else f"step {worst}")
                tried.append(f"{got['hover_mm']:.0f} mm (asked {hover:.0f}): {where} puts the {names[j]} "
                             f"{min(q[j] - JOINT_LO[j], JOINT_HI[j] - q[j]):.1f} steps from its end"
                             f"  ||  there: {room_text(q, check)}"
                             f"  ||  hover: {room_text(got['hover'], check)}"
                             f"  ||  grasp: {room_text(got['grasp'], check)}")
                continue
            if accept is not None and not accept(got):
                tried.append(f"{got['hover_mm']:.0f} mm (asked {hover:.0f}): too close to the block to get to"
                             f"  ||  hover: {room_text(got['hover'], check)}"
                             f"  ||  grasp: {room_text(got['grasp'], check)}")
                continue
            got["tried"] = tried + [f"{got['hover_mm']:.0f} mm: taken"]
            return got

        if blind_best is not None:
            break                                         # a blind straight plan beats pulling the aim in further
        # A joint-space plan doesn't stop the pull ladder: a straight way in aimed a few mm nearer the tower is
        # better (joint-space grabs held 0 times in 9, and at 270 mm every block planned straight once pulled in).
    if blind_best is not None and (accept is None or accept(blind_best)) \
            and min(limit_room(q) for q in blind_best["path"]) >= room_min:
        blind_best["tried"] = tried + [f"{blind_best['hover_mm']:.0f} mm: taken, though the gripper camera "
                                       f"cannot see the block from it"]
        blind_best["blind"] = True
        return blind_best
    if joint_best is not None:
        joint_best["tried"] = tried + [f"{joint_best['hover_mm']:.0f} mm: taken, the joint-space way in (no straight "
                                       f"one at any hover)"]
        joint_best["blind"] = not joint_best["sees"]
        return joint_best
    return {"why": ["no hover worked, roomiest first:"] + tried,       # never return a plan that was turned down
            "tried": tried}


def plan(w, check, cube, margin, grip=GRIP_OPEN, depth=DEPTH_MM, hover_mm=HOVER_MM, grasp_margin=GRASP_MARGIN_MM,
         see=True, pull=0.0, joint=True):
    """A grab plan for a block from cube.locate: dict(pitch, face_deg, hover, grasp, path, min_gap_mm, ...), or
    dict(why=...) if nothing works.

    check: a tablecheck.TableCheck. margin: table margin for the hover (mm). grasp_margin: for the way in (mm, at
    most margin). see: prefer a hover the gripper camera can see the block's top tag from; plan["sees"] says
    whether it got one. pull: aim this far from the middle toward the tower."""
    grasp_margin = min(grasp_margin, margin)
    target = np.asarray(cube["xyz"], float) - [0.0, 0.0, depth]
    faces = [math.radians(cube["yaw_deg"]), math.radians(cube["yaw_deg"] + 90.0)]
    view = top_tag(target, cube["yaw_deg"], depth) if see else None
    why, blind = [], None
    best = None
    radial = target[:2] - w.axis[:2]                       # pulling in = straight back toward the tower
    radial = radial / max(1e-9, np.linalg.norm(radial))
    aim = target.copy()
    aim[:2] -= radial * pull
    for face in faces:
        branches = reach_all(w, aim, face, grip)
        if not branches:
            why.append(f"fingers square to the {math.degrees(face):.0f} deg face: out of reach")
            continue
        for g in branches:
            got = way_in(w, check, aim, g, face, grip, hover_mm, margin, grasp_margin, view=view, joint=joint)
            if isinstance(got, dict):
                got["pull_mm"] = pull
                got["depth_mm"] = depth                       # so replan aims the same way
                # Take the roomiest way in, not the first visible one: the branches are sorted by room at the
                # grasp, which says nothing about the hover. Stop early once one is clearly good, though: checking
                # every branch costs more IK than it's worth on the Pi.
                if got["sees"]:
                    best = got if best is None or got["room"] > best["room"] else best
                    if best["room"] >= ROOM_PLENTY:
                        return best
                else:
                    blind = got if blind is None or got["room"] > blind["room"] else blind
            else:
                why.append(got)
    return best or blind or {"why": why or ["nothing tried"]}


def replan(w, check, cube, prev, margin, depth=None, hover_mm=None, grasp_margin=GRASP_MARGIN_MM):
    """_replan at the squareness the plan was made with. If that fails and the plan was made at the strict
    tolerance, try the loose one: a few mm of block movement can push a grasp that was just inside the tolerance
    just outside it (6 of 8 failed re-plans one day; all 6 worked at the loose tolerance)."""
    tol = prev.get("square_tol", SQUARE_TOL_DEG)
    with square_tol(tol):
        got = _replan(w, check, cube, prev, margin, depth, hover_mm, grasp_margin)
    if "why" in got and tol < SQUARE_TOL_LOOSE_DEG:
        with square_tol(SQUARE_TOL_LOOSE_DEG):
            loose = _replan(w, check, cube, prev, margin, depth, hover_mm, grasp_margin)
        if "why" not in loose:
            got, tol = loose, SQUARE_TOL_LOOSE_DEG
    if tol != SQUARE_TOL_DEG and "why" not in got:
        got["square_tol"] = tol
    return got


def _replan(w, check, cube, prev, margin, depth=None, hover_mm=None, grasp_margin=GRASP_MARGIN_MM):
    """The same grab for a block that has been located again (closer up): same face, pitch, way in, depth, pull and
    hover height, solved from the previous grasp so the joints only shift by what the new position needs. Using the
    defaults instead of the plan's own depth and hover made re-plans fail on blocks that had barely moved."""
    grasp_margin = min(grasp_margin, margin)
    depth = prev.get("depth_mm", DEPTH_MM) if depth is None else depth
    pull = float(prev.get("pull_mm", 0.0))
    hover_mm = prev.get("hover_mm", HOVER_MM) if hover_mm is None else hover_mm
    target = np.asarray(cube["xyz"], float) - [0.0, 0.0, depth]
    if pull:
        radial = target[:2] - w.axis[:2]
        target[:2] -= radial / max(1e-9, np.linalg.norm(radial)) * pull
    face, grip = math.radians(prev["face_deg"]), prev["grip"]
    g = reach(w, target, prev["pitch"], face, grip, near=prev["grasp"], ws=SQUARE_WEIGHT)
    if g is None:
        g = reach(w, target, prev["pitch"], face, grip, near=prev["grasp"], ws=SQUARE_WEIGHT, wp=PATH_PITCH_WEIGHT)
    if g is None:
        return {"why": ["the new position cannot be reached the same way"]}
    first = [a for a in APPROACHES if a[0] == prev["approach"]] + [a for a in APPROACHES if a[0] != prev["approach"]]
    first = first or list(APPROACHES)      # the joint-space ways in aren't in APPROACHES; way_in falls back to them
    view = top_tag(target, cube["yaw_deg"], depth) if "yaw_deg" in cube else None
    got = way_in(w, check, target, g, face, grip, hover_mm, margin, grasp_margin, approaches=first, view=view)
    if isinstance(got, dict):
        got["depth_mm"], got["pull_mm"] = depth, pull         # so the set-down aims the same way
        got["hover_mm"] = hover_mm
        return got
    return {"why": [got]}


# The precomputed plan table (tools/build_plan_table.py, tools/grow_plan_table.py). Planning from scratch took
# ~15 s of a round on the Pi, for a mat the laptop can plan all of ahead of time. The table holds plan_best's
# answers on a grid of block positions and turns; the Pi re-plans the nearest entries for the block it sees and
# keeps one only if it passes every check plan_best would make. So the table only changes how fast a plan comes,
# never which checks it passes.
PLAN_TABLE_FILE = "plan_table.json"
PLAN_TABLE_DEG_MM = 1.5          # for "nearest entry", a degree of block turn counts as this many mm
PLAN_TABLE_STAMPED = ("world.json", "table_model.json", "arm_map.npz")   # a table is only valid for these exact files


def plan_table_stamp(data_dir):
    """md5 of each file the planner's answers depend on."""
    import hashlib
    out = {}
    for f in PLAN_TABLE_STAMPED:
        with open(os.path.join(data_dir, f), "rb") as fh:
            out[f] = hashlib.md5(fh.read()).hexdigest()
    return out


def load_plan_table(data_dir):
    """The plan table in data_dir, or None (printing why) if there isn't one or it was built from other files."""
    import json
    path = os.path.join(data_dir, PLAN_TABLE_FILE)
    if not os.path.exists(path):
        return None
    t = json.load(open(path))
    stale = [f for f, m in plan_table_stamp(data_dir).items() if t.get("stamp", {}).get(f) != m]
    if stale:
        print(f"   (plan table not used: built for a different {', '.join(stale)} -- rebuild it with "
              f"tools/build_plan_table.py)")
        return None
    return t


def _face_like(seed_face, dyaw, yaw):
    """Of the two faces plan() would try for a block at yaw, the one closest to the seed's face turned by the change
    in yaw (faces repeat every 180 deg)."""
    want = seed_face + dyaw
    return min((yaw, yaw + 90.0), key=lambda f: abs((f - want + 90.0) % 180.0 - 90.0))


def plan_seeded(w, check, cube, margin, table, accept=None, room_min=0.5, tries=5, depth=DEPTH_MM,
                grasp_margin=GRASP_MARGIN_MM):
    """A plan from the nearest table entries, re-planned for this block, or None (then use plan_best). Only used if
    the table was built with the same margins and depth."""
    if not table or table.get("margin") != margin or table.get("depth_mm") != depth \
            or table.get("grasp_margin") != grasp_margin:
        return None
    xy, yaw = np.asarray(cube["xyz"], float)[:2], float(cube["yaw_deg"])

    def dyaw(e):                                          # a block's turn repeats every 90 deg
        return (yaw - e["yaw"] + 45.0) % 90.0 - 45.0

    near = sorted(table["plans"], key=lambda e: float(np.hypot(*(xy - e["xy"]))) + PLAN_TABLE_DEG_MM * abs(dyaw(e)))
    for e in near[:tries]:
        seed = dict(e["plan"], face_deg=_face_like(e["plan"]["face_deg"], dyaw(e), yaw))
        got = replan(w, check, cube, seed, margin, grasp_margin=grasp_margin)
        if "why" in got or not got.get("sees", True) or min(limit_room(q) for q in got["path"]) < room_min:
            continue
        if got.get("approach") in JOINT_APPROACHES:         # plan_best tries those last; let it decide
            continue
        if accept is not None and not accept(got):
            continue
        got["seeded"] = f"{float(np.hypot(*(xy - e['xy']))):.0f} mm, {dyaw(e):+.0f} deg from a table plan"
        return got
    return None
