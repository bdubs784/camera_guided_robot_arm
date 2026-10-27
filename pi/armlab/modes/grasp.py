"""
grasp: find the block with the cameras, plan a grab with the world model, and do it.

    python3 arm.py grasp --dry   --table-margin 20    find the block and plan the grab; nothing moves
    python3 arm.py grasp --hover --table-margin 20    go to the hover above it, look up close, then home
    python3 arm.py grasp --grab  --table-margin 20    the whole grab: hover, close look, re-plan, slow way in,
                                                      close, lift, check it came up, set it down, home
    (--grasp-margin 12: the table margin for the last part of the way in, where the fingers go down beside it)

If no camera sees the block from home, the arm goes up to SEARCH_POSE and the base sweeps. Every look goes into a
block filter (cube.BlockFilter). At the hover, if the gripper camera can't see the block, the arm moves to the
nearest pose the model says can see it; if it still can't, it goes on the filter's estimate when that is within
BLIND_MAX_MM.

Every pose is checked against the table (the exact table check), and every move except the way in is also checked
against the block. Ctrl+C stops the arm where it is.
"""
import os
import sys
import time

import numpy as np

from ..config import HOME_FINISH_DEG, PICK_SLOW, ROLL_RANGE, ROLL_SLIDER
from ..cube import CUBE_MM, BlockFilter, locate
from ..errors import ArmError
from ..grasp import (DEPTH_MM, GRIP_CLOSE, GRIP_OPEN, hand, plan_best, replan, set_limits)
from .. import grasp as _g
from ..hardware import arm_angles
from ..motion import base_ready, check_ready, move_base, move_to, pan_move, travel
from ..safety import Zone
from .. import live
from .. import trace
from ..state import S
from .touch import camera_dq
from .where import sightings_now

BASE_CAMS = ("torre", "webcam")
TRAVEL_GRIP = 80.0
HOVER_GRIP = 40.0                 # open wide before crossing to the hover so the fingers pass either side of the block
HIGH_POSE = [36.6, 75.0, 60.0]    # fingertip ~130 mm up; base turns happen here (at home the hand is lower than the block)
CUBE_R_MM = 40.0                  # the block as an upright cylinder for the clearance checks (covers its corners
CUBE_CLEAR_MM = 25.0              # at any turn), and how far every move except the way in has to stay from it
_PLAN_TABLE = {}
START_TOL_DEG = 2.0              # a grab starting within this of home or HIGH_POSE starts where it is
STAY_UP = {}                     # show --stay-up: {"base": the round's home base angle} while the arm waits up high


def plan_table():
    """The precomputed plan table (tools/build_plan_table.py), loaded once. None if missing or built for a different
    world model, table model or safety map."""
    if "t" not in _PLAN_TABLE:
        _PLAN_TABLE["t"] = _g.load_plan_table(S.data_dir)
        if _PLAN_TABLE["t"]:
            print(f"   (plan table: {len(_PLAN_TABLE['t']['plans'])} plans, built {_PLAN_TABLE['t'].get('built', '?')})")
    return _PLAN_TABLE["t"]


HOVER_CLEAR_MM = 12.0             # the hover sits right over the block, so it gets a smaller clearance than other moves
SLACK_TABLE_MM = 5.0              # the move checks use slider values and the planner uses real poses; allow this gap
SLOW_TOL_MM = 2.0                 # the arm reads ~1 step low after closing on the block; don't refuse the lift over it
STEP_DEG = 2.0
LIFTED_MM = 20.0                  # the block's tag must rise at least this much for a grab to count
MOVED_MM = 25.0                   # the close-up look may move the block at most this much from the first look
CARRY_SPEED = 30.0                # deg/s carrying the block up high (moves beside the block stay at PICK_SLOW)
RELEASE_GRIP = 100.0              # set-down: open only this far, lift off, then open wide (opening wide in one go
                                  # swung the fingers and dragged the block)
SQUEEZE_GLIDE_S = 0.25
SQUEEZE_TO = 120.0                # squeeze check: close from GRIP_CLOSE to this; on the block the fingers stop, so
SQUEEZE_HELD = 0.4                # they close less than this share of what the model expects for an empty hand
CARRY_TO_R = None                 # show: carry the block toward this distance from the base axis
CARRY_MAX_MM = 70.0               # at most this far per round (it walks there over several rounds)
RADIAL_STEPS_MM = (50.0, 35.0, 25.0, 18.0, 12.0, 8.0)   # shorter carries to fall back on (see radial_shifts)
CARRY_MM = 10.0                   # without show: set it down this far from where it was
CARRY_FOUND_MM = 4.0              # found back within this of the new spot = carried
PRACTICE_R = (230.0, 350.0)       # where blocks may be set down: this far from the base axis (mm)...
PRACTICE_BEARING = (-45.0, 50.0)  # ...and between these bearings (deg, 0 = straight out). Past ~+55 is out of the
                                  # base's range anyway
SEARCH_POSE = [51.0, 55.0, 96.0]   # where the arm waits while the base searches: the one pose where the arm hides
                                   # nothing from the tower camera (checked over 88 spots on the table)
SEARCH_CLEAR_MM = 60.0             # table clearance the search pose needs at every base angle
SEARCH_DPS = 12.0                  # base speed while searching (the firmware's smooth base speed)
SEARCH_EDGE_PX = 150.0             # a spot only counts as in a camera's view this far inside the picture; tags at the
                                   # very edge were often missed
SEARCH_PAUSE_CAMS = ("gripper",)   # paused while searching: the gripper camera only sees table from up there, and
                                   # its CPU is better spent on the webcam and tower camera
IN_VIEW_EDGE_PX = 60.0            # the least margin look_pose will accept...
LOOK_EDGE_PX = 200.0              # ...and what it aims for. 60 px wasn't enough: a few steps of shoulder slack pushed
                                  # the tag out of the picture
LOOK_SPAN_DEG = 24.0              # if the hover can't see the block: search this far around it (each joint)
LOOK_STEP_DEG = 4.0               # in these steps for a pose the model says can see the tag
BLIND_MAX_MM = 4.0                # still not seen: trust the block filter if it is this sure (1 sigma)
SIDE_TAGS = (250, 251, 252, 253)  # tags on the block's four sides: at the lift the hand hides the top tag


class Stuck(ArmError):
    """The arm could not back out of the way in, so it is left where it is (driving it out by hand is safer)."""


def polar(w, xyz):
    """[mm out from the base axis, bearing deg] of a point, for the demo page's map, or None."""
    try:
        import math
        v = np.asarray(xyz, float)[:2] - w.axis[:2]
        return [round(float(np.linalg.norm(v)), 1), round(math.degrees(math.atan2(v[1], v[0])), 2)]
    except Exception:
        return None


def live_block(w, cube, blk=None, off=None, fit=None):
    """Send the block's position and uncertainty to the demo page (and, with off/fit, how well the cameras agree)."""
    try:
        import math
        xyz = np.asarray(cube["xyz"], float)
        v = xyz[:2] - w.axis[:2]
        b = {"out_mm": float(np.linalg.norm(v)), "bearing_deg": math.degrees(math.atan2(v[1], v[0])),
             "yaw_deg": float(cube.get("yaw_deg", 0.0)), "cams": dict(cube.get("cams") or {})}
        if blk is not None:
            b.update(sigma_mm=float(blk.sigma_mm()), looks=int(blk.looks))
        live.update(block=b)
        if fit is not None:
            live.update(agree={"fit_px": float(fit), "base_off_deg": None if off is None else float(off)})
    except Exception:
        pass


def log_look(a, seen):
    """Keep one look's raw data for the run file (see LOOKS_RAW)."""
    try:
        LOOKS_RAW.append({"t": round(time.time(), 2), "joints": [round(float(v), 2) for v in a],
                          "seen": [[c, int(t), np.round(np.asarray(px, float), 1).tolist()] for c, t, px in seen]})
    except Exception:
        pass


def pause(secs, why="wait"):
    """time.sleep that shows up in the trace (and so in tools/timeline.py) by name."""
    with trace.span("wait", why):
        time.sleep(secs)


def find_cube(w, cams=None, dq=None):
    """_find_cube, traced."""
    with trace.span("camera", "look"):
        return _find_cube(w, cams, dq)


LAST_LOOK = {}                    # the latest look's base count and the base offset its table tags measured


def _find_cube(w, cams=None, dq=None):
    """(arm_angles, base offset, table fit px, block or None, sightings) from the cameras right now (cams: only
    these). dq: the arm's measured slack, needed for the gripper camera's position."""
    a = arm_angles()
    seen_all = sightings_now()
    log_look(a, seen_all)
    off, fit = w.fit_base_offset(seen_all, a[:4], a[5])
    LAST_LOOK.update(base=float(a[5]), off=float(off), fit=float(fit), t=time.time())
    seen = seen_all if cams is None else [s for s in seen_all if s[0] in cams]
    return a, off, fit, locate(w, seen, a[:4], a[5], base_offset=off, grip=a[4], dq=dq), seen_all


def _sweep_to(end):
    """Turn the base slowly to `end` while the cameras watch for the block's tag. Returns the base angle where it
    was first seen (the base stops there), or None."""
    from ..hardware import arm_stop, base_angle, http_json
    if not S.base_speed_cmd:                                 # old firmware has no smooth base moves: look in steps
        b = base_angle()
        for t in list(np.arange(b, end, 4.0 if end > b else -4.0)[1:]) + [end]:
            move_base(float(t), closed=False)
            if any(tag == 7 for _, tag, _ in sightings_now()):
                return float(t)
        return None
    S.moving = True
    http_json(S.esp_url + f"/setall?a=,,,,,{end:.3f}&bs={SEARCH_DPS:g}")
    S.base_cmd = end
    t_end = time.time() + abs(end - base_angle()) / SEARCH_DPS + 10.0
    try:
        while time.time() < t_end:
            b = base_angle()
            if any(tag == 7 for _, tag, _ in sightings_now()):
                arm_stop()
                return b
            if abs(b - end) < 0.3:
                return None
            time.sleep(0.05)
        raise ArmError(f"The base did not get to {end:+.0f} during the search (it is at {base_angle():+.1f}).")
    finally:
        arm_stop()
        S.base_cmd = base_angle()


_SEARCH_BASES = {}


def _base_cam_px(w, X, b):
    """{camera: pixel coords of the table points X} for each base camera, with the arm at SEARCH_POSE and the base
    at b. NaN where a point is behind the camera."""
    q = list(SEARCH_POSE) + [ROLL_SLIDER, TRAVEL_GRIP, float(b)]
    X = np.atleast_2d(np.asarray(X, float))
    out = {}
    for cam in BASE_CAMS:
        if cam not in w.cams:
            continue
        R, t = w.camera_pose(cam, q[:4], q[5])
        px = np.array(w.project(cam, X, q[:4], q[5]), float).reshape(-1, 2)
        px[(X @ R.T + t)[:, 2] <= 0] = np.nan
        out[cam] = px
    return out


def _px_room(cam, px, edge=SEARCH_EDGE_PX):
    """How far inside the picture each point is, minus edge (>= 0 means in view, NaN counts as out)."""
    wide, high = _g.CAM_SIZE_PX[cam]
    room = np.minimum(np.minimum(px[:, 0], wide - px[:, 0]), np.minimum(px[:, 1], high - px[:, 1])) - edge
    return np.where(np.isfinite(room), room, -1e9)


def search_stops(w):
    """The base angles the search stops at for a still look: a greedy cover, so that every spot in the practice
    area is well inside some base camera's picture from at least one stop. [] if no base camera sees any of it.

    A moving sweep missed blocks (the webcam detects ~2.6 frames/s and frames smear while turning); three still
    looks cover the whole area on the current world model."""
    from ..grasp import BASE_RANGE
    key = (id(w), tuple(PRACTICE_R), tuple(PRACTICE_BEARING))
    if key not in _SEARCH_BASES:
        import math
        ax = w.axis[:2]
        pts = []
        for r in np.arange(PRACTICE_R[0] - 20.0, PRACTICE_R[1] + 20.0 + 1e-6, 15.0):
            for bb in np.arange(PRACTICE_BEARING[0] - 5.0, PRACTICE_BEARING[1] + 5.0 + 1e-6, 4.0):
                x, y = ax + r * np.array([math.cos(math.radians(bb)), math.sin(math.radians(bb))])
                pts.append([x, y, w.table_height(x, y) + CUBE_MM])
        X = np.array(pts)
        bases = np.arange(BASE_RANGE[0], BASE_RANGE[1] + 1e-6, 1.0)
        V = np.array([np.any([_px_room(c, px) >= 0 for c, px in _base_cam_px(w, X, b).items()] or
                             [np.zeros(len(X), bool)], axis=0) for b in bases])      # [base, spot]
        left, stops = V.any(axis=0), []
        while left.any():
            k = int(np.argmax((V & left).sum(axis=1)))
            stops.append(float(bases[k]))
            left = left & ~V[k]
        _SEARCH_BASES[key] = sorted(stops)
    return list(_SEARCH_BASES[key])


def aim_base_at(w, xyz):
    """The base angle that puts the point xyz deepest inside a base camera's picture from SEARCH_POSE, or None."""
    from ..grasp import BASE_RANGE
    best, best_room = None, 0.0
    for b in np.arange(BASE_RANGE[0], BASE_RANGE[1] + 1e-6, 1.0):
        for c, px in _base_cam_px(w, xyz, b).items():
            room = float(_px_room(c, px)[0])
            if room >= best_room:
                best, best_room = float(b), room
    return best


def _pause_cams(cams, secs=60.0):
    """Ask the camera server to stop detecting tags on these cameras for up to secs (() resumes). Failures are only
    printed; older camera servers ignore the request."""
    from ..hardware import http_json
    try:
        http_json(S.cam_url + f"/rate?pause={','.join(cams)}&secs={secs:g}", tries=1)
    except Exception as e:
        print(f"   (could not change camera detection: {e})")


def search(w, zone, check, speed):
    """No camera sees the block: go up to SEARCH_POSE, look once, look where it was last seen, then stop and look
    from each search stop, then sweep the whole base range. Returns the block, or None (left at the search pose)."""
    from ..grasp import BASE_RANGE
    from .survey import set_hand
    level = (ROLL_SLIDER, TRAVEL_GRIP)
    start = arm_angles()
    gap = check.gaps(SEARCH_POSE + list(level), None)[0]
    if not zone.contains(SEARCH_POSE) or not zone.path_ok(start[:3], SEARCH_POSE) or gap < SEARCH_CLEAR_MM:
        raise ArmError(f"No camera sees the block, and the search pose {SEARCH_POSE} is not clear "
                       f"({gap:.0f} mm from the table/plywood, keeps {SEARCH_CLEAR_MM:.0f}). Put the block where the "
                       "tower camera or the webcam can see it.")
    print("No camera sees the block from here: searching (up to the search pose, then the base sweeps slowly)...")
    set_hand(*level)
    move_to(SEARCH_POSE, zone, speed)
    # Often the arm itself was blocking the view; raising it can put the block in plain sight, so look before
    # turning the base away from it.
    pause(0.4, "search: still look")
    got = find_cube(w)[3]
    if got is not None:
        print(f"   seen from the search pose, without sweeping (base {arm_angles()[5]:+.1f})")
        return got
    lo, hi = BASE_RANGE
    if LAST_BLOCK.get("xyz") is not None:
        t = aim_base_at(w, LAST_BLOCK["xyz"])
        if t is not None and abs(t - float(arm_angles()[5])) > 1.0:
            print(f"   looking where it was last seen first (base {t:+.1f})")
            move_base(t)
            pause(0.5, "search: still look")
            got = find_cube(w)[3]
            if got is not None:
                print(f"   found it at base {t:+.1f}")
                return got
    b = float(arm_angles()[5])
    stops = search_stops(w)
    if stops and abs(b - stops[-1]) < abs(b - stops[0]):    # start from the nearer end
        stops = stops[::-1]
    ends = [(t, True) for t in stops]                       # then sweep the whole range as a fallback
    far = hi if not stops or stops[-1] >= stops[0] else lo
    ends += [(far, False), (lo if far == hi else hi, False)]
    _pause_cams(SEARCH_PAUSE_CAMS)
    try:
        for end, still in ends:
            print(f"   {'turning the base to look from' if still else 'sweeping the base to'} {end:+.0f}...")
            with trace.span("base", f"{'search stop' if still else 'sweep to'} {end:+.0f}"):
                hit = _sweep_to(end)                        # it watches on the way there too
            if hit is None and still:
                # don't wait for the gripper camera here: it's paused, so the wait would time out every stop
                wait_fresh(time.time() + 0.3, cams=BASE_CAMS)
                got = find_cube(w)[3]
                if got is not None:
                    print(f"   found it at base {end:+.1f}")
                    return got
                continue
            if hit is None:
                continue
            for d in (0.0, 3.0, -3.0, 6.0, -6.0):          # the cameras lag the base a little: look either side
                t = min(max(hit + d, lo), hi)
                move_base(t)
                pause(0.5, "search: still look")
                got = find_cube(w)[3]
                if got is not None:
                    print(f"   found it at base {t:+.1f}")
                    return got
            print(f"   glimpsed it near base {hit:+.1f} but could not get a still look; carrying on")
        return None
    finally:
        _pause_cams(())


def cube_text(cube, w=None):
    x, y, _ = cube["xyz"]
    reach = ""
    if w is not None:
        r = float(np.hypot(x - w.axis[0], y - w.axis[1]))
        bearing = float(np.degrees(np.arctan2(y - w.axis[1], x - w.axis[0])))
        reach = f", {r:5.1f} mm out from the base axis (bearing {bearing:+5.1f} deg)"
    if "filter_mm" in cube:
        return (f"x {x:6.1f}  y {y:6.1f} mm{reach}, turned {cube['yaw_deg']:+5.1f} deg  "
                f"(block filter: +-{cube['filter_mm']:.1f} mm from {cube['looks']} looks)")
    out = (f"x {x:6.1f}  y {y:6.1f} mm{reach}, turned {cube['yaw_deg']:+5.1f} deg, "
           f"top {cube['above_table_mm']:5.1f} mm up  ("
           + ", ".join(f"{c} {e:.1f} px" for c, e in cube["cams"].items()) + ")")
    tops = cube.get("tops") or {}                 # each camera's own estimate of the top's height (the gripper
    if tops:                                      # camera tends to read it a few mm low)
        out += ("\n      top above the table, camera by camera "
                f"(the block is {CUBE_MM:.1f} mm): "
                + ",  ".join(f"{c} {h:.1f} ({h - CUBE_MM:+.1f})" for c, h in sorted(tops.items())))
    return out


def joints_text(q):
    return (f"shoulder {q[0]:5.1f}  elbow {q[1]:5.1f}  wrist {q[2]:5.1f}  rotate {q[3]:5.1f}  gripper {q[4]:5.1f}  "
            f"base {q[5]:+6.1f}")


def plan_text(w, p, margin, grasp_margin):
    tip, _ = hand(w, p["grasp"])
    return "\n".join([
        f"Plan: hand pitched {p['pitch']:.0f} deg down, fingers {p['off_square_deg']:+.0f} deg off square to the "
        f"{p['face_deg']:.0f} deg face, coming in {p['approach']}",
        "   hover:  " + joints_text(p["hover"]),
        "   grasp:  " + joints_text(p["grasp"]),
        f"   then close to {GRIP_CLOSE:.0f}, lift 40 mm.   Table/plywood: {p['hover_gap_mm']:.0f} mm at the hover (margin "
        f"{margin:.0f}), {p['min_gap_mm']:.0f} mm at the closest on the way in (grasp margin {min(grasp_margin, margin):.0f}); "
        f"fingertip {np.linalg.norm(tip - p['target']):.1f} mm from the grasp point, {len(p['path'])} checked steps; the "
        f"gripper camera should {'see' if p.get('sees', True) else 'NOT see'} the block from the hover"])


def cube_gap(check, cube, q5, base):
    """Clearance (mm) from the arm's checked points to the block, treated as an upright cylinder (negative = inside)."""
    X = None
    if getattr(check, "w", None) is not None:          # the compiled version of check.points, when numba is there
        try:
            from .. import fastik
            X = fastik.points(check, q5, float(base))
        except Exception:
            X = None
    if X is None:
        X = check.points(q5, base)
    c = np.asarray(cube["xyz"], float)
    dr = np.linalg.norm(X[:, :2] - c[:2], axis=1) - CUBE_R_MM
    dz = X[:, 2] - c[2]
    out = np.where((dr > 0) | (dz > 0), np.hypot(np.maximum(dr, 0), np.maximum(dz, 0)), np.maximum(dr, dz))
    return float(out.min())


def arm_path(a, b, step=STEP_DEG):
    a, b = np.asarray(a[:3], float), np.asarray(b[:3], float)
    n = int(np.abs(b - a).max() / step) + 2
    return [a + (b - a) * t for t in np.linspace(0.0, 1.0, n)]


def check_arm_move(check, cube, a, b, hand_, base, leaving=False, arriving=False, what="the move", between=False):
    """Refuse the straight arm move a -> b unless it keeps CUBE_CLEAR_MM from the block. Three exceptions:
    leaving (starts close, e.g. home): never closer than the start, and end clear;
    arriving at the hover (which is over the block): never closer than the end, which needs HOVER_CLEAR_MM;
    between two poses over the block: never closer than the nearer end."""
    gaps = [cube_gap(check, cube, list(q) + list(hand_), base) for q in arm_path(a, b)]
    if between:
        ok = min(gaps) >= min(gaps[0], gaps[-1]) - 2.0
    elif leaving:
        ok = min(gaps) >= min(CUBE_CLEAR_MM, gaps[0] - 2.0) and gaps[-1] >= CUBE_CLEAR_MM
    elif arriving:
        ok = min(gaps) >= min(CUBE_CLEAR_MM, gaps[-1] - 2.0) and gaps[-1] >= HOVER_CLEAR_MM
    else:
        ok = min(gaps) >= CUBE_CLEAR_MM
    if not ok:
        raise ArmError(f"refused: {what} would come {min(gaps):.0f} mm from the block (it starts {gaps[0]:.0f} and ends "
                       f"{gaps[-1]:.0f} mm away). Move the block a bit and try again.")


def check_base_turn(check, cube, q5, b0, b1):
    for b in np.linspace(b0, b1, int(abs(b1 - b0) / STEP_DEG) + 2):
        g = cube_gap(check, cube, q5, b)
        if g < CUBE_CLEAR_MM:
            raise ArmError(f"refused: turning the base to {b1:+.1f} would sweep the hand {g:.0f} mm from the block")
PAN_ARM_STEP_DEG = STEP_DEG       # grid spacing for check_pan_move, the same spacing the single-move checks use
PAN_BASE_STEP_DEG = STEP_DEG      # (an 80 deg turn with a 44 deg arm move is ~950 points, ~0.25 ms each)


def check_pan_move(zone, check, cube, a, b, hand_, b0, b1, what="the move", leaving=False, arriving=False,
                   between=False, runup=HOME_FINISH_DEG):
    """Check the arm moving a -> b while the base turns b0 -> b1 at the same time.

    The two aren't synchronised (the arm and the base run on separate clocks), so the arm can be anywhere on its line
    at any point of the turn. So check every (arm step, base step) pair: a grid. Each point gets the usual checks
    (safety map + exact table at that base angle, and the block with check_arm_move's leaving/arriving/between rules).
    The grid's edges are "arm then turn" and "turn then arm", so anything that passes here also passes the two
    single-move checks either way round. Returns the smallest table/tower/arm gap on the grid.

    hand_: one (rotate, grip), or a list of them if the hand moves during the pan (each is checked on the whole grid).
    runup: the base's slack-taking finish first goes to b1 - runup, so a downward turn overshoots b1 by that much."""
    hands = [tuple(hand_)] if np.ndim(hand_) == 1 else [tuple(h) for h in hand_]
    arm = arm_path(a, b, PAN_ARM_STEP_DEG)
    run = b1 - runup
    bases = list(np.linspace(b0, run, int(abs(run - b0) / PAN_BASE_STEP_DEG) + 2)) + ([b1] if runup else [])
    rows = np.array([list(q) + list(h) + [bb] for h in hands for q in arm for bb in bases], float)
    shape = (len(hands), len(arm), len(bases))
    with trace.span("compute", f"pan check {len(hands)}x{len(arm)}x{len(bases)}"):
        room = zone.gaps(rows).reshape(shape)
        k = int(np.argmin(room))
        if room.flat[k] < zone.margin:
            raise ArmError(f"refused: {what} with the base turning {b0:+.1f} -> {b1:+.1f} at the same time passes "
                           f"within {room.flat[k]:.0f} mm of something (the margin is {zone.margin:.0f} mm), at "
                           f"{joints_text(rows[k])}")
        if cube is None or cube.get("xyz") is None:
            return float(room.min())
        gaps = np.array([cube_gap(check, cube, r[:5], r[5]) for r in rows]).reshape(shape)
    first, last, worst = gaps[0, 0, 0], gaps[-1, -1, -1], float(gaps.min())
    if between:
        ok = worst >= min(first, last) - 2.0
    elif leaving:
        ok = worst >= min(CUBE_CLEAR_MM, first - 2.0) and last >= CUBE_CLEAR_MM
    elif arriving:
        ok = worst >= min(CUBE_CLEAR_MM, last - 2.0) and last >= HOVER_CLEAR_MM
    else:
        ok = worst >= CUBE_CLEAR_MM
    if not ok:
        raise ArmError(f"refused: {what} with the base turning {b0:+.1f} -> {b1:+.1f} at the same time would come "
                       f"{worst:.0f} mm from the block (it starts {first:.0f} and ends {last:.0f} mm away), with the "
                       f"arm at {joints_text(rows[int(np.argmin(gaps))])}")
    return float(room.min())


PAN_OVERLAP = False               # --pan-overlap: move the arm while the base turns, where check_pan_move allows it


def pan_check_later(*a, **k):
    """Run check_pan_move in the background while the previous move is still going; pan_ok() collects the answer."""
    return CarryPlan(lambda: check_pan_move(*a, **k))


def pan_ok(job):
    """True = move the base and arm together. False = the check refused (or failed); the caller then does the two
    moves one after the other, with their own checks."""
    if job is None:
        return False
    with trace.span("wait", "pan check (finishing)"):
        try:
            job.get()
            return True
        except Exception as e:
            print(f"   (base and arm one after the other: {e})")
            return False


def go_to_hover(zone, check, cube, p, speed):
    """Up, turn the base, across, then the hand: all checked against the table and the block before anything moves.
    Returns the base angle it started from."""
    from .survey import exact_ok, set_hand
    h = p["hover"]
    start = arm_angles()
    b0 = float(S.base_cmd if S.base_cmd is not None else start[5])
    level = (ROLL_SLIDER, TRAVEL_GRIP)
    wide = (ROLL_SLIDER, HOVER_GRIP)
    if not zone.contains(HIGH_POSE) or not zone.path_ok(start[:3], HIGH_POSE) or not zone.path_ok(HIGH_POSE, h[:3]):
        raise ArmError("refused: the moves up and across to the hover are not clear of the table/tower/arm")
    check_arm_move(check, cube, start[:3], HIGH_POSE, level, b0, leaving=True, what="the move up from home")
    check_base_turn(check, cube, HIGH_POSE + list(level), b0, h[5])
    check_arm_move(check, cube, HIGH_POSE, h[:3], wide, h[5], arriving=True, what="the move across to the hover")
    hand_now = [list(h[:3]) + [r, g] for r, g in zip(np.linspace(wide[0], h[3], 6), np.linspace(wide[1], h[4], 6))]
    if min(cube_gap(check, cube, q, h[5]) for q in hand_now) < HOVER_CLEAR_MM or \
            not exact_ok(zone, h[:3], h[5], [wide, (h[3], h[4])]):
        raise ArmError("refused: turning the hand and opening the gripper at the hover comes too close to the block or table")
    turn = abs(h[5] - TURNED.get("base", 1e9)) > TURN_SAME_DEG
    pan = None
    # With the closed-loop base its corrections would happen at the hover, over the block, so no overlap then.
    if PAN_OVERLAP and turn and not HOVER_BASE_CLOSED:
        opening = [tuple(v) for v in np.linspace(start[3:5], wide, 3)]   # the fingers open during the move
        pan = pan_check_later(zone, check, cube, HIGH_POSE, h[:3], opening, b0, h[5], arriving=True,
                              what="the move across to the hover")
    print("\nGoing to the hover (Ctrl+C stops the arm): up, base, open the hand, across, then the hand.")
    move_to(HIGH_POSE, zone, speed)
    set_hand(ROLL_SLIDER, HOVER_GRIP, wait=False)            # open wide up here, well clear of the block
    if pan_ok(pan):
        TURNED.clear()
        pan_move(h[:3], h[5], zone, speed, closed=False, settle=False, slow_finish=True)
        set_hand(ROLL_SLIDER, HOVER_GRIP)
    else:
        if turn:
            move_base(h[5], closed=HOVER_BASE_CLOSED, settle=False)   # usually already there (turn_toward)
        TURNED.clear()
        set_hand(ROLL_SLIDER, HOVER_GRIP)                    # make sure it got there (instant if it did)
        move_to(h[:3], zone, speed, slow_finish=True)
    set_hand(h[3], h[4])
    pause(HOVER_SETTLE_S, "settle at the hover")
    wait_fresh(time.time(), cams=HOVER_FRESH_CAMS if getattr(S, "hand_rel", False) else None)   # frames after it
    return b0


HOVER_SETTLE_S = 0.5              # let the arm stop swaying before the close look (blur shows up as missed looks)
HOVER_FRESH_CAMS = ("gripper", "torre")   # --hand-rel only reads these two at the hover. The webcam is slow (frames
                                          # ~0.4 s late) and it's on the base, which isn't moving, so skip waiting for it


LAST_BLOCK = {}                   # where the block was last seen (grab_from_hover updates it after the carry)
SLACK_GP = {}                     # this round's slack-model prediction vs the cameras' measurement (for the run file)
SLACK_GP_MIN_TAGS = 4             # only scored when the cameras saw at least this many arm tags


def find_block_anywhere(args):
    """Where the block is now: a look from home, else the full search, then back home. (block or None, world).
    Used by show --final-look."""
    from ..tablecheck import load
    S.exact_table = True
    check = load()
    w = check.w
    zone = Zone(args.margin)
    set_limits(zone)
    LOOKS_RAW.clear()
    a, off, fit, cube, _ = find_cube(w)
    if cube is not None:
        return cube, w
    check_ready(zone)
    base_ready(zone)
    b_home = float(S.base_cmd if S.base_cmd is not None else a[5])
    try:
        cube = search(w, zone, check, args.speed)
    finally:
        go_home(zone, b_home, args.speed)
    return cube, w


# Where the arm waits between rounds so a fixed camera can see the block (park_spot). The numbers come from every
# look on record: the webcam sees best from the search pose and mostly finds close blocks; the tower camera sees
# best from up high and mostly finds far ones; the arm standing in front of a camera is what hides the block.
PARK_FAR_MM = 340.0               # from this far out try the tower camera first, nearer try the webcam first
PARK_POSE_OF = {"webcam": "search", "torre": "high"}
PARK_CLEAR_PX = 100.0             # the block counts as hidden if any part of the arm (nearer the camera) projects
                                  # within this of it in the picture
PARK_PREFER_PX = 200.0            # prefer views at least this clear: 100-200 px views were missed 3 times in 4
PARK_STEP_DEG = 3.0               # base angles tried for the waiting spot


def view_clear_px(w, check, cam, q, xyz):
    """In cam's picture with the arm at q: how far (px) the block's top tag is from the nearest part of the arm that
    is in front of it. inf if nothing is in front, None if the tag isn't in the picture at all."""
    from ..grasp import in_view, top_tag
    if cam not in w.cams:
        return None
    X = top_tag(np.asarray(xyz, float), 0.0, 0.0)
    if not in_view(w, X, q, cam, edge=40):
        return None
    R, t = w.camera_pose(cam, q[:4], q[5])
    A = check.points(list(q[:5]), q[5])
    nearer = np.linalg.norm(A @ R.T + t, axis=1) < np.linalg.norm(R @ np.asarray(xyz, float) + t)
    if not nearer.any():
        return float("inf")
    pt = w.project(cam, X, q[:4], q[5]).mean(0)
    return float(np.linalg.norm(w.project(cam, A[nearer], q[:4], q[5]) - pt, axis=1).min())


def park_spot(w, check, xyz, first=False):
    """Where to wait so a fixed camera sees a block at xyz clear of the arm: (pose, base, camera), or None.

    Of the base angles with a clear view, take the one nearest where the next round will turn to (saves base
    travel); the pose must also keep CUBE_CLEAR_MM from the block. first=True: return any angle that works (show's
    set-down planner only needs to know there is one)."""
    from ..grasp import BASE_RANGE, base_toward
    if xyz is None:
        return None
    cube = {"xyz": np.asarray(xyz, float), "yaw_deg": 0.0}
    far = float(np.hypot(*(cube["xyz"][:2] - w.axis[:2]))) >= PARK_FAR_MM
    try:
        toward = float(base_toward(w, cube["xyz"]))
    except Exception:
        toward = None
    cams = ("torre", "webcam") if far else ("webcam", "torre")
    views = {}                                         # cam -> [(clearance px, base)] that clear PARK_CLEAR_PX
    for cam in cams:
        pose = SEARCH_POSE if PARK_POSE_OF[cam] == "search" else HIGH_POSE
        views[cam] = []
        for b in np.arange(BASE_RANGE[0], BASE_RANGE[1] + 0.1, PARK_STEP_DEG):
            q = list(pose) + [ROLL_SLIDER, TRAVEL_GRIP, float(b)]
            c = view_clear_px(w, check, cam, q, xyz)
            if c is None or c < PARK_CLEAR_PX or cube_gap(check, cube, q[:5], float(b)) < CUBE_CLEAR_MM:
                continue
            if first:
                return list(pose), float(b), cam
            views[cam].append((float(c), float(b)))
    # a view at least PARK_PREFER_PX clear, nearest the next turn; failing that, the clearest view there is
    for cam in cams:
        good = [v for v in views[cam] if v[0] >= PARK_PREFER_PX]
        if good:
            c, b = min(good, key=lambda v: (abs(v[1] - toward) if toward is not None else -v[0]))
            return list(SEARCH_POSE if PARK_POSE_OF[cam] == "search" else HIGH_POSE), b, cam
    for cam in cams:
        if views[cam]:
            c, b = max(views[cam])
            return list(SEARCH_POSE if PARK_POSE_OF[cam] == "search" else HIGH_POSE), b, cam
    return None


def go_home(zone, b0, speed, check=None, stay_up=False, park=None):
    """From above the block: level the hand, go up, turn the base back, go home.

    With check and a known block, refuse the way down to home if it would come closer than HOVER_CLEAR_MM to the
    block (a set-down can leave it under the home fingertip) and stop up high instead.
    stay_up (show --stay-up): stop at HIGH_POSE instead of home and wait there.
    park: wait at (pose, base, camera) from park_spot instead, where a fixed camera sees the block past the arm."""
    from .survey import set_hand
    moves_on = bool(stay_up and park is not None and park[0] != list(HIGH_POSE))   # an arm move follows, not a look
    pan = None
    if PAN_OVERLAP and moves_on and check is not None and LAST_BLOCK.get("xyz") is not None:
        b_now = float(S.base_cmd if S.base_cmd is not None else arm_angles()[5])
        pan = pan_check_later(zone, check, LAST_BLOCK, HIGH_POSE, park[0], (ROLL_SLIDER, TRAVEL_GRIP), b_now, park[1],
                              between=True, what="the move to wait where the " + park[2] + " sees the block")
    set_hand(ROLL_SLIDER, TRAVEL_GRIP, wait=False)           # level the hand while rising away from the block
    move_to(HIGH_POSE, zone, speed)
    set_hand(ROLL_SLIDER, TRAVEL_GRIP)
    if pan_ok(pan):
        STAY_UP["base"] = b0
        try:
            pan_move(park[0], park[1], zone, speed, closed=False, settle=False)
        except ArmError as e:
            print(f"   (stopped on the way to wait where the {park[2]} sees the block: {e})")
        return
    move_base(park[1] if stay_up and park is not None else b0, closed=False, settle=not moves_on)
    if stay_up:
        STAY_UP["base"] = b0
        if park is not None and park[0] != list(HIGH_POSE):
            try:
                if check is not None and LAST_BLOCK.get("xyz") is not None:
                    check_arm_move(check, LAST_BLOCK, HIGH_POSE, park[0], (ROLL_SLIDER, TRAVEL_GRIP), park[1],
                                   between=True, what="the move to wait where the " + park[2] + " sees the block")
                move_to(park[0], zone, speed)
            except ArmError as e:
                print(f"   (waiting up high instead: {e})")
        return
    STAY_UP.clear()
    if check is not None and LAST_BLOCK.get("xyz") is not None:
        level = (ROLL_SLIDER, TRAVEL_GRIP)
        gaps = [cube_gap(check, LAST_BLOCK, list(q) + list(level), b0) for q in arm_path(HIGH_POSE, zone.home)]
        if min(gaps) < min(HOVER_CLEAR_MM, LAST_BLOCK.get("home_gap", HOVER_CLEAR_MM) - 2.0):
            raise ArmError(f"Stopped up high: going down to home would come {min(gaps):.0f} mm from the block (it was "
                           "moved). Move the block away from under the hand, then:  python3 arm.py home")
    travel(zone.home, zone, speed)


def arm_slack(w, a, off, seen):
    """The arm's real joint angles minus the slider values, fitted to the arm tags the cameras see. (dq, px, tags)."""
    row = {"joints": list(a[:5]), "base": a[5], "base_offset": off,
           "tags": [{"cam": c, "id": t, "corners": np.asarray(px).tolist()} for c, t, px in seen]}
    return camera_dq(w, row)


SLACK_CAM = "torre"               # --hand-rel: measure the slack from this camera's arm tags only
SLACK_MIN_TAGS = 2                # and only if it sees at least this many; otherwise no correction


def slack_now(w, a, off, seen):
    """arm_slack, or with --hand-rel the tower camera's arm tags only (no correction if it sees too few).

    Fitted from each camera alone at the same hovers, the webcam disagreed with the tower camera by up to 7 deg on
    a joint (only 2-5 arm tags each). At 300 mm out, 2-3 deg is 10-15 mm at the fingertip, about the whole margin
    the open fingers have, so a bad fit is worse than none."""
    if not getattr(S, "hand_rel", False):
        return arm_slack(w, a, off, seen)
    dq, px, n = arm_slack(w, a, off, [s for s in seen if s[0] == SLACK_CAM])
    if n < SLACK_MIN_TAGS:
        return np.zeros_like(np.asarray(dq, float)), px, n
    return dq, px, n


def hand_rel(w, a, off, seen):
    """--hand-rel: the block from the gripper camera alone, located with NO slack correction, or None.

    The camera is bolted to the hand, so joint slack moves the camera and the fingers together. Locate the block
    as the sliders believe the arm is, grab with the same sliders, and the error cancels out."""
    if not getattr(S, "hand_rel", False) or not gripper_sees(seen):
        return None
    c = locate(w, [s for s in seen if s[0] == "gripper"], a[:4], a[5], base_offset=off, grip=a[4], dq=None)
    if c is not None:
        c["close"] = True
    return c


def gripper_sees(seen):
    """Did the gripper camera itself see the block's top tag? A close look needs it to: once the webcam's reading
    at the hover got used as the "close look" and the fingers closed short of the block."""
    from ..config import TARGET_TAG
    return any(s[0] == "gripper" and s[1] == TARGET_TAG for s in seen)


def look_up_close(w, cube):
    """At the hover: measure the arm's slack from its tags, then locate the block from every camera with that slack
    taken out. Prints what it saw. Returns (block or None, dq)."""
    a, off, fit, _, seen = find_cube(w)
    pred = None
    try:                                                 # the slack model's prediction, before anything is measured
        from .. import slackgp
        pred = slackgp.predict(a[:5], a[5], S.came_from)
    except Exception:
        pred = None
    dq, px, n = slack_now(w, a, off, seen)
    if pred is not None:
        # Log the prediction against what the cameras measure, using every camera's arm tags (that's how the training
        # data was measured; the tower-only fit used for the grab often has too few tags to measure the wrist).
        try:
            mq, mpx, mn = arm_slack(w, a, off, seen)
        except Exception:
            mq, mn = None, 0
        if mq is not None and mn >= SLACK_GP_MIN_TAGS:
            print("   slack model (Gaussian process) predicted " + "  ".join(
                f"{k} {pred[k]:+.1f}" for k in ("shoulder", "elbow", "wrist")) + f"  -- measured ({mn} arm tags) " +
                "  ".join(f"{k} {v:+.1f}" for k, v in zip(("shoulder", "elbow", "wrist"), mq[:3])))
            SLACK_GP.update(predicted=[round(pred[k], 2) for k in ("shoulder", "elbow", "wrist")],
                            measured=[round(float(v), 2) for v in mq[:3]], tags=int(mn), fit_px=round(float(mpx), 2))
            live.slack_pair(mq[:3], pred)
    print(f"   the arm's real joints vs its sliders (from {n} {'tower ' if getattr(S, 'hand_rel', False) else ''}arm "
          f"tags, fit {px:.1f} px): shoulder {dq[0]:+.1f}  elbow "
          f"{dq[1]:+.1f}  wrist {dq[2]:+.1f}  rotate {dq[3]:+.1f}")
    got = {}
    for name, cams in (("base cameras", BASE_CAMS), ("gripper camera", ("gripper",)), ("all cameras", None)):
        c = locate(w, [s for s in seen if cams is None or s[0] in cams], a[:4], a[5], base_offset=off, grip=a[4], dq=dq)
        if c is not None:
            c["close"] = True
        got[name] = c
        print(f"   {name:15s} " + (cube_text(c, w) if c is not None else "does not see the block's tag"))
    bc, gc, al = got["base cameras"], got["gripper camera"], got["all cameras"]
    if gc is None and "gripper" in w.cams:                    # where the model expected the tag, to debug misses
        from ..grasp import DEPTH_MM, top_tag
        X = top_tag(np.asarray(cube["xyz"]) - [0.0, 0.0, DEPTH_MM], cube["yaw_deg"])
        px = w.project("gripper", X, a[:4], a[5], base_offset=off, dq=dq)
        K = w.cams["gripper"]["K"]
        print(f"   (the model expected the tag in the gripper picture at x {px[:, 0].min():.0f}-{px[:, 0].max():.0f}, "
              f"y {px[:, 1].min():.0f}-{px[:, 1].max():.0f} of {2 * K[0, 2]:.0f}x{2 * K[1, 2]:.0f})")
    if bc is not None and gc is not None:
        d = gc["xyz"] - bc["xyz"]
        print(f"   gripper camera vs base cameras: {np.linalg.norm(d[:2]):.1f} mm sideways, {d[2]:+.1f} mm in height")
    if al is not None:
        d = al["xyz"] - cube["xyz"]
        print(f"   moved from the first look by {np.linalg.norm(d[:2]):.1f} mm sideways, {d[2]:+.1f} mm in height")
        al["gripper"] = gc is not None
        al["hand_rel"] = hand_rel(w, a, off, seen)
    return al, dq


def look_pose(w, check, h, est, dq, margin):
    """The pose nearest hover h (within LOOK_SPAN_DEG on shoulder/elbow/wrist) that the model says lets the gripper
    camera see the block's top tag, clear of the table and the block. None if there isn't one."""
    from ..grasp import DEPTH_MM, JOINT_HI, JOINT_LO, in_view, top_tag
    view = top_tag(np.asarray(est["xyz"]) - [0.0, 0.0, DEPTH_MM], est["yaw_deg"])
    steps = np.arange(-LOOK_SPAN_DEG, LOOK_SPAN_DEG + 0.1, LOOK_STEP_DEG)
    D = np.array(np.meshgrid(steps, steps, steps, indexing="ij")).reshape(3, -1).T
    order = D[np.argsort(np.abs(D).sum(axis=1))][1:]       # skip the hover itself: we just looked from there
    # Ask for the tag well inside the picture first. The nearest pose that barely fits put the tag a few px from the
    # edge, and the shoulder's real slack then carried it out of frame (three hovers in a row saw nothing).
    for edge in (LOOK_EDGE_PX, LOOK_EDGE_PX / 2.0, IN_VIEW_EDGE_PX):
        for d in order:
            q = [h[i] + d[i] for i in range(3)]
            if any(q[i] < JOINT_LO[i] or q[i] > JOINT_HI[i] for i in range(3)):
                continue
            true = [q[i] + dq[i] for i in range(3)] + [h[3] + dq[3], h[4], h[5]]
            if not in_view(w, view, true, edge=edge):
                continue
            if check.gaps(q + [h[3], h[4]], h[5])[0] < margin \
                    or cube_gap(check, est, q + [h[3], h[4]], h[5]) < HOVER_CLEAR_MM:
                continue
            if edge > IN_VIEW_EDGE_PX:
                print(f"   (a look pose with {edge:.0f} px of picture to spare)")
            return q
    return None


LOOK_PAN_DEG = 4.0                # the extra looks pan the base this far, then twice it, to one side


def look_around(w, blk, b_now, speed, zone, check, cube, extra=2):
    """A couple more looks at the block a few degrees of base apart, from the pose it was just found from. Each goes
    into the block filter (still=True: nothing touched the block in between). Returns how many saw it.

    Looks from up at HIGH_POSE never saw the block (the arm stands in front of the webcam up there), so stay at the
    search pose, or go there first. Pan to whichever side the model says keeps the arm out of the camera's view, and
    check every base turn against the block (a pan at home once swept the hand into it)."""
    from ..grasp import BASE_RANGE
    level = (ROLL_SLIDER, TRAVEL_GRIP)
    start = arm_angles()
    pose = next((list(q) for q in (SEARCH_POSE, HIGH_POSE) if max(abs(start[i] - q[i]) for i in range(3)) <= 1.0), None)
    if pose is None:
        pose = list(SEARCH_POSE)
        if not zone.contains(pose) or not zone.path_ok(start[:3], pose) \
                or check.gaps(pose + list(level), b_now)[0] < SEARCH_CLEAR_MM:
            print("   (extra looks skipped: the move to the look pose is not clear of the table/tower/arm)")
            return 0
        check_arm_move(check, cube, start[:3], pose, level, b_now, leaving=True, what="the move up to look")
        move_to(pose, zone, speed)
        wait_fresh(time.time() + 0.3)                # the arm sways for a moment after stopping
        c = find_cube(w)[3]
        if c is not None:
            blk.see(c, still=True)
    q5 = pose + list(level)
    cams = [c for c in ("webcam", "torre") if c in w.cams]

    def clear(b):                                    # how clear of the arm the best camera's view is at base b
        got = [view_clear_px(w, check, c, q5 + [b], cube["xyz"]) for c in cams]
        return max([g for g in got if g is not None], default=0.0)
    sides = sorted((-1.0, 1.0), key=lambda sg: -min(clear(b_now + sg * k * LOOK_PAN_DEG) for k in (1, 2)))
    got = 0
    for k in range(1, max(0, int(extra)) + 1):          # --looks N: the first sighting plus N-1 of these
        t = min(max(b_now + sides[0] * k * LOOK_PAN_DEG, BASE_RANGE[0]), BASE_RANGE[1])
        if abs(t - b_now) < 1.0 or clear(t) < PARK_CLEAR_PX:
            continue
        gaps = [cube_gap(check, cube, q5, b) for b in np.linspace(b_now, t, int(abs(t - b_now) / STEP_DEG) + 2)]
        if min(gaps) < CUBE_CLEAR_MM:
            print(f"   (look from base {t:+.1f} skipped: the turn would bring the hand {min(gaps):.0f} mm from the block)")
            continue
        move_base(t, closed=False)                   # open loop is fine: each look measures the base angle itself
        wait_fresh(time.time())
        c = find_cube(w)[3]
        if c is None:
            continue
        off = blk.see(c, still=True)
        got += 1
        print(f"   look from base {t:+.1f}: {off:.1f} mm from the estimate so far")
    return got


TURNED = {}                       # turn_toward's angle, so the hover can skip a base move it has already made
HOVER_BASE_CLOSED = False         # --hover-base-closed: closed-loop base turns on the way to the hover. Off by
                                  # default: the loop kept nudging (the base lands within ~0.6-0.9 deg and it asked
                                  # for 0.3), ~9.5 s a round, and the close look measures the base anyway
TURN_SAME_DEG = 0.5               # if the hover's base angle is this close to the current one, don't turn at all


def turn_toward(w, check, cube):
    """Start turning the base toward the block as soon as the looks are done, while the plan is still computing.

    The hover's base angle depends only on where the block is (base_toward), not on the plan, so this can start
    early and the hover is left with a degree or two instead of a 25-45 deg turn. Only from the search pose or up
    high (clear of the table at every angle), and the turn is checked against the block. Returns the base angle,
    or None if it didn't turn."""
    from ..grasp import BASE_RANGE, DEPTH_MM, base_toward
    a = arm_angles()
    pose = next((list(q) for q in (SEARCH_POSE, HIGH_POSE) if max(abs(a[i] - q[i]) for i in range(3)) <= 1.0), None)
    if pose is None:
        return None
    b_now = float(S.base_cmd if S.base_cmd is not None else a[5])
    t = float(np.clip(base_toward(w, np.asarray(cube["xyz"], float) - [0.0, 0.0, DEPTH_MM]), *BASE_RANGE))
    if abs(t - b_now) < 1.0:
        return b_now
    q5 = pose + [ROLL_SLIDER, TRAVEL_GRIP]
    gaps = [cube_gap(check, cube, q5, b) for b in np.linspace(b_now, t, int(abs(t - b_now) / STEP_DEG) + 2)]
    if min(gaps) < CUBE_CLEAR_MM:
        print(f"   (not turning toward the block early: the turn would bring the hand {min(gaps):.0f} mm from it)")
        return None
    print(f"   turning toward the block (base {b_now:+.1f} -> {t:+.1f}) while the plan finishes")
    move_base(t, closed=HOVER_BASE_CLOSED, settle=False)   # open loop by default: the close look measures it
    TURNED["base"] = t
    return t


HUNT_WRIST_DEG = (2.0, 4.0, 6.0, 9.0, 12.0, 16.0, 20.0)   # how far to tip the wrist either way while hunting
HUNT_ELBOW_DEG = 5.0              # once the wrist hits its limit, keep going with up to this much elbow (it tips
                                  # the hand the same way)
HUNT_SETTLE_S = 0.6


def hunt_for_tag(w, check, zone, p, blk, margin, which="both"):
    """Tip the wrist either side of the hover and take a look at each step until the gripper camera sees the block.

    look_pose predicts where the tag will be and goes there, which only works when the model of the gripper camera
    is right. When it wasn't, the tag was nowhere in the picture run after run (watching the camera feed, it was
    creeping in at the bottom). So this measures instead of predicting. Every pose still goes through the table and
    block checks first. Only runs after the model has already failed to find the block."""
    from ..grasp import JOINT_HI, JOINT_LO
    dirs = {"back": (-1,), "out": (1,), "both": (-1, 1)}.get(which, (-1, 1))
    est = blk.cube()
    h = p["hover"]
    level = (h[3], h[4])
    saved, c = zone.hand, None
    zone.hand = level
    room = min(h[2] - JOINT_LO[2], JOINT_HI[2] - h[2])
    print(f"   hunting for the tag: sweeping the wrist {'/'.join('back' if x < 0 else 'out' for x in dirs)} "
          f"from {h[2]:.1f} (it has {room:.0f} steps of room)")
    try:
        # one clean sweep each way rather than alternating sides, so the tag can be seen walking across the picture
        for sign in dirs:
            for step in HUNT_WRIST_DEG:
                d = sign * step
                wrist = min(JOINT_HI[2], max(JOINT_LO[2], h[2] + d))
                over = abs(h[2] + d - wrist)              # whatever the wrist couldn't take goes to the elbow
                elbow = h[1] + sign * min(over, HUNT_ELBOW_DEG)
                if over > HUNT_ELBOW_DEG or elbow < JOINT_LO[1] or elbow > JOINT_HI[1]:
                    continue
                q = [h[0], elbow, wrist]
                if check.gaps(q + list(level), h[5])[0] < margin \
                        or cube_gap(check, est, q + list(level), h[5]) < HOVER_CLEAR_MM:
                    continue
                try:
                    check_arm_move(check, est, h[:3], q, level, h[5], between=True, what="tipping the wrist to look")
                except ArmError:
                    continue
                move_to(q, zone, PICK_SLOW)
                pause(HUNT_SETTLE_S, "hunt settle")
                a, off, _, _, seen = find_cube(w)
                dq2, _, _ = slack_now(w, a, off, seen)
                got = locate(w, seen, a[:4], a[5], base_offset=off, grip=a[4], dq=dq2)
                if got is not None:
                    got["hand_rel"] = hand_rel(w, a, off, seen)
                if not gripper_sees(seen):                # a base camera seeing it isn't a close look
                    got = None
                how = f"wrist {q[2]:+6.1f}" + (f" elbow {q[1]:+6.1f}" if abs(q[1] - h[1]) > 0.05 else "")
                print(f"      {how} ({d:+.0f}): " + ("SEEN -- " + cube_text(got, w)
                                                     if got is not None else "nothing"))
                if got is not None:
                    got["close"] = True
                    return got
    finally:
        try:
            move_to(h[:3], zone, PICK_SLOW)               # always come back to the hover the plan was made for
        finally:
            zone.hand = saved
            pause(0.4, "back at the hover")
    return c


def back_off_look(w, check, zone, p, blk, dq, margin):
    """The hover can't see the block: look from the nearest pose that can (look_pose), then come back."""
    return look_from_near(w, check, zone, p["hover"], blk.cube(), dq, margin, need_gripper=True)


def look_from_near(w, check, zone, h, est, dq, margin, need_gripper=False):
    """From hover h, move to the nearest pose that should see a block at est, look, and come back to h. Returns the
    block or None. Used for the pickup's close look and for checking a set-down. need_gripper: only count the
    gripper camera's own sighting."""
    look = look_pose(w, check, h, est, dq, margin)
    if look is None:
        print(f"   no arm pose within {LOOK_SPAN_DEG:.0f} deg of the hover would see it either")
        return None
    level = (h[3], h[4])
    check_arm_move(check, est, h[:3], look, level, h[5], between=True, what="moving to look")
    print(f"   moving to look again from shoulder {look[0]:.1f}  elbow {look[1]:.1f}  wrist {look[2]:.1f}...")
    saved, c = zone.hand, None
    zone.hand = level
    try:
        move_to(look, zone, PICK_SLOW)
        pause(0.8, "look pose settle")
        a, off, _, _, seen = find_cube(w)
        dq2, _, _ = slack_now(w, a, off, seen)
        c = locate(w, seen, a[:4], a[5], base_offset=off, grip=a[4], dq=dq2)
        if c is not None and need_gripper:
            c["hand_rel"] = hand_rel(w, a, off, seen)
        if c is not None and need_gripper and not gripper_sees(seen):
            print("   from there: only the base cameras see it (" + cube_text(c, w) + "), not the gripper camera")
            c = None
        else:
            if c is not None:
                c["close"] = True
            print("   from there: " + (cube_text(c, w) if c is not None else "still does not see the block's tag"))
    finally:
        check_arm_move(check, est, look, h[:3], level, h[5], between=True, what="the move back to the hover")
        move_to(h[:3], zone, PICK_SLOW)
        zone.hand = saved
        pause(0.5, "back at the hover")
    return c


def by_hand_at_hover(w, check, zone, p, blk):
    """--pause-hover: stop and let the sliders be driven by hand until a camera sees the block, then carry on.
    Nothing here commands the arm while it's being driven; the one move back to the hover is checked like any other."""
    from ..grasp import DEPTH_MM, top_tag
    from .hovertune import pause_at_hover
    est = blk.cube()
    h = p["hover"]
    X = top_tag(np.asarray(est["xyz"]) - [0.0, 0.0, DEPTH_MM], est["yaw_deg"])
    first, path = pause_at_hover(w, check, est, h, X)
    now = arm_angles()
    if max(abs(now[i] - h[i]) for i in range(3)) > 0.5:
        print("   putting the arm back at the hover...")
        saved = zone.hand
        zone.hand = (h[3], h[4])
        try:
            check_arm_move(check, est, list(now[:3]), h[:3], (h[3], h[4]), h[5],
                           between=True, what="back to the hover after the pause")
            move_to(h[:3], zone, PICK_SLOW)
        except ArmError as e:
            zone.hand = saved
            raise Stuck(f"Cannot get back to the hover from where the arm was left ({e}). Drive it back by hand, "
                        f"then run the grab again. Nothing is in the fingers.")
        zone.hand = saved
        pause(0.4, "settle")
    if first is None:
        return None
    a, off, _, _, seen = find_cube(w)                    # look once more now that it's back at the hover
    dq2, _, _ = arm_slack(w, a, off, seen)
    c = locate(w, seen, a[:4], a[5], base_offset=off, grip=a[4], dq=dq2)
    if c is not None:
        c["close"] = True
        print("   and from the hover it is seen now: " + cube_text(c, w))
    return c


CLOSE_IN_MM = 0.0                 # --close-in: aim the grasp this far toward the tower for blocks nearer than
CLOSE_IN_FROM_MM = 320.0          # this. Off by default since the fingertip calibration fix.


FACE_FIX_MIN_MM = 2.0             # at the bottom, slide the fingers along the face if they're this far off...
FACE_FIX_MAX_MM = 12.0            # ...but ignore readings bigger than this (a misread, not something to chase)


def face_axis(w, q):
    """Horizontal unit vector along the block face the fingers grip (perpendicular to the way they close)."""
    from ..grasp import hand
    y = hand(w, list(q))[1][:, 1]
    a = np.array([-y[1], y[0]])
    return a / max(1e-9, np.linalg.norm(a))


def face_offset(w, aim):
    """How far (mm) the block is from `aim` along the gripped face, read by the gripper camera alone with no slack
    correction (the same frame the hand-relative aim was made in). None if it can't see the tag."""
    from ..config import TARGET_TAG
    a, off, _, _, seen = find_cube(w)
    g = [s for s in seen if s[0] == "gripper" and s[1] == TARGET_TAG]
    if not g:
        return None, a
    c = locate(w, g, a[:4], a[5], base_offset=off, grip=a[4], dq=None)
    if c is None:
        return None, a
    d = np.asarray(c["xyz"], float)[:2] - np.asarray(aim["xyz"], float)[:2]
    return float(d @ face_axis(w, a)), a


def go_watching(poses, go, w, check, zone, near, p2, margin, args, rot):
    """--hand-rel: go in stop by stop, reading at each stop how far the block is from the aim along the face the
    fingers grip, with later (closer, sharper) readings weighted more. At the bottom, if the fingers are
    FACE_FIX_MIN_MM or more off, re-solve the grasp for the shifted aim and make one small move before closing.

    Misses close in were along the face, not in depth: the pads caught the block's far corner. Across the fingers
    there's ~13 mm of play and the squeeze centres it; along the face a few mm decides the grab, and the error grows
    on the way down after the last look (the slack changes as the arm reaches)."""
    ref = LAST_BLOCK.get("near") or near              # the close look as seen; --close-in's nudge (in near) is
    a_g = face_axis(w, p2["grasp"])                       # deliberate, so only correct what's beyond it
    meant = float((np.asarray(ref["xyz"], float)[:2] - np.asarray(near["xyz"], float)[:2]) @ a_g)
    est, wsum = 0.0, 0.0
    for k, q in enumerate(poses):
        go([q])
        wait_fresh(time.time(), cams=("gripper",))   # a picture taken after the arm stopped
        with trace.span("camera", "face check"):
            d, _ = face_offset(w, near)
        d = None if d is None else d - meant
        if d is not None and abs(d) <= FACE_FIX_MAX_MM:
            wk = float(k + 1)
            est, wsum = est + wk * d, wsum + wk
            print(f"      along the face: {d:+.1f} mm from the aim")
    if wsum == 0.0:
        print("      (the gripper camera did not see the block on the way in: closing as planned)")
        return
    d = est / wsum
    if abs(d) < FACE_FIX_MIN_MM:
        print(f"      fingers on the face ({d:+.1f} mm): closing")
        return
    shifted = dict(near)
    shifted["xyz"] = np.asarray(near["xyz"], float) + np.append(a_g * d, 0.0)
    with trace.span("compute", "face fix"):
        p3 = replan(w, check, shifted, p2, margin, grasp_margin=args.grasp_margin)
    if "why" in p3:
        print(f"      the fingers are {d:+.1f} mm off along the face but the shifted grasp cannot be reached: closing")
        return
    q = list(p3["grasp"][:3])
    if max(abs(q[i] - arm_angles()[i]) for i in range(3)) > 6.0:
        print(f"      the fix for {d:+.1f} mm wants more than 6 deg of joint: not risking it, closing")
        return
    print(f"      the fingers are {d:+.1f} mm off along the face: moving them onto it")
    move_to(q, zone, PICK_SLOW)


def aim_in_close(w, near, mm):
    """The block to aim the grasp at: `near` moved `mm` toward the base axis if it's nearer than CLOSE_IN_FROM_MM
    (--close-in; 0 = off), otherwise unchanged."""
    xy = np.asarray(near["xyz"], float)[:2] - w.axis[:2]
    r = float(np.linalg.norm(xy))
    if not mm or r >= CLOSE_IN_FROM_MM or r < 1.0:
        return near
    got = dict(near)
    got["xyz"] = np.asarray(near["xyz"], float) - np.append(xy / r * mm, 0.0)
    print(f"   aiming {mm:.0f} mm nearer the tower than the close look ({r:.0f} mm out; --close-in)")
    return got


def look_close(w, check, zone, cube, p, blk, margin):
    """At the hover: look up close; if the gripper camera doesn't see the block, back off and look; then hunt for it;
    then fall back on the block filter if it's sure enough. Every look goes into the filter. (block or None, dq)."""
    near, dq = look_up_close(w, cube)
    if near is not None and not near.get("gripper", True) and "gripper" in w.cams:
        blk.see(near)                                    # the base cameras' reading still goes in the filter
        print("   the gripper camera does not see the block from the hover: that is not a close look -- looking for it")
        near = None
    if near is None:
        near = back_off_look(w, check, zone, p, blk, dq, margin)
    if near is None:                                     # the model can't say where to look: go and find out
        near = hunt_for_tag(w, check, zone, p, blk, margin, which=S.hunt_wrist)
    if near is None and getattr(S, "pause_hover", False):
        near = by_hand_at_hover(w, check, zone, p, blk)
    if near is not None:
        hr = near.pop("hand_rel", None)
        d = blk.see(near)
        print(f"   block filter: +-{blk.sigma_mm():.1f} mm from {blk.looks} looks (this look {d:.1f} mm from the estimate)")
        if hr is not None:                               # --hand-rel: grab where the gripper camera saw it, no slack
            off = float(np.linalg.norm(np.asarray(hr["xyz"])[:2] - np.asarray(near["xyz"])[:2]))
            print(f"   hand-relative: the gripper camera's own reading, no slack ({off:.1f} mm from the slack-corrected "
                  f"one); the grab goes in without the slack taken out")
            return hr, np.zeros_like(np.asarray(dq, float))
        return blk.cube(), dq
    if blk.sigma_mm() <= BLIND_MAX_MM:
        print(f"   not seen up close: going on the block filter (+-{blk.sigma_mm():.1f} mm from {blk.looks} looks; the "
              f"open fingers have ~13 mm either side)")
        return blk.cube(), dq
    print(f"   not seen up close, and the block filter is only +-{blk.sigma_mm():.1f} mm sure (needs {BLIND_MAX_MM:.0f})")
    return None, dq


def carry_verdict(off_start, off_want, d):
    """(carried, placed, text) for a block found off_start mm from where it was picked up and off_want mm from where
    it was put down, after a carry of d.

    carried: it moved at least half the carry and ended nearer the new spot than the old one, i.e. the fingers took
    it there. placed: also within CARRY_FOUND_MM of the aim. These used to be one test, so a good grab that landed
    8 mm off the aim (the base is ~0.25 deg off on small turns) was scored as a failure."""
    n = float(np.linalg.norm(np.asarray(d, float)[:2]))
    carried = bool(n > 0 and off_start >= max(CARRY_FOUND_MM, 0.5 * n) and off_want < n)
    placed = bool(carried and off_want <= CARRY_FOUND_MM)
    text = (f"found {off_start:.1f} mm from where it was, {off_want:.1f} mm from where it was put"
            + ("" if not carried else " -> CARRIED" + ("" if placed else f" (set down {off_want:.0f} mm off the aim)")))
    return carried, placed, text


def look_again(w):
    """find_cube's block; if it sees none, wait for frames taken after now and look once more (a look right after
    the arm stops can read a frame from during the move)."""
    c = find_cube(w)[3]
    if c is None:
        wait_fresh(time.time())
        c = find_cube(w)[3]
    return c


def home_check(w, cube, blk):
    """Back home, look with the same cameras from the same pose as the first look (their errors mostly cancel).
    True if the block is where it was put, None if they can't see it."""
    d = blk.notes.get("carry")
    c = look_again(w)
    if c is None:
        print("From home: no camera sees the block")
        return None
    want = np.asarray(cube["xyz"], float) + d
    off_want = float(np.linalg.norm(c["xyz"][:2] - want[:2]))
    off_start = float(np.linalg.norm(c["xyz"][:2] - np.asarray(cube["xyz"])[:2]))
    ok, _, text = carry_verdict(off_start, off_want, d)
    print("From home (same cameras as the first look): " + text)
    return ok


def commanded(q, dq):
    """Slider values that should put the arm at the true pose q, given the measured slack dq (real minus slider)."""
    from ..grasp import MAP_HI, MAP_LO
    return [min(MAP_HI[i], max(MAP_LO[i], q[i] - dq[i])) for i in range(3)] + [q[3] - dq[3], q[4], q[5]]


def lift_seen(w, dq=None):
    """How far above its resting height the block's highest visible tag is right now (mm), plus which tag and
    camera; (None, None, None) if no camera sees any. At rest the top tag is at CUBE_MM and the side tags at half that."""
    from ..config import TAG_MM
    a = arm_angles()
    seen = sightings_now()
    off, _ = w.fit_base_offset(seen, a[:4], a[5])
    best = (None, None, None)
    for c, t, px in seen:
        if c not in w.cams or (t != 7 and t not in SIDE_TAGS):
            continue
        got = w.locate_tag(c, px, TAG_MM.get(t, 20.0), a[:4], a[5], base_offset=off, dq=dq)
        if got is None:
            continue
        z = got[1][2] - w.table_height(got[1][0], got[1][1])
        rise = z - (CUBE_MM if t == 7 else CUBE_MM / 2.0)
        if best[0] is None or rise > best[0]:
            best = (float(rise), t, c)
    return best


def snap_block(args, w, cube, a, off, fit):
    """--snap: save what the cameras see of the block without planning anything. A snap takes a couple of seconds
    instead of a minute of planning, so many block positions can be collected and planned later off the arm
    (tools/replay_grasp.py)."""
    import json as _json, time as _time
    import numpy as _np
    try:
        r = float(_np.hypot(cube["xyz"][0] - w.axis[0], cube["xyz"][1] - w.axis[1]))
        bearing = float(_np.degrees(_np.arctan2(cube["xyz"][1] - w.axis[1], cube["xyz"][0] - w.axis[0])))
        out = {"when": _time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "snap",
               "world_made": w.d.get("made"), "tip_in_G": [round(float(v), 2) for v in w.tip_in_G],
               "block": {"xyz": [round(float(v), 2) for v in cube["xyz"]],
                         "yaw_deg": round(float(cube["yaw_deg"]), 2),
                         "above_table_mm": round(float(cube.get("above_table_mm", 0.0)), 2),
                         "out_mm": round(r, 1), "bearing_deg": round(bearing, 1),
                         "cams": {k: round(float(v), 2) for k, v in (cube.get("cams") or {}).items()},
                         "tops": {k: round(float(v), 2) for k, v in (cube.get("tops") or {}).items()}},
               "seen_from": {"joints": [round(float(v), 2) for v in a],
                             "base_offset_deg": round(float(off), 3), "table_fit_px": round(float(fit), 2)},
               "args": {k: v for k, v in sorted(vars(args).items()) if not callable(v)}}
        d = os.path.join(S.data_dir, "grasp_runs")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, _time.strftime("snap_%Y%m%d_%H%M%S.json"))
        _json.dump(out, open(path, "w"), indent=1)
        print(f"   {r:.0f} mm out, bearing {bearing:+.0f}, turned {cube['yaw_deg']:+.0f}  ->  saved {path}")
    except Exception as e:
        print(f"   (could not save: {e})")


def save_run(args, w, cube, p, margin, seen_from=None):
    """Save every run, planned or refused: the code version, settings, the block, the plan. Without this, working
    out why a grab failed meant scrolling the terminal and guessing what code was installed.

    seen_from: (arm angles, base offset, table fit px) of the look that located the block."""
    import hashlib, json as _json, time as _time
    try:
        import inspect
        from .. import tablecheck as _tc, world as _wd, cube as _cb
        ver = {}
        for nm, mod in (("grasp.py", _g), ("tablecheck.py", _tc), ("world.py", _wd), ("cube.py", _cb),
                        ("modes/grasp.py", sys.modules[__name__])):
            f = inspect.getsourcefile(mod)
            ver[nm] = hashlib.md5(open(f, "rb").read()).hexdigest()[:10] if f else "?"
        from ..version import stamp
        out = {"when": _time.strftime("%Y-%m-%d %H:%M:%S"), "code": ver, "version": stamp(),
               "world_made": w.d.get("made"), "tip_in_G": [round(float(v), 2) for v in w.tip_in_G],
               "args": {k: v for k, v in sorted(vars(args).items()) if not callable(v)},
               "limits": {"MAP_LO": list(_g.MAP_LO), "MAP_HI": list(_g.MAP_HI),
                          "JOINT_LO": [round(v, 2) for v in _g.JOINT_LO],
                          "JOINT_HI": [round(v, 2) for v in _g.JOINT_HI],
                          "SLACK_ROOM": [round(v, 2) for v in _g.SLACK_ROOM],
                          "CUBE_R_MM": CUBE_R_MM, "CUBE_CLEAR_MM": CUBE_CLEAR_MM,
                          "HOVER_CLEAR_MM": HOVER_CLEAR_MM, "HOVER_LADDER": list(_g.HOVER_LADDER),
                          "PATH_PITCH_DRIFT": _g.PATH_PITCH_DRIFT, "table_margin": margin},
               "block": None if cube is None else
                        {"xyz": [round(float(v), 1) for v in cube["xyz"]],
                         "yaw_deg": round(float(cube["yaw_deg"]), 1),
                         "above_table_mm": round(float(cube.get("above_table_mm", 0.0)), 1),
                         "cams": {k: round(float(v), 1) for k, v in (cube.get("cams") or {}).items()},
                         "tops": {k: round(float(v), 1) for k, v in (cube.get("tops") or {}).items()}},
               "tried": p.get("tried", []),
               "planned": "why" not in p}
        if seen_from is not None:
            a, off, fit = seen_from
            out["seen_from"] = {"joints": [round(float(v), 2) for v in a], "base_offset_deg": round(float(off), 3),
                                "table_fit_px": round(float(fit), 2)}
        if "why" in p:
            out["why"] = p["why"]
        else:
            out["plan"] = {"hover_mm": round(float(p["hover_mm"]), 1), "pitch": round(float(p["pitch"]), 1),
                           "face_deg": round(float(p["face_deg"]), 1), "approach": p["approach"],
                           "hover": [round(float(v), 2) for v in p["hover"]],
                           "grasp": [round(float(v), 2) for v in p["grasp"]],
                           "room": round(float(min(_g.limit_room(q) for q in p["path"])), 2),
                           "min_gap_mm": round(float(p["min_gap_mm"]), 1),
                           "hover_gap_mm": round(float(p["hover_gap_mm"]), 1), "sees": bool(p["sees"])}
        d = os.path.join(S.data_dir, "grasp_runs")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, _time.strftime("grasp_%Y%m%d_%H%M%S.json"))
        _json.dump(out, open(path, "w"), indent=1)
        LAST_RUN_PATH[0] = path
        print(f"   run saved: {path}")
    except Exception as e:
        print(f"   (could not save the run: {e})")


class Laps:
    """How long each phase of a grab took. lap(name) records the time since the last lap; the rows are printed
    after the grab and saved in the run file and show's table."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = self.t = time.time()
        self.rows = []
        trace.reset()
        TURNED.clear()
        live.update(step="Finding the block", laps=[], t_round0=self.t0, result=None, grip={}, place={}, slack=None,
                    safety=None, plan=None, slack_pred=None)
        SLACK_GP.clear()

    def lap(self, name):
        now = time.time()
        self.rows.append([name, round(now - self.t, 1)])
        self.t = now
        live.lap(name, self.rows)

    def text(self):
        return "  ".join(f"{n} {s:.0f}" for n, s in self.rows) + f"  = {time.time() - self.t0:.0f} s"


LAPS = Laps()
LOOKS_RAW = []                    # every look this grab made (time, joints, raw tag corners per camera), so a run
                                  # can be re-checked offline under a newer world model
LAST_RUN_PATH = [None]            # where save_run put this run, so finish_run can add what happened


def finish_run(result, blk=None, near=None, cube=None):
    """Add what actually happened (held, squeeze, where it landed, the timings and trace) to the run file save_run
    wrote before the arm moved."""
    import json as _json
    path = LAST_RUN_PATH[0]
    if not path or not os.path.exists(path):
        return
    try:
        d = _json.load(open(path))
        n = dict(blk.notes) if blk is not None and getattr(blk, "notes", None) else {}
        got = {"result": result, "held": n.get("held"), "squeeze": n.get("squeeze"), "carried": n.get("carried"),
               "spacing_mm": {k: (None if v is None else round(float(v), 2)) for k, v in SPACING.items()},
               "lift_mm": n.get("lift_mm"), "lifted": n.get("lifted"),
               "placed_off_mm": n.get("placed_off_mm"), "let_go_off_mm": n.get("let_go_off_mm"),
               "moved_after_release_mm": n.get("moved_after_release_mm"),
               "carry_plan": n.get("carry_plan"), "released_xyz": n.get("released_xyz"), "found_xyz": n.get("found_xyz"),
               "times_s": [list(r) for r in LAPS.rows], "total_s": round(time.time() - LAPS.t0, 1),
               "t0": round(LAPS.t0, 3), "trace": trace.dump(),
               "looks_raw": list(LOOKS_RAW), "slack_gp": dict(SLACK_GP) or None}
        if n.get("carry") is not None:
            got["carried_mm"] = [round(float(v), 1) for v in n["carry"]]
        if near is not None and cube is not None:            # how far the close-up look moved the estimate
            import numpy as _np
            got["close_look_moved_mm"] = round(float(_np.linalg.norm(
                _np.asarray(near["xyz"], float)[:2] - _np.asarray(cube["xyz"], float)[:2])), 1)
            got["close_look_xyz"] = [round(float(v), 1) for v in near["xyz"]]
        if blk is not None:
            got["filter_sigma_mm"] = round(float(blk.sigma_mm()), 1)
            got["looks"] = blk.looks
        d["outcome"] = got
        _json.dump(d, open(path, "w"), indent=1)
    except Exception as e:
        print(f"   (could not record the outcome: {e})")


def come_down(args):
    """show --stay-up, after the last round: from waiting up high back down to home (checked against the block)."""
    if STAY_UP:
        from ..tablecheck import load
        go_home(Zone(args.margin), STAY_UP["base"], args.speed, load())


def mode_grasp(args):
    from ..tablecheck import load
    if not (args.dry or args.hover or args.grab or args.snap):
        raise ArmError("Say --snap (look and save, no planning), --dry (plan only), --hover (go above the block "
                       "and stop) or --grab (the whole grab).")
    S.exact_table = True
    check = load()
    w = check.w
    if "tip_in_G" not in w.d:
        print("   (world.json has no measured fingertip yet: run  python3 arm.py fit_touch  first; using the guess)")
    global CUBE_R_MM, CUBE_CLEAR_MM, HOVER_CLEAR_MM, PICK_SLOW, HOVER_BASE_CLOSED, PAN_OVERLAP
    HOVER_BASE_CLOSED = bool(getattr(args, "hover_base_closed", False))
    PAN_OVERLAP = bool(getattr(args, "pan_overlap", False))
    from . import survey as _survey
    _survey.HAND_FAST = True              # quicker gripper/rotate moves in grasp and show only
    for name, got in (("CUBE_R_MM", args.cube_r), ("CUBE_CLEAR_MM", args.cube_clear),
                      ("HOVER_CLEAR_MM", args.hover_clear), ("PICK_SLOW", getattr(args, "slow", None))):
        if got is not None:
            print(f"   {name} {globals()[name]:.0f} -> {got:.0f} (from the command line)")
            globals()[name] = float(got)
    margin = S.table_margin_mm if S.table_margin_mm is not None else args.margin
    zone = Zone(args.margin)
    lo, hi = set_limits(zone)             # before any planning: the planner's joint range comes from the safety map
    print(f"   planner joints: shoulder {lo[0]:.0f}-{hi[0]:.0f}  elbow {lo[1]:.0f}-{hi[1]:.0f}  "
          f"wrist {lo[2]:.0f}-{hi[2]:.0f}  (safety map {zone.lo[0]:.0f}-{zone.hi[0]:.0f}, keeping "
          f"{_g.SLACK_ROOM[0]:.1f}/{_g.SLACK_ROOM[1]:.1f}/{_g.SLACK_ROOM[2]:.1f} steps clear = {_g.SLACK_K:g} sigma "
          f"of the measured slack)")
    if not args.snap and not args.dry:
        # A run stopped with Ctrl+C can leave the wrist rotated (only go_home straightens it), and the next run then
        # had to be rescued by hand. Straighten it first.
        from .survey import set_hand
        now = arm_angles()
        if abs(float(now[3]) - ROLL_SLIDER) > 1.0:
            print(f"   wrist rotate is at {now[3]:.0f}: putting it back to {ROLL_SLIDER:.0f} before starting")
            set_hand(ROLL_SLIDER, float(now[4]))
        # A stopped run can also leave the arm at a hover beside the block, and the next grab would plan and set off
        # from there. Unless it's at home, HIGH_POSE or the search pose, go up to HIGH_POSE first.
        now = arm_angles()
        if not any(max(abs(now[i] - q[i]) for i in range(3)) <= START_TOL_DEG for q in (zone.home, HIGH_POSE,
                                                                                         SEARCH_POSE)):
            check_ready(zone)
            print(f"   the arm was left mid-way (shoulder {now[0]:.0f} elbow {now[1]:.0f} wrist {now[2]:.0f}): up "
                  f"high before looking")
            set_hand(ROLL_SLIDER, TRAVEL_GRIP)
            move_to(HIGH_POSE, zone, args.speed)
    LAPS.reset()
    LOOKS_RAW.clear()
    a, off, fit, cube, _ = find_cube(w)
    if cube is None:                                     # maybe a frame from before the arm stopped: look once more
        wait_fresh(time.time())
        a, off, fit, cube, _ = find_cube(w)
    b_home = None
    if cube is None and STAY_UP and not args.dry:
        # waiting up high and nothing sees the block: go down to home, where a search starts from
        print("   (up high and no camera sees the block: going down to home first)")
        go_home(Zone(args.margin), STAY_UP["base"], args.speed)
        a, off, fit, cube, _ = find_cube(w)
    if cube is None:
        if args.dry:
            raise ArmError("No camera sees the block's tag (7). Put it on the mat where the tower camera or the webcam "
                           "can see it (--hover or --grab search for it).")
        check_ready(zone)
        base_ready(zone)
        b_home = float(S.base_cmd if S.base_cmd is not None else a[5])
        cube = search(w, zone, check, args.speed)
        if cube is None:
            go_home(zone, b_home, args.speed)
            raise ArmError("Searched the whole base range: no camera saw the block's tag (7). Is it face up and on the mat?")
        a, off, fit, _, _ = find_cube(w)
    LAPS.lap("find" if b_home is None else "search")
    if cube is not None:
        live_block(w, cube, off=off, fit=fit)
        live.event(f"found the block {np.linalg.norm(np.asarray(cube['xyz'])[:2] - w.axis[:2]):.0f} mm out")
    if args.snap:                                        # look and save only; plan later, off the arm
        print("Block: " + cube_text(cube, w))
        snap_block(args, w, cube, a, off, fit)
        return
    reach_in = float(getattr(args, "reach_in", 0.0) or 0.0)
    if reach_in:
        # --reach-in: shift the whole grab this far toward the tower. A stopgap from before the fingertip was
        # calibrated (the fingers closed ~2 mm too far out); leave it at 0.
        d = np.asarray(cube["xyz"], float)[:2] - w.axis[:2]
        n = float(np.linalg.norm(d))
        if n > 1.0:
            cube = dict(cube)
            cube["xyz"] = np.asarray(cube["xyz"], float) - np.array([d[0] / n * reach_in, d[1] / n * reach_in, 0.0])
            print(f"   grabbing {reach_in:.1f} mm closer to the tower than the block reads (--reach-in)")
    print("Block: " + cube_text(cube, w) + f"\n      (base {off:+.2f} deg from the world's zero, table fit {fit:.1f} px)")
    if abs(cube["above_table_mm"] - CUBE_MM) > 12:
        print(f"   (the tag is {cube['above_table_mm']:.0f} mm up, not ~{CUBE_MM:.0f}: on the plywood, tipped over, or a misread)")

    def hover_clear_of_block(pp):
        """Reject plans whose hover (crossing with the fingers wide, or turning the hand there) comes within
        HOVER_CLEAR_MM of the block: the grab would refuse them anyway, so keep searching instead."""
        h = pp["hover"]
        poses = [("crossing, fingers wide", list(h[:3]) + [ROLL_SLIDER, HOVER_GRIP]),
                 ("at the hover, hand turned", list(h[:3]) + [h[3], h[4]])]
        worst, why_ = None, ""
        for what, q in poses:
            g = cube_gap(check, cube, q, h[5])
            if worst is None or g < worst:
                worst, why_ = g, what
        if args.why and worst < HOVER_CLEAR_MM:          # --why: say which part of the hand is too close, and where
            q = dict(poses)[why_]
            X = check.points(q, h[5])
            c = np.asarray(cube["xyz"], float)
            dr = np.linalg.norm(X[:, :2] - c[:2], axis=1) - CUBE_R_MM
            dz = X[:, 2] - c[2]
            k = int(np.argmin(np.where((dr > 0) | (dz > 0), np.hypot(np.maximum(dr, 0), np.maximum(dz, 0)),
                                       np.maximum(dr, dz))))
            print(f"      hover {pp['hover_mm']:.0f} mm, {why_}: {worst:+.1f} mm from the block "
                  f"(needs {HOVER_CLEAR_MM:.0f}). Closest bit of the hand is {dr[k] + CUBE_R_MM:.0f} mm from the "
                  f"block's middle ({dr[k]:+.0f} past its {CUBE_R_MM:.0f} mm cylinder) and {dz[k]:+.0f} mm above "
                  f"its top.  shoulder {h[0]:.0f} elbow {h[1]:.0f} wrist {h[2]:.0f} rotate {h[3]:.0f}")
        return worst >= HOVER_CLEAR_MM

    if args.pitch_drift is not None:
        _g.PATH_PITCH_DRIFT = float(args.pitch_drift)
        print(f"   PATH_PITCH_DRIFT -> {_g.PATH_PITCH_DRIFT:.0f} deg (from the command line)")
    kw = {} if args.grasp_depth is None else {"depth": float(args.grasp_depth)}

    def first_plan(cube):
        """Try the precomputed plan table first (plan_seeded keeps a table plan only if it passes every check
        plan_best would make), else plan from scratch."""
        t0 = time.time()
        got = _g.plan_seeded(w, check, cube, margin, plan_table(), accept=hover_clear_of_block,
                             grasp_margin=args.grasp_margin, **kw)
        src = "the precomputed plan table"
        if got is not None:
            print(f"   (planned from the plan table: {got['seeded']})")
        else:
            got = plan_best(w, check, cube, margin, grasp_margin=args.grasp_margin, accept=hover_clear_of_block,
                            hover_floor=HOVER_CLEAR_MM, **kw)
            src = "scratch"
        if "why" not in got:                             # for the demo page: where the plan came from and how fast
            ms = (time.time() - t0) * 1000.0
            live.update(plan={"source": src, "ms": ms, "approach": got.get("approach"), "hover_mm": got.get("hover_mm")})
            live.event(f"grasp planned from {src} in {ms:.0f} ms ({got.get('approach')})")
        return got

    # Plan in the background while the extra looks happen: the first plan only needs the first look, and the looks
    # only turn the base a few degrees. Afterwards, re-plan from it for whatever the looks added.
    looking = not args.dry
    n_looks, b_look, p_looks = 0, None, None
    if looking:
        job = CarryPlan(lambda c=dict(cube): first_plan(c))
        blk = BlockFilter()
        blk.see(cube)
        if b_home is None:                # (the search already readied the controller and base)
            check_ready(zone)
            base_ready(zone)
        b_look = float(S.base_cmd if S.base_cmd is not None else arm_angles()[5])
        if args.looks > 1:
            n_looks = look_around(w, blk, b_look, args.speed, zone, check, cube, extra=args.looks - 1)
        LAPS.lap("looks")
        if n_looks:                       # the re-plan for what the looks added, also in the background
            c_looks = blk.cube()
            job_looks = CarryPlan(lambda c=c_looks: (lambda p0: p0 if "why" in p0 else
                                                     replan(w, check, c, p0, margin, grasp_margin=args.grasp_margin))(job.get()))
        turn_toward(w, check, blk.cube())
        LAPS.lap("turn")
        with trace.span("compute", "first plan (finishing)"):
            p = job.get()
        if n_looks:
            with trace.span("compute", "re-plan after the looks (finishing)"):
                p_looks = job_looks.get()
    else:
        with trace.span("compute", "first plan"):
            p = first_plan(cube)
    LAPS.lap("plan")
    if "why" in p:
        print("No grab plan:")
        for r in p["why"]:
            print("   " + r)
        save_run(args, w, cube, p, margin, seen_from=(a, off, fit))
        if b_home is not None:                               # it searched, so go back home from the search pose
            go_home(zone, b_home, args.speed)                #   (a dry run never searches, so it still moves nothing)
        return
    save_run(args, w, cube, p, margin, seen_from=(a, off, fit))
    print(plan_text(w, p, margin, args.grasp_margin))
    if args.dry:
        print("   (dry run: nothing moved)")
        return

    LAST_BLOCK.clear()
    LAST_BLOCK.update(cube)
    if not looking:
        blk = BlockFilter()
        blk.see(cube)
        if b_home is None:
            check_ready(zone)
            base_ready(zone)
    b_start = b_home if b_home is not None else float(S.base_cmd if S.base_cmd is not None else arm_angles()[5])
    b_start = STAY_UP.get("base", b_start)                  # waiting up high: home is still the first round's
    if looking:
        if n_looks:
            moved = float(np.linalg.norm(blk.cube()["xyz"][:2] - np.asarray(cube["xyz"])[:2]))
            cube = blk.cube()                               # the closures below read `cube`, so they follow this
            LAST_BLOCK.clear()
            LAST_BLOCK.update(cube)
            print(f"   from {blk.looks} looks: {cube_text(cube, w)}")
            live_block(w, cube, blk)
            live.event(f"{blk.looks} looks agree to +-{blk.sigma_mm():.1f} mm")
            print(f"   ({moved:.1f} mm from the single look the plan above was made on, +-{blk.sigma_mm():.1f} mm)")
            # Re-plan from the first plan rather than from scratch: a second full plan_best here took ~13 s for an
            # estimate the looks had moved by only 0.5-4 mm. Fall back to the full search if the re-plan's hover
            # can't see or clear the block.
            p_new = p_looks if p_looks is not None else replan(w, check, cube, p, margin, grasp_margin=args.grasp_margin)
            if "why" not in p_new and p_new.get("sees", True) and hover_clear_of_block(p_new):
                p = p_new
            else:
                p = first_plan(cube)
            if "why" in p:
                raise ArmError("No grab plan from the averaged looks: " + "; ".join(str(x) for x in p["why"][:2]))
            print(plan_text(w, p, margin, args.grasp_margin))
    # How close the way home already comes to the block. show uses this to keep set-downs out from under it.
    level = (ROLL_SLIDER, TRAVEL_GRIP)
    HOME_SWEEP.update(home=list(zone.home), base=b_start)
    LAST_BLOCK["home_gap"] = min(cube_gap(check, cube, list(q) + list(level), b_start)
                                 for q in arm_path(HIGH_POSE, zone.home))
    b0 = go_to_hover(zone, check, cube, p, args.speed)
    LAPS.lap("to hover")
    if looking:                                             # the looks moved the base; home is where it started
        b0 = b_look
    b0 = b0 if b_home is None else b_home                   # after a search: home is where the search started
    b0 = STAY_UP.get("base", b0)                            # waiting up high between rounds: the same home every round
    try:
        print("At the hover. The block now:")
        near, dq = look_close(w, check, zone, cube, p, blk, margin)
        if near is not None:
            live_block(w, near, blk)
        if dq is not None:
            live.update(slack={"shoulder": float(dq[0]), "elbow": float(dq[1]), "wrist": float(dq[2]),
                               "rotate": float(dq[3]) if len(dq) > 3 else None})
        LAPS.lap("close look")
        if near is not None:
            LAST_BLOCK["near"] = dict(near)               # the close-up look, for the run file
            near = aim_in_close(w, near, getattr(args, "close_in", CLOSE_IN_MM))
        if args.hover:
            finish_run("hover only", blk, near, cube)
            input("\nLook at the hand over the block. Enter = back home: ")
            go_home(zone, b0, args.speed, check)         # go home here: returning straight away left the wrist
            return                                       # rotated, and the next run then refused to start
        said = grab_from_hover(w, check, zone, cube, near, dq, p, margin, args, blk=blk)
        print("\n" + said)
        if "carry" not in blk.notes:                      # it stopped before the carry: a refusal is worth
            finish_run(said, blk, near, cube)             # recording too
    except Stuck:
        raise
    except BaseException as e:
        print(f"\nStopped: {e or type(e).__name__}. Going back home...")
        try:                                             # keep what this round got (trace, looks): a round
            finish_run(f"stopped: {e or type(e).__name__}", blk, LAST_BLOCK.get("near"), cube)   # stopped by hand
        except Exception:                                #   is usually the interesting one
            pass
        go_home(zone, b0, args.speed, check)
        raise
    # --stay-up applies after a miss too: the block is still about where the close look saw it, so wait where a fixed
    # camera can see that spot. Going home after misses meant a 10-25 s search every next round.
    stay = bool(getattr(args, "stay_up", False))
    seen_at = (blk.notes.get("found_xyz") or blk.notes.get("released_xyz") if "carry" in blk.notes
               else (LAST_BLOCK.get("near") or {}).get("xyz", LAST_BLOCK.get("xyz")))
    park = park_spot(w, check, seen_at) if stay else None
    go_home(zone, b0, args.speed, check, stay_up=stay, park=park)
    LAPS.lap("up high" if stay else "home")
    if "carry" in blk.notes:
        home = home_check(w, cube, blk)
        LAPS.lap("home check")
        n = blk.notes
        ok = n.get("held") is not False and (n.get("carried") is True or home is True)
        line = ("GRABBED" if ok else "NOT confirmed") + (
            f" (squeeze {n.get('squeeze')}, up close "
            f"{'carried' if n.get('carried') else 'not confirmed'}, from home "
            f"{'carried' if home else 'not seen' if home is None else 'not confirmed'})")
        print("RESULT: " + line)
        live.update(result=line.split(" (")[0])
        print("   timing (s): " + LAPS.text())
        finish_run(line, blk, LAST_BLOCK.get("near"), cube)


def finger_spacing(w):
    """Distance between the two finger tags right now (mm), from the gripper camera, or None. Only the distance is
    used, so the arm's slack doesn't matter."""
    from ..config import TAG_MM
    a = arm_angles()
    at = {}
    for c, t, px in sightings_now():
        if c == "gripper" and t in (31, 32):
            got = w.locate_tag(c, px, TAG_MM.get(t, 11.7), a[:4], a[5])
            if got is not None:
                at[t] = got[1]
    return float(np.linalg.norm(at[31] - at[32])) if len(at) == 2 else None


def model_spacing(w, grip):
    return float(np.linalg.norm((w.tags[31]["t"] + w.finger_offset_side("L", grip))
                                - (w.tags[32]["t"] + w.finger_offset_side("R", grip))))


SPACING = {}                      # finger-tag spacings this grab read (mm), logged to the run file: at the bottom
                                  # after closing, and at the hover before and during the squeeze. Logged to see
                                  # whether one reading could replace the squeeze (it can't: held and empty overlap)


def squeeze_check(w):
    """Close a little harder (GRIP_CLOSE -> SQUEEZE_TO) and see whether the fingers move; on the block they stop.
    Returns (held: True/False, or None if the gripper camera can't see both finger tags; closed mm; expected mm)."""
    from .survey import set_hand
    s1 = finger_spacing(w)
    set_hand(grip=SQUEEZE_TO, secs=SQUEEZE_GLIDE_S)
    pause(0.2, "settle")
    wait_fresh(time.time(), cams=("gripper",))           # need a picture of the fingers after they stopped
    s2 = finger_spacing(w)
    set_hand(grip=GRIP_CLOSE, secs=SQUEEZE_GLIDE_S, wait=False)
    expect = model_spacing(w, GRIP_CLOSE) - model_spacing(w, SQUEEZE_TO)
    SPACING.update(hover=s1, squeezed=s2, expect_closed=expect)
    if s1 is None or s2 is None:
        return None, None, expect
    closed = s1 - s2
    return closed < SQUEEZE_HELD * expect, closed, expect


def in_practice_area(w, xyz):
    ax = w.axis[:2]
    d = np.asarray(xyz[:2], float) - ax
    r, b = float(np.linalg.norm(d)), float(np.degrees(np.arctan2(d[1], d[0])))
    return PRACTICE_R[0] <= r <= PRACTICE_R[1] and PRACTICE_BEARING[0] <= b <= PRACTICE_BEARING[1]


def practice_off(w, xyz):
    """How far (mm) a spot is from the middle of the practice area."""
    ax = w.axis[:2]
    r, b = np.mean(PRACTICE_R), np.radians(np.mean(PRACTICE_BEARING))
    return float(np.linalg.norm(np.asarray(xyz[:2], float) - (ax + r * np.array([np.cos(b), np.sin(b)]))))


def carry_ok(w, near, d):
    """Can the block be carried by d? Yes if it stays in the practice area, or gets closer to its middle."""
    new = np.asarray(near["xyz"], float) + d
    return in_practice_area(w, new) or practice_off(w, new) < practice_off(w, near["xyz"])


def pan_shift(w, check, near, p2, margin, grasp_margin, rng):
    """The simplest carry: the same arm poses with the base turned, so the block moves CARRY_MM around the base
    axis. Always reachable with the same way in. (shift xyz, plan) or (None, None)."""
    from ..grasp import BASE_RANGE, hand
    g = p2["grasp"]
    tip0, _ = hand(w, g)
    per_deg = float(np.linalg.norm((hand(w, list(g[:5]) + [g[5] + 1.0])[0] - tip0)[:2]))
    if per_deg < 1.0:
        return None, None
    turn = CARRY_MM / per_deg
    signs = [1.0, -1.0]
    rng.shuffle(signs)
    for sgn in signs:
        path = [list(q[:5]) + [q[5] + sgn * turn] for q in p2["path"]]
        if not BASE_RANGE[0] <= path[0][5] <= BASE_RANGE[1]:
            continue
        d = hand(w, path[-1])[0] - tip0
        d[2] = 0.0
        if not carry_ok(w, near, d):
            continue
        gaps = [check.gaps(q[:5], q[5])[0] for q in path]
        if gaps[0] < margin or min(gaps) < min(grasp_margin, margin):
            continue
        return d, {"path": path, "pan_deg": sgn * turn}
    return None, None


def settable(check, plan, dq, floor):
    """Can this set-down actually be executed? Checks the poses that will be SENT (with the slack taken out), not the
    planned ones. Those read ~5 mm closer to the table, so a set-down that passed on the planned poses could be
    refused halfway down, with the block stuck in the fingers."""
    if check is None or dq is None:
        return True
    for q in plan["path"]:
        c = commanded(q, dq)
        if check.gaps(list(c[:5]), float(c[5]))[0] < floor:
            return False
    return True


def radial_shifts(w, near, want_r):
    """Carry vectors that take the block toward want_r from the base axis: straight in or out first, then angled
    off it. Empty if it's already there."""
    import math
    ax = w.axis[:2]
    v = np.asarray(near["xyz"][:2], float) - ax
    r = float(np.linalg.norm(v))
    if r < 1e-6:
        return []
    u = v / r
    step = float(np.clip(want_r - r, -CARRY_MAX_MM, CARRY_MAX_MM))
    if abs(step) < 2.0:                                     # close enough: the ordinary carry can nudge it
        return []
    # The whole way first, then shorter steps. A big change of distance needs a different hand pitch, and at the
    # same pitch only small moves were possible, so taking the longest step that works walks it there over a few rounds.
    mags = [abs(step)] + [m for m in RADIAL_STEPS_MM if m < abs(step) - 1.0]
    out = []
    for m in mags:
        for ang in (0.0, 12.0, -12.0, 25.0, -25.0):
            c, sn = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            out.append(np.array([u[0] * c - u[1] * sn, u[0] * sn + u[1] * c, 0.0]) * np.sign(step) * m)
    return out


PLACE_PITCH_TOL_DEG = 20.0        # a set-down may tilt the hand this far from the pickup's pitch. Moving the block
                                  # in or out needs it (the reachable pitch changes with distance), and a 40 mm cube
                                  # set down at well under 45 deg rocks back onto its base
PLACE_SHOULDER_MAX = 52.0         # set-downs prefer the shoulder no higher than this...
PLACE_ROOM_MIN = 8.0              # ...and every joint this many steps from its end (see firm())
# Set-downs land further out than aimed, more so further out (probably sag with the arm stretched), so aim inward:
# 7.3 mm + 0.19 mm per mm beyond 300 mm. Fitted on 16 set-downs; see docs/decisions/set-down-aim.md
PLACE_AIM_IN_MM = 7.3
PLACE_AIM_IN_PER_MM = 0.19
PLACE_AIM_IN_PER_DEG = 0.0        # was 0.79 per degree of pitch change; that fit was wrong (it came from tight set-downs)
PLACE_HOME_CLEAR_MM = 40.0        # don't set the block down under the arm's way home: go_home would stop to avoid it,
                                  # but in show that ends the round with an error on camera
PLACE_TRIES_PER_R = 6             # bearings planned in full at one distance before trying a nearer one
HOME_SWEEP = {}                   # set by mode_grasp: the home pose and the base angle this round goes home at
PLACE_OK_MM = 10.0                # found within this of where the fingers let go = it stayed where it was put


PLACE_STEP_MM = 10.0              # set-down distances tried, from the target back toward the block, this far apart
PLACE_PANS_DEG = (22.0, 16.0, 28.0, 19.0, 25.0, 30.0, 35.0, 12.0, 40.0, 45.0, 8.0, 55.0, 70.0)
                                  # base swings tried at each distance: 15-30 deg first (good for the cameras)


def place_spots(w, near, want_r, rng, check=None):
    """show's set-down candidates, best first: [(distance, [(pan, xy), ...]), ...]. Distances step from want_r back
    toward the block's current distance (most progress first), and at each one every base swing in PLACE_PANS_DEG,
    both ways, that stays in the practice area.

    It used to try only want_r, halfway and the same distance, 15-30 deg either side; for a block near the edge of
    the area every one of those was ruled out, and the block never moved."""
    import math
    ax = w.axis[:2]
    v = np.asarray(near["xyz"][:2], float) - ax
    r, b0 = float(np.linalg.norm(v)), math.degrees(math.atan2(v[1], v[0]))
    z0 = float(near["xyz"][2]) - w.table_height(*near["xyz"][:2])    # the top tag's height above the table
    lo, hi = PRACTICE_R
    want = float(np.clip(want_r, lo, hi))
    n = int(abs(want - r) // PLACE_STEP_MM)
    radii = [want + np.sign(r - want) * k * PLACE_STEP_MM for k in range(n + 1)] + [r]
    signs = [1.0, -1.0]
    rng.shuffle(signs)
    out = []
    for rr in dict.fromkeys(round(float(x), 1) for x in radii):
        if not lo <= rr <= hi:
            continue
        spots = []
        for pan in PLACE_PANS_DEG:
            for sg in signs:
                bb = b0 + sg * pan
                if PRACTICE_BEARING[0] <= bb <= PRACTICE_BEARING[1]:
                    xy = ax + rr * np.array([math.cos(math.radians(bb)), math.sin(math.radians(bb))])
                    # only where a fixed camera can find it again (otherwise the next round starts with a search)
                    top = [xy[0], xy[1], z0 + w.table_height(xy[0], xy[1])]
                    if check is None or park_spot(w, check, top, first=True) is not None:
                        spots.append((sg * pan, xy))
        if spots:
            out.append((rr, spots))
    return out


def firm(q):
    """Is this a set-down pose the arm actually holds? Shoulder no higher than PLACE_SHOULDER_MAX and every joint
    PLACE_ROOM_MIN from its limit. Set-downs with the shoulder raised high or a joint at its stop landed 20-61 mm
    further out (the raised arm sags under the block's weight; a joint at its stop doesn't get there)."""
    from ..grasp import limit_room
    return q[0] <= PLACE_SHOULDER_MAX and limit_room(q) >= PLACE_ROOM_MIN


def place_plan(w, check, spot_xy, pan_deg, near, p2, margin, grasp_margin, dq, floor, tight=False):
    """A set-down plan at spot_xy (or None): the hand within PLACE_PITCH_TOL_DEG of the pickup's pitch, any wrist
    turn, the same kind of way in. Only firm() poses, unless tight (and even then a firm one wins)."""
    from ..grasp import GRIP_OPEN as OPEN, SQUARE_WEIGHT, APPROACHES, HOVER_MM, reach, way_in, base_toward
    import math
    near_xyz = np.asarray(near["xyz"], float)
    top = near_xyz[2] + w.table_height(spot_xy[0], spot_xy[1]) - w.table_height(near_xyz[0], near_xyz[1])
    depth = float(p2.get("depth_mm", 10.0))
    target = np.array([spot_xy[0], spot_xy[1], top - depth])
    seed = list(p2["grasp"])
    seed[5] = base_toward(w, target)
    approach = [a for a in APPROACHES if a[0] == p2["approach"]] or list(APPROACHES)
    t = PLACE_PITCH_TOL_DEG
    r_now = float(np.linalg.norm(near_xyz[:2] - w.axis[:2]))
    r_to = float(np.linalg.norm(np.asarray(spot_xy, float) - w.axis[:2]))
    way = 1.0 if r_to < r_now else -1.0                  # moving in needs a steeper hand, out a shallower one
    pitches = [0.0] + [way * k for k in (5.0, 10.0, 15.0, 20.0) if k <= t] + [-way * 5.0]
    mid = 0.5 * (ROLL_RANGE[0] + ROLL_RANGE[1])
    from ..grasp import top_tag
    blind, loose = None, []                              # firm but blind; non-firm ones (only used if tight)
    out_dir = (np.asarray(spot_xy, float) - w.axis[:2]) / max(1e-9, r_to)
    for dpitch in pitches:
        cands = []
        aim = target.copy()                              # where the hand goes so the block lands on the spot
        aim[:2] -= out_dir * max(0.0, PLACE_AIM_IN_MM + PLACE_AIM_IN_PER_MM * (r_to - 300.0)
                                 + PLACE_AIM_IN_PER_DEG * dpitch)
        for k, dface in enumerate((0.0, 45.0, -45.0, 15.0, -15.0, 30.0, -30.0)):   # the block turns with the base
            face = math.radians(p2["face_deg"] + pan_deg + dface)
            g = reach(w, aim, p2["pitch"] + dpitch, face, OPEN, near=seed, ws=SQUARE_WEIGHT)
            if g is not None:
                cands.append((abs(g[3] - mid), face, g))
            if k == 2 and not cands:                     # three wide turns all failed: at this pitch the rest
                break                                    # won't reach either
        # Prefer the wrist rotate mid-range: the rotate at set-down decides how the block's faces end up, and so
        # whether the next pickup can get square to it (the wrist only turns ~60 deg). Two missed pickups came
        # right after set-downs with the rotate pinned at its end.
        # Also prefer a hover the gripper camera can see the spot from, so the check after letting go is a real look.
        for _, face, g in sorted(cands, key=lambda c: c[0]):
            view = top_tag(target, math.degrees(face) - 90.0, depth)
            got = way_in(w, check, aim, g, face, OPEN, HOVER_MM, margin, grasp_margin, approaches=approach,
                         view=view)
            if isinstance(got, dict) and settable(check, got, dq, floor):
                got["spot"] = [float(spot_xy[0]), float(spot_xy[1]), float(top)]
                got["spot_r_mm"] = r_to
                got["pan_deg"] = float(pan_deg)
                got["via_high"] = True
                if not firm(got["grasp"]):
                    loose.append(got)
                    continue
                if got.get("sees"):
                    return got
                blind = blind or got
    if blind is not None:                                # works but can't see: the check afterwards steps aside to look
        return blind
    if tight and loose:
        return max(loose, key=lambda g: bool(g.get("sees")))
    return None


def home_clear(check, xy, top):
    """Clearance (mm) between a block set down at xy and the arm on its way from HIGH_POSE down to home (at the base
    angle this round goes home at), or None if that isn't known yet."""
    if check is None or not HOME_SWEEP:
        return None
    block = {"xyz": [float(xy[0]), float(xy[1]), float(top)]}
    level = (ROLL_SLIDER, TRAVEL_GRIP)
    return min(cube_gap(check, block, list(q) + list(level), HOME_SWEEP["base"])
               for q in arm_path(HIGH_POSE, HOME_SWEEP["home"]))


def reachable_at(w, near, p2, xy, pan_deg=0.0):
    """Can the hand reach a set-down at xy at all? Tries the same pitches and wrist turns place_plan would, but only
    the cheap grasp solve, so place_shift can skip a whole distance without planning each bearing.

    This used to try the pickup's face with no base swing and only 5-10 deg shallower (carrying a block out needs up
    to 20), so it ruled out every distance in rounds where a set-down existed; a third of rounds put the block back
    where it was. Found by replaying the planner on recorded rounds (tools/replay_place.py)."""
    from ..grasp import GRIP_OPEN as OPEN, SQUARE_WEIGHT, reach, base_toward
    import math
    near_xyz = np.asarray(near["xyz"], float)
    target = np.array([xy[0], xy[1], near_xyz[2] - float(p2.get("depth_mm", 10.0))])
    seed = list(p2["grasp"])
    seed[5] = base_toward(w, target)
    for dp in (0.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0):
        if abs(dp) > PLACE_PITCH_TOL_DEG:
            continue
        for k, dface in enumerate((0.0, 45.0, -45.0, 15.0, -15.0, 30.0, -30.0)):    # same as place_plan
            face = math.radians(p2["face_deg"] + pan_deg + dface)
            if reach(w, target, p2["pitch"] + dp, face, OPEN, near=seed, ws=SQUARE_WEIGHT) is not None:
                return True
            if k == 2:
                break
    return False


PICKABLE_SPREAD_DEG = 6.0         # the next pickup must reach the block this far either side of the predicted turn


def turn_after_set_down(plan):
    """The block's turn (yaw, -45..45) after this set-down: square to the fingers. Matched the next round's measured
    turn within ~5 deg in 10 of 14 set-downs (the release twisted it in the others)."""
    return ((float(plan["face_deg"]) + 45.0) % 90.0) - 45.0


def pickable_after(w, near, p2, plan):
    """Will the next pickup be able to square the fingers to the block where (and how turned) this set-down leaves
    it? A block left turned 35-40 deg to the arm's bearing once couldn't be grasped from any hover (the wrist
    rotate can't square to either face there) and the round was lost. Only checks the grasp's reach (cheap)."""
    from ..grasp import reach_all
    import math
    yaw = turn_after_set_down(plan)
    depth = float(p2.get("depth_mm", 10.0))
    target = np.asarray(plan["spot"], float) - [0.0, 0.0, depth]
    from ..grasp import SQUARE_TOL_LOOSE_DEG, square_tol
    if straight_pickup_near(plan["spot"], yaw) is False:
        return False
    with square_tol(SQUARE_TOL_LOOSE_DEG):              # the pickup may meet it this far off square if it must
        return all(any(reach_all(w, target, math.radians(yaw + d + f), p2["grip"]) for f in (0.0, 90.0))
                   for d in (-PICKABLE_SPREAD_DEG, 0.0, PICKABLE_SPREAD_DEG))


PICKABLE_TABLE_MM = 20.0          # a set-down needs a plan-table entry this close to it (one grid step)...
PICKABLE_TABLE_DEG = 10.0         # ...at a block turn this close, with a straight way in


def straight_pickup_near(xy, yaw):
    """Does the plan table have a straight way in (not the joint-space one) for a block left at xy, turned yaw?
    None without a table. The joint-space way in held the block 0 times in 9, so don't leave the block where that's
    the only option (a third of the table's entries inside ~280 mm)."""
    t = plan_table()
    if not t:
        return None
    if "_straight" not in t:
        keep = [e for e in t["plans"] if e["plan"].get("approach") not in _g.JOINT_APPROACHES]
        t["_straight"] = (np.array([e["xy"] for e in keep], float).reshape(-1, 2),
                          np.array([e["yaw"] for e in keep], float))
    P, Y = t["_straight"]
    if not len(P):
        return False
    near = np.hypot(*(P - np.asarray(xy, float)[:2]).T) <= PICKABLE_TABLE_MM
    dyaw = np.abs((Y - float(yaw) + 45.0) % 90.0 - 45.0)
    return bool(np.any(near & (dyaw <= PICKABLE_TABLE_DEG)))


def place_shift(w, check, near, p2, margin, grasp_margin, rng, dq=None, floor=0.0):
    """_place_shift with firm set-downs only first, then allowing non-firm ones (see firm())."""
    for tight in (False, True):
        got = _place_shift(w, check, near, p2, margin, grasp_margin, rng, dq, floor, tight)
        if got[0] is not None:
            return got
    return None, None


def _place_shift(w, check, near, p2, margin, grasp_margin, rng, dq=None, floor=0.0, tight=False):
    """show's carry: (shift xyz, plan) to the set-down that gets nearest CARRY_TO_R, or (None, None). For each
    distance: a cheap reach check, then the bearings (skipping spots under the way home), at most PLACE_TRIES_PER_R
    planned in full."""
    top0 = float(np.asarray(near["xyz"], float)[2])
    for rr, spots in place_spots(w, near, CARRY_TO_R, rng, check=check):
        spots = [(pan, xy) for pan, xy in spots
                 if (lambda g: g is None or g >= PLACE_HOME_CLEAR_MM)(home_clear(check, xy, top0))]
        if not spots or not reachable_at(w, near, p2, spots[0][1], spots[0][0]):
            continue
        for pan, xy in spots[:PLACE_TRIES_PER_R]:
            plan = place_plan(w, check, xy, pan, near, p2, margin, grasp_margin, dq, floor, tight=tight)
            if plan is not None and not pickable_after(w, near, p2, plan):
                continue                                  # it would leave the block where nothing can grab it
            if plan is not None:
                d = np.array(plan["spot"]) - np.asarray(near["xyz"], float)
                d[2] = 0.0
                return d, plan
    return None, None


def shift_plan(w, check, p2, d, margin, grasp_margin, dq, floor):
    """The set-down plan for carrying the block by d, or None if it can't be reached or executed."""
    from ..grasp import GRIP_OPEN as OPEN, SQUARE_WEIGHT, APPROACHES, HOVER_MM, reach, way_in
    import math
    face = math.radians(p2["face_deg"])
    approach = [a for a in APPROACHES if a[0] == p2["approach"]] or list(APPROACHES)
    t2 = np.asarray(p2["target"], float) + d
    g2 = reach(w, t2, p2["pitch"], face, OPEN, near=p2["grasp"], ws=SQUARE_WEIGHT)
    if g2 is None:
        return None
    got = way_in(w, check, t2, g2, face, OPEN, HOVER_MM, margin, grasp_margin, approaches=approach)
    return got if isinstance(got, dict) and settable(check, got, dq, floor) else None


def pick_shift(w, check, near, p2, margin, grasp_margin, rng, dq=None, floor=0.0):
    """A spot CARRY_MM from the block (inside the practice area, or toward its middle) that can be set down the same
    way it was picked up: (shift xyz, plan) or (None, None)."""
    import math
    if CARRY_TO_R is not None:                              # show: move it to a new distance, not just a new spot
        for d in radial_shifts(w, near, CARRY_TO_R):        # (pan_shift can't: it keeps the distance)
            if not carry_ok(w, near, d):
                continue
            plan = shift_plan(w, check, p2, d, margin, grasp_margin, dq, floor)
            if plan is not None:
                return d, plan
    got = pan_shift(w, check, near, p2, margin, grasp_margin, rng)
    if got[0] is not None and settable(check, got[1], dq, floor):
        return got
    angles = list(np.arange(0.0, 360.0, 45.0))
    rng.shuffle(angles)
    for ang in angles:
        d = CARRY_MM * np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang)), 0.0])
        if not carry_ok(w, near, d):
            continue
        plan = shift_plan(w, check, p2, d, margin, grasp_margin, dq, floor)
        if plan is not None:
            return d, plan
    return None, None


# Merging stops: a planned path is ~10 poses 5 mm apart, and moving to each one separately (send, wait, next)
# was most of the slow part of a round. A stop can be skipped if one straight joint move past it keeps the
# fingertip within MERGE_MM of the planned path and no closer to the table than the plan.
MERGE_MM = 1.0
MERGE_SAMPLES = 3                 # check at least this many points per skipped step...
MERGE_STEP_DEG = 0.5              # ...and at least every this much of the biggest joint's travel
MERGE_KEEP_LAST = 1               # the last hop into a set-down or a grasp stays a separate move


def _off_polyline(p, pts):
    """Distance (mm) from point p to the polyline through pts."""
    best = float("inf")
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def merged_stops(w, check, true_path, start, end, keep_last=0):
    """_merged_stops, traced."""
    with trace.span("compute", "merge stops"):
        return _merged_stops(w, check, true_path, start, end, keep_last)


def _merged_stops(w, check, true_path, start, end, keep_last=0):
    """The indices to stop at going along true_path from start to end (either direction; end included, start not),
    with every stop dropped that one longer straight move can safely skip (see MERGE_MM). Typically ~3 stops instead
    of ~10. keep_last: this many hops at the end stay separate."""
    step = 1 if end > start else -1
    idx = list(range(start + step, end + step, step))
    if w is None or check is None or len(idx) <= 1:
        return idx
    from ..grasp import hand
    path = [np.asarray(q, float) for q in true_path]
    tips = {}

    def tip(i):
        if i not in tips:
            tips[i] = hand(w, list(path[i]))[0]
        return tips[i]
    gap_at = {}

    def gap(i):
        if i not in gap_at:
            gap_at[i] = check.gaps(list(path[i][:5]), float(path[i][5]))[0]
        return gap_at[i]

    def ok(i, j):                                    # one straight move from i to j, past everything between
        lo, hi = min(i, j), max(i, j)
        line = [tip(k) for k in range(lo, hi + 1)]
        floor = min(gap(k) for k in range(lo, hi + 1)) - 0.5
        # A straight joint move bulges away from the planned line in the middle; 3 samples per hop missed it once
        # (7.7 mm off and 2.3 mm nearer the table). So sample the fingertip finely (cheap), and run the expensive
        # table check at two points per original hop: with the fingertip held to the plan the clearance changes
        # smoothly in between.
        span = float(np.abs(path[j][:4] - path[i][:4]).max())
        n = max(MERGE_SAMPLES * (hi - lo), int(np.ceil(span / MERGE_STEP_DEG)), 8)
        qs = [path[i] + (path[j] - path[i]) * (m / n) for m in range(1, n)]
        if any(_off_polyline(hand(w, list(q))[0], line) > MERGE_MM for q in qs):
            return False
        for m in range(1, 2 * (hi - lo)):
            q = path[i] + (path[j] - path[i]) * (m / (2.0 * (hi - lo)))
            if check.gaps(list(q[:5]), float(q[5]))[0] < floor:
                return False
        return True
    out, i = [], start
    last_free = idx[-1 - keep_last] if keep_last and len(idx) > keep_last else idx[-1]
    while i != end:
        if (last_free - i) * step <= 0:              # into the kept hops: one at a time
            j = i + step
        else:                                        # binary search for the farthest stop one move can reach
            good, lo_k, hi_k = i + step, 2, (last_free - i) * step
            while lo_k <= hi_k:
                mid = (lo_k + hi_k) // 2
                if ok(i, i + step * mid):
                    good, lo_k = i + step * mid, mid + 1
                else:
                    hi_k = mid - 1
            j = good
        out.append(j)
        i = j
    return out


def wait_fresh(after, secs=1.2, cams=None):
    """_wait_fresh, traced."""
    with trace.span("wait", "fresh frame"):
        return _wait_fresh(after, secs, cams)


def _wait_fresh(after, secs=1.2, cams=None):
    """Wait (at most secs) until every camera, or just `cams`, has a frame captured after `after` (Pi clock).
    Better than a fixed sleep: frames are 0.1-0.4 s old by the time they're read, depending on the camera."""
    from ..hardware import http_json
    t_end = time.time() + secs
    while time.time() < t_end:
        try:                                         # short timeouts so a slow server can't hold us past secs
            got = http_json(S.cam_url + "/tags", timeout=max(0.1, min(0.5, t_end - time.time())), tries=1)
            if all(not isinstance(c, dict) or c.get("t_cap", 0.0) > after for n, c in got.items()
                   if cams is None or n in cams):
                return
        except Exception:
            return
        time.sleep(0.05)


class CarryPlan:
    """Run a function in a background thread; get() waits for the answer (and re-raises its error).

    Used to plan the set-down while the hand lifts the block and does the squeeze check (everything the plan needs
    is known before the lift), which took 2-12 s a round with nothing moving."""

    def __init__(self, fn):
        import threading
        self.out, self.err = None, None

        def run():
            try:
                self.out = fn()
            except Exception as e:
                self.err = e
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()

    def get(self):
        self.t.join()
        if self.err is not None:
            raise self.err
        return self.out


CLOSE_SETTLE_S = 0.25             # pause after the gripper closes, before lifting (set_grip already waited for it)
RELEASE_LIFT_SPEED = CARRY_SPEED  # after letting go nothing is held, so lift away at carry speed
RELEASE_LIFT_MM = 20.0            # after letting go, lift this far with the hand barely open before opening it wide
LIFT_STEP_MM = 5.0                # the straight-up lift after a joint-space set-down, in steps this long
LIFT_SIDEWAYS_MM = 1.0            # ...that may wander at most this far off vertical


def release_lift(w, check, plan2, dq, floor):
    """The commanded poses for lifting straight up after a joint-space set-down (vertical_lift), or None to fall
    back to backing out along the way in. Every pose keeps `floor` from the table, and the move from the top to the
    set-down's hover (hand wide open) must pass check_arm_move as "leaving". Checked before the set-down starts."""
    if check is None:
        return None
    lift = None
    for mm in (RELEASE_LIFT_MM, float(plan2.get("depth_mm", DEPTH_MM)) + 3.0):   # else just clear of the block's top
        lift = vertical_lift(w, plan2["grasp"], RELEASE_GRIP, mm)
        if lift is not None:
            break
    if lift is None:
        return None
    cmd = [commanded(q, dq) for q in lift]
    if any(check.gaps(list(c[:3]) + [c[3], RELEASE_GRIP], float(c[5]))[0] < floor for c in cmd):
        return None
    hover = commanded(plan2["path"][0], dq)
    block = {"xyz": np.asarray(plan2["spot"], float)}
    try:
        check_arm_move(check, block, cmd[-1][:3], hover[:3], (cmd[-1][3], GRIP_OPEN), float(hover[5]), leaving=True,
                       what="the move up from the set-down")
    except ArmError:
        return None
    return cmd


def vertical_lift(w, q0, grip, mm=RELEASE_LIFT_MM):
    """True poses that lift the fingertip straight up from q0, in LIFT_STEP_MM steps up to mm, using only shoulder,
    elbow and wrist; or None.

    Backing out along the joint-space arc after a set-down swung the fingertips 14-22 mm outward while they were
    still beside the block, and the block went with them (found 14-60 mm out). Straight-in set-downs that lift
    vertically landed within ~8 mm."""
    from ..grasp import straight_up
    return straight_up(w, q0, grip, mm, step=LIFT_STEP_MM, sideways=LIFT_SIDEWAYS_MM)
STEP_MM_PATH = 5.0                # spacing of the planned paths (grasp.STEP_MM)


def _choose_carry(w, check, near, p2, margin, grasp_margin, rng, dq, floor):
    """(shift, set-down plan): show's set-down if there is one, else the small nudge, else (None, None)."""
    d, plan2 = (place_shift(w, check, near, p2, margin, grasp_margin, rng, dq=dq, floor=floor)
                if CARRY_TO_R is not None else (None, None))
    if d is None:
        d, plan2 = pick_shift(w, check, near, p2, margin, grasp_margin, rng, dq=dq, floor=floor)
    return d, plan2


def let_go_point(w, dq, depth):
    """Where the block's top centre is as the fingers open: the fingertip point (from the real joints, sliders plus
    slack) raised by the grasp depth. None if it can't be worked out.

    Uses the OPEN fingers' fingertip point, the one the grab and set-down are planned with. Using the closed
    fingers' point (21.6 mm lower and 3-4 mm across in the finger model) made every logged release ~20 mm low."""
    try:
        a = arm_angles()
        tip = w.tip(list(a[:4]), float(a[5]), grip=GRIP_OPEN, dq=dq)
        return np.array([tip[0], tip[1], tip[2] + depth], float)
    except Exception:
        return None


def refind(w):
    """The block from every camera at the hover, with the arm's slack measured again (no printing)."""
    a, off, fit, _, seen = find_cube(w)
    dq, _, _ = arm_slack(w, a, off, seen)
    return locate(w, seen, a[:4], a[5], base_offset=off, grip=a[4], dq=dq)


BASE_FIX_DEG = 0.5                # if the re-plan wants the base this far from where it is, turn it before going in


def turn_base_for(w, check, near, p2):
    """Turn the base to the re-plan's angle before going in.

    The re-plan after the close look solves the base angle too, but the way in only moves shoulder, elbow and
    wrist, so that correction was being dropped: replayed on 98 grabs it left the fingers a median 0.8 mm off
    sideways, over 3 mm in 16 of them (up to 19). Closed loop, and checked against the block first."""
    from ..config import BASE_LIMIT_DEG, HOME_FINISH_DEG
    want = float(p2["grasp"][5])
    now = float(S.base_cmd if S.base_cmd is not None else arm_angles()[5])
    if LAST_LOOK.get("fit", 99.0) < 10.0 and time.time() - LAST_LOOK.get("t", 0.0) < 60.0:
        now = LAST_LOOK["base"] + LAST_LOOK["off"]           # where the close look measured it really is
    if check is None or abs(want - now) < BASE_FIX_DEG or abs(want) > BASE_LIMIT_DEG:
        return
    q5 = list(arm_angles()[:5])
    sweep = np.linspace(min(now, want - HOME_FINISH_DEG), max(now, want), 12)
    floor = min(HOVER_CLEAR_MM, cube_gap(check, near, q5, now) - 2.0)
    if min(cube_gap(check, near, q5, b) for b in sweep) < floor:
        print(f"   (the re-plan wants the base {want - now:+.2f} deg, but the turn passes too close to the block: "
              f"going in without it)")
        return
    print(f"   turning the base {want - now:+.2f} deg for the re-planned grasp")
    move_base(want)


LIFT_SLOW_MM = 20.0               # holding the block, lift the first 20 mm slowly (beside the block), the rest faster


def lift_clear_index(w, path):
    """The path index (hover = 0) where the fingertip is first LIFT_SLOW_MM above the grasp, coming out; 0 if never."""
    from ..grasp import hand
    if w is None:
        return 0
    z0 = hand(w, path[-1])[0][2]
    for k in range(len(path) - 1, -1, -1):
        if hand(w, path[k])[0][2] - z0 >= LIFT_SLOW_MM:
            return k
    return 0


def grab_from_hover(w, check, zone, cube, near, dq, p, margin, args, rng=None, blk=None):
    """The grab itself, starting at the hover: re-plan from the close look, go in slowly, close, lift back to the
    hover, squeeze check, carry and set it down, back out and check where it landed. Returns a one-line result; the
    arm ends at a hover with the hand open. If anything stops it partway, it backs out the way it came (or raises
    Stuck if it can't)."""
    import random
    from .survey import set_grip, set_hand
    rng = rng or random.Random()
    if near is None:
        return "NOT grabbed: no camera saw the block from the hover"
    moved = float(np.linalg.norm(near["xyz"][:2] - np.asarray(cube["xyz"])[:2]))
    if moved > MOVED_MM:
        return f"NOT grabbed: the block looks {moved:.0f} mm from where it was (moved, or a misread): try again"
    p2 = replan(w, check, near, p, margin, grasp_margin=args.grasp_margin)
    LAPS.lap("re-plan")
    if "why" in p2:
        return "NOT grabbed: re-planned from up close, no plan (" + p2["why"][0] + ")"
    turn_base_for(w, check, near, p2)
    path = [commanded(q, dq) for q in p2["path"]]            # true poses -> slider values (slack taken out)
    path[0] = list(arm_angles()[:3]) + list(path[0][3:])     # start from where the arm really is
    print("   re-planned grasp: " + joints_text(p2["grasp"]) + f"\n   commanded with the slack out: {joints_text(path[-1])}")
    rot = path[0][3]
    gm = min(args.grasp_margin, margin)
    saved = (zone.table_margin, zone.table_extra, zone.hand)

    def allow(true_path, cmd_path):
        """(slip, shut) mm of extra table clearance to allow the move checks. slip: the planner checked the real
        poses but the move checks see the slider values. shut: holding the block, the checks see the hand closed to
        GRIP_CLOSE (fingertips ~2 mm lower than they really are, since they stop at the block's sides)."""
        if check is None or getattr(zone, "exact", None) is None:
            return 0.0, 0.0
        hand_was = zone.hand
        pts = np.array([c[:3] for c in cmd_path], float)
        zone.hand = (rot, GRIP_OPEN)
        zg = zone._lookup(pts, "base")
        zone.hand = (rot, GRIP_CLOSE)
        zc = zone._lookup(pts, "base")
        zone.hand = hand_was
        sl = max([0.0] + [check.gaps(t[:5], t[5])[0] - g for t, g in zip(true_path, zg)])
        return min(sl + 0.5, SLACK_TABLE_MM), min(max([0.0] + list(zg - zc)), SLACK_TABLE_MM)

    def margins(extra):                                      # the slow moves near the table use the grasp margin
        extra += SLOW_TOL_MM
        zone.table_margin, zone.table_extra = gm - extra, zone.margin - gm + extra
    slip, shut = allow(p2["path"], path)
    if slip > 1.0:
        print(f"   (the slider values read {slip:.1f} mm closer to the table than the real poses, which keep "
              f"{gm:.0f}: the move checks allow that)")
    margins(slip)
    trail = [path[0]]                                        # poses since the last hover, to back out along

    def go(poses):                                           # going in: each pose goes on the trail
        for q in poses:
            move_to(q[:3], zone, PICK_SLOW)
            trail.append(q)

    def come_out(poses, speed=None):                         # coming out: each pose comes off the trail, so a stop
        for q in poses:                                      # partway out backs out further, never back in
            move_to(q[:3], zone, speed or PICK_SLOW)
            if len(trail) > 1:
                trail.pop()

    def at_hover(q):
        trail[:] = [q]
    held, carried, note = None, None, ""
    try:
        zone.hand = (rot, GRIP_OPEN)                         # the checks see the hand as it is: turned and open
        set_hand(rot, GRIP_OPEN)
        print("   going in (slowly)...")
        ins = merged_stops(w, check, p2["path"], 0, len(path) - 1, keep_last=MERGE_KEEP_LAST)
        if getattr(S, "hand_rel", False):
            go_watching([path[k] for k in ins], go, w, check, zone, near, p2, margin, args, rot)
        else:
            go([path[k] for k in ins])
        LAPS.lap("in")
        print(f"   closing to {GRIP_CLOSE:.0f}...")
        zone.hand = (rot, GRIP_CLOSE)
        set_grip(GRIP_CLOSE)
        pause(CLOSE_SETTLE_S, "settle")
        SPACING.clear()
        try:                                                 # logged only, nothing depends on it
            SPACING["bottom"] = finger_spacing(w)
        except Exception:
            pass
        LAPS.lap("close")
        floor = max(0.0, gm - shut - SLOW_TOL_MM)
        carry_job = CarryPlan(lambda: _choose_carry(w, check, near, p2, margin, args.grasp_margin, rng, dq, floor))
        print("   lifting it out to the hover...")
        margins(slip + shut)
        n = len(path) - 1
        k_up = lift_clear_index(w, p2["path"])               # slow beside the block, faster once clear of it
        come_out([path[k] for k in merged_stops(w, check, p2["path"], n, k_up)])
        if k_up > 0:
            come_out([path[k] for k in merged_stops(w, check, p2["path"], k_up, 0)], speed=CARRY_SPEED)
        at_hover(path[0])
        LAPS.lap("lift")
        held, closed, expect = squeeze_check(w)
        live.update(grip={"held": held, "closed_mm": None if closed is None else float(closed),
                          "empty_mm": None if expect is None else float(expect),
                          "spacing": {k: v for k, v in SPACING.items() if isinstance(v, (int, float))}})
        live.event("the squeeze says " + ("it's holding the block" if held else "the fingers are empty" if held is False
                                          else "it can't see both fingers"))
        print(f"   squeeze check ({GRIP_CLOSE:.0f} -> {SQUEEZE_TO:.0f}): " + (
            f"the fingers closed {closed:.1f} mm (empty would be {expect:.1f}) -> " + ("HOLDING it" if held else "nothing in them")
            if held is not None else "the gripper camera cannot see both finger tags"))
        # A second opinion from the cameras: the squeeze reads the finger tags, which the gripper camera often can't
        # see at the hover, so also check whether the block itself rose.
        rise, rtag, rcam = lift_seen(w, dq)
        if rise is None:
            lift = None
            print("   lift check: no camera sees the block's tag from the hover")
        else:
            lift = bool(rise >= LIFTED_MM)
            print(f"   lift check: tag {rtag} from {rcam} is {rise:.0f} mm above where it rests -> "
                  + ("UP IN THE HAND" if lift else "still on the table"))
        if lift and held is None:                            # couldn't read the fingers, but the block is in the air
            held = True
        if blk is not None:
            blk.notes.update(lift_mm=None if rise is None else float(rise), lifted=lift)
        if held is False:
            set_grip(GRIP_OPEN)
            zone.hand = (rot, GRIP_OPEN)
            return ("NOT grabbed: the squeeze check says the fingers are empty"
                    + ("" if not lift else f" -- though {rcam} sees the block {rise:.0f} mm up, so look at it"))
        LAPS.lap("squeeze")
        try:
            with trace.span("compute", "carry plan (finishing)"):
                d, plan2 = carry_job.get()                   # planned during the lift; usually ready already
        except Exception as e:
            print(f"   (the background set-down plan failed: {e}; planning it now)")
            d, plan2 = _choose_carry(w, check, near, p2, margin, args.grasp_margin, rng, dq, floor)
        if d is None:
            print("   no spot 10 mm away to set it down in the practice area: setting it back where it was")
            d, plan2 = np.zeros(3), p2
            if not settable(check, plan2, dq, max(0.0, gm - shut - SLOW_TOL_MM)):
                print("   (and the way back down is tight at the table once the hand is shut: it may refuse "
                      "part way. If it does, the block is in the fingers -- open the gripper by hand.)")
        LAPS.lap("carry plan")
        if plan2.get("via_high"):
            live.update(place={"spot_r_mm": plan2.get("spot_r_mm"), "pan_deg": plan2.get("pan_deg"),
                               "pitch": plan2.get("pitch"), "spot": polar(w, plan2.get("spot"))})
        if blk is not None:
            blk.notes["carry_plan"] = {k: plan2.get(k) for k in ("spot", "spot_r_mm", "pan_deg", "pitch", "face_deg",
                                                                 "approach") if plan2.get(k) is not None}
        path2 = [commanded(q, dq) for q in plan2["path"]]
        lift2 = release_lift(w, check, plan2, dq, floor) if plan2.get("approach") in _g.JOINT_APPROACHES else None
        if plan2.get("via_high"):
            # show's carry: up holding the block, swing the base up there, then across and down. Turning the base at
            # the hover (arm stretched out, holding the block) lost small turns and the set-downs landed 10-40 mm off.
            print(f"   carrying it to {plan2.get('spot_r_mm', 0.0):.0f} mm out, {plan2['pan_deg']:+.0f} deg round "
                  f"(hand {plan2['pitch']:.0f} deg down): up, turn, across, down...")
            zone.table_margin, zone.table_extra = saved[0], saved[1]       # moves in free space: full margins
            pan = None
            if PAN_OVERLAP:                                  # the block is in the hand, so only table/tower/arm
                b_now = float(S.base_cmd if S.base_cmd is not None else arm_angles()[5])   # checks; the wrist
                turning = [(r, GRIP_CLOSE) for r in np.linspace(rot, path2[0][3], 3)]      # turns on the way
                pan = pan_check_later(zone, check, None, HIGH_POSE, path2[0][:3], turning, b_now, path2[0][5],
                                      what="the carry across to the set-down")
            move_to(HIGH_POSE, zone, CARRY_SPEED)
            rot = path2[0][3]
            zone.hand = (rot, GRIP_CLOSE)
            set_hand(rot, GRIP_CLOSE, wait=False)            # turn the wrist for the set-down while the base swings
            # Open-loop base: the set-down only needs to land about the right distance out, and the check after
            # letting go measures where it really ended up. Closed loop added 1.5-3 s to each carry.
            if pan_ok(pan):
                pan_move(path2[0][:3], path2[0][5], zone, CARRY_SPEED, closed=False, settle=False, slow_finish=True)
                set_hand(rot, GRIP_CLOSE)
            else:
                move_base(path2[0][5], closed=False, settle=False)   # an arm move follows, not a look
                set_hand(rot, GRIP_CLOSE)
                move_to(path2[0][:3], zone, CARRY_SPEED, slow_finish=True)
        else:
            how = f" by turning the base {plan2['pan_deg']:+.1f} deg" if "pan_deg" in plan2 else ""
            print(f"   carrying it {np.linalg.norm(d):.0f} mm (x {d[0]:+.1f}, y {d[1]:+.1f}){how} and setting it down...")
            if abs(path2[0][5] - path[0][5]) > 0.05:
                move_base(path2[0][5])
            move_to(path2[0][:3], zone, PICK_SLOW)
        LAPS.lap("carry")
        slip2, shut2 = allow(plan2["path"], path2)
        at_hover(path2[0])
        margins(slip2 + shut2)                               # going down holding it: the hand reads shut
        go([path2[k] for k in merged_stops(w, check, plan2["path"], 0, len(path2) - 1, keep_last=MERGE_KEEP_LAST)])
        LAPS.lap("down")
        released = let_go_point(w, dq, float(p2.get("depth_mm", DEPTH_MM)))
        # Let go gently: open just enough to release (RELEASE_GRIP), lift clear, and only then open wide. Opening
        # straight to wide swings the fingers out on their arcs while still touching the block; on film that turned
        # it ~10 deg every set-down and left it 10-40 mm from where it was put.
        set_grip(RELEASE_GRIP)
        zone.hand = (rot, RELEASE_GRIP)
        pause(0.25, "after release")
        margins(slip2)
        n2 = len(path2) - 1
        lift = lift2 if plan2.get("approach") in _g.JOINT_APPROACHES else None
        if lift is not None:                                 # joint-space set-down: straight up, not back along the
            for q in lift:                                   # arc (which dragged the block outward)
                move_to(q[:3], zone, RELEASE_LIFT_SPEED)
            trail[:] = [path2[0], lift[-1]]                  # backing out from here goes to the hover, not down
            zone.hand = (rot, GRIP_OPEN)
            set_hand(grip=GRIP_OPEN, wait=False)             # fingertips are above the block now: open on the way up
            move_to(path2[0][:3], zone, CARRY_SPEED)
        else:
            clear = max(0, n2 - int(np.ceil(RELEASE_LIFT_MM / STEP_MM_PATH)))
            come_out([path2[k] for k in merged_stops(w, check, plan2["path"], n2, clear)], speed=RELEASE_LIFT_SPEED)
            zone.hand = (rot, GRIP_OPEN)
            set_hand(grip=GRIP_OPEN, wait=False)
            if clear > 0:                                    # clear of the block: the rest of the way up is free space
                come_out([path2[k] for k in merged_stops(w, check, plan2["path"], clear, 0)], speed=CARRY_SPEED)
        set_hand(grip=GRIP_OPEN)                             # make sure it got there (instant if it did)
        at_hover(path2[0])
        wait_fresh(time.time() + 0.15, cams=HOVER_FRESH_CAMS if getattr(S, "hand_rel", False) else None)
        LAPS.lap("let go")
        if blk is not None:
            blk.moved(d)
            if released is not None:
                blk.notes["released_xyz"] = [round(float(v), 1) for v in released]
            blk.notes.update(carry=np.asarray(d, float), held=held,
                             squeeze="held" if held else "not seen" if held is None else "empty")
        found = refind(w)
        if found is None and released is not None and plan2.get("via_high"):
            print("   not in view from the hover: looking from a pose aimed at where it was let go...")
            try:
                found = look_from_near(w, check, zone, path2[0], {"xyz": released,
                                       "yaw_deg": float(plan2.get("face_deg", 0.0)) - 90.0}, dq, margin)
            except ArmError as e:
                print(f"   (could not look again: {e})")
        LAPS.lap("check")
        if found is not None and blk is not None:
            blk.notes["found_xyz"] = [round(float(v), 1) for v in found["xyz"]]
        if found is not None:
            LAST_BLOCK.update(found)
        else:                                                # best guess: where the fingers let go of it
            LAST_BLOCK.update({"xyz": released if released is not None else np.asarray(near["xyz"], float) + d})
        if found is None:
            note = "could not see the block afterwards"
        else:
            want = np.asarray(near["xyz"], float) + d
            off_want = float(np.linalg.norm(found["xyz"][:2] - want[:2]))
            off_start = float(np.linalg.norm(found["xyz"][:2] - np.asarray(near["xyz"])[:2]))
            carried, placed, note = carry_verdict(off_start, off_want, d)
            if released is not None:                         # did it stay where the fingers let go of it?
                let_go = float(np.linalg.norm(released[:2] - want[:2]))
                after = float(np.linalg.norm(found["xyz"][:2] - released[:2]))
                note += (f"; let go {let_go:.0f} mm from the aim, found {after:.1f} mm from there"
                         + (" -> STAYED PUT" if after <= PLACE_OK_MM else " -> MOVED after it was let go"))
                if blk is not None:
                    blk.notes.update(let_go_off_mm=round(let_go, 1), moved_after_release_mm=round(after, 1))
                place = dict(live.snapshot().get("place") or {})
                place.update(let_go_off_mm=let_go, moved_after_mm=after, let_go=polar(w, released),
                             found=polar(w, found["xyz"]),
                             off_spot_mm=None if plan2.get("spot") is None else
                             float(np.linalg.norm(found["xyz"][:2] - np.asarray(plan2["spot"], float)[:2])))
                live.update(place=place)
                live.event(f"landed {after:.0f} mm from where it was let go")
            if blk is not None:
                blk.notes["carried"] = carried
                blk.notes["placed_off_mm"] = round(off_want, 1)
                blk.see(found)
            print("   " + note)
    finally:
        try:                                                 # whatever happened: back out the way it came
            if len(trail) > 1:
                set_grip(GRIP_OPEN)
                zone.hand = (rot, GRIP_OPEN)
                for q in trail[-2::-1]:
                    move_to(q[:3], zone, PICK_SLOW)
        except Exception as e:
            raise Stuck(f"Could not back out of the way in ({e}). The arm is left where it is: open the gripper and "
                        "drive it up and away from the block with the controller.") from e
        finally:
            zone.table_margin, zone.table_extra, zone.hand = saved
    ok = held is not False and carried is True
    return ("GRABBED" if ok else "NOT confirmed") + f": squeeze {'held' if held else 'not seen' if held is None else 'empty'}, {note}"
