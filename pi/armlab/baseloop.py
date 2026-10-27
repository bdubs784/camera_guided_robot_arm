"""
Closed-loop base moves. The worm gear's play (about 0.85 deg on average between arriving going + and going -, up to
1.8) and where it settles (+-0.6 deg from stop to stop, in backlash tests) leave the base up to ~1 deg from its
count. The tower and webcam turn with the base and see the table tags, so with a world.json they measure the real
angle to about 0.1 deg: after the move, measure, nudge by what is left (arriving going +), measure again.

move_base does this when S.base_closed is on (grasp, show; --open-base turns it off). The count then
differs from the target by the nudges; S.base_cmd keeps the target (the angle the program asked for, in world terms)
and S.base_fix the difference (target minus count).
"""
import os
import time

import numpy as np

from .state import S

LOOP_TOL_DEG = 0.3                      # done when the base is this close to the target. At 0.1 the loop used all
                                        # 5 nudges in most moves (12-32 s): the base only takes part of each nudge,
                                        # so the error creeps instead of closing. 0.3 deg is ~1.5 mm at 300 mm out,
                                        # and the close look at the hover corrects the rest.
LOOP_TRIES = 5                          # nudges at most; 3 left basehome 0.43 deg off
NUDGE_RUNUP_DEG = 3.0                   # every nudge backs off this far and comes forward to the target in one move.
                                        # A +1.25 deg nudge from rest did not turn the base at all: the worm winds up
                                        # small moves from rest without turning. Already moving when it arrives, it
                                        # gets there. (HOME_FINISH_DEG, the run-up for ordinary moves, is 1.5.)
FIT_PX = 10.0                           # a table-tag fit worse than this (arm in the way, a misread) is not used
MAX_DEG = 2.5                           # more than this off means something else is wrong: not nudged
READS = 3

_cache = {}


def world():
    """The world model from ~/arm_data/world.json (reloaded if the file changes), or None."""
    p = os.path.join(S.data_dir, "world.json")
    if not os.path.exists(p):
        return None
    m = os.path.getmtime(p)
    if _cache.get("mtime") != m:
        from .world import World
        _cache.update(mtime=m, w=World(p))
    return _cache["w"]


def default_read():
    from .hardware import arm_angles
    from .modes.where import sightings_now
    return arm_angles(), sightings_now()


def world_offset(w, read, reads=READS, pause=0.3, base=None, tilt=False):
    """_world_offset, traced (trace.py: the closed loop's camera checks)."""
    from . import trace
    with trace.span("camera", "base check"):
        return _world_offset(w, read, reads, pause, base, tilt)


def _world_offset(w, read, reads=READS, pause=0.3, base=None, tilt=False):
    """How far the base is from the world's angle for its count (deg, real minus count), from the table tags the
    base-mounted cameras see: the median of several readings. read() -> (arm_angles(), sightings_now()). base: the
    count to measure against (default the reported one, which is rounded to 0.1). Returns (offset or None, readings)."""
    offs, tilts = [], []
    for k in range(reads):
        a, seen = read()
        off, fit = w.fit_base_offset(seen, a[:4], a[5] if base is None else base, tilt=tilt)
        if np.isfinite(fit) and fit < FIT_PX and abs(off) < 2.9:         # 2.9: the fit's own search is +-3
            offs.append(off)
            if tilt:
                tilts.append(w.last_tilt_deg)
                w.tower_tilt_joints = list(a)
    if tilts:                                                            # the tower's lean, for whoever wants it
        w.tower_tilt_deg = tuple(float(v) for v in np.median(np.array(tilts), axis=0))
        if pause and k < reads - 1:
            time.sleep(pause)
    return (float(np.median(offs)) if offs else None), len(offs)


def _warn(msg):
    if not getattr(S, "base_loop_warned", False):
        print(msg)
        S.base_loop_warned = True


def close_base(target, move, read=default_read, w=None, pause=0.3, tilt=False):
    """_close_base, traced (trace.py)."""
    from . import trace
    with trace.span("base loop", f"to {target:.1f}"):
        return _close_base(target, move, read, w, pause, tilt)


def _close_base(target, move, read=default_read, w=None, pause=0.3, tilt=False):
    """After the base was moved to count `target`: nudge it until it is within LOOP_TOL_DEG of `target` in world terms.
    move(count, finish_from_below, runup) moves the base by its count. Returns (count it ended at, error left deg), or None
    when the cameras could not check it (the base is then left where the count put it)."""
    w = w if w is not None else world()
    if w is None:
        return None
    cmd, err = float(target), None
    for k in range(LOOP_TRIES + 1):
        off, _ = world_offset(w, read, pause=pause, base=cmd, tilt=tilt)
        if off is None:
            _warn("   (base: the tower and webcam could not see enough table tags to check it, so it stays where its "
                  "count put it, about +-1 deg)")
            break
        err = cmd + off - target                             # real minus wanted
        from . import trace
        with trace.span("base error", f"{err:+.2f} deg after {k} nudge(s)"):
            pass                                             # each check's reading, for tools/timeline.py --events
        if abs(err) <= LOOP_TOL_DEG or k == LOOP_TRIES:
            break
        # Not scaled by how much the last nudge "took": the worm also lands a little differently at every stop, and
        # that reads as a gain and overcorrects. The run-up is what deals with the swallowed moves.
        nxt = cmd - err
        if abs(nxt - target) > MAX_DEG:
            _warn(f"   (base: the table tags say it is {err:+.2f} deg off, more than {MAX_DEG:g}: not nudged. "
                  "Run  python3 arm.py basehome)")
            break
        cmd = nxt
        move(cmd, True, NUDGE_RUNUP_DEG)                     # back off, then arrive moving in the + direction
    if err is not None and abs(err) > LOOP_TOL_DEG and abs(cmd - target) <= MAX_DEG:
        _warn(f"   (base: still {err:+.2f} deg off after {LOOP_TRIES} nudges)")
    S.base_cmd = float(target)
    S.base_fix = float(target) - cmd
    return (cmd, err) if err is not None else None
