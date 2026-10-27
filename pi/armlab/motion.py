"""
Moving the arm: checked straight moves and routes, the base, the gripper, and the start-of-run checks.
Every move is checked against the safety map before it starts and watched while it runs.
"""
import json
import os
import time

import numpy as np

from .config import (
    BASE_DPS,
    ARRIVE_TOL, AUTO_LO, BASE_LIMIT_DEG, FINISH_DEG, FINISH_SPEED, GRIP_TIME_S,
    HOME_FINISH_DEG, JOINT_NAMES, MAX_SPEED, MOVE_LIMIT_S, MOVING, NORMAL_SPEED,
    OFF_PATH_DEG, PICK_SLOW, ROLL_SLIDER, ROLL_TOL, RUN_SPEED, SPEED_SETTING, FW_SPEED_BASE, STALL_S, TICK_CAP_DEG,
    TIMING_FILE, WATCH_SLACK_MM,
)
from .errors import ArmError
from .hardware import arm_angles, arm_state, arm_stop, base_angle, http_json
from .safety import gap_text, off_path, pose_text
from . import live, trace
from .state import S
from .util import ask


def check_ready(zone):
    """Refuse to start unless the joints are powered, the hand is level and the arm is in an allowed pose."""
    joints = arm_state()
    off = [JOINT_NAMES[j] for j in MOVING if not joints[j]["on"]]
    if off:
        raise ArmError("These joints are switched OFF on the arm page: " + ", ".join(off) +
                       ". Turn them on (watch for the power-on snap), then run this again.")
    zone.keep_within(joints)
    try:
        S.slider_range = tuple((float(joints[j]["min"]), float(joints[j]["max"])) for j in MOVING)
        S.auto_hi = tuple(min(S.auto_hi[j], S.slider_range[j][1]) for j in range(3))
    except (KeyError, TypeError, ValueError):
        pass
    now = arm_angles()
    if S.base_cmd is None and len(now) > 5:
        S.base_cmd = float(now[5])                         # the exact table check needs the base angle (zone._state)
    if abs(now[3] - ROLL_SLIDER) > ROLL_TOL:
        raise ArmError(f"Wrist rotate is at {now[3]:.0f}. The safety map was drawn with the hand level "
                       f"(wrist rotate {ROLL_SLIDER:.0f}).\nSet wrist rotate to {ROLL_SLIDER:.0f} with the "
                       "controller or the slider, then run this again.")
    if not zone.contains(zone.home):
        raise ArmError(f"The home pose ({pose_text(zone.home)}) is not allowed: {zone.why_not(zone.home)}.\n"
                       "Drive to a better one and run:  python3 arm.py home")
    if not zone.contains(now):
        raise ArmError(f"The arm is not in an allowed pose right now: {zone.why_not(now)}.\n"
                       f"   now:  {pose_text(now)}\n"
                       "Drive it somewhere with more room using the controller, then run this again.\n"
                       "(python3 arm.py where  shows the clear space live while you drive.)")
    # Raise the ESP32's own speed cap for the run, so it never makes one joint lag behind the others.
    http_json(S.esp_url + f"/speed?v={SPEED_SETTING}")
    S.speed_changed = True
    try:                                                   # firmware 4.7+: clock-timed moves (A/B tested on the arm:
        ver = http_json(S.esp_url + "/ver", tries=1)                                        # settle about 2x faster)
        S.prof = 1 if 1 in ver.get("prof", []) else 0
        S.base_speed_cmd = bool(ver.get("bs"))           # firmware 4.11+: smooth slow base moves (bs=)
    except ArmError:
        S.prof = 0
    if S.prof:
        S.time_scale = 1.0                                   # clock-timed moves take exactly as long as asked
    else:
        try:                                               # start from the timing measured last time
            S.time_scale = min(6.0, max(1.0, float(json.load(open(os.path.join(S.data_dir, TIMING_FILE)))["time_scale"])))
        except (OSError, ValueError, KeyError):
            pass
    return now


def restore_speed():
    """Put the ESP32's Speed setting back to normal, so the sliders and controller behave as usual."""
    if S.speed_changed:
        try:
            http_json(S.esp_url + f"/speed?v={NORMAL_SPEED}", tries=2)
            if not S.prof:
                json.dump({"time_scale": round(S.time_scale, 2)}, open(os.path.join(S.data_dir, TIMING_FILE), "w"))
        except (ArmError, OSError):
            pass


def move_to(target, zone, speed=RUN_SPEED, slow_finish=False):
    """One straight move of shoulder, elbow and wrist to `target`. Checked before and watched during."""
    target = [round(float(v), 1) for v in target[:3]]
    now = arm_angles()
    start = now[:3]
    far = max(abs(target[i] - start[i]) for i in range(3))
    if far <= ARRIVE_TOL:
        return now
    if abs(now[3] - ROLL_SLIDER) > ROLL_TOL and zone.exact is None:     # the exact check uses the real rotate
        raise ArmError(f"Wrist rotate has moved to {now[3]:.0f}; the safety map needs it near "
                       f"{ROLL_SLIDER:.0f}. Put it back and run this again.")
    if not zone.contains(target):
        raise ArmError(f"refused: will not go to {pose_text(target)} because {zone.why_not(target)}")
    with trace.span("compute", "move check"):
        worst = zone.path_gap(start, target)
    live.check(worst, zone.margin)                         # the demo page's safety strip
    if worst < zone.margin:
        raise ArmError(f"refused: the straight move from {pose_text(start)} to {pose_text(target)} "
                       f"passes within {worst:.0f} mm of something (the margin is {zone.margin:.0f} mm)")
    if S.confirm_left > 0:
        print(f"\n  next move: {pose_text(start)}  ->  {pose_text(target)}")
        print(f"     at the end: {gap_text(zone, target)}")
        print(f"     tightest spot on the way: {worst:.0f} mm clear")
        if ask(f"     Enter = go,  q = quit   ({S.confirm_left} more to confirm): ") == "q":
            raise ArmError("Stopped before that move.")
        S.confirm_left -= 1

    if slow_finish and far > FINISH_DEG + 2:                  # most of the way at speed, the last part slowly, on the same line
        part = 1.0 - FINISH_DEG / far
        mid = [round(start[i] + (target[i] - start[i]) * part, 1) for i in range(3)]
        _glide(mid, zone, speed, start, target)
        return _glide(target, zone, FINISH_SPEED, start, target)
    return _glide(target, zone, speed, start, target)


GLIDE_POLL_S = 0.05            # how often a glide reads where the arm is (was 0.15: ~75 ms lost at the end of every move;
                               # over the cable a read is a few ms)


def _glide(dest, zone, speed, line_from, line_to):
    """Send one glide along the already-checked line and wait for it, watching where the arm really goes."""
    with trace.span("arm", "to " + " ".join(f"{v:.1f}" for v in dest[:3])):
        return _glide_now(dest, zone, speed, line_from, line_to)


def _glide_now(dest, zone, speed, line_from, line_to):
    now = arm_angles()
    start = now[:3]
    far = max(abs(dest[i] - start[i]) for i in range(3))
    if far <= ARRIVE_TOL:
        return now
    for i in range(3):                                        # which way each joint arrives (slackgp.py)
        if abs(dest[i] - start[i]) > 0.05:
            S.came_from[i] = float(start[i])
    want_s = far / min(speed, MAX_SPEED)                      # how long the move should really take
    ms = int(max(200, want_s * 1000 / S.time_scale))            # the ESP32 runs slow by time_scale, so ask for less
    ms = max(ms, int(far / TICK_CAP_DEG * 10) + 1)            # but never so little that it would cap one joint
    if S.prof:                                                # clock-timed: no joint may hit its own speed cap either
        caps = [b * SPEED_SETTING for b in FW_SPEED_BASE]
        ms = max(ms, int(1100 * max(1.875 * abs(dest[i] - start[i]) / caps[i] for i in range(3))) + 1)
    S.moving = True
    http_json(S.esp_url + "/setall?a=" + ",".join(f"{v:.1f}" for v in dest) + f"&ms={ms}" + ("&prof=1" if S.prof else ""))
    # Wait for as long as the arm keeps making progress, and watch where it actually goes.
    t_start = t_progress = time.time()
    last = start
    problem = None
    while time.time() - t_start < MOVE_LIMIT_S:
        time.sleep(GLIDE_POLL_S)
        now = arm_angles()
        trace.sample(now)
        if all(abs(now[i] - dest[i]) <= ARRIVE_TOL for i in range(3)):
            took = time.time() - t_start
            if ms >= 500 and not S.prof:                        # learn how slow the ESP32 really is
                S.time_scale = min(6.0, max(1.0, 0.5 * S.time_scale + 0.5 * max(took - 0.1, 0.05) * 1000 / ms))
            if took > 3 * want_s + 2:
                print(f"\n  (note: that move took {took:.1f} s, wanted {want_s:.1f} s.)")
            return now
        stray = off_path(now, line_from, line_to)
        if stray > OFF_PATH_DEG:
            problem = (f"The arm strayed {stray:.0f} degrees from the planned path, so I stopped it.\n"
                       "The joints are not moving together. Was the controller armed, or a slider moved?")
            break
        room = zone.gap(now)
        if room < zone.margin - WATCH_SLACK_MM:
            problem = (f"Mid-move the arm had only {room:.0f} mm of clear space on the map, so I stopped it.")
            break
        if max(abs(now[i] - last[i]) for i in range(3)) >= 0.1:
            t_progress, last = time.time(), now[:3]
        elif time.time() - t_progress > STALL_S:
            break
    arm_stop()
    if problem:
        raise ArmError(problem + f"\n   it is at:  {pose_text(now)}")
    moved = max(abs(now[i] - start[i]) for i in range(3))
    raise ArmError(
        f"The arm stopped {'part-way' if moved > 0.5 else 'without moving'} and did not reach "
        f"{pose_text(dest)}.\n   it is at:  {pose_text(now)}\n"
        "Likely causes:\n"
        "   - STOP or the B button was pressed on the arm page or controller (that cancels the move)\n"
        "   - the controller was still armed and a stick was touched\n"
        "   - a servo was switched off on the arm page\n"
        "Nothing is wrong with the map. Just run the same command again.")


def travel(target, zone, speed=RUN_SPEED, slow_finish=False):
    """
    Go to target by a route the map allows: straight there, or by way of home.
    With slow_finish, the last few degrees are done slowly (used wherever the arm is about to be measured).
    Returns (angles on arrival, pose the last straight move started from).
    """
    now = arm_angles()
    legs = zone.route(now[:3], target)
    if legs is None:
        why = zone.why_not(target) if not zone.contains(target) else \
            "neither the straight move nor the move by way of home is clear"
        raise ArmError(f"refused: cannot get from {pose_text(now)} to {pose_text(target)}: {why}")
    came_from = now[:3]
    for k, leg in enumerate(legs):
        came_from = now[:3]
        now = move_to(leg, zone, speed, slow_finish and k == len(legs) - 1)
    return now, came_from


BASE_MOVE_DPS = BASE_DPS       # firmware 4.11+: every automatic base move runs smoothly at this speed (config)
BASE_GLIDE_DPS = 6.0           # automatic base moves longer than a few degrees go no faster than this. The stepper
BASE_GLIDE_TICK_S = 0.2        # missed whole moves at its full 20 deg/s with the arm reaching out (once ending up
                               # 90 deg from where it thought it was); small steps do not.


def glide_base(target):
    """Turn the base to near `target` in small steps at BASE_GLIDE_DPS (the last few degrees are left to the caller)."""
    now = base_angle()
    far = target - now
    if abs(far) <= 3.0:
        return
    n = int(abs(far) / (BASE_GLIDE_DPS * BASE_GLIDE_TICK_S)) + 1
    S.moving = True
    for k in range(1, n + 1):
        t0 = time.time()
        b = now + far * k / n
        http_json(S.esp_url + f"/setall?a=,,,,,{b:.3f}")
        S.base_cmd = b
        time.sleep(max(0.0, BASE_GLIDE_TICK_S - (time.time() - t0)))


def move_base(target, finish_from_below=True, closed=None, runup=None, settle=True):
    """Turn the base to `target` degrees and wait. The last part is always done in the + direction (worm slack).
    closed (default S.base_closed, on in grab/pick/find/scan/demo): then the tower and webcam check it against the
    world model and it is nudged onto the target (baseloop.py). settle=False: no 0.6 s pause at the end -- for an
    open-loop turn followed by an arm move, not a look (the closed loop always settles before it looks)."""
    closed = (S.base_closed if closed is None else closed) and finish_from_below
    _move_base_count(target, finish_from_below, runup, settle=settle or closed)
    if closed:
        from .baseloop import close_base
        close_base(target, _move_base_count)


BASE_SETTLE_S = 0.4               # after a base turn, before a look; the base stops smoothly at 12 deg/s. Too
                                  # short shows up as more closed-loop nudges or "not seen"


def _move_base_count(target, finish_from_below=True, runup=None, settle=True):
    """move_base by the motor count alone. runup: how far below the target the last, + direction leg starts
    (default HOME_FINISH_DEG)."""
    if abs(target) > BASE_LIMIT_DEG:
        raise ArmError(f"refused: base {target:.1f} is past the cable limit of {BASE_LIMIT_DEG:.0f} either way.")
    legs = [target - (HOME_FINISH_DEG if runup is None else runup), target] if finish_from_below else [target]
    if not S.base_speed_cmd:
        glide_base(legs[0])                                # old firmware: most of the way in small steps
    for leg in legs:
        with trace.span("base", f"leg {leg:.1f}"):
            start = base_angle()
            S.moving = True
            speed = f"&bs={BASE_MOVE_DPS:g}" if S.base_speed_cmd else ""     # 4.11+: one smooth move at this speed
            http_json(S.esp_url + f"/setall?a=,,,,,{leg:.3f}" + speed)
            S.base_cmd = leg
            t_end = time.time() + abs(leg - start) / min(10.0, BASE_MOVE_DPS) + 4.0
            hold = None
            while time.time() < t_end:
                a = arm_angles()
                trace.sample(a)
                hold = hold or a[:3]
                moved = max(abs(a[i] - hold[i]) for i in range(3))
                if moved > 2.0:                                # nothing should move the arm during a base move
                    arm_stop()
                    raise ArmError(f"stopped: the arm moved {moved:.1f} deg during a base-only move "
                                   f"(from {pose_text(hold)} to {pose_text(a)}). Something else sent it a command.")
                if abs(a[5] - leg) <= 0.06:                    # the reading is rounded to 0.1
                    break
                time.sleep(0.1)
            if abs(base_angle() - leg) > 0.15:
                arm_stop()
                raise ArmError(f"The base did not reach {leg:.1f} (it is at {base_angle():.1f}). Is the base switched on?")
    if settle:
        with trace.span("wait", "base settle"):
            time.sleep(BASE_SETTLE_S)                      # let the arm stop swaying before the camera looks


def pan_move(target, b1, zone, speed=RUN_SPEED, closed=False, settle=True, slow_finish=False):
    """The arm to `target` (shoulder, elbow, wrist) while the base turns to b1, together (--pan-overlap). The caller
    checks the pair first (modes/grasp.check_pan_move: every arm point at every base angle). The base's long leg is
    sent first and not waited for (to b1 - HOME_FINISH_DEG, at BASE_MOVE_DPS); the arm then goes by move_to, so its
    own checks and the live watcher run as for any move (the watcher reads the real base, which the exact table
    check uses); then it waits for the base, and move_base does the + finish leg and the closed loop, exactly as the
    one-after-the-other way ends. Not move_base for the long leg: its "the arm moved during a base-only move" guard
    stays for base-only moves. Old firmware (no smooth base moves) cannot send the base without waiting: base,
    then arm."""
    if abs(b1) > BASE_LIMIT_DEG:
        raise ArmError(f"refused: base {b1:.1f} is past the cable limit of {BASE_LIMIT_DEG:.0f} either way.")
    if not S.base_speed_cmd:
        move_base(b1, closed=closed, settle=False)
        return move_to(target, zone, speed, slow_finish)
    leg = b1 - HOME_FINISH_DEG
    start = base_angle()
    S.moving = True
    with trace.span("base", f"pan leg {leg:.1f} (arm moving)"):
        http_json(S.esp_url + f"/setall?a=,,,,,{leg:.3f}&bs={BASE_MOVE_DPS:g}")
    S.base_cmd = leg
    t_end = time.time() + abs(leg - start) / min(10.0, BASE_MOVE_DPS) + 4.0
    try:
        now = move_to(target, zone, speed, slow_finish)
    except BaseException:
        arm_stop()                                         # the base is still turning: stop that too
        raise
    with trace.span("base", f"pan leg {leg:.1f} (arm there)"):
        hold = now[:3]
        while time.time() < t_end:
            a = arm_angles()
            trace.sample(a)
            moved = max(abs(a[i] - hold[i]) for i in range(3))
            if moved > 2.0:                                # the arm has arrived: nothing should move it now
                arm_stop()
                raise ArmError(f"stopped: the arm moved {moved:.1f} deg while the base finished its turn "
                               f"(from {pose_text(hold)} to {pose_text(a)}). Something else sent it a command.")
            if abs(a[5] - leg) <= 0.06:
                break
            time.sleep(0.1)
        if abs(base_angle() - leg) > 0.15:
            arm_stop()
            raise ArmError(f"The base did not reach {leg:.1f} (it is at {base_angle():.1f}). Is the base switched on?")
    move_base(b1, closed=closed, settle=settle)            # the + finish leg (its first leg is where it already is)
    return arm_angles()


def move_unchecked(target, speed=PICK_SLOW):
    """
    A slow, straight move that the safety map does not check, for the taught way in and out of the homing pose
    (the map's hand is a big box, so it refuses poses the real hand is fine at; you checked these by eye).
    """
    target = auto_clamp(target) + list(target[3:])
    now = arm_angles()
    start = now[:3]
    far = max(abs(target[i] - start[i]) for i in range(3))
    if far <= ARRIVE_TOL:
        return now
    ms = max(int(max(200, far / speed * 1000 / S.time_scale)), int(far / TICK_CAP_DEG * 10) + 1)
    S.moving = True
    http_json(S.esp_url + "/setall?a=" + ",".join(f"{v:.1f}" for v in target[:3]) + f"&ms={ms}" + ("&prof=1" if S.prof else ""))
    t_end = time.time() + 3 * far / speed + 5
    while time.time() < t_end:
        time.sleep(0.15)
        now = arm_angles()
        if all(abs(now[i] - target[i]) <= ARRIVE_TOL for i in range(3)):
            return now
    arm_stop()
    raise ArmError(f"The arm did not reach {pose_text(target)} (it is at {pose_text(now)}). It has been stopped.")


def base_ready(zone):
    joints = arm_state()
    if len(joints) < 6 or not joints[5]["on"]:
        raise ArmError("The base is switched OFF on the arm page. Turn it on and run this again.")


def auto_clamp(q):
    """A pose pulled back inside AUTO_LO/AUTO_HI (shoulder, elbow, wrist)."""
    return [float(np.clip(q[j], AUTO_LO[j], S.auto_hi[j])) for j in range(3)]


def set_gripper(value, secs=GRIP_TIME_S):
    """Glide the gripper to a slider value and wait for it."""
    if abs(arm_angles()[4] - value) <= 0.6:
        return
    S.moving = True
    http_json(S.esp_url + f"/setall?a=,,,,{value:.1f}&ms={int(max(200, secs * 1000 / S.time_scale))}" + ("&prof=1" if S.prof else ""))
    t_end = time.time() + 4 * secs + 3
    while time.time() < t_end and abs(arm_angles()[4] - value) > 0.6:
        time.sleep(0.15)
    time.sleep(0.5)                                         # let the fingers finish squeezing or letting go


