"""
basehome: find the base's true zero from a tag on the plywood, seen by the gripper camera. Then, with a world.json, a
finish against the world model: back at home, the tower and webcam (they turn with the base, so the arm's own slack
cannot fool them) measure how far the base still is from the world's zero off the table tags, the base turns by that
and is zeroed again. The gripper camera's view of the homing tag moves with the shoulder's slack, which changes from
session to session, so the tag alone left the base up to ~1.3 deg off. --no-world skips the finish.
"""
import json
import math
import os
import select
import sys
import time

import numpy as np

from ..cameras import gripper_tag
from ..config import HOME_TAG_ID, BASE_HOME_FILE, HOME_SEARCH_DEG, HOME_TOL_DEG, HOMING_STAGE, SPEED_SETTING
from ..errors import ArmError
from ..hardware import arm_angles, base_angle, http_json
from ..motion import base_ready, check_ready, move_base, move_unchecked, restore_speed, set_gripper, travel
from ..safety import Zone, pose_text
from ..state import S
from ..util import ask
from ..baseloop import FIT_PX as WORLD_FIT_PX, close_base, world_offset
from .where import sightings_now

WORLD_MAX_DEG = 2.5                     # more than this off means something else is wrong: not applied


def go_to_homing_pose(zone, h, speed):
    """Through the stage pose, then one joint at a time: shoulder, elbow, wrist."""
    travel(list(HOMING_STAGE), zone, speed, slow_finish=True)
    here = list(HOMING_STAGE)
    for j in (0, 1, 2):
        here[j] = h["pose"][j]
        move_unchecked(here)


def leave_homing_pose(zone, h, speed):
    """The same way back, reversed: wrist up to the stage value, then elbow, then shoulder, then normal moves."""
    here = [float(v) for v in arm_angles()[:3]]
    for j in (2, 1, 0):
        here[j] = HOMING_STAGE[j]
        move_unchecked(here)
    travel(zone.home, zone, speed)


FINISH_POSE = (36.6, 75.0, 60.0)        # grasp.HIGH_POSE: the world finish is measured and nudged with the arm up.
                                        # Up here, +1.0 deg asked turned +0.77 and +2.0 turned +2.21. At home (arm
                                        # stretched out low, the hardest load on the base, tower leaning 1.8 deg)
                                        # three runs nudged ~1.3 deg and the base did not turn at all.


def finish_with_world(read=lambda: (arm_angles(), sightings_now()), zone=None, speed=None):
    """Turn the base onto the world's zero and zero the count there. Returns the offset fixed, or None.
    zone: go up to FINISH_POSE for it and back home after (without one, it is done wherever the arm is)."""
    if zone is not None:
        travel(list(FINISH_POSE), zone, speed)
        try:
            return finish_with_world(read)
        finally:
            travel(zone.home, zone, speed)
    wp = os.path.join(S.data_dir, "world.json")
    if not os.path.exists(wp):
        print("   (no world.json, so no world-model finish)")
        return None
    from ..world import World
    w = World(wp)
    time.sleep(1.0)                                         # fresh camera frames after the move home
    off, n = world_offset(w, read, tilt=True)
    if off is None:
        print("   World-model finish skipped: the tower and webcam could not fit the table tags well enough "
              f"(need a fit under {WORLD_FIT_PX:g} px). Is the arm blocking their view?")
        return None
    if abs(off) > WORLD_MAX_DEG:
        print(f"   World-model finish NOT applied: the table tags say the base is {off:+.2f} deg off, more than "
              f"{WORLD_MAX_DEG:g}. Check the cameras and the mat haven't moved, then run  python3 arm.py where")
        return None
    if abs(off) < HOME_TOL_DEG:
        print(f"   World-model check: {off:+.2f} deg from the world's zero ({n} readings), nothing to fix.")
        say_tilt(w)
        return 0.0
    close_base(0.0, lambda c, from_below, runup=None: move_base(c, from_below, closed=False, runup=runup), read=read, w=w, tilt=True)   # nudge until
    end = -S.base_fix                                       # within 0.1 deg: the worm settles +-0.6 deg each stop
    http_json(S.esp_url + "/basezero")
    S.base_fix, S.base_cmd = 0.0, 0.0
    time.sleep(1.0)
    after, n2 = world_offset(w, read, tilt=True)
    print(f"   World-model finish: the base was {off:+.2f} deg from the world's zero ({n} readings); nudged to count "
          f"{end:+.2f} and zeroed there; now " + (f"{after:+.2f} deg ({n2} readings)." if after is not None
                                                     else "(no good reading to confirm)."))
    say_tilt(w)
    return off


LEAN_REF = (0.032, -1.41)               # the lean along the arm, deg = k * shoulder + b, from the surveys the lean
                                        # model was fitted on (it depends on the shoulder: -1.2 at home, -0.2 up)


def say_tilt(w):
    """How far the tower leans relative to the table right now, as the tilted fit found it, and a line in
    ~/arm_data/tower_tilt.csv so the lean can be followed day to day (it looked like it was getting worse). The
    surveys the lean model came from put it at -1.22 deg at home."""
    t = getattr(w, "tower_tilt_deg", None)
    if t is None:
        return
    mag = float(np.hypot(*t))
    j = getattr(w, "tower_tilt_joints", None)
    then = "" if not j else f" -- the model at this shoulder: ~{abs(LEAN_REF[0] * j[0] + LEAN_REF[1]):.1f}"
    print(f"   tower lean vs the table: {mag:.2f} deg (about x {t[0]:+.2f}, y {t[1]:+.2f}){then}")
    try:
        path = os.path.join(S.data_dir, "tower_tilt.csv")
        new = not os.path.exists(path)
        with open(path, "a") as f:
            if new:
                f.write("when,joints,tilt_x_deg,tilt_y_deg,magnitude_deg\n")
            a = getattr(w, "tower_tilt_joints", None) or arm_angles()
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{' '.join(f'{v:.1f}' for v in a[:4])},{t[0]:.3f},{t[1]:.3f},{mag:.3f}\n")
    except Exception as e:
        print(f"   (could not log the lean: {e})")


WORLD_SWEEP_DEG = 24.0                  # --world-only: how far either way to turn the base looking for an angle the
WORLD_SWEEP_STEP = 3.0                  # table tags can be read from, in steps this big


def hunt_world_offset(w, read=lambda: (arm_angles(), sightings_now())):
    """The base's offset from the world's zero (deg, real minus count), turning the base to find the tags if it
    has to. Returns (offset, readings) or (None, 0), and leaves the base wherever it got a reading.

    Turning the base does NOT change the offset -- the count and the real angle move together, so the offset is a
    property of where the count was zeroed, not of where the base is pointing. What turning fixes is a different
    problem: the tower and webcam turn with the base, so from some angles they are not looking at enough of the
    mat to fit anything at all. The hunt is for the tags, not for the offset.
    """
    off, n = world_offset(w, read, tilt=True)
    if off is not None:
        return off, n
    start = base_angle()
    print(f"   the table tags cannot be read from base {start:+.1f}: turning up to +-{WORLD_SWEEP_DEG:.0f} deg to "
          "look for them")
    k = 1
    while k * WORLD_SWEEP_STEP <= WORLD_SWEEP_DEG:
        for d in (k * WORLD_SWEEP_STEP, -k * WORLD_SWEEP_STEP):
            move_base(start + d)
            time.sleep(0.6)
            off, n = world_offset(w, read, tilt=True)
            if off is not None:
                print(f"   read them at base {start + d:+.1f} ({n} readings)")
                return off, n
        k += 1
    move_base(start)
    return None, 0


def mode_world_zero(args, zone):
    """Zero the base on the world model's table tags alone -- no plywood tag, no gripper camera, no taught pose.

    Why not the plywood tag: it is seen by the GRIPPER camera, so where it appears depends on the shoulder's
    slack, which changes from session to
    session; the table tags are seen by the tower and webcam, which turn with the base and so cannot be fooled by
    the arm at all. The world's zero is also the frame the planner, the safety map and every grab already work
    in, so homing straight onto it is the one way the two cannot disagree.

    It does have a limit the plywood tag does not: fit_base_offset searches +-3 deg, so if the count is further
    out than that -- the base unplugged and turned by hand, a fresh ESP32 -- there is nothing for it to lock
    onto, and the plywood tag has to get it close first. That is said plainly rather than guessed at.
    """
    wp = os.path.join(S.data_dir, "world.json")
    if not os.path.exists(wp):
        raise ArmError("No world.json, so there is no world zero to home to. Use  python3 arm.py basehome  instead.")
    from ..world import World
    w = World(wp)
    print("Base zero from the world model's table tags (no plywood tag). Press START on the controller so it is "
          "disarmed.")
    travel(list(FINISH_POSE), zone, args.speed)             # up: the base turns properly with the arm up (and the
    #                                                         cameras need it out of their way); home again at the end
    time.sleep(1.0)
    off, n = hunt_world_offset(w)
    if off is None:
        raise ArmError(
            f"The tower and webcam could not read the table tags well enough (under {WORLD_FIT_PX:g} px) from "
            f"anywhere within {WORLD_SWEEP_DEG:.0f} deg. Either the mat is not in view -- check with  python3 "
            "arm.py where  -- or the count is more than 3 deg out, which is further than the fit can search. "
            "In that second case run  python3 arm.py basehome  once off the plywood tag and then this again.")
    print(f"   the table tags say the base is {off:+.2f} deg from the world's zero ({n} readings)")
    move_base(-off)                                         # the count whose world angle is zero: real = count + off
    close_base(0.0, lambda c, from_below, runup=None: move_base(c, from_below, closed=False, runup=runup), w=w, tilt=True)
    http_json(S.esp_url + "/basezero")
    S.base_fix, S.base_cmd = 0.0, 0.0
    time.sleep(1.0)
    after, n2 = world_offset(w, lambda: (arm_angles(), sightings_now()), tilt=True)
    print("   Zeroed there. Now " + (f"{after:+.2f} deg from the world's zero ({n2} readings)." if after is not None
                                     else "(no good reading to confirm)."))
    if after is not None and abs(after) > HOME_TOL_DEG:
        print(f"   (that is more than {HOME_TOL_DEG:g} deg: the worm settles about +-0.6 deg at each stop, so run "
              "it again if you want it tighter)")
    say_tilt(w)
    travel(zone.home, zone, args.speed)


def mode_basehome(args):
    zone = Zone(args.margin)
    check_ready(zone)
    base_ready(zone)
    if getattr(args, "world_only", False):
        return mode_world_zero(args, zone)
    path = os.path.join(S.data_dir, BASE_HOME_FILE)
    tag = args.home_tag
    if tag is None:                                    # the tag taught last time, else 0
        try:
            tag = int(json.load(open(path))["tag"])
        except (OSError, ValueError, KeyError):
            tag = HOME_TAG_ID
    if args.teach:
        print("Teach the base homing.")
        print("1. Turn the base until it sits on your cable-neutral mark (use the arm page), then press Enter here.")
        if ask("   Enter = it is on the mark,  q = quit: ") == "q":
            return
        http_json(S.esp_url + "/basezero")
        restore_speed()
        print(f"\n2. Drive the arm (controller or sliders, NOT the base) so the gripper camera looks at tag {tag} on the plywood,")
        print("   nicely in the middle of its picture. This line shows what the gripper camera sees. Press Enter when it says SEEN.")
        while True:
            seen = gripper_tag(tag, frames=1, timeout=0.6) is not None
            print(f"\r   tag {tag}: " + ("SEEN      " if seen else "not seen  "), end="", flush=True)
            if not select.select([sys.stdin], [], [], 0.4)[0]:
                continue
            line = sys.stdin.readline()
            if not line or line.strip().lower() == "q":
                raise ArmError("Stopped.")
            now = arm_angles()
            pose = [round(v, 1) for v in now[:3]]
            if gripper_tag(tag) is None:
                print("\n   The gripper camera cannot see it steadily from here. Try another pose.")
                continue
            break
        print(f"\n   Homing pose: {pose_text(pose)}, wrist rotate {now[3]:.0f}, gripper {now[4]:.0f}")
        print("\n3. Press START on the controller so it is disarmed. The base will turn 2 degrees each way to measure how the tag moves.")
        if ask("   Enter = go,  q = quit: ") == "q":
            return
        http_json(S.esp_url + f"/speed?v={SPEED_SETTING}")
        move_base(2.0)
        a = gripper_tag(tag)
        move_base(-2.0)
        b = gripper_tag(tag)
        move_base(0.0)
        c0 = gripper_tag(tag)
        if a is None or b is None or c0 is None:
            raise ArmError("The tag went out of view while the base turned 2 degrees. Pick a pose with the tag nearer the middle of the picture.")
        gain = (a - b) / 4.0                               # pixels the tag moves per degree of base
        if float(np.linalg.norm(gain)) < 2.0:
            raise ArmError(f"The tag only moves {np.linalg.norm(gain):.1f} pixels per degree of base, too little to home accurately. "
                           "Pick a pose further from the base's centre, or closer to the tag.")
        json.dump({"tag": tag, "pose": pose, "wrist_rotate": round(now[3], 1), "gripper": round(now[4], 1),
                   "centre": [round(float(v), 2) for v in c0], "px_per_deg": [round(float(v), 3) for v in gain],
                   "made": time.strftime("%Y-%m-%d %H:%M")}, open(path, "w"), indent=1)
        print(f"\nSaved. The tag moves {np.linalg.norm(gain):.1f} pixels per degree of base, so homing can tell about "
              f"{1.0 / np.linalg.norm(gain):.2f} of a degree apart (before camera noise).")
        leave_homing_pose(zone, None, args.speed)
        print("Now try it:  python3 arm.py basehome      (or  basehome --check  to measure how repeatable it is)")
        return

    if not os.path.exists(path):
        raise ArmError("Base homing has not been taught yet. Run:  python3 arm.py basehome --teach")
    h = json.load(open(path))
    tag, c0, gain = h["tag"], np.array(h["centre"]), np.array(h["px_per_deg"])
    print(f"Base homing from tag {tag} (taught {h['made']}). Press START on the controller so it is disarmed.")
    if abs(base_angle()) > 1.0:                        # the base was left elsewhere (a scan, a demo): back to where
        print(f"   turning the base from {base_angle():+.0f} back to 0 first (arm at home)")   # homing was taught
        travel(zone.home, zone, args.speed)
        move_base(0.0)
    go_to_homing_pose(zone, h, args.speed)
    set_gripper(h["gripper"], secs=1.0)

    def home_once():
        start = base_angle()
        here = start
        c = gripper_tag(tag)
        if c is None:                                      # search, never more than HOME_SEARCH_DEG either way
            for off in (10, -10, 20, -20, 30, -30):
                if abs(off) > HOME_SEARCH_DEG:
                    break
                move_base(start + off)
                here = start + off
                c = gripper_tag(tag)
                if c is not None:
                    break
            if c is None:
                move_base(start)
                raise ArmError(f"Tag {tag} was not found within {HOME_SEARCH_DEG:.0f} degrees either way. Turn the base by hand "
                               "(arm page) to roughly the cable-neutral mark and run this again.")
        for _ in range(8):
            delta = -float((c - c0) @ gain / (gain @ gain))
            if abs(delta) <= HOME_TOL_DEG:
                return here, float(np.linalg.norm(c - c0))
            here += delta
            move_base(here)
            c = gripper_tag(tag)
            if c is None:
                raise ArmError("The tag dropped out of view while homing. Try again, and check the gripper camera.")
        return here, float(np.linalg.norm(c - c0))

    found, px = home_once()
    http_json(S.esp_url + "/basezero")
    print(f"   Homed. The base's own count was {found:+.2f} degrees off, and is now reset to 0 "
          f"(tag within {px:.1f} pixels of its taught spot).")
    if args.check:
        ends = []
        for side in (30.0, -30.0, 30.0, -30.0):
            move_base(side)
            end, _ = home_once()
            ends.append(end)
            print(f"   from {side:+.0f}: homed at {end:+.3f} degrees")
        spread = max(ends) - min(ends)
        print(f"\nHoming repeats to within {spread:.2f} degrees ({spread * math.pi / 180 * 350:.1f} mm sideways at 350 mm reach).")
        move_base(0.0)
    leave_homing_pose(zone, h, args.speed)
    if not getattr(args, "no_world", False):
        finish_with_world(zone=zone, speed=args.speed)
