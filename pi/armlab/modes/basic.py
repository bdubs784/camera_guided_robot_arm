"""
status, where, home, go.
"""
import json
import os
import time

from ..cameras import read_hand
from ..config import HAND_TAG, HOME_FILE, JOINT_NAMES, ROLL_SLIDER, ROLL_TOL
from ..errors import ArmError
from ..hardware import arm_angles, arm_state, camera_status, http_json
from ..motion import check_ready, travel
from ..safety import Zone, gap_text, pose_text
from ..state import S


def mode_status(args):
    joints = arm_state()
    now = arm_angles()
    print(f"ESP32 at {S.esp_url}: OK")
    for j, name in enumerate(JOINT_NAMES):
        info = joints[j]
        print(f"   {name:13s} {now[j]:7.1f} deg   {'ON ' if info['on'] else 'off'}   limits {info['min']} to {info['max']}")
    try:                                             # the stored servo trims: a reseated horn must NOT be trimmed
        trim = http_json(S.esp_url + "/trim", tries=1)["trim"]   # back out, or the trim gives the new room away
        print("   trims  " + "  ".join(f"{JOINT_NAMES[j]} {float(t):+.2f}" for j, t in enumerate(trim))
              + ("" if any(abs(float(t)) > 0.005 for t in trim) else "   (all zero)"))
    except (ArmError, KeyError, IndexError, ValueError, TypeError):
        print("   trims  not reported (firmware older than 4.8)")
    st = camera_status()
    print(f"Tower camera at {S.cam_url}: OK   ({'calibrated, positions in mm' if st.get('calibrated') else 'NOT calibrated'})")
    m = read_hand(reads=3, timeout=3)
    if m:
        print("   hand tag at x {:.0f}  y {:.0f}  z {:.0f} mm".format(*m["pos"]))
    else:
        print(f"   hand tag (id {HAND_TAG}) not visible right now")
    zone = Zone(args.margin)
    zone.keep_within(joints)
    print(f"Safety map: {zone.describe()}")
    print(f"   it covers  {zone.describe_range()}")
    print(f"   here:  {gap_text(zone, now)}   -> {'ALLOWED' if zone.contains(now) else 'NOT allowed'}")
    exact_here(now)
    print(f"   home:  {pose_text(zone.home)}   {zone.gap(zone.home):.0f} mm clear"
          f"   -> {'allowed' if zone.contains(zone.home) else 'NOT allowed'}")
    if abs(now[3] - ROLL_SLIDER) > ROLL_TOL:
        print(f"   NOTE: wrist rotate is {now[3]:.0f}. The map needs it near {ROLL_SLIDER:.0f} (hand level).")
    block_report(now)


def exact_here(now):
    """The exact table check's gap at this pose, beside the map's.

    The safety map is indexed by shoulder/elbow/wrist only, so it cannot know the base angle and has to assume the
    tallest thing at each distance is there at every bearing -- the stepper becomes a wall all the way round. The
    exact check (tablecheck.py, what grasp uses) knows the base angle and reads the real bearing, so its number is
    the honest one for where the arm actually is."""
    try:
        from ..tablecheck import load
        ck = load()
    except Exception:
        return
    try:
        gap, low = ck.gaps(list(now[:5]), float(now[5]))
    except Exception:
        return
    print(f"   exact table check (knows the base angle): {gap:.0f} mm to the table/plywood/base, "
          f"lowest point {low:.0f} mm above the table")


def block_report(now):
    """Where the block is, and where the fingertip is in relation to it: for driving to a hover by hand.

    Nothing here moves the arm. The numbers are the ones a hover is judged by -- how far the fingertip is above
    the block's top, how far it still has to go sideways, and whether the gripper camera can see the tag from
    where the hand is now."""
    import numpy as np
    path = os.path.join(S.data_dir, "world.json")
    if not os.path.exists(path):
        return
    try:
        from ..cube import locate
        from ..world import World
        from .where import sightings_now
        w = World(path)
        seen = sightings_now()
        cube = locate(w, seen, now[:4], now[5], grip=now[4])
    except (ArmError, OSError, ValueError, KeyError):
        return
    if cube is None:
        print("Block: no camera sees its tag from here.")
        return
    bx, by, bz = cube["xyz"]
    r = float(np.hypot(bx - w.axis[0], by - w.axis[1]))
    bearing = float(np.degrees(np.arctan2(by - w.axis[1], bx - w.axis[0])))
    print(f"Block: {r:5.1f} mm out from the base axis (bearing {bearing:+5.1f} deg), top "
          f"{cube['above_table_mm']:.1f} mm up, turned {cube['yaw_deg']:+.1f} deg"
          + ("   (" + ", ".join(f"{c} {e:.1f} px" for c, e in cube["cams"].items()) + ")"))
    try:
        tip = w.tip(list(now[:4]), float(now[5]), grip=float(now[4]))
    except Exception:
        return
    d = np.array([bx, by, bz]) - tip
    flat = float(np.hypot(d[0], d[1]))
    print(f"   fingertip is {flat:5.1f} mm sideways from the block and {-d[2]:+6.1f} mm above its top"
          f"   (fingertip at x {tip[0]:.0f} y {tip[1]:.0f} z {tip[2]:.0f})")
    sees = [c for c in cube["cams"] if c == "gripper"]
    print("   the gripper camera " + ("SEES the tag from here" if sees else "does not see the tag from here")
          + f";  seen by: {', '.join(cube['cams']) or 'nothing'}")


def mode_where(args):
    zone = Zone(args.margin)
    zone.keep_within(arm_state())
    print(f"Safety map: {zone.describe()}")
    print("Drive the arm with the controller and compare these numbers with what you see.")
    print("Clear space in mm to each thing. Ctrl+C to finish.\n")
    while True:
        now = arm_angles()
        ok = "ALLOWED    " if zone.contains(now) else "NOT allowed"
        roll = "" if abs(now[3] - ROLL_SLIDER) <= ROLL_TOL else f"   wrist rotate {now[3]:.0f}: map needs {ROLL_SLIDER:.0f}!"
        print(f"\r  {now[0]:5.1f} {now[1]:5.1f} {now[2]:5.1f}   {gap_text(zone, now)}   {ok}{roll}      ",
              end="", flush=True)
        time.sleep(0.2)


def mode_home(args):
    zone = Zone(args.margin)
    zone.keep_within(arm_state())
    now = arm_angles()
    if not zone.contains(now):
        raise ArmError(f"This pose cannot be the home pose: {zone.why_not(now)}.")
    seen = read_hand(reads=3, timeout=3) is not None
    os.makedirs(S.data_dir, exist_ok=True)
    json.dump({"pose": [round(v, 1) for v in now[:3]], "saved": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(S.data_dir, HOME_FILE), "w"))
    print(f"Home pose is now  {pose_text(now)}   ({zone.gap(now):.0f} mm clear)")
    print(f"   hand tag {'seen' if seen else 'NOT seen'} by the tower camera here."
          + ("" if seen else "  The runs need it seen at home: pick another pose."))


def mode_go(args):
    zone = Zone(args.margin)
    check_ready(zone)
    target = [float(v) for v in args.pose] if args.pose else zone.home
    if len(target) != 3:
        raise ArmError("Give three numbers: shoulder elbow wrist.   Example:  python3 arm.py go 100 60 70")
    print(f"Going to  {pose_text(target)}   ({gap_text(zone, target)})")
    now, _ = travel(target, zone, args.speed)
    seen = read_hand(reads=2, timeout=2) is not None
    print(f"There.  {gap_text(zone, now)}   hand tag {'seen' if seen else 'NOT seen'}.")
