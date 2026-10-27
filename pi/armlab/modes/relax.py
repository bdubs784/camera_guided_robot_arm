"""
relax: park the arm in its rest pose (a pose it can sit in limp without falling or buzzing), then switch the servos
off. wake: switch them back on one at a time (they hold the rest pose, so nothing flings) and go home slowly.
The rest pose lives on the ESP32 (firmware 4.15+, /rest), so after a power cut the servos are commanded to it too.

    python3 arm.py relax            go home, base to 0, ease down to the rest pose, servos off
    python3 arm.py relax --teach    the pose the arm is in now becomes the rest pose (drive it there with the sliders)
    python3 arm.py wake             servos on (shoulder first, gripper last), holding the rest pose
    python3 arm.py wake --home      ...then ease up to home
"""
import time

from ..config import JOINT_NAMES, PICK_SLOW
from ..errors import ArmError
from ..hardware import arm_angles, arm_state, http_json
from ..motion import check_ready, move_base, move_unchecked, travel
from ..safety import Zone, pose_text
from ..state import S

EASE_DPS = PICK_SLOW            # how fast it eases down into the rest pose and back up out of it (deg/s)


def rest_pose():
    try:
        r = http_json(S.esp_url + "/rest")
    except ArmError:
        r = {}
    if "rest" not in r:
        raise ArmError("The ESP32 has no rest pose (it needs firmware 4.15). Flash it, then run this again.")
    rest = [float(v) for v in r["rest"]]
    if [round(v) for v in rest[:3]] == [90, 90, 90]:
        raise ArmError("The rest pose on the ESP32 is still the factory 90/90/90. Set yours first:\n"
                       f"   curl \"{S.esp_url}/rest?a=104,79,62,42,80\"   (or drive there and: python3 arm.py relax --teach)")
    return rest


def wait_joints(target, idx, secs):
    t_end = time.time() + secs
    while time.time() < t_end:
        now = arm_angles()
        if all(abs(now[i] - target[i]) <= 1.0 for i in idx):
            return
        time.sleep(0.15)


def mode_relax(args):
    if args.teach:
        now = arm_angles()
        r = http_json(S.esp_url + "/rest?set=1")
        print(f"Rest pose saved on the ESP32: {', '.join(f'{JOINT_NAMES[i]} {v:.0f}' for i, v in enumerate(r['rest']))}"
              f"  (the arm is at {pose_text(now)})")
        return
    rest = rest_pose()
    zone = Zone(args.margin)
    check_ready(zone)
    out = [f"{JOINT_NAMES[i]} {rest[i]:.1f} (map {zone.lo[i]:.0f} to {zone.hi[i]:.0f})"
           for i in range(3) if not zone.lo[i] <= rest[i] <= zone.hi[i]]
    if out:                               # the rest pose lives on the ESP32, outside this repo, in slider numbers: a
        raise ArmError(                   # reseat or a limit change leaves it pointing somewhere else entirely
            "The rest pose on the ESP32 is outside the safety map: " + ", ".join(out) + "."
            + "\nIt was taught against a different slider-to-angle mapping, so easing down to it would put"
              " the arm somewhere other than where it was taught.\n"
              "Drive to the rest position and  python3 arm.py relax --teach, or set it directly:"
              "  /rest?a=<shoulder>,<elbow>,<wrist>,<rotate>,<gripper>")
    print(f"Relaxing: home, base to 0, then easing down to the rest pose {pose_text(rest[:3])}.")
    travel(zone.home, zone, PICK_SLOW * 2)
    if not args.keep_base:
        try:
            move_base(0.0)
        except ArmError as e:
            print(f"   (base left where it is: {e})")
    move_unchecked(rest[:3], EASE_DPS)                    # the slow straight way down you checked by eye
    now = arm_angles()
    secs = max(1.0, max(abs(now[3] - rest[3]), abs(now[4] - rest[4])) / 40.0)
    http_json(S.esp_url + f"/setall?a=,,,{rest[3]:.1f},{rest[4]:.1f}&ms={int(secs * 1000)}"
              + ("&prof=1" if S.prof else ""))
    wait_joints(rest, (3, 4), secs + 2.0)
    time.sleep(0.5)
    for j in range(5):
        http_json(S.esp_url + f"/power?j={j}&on=0")
    print("Relaxed: servos OFF. To wake it:  python3 arm.py wake   (or switch the servos on from the arm page: "
          "they hold the rest pose, no fling)")


def mode_wake(args):
    rest = rest_pose()
    joints = arm_state()
    off = [j for j in range(5) if not joints[j]["on"]]
    if off:
        cmd = arm_angles()
        if any(abs(cmd[j] - rest[j]) > 2.0 for j in off):
            print("   (the servos are commanded away from the rest pose: they will go there when switched on)")
            http_json(S.esp_url + "/setall?a=" + ",".join(f"{v:.1f}" for v in rest))
        for j in sorted(off):                               # shoulder first: lifts the arm clear before the rest
            http_json(S.esp_url + f"/power?j={j}&on=1")
            print(f"   {JOINT_NAMES[j]} on")
            time.sleep(0.4)
        time.sleep(0.5)
    if not args.home:
        print(f"Awake, holding the rest pose {pose_text(rest[:3])}.  (python3 arm.py wake --home also goes home)")
        return
    zone = Zone(args.margin)
    zone.keep_within(arm_state())
    print(f"Easing up from the rest pose to home {pose_text(zone.home)}...")
    move_unchecked(zone.home, EASE_DPS)                   # the same slow straight line relax came down
    check_ready(zone)
    print("Awake and home.")
