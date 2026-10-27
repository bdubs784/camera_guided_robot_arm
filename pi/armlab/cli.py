"""
The command line: python3 arm.py <mode> [options]. Each mode lives in modes/.
"""
import argparse
import os
import signal
import sys

from .config import MARGIN_MM, RUN_SPEED, SURVEY_SPEED, TARGET_TAG
from .errors import ArmError
from .hardware import arm_stop
from .modes.basehome import mode_basehome
from .modes.survey import mode_rollsweep, mode_safetyrun, mode_survey
from .modes.where import mode_where as mode_where_world
from .modes.track import mode_track
from .modes.touch import mode_fit_touch, mode_touch
from .modes.relax import mode_relax, mode_wake
from . import link
from .modes.basic import mode_go, mode_home, mode_status
from .modes.grasp import mode_grasp
from .modes.show import mode_show
from .motion import restore_speed
from .state import S


MODES = {
    # everyday
    "status": mode_status, "where": mode_where_world, "home": mode_home, "go": mode_go, "relax": mode_relax, "wake": mode_wake,
    # the camera-guided grab and the demo loop
    "grasp": mode_grasp, "show": mode_show, "basehome": mode_basehome,
    # measuring and calibrating the arm
    "survey": mode_survey, "rollsweep": mode_rollsweep, "safetyrun": mode_safetyrun, "touch": mode_touch,
    "fit_touch": mode_fit_touch, "track": mode_track,
}


CLOSED_BASE_MODES = {"grasp", "show"}   # base moves checked by the cameras (baseloop.py)


def main():
    modes = MODES
    p = argparse.ArgumentParser(description="Control program for the camera-guided robot arm.")
    p.add_argument("mode", choices=list(modes))
    p.add_argument("pose", nargs="*", help="go: shoulder elbow wrist (leave out to go home)")
    p.add_argument("--esp", default=os.environ.get("ARM_ESP", S.esp_url), help="ESP32 address")
    p.add_argument("--cam", default=os.environ.get("ARM_CAM", S.cam_url), help="camera program address")
    p.add_argument("--data", default=os.environ.get("ARM_DATA", S.data_dir), help="folder for files")
    p.add_argument("--margin", type=float, default=MARGIN_MM, help="mm of clear space to keep (default 40)")
    p.add_argument("--table-margin", type=float, default=None,
                   help="mm of clear space to the table, plywood and base only (default: same as --margin)")
    p.add_argument("--confirm", type=int, default=0, help="wait for Enter before each of the first N moves")
    p.add_argument("--poses", type=int, default=300, help="collect: number of poses")
    p.add_argument("--speed", type=float, default=None, help="deg/s for automatic moves (default 30; survey 100; max 100)")
    p.add_argument("--seed", type=int, default=None, help="random seed (same seed = same poses; default: a new one every run)")
    p.add_argument("--tag", type=int, default=TARGET_TAG, help="the cube's tag (grab, find, demo) or follow's target (default 7)")
    p.add_argument("--teach", action="store_true", help="basehome / grab: show it once by hand")
    p.add_argument("--check", action="store_true", help="basehome: home from both sides several times and report how repeatable it is")
    p.add_argument("--force", action="store_true", help="fit_touch: write into world.json even if the fit is not pinned down")
    p.add_argument("--cube-r", type=float, default=None,
                   help="grasp: the block's cautious cylinder radius, mm (default 40: the 39.7 mm square's corners "
                        "are 28 mm out, plus 12)")
    p.add_argument("--cube-clear", type=float, default=None,
                   help="grasp: clearance every move but the way in keeps from that cylinder, mm (default 25)")
    p.add_argument("--hover-clear", type=float, default=None,
                   help="grasp: clearance the hover itself keeps, mm (default 12; the hover sits over the block with "
                        "the fingers either side of it, so this is the one to turn down first)")
    p.add_argument("--slow", type=float, default=None,
                   help="grasp, show: deg/s for the moves beside the block -- in, lift, down, let go (default 8). "
                        "Those four were ~19 s of a 75 s round at 8")
    p.add_argument("--pause-hover", action="store_true",
                   help="grasp: when no camera sees the block at the hover, stop and let you drive the sliders "
                        "until one does, recording the arm and both cameras the whole time "
                        "(~/arm_data/hover_tune/). Ctrl-C to go on. Nothing moves the arm while it is paused")
    p.add_argument("--hunt", choices=("back", "out", "both"), default="both",
                   help="grasp: when no camera sees the block at the hover, sweep the wrist this way looking for "
                        "it. One clean sweep each way by default; back or out alone isolates the behaviour")
    p.add_argument("--looks", type=int, default=3,
                   help="grasp: look at the block from this many base angles and plan off the average. One look "
                        "is not a measurement: the same webcam once put the same block 19.1 mm apart from two "
                        "base angles 12 deg apart. 1 = the old behaviour")
    p.add_argument("--close-in", type=float, default=0.0,
                   help="grasp, show: for blocks nearer than 320 mm, aim the final grasp this many mm toward the tower "
                        "from where the close look put it (0 = off, the default: the misses it was for were across the "
                        "fingers, not toward the tower -- the fingertip point sat 10 mm off the fingers' midline)")
    p.add_argument("--reach-in", type=float, default=0.0,
                   help="grasp: grab this many mm closer to the base axis than the block reads. The leftover "
                        "fingertip error, measured on the arm; it was about 2 mm too far out at 257 mm")
    p.add_argument("--grasp-depth", type=float, default=None,
                   help="grasp: how far below the block's top the fingertips go, mm (default 20). Lower grips "
                        "higher on the block and lets a shorter hover clear it")
    p.add_argument("--pitch-drift", type=float, default=None,
                   help="grasp: how far the hand's angle may wander on the way in, deg (default 35). It only has "
                        "to be right where the fingers close")
    p.add_argument("--snap", action="store_true",
                   help="grasp: just look at the block and save what the cameras saw, no planning. Seconds "
                        "instead of a minute, so a lot of block positions can be gathered and planned later")
    p.add_argument("--why", action="store_true",
                   help="grasp: say what each hover was rejected for, and which point of the hand was closest")
    p.add_argument("--grasp-margin", type=float, default=12.0,
                   help="grasp: table margin (mm) for the slow last part of the way in, where the fingers go down beside the cube")
    p.add_argument("--dry", action="store_true", help="grasp: find the cube and plan the grab, but do not move")
    p.add_argument("--grab", action="store_true", help="grasp: the whole grab (hover, close-up look, way in, close, lift, check, set down)")
    p.add_argument("--hover", action="store_true", help="grasp: go to the hover above the cube and stop there (checks the aim)")
    p.add_argument("--exact-table", action="store_true",
                   help="table/plywood clearance from the exact check (world model + measured hand) instead of the map")
    p.add_argument("--open-base", action="store_true",
                   help="grab/pick/find/scan/demo: base moves by the count alone (no camera check and nudge)")
    p.add_argument("--no-world", action="store_true", help="basehome: skip the finish against the world model (table tags)")
    p.add_argument("--world-only", action="store_true",
                   help="basehome: zero the base on the world model's table tags alone, with no plywood tag and "
                        "no taught pose (needs world.json)")
    p.add_argument("--home-tag", type=int, default=None, help="basehome --teach: id of the homing tag (default: the one taught last time)")
    p.add_argument("--auto", action="store_true", help="pick: work the grasp out from the finger calibration instead of a taught grasp")
    p.add_argument("--depth", type=float, default=15.0, help="pick --auto: how far below the object's tag the pads grip, mm")
    p.add_argument("--at", type=int, default=1, help="fingers: how many arm poses to measure at (1 = home only)")
    p.add_argument("--find", action="store_true", help="grab: find the cube with the tower camera and turn to it first")
    p.add_argument("--joint", choices=["shoulder", "elbow", "wrist"], default="shoulder", help="trim: the joint whose servo was swapped")
    p.add_argument("--hold", action="store_true", help="pick: pause with the object held up until you press Enter")
    p.add_argument("--dps", type=float, default=None, help="pantest / find: base pan speed, deg/s")
    p.add_argument("--link", choices=["auto", "cable", "wifi"], default=os.environ.get("ARM_LINK", "auto"), help="how to talk to the ESP32: the Pi cable if it answers (auto), cable only, or WiFi only")
    p.add_argument("--home", action="store_true", help="wake: after switching on, ease up to the home pose")
    p.add_argument("--keep-base", action="store_true", help="relax: leave the base where it is (default: turn it to 0)")
    p.add_argument("--sw", default=None, help="noisetest: servo write intervals to compare, ms, e.g. 0,20,50")
    p.add_argument("--rounds", type=int, default=10, help="demo, show: how many find-grab-move rounds")
    p.add_argument("--radii", default=None,
                   help="show: distances from the base axis to walk the block along, mm "
                        "(default 240,290,340,375). The demo aims each set-down at the next one.")
    p.add_argument("--no-save", action="store_true", help="show: do not write the scoreboard to ~/arm_data/show")
    p.add_argument("--hand-rel", action="store_true",
                   help="grasp, show: grab where the gripper camera sees the block, without the slack correction "
                        "(slack from the tower's arm tags alone when it does not see it)")
    p.add_argument("--film", action="store_true",
                   help="show: film the cameras (2 frames a second each) into ~/arm_data/film/<time> for the whole show")
    p.add_argument("--hover-base-closed", action="store_true",
                   help="grasp, show: turn the base toward the block and to the hover closed loop (the camera-checked "
                        "nudges), the old behaviour. Default: open loop there; the base is closed-loop only just before "
                        "going in, from where the close look measured it")
    p.add_argument("--final-look", action="store_true",
                   help="show: after the last round, find where the block ended up (search if need be)")
    p.add_argument("--pan-overlap", action="store_true",
                   help="grasp, show: move the arm while the base turns (to the hover, show's carry, to the park "
                        "pose), each pair checked at every arm point and base angle first (check_pan_move); refused: "
                        "one after the other, as without it")
    p.add_argument("--stay-up", action="store_true",
                   help="show: between rounds wait up high at the home base angle instead of going down to home "
                        "(the next round finds the block and looks from there); home at the end")
    p.add_argument("--rest", type=float, default=20.0, help="demo: seconds to rest between rounds (servos cool down)")
    p.add_argument("--loop", action="store_true", help="where: keep updating")
    p.add_argument("--log", action="store_true", help="track: save the estimates to ~/arm_data/track_<time>.jsonl")
    p.add_argument("--minutes", type=float, default=15.0, help="survey: how long to collect (rests not counted)")
    p.add_argument("--chunk", type=float, default=10.0, help="survey: minutes of work between rests")
    p.add_argument("--rest-min", type=float, default=4.0, help="survey: minutes to rest limp between chunks")
    p.add_argument("--long-rest-min", type=float, default=10.0, help="survey: the longer limp rest...")
    p.add_argument("--roll-full-mm", type=float, default=None, help="survey/safetyrun: clearance to the table (mm) above which the wrist rotates its full range (default 40)")
    p.add_argument("--long-every", type=int, default=3, help="survey: ...after every this many chunks")
    p.add_argument("--shoulder-range", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                   help="survey: only poses with the shoulder in this range (slider), to fill in one stretch of it")
    p.add_argument("--heat-guard", choices=("on", "off"), default="on", help="survey: rest when the Pi (or shoulder) runs hot")
    p.add_argument("--shoulder-temp", action="store_true", help="survey: use the shoulder sensor (TMP36, firmware 4.19)")
    p.add_argument("--pi-eco", type=float, default=80.0, help="survey: Pi deg C: slow the cameras between poses")
    p.add_argument("--pi-limp", type=float, default=81.0, help="survey: Pi deg C: stop and rest limp...")
    p.add_argument("--pi-resume", type=float, default=71.0, help="survey: ...until the Pi is this cool")
    p.add_argument("--shoulder-hot", type=float, default=55.0, help="survey: shoulder deg C: stop and rest limp...")
    p.add_argument("--shoulder-resume", type=float, default=42.0, help="survey: ...until the shoulder is this cool")
    args = p.parse_args()
    if args.speed is None and args.mode != "show":          # show picks its own (show.SHOW_SPEED)
        args.speed = SURVEY_SPEED if args.mode == "survey" else RUN_SPEED
    S.esp_url, S.cam_url, S.data_dir = args.esp.rstrip("/"), args.cam.rstrip("/"), os.path.expanduser(args.data)
    S.confirm_left = args.confirm
    S.exact_table = args.exact_table
    S.base_closed = args.mode in CLOSED_BASE_MODES and not args.open_base
    S.pan_dps = args.dps
    if link.setup(args.link) == "cable":
        print("(ESP32 over the cable)")
    if args.margin < 20:
        sys.exit("A margin under 20 mm is not allowed: the map itself is only good to a few mm.")
    S.hunt_wrist = getattr(args, "hunt", "both")
    S.pause_hover = bool(getattr(args, "pause_hover", False))
    S.hand_rel = bool(getattr(args, "hand_rel", False))
    if args.table_margin is not None:
        if not 15 <= args.table_margin <= args.margin:
            sys.exit("--table-margin must be between 15 and the main margin.")
        S.table_margin_mm = args.table_margin

    def on_ctrl_c(signum, frame):
        if not S.moving:
            print("\n")
            sys.exit(0)
        ok = arm_stop()
        restore_speed()
        print("\n\nStopped by Ctrl+C. " + ("The arm is holding where it is." if ok else
              "COULD NOT REACH THE ESP32 TO STOP IT - use the STOP button on the arm page."))
        sys.exit(1)
    signal.signal(signal.SIGINT, on_ctrl_c)

    try:
        modes[args.mode](args)
        restore_speed()
    except ArmError as e:
        if S.moving:
            arm_stop()
        restore_speed()
        print("\n" + str(e))
        sys.exit(2)
