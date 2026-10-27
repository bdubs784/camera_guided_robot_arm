"""
survey: the big data collection for the "big solve". The arm visits random safe poses at random base angles and,
at each one, records every tag every camera sees (raw corner pixels, several frames each) with the joint values.
Nothing needs to be in a known place: the solver on the laptop works out the arm's geometry, where each camera
sits, and where each tag is (on the arm or fixed on the table) all at once, from these pictures alone.

It runs in chunks (default 15 minutes) with a rest at home in between, so the servos cool down, and prints a
coverage report after every chunk: for each tag, how many poses each camera saw it in, and how often two or
more cameras saw it at the same time (what the solve needs to tie the cameras together).

    python3 arm.py survey --minutes 45                 # three 15-minute chunks
    python3 arm.py survey --minutes 30 --chunk 10 --rest-min 5

Saved to ~/arm_data/survey/<time>/: poses.jsonl (one line per pose) and meta.json (lens calibrations, tag sizes,
base range, settings). Copy the folder to the laptop for tools/big_solve.py.

Also, all the time (moving or not), motion.jsonl.gz: every camera frame's tags (hand tags with corners, others as
centres) and the ESP32 telemetry ~20 times a second. "pt" = Pi clock; telemetry "t"/"ms" = ESP32 clock. Not used by the solve (frames taken while
moving are blurred and late); it is for tuning the Kalman filter: camera delay, servo lag, gear-play flips.
"""
import gzip
import json
import os
import random
import threading
import time

import numpy as np

from ..config import (HOME_TAG, JOINT_NAMES, MOVING, ROLL_RANGE, ROLL_SLIDER, SCAN_FILE, SURVEY_BASE_DEFAULT, SURVEY_FRAMES,
                      TAG_MM)
from ..errors import ArmError
from ..hardware import all_cameras, arm_angles, http_json
from ..motion import base_ready, check_ready, move_base, move_to, travel
from ..safety import Zone
from ..state import S

CAMERAS = ("torre", "gripper", "webcam")


# ---------------------------------------------------------------- heat: the Pi and the shoulder servo
def pi_temp():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0
    except (OSError, ValueError):
        return None


def shoulder_temp():
    """Shoulder servo temperature from the TMP36 (firmware 4.19 /tel "st"), or None."""
    try:
        v = http_json(S.esp_url + "/tel", timeout=0.5, tries=1).get("st")
        return None if v is None else float(v)
    except (ArmError, TypeError, ValueError):
        return None


def camera_rate(scale):
    """Ask the camera server to read tags at this share of its full rate (camera server with /rate; else nothing)."""
    try:
        http_json(S.cam_url + f"/rate?scale={scale:g}", timeout=1.0, tries=1)
    except ArmError:
        pass


class Heat:
    """Watches the Pi and (with --shoulder-temp) the shoulder; says when to ease off or rest."""

    def __init__(self, args):
        self.on = getattr(args, "heat_guard", "on") != "off"
        self.use_shoulder = bool(getattr(args, "shoulder_temp", False))
        self.a = args
        self.eco = False

    def readings(self):
        return pi_temp(), (shoulder_temp() if self.use_shoulder else None)

    def text(self):
        p, sh = self.readings()
        return (f"pi {p:.0f}C" if p is not None else "pi ?") + (f"  shoulder {sh:.0f}C" if sh is not None else "")

    def after_pose(self):
        """'rest' (stop and rest limp now), or None. Also switches the cameras' between-pose eco mode."""
        if not self.on:
            return None
        p, sh = self.readings()
        if p is not None and p >= self.a.pi_limp:
            return f"the Pi is at {p:.0f}C"
        if sh is not None and sh >= self.a.shoulder_hot:
            return f"the shoulder is at {sh:.0f}C"
        want_eco = p is not None and p >= self.a.pi_eco
        if want_eco != self.eco:
            self.eco = want_eco
            print(f"\n   (Pi at {p:.0f}C: cameras {'slowed between poses' if want_eco else 'back to full speed'})")
        return None

    def cooled(self):
        p, sh = self.readings()
        ok_p = p is None or p <= self.a.pi_resume
        ok_s = sh is None or sh <= self.a.shoulder_resume
        return ok_p and ok_s, p, sh


def rest_limp(zone, args, motion, minutes, heat=None, why=""):
    """Park in the rest pose, servos off, motion log paused (the cameras go idle): the shoulder and the Pi both cool.
    Rests at least `minutes`; with a heat reason, until the temperatures are back down too. Then wake and go home."""
    from .relax import mode_relax, mode_wake
    import types
    ns = types.SimpleNamespace(teach=False, keep_base=False, margin=args.margin, home=True)
    print(f"   resting limp {minutes:g} min" + (f" ({why})" if why else "") + "...")
    motion.pause()
    try:
        mode_relax(ns)
        t0 = time.time()
        while True:
            time.sleep(10)
            waited = (time.time() - t0) / 60
            if waited < minutes:
                continue
            if heat is None or not heat.on:
                break
            ok, p, sh = heat.cooled()
            if ok:
                break
            print("\r      still cooling: " + heat.text() + f"  ({waited:.0f} min)".ljust(20), end="", flush=True)
        print()
        mode_wake(ns)
        set_grip(GRIP_SURVEYED)                   # the rest pose may hold these elsewhere: back to the travel values
        set_roll(ROLL_SLIDER)
    finally:
        motion.resume()


def mode_survey(args):
    set_roll_full(args)
    zone = Zone(args.margin)
    check_ready(zone)
    base_ready(zone)
    lo, hi = base_range()
    seed = args.seed if args.seed is not None else int(time.time()) % 100000   # new poses every run unless asked
    rng = random.Random(seed)
    out_dir = os.path.join(S.data_dir, "survey", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir)
    json.dump(meta(lo, hi, args, seed), open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    out = open(os.path.join(out_dir, "poses.jsonl"), "w")
    motion = MotionLog(os.path.join(out_dir, "motion.jsonl.gz"))
    motion.start()
    heat = Heat(args)
    band = tuple(args.shoulder_range) if getattr(args, "shoulder_range", None) else None
    print(f"Survey: random safe poses{f', shoulder {band[0]:g} to {band[1]:g} only' if band else ''}, "
          f"base {lo:+.0f} to {hi:+.0f}, {args.minutes:.0f} minutes in chunks of {args.chunk:g} "
          f"with {args.rest_min:g}-minute limp rests ({args.long_rest_min:g} min after every {args.long_every}). "
          f"Saving to {out_dir}")
    print("Heat guard: " + ("off (measuring)" if not heat.on else
          f"Pi eases off at {args.pi_eco:g}C, rests at {args.pi_limp:g}C until {args.pi_resume:g}C"
          + (f"; shoulder rests at {args.shoulder_hot:g}C until {args.shoulder_resume:g}C" if heat.use_shoulder else
             "; shoulder sensor not used (--shoulder-temp to use it)")))
    print("Press START on the controller so it is disarmed. Ctrl+C stops the arm (everything so far is kept).\n")
    t_start = time.time()
    seen = {}                                     # tag -> {camera: poses seen in, "2+": poses seen by two or more}
    t_end = time.time() + args.minutes * 60
    try:
        n, chunk = 0, 0
        while time.time() < t_end:
            chunk += 1
            chunk_end = min(t_end, time.time() + args.chunk * 60)
            print(f"---- chunk {chunk} ----")
            hot = None
            while time.time() < chunk_end:
                n += 1
                camera_rate(1.0)                          # full speed for the pose's pictures
                rec = visit(zone, rng, lo, hi, args.speed, band=band)
                rec["i"] = n
                rec["pi_c"], rec["shoulder_c"] = heat.readings()
                hot = heat.after_pose()
                camera_rate(0.35 if heat.eco else 1.0)    # between poses: slowed only when the Pi runs hot
                out.write(json.dumps(rec) + "\n")
                out.flush()
                count(seen, rec)
                tags = sorted({t["id"] for c in rec["cams"].values() for f in c for t in f})
                print(f"\r  pose {n}: base {rec['base']:+6.1f}  rotate {rec['joints'][3]:5.1f}  grip {rec['joints'][4]:5.1f}  "
                      f"{len(tags)} tags  {heat.text()}  ({(time.time() - t_start) / 60:.1f} min)".ljust(90), end="", flush=True)
                if hot:
                    break
            print()
            report(seen, n)
            if time.time() < t_end or hot:
                long = chunk % args.long_every == 0
                mins = args.long_rest_min if long else args.rest_min
                t_rest = time.time()
                rest_limp(zone, args, motion, mins, heat, why=hot or ("long break" if long else ""))
                t_end += time.time() - t_rest             # rests do not count towards the survey time
    finally:                                      # Ctrl+C or an error: still save everything cleanly
        out.close()
        motion.stop()
        camera_rate(1.0)
        try:
            set_grip(GRIP_SURVEYED)               # never leave the gripper turned or opened
            set_roll(ROLL_SLIDER)
        except Exception:
            pass
        import sys as _sys
        if not isinstance(_sys.exc_info()[1], (KeyboardInterrupt, SystemExit)):   # finished, or failed: rest limp
            try:                                                                   # (Ctrl+C: hold where it is)
                from .relax import mode_relax
                import types
                mode_relax(types.SimpleNamespace(teach=False, keep_base=False, margin=args.margin, home=True))
            except Exception as e:
                print(f"\n   could not relax the arm ({e}). Run:  python3 arm.py relax")
    print(f"\nDone: {n} poses in {out_dir}\nCopy it to the laptop:  scp -r pi@raspberrypi:{out_dir} .")


HAND_TAGS = {9, 31, 32, 241} | set(range(280, 312)) | set(range(320, 352)) | set(range(360, 380))
# ^ tags on the arm (hand + the wedges): their corners are kept in the motion log (the arm filter measures the arm with them)


class MotionLog(threading.Thread):
    """Records every new camera frame (tags) and the ESP32's telemetry, ~20 times a second, until stop()."""

    def __init__(self, path, period=0.05):
        super().__init__(daemon=True)
        self.path, self.period, self.halt = path, period, threading.Event()
        self.paused = threading.Event()
        self.frames = self.tels = 0

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()

    def stop(self):
        self.halt.set()
        self.join(timeout=3.0)
        print(f"   motion log: {self.frames} camera frames, {self.tels} telemetry readings -> {self.path}")

    def run(self):
        last = {}
        with gzip.open(self.path, "wt") as f:
            while not self.halt.is_set():
                if self.paused.is_set():                  # a rest: no polling, so the camera server goes idle
                    time.sleep(0.5)
                    continue
                t0 = time.time()
                try:
                    tel = http_json(S.esp_url + "/tel", timeout=0.5, tries=1)
                    f.write(json.dumps({"k": "tel", "pt": round(t0, 4), **{k: tel[k] for k in
                            ("t", "ms", "p", "tg", "mv", "b", "bt", "br", "st") if k in tel}}) + "\n")
                    self.tels += 1
                except (ArmError, KeyError, TypeError):
                    pass
                try:
                    cams = all_cameras()
                except ArmError:
                    cams = {}
                for c in CAMERAS:
                    info = cams.get(c)
                    if not isinstance(info, dict):
                        continue
                    fr = info.get("frames_checked")
                    if fr is None or fr == last.get(c):
                        continue
                    last[c] = fr
                    tags = []
                    for t in info.get("tags", []):
                        row = {"id": t.get("id"), "center": t.get("center")}
                        if (t.get("id") in HAND_TAGS or c == "gripper") and t.get("corners"):   # gripper: all corners
                            row["corners"] = t["corners"]
                        tags.append(row)
                    f.write(json.dumps({"k": "cam", "c": c, "pt": round(time.time(), 4), "f": fr,
                                        "t_cap": info.get("t_cap") or next((t.get("t_cap") for t in info.get("tags", [])
                                                                            if t.get("t_cap")), None),
                                        "tags": tags}) + "\n")
                    self.frames += 1
                if self.frames % 100 == 0:
                    f.flush()
                time.sleep(max(0.0, self.period - (time.time() - t0)))


def base_range():
    try:
        sc = json.load(open(os.path.join(S.data_dir, SCAN_FILE)))
        return float(sc["base_lo"]), float(sc["base_hi"])
    except (OSError, ValueError, KeyError):
        return -SURVEY_BASE_DEFAULT, SURVEY_BASE_DEFAULT


LOOK_SHARE = 0.5               # this share of the poses are near a taught look pose: the gripper camera sees the table
LOOK_JITTER_DEG = 12.0


def look_pose(zone, rng):
    """A pose near one of the taught look poses (scan --teach), so the gripper camera looks at the table. Or None."""
    try:
        poses = json.load(open(os.path.join(S.data_dir, SCAN_FILE)))["poses"]
    except (OSError, ValueError, KeyError):
        return None
    for _ in range(200):
        base = rng.choice(poses)
        p = [round(base[k] + rng.uniform(-LOOK_JITTER_DEG, LOOK_JITTER_DEG), 1) for k in range(3)]
        if zone.contains(p) and zone.path_ok(zone.home, p):
            return p
    return None


SURVEY_ELBOW_MAX = 120.0  # above ~125 the elbow spring hits itself and holds the elbow ~6.6 steps short (a partial
                          # stall that heats the servo and is not gear play). Survey poses and their approaches stay below.


SURVEY_TABLE_MM = 55.0    # survey poses keep this much map clearance to the table (map 40 mm was ~13-33 mm real)
GRIP_CLOSE_TABLE_MM = 75.0  # closing past GRIP_CLOSE_MAX pushes the fingers ~20 mm forward: only with this much room
GRIP_CLOSE_MAX = 100.0


def table_gap(zone, pose):
    return zone.detail(pose)["base"] + zone.table_extra


def survey_table_mm(zone):
    """How much table clearance survey poses keep, on the same scale as table_gap. The map's 55 was a guess at its
    own error; the exact check (--exact-table) is measured, so there it is just the table margin."""
    return SURVEY_TABLE_MM if zone.exact is None else zone.margin


EXACT_TRIES = 30            # --exact-table: tries for a base angle / rotate + gripper that the exact check allows
EXACT_STEPS = 6             # points checked along each hand move


def exact_ok(zone, pose, base, hands):
    """--exact-table: the arm at pose (shoulder, elbow, wrist) and base, with the hand going through `hands`
    [(rotate, gripper), ...] in straight steps, keeps the table margin all the way."""
    if zone.exact is None:
        return True
    need = zone.table_margin
    for (r0, g0), (r1, g1) in zip(hands[:-1], hands[1:]):
        for t in np.linspace(0.0, 1.0, EXACT_STEPS):
            q = list(pose[:3]) + [r0 + (r1 - r0) * t, g0 + (g1 - g0) * t, base]
            if zone.exact.gaps(q[:5], q[5])[0] < need:
                return False
    return True


def exact_base(zone, target, lo, hi, rng, now_base):
    """--exact-table: a base angle where the arm at target (hand level, travel grip) keeps the table margin, checked
    every few degrees of the turn there. None if no try works."""
    travel = [(ROLL_SLIDER, GRIP_SURVEYED)] * 2
    for _ in range(EXACT_TRIES):
        b = round(rng.uniform(lo, hi), 1)
        if all(exact_ok(zone, target, x, travel) for x in np.linspace(now_base, b, max(2, int(abs(b - now_base) / 5) + 2))):
            return b
    return None


def exact_hand(zone, target, base, rng):
    """--exact-table: (roll, roll_from, grip, grip_from) for this pose: the full rotate and gripper ranges, each
    arriving from a random side, wherever the exact check allows the whole hand move (level and 80 -> from -> there ->
    back). Falls back to level and 80 (None, None, None, None)."""
    for _ in range(EXACT_TRIES):
        roll, roll_from = pick_roll(zone, target, rng, full=True)
        grip, grip_from = pick_grip(rng)
        r, rf = (ROLL_SLIDER, ROLL_SLIDER) if roll is None else (roll, roll_from)
        g, gf = (GRIP_SURVEYED, GRIP_SURVEYED) if grip is None else (grip, grip_from)
        home = (ROLL_SLIDER, GRIP_SURVEYED)
        if exact_ok(zone, target, base, [home, (rf, gf), (r, g), home]):
            return roll, roll_from, grip, grip_from
    return None, None, None, None


def survey_pose(zone, rng, band=None):
    """A random survey pose (half of them near a taught look pose) with the elbow at or below SURVEY_ELBOW_MAX.
    band: (lo, hi) to keep the shoulder inside, for surveying one stretch of it (--shoulder-range). The taught look
    poses sit where they sit, so a band skips them and takes random poses only."""
    for _ in range(500):
        look = look_pose(zone, rng) if (band is None and rng.random() < LOOK_SHARE) else None
        target = look or zone.random_pose(rng)
        if band is not None and not (band[0] <= target[0] <= band[1]):
            continue
        if target[1] <= SURVEY_ELBOW_MAX and table_gap(zone, target) >= survey_table_mm(zone):
            return target
    raise ArmError("could not find a survey pose with the elbow under its cap"
                   + (f" and the shoulder in {band[0]:g}-{band[1]:g}" if band else ""))


def survey_approach(zone, target, rng, band=None):
    """approach_pose, but never past the elbow cap (None = arrive straight). band: keep the shoulder inside it, so a
    banded survey still arrives from both sides without leaving the stretch it is measuring."""
    for _ in range(20):
        away = zone.approach_pose(target, rng)
        if away is None:
            return None
        if band is not None and not (band[0] <= away[0] <= band[1]):
            continue
        if away[1] <= SURVEY_ELBOW_MAX and table_gap(zone, away) >= survey_table_mm(zone):
            return away
    return None


def visit(zone, rng, lo, hi, speed, tries=6, band=None):
    """Go to a random safe pose (from a random nearby direction) and base angle, settle, and record every camera.
    A pose the map will not route to from where the arm ended up is skipped (back out the way it came, pick another)."""
    for _ in range(tries):
        target = survey_pose(zone, rng, band)
        away = survey_approach(zone, target, rng, band)
        back = None
        if away is not None:                      # arrive from a random side: the solve learns the gear play too
            try:
                _, back = travel(away, zone, speed)   # back = start of the straight leg it came in on
            except ArmError as e:
                if not str(e).startswith("refused"):
                    raise
        came = arm_angles()[:3]
        try:
            travel(target, zone, speed, slow_finish=True)
        except ArmError as e:                     # refused before moving (the arm stopped a bit off the map's pose)
            if not str(e).startswith("refused"):
                raise
            print(f"\n   (skipped a pose: {str(e).splitlines()[0][9:]})")
            if back is not None:
                move_to(back, zone, speed)        # the straight leg it just came in on is clear both ways
            continue
        if zone.exact is None:
            base = round(rng.uniform(lo, hi), 1)
        else:
            base = exact_base(zone, target, lo, hi, rng, S.base_cmd if S.base_cmd is not None else arm_angles()[5])
            if base is None:                      # nowhere to turn to with the margin kept: stay at this base
                base = round(float(S.base_cmd if S.base_cmd is not None else arm_angles()[5]), 1)
        from_below = True if (not BASE_EITHER_SIDE or base >= hi - 2) else rng.random() < 0.5
        if from_below:
            move_base(base)                       # last few degrees in the + direction (the usual way)
        else:
            move_base(min(hi, base + 2.5), finish_from_below=False)   # overshoot, then come back down to it
            move_base(base, finish_from_below=False)
        if zone.exact is None:
            roll, roll_from = pick_roll(zone, target, rng)
            grip, grip_from = pick_grip(rng, GRIP_RANGE[1] if table_gap(zone, target) >= GRIP_CLOSE_TABLE_MM else GRIP_CLOSE_MAX)
        else:
            roll, roll_from, grip, grip_from = exact_hand(zone, target, base, rng)
        if roll is not None or grip is not None:  # turn and open/close here (never while travelling), both at once,
            set_hand(roll_from, grip_from)        # each arriving from a random side
            set_hand(roll, grip)
        time.sleep(0.6)                           # the arm sways after the base stops
        a = arm_angles()
        rec = {"t": round(time.time(), 2), "cmd": target, "joints": a,
               "came_from": came + [ROLL_SLIDER if roll_from is None else roll_from,
                                    GRIP_SURVEYED if grip_from is None else grip_from],
               "base": base, "base_from": "below" if from_below else "above", "cams": capture(SURVEY_FRAMES)}
        if zone.exact is not None:                # the exact check's table gap at the pose as it really is (logged)
            rec["table_gap_mm"] = round(zone.exact.gaps(a[:5], a[5])[0], 1)
        if roll is not None or grip is not None:  # level hand and usual grip again before the arm moves anywhere
            set_hand(ROLL_SLIDER, GRIP_SURVEYED)
        return rec
    raise ArmError(f"{tries} poses in a row could not be reached from {came}: stopping (everything so far is kept).")


GRIP_SURVEYED = 80.0      # the gripper value every survey before v2 used (and the one the arm travels at)
ROLL_SHARE = 0.9          # share of poses where the wrist rotate is turned (the rest stay level, tying in the old surveys)
ROLL_FULL_TABLE_MM = 40.0 # the map's clear space to the table/plywood at least this: the full rotate range (0-112)
                          # (40 = the map's own minimum, so full rotate everywhere; checked by eye; --roll-full-mm)
ROLL_HALF = (14.0, 70.0)  # otherwise half the range, centred on level (the map never allows less than its 40 mm)
GRIP_SHARE = 0.8          # share of poses where the gripper is moved off the surveys' usual 80
GRIP_RANGE = (40.0, 150.0)  # open wide .. closed (finger gap 65 .. 0 mm); the fingers barely move up or down
BASE_EITHER_SIDE = True   # arrive at the base angle from a random side (its slack differs by side, like the joints')


def pick_roll(zone, pose, rng, full=False):
    """(rotate value, value to arrive from) for this pose, or (None, None) to leave the hand level. The full range
    where the safety map has at least ROLL_FULL_TABLE_MM to the table; half the range otherwise. The real turn is only about -23..+38 deg from level (0.55 deg per step), so
    the fingertips drop at most ~25 mm."""
    if rng.random() > ROLL_SHARE:
        return None, None
    if not full:                                            # clear space to the table/base (the map keeps the fingers
        full = zone.detail(pose)["base"] + zone.table_extra >= ROLL_FULL_TABLE_MM   # off the arm itself)
    lo, hi = ROLL_RANGE if full else ROLL_HALF
    roll = round(rng.uniform(lo, hi), 1)
    side = rng.choice([-1, 1]) * rng.uniform(5, 12)          # arrive from a random side: the solve learns its play
    roll_from = round(float(min(hi, max(lo, roll + side))), 1)
    return roll, roll_from


def pick_grip(rng, hi_cap=None):
    """(gripper value, value to arrive from), or (None, None) to leave it at 80."""
    if rng.random() > GRIP_SHARE:
        return None, None
    lo, hi = GRIP_RANGE
    if hi_cap is not None:
        hi = min(hi, hi_cap)
    g = round(rng.uniform(lo, hi), 1)
    side = rng.choice([-1, 1]) * rng.uniform(5, 12)
    return g, round(float(min(hi, max(lo, g + side))), 1)


def _timed(ms):
    """Clock-timed hand moves (firmware 4.7+): without prof=1 the ESP32's old tick glide ran them ~4x slow
    (grip asked 0.6 s took 2.3 s, rotate asked 0.8 s took 3.1 s)."""
    return f"&ms={int(ms)}" + ("&prof=1" if S.prof else "")


def set_hand(roll=None, grip=None, secs=0.8, wait=True):
    """_set_hand, traced (trace.py)."""
    from .. import trace
    with trace.span("hand", f"rotate {roll} grip {grip}" + ("" if wait else " (no wait)")):
        return _set_hand(roll, grip, secs, wait)


HAND_FAST = False                 # grasp/show set it: hand moves glide in HAND_FAST_S and settle HAND_FAST_SETTLE_S
HAND_FAST_S = 0.3                 # (the default glide plus settle was ~0.7-1.0 s a move, 4.8 s a round of show;
HAND_FAST_SETTLE_S = 0.1          # the servo itself crosses the gripper's range in ~0.2 s)


def _set_hand(roll=None, grip=None, secs=0.8, wait=True):
    """Turn the wrist rotate and/or move the gripper together (None = leave it), and wait for both. wait=False:
    send it and return, so it can happen while the base or the arm moves (call it again with wait to make sure)."""
    if HAND_FAST:
        secs = min(secs, HAND_FAST_S)
    want = {}
    if roll is not None:
        want[3] = float(min(ROLL_RANGE[1], max(ROLL_RANGE[0], roll)))
    if grip is not None:
        want[4] = float(min(GRIP_RANGE[1], max(GRIP_RANGE[0], grip)))
    now = arm_angles()
    want = {j: v for j, v in want.items() if abs(now[j] - v) > 0.6}
    if not want:
        return
    vals = ["", "", "", "", ""]
    for j, v in want.items():
        vals[j] = f"{v:.1f}"
        S.came_from[j] = float(now[j])                   # which way it arrives (slackgp.py)
    http_json(S.esp_url + "/setall?a=" + ",".join(vals) + _timed(secs * 1000))
    if not wait:
        return
    t_end = time.time() + 3 * secs + 3
    while time.time() < t_end and any(abs(arm_angles()[j] - v) > 0.6 for j, v in want.items()):
        time.sleep(0.05)
    time.sleep(HAND_FAST_SETTLE_S if HAND_FAST else 0.25)


def set_grip(value, secs=0.6):
    """Move the gripper to `value` (clamped to GRIP_RANGE) and wait for it."""
    set_hand(grip=value, secs=secs)


def set_roll(value, secs=0.8):
    """Turn the wrist rotate to `value` (clamped to ROLL_RANGE) and wait for it."""
    set_hand(roll=value, secs=secs)

def capture(frames, timeout=3.0):
    """{camera: [frame, ...]}, each frame a list of tags {id, corners, center, side_px, pos_mm?, t_cap?}: new frames only."""
    got = {c: [] for c in CAMERAS}
    last = {c: None for c in CAMERAS}
    t_end = time.time() + timeout
    while time.time() < t_end and any(len(got[c]) < frames for c in CAMERAS):
        try:
            cams = all_cameras()
        except ArmError:
            time.sleep(0.2)
            continue
        for c in CAMERAS:
            info = cams.get(c)
            if not isinstance(info, dict) or len(got[c]) >= frames:
                continue
            f = info.get("frames_checked")
            if f is not None and f == last[c]:
                continue
            last[c] = f
            got[c].append([{k: t[k] for k in ("id", "corners", "center", "side_px", "pos_mm", "t_cap") if k in t}
                           for t in info.get("tags", []) if t.get("corners")])
        time.sleep(0.05)
    return {c: v for c, v in got.items() if v}


def count(seen, rec):
    per_tag = {}
    for cam, frames in rec["cams"].items():
        for tid in {t["id"] for f in frames for t in f}:
            per_tag.setdefault(tid, set()).add(cam)
    for tid, cams in per_tag.items():
        row = seen.setdefault(tid, {"torre": 0, "gripper": 0, "webcam": 0, "2+": 0})
        for c in cams:
            row[c] += 1
        row["2+"] += len(cams) >= 2


def report(seen, n):
    """Coverage so far: which tags each camera saw, and how often two cameras saw the same tag at once."""
    print(f"   coverage after {n} poses (poses each camera saw the tag in; '2+' = seen by two or more at once):")
    print("      tag    tower  gripper  webcam    2+")
    for tid in sorted(seen):
        r = seen[tid]
        print(f"     {tid:4d}   {r['torre']:6d}  {r['gripper']:7d}  {r['webcam']:6d}  {r['2+']:4d}  ({100 * r['2+'] / max(n, 1):.0f}%)")


def meta(lo, hi, args, seed=None):
    try:                                           # the gripper camera's focus now (its lens numbers depend on it)
        focus = http_json(S.cam_url + "/focus", timeout=2.0, tries=1).get("lens_position")
    except ArmError:
        focus = None
    calibs = {}
    for c in CAMERAS:
        try:
            calibs[c] = json.load(open(os.path.expanduser(f"~/calib_{c}.json")))
        except (OSError, ValueError):
            calibs[c] = None
    sizes = dict(TAG_MM)
    try:                                           # sizes you measured override the printed ones
        sizes.update({int(k): float(v) for k, v in json.load(open(os.path.join(S.data_dir, "tag_sizes.json"))).items()})
    except (OSError, ValueError):
        pass
    return {"made": time.strftime("%Y-%m-%d %H:%M"), "base_range": [lo, hi], "calibrations": calibs,
            "tag_mm": {str(k): v for k, v in sizes.items()}, "home_tag": HOME_TAG, "joint_names": JOINT_NAMES,
            "moving": list(MOVING), "slider_range": [list(r) for r in S.slider_range], "margin": args.margin,
            "table_margin": S.table_margin_mm, "frames_per_pose": SURVEY_FRAMES,
            # How fast the arm was driven: a plausible reason two surveys disagree. One survey needed four times
            # the per-pose gear-play correction of the ones before it (shoulder 1.57 vs 0.37) and there was no
            # record of the speed to check. A joint that arrives slowly can stop short of its target on stiction
            # where a faster one carries through, which is exactly this shape of error.
            "speed_dps": args.speed, "base_dps": args.dps, "pan_dps": S.pan_dps,
            "seed": seed, "gripper_focus": focus}


# ---------------------------------------------------------------- rotate sweep
SWEEP_POSES = 6           # high, open poses to sweep the wrist rotate at
SWEEP_STEPS = 8           # rotate values per direction (up through the range, then back down between them)
SWEEP_TABLE_MM = 90.0     # extra clear space (beyond the margin) to the table for a sweep pose...
SWEEP_SELF_MM = 25.0      # ...and to the arm itself and the tower


def sweep_poses(zone, rng, n=SWEEP_POSES):
    """n poses where the hand is far from the table, the tower and the arm, spread apart from each other."""
    for extra in (1.0, 0.7, 0.5):                     # relax a little if the map has few such poses
        ok = []
        for _ in range(3000):
            p = zone.random_pose(rng)
            d = zone.detail(p)
            if d["base"] + zone.table_extra >= zone.margin + SWEEP_TABLE_MM * extra and \
                    d["self"] >= zone.margin + SWEEP_SELF_MM * extra and d["tower"] >= zone.margin + SWEEP_SELF_MM * extra:
                ok.append(p)
        if len(ok) >= n:
            break
    if not ok:
        raise ArmError("The map has no pose open enough to sweep the wrist rotate safely.")
    picked = [ok[rng.randrange(len(ok))]]
    while len(picked) < min(n, len(ok)):              # farthest-point: spread the poses out
        far = max(ok, key=lambda p: min(sum((a - b) ** 2 for a, b in zip(p, q)) for q in picked))
        picked.append(far)
    return picked


def mode_rollsweep(args):
    """
    rollsweep: calibrate the wrist rotate. At a few high, open poses (and random base angles) it steps the rotate up
    through its whole range (0-112) and back down between those values, taking pictures at every step: the solve gets
    big turns from both directions, which pins down the rotate's degrees per step, its axis and its slack. Same folder
    format as a survey, so it is solved together with them:  big_solve.py comb4 <surveys...> <this folder>
        python3 arm.py rollsweep            (about 5 minutes)
    """
    zone = Zone(args.margin)
    check_ready(zone)
    base_ready(zone)
    lo, hi = base_range()
    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    rng = random.Random(seed)
    poses = sweep_poses(zone, rng)
    r_lo, r_hi = ROLL_RANGE
    step = (r_hi - r_lo) / (SWEEP_STEPS - 1)
    up = [r_lo + k * step for k in range(SWEEP_STEPS)]
    down = [v + step / 2 for v in reversed(up[:-1])]        # back down, between the up values
    out_dir = os.path.join(S.data_dir, "survey", time.strftime("%Y%m%d_%H%M%S") + "_rollsweep")
    os.makedirs(out_dir)
    json.dump(meta(lo, hi, args, seed), open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    out = open(os.path.join(out_dir, "poses.jsonl"), "w")
    motion = MotionLog(os.path.join(out_dir, "motion.jsonl.gz"))
    motion.start()
    print(f"Rotate sweep: {len(poses)} open poses, rotate {r_lo:.0f} -> {r_hi:.0f} and back at each "
          f"({len(up) + len(down)} pictures per pose). Saving to {out_dir}")
    print("Press START on the controller so it is disarmed. Ctrl+C stops the arm (everything so far is kept).\n")
    n = 0
    try:
        for k, target in enumerate(poses):
            set_roll(ROLL_SLIDER)                     # level before any travel
            travel(target, zone, args.speed, slow_finish=True)
            base = round(rng.uniform(lo, hi), 1)
            move_base(base)
            came = arm_angles()[:3]
            set_roll(r_lo)                            # start the up sweep from the bottom
            prev = r_lo
            for v in up + down:
                set_roll(v)
                time.sleep(0.3)
                a = arm_angles()
                rec = {"t": round(time.time(), 2), "cmd": target, "joints": a, "came_from": came + [round(prev, 1)],
                       "base": base, "cams": capture(SURVEY_FRAMES), "sweep": k}
                n += 1
                rec["i"] = n
                out.write(json.dumps(rec) + "\n")
                out.flush()
                tags = sorted({t["id"] for c in rec["cams"].values() for f in c for t in f})
                print(f"\r  pose {k + 1}/{len(poses)}  rotate {a[3]:5.1f}  {len(tags)} tags seen".ljust(60), end="", flush=True)
                prev = v
            print()
        set_roll(ROLL_SLIDER)
        travel(zone.home, zone, args.speed)
        move_base(0.0)
    finally:
        out.close()
        motion.stop()
        try:
            set_roll(ROLL_SLIDER)
        except Exception:
            pass
    print(f"\nDone: {n} pictures at {len(poses)} poses in {out_dir}\nCopy it to the laptop:  scp -r pi@raspberrypi:{out_dir} .")


# ---------------------------------------------------------------- supervised safety run
def risky_poses(zone, rng, n_sample=6000):
    """The poses survey v2 could pick that deserve a look: lowest over the table, just at the full-rotate limit,
    most folded (closest to itself), closest to the tower, and the far reach."""
    cands = []
    for _ in range(n_sample):
        p = zone.random_pose(rng)
        if p[1] > SURVEY_ELBOW_MAX or table_gap(zone, p) < SURVEY_TABLE_MM:   # the survey never goes there
            continue
        d = zone.detail(p)
        cands.append((p, d["base"] + zone.table_extra, d["self"], d["tower"]))
    picks = []
    low = min(cands, key=lambda c: c[1])
    full_low = low[1] >= ROLL_FULL_TABLE_MM
    picks.append((f"lowest over the table ({'full' if full_low else 'half'} rotate range)", low))
    at_limit = [c for c in cands if c[1] >= ROLL_FULL_TABLE_MM]
    if at_limit and not full_low:
        picks.append((f"just at the full-rotate limit ({ROLL_FULL_TABLE_MM:g} mm)", min(at_limit, key=lambda c: c[1])))
    picks.append(("most folded (closest to itself)", min(cands, key=lambda c: c[2])))
    picks.append(("closest to the tower", min(cands, key=lambda c: c[3])))
    picks.append(("shoulder far down (long low reach)", min(cands, key=lambda c: c[0][0])))
    picks.append(("shoulder far up", max(cands, key=lambda c: c[0][0])))
    return picks


def set_roll_full(args):
    global ROLL_FULL_TABLE_MM
    v = getattr(args, "roll_full_mm", None)
    if v is not None:
        ROLL_FULL_TABLE_MM = float(v)


def mode_safetyrun(args):
    """
    safetyrun: before a long survey v2, see the extremes once with your own eyes. It goes (slowly) to the riskiest
    poses the survey could pick and at each one, pausing before every step so you can look:
        rotate to both ends of what that pose allows, then the gripper wide open and closed, then back to normal.
    At every prompt: Enter = do it, s = skip this pose, q = stop. Ctrl+C stops the arm at any time.
        python3 arm.py safetyrun
    """
    set_roll_full(args)
    zone = Zone(args.margin)
    check_ready(zone)
    base_ready(zone)
    lo, hi = base_range()
    rng = random.Random(args.seed if args.seed is not None else 7)
    speed = min(args.speed, 15.0)
    picks = risky_poses(zone, rng)
    print(f"Safety run: {len(picks)} poses, moving at {speed:.0f} deg/s. Enter = go on, s = skip pose, q = stop.\n")

    def ask(what):
        a = input(f"      {what}?  [Enter / s / q] ").strip().lower()
        if a == "q":
            raise KeyboardInterrupt
        return a != "s"

    try:
        for k, (why, (pose, table, selfgap, tower)) in enumerate(picks):
            full = table >= ROLL_FULL_TABLE_MM
            r_lo, r_hi = ROLL_RANGE if full else ROLL_HALF
            print(f"  {k + 1}/{len(picks)}  {why}: shoulder {pose[0]:.0f} elbow {pose[1]:.0f} wrist {pose[2]:.0f}  "
                  f"(map: {table:.0f} mm to the table, {selfgap:.0f} to itself, {tower:.0f} to the tower; "
                  f"rotate {r_lo:.0f}-{r_hi:.0f})")
            if not ask("go there"):
                continue
            set_grip(GRIP_SURVEYED)
            set_roll(ROLL_SLIDER)
            travel(pose, zone, speed, slow_finish=True)
            move_base(round(rng.uniform(lo, hi), 1))
            for label, fn, v in (("rotate to " + f"{r_lo:.0f}", set_roll, r_lo), ("rotate to " + f"{r_hi:.0f}", set_roll, r_hi),
                                 ("rotate back to level", set_roll, ROLL_SLIDER), ("gripper wide open (40)", set_grip, 40.0),
                                 ("gripper closed (150)" if table >= GRIP_CLOSE_TABLE_MM else "gripper closed as far as allowed (100)", set_grip,
                                  150.0 if table >= GRIP_CLOSE_TABLE_MM else GRIP_CLOSE_MAX), ("gripper back to 80", set_grip, GRIP_SURVEYED)):
                if label.startswith(("rotate back", "gripper back")):
                    fn(v, secs=1.5)
                    continue
                if not ask(label):
                    continue
                fn(v, secs=1.5)                    # slow
        print("\nDone. Anything that looked close: note which pose and which step.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            set_grip(GRIP_SURVEYED)
            set_roll(ROLL_SLIDER)
            travel(zone.home, zone, speed)
        except Exception:
            pass
