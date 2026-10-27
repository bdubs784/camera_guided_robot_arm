#!/usr/bin/env python3
"""
arm_probe.py: measurements for the smooth-motion program. Data first: nothing in the
firmware's motion changes until these numbers say what to change.

    python3 arm_probe.py wifi                       how long the ESP32 takes to answer (WiFi, and a cable if given)
    python3 arm_probe.py lag                        time from a move command until each camera sees the hand move
    python3 arm_probe.py step --quick               a 2-minute first pass of step responses
    python3 arm_probe.py step                       the full step-response run (about 20-30 min, runs alone)
    python3 arm_probe.py step --profiles 0,1 --ms 600   A/B test: today's moves vs the v4.7 smooth moves
    python3 arm_probe.py timing --reps 3             does the ESP32 finish moves in the time asked? (old vs clock-timed)
    python3 arm_probe.py report <session folder>    print the numbers of a finished session again

Every run saves everything raw in ~/arm_data/sessions/<date_time>_<mode>/ :
    meta.json       versions, settings, what was asked
    frames_<cam>.jsonl  every detected camera frame (tags with corners, capture time, sharpness, motion)
    moves.jsonl     every command sent, with the Pi's and the ESP32's clock
    summary.json    the numbers printed at the end

Needs the camera program v4 (for /frames) and works best with firmware v4.7 (for /tel, /ver, id=, prof=).
With older versions it still runs, with less exact timing.
"""
import argparse
import json
import math
import os
import signal
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

import armlab
from armlab.config import HAND_TAG, MARGIN_MM, RUN_SPEED, STILL_MM
from armlab.errors import ArmError
from armlab.hardware import arm_angles, arm_stop, http_json
from armlab.motion import check_ready, restore_speed, travel
from armlab.safety import Zone, pose_text
from armlab.state import S

LAG_STEP_DEG = 3.0          # lag test: wrist step size
LAG_REPS = 10
MOVED_MM = 0.5              # the hand counts as "moving" once it is this far from where it rested
MOVED_PX = 1.5              # the same, in pixels, when a camera has no mm
SETTLED_MM = 0.5            # "settled" = stays within this of where it ends up
HOLD_S = float(os.environ.get("PROBE_HOLD_S", 3.0))   # how long each step is filmed
PRE_S = float(os.environ.get("PROBE_PRE_S", 0.4))     # still time recorded before each command


# ---------------- small helpers ----------------
def get_json(url, timeout=2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def firmware_info():
    """{"fw": "4.7", ...} on firmware 4.7+, None on older firmware."""
    try:
        v = get_json(S.esp_url + "/ver")
        return v if v.get("fw") else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def stats_ms(xs):
    xs = sorted(xs)
    if not xs:
        return "no data"
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return (f"median {statistics.median(xs):.1f} ms   95% {q(0.95):.1f} ms   99% {q(0.99):.1f} ms   "
            f"worst {xs[-1]:.1f} ms   ({len(xs)} samples)")


class Session:
    """A folder for one run. Every file is plain JSON lines, written as it happens."""

    def __init__(self, what, args):
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        self.path = os.path.join(S.data_dir, "sessions", f"{stamp}_{what}")
        os.makedirs(self.path, exist_ok=True)
        self.files, self.lock = {}, threading.Lock()
        meta = {"what": what, "started": time.time(), "args": vars(args), "esp": S.esp_url, "cam": S.cam_url,
                "firmware": firmware_info(), "arm_lab_mtime": os.path.getmtime(armlab.__file__),
                "calib_files": sorted(f for f in os.listdir(os.path.expanduser("~")) if f.startswith("calib_"))}
        json.dump(meta, open(os.path.join(self.path, "meta.json"), "w"), indent=1)

    def write(self, name, obj):
        with self.lock:
            f = self.files.get(name)
            if f is None:
                f = self.files[name] = open(os.path.join(self.path, name), "a")
            f.write(json.dumps(obj) + "\n")
            f.flush()

    def close(self):
        with self.lock:
            for f in self.files.values():
                f.close()
            self.files = {}


class FrameRecorder:
    """
    Pulls every detected frame from the camera program in the background and writes it to the session.
    Uses /frames (camera program v4). With the old program it falls back to /tags (newest frame only).
    """

    def __init__(self, session, period=0.15):
        self.s, self.period = session, period
        self.since = {}
        self.stop_flag = False
        self.latest = {}
        self.lock = threading.Lock()
        try:
            get_json(S.cam_url + "/frames?since=999999999")
            self.mode = "frames"
        except (urllib.error.URLError, OSError, ValueError):
            self.mode = "tags"
            print("   (old camera program: only the newest frame is recorded. Install camera program v4 for every frame.)")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        last_n = {}
        while not self.stop_flag:
            try:
                if self.mode == "frames":
                    names = list(self.since) or list(get_json(S.cam_url + "/tags").keys())
                    for cam in names:
                        since = self.since.get(cam, None)
                        if since is None:                       # start from now, not from old frames
                            r = get_json(S.cam_url + f"/frames?cam={cam}&since=999999999")
                            self.since[cam] = r.get(cam, {}).get("last", 0)
                            continue
                        r = get_json(S.cam_url + f"/frames?cam={cam}&since={since}").get(cam, {})
                        if r.get("last", since) < since:        # the camera program restarted or was reset
                            self.since[cam] = 0
                            continue
                        for fr in r.get("frames", []):
                            fr["cam"] = cam
                            self.s.write(f"frames_{cam}.jsonl", fr)
                            self.since[cam] = max(self.since[cam], fr["n"])
                            with self.lock:
                                self.latest[cam] = fr
                else:
                    t_pi = time.time()
                    for cam, st in get_json(S.cam_url + "/tags").items():
                        n = st.get("frames_checked")
                        if n == last_n.get(cam):
                            continue
                        last_n[cam] = n
                        fr = {"cam": cam, "n": n, "t_cap": st.get("t_cap", t_pi), "t_det": t_pi, "tags": st.get("tags", []),
                              "sharp": st.get("sharp"), "motion": st.get("motion")}
                        self.s.write(f"frames_{cam}.jsonl", fr)
                        with self.lock:
                            self.latest[cam] = fr
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(self.period if self.mode == "frames" else 0.02)

    def newest(self, cam):
        with self.lock:
            return self.latest.get(cam)

    def stop(self):
        self.stop_flag = True
        self.thread.join(timeout=2)


def send(cmd, sess, info):
    """Send one move command, log it with both clocks. Returns the ESP32's reply."""
    S.moving = True
    t_send = time.time()
    reply = http_json(S.esp_url + cmd, timeout=2.0, tries=1)
    t_reply = time.time()
    rec = dict(info, cmd=cmd, t_send=t_send, t_reply=t_reply, esp_t=reply.get("t"), reply=reply)
    sess.write("moves.jsonl", rec)
    return rec


def wait_until_quiet(rec, cam="torre", need=0.4, timeout=4.0):
    """Wait until the camera's motion score has been low for `need` seconds (or the tag is still)."""
    t_end, quiet_since, last_pos = time.time() + timeout, None, None
    while time.time() < t_end:
        fr = rec.newest(cam)
        still = False
        if fr is not None:
            tag = next((t for t in fr.get("tags", []) if t["id"] == HAND_TAG), None)
            if tag is not None:
                p = tag.get("pos_mm") or tag["center"]
                still = last_pos is not None and math.dist(p, last_pos) < (STILL_MM if "pos_mm" in tag else 1.0)
                last_pos = p
            elif fr.get("motion") is not None:
                still = fr["motion"] < 0.5
        if still:
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since >= need:
                return True
        else:
            quiet_since = None
        time.sleep(0.05)
    return False


# ---------------- analysis (also used by "report") ----------------
def load_frames(path, cam):
    out = []
    p = os.path.join(path, f"frames_{cam}.jsonl")
    if os.path.exists(p):
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    out.sort(key=lambda f: f.get("t_cap") or 0)
    return out


def hand_track(frames):
    """[(t_cap, position)] of the hand tag: mm if the camera gives mm, else pixel centre."""
    tr, unit = [], None
    for f in frames:
        tag = next((t for t in f.get("tags", []) if t["id"] == HAND_TAG), None)
        if tag is None:
            continue
        if "pos_mm" in tag:
            tr.append((f["t_cap"], np.array(tag["pos_mm"], float)))
            unit = "mm"
        else:
            tr.append((f["t_cap"], np.array(tag["center"], float)))
            unit = unit or "px"
    return tr, unit or "px"


def smooth(points, k=3):
    """Running median over k readings, per axis: takes out single-frame tag noise without hiding real motion."""
    arr = np.array(points)
    if len(arr) < k:
        return arr
    h = k // 2
    out = arr.copy()
    for i in range(len(arr)):
        out[i] = np.median(arr[max(0, i - h):i + h + 1], axis=0)
    return out


def analyse_move(track, unit, t_send, hold=HOLD_S):
    """Start lag, settle time, overshoot and noise for one step, from the hand-tag track."""
    before = [p for t, p in track if t_send - PRE_S <= t < t_send]
    after = [(t, p) for t, p in track if t_send <= t <= t_send + hold]
    if len(before) < 2 or len(after) < 8:
        return None
    start = np.median(np.array(before), axis=0)
    ts = [t for t, p in after]
    ps = smooth([p for t, p in after])
    tail_idx = [i for i, t in enumerate(ts) if t >= t_send + hold - 0.6]
    if len(tail_idx) < 2:
        return None
    final = np.median(ps[tail_idx], axis=0)
    noise = float(np.std(np.array(before), axis=0).max())      # how much the reading jitters while the arm is still
    moved_thr = MOVED_MM if unit == "mm" else MOVED_PX
    t_move = next((t for t, p in zip(ts, ps) if np.linalg.norm(p - start) > moved_thr), None)
    if t_move is None:
        return {"moved": False}
    dist = float(np.linalg.norm(final - start))
    u = (final - start) / dist if dist > 1e-6 else None
    over = max((float((p - final) @ u) for t, p in zip(ts, ps) if t >= t_move), default=0.0) if u is not None else 0.0
    band = max(SETTLED_MM if unit == "mm" else MOVED_PX, 3 * noise)    # "settled" line sits above the noise
    late = [t for t, p in zip(ts, ps) if np.linalg.norm(p - final) > band]
    t_settle = (late[-1] - t_send) if late else (t_move - t_send)
    one = 1.0 if unit == "mm" else 3 * MOVED_PX
    late1 = [t for t, p in zip(ts, ps) if np.linalg.norm(p - final) > one]
    t_1 = (late1[-1] - t_send) if late1 else (t_move - t_send)
    return {"moved": True, "start_lag_ms": round((t_move - t_send) * 1000, 1), "settle_s": round(t_settle, 3),
            "within1_s": round(t_1, 3), "overshoot": round(max(over, 0.0), 2), "travel": round(dist, 2),
            "tail_jitter": round(noise, 3), "unit": unit}


def report(path, quiet=False):
    frames = load_frames(path, "torre")
    track, unit = hand_track(frames)
    moves = [json.loads(l) for l in open(os.path.join(path, "moves.jsonl"))] if os.path.exists(os.path.join(path, "moves.jsonl")) else []
    rows = []
    for m in moves:
        if not m.get("analyse"):
            continue
        r = analyse_move(track, unit, m["t_send"], m.get("hold", HOLD_S))
        if r and r.get("moved"):
            rows.append(dict(r, joint=m.get("joint"), size=m.get("size"), ms=m.get("ms"), prof=m.get("prof", 0),
                             pose=m.get("pose"), dir=m.get("dir")))
    summary = {"moves": len(rows), "unit": unit, "groups": []}
    keys = sorted({(r["joint"], r["size"], r["ms"], r["prof"]) for r in rows}, key=str)
    if not quiet:
        print(f"\nSession {os.path.basename(path)}: {len(rows)} analysed moves, hand tag in {unit}.")
        print(f"{'joint':8s} {'size':>5s} {'ms':>5s} {'prof':>4s} {'n':>3s} {'start lag':>10s} {'settle':>8s} "
              f"{'within 1':>9s} {'overshoot':>10s} {'noise':>8s}")
    for k in keys:
        g = [r for r in rows if (r["joint"], r["size"], r["ms"], r["prof"]) == k]
        med = lambda f: statistics.median([r[f] for r in g])
        grp = {"joint": k[0], "size": k[1], "ms": k[2], "prof": k[3], "n": len(g), "start_lag_ms": med("start_lag_ms"),
               "settle_s": med("settle_s"), "within1_s": med("within1_s"), "overshoot": med("overshoot"),
               "tail_jitter": med("tail_jitter")}
        summary["groups"].append(grp)
        if not quiet:
            print(f"{k[0]:8s} {k[1]:5.1f} {k[2]:5.0f} {k[3]:4d} {len(g):3d} {grp['start_lag_ms']:8.0f}ms "
                  f"{grp['settle_s']:7.2f}s {grp['within1_s']:8.2f}s {grp['overshoot']:8.2f}{unit:>2s} {grp['tail_jitter']:6.2f}{unit:>2s}")
    json.dump(summary, open(os.path.join(path, "summary.json"), "w"), indent=1)
    return summary


# ---------------- modes ----------------
def mode_wifi(args):
    fw = firmware_info()
    url = S.esp_url + ("/tel" if fw else "/setall?a=")
    print(f"Firmware: {fw['fw'] if fw else 'older than 4.7'}.  Asking {url} {args.n} times...")
    rtts, fails = [], 0
    for _ in range(args.n):
        t0 = time.perf_counter()
        try:
            get_json(url, timeout=2.0)
            rtts.append((time.perf_counter() - t0) * 1000)
        except (urllib.error.URLError, OSError, ValueError):
            fails += 1
        time.sleep(0.02)
    print(f"WiFi:   {stats_ms(rtts)}   failed {fails}")
    if args.serial:
        try:
            import serial
        except ImportError:
            sys.exit("pyserial is missing:  sudo apt install python3-serial")
        port = serial.Serial()
        port.port, port.baudrate, port.timeout = args.serial, args.baud, 1.0
        port.dtr = False                        # do NOT reset the ESP32 when the port opens
        port.rts = False
        port.open()
        time.sleep(0.2)
        port.reset_input_buffer()
        crtts, cfails = [], 0
        for _ in range(args.n):
            t0 = time.perf_counter()
            port.write(b"/tel\n")
            got = None
            while time.perf_counter() - t0 < 1.0:
                line = port.readline().decode(errors="ignore").strip()
                if line.startswith("{") and '"tel"' in line:
                    got = line
                    break
            if got:
                crtts.append((time.perf_counter() - t0) * 1000)
            else:
                cfails += 1
        port.close()
        print(f"Cable:  {stats_ms(crtts)}   failed {cfails}")


def mode_lag(args):
    zone = Zone(args.margin)
    check_ready(zone)
    sess = Session("lag", args)
    rec = FrameRecorder(sess)
    here = arm_angles()[:3]
    print(f"Lag test at {pose_text(here)}: {args.reps} small wrist moves of {LAG_STEP_DEG:.0f} deg.")
    for k in range(args.reps):
        target = list(here)
        target[2] = here[2] + (LAG_STEP_DEG if k % 2 == 0 else 0.0)
        if not zone.contains(target):
            raise ArmError(f"The test move to {pose_text(target)} is not allowed here. Go home first.")
        wait_until_quiet(rec)
        time.sleep(PRE_S)
        send(f"/setall?a=,,{target[2]:.1f},,,&id={k}", sess,
             {"k": k, "joint": "wrist", "size": LAG_STEP_DEG, "ms": 0, "prof": 0, "pose": here, "analyse": True, "hold": 1.5})
        time.sleep(1.5)
    rec.stop()
    sess.close()
    # analyse: torre by the hand tag, other cameras by their motion score
    moves = [json.loads(l) for l in open(os.path.join(sess.path, "moves.jsonl"))]
    print("\nTime from sending the command until the camera saw the hand start to move:")
    for cam in ("torre", "gripper", "webcam"):
        frames = load_frames(sess.path, cam)
        if not frames:
            continue
        lags_cap, lags_seen = [], []
        track, unit = hand_track(frames)
        for m in moves:
            t0 = m["t_send"]
            if cam == "torre" and track:
                r = analyse_move(track, unit, t0, 1.5)
                if r and r.get("moved"):
                    lags_cap.append(r["start_lag_ms"])
                    t_move = t0 + r["start_lag_ms"] / 1000
                    seen = next((f["t_det"] for f in frames if f.get("t_cap") and f["t_cap"] >= t_move - 1e-4), None)
                    if seen:
                        lags_seen.append((seen - t0) * 1000)
                continue
            base = [f["motion"] for f in frames if f.get("motion") is not None and t0 - PRE_S <= f["t_cap"] < t0]
            if len(base) < 2:
                continue
            thr = max(1.0, 4 * statistics.median(base) + 0.5)
            hit = next((f for f in frames if f.get("motion") is not None and t0 <= f["t_cap"] <= t0 + 1.5
                        and f["motion"] > thr), None)
            if hit:
                lags_cap.append((hit["t_cap"] - t0) * 1000)
                lags_seen.append((hit["t_det"] - t0) * 1000)
        if lags_cap:
            print(f"  {cam:8s} captured after {statistics.median(lags_cap):6.0f} ms,   available to the code after "
                  f"{statistics.median(lags_seen) if lags_seen else float('nan'):6.0f} ms   ({len(lags_cap)} moves)")
        else:
            print(f"  {cam:8s} could not see the moves")
    print(f"\nSaved in {sess.path}")


def pick_poses(zone, n, sizes, rng):
    """Poses spread over the safety map where a step of every size, either way, on every joint stays allowed."""
    axes = zone.axes
    cands = []
    for s in axes[0][::2]:
        for e in axes[1][::2]:
            for w in axes[2][::2]:
                p = [float(s), float(e), float(w)]
                if not zone.contains(p) or not zone.path_ok(zone.home, p):
                    continue
                if zone.gap(p) < zone.margin + 15:               # keep away from the edges of the map
                    continue
                ok = True
                for j in range(3):
                    for d in (max(sizes), -max(sizes)):
                        q = list(p)
                        q[j] += d
                        if not zone.contains(q):
                            ok = False
                if ok:
                    cands.append(p)
    if not cands:
        raise ArmError("No pose on the map leaves room for the test steps. Try smaller --sizes.")
    pts = np.array(cands)
    picked = [int(np.argmin(np.abs(pts - np.array(zone.home)).sum(axis=1)))]
    d = np.abs(pts - pts[picked[0]]).max(axis=1)
    while len(picked) < min(n, len(cands)):                      # farthest-point: spread them out
        i = int(np.argmax(d + rng.random(len(d)) * 0.5))
        picked.append(i)
        d = np.minimum(d, np.abs(pts - pts[i]).max(axis=1))
    order, left = [picked[0]], set(picked[1:])                   # visit nearest-first to keep travel short
    while left:
        last = pts[order[-1]]
        nxt = min(left, key=lambda i: np.abs(pts[i] - last).max())
        order.append(nxt)
        left.remove(nxt)
    return [list(pts[i]) for i in order]


def mode_step(args):
    zone = Zone(args.margin)
    check_ready(zone)
    fw = firmware_info()
    profiles = [int(x) for x in args.profiles.split(",")]
    if any(p >= 1 for p in profiles) and not fw:
        raise ArmError("prof 1 or 2 needs firmware v4.7 on the ESP32.")
    if args.quick:
        args.poses, args.sizes, args.ms = 4, "5", "300"
    sizes = [float(x) for x in args.sizes.split(",")]
    mss = [float(x) for x in args.ms.split(",")]
    joints = [j.strip() for j in args.joints.split(",")]
    jidx = {"shoulder": 0, "elbow": 1, "wrist": 2}
    rng = np.random.default_rng(args.seed)
    poses = pick_poses(zone, args.poses, sizes, rng)
    n_moves = len(poses) * len(joints) * len(sizes) * len(mss) * len(profiles) * 2
    print(f"Step responses: {len(poses)} poses x {len(joints)} joints x {len(sizes)} sizes x {len(mss)} speeds x "
          f"{len(profiles)} profiles x out-and-back = {n_moves} moves, about {n_moves * (HOLD_S + 0.8) / 60:.0f} min "
          "plus travel.\nPress START on the controller so it is disarmed. Ctrl+C stops the arm.")
    sess = Session("step", args)
    rec = FrameRecorder(sess)
    k = 0
    try:
        for pi, pose in enumerate(poses):
            print(f"\npose {pi + 1}/{len(poses)}: {pose_text(pose)}")
            travel(pose, zone, RUN_SPEED, slow_finish=True)
            wait_until_quiet(rec, need=0.6, timeout=5)
            fr = rec.newest("torre")
            if not fr or not any(t["id"] == HAND_TAG for t in fr.get("tags", [])):
                print("   hand tag not visible from the tower here: skipped")
                sess.write("moves.jsonl", {"skip": True, "pose": pose, "why": "hand tag not visible"})
                continue
            for joint in joints:
                for size in sizes:
                    sign = 1 if rng.random() < 0.5 else -1          # same step for every speed and profile (fair A/B)
                    for ms in mss:
                        for prof in profiles:
                            out = list(pose)
                            out[jidx[joint]] += size * sign
                            if not zone.contains(out) or not zone.path_ok(pose, out):
                                continue
                            for leg, (frm, to) in enumerate(((pose, out), (out, pose))):
                                wait_until_quiet(rec, need=0.3, timeout=3)
                                time.sleep(PRE_S)
                                a = ",".join(f"{v:.1f}" for v in to) + ",,,"
                                extra = f"&ms={ms:.0f}" if ms > 0 else ""
                                if prof >= 1:
                                    extra += f"&prof={prof}"
                                    if prof == 2:
                                        extra += f"&f={args.f}&z={args.z}"
                                send(f"/setall?a={a}{extra}&id={k}", sess,
                                     {"k": k, "pose": pose, "joint": joint, "size": size, "ms": ms, "prof": prof,
                                      "dir": "out" if leg == 0 else "back", "from": frm, "to": to, "analyse": True, "hold": HOLD_S})
                                k += 1
                                time.sleep(HOLD_S)
                            print(f"   {joint} {size:.0f} deg, {ms:.0f} ms, prof {prof}: done", flush=True)
        travel(zone.home, zone, RUN_SPEED)
    finally:
        rec.stop()
        sess.close()
    report(sess.path)
    print(f"\nSaved in {sess.path}")



def mode_timing(args):
    """
    Does the ESP32 finish a move in the time it was asked? Watches the ESP32's own commanded position (/tel),
    so the servos, springs and cameras play no part. Compares today's moves (prof 0) with the clock-timed ones (prof 1).
    """
    if not firmware_info():
        raise ArmError("This test needs firmware v4.7 on the ESP32.")
    zone = Zone(args.margin)
    check_ready(zone)
    here = arm_angles()[:3]
    j, size, ms = 2, 5.0, 300.0
    out = list(here)
    out[j] += size
    if not zone.contains(out):
        out[j] = here[j] - size
    if not zone.contains(out):
        raise ArmError("No room for a 5 degree wrist move here. Go home first:  python3 arm.py go")
    print(f"Wrist moves of {size:.0f} deg asked to take {ms / 1000:.1f} s, at {pose_text(here)}. Watching the ESP32's own clock...")
    sess = Session("timing", args)
    results = {0: [], 1: []}
    for rep in range(args.reps):
        for prof in (0, 1):
            for to in (out, here):
                time.sleep(0.6)
                a = ",".join(f"{v:.1f}" for v in to) + ",,,"
                extra = f"&ms={ms:.0f}" + (f"&prof={prof}" if prof else "")
                rec = send(f"/setall?a={a}{extra}&id={rep}", sess, {"prof": prof, "to": to, "timing": True})
                t_end, done = time.time() + 5.0, None
                while time.time() < t_end:
                    tel = get_json(S.esp_url + "/tel", timeout=1.0)
                    if not tel["mv"][j] and abs(tel["p"][j] - to[j]) < 0.05:
                        done = (tel["t"] - rec["esp_t"]) / 1e6 if rec.get("esp_t") else time.time() - rec["t_send"]
                        break
                    time.sleep(0.05)
                results[prof].append(done)
                sess.write("moves.jsonl", {"timing_result": done, "prof": prof})
    sess.close()
    for prof, name in ((0, "today's moves (prof 0)"), (1, "clock-timed moves (prof 1)")):
        ok = [d for d in results[prof] if d is not None]
        if ok:
            print(f"  {name:28s} finished after {statistics.median(ok):.2f} s  (asked {ms / 1000:.2f} s, {len(ok)} moves, "
                  f"slowest {max(ok):.2f} s)")
        else:
            print(f"  {name:28s} never reported finishing")
    print(f"Saved in {sess.path}")

def mode_report(args):
    if not args.path:
        sys.exit("Which session?  python3 arm_probe.py report ~/arm_data/sessions/<folder>")
    report(os.path.expanduser(args.path))


def main():
    p = argparse.ArgumentParser(description="Measurements for smooth motion: WiFi, lag, step responses.")
    p.add_argument("mode", choices=["wifi", "lag", "step", "timing", "report"])
    p.add_argument("path", nargs="?", help="report: the session folder")
    p.add_argument("--esp", default=os.environ.get("ARM_ESP", S.esp_url))
    p.add_argument("--cam", default=os.environ.get("ARM_CAM", S.cam_url))
    p.add_argument("--data", default=os.environ.get("ARM_DATA", S.data_dir))
    p.add_argument("--margin", type=float, default=MARGIN_MM)
    p.add_argument("--table-margin", type=float, default=None)
    p.add_argument("--n", type=int, default=200, help="wifi: how many requests")
    p.add_argument("--serial", default=None, help="wifi: also test a cable, e.g. /dev/serial0 or /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=921600, help="wifi --serial: 921600 for the Pi UART, 115200 for USB")
    p.add_argument("--reps", type=int, default=LAG_REPS, help="lag: number of test moves; timing: repeats (use 3)")
    p.add_argument("--quick", action="store_true", help="step: short first pass (4 poses, one size, one speed)")
    p.add_argument("--poses", type=int, default=12, help="step: number of poses")
    p.add_argument("--joints", default="shoulder,elbow,wrist", help="step: which joints")
    p.add_argument("--sizes", default="3,8", help="step: step sizes in degrees")
    p.add_argument("--ms", default="150,600", help="step: move durations in ms (0 = the ESP32's full speed)")
    p.add_argument("--profiles", default="0", help="step: 0 = today's moves, 1 = smooth (v4.7), 2 = smooth + wobble cancel")
    p.add_argument("--f", type=float, default=2.0, help="step --profiles 2: wobble frequency, Hz")
    p.add_argument("--z", type=float, default=0.05, help="step --profiles 2: wobble damping, 0..0.7")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    S.esp_url, S.cam_url, S.data_dir = args.esp.rstrip("/"), args.cam.rstrip("/"), os.path.expanduser(args.data)
    if args.table_margin is not None:
        if not 15 <= args.table_margin <= args.margin:
            sys.exit("--table-margin must be between 15 and the main margin.")
        S.table_margin_mm = args.table_margin

    def on_ctrl_c(signum, frame):
        if S.moving:
            ok = arm_stop()
            restore_speed()
            print("\n\nStopped by Ctrl+C. " + ("The arm is holding where it is." if ok else
                  "COULD NOT REACH THE ESP32 TO STOP IT - use the STOP button on the arm page."))
        sys.exit(1)
    signal.signal(signal.SIGINT, on_ctrl_c)

    try:
        {"wifi": mode_wifi, "lag": mode_lag, "step": mode_step, "timing": mode_timing, "report": mode_report}[args.mode](args)
        restore_speed()
    except ArmError as e:
        if S.moving:
            arm_stop()
        restore_speed()
        print("\n" + str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
