"""
track: the arm filter, live. Reads the ESP32's telemetry and every new camera frame ~20 times a second and keeps
an estimate of each joint's real offset from what it was told (its slack), with how sure it is. Prints the
fingertip with the slack taken out. No movement: drive the arm with the controller or the page while it runs.

    python3 arm.py track                 (Ctrl+C to stop)
    python3 arm.py track --log           also save everything to ~/arm_data/track_<time>.jsonl for checking later
"""
import json
import os
import time
from collections import deque

import numpy as np

from ..errors import ArmError
from ..hardware import all_cameras, http_json
from ..state import S
from ..track import NAMES, ArmFilter
from ..world import World

CAMERAS = ("torre", "gripper", "webcam")
PERIOD = 0.05


def mode_track(args):
    path = os.path.join(S.data_dir, "world.json")
    if not os.path.exists(path):
        raise ArmError(f"No world model at {path}: copy the big solve's world json there first.")
    w = World(path)
    f = ArmFilter(w)
    hist = deque(maxlen=200)                 # (pi time, joints[4], base) telemetry, to look up the joints at t_cap
    last = {}
    log = open(os.path.join(S.data_dir, time.strftime("track_%Y%m%d_%H%M%S.jsonl")), "w") if getattr(args, "log", False) else None
    print("Tracking (Ctrl+C to stop). Offsets = real minus commanded: shoulder/elbow/wrist/rotate in slider steps, base in deg.")
    t_print = 0.0
    try:
        while True:
            t0 = time.time()
            try:
                tel = http_json(S.esp_url + "/tel", timeout=0.5, tries=1)
                joints, base = [float(v) for v in tel["p"][:4]], float(tel["b"])
                hist.append((t0, joints, base, float(tel["p"][4])))
                mv = list(tel.get("mv", [0] * 5)) + [0] * 5
                f.step(t0, mv[:4] + [tel.get("br", 0), mv[4]])   # ... rotate, base, gripper
            except (ArmError, KeyError, TypeError, ValueError):
                pass
            try:
                cams = all_cameras()
            except ArmError:
                cams = {}
            for c in CAMERAS:
                info = cams.get(c)
                if not isinstance(info, dict) or not hist:
                    continue
                fr = info.get("frames_checked")
                if fr is None or fr == last.get(c):
                    continue
                last[c] = fr
                t_cap = info.get("t_cap") or next((t.get("t_cap") for t in info.get("tags", []) if t.get("t_cap")), None) or t0
                h = min(hist, key=lambda r: abs(r[0] - t_cap))          # the joints when the picture was taken
                tags = [(t["id"], t.get("corners"), t.get("center")) for t in info.get("tags", [])
                        if t.get("corners") or t.get("center")]
                n = f.see(c, tags, h[1], h[2], grip=h[3])
                if log and n:
                    log.write(json.dumps({"t": round(t_cap, 3), "c": c, "n": n, "x": np.round(f.x, 3).tolist()}) + "\n")
            if time.time() - t_print > 0.5 and hist:
                t_print = time.time()
                _, joints, base, grip = hist[-1]
                tip = w.tip(joints, base, grip=grip, dq=f.offsets())
                s = f.sigma()
                print("\r" + "  ".join(f"{n[:4]} {x:+5.2f}±{e:.2f}" for n, x, e in zip(NAMES, f.x, s))
                      + f"   tip {tip[0]:6.1f} {tip[1]:6.1f} {tip[2]:6.1f} mm, "
                      f"{tip[2] - w.table_height(tip[0], tip[1]):5.1f} above table   ", end="", flush=True)
            time.sleep(max(0.0, PERIOD - (time.time() - t0)))
    except KeyboardInterrupt:
        print(f"\nUsed {f.used} tag sightings, skipped {f.skipped} as misreads.")
    finally:
        if log:
            log.close()
