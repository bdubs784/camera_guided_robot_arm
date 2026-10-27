"""
Stop at the hover and let a person drive the sliders until the gripper camera sees the block, recording all of it.

`grasp --grab --pause-hover` calls pause_at_hover() once the automatic looks have had their go. Nothing here
commands the arm: it reads where the arm is while it is driven by hand, so there is no move for a check to
approve and no way for this to put the arm anywhere.

Why it exists. For a whole day the gripper camera did not see the block once, at any hover. The model said it
should: 166 mm from the tag, 234 px across, 66 px inside the frame -- an easy detection. Sweeping the wrist
through 30 slider steps found nothing at 250 mm. So the model is not slightly off about where that camera points,
it is describing a camera somewhere else, and every fix aimed by the model has been aimed at the wrong place.

What settles it is the pose where the tag IS seen, measured rather than predicted. Each sample keeps the arm's
real joints, which cameras saw the tag, and where the model said it would be in the picture, so the difference
between the two can be fitted afterwards instead of guessed at. If the tag comes into view at a repeatable
offset, that is the gripper camera's pose and it can be corrected properly; if it comes in somewhere different
every time, the camera's position is wrong rather than its angle and the solve has to be redone.

A pose found here is handed straight back to the grab, so a hover you rescue by hand becomes the close-up look
and the grab carries on from it.
"""
import json
import os
import signal
import time

import numpy as np

from ..hardware import arm_angles
from ..state import S
from .where import sightings_now

TUNE_HZ = 2.0
BLOCK_TAG = 7


def _predicted(w, X, q):
    """Where the model says the block's top tag is in the gripper picture, and how far that is inside it."""
    if X is None or "gripper" not in w.cams:
        return None
    try:
        px = w.project("gripper", X, q[:4], q[5])
    except Exception:
        return None
    if not np.isfinite(px).all():
        return None
    K = w.cams["gripper"]["K"]
    wide, high = 2.0 * K[0, 2], 2.0 * K[1, 2]
    return {"x": [round(float(px[:, 0].min()), 1), round(float(px[:, 0].max()), 1)],
            "y": [round(float(px[:, 1].min()), 1), round(float(px[:, 1].max()), 1)],
            "edge": round(float(min(px[:, 0].min(), wide - px[:, 0].max(),
                                    px[:, 1].min(), high - px[:, 1].max())), 1)}


def _seen_now(w, seen):
    """Which cameras have the block's tag right now, and where its corners are in each picture."""
    out = {}
    for cam, tag, px in seen:
        if tag == BLOCK_TAG:
            a = np.asarray(px, float)
            out[cam] = {"x": [round(float(a[:, 0].min()), 1), round(float(a[:, 0].max()), 1)],
                        "y": [round(float(a[:, 1].min()), 1), round(float(a[:, 1].max()), 1)]}
    return out


def pause_at_hover(w, check, est, hover, X):
    """Record while the arm is driven by hand. Returns (the pose the tag was first seen from, the saved file).

    hover: the pose the grab wants to come back to. X: the block's top tag corners, for the model's prediction.
    """
    rows, first = [], None
    print()
    print("PAUSED at the hover. Drive the sliders until the gripper camera sees the block.")
    print("   nothing here moves the arm. Ctrl-C when you are done (or when you give up) and the grab goes on.")
    print("   hover was shoulder %.1f  elbow %.1f  wrist %.1f  rotate %.1f" % tuple(hover[:4]))
    print()
    stop = []
    was = signal.signal(signal.SIGINT, lambda *_: stop.append(True))
    try:
        while not stop:
            time.sleep(1.0 / TUNE_HZ)
            q = arm_angles()
            seen = sightings_now()
            got = _seen_now(w, seen)
            pred = _predicted(w, X, list(q[:5]) + [q[5]])
            row = {"t": round(time.time(), 2), "pose": [round(float(v), 2) for v in q[:5]],
                   "base": round(float(q[5]), 2),
                   "off_hover": [round(float(q[i] - hover[i]), 2) for i in range(3)],
                   "sees": got, "model_says": pred}
            try:
                row["table_gap"] = round(float(check.gaps(list(q[:5]), float(q[5]))[0]), 1)
            except Exception:
                row["table_gap"] = None
            rows.append(row)
            if got and first is None:
                first = row
                print("\n   SEEN by %s at shoulder %+.1f elbow %+.1f wrist %+.1f  (%s from the hover)"
                      % (", ".join(sorted(got)), q[0], q[1], q[2],
                         " ".join("%+.1f" % v for v in row["off_hover"])))
                if pred:
                    print("   the model put the tag at x %s y %s; it is really at %s"
                          % (pred["x"], pred["y"], {c: v["x"] for c, v in got.items()}))
            print("   s%6.1f e%6.1f w%6.1f  (%+5.1f %+5.1f %+5.1f)  table %5s  %s      "
                  % (q[0], q[1], q[2], row["off_hover"][0], row["off_hover"][1], row["off_hover"][2],
                     "  ?  " if row["table_gap"] is None else "%5.1f" % row["table_gap"],
                     ("SEES: " + ",".join(sorted(got))) if got else "no tag"), end="\r")
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, was)
    print()

    name = os.path.join(S.data_dir, "hover_tune", "tune_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.dirname(name), exist_ok=True)
    d = {"kind": "hover_tune", "made": time.strftime("%Y-%m-%d %H:%M:%S"), "world": w.d.get("made"),
         "hover": [round(float(v), 2) for v in hover[:6]],
         "block_est": None if est is None else {"xyz": [round(float(v), 1) for v in est["xyz"]],
                                                "yaw_deg": round(float(est["yaw_deg"]), 1)},
         "seen_first": first, "steps": rows}
    json.dump(d, open(name, "w"), indent=1)
    hits = [r for r in rows if r["sees"]]
    print("   %d samples, %d with the tag in some camera -> %s" % (len(rows), len(hits), name))
    if first is None:
        print("   the tag was never seen: that is a result too, and it is in the file.")
    return first, name
