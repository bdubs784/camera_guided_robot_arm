#!/usr/bin/env python3
"""
Where the fingers really close, from the gripper camera's own film -- and how far the world model's fingertip point
is from it.

    python tools/finger_center.py <data dir> <film folder(s)> --runs <grasp_runs folder(s)> [--write world_fixed.json]

The gripper camera and the fingers ride on the same part, so the block's top tag in that camera's picture says where
the block is relative to the fingers with no world model in between. Two moments are read from the film (show
--film, 2 frames/s) for every grab with a trace:

  held     the first frame after a good squeeze: the block sits between the pads, so its tag's x (across the
           fingers) IS the fingers' centre line
  before   the last frame before the fingers start to close: where the grab put the block relative to that line

and compares the first with where the model puts its fingertip point (world.json tip_in_G at the closed grip) in the
same camera frame. The difference across the fingers is the fingertip point's lateral error.

Result: held blocks sat at camera x -0.3 +- 0.6 mm (6 grabs); the model put the fingertip at +10.2 -- 10 mm toward
finger R (tag 32), on that finger rather than between them. Blocks before the close sat at +9.7 on the grabs and
+15 to +22 on every empty squeeze: the aim was right by the model and the fingers were 10 mm off, so only the grabs
whose other errors happened to lean the other way caught the block.

--write: a world.json with tip_in_G moved by the measured lateral error (its old value kept under "tip_fix"). Any
change to world.json needs the plan table rebuilt (tools/build_plan_table.py): the Pi ignores a stale one.
"""
import argparse
import bisect
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pi"))

TAG_MM = 30.0
FILM_W = 1152.0                      # the film is saved 640 wide; the lens calibration is for 1152 x 864


def detector():
    import cv2
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    p.aprilTagQuadDecimate = 1.0
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), p)


_DET = []


def _frame(f):
    """(time, {tag: corners}) for one film frame that shows the block's tag, else None."""
    import cv2
    if not _DET:
        _DET.append(detector())
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    c, ids, _ = _DET[0].detectMarkers(im)
    if ids is None or 7 not in ids.ravel():
        return None
    s = FILM_W / im.shape[1]
    return (float(re.search(r"gripper_([\d.]+)\.jpg", f).group(1)),
            {int(i): cc.reshape(4, 2) * s for i, cc in zip(ids.ravel(), c)})


def read_film(folders, jobs):
    """{time: {tag: corners (1152-wide pixels)}} for every gripper frame that shows the block's tag (7)."""
    import multiprocessing as mp
    files = [f for d in folders for f in glob.glob(os.path.join(d, "**", "gripper_*.jpg"), recursive=True)]
    with mp.Pool(jobs) as pool:
        return dict(r for r in pool.imap_unordered(_frame, files, chunksize=64) if r)


def tag_in_camera(w, corners):
    """The tag's centre in the gripper camera's frame (mm): x across the fingers, y down the picture, z out."""
    import cv2
    c = w.cams["gripper"]
    h = TAG_MM / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    ok, _, t = cv2.solvePnP(obj, np.asarray(corners, np.float32).reshape(4, 1, 2), c["K"], c["dist"],
                            flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return t.ravel() if ok else None


def moments(run, film, times):
    """(before, held, grip): the tag in the camera just before the close and just after it, from the run's trace."""
    o = run.get("outcome") or {}
    ev = (o.get("trace") or {}).get("events") or []
    k = next((i for i, e in enumerate(ev) if e[2] == "hand" and re.match(r"rotate None grip 1[01]\d", e[3])), None)
    if k is None:
        return None, None, None
    grip = float(re.search(r"grip ([\d.]+)", ev[k][3]).group(1))
    arm_end = max([e[1] for e in ev[:k] if e[2] == "arm"], default=ev[k][0])

    def pick(t0, t1, last):
        i0, i1 = bisect.bisect_left(times, t0), bisect.bisect_right(times, t1)
        span = range(i1 - 1, i0 - 1, -1) if last else range(i0, i1)
        return next((film[times[i]][7] for i in span), None)
    return pick(arm_end - 0.2, ev[k][0] + 0.3, True), pick(ev[k][1] + 0.2, ev[k][1] + 1.5, False), grip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("film", nargs="+")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--write", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    from armlab.world import World
    w = World(os.path.join(args.data_dir, "world.json"))
    c = w.cams["gripper"]
    film = read_film(args.film, args.jobs)
    times = sorted(film)
    print(f"{len(times)} film frames show the block's tag")
    held, before = [], {"GRABBED": [], "NOT confirmed": [], "empty": []}
    grip_closed = None
    for d in args.runs:
        for f in sorted(glob.glob(os.path.join(d, "grasp_*.json"))):
            run = json.load(open(f))
            res = str((run.get("outcome") or {}).get("result", ""))
            b, h, grip = moments(run, film, times)
            if grip is None:
                continue
            kind = "GRABBED" if res.startswith("GRABBED") else "NOT confirmed" if res.startswith("NOT confirmed") \
                else "empty" if "empty" in res else None
            if kind and b is not None:
                p = tag_in_camera(w, b)
                if p is not None:
                    before[kind].append(p[0])
            if kind in ("GRABBED", "NOT confirmed") and h is not None:
                p = tag_in_camera(w, h)
                if p is not None:
                    held.append(p)
                    grip_closed = grip
    if not held:
        sys.exit("no held block in the film: needs traced grabs filmed with show --film")
    H = np.array(held)
    tip = c["R"] @ (w.tip_in_G + w.tip_shift(grip_closed)) + c["t"]
    err = float(np.median(H[:, 0]) - tip[0])
    print(f"held blocks ({len(H)}): tag at camera x {np.median(H[:, 0]):+.1f} mm (spread {H[:, 0].std():.1f}) -- the "
          f"fingers' centre line")
    print(f"the model's fingertip point at grip {grip_closed:.0f}: camera x {tip[0]:+.1f}  ->  {err:+.1f} mm across "
          f"the fingers")
    for k, v in before.items():
        if v:
            print(f"   before the close, {k:13s} ({len(v):2d}): x {np.mean(v):+5.1f} (min {min(v):+5.1f}, max {max(v):+5.1f})")
    if args.write:
        # camera x in the gripper frame G is c["R"][0]; moving the point by -err along it brings it onto the line
        d = json.load(open(os.path.join(args.data_dir, "world.json")))
        old = list(d["tip_in_G"])
        d["tip_in_G"] = list(np.asarray(old, float) + err * np.asarray(c["R"][0], float))
        d["tip_fix"] = {"was": old, "lateral_mm": err, "held": len(H),
                        "doc": "tools/finger_center.py: tip_in_G moved onto the fingers' centre line seen on film"}
        json.dump(d, open(args.write, "w"), indent=1)
        print(f"   tip_in_G {np.round(old, 1)} -> {np.round(d['tip_in_G'], 1)}  ->  {args.write}")


if __name__ == "__main__":
    main()
