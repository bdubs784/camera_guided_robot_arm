#!/usr/bin/env python3
"""
One-time lens calibration for an arm camera, using the printed calibration board.

    python3 calibrate_cam.py                       # calibrates the torre camera (30 mm board)
    python3 calibrate_cam.py gripper --small       # the gripper camera, with the HALF-SIZE board (15 mm squares)

The gripper camera's lens is locked at a set focus (about 10 cm away by default), so it must be calibrated
with a board small enough to show 10+ corners at that distance: print charuco_15mm.pdf at 100% (no scaling)
and check one square measures 15 mm. Open the gripper wide (about 40) so the fingers are out of the way.
The lens position used is saved with the result: if you ever change the focus, calibrate again.

Run it on the Pi while the camera stream (armcams) is running. It takes frames
from the stream, so nothing has to be stopped.

What you do:
    Hold the printed board in front of the camera, keep it still for a moment,
    and the script captures it. Then move the board somewhere new: closer,
    farther, left, right, top, bottom, and TILTED at different angles.
    The camera and the arm stay still. Only the board moves.

It saves the result to ~/calib_<camera>.json. Press Ctrl+C to finish early.
"""

import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

# ---------------- settings ----------------
ARGS       = [a for a in sys.argv[1:] if not a.startswith("--")]
CAM        = ARGS[0] if ARGS else "torre"
URL        = f"http://localhost:8080/raw/{CAM}"
SQUARES_X, SQUARES_Y = 7, 5        # must match the printed board
SQUARE_MM, MARKER_MM = (15.0, 11.0) if "--small" in sys.argv else (30.0, 22.0)   # --small: the half-size board
TARGET_VIEWS = 20                  # how many different board positions to collect
MAX_VIEWS    = 35
MIN_CORNERS  = 10                  # corners needed in a frame to use it (board has 24)
STILL_PX     = 2.5                 # board must move less than this between two looks
NEW_VIEW_PX  = 45                  # and differ this much from every saved view
OUT_FILE   = os.path.expanduser(f"~/calib_{CAM}.json")
IMAGE_DIR  = os.path.expanduser(f"~/calib_{CAM}_images")

DICT  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM, MARKER_MM, DICT)
DETECTOR = cv2.aruco.CharucoDetector(BOARD)


def fetch_frame():
    """Newest full-size frame from the camera stream, as a grey image."""
    with urllib.request.urlopen(URL, timeout=5) as r:
        data = np.frombuffer(r.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def find_board(gray):
    """Return {corner_id: (x, y)} for the board corners seen in this frame."""
    corners, ids, _, _ = DETECTOR.detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return {}
    pts = np.asarray(corners, np.float32).reshape(-1, 2)
    return {int(i): p for p, i in zip(pts, np.asarray(ids).flatten())}


def mean_shift(a, b):
    """Average distance between the same corners in two views (None if too few shared)."""
    common = set(a) & set(b)
    if len(common) < 6:
        return None
    return float(np.mean([np.linalg.norm(a[k] - b[k]) for k in common]))


def coverage(views, size):
    """3 x 3 grid of the picture: which parts have had board corners in them."""
    w, h = size
    grid = np.zeros((3, 3), bool)
    for v in views:
        for x, y in v.values():
            grid[min(2, int(3 * y / h)), min(2, int(3 * x / w))] = True
    return grid


def show_coverage(grid):
    return "\n".join("      " + "".join("[x]" if c else "[ ]" for c in row) for row in grid)


def calibrate(views, size):
    obj_pts, img_pts = [], []
    for v in views:
        ids = np.array(sorted(v), np.int32).reshape(-1, 1)
        pts = np.array([v[int(i)] for i in ids.flatten()], np.float32).reshape(-1, 1, 2)
        o, i = BOARD.matchImagePoints(pts, ids)
        obj_pts.append(o)
        img_pts.append(i)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    per_view = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj - i) ** 2, axis=2)))))
    return rms, K, dist, per_view


def main():
    print(f"Calibrating the '{CAM}' camera from {URL}")
    try:
        first = fetch_frame()
    except Exception as e:
        sys.exit(f"Could not get a frame ({e}).\n"
                 "Is the camera stream running?  systemctl is-active armcams.service")
    size = (first.shape[1], first.shape[0])
    print(f"Frame size {size[0]} x {size[1]}. Hold the board in view and keep it still.\n")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    views, prev, last_msg = [], {}, ""
    try:
        while len(views) < MAX_VIEWS:
            gray = fetch_frame()
            cur = find_board(gray)
            msg = ""
            if not cur:
                msg = "no board in view (show the whole board, avoid glare)"
            else:
                moved = mean_shift(cur, prev) if prev else None
                if moved is None or moved > STILL_PX:
                    msg = "board seen - hold it still..."
                else:
                    shifts = [mean_shift(cur, v) for v in views]
                    if any(s is not None and s < NEW_VIEW_PX for s in shifts):
                        msg = "already have this position - move or tilt the board"
                    else:
                        views.append(cur)
                        cv2.imwrite(os.path.join(IMAGE_DIR, f"view_{len(views):02d}.png"), gray)
                        grid = coverage(views, size)
                        print(f"\n  captured {len(views)}/{TARGET_VIEWS}"
                              f"   ({len(cur)} corners)   picture areas covered:")
                        print(show_coverage(grid))
                        empty = int((~grid).sum())
                        if empty:
                            print(f"      {empty} area(s) still empty: put the board there next")
                        last_msg = ""
                        if len(views) >= TARGET_VIEWS and empty <= 1:
                            break
            if msg and msg != last_msg:
                print("  " + msg)
                last_msg = msg
            prev = cur
            time.sleep(0.6)
    except KeyboardInterrupt:
        print("\nStopping capture.")

    if len(views) < 8:
        sys.exit(f"Only {len(views)} views captured; need at least 8. Run it again.")

    print(f"\nWorking out the lens numbers from {len(views)} views...")
    rms, K, dist, per_view = calibrate(views, size)
    # Drop any view that fits much worse than the rest (blur, bent board), then redo.
    limit = max(2.0 * float(np.median(per_view)), 1.0)
    keep = [v for v, e in zip(views, per_view) if e <= limit]
    if 8 <= len(keep) < len(views):
        print(f"  dropped {len(views) - len(keep)} poor view(s) and recalculated")
        views = keep
        rms, K, dist, per_view = calibrate(views, size)

    lens = None
    try:                                     # the gripper camera's focus setting: the result is only valid for it
        with urllib.request.urlopen("http://localhost:8080/focus", timeout=3) as r:
            info = json.loads(r.read().decode())
        lens = info.get("lens_position") if CAM == "gripper" else None
    except Exception:
        pass
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hfov = 2 * np.degrees(np.arctan(size[0] / (2 * fx)))
    result = {
        "camera": CAM,
        "image_size": list(size),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.flatten().tolist(),
        "rms_px": float(rms),
        "views_used": len(views),
        "board": {"squares": [SQUARES_X, SQUARES_Y], "square_mm": SQUARE_MM, "marker_mm": MARKER_MM},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lens_position": lens,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {OUT_FILE}")
    print(f"  average error : {rms:.2f} pixels", end="   ")
    print("(excellent)" if rms < 0.5 else "(good)" if rms < 1.0 else
          "(usable, but worth redoing with a flatter board and more tilted views)" if rms < 2.0 else
          "(poor - redo it: keep the board flat and still, avoid glare)")
    print(f"  focal length  : {fx:.0f} x {fy:.0f} pixels")
    print(f"  picture center: {cx:.0f}, {cy:.0f}   (frame is {size[0]} x {size[1]})")
    print(f"  field of view : {hfov:.1f} degrees wide")
    grid = coverage(views, size)
    if (~grid).any():
        print("  note: some picture areas had no board in them, so accuracy is lower there:")
        print(show_coverage(grid))


if __name__ == "__main__":
    main()
