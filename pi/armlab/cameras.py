"""
Reading tags from the cameras: the tower camera's 3D positions and the gripper camera's picture positions,
always from fresh frames, averaged or median-filtered over a few of them.
"""
import math
import time

import numpy as np

from .config import FINGER_TAGS, HAND_TAG, READS, READ_TIMEOUT_S
from .errors import ArmError
from .hardware import all_cameras, camera_status


def read_hand(reads=READS, timeout=READ_TIMEOUT_S):
    """
    Measure the hand tag with the tower camera. Each reading comes from a NEW camera frame.
    Returns None if the tag was not seen, else a dict with the mean position in mm, the spread
    between readings, how many readings were used, the tag's rotation, and any finger tags seen.
    """
    got, rvec, fingers = [], None, {}
    last_frame = None
    t_end = time.time() + timeout
    while len(got) < reads and time.time() < t_end:
        try:
            st = camera_status()
        except ArmError:
            time.sleep(0.2)
            continue
        if st.get("frames_checked") == last_frame:     # same frame as last time: wait for a new one
            time.sleep(0.03)
            continue
        last_frame = st.get("frames_checked")
        for tag in st.get("tags", []):
            if "pos_mm" not in tag:
                continue
            if tag["id"] == HAND_TAG:
                got.append(tag["pos_mm"])
                rvec = tag.get("rvec")
            elif tag["id"] in FINGER_TAGS:
                fingers[str(tag["id"])] = tag["pos_mm"]
    if not got:
        return None
    n = len(got)
    mean = [sum(p[i] for p in got) / n for i in range(3)]
    std = [math.sqrt(sum((p[i] - mean[i]) ** 2 for p in got) / n) for i in range(3)]
    return {"pos": mean, "std": std, "n": n, "rvec": rvec, "fingers": fingers}


def gripper_tag(tag_id, frames=6, timeout=3.0):
    """Where a tag is in the gripper camera's picture: (centre x, centre y) in pixels, median of a few new frames. None if not seen."""
    got, last, t_end = [], None, time.time() + timeout
    while len(got) < frames and time.time() < t_end:
        try:
            g = all_cameras().get("gripper") or {}
        except ArmError:
            time.sleep(0.2)
            continue
        frame = g.get("frames_checked")
        if frame is None:                                  # this camera does not number its frames: just space the reads out
            time.sleep(0.12)
        elif frame == last:
            time.sleep(0.03)
            continue
        last = frame
        for t in g.get("tags", []):
            if t.get("id") != tag_id:
                continue
            pts = t.get("corners") or t.get("px") or t.get("center") or t.get("centre")
            if pts is not None:
                got.append(np.mean(np.atleast_2d(np.array(pts, dtype=float)), axis=0))
    return np.median(got, axis=0) if len(got) >= min(frames, max(2, frames // 2)) else None


