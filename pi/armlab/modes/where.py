"""
where: where the arm really is, from the world model (the big solve's world.json in ~/arm_data/): the fingertip in
table millimetres, its height above the table, how far the base really is from its count (measured from the table
tags the tower camera and webcam see right now), and how well every camera's view agrees with the model.
No movement.

    python3 arm.py where            once
    python3 arm.py where --loop     keep updating (Ctrl+C to stop)

Without a world.json it falls back to the old self-model answer.
"""
import os
import time

import numpy as np

from ..errors import ArmError
from ..hardware import all_cameras, arm_angles
from ..state import S
from ..world import World

WORLD_FILE = "world.json"


def sightings_now():
    try:
        cams = all_cameras()
    except ArmError:
        return []
    out = []
    for c, info in cams.items():
        if not isinstance(info, dict):
            continue
        for t in info.get("tags", []):
            if t.get("corners") and len(t["corners"]) == 4:
                out.append((c, int(t["id"]), np.asarray(t["corners"], float)))
    return out


def report(w, verbose=True):
    a = arm_angles()
    joints, base = a[:4], a[5]
    seen = sightings_now()
    off, fit = w.fit_base_offset(seen, joints, base)
    tip = w.tip(joints, base, grip=a[4], base_offset=off)
    height = tip[2] - w.table_height(tip[0], tip[1])
    print(f"joints  shoulder {a[0]:.1f}  elbow {a[1]:.1f}  wrist {a[2]:.1f}  rotate {a[3]:.1f}   base count {base:+.1f}")
    print(f"base    really {base + off + w.base_zero:+.2f} deg ({off:+.2f} vs the world's zero"
          + (f", from table tags, fit {fit:.1f} px)" if np.isfinite(fit) else ", no table tags in view: not checked)"))
    print(f"tip     x {tip[0]:7.1f}  y {tip[1]:7.1f}  z {tip[2]:7.1f} mm   ->  {height:6.1f} mm above the table")
    if verbose:
        per = {}
        for c, t, px in seen:
            if c in w.cams and t in w.tags:
                e = np.linalg.norm(w.predict_tag(t, c, joints, base, grip=a[4], base_offset=off) - px, axis=1).mean()
                per.setdefault(c, []).append((e, t))
        for c, rows in sorted(per.items()):
            rows.sort()
            arm = [f"{t}:{e:.0f}" for e, t in rows if w.tags[t]["part"] != 0][:8]
            print(f"   {c:8s} {len(rows):2d} tags, model agrees to {np.median([e for e, _ in rows]):5.1f} px median"
                  + (f"   arm tags {' '.join(arm)}" if arm else ""))
    return tip


def mode_where(args):
    path = os.path.join(S.data_dir, WORLD_FILE)
    if not os.path.exists(path):
        from .basic import mode_where as old
        print(f"(no {path}: the old self-model answer. Copy the big solve's world json there for the new one.)")
        return old(args)
    w = World(path)
    print(f"World model {path} ({w.d.get('made', '?')}). Fingertip point on the gripper: {np.round(w.tip_in_G, 1)} mm"
          + ("" if "tip_in_G" in w.d else " (a guess until the touch test)"))
    if not getattr(args, "loop", False):
        report(w)
        return
    try:
        while True:
            print()
            report(w, verbose=False)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print()
