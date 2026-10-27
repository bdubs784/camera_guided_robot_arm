"""
touch: record fingertip touches, for measuring the fingertip point, the plywood and the table's real edges.
You drive the arm (controller or page) until the CLOSED fingertips just touch something, then press Enter and
type what it is. Each touch saves the joints, the base, every camera's tags right then, and (with a world.json) the
model's fingertip. No movement.

Labels (type anything; "... tag N" touches are fitted, the rest are checks that fit_touch reports as heights above
the table plane):
    tag 120          the closed FINGERTIPS on the CENTRE of mat tag 120 (calibrates the fingertip point; do 3-4 of
                     these, spread out: near and far, left and right, different hand tilts). Touch, don't press.
    cam tag 120      any other point of the hand on a tag centre, named (here "cam"): measures that point, so the
                     safety check knows the hand's real shape. Do 2-3 per point, from different angles. The name can
                     say which part the point is on:  G:cam (gripper body, the default), H:wristbottom (the wrist
                     block, does not turn with rotate), L:outer / R:outer (a finger: moves with the gripper value)
    mat              anywhere on the mat / table top
    ply top          on top of the plywood          ply edge   on a plywood edge or corner
    tower            the tower base                 low        a low, far reach (droop check)

    python3 arm.py touch            (q + Enter to finish)
    then:  python3 arm.py fit_touch    (fits the fingertip and the named points, writes them into world.json)

fit_touch measures each touch's real joint angles from the arm tags the cameras saw (a touch that presses bends the
arm: the elbow gave up to 10 steps), and writes into world.json only when the fit is pinned down (leaving any
one touch out moves it less than FIT_STABLE_MM); --force writes anyway.
"""
import json
import os
import re
import time

import numpy as np

from ..config import FINGER_TAGS
from ..hardware import arm_angles
from ..state import S
from .where import sightings_now


def mode_touch(args):
    out_path = os.path.join(S.data_dir, time.strftime("touch_%Y%m%d_%H%M%S.jsonl"))
    w = None
    wp = os.path.join(S.data_dir, "world.json")
    if os.path.exists(wp):
        from ..world import World
        w = World(wp)
    print(__doc__.split("    python3 arm.py touch")[0].strip())      # what to do, and the labels fit_touch uses
    print(f"\nSaving to {out_path}.  Enter a label after each touch ('q' to finish).")
    n = 0
    with open(out_path, "w") as f:
        while True:
            label = input(f"\ntouch {n + 1} label (e.g. 'tag 120', 'ply top', 'q'): ").strip()
            if label.lower() in ("q", "quit", "exit"):
                break
            if not label:
                continue
            a = arm_angles()
            seen = sightings_now()
            row = {"t": round(time.time(), 2), "label": label, "joints": a[:5], "base": a[5],
                   "tags": [{"cam": c, "id": t, "corners": px.tolist()} for c, t, px in seen]}
            if w is not None:
                off, fit = w.fit_base_offset(seen, a[:4], a[5])
                tip = w.tip(a[:4], a[5], grip=a[4], base_offset=off)
                row["model_tip"] = np.round(tip, 2).tolist()
                row["base_offset"] = round(off, 3)
                print(f"   saved. Model fingertip x {tip[0]:.1f} y {tip[1]:.1f} z {tip[2]:.1f} mm, "
                      f"{tip[2] - w.table_height(tip[0], tip[1]):.1f} mm above the table plane (before fitting the tip)")
            else:
                print("   saved.")
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += 1
    print(f"\n{n} touches in {out_path}. Next: python3 arm.py fit_touch")


TAG_LABEL = re.compile(r"^(?:(\S+)\s+)?tag\s+(\d+)$", re.I)   # "tag 120" (the fingertip) or "cam tag 120"
PART_INDEX = {"G": 5, "H": 3}                                     # world.parts(): wrist block 3, gripper 5


def parse_label(label):
    """'tag 120' -> ('tip', 'G', 120); 'cam tag 120' -> ('cam', 'G', 120); 'L:outer tag 7' -> ('L:outer', 'L', 7);
    anything else -> None (a check touch)."""
    m = TAG_LABEL.match(label.strip())
    if not m:
        return None
    name = m.group(1) or "tip"
    part = name.split(":", 1)[0].upper() if ":" in name else "G"
    return name, part, int(m.group(2))


GRIP_SURVEYED_RANGE = (40.0, 150.0)  # the surveys' gripper values; beyond them the finger model is a guess (the finger
                                     # tags were 40-50 px off at 160, fully closed), so finger tags are not used
DQ_PRIOR_STEPS = 15.0                # how far a touch may have bent a joint (soft limit, slider steps)
HUBER_PX = 5.0


def arm_sightings(w, r):
    """The arm tags (not the table's) a touch row saw: [(camera, tag, corners 4x2)]."""
    grip = r["joints"][4] if len(r["joints"]) > 4 else None
    fingers_ok = bool(w.fingers) and grip is not None and GRIP_SURVEYED_RANGE[0] <= grip <= GRIP_SURVEYED_RANGE[1]
    return [(t["cam"], int(t["id"]), np.asarray(t["corners"], float)) for t in r.get("tags", [])
            if int(t["id"]) in w.tags and t["cam"] in w.cams and w.tags[int(t["id"])]["part"] != 0
            and (fingers_ok or int(t["id"]) not in FINGER_TAGS)]


def camera_dq(w, r, iters=12):
    """How far each joint really was from its slider value at this touch (shoulder, elbow, wrist, rotate in slider
    steps, then 0 for the base, which base_offset already has), from the arm tags the cameras saw: a robust
    least-squares fit. Returns (dq (5,), median corner error px, tags used); no arm tags seen: zeros."""
    obs = arm_sightings(w, r)
    if len(obs) < 2:
        return np.zeros(5), float("nan"), len(obs)
    j, b, off = r["joints"][:4], r["base"], r.get("base_offset", 0.0)
    grip = r["joints"][4] if len(r["joints"]) > 4 else None
    z = np.concatenate([px.ravel() for _, _, px in obs])

    def pred(x):
        dq = np.append(x, 0.0)
        return np.concatenate([w.predict_tag(t, c, j, b, grip=grip, base_offset=off, dq=dq).ravel() for c, t, _ in obs])
    x = np.zeros(4)
    for _ in range(iters):
        h = pred(x)
        e = np.nan_to_num(z - h)
        J = np.column_stack([(pred(x + d) - h) / 1e-3 for d in np.eye(4) * 1e-3])
        wt = np.minimum(1.0, HUBER_PX / np.maximum(np.abs(e), 1e-9))       # big misses count less (a misread)
        A = (J * wt[:, None]).T @ J + np.eye(4) / DQ_PRIOR_STEPS ** 2
        step = np.linalg.solve(A, (J * wt[:, None]).T @ e - x / DQ_PRIOR_STEPS ** 2)
        x += step
        if np.abs(step).max() < 1e-3:
            break
    err = np.linalg.norm((z - pred(x)).reshape(-1, 4, 2), axis=2).mean(1)
    return np.append(x, 0.0), float(np.median(err)), len(obs)


def point_frame(w, r, part):
    """(R, o, shift) for a touch row: the point p (mm, in its part) is at R @ (p + shift) + o in table mm.
    Finger points (L/R) ride on the gripper and slide with the gripper value (the finger model)."""
    grip = r["joints"][4] if len(r["joints"]) > 4 else None
    R, o = w.parts(r["joints"][:4], r["base"], base_offset=r.get("base_offset", 0.0), dq=r.get("dq"))[PART_INDEX.get(part, 5)]
    shift = w.finger_offset_side(part, grip) if part in ("L", "R") else np.zeros(3)
    return R, o, shift


def table_plane(w):
    """The table as (unit normal, offset): points on it satisfy n . X = c."""
    c0 = w.table_height(0.0, 0.0)
    a = w.table_height(1.0, 0.0) - c0
    b = w.table_height(0.0, 1.0) - c0
    n = np.array([-a, -b, 1.0])
    k = np.linalg.norm(n)
    return n / k, c0 / k


def fit_point(w, rows, part, tip=False, planes=()):
    """Least squares: the point (in its part's frame) that lands on each touched tag's centre.

    planes: touches that landed on a FLAT thing instead of a tag centre, as (row, height above the table in mm).
    A tag centre pins the point in all three directions but carries the aiming error ("was that really the
    centre?"); a flat touch has no aiming error at all but only says the point is somewhere on that plane, so it
    is one equation instead of three. Many quick flat touches beat a few careful tag ones, as long as the hand
    was held at a good spread of angles -- at one angle they all say the same thing and two directions stay
    loose. Returns (point, misses mm); the flat touches' misses are heights above where they should have sat."""
    A, b = [], []
    for r, tid in rows:
        R, o, shift = point_frame(w, r, part)
        if tip:
            shift = w.tip_shift(r["joints"][4] if len(r["joints"]) > 4 else None)
        A.append(R)                                                # R @ (p + shift) + o = tag centre
        b.append(w.tags[tid]["t"] - o - R @ shift)
    n_tag = len(rows)
    if len(planes):
        nrm, c = table_plane(w)
        for r, height in planes:
            R, o, shift = point_frame(w, r, part)
            if tip:
                shift = w.tip_shift(r["joints"][4] if len(r["joints"]) > 4 else None)
            A.append((nrm @ R)[None, :])                           # n . (R @ (p + shift) + o) = c + height
            b.append(np.array([c + height - nrm @ o - (nrm @ R) @ shift]))
    A, b = np.vstack(A), np.concatenate(b)
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = A @ p - b
    miss = list(np.linalg.norm(res[:3 * n_tag].reshape(-1, 3), axis=1)) + list(np.abs(res[3 * n_tag:]))
    return p, np.array(miss)


def fit_spread_deg(w, rows, part, planes=(), tip=False):
    """How well the touches pin the point down in every direction: the smallest and largest singular value of the
    fit. Flat touches at one hand angle look stable when a touch is left out (they all agree) while leaving two
    directions free, so leave-one-out alone cannot catch it -- this can."""
    A = []
    for r, _ in rows:
        R, _, _ = point_frame(w, r, part)
        A.append(R)
    if len(planes):
        nrm, _ = table_plane(w)
        for r, _ in planes:
            R, _, _ = point_frame(w, r, part)
            A.append((nrm @ R)[None, :])
    if not A:
        return float("inf")
    sv = np.linalg.svd(np.vstack(A), compute_uv=False)
    return float(sv[0] / max(sv[-1], 1e-12))


FLAT_LABELS = ("mat", "table")   # touches on the table top: fitted as "the point sits on this plane"
COND_WARN = 25.0                 # above this the touches do not pin the point down in every direction
MISS_WARN_MM = 8.0                       # a touch on a tag centre fits to a few mm (joints from the cameras:
                                         # mostly 1-7 mm); more means a touch was off
FIT_STABLE_MM = 3.0                      # world.json is written only if leaving out any one touch moves the tip less


def loo_spread(w, rows, part, tip=False, planes=()):
    """How far the fitted point moves (mm, worst case) when any one touch is left out (needs 3+ touches)."""
    planes = list(planes)
    all_rows = [("tag", k) for k in range(len(rows))] + [("flat", k) for k in range(len(planes))]
    if len(all_rows) < 3:
        return float("inf")
    p, _ = fit_point(w, rows, part, tip, planes)
    out = 0.0
    for kind, k in all_rows:
        rr = rows[:k] + rows[k + 1:] if kind == "tag" else rows
        pp = planes if kind == "tag" else planes[:k] + planes[k + 1:]
        if 3 * len(rr) + len(pp) < 3:
            continue
        out = max(out, float(np.linalg.norm(fit_point(w, rr, part, tip, pp)[0] - p)))
    return out


def miss_note(miss, nums=None):
    """nums: each miss's touch number as fit_touch lists them (default 1, 2, ...)."""
    nums = nums or list(range(1, len(miss) + 1))
    bad = [nums[k] for k, e in enumerate(miss) if e > MISS_WARN_MM]
    return "" if not bad else (f"   <- touch {', '.join(map(str, bad))} disagree{'s' if len(bad) == 1 else ''} with the "
                               f"others by more than {MISS_WARN_MM:g} mm: not on the tag centre, pressing, or the wrong tag "
                               "number? Redo those touches.")


def mode_fit_touch(args):
    """The fingertip point from the 'tag N' touches, and every named point from '<name> tag N' touches: the point on
    the hand that lands on each touched tag's centre. Written into world.json ("tip_in_G", "points")."""
    from ..world import World
    wp = os.path.join(S.data_dir, "world.json")
    w = World(wp)
    files = sorted(p for p in os.listdir(S.data_dir) if p.startswith("touch_") and p.endswith(".jsonl"))
    rows = [json.loads(l) for p in files for l in open(os.path.join(S.data_dir, p)) if l.strip()]
    print("Each touch's real joint angles, from the arm tags the cameras saw (slider steps off the sliders):")
    for k, r in enumerate(rows):
        dq, px, n = camera_dq(w, r)
        r["dq"], r["n"] = dq, k + 1
        note = "   <- pressing? the arm bent a lot" if np.abs(dq[:3]).max() > 5 else ""
        print(f"   {k + 1:2d} {r['label']:14s} shoulder {dq[0]:+5.1f}  elbow {dq[1]:+5.1f}  wrist {dq[2]:+5.1f}  "
              f"rotate {dq[3]:+5.1f}   ({n} tags, fit {px:.1f} px){note}" if n >= 2 else
              f"   {k + 1:2d} {r['label']:14s} (too few arm tags seen: slider values used)")
    groups = {}                                                    # name -> (part, [(row, tag)])
    for r in rows:
        got = parse_label(r["label"])
        if got is None:
            continue
        name, part, tid = got
        if tid not in w.tags:
            print(f"   ({r['label']}: tag {tid} is not in the world model, skipped)")
            continue
        if part not in ("G", "H", "L", "R"):
            print(f"   ({r['label']}: part '{part}' is not one of G, H, L, R, skipped)")
            continue
        groups.setdefault(name, (part, []))[1].append((r, tid))
    flat = [(r, 0.0) for r in rows if r["label"].strip().lower() in FLAT_LABELS]
    tip_rows = groups.get("tip", ("G", []))[1]
    if 3 * len(tip_rows) + len(flat) < 6:
        print(f"Not enough to fit the fingertip: {len(tip_rows)} 'tag N' touches and {len(flat)} flat ones "
              f"({'/'.join(FLAT_LABELS)}). Either two tag centres, or half a dozen flat touches at a good spread "
              "of hand angles.")
        return
    cond = fit_spread_deg(w, tip_rows, "G", flat, tip=True)
    if flat:
        print(f"Fingertip fit from {len(tip_rows)} tag touches + {len(flat)} flat touches "
              f"(spread of directions: {cond:.0f}, under {COND_WARN:g} is good)")
    if cond > COND_WARN:
        print("   The hand was held at much the same angle every time, so the fit is loose in some directions"
              " even though the touches agree with each other. Add flat touches with the wrist rotated and"
              " tilted differently.")
    d = json.load(open(wp))
    tip, miss = fit_point(w, tip_rows, "G", tip=True, planes=flat)
    spread = loo_spread(w, tip_rows, "G", tip=True, planes=flat)
    print(f"Fingertip point on the gripper: {np.round(tip, 1)} mm (was {np.round(w.tip_in_G, 1)}); leaving out any one "
          f"touch moves it up to {spread:.1f} mm")
    nums = [r["n"] for r, _ in tip_rows] + [r["n"] for r, _ in flat]
    print("   miss at each touch after the fit (mm): " + "  ".join(f"#{n} {e:.1f}" for n, e in zip(nums, miss))
          + miss_note(miss, nums))
    if spread > FIT_STABLE_MM and not getattr(args, "force", False):
        print(f"NOT written into world.json: the tip is not pinned down to {FIT_STABLE_MM:g} mm (needs 3+ good "
              "'tag N' touches, or more flat ones at a wider spread of hand angles). Add touches, or "
              "fit_touch --force  to write it anyway.")
        return
    d["tip_in_G"] = tip.tolist()
    d["tip_fit"] = {"touches": len(miss), "rms_mm": float(np.sqrt(np.mean(miss ** 2)))}
    points = d.get("points", {})
    for name, (part, grp) in sorted(groups.items()):
        if name == "tip":
            continue
        if len(grp) < 2:
            print(f"   {name}: only 1 touch, needs at least 2 (from different angles): not fitted")
            continue
        p, miss = fit_point(w, grp, part)
        warn = "" if len(grp) >= 3 else "   (2 touches: do a third from another angle to be sure)"
        nums = [r["n"] for r, _ in grp]
        print(f"Point {name} on {part}: {np.round(p, 1)} mm, miss at each touch " + "  ".join(f"#{n} {e:.1f}" for n, e in zip(nums, miss))
              + " mm" + warn + miss_note(miss, nums))
        points[name] = {"part": part, "xyz_mm": p.tolist(), "touches": len(grp), "rms_mm": float(np.sqrt(np.mean(miss ** 2)))}
    if points:
        d["points"] = points
    json.dump(d, open(wp, "w"), indent=1)
    w = World(wp)
    print(f"Written into {wp}. Every other touch: model height above the table with the new tip:")
    for r in rows:
        if parse_label(r["label"]) is not None:
            continue
        tipw = w.tip(r["joints"][:4], r["base"], grip=r["joints"][4] if len(r["joints"]) > 4 else None,
                     base_offset=r.get("base_offset", 0.0), dq=r.get("dq"))
        print(f"   {r['label']:12s} x {tipw[0]:7.1f} y {tipw[1]:7.1f} z {tipw[2]:7.1f}  "
              f"-> {tipw[2] - w.table_height(tipw[0], tipw[1]):6.1f} mm above the table plane")
