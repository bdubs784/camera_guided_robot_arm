#!/usr/bin/env python3
"""
Run a saved block (grasp --snap) through the whole grab, off the arm: the plan AND every check the real grab
makes before it moves.

    python replay_grasp.py <arm_data dir> <snap_*.json or a folder of them> [--table-margin 15]

For a whole day every failure was a plan the planner liked and a MOVE check then refused -- the hover too close to the
block, the way across not clear -- and each one cost a minute of arm time to find. Those checks are pure
functions of the world model, the table model and the safety map, so they can all be run here: snap a lot of
block positions in a couple of minutes, then find out offline which ones would actually grab, and why not.

Checked, in the order the grab does it:
    1. plan_best             a way in exists, the camera sees from the hover, the joints have room
    2. up to HIGH_POSE       allowed by the map, and clear of the block leaving home
    3. the base turn         clear of the block at every angle it turns through
    4. across to the hover   clear of the block, arriving with the fingers wide
    5. the hand at the hover turning and opening there stays clear of the block and the table
    6. the way in            every step keeps the table margin
Not checked: whether the fingers actually hold it. That needs the arm.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pi"))


def corrected_top(w, b):
    """The block top with each camera's known error taken off, weighted by how well its tag fitted."""
    from armlab.cube import BIAS_MAX_PX, top_bias
    num = den = 0.0
    for cam, h in (b.get("tops") or {}).items():
        px = (b.get("cams") or {}).get(cam, 99.0)
        bias = top_bias(w, cam, b["out_mm"])
        if px > BIAS_MAX_PX or bias is None:
            continue
        wgt = 1.0 / max(px, 0.5) ** 2
        num += wgt * (h - bias)
        den += wgt
    return None if den <= 0 else num / den


def try_one(G, MG, exact_ok, zone, check, w, cube, table_margin, grasp_margin, depth, roll):
    """The plan and then every move check the grab makes, in order. Says where it stopped and why."""
    level = (roll, MG.TRAVEL_GRIP)
    wide = (roll, MG.HOVER_GRIP)

    def accept(pp):
        h = pp["hover"]
        poses = [list(h[:3]) + list(wide), list(h[:3]) + [h[3], h[4]]]
        return min(MG.cube_gap(check, cube, q, h[5]) for q in poses) >= MG.HOVER_CLEAR_MM

    kw = {} if depth is None else {"depth": float(depth)}
    # plan_best binds `ladder=HOVER_LADDER` as a default when the module is imported, so setting
    # G.HOVER_LADDER afterwards does not reach it. Pass it through or the ladder is whatever it was at import.
    p = G.plan_best(w, check, cube, table_margin, grasp_margin=grasp_margin, accept=accept,
                    hover_floor=MG.HOVER_CLEAR_MM, ladder=G.HOVER_LADDER, **kw)
    if "why" in p:
        first = p["why"][1] if len(p["why"]) > 1 else p["why"][0]
        return {"ok": False, "stage": "no plan", "note": first[:90]}

    h = p["hover"]
    room = float(min(G.limit_room(q) for q in p["path"]))
    note = "hover %.0f mm, shoulder %.0f, room %.1f" % (p["hover_mm"], h[0], room)
    start = list(MG.HIGH_POSE)

    if not zone.contains(MG.HIGH_POSE):
        return {"ok": False, "stage": "HIGH_POSE not allowed by the map", "note": note}
    if not zone.path_ok(start[:3], MG.HIGH_POSE) or not zone.path_ok(MG.HIGH_POSE, h[:3]):
        return {"ok": False, "stage": "the map will not route up or across", "note": note}

    try:
        MG.check_arm_move(check, cube, start[:3], MG.HIGH_POSE, level, h[5], leaving=True, what="up from home")
    except Exception as e:
        return {"ok": False, "stage": "the move up from home", "note": note + " | " + str(e)[:70]}
    try:
        MG.check_base_turn(check, cube, list(MG.HIGH_POSE) + list(level), 0.0, h[5])
    except Exception as e:
        return {"ok": False, "stage": "the base turn", "note": note + " | " + str(e)[:70]}
    try:
        MG.check_arm_move(check, cube, MG.HIGH_POSE, h[:3], wide, h[5], arriving=True, what="across to the hover")
    except Exception as e:
        return {"ok": False, "stage": "the move across to the hover", "note": note + " | " + str(e)[:70]}

    hand_now = [list(h[:3]) + [r, g] for r, g in zip(np.linspace(wide[0], h[3], 6), np.linspace(wide[1], h[4], 6))]
    if min(MG.cube_gap(check, cube, q, h[5]) for q in hand_now) < MG.HOVER_CLEAR_MM:
        return {"ok": False, "stage": "turning the hand at the hover (block)", "note": note}
    if not exact_ok(zone, h[:3], h[5], [wide, (h[3], h[4])]):
        return {"ok": False, "stage": "turning the hand at the hover (table)", "note": note}

    for q in p["path"]:
        if check.gaps(list(q[:5]), float(q[5]))[0] < grasp_margin:
            return {"ok": False, "stage": "the way in (table)", "note": note}
    return {"ok": True, "stage": "", "note": note}


_W = {}


def _init(data_dir, opt):
    """Each worker builds the world, the table check and the safety map once, then reuses them."""
    from armlab.state import S
    S.data_dir = data_dir
    S.exact_table = True
    S.table_margin_mm = opt["table_margin"]
    from armlab import grasp as G
    from armlab.config import ROLL_SLIDER
    from armlab.modes import grasp as MG
    from armlab.modes.survey import exact_ok
    from armlab.safety import Zone
    from armlab.tablecheck import load as load_check
    for k, v in (("hover_clear", "HOVER_CLEAR_MM"), ("cube_clear", "CUBE_CLEAR_MM"),
                 ("cube_r", "CUBE_R_MM")):
        if opt.get(k) is not None:
            setattr(MG, v, opt[k])
    for k, v in (("pitch_drift", "PATH_PITCH_DRIFT"), ("square_tol", "SQUARE_TOL_DEG")):
        if opt.get(k) is not None:
            setattr(G, v, opt[k])
    if opt.get("out_approach"):
        o = float(opt["out_approach"])
        G.APPROACHES = tuple(G.APPROACHES) + (("out and up", 0.0, o), ("out, up and along", 0.4, o))
    if opt.get("ladder"):
        G.HOVER_LADDER = tuple(float(x) for x in opt["ladder"].split(","))
    check = load_check()
    zone = Zone(opt["margin"])
    lo, hi = G.set_limits(zone)
    if opt.get("shoulder_hi") is not None:
        G.JOINT_HI = (opt["shoulder_hi"],) + tuple(G.JOINT_HI[1:])
        hi = G.JOINT_HI
    _W.update(G=G, MG=MG, exact_ok=exact_ok, zone=zone, check=check, w=check.w, roll=ROLL_SLIDER,
              opt=opt, lo=lo, hi=hi)


def _work(f):
    d = json.load(open(f))
    if d.get("kind") != "snap":
        return None
    b = d["block"]
    o, w = _W["opt"], _W["w"]
    xyz = np.asarray(b["xyz"], float)
    if not o["raw_height"]:
        fixed = corrected_top(w, b)
        if fixed is not None:
            xyz = xyz.copy()
            xyz[2] = w.table_height(xyz[0], xyz[1]) + fixed
    if o.get("shorter"):
        xyz = xyz.copy()
        xyz[2] -= float(o["shorter"])
    cube = {"xyz": xyz, "yaw_deg": float(b["yaw_deg"])}
    got = try_one(_W["G"], _W["MG"], _W["exact_ok"], _W["zone"], _W["check"], w, cube,
                  o["table_margin"], o["grasp_margin"], o["grasp_depth"], _W["roll"])
    got.update(out_mm=b["out_mm"], bearing=b["bearing_deg"], yaw=b["yaw_deg"], file=os.path.basename(f))
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="a copy of ~/arm_data (world.json, table_model.json, arm_map.npz, home_pose.json)")
    ap.add_argument("snaps", nargs="+", help="snap_*.json files, or a folder holding them")
    ap.add_argument("--margin", type=float, default=40.0)
    ap.add_argument("--table-margin", type=float, default=15.0)
    ap.add_argument("--grasp-margin", type=float, default=12.0)
    ap.add_argument("--hover-clear", type=float, default=None)
    ap.add_argument("--cube-clear", type=float, default=None)
    ap.add_argument("--out-approach", type=float, default=None,
                    help="also try backing off this much radially outward from the base axis, 0..0.8. The stock "
                         "approaches blend 'back along the fingers' with 'straight up', which at the near-"
                         "vertical pitch a close block needs are the same direction -- so the hover always "
                         "lands straight above the grasp and needs vertical room the arm does not have")
    ap.add_argument("--shorter", type=float, default=None,
                    help="pretend the block is this many mm shorter, to see what a shorter one would buy. The "
                         "probe says a lower grasp has more room above it (depth 20 leaves a median 10 mm of "
                         "lift, depth 10 only 5), so block height may be what the close range really needs")
    ap.add_argument("--cube-r", type=float, default=None,
                    help="the radius of the no-go cylinder standing on the block, mm (default 40). The real "
                         "block is about 28 mm half-diagonal, so 40 refuses hands that would clear it")
    ap.add_argument("--grasp-depth", type=float, default=None)
    ap.add_argument("--pitch-drift", type=float, default=None)
    ap.add_argument("--square-tol", type=float, default=None,
                    help="how far off square to a block face the fingers may close, deg (default 20). Squeezing "
                         "turns the block square, which is why there is a tolerance at all")
    ap.add_argument("--shoulder-hi", type=float, default=None,
                    help="let the shoulder go this high instead of the map's ceiling. The stall is posture "
                         "dependent -- it holds past 66 with the elbow folded, which is the close-block case -- "
                         "and the map carries one number for every posture")
    ap.add_argument("--raw-height", action="store_true",
                    help="use the block height as the snapshot recorded it, without taking the cameras known "
                         "error off (the snapshots predate cam_top_bias)")
    ap.add_argument("--ladder", default=None,
                    help="the hover heights to try, comma separated. The default ladder bottoms out at 15 mm, "
                         "and close blocks have a median 10 mm of lift above the grasp -- so for those the "
                         "planner has been asking for a height they cannot give, whatever the clearance floor")
    ap.add_argument("--jobs", type=int, default=None, help="how many cores (default: all of them)")
    ap.add_argument("--out", default=None, help="write the results as json")
    a = ap.parse_args()

    opt = {k: getattr(a, k) for k in ("margin", "table_margin", "grasp_margin", "hover_clear", "cube_clear",
                                      "grasp_depth", "pitch_drift", "square_tol", "shoulder_hi", "raw_height")}
    data_dir = os.path.abspath(a.data_dir)

    files = []
    for sp in a.snaps:
        files += sorted(glob.glob(os.path.join(sp, "snap_*.json"))) if os.path.isdir(sp) else [sp]

    # The blocks do not depend on each other, and planning one is 20-40 s, so run them across the cores.
    n = max(1, min(a.jobs or (os.cpu_count() or 2), len(files)))
    print("%d blocks on %d cores   table margin %.0f   depth %s   shoulder ceiling %s"
          % (len(files), n, a.table_margin,
             "default" if a.grasp_depth is None else "%.0f" % a.grasp_depth,
             "map" if a.shoulder_hi is None else "%.0f" % a.shoulder_hi))
    print()

    rows = []
    if n == 1:
        _init(data_dir, opt)
        got = [_work(f) for f in files]
    else:
        import multiprocessing as mp
        with mp.Pool(n, initializer=_init, initargs=(data_dir, opt)) as pool:
            got = pool.map(_work, files)
    for r in got:
        if r is None:
            continue
        rows.append(r)
        print("  %4.0f mm %+5.0f turn %+5.0f   %s  %s"
              % (r["out_mm"], r["bearing"], r["yaw"], "GRABS" if r["ok"] else "-    ", r["note"]))

    ok = sum(1 for r in rows if r["ok"])
    print()
    print("%d of %d would get all the way to the fingers closing" % (ok, len(rows)))
    by = {}
    for r in rows:
        if not r["ok"]:
            by[r["stage"]] = by.get(r["stage"], 0) + 1
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print("   %3d stopped at: %s" % (v, k))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print("\nwritten to " + a.out)


if __name__ == "__main__":
    main()
