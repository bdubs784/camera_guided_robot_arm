#!/usr/bin/env python3
"""
A dense plan table, grown from a sparse one the way the Pi uses it.

    python tools/grow_plan_table.py <arm_data dir> --seeds <plan_table.json> [--step 10] [--yaws 12] [--jobs 8]

build_plan_table.py runs plan_best from scratch at every grid point: 10-60 s a point close in, where the block's middle
has no straight way in and the planner walks the hover ladder at every pull (a 272 mm block took 139 s on
the Pi when the table had nothing near it). But the Pi never uses the table that way: it takes the nearest entries
and re-plans them for the block it sees (grasp.plan_seeded, ~0.3 s on the laptop). So grow the table with that same
call: every point of a fine grid tries plan_seeded from the entries already known (the seed table, then everything
grown so far); only points nothing seeds get plan_best from scratch. Each entry is then one the Pi's own lookup is
known to reach from its neighbours, and a 10 mm x 12-turn grid costs minutes instead of hours.

Only straight ways in that the gripper camera sees from the hover are kept (plan_seeded's own rule): the joint-space
way in held the block 0 times in 9 on record. Margins and depth must match the grab's (show/grasp: --grasp-depth 10,
table margin 15, grasp margin 12); the seed table must be built for the same world.json / table_model / arm_map.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
sys.path.insert(0, HERE)

from build_plan_table import KEEP, _S, _setup, hover_clear      # noqa: E402  (same workers as the plain build)


def _seeds(table):
    _S["table"] = table


def _grow(job):
    x, y, yaw, scratch = job
    G, MG, check, w, o = _S["G"], _S["MG"], _S["check"], _S["w"], _S["opt"]
    cube = {"xyz": np.array([x, y, w.table_height(x, y) + MG.CUBE_MM]), "yaw_deg": yaw}
    ok = lambda pp: hover_clear(MG, check, cube, pp)
    p = G.plan_seeded(w, check, cube, o["table_margin"], _S["table"], accept=ok, tries=_S["tries"],
                      depth=o["grasp_depth"], grasp_margin=o["grasp_margin"])
    if p is None and scratch:
        p = G.plan_best(w, check, cube, o["table_margin"], grasp_margin=o["grasp_margin"], accept=ok,
                        hover_floor=MG.HOVER_CLEAR_MM, depth=o["grasp_depth"])
        if "why" in p or not p.get("sees", True) or p.get("approach") in G.JOINT_APPROACHES:
            p = None
    if p is None:
        return None
    return {"xy": [round(x, 1), round(y, 1)], "yaw": yaw,
            "plan": {k: (np.asarray(p[k], float).round(3).tolist() if k == "grasp" else p[k]) for k in KEEP}}


def grid(axis, step, r_min, r_max, b0, b1):
    pts = []
    for x in np.arange(axis[0] - r_max, axis[0] + r_max + 1e-9, step):
        for y in np.arange(axis[1] - r_max, axis[1] + r_max + 1e-9, step):
            r = math.hypot(x - axis[0], y - axis[1])
            b = math.degrees(math.atan2(y - axis[1], x - axis[0]))
            if r_min <= r <= r_max and b0 <= b <= b1:
                pts.append((float(x), float(y)))
    return pts


def run(pool_n, data_dir, opt, table, jobs, tries):
    # workers are replaced every 100 jobs: the planner's caches grow, and the laptop has ~15 GB for all of them
    with mp.Pool(pool_n, initializer=_init_all, initargs=(data_dir, opt, table, tries), maxtasksperchild=100) as pool:
        return list(pool.imap(_grow, jobs, chunksize=8))


def _init_all(d, o, t, tries):
    _setup(d, o)
    _S["tries"] = tries
    _seeds(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--seeds", nargs="+", required=True, help="plan table(s) built for the same files")
    ap.add_argument("--step", type=float, default=10.0)
    ap.add_argument("--yaws", type=int, default=12)
    ap.add_argument("--r-min", type=float, default=250.0)
    ap.add_argument("--r-max", type=float, default=390.0)
    ap.add_argument("--bearings", default="-50,67", help="degrees round the base axis, from,to")
    ap.add_argument("--tries", type=int, default=8, help="nearest entries plan_seeded tries per point")
    ap.add_argument("--rounds", type=int, default=3, help="seeded passes before the from-scratch one")
    ap.add_argument("--no-scratch", action="store_true", help="skip plan_best for points nothing seeds")
    ap.add_argument("--grasp-depth", type=float, default=10.0)
    ap.add_argument("--table-margin", type=float, default=15.0)
    ap.add_argument("--grasp-margin", type=float, default=12.0)
    ap.add_argument("--hover-clear", type=float, default=0.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from armlab import grasp as G
    from armlab.world import World
    w = World(os.path.join(args.data_dir, "world.json"))
    stamp = G.plan_table_stamp(args.data_dir)
    seeds = []
    for f in args.seeds:
        t = json.load(open(f))
        if t.get("stamp") != stamp or t.get("depth_mm") != args.grasp_depth or t.get("margin") != args.table_margin \
                or t.get("grasp_margin") != args.grasp_margin:
            sys.exit(f"{f} was built for other files or margins than {args.data_dir}: rebuild it or pick another")
        seeds += [e for e in t["plans"] if e["plan"].get("approach") not in G.JOINT_APPROACHES]
    b0, b1 = (float(v) for v in args.bearings.split(","))
    pts = grid(w.axis[:2], args.step, args.r_min, args.r_max, b0, b1)
    yaws = [round(-45.0 + 90.0 * (k + 0.5) / args.yaws, 2) for k in range(args.yaws)]
    todo = [(x, y, yaw) for x, y in pts for yaw in yaws]
    print(f"{len(pts)} points ({args.step:.0f} mm, {args.r_min:.0f}-{args.r_max:.0f} mm, bearings {b0:.0f}..{b1:.0f}) "
          f"x {len(yaws)} turns = {len(todo)}; {len(seeds)} seed plans; {args.jobs} workers")
    opt = {"margin": 40.0, "table_margin": args.table_margin, "grasp_margin": args.grasp_margin,
           "grasp_depth": args.grasp_depth, "hover_clear": args.hover_clear, "cube_clear": None, "cube_r": None,
           "pitch_drift": None, "square_tol": None, "shoulder_hi": None, "out_approach": None, "ladder": None,
           "raw_height": False, "shorter": None}
    base = {"stamp": stamp, "margin": args.table_margin, "grasp_margin": args.grasp_margin,
            "depth_mm": args.grasp_depth, "step_mm": args.step, "hover_clear_mm": args.hover_clear}
    out = args.out or os.path.join(args.data_dir, G.PLAN_TABLE_FILE)
    grown, new, t0 = [], [], time.time()
    k = 0
    for scratch in [False] * args.rounds + ([] if args.no_scratch else [True]):
        if not todo or (scratch is False and k and not new):          # a seeded pass that added nothing: on
            continue                                                  #   to the from-scratch one
        k += 1
        table = dict(base, plans=seeds + grown)
        got = run(args.jobs, args.data_dir, opt, table, [j + (scratch,) for j in todo], args.tries)
        new = [g for g in got if g]
        grown += new
        todo = [j for j, g in zip(todo, got) if not g]
        print(f"   pass {k} ({'from scratch' if scratch else 'seeded'}): +{len(new)}, {len(todo)} left, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        json.dump(dict(base, built=time.strftime("%Y-%m-%d %H:%M") + f" grown, pass {k}", plans=grown),
                  open(out, "w"))                            # after every pass: usable before the build ends
    json.dump(dict(base, built=time.strftime("%Y-%m-%d %H:%M") + " grown", plans=grown), open(out, "w"))
    print(f"{len(grown)} plans ({len(todo)} points with none) -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
