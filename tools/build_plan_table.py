#!/usr/bin/env python3
"""
The plan table: plan_best's answer for a grid of block positions over the mat, at a few block yaws.

    python tools/build_plan_table.py <arm_data dir> [--grasp-depth 10] [--step 20] [--yaws 6] [--jobs 8]

On the Pi a first plan from scratch was 15.6 s of a 107 s show round. With plan_table.json in
~/arm_data the grab re-plans the nearest entry for the block it actually sees (grasp.plan_seeded, ~1.4 s) and keeps
it only if it passes everything plan_best would ask; otherwise it plans from scratch as before. The table is stamped
with the md5 of world.json, table_model.json and arm_map.npz, and the Pi ignores it when any of them changed --
rebuild it after every world model / table model / safety map.

The grid covers the sector of the mat blocks have actually been on (every grasp run and snapshot found under
--blocks), widened by --pad, between --r-min and --r-max from the base axis. Points the planner cannot plan are
left out (the Pi plans those from scratch, and will fail the same way).

Margins and depth must match the grab's (show/grasp: --grasp-depth 10, table margin 15, grasp margin 12); the Pi
only uses a table built with its own.
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pi"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_S = {}
KEEP = ("pitch", "face_deg", "grip", "grasp", "approach", "hover_mm", "depth_mm", "pull_mm")   # what replan needs


def _setup(data_dir, opt):
    from replay_grasp import _init, _W
    _init(data_dir, opt)
    _S.update(_W)
    _S["opt"] = opt


def hover_clear(MG, check, cube, pp):
    """The grab's own acceptance test (hover_clear_of_block in modes/grasp.py): crossing with the fingers wide and
    turning the hand at the hover both stay HOVER_CLEAR_MM from the block."""
    h = pp["hover"]
    poses = (list(h[:3]) + [MG.ROLL_SLIDER, MG.HOVER_GRIP], list(h[:3]) + [h[3], h[4]])
    return min(MG.cube_gap(check, cube, q, h[5]) for q in poses) >= MG.HOVER_CLEAR_MM


def _one(job):
    x, y, yaw = job
    G, MG, check, w, o = _S["G"], _S["MG"], _S["check"], _S["w"], _S["opt"]
    cube = {"xyz": np.array([x, y, w.table_height(x, y) + MG.CUBE_MM]), "yaw_deg": yaw}
    p = G.plan_best(w, check, cube, o["table_margin"], grasp_margin=o["grasp_margin"],
                    accept=lambda pp: hover_clear(MG, check, cube, pp), hover_floor=MG.HOVER_CLEAR_MM,
                    depth=o["grasp_depth"])
    if "why" in p or not p.get("sees", True):          # plan_seeded only keeps plans that see the block
        return None
    return {"xy": [round(x, 1), round(y, 1)], "yaw": yaw,
            "plan": {k: (np.asarray(p[k], float).round(3).tolist() if k == "grasp" else p[k]) for k in KEEP}}


def grid(w, blocks, step, r_min, r_max, pad_deg):
    """xy points every `step` mm inside the sector [r_min, r_max] x (the bearings blocks have been at, +-pad_deg)."""
    ang = [math.degrees(math.atan2(b[1] - w.axis[1], b[0] - w.axis[0])) for b in blocks]
    a0, a1 = min(ang) - pad_deg, max(ang) + pad_deg
    pts = []
    for x in np.arange(w.axis[0] - r_max, w.axis[0] + r_max + 1e-9, step):
        for y in np.arange(w.axis[1] - r_max, w.axis[1] + r_max + 1e-9, step):
            r = math.hypot(x - w.axis[0], y - w.axis[1])
            a = math.degrees(math.atan2(y - w.axis[1], x - w.axis[0]))
            if r_min <= r <= r_max and a0 <= a <= a1:
                pts.append((float(x), float(y)))
    return pts, (a0, a1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="a copy of ~/arm_data (world.json, table_model.json, arm_map.npz, home_pose.json)")
    ap.add_argument("--blocks", nargs="+", default=[], help="folders of grasp runs / snapshots: where blocks have been")
    ap.add_argument("--grasp-depth", type=float, default=10.0)
    ap.add_argument("--table-margin", type=float, default=15.0)
    ap.add_argument("--grasp-margin", type=float, default=12.0)
    ap.add_argument("--hover-clear", type=float, default=None,
                    help="the grab's --hover-clear (show runs 0): with the default 12 the table has no straight way in "
                         "close to the tower, and those blocks fell back to a 30-40 s plan from scratch")
    ap.add_argument("--step", type=float, default=20.0, help="grid spacing, mm")
    ap.add_argument("--yaws", type=int, default=6, help="block yaws per point, evenly over the 90 deg it repeats in")
    ap.add_argument("--r-min", type=float, default=150.0)
    ap.add_argument("--r-max", type=float, default=460.0)
    ap.add_argument("--pad", type=float, default=10.0, help="degrees past the bearings blocks have been at")
    ap.add_argument("--jobs", type=int, default=8, help="workers (each loads the world and the safety map)")
    ap.add_argument("--out", default=None, help="default: <data_dir>/plan_table.json")
    args = ap.parse_args()

    from armlab import grasp as G
    from armlab.world import World
    w = World(os.path.join(args.data_dir, "world.json"))
    blocks = []
    for d in args.blocks:
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                b = json.load(open(f)).get("block")
            except (ValueError, OSError):
                continue
            if b and "xyz" in b:
                blocks.append(b["xyz"][:2])
    if not blocks:
        sys.exit("no blocks found under --blocks: give the grasp_runs / snaps folders, so the grid covers the mat")
    pts, (a0, a1) = grid(w, blocks, args.step, args.r_min, args.r_max, args.pad)
    yaws = [round(-45.0 + 90.0 * (k + 0.5) / args.yaws, 2) for k in range(args.yaws)]
    jobs = [(x, y, yaw) for x, y in pts for yaw in yaws]
    print(f"{len(pts)} points ({args.step:.0f} mm, {args.r_min:.0f}-{args.r_max:.0f} mm out, bearings "
          f"{a0:.0f} to {a1:.0f} deg) x {len(yaws)} yaws = {len(jobs)} plans on {args.jobs} workers")
    opt = {"margin": 40.0, "table_margin": args.table_margin, "grasp_margin": args.grasp_margin,
           "grasp_depth": args.grasp_depth, "hover_clear": args.hover_clear, "cube_clear": None, "cube_r": None,
           "pitch_drift": None, "square_tol": None, "shoulder_hi": None, "out_approach": None, "ladder": None,
           "raw_height": False, "shorter": None}
    t0, plans = time.time(), []
    with mp.Pool(args.jobs, initializer=_setup, initargs=(args.data_dir, opt)) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=4), 1):
            if r:
                plans.append(r)
            if k % 100 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"   {k}/{len(jobs)}  {len(plans)} planned  {el / 60:.1f} min, ~{el / k * (len(jobs) - k) / 60:.0f} min left",
                      flush=True)
    out = args.out or os.path.join(args.data_dir, G.PLAN_TABLE_FILE)
    json.dump({"stamp": G.plan_table_stamp(args.data_dir), "margin": args.table_margin,
               "grasp_margin": args.grasp_margin, "depth_mm": args.grasp_depth, "step_mm": args.step,
               "hover_clear_mm": args.hover_clear,
               "built": time.strftime("%Y-%m-%d %H:%M"), "plans": plans}, open(out, "w"))
    print(f"{len(plans)} of {len(jobs)} planned -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
