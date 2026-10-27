#!/usr/bin/env python3
"""
show's set-down plan, replayed off the arm for every round of saved shows: where it found a spot, or why every
candidate was turned down.

    python tools/replay_place.py <arm_data copy> <show json> [<show json> ...] [--workers 6]

Each round: the close look's block, re-planned (plan_best + replan, as the grab does), then place_shift toward the
round's want_r with show's practice area (show.py: min radius - AREA_PAD_MM .. max radius). Every candidate spot
is counted at the step that turned it down: under the way home (home_clear), the hand cannot get there
(reachable_at), no set-down plan (place_plan), or it would leave the block where nothing grabs it (pickable_after).
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
sys.path.insert(0, HERE)

_W = {}


def _init(data_dir):
    import replay_grasp
    replay_grasp._init(data_dir, {"table_margin": 15.0, "margin": 40.0})
    _W.update(replay_grasp._W)


def _work(job):
    import random
    G, MG, check, w = _W["G"], _W["MG"], _W["check"], _W["w"]
    MG.PRACTICE_R, MG.CARRY_TO_R = tuple(job["area"]), job["want"]
    MG.HOME_SWEEP.clear()
    if job.get("home_base") is not None:                 # the way home this round would take (home_clear)
        MG.HOME_SWEEP.update(home=list(_W["zone"].home), base=float(job["home_base"]))
    near = {"xyz": np.asarray(job["xyz"], float), "yaw_deg": job["yaw"]}
    t0 = time.time()
    p = G.plan_best(w, check, near, 15.0, grasp_margin=12.0, hover_floor=MG.HOVER_CLEAR_MM, ladder=G.HOVER_LADDER,
                    depth=job["depth"])
    if "why" in p:
        return dict(job, result="no pickup plan", secs=time.time() - t0)
    p2 = G.replan(w, check, near, p, 15.0, grasp_margin=12.0)
    if "why" in p2:
        return dict(job, result="no re-plan", secs=time.time() - t0)
    n = {"spots": 0, "home": 0, "reach": 0, "plan": 0, "grab_after": 0}
    orig = {k: getattr(MG, k) for k in ("home_clear", "reachable_at", "place_plan", "pickable_after")}
    radii = []

    def home_clear(check_, xy, top):
        g = orig["home_clear"](check_, xy, top)
        n["spots"] += 1
        if g is not None and g < MG.PLACE_HOME_CLEAR_MM:
            n["home"] += 1
        return g

    def reachable_at(w_, near_, p2_, xy, *a):
        ok = orig["reachable_at"](w_, near_, p2_, xy, *a)
        radii.append((round(float(np.linalg.norm(np.asarray(xy) - w_.axis[:2]))), bool(ok)))
        if not ok:
            n["reach"] += 1
        return ok

    def place_plan(*a, **k):
        got = orig["place_plan"](*a, **k)
        if got is None:
            n["plan"] += 1
        return got

    def pickable_after(*a, **k):
        ok = orig["pickable_after"](*a, **k)
        if not ok:
            n["grab_after"] += 1
        return ok
    MG.home_clear, MG.reachable_at, MG.place_plan, MG.pickable_after = home_clear, reachable_at, place_plan, \
        pickable_after
    try:
        d, plan2 = MG.place_shift(w, check, near, p2, 15.0, 12.0, random.Random(job["round"]), dq=None, floor=9.0)
    finally:
        for k, v in orig.items():
            setattr(MG, k, v)
    out = dict(job, secs=round(time.time() - t0, 1), counts=n, radii=radii, pick_approach=p2.get("approach"),
               pick_pitch=round(float(p2["pitch"]), 1))
    if plan2 is None:
        out["result"] = "NO SPOT"
    else:
        out.update(result="spot", spot_r=round(float(plan2["spot_r_mm"])), approach=plan2.get("approach"),
                   pitch=round(float(plan2["pitch"]), 1), firm=bool(MG.firm(plan2["grasp"])))
        if plan2.get("approach") in G.JOINT_APPROACHES:      # the straight-up lift after letting go (release_lift)
            out["lift"] = MG.release_lift(w, check, plan2, np.zeros(6), 9.0) is not None
            tips = np.array([w.tip(q[:4], q[5], grip=G.GRIP_OPEN) for q in plan2["path"]])
            h = np.linalg.norm(np.diff(tips[:, :2], axis=0), axis=1)
            k = len(h)
            while k > 0 and h[k - 1] < 1.5:
                k -= 1
            out["straight"] = float(tips[k, 2] - tips[-1, 2])  # how far the last part of the way in is straight down
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("shows", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--depth", type=float, default=None, help="grasp depth (default: each round's --grasp-depth)")
    a = ap.parse_args()
    from armlab.modes import show as SH
    jobs = []
    for f in a.shows:
        d = json.load(open(f))
        radii = d.get("radii") or list(SH.RADII)
        area = (min(radii) - SH.AREA_PAD_MM, max(radii))
        runs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(f))), "grasp_runs")
        for r in d.get("rows") or []:
            path = os.path.join(runs, r.get("run") or "")
            if not r.get("run") or not os.path.exists(path):
                continue
            g = json.load(open(path))
            o = g.get("outcome") or {}
            xyz = o.get("close_look_xyz") or (g.get("block") or {}).get("xyz")
            if xyz is None:
                continue
            cp = o.get("carry_plan") or {}
            looks = o.get("looks_raw") or []
            home_base = looks[0]["joints"][5] if looks and looks[0].get("joints") else None
            jobs.append({"show": os.path.basename(f)[9:15], "round": r["round"], "want": r["want_r_mm"], "area": area,
                         "xyz": xyz, "yaw": (g.get("block") or {}).get("yaw_deg", 0.0),
                         "r": round(float(np.hypot(xyz[0] + 33.265, xyz[1] + 22.613))),
                         "arm_spot": cp.get("spot_r_mm"), "arm_approach": cp.get("approach"),
                         "home_base": home_base,
                         "depth": a.depth or (g.get("args") or {}).get("grasp_depth") or 20.0})
    with Pool(a.workers, initializer=_init, initargs=(a.data_dir,)) as pool:
        res = pool.map(_work, jobs)
    print(" show   rnd  block r  want | on the arm          | replay: result  spot  set-down way in       firm  "
          "| spots home reach plan grab-after | s")
    for x in res:
        arm = (f"{x['arm_spot']:4.0f} {str(x['arm_approach'])[:14]:14s}" if x.get("arm_spot")
               else "back in place      ")
        c = x.get("counts") or {}
        print(f" {x['show']} {x['round']:3d}  {x['r']:6d}  {x['want']:5.0f} | {arm} | {x['result']:7s} "
              f"{x.get('spot_r', 0):5.0f}  {str(x.get('approach', '-'))[:20]:20s} {str(x.get('firm', '-')):5s} | "
              f"{c.get('spots', 0):5d} {c.get('home', 0):4d} {c.get('reach', 0):5d} {c.get('plan', 0):4d} "
              f"{c.get('grab_after', 0):10d} | {x['secs']:5.1f}"
              + ("" if "lift" not in x else "  straight-up lift %s, straight %.0f mm" % ("ok" if x["lift"] else "NO",
                                                                                           x["straight"])))


if __name__ == "__main__":
    main()
