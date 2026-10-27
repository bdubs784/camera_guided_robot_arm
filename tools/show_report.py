#!/usr/bin/env python3
"""
One show, round by round: what each change should show if it works -- or if it is hurting.

    python tools/show_report.py <show json> [<grasp_runs folder>]     (default: grasp_runs next to the show folder)

Per round: where the block was picked up and which way in; the squeeze and the logged finger spacings; where the
set-down was aimed, its way in, how far the block ended up from where the fingers let go (joint-space set-downs
used to drag it 14-60 mm outward); whether the look after the set-down saw it (park views) and how long the next
round searched; and the round's time. Then the means per phase, including the time spent in the move checks.
"""
import argparse
import json
import os

import numpy as np


def radius(axis, xyz):
    return float(np.hypot(xyz[0] - axis[0], xyz[1] - axis[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("show")
    ap.add_argument("runs", nargs="?", default=None)
    args = ap.parse_args()
    show = json.load(open(args.show))
    runs = args.runs or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.show))), "grasp_runs")
    data = os.path.dirname(os.path.dirname(os.path.abspath(args.show)))
    try:
        axis = json.load(open(os.path.join(data, "world.json")))["base"]["axis_xy_mm"]
    except (OSError, KeyError, ValueError):
        axis = (-33.265, -22.613)
    print(f"{os.path.basename(args.show)}  build {show.get('version', {}).get('build', '?')}  radii {show.get('radii')}")
    print(" rnd  pick r  way in            squeeze  spacing b/h/s mm    | aim r  set-down way in      drift  | "
          "home look   next search | s")
    rows = show.get("rows") or []
    phases, checks, cpu, pans, pan_waits, off_aim = {}, [], [], [], [], []
    for i, r in enumerate(rows):
        path = os.path.join(runs, r.get("run") or "")
        g = json.load(open(path)) if r.get("run") and os.path.exists(path) else {}
        o = g.get("outcome") or {}
        p = g.get("plan") or {}
        cp = o.get("carry_plan") or {}
        sp = o.get("spacing_mm") or {}
        rel, fnd = o.get("released_xyz"), o.get("found_xyz")
        drift = f"{radius(axis, fnd) - radius(axis, rel):+5.1f}" if rel and fnd else "    -"
        if cp.get("spot") and fnd:                           # where it landed vs where the round aimed it
            off_aim.append((float(np.hypot(fnd[0] - cp["spot"][0], fnd[1] - cp["spot"][1])),
                            radius(axis, fnd) - radius(axis, cp["spot"])))
        res = str(r.get("result") or r.get("error") or "")
        home = res.split("from home ")[-1].rstrip(")") if "from home" in res else "-"
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        nsearch = dict((nxt or {}).get("times_s") or []).get("search") if nxt else None
        b = g.get("block") or {}
        print(f" {r['round']:3d}  {radius(axis, b['xyz']) if b.get('xyz') else 0:6.0f}  {str(p.get('approach', '-'))[:16]:16s}  "
              f"{str(o.get('squeeze'))[:7]:7s}  {sp.get('bottom') or 0:5.1f}/{sp.get('hover') or 0:5.1f}/{sp.get('squeezed') or 0:5.1f}"
              f"     | {cp.get('spot_r_mm') or 0:5.0f}  {str(cp.get('approach', '-'))[:19]:19s} {drift}  | "
              f"{home[:11]:11s} {('%5.1f' % nsearch) if nsearch else '    -'}       | {r.get('secs') or 0:4.0f}")
        for name, secs in o.get("times_s") or []:
            phases.setdefault(name, []).append(secs)
        cpu += (o.get("trace") or {}).get("cpu") or []
        for e in (o.get("trace") or {}).get("events") or []:
            if e[2] == "compute" and e[3] == "move check":
                checks.append(e[1] - e[0])
            elif e[2] == "compute" and str(e[3]).startswith("pan check"):     # --pan-overlap: the grid check, run in
                pans.append(e[1] - e[0])                                       # the background during the move before
            elif e[2] == "wait" and e[3] == "pan check (finishing)":           # what the arm waited for it after that
                pan_waits.append(e[1] - e[0])
    held = sum(1 for r in rows if r.get("held"))
    ok = sum(1 for r in rows if str(r.get("result", "")).startswith("GRABBED"))
    secs = [r.get("secs") or 0 for r in rows]
    print(f"\nheld {held}/{len(rows)}, confirmed {ok}/{len(rows)}, mean {np.mean(secs) if secs else 0:.1f} s a round")
    if off_aim:
        d, r = np.array(off_aim).T
        print(f"set-downs vs the aimed spot: {np.mean(d):.1f} mm mean, {np.max(d):.1f} worst; radially {np.mean(r):+.1f} "
              f"+- {np.std(r, ddof=1) if len(r) > 1 else 0:.1f} mm ({len(d)} found afterwards)")
    if phases:
        print("phase means (s): " + ", ".join(f"{k} {np.mean(v):.1f}" for k, v in
                                               sorted(phases.items(), key=lambda kv: -np.mean(kv[1]))))
    if cpu:
        c = np.array([[r[1], r[2], r[3] if r[3] is not None else np.nan] for r in cpu], float)
        print(f"cpu: Pi busy {np.mean(c[:, 0]):.0f}% mean ({np.percentile(c[:, 0], 90):.0f}% p90), this program "
              f"{np.mean(c[:, 1]):.0f}% of one core, temp max {np.nanmax(c[:, 2]):.1f} C")
    if checks:
        print(f"move checks: {sum(checks) / max(1, len(rows)):.1f} s a round ({len(checks)} checks, "
              f"median {np.median(checks) * 1000:.0f} ms)")
    if pans:
        print(f"pan checks: {sum(pans) / max(1, len(rows)):.2f} s a round in the background ({len(pans)} checks, median "
              f"{np.median(pans) * 1000:.0f} ms, max {max(pans) * 1000:.0f}); the arm waited for them "
              f"{sum(pan_waits) / max(1, len(rows)):.2f} s a round (max {max(pan_waits or [0]) * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
