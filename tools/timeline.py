#!/usr/bin/env python3
"""
Where each phase of a grab spent its time, from the trace the grab now records (armlab/trace.py).

    python tools/timeline.py <show json or grasp run files> [--round N] [--events]

For every phase (the same names as the timing table) it splits the time into what the arm was doing:

  arm        a shoulder/elbow/wrist glide          base       a base leg (the motor turning)
  hand       wrist rotate / gripper                camera     looking: a find, a closed-loop base check
  compute    planning, merging stops               wait       a fixed pause or waiting for a fresh frame
  base loop  inside the closed base loop but none of the above
  nothing    no traced activity at all -- the first place to look for time to win

When two overlap, the one first in that list counts (a hand move during a base turn counts as base). --events lists a
round's events in order; --round picks the round (default all, averaged).
"""
import argparse
import json
import os
import sys
from collections import defaultdict

ORDER = ("arm", "base", "hand", "camera", "compute", "wait", "base loop")
STEP_S = 0.01


def rounds_from(paths):
    out = []
    for p in paths:
        d = json.load(open(p))
        if "rows" in d:                                          # a show
            for r in d["rows"]:
                if r.get("trace") and r.get("t0"):
                    out.append((f"{os.path.basename(p)} round {r['round']}", r))
        elif (d.get("outcome") or {}).get("trace"):              # a grasp run file
            out.append((os.path.basename(p), d["outcome"]))
    return out


def phases(r):
    t, out = float(r["t0"]), []
    for name, secs in r["times_s"]:
        out.append((name, t, t + float(secs)))
        t += float(secs)
    return out


def split(events, t0, t1):
    """seconds per kind in [t0, t1], each instant counted once, by ORDER."""
    n = max(1, int(round((t1 - t0) / STEP_S)))
    label = [None] * n
    rank = {k: i for i, k in enumerate(ORDER)}
    for a, b, kind, _ in events:
        if b <= t0 or a >= t1 or kind not in rank:
            continue
        i0, i1 = max(0, int((a - t0) / STEP_S)), min(n, int((b - t0) / STEP_S) + 1)
        for i in range(i0, i1):
            if label[i] is None or rank[kind] < rank[label[i]]:
                label[i] = kind
    got = defaultdict(float)
    for x in label:
        got[x or "nothing"] += STEP_S
    return got


def waits(events, t0, t1):
    got = defaultdict(float)
    for a, b, kind, what in events:
        if kind == "wait" and b > t0 and a < t1:
            got[what] += min(b, t1) - max(a, t0)
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--round", type=int, default=None, help="only this show round")
    ap.add_argument("--events", action="store_true", help="list the round's events in order")
    args = ap.parse_args()
    rs = rounds_from(args.files)
    if args.round is not None:
        rs = [x for x in rs if x[0].endswith(f"round {args.round}")]
    if not rs:
        sys.exit("no traced rounds in those files")
    total = defaultdict(lambda: defaultdict(float))
    wait_total = defaultdict(lambda: defaultdict(float))
    count = defaultdict(int)
    for name, r in rs:
        ev = r["trace"]["events"]
        if args.events:
            print(f"\n== {name}")
            for a, b, kind, what in sorted(ev):
                print(f"   {a - r['t0']:7.2f} {b - a:6.2f} s  {kind:9s} {what}")
        for ph, a, b in phases(r):
            for k, v in split(ev, a, b).items():
                total[ph][k] += v
            for k, v in waits(ev, a, b).items():
                wait_total[ph][k] += v
            count[ph] += 1
    cols = list(ORDER) + ["nothing"]
    print(f"\nmean seconds per phase over {len(rs)} round(s):")
    print("   %-12s %6s  " % ("phase", "total") + " ".join("%9s" % c for c in cols))
    for ph in sorted(total, key=lambda p: -sum(total[p].values())):
        n = count[ph]
        print("   %-12s %6.1f  " % (ph, sum(total[ph].values()) / n) + " ".join(
            "%9s" % ("%.1f" % (total[ph][c] / n) if total[ph][c] / n >= 0.05 else "") for c in cols))
    print("\nthe waits, by name (mean seconds per round):")
    for ph in sorted(wait_total, key=lambda p: -sum(wait_total[p].values())):
        top = sorted(wait_total[ph].items(), key=lambda kv: -kv[1])
        print("   %-12s " % ph + ", ".join(f"{k} {v / count[ph]:.1f}" for k, v in top if v / count[ph] >= 0.05))


if __name__ == "__main__":
    main()
