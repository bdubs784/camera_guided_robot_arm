"""
show: the demo loop. Round after round with nobody in the way -- find the block, grab it, carry it to a NEW
distance from the base, set it down, and go straight into the next grab there.

The point is to show the arm's whole range. Each round is given a distance to aim for, and the set-down walks the
block toward it, so the recording shows grabs from ~270 to ~365 mm out instead of the same reach every time. A
round that can't move it the whole way moves it as far as it can and the next round carries on from there.

    python3 arm.py show --rounds 8
    python3 arm.py show --rounds 12 --radii 250,300,340,385 --no-save

Everything a grab does is unchanged: the same planner, the same checks, the same squeeze and lift confirmation.
This only chooses where the block is put down and keeps score. A round that fails is printed and the loop goes
on to the next one -- the arm comes home by itself -- so a bad grab does not end the recording.
"""
import json
import os
import time

import numpy as np

from ..errors import ArmError
from ..state import S
from . import grasp as G
from .. import live
from .grasp import Stuck

# The distances the block is walked along, out and back. Inside 255 mm only the joint-space way in plans (it has
# never held the block), and set-downs scatter ~+-10 mm, so the near end is 270. Pickups hold out to ~385 mm, but
# set-downs land further out the further out they are and one aimed at 375 slid out of reach, so the far end is 365.
RADII = (270.0, 305.0, 340.0, 365.0)
SHOW_SPEED = 45.0                      # deg/s with nothing in the hand (--speed overrides; the default elsewhere is 30)
SHOW_SLOW = 16.0                       # deg/s beside the block (--slow overrides; PICK_SLOW elsewhere is 8)
GIVE_UP_AFTER = 3                     # consecutive rounds that go wrong before the loop stops on its own
AREA_PAD_MM = 15.0                     # let the block be set down this far outside the radii, so the practice-area
                                       # check does not refuse the very spot a round is aiming at


def ladder(radii, rounds):
    """The radius each round aims at: out along the ladder, then back down it, repeating. Out and back rather than
    jumping to a random one, because the block can only be carried CARRY_MAX_MM a round -- walking it keeps every
    round's carry reachable, and on camera it reads as the arm working its way out to its limit and back."""
    walk = list(radii) + list(radii)[-2:0:-1]
    return [walk[i % len(walk)] for i in range(rounds)]


def mode_show(args):
    radii = [float(v) for v in str(args.radii).split(",") if v.strip()] if args.radii else list(RADII)
    if len(radii) < 2:
        raise ArmError("--radii wants at least two distances, e.g. --radii 250,300,350,385")
    want = ladder(radii, args.rounds)
    if getattr(args, "speed", None) is None:     # show's own speeds unless given on the command line
        args.speed = SHOW_SPEED
    if getattr(args, "slow", None) is None:
        args.slow = SHOW_SLOW
    lo, hi = min(radii) - AREA_PAD_MM, max(radii)       # no padding past the far end: that is the edge of reach
    was_area, was_to = G.PRACTICE_R, G.CARRY_TO_R
    G.PRACTICE_R = (lo, hi)                       # the carry may put it anywhere on the ladder, not just 230-350
    print(f"Demo: {args.rounds} rounds, walking the block {'/'.join('%.0f' % r for r in radii)} mm from the base "
          f"axis and back.\n   (the set-down area is {lo:.0f}-{hi:.0f} mm; everything else is an ordinary grab)")
    args.grab, args.dry, args.hover, args.snap = True, False, False, False
    from ..version import stamp
    print(f"   build {stamp()['build']}  (the code's hash; the saved run has it with every flag)")
    url = live.serve()
    if url:
        print(f"   demo page: {url}   (the cameras stay at :8080)")
    live.update(rounds=args.rounds, score={"picked": 0, "rounds": 0}, radii=list(radii), history=[],
                bearings=list(G.PRACTICE_BEARING))
    film = None
    if getattr(args, "film", False):                # filmed from here to the scoreboard, however the show ends
        from ..film import Film
        film = Film().start()
        print(f"   filming the cameras into {film.out}")
    rows, bad, final = [], 0, None
    t0 = time.time()
    try:
        for i, r in enumerate(want, 1):
            G.CARRY_TO_R = r
            live.update(round=i, want_r=r)
            live.event(f"round {i}: set it down ~{r:.0f} mm out")
            print(f"\n{'=' * 78}\nROUND {i} of {args.rounds}: grab it, then put it down near {r:.0f} mm out\n{'=' * 78}")
            t1 = time.time()
            row = {"round": i, "want_r_mm": r}
            try:
                G.mode_grasp(args)
                row.update(_outcome())
            except Stuck:
                raise                              # the arm is stranded: a human has to free it, so stop here
            except KeyboardInterrupt:              # stopped by hand: keep the round, it is the interesting one
                row.update(_outcome())
                row["error"] = "stopped by hand (Ctrl+C)"
                row["secs"] = round(time.time() - t1, 1)
                rows.append(row)
                raise
            except Exception as e:
                row.update(_outcome())
                row["error"] = str(e) or type(e).__name__
                print(f"   round {i} stopped: {row['error']}")
                bad += 1
                if bad >= GIVE_UP_AFTER:           # the same thing failing over and over is not a demo: something
                    print(f"\n{bad} rounds in a row went wrong. Stopping rather than filming the same failure.")
                    rows.append(row)
                    break
            else:
                bad = 0 if row.get("result") else bad + 1   # a round with no plan is a round gone wrong too
                if bad >= GIVE_UP_AFTER:
                    print(f"\n{bad} rounds in a row went wrong ({row.get('error', 'no result')}). Stopping.")
                    row["secs"] = round(time.time() - t1, 1)
                    rows.append(row)
                    break
            row["secs"] = round(time.time() - t1, 1)
            rows.append(row)
            _live_score(rows)
            print(f"   round {i}: {row.get('result', 'no result recorded')}  ({row['secs']:.0f} s)")
        G.come_down(args)                          # --stay-up: the last round left it waiting up high
        if rows and getattr(args, "final_look", False):   # off by default: a full search at the end wastes time
            final = _last_look(args, rows[-1])
    except KeyboardInterrupt:
        print("\nStopped by hand." + ("  The arm is up high: python3 arm.py home" if G.STAY_UP else ""))
    finally:
        G.PRACTICE_R, G.CARRY_TO_R = was_area, was_to
        if film is not None:
            out, n = film.stop()
            print(f"   film: {n} frames in {out}")
        _scoreboard(rows, time.time() - t0, None if args.no_save else _save(args, radii, rows, final))


def _live_score(rows):
    """The demo page's scoreboard."""
    held = [r for r in rows if r.get("held") and not r.get("error")]
    reach = [r["out_mm"] for r in held if "out_mm" in r]
    live.update(history=[[r.get("round"), r.get("secs"), "ok" if _good(r) else "held" if r.get("held") else "miss",
                          r.get("out_mm"), r.get("want_r_mm")] for r in rows])
    live.update(score={"picked": len(held), "rounds": len(rows), "rmin": min(reach) if reach else None,
                       "rmax": max(reach) if reach else None, "placed": len([r for r in rows if _good(r)])})


def _outcome():
    """What the run file says happened, if mode_grasp got far enough to write one."""
    path = G.LAST_RUN_PATH[0]
    if not path or not os.path.exists(path):
        return {}
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return {}
    got = dict(d.get("outcome") or {})
    got["run"] = os.path.basename(path)
    if not got.get("result") and d.get("why"):           # no plan at all: record why instead of a blank round
        got["error"] = "no plan: " + str((d["why"] or ["?"])[0])[:160]
    b = d.get("block") or {}
    if "out_mm" in b:
        got["out_mm"] = round(float(b["out_mm"]), 1)
    elif b.get("xyz"):                                    # grab runs keep the block's xyz, not its distance
        try:
            ax = json.load(open(os.path.join(S.data_dir, "world.json")))["base"]["axis_xy_mm"]
            got["out_mm"] = round(float(np.hypot(b["xyz"][0] - ax[0], b["xyz"][1] - ax[1])), 1)
        except (OSError, ValueError, KeyError):
            pass
    return got


def _last_look(args, last):
    """After the last round: where the block really ended up (find_block_anywhere), against where that round aimed
    and where its fingers let go. Saved in the show json as "final"."""
    print(f"\n{'=' * 78}\nWhere did the last set-down leave it? Looking (from home, else the full search)...")
    try:
        cube, w = G.find_block_anywhere(args)
    except Exception as e:
        print(f"   could not look: {e}")
        return {"error": str(e)}
    if cube is None:
        print("   no camera found the block anywhere")
        return {"found": False}
    xyz = np.asarray(cube["xyz"], float)
    out = {"found": True, "xyz": [round(float(v), 1) for v in xyz], "yaw_deg": round(float(cube["yaw_deg"]), 1),
           "out_mm": round(float(np.linalg.norm(xyz[:2] - w.axis[:2])), 1), "looks_raw": list(G.LOOKS_RAW)}
    say = f"   found it {out['out_mm']:.0f} mm out, turned {out['yaw_deg']:+.0f} deg"
    for key, name in (("released_xyz", "where the fingers let go"), ("found_xyz", "where the close-up check saw it")):
        ref = last.get(key)
        if ref:
            dd = float(np.linalg.norm(xyz[:2] - np.asarray(ref, float)[:2]))
            out[key.replace("_xyz", "_off_mm")] = round(dd, 1)
            say += f"; {dd:.0f} mm from {name}"
    spot = (last.get("carry_plan") or {}).get("spot")
    if spot:
        dd = float(np.linalg.norm(xyz[:2] - np.asarray(spot, float)[:2]))
        out["aim_off_mm"] = round(dd, 1)
        say += f"; {dd:.0f} mm from where it was aimed"
    print(say)
    return out


def _save(args, radii, rows, final=None):
    d = os.path.expanduser("~/arm_data/show")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, time.strftime("%Y%m%d_%H%M%S") + ".json")
    from ..version import stamp
    json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "version": stamp(),       # what code and data, which flags
               "args": {k: v for k, v in sorted(vars(args).items()) if not callable(v)},
               "radii": radii, "rounds": args.rounds, "rows": rows, "final": final}, open(path, "w"), indent=1,
              default=str)
    return path


def _good(r):
    """A round that counts. The run file is written before the arm moves and filled in as it goes, so a round that
    threw part way can still have GRABBED sitting in it from before whatever went wrong -- the error wins, or the
    scoreboard flatters itself."""
    return not r.get("error") and str(r.get("result", "")).startswith("GRABBED")


def _scoreboard(rows, secs, path):
    print(f"\n{'=' * 78}\nDEMO: {len(rows)} rounds in {secs / 60.0:.1f} min")
    print(" rnd  grabbed at  aimed for  result")
    for r in rows:
        said = r.get("result", "no result")
        if r.get("error"):
            said = r["error"] + (f"   (the run file says {said})" if str(said).startswith("GRABBED") else "")
        print("  %2d   %7s mm  %6.0f mm  %s" % (r["round"], r.get("out_mm", "?"), r["want_r_mm"], said))
    if rows:
        ok = [r for r in rows if _good(r)]
        held = [r for r in rows if r.get("held") and not r.get("error")]    # the squeeze said it had it
        reach = [r["out_mm"] for r in held if "out_mm" in r]
        print(f"\n  picked up {len(held)} of {len(rows)}"
              + (f", from {min(reach):.0f} to {max(reach):.0f} mm out" if reach else "")
              + f"; set down on the new spot and confirmed {len(ok)} of {len(rows)}")
    timed = [r for r in rows if r.get("times_s")]          # where the time goes, phase by phase (Laps)
    if timed:
        order, by = [], {}
        for r in timed:
            for name, secs in r["times_s"]:
                if name not in by:
                    order.append(name)
                by.setdefault(name, []).append(secs)
        total = float(np.mean([r.get("total_s", 0.0) for r in timed]))
        print(f"\n  where a round's {total:.0f} s goes (mean over {len(timed)} rounds; min-max):")
        for name in sorted(order, key=lambda n: -np.mean(by[n])):
            v = by[name]
            print(f"    {name:12s} {np.mean(v):6.1f} s  {100 * np.mean(v) / max(total, 1e-9):4.0f}%   ({min(v):.0f}-{max(v):.0f})")
    if path:
        print(f"  saved {path}")
