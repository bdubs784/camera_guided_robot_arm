#!/usr/bin/env python3
"""
Pack what the Pi's exact table check needs into one file, table_model.json: the touch-checked hand (fit_hand.py),
the CAD arm links placed in the world model (register_cad.py), and the plywood + base as their height above the table
at each distance from the base axis (spun round it: the tallest thing at each distance, like the old map).

    python export_table_model.py cad_in_world.json hand_model.json [--out table_model.json]
    then:  scp table_model.json pi@raspberrypi:~/arm_data/
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CAD = os.path.join(HERE, "..", "hardware", "cad", "rigid_parts")
CAD_TABLE_Z, BIN = -19.0, 5.0                             # the CAD's table top (mm), radius bins of the spun profile
PLY_SPACING = 3.0                                         # the plywood/base surfaces are sampled this finely (mm)
ANGLE_BIN = 15.0                                          # ...and binned by bearing this coarsely (deg)
HAND_POINTS, LINK_POINTS = 700, 250
PLANNER_BIAS_MM = 3.0                                     # at the touches the real hand sat 2.7 mm lower than commanded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cad")
    ap.add_argument("hand")
    ap.add_argument("--out", default="table_model.json")
    a = ap.parse_args()
    from build_arm_map import ArmModel, load_stl, sample
    cad, hm = json.load(open(a.cad)), json.load(open(a.hand))
    rng = np.random.default_rng(0)

    def thin(p, n):
        p = np.asarray(p, float)
        return p if len(p) <= n else p[rng.choice(len(p), n, replace=False)]
    A = ArmModel(CAD, spacing=20.0, hand_rolls=(0,))
    # Spread points over the plywood and base SURFACES, not their mesh corners. The whole assembly is 193 unique
    # vertices across 41 radius bins, so binning the corners left most bins empty and gave the few that held a
    # corner its full height: a profile of spikes and holes. It once claimed a 59 mm ring at 196 mm out (one
    # stray vertex) and nothing at all at 180 and 200, and the grab refused poses against the phantom.
    tris = load_stl(os.path.join(CAD, "meshes", "plywood_and_base_1.stl"))
    P = sample(tris, PLY_SPACING, 0)
    ax = A.anch0["base"][:2]
    r = np.linalg.norm(P[:, :2] - ax, axis=1)
    th = np.degrees(np.arctan2(P[:, 1] - ax[1], P[:, 0] - ax[0])) % 360.0
    nr, nt = int(np.ceil(r.max() / BIN)) + 1, int(round(360.0 / ANGLE_BIN))
    # Height by distance AND bearing, not by distance alone. Spinning the tallest thing at each radius all the way
    # round turns the stepper (69 mm, bearing +30..+90) and the base frame into walls in directions where there is
    # nothing: three 30 deg sectors have no tall structure at all and the grab was being refused in them.
    H2 = np.zeros((nr, nt))
    np.maximum.at(H2, ((r / BIN).astype(int), (th / ANGLE_BIN).astype(int) % nt), P[:, 2] - CAD_TABLE_Z)
    H2 = np.maximum(H2, 0.0)
    H = H2.max(axis=1)                                       # the old spun profile, kept for older Pi code
    print(f"   plywood/base sampled at {PLY_SPACING:g} mm: {len(P)} points over {len(tris)} triangles; "
          f"{int((H2 > 0.5).sum())} of {H2.size} (radius x bearing) cells hold something, "
          f"{int((H2.max(axis=0) < 0.5).sum())} of {nt} bearings are clear")
    out = {"made": time.strftime("%Y-%m-%d %H:%M"), "planner_bias_mm": PLANNER_BIAS_MM,
           "hand": {"ref_grip": hm["ref_grip"], "fit": hm.get("fit"),
                    "parts": {k: np.round(thin(v, HAND_POINTS if k == "G" else 10 ** 6), 1).tolist()
                              for k, v in hm["parts"].items()}},
           "links": {p["frame"]: np.round(thin(p["points_mm"], LINK_POINTS), 1).tolist()
                     for b, p in cad["parts"].items() if b != "hand"},
           "plywood": {"bin_mm": BIN, "height_mm": np.round(H, 1).tolist(),
                       "angle_bin_deg": ANGLE_BIN, "height_by_angle_mm": np.round(H2, 1).tolist(),
                       "doc": "height above the table of the plywood + base. height_mm: the tallest at distance "
                              "bin*k..bin*(k+1) from the base axis, whatever the bearing (the old spun profile). "
                              "height_by_angle_mm[k][j]: the same but only for bearings "
                              "angle_bin*j..angle_bin*(j+1), measured in the table frame, which is what the base "
                              "and plywood are fixed in"},
           "doc": "hand parts in G (L/R slide with the finger model from ref_grip), links in their world frames (U, F, H)"}
    json.dump(out, open(a.out, "w"))
    n = sum(len(v) for v in out["hand"]["parts"].values()) + sum(len(v) for v in out["links"].values())
    print(f"Saved {a.out}: {n} points, plywood/base up to {H.max():.0f} mm within {len(H) * BIN:.0f} mm of the axis")


if __name__ == "__main__":
    main()
