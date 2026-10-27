#!/usr/bin/env python3
"""
Build the arm's safety map from the Fusion export.

    pip install mujoco numpy scipy
    python build_arm_map.py  path/to/rigid_parts            (the folder with the .xml and meshes/)

It reads the exported model (shapes, link lengths, joints), and for every combination of
shoulder / elbow / wrist slider values on a grid it works out how close the arm comes to
  * the table, the plywood and the fixed base
  * the tower
  * itself
The result is one file, arm_map.npz. Copy that to the Pi (~/arm_data/arm_map.npz); arm.py
uses it to refuse any move that would bring the arm too close to something.

Re-run this whenever the drawing or the numbers below change.
"""
import argparse, glob, json, multiprocessing, os, re, struct, sys, time
import numpy as np
import mujoco
from scipy.spatial import cKDTree

# ---------------- measured on the real arm ----------------
# The arm is level and straight at these slider values...
# (Shoulder was 123 until the servo was replaced. Three table-touch points with the new servo
#  came out 7, 4.5 and 4.6 lower than before, so the new servo reads about 5.5 lower for the same real angle.)
REF_SLIDER = {"shoulder": 117.5, "elbow": 54.0, "wrist": 68.0}
# ...where the tape measure says the links really sit at these angles (degrees above level):
REF_ANGLE = {"upper_arm": -3.3, "forearm": 0.0, "hand": 0.0}
# Real degrees turned per slider degree. + means "slider up moves the far end of that link up".
# Shoulder: 2:1 planetary gearbox, so exactly 0.50 (the tape measure gave 0.52)
# Elbow:    -30 on the slider lifted the forearm 30.7 degrees  -> -1.02
# Wrist:    +30 on the slider lifted the hand about 27.5 deg   -> 0.92
# Real degrees per slider step. Each joint has two values: (slider below the level-pose value, slider above it).
# Reaching down and out was fitted to four table-touch points; the raised side is from a tape measure.
DEG_PER_SLIDER = {"shoulder": (0.455, 0.526), "elbow": (-1.027, -0.926), "wrist": (0.960, 0.960)}
# The drawing has the shoulder pivot 263 mm above the plywood; the tape says 244.5.
SHOULDER_DROP_MM = 18.5
PLYWOOD_MM = 19.0            # 3/4 inch
# Slider ranges to map (wrist stops at 148: above that the camera meets the elbow spring mount).
LIMITS = {"shoulder": (0, 160), "elbow": (0, 160), "wrist": (0, 148)}
# Tape-measure checks: (shoulder, elbow, wrist) -> measured height above the plywood, mm
# (taken with the old shoulder servo at 123 and 150; written here as the new servo's values, 5.5 lower)
CHECKS = [((117.5, 54, 68), "elbow hinge", 238), ((117.5, 54, 68), "wrist hinge", 238),
          ((144.5, 54, 68), "elbow hinge", 265), ((144.5, 54, 68), "wrist hinge", 289),
          ((117.5, 24, 68), "wrist hinge", 289)]

JOINT_NAMES = {"base": "Revolute-4", "shoulder": "Revolute-7", "elbow": "Revolute-12", "wrist": "Revolute-14", "roll": "Revolute-13"}
BODIES = ("tower", "upper_arm", "forearm", "wrist", "hand", "plywood_and_base", "floor")


def load_stl(path):
    b = open(path, "rb").read(); n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b, offset=84, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
    return a["v"].astype(float)                                   # (triangles, 3 corners, xyz) in mm


def sample(tris, spacing, seed):
    """Points spread over a surface, roughly `spacing` mm apart, plus every corner."""
    rng = np.random.default_rng(seed)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    idx = np.repeat(np.arange(len(tris)), np.ceil(area / spacing ** 2).astype(int))
    u, v = rng.random(len(idx)), rng.random(len(idx)); flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return np.vstack([a[idx] + u[:, None] * (b - a)[idx] + v[:, None] * (c - a)[idx], tris.reshape(-1, 3)])


class ArmModel:
    def __init__(self, folder, spacing=8.0, hand_rolls=(-20, 0, 20)):
        xml = glob.glob(os.path.join(folder, "*.xml"))
        if not xml:
            sys.exit(f"No .xml file in {folder}. Point this at the exported folder (the one with meshes/ in it).")
        src = open(xml[0]).read()
        src = re.sub(r"<asset>.*?</asset>", "", src, flags=re.S); src = re.sub(r"\s*<geom [^>]*/>", "", src)
        self.m = mujoco.MjModel.from_xml_string(src); self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        self.J = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, v) for k, v in JOINT_NAMES.items()}
        self.B = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n + "_1") for n in BODIES}
        missing = [k for k, v in {**self.J, **self.B}.items() if v < 0]
        if missing:
            sys.exit(f"The export is missing these parts or joints: {missing}. Their names must match the earlier export.")
        self.QA = {k: m.jnt_qposadr[v] for k, v in self.J.items()}
        mujoco.mj_forward(m, d)
        anch = {k: d.xanchor[j] * 1000 for k, j in self.J.items()}; axs = {k: d.xaxis[j].copy() for k, j in self.J.items()}
        self.anch0 = anch
        F0 = self.fk(np.zeros(m.nq))
        P = {n: sample(load_stl(os.path.join(folder, "meshes", n + "_1.stl")), spacing, i) for i, n in enumerate(BODIES)}
        self.P = P
        self.table = self.world(F0, "floor", P["floor"])[:, 2].max()
        pb = self.world(F0, "plywood_and_base", P["plywood_and_base"])
        self.plywood = PLYWOOD_MM
        fwd = np.cross([0, 0, 1], axs["shoulder"]); fwd /= np.linalg.norm(fwd)
        self.fwd = fwd if fwd @ (anch["elbow"] - anch["shoulder"]) > 0 else -fwd
        elev = lambda v: np.degrees(np.arctan2(v[2], v @ self.fwd))
        # which way each model joint turns
        def lifts(joint, height_fn):
            q = np.zeros(m.nq); d.qpos[:] = q; mujoco.mj_forward(m, d); z0 = height_fn()
            q[self.QA[joint]] = 0.05; d.qpos[:] = q; mujoco.mj_forward(m, d); z1 = height_fn()
            return 1.0 if z1 > z0 else -1.0
        hw = self.world(F0, "hand", P["hand"]); far_local = P["hand"][np.linalg.norm(hw - anch["wrist"], axis=1).argmax()]
        def tip_z(): return (d.xpos[self.B["hand"]] * 1000 + far_local @ d.xmat[self.B["hand"]].reshape(3, 3).T)[2]
        self.sgn = {"shoulder": lifts("shoulder", lambda: d.xanchor[self.J["elbow"]][2]),
                    "elbow": lifts("elbow", lambda: d.xanchor[self.J["wrist"]][2]),
                    "wrist": lifts("wrist", tip_z)}
        d.qpos[:] = 0; mujoco.mj_forward(m, d)
        roll_axis = axs["roll"] if axs["roll"] @ self.fwd > 0 else -axs["roll"]
        e_up, e_fo, e_ha = elev(anch["elbow"] - anch["shoulder"]), elev(anch["wrist"] - anch["elbow"]), elev(roll_axis)
        self.drawn = {"shoulder": e_up, "elbow": e_fo - e_up, "wrist": e_ha - e_fo}        # real angles of the pose as drawn
        self.lengths = [float(np.linalg.norm(anch["elbow"] - anch["shoulder"])), float(np.linalg.norm(anch["wrist"] - anch["elbow"])),
                        float(np.linalg.norm(hw - anch["wrist"], axis=1).max())]
        # ---- obstacles and arm points ----
        self.tower_tree = cKDTree(self.world(F0, "tower", P["tower"]))
        self.upper_tree = cKDTree(P["upper_arm"])
        wp_fore = self.local(F0, "forearm", anch["wrist"][None])[0]
        self.fore_tree = cKDTree(P["forearm"][np.linalg.norm(P["forearm"] - wp_fore, axis=1) > 45])
        rolls = []
        for r in hand_rolls:                       # the hand at a few wrist-rotate angles around the drawn one
            q = np.zeros(m.nq); q[self.QA["roll"]] = np.radians(r); Fr = self.fk(q)
            rolls.append(self.local(Fr, "wrist", self.world(Fr, "hand", P["hand"])))
        self.hand = np.vstack(rolls)
        wp_wrist = self.local(F0, "wrist", anch["wrist"][None])[0]
        self.hand_far = self.hand[np.linalg.norm(self.hand - wp_wrist, axis=1) > 70]
        # plywood + fixed base, spun round the base axis: the tallest thing at each distance from the axis
        self.cxy = anch["base"][:2]
        pr = np.linalg.norm(pb[:, :2] - self.cxy, axis=1); ph = pb[:, 2] - self.table
        self.BIN = 5.0; nb = int(np.ceil(pr.max() / self.BIN)) + 1
        self.H = np.zeros(nb); np.maximum.at(self.H, (pr / self.BIN).astype(int), ph)
        self.r_lo = np.arange(nb) * self.BIN; self.r_hi = self.r_lo + self.BIN

    def fk(self, q):
        self.d.qpos[:] = q; mujoco.mj_kinematics(self.m, self.d)
        return {n: (self.d.xpos[b] * 1000.0, self.d.xmat[b].reshape(3, 3).copy()) for n, b in self.B.items()}

    @staticmethod
    def world(F, n, pts): p, R = F[n]; return p + pts @ R.T
    @staticmethod
    def local(F, n, pts): p, R = F[n]; return (pts - p) @ R

    def real_angles(self, s, e, w):
        """Real joint angles (deg) for slider values: upper arm above level, elbow bend, wrist bend."""
        def turn(joint, value):                      # degrees turned away from the level pose
            step = value - REF_SLIDER[joint]
            return DEG_PER_SLIDER[joint][0 if step < 0 else 1] * step
        up = REF_ANGLE["upper_arm"] + turn("shoulder", s)
        el = (REF_ANGLE["forearm"] - REF_ANGLE["upper_arm"]) + turn("elbow", e)
        wr = (REF_ANGLE["hand"] - REF_ANGLE["forearm"]) + turn("wrist", w)
        return {"shoulder": up, "elbow": el, "wrist": wr}

    def q(self, s, e, w):
        real = self.real_angles(s, e, w); q = np.zeros(self.m.nq)
        for k in ("shoulder", "elbow", "wrist"):
            q[self.QA[k]] = self.sgn[k] * np.radians(real[k] - self.drawn[k])
        return q

    def hinge_heights(self, s, e, w):
        """Heights above the plywood of the elbow hinge and wrist pitch hinge, with the shoulder correction."""
        self.d.qpos[:] = self.q(s, e, w); mujoco.mj_forward(self.m, self.d)
        z = lambda j: self.d.xanchor[self.J[j]][2] * 1000 - self.table - self.plywood - SHOULDER_DROP_MM
        return {"shoulder hinge": z("shoulder"), "elbow hinge": z("elbow"), "wrist hinge": z("wrist")}

    def clearances(self, s, e, w):
        """(lowest point above the table, gap to table/plywood/base, gap to tower, gap to itself), all mm."""
        F = self.fk(self.q(s, e, w))
        up = self.world(F, "upper_arm", self.P["upper_arm"]); fo = self.world(F, "forearm", self.P["forearm"])
        wr = self.world(F, "wrist", self.P["wrist"]); ha = self.world(F, "wrist", self.hand)
        arm = np.vstack([up, fo, wr, ha])
        z = arm[:, 2] - self.table - SHOULDER_DROP_MM            # the real arm sits lower than the drawing
        r = np.linalg.norm(arm[:, :2] - self.cxy, axis=1)
        low = float(z.min())
        near = z < self.H.max() + 150
        base = low
        if near.any():
            rn, zn = r[near][:, None], z[near][:, None]
            dr = np.maximum(np.maximum(self.r_lo[None] - rn, rn - self.r_hi[None]), 0); dz = np.maximum(zn - self.H[None], 0)
            base = min(float(np.hypot(dr, dz).min()), low)
        tower = float(self.tower_tree.query(np.vstack([fo, wr, ha]))[0].min())
        s1 = float(self.upper_tree.query(self.local(F, "upper_arm", np.vstack([wr, ha])))[0].min())
        s2 = float(self.fore_tree.query(self.local(F, "forearm", self.world(F, "wrist", self.hand_far)))[0].min())
        return low, base, tower, min(s1, s2)


_ARM = None
def _start_worker(folder):
    global _ARM
    _ARM = ArmModel(folder)


def _one_shoulder_row(job):
    i, s, E, W = job
    out = np.zeros((len(E), len(W), 4), np.float32)
    for j, e in enumerate(E):
        for k, w in enumerate(W):
            out[j, k] = _ARM.clearances(s, e, w)
    return i, out


def main():
    ap = argparse.ArgumentParser(description="Build the arm's safety map from the Fusion export.")
    ap.add_argument("folder", help="exported folder (contains the .xml and meshes/)")
    ap.add_argument("--out", default="arm_map.npz")
    ap.add_argument("--step", type=int, default=4, help="grid step in slider degrees (smaller = finer and slower)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="CPU cores to use")
    ap.add_argument("--pose", action="append", default=[], help="just report one pose, e.g. --pose 123,54,68 (repeatable)")
    args = ap.parse_args()
    A = ArmModel(args.folder)
    print(f"Model read. Links: upper arm {A.lengths[0]:.0f} mm, forearm {A.lengths[1]:.0f} mm, wrist hinge to far tip {A.lengths[2]:.0f} mm.")
    print(f"Plywood is {A.plywood:.0f} mm thick. Shoulder in the drawing: {A.anch0['shoulder'][2] - A.table - A.plywood:.0f} mm above the plywood "
          f"(corrected by -{SHOULDER_DROP_MM} mm to match the tape).")
    print("\nCheck against the tape measure (heights above the plywood):")
    worst = 0
    for pose, what, meas in CHECKS:
        pred = A.hinge_heights(*pose)[what]; worst = max(worst, abs(pred - meas))
        print(f"   pose {pose}: {what:12s} model {pred:5.0f} mm   tape {meas} mm   difference {pred - meas:+.0f}")
    print(f"   biggest difference: {worst:.0f} mm" + ("  (good)" if worst <= 8 else "  (more than expected - check the CAD placement)"))
    if args.pose:
        print()
        for t in args.pose:
            s, e, w = map(float, t.split(","))
            low, base, tow, sf = A.clearances(s, e, w); ra = A.real_angles(s, e, w)
            print(f"pose {t}: upper arm {ra['shoulder']:+.0f} deg, lowest point {low:.0f} mm above the table | "
                  f"gap to table/base {base:.0f} | tower {tow:.0f} | itself {sf:.0f}")
        return
    ax = {k: np.arange(lo, hi + 1, args.step) for k, (lo, hi) in LIMITS.items()}
    S, E, W = ax["shoulder"], ax["elbow"], ax["wrist"]
    G = np.zeros((len(S), len(E), len(W), 4), np.float32)
    print(f"\nBuilding the map: {G.shape[0] * G.shape[1] * G.shape[2]} poses on {args.workers} CPU cores...")
    t0 = time.time(); n_done = 0
    jobs = [(i, float(s), E, W) for i, s in enumerate(S)]
    with multiprocessing.Pool(args.workers, initializer=_start_worker, initargs=(args.folder,)) as pool:
        for i, out in pool.imap_unordered(_one_shoulder_row, jobs):
            G[i] = out; n_done += 1
            done = n_done / len(S); el = time.time() - t0
            print(f"\r   {done * 100:3.0f}%   about {el / done * (1 - done) / 60:.1f} min left   ", end="", flush=True)
    print(f"\rDone in {(time.time() - t0) / 60:.1f} min.                         ")
    gap = np.minimum(np.minimum(G[..., 1], G[..., 2]), G[..., 3])
    for margin in (20, 30, 40, 50):
        ok = gap >= margin
        print(f"   with a {margin} mm safety margin: {ok.mean() * 100:4.1f}% of poses allowed "
              f"(blocked by table/base {(G[..., 1] < margin).mean() * 100:.0f}%, tower {(G[..., 2] < margin).mean() * 100:.0f}%, itself {(G[..., 3] < margin).mean() * 100:.0f}%)")
    ok = gap >= 40
    if ok.any():
        print("   at 40 mm the arm can use: shoulder %d-%d, elbow %d-%d, wrist %d-%d (not every combination)" % (
            S[ok.any((1, 2))].min(), S[ok.any((1, 2))].max(), E[ok.any((0, 2))].min(), E[ok.any((0, 2))].max(), W[ok.any((0, 1))].min(), W[ok.any((0, 1))].max()))
    np.savez_compressed(args.out, s=S, e=E, w=W,
                        low=np.clip(G[..., 0], -500, 2000).astype(np.int16), base=np.clip(G[..., 1], 0, 255).astype(np.uint8),
                        tower=np.clip(G[..., 2], 0, 255).astype(np.uint8), self=np.clip(G[..., 3], 0, 255).astype(np.uint8),
                        meta=json.dumps({"ref_slider": REF_SLIDER, "ref_angle": REF_ANGLE, "deg_per_slider": DEG_PER_SLIDER,
                                         "shoulder_drop_mm": SHOULDER_DROP_MM, "limits": LIMITS, "links_mm": A.lengths,
                                         "made": time.strftime("%Y-%m-%d %H:%M")}))
    print(f"\nSaved {args.out}  ({os.path.getsize(args.out) / 1024:.0f} kB). Copy it to the Pi as ~/arm_data/arm_map.npz")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
