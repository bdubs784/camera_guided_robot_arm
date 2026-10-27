#!/usr/bin/env python3
"""
The big solve: one consistent 3D model of the arm, its three cameras and every tag, from a survey alone.

    python tools/big_solve.py <survey folder> --model self_model_pi.json        # fit, report, save world.json
    python tools/big_solve.py <survey folder> --model self_model_pi.json --holdout 0.2   # also test on unseen poses
    python tools/big_solve.py --synthetic                                         # self-test on made-up data

What is solved (bundle adjustment: every unknown at once, by matching every tag corner every camera saw):
  * the arm: link lengths, each joint's zero and gearing (slider degrees -> real degrees)
  * the base: where its axis is, and how its angle relates to the stepper count
  * each camera's pose: tower camera and webcam ride on the tower, the gripper camera rides on the hand
  * each tag's pose on whatever it is stuck to: the table (world), the tower, the upper arm, the forearm or the hand.
    Which one is worked out automatically (the one that keeps the tag still).
  * a small correction of each pose's joint angles (gear play), kept small by a prior

Frames: T = the tower frame (turns with the base): origin at the shoulder hinge, x along the arm, z up the base
axis. W = the table: T at base 0. The arm moves in T's x-z plane. Lens numbers come from calibrate_cam.py.
Residuals are corner reprojection errors in pixels; a robust loss keeps a few bad detections from mattering.
"""
import argparse
import json
import os
import sys
import time

import threading

import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:                                     # capping BLAS threads is worth a lot here (see blas_threads)
    from threadpoolctl import threadpool_limits as _blas
except ImportError:                      # without it, numpy keeps its default and the solve is just slower
    import contextlib

    def _blas(*a, **k):
        return contextlib.nullcontext()


FILE_PROGRESS_S = 30.0               # how often the progress line is written when the output is a file


class Progress:
    """One live line at the bottom: overall bar, which stage, step, current corner error, time. Other prints go above it."""

    def __init__(self, stages=5, width=28):
        self.stages, self.width, self.stage, self.label = stages, width, 0, ""
        self.t0, self.last, self.shown = time.time(), 0.0, False
        self.step, self.err, self.calls_per_step, self.calls = 0, float("nan"), 1, 0
        self.on = sys.stdout.isatty()
        self.last_file = 0.0                 # when the progress line last went into a redirected log
        self.finished = False

    def start_stage(self, label, calls_per_step):
        self.stage += 1
        self.stages = max(self.stages, self.stage)
        self.label, self.calls_per_step, self.calls, self.step = label, max(1, calls_per_step), 0, 0
        self.draw(force=True)

    def tick(self, err):
        self.calls += 1
        self.step = self.calls // self.calls_per_step
        self.err = err
        self.draw()

    def draw(self, force=False):
        if self.finished:
            return
        now = time.time()
        if not force and now - self.last < 0.3:
            return
        self.last = now
        inside = min(self.step / 40.0, 0.95)                    # most stages settle within ~40 steps
        frac = min((self.stage - 1 + inside) / self.stages, 1.0)
        n = int(frac * self.width)
        m, sec = divmod(int(now - self.t0), 60)
        line = (f"[{'#' * n}{'.' * (self.width - n)}] {frac * 100:3.0f}%  stage {self.stage}/{self.stages} {self.label}"
                f"  step {self.step}  error {self.err:5.1f} px  {m}:{sec:02d}")
        if self.on:
            sys.stdout.write("\r" + line[:118].ljust(118))
            sys.stdout.flush()
            self.shown = True
        elif force or now - self.last_file >= FILE_PROGRESS_S:
            # Redirected to a file this printed only at a stage boundary, so a run watched through its log showed
            # "step 0, nan px" frozen for half an hour with no way to tell working from hung. A terminal redraws
            # one line in place; a file cannot, so it gets a fresh line every so often instead.
            self.last_file = now
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def clear(self):
        if self.shown:
            sys.stdout.write("\r" + " " * 118 + "\r")
            sys.stdout.flush()
            self.shown = False

    def done(self):
        self.stage, self.step = self.stages, 10 ** 6
        self.draw(force=True)
        if self.on:
            sys.stdout.write("\n")
        self.shown, self.finished = False, True


PROGRESS = None

FRAMES = ("W", "U", "F", "H", "T", "G")   # table, upper arm, forearm, hand (wrist block), tower, gripper (turns with rotate)
T_IDX, H_IDX, G_IDX = 4, 3, 5
CAM_PARENT = {"torre": "T", "webcam": "T", "gripper": "G", "gripper2": "G"}   # gripper2: same camera after --moved
FOCAL_FREE = ("gripper", "gripper2")            # its focus is moved by hand (/focus), which shifts the focal length a little: solved
FINGER_TAGS = {31, 32}               # on the fingers: they move when the gripper opens or closes...
GRIP_SURVEYED, GRIP_TOL = 80.0, 1.5  # ...the finger model measures them from where they are at this gripper value
AFTER = 100000                       # a tag that moved gets a second id, AFTER + id, for the sightings after the move
FINGER_OF = {31: 0, 32: 1, AFTER + 31: 0, AFTER + 32: 1}   # which finger each finger tag rides on (0 = left, 1 = right)
FINGER_MODEL = True                  # --no-fingers: the old way (finger tags used only at GRIP_SURVEYED)
MOVING_TAGS = {7}                    # the cube moves between surveys/rounds: not a landmark unless --static
TAG_DEFAULT_MM = 30.0
JOINT_PRIOR_DEG = 0.7                # gear play: per-pose joint corrections are kept about this small
BASE_PRIOR_DEG = 0.3                 # the base's own per-pose correction (worm gear slack, missed steps), kept this small
ND = 6                               # per-pose corrections: shoulder, elbow, wrist, wrist rotate, base, gripper
DQ_ROT, DQ_BASE, DQ_GRIP = 3, 4, 5
ROT_PRIOR_STEPS = 1.5                # the rotate's slack (a sweep: ~2 steps between going up and coming down)
GRIP_PRIOR_STEPS = 1.5               # the gripper's slack, in slider steps (only seen through the finger tags)
PRIORS = np.array([JOINT_PRIOR_DEG] * 3 + [ROT_PRIOR_STEPS, BASE_PRIOR_DEG, GRIP_PRIOR_STEPS])
# The finger model (survey v2 moves the gripper in most poses). The gripper is a parallel linkage: each finger slides
# along an arc without turning (finger tags move 300+ px in the gripper camera as it opens, their tilt
# changes ~1 deg). So a finger tag sits where it was at GRIP_SURVEYED plus an arc offset in the gripper's frame G:
#     angle = k d + k2 d^2  (degrees, d = gripper slider - GRIP_SURVEYED),
#     offset = r [(cos(phi0 + angle) - cos phi0) u + (sin(phi0 + angle) - sin phi0) v]
# with u, v spanning the plane the fingers swing in (normal n = (ax, ay, 1), shared by both fingers).
FING_NAMES = ("ax", "ay", "r_L", "phi0_L", "k_L", "k2_L", "r_R", "phi0_R", "k_R", "k2_R")
N_FING = len(FING_NAMES)
ARM_NAMES = ("th1", "k1", "th2", "k2", "th3", "k3", "L1", "L2",
             "th4", "k4", "ax_u", "ax_v", "ax_y", "ax_z", "k4b")
# wrist rotate: zero, deg per step, axis tilt (2), axis offset (2, mm), bend (deg per step^2: the servo turns less per
# step at one end of its range: about 10 steps short at 112)
N_ARM = len(ARM_NAMES)
ROT_DEFAULT = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
AXIS_PRIOR = np.array([1.0, 1.0, 30.0, 30.0])  # rotate axis tilt and offset: only a very loose hold (a tight one hid the real tilt)
ROLL_REF = 42.0                      # the wrist rotate the early surveys were all taken at


# ---------------------------------------------------------------- small geometry helpers
def rotvec_to_mat(r):
    """Rodrigues, vectorised: (..., 3) -> (..., 3, 3)."""
    r = np.asarray(r, float)
    th = np.linalg.norm(r, axis=-1)[..., None, None]
    k = r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
    K = np.zeros(r.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    I = np.broadcast_to(np.eye(3), K.shape)
    return I + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def mat_to_rotvec(R):
    import cv2
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def rot_xz(a):
    """Rotation in the arm's plane by angle a (rad): x -> (cos a, 0, sin a). Vectorised over a."""
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 2], R[..., 1, 1], R[..., 2, 0], R[..., 2, 2] = c, -s, 1.0, s, c
    return R


def rot_z(a):
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1.0
    return R


def tag_corners(size):
    """A tag's corners in its own frame, in OpenCV's order (top-left, top-right, bottom-right, bottom-left)."""
    h = size / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def project(p, K, dist):
    """Camera-frame points (..., 3) -> pixels (..., 2), OpenCV's 5-number lens model."""
    z = np.maximum(p[..., 2], 1e-6)
    x, y = p[..., 0] / z, p[..., 1] / z
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=-1)


def kabsch(a, b):
    """R, t with b ~= R a + t (least squares, rows are points)."""
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def mean_rotation(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def finger_plane(ax, ay):
    """Unit normal n of the plane the fingers swing in, and u, v spanning it (u as close to G's x as possible).
    Vectorised: ax, ay (N,) -> n, u, v (N, 3)."""
    ax, ay = np.atleast_1d(np.asarray(ax, float)), np.atleast_1d(np.asarray(ay, float))
    n = np.stack([ax, ay, np.ones_like(ax)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    u = np.array([1.0, 0.0, 0.0]) - n[:, :1] * n
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    return n, u, np.cross(n, u)


def finger_offset(fp, d):
    """Where a finger tag is (mm, in G) relative to where it is at GRIP_SURVEYED. Vectorised over N sightings:
    fp (N, 6) = ax, ay, r, phi0 (deg), k (deg/step), k2 (deg/step^2) of that tag's finger; d (N,) = grip - GRIP_SURVEYED."""
    fp = np.atleast_2d(fp)
    _, u, v = finger_plane(fp[:, 0], fp[:, 1])
    r, phi0 = fp[:, 2:3], np.radians(fp[:, 3:4])
    a = np.radians(fp[:, 4] * d + fp[:, 5] * d * d)[:, None]
    return r * ((np.cos(phi0 + a) - np.cos(phi0)) * u + (np.sin(phi0 + a) - np.sin(phi0)) * v)


# ---------------------------------------------------------------- the model
class World:
    """Everything that is solved, packed into / out of one parameter vector."""

    def __init__(self, cams, tags, n_poses, ref, calib):
        self.cams = list(cams)                      # camera names, in parameter order
        self.tags = list(tags)                      # tag ids, in parameter order
        self.tag_frame = {}                         # tag id -> index into FRAMES
        self.tag_size = {}
        self.ref = np.asarray(ref, float)           # slider values the joint zeros are measured from
        self.calib = calib                          # name -> (K, dist)
        self.n = n_poses
        self.arm = np.concatenate([np.zeros(8), ROT_DEFAULT])
        self.rot_varies = False                     # does any pose have the wrist rotate away from the others?
        self.rot_ambiguous = False                  # the quick test could not tell which way rotate turns
        self.base = np.array([1.0, 0.0, 0.0])       # degrees of real turn per degree of base count; axis x, y in T
        self.cam = {c: np.zeros(6) for c in self.cams}
        self.fs = {c: 1.0 for c in self.cams if c in FOCAL_FREE}   # focal length scale vs the lens calibration
        self.tag = {t: np.zeros(6) for t in self.tags}
        self.dq = np.zeros((n_poses, ND))           # per-pose corrections (deg): shoulder, elbow, wrist, base
        self.sv = np.zeros(n_poses, int)            # which survey each pose came from (basehome is re-run between them)
        self.boff = np.zeros(1)                     # base count offset per survey (deg); the first is the reference
        self.fing = np.zeros(N_FING)                # the finger model (FING_NAMES); all zero = fingers never move
        self.grip_varies = False                    # does any pose have the gripper away from GRIP_SURVEYED?
        self.fing_on = False                        # is the finger model being solved (set once it has a first guess)?

    def finger_params(self, t):
        """(6,) ax, ay, r, phi0, k, k2 for a finger tag's finger, or None for any other tag."""
        f = FINGER_OF.get(t)
        if f is None:
            return None
        return np.concatenate([self.fing[:2], self.fing[2 + 4 * f: 6 + 4 * f]])

    # ----- parameter vector
    def layout(self):
        sizes = [("arm", N_ARM), ("base", 3), ("boff", len(self.boff)), ("fing", N_FING)] \
                + [(("cam", c), 6) for c in self.cams] + [(("fs", c), 1) for c in self.fs] \
                + [(("tag", t), 6) for t in self.tags] + [("dq", ND * self.n)]
        out, i = {}, 0
        for k, n in sizes:
            out[k] = slice(i, i + n)
            i += n
        return out, i

    def pack(self):
        lay, n = self.layout()
        x = np.zeros(n)
        x[lay["arm"]], x[lay["base"]], x[lay["dq"]] = self.arm, self.base, self.dq.ravel()
        x[lay["boff"]] = self.boff
        x[lay["fing"]] = self.fing
        for c in self.cams:
            x[lay[("cam", c)]] = self.cam[c]
        for c in self.fs:
            x[lay[("fs", c)]] = self.fs[c]
        for t in self.tags:
            x[lay[("tag", t)]] = self.tag[t]
        return x

    def unpack(self, x):
        lay, _ = self.layout()
        self.arm, self.base = x[lay["arm"]].copy(), x[lay["base"]].copy()
        self.dq = x[lay["dq"]].reshape(-1, ND).copy()
        self.boff = x[lay["boff"]].copy()
        self.fing = x[lay["fing"]].copy()
        for c in self.cams:
            self.cam[c] = x[lay[("cam", c)]].copy()
        for c in self.fs:
            self.fs[c] = float(x[lay[("fs", c)]][0])
        for t in self.tags:
            self.tag[t] = x[lay[("tag", t)]].copy()

    # ----- forward model
    def angles(self, q, dq=None):
        """Link angles (rad) a1 (upper arm), a2 (forearm), a3 (hand) from the arm's plane x axis, per pose."""
        th1, k1, th2, k2, th3, k3 = self.arm[:6]
        q = np.atleast_2d(np.asarray(q, float))[:, :3] + (0.0 if dq is None else np.asarray(dq)[:, :3])
        d = q - self.ref[:3]
        a1 = np.radians(th1 + k1 * d[:, 0])
        a2 = a1 + np.radians(th2 + k2 * d[:, 1])
        a3 = a2 + np.radians(th3 + k3 * d[:, 2])
        return a1, a2, a3

    def rotate_angle(self, q, dq=None):
        """Wrist rotate angle (rad) away from the reference, per pose (0 for data without a rotate column)."""
        q = np.atleast_2d(np.asarray(q, float))
        r = q[:, 3] if q.shape[1] > 3 else np.full(len(q), self.ref[3] if len(self.ref) > 3 else ROLL_REF)
        if dq is not None and np.asarray(dq).shape[1] > DQ_ROT:
            r = r + np.asarray(dq)[:, DQ_ROT]
        th4, k4, k4b = self.arm[8], self.arm[9], self.arm[14]
        d = r - (self.ref[3] if len(self.ref) > 3 else ROLL_REF)
        return np.radians(th4 + k4 * d + k4b * d * d)

    def rotate_axis(self):
        u, v, ay, az = self.arm[10:14]
        n = np.array([1.0, u, v])
        return n / np.linalg.norm(n), np.array([0.0, ay, az])

    def frames(self, q, base, dq=None):
        """Rotations (N,6,3,3) and origins (N,6,3) of W, U, F, H, T itself and G in the tower frame T, for each pose."""
        a1, a2, a3 = self.angles(q, dq)
        L1, L2 = self.arm[6], self.arm[7]
        N = len(a1)
        R, o = np.zeros((N, 6, 3, 3)), np.zeros((N, 6, 3))
        R[:, 4] = np.eye(3)
        g, cx, cy = self.base
        base = np.asarray(base, float)
        if len(self.boff) > 1 and base.shape == self.sv.shape:
            base = base + self.boff[self.sv]                 # each survey's own base zero (re-homed in between)
        if dq is not None and np.asarray(dq).shape[1] > DQ_BASE:
            base = base + np.asarray(dq)[:, DQ_BASE]         # this pose's base slack
        Rw = rot_z(-np.radians(g * base))
        c = np.array([cx, cy, 0.0])
        R[:, 0], o[:, 0] = Rw, c - Rw @ c
        R[:, 1] = rot_xz(a1)
        elbow = L1 * np.stack([np.cos(a1), np.zeros(N), np.sin(a1)], 1)
        R[:, 2], o[:, 2] = rot_xz(a2), elbow
        R[:, 3], o[:, 3] = rot_xz(a3), elbow + L2 * np.stack([np.cos(a2), np.zeros(N), np.sin(a2)], 1)
        n, pa = self.rotate_axis()                           # the gripper turns about this axis, fixed in the hand
        Rr = rotvec_to_mat(self.rotate_angle(q, dq)[:, None] * n[None])
        R[:, 5] = R[:, 3] @ Rr
        o[:, 5] = o[:, 3] + np.einsum("nij,nj->ni", R[:, 3], pa[None] - np.einsum("nij,j->ni", Rr, pa))
        return R, o

    def corners_in_parent(self, t):
        R = rotvec_to_mat(self.tag[t][:3])
        return tag_corners(self.tag_size[t]) @ R.T + self.tag[t][3:]

    def grip_d(self, q, dq=None):
        """Gripper slider minus GRIP_SURVEYED, per pose (0 for data without a gripper column)."""
        q = np.atleast_2d(np.asarray(q, float))
        g = q[:, 4] if q.shape[1] > 4 else np.full(len(q), GRIP_SURVEYED)
        if dq is not None and np.asarray(dq).shape[1] > DQ_GRIP:
            g = g + np.asarray(dq)[:, DQ_GRIP]
        return g - GRIP_SURVEYED

    def predict(self, obs, q, base, dq=None):
        """Pixel corners (K,4,2) for observation arrays obs = dict(pose, cam, tag) (indices)."""
        Rf, of = self.frames(q, base, dq)
        gd = self.grip_d(q, dq)
        out = np.zeros((len(obs["pose"]), 4, 2))
        for ci, c in enumerate(self.cams):
            sel = obs["cam"] == ci
            if not sel.any():
                continue
            K, dist = self.calib[c]
            if c in self.fs:                                           # focal length fine-tuned (focus moved)
                K = K.copy()
                K[0, 0] *= self.fs[c]
                K[1, 1] *= self.fs[c]
            Rc = rotvec_to_mat(self.cam[c][:3])
            tc = self.cam[c][3:]
            for ti, t in enumerate(self.tags):
                s2 = sel & (obs["tag"] == ti)
                if not s2.any():
                    continue
                i = obs["pose"][s2]
                f = self.tag_frame[t]
                X = self.corners_in_parent(t)[None]                    # (1,4,3) in its parent frame
                fp = self.finger_params(t)
                if fp is not None:                                      # a finger: slid along its arc at this grip
                    X = X + finger_offset(np.tile(fp, (len(i), 1)), gd[i])[:, None, :]
                pT = np.einsum("nij,nkj->nki", Rf[i, f], np.broadcast_to(X, (len(i), 4, 3))) + of[i, f][:, None, :]
                pf = FRAMES.index(CAM_PARENT[c])
                if pf != T_IDX:                                         # into the camera's own part first
                    pT = np.einsum("nji,nkj->nki", Rf[i, pf], pT - of[i, pf][:, None, :])
                pc = pT @ Rc.T + tc
                out[s2] = project(pc, K, dist)
        return out

    def hand_point(self, q, base=0.0, p_hand=(0.0, 0.0, 0.0), dq=None):
        """A point fixed to the hand, in the tower frame, for each pose."""
        R, o = self.frames(q, np.zeros(len(np.atleast_2d(q))) + base, dq)
        return np.einsum("nij,j->ni", R[:, G_IDX], np.asarray(p_hand, float)) + o[:, G_IDX]

    # ----- save
    def to_json(self):
        return {"arm": dict(zip(ARM_NAMES, map(float, self.arm))), "ref_slider": self.ref.tolist(),
                "base": {"gain": float(self.base[0]), "axis_xy_mm": self.base[1:].tolist()},
                "cameras": {c: {"parent": CAM_PARENT[c], "rvec": self.cam[c][:3].tolist(), "t_mm": self.cam[c][3:].tolist(),
                                "focal_scale": float(self.fs.get(c, 1.0)),
                                "camera_matrix": np.asarray(self.calib[c][0], float).tolist(),
                                "dist_coeffs": np.asarray(self.calib[c][1], float).ravel().tolist()} for c in self.cams},
                "base_zero_deg": float(self.boff[-1]) if len(self.boff) else 0.0,   # the newest survey's base zero
                **({"fingers": {"ref_grip": GRIP_SURVEYED, "axis": self.fing[:2].tolist(),
                                "tags": {str(t): "LR"[f] for t, f in FINGER_OF.items() if t in self.tag_frame},
                                **{side: dict(zip(("r_mm", "phi0_deg", "k_deg_per_step", "k2_deg_per_step2"),
                                                  map(float, self.fing[2 + 4 * f: 6 + 4 * f])))
                                   for f, side in enumerate("LR")},
                                "doc": "finger tag position in G = its t_mm (at ref_grip) + r [(cos(phi0+a) - cos phi0) u + "
                                       "(sin(phi0+a) - sin phi0) v], a = k d + k2 d^2 deg, d = grip - ref_grip; "
                                       "n = normalised (axis[0], axis[1], 1), u = G's x made perpendicular to n, v = n x u"}}
                   if self.fing_on else {}),
                "tags": {str(t): {"on": FRAMES[self.tag_frame[t]], "size_mm": self.tag_size[t],
                                  "rvec": self.tag[t][:3].tolist(), "t_mm": self.tag[t][3:].tolist()} for t in self.tags},
                "frames_doc": "T: tower frame, origin shoulder hinge, x along the arm, z up the base axis. "
                              "W: table (T at base 0). U/F/H: upper arm / forearm / hand, x along the link."}


# ---------------------------------------------------------------- loading a survey
def load_survey(folder, static=()):
    meta = json.load(open(os.path.join(folder, "meta.json")))
    calib = {}
    for c, d in (meta.get("calibrations") or {}).items():
        if d:
            calib[c] = (np.array(d["camera_matrix"], float), np.array(d["dist_coeffs"], float))
    sizes = {int(k): float(v) for k, v in (meta.get("tag_mm") or {}).items()}
    poses = [json.loads(l) for l in open(os.path.join(folder, "poses.jsonl")) if l.strip()]
    q = np.array([pad_q(p["joints"][:5]) for p in poses], float).reshape(-1, 5)   # shoulder, elbow, wrist, rotate, gripper
    came = np.array([pad_q(p.get("came_from", p["joints"][:3])) for p in poses], float).reshape(-1, 5)
    base = np.array([p["base"] for p in poses], float)
    sv = np.array([int(p.get("sv", 0)) for p in poses], int)
    obs = []                                         # (pose, cam, tag, corners 4x2 median over frames)
    for i, p in enumerate(poses):
        grip_ok = len(p["joints"]) < 5 or abs(float(p["joints"][4]) - GRIP_SURVEYED) <= GRIP_TOL
        for c, frames in p["cams"].items():
            if c not in calib:
                continue
            by_tag = {}
            for fr in frames:
                for t in fr:
                    if len(t.get("corners") or []) == 4:
                        by_tag.setdefault(t["id"], []).append(t["corners"])
            for tid, cs in by_tag.items():
                if tid in MOVING_TAGS and tid not in static:
                    continue
                if tid in FINGER_TAGS and not grip_ok and not FINGER_MODEL:   # --no-fingers: only at the usual grip
                    continue
                obs.append((i, c, tid, np.median(np.array(cs, float), axis=0)))
    return {"meta": meta, "calib": calib, "sizes": sizes, "q": q, "came": came, "base": base, "obs": obs, "sv": sv}


def pad_q(v):
    """Joint values -> exactly 5: shoulder, elbow, wrist, rotate (default level), gripper (default GRIP_SURVEYED)."""
    v = [float(x) for x in v[:5]]
    return v + [ROLL_REF, GRIP_SURVEYED][len(v) - 3:] if len(v) < 5 else v


def finger_moved(S, o):
    """Is this sighting a finger tag in a pose where the gripper was away from GRIP_SURVEYED?"""
    return o[2] in FINGER_OF and abs(float(S["q"][o[0], 4]) - GRIP_SURVEYED) > GRIP_TOL


def load_surveys(folders, static=(), moved_from=None, moved_tags=()):
    """Several survey folders as one: poses appended, each folder's surveys get their own base zero (sv).
    moved_from: a folder name; from that folder on (in the order given) the gripper camera is treated as a separate
    camera "gripper2" with its own place on the gripper (once the fingers hit the table and shifted it)."""
    out = None
    for k, folder in enumerate(folders):
        S = load_survey(folder, static)
        if len(S["q"]) == 0:                         # a survey stopped before its first pose
            print(f"   skipping {folder}: no poses in it")
            continue
        if moved_from is not None and k >= moved_from:
            S["obs"] = [(i, "gripper2" if c == "gripper" else c, AFTER + t if t in moved_tags else t, px)
                        for (i, c, t, px) in S["obs"]]
            for t in moved_tags:
                if t in S["sizes"]:
                    S["sizes"][AFTER + t] = S["sizes"][t]
            if "gripper" in S["calib"]:
                S["calib"]["gripper2"] = S["calib"]["gripper"]
        if out is None:
            out = S
            continue
        n0, sv0 = len(out["q"]), int(out["sv"].max()) + 1
        for k in ("q", "came"):
            out[k] = np.vstack([out[k], S[k]])              # both padded to 5 columns by load_survey
        out["base"] = np.concatenate([out["base"], S["base"]])
        out["sv"] = np.concatenate([out["sv"], S["sv"] + sv0])
        out["obs"] += [(i + n0, c, t, px) for (i, c, t, px) in S["obs"]]
        out["sizes"].update(S["sizes"])
        for c, v in S["calib"].items():
            out["calib"].setdefault(c, v)
    return out


def obs_arrays(world, obs):
    ci = {c: k for k, c in enumerate(world.cams)}
    ti = {t: k for k, t in enumerate(world.tags)}
    keep = [o for o in obs if o[1] in ci and o[2] in ti]
    return {"pose": np.array([o[0] for o in keep], int), "cam": np.array([ci[o[1]] for o in keep], int),
            "tag": np.array([ti[o[2]] for o in keep], int), "px": np.array([o[3] for o in keep], float)}


# ---------------------------------------------------------------- initial guess
def from_hand_model(path):
    """Arm geometry, the hand tag and the tower camera from the trained self-model (self_model_pi.json)."""
    d = json.load(open(path))
    k1, k2, k3, e0, w0, L1, L2, tx, ty = d["x_tuned"][:9]
    up = float(d["upper_arm_level_deg"])
    arm = np.array([up, k1, -up + e0, k2, w0, k3, L1, L2])
    return arm, np.array(d["ref_slider"], float), np.array([tx, 0.0, ty]), np.array(d["x_tuned"][9:15], float)


def tag_pose_in_cam(corners, size, calib):
    """(R, t) of a tag in the camera frame from its corners, or None. Checks the answer reprojects (IPPE can pick the
    mirror solution for a tag seen almost face-on); falls back to the iterative method if not."""
    import cv2
    K, dist = calib
    obj = tag_corners(size).astype(np.float32)
    img = np.asarray(corners, np.float32).reshape(4, 1, 2)
    for flag in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_ITERATIVE):
        ok, r, t = cv2.solvePnP(obj, img, K, dist, flags=flag)
        if ok and t.ravel()[2] > 0:
            back = cv2.projectPoints(obj, r, t, K, dist)[0].reshape(4, 2)
            if np.abs(back - img.reshape(4, 2)).max() < 3.0:
                return cv2.Rodrigues(r)[0], t.ravel()
    return None


def tag_poses_in_cam(corners, size, calib):
    """Both (R, t) a square tag can have from its corners. A tag seen at a steep angle looks almost the same tilted
    either of two ways (the IPPE pair); the first guess must try both or it can get stuck on the wrong one."""
    import cv2
    K, dist = calib
    obj = tag_corners(size).astype(np.float32)
    img = np.asarray(corners, np.float32).reshape(4, 1, 2)
    out = []
    try:
        n, rs, ts, _ = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        n, rs, ts = 0, [], []
    for r, t in zip(rs[:n], ts[:n]):
        if t.ravel()[2] > 0:
            out.append((cv2.Rodrigues(r)[0], t.ravel()))
    if not out:
        p = tag_pose_in_cam(corners, size, calib)
        out = [p] if p is not None else []
    return out


def base_direction(w, S, torre, hand_tag, verbose=True):
    """
    Which way does a positive base count turn the tower? Try both: the right one keeps the table's tags still
    (seen from the tower camera, put back into table coordinates). The real arm turned out to turn opposite to the
    synthetic test's convention, and with the wrong sign every table tag got pinned to the tower instead.
    """
    if torre is None:
        return 1.0
    Rc, tc = torre
    by_tag = {}
    for (i, c, t, px) in S["obs"]:
        if c != "torre" or t == hand_tag:
            continue
        p = tag_pose_in_cam(px, w.tag_size.get(t, TAG_DEFAULT_MM), S["calib"][c])
        if p is not None:
            by_tag.setdefault(t, []).append((i, Rc.T @ (p[1] - tc)))
    score = {}
    for g in (1.0, -1.0):
        w.base[0] = g
        Rf, of = w.frames(S["q"], S["base"])
        spreads = []
        for rows in by_tag.values():
            if len(rows) < 3 or np.ptp([S["base"][i] for i, _ in rows]) < 5.0:
                continue                              # needs the base to have turned while it was seen
            pts = np.array([Rf[i, 0].T @ (tt - of[i, 0]) for i, tt in rows])
            spreads.append(np.linalg.norm(pts - np.median(pts, 0), axis=1).mean())
        score[g] = float(np.median(spreads)) if spreads else float("inf")
    g = min(score, key=score.get)
    if verbose:
        print(f"   base direction: {'+' if g > 0 else '-'} (table tags hold still to {score[g]:.0f} mm that way, "
              f"{score[-g]:.0f} mm the other)")
    return g


def rotate_direction(w, S, torre, hand_tag, verbose=True):
    """Which way does a positive wrist-rotate step turn the gripper? Try both: the right one keeps the hand tag still
    in the gripper's own frame (seen from the tower camera)."""
    if torre is None:
        return 1.0
    Rc, tc = torre
    rows = []
    for (i, c, t, px) in S["obs"]:
        if c == "torre" and t == hand_tag:
            p = tag_pose_in_cam(px, w.tag_size.get(t, TAG_DEFAULT_MM), S["calib"][c])
            if p is not None:
                rows.append((i, Rc.T @ (p[1] - tc)))
    score = {}
    for k in (1.0, -1.0):
        w.arm[9] = k
        Rf, of = w.frames(S["q"], S["base"])
        pts = np.array([Rf[i, G_IDX].T @ (tt - of[i, G_IDX]) for i, tt in rows]) if rows else np.zeros((0, 3))
        score[k] = float(np.linalg.norm(pts - np.median(pts, 0), axis=1).mean()) if len(pts) > 2 else float("inf")
    k = min(score, key=score.get)
    w.rot_ambiguous = not (score[-k] > 1.5 * score[k])
    if verbose:
        print(f"   wrist rotate direction: {'+' if k > 0 else '-'} (hand tag holds still to {score[k]:.0f} mm that way, "
              f"{score[-k]:.0f} mm the other)")
    return k


PINNED = {}                          # tag -> part index, from --on (you know where you stuck it)


def initialise(S, model_path, hand_tag=9, verbose=True, rot_sign=None):
    """A starting point good enough for the solver: tower camera and arm from the self-model, then everything else
    chained from there (tags the tower camera sees, then the webcam from those, then the gripper camera...)."""
    arm, ref, hand_tag_pos, cam_x = from_hand_model(model_path)
    cams = [c for c in ("torre", "webcam", "gripper", "gripper2") if c in S["calib"] and any(o[1] == c for o in S["obs"])]
    tags = sorted({o[2] for o in S["obs"]})
    w = World(cams, tags, len(S["q"]), np.concatenate([ref[:3], [ROLL_REF]]), S["calib"])
    w.arm = np.concatenate([arm[:8], ROT_DEFAULT])
    w.rot_varies = S["q"].shape[1] > 3 and float(np.ptp(S["q"][:, 3])) > 2.0
    w.grip_varies = S["q"].shape[1] > 4 and float(np.ptp(S["q"][:, 4])) > 2.0
    w.sv = S.get("sv", np.zeros(len(S["q"]), int))
    w.boff = np.zeros(int(w.sv.max()) + 1)
    for t in tags:
        w.tag_size[t] = S["sizes"].get(t, TAG_DEFAULT_MM)
    known_cam = {}
    if "torre" in cams:
        known_cam["torre"] = (rotvec_to_mat(cam_x[:3]), cam_x[3:])
    w.base[0] = base_direction(w, S, known_cam.get("torre"), hand_tag, verbose)
    if w.rot_varies:
        w.arm[9] = rot_sign if rot_sign is not None else rotate_direction(w, S, known_cam.get("torre"), hand_tag, verbose)
    Rf, of = w.frames(S["q"], S["base"])
    # tag poses seen from each camera, in that camera's frame
    seen = {}
    for (i, c, t, px) in S["obs"]:
        p = tag_pose_in_cam(px, w.tag_size[t], S["calib"][c])
        if p is not None:
            seen.setdefault(c, []).append((i, t, p))
    placed = {}                                       # tag -> (frame index, R, t) in that frame

    def to_T(c, i, R, t):
        Rc, tc = known_cam[c]
        R1, t1 = Rc.T @ R, Rc.T @ (t - tc)            # camera -> camera's parent
        pf = FRAMES.index(CAM_PARENT[c])
        if pf != T_IDX:
            R1, t1 = Rf[i, pf] @ R1, Rf[i, pf] @ t1 + of[i, pf]
        return R1, t1

    def place_tags_from(c):
        by_tag = {}
        for i, t, (R, tt) in seen.get(c, []):
            if t in placed:
                continue
            by_tag.setdefault(t, []).append(to_T(c, i, R, tt) + (i,))
        for t, rows in by_tag.items():
            if t == hand_tag:
                best = (0.0, G_IDX)
            elif t in PINNED:
                best = (0.0, PINNED[t])
            else:
                spreads = []
                for f in range(len(FRAMES)):
                    pts = np.array([Rf[i, f].T @ (tt - of[i, f]) for _, tt, i in rows])
                    spreads.append((np.linalg.norm(pts - np.median(pts, 0), axis=1).mean() if len(rows) > 1 else 0.0, f))
                best = min(spreads)
            f = best[1]
            pts = np.array([Rf[i, f].T @ (tt - of[i, f]) for _, tt, i in rows])
            Rs = np.array([Rf[i, f].T @ R for R, _, i in rows])
            placed[t] = (f, mean_rotation(Rs), np.median(pts, 0))

    def locate_camera(c):
        """Camera pose in its parent frame from tags already placed (centres matched, Kabsch)."""
        a, b = [], []
        for i, t, (R, tt) in seen.get(c, []):
            if t not in placed:
                continue
            f, Rt, pt = placed[t]
            pT = Rf[i, f] @ pt + of[i, f]
            pf = FRAMES.index(CAM_PARENT[c])
            if pf != T_IDX:
                pT = Rf[i, pf].T @ (pT - of[i, pf])
            a.append(pT)
            b.append(tt)
        if len(a) < 4:
            return False
        Rc, tc = kabsch(np.array(a), np.array(b))
        known_cam[c] = (Rc, tc)
        return True

    if "torre" in known_cam:
        place_tags_from("torre")
    for _ in range(3):                                # chain: webcam, gripper, then their tags, until all placed
        for c in cams:
            if c not in known_cam and locate_camera(c):
                if verbose:
                    print(f"   initial {c} camera pose from {sum(1 for x in seen.get(c, []) if x[1] in placed)} tag sightings")
            if c in known_cam:
                place_tags_from(c)
    if hand_tag in placed:                            # the self-model knows the hand tag better than one camera does
        f, R, _ = placed[hand_tag]
        placed[hand_tag] = (G_IDX, R, hand_tag_pos)
    for c in cams:
        if c not in known_cam:
            raise SystemExit(f"Could not get a first guess for the {c} camera: it never saw enough tags the others saw.")
        w.cam[c] = np.concatenate([mat_to_rotvec(known_cam[c][0]), known_cam[c][1]])
    unplaced = [t for t in tags if t not in placed]
    for t in unplaced:
        w.tags.remove(t)
    for t in w.tags:
        f, R, p = placed[t]
        w.tag_frame[t] = f
        w.tag[t] = np.concatenate([mat_to_rotvec(R), p])
    if verbose:
        print("   tags: " + ", ".join(f"{t} on {FRAMES[w.tag_frame[t]]}" for t in w.tags)
              + (f"   (not placed: {unplaced})" if unplaced else ""))
    return w


# ---------------------------------------------------------------- solving
def residuals_fn(world, O, q, base, prior=PRIORS):
    def f(x):
        world.unpack(x)
        pix = world.predict(O, q, base, world.dq)
        d = (pix - O["px"]).ravel()
        if PROGRESS is not None:
            PROGRESS.tick(float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan"))
        return np.concatenate([d, (world.dq / prior).ravel()])
    return f


def sparsity(world, O):
    from scipy.sparse import lil_matrix
    lay, n = world.layout()
    rows = 8 * len(O["pose"]) + ND * world.n
    A = lil_matrix((rows, n), dtype=int)
    for k in range(len(O["pose"])):
        r = slice(8 * k, 8 * k + 8)
        A[r, lay["arm"]] = 1
        A[r, lay["base"]] = 1
        A[r, lay["boff"].start + int(world.sv[O["pose"][k]])] = 1
        A[r, lay[("cam", world.cams[O["cam"][k]])]] = 1
        if world.cams[O["cam"][k]] in world.fs:
            A[r, lay[("fs", world.cams[O["cam"][k]])]] = 1
        A[r, lay[("tag", world.tags[O["tag"][k]])]] = 1
        if world.tags[O["tag"][k]] in FINGER_OF:
            A[r, lay["fing"]] = 1
        i = O["pose"][k]
        A[r, lay["dq"].start + ND * i: lay["dq"].start + ND * i + ND] = 1
    for i in range(world.n):
        for j in range(ND):
            A[8 * len(O["pose"]) + ND * i + j, lay["dq"].start + ND * i + j] = 1
    return A


# ---------------------------------------------------------------- fast Jacobian
# Each tag sighting depends on only a few dozen of the unknowns: the arm, the base (3), that survey's base zero (1),
# that camera's pose (6) and focal scale (1), that tag's pose (6), that pose's joint corrections, and for a finger tag
# its finger's model (6). So instead of letting scipy nudge the unknowns in hundreds of groups (one full camera-model
# run each), every sighting gets its own copy of its numbers and all sightings are nudged together: two vectorised
# runs per local number give the whole Jacobian.
LOCAL = N_ARM + 3 + 1 + 6 + 1 + 6 + ND + 6   # arm, base, survey base zero, camera, focal scale, tag, pose corrections, finger
SL_ARM, SL_BASE, SL_BOFF, SL_CAM, SL_FS, SL_TAG = 0, N_ARM, N_ARM + 3, N_ARM + 4, N_ARM + 10, N_ARM + 11
SL_DQ = N_ARM + 17
SL_FING = SL_DQ + ND                          # ax, ay, then r, phi0, k, k2 of the sighting's finger (zeros: not a finger)


def _rot(r):
    """Rodrigues for (N, 3) rotation vectors -> (N, 3, 3)."""
    th = np.linalg.norm(r, axis=1)
    k = r / np.maximum(th, 1e-12)[:, None]
    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K = K - np.swapaxes(K, 1, 2)
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def _rxz(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, -s, 1.0, s, c
    return R


class FastJac:
    """Residuals and their sparse Jacobian for one set of sightings O (same model as World.predict)."""

    def __init__(self, world, O, q, base, prior=PRIORS):
        lay, self.n_x = world.layout()
        N = len(O["pose"])
        self.N, self.world = N, world
        i = O["pose"]
        idx = -np.ones((N, LOCAL), int)                                  # where each local number lives in x
        idx[:, SL_ARM:SL_ARM + N_ARM] = lay["arm"].start + np.arange(N_ARM)
        idx[:, SL_BASE:SL_BASE + 3] = lay["base"].start + np.arange(3)
        if len(world.boff) > 1:
            idx[:, SL_BOFF] = lay["boff"].start + world.sv[i]
        cams = [world.cams[c] for c in O["cam"]]
        idx[:, SL_CAM:SL_CAM + 6] = np.array([lay[("cam", c)].start for c in cams])[:, None] + np.arange(6)
        idx[:, SL_FS] = [lay[("fs", c)].start if c in world.fs else -1 for c in cams]
        tags = [world.tags[t] for t in O["tag"]]
        idx[:, SL_TAG:SL_TAG + 6] = np.array([lay[("tag", t)].start for t in tags])[:, None] + np.arange(6)
        idx[:, SL_DQ:SL_DQ + ND] = lay["dq"].start + ND * i[:, None] + np.arange(ND)
        fing = np.array([FINGER_OF.get(t, -1) for t in tags])
        self.is_finger = fing >= 0
        if self.is_finger.any():                                         # a finger tag: its finger's arc
            f0 = lay["fing"].start
            idx[self.is_finger, SL_FING:SL_FING + 2] = f0 + np.arange(2)
            idx[self.is_finger, SL_FING + 2:SL_FING + 6] = f0 + 2 + 4 * fing[self.is_finger][:, None] + np.arange(4)
        self.idx = idx
        self.fill = np.zeros(LOCAL)
        self.fill[SL_FS] = 1.0                                           # focal scale of a camera that has none
        qq = np.asarray(q, float)[i]
        if qq.shape[1] < 4:
            qq = np.column_stack([qq[:, :3], np.full(N, world.ref[3])])
        self.q = qq[:, :4]
        self.grip = qq[:, 4] if qq.shape[1] > 4 else np.full(N, GRIP_SURVEYED)
        self.base = np.asarray(base, float)[i]
        self.ref = world.ref[:4]
        self.frame = np.array([world.tag_frame[t] for t in tags])
        self.parent = np.array([FRAMES.index(CAM_PARENT[c]) for c in cams])
        self.X = np.array([tag_corners(world.tag_size[t]) for t in tags])         # (N, 4, 3)
        Ks = {c: world.calib[c] for c in set(cams)}
        self.fxy = np.array([[Ks[c][0][0, 0], Ks[c][0][1, 1]] for c in cams])
        self.cxy = np.array([[Ks[c][0][0, 2], Ks[c][0][1, 2]] for c in cams])
        self.dist = np.array([Ks[c][1][:5] for c in cams])
        self.px = O["px"]
        self.n_dq = ND * world.n
        self.prior_vec = np.tile(np.asarray(prior, float), world.n)
        self.dq_start = lay["dq"].start
        self.ax_start = lay["arm"].start + 10
        rows = (np.arange(N)[:, None] * 8 + np.arange(8))[:, :, None]      # (N, 8, 1)
        ok = idx >= 0
        self.cols = np.broadcast_to(np.where(ok, idx, 0)[:, None, :], (N, 8, LOCAL))
        self.rows = np.broadcast_to(rows, (N, 8, LOCAL))
        self.ok = np.broadcast_to(ok[:, None, :], (N, 8, LOCAL))

    def local(self, x):
        return np.where(self.idx >= 0, x[np.maximum(self.idx, 0)], self.fill[None])

    def pixels(self, P):
        """(N, LOCAL) local numbers -> (N, 4, 2) predicted corner pixels."""
        th1, k1, th2, k2, th3, k3, L1, L2, th4, k4, au, av, ay, az, k4b = P[:, SL_ARM:SL_ARM + N_ARM].T
        g, cx, cy = P[:, SL_BASE:SL_BASE + 3].T
        dq = P[:, SL_DQ:SL_DQ + ND]
        d = self.q + dq[:, :4] - self.ref
        a1 = np.radians(th1 + k1 * d[:, 0])
        a2 = a1 + np.radians(th2 + k2 * d[:, 1])
        a3 = a2 + np.radians(th3 + k3 * d[:, 2])
        a4 = np.radians(th4 + k4 * d[:, 3] + k4b * d[:, 3] ** 2)
        N = len(P)
        R = np.zeros((N, 6, 3, 3))
        o = np.zeros((N, 6, 3))
        b = np.radians(g * (self.base + P[:, SL_BOFF] + dq[:, DQ_BASE]))
        cb, sb = np.cos(-b), np.sin(-b)
        R[:, 0, 0, 0], R[:, 0, 0, 1], R[:, 0, 1, 0], R[:, 0, 1, 1], R[:, 0, 2, 2] = cb, -sb, sb, cb, 1.0
        C = np.stack([cx, cy, np.zeros(N)], 1)
        o[:, 0] = C - np.einsum("nij,nj->ni", R[:, 0], C)
        R[:, 1] = _rxz(a1)
        elbow = L1[:, None] * np.stack([np.cos(a1), np.zeros(N), np.sin(a1)], 1)
        R[:, 2], o[:, 2] = _rxz(a2), elbow
        R[:, 3] = _rxz(a3)
        o[:, 3] = elbow + L2[:, None] * np.stack([np.cos(a2), np.zeros(N), np.sin(a2)], 1)
        R[:, 4] = np.eye(3)
        nax = np.stack([np.ones(N), au, av], 1)
        nax = nax / np.linalg.norm(nax, axis=1, keepdims=True)
        pa = np.stack([np.zeros(N), ay, az], 1)
        Rr = _rot(a4[:, None] * nax)
        R[:, 5] = R[:, 3] @ Rr
        o[:, 5] = o[:, 3] + np.einsum("nij,nj->ni", R[:, 3], pa - np.einsum("nij,nj->ni", Rr, pa))
        n = np.arange(N)
        Rf, of = R[n, self.frame], o[n, self.frame]
        Rt = _rot(P[:, SL_TAG:SL_TAG + 3])
        Xp = np.einsum("nij,nkj->nki", Rt, self.X) + P[:, None, SL_TAG + 3:SL_TAG + 6]   # corners in the tag's part
        h = self.is_finger
        if h.any():                                                      # finger tags: slid along their arc
            Xp[h] += finger_offset(P[h, SL_FING:SL_FING + 6], self.grip[h] + dq[h, DQ_GRIP] - GRIP_SURVEYED)[:, None]
        pT = np.einsum("nij,nkj->nki", Rf, Xp) + of[:, None]
        for pf in np.unique(self.parent):
            if pf == T_IDX:
                continue
            h = self.parent == pf
            pT[h] = np.einsum("nji,nkj->nki", R[h, pf], pT[h] - o[h, pf][:, None])
        Rc = _rot(P[:, SL_CAM:SL_CAM + 3])
        pc = np.einsum("nij,nkj->nki", Rc, pT) + P[:, None, SL_CAM + 3:SL_CAM + 6]
        z = np.maximum(pc[..., 2], 1e-6)
        xx, yy = pc[..., 0] / z, pc[..., 1] / z
        k1_, k2_, p1, p2, k3_ = [v[:, None] for v in self.dist.T]
        r2 = xx * xx + yy * yy
        rad = 1 + k1_ * r2 + k2_ * r2 * r2 + k3_ * r2 ** 3
        xd = xx * rad + 2 * p1 * xx * yy + p2 * (r2 + 2 * xx * xx)
        yd = yy * rad + p1 * (r2 + 2 * yy * yy) + 2 * p2 * xx * yy
        fs = P[:, SL_FS:SL_FS + 1]
        return np.stack([self.fxy[:, :1] * fs * xd + self.cxy[:, :1], self.fxy[:, 1:] * fs * yd + self.cxy[:, 1:]], -1)

    def residuals(self, x):
        d = (self.pixels(self.local(x)) - self.px).reshape(-1)
        if PROGRESS is not None:
            PROGRESS.tick(float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan"))
        dq = x[self.dq_start:self.dq_start + self.n_dq]
        extra = x[self.ax_start:self.ax_start + 4] / AXIS_PRIOR if self.world.rot_varies else np.zeros(0)
        return np.concatenate([d, dq / self.prior_vec, extra])

    def jacobian(self, x):
        from scipy.sparse import coo_matrix
        P = self.local(x)
        J = np.zeros((self.N, 8, LOCAL))
        cols = [j for j in range(LOCAL) if (self.idx[:, j] >= 0).any()]

        def column(j):                                                   # central differences, all sightings at once
            h = 1e-6 * (1.0 + np.abs(P[:, j]))
            Pw = getattr(_scratch, "P", None)              # one scratch per THREAD, not per column: this loop is
            if Pw is None or Pw.shape != P.shape:          # bandwidth-bound, and copying P 44 times a Jacobian
                Pw = _scratch.P = np.empty_like(P)         # was most of what it was doing
            np.copyto(Pw, P)
            Pw[:, j] = P[:, j] + h
            plus = self.pixels(Pw)
            Pw[:, j] = P[:, j] - h
            J[:, :, j] = ((plus - self.pixels(Pw)).reshape(self.N, 8)) / (2 * h)[:, None]

        # One column per thread. They are independent -- each reads P and writes its own slice of J, and numpy
        # drops the GIL for the elementwise work in pixels() -- so this is the same arithmetic on more cores.
        # Worth doing since the thread cap: capping BLAS cut the dense solve about sevenfold and left this as
        # the slowest part of a step.
        pool = jac_pool(min(JAC_THREADS, len(cols)))
        if pool is not None:
            list(pool.map(column, cols))
        else:
            for j in cols:
                column(j)
        ok = self.ok
        n_ax = 4 if self.world.rot_varies else 0
        rows = np.concatenate([self.rows[ok], 8 * self.N + np.arange(self.n_dq + n_ax)])
        cols = np.concatenate([self.cols[ok], self.dq_start + np.arange(self.n_dq), self.ax_start + np.arange(n_ax)])
        data = np.concatenate([J[ok], 1.0 / self.prior_vec, 1.0 / AXIS_PRIOR[:n_ax]])
        return coo_matrix((data, (rows, cols)), shape=(8 * self.N + self.n_dq + n_ax, self.n_x)).tocsr()


class _Result:
    def __init__(self, x, fun):
        self.x, self.fun = x, fun


JAC_THREADS = min(4, os.cpu_count() or 1)
# Threads for the Jacobian's independent columns. 8 was a guess by analogy with blas_threads; measured on a
# 60-pose synthetic (24 logical cores), every count was tried and 4 won:
#     1 thread 13.6 s    4 threads 10.4 s    8 threads 11.4 s    16 threads 18.3 s
# The reason it stops paying so early is memory bandwidth, not thread setup: making the pool once instead of
# per call, and keeping one scratch array per thread instead of copying the pose array for all 44 columns,
# together bought about 7% and did not move where the optimum sits. The work is elementwise passes over big
# arrays and 24 cores share one memory controller, so past about 4 threads there is no more throughput to have.


_POOL = {}
_scratch = threading.local()          # each thread keeps one working copy of the pose array


def jac_pool(n):
    """One thread pool, kept. None when there is nothing to spread.

    It used to be made and thrown away inside every jacobian() call -- spawning threads, using them for a few
    milliseconds and tearing them down, hundreds of times a solve. The symptom: fewer threads
    looked faster on a small problem, which is what you see when setup costs more than the work.
    """
    if n < 2:
        return None
    if n not in _POOL:
        _POOL[n] = ThreadPoolExecutor(max_workers=n)
    return _POOL[n]


def blas_threads(n):
    """How many BLAS threads to give a dense n x n solve. Fewer than you would think.

    Measured on the dev laptop (24 logical cores, scipy-openblas 0.3.31):

        n= 400   1 thread 0.003 s    8 threads 0.023 s    24 threads 0.166 s
        n=1200   1 thread 0.047 s    8 threads 0.026 s    24 threads 1.662 s
        n=2400   1 thread 0.379 s    8 threads 0.153 s    24 threads 3.259 s
        n=3900   1 thread 1.508 s    8 threads 0.612 s    24 threads 4.089 s

    OpenBLAS takes every core it can find by default, and at 24 it spends far longer handing work between
    threads than doing it -- 7x worse than 8 threads on the biggest matrix here and 64x worse on a middling
    one. A profile of a 40-pose solve had 26 of its 50 seconds inside np.linalg.solve on a 400-square matrix,
    which should take milliseconds. Nothing about the arithmetic changes; only how many threads run it.
    """
    return 1 if n < 800 else 8


def lm_solve(fun, jac, x0, free, f_scale=2.0, max_iter=150, tol=1e-7):
    """
    Levenberg-Marquardt with the same soft_l1 robust loss as before, on the free unknowns only. Each step solves one
    small (unknowns x unknowns) system, which is what makes bundle adjustment fast.
    """
    x = x0.copy()

    def cost(r):
        return float(np.sum(2.0 * f_scale ** 2 * (np.sqrt(1.0 + (r / f_scale) ** 2) - 1.0)))
    r = fun(x)
    c = cost(r)
    lam = 1e-3
    free_cols = np.flatnonzero(free)             # the same every iteration: work it out once
    A = None
    for _ in range(max_iter):
        J = jac(x).tocsc()[:, free_cols].tocsr()  # CSC first: slicing columns of a CSR scans every row
        w = 1.0 / np.sqrt(1.0 + (r / f_scale) ** 2)                 # soft_l1 as weights (big misses count less)
        Jw = J.copy()                            # same as J.multiply(w[:, None]): scale each row by its weight,
        Jw.data *= np.repeat(w, np.diff(J.indptr))   # in place on the data, with no new sparsity structure built
        H = (J.T @ Jw).toarray()
        g = Jw.T @ r
        diag = np.einsum("ii->i", H)             # a view of H's diagonal, not a copy
        d = np.maximum(diag, 1e-9 * max(1.0, float(diag.max())))
        if A is None or A.shape != H.shape:
            A = np.empty_like(H)
        improved = False
        for _ in range(12):
            try:
                np.copyto(A, H)                  # H + lam * diag(d), without building an n x n diagonal matrix
                np.einsum("ii->i", A)[:] += lam * d
                with _blas(limits=blas_threads(A.shape[0]), user_api="blas"):
                    step = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            xn = x.copy()
            xn[free] += step
            rn = fun(xn)
            cn = cost(rn) if np.all(np.isfinite(rn)) else np.inf
            if cn < c:
                done = (c - cn) < tol * c
                x, r, c = xn, rn, cn
                lam = max(lam / 3.0, 1e-9)
                improved = True
                break
            lam *= 5.0
        if not improved or done:
            break
    return _Result(x, r)


FAST = True                          # --slow: let scipy estimate the Jacobian itself (the old way)


def solve(world, S, poses=None, verbose=True, fix_arm=False, fix_dq=False):
    from scipy.optimize import least_squares
    O = obs_arrays(world, S["obs"] if poses is None else [o for o in S["obs"] if o[0] in poses])
    if FAST:
        fj = FastJac(world, O, S["q"], S["base"])
        f, jac = fj.residuals, fj.jacobian
    else:
        f, jac = residuals_fn(world, O, S["q"], S["base"]), "2-point"
    x0 = world.pack()
    lay, _ = world.layout()
    lo, hi = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    if fix_arm:
        lo[lay["arm"]], hi[lay["arm"]] = x0[lay["arm"]] - 1e-9, x0[lay["arm"]] + 1e-9
    a0 = lay["arm"].start
    fixed_rot = [8] if world.rot_varies else list(range(8, 15))   # th4 is never needed (the gripper's own parts absorb it)
    for j in fixed_rot:
        lo[a0 + j], hi[a0 + j] = x0[a0 + j] - 1e-9, x0[a0 + j] + 1e-9
    if not world.rot_varies:                          # no rotate data: no rotate corrections either
        d0 = lay["dq"].start + DQ_ROT
        lo[d0::ND][:world.n], hi[d0::ND][:world.n] = x0[d0::ND][:world.n] - 1e-9, x0[d0::ND][:world.n] + 1e-9
    if not world.fing_on:                             # no finger model (yet): fingers and gripper slack stay put
        lo[lay["fing"]], hi[lay["fing"]] = x0[lay["fing"]] - 1e-9, x0[lay["fing"]] + 1e-9
        d0 = lay["dq"].start + DQ_GRIP
        lo[d0::ND][:world.n], hi[d0::ND][:world.n] = x0[d0::ND][:world.n] - 1e-9, x0[d0::ND][:world.n] + 1e-9
    elif fix_arm:                                     # cameras + tags stage: the finger model holds too
        lo[lay["fing"]], hi[lay["fing"]] = x0[lay["fing"]] - 1e-9, x0[lay["fing"]] + 1e-9
    b0 = lay["boff"].start                           # the first survey's base zero is the reference
    lo[b0], hi[b0] = x0[b0] - 1e-9, x0[b0] + 1e-9
    if fix_dq:                                        # no per-pose corrections yet: they must not soak up a bad tag
        lo[lay["dq"]], hi[lay["dq"]] = x0[lay["dq"]] - 1e-9, x0[lay["dq"]] + 1e-9
    t0 = time.time()
    A = None if FAST else sparsity(world, O)
    if PROGRESS is not None:
        if FAST:
            calls = 1
        else:
            from scipy.optimize._numdiff import group_columns
            calls = int(group_columns(A.tocsc()).max()) + 2
        PROGRESS.start_stage("everything" if not fix_arm else "cameras + tags", calls)
    r0 = f(x0)
    if FAST:
        res = lm_solve(f, jac, x0, (hi - lo) > 1e-6)
    else:
        res = least_squares(f, x0, jac=jac, jac_sparsity=A, loss="soft_l1", f_scale=2.0, x_scale="jac",
                        bounds=(lo, hi), max_nfev=200, verbose=0)
    if not np.all(np.isfinite(res.fun)) or rms(res.fun) > rms(r0):
        if verbose:                                   # it ran away (a badly pinned tag can do that): keep what we had
            print("   (that solve ran away: kept the previous answer)")
        return O, res
    world.unpack(res.x)                               # (the fast path does not unpack as it goes)
    if verbose:
        n = 8 * len(O["pose"])
        print(f"   solved in {time.time() - t0:.0f} s: corner error {rms(r0[:n]):.2f} px -> {rms(res.fun[:n]):.2f} px (rms), "
              f"{len(O['pose'])} tag sightings, {len(res.x)} unknowns")
    return O, res


def prune_tags(world, S, poses, limit_px=25.0, keep=(9, 31, 32, AFTER + 31, AFTER + 32), verbose=True):
    """
    Drop tags that fit nowhere: their sightings stay far off whatever part they were pinned to (a tag that was
    moved during the survey, or pinned to the wrong part by the first guess). In one survey tag 61 got pinned to
    the forearm, was ~200 px off, and the per-pose joint corrections bent every pose (by up to 40 deg) to chase it.
    """
    # A tag named with --on is kept: someone looked at the arm and said where it is, and this runs while the arm
    # is still frozen at its first guess, so a tag on a link can only fit as well as that guess puts the link.
    # Tags 320, 321, 360 and 322 were pinned to the forearm, wrist block and gripper and dropped anyway at
    # 36-55 px, which left the forearm and wrist block with no tags at all and the arm's geometry resting on the
    # three hand tags -- L1 and th1 then wandered between solves because nothing pinned them down.
    # The danger this guards against is real (see the tag 61 story above), so the pinned tags' errors are
    # printed at the end whatever happens: a pin on the wrong part should be obvious, not silent.
    keep = tuple(keep) + tuple(PINNED)
    O = obs_arrays(world, [o for o in S["obs"] if o[0] in poses])
    err = np.linalg.norm(world.predict(O, S["q"], S["base"], world.dq) - O["px"], axis=-1).mean(axis=1)
    kept_pinned = []
    drop = []
    for ti, t in enumerate(list(world.tags)):
        e = err[O["tag"] == ti]
        if t in PINNED and len(e):
            kept_pinned.append((t, float(np.median(e)), FRAMES[world.tag_frame[t]]))
        if t not in keep and len(e) and float(np.median(e)) > limit_px:
            drop.append((t, float(np.median(e)), FRAMES[world.tag_frame[t]]))
    for t, _, _ in drop:
        world.tags.remove(t)
    if verbose and drop:
        print("   dropped (fit nowhere): " + ", ".join(f"tag {t} ({e:.0f} px off on {f})" for t, e, f in drop))
    if verbose and kept_pinned:
        bad = [r for r in kept_pinned if r[1] > limit_px]
        print("   kept because you pinned them: "
              + ", ".join(f"tag {t} ({e:.0f} px on {f})" for t, e, f in sorted(kept_pinned)))
        if bad:
            print("      (%s over the %.0f px the others are held to -- if one of those is on the wrong part it "
                  "will drag the arm, so check it before trusting the result)"
                  % (", ".join("tag %d" % t for t, _, _ in bad), limit_px))
    return [t for t, _, _ in drop]


def tag_errors(world, S, poses):
    """{tag: median corner error (px) over its sightings in these poses}."""
    O = obs_arrays(world, [o for o in S["obs"] if o[0] in poses])
    err = np.linalg.norm(world.predict(O, S["q"], S["base"], world.dq) - O["px"], axis=-1).mean(axis=1)
    return {t: float(np.median(err[O["tag"] == ti])) for ti, t in enumerate(world.tags) if (O["tag"] == ti).any()}


def reassign_tags(world, S, poses, limit_px=25.0, keep=(9,), verbose=True, only=None):
    """
    A tag that fits badly may be pinned to the wrong part (first guesses come from few sightings), or tilted the
    wrong way (a tag seen at a steep angle can look tilted either of two ways). With the cameras now settled, place
    it afresh on each part in turn (table, upper arm, forearm, hand, tower), trying both tilts every sighting allows,
    and keep whichever fits best.
    """
    errs = tag_errors(world, S, poses)
    bad = [t for t, e in errs.items() if e > limit_px and t not in keep and t not in PINNED and (only is None or t in only)]
    if not bad:
        return []
    Rf, of = world.frames(S["q"], S["base"], world.dq)
    moved = []
    for t in bad:
        obs_t = [o for o in S["obs"] if o[2] == t and o[0] in poses]
        O_t = obs_arrays(world, obs_t)

        def err_now():
            pix = world.predict(O_t, S["q"], S["base"], world.dq)
            return float(np.median(np.linalg.norm(pix - O_t["px"], axis=-1).mean(axis=1)))

        rows = []                                      # (pose, [both (R, t) in the camera's parent frame...])
        for (i, c, _, px) in obs_t:
            Rc, tc = rotvec_to_mat(world.cam[c][:3]), world.cam[c][3:]
            sols = []
            for R, tt in tag_poses_in_cam(px, world.tag_size[t], world.calib[c]):
                R1, t1 = Rc.T @ R, Rc.T @ (tt - tc)
                pf = FRAMES.index(CAM_PARENT[c])
                if pf != T_IDX:
                    R1, t1 = Rf[i, pf] @ R1, Rf[i, pf] @ t1 + of[i, pf]
                sols.append((R1, t1))
            if sols:
                rows.append((i, sols))
        if len(rows) < 2:
            continue
        old = (world.tag_frame[t], world.tag[t].copy())
        best = (errs[t], old[0], old[1])
        for f in range(len(FRAMES)):
            world.tag_frame[t] = f
            pts = np.array([Rf[i, f].T @ (sols[0][1] - of[i, f]) for i, sols in rows])
            p_med = np.median(pts, 0)
            # every orientation any sighting allows is a candidate; keep the one that fits all sightings best
            cands = [Rf[i, f].T @ R for i, sols in rows for R, _ in sols]
            step = max(1, len(cands) // 40)            # at most ~40 tries per part
            for Rcand in cands[::step]:
                world.tag[t] = np.concatenate([mat_to_rotvec(Rcand), p_med])
                e = err_now()
                if e < best[0]:
                    best = (e, f, world.tag[t].copy())
        world.tag_frame[t], world.tag[t] = best[1], best[2]
        if best[1] != old[0] or best[0] < errs[t] - 1.0:
            moved.append((t, FRAMES[old[0]], FRAMES[best[1]], errs[t], best[0]))
    if verbose and moved:
        print("   re-placed: " + ", ".join(f"tag {t} {a}->{b} ({e0:.0f}->{e1:.0f} px)" for t, a, b, e0, e1 in moved))
    return moved


def rms(v):
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")


def report(world, S, O, label=""):
    """Pixel error per camera and per tag."""
    pix = world.predict(O, S["q"], S["base"], world.dq)
    err = np.linalg.norm(pix - O["px"], axis=-1).mean(axis=1)           # mean corner error per sighting (px)
    out = {"cameras": {}, "tags": {}}
    for ci, c in enumerate(world.cams):
        e = err[O["cam"] == ci]
        if len(e):
            out["cameras"][c] = {"sightings": int(len(e)), "median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90))}
    for ti, t in enumerate(world.tags):
        e = err[O["tag"] == ti]
        if len(e):
            out["tags"][str(t)] = {"on": FRAMES[world.tag_frame[t]], "sightings": int(len(e)), "median_px": float(np.median(e))}
    print(f"   {label}corner error per camera:  " + "   ".join(
        f"{c} {v['median_px']:.2f} px median ({v['sightings']} sightings)" for c, v in out["cameras"].items()))
    return out


def holdout_error(world, S, test, train_dq_mean=None):
    """For poses the solve never saw: how far off is the predicted hand (mm, from the tower camera's view)?
    Joint corrections cannot be known for unseen poses, so they are zero here."""
    O = obs_arrays(world, [o for o in S["obs"] if o[0] in test])
    if not len(O["pose"]):
        return None
    dq = np.zeros_like(world.dq)
    pix = world.predict(O, S["q"], S["base"], dq)
    e = np.linalg.norm(pix - O["px"], axis=-1).mean(axis=1)
    fing = np.array([world.tags[t] in FINGER_OF for t in O["tag"]])
    out = {"sightings": int(len(e)), "median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90)),
           "cameras": {c: float(np.median(e[O["cam"] == ci])) for ci, c in enumerate(world.cams) if (O["cam"] == ci).any()}}
    if fing.any() and (~fing).any():                  # finger tags are the hardest to predict (up close, gripper slack
        out["without_fingers_px"] = float(np.median(e[~fing]))          # unknown): show the rest on its own too
        out["fingers_px"] = float(np.median(e[fing]))
    return out


# ---------------------------------------------------------------- finger model: first guess
def init_fingers(world, S, poses, verbose=True):
    """
    A first guess of the finger model from the gripper camera, which rides on G and sees the finger tags up close:
    every sighting of a finger tag, put into G, traces that finger's arc as the gripper value changes. Fit the plane
    both arcs lie in, a circle per finger, and how far round the circle each gripper value is. Returns True if it
    could (enough sightings over a wide enough range of gripper values), else False (the solve carries on without it).
    """
    pts = {0: [], 1: []}
    for (i, c, t, px) in S["obs"]:
        f = FINGER_OF.get(t)
        if f is None or i not in poses or c not in world.cam or CAM_PARENT.get(c) != "G" or t not in world.tag:
            continue
        p = tag_pose_in_cam(px, world.tag_size[t], world.calib[c])
        if p is None:
            continue
        Rc, tc = rotvec_to_mat(world.cam[c][:3]), world.cam[c][3:]
        pts[f].append((float(S["q"][i, 4]) - GRIP_SURVEYED, Rc.T @ (p[1] - tc) - world.tag[t][3:3 + 3], t))
    fits = {}
    for f, rows in pts.items():
        d = np.array([r[0] for r in rows])
        if len(rows) < 12 or np.ptp(d) < 30.0:
            if verbose:
                print(f"   finger model: not enough {'LR'[f]} finger sightings over a range of gripper values "
                      f"({len(rows)} sightings, range {np.ptp(d) if len(d) else 0:.0f} steps): skipped")
            return False
        P = np.array([r[1] for r in rows])
        fits[f] = (d, P)
    normals = []
    for f, (d, P) in fits.items():
        n = np.linalg.svd(P - P.mean(0))[2][2]
        normals.append(n if n[2] >= 0 else -n)
    n = np.mean(normals, 0)
    n /= np.linalg.norm(n)
    if n[2] < 0.5:
        if verbose:
            print(f"   finger model: the fingers swing in a plane far from the one expected (normal {np.round(n, 2)}): skipped")
        return False
    world.fing[:2] = n[:2] / n[2]
    _, u, v = finger_plane(world.fing[0], world.fing[1])
    u, v = u[0], v[0]
    for f, (d, P) in fits.items():
        x, y = P @ u, P @ v
        keep = np.ones(len(x), bool)
        for _ in range(2):                                           # fit, drop the misreads, fit again
            A = np.column_stack([x[keep], y[keep], np.ones(keep.sum())])
            D, E, F = np.linalg.lstsq(A, -(x[keep] ** 2 + y[keep] ** 2), rcond=None)[0]
            cx, cy = -D / 2, -E / 2
            r = float(np.sqrt(max(cx * cx + cy * cy - F, 1e-9)))
            keep = np.abs(np.hypot(x - cx, y - cy) - r) < 2.5
        phi0 = np.arctan2(-cy, -cx)                                  # the tag at GRIP_SURVEYED sits at the origin
        a = np.degrees(np.angle(np.exp(1j * (np.arctan2(y - cy, x - cx) - phi0))))
        k, k2 = np.linalg.lstsq(np.column_stack([d[keep], d[keep] ** 2]), a[keep], rcond=None)[0]
        if not 5.0 < r < 120.0:
            if verbose:
                print(f"   finger model: the {'LR'[f]} finger's arc came out {r:.0f} mm across: skipped")
            return False
        world.fing[2 + 4 * f: 6 + 4 * f] = [r, np.degrees(phi0), k, k2]
        if verbose:
            print(f"   finger model first guess, {'LR'[f]} finger: arc radius {r:.1f} mm, {k:+.3f} deg per step "
                  f"({keep.sum()} of {len(x)} gripper-camera sightings)")
    world.fing_on = True
    return True


# ---------------------------------------------------------------- synthetic self-test
TRUE_FINGERS = np.array([0.03, -0.04, 31.0, 90.0, -0.80, 0.0006, 31.0, -90.0, 0.80, -0.0006])   # synthetic test's fingers


def synthetic(n_poses=250, seed=0, noise_px=0.4, play_deg=0.4, rotate=False, grip=False):
    """Make a survey from a known world (with noise and gear play), then check the solver gets the world back.
    grip: the gripper opens and closes too (survey v2), with fingers on arcs (TRUE_FINGERS) and gripper slack."""
    rng = np.random.default_rng(seed)
    K = np.array([[1137.0, 0, 640], [0, 1137.0, 480], [0, 0, 1]])
    calib = {"torre": (K, np.zeros(5)),
             "webcam": (np.array([[1876.0, 0, 888], [0, 1882.0, 443], [0, 0, 1]]), np.array([0.067, -0.39, -0.015, -0.004, 0.79])),
             "gripper": (np.array([[1196.0, 0, 555], [0, 1193.0, 410], [0, 0, 1]]), np.zeros(5))}
    table = [50, 51, 52, 53, 54, 55] + list(range(100, 112))                  # landmarks and a few mat tags
    tags = [9, 31, 32, 40, 41, 70, 71, 72, 73] + table
    truth = World(["torre", "webcam", "gripper"], tags, n_poses, [123.0, 54.0, 68.0, ROLL_REF], calib)
    truth.arm = np.concatenate([[-3.3, 0.436, -11.0, -1.0, 9.6, 1.0, 230.0, 190.0],
                                [0.0, -0.97, 0.03, -0.20, 4.0, -6.0, 0.002] if rotate else ROT_DEFAULT])
    truth.rot_varies = rotate
    truth.base = np.array([1.0, -12.0, 6.0])

    def look_at(eye, target, up=(0, 0, 1)):
        z = np.asarray(target, float) - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])                      # world -> camera rows
        return np.concatenate([mat_to_rotvec(R), -R @ eye])

    truth.cam["torre"] = look_at(np.array([-60.0, 0.0, 180.0]), [300, 0, -150])
    truth.cam["webcam"] = look_at(np.array([-50.0, 30.0, 160.0]), [320, 0, -190])
    # gripper camera on the hand, 30 mm below the hand's axis, looking forward and down (in the hand frame)
    eye = np.array([40.0, 0.0, -20.0])
    z = np.array([np.cos(np.radians(-40)), 0.0, np.sin(np.radians(-40))])
    x = np.array([0.0, -1.0, 0.0])
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    truth.cam["gripper"] = np.concatenate([mat_to_rotvec(R), -R @ eye])
    for t in tags:
        truth.tag_size[t] = 19.5 if t == 9 else 11.7 if t in (31, 32) else 20.0 if t in (40, 41) else 25.0 if t >= 70 else 30.0
    face_up = np.zeros(3)                                    # facing up: a tag's z axis points out of its face
    face_back = np.array([0.0, -np.pi / 2, 0.0])              # facing back along the hand (the finger tags)
    for t in table:
        a = np.radians(rng.uniform(-60, 60))
        r = rng.uniform(300, 480)
        truth.tag_frame[t] = 0
        truth.tag[t] = np.concatenate([face_up + rng.normal(0, 0.05, 3), [r * np.cos(a), r * np.sin(a), -330.0]])
    for t, (f, p) in {9: (G_IDX, [130.0, 0, 80.0]), 31: (G_IDX, [120.0, 25, -60]), 32: (G_IDX, [120.0, -25, -60]),
                      40: (3, [60.0, 25, 40]), 41: (3, [60.0, -25, 40]),
                      70: (1, [80.0, 0, 25]), 71: (1, [150.0, 0, 25]), 72: (2, [60.0, 0, 25]), 73: (2, [120.0, 0, 25])}.items():
        truth.tag_frame[t] = f
        truth.tag[t] = np.concatenate([(face_back if t in (31, 32) else face_up) + rng.normal(0, 0.1, 3), p])
    q = np.column_stack([rng.uniform(80, 150, 4 * n_poses), rng.uniform(40, 150, 4 * n_poses), rng.uniform(30, 160, 4 * n_poses)])
    q = q[truth.frames(q, np.zeros(len(q)))[1][:, 3, 2] > -280][:n_poses]     # hand kept above the table
    base = rng.uniform(-40, 40, n_poses)
    roll = np.where(rng.random(n_poses) < 0.8, rng.uniform(10, 100, n_poses), ROLL_REF) if rotate else np.full(n_poses, ROLL_REF)
    grips = np.where(rng.random(n_poses) < 0.8, rng.uniform(40, 150, n_poses), GRIP_SURVEYED) if grip else np.full(n_poses, GRIP_SURVEYED)
    q = np.column_stack([q, roll, grips])
    if grip:
        truth.fing = TRUE_FINGERS.copy()
        truth.grip_varies = truth.fing_on = True
    truth.dq = np.column_stack([rng.normal(0, play_deg, (n_poses, 3)),
                                rng.normal(0, play_deg, n_poses) if rotate else np.zeros(n_poses), rng.normal(0, 0.1, n_poses),
                                rng.normal(0, 1.0, n_poses) if grip else np.zeros(n_poses)])
    obs = []
    all_obs = {"pose": [], "cam": [], "tag": []}
    for i in range(n_poses):
        for ci, c in enumerate(truth.cams):
            for ti, t in enumerate(tags):
                all_obs["pose"].append(i)
                all_obs["cam"].append(ci)
                all_obs["tag"].append(ti)
    A = {k: np.array(v) for k, v in all_obs.items()}
    pix = truth.predict(A, q, base, truth.dq)
    # visibility: in the picture, in front of the camera, facing it (roughly), not too small
    for k in range(len(A["pose"])):
        c, t, i = truth.cams[A["cam"][k]], tags[A["tag"][k]], A["pose"][k]
        w_, h_ = (1920, 1080) if c == "webcam" else (1152, 864) if c == "gripper" else (1280, 960)
        p = pix[k]
        side = np.linalg.norm(p[0] - p[1])
        if np.all((p[:, 0] > 5) & (p[:, 0] < w_ - 5) & (p[:, 1] > 5) & (p[:, 1] < h_ - 5)) and 12 < side < 400 \
                and cross2(p[1] - p[0], p[3] - p[0]) > 0:            # corners clockwise = facing the camera
            obs.append((i, c, t, p + rng.normal(0, noise_px, (4, 2))))
    S = {"calib": calib, "sizes": dict(truth.tag_size), "q": q, "came": q, "base": base, "obs": obs}
    return truth, S


def self_test(rotate=False, grip=False):
    print("Synthetic self-test: make a survey from a known arm, cameras and tags, then solve it back."
          + (" With the wrist rotate turning." if rotate else "") + (" With the gripper opening and closing." if grip else ""))
    truth, S = synthetic(rotate=rotate, grip=grip)
    print(f"   {len(S['q'])} poses, {len(S['obs'])} tag sightings")
    S["obs_all"] = S["obs"]
    S["obs"] = [o for o in S["obs"] if not finger_moved(S, o)]
    pinned_before = dict(PINNED)
    PINNED.update({31: G_IDX, 32: G_IDX})
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_synthetic_model.json")
    tx = truth.tag[9][3:]
    rv = truth.cam["torre"]
    th1, k1, th2, k2, th3, k3, L1, L2 = truth.arm[:8]
    json.dump({"x_tuned": [k1 * 1.03, k2 * 0.97, k3 * 1.02, th2 + th1 + 2.0, th3 - 3.0, L1 - 8, L2 + 6, tx[0] + 5, tx[2] - 5,
                           *(rv[:3] + 0.03), *(rv[3:] + np.array([10.0, -8, 6]))],
               "ref_slider": truth.ref.tolist(), "upper_arm_level_deg": th1 + 4.0}, open(tmp, "w"))
    w = initialise(S, tmp)
    os.remove(tmp)
    PINNED.clear()
    PINNED.update(pinned_before)
    if grip:                                          # the same stages as main(): without the moved fingers first
        solve(w, S, fix_arm=True, fix_dq=True)
        solve(w, S)
        if not init_fingers(w, dict(S, obs=S["obs_all"]), set(range(len(S["q"])))):
            raise SystemExit("   FAILED: no first guess for the finger model")
        S["obs"] = S["obs_all"]
    O, res = solve(w, S)
    # Compare where the hand tag is as the tower camera sees it (the solve's own axes are only defined up to a
    # sideways shift of the arm's plane, which nothing physical depends on).
    def in_tower(world, dq):
        p = world.hand_point(S["q"], p_hand=world.tag[9][3:], dq=dq)
        return p @ rotvec_to_mat(world.cam["torre"][:3]).T + world.cam["torre"][3:]
    err = np.linalg.norm(in_tower(w, w.dq) - in_tower(truth, truth.dq), axis=1)
    print(f"   hand tag position vs the truth (in the tower camera's view): {np.median(err):.2f} mm median, {err.max():.2f} mm worst")
    err0 = np.linalg.norm(in_tower(w, None) - in_tower(truth, truth.dq), axis=1)
    print(f"   ...without the per-pose gear-play corrections (what a new pose gets): {np.median(err0):.2f} mm median")
    print("   arm:   " + "  ".join(f"{n} {a:.3f} (true {b:.3f})" for n, a, b in zip(ARM_NAMES, w.arm, truth.arm)))
    print(f"   base:  gain {w.base[0]:.4f} (true {truth.base[0]:.4f}), axis {np.round(w.base[1:], 1)} (true {truth.base[1:]})")
    dq_err = np.abs(w.dq - truth.dq)
    print(f"   gear play recovered to {np.median(dq_err[:, :5]):.2f} deg median"
          + (f", gripper slack to {np.median(dq_err[:, DQ_GRIP]):.2f} steps median (true spread {truth.dq[:, DQ_GRIP].std():.2f})"
             if grip else ""))
    if grip:                                          # each finger tag over every pose (its own gripper value), vs the truth,
        def finger_in_tower(world, t, dq):            # in the tower camera's view like the hand tag above
            p = world.tag[t][3:] + finger_offset(np.tile(world.finger_params(t), (len(S["q"]), 1)), world.grip_d(S["q"], dq))
            Rf, of = world.frames(S["q"], np.zeros(len(S["q"])), dq)
            pT = np.einsum("nij,nj->ni", Rf[:, G_IDX], p) + of[:, G_IDX]
            return pT @ rotvec_to_mat(world.cam["torre"][:3]).T + world.cam["torre"][3:]
        for f, t in ((0, 31), (1, 32)):
            e = np.linalg.norm(finger_in_tower(w, t, w.dq) - finger_in_tower(truth, t, truth.dq), axis=1)
            travel = [np.linalg.norm(finger_offset(m.finger_params(t), np.array([150.0 - GRIP_SURVEYED]))[0]
                                     - finger_offset(m.finger_params(t), np.array([40.0 - GRIP_SURVEYED]))[0]) for m in (w, truth)]
            print(f"   finger {t}: position vs the truth {np.median(e):.2f} mm median, {e.max():.2f} worst; travel 40->150 "
                  f"{travel[0]:.2f} mm (true {travel[1]:.2f});  radius {w.fing[2 + 4 * f]:.1f} (true {truth.fing[2 + 4 * f]:.1f}), "
                  f"k {w.fing[4 + 4 * f]:+.3f} (true {truth.fing[4 + 4 * f]:+.3f})")
    return float(np.median(err))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("survey", nargs="*", help="survey folder(s) made by  python3 arm.py survey  (several are solved together)")
    ap.add_argument("--model", help="self_model_pi.json, for the first guess of the arm and the tower camera")
    ap.add_argument("--holdout", type=float, default=0.0, help="share of poses kept back to test on (e.g. 0.2)")
    ap.add_argument("--static", type=int, nargs="*", default=[], help="tags normally treated as moving that stayed put (e.g. 7)")
    ap.add_argument("--ignore", type=int, nargs="*", default=[61], help="tags to leave out entirely (default 61: two "
                    "different physical tags share that number, a small one on the hand and one on the couch armrest)")
    ap.add_argument("--out", default=None, help="where to save world.json (default: in the survey folder)")
    ap.add_argument("--synthetic", action="store_true", help="self-test on made-up data")
    ap.add_argument("--slow", action="store_true", help="the old way: scipy estimates the Jacobian (much slower)")
    ap.add_argument("--moved", default=None, metavar="FOLDER",
                    help="the gripper camera got knocked before this survey folder: it and the later folders get their "
                         "own gripper camera position (gripper2)")
    ap.add_argument("--moved-tags", type=int, nargs="*", default=[31, 32],
                    help="with --moved: tags that may have moved too and get their own position after (default 31 32)")
    ap.add_argument("--no-fingers", action="store_true",
                    help="no finger model: use the finger tags only at gripper 80 (the old way)")
    ap.add_argument("--rotate", choices=("+", "-", "auto"), default="auto",
                    help="which way the wrist rotate turns. When the quick test cannot tell, the solve settles "
                         "the cameras BOTH ways and keeps the better, which doubles the first stage -- 11 min on "
                         "419 poses. Every solve that could decide chose +. Give it here to skip that")
    ap.add_argument("--on", nargs="*", default=[], metavar="TAG=PART",
                    help="pin tags to parts you know, e.g.  --on 322=G 360=H  (W table, U upper arm, F forearm, "
                         "H wrist block, G gripper (turns with rotate), T tower)")
    a = ap.parse_args()
    global FAST, FINGER_MODEL
    FAST = not a.slow
    FINGER_MODEL = not a.no_fingers
    for item in a.on:
        t, _, part = item.partition("=")
        if part.upper() not in FRAMES:
            ap.error(f"--on {item}: the part must be one of {', '.join(FRAMES)}")
        PINNED[int(t)] = FRAMES.index(part.upper())
    if PINNED:
        print("   pinned: " + ", ".join(f"tag {t} on {FRAMES[f]}" for t, f in PINNED.items()))
    if FINGER_MODEL:                                  # the finger tags ride on the gripper (the finger model moves them)
        for t in (31, 32):
            PINNED.setdefault(t, G_IDX)
    if a.synthetic:
        self_test()
        self_test(rotate=True)
        if FINGER_MODEL:
            self_test(rotate=True, grip=True)
        return
    if not a.survey or not a.model:
        ap.error("give the survey folder and --model")
    global PROGRESS
    PROGRESS = Progress()
    import builtins
    _print = builtins.print

    def print_above(*args, **kw):                     # normal messages go above the live bar
        PROGRESS.clear()
        _print(*args, **kw)
        if PROGRESS.stage and PROGRESS.on:
            PROGRESS.draw(force=True)
    builtins.print = print_above
    moved = None
    if a.moved:
        names = [os.path.basename(os.path.normpath(f)) for f in a.survey]
        key = os.path.basename(os.path.normpath(a.moved))
        if key not in names:
            ap.error(f"--moved {a.moved}: not one of the survey folders given")
        moved = names.index(key)
        print(f"   gripper camera: own position from {key} on (gripper2)")
    S = load_surveys(a.survey, set(a.static), moved, set(a.moved_tags) if moved is not None else ())
    if moved is not None and a.moved_tags:
        print(f"   ...and tags {a.moved_tags} (after the move they show as {AFTER}+id)")
        for t in a.moved_tags:
            if t in PINNED:
                PINNED[AFTER + t] = PINNED[t]
    if a.ignore:
        S["obs"] = [o for o in S["obs"] if o[2] not in set(a.ignore)]
        print(f"   left out tags {sorted(a.ignore)}")
    print(f"Survey {' + '.join(a.survey)}: {len(S['q'])} poses, {len(S['obs'])} tag sightings, cameras {sorted(S['calib'])}")
    S["obs_all"] = S["obs"]                           # finger tags at other gripper values wait for the finger model:
    S["obs"] = [o for o in S["obs"] if not finger_moved(S, o)]      # first solve the way it always was
    if len(S["obs"]) < len(S["obs_all"]):
        print(f"   finger tags at other gripper values: {len(S['obs_all']) - len(S['obs'])} sightings, "
              "used once the finger model has a first guess")
    w = initialise(S, a.model)
    rng = np.random.default_rng(1)
    poses = np.arange(len(S["q"]))
    test = set(rng.choice(poses, int(len(poses) * a.holdout), replace=False).tolist()) if a.holdout > 0 else set()
    train = set(poses.tolist()) - test
    print("Solving (arm fixed first, to settle the cameras and tags) ...")
    if w.rot_varies and a.rotate in ("+", "-"):
        sign = 1.0 if a.rotate == "+" else -1.0
        print(f"   wrist rotate direction: {a.rotate} (given on the command line, not worked out)")
        w = initialise(S, a.model, verbose=False, rot_sign=sign)
        solve(w, S, train, fix_arm=True, fix_dq=True)
    elif w.rot_varies and w.rot_ambiguous:
        print("   the rotate direction was too close to call: settling the cameras both ways and keeping the better one")
        tries = []
        for sign in (1.0, -1.0):
            ws = initialise(S, a.model, verbose=False, rot_sign=sign)
            Os, rs = solve(ws, S, train, fix_arm=True, fix_dq=True)
            e = rms(rs.fun[:8 * len(Os["pose"])])
            print(f"      rotate {'+' if sign > 0 else '-'}: {e:.2f} px")
            tries.append((e, sign, ws))
        _, sign, w = min(tries, key=lambda t: t[0])
        print(f"   wrist rotate direction: {'+' if sign > 0 else '-'}")
    else:
        solve(w, S, train, fix_arm=True, fix_dq=True)
    reassign_tags(w, S, train)
    if prune_tags(w, S, train, limit_px=100.0):                     # the hopeless ones first...
        solve(w, S, train, fix_arm=True, fix_dq=True)
    reassign_tags(w, S, train, verbose=False)
    if prune_tags(w, S, train, limit_px=25.0):                      # ...then any that still fit nowhere, before the
        solve(w, S, train, fix_arm=True, fix_dq=True)               # arm is freed (they can make it run away)
    if w.rot_varies:                                  # hand vs gripper: with rotate data the two finally differ, so check
        hand = [t for t in w.tags if w.tag_frame[t] in (H_IDX, G_IDX)]   # every tag there (first guesses could not tell)
        if reassign_tags(w, S, train, limit_px=2.0, only=hand):
            solve(w, S, train, fix_arm=True, fix_dq=True)
    print("Solving everything ...")
    O, _ = solve(w, S, train)
    if w.rot_varies:
        hand = [t for t in w.tags if w.tag_frame[t] in (H_IDX, G_IDX)]
        if reassign_tags(w, S, train, limit_px=2.0, only=hand):
            print("Solving everything again with those moved ...")
            O, _ = solve(w, S, train)
    if FINGER_MODEL and w.grip_varies and len(S["obs_all"]) > len(S["obs"]):
        if init_fingers(w, dict(S, obs=S["obs_all"]), train):
            S["obs"] = S["obs_all"]
            print("Solving everything with the finger model ...")
            O, _ = solve(w, S, train)
    reassign_tags(w, S, train, keep=(9, 31, 32, AFTER + 31, AFTER + 32))
    if prune_tags(w, S, train, limit_px=25.0):
        print("Solving everything again without them ...")
        O, _ = solve(w, S, train)
    PROGRESS.done()
    rep = report(w, S, O)
    if test:
        h = holdout_error(w, S, test)
        print(f"   held-out poses ({len(test)}): corner error {h['median_px']:.2f} px median, {h['p90_px']:.2f} px 90th percentile")
        print("      per camera: " + "   ".join(f"{c} {v:.2f}" for c, v in h["cameras"].items())
              + (f"   |  finger tags {h['fingers_px']:.2f}, everything else {h['without_fingers_px']:.2f} px"
                 if "fingers_px" in h else ""))
        rep["holdout"] = h
    out = a.out or os.path.join(a.survey[0], "world.json")
    data = w.to_json()
    data["report"] = rep
    data["survey"] = [os.path.abspath(f) for f in a.survey]
    data["poses"] = {                                # per pose, for training the slack model: what was commanded, where
        "joints": np.round(S["q"], 2).tolist(),     # each joint came from, and the slack the solve found (deg / steps)
        "came_from": np.round(S["came"], 2).tolist(), "base": np.round(S["base"], 2).tolist(),
        "survey": S["sv"].tolist(), "slack": np.round(w.dq, 4).tolist(), "held_out": sorted(int(i) for i in test),
        "joint_columns": ["shoulder", "elbow", "wrist", "rotate", "gripper"],
        "slack_columns": ["shoulder", "elbow", "wrist", "rotate", "base", "gripper"]}
    data["made"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(data, open(out, "w"), indent=1)
    print(f"Saved {out}")
    print("   arm: " + "  ".join(f"{n} {v:.3f}" for n, v in zip(ARM_NAMES, w.arm)))
    if len(w.boff) > 1:
        print("   base zero per survey vs the first (re-homed in between): " + ", ".join(f"{v:+.2f}" for v in w.boff) + " deg")
    for c, v in w.fs.items():
        print(f"   {c} camera focal length: {v:.4f} x its lens calibration ({(v - 1) * 100:+.2f}%)")
    play = np.abs(w.dq).mean(0)
    print(f"   gear play (mean per-pose correction): shoulder {play[0]:.2f}  elbow {play[1]:.2f}  wrist {play[2]:.2f}  "
          + (f"rotate {play[DQ_ROT]:.2f}  " if w.rot_varies else "")
          + (f"gripper {play[DQ_GRIP]:.2f}  " if w.fing_on else "")
          + f"base {play[DQ_BASE]:.2f} deg (base: 90% under {np.percentile(np.abs(w.dq[:, DQ_BASE]), 90):.2f})")
    if w.fing_on:
        for f, side in enumerate(("left (31)", "right (32)")):
            r, phi0, k, k2 = w.fing[2 + 4 * f: 6 + 4 * f]
            span = [finger_offset(np.concatenate([w.fing[:2], w.fing[2 + 4 * f: 6 + 4 * f]]), np.array([g - GRIP_SURVEYED]))[0]
                    for g in (40.0, 150.0)]
            print(f"   finger {side}: arc radius {r:.1f} mm, {k:+.3f} deg per step ({k2:+.5f} per step^2); "
                  f"from gripper 40 to 150 it moves {np.linalg.norm(span[1] - span[0]):.1f} mm")
        n, _, _ = finger_plane(w.fing[0], w.fing[1])
        print(f"   fingers swing in a plane tilted {np.degrees(np.arccos(n[0, 2])):.1f} deg from the gripper's x-y plane")
    if w.rot_varies:
        n, pa = w.rotate_axis()
        d_lo, d_hi = 0.0 - ROLL_REF, 112.0 - ROLL_REF
        print(f"   wrist rotate: {w.arm[9]:.3f} real deg per step at level, {w.arm[9] + 2 * w.arm[14] * d_lo:.3f} at 0, "
              f"{w.arm[9] + 2 * w.arm[14] * d_hi:.3f} at 112; axis tilted {np.degrees(np.arccos(n[0])):.1f} deg from the "
              f"hand's line ({np.degrees(np.arctan(w.arm[11])):+.1f} up/down, {np.degrees(np.arctan(w.arm[10])):+.1f} sideways), "
              f"{np.hypot(pa[1], pa[2]):.1f} mm off it")


if __name__ == "__main__":
    sys.exit(main())
