"""
The arm filter (a Kalman filter): what the arm's joints REALLY are right now, from what they were told (ESP32
telemetry) plus what the three cameras see (every tag on the arm, and the table tags for the base).

The model (world.json) already knows the arm's shape exactly. What it cannot know in advance is each joint's slack:
gear play, droop, the rotate servo's dead band, the base's worm-gear give. The big solve measured that as about
1 deg on the shoulder/elbow/wrist, ~0.7 steps on the rotate and ~0.25 deg on the base, different after every move.
So the filter's state is just those offsets (true = commanded + offset), in the solver's order:

    shoulder, elbow, wrist (slider steps), rotate (slider steps), base (deg), gripper (slider steps)

The gripper's slack (~2 steps in survey v2) only shows through the finger tags, so it needs a finger model in
world.json; without one it simply stays at zero.

    predict: offsets drift a little while a joint holds still, and can jump (re-settle in the slack) when it moves
    update:  every camera frame: where its tags should be with the current offsets vs where they are; nudge the
             offsets to agree (a sighting far off is ignored as a misread)

    f = ArmFilter(world)
    f.step(t, joints, base, moving)                 every telemetry reading
    f.see(camera, tags, joints, base)               every camera frame (tags: [(id, corners 4x2 or None, centre)])
    f.offsets(), f.sigma()                          the estimate and how sure it is (1 sigma)
    world.tip(joints, base, dq=f.offsets())         the fingertip with the slack taken out
"""
import numpy as np

NAMES = ("shoulder", "elbow", "wrist", "rotate", "base", "gripper")
N = len(NAMES)
START_SIGMA = np.array([2.0, 2.0, 2.0, 2.0, 1.0, 2.0])     # before any camera has seen anything
DRIFT = np.array([0.05, 0.05, 0.05, 0.05, 0.02, 0.05])      # per second while holding still (1 sigma)
JUMP = np.array([1.0, 1.4, 0.9, 1.0, 0.4, 2.0])            # a move can re-settle the slack by about this much
CORNER_PX = {"torre": 2.0, "webcam": 2.0, "gripper": 3.0}     # how well a camera's tag corners fit (1 sigma, px)
CENTRE_PX = 3.0                                      # tags the camera server reports as a centre only
GATE_PX = 25.0                                       # a sighting further off than this is a misread: skipped
FINGER_TAGS = {31, 32}                               # on the fingers: they move when the gripper opens or closes...
GRIP_SURVEYED = 80.0                                 # ...without a finger model in world.json they are only used here
GRIP_TOL = 1.5


class ArmFilter:
    def __init__(self, world, start=None):
        self.w = world
        self.x = np.zeros(N)
        if start is not None:
            self.x[:len(start)] = np.asarray(start, float)
        self.P = np.diag(START_SIGMA ** 2)
        self.t = None
        self.was_moving = np.zeros(N, bool)
        self.skipped = 0
        self.used = 0

    def offsets(self):
        return self.x.copy()

    def sigma(self):
        return np.sqrt(np.diag(self.P))

    # ---------------------------------------------------------------- predict
    def step(self, t, moving):
        """Time passes. moving: flags (shoulder, elbow, wrist, rotate, base, gripper) from the telemetry; a shorter list
        (the old 5) leaves the rest as not moving."""
        moving = np.concatenate([np.asarray(moving, bool), np.zeros(N, bool)])[:N]
        seen = np.ones(N, bool)
        seen[5] = bool(self.w.fingers)                      # no finger model: nothing measures the gripper's slack,
        if self.t is not None:                              #   so its uncertainty is left alone instead of growing
            dt = max(0.0, t - self.t)
            self.P += np.diag(np.where(seen, DRIFT ** 2 * dt, 0.0))
            stopped = self.was_moving & ~moving & seen       # a joint that just finished a move: slack re-settled
            if stopped.any():
                self.P += np.diag(np.where(stopped, JUMP ** 2, 0.0))
        self.t = t
        self.was_moving = moving

    # ---------------------------------------------------------------- update
    def _measure(self, cam, tags, joints, base, x, grip=None):
        out = []
        for tid, corners, centre in tags:
            if tid not in self.w.tags:
                continue
            if corners is not None:
                out.append(self.w.predict_tag(tid, cam, joints, base, grip=grip, dq=x).ravel())
            else:
                c = self.w.tag_corners_table(tid, joints, base, grip=grip, dq=x).mean(0)
                out.append(self.w.project(cam, c, joints, base, dq=x).ravel())
        return out

    def see(self, cam, tags, joints, base, grip=None):
        """One camera frame. tags: [(id, corners (4x2) or None, centre (2,))]. joints: the 4 commanded joint values
        when the picture was taken, base: the base count then, grip: the gripper value then (with a finger model in
        world.json the finger tags are used at any gripper value; without one only at the value the surveys had).
        Returns how many sightings were used."""
        if cam not in self.w.cams:
            return 0
        if self.w.fingers and grip is not None:
            fingers_ok = True
        else:
            fingers_ok = grip is None or abs(grip - GRIP_SURVEYED) <= GRIP_TOL
        tags = [(int(t), None if c is None else np.asarray(c, float), None if m is None else np.asarray(m, float))
                for t, c, m in tags if int(t) in self.w.tags and (fingers_ok or int(t) not in FINGER_TAGS)]
        if not tags:
            return 0
        pred = self._measure(cam, tags, joints, base, self.x, grip)
        z, h, sig, keep = [], [], [], []
        for k, ((tid, corners, centre), p) in enumerate(zip(tags, pred)):
            obs = corners.ravel() if corners is not None else centre.ravel()
            err = np.linalg.norm((obs - p).reshape(-1, 2), axis=1).mean()
            if not np.isfinite(err) or err > GATE_PX:
                self.skipped += 1
                continue
            keep.append(k)
            z.append(obs)
            h.append(p)
            sig.append(np.full(len(obs), CORNER_PX.get(cam, 3.0) if corners is not None else CENTRE_PX))
        if not keep:
            return 0
        sub = [tags[k] for k in keep]
        z, h, sig = np.concatenate(z), np.concatenate(h), np.concatenate(sig)
        H = np.zeros((len(z), N))
        for j in range(N):                                   # how each offset moves every predicted pixel
            dx = np.zeros(N)
            dx[j] = 1e-3
            hp = np.concatenate(self._measure(cam, sub, joints, base, self.x + dx, grip))
            H[:, j] = (hp - h) / 1e-3
        R = np.diag(sig ** 2)
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ (z - h)
        self.P = (np.eye(N) - K @ H) @ self.P
        self.P = (self.P + self.P.T) / 2
        self.used += len(keep)
        return len(keep)
