#!/usr/bin/env python3
"""
Train the gear-slack model (a Gaussian process per joint) on a world's solved survey poses and export it as plain
numbers the Pi can evaluate with numpy alone (its scikit-learn/scipy are not usable).

    python tools/train_slack_gp.py <world.json> <out slack_gp.json>

Features and kernel as tools/slack_check.py: the commanded joints (shoulder, elbow, wrist, rotate, gripper), the base
angle, and the direction each joint arrived from (+1 up, -1 down, 0 did not move); ConstantKernel * RBF (one length
scale per feature) + WhiteKernel, inputs standardised, targets normalised. Predicting = k(x, X_train) @ alpha. The
file is stamped with the world's md5: armlab/slackgp.py ignores a model made for another world.

Validation: on an earlier world (6 surveys, leave-one-survey-out) a GP explained shoulder 62%, elbow 73%, wrist 46% of the slack.
This world's poses are two surveys of the same poses, so there is no honest held-out split here; the shows are the
test -- every hover predicts the slack before the cameras measure it (the demo page keeps the running score).
"""
import hashlib
import json
import sys
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

JOINTS = ("shoulder", "elbow", "wrist", "rotate")


def features(J, C, B):
    """[commanded joints (5), base, direction each joint arrived from (5)] -- slack_check's X."""
    J, C = np.atleast_2d(J), np.atleast_2d(C)
    return np.column_stack([J, np.asarray(B, float).reshape(-1), np.sign(np.round(J - C, 1))])


def main():
    path, out = sys.argv[1], sys.argv[2]
    raw = open(path, "rb").read()
    d = json.loads(raw)["poses"]
    J, C, B = np.array(d["joints"], float), np.array(d["came_from"], float), np.array(d["base"], float)
    Y, held = np.array(d["slack"], float), set(d["held_out"])
    keep = np.array([i not in held for i in range(len(J))])
    X, Y = features(J[keep], C[keep], B[keep]), Y[keep]
    cols = list(d["slack_columns"])
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    model = {"world_md5": hashlib.md5(raw).hexdigest(), "made": time.strftime("%Y-%m-%d %H:%M"), "n": int(len(X)),
             "x_mean": sc.mean_.tolist(), "x_scale": sc.scale_.tolist(), "X": np.round(Xs, 6).tolist(), "joints": {}}
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(X))
    for name in JOINTS:
        y = Y[:, cols.index(name)]
        kern = ConstantKernel(1.0) * RBF(np.ones(X.shape[1])) + WhiteKernel(0.5)
        gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=1, random_state=0).fit(Xs, y)
        k = gp.kernel_
        model["joints"][name] = {"alpha": np.round(gp.alpha_, 8).tolist(), "y_mean": float(gp._y_train_mean),
                                 "y_std": float(gp._y_train_std), "c": float(k.k1.k1.constant_value),
                                 "ls": np.atleast_1d(k.k1.k2.length_scale).tolist(), "noise": float(k.k2.noise_level)}
        pred = np.zeros(len(y))                          # random 5-fold: optimistic (same poses twice), a sanity check
        for f in range(5):
            m = GaussianProcessRegressor(gp.kernel_, normalize_y=True, optimizer=None).fit(Xs[fold != f], y[fold != f])
            pred[fold == f] = m.predict(Xs[fold == f])
        print(f"  {name:8s} sd {y.std():.2f} steps, length scales {np.round(model['joints'][name]['ls'], 1).tolist()}, "
              f"random 5-fold explained {100 * (1 - np.mean((pred - y) ** 2) / np.var(y)):.0f}% (optimistic)")
    json.dump(model, open(out, "w"))
    print(f"wrote {out}: {len(X)} poses, world {model['world_md5'][:8]}")


if __name__ == "__main__":
    main()
