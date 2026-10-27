"""
The gear-slack model, evaluated with numpy alone: a Gaussian process per joint trained on the laptop
(tools/train_slack_gp.py) from the big solve's per-pose slack, exported as numbers to <data dir>/slack_gp.json.

    predict(joints5, base, came_from5) -> {"shoulder": steps, "elbow": ..., "wrist": ..., "rotate": ...} or None

Features as the training: the commanded joints, the base angle, and the direction each joint arrived from. The grab
asks it at the hover, BEFORE the cameras measure the slack there, so every round is a held-out test of the model
(the demo page keeps the running score). Nothing here moves the arm or changes a plan.
"""
import hashlib
import json
import os

import numpy as np

from .state import S

FILE = "slack_gp.json"
_M = {}


def load():
    """The model for this world, or None (no file, or made for another world.json). Cached."""
    if "m" in _M:
        return _M["m"]
    m = None
    try:
        path = os.path.join(S.data_dir, FILE)
        world = hashlib.md5(open(os.path.join(S.data_dir, "world.json"), "rb").read()).hexdigest()
        d = json.load(open(path))
        if d.get("world_md5") == world:
            d["_X"] = np.asarray(d["X"], float)
            d["_mu"], d["_sd"] = np.asarray(d["x_mean"], float), np.asarray(d["x_scale"], float)
            for j in d["joints"].values():
                j["_alpha"], j["_ls"] = np.asarray(j["alpha"], float), np.asarray(j["ls"], float)
            m = d
    except (OSError, ValueError, KeyError):
        m = None
    _M["m"] = m
    return m


def features(joints5, base, came_from5):
    j = np.asarray(joints5, float)[:5]
    c = np.asarray([j[i] if v is None else v for i, v in enumerate(list(came_from5)[:5])], float)
    return np.concatenate([j, [float(base)], np.sign(np.round(j - c, 1))])


def predict(joints5, base, came_from5, model=None):
    """Predicted slack (slider steps, real minus commanded) per joint, or None without a model."""
    m = model or load()
    if m is None:
        return None
    x = (features(joints5, base, came_from5) - m["_mu"]) / m["_sd"]
    out = {}
    for name, j in m["joints"].items():
        d = (m["_X"] - x) / j["_ls"]
        k = j["c"] * np.exp(-0.5 * np.sum(d * d, axis=1))
        out[name] = float(k @ j["_alpha"] * j["y_std"] + j["y_mean"])
    return out
