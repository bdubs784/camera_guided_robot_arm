"""
Is the arm's slack learnable? Reads the per-pose slack the big solve found (world json "poses") and tries to predict
it from the commanded joints, the base angle and which direction each joint came from. Every survey is predicted by
models trained only on the OTHER surveys (no cheating by memorising a session).

    pip install numpy scikit-learn
    python slack_check.py solve14_world.json

Prints, per model and joint: the prediction error (rmse) and the share of the slack it explains. "none" is today
(no correction). Anything clearly above 0% means a model can take that much slack out before the cameras even look.
"""
import json
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "solve14_world.json"))["poses"]
J, C, B = np.array(d["joints"]), np.array(d["came_from"]), np.array(d["base"])
w = min(J.shape[1], C.shape[1])                                  # v2 surveys also record rotate and gripper: compare like
J, C = J[:, :w], C[:, :w]                                        #   with like (older solves saved 4 joint columns)
sv, Y, held = np.array(d["survey"]), np.array(d["slack"]), set(d["held_out"])
keep = np.array([i not in held for i in range(len(J))])          # held-out poses have no solved slack
J, C, B, sv, Y = J[keep], C[keep], B[keep], sv[keep], Y[keep]
group = np.where(sv == 1, 0, sv)                                  # surveys 0 and 1 repeated the same poses: one group
if len(set(group)) < 2:                                          # (that was solve14's layout. The 623-pose world
    group = sv                                                   #  has just surveys 0 and 1 -- merged, there was nothing
                                                                 #  left to train on when testing either)
direction = np.sign(np.round(J - C, 1))                          # +1 arrived going up, -1 going down, 0 did not move
X = np.column_stack([J, B, direction])
NAMES = d.get("slack_columns", ["shoulder", "elbow", "wrist", "rotate", "base"])[:Y.shape[1]]
COLS = [j for j in range(len(NAMES)) if Y[:, j].std() > 1e-6]  # skip a slack the solve did not use (no rotate/gripper data)
JOINT_COL = {"shoulder": 0, "elbow": 1, "wrist": 2, "rotate": 3, "gripper": 4}   # its commanded joint, for "came from"

MODELS = {
    "none (today)": None,
    "average": "mean",
    "linear": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "boosted trees": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                                           min_samples_leaf=10),
    "gaussian process": lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
        ConstantKernel(1.0) * RBF(np.ones(X.shape[1])) + WhiteKernel(0.5), normalize_y=True)),
    "neural net": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 32), alpha=1e-2, max_iter=3000,
                                                                       early_stopping=True, random_state=0)),
}

print(f"{len(Y)} poses from {len(set(group))} surveys. Typical slack (1 sigma): "
      + "  ".join(f"{NAMES[j]} {Y[:, j].std():.2f}" for j in COLS) + "  (steps; base in deg)")
print("Each survey predicted by a model trained on the others.  rmse (share of slack explained):\n")
def bar(name, done, total, t0, width=30):
    n = int(width * done / total)
    sys.stdout.write(f"\r  {name:17s} [{'#' * n}{'.' * (width - n)}] {100 * done / total:3.0f}%  {time.time() - t0:4.0f} s")
    sys.stdout.flush()


groups = sorted(set(group))
for name, make in MODELS.items():
    t0 = time.time()
    cells = []
    total, done = len(COLS) * len(groups), 0
    for j in COLS:
        pred = np.zeros(len(Y))
        for g in groups:
            bar(name, done, total, t0)
            done += 1
            te, tr = group == g, group != g
            if make is None:
                p = np.zeros(te.sum())
            elif make == "mean":
                p = np.full(te.sum(), Y[tr, j].mean())
            else:
                m = make()
                m.fit(X[tr], Y[tr, j])
                p = m.predict(X[te])
            pred[te] = p
        e = pred - Y[:, j]
        cells.append(f"{NAMES[j][:5]:5s} {np.sqrt(np.mean(e ** 2)):4.2f} ({100 * (1 - np.mean(e ** 2) / np.var(Y[:, j])):+4.0f}%)")
    sys.stdout.write("\r" + " " * 70 + "\r")
    print(f"  {name:17s} " + "   ".join(cells) + f"   [{time.time() - t0:.0f} s]")

# what direction alone says: average slack when a joint arrived going up vs going down
print("\nAverage slack by the direction each joint arrived from (the gear-play signature):")
for j in COLS:
    c = JOINT_COL.get(NAMES[j])
    if c is None or c >= direction.shape[1]:
        continue
    up, dn = Y[direction[:, c] > 0, j], Y[direction[:, c] < 0, j]
    if len(up) > 5 and len(dn) > 5:
        print(f"  {NAMES[j]:8s} arrived going up {up.mean():+.2f} ({len(up)})   going down {dn.mean():+.2f} ({len(dn)})"
              f"   gap {up.mean() - dn.mean():+.2f}")
