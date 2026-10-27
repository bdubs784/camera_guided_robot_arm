"""
Tests for the parts that are pure maths: no arm, no cameras. Run on any computer:  python3 -m pytest pi/tests
"""
import os

import pytest

from armlab.config import FINGER_TAGS
from armlab.motion import auto_clamp
from armlab.safety import off_path
from armlab.state import S

HERE = os.path.dirname(os.path.abspath(__file__))
L, R = FINGER_TAGS


def test_off_path():
    assert off_path([5, 5, 0], [0, 0, 0], [10, 0, 0]) == pytest.approx(5.0)
    assert off_path([20, 0, 0], [0, 0, 0], [10, 0, 0]) == pytest.approx(10.0)    # past the end: distance to the end


def test_auto_clamp_uses_reported_limits():
    old = S.auto_hi
    try:
        S.auto_hi = (150.0, 159.0, 169.0)               # a trimmed shoulder reports a lower top
        assert auto_clamp([155.0, 0.0, 200.0]) == [150.0, 1.0, 169.0]
    finally:
        S.auto_hi = old


# ---------------- the reach map ----------------
def test_every_mode_names_only_things_that_exist():
    """Each mode module's functions only call names that are defined or imported (once an edit deleted
    basehome's go_to_homing_pose and nothing noticed until it ran on the arm)."""
    import ast
    import builtins
    import glob
    import os
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "armlab")
    bad = []
    for path in glob.glob(os.path.join(here, "**", "*.py"), recursive=True):
        tree = ast.parse(open(path).read())
        known = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}     # every module has these
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                known.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                known.add(node.name)
                known.update(a.arg for a in node.args.args + node.args.kwonlyargs) if hasattr(node, "args") else None
                if hasattr(node, "args"):
                    if node.args.vararg:
                        known.add(node.args.vararg.arg)
                    if node.args.kwarg:
                        known.add(node.args.kwarg.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                known.add(node.id)
            elif isinstance(node, (ast.For, ast.comprehension)):
                for n in ast.walk(node.target):
                    if isinstance(n, ast.Name):
                        known.add(n.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                known.add(node.name)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        for n in ast.walk(item.optional_vars):
                            if isinstance(n, ast.Name):
                                known.add(n.id)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                known.update(node.names)
            elif isinstance(node, ast.Lambda):
                known.update(a.arg for a in node.args.args)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in known:
                bad.append(f"{os.path.relpath(path, here)}:{node.lineno} {node.id}")
    assert not bad, "undefined names: " + ", ".join(bad)


def test_hard_coded_poses_are_inside_the_automatic_limits():
    """Any pose written into the code must be one the arm may actually be sent to.

    Reseating the shoulder horn moved the shoulder about +90 slider steps, so every number written before it points the
    arm roughly 40 degrees higher than whoever wrote it meant. auto_clamp catches that on the way out, and the
    safety map refuses it for checked moves, but a clamped pose is a quiet wrong answer rather than a loud one.
    This makes a stale constant fail here instead.
    """
    from armlab.config import AUTO_HI_DEFAULT, AUTO_LO, HOME_POSE, HOMING_STAGE
    from armlab.modes.grasp import HIGH_POSE, SEARCH_POSE

    for name, pose in (("HOME_POSE", HOME_POSE), ("HOMING_STAGE", HOMING_STAGE),
                       ("HIGH_POSE", HIGH_POSE), ("SEARCH_POSE", SEARCH_POSE)):
        for j, v in enumerate(pose[:3]):
            assert AUTO_LO[j] <= v <= AUTO_HI_DEFAULT[j], (
                f"{name}[{j}] is {v}, outside AUTO_LO/AUTO_HI_DEFAULT ({AUTO_LO[j]}..{AUTO_HI_DEFAULT[j]}). "
                "A pose from before a mapping change?")
