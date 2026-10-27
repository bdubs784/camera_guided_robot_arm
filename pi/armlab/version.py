"""
What code and data a run was made with, so offline experiments know where a run came from. Every .py in armlab
hashed (md5, 10 hex), one "build" hash over all of them -- two runs with the same
build ran the same code -- and the data files the planner and the checks read.
"""
import hashlib
import os

from .state import S

DATA_FILES = ("world.json", "table_model.json", "arm_map.npz", "home_pose.json", "base_home.json", "plan_table.json")


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:10]


def stamp():
    """{"build": ..., "code": {file: md5}, "data": {file: md5 or None}}."""
    root = os.path.dirname(os.path.abspath(__file__))
    code = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            if f.endswith(".py"):
                p = os.path.join(d, f)
                code[os.path.relpath(p, root).replace(os.sep, "/")] = _md5(p)
    build = hashlib.md5("".join(f"{k}:{v};" for k, v in sorted(code.items())).encode()).hexdigest()[:10]
    data = {}
    for f in DATA_FILES:
        p = os.path.join(S.data_dir, f)
        data[f] = _md5(p) if os.path.exists(p) else None
    return {"build": build, "code": code, "data": data}
