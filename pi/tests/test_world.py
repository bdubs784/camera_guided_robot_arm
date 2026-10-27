"""World model (world.py): self-consistency on a small made-up world (the solver cross-check lives in tools/)."""
import json
import os
import tempfile

import numpy as np

from armlab.world import World, tag_corners


def make_world():
    return World(make_world_path())


def make_world_path():
    d = {"arm": {"th1": -3.0, "k1": 0.47, "th2": 4.4, "k2": -1.0, "th3": -5.0, "k3": 1.03, "L1": 127.0, "L2": 92.0,
                 "th4": 0.0, "k4": 0.55, "ax_u": 0.0, "ax_v": -0.02, "ax_y": 0.0, "ax_z": 4.0, "k4b": -0.0005},
         "ref_slider": [123.0, 54.0, 68.0, 42.0], "base": {"gain": -1.0, "axis_xy_mm": [-30.0, -20.0]},
         "cameras": {"torre": {"parent": "T", "rvec": [1.7, -1.7, 0.76], "t_mm": [-13.0, 194.0, 109.0],
                               "camera_matrix": [[1137, 0, 645], [0, 1138, 494], [0, 0, 1]], "dist_coeffs": [0, 0, 0, 0, 0]},
                     "gripper": {"parent": "G", "rvec": [1.67, -1.65, 0.88], "t_mm": [-16.0, 117.0, -25.0],
                                 "camera_matrix": [[1196, 0, 555], [0, 1193, 410], [0, 0, 1]], "dist_coeffs": [0, 0, 0, 0, 0]}},
         "tags": {"9": {"on": "G", "size_mm": 19.5, "rvec": [1.4, -0.07, -2.8], "t_mm": [99.0, -10.0, 113.0]},
                  "31": {"on": "G", "size_mm": 11.7, "rvec": [0.7, -0.5, -1.5], "t_mm": [187.0, 20.0, 34.0]},
                  "32": {"on": "G", "size_mm": 11.7, "rvec": [-0.1, -0.8, -0.1], "t_mm": [190.0, -58.0, 37.0]},
                  "120": {"on": "W", "size_mm": 30.0, "rvec": [0.0, 0.0, -2.2], "t_mm": [358.0, -8.0, -247.0]},
                  "121": {"on": "W", "size_mm": 30.0, "rvec": [0.0, 0.0, -2.2], "t_mm": [384.0, 124.0, -245.0]},
                  "100": {"on": "W", "size_mm": 30.0, "rvec": [0.0, 0.0, -2.3], "t_mm": [94.0, -411.0, -254.0]}}}
    path = os.path.join(tempfile.mkdtemp(), "world.json")
    json.dump(d, open(path, "w"))
    return path


def test_rotate_turns_the_gripper_not_the_wrist():
    w = make_world()
    a, b = w.parts([100, 80, 70, 42], 10.0), w.parts([100, 80, 70, 90], 10.0)
    assert np.allclose(a[3][0], b[3][0]) and np.allclose(a[3][1], b[3][1])      # wrist block unchanged
    assert not np.allclose(a[5][0], b[5][0])                                       # gripper turned
    assert abs(np.linalg.norm(w.tip([100, 80, 70, 42], 10.0) - w.tip([100, 80, 70, 90], 10.0))) > 1.0


def test_locate_inverts_predict():
    w = make_world()
    joints, base = [120, 70, 60, 42], 15.0
    for cam in ("torre", "gripper"):
        px = w.predict_tag(120, cam, joints, base)
        if not np.all(np.isfinite(px)):
            continue
        r = w.locate_tag(cam, px, 30.0, joints, base)
        assert r is not None
        assert np.linalg.norm(r[1] - w.tags[120]["t"]) < 0.5


def test_base_offset_is_found():
    w = make_world()
    joints, base = [130, 60, 70, 42], 20.0
    seen = [("torre", t, w.predict_tag(t, "torre", joints, base, base_offset=0.8)) for t in (120, 121, 100)]
    off, err = w.fit_base_offset(seen, joints, base)
    assert abs(off - 0.8) < 0.02 and err < 0.5


def test_table_height_from_mat():
    w = make_world()
    assert abs(w.table_height(94.0, -411.0) + 254.0) < 1e-6
    assert tag_corners(30.0).shape == (4, 3)


def test_filter_finds_a_joint_offset():
    from armlab.track import ArmFilter
    w = make_world()
    true = np.array([0.0, 1.5, -1.0, 0.0, 0.3])
    f = ArmFilter(w)
    rng = np.random.default_rng(0)
    for k in range(40):
        joints, base = [110 + k % 5, 70, 60, 42], 15.0 + k % 7
        f.step(k * 0.1, [0, 0, 0, 0, 0])
        for cam in ("torre", "gripper"):
            tags = []
            for t in (9, 31, 32, 120, 121, 100):
                px = w.predict_tag(t, cam, joints, base, dq=true)
                if np.all(np.isfinite(px)) and np.all(np.abs(px) < 3000):
                    tags.append((t, px + rng.normal(0, 0.5, px.shape), None))
            f.see(cam, tags, joints, base)
    assert np.all(np.abs(f.offsets()[:5] - true) < 0.3), f.offsets()      # (6th: gripper slack, no finger model here)


def test_filter_ignores_finger_tags_when_the_gripper_moved():
    from armlab.track import ArmFilter
    w = make_world()
    joints, base = [110, 70, 60, 42], 15.0
    px = w.predict_tag(31, "torre", joints, base) + 40.0          # fingers opened: the tag is somewhere else
    f = ArmFilter(w)
    assert f.see("torre", [(31, px, None)], joints, base, grip=120.0) == 0


FINGERS = {"ref_grip": 80.0, "axis": [0.03, -0.04], "tags": {"31": "L", "32": "R"},
           "L": {"r_mm": 31.0, "phi0_deg": 90.0, "k_deg_per_step": -0.8, "k2_deg_per_step2": 0.0006},
           "R": {"r_mm": 31.0, "phi0_deg": -90.0, "k_deg_per_step": 0.8, "k2_deg_per_step2": -0.0006}}


def make_finger_world():
    w = make_world()
    d = dict(w.d, fingers=FINGERS)
    path = os.path.join(tempfile.mkdtemp(), "world.json")
    json.dump(d, open(path, "w"))
    w = World(path)
    w._path = path
    return w


def w_path_of(w):
    return w._path


def test_finger_tags_slide_with_the_gripper():
    w = make_finger_world()
    joints, base = [110, 70, 60, 42], 15.0
    at_ref = w.tag_corners_table(31, joints, base, grip=80.0)
    assert np.allclose(at_ref, w.tag_corners_table(31, joints, base))           # no grip given: where world.json has it
    closed = w.tag_corners_table(31, joints, base, grip=140.0)
    shift = closed - at_ref
    assert np.allclose(shift, shift[0])                                           # slid, not turned
    assert 20.0 < np.linalg.norm(shift[0]) < 40.0
    assert np.allclose(w.tag_corners_table(9, joints, base, grip=140.0), w.tag_corners_table(9, joints, base))
    assert np.linalg.norm(w.tip(joints, base, grip=140.0) - w.tip(joints, base, grip=80.0)) > 5.0
    assert np.allclose(w.tip(joints, base), w.tip(joints, base, grip=80.0))


def test_finger_model_matches_the_solver():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
    try:
        import big_solve
    except ImportError:
        return
    fp = {side: np.array([*FINGERS["axis"], *(FINGERS[side][k] for k in ("r_mm", "phi0_deg", "k_deg_per_step",
                                                                         "k2_deg_per_step2"))]) for side in "LR"}
    from armlab.world import finger_offset
    for side in "LR":
        for g in (40.0, 80.0, 115.0, 150.0):
            a = finger_offset(FINGERS, side, g)
            b = big_solve.finger_offset(fp[side], np.array([g - 80.0]))[0]
            assert np.allclose(a, b, atol=1e-9), (side, g, a, b)


def test_filter_uses_finger_tags_at_any_grip_with_a_finger_model():
    from armlab.track import ArmFilter
    w = make_finger_world()
    joints, base = [110, 70, 60, 42], 15.0
    px = w.predict_tag(31, "torre", joints, base, grip=120.0)
    f = ArmFilter(w)
    assert f.see("torre", [(31, px, None)], joints, base, grip=120.0) == 1
    assert np.all(np.abs(f.offsets()) < 0.05)                                     # it fits as is: nothing to correct


def test_filter_finds_the_gripper_slack():
    from armlab.track import ArmFilter
    w = make_finger_world()
    true = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])                               # the gripper sits 2 steps past its command
    f = ArmFilter(w)
    rng = np.random.default_rng(1)
    for k in range(30):
        joints, base, grip = [110 + k % 5, 70, 60, 42], 15.0, 60.0 + 3 * k
        f.step(k * 0.1, [0] * 6)
        for cam in ("torre", "gripper"):
            tags = []
            for t in (9, 31, 32, 120, 121):
                px = w.predict_tag(t, cam, joints, base, grip=grip, dq=true)
                if np.all(np.isfinite(px)) and np.all(np.abs(px) < 3000):
                    tags.append((t, px + rng.normal(0, 0.5, px.shape), None))
            f.see(cam, tags, joints, base, grip=grip)
    assert abs(f.offsets()[5] - 2.0) < 0.3, f.offsets()
    assert np.all(np.abs(f.offsets()[:5]) < 0.3), f.offsets()


def test_touch_fit_recovers_named_points():
    """Touches where a known point on the hand sits on a tag centre: fit_point must give that point back, for the
    fingertip (moves with the grip), a gripper-body point, a wrist-block point and a finger point."""
    from armlab.modes.touch import fit_point, parse_label
    w = make_finger_world()
    rng = np.random.default_rng(3)
    truth = {"tip": ("G", np.array([205.0, -18.0, 15.0])), "cam": ("G", np.array([20.0, 120.0, -30.0])),
             "H:bottom": ("H", np.array([40.0, -10.0, -35.0])), "L:outer": ("L", np.array([195.0, 30.0, 20.0]))}
    for name, (part, p_true) in truth.items():
        rows = []
        for k in range(4):
            joints = [rng.uniform(80, 140), rng.uniform(40, 110), rng.uniform(30, 120), rng.uniform(20, 90)]
            grip = rng.uniform(60, 150)
            r = {"joints": joints + [grip], "base": rng.uniform(-30, 30), "base_offset": 0.0}
            if name == "tip":
                at = w.point("G", p_true + w.tip_shift(grip), joints, r["base"])
            else:
                R, o = w.parts(joints, r["base"])[3 if part == "H" else 5]
                shift = w.finger_offset_side(part, grip) if part in ("L", "R") else np.zeros(3)
                at = R @ (p_true + shift) + o
            tid = 900 + k
            w.tags[tid] = {"part": 0, "R": np.eye(3), "t": at, "size": 30.0}
            rows.append((r, tid))
        p, miss = fit_point(w, rows, part, tip=(name == "tip"))
        assert np.allclose(p, p_true, atol=1e-6), (name, p, p_true)
        assert miss.max() < 1e-6
    assert parse_label("tag 120") == ("tip", "G", 120)
    assert parse_label("cam tag 7") == ("cam", "G", 7)
    assert parse_label("L:outer tag 101") == ("L:outer", "L", 101)
    assert parse_label("ply edge") is None


def test_basehome_world_finish_lands_on_the_world_zero(monkeypatch):
    from armlab.modes import basehome
    from armlab.state import S
    path = make_world_path()
    w = World(path)
    sim = {"count": 0.0, "off": -1.3}                   # real base = count + off (off as fit_base_offset reports it)

    def read():
        a = [130, 60, 70, 42, 80, sim["count"]]
        seen = [("torre", t, w.predict_tag(t, "torre", a[:4], a[5], base_offset=sim["off"])) for t in (120, 121, 100)]
        seen.append(("torre", 121, np.zeros((4, 2))))      # a misread: the median ignores it
        return a, seen

    def move_base(x, from_below=True, closed=None, runup=None):     # the base follows the count, but the worm settles a bit off,
        sim["count"] = x                                   # differently at each stop
        sim["off"] = -1.3 if x == 0.0 else 0.25 * np.sin(7 * x) - 0.15

    def http_json(url):
        assert url.endswith("/basezero")
        sim["off"] += sim["count"]
        sim["count"] = 0.0

    monkeypatch.setattr(basehome, "move_base", move_base)
    monkeypatch.setattr(basehome, "http_json", http_json)
    monkeypatch.setattr(basehome.time, "sleep", lambda s: None)
    monkeypatch.setattr(S, "data_dir", os.path.dirname(path))
    fixed = basehome.finish_with_world(read)
    assert abs(fixed + 1.3) < 0.05 and sim["count"] == 0.0 and abs(sim["off"]) <= 0.12


def test_closed_loop_base_lands_on_the_target():
    from armlab import baseloop
    from armlab.state import S
    w = make_world()
    sim = {"count": 0.0, "real": 0.0, "moves": 0}

    def move(count, from_below, runup=None):           # worm play: arriving going + leaves it 0.8 short, plus a
        sim["count"] = count                           # settle error that depends on where it stops
        sim["real"] = count - (0.8 if from_below or count >= sim.get("last", count) else 0.0) + 0.3 * np.sin(count)
        sim["last"] = count
        sim["moves"] += 1

    def read():
        a = [130, 60, 70, 42, 80, round(sim["count"], 1)]
        seen = [("torre", t, w.predict_tag(t, "torre", a[:4], sim["count"], base_offset=sim["real"] - sim["count"]))
                for t in (120, 121, 100)]
        return a, seen

    S.base_loop_warned = False
    for target in (-20.0, 7.3, 31.0):
        move(target, True)
        sim["moves"] = 0
        got = baseloop.close_base(target, move, read=read, w=w, pause=0)
        assert got is not None and abs(sim["real"] - target) <= baseloop.LOOP_TOL_DEG + 0.02, (target, sim)
        assert sim["moves"] <= baseloop.LOOP_TRIES and S.base_cmd == target


def test_closed_loop_base_leaves_it_when_the_tags_are_not_seen():
    from armlab import baseloop
    w = make_world()
    moved = []
    got = baseloop.close_base(5.0, lambda c, b, r=None: moved.append(c), read=lambda: ([130, 60, 70, 42, 80, 5.0], []), w=w, pause=0)
    assert got is None and not moved


def test_touch_fit_measures_a_bent_arm_from_the_cameras():
    from armlab.modes.touch import camera_dq
    w = make_finger_world()
    joints, base, true = [120.0, 70.0, 60.0, 42.0, 80.0], 10.0, np.array([1.0, -6.0, 0.5, 0.0, 0.0])
    tags = [{"cam": "torre", "id": t, "corners": w.predict_tag(t, "torre", joints[:4], base, grip=80.0, dq=true).tolist()}
            for t in (9, 31, 32, 120)]
    dq, px, n = camera_dq(w, {"joints": joints, "base": base, "tags": tags})
    assert n == 3 and px < 0.5                                     # the table tag is not an arm tag
    tip_true = w.tip(joints[:4], base, grip=80.0, dq=true)
    assert np.linalg.norm(w.tip(joints[:4], base, grip=80.0, dq=dq) - tip_true) < 1.0
    _, _, n2 = camera_dq(w, {"joints": joints[:4] + [160.0], "base": base, "tags": tags})
    assert n2 == 1                                                 # past the surveyed grip: finger tags not used


def make_table_model(w, path):
    """A tiny table model: a gripper body point, a fingertip on each finger, one wrist-block point."""
    tip = w.tip_in_G
    d = {"made": "test", "planner_bias_mm": 3.0,
         "hand": {"ref_grip": 80.0, "parts": {"G": [[100.0, 0.0, 0.0]], "L": [list(tip + [0, 10, 0])], "R": [list(tip - [0, 10, 0])]}},
         "links": {"H": [[20.0, 0.0, 0.0]]},
         "plywood": {"bin_mm": 5.0, "height_mm": [125.0] * 10 + [19.0] * 30}}
    json.dump(d, open(path, "w"))


def test_exact_table_check_follows_the_arm_down():
    from armlab.tablecheck import TableCheck
    w = make_finger_world()
    path = os.path.join(tempfile.mkdtemp(), "table_model.json")
    make_table_model(w, path)
    t = TableCheck(w, path)
    hi, lo = t.gaps([120, 70, 60, 42, 80], 10.0), t.gaps([100, 70, 60, 42, 80], 10.0)
    assert lo[1] < hi[1] - 5                                       # shoulder down: the lowest point drops
    p = t.points([120, 70, 60, 42, 80], 10.0)
    a, b, c = t.plane
    assert abs(hi[1] - ((p[:, 2] - (a * p[:, 0] + b * p[:, 1] + c)).min() - 3.0)) < 1e-6
    p150 = t.points([120, 70, 60, 42, 150], 10.0)
    assert np.linalg.norm(p150[1] - p[1]) > 5 and np.allclose(p150[0], p[0])   # fingers slide, the body does not
    assert t.gaps([120, 70, 60, 42, 80], None)[0] <= hi[0] + 1e-9            # unknown base: the worst angle


def test_zone_uses_the_exact_table_check(monkeypatch):
    from armlab import safety
    from armlab.state import S
    w = make_finger_world()
    d = tempfile.mkdtemp()
    shutil = __import__("shutil")
    shutil.copy(w_path_of(w), os.path.join(d, "world.json"))
    make_table_model(w, os.path.join(d, "table_model.json"))
    ax = np.arange(0, 161, 40.0)
    full = np.full((len(ax), len(ax), len(ax)), 200, np.uint8)
    np.savez(os.path.join(d, "arm_map.npz"), s=ax, e=ax, w=ax, base=full, tower=full, self=full,
             low=np.full(full.shape, 500, np.int16), meta=json.dumps({"made": "test"}))
    monkeypatch.setattr(S, "data_dir", d)
    monkeypatch.setattr(S, "exact_table", True)
    monkeypatch.setattr(S, "base_cmd", 10.0)
    z = safety.Zone(margin=40)
    from armlab.tablecheck import TableCheck
    t = TableCheck(w, os.path.join(d, "table_model.json"))
    pose = [120, 70, 60]
    want = max(t.gaps([120, 70, 60, 42, 80], 10.0)[0], 0.0)
    assert abs(z.detail(pose)["base"] - want) < 1e-6 and abs(z.gap(pose) - min(want, 200)) < 1e-6
    six = [120, 70, 60, 90, 150, -30.0]                            # a full arm_angles() reading: its own hand and base
    assert abs(z.detail(six)["base"] - max(t.gaps(six[:5], six[5])[0], 0.0)) < 1e-6
    monkeypatch.setattr(S, "exact_table", False)
    assert safety.Zone(margin=40).detail(pose)["base"] == 200.0    # without the flag: the map, as before


def test_survey_picks_only_what_the_exact_check_allows():
    import random as _random
    from armlab.modes import survey

    class FakeExact:                                   # "table gap" = 60 minus how far rotate/grip/base stray
        def gaps(self, q, base):
            g = 60.0 - abs(q[3] - 42.0) * 0.5 - max(0.0, q[4] - 100.0) - abs(base) * 0.3
            return g, g

    class FakeZone:
        exact, table_margin = FakeExact(), 20.0

    z, rng = FakeZone(), _random.Random(1)
    for _ in range(30):
        roll, roll_from, grip, grip_from = survey.exact_hand(z, [100, 60, 60], 0.0, rng)
        r, rf = (42.0, 42.0) if roll is None else (roll, roll_from)
        g, gf = (80.0, 80.0) if grip is None else (grip, grip_from)
        assert survey.exact_ok(z, [100, 60, 60], 0.0, [(42.0, 80.0), (rf, gf), (r, g), (42.0, 80.0)])
        b = survey.exact_base(z, [100, 60, 60], -60.0, 90.0, rng, 0.0)
        assert b is None or 60.0 - abs(b) * 0.3 >= 20.0
    z.table_margin = 100.0                             # nothing fits: level hand, usual grip, stay at this base
    assert survey.exact_hand(z, [100, 60, 60], 0.0, rng) == (None, None, None, None)
    assert survey.exact_base(z, [100, 60, 60], -60.0, 90.0, rng, 0.0) is None


def test_cube_is_found_from_the_cameras():
    from armlab import cube as cubemod
    w = make_finger_world()
    joints, base = [120.0, 70.0, 60.0, 42.0], 10.0
    true = np.array([360.0, 10.0, w.table_height(360.0, 10.0) + cubemod.CUBE_MM, np.radians(17.0)])
    X = cubemod._corners(true)
    seen = [("torre", 7, w.project("torre", X, joints, base))]
    c = cubemod.locate(w, seen, joints, base)
    assert c is not None and np.linalg.norm(c["xyz"] - true[:3]) < 1.0 and abs(c["yaw_deg"] - 17.0) < 0.5
    assert abs(c["above_table_mm"] - cubemod.CUBE_MM) < 1.0 and c["px"] < 0.1
    assert cubemod.locate(w, [("torre", 9, X[:, :2])], joints, base) is None      # no cube tag: nothing


def test_grasp_solve_puts_the_fingertip_on_the_point():
    import math
    from armlab import grasp
    w = make_finger_world()
    q0 = [110.0, 70.0, 60.0, 50.0, 60.0, 12.0]
    tip, R = grasp.hand(w, q0)
    y = R[:, 1]
    face = math.atan2(y[1], y[0])                                  # the fingers square to this face at q0
    q, r = grasp.solve(w, tip, grasp.pitch_of(w, q0), face, 60.0, [100.0, 60.0, 70.0, 42.0, 60.0, 5.0])
    assert grasp.solved(r) and np.linalg.norm(grasp.hand(w, q)[0] - tip) < 1.0


def _grab_rig(monkeypatch, refuse_at=None, held=True, found_shift=10.0, lift_mm=None):
    import types
    from armlab.modes import grasp as G
    from armlab.modes import survey
    from armlab.errors import ArmError
    path = [[100.0 + k, 50.0, 60.0, 30.0, 55.0, 10.0] for k in range(13)]          # 0 = hover, 12 = grasp
    path2 = [[130.0 + k, 50.0, 60.0, 30.0, 55.0, 10.0] for k in range(13)]         # the set-down, 10 mm over
    moves, grips = [], []

    def move_to(q, zone, speed=None, slow_finish=False):
        if refuse_at is not None and round(q[0]) == refuse_at and len(moves) < 6:
            raise ArmError("refused: test")
        moves.append(round(q[0]))
    monkeypatch.setattr(G, "move_to", move_to)
    monkeypatch.setattr(survey, "set_hand", lambda roll=None, grip=None, secs=0.8, wait=True:
                        None if grip is None or (grips and grips[-1] == grip) else grips.append(grip))
    monkeypatch.setattr(survey, "set_grip", lambda v, secs=0.6: grips.append(v))
    monkeypatch.setattr(G, "replan", lambda *a, **k: {"path": path, "grasp": path[-1], "target": np.zeros(3),
                                                      "face_deg": 0.0, "pitch": 50.0, "approach": "from above"})
    monkeypatch.setattr(G, "squeeze_check", lambda w: (held, 1.0 if held else 7.0, 7.0))
    monkeypatch.setattr(G, "finger_spacing", lambda w: 64.0)
    rise = (30.0 if held else 0.0) if lift_mm is None else lift_mm
    monkeypatch.setattr(G, "lift_seen", lambda w, dq=None: (rise, 7, "webcam"))
    monkeypatch.setattr(G, "pick_shift", lambda *a, **k: (np.array([10.0, 0.0, 0.0]), {"path": path2}))
    near = {"xyz": np.array([300.0, 0.0, -200.0])}
    monkeypatch.setattr(G, "refind", lambda w: {"xyz": near["xyz"] + [found_shift, 0.0, 0.0]})
    monkeypatch.setattr(G.time, "sleep", lambda s: None)
    monkeypatch.setattr(G, "wait_fresh", lambda after, secs=1.2, cams=None: None)
    monkeypatch.setattr(G, "arm_angles", lambda: path[0])
    zone = types.SimpleNamespace(table_margin=20.0, table_extra=20.0, margin=40.0, hand=(42.0, 80.0))
    args = types.SimpleNamespace(grasp_margin=12.0)
    out = None
    try:
        out = G.grab_from_hover(None, None, zone, near, near, np.zeros(5), {}, 20.0, args)
    except ArmError:
        pass
    assert (zone.table_margin, zone.table_extra, zone.hand) == (20.0, 20.0, (42.0, 80.0))      # put back after
    return G, out, moves, grips


def test_grab_goes_in_lifts_carries_10mm_and_finds_it(monkeypatch):
    G, out, moves, grips = _grab_rig(monkeypatch)
    assert out.startswith("GRABBED") and "CARRIED" in out
    assert moves == (list(range(101, 113)) + list(range(111, 99, -1)) + [130] + list(range(131, 143))
                     + list(range(141, 129, -1)))
    assert G.GRIP_CLOSE in grips and grips[-1] == G.GRIP_OPEN     # closed on it, and open wide last


def test_grab_stops_when_the_squeeze_says_empty(monkeypatch):
    G, out, moves, grips = _grab_rig(monkeypatch, held=False)
    assert out.startswith("NOT grabbed") and moves == list(range(101, 113)) + list(range(111, 99, -1))
    assert grips[-1] == G.GRIP_OPEN


def test_a_camera_seeing_the_block_up_stands_in_for_the_squeeze(monkeypatch):
    """The gripper camera often cannot see both finger tags at the hover, which leaves the squeeze check with no
    answer. Watching the BLOCK rise instead settles it, so the pickup is confirmed with the webcam."""
    G, out, moves, grips = _grab_rig(monkeypatch, held=None, lift_mm=30.0)
    assert out.startswith("GRABBED")                       # it carried on and carried it, rather than giving up


def test_the_lift_check_does_not_overrule_an_empty_squeeze(monkeypatch):
    G, out, moves, grips = _grab_rig(monkeypatch, held=False, lift_mm=30.0)
    assert out.startswith("NOT grabbed") and "so look at it" in out


def test_radial_shifts_aim_the_carry_at_the_wanted_distance(monkeypatch):
    import numpy as np
    from armlab.modes import grasp as G
    import types as _t
    w = _t.SimpleNamespace(axis=np.array([0.0, 0.0, 0.0]))
    near = {"xyz": np.array([250.0, 0.0, -200.0])}
    out = G.radial_shifts(w, near, 340.0)                  # 90 mm further from the axis
    assert out and abs(float(np.linalg.norm(out[0])) - 70.0) < 1e-6     # capped at CARRY_MAX_MM
    assert out[0][0] > 0 and abs(out[0][1]) < 1e-9                      # straight out first
    assert float(np.linalg.norm(G.radial_shifts(w, near, 200.0)[0] + [50.0, 0.0, 0.0])) < 1e-6   # and inward
    assert G.radial_shifts(w, near, 251.0) == []                        # already there: leave it to the nudge


def test_grab_not_confirmed_when_the_block_did_not_move(monkeypatch):
    G, out, moves, grips = _grab_rig(monkeypatch, found_shift=0.5)
    assert out.startswith("NOT confirmed")


def test_grab_backs_out_when_a_move_is_refused(monkeypatch):
    G, out, moves, grips = _grab_rig(monkeypatch, refuse_at=106)
    assert out is None and moves == [101, 102, 103, 104, 105, 104, 103, 102, 101, 100]      # backed out to the hover


def test_lift_check_reads_a_side_tag_height(monkeypatch):
    from armlab.modes import grasp as G
    from armlab.world import tag_corners, rotvec
    w = make_finger_world()
    joints, base = [120.0, 70.0, 60.0, 42.0, 80.0, 10.0], 10.0
    up = 35.0                                                       # the block lifted 35 mm
    c = np.array([360.0, 10.0, w.table_height(360.0, 10.0) + G.CUBE_MM / 2 + up])
    X = tag_corners(20.0) @ rotvec([np.pi / 2, 0.0, 0.0]).T + c     # a side tag, standing up
    px = w.project("torre", X, joints[:4], base)
    monkeypatch.setattr(G, "arm_angles", lambda: joints)
    monkeypatch.setattr(G, "sightings_now", lambda: [("torre", 251, px)])
    rise, tag, cam = G.lift_seen(w)
    assert tag == 251 and cam == "torre" and abs(rise - up) < 1.0


def test_search_sweeps_the_base_until_a_camera_sees_the_block(monkeypatch):
    import types
    from armlab import hardware
    from armlab.modes import grasp as G
    from armlab.modes import survey
    from armlab.state import S
    base = {"b": 0.0, "goal": 0.0}

    def base_angle():                                        # the base creeps toward its goal at each reading
        b = base["b"]
        base["b"] += max(-1.0, min(1.0, base["goal"] - b))
        return b
    calls = []

    def http_json(url, tries=3):
        if "/setall" in url:
            base["goal"] = float(url.split(",,,,,")[1].split("&")[0])
        calls.append(url)
        return {"ok": 1}
    monkeypatch.setattr(hardware, "base_angle", base_angle)
    monkeypatch.setattr(hardware, "http_json", http_json)
    monkeypatch.setattr(hardware, "arm_stop", lambda: base.update(goal=base["b"]) or True)
    monkeypatch.setattr(G, "arm_angles", lambda: [100.0, 60.0, 60.0, 42.0, 80.0, base["b"]])
    monkeypatch.setattr(G, "sightings_now", lambda: [("webcam", 7, np.zeros((4, 2)))] if base["b"] < -20 else [])
    monkeypatch.setattr(G, "move_to", lambda q, zone, speed=None, slow_finish=False: calls.append(("arm", list(q))))
    monkeypatch.setattr(G, "move_base", lambda t, fb=True, closed=None, settle=True: base.update(b=t, goal=t))
    monkeypatch.setattr(survey, "set_hand", lambda roll=None, grip=None, secs=0.8, wait=True: None)
    monkeypatch.setattr(G.time, "sleep", lambda s: None)
    monkeypatch.setattr(G, "find_cube", lambda w: (None, 0.0, 1.0, {"xyz": np.zeros(3)} if base["b"] < -20 else None, []))
    monkeypatch.setattr(S, "base_speed_cmd", True)
    monkeypatch.setattr(G, "search_stops", lambda w: [30.0, -15.0])   # still looks that see the practice area
    monkeypatch.setattr(G, "aim_base_at", lambda w, xyz: 10.0)        # where it was last: not there
    monkeypatch.setattr(G, "LAST_BLOCK", {"xyz": np.zeros(3)})
    import types as _t
    zone = _t.SimpleNamespace(contains=lambda q: True, path_ok=lambda a, b: True)
    check = types.SimpleNamespace(gaps=lambda q, b: (100.0, 110.0))
    got = G.search(None, zone, check, 10.0)
    assert got is not None and base["b"] < -20
    assert ("arm", G.SEARCH_POSE) in calls
    assert any(f"bs={G.SEARCH_DPS:g}" in c for c in calls if isinstance(c, str))     # the slow sweep
    sweeps = [float(c.split(",,,,,")[1].split("&")[0]) for c in calls if isinstance(c, str) and "/setall" in c]
    assert sweeps[:3] == [30.0, -15.0, -57.0]     # from base 10 (the last-seen look): the stops, nearer first, the rest
    pauses = [c for c in calls if isinstance(c, str) and "/rate?pause" in c]
    assert pauses[0].startswith(S.cam_url + "/rate?pause=gripper") and pauses[-1].startswith(S.cam_url + "/rate?pause=&")
    check_low = types.SimpleNamespace(gaps=lambda q, b: (40.0, 50.0))
    try:
        G.search(None, zone, check_low, 10.0)
        assert False, "a search pose near the table must be refused"
    except G.ArmError:
        pass


def test_block_filter_fuses_looks_gates_moves_and_carries():
    from armlab.cube import BlockFilter
    f = BlockFilter()
    f.see({"xyz": np.array([300.0, 0.0, 50.0]), "yaw_deg": 44.0, "cams": {"torre": 1.0}, "above_table_mm": 50.8})
    assert abs(f.sigma_mm() - 2.0) < 1e-6
    f.see({"xyz": np.array([302.0, 0.0, 50.0]), "yaw_deg": -44.0, "cams": {"gripper": 0.3}, "above_table_mm": 50.8,
           "close": True})
    c = f.cube()
    assert 301.5 < c["xyz"][0] < 302.0 and f.sigma_mm() < 1.0       # leans to the close-up gripper look
    assert abs(abs(c["yaw_deg"]) - 45.0) < 2.0                       # 44 and -44 are 2 deg apart on a square
    d = f.see({"xyz": np.array([340.0, 0.0, 50.0]), "yaw_deg": 0.0, "cams": {"torre": 1.0}})
    assert d > 30 and abs(f.cube()["xyz"][0] - 340.0) < 1e-6         # moved: starts again from the new look
    s0 = f.sigma_mm()
    f.moved([10.0, 0.0, 0.0])
    assert abs(f.cube()["xyz"][0] - 350.0) < 1e-6 and f.sigma_mm() > s0


def test_look_close_goes_on_the_filter_only_when_it_is_sure(monkeypatch):
    from armlab.cube import BlockFilter
    from armlab.modes import grasp as G
    monkeypatch.setattr(G, "look_up_close", lambda w, cube: (None, np.zeros(5)))
    monkeypatch.setattr(G, "back_off_look", lambda *a: None)
    monkeypatch.setattr(G, "hunt_for_tag", lambda *a, **k: None)      # the last looker in the chain, stubbed like the rest
    for px, ok in ((1.0, True), (3.0, False)):                      # torre fit 1 px: +-2 mm; 3 px: +-6 mm
        f = BlockFilter()
        f.see({"xyz": np.array([300.0, 0.0, 50.0]), "yaw_deg": 0.0, "cams": {"torre": px}})
        near, dq = G.look_close(None, None, None, {}, {}, f, 20.0)
        assert (near is not None) == ok
    monkeypatch.setattr(G, "back_off_look", lambda *a: {"xyz": np.array([301.0, 0.0, 50.0]), "yaw_deg": 0.0,
                                                         "cams": {"gripper": 0.5}})
    near, _ = G.look_close(None, None, None, {}, {}, f, 20.0)       # the back-off look found it
    assert near is not None and f.looks == 2


def test_the_demo_walks_the_block_out_and_back(monkeypatch):
    """Out along the radii then back down them, so each round's carry is within CARRY_MAX_MM of the last."""
    from armlab.modes.show import ladder
    from armlab.modes import grasp as G
    walk = ladder([240.0, 290.0, 340.0, 385.0], 9)
    assert walk[:7] == [240.0, 290.0, 340.0, 385.0, 340.0, 290.0, 240.0]
    assert walk[7] == 290.0                                   # and round again, not stuck at the near end
    assert max(abs(b - a) for a, b in zip(walk, walk[1:])) <= G.CARRY_MAX_MM


def test_world_zero_hunts_for_the_tags_without_changing_the_offset(monkeypatch):
    """Turning the base brings the mat into view; it does not move the base's offset from the world's zero, so
    the offset that comes back is the one the tags gave, not one adjusted by how far the base turned."""
    from armlab.modes import basehome as B
    turned = []
    monkeypatch.setattr(B, "base_angle", lambda: 0.0)
    monkeypatch.setattr(B, "move_base", lambda c, *a, **k: turned.append(round(float(c), 1)))
    monkeypatch.setattr(B.time, "sleep", lambda s: None)
    seen = {"n": 0}

    def fake_offset(w, read, **kw):
        seen["n"] += 1
        return (0.75, 4) if seen["n"] == 3 else (None, 0)   # readable only from the second angle tried

    monkeypatch.setattr(B, "world_offset", fake_offset)
    off, n = B.hunt_world_offset(None, read=lambda: (None, None))
    assert (off, n) == (0.75, 4)                            # the tags' answer, untouched by the +-3 deg of hunting
    assert turned == [3.0, -3.0]                            # out one step each way, stopping as soon as it reads


def test_world_zero_gives_up_and_puts_the_base_back(monkeypatch):
    from armlab.modes import basehome as B
    turned = []
    monkeypatch.setattr(B, "base_angle", lambda: 5.0)
    monkeypatch.setattr(B, "move_base", lambda c, *a, **k: turned.append(round(float(c), 1)))
    monkeypatch.setattr(B.time, "sleep", lambda s: None)
    monkeypatch.setattr(B, "world_offset", lambda w, read, **kw: (None, 0))
    assert B.hunt_world_offset(None, read=lambda: (None, None)) == (None, 0)
    assert turned[-1] == 5.0                                # back where it started, not left mid-sweep
    assert max(abs(c - 5.0) for c in turned) <= B.WORLD_SWEEP_DEG


def test_the_demo_scoreboard_does_not_credit_a_round_that_threw():
    """The run file is filled in as the grab goes, so a round that threw on the way home still has GRABBED in it.
    Counting that would make the scoreboard -- the thing the demo is for -- flatter itself."""
    from armlab.modes.show import _good
    assert _good({"result": "GRABBED (squeeze held)"})
    assert not _good({"result": "GRABBED (squeeze held)", "error": "refused going home"})
    assert not _good({"result": "NOT confirmed: squeeze empty"})
    assert not _good({})


def test_tower_lean_tips_the_arm_and_cameras_but_not_the_table():
    """tower_lean: the tower carries the base cameras and the shoulder, so both turn by k*shoulder + b
    relative to the table; the table and its tags stay put. At the shoulder where the lean is zero nothing moves."""
    w0 = make_world()
    d = dict(w0.d, tower_lean={"deg_per_shoulder_step": 0.032, "deg_at_shoulder_0": -1.41})
    path = os.path.join(tempfile.mkdtemp(), "world.json")
    json.dump(d, open(path, "w"))
    w = World(path)
    s0 = 1.41 / 0.032                                      # lean is zero here
    for b in (0.0, -30.0):
        for got, want in zip(w.parts([s0, 70.0, 60.0, 42.0], b), w0.parts([s0, 70.0, 60.0, 42.0], b)):
            assert np.allclose(got[0], want[0]) and np.allclose(got[1], want[1])
    q = [5.0, 75.0, 60.0, 42.0]                            # shoulder low: the tower leans -1.25 deg
    P, P0 = w.parts(q, 0.0), w0.parts(q, 0.0)
    assert np.allclose(P[0][0], P0[0][0]) and np.allclose(P[0][1], P0[0][1])          # the table does not move
    tip, tip0 = w.tip(q, 0.0), w0.tip(q, 0.0)
    r = np.linalg.norm(tip0[:2] - w0.axis[:2])
    assert abs(abs(tip[2] - tip0[2]) - r * np.radians(1.25)) < 0.15 * r * np.radians(1.25)   # it swings by the lean
    for k in range(6):                                     # frame() agrees with parts()
        R, o = w.frame(k, q, 0.0)
        assert np.allclose(R, P[k][0]) and np.allclose(o, P[k][1])
    Rc, tc = w.camera_pose("torre", q, 0.0)                # and the camera leans with the arm it rides beside
    Rc0, tc0 = w0.camera_pose("torre", q, 0.0)
    assert not np.allclose(Rc, Rc0)


def test_base_offset_measured_around_a_leaning_tower():
    """fit_base_offset(tilt=True): the tower leans under the arm, which a turn alone cannot match, so the
    plain fit is dragged off and basehome's finish gave up at 27 px. Lean the tower 1.6 deg (generated with
    tower_lean itself), put the base 1.3 deg off, and the tilted fit should hand back the 1.3."""
    w = make_world()
    d = dict(w.d, tower_lean={"deg_per_shoulder_step": 0.0, "deg_at_shoulder_0": 1.6})
    path = os.path.join(tempfile.mkdtemp(), "world.json")
    json.dump(d, open(path, "w"))
    leaning = World(path)
    a = [130, 60, 70, 42, 80, 0.0]
    tags = [t for t, g in w.tags.items() if g["part"] == 0]
    seen = [("torre", t, leaning.predict_tag(t, "torre", a[:4], a[5], base_offset=-1.3)) for t in tags]
    off0, fit0 = w.fit_base_offset(seen, a[:4], a[5])
    off, fit = w.fit_base_offset(seen, a[:4], a[5], tilt=True)
    assert abs(off + 1.3) < 0.05 and fit < 0.5                 # the turn, clean
    assert fit < fit0                                          # and a better fit than turning alone could manage
    assert abs(np.hypot(*w.last_tilt_deg) - 1.6) < 0.2          # the lean it found is the one put in
    assert w.fit_base_offset(seen, a[:4], a[5]) == (off0, fit0)  # the plain fit is untouched by the option


def test_extra_looks_stay_where_the_block_was_found_and_never_turn_into_it(monkeypatch):
    """Every look taken up at HIGH_POSE saw nothing (the arm in front of the webcam), so the looks are taken
    from the pose the block was found from, panning LOOK_PAN_DEG and twice that to the side the model says keeps the
    block clear of the arm. From home (hand low over the table) it goes to the search pose first; a pan at home
    swept the hand into the block, so every turn is still checked and a close one is skipped."""
    from armlab.modes import grasp as G
    log = []
    home = [5.8, 75.0, 60.0, 42.0, 80.0, -18.0]
    monkeypatch.setattr(G, "arm_angles", lambda: home if not log else G.SEARCH_POSE + [42.0, 80.0, -18.0])
    monkeypatch.setattr(G, "move_to", lambda q, zone, speed, slow_finish=False: log.append(("arm", list(q))))
    monkeypatch.setattr(G, "move_base", lambda b, *a, **k: log.append(("base", round(float(b), 1))))
    monkeypatch.setattr(G, "wait_fresh", lambda after, secs=1.2, cams=None: None)
    monkeypatch.setattr(G, "check_arm_move", lambda *a, **k: None)
    monkeypatch.setattr(G, "find_cube", lambda w: (None, 0.0, 1.0, {"xyz": np.array([300.0, 90.0, -200.0])}, []))
    monkeypatch.setattr(G, "view_clear_px", lambda w, check, cam, q, xyz: 300.0 if q[5] < -18.0 else 20.0)  # - side clear
    monkeypatch.setattr(G, "cube_gap", lambda check, cube, q, b: 80.0 if b > -24.0 else 10.0)  # but -8 deg comes close
    import types as _t
    zone = _t.SimpleNamespace(contains=lambda q: True, path_ok=lambda a, b: True)
    check = _t.SimpleNamespace(gaps=lambda q, b: (100.0, 0.0))
    w = _t.SimpleNamespace(cams={"webcam": {}, "torre": {}})

    class Blk:
        def see(self, c, still=False):
            return 0.0
    got = G.look_around(w, Blk(), -18.0, 20.0, zone, check, {"xyz": np.array([300.0, 90.0, -200.0])})
    assert log[0] == ("arm", list(G.SEARCH_POSE))                     # to the search pose before anything turns
    assert ("base", -22.0) in log and ("base", -26.0) not in log      # the clear side; the close turn skipped
    assert not any(e[0] == "base" and e[1] > -18.0 for e in log)       # never the side the arm would hide it
    assert got == 1
    log.clear(); log.append(("start", 0))                              # already at the search pose: no move at all
    G.look_around(w, Blk(), -18.0, 20.0, zone, check, {"xyz": np.array([300.0, 90.0, -200.0])})
    assert not any(e[0] == "arm" for e in log)

def test_nudges_land_on_a_base_that_swallows_small_moves():
    """basehome's finish nudged +1.25 deg and the base did not turn; three ~1 deg nudges recovered 0.68 of
    1.11. A base that loses the first 0.9 deg of any move from rest: forward-only nudges stall, a run-up does not."""
    from armlab import baseloop
    from armlab.state import S
    w = make_world()
    sim = {"count": 0.0, "real": -1.11, "moves": 0}

    def leg(to):                                        # one move from rest: the first 0.9 deg winds up, no turn
        d = to - sim["count"]
        sim["real"] += np.sign(d) * max(0.0, abs(d) - 0.9)
        sim["count"] = to

    def move(count, from_below, runup=None):
        if from_below:
            leg(count - (1.5 if runup is None else runup))
        leg(count)
        sim["moves"] += 1

    def read():
        a = [130, 60, 70, 42, 80, round(sim["count"], 1)]
        seen = [("torre", t, w.predict_tag(t, "torre", a[:4], sim["count"], base_offset=sim["real"] - sim["count"]))
                for t in (120, 121, 100)]
        return a, seen

    S.base_loop_warned = False
    got = baseloop.close_base(0.0, move, read=read, w=w, pause=0)
    assert got is not None and abs(sim["real"]) <= baseloop.LOOP_TOL_DEG + 0.02, sim
    assert sim["moves"] <= baseloop.LOOP_TRIES


def test_replan_aims_at_the_plans_own_depth_and_pull(monkeypatch):
    """replan used to take the default DEPTH_MM whatever the plan used, so a grab planned with --grasp-depth 10 was
    re-planned 10 mm deeper after the close-up look and failed at 380 mm out. It must aim where the plan did."""
    import armlab.grasp as G
    w = make_world()
    got = {}

    def reach(w_, target, *a, **k):
        got.setdefault("t", np.array(target, float))
        return None
    monkeypatch.setattr(G, "reach", reach)
    cube = {"xyz": np.array([300.0, 40.0, -200.0]), "yaw_deg": 0.0}
    prev = {"face_deg": 90.0, "grip": 55.0, "pitch": 40.0, "grasp": [30, 80, 60, 42, 55, 0], "approach": "from above",
            "depth_mm": 10.0, "pull_mm": 15.0}
    assert "why" in G.replan(w, None, cube, prev, 15.0)
    radial = cube["xyz"][:2] - w.axis[:2]
    want = cube["xyz"] - [0, 0, 10.0]
    want[:2] -= radial / np.linalg.norm(radial) * 15.0
    assert np.allclose(got["t"], want)                                 # 10 mm down, 15 mm toward the tower


def test_a_carry_that_moved_the_block_counts_even_if_it_landed_a_bit_off():
    """Held, lifted, moved 12 mm, landed 8 mm off a 10 mm carry -> was NOT confirmed. Carried; not placed."""
    from armlab.modes.grasp import carry_verdict
    d = [3.2, -9.5, 0.0]
    assert carry_verdict(11.6, 8.1, d)[:2] == (True, False)
    assert carry_verdict(10.4, 1.5, d)[:2] == (True, True)              # on the aim
    assert carry_verdict(1.2, 9.8, d)[:2] == (False, False)             # never moved
    assert carry_verdict(14.0, 19.0, d)[:2] == (False, False)           # moved, but the wrong way: not the carry
    assert carry_verdict(36.0, 30.0, [70.0, 0.0, 0.0])[:2] == (True, False)   # show's long carries too


def test_show_carry_goes_up_turns_up_there_then_sets_down(monkeypatch):
    """show's carry: up to HIGH_POSE holding the block, the base turned up there (closed loop, where the base
    turns by what it is told), then across and down -- not the base turned with the arm out at the hover."""
    from armlab.modes import grasp as G
    events = []
    down = [[150.0 + k, 60.0, 60.0, 50.0, 55.0, 25.0] for k in range(4)]
    monkeypatch.setattr(G, "CARRY_TO_R", 300.0)
    monkeypatch.setattr(G, "place_shift", lambda *a, **k: (np.array([-20.0, 30.0, 0.0]),
                                                           {"path": down, "via_high": True, "spot": [280.0, 30.0, -200.0],
                                                            "pan_deg": 22.0, "pitch": 60.0}))
    monkeypatch.setattr(G, "move_base", lambda b, *a, **k: events.append(("base", round(float(b), 1))))
    G2, out, moves, grips = _grab_rig(monkeypatch)
    high = round(G.HIGH_POSE[0])
    assert high in moves and 150 in moves
    assert moves.index(high) < moves.index(150)                  # up before going across to the set-down
    assert events == [("base", 25.0)]                            # the base turned once, to the set-down's angle
    assert G.RELEASE_GRIP in grips and grips.index(G.RELEASE_GRIP) < len(grips) - 1    # let go a little, then
    assert grips[-1] == G.GRIP_OPEN                                                   # open wide, last
    booked = [n for n, _ in G.LAPS.rows]                                            # and every phase was timed
    for name in ("re-plan", "in", "close", "lift", "squeeze", "carry plan", "carry", "down", "let go", "check"):
        assert name in booked, (name, booked)


def test_show_looks_for_the_block_after_its_last_round(monkeypatch):
    """The last round's set-down has no next round to check it: show finds the block and says how
    far it is from where the fingers let go and from where it was aimed."""
    import types as _t
    from armlab.modes import show as SH
    w = _t.SimpleNamespace(axis=np.array([0.0, 0.0, 0.0]))
    monkeypatch.setattr(SH.G, "find_block_anywhere", lambda args: ({"xyz": [300.0, 4.0, -200.0], "yaw_deg": 10.0}, w))
    last = {"released_xyz": [300.0, 0.0, -200.0], "carry_plan": {"spot": [310.0, 0.0, -200.0]}}
    got = SH._last_look(_t.SimpleNamespace(), last)
    assert got["found"] and abs(got["released_off_mm"] - 4.0) < 0.06 and abs(got["out_mm"] - 300.03) < 0.1
    assert abs(got["aim_off_mm"] - np.hypot(10.0, 4.0)) < 0.06            # saved to 0.1 mm


def test_demo_page_serves_the_live_state():
    """armlab.live: the page and /live come up, and a lap moves the banner on to the next step."""
    import urllib.request
    from armlab import live
    url = live.serve(port=18090)
    assert url is not None
    live.lap("squeeze", [["squeeze", 2.0]])
    live.update(grip={"held": True})
    d = json.loads(urllib.request.urlopen("http://127.0.0.1:18090/live", timeout=5).read())
    assert d["step"] == "Choosing where to put it" and d["grip"]["held"] is True and d["laps"] == [["squeeze", 2.0]]
    page = urllib.request.urlopen("http://127.0.0.1:18090/", timeout=5).read().decode()
    assert "Arm Live" in page and "/live" in page


def test_plan_table_seeds_the_nearest_plan_and_keeps_only_what_passes(monkeypatch, tmp_path):
    """The plan table: the nearest entry is re-planned for the block and kept only if it passes what
    plan_best would ask; a table for other files, other margins or another depth is never used."""
    import armlab.grasp as G
    w = make_world()
    for f in G.PLAN_TABLE_STAMPED:
        (tmp_path / f).write_bytes(f.encode())
    plan = {"pitch": 40.0, "face_deg": 10.0, "grip": 55.0, "grasp": [30, 80, 60, 42, 55, 0], "approach": "from above",
            "hover_mm": 35.0, "depth_mm": 10.0, "pull_mm": 0.0}
    table = {"stamp": G.plan_table_stamp(str(tmp_path)), "margin": 15.0, "grasp_margin": 12.0, "depth_mm": 10.0,
             "plans": [{"xy": [300.0, 40.0], "yaw": 10.0, "plan": plan},
                       {"xy": [200.0, 40.0], "yaw": 10.0, "plan": dict(plan, face_deg=99.0)}]}
    (tmp_path / G.PLAN_TABLE_FILE).write_text(json.dumps(table))
    assert G.load_plan_table(str(tmp_path))["plans"]
    (tmp_path / "world.json").write_bytes(b"another world")
    assert G.load_plan_table(str(tmp_path)) is None                       # stale: built for another world model

    seen = []
    good = {"path": [[30, 80, 60, 42, 55, 0]] * 3, "sees": True}
    monkeypatch.setattr(G, "limit_room", lambda q: 5.0)
    monkeypatch.setattr(G, "replan", lambda w_, ch, c, prev, m, grasp_margin=12.0: seen.append(prev) or dict(good))
    cube = {"xyz": np.array([305.0, 42.0, -200.0]), "yaw_deg": 14.0}
    got = G.plan_seeded(w, None, cube, 15.0, table, depth=10.0)
    assert got is not None and seen[0]["face_deg"] == 14.0                # the nearest, its face turned with the block
    assert "5 mm, +4 deg" in got["seeded"]
    assert G.plan_seeded(w, None, cube, 15.0, table, depth=20.0) is None  # another depth: not this table
    assert G.plan_seeded(w, None, cube, 40.0, table, depth=10.0) is None  # other margins: not this table
    seen.clear()
    assert G.plan_seeded(w, None, cube, 15.0, table, depth=10.0, accept=lambda p: False) is None
    assert len(seen) == 2                                                 # tried both, the caller refused both
    good["sees"] = False
    assert G.plan_seeded(w, None, cube, 15.0, table, depth=10.0) is None  # blind: plan_best from scratch instead
    # Faces repeat every 180 deg and block yaw every 90: a seed at yaw 40 for a block now at -42 is 8 deg further
    # round (-42 = 48 - 90). Its face 130 becomes 138 = -42 + 180; its face 40 becomes 48 = -42 + 90.
    assert G._face_like(130.0, 8.0, -42.0) == -42.0
    assert G._face_like(40.0, 8.0, -42.0) == 48.0


def test_stay_up_waits_high_at_home_angle_and_comes_down_at_the_end(monkeypatch):
    """show --stay-up: between rounds the arm rises, turns the base back to the round's home angle and
    waits there instead of going down to home; the next round uses that same home angle; the end of the show goes
    down to home properly."""
    import armlab.modes.grasp as G
    from armlab.modes import survey
    log = []
    monkeypatch.setattr(survey, "set_hand", lambda roll=None, grip=None, secs=0.8, wait=True: None)
    monkeypatch.setattr(G, "move_to", lambda q, zone, speed, **k: log.append(("to", list(q))))
    monkeypatch.setattr(G, "move_base", lambda b, closed=True, settle=True: log.append(("base", b)))
    monkeypatch.setattr(G, "travel", lambda q, zone, speed: log.append(("home", list(q))))
    zone = type("Z", (), {"home": [10.0, 20.0, 30.0]})()
    G.STAY_UP.clear()
    G.go_home(zone, -12.0, 8.0, stay_up=True)
    assert log == [("to", G.HIGH_POSE), ("base", -12.0)] and G.STAY_UP == {"base": -12.0}
    log.clear()
    G.go_home(zone, -12.0, 8.0)
    assert log[-1] == ("home", [10.0, 20.0, 30.0]) and not G.STAY_UP


def test_stay_up_waits_where_a_camera_sees_the_block_clear_of_the_arm(monkeypatch):
    """Waiting centred on the block, the arm stood in front of the camera and the first look missed 6
    rounds in 7 (22-53 s searches). park_spot: the webcam from the search pose first, then the tower from up high, at
    the base angle that keeps the block furthest from the arm in the picture, never closer than CUBE_CLEAR_MM to it."""
    import armlab.modes.grasp as MG

    def clear(w, check, cam, q, xyz):                    # webcam: clear only at -30..-20; tower: anywhere, 300 px
        if cam == "webcam":
            return 250.0 + 10.0 * (q[5] + 25.0 - abs(q[5] + 25.0)) if -30.0 <= q[5] <= -20.0 else 50.0
        return 300.0
    monkeypatch.setattr(MG, "view_clear_px", clear)
    monkeypatch.setattr(MG, "cube_gap", lambda check, cube, q5, b: 100.0)
    W = type("W", (), {"axis": np.zeros(3)})()
    pose, b, cam = MG.park_spot(W, None, [300.0, 0.0, -150.0])
    assert cam == "webcam" and pose == list(MG.SEARCH_POSE) and -30.0 <= b <= -20.0
    import armlab.grasp as P
    monkeypatch.setattr(P, "base_toward", lambda w, t: -21.0)            # the next round turns to -21: wait nearest it
    pose, b, cam = MG.park_spot(W, None, [300.0, 0.0, -150.0])
    assert cam == "webcam" and b == -21.0
    # views 100-200 px clear were missed 3 of 3; one 200+ px clear wins, on either camera
    monkeypatch.setattr(MG, "view_clear_px", lambda w, check, cam, q, xyz: 150.0 if cam == "webcam" else 260.0)
    pose, b, cam = MG.park_spot(W, None, [300.0, 0.0, -150.0])
    assert cam == "torre" and pose == list(MG.HIGH_POSE)
    monkeypatch.setattr(MG, "view_clear_px", lambda w, check, cam, q, xyz:
                        (160.0 if abs(q[5] - 9.0) < 0.1 else 120.0) if cam == "webcam" else None)
    pose, b, cam = MG.park_spot(W, None, [300.0, 0.0, -150.0])   # nothing 200 clear: the clearest there is
    assert cam == "webcam" and abs(b - 9.0) < 0.1
    monkeypatch.setattr(MG, "view_clear_px", lambda w, check, cam, q, xyz: 300.0 if cam == "torre" else None)
    pose, b, cam = MG.park_spot(W, None, [400.0, 0.0, -150.0])
    assert cam == "torre" and pose == list(MG.HIGH_POSE)
    monkeypatch.setattr(MG, "cube_gap", lambda check, cube, q5, b: 1.0)       # every waiting pose too near the block
    assert MG.park_spot(W, None, [400.0, 0.0, -150.0]) is None

def test_film_starts_and_stops_with_the_show(monkeypatch, tmp_path):
    """show --film: the recorder runs for exactly as long as the show, with nothing to guess up front."""
    from armlab import film as F
    monkeypatch.setattr(F, "FILM_STREAMS", ())
    monkeypatch.setattr(F.Film, "_save", lambda self, cam, jpg: setattr(self, "frames", self.frames + 1))
    monkeypatch.setattr(F.urllib.request, "urlopen", lambda url, timeout=5: type("R", (), {"read": lambda s: b"jpg"})())
    monkeypatch.setattr(F, "FILM_HZ", 50.0)
    f = F.Film(root=str(tmp_path)).start()
    import time
    time.sleep(0.2)
    out, n = f.stop()
    assert os.path.isdir(out) and n > 0
    assert not any(t.is_alive() for t in f._threads)                   # stopped when told, not after a guessed time


def test_a_base_camera_reading_at_the_hover_is_not_the_close_look(monkeypatch):
    """At 270 mm: at the hover only the webcam saw the block; that reading was taken as the close look,
    the back-off look and the hunt never ran, and the fingers closed short of it. Now it goes into the filter and the
    grab goes looking for it with the gripper camera."""
    import armlab.modes.grasp as G
    assert G.gripper_sees([("gripper", 7, None)]) and not G.gripper_sees([("webcam", 7, None), ("gripper", 31, None)])
    base_only = {"xyz": np.array([195.0, 120.0, -200.0]), "yaw_deg": 18.0, "gripper": False, "cams": {"webcam": 1.0}}
    calls = []
    monkeypatch.setattr(G, "look_up_close", lambda w, cube: (dict(base_only), [0.0] * 4))
    monkeypatch.setattr(G, "back_off_look", lambda *a: calls.append("back off") or None)
    monkeypatch.setattr(G, "hunt_for_tag", lambda *a, **k: calls.append("hunt") or {"xyz": np.array([196.0, 121.0, -200.0]),
                                                                                    "yaw_deg": 18.0, "cams": {"gripper": 1.0}, "close": True})
    blk = G.BlockFilter()
    blk.see({"xyz": np.array([195.0, 120.0, -200.0]), "yaw_deg": 18.0, "cams": {"torre": 1.0}})
    w = type("W", (), {"cams": {"gripper": {}}})()
    near, dq = G.look_close(w, None, None, base_only, {"hover": [0] * 6}, blk, 15.0)
    assert calls == ["back off", "hunt"] and near is not None and blk.looks == 3   # webcam reading + hunt both counted


def test_plan_best_takes_a_straight_way_in_at_a_lower_hover_over_a_joint_one_higher(monkeypatch):
    """At 270 mm: the ladder starts high, and the new world model accepted a 60 mm "elbow then wrist"
    hover before reaching the straight "from above" one at 25 mm. The gripper camera read the block from 0 of 4
    joint-retreat hovers (10 of 10 straight), and all 4 of those grabs missed. Straight first, at every hover."""
    import armlab.grasp as G

    def plan(w, check, cube, margin, hover_mm=60.0, pull=0.0, **kw):
        a = G.JOINT_APPROACH if hover_mm > 30 else "from above"
        return {"approach": a, "sees": True, "hover_mm": hover_mm, "path": [[30.0, 80.0, 60.0]], "room": 5.0}
    monkeypatch.setattr(G, "plan", plan)
    monkeypatch.setattr(G, "limit_room", lambda q: 5.0)
    got = G.plan_best(None, None, {"xyz": np.zeros(3)}, 15.0, ladder=(60.0, 45.0, 25.0), hover_floor=0.0, depth=10.0)
    assert got["approach"] == "from above" and got["hover_mm"] == 25.0
    monkeypatch.setattr(G, "plan", lambda *a, hover_mm=60.0, **k: dict(plan(*a, hover_mm=60.0, **k), hover_mm=hover_mm))
    got = G.plan_best(None, None, {"xyz": np.zeros(3)}, 15.0, ladder=(60.0, 45.0, 25.0), hover_floor=0.0, depth=10.0)
    assert got["approach"] == G.JOINT_APPROACH and got["hover_mm"] == 60.0     # no straight one anywhere: the joint one


def test_hand_rel_grabs_on_the_gripper_cameras_reading_without_slack(monkeypatch):
    """--hand-rel: the slack from 2-5 arm tags a camera swung 2-7 deg between cameras on the same pose. The
    slack comes from the tower's arm tags alone (none when it sees under 2), and when the gripper camera sees the block
    the grab uses its reading taken with no slack and goes in with none: camera and fingers ride the same hand."""
    import armlab.modes.grasp as G
    from armlab.state import S
    monkeypatch.setattr(S, "hand_rel", True, raising=False)
    got = []
    monkeypatch.setattr(G, "arm_slack", lambda w, a, off, seen: got.append([s[0] for s in seen]) or
                        (np.array([1.0, 2.0, 3.0, 4.0]), 2.0, len(seen)))
    seen = [("torre", 320, None), ("webcam", 321, None), ("webcam", 9, None)]
    dq, _, n = G.slack_now(None, [0] * 6, 0.0, seen)
    assert got[-1] == ["torre"] and not dq.any()                             # one tower tag: no correction
    dq, _, n = G.slack_now(None, [0] * 6, 0.0, seen + [("torre", 360, None)])
    assert got[-1] == ["torre", "torre"] and dq[1] == 2.0
    hr = {"xyz": np.array([301.0, 2.0, -150.0]), "yaw_deg": 0.0, "cams": {"gripper": 1.0}, "close": True}
    near = dict(hr, xyz=np.array([300.0, 0.0, -150.0]), gripper=True, hand_rel=hr)
    monkeypatch.setattr(G, "look_up_close", lambda w, cube: (dict(near), np.array([1.0, 2.0, 3.0, 4.0])))
    blk = G.BlockFilter()
    out, dq = G.look_close(type("W", (), {"cams": {"gripper": {}}})(), None, None, near, {"hover": [0] * 6}, blk, 15.0)
    assert out is hr and not dq.any() and blk.looks == 1                    # the filter still gets the usual reading


def test_a_set_down_never_leaves_the_block_at_a_turn_the_pickup_cannot_square_to(monkeypatch):
    """A block at 310 mm whose faces sat 35-40 deg off the arm's bearing was out of reach at every hover (the
    wrist rotate cannot square to either face there). A set-down must leave it pickable at its predicted turn and
    PICKABLE_SPREAD_DEG either side (the release sometimes twists it)."""
    import armlab.modes.grasp as G
    import armlab.grasp as P
    plan = {"face_deg": 100.0, "spot": [300.0, 0.0, -150.0]}              # -> the block at +10 deg
    assert G.turn_after_set_down(plan) == 10.0
    dead = lambda w, t, face, grip: [] if 12.0 <= (np.degrees(face) % 90.0) <= 20.0 else [1]
    monkeypatch.setattr(P, "reach_all", dead)
    assert not G.pickable_after(None, {}, {"grip": 55.0, "depth_mm": 10.0}, plan)      # +16 falls in the dead band
    assert G.pickable_after(None, {}, {"grip": 55.0, "depth_mm": 10.0}, dict(plan, face_deg=130.0))  # +40: clear of it


def test_plan_best_loosens_squareness_only_when_nothing_squares(monkeypatch):
    """No plan within SQUARE_TOL_DEG -> the same again (the squeeze squares the block up) at
    SQUARE_TOL_LOOSE_DEG, marked so the re-plans keep to it; the tolerance is put back either way."""
    import armlab.grasp as G
    seen = []

    def inner(w, check, cube, margin, **kw):
        seen.append(G.SQUARE_TOL_DEG)
        return {"why": ["out of reach"]} if G.SQUARE_TOL_DEG < 15 else {"approach": "from above", "off_square_deg": 14.8}
    monkeypatch.setattr(G, "_plan_best", inner)
    got = G.plan_best(None, None, {}, 15.0)
    assert seen == [12.0, G.SQUARE_TOL_LOOSE_DEG] and got["square_tol"] == G.SQUARE_TOL_LOOSE_DEG
    assert G.SQUARE_TOL_DEG == 12.0
    seen.clear()
    monkeypatch.setattr(G, "_plan_best", lambda *a, **k: seen.append(G.SQUARE_TOL_DEG) or {"approach": "from above"})
    assert "square_tol" not in G.plan_best(None, None, {}, 15.0) and seen == [12.0]     # square enough: untouched


def test_trace_records_spans_and_samples():
    """Every traced span keeps its start,
    end, kind and name, even when what it wraps raises; samples keep the joints with the time."""
    from armlab import trace
    trace.reset()
    with trace.span("arm", "to 1 2 3"):
        trace.sample([1, 2, 3, 42, 80, -10])
    try:
        with trace.span("wait", "settle"):
            raise ValueError("boom")
    except ValueError:
        pass
    d = trace.dump()
    assert [e[2:] for e in d["events"]] == [["arm", "to 1 2 3"], ["wait", "settle"]]
    assert d["samples"][0][1:] == [1.0, 2.0, 3.0, 42.0, 80.0, -10.0] and all(e[1] >= e[0] for e in d["events"])
    trace.reset()
    d = trace.dump()
    assert d["events"] == [] and d["samples"] == [] and isinstance(d["cpu"], list)   # cpu: sampled on Linux only


def test_version_stamp_names_the_code_and_data(tmp_path, monkeypatch):
    """Run files must say what they ran. Every armlab .py hashed, one build hash over them, the data files."""
    from armlab import version
    from armlab.state import S
    monkeypatch.setattr(S, "data_dir", str(tmp_path))
    (tmp_path / "world.json").write_text("{}")
    st = version.stamp()
    assert "modes/show.py" in st["code"] and "grasp.py" in st["code"] and len(st["build"]) == 10
    assert st["data"]["world.json"] and st["data"]["plan_table.json"] is None
    assert version.stamp()["build"] == st["build"]                       # same code, same build


def test_the_base_turns_toward_the_block_while_the_plan_finishes(monkeypatch):
    """The base starts toward the hover as soon as the looks are done: from the search pose or up high
    the base turns to base_toward(block) while the plan computes, its sweep checked against the block; from elsewhere
    it does not. Open loop unless --hover-base-closed (the close look measures where it really is)."""
    import armlab.modes.grasp as G
    import armlab.grasp as P
    from armlab.state import S
    log = []
    monkeypatch.setattr(S, "base_cmd", -30.0, raising=False)
    monkeypatch.setattr(P, "base_toward", lambda w, t: 10.0)
    monkeypatch.setattr(G, "move_base", lambda b, closed=True, settle=True: log.append((b, closed)))
    monkeypatch.setattr(G, "cube_gap", lambda check, cube, q5, b: 80.0)
    monkeypatch.setattr(G, "arm_angles", lambda: list(G.SEARCH_POSE) + [42.0, 80.0, -30.0])
    G.TURNED.clear()
    cube = {"xyz": np.array([300.0, 50.0, -150.0])}
    assert G.turn_toward(None, None, cube) == 10.0 and log == [(10.0, False)] and G.TURNED["base"] == 10.0
    log.clear()
    monkeypatch.setattr(G, "HOVER_BASE_CLOSED", True)
    assert G.turn_toward(None, None, cube) == 10.0 and log == [(10.0, True)]
    log.clear()
    monkeypatch.setattr(G, "cube_gap", lambda check, cube, q5, b: 80.0 if b < 0 else 5.0)    # the far end comes close
    assert G.turn_toward(None, None, cube) is None and not log
    monkeypatch.setattr(G, "arm_angles", lambda: [5.8, 75.0, 60.0, 42.0, 80.0, -30.0])         # at home: not from here
    monkeypatch.setattr(G, "cube_gap", lambda check, cube, q5, b: 80.0)
    assert G.turn_toward(None, None, cube) is None and not log


def test_close_blocks_are_aimed_a_few_mm_toward_the_tower():
    """Close in, the fingers closed on the block's far edge though the gripper camera saw it. Blocks
    nearer than CLOSE_IN_FROM_MM get the final aim moved --close-in mm toward the base axis; far ones are untouched."""
    import armlab.modes.grasp as G
    w = type("W", (), {"axis": np.zeros(3)})()
    near = {"xyz": np.array([300.0, 0.0, -150.0]), "yaw_deg": 5.0}
    got = G.aim_in_close(w, near, 4.0)
    assert np.allclose(got["xyz"], [296.0, 0.0, -150.0]) and np.allclose(near["xyz"], [300.0, 0.0, -150.0])
    assert G.aim_in_close(w, {"xyz": np.array([350.0, 0.0, -150.0])}, 4.0)["xyz"][0] == 350.0     # far: unchanged
    assert G.aim_in_close(w, near, 0.0) is near                                                   # 0: off


def test_plan_best_pulls_the_aim_in_for_a_straight_way_in_before_taking_a_joint_one(monkeypatch):
    """In a sweep at 270 mm, 7 of 25 blocks got the joint-space way in at pull 0 while every one of them plans a
    straight way in aimed 5-10 mm nearer the tower. A joint-space plan must not stop the pull ladder."""
    import armlab.grasp as G

    def plan(w, check, cube, margin, hover_mm=60.0, pull=0.0, **kw):
        a = "from above" if pull >= 5.0 else G.JOINT_APPROACH
        return {"approach": a, "sees": True, "hover_mm": hover_mm, "path": [[30.0, 80.0, 60.0]], "room": 5.0,
                "pull_mm": pull}
    monkeypatch.setattr(G, "plan", plan)
    monkeypatch.setattr(G, "limit_room", lambda q: 5.0)
    got = G.plan_best(None, None, {"xyz": np.zeros(3)}, 15.0, ladder=(60.0, 45.0), hover_floor=0.0, depth=10.0)
    assert got["approach"] == "from above" and got["pull_mm"] == 5.0


def test_the_gripper_picture_ends_where_the_hand_starts_hiding_it(monkeypatch):
    """The gripper camera read the block's tag 48 of 49 times with its lowest corner above 500 px and 5 of
    117 below 650 -- the hand covers the bottom of its picture. in_view stops the picture at GRIPPER_USABLE_Y_PX."""
    import armlab.grasp as G
    K = np.array([[1000.0, 0, 576.0], [0, 1000.0, 432.0], [0, 0, 1.0]])
    w = type("W", (), {"cams": {"gripper": {"K": K}}})()
    w.camera_pose = lambda cam, q, b: (np.eye(3), np.zeros(3))
    w.project = lambda cam, X, q, b: np.asarray(X)[:, :2] / np.asarray(X)[:, 2:] * 1000.0 + [576.0, 432.0]
    at = lambda v: np.array([[0.0, v, 1000.0]] * 4)                     # all four corners at picture row 432 + v
    assert G.in_view(w, at(100.0), [0] * 6)                              # row 532: above the line
    assert not G.in_view(w, at(150.0), [0] * 6)                          # row 582: the hand hides it
    monkeypatch.setattr(G, "GRIPPER_USABLE_Y_PX", None)
    assert G.in_view(w, at(150.0), [0] * 6)                              # None: the whole picture, as before


def test_a_set_down_needs_a_straight_pickup_in_the_plan_table(monkeypatch):
    """A set-down where the plan table has only the joint-space way in (0 held of 9 on record) is passed over."""
    import armlab.modes.grasp as G
    import armlab.grasp as P
    t = {"plans": [{"xy": [300.0, 0.0], "yaw": 7.5, "plan": {"approach": "from above"}},
                   {"xy": [260.0, 0.0], "yaw": 7.5, "plan": {"approach": P.JOINT_APPROACH}}]}
    monkeypatch.setattr(G, "plan_table", lambda: t)
    assert G.straight_pickup_near([305.0, 5.0], 10.0) is True
    assert G.straight_pickup_near([305.0, 5.0], 30.0) is False           # a turn the table has no straight plan for
    assert G.straight_pickup_near([262.0, 0.0], 7.5) is False            # only the joint-space one there
    monkeypatch.setattr(G, "plan_table", lambda: None)
    assert G.straight_pickup_near([262.0, 0.0], 7.5) is None             # no table: nothing to say


def test_the_base_turns_for_the_replanned_grasp_only_when_it_is_worth_it(monkeypatch):
    """The re-plan after the close look solves the base, the way in moves only shoulder/elbow/wrist: a base change of
    BASE_FIX_DEG or more is made at the hover (closed loop), unless the turn would come closer to the block."""
    import armlab.modes.grasp as G
    from armlab.state import S
    turned = []
    monkeypatch.setattr(G, "move_base", lambda t, *a, **k: turned.append(t))
    monkeypatch.setattr(G, "arm_angles", lambda: [50.0, 60.0, 40.0, 42.0, 55.0, 10.0])
    monkeypatch.setattr(S, "base_cmd", 10.0)
    monkeypatch.setattr(G, "cube_gap", lambda check, cube, q5, b: 20.0)
    G.LAST_LOOK.clear()
    G.turn_base_for(None, object(), {"xyz": np.zeros(3)}, {"grasp": [0, 0, 0, 0, 0, 10.3]})
    assert turned == []                                                   # 0.3 deg: not worth a turn
    G.LAST_LOOK.update(base=10.0, off=-0.4, fit=3.0, t=G.time.time())   # the close look: really at 9.6
    G.turn_base_for(None, object(), {"xyz": np.zeros(3)}, {"grasp": [0, 0, 0, 0, 0, 10.3]})
    assert turned == [10.3]
    turned.clear()
    G.LAST_LOOK.clear()
    G.turn_base_for(None, object(), {"xyz": np.zeros(3)}, {"grasp": [0, 0, 0, 0, 0, 11.0]})
    assert turned == [11.0]
    monkeypatch.setattr(G, "cube_gap", lambda check, cube, q5, b: 20.0 - abs(b - 10.0) * 10.0)
    G.turn_base_for(None, object(), {"xyz": np.zeros(3)}, {"grasp": [0, 0, 0, 0, 0, 11.0]})
    assert turned == [11.0]                                               # the turn closes on the block: not made


def test_a_set_down_aims_nearer_the_base_by_what_set_downs_overshoot(monkeypatch):
    """Over 32 show set-downs the block landed 14.5 mm + 0.79 mm per degree of steeper hand further out than the
    spot. place_plan aims the hand that much nearer the base axis; the plan keeps the spot itself."""
    import armlab.modes.grasp as G
    import armlab.grasp as P
    aims = []

    def reach(w, target, pitch, face, grip, near=None, ws=1.0, wp=1.0):
        aims.append((np.array(target, float), pitch))
        return [0.0, 0.0, 0.0, 56.0, 55.0, 0.0]
    monkeypatch.setattr(P, "reach", reach)
    monkeypatch.setattr(P, "way_in", lambda *a, **k: {"sees": True, "path": [[0] * 6], "grasp": [30.0, 60.0, 40.0]})
    monkeypatch.setattr(P, "base_toward", lambda w, t: 0.0)
    monkeypatch.setattr(G, "settable", lambda *a: True)
    monkeypatch.setattr(G, "PLACE_AIM_IN_MM", 14.5)
    monkeypatch.setattr(G, "PLACE_AIM_IN_PER_MM", 0.0)                     # 0 since the firm-set-down finding
    w = type("W", (), {"axis": np.zeros(3), "table_height": lambda self, x, y: -250.0})()
    near = {"xyz": np.array([300.0, 0.0, -199.2])}
    p2 = {"grasp": [0] * 6, "approach": "from above", "pitch": 60.0, "face_deg": 0.0, "depth_mm": 10.0}
    got = G.place_plan(w, None, [250.0, 0.0], 0.0, near, p2, 15.0, 12.0, None, 0.0)
    target, pitch = aims[0]
    assert pitch == 60.0 and abs(target[0] - (250.0 - G.PLACE_AIM_IN_MM)) < 1e-6 and abs(target[1]) < 1e-9
    assert got["spot"][:2] == [250.0, 0.0]


def test_a_blind_plan_the_grab_would_refuse_does_not_stop_the_pull_ladder(monkeypatch):
    """With the gripper picture ending where the hand hides it, blocks at 381-385 mm had a blind 60 mm hover at
    pull 0 that failed the room test; kept anyway it stopped the pull ladder and no plan came back, where a 5 mm pull
    gives one the camera sees."""
    import armlab.grasp as G

    def plan(w, check, cube, margin, hover_mm=60.0, pull=0.0, **kw):
        return {"approach": "from above", "sees": pull >= 5.0, "hover_mm": hover_mm, "path": [[pull, 80.0, 60.0]],
                "room": 5.0, "pull_mm": pull}
    monkeypatch.setattr(G, "plan", plan)
    monkeypatch.setattr(G, "limit_room", lambda q: 0.1 if q[0] == 0.0 else 5.0)     # the pull-0 plan has no room
    got = G.plan_best(None, None, {"xyz": np.zeros(3)}, 15.0, ladder=(60.0,), hover_floor=0.0, depth=10.0)
    assert "why" not in got and got["sees"] and got["pull_mm"] == 5.0


def test_the_compiled_solve_gives_the_python_answer():
    """fastik.py (numba) is grasp.solve compiled -- same kinematics, same steps. A from-scratch plan was 96%
    solve(); compiled it is ~15x faster. Where numba is missing the test has nothing to compare."""
    import math
    import armlab.grasp as G
    from armlab import fastik
    if fastik.numba is None:
        return
    w = make_finger_world()
    assert fastik.params(w) is not None                      # the compiled path really runs here
    rng = np.random.default_rng(3)
    for _ in range(40):
        q0 = [rng.uniform(80, 140), rng.uniform(40, 100), rng.uniform(30, 110), rng.uniform(0, 112), 60.0,
              rng.uniform(-30, 30)]
        target = G.hand(w, q0)[0] + rng.normal(0, 10, 3)
        start = [110.0, 70.0, 60.0, 42.0, 60.0, q0[5]]
        args = (w, target, rng.uniform(30, 80), math.radians(rng.uniform(-90, 90)), 60.0, start)
        G.FAST_IK = False
        qp, rp = G.solve(*args, wp=0.3, ws=0.3)
        G.FAST_IK = True
        qf, rf = G.solve(*args, wp=0.3, ws=0.3)
        assert max(abs(a - b) for a, b in zip(qp, qf)) < 1e-6 and np.abs(np.asarray(rp) - rf).max() < 1e-6


def test_a_set_down_is_firm_only_with_the_shoulder_low_and_room_at_every_joint():
    """Set-downs with the shoulder at 56-69 or a joint at its end landed 20-61 mm out from where the
    fingers let go; with 10+ steps of room, within 7 mm. place_plan keeps those for a second pass (tight=True)."""
    import armlab.modes.grasp as G
    assert G.firm([30.0, 60.0, 40.0])
    assert not G.firm([60.0, 90.0, 40.0])                               # shoulder high
    assert not G.firm([30.0, 60.0, 3.0])                                # wrist at its end


def test_after_a_joint_space_set_down_the_fingers_lift_straight_up():
    """Seen on film: lifting back along the joint-space arc after letting go dragged the block 14-60 mm
    outward. vertical_lift: poses straight up from the release, only shoulder/elbow/wrist moving."""
    import armlab.grasp as G
    import armlab.modes.grasp as MG
    w = make_finger_world()
    q0 = [110.0, 70.0, 60.0, 50.0, 100.0, 12.0]
    lift = MG.vertical_lift(w, q0, 100.0, 20.0)
    assert lift is not None and len(lift) == 4
    tip0 = G.hand(w, q0)[0]
    for k, q in enumerate(lift, 1):
        tip = G.hand(w, q)[0]
        assert abs(tip[2] - tip0[2] - 5.0 * k) < 0.5 and np.linalg.norm((tip - tip0)[:2]) < MG.LIFT_SIDEWAYS_MM
        assert q[3:] == [50.0, 100.0, 12.0]                              # rotate, grip, base untouched
    assert MG.release_lift(w, None, {"grasp": q0}, None, 0.0) is None    # no table check: the old way out


def test_a_grab_logs_the_finger_spacing_at_the_bottom(monkeypatch):
    """Logged only: the spacing just after closing goes into SPACING (and the run file) for deciding later whether
    one read can replace the squeeze."""
    G, out, moves, grips = _grab_rig(monkeypatch)
    assert G.SPACING.get("bottom") == 64.0


def test_a_joint_space_way_in_ends_straight_down_onto_the_grasp():
    """The plain joint-space way in slid the fingertips ~15 mm along the face in its last 10 mm (0 of
    45 held). joint_ways_in_tail: the retreat above, the last TAIL_MM straight down onto the grasp."""
    import armlab.grasp as G
    w = make_finger_world()
    g = [110.0, 70.0, 60.0, 50.0, 55.0, 12.0]
    tip_g = G.hand(w, g)[0]
    paths = list(G.joint_ways_in_tail(w, tip_g, g, 55.0, 60.0))
    assert paths
    for path in paths:
        assert path[-1] == list(g)
        tail = [G.hand(w, q)[0] for q in path if G.hand(w, q)[0][2] - tip_g[2] <= 10.5]
        assert len(tail) >= 3 and max(np.linalg.norm((t - tip_g)[:2]) for t in tail) < 1.0


def test_straight_up_stays_inside_the_joint_limits(monkeypatch):
    """The three arm joints have one to spare on a vertical line: it is spent away from the joint nearest its end."""
    import armlab.grasp as G
    w = make_finger_world()
    q0 = [110.0, 70.0, 60.0, 50.0, 55.0, 12.0]
    free = G.straight_up(w, q0, 55.0, 15.0)
    assert free is not None
    lo = list(G.JOINT_LO)
    lo[2] = min(q[2] for q in free) + 2.0                                 # a wrist floor the free answer goes under
    monkeypatch.setattr(G, "JOINT_LO", tuple(lo))
    got = G.straight_up(w, q0, 55.0, 15.0)
    if got is not None:                                                    # when it can, it keeps the wrist above it
        assert all(q[2] >= lo[2] - 1e-9 for q in got)


def test_the_lift_slows_only_beside_the_block():
    """Holding the block: PICK_SLOW until the fingertip is LIFT_SLOW_MM above the grasp, CARRY_SPEED above that."""
    import armlab.grasp as P
    import armlab.modes.grasp as G
    w = make_finger_world()
    g = [110.0, 70.0, 60.0, 50.0, 55.0, 12.0]
    path = [list(q) for q in reversed([g] + P.straight_up(w, g, 55.0, 40.0))]   # hover first, 40 mm straight up
    k = G.lift_clear_index(w, path)
    z = [P.hand(w, q)[0][2] for q in path]
    assert z[k] - z[-1] >= G.LIFT_SLOW_MM and z[k + 1] - z[-1] < G.LIFT_SLOW_MM
    assert G.lift_clear_index(None, path) == 0                           # no world: all of it slow, as before


def test_pan_move_checks_the_middle_of_the_grid(monkeypatch):
    """Arm and base moving together: a spot that is only tight with the arm half-way AND the base half-way is
    refused, though either order one after the other (the grid's edges) is clear."""
    import armlab.modes.grasp as G
    from armlab.errors import ArmError

    class FakeZone:                                    # tight only around shoulder 50 with the base at 20
        margin = 30.0
        def gaps(self, rows):
            rows = np.atleast_2d(rows)
            return 100.0 - 80.0 * np.exp(-((rows[:, 0] - 50.0) ** 2 + (rows[:, 5] - 20.0) ** 2) / 20.0)
    z, hand_ = FakeZone(), (42.0, 80.0)
    monkeypatch.setattr(G, "cube_gap", lambda check, cube, q5, b: 80.0)
    a, b = [30.0, 75.0, 60.0], [70.0, 75.0, 60.0]
    for q in (a, b):                                   # base turn at either end: clear
        assert z.gaps([q + list(hand_) + [bb] for bb in np.linspace(0.0, 40.0, 41)]).min() > z.margin
    for bb in (0.0, 40.0):                             # arm move at either base: clear
        assert z.gaps([list(q) + list(hand_) + [bb] for q in G.arm_path(a, b, 1.0)]).min() > z.margin
    try:
        G.check_pan_move(z, None, {"xyz": [300.0, 0.0, -150.0]}, a, b, hand_, 0.0, 40.0)
        assert False, "the middle of the grid should be refused"
    except ArmError as e:
        assert "base turning +0.0 -> +40.0" in str(e)
        assert abs(float(str(e).split("base")[-1]) - 20.0) <= G.PAN_BASE_STEP_DEG      # the nearest grid point to it
    assert G.check_pan_move(z, None, None, a, b, hand_, 0.0, 6.0) > z.margin    # a short turn clear of it: allowed


def test_pan_move_block_rules(monkeypatch):
    import armlab.modes.grasp as G
    from armlab.errors import ArmError

    class OpenZone:
        margin = 30.0
        def gaps(self, rows):
            return np.full(len(np.atleast_2d(rows)), 90.0)
    a, b, hand_, cube = [36.6, 75.0, 60.0], [60.0, 90.0, 70.0], (42.0, 80.0), {"xyz": [300.0, 0.0, -150.0]}
    # the block gap falls as the arm goes out and as the base turns toward it: the corner (both at the end) is the
    # nearest, 15 mm -- an arrival at the hover (>= 12 at the end, never nearer than the end) is fine, a plain move is not
    monkeypatch.setattr(G, "cube_gap", lambda check, c, q5, base: 75.0 - (q5[0] - 36.6) * 1.5 - base)
    assert G.check_pan_move(OpenZone(), None, cube, a, b, hand_, 0.0, 25.0, arriving=True) == 90.0
    try:
        G.check_pan_move(OpenZone(), None, cube, a, b, hand_, 0.0, 25.0)
        assert False, "15 mm from the block on a plain move should be refused"
    except ArmError as e:
        assert "from the block" in str(e)
    # nearest in the middle of the grid only (the edges stay 40+ mm away): refused even as an arrival
    monkeypatch.setattr(G, "cube_gap", lambda check, c, q5, base: 40.0 if abs(q5[0] - 48.0) > 3 or abs(base - 12.0) > 3
                        else 5.0)
    try:
        G.check_pan_move(OpenZone(), None, cube, a, b, hand_, 0.0, 25.0, arriving=True)
        assert False
    except ArmError as e:
        assert "5 mm from the block" in str(e)


def test_pan_move_uses_the_zone_and_exact_table_at_every_base(monkeypatch):
    """With the real Zone and the exact table check, the grid's answer is the smallest zone.gap over every
    (arm point, base angle) pair -- the same check a single pose gets."""
    import armlab.modes.grasp as G
    from armlab import safety
    from armlab.state import S
    w = make_finger_world()
    d = tempfile.mkdtemp()
    __import__("shutil").copy(w_path_of(w), os.path.join(d, "world.json"))
    make_table_model(w, os.path.join(d, "table_model.json"))
    ax = np.arange(0, 161, 40.0)
    full = np.full((len(ax), len(ax), len(ax)), 200, np.uint8)
    np.savez(os.path.join(d, "arm_map.npz"), s=ax, e=ax, w=ax, base=full, tower=full, self=full,
             low=np.full(full.shape, 500, np.int16), meta=json.dumps({"made": "test"}))
    monkeypatch.setattr(S, "data_dir", d)
    monkeypatch.setattr(S, "exact_table", True)
    monkeypatch.setattr(S, "base_cmd", 10.0)
    z = safety.Zone(margin=0.0)
    a, b, hand_, b0, b1 = [120.0, 70.0, 60.0], [100.0, 70.0, 60.0], (42.0, 80.0), -20.0, 30.0
    want = min(z.gap(list(q) + list(hand_) + [bb]) for q in G.arm_path(a, b, G.PAN_ARM_STEP_DEG)
               for bb in np.linspace(b0, b1, int(abs(b1 - b0) / G.PAN_BASE_STEP_DEG) + 2))
    assert abs(G.check_pan_move(z, None, None, a, b, hand_, b0, b1) - want) < 1e-6


def test_cube_gap_compiled_points_match_the_python_ones(monkeypatch):
    import armlab.modes.grasp as G
    from armlab import fastik
    from armlab.tablecheck import TableCheck
    if fastik.numba is None:
        return
    w = make_finger_world()
    path = os.path.join(tempfile.mkdtemp(), "table_model.json")
    make_table_model(w, path)
    t = TableCheck(w, path)
    rng, cube = np.random.default_rng(3), {"xyz": [250.0, 40.0, -150.0]}
    for _ in range(30):
        q5 = [rng.uniform(60, 140), rng.uniform(40, 100), rng.uniform(40, 90), rng.uniform(20, 90), rng.uniform(40, 150)]
        b = rng.uniform(-60, 90)
        assert np.allclose(fastik.points(t, q5, b), t.points(q5, b), atol=1e-6)
        fast = G.cube_gap(t, cube, q5, b)
        monkeypatch.setattr(fastik, "points", lambda *a: None)
        assert abs(G.cube_gap(t, cube, q5, b) - fast) < 1e-6
        monkeypatch.undo()


def test_pan_move_sends_the_base_then_the_arm_then_the_finish_leg(monkeypatch):
    """motion.pan_move: the base's long leg is sent (not waited for), the arm goes by move_to (its checks and live
    watcher), then it waits for the base, and move_base does the + finish leg -- move_base is not used for the long
    leg (its "arm moved during a base-only move" guard)."""
    from armlab import motion as M
    from armlab.state import S
    log, base = [], {"at": 0.0}
    monkeypatch.setattr(S, "base_speed_cmd", True, raising=False)
    monkeypatch.setattr(S, "esp_url", "http://esp", raising=False)

    def http(url, **k):
        log.append(("http", url))
        if "a=,,,,," in url:
            base["at"] = float(url.split("a=,,,,,")[1].split("&")[0])   # the fake base gets there at once
        return {}
    monkeypatch.setattr(M, "http_json", http)
    monkeypatch.setattr(M, "base_angle", lambda: base["at"])
    monkeypatch.setattr(M, "arm_angles", lambda: [50.0, 80.0, 60.0, 42.0, 80.0, base["at"]])
    monkeypatch.setattr(M, "move_to", lambda t, zone, speed=0, slow_finish=False:
                        log.append(("arm", list(t), base["at"])) or [50.0, 80.0, 60.0, 42.0, 80.0, base["at"]])
    monkeypatch.setattr(M, "move_base", lambda b, closed=None, settle=True, **k: log.append(("finish", b, closed, settle)))
    M.pan_move([50.0, 80.0, 60.0], 30.0, None, 20.0, closed=False, settle=False)
    assert log[0][0] == "http" and "a=,,,,,28.500" in log[0][1] and "&bs=" in log[0][1]   # long leg, smooth, first
    assert log[1] == ("arm", [50.0, 80.0, 60.0], 28.5)
    assert log[2] == ("finish", 30.0, False, False) and len(log) == 3
    # the arm move refused or stopped part way: the base, still turning, is stopped too
    log.clear()
    stops = []
    monkeypatch.setattr(M, "arm_stop", lambda: stops.append(1))

    def refuse(*a, **k):
        raise M.ArmError("refused: test")
    monkeypatch.setattr(M, "move_to", refuse)
    try:
        M.pan_move([50.0, 80.0, 60.0], 10.0, None)
        assert False
    except M.ArmError:
        pass
    assert stops and not any(e[0] == "finish" for e in log)
    # old firmware (no smooth base moves): base, then arm, one after the other
    log.clear()
    monkeypatch.setattr(S, "base_speed_cmd", False)
    monkeypatch.setattr(M, "move_to", lambda t, zone, speed=0, slow_finish=False: log.append(("arm", list(t))))
    M.pan_move([50.0, 80.0, 60.0], 5.0, None)
    assert [e[0] for e in log] == ["finish", "arm"]


def _hover_setup(monkeypatch, log):
    import armlab.modes.grasp as G
    from armlab.modes import survey
    from armlab.state import S
    monkeypatch.setattr(S, "base_cmd", 0.0, raising=False)
    monkeypatch.setattr(G, "arm_angles", lambda: [5.8, 75.0, 60.0, 42.0, 80.0, 0.0])
    monkeypatch.setattr(G, "check_arm_move", lambda *a, **k: log.append("check arm"))
    monkeypatch.setattr(G, "check_base_turn", lambda *a, **k: log.append("check base"))
    monkeypatch.setattr(G, "cube_gap", lambda *a, **k: 80.0)
    monkeypatch.setattr(survey, "exact_ok", lambda *a, **k: True)
    monkeypatch.setattr(survey, "set_hand", lambda *a, **k: None)
    monkeypatch.setattr(G, "move_to", lambda t, zone, speed=0, slow_finish=False: log.append(("arm", list(t[:3]))))
    monkeypatch.setattr(G, "move_base", lambda b, **k: log.append(("base", b)))
    monkeypatch.setattr(G, "pan_move", lambda t, b, zone, speed=0, **k: log.append(("pan", list(t[:3]), b)))
    monkeypatch.setattr(G, "pause", lambda *a, **k: None)
    monkeypatch.setattr(G, "wait_fresh", lambda *a, **k: None)
    G.TURNED.clear()

    class Z:
        def contains(self, p):
            return True

        def path_ok(self, a, b):
            return True
    hover = [60.0, 90.0, 70.0, 50.0, 60.0, 25.0]
    return G, Z(), {"xyz": [300.0, 0.0, -150.0]}, {"hover": hover}


def test_go_to_hover_pan_overlap(monkeypatch):
    log = []
    G, z, cube, p = _hover_setup(monkeypatch, log)
    called = []
    monkeypatch.setattr(G, "check_pan_move", lambda *a, **k: called.append((a, k)) or 90.0)
    # the flag off: exactly the old moves, and no pan check at all
    monkeypatch.setattr(G, "PAN_OVERLAP", False)
    G.go_to_hover(z, None, cube, p, 20.0)
    assert [e for e in log if isinstance(e, tuple)] == [("arm", G.HIGH_POSE), ("base", 25.0), ("arm", [60.0, 90.0, 70.0])]
    assert not called
    # on: up, then across while the base turns, checked with the fingers opening (80 -> 40) as an arrival
    log.clear()
    monkeypatch.setattr(G, "PAN_OVERLAP", True)
    G.go_to_hover(z, None, cube, p, 20.0)
    assert [e for e in log if isinstance(e, tuple)] == [("arm", G.HIGH_POSE), ("pan", [60.0, 90.0, 70.0], 25.0)]
    a, k = called[0]
    assert [h[1] for h in a[5]] == [80.0, 60.0, 40.0] and (a[6], a[7]) == (0.0, 25.0) and k["arriving"]
    assert "check arm" in log and "check base" in log            # the one-after-the-other checks still run first
    # refused: one after the other, as with the flag off
    log.clear()

    def refuse(*a, **k):
        raise G.ArmError("refused: test grid")
    monkeypatch.setattr(G, "check_pan_move", refuse)
    G.go_to_hover(z, None, cube, p, 20.0)
    assert [e for e in log if isinstance(e, tuple)] == [("arm", G.HIGH_POSE), ("base", 25.0), ("arm", [60.0, 90.0, 70.0])]
    # closed-loop base to the hover: its nudges would come over the block, so no overlap
    log.clear()
    monkeypatch.setattr(G, "check_pan_move", lambda *a, **k: 90.0)
    monkeypatch.setattr(G, "HOVER_BASE_CLOSED", True)
    G.go_to_hover(z, None, cube, p, 20.0)
    assert ("pan", [60.0, 90.0, 70.0], 25.0) not in log


def test_go_home_to_the_park_pose_pan_overlap(monkeypatch):
    log = []
    G, z, cube, p = _hover_setup(monkeypatch, log)
    monkeypatch.setitem(G.LAST_BLOCK, "xyz", [300.0, 0.0, -150.0])
    park = ([40.0, 70.0, 55.0], 45.0, "torre")
    calls = []
    monkeypatch.setattr(G, "check_pan_move", lambda *a, **k: calls.append(k) or 90.0)
    old = [("arm", G.HIGH_POSE), ("base", 45.0), ("arm", [40.0, 70.0, 55.0])]
    monkeypatch.setattr(G, "PAN_OVERLAP", False)
    G.go_home(z, -10.0, 20.0, check=object(), stay_up=True, park=park)
    assert [e for e in log if isinstance(e, tuple)] == old and not calls
    log.clear()
    monkeypatch.setattr(G, "PAN_OVERLAP", True)
    G.go_home(z, -10.0, 20.0, check=object(), stay_up=True, park=park)
    assert [e for e in log if isinstance(e, tuple)] == [("arm", G.HIGH_POSE), ("pan", [40.0, 70.0, 55.0], 45.0)]
    assert calls[0]["between"] and G.STAY_UP["base"] == -10.0
    log.clear()

    def refuse(*a, **k):
        raise G.ArmError("refused: test grid")
    monkeypatch.setattr(G, "check_pan_move", refuse)
    G.go_home(z, -10.0, 20.0, check=object(), stay_up=True, park=park)
    assert [e for e in log if isinstance(e, tuple)] == old
    G.STAY_UP.clear()


def test_pan_overlap_is_off_unless_asked(monkeypatch):
    import argparse
    from armlab import cli
    import armlab.modes.grasp as G
    assert G.PAN_OVERLAP is False
    got, real = [], argparse.ArgumentParser.parse_args

    def parse(self, argv=None, ns=None):              # cli.main's own parser, stopped before it runs anything
        got.extend([real(self, ["show"]), real(self, ["show", "--pan-overlap"])])
        raise SystemExit(0)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    try:
        cli.main()
    except SystemExit:
        pass
    assert got[0].pan_overlap is False and got[1].pan_overlap is True


def test_pan_check_covers_the_worm_slack_runup(monkeypatch):
    """Turning down to b1 the base first goes to b1 - HOME_FINISH_DEG (then + to b1): that overshoot is on the grid."""
    import armlab.modes.grasp as G
    seen = []

    class Z:
        margin = 30.0
        def gaps(self, rows):
            seen.extend(np.atleast_2d(rows)[:, 5])
            return np.full(len(np.atleast_2d(rows)), 90.0)
    G.check_pan_move(Z(), None, None, [40.0, 75.0, 60.0], [50.0, 80.0, 60.0], (42.0, 80.0), 20.0, 0.0)
    assert abs(min(seen) - (0.0 - G.HOME_FINISH_DEG)) < 1e-9 and max(seen) == 20.0 and 0.0 in seen


def test_live_page_safety_and_snapshot():
    """The demo page's data: move checks keep the tightest clearance this round; the snapshot carries the Pi's clock."""
    import time
    from armlab import live
    live.update(safety=None)
    for g in (60.0, 34.0, 71.0):
        live.check(g, 15.0)
    s = live.snapshot()
    assert s["safety"] == {"last": 71.0, "min": 34.0, "margin": 15.0, "n": 3} and abs(s["now"] - time.time()) < 5
    page = live.PAGE
    assert "renderJoints" in page and "innerHTML +=" not in page   # an appended
    # text grew every poll until the block was first seen and squeezed the camera views


def test_slack_gp_predicts_like_the_formula_and_logs_for_the_page():
    """slackgp: k(x, X) @ alpha, de-normalised -- checked on a two-pose model by hand; live keeps the pairs."""
    import numpy as np
    from armlab import live, slackgp
    X = np.array([[0.0] * 11, [1.0] * 11])
    j = {"_alpha": np.array([1.0, -1.0]), "_ls": np.ones(11), "c": 2.0, "y_mean": 0.5, "y_std": 2.0}
    m = {"_X": X, "_mu": np.zeros(11), "_sd": np.ones(11), "joints": {"shoulder": j, "elbow": j, "wrist": j}}
    joints, came = [0.0] * 5, [0.0] * 5                     # features all 0: x = X[0]
    k = np.array([2.0, 2.0 * np.exp(-0.5 * 11)])
    want = float(k @ j["_alpha"] * 2.0 + 0.5)
    got = slackgp.predict(joints, 0.0, came, model=m)
    assert abs(got["shoulder"] - want) < 1e-9 and set(got) == {"shoulder", "elbow", "wrist"}
    assert list(slackgp.features([10, 20, 30, 40, 50], 5, [12, 20, None, 39, 50])[6:]) == [-1, 0, 0, 1, 0]
    live.update(slack_log=[])
    live.slack_pair([1.0, -0.5, 0.2], got)
    s = live.snapshot()
    assert s["slack_log"][-1][0] == [1.0, -0.5, 0.2] and abs(s["slack_pred"]["elbow"] - want) < 1e-9
