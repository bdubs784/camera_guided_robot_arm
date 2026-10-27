#!/usr/bin/env python3
"""
Open the simulated arm in the MuJoCo viewer, driven by the same numbers as the arm page.

    python view_arm.py                 # starts at the level pose 123 54 68
    python view_arm.py 100 60 87       # shoulder elbow wrist
    python view_arm.py 100 60 87 --grip 140 --base 30

In the viewer, open "Control" on the right: each slider is the arm-page value for that joint
(gripper 160 = closed, 40 = wide open; base is in degrees). Needs:  pip install mujoco
"""
import argparse, os, mujoco, mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "arm_twin.xml"))
    return m, mujoco.MjData(m)


def set_pose(m, d, values):
    """values: {actuator name: arm-page value}. Puts the joints there straight away and holds them."""
    for name, v in values.items():
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        d.ctrl[a] = v
        j = m.actuator_trnid[a, 0]
        # each actuator pulls its joint toward  (gain * value + bias0) / -bias1
        d.qpos[m.jnt_qposadr[j]] = (m.actuator_gainprm[a, 0] * v + m.actuator_biasprm[a, 0]) / -m.actuator_biasprm[a, 1]
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gear_l"); r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gear_r")
    d.qpos[m.jnt_qposadr[r]] = -d.qpos[m.jnt_qposadr[g]]          # the two gears mesh
    mujoco.mj_forward(m, d)
    for _ in range(300):                                           # let the finger linkage settle
        mujoco.mj_step(m, d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pose", nargs="*", type=float, default=[123, 54, 68], help="shoulder elbow wrist")
    p.add_argument("--roll", type=float, default=42); p.add_argument("--grip", type=float, default=100)
    p.add_argument("--base", type=float, default=0)
    a = p.parse_args()
    if len(a.pose) != 3:
        raise SystemExit("Give three numbers: shoulder elbow wrist")
    m, d = load()
    set_pose(m, d, {"shoulder": a.pose[0], "elbow": a.pose[1], "wrist": a.pose[2], "wrist_rotate": a.roll,
                    "base_deg": a.base, "gripper": a.grip})
    mujoco.viewer.launch(m, d)
