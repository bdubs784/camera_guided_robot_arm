#!/usr/bin/env python3
"""
Control program for the camera-guided robot arm (runs on the Raspberry Pi).

    python3 arm.py status      check the ESP32, the cameras and the safety map
    python3 arm.py tags        what every camera sees right now (checks stickers: duplicates, unknown, too small)
    python3 arm.py where       live readout: clear space to the table, tower and arm, as you drive
    python3 arm.py go          go to the home pose   (or:  go 100 60 70  for shoulder, elbow, wrist)
    python3 arm.py home        make the pose the arm is in now the home pose
    python3 arm.py grab        grab the cube, steered by the gripper camera   (grab --teach once first)
    python3 arm.py grab --anywhere    find the cube on the table first (scan --teach once first)
    python3 arm.py scan        show or teach (--teach) the look poses the scan uses
    python3 arm.py find        scan the table for the cube and say where it is (no grab)
    python3 arm.py demo --rounds 10   find it, grab it, set it down somewhere new, again and again (scoreboard)
    python3 arm.py basehome    home the base from a tag, with the gripper camera   (--teach once first)
    python3 arm.py trim        after a servo swap: measure the new horn's offset and store a trim in the ESP32
    python3 arm.py tags3d      live: every tag the gripper camera sees, in mm
    python3 arm.py survey      big data collection for the big solve (all cameras, random poses, in chunks)
    python3 arm.py repeat      repeatability test          python3 arm.py collect   collect self-model data
    python3 arm.py follow      the hand follows a tag on a stick     python3 arm.py fingers   measure the fingers
    python3 arm.py teach / look / pick    the older pick, steered by the tower camera and the self-model

Ctrl+C at any time tells the ESP32 to hold where it is, then quits. Files live in ~/arm_data/.
"""
import importlib.util
import sys

if importlib.util.find_spec("numpy") is None:
    sys.exit("This needs numpy. Install it with:  sudo apt install python3-numpy")

from armlab.cli import main

if __name__ == "__main__":
    main()
