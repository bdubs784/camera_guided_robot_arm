"""
Run-time state shared by the whole program: where the ESP32 and the camera service are, and what this run has
changed on the arm. One object, `S`, so every module reads and writes the same values. The command line sets the
addresses; check_ready() fills in the limits the firmware reports (they shrink when a joint has a trim).
"""
import os

from .config import AUTO_HI_DEFAULT, SLIDER_RANGE_DEFAULT


class State:
    def __init__(self):
        self.esp_url = "http://192.168.1.226"          # the ESP32 arm controller
        self.cam_url = "http://localhost:8080"         # the camera service on this Pi
        self.data_dir = os.path.expanduser("~/arm_data")
        self.table_margin_mm = None                    # clear space to the table, plywood and base; None = same as the margin
        self.hunt_wrist = "both"                       # grasp: which way hunt_for_tag sweeps the wrist
        self.pause_hover = False                       # grasp: stop at the hover and let the sliders be driven by hand                       # grasp: which way hunt_for_tag sweeps the wrist to find the tag
        self.slider_range = SLIDER_RANGE_DEFAULT       # shoulder, elbow, wrist limits (the firmware's, after trims)
        self.auto_hi = AUTO_HI_DEFAULT                 # highest value automatic moves use per joint
        self.prof = 0                                  # 1 = firmware 4.7+: clock-timed moves (prof=1)
        self.time_scale = 1.0                          # how much slower than asked the ESP32 moves (old firmware only)
        self.speed_changed = False                     # True once this run changed the ESP32's Speed setting
        self.moving = False                            # Ctrl+C only sends STOP when this program is driving the arm
        self.base_cmd = None                           # the base angle last asked for (the ESP32 reports only 0.1 deg)
        self.exact_table = False                       # --exact-table: Zone's table gap from tablecheck.py
        self.base_closed = False                       # move_base checks the base with the cameras and nudges it (baseloop.py)
        self.base_fix = 0.0                            # target minus count after the last closed-loop base move
        self.base_speed_cmd = False                    # firmware 4.11+: /setall takes bs= (smooth slow base moves)
        self.pan_dps = None                            # --dps: base pan speed for the scan (else the config value)
        self.confirm_left = 0                          # moves that still wait for Enter first (--confirm)
        self.came_from = [None] * 5                    # each joint's commanded value before its last move (the gear
                                                       # slack depends on the direction it arrived from: slackgp.py)


S = State()
