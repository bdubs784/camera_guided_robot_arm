"""
Every tuning number in one place, grouped by what uses it. Angles are the ESP32's slider values (degrees),
distances are mm, times are seconds. Files named here live in the data folder (~/arm_data by default).
"""

# ---------------- tags ----------------
HAND_TAG = 9                           # tag on the hand that the tower camera measures
FINGER_TAGS = (31, 32)                 # one on each finger; the gripper camera steers by them
HOME_TAG = 0                           # tag on the base, seen by the gripper camera
TARGET_TAG = 7                         # the cube
TAG_MM = {7: 30.0, 31: 11.7, 32: 11.7, 9: 19.5, 0: 39.0}  # printed tag sizes (black square), mm
TAG_MM.update({i: 20.0 for i in range(40, 44)})   # hand cluster (tags_mat_and_stickers.pdf, page 5)
TAG_MM.update({i: 30.0 for i in range(50, 60)})   # landmarks around the table and tower (page 4)
TAG_MM.update({i: 25.0 for i in range(70, 78)})   # arm links (page 4)
TAG_MM.update({i: 30.0 for i in range(100, 136)}) # the tape-down mat (pages 1-3)
TAG_MM.update({i: 30.0 for i in range(200, 216)}) # tags_safe_200plus.pdf: landmarks
TAG_MM.update({i: 25.0 for i in range(220, 244)}) # ...arm and tower
TAG_MM.update({i: 20.0 for i in range(250, 266)}) # ...small
TAG_MM.update({i: 9.0 for i in range(280, 312)})  # tags_wedge_9mm.pdf: the printed wedges on the arm
TAG_MM.update({i: 14.0 for i in range(320, 352)})  # tags_wedge_14mm.pdf: no border, cut on the black edge
TAG_MM.update({i: 28.0 for i in range(360, 380)})  # tags_wedge_28mm.pdf: the big wedge, no border
TAG_MM.update({i: 15.0 for i in range(400, 526)})  # the left-side mat (tools/make_mat.py): small tags on a 25 mm
                                                  # pitch, so the gripper camera sees many at once up close
TAG_MM[61] = 10.0                              # old small sheet (51-72 are 10 mm): the one on the hand

# ---------------- the arm ----------------
MOVING = (0, 1, 2)                     # joints the automatic moves use: shoulder, elbow, wrist
JOINT_NAMES = ["shoulder", "elbow", "wrist", "wrist rotate", "gripper", "base"]
SLIDER_RANGE_DEFAULT = ((0, 160), (0, 160), (0, 170))   # until the ESP32 reports its own (trimmed) limits
AUTO_HI_DEFAULT = (75.0, 159.0, 169.0)   # automatic moves stay inside these. After the shoulder horn was reseated
AUTO_LO = (1.0, 1.0, 1.0)                #   (+90 steps) the safety map stops at 72, so a shoulder value above ~75 is an
                                         #   old pose that would point the arm ~40 deg too high. Every Pi move is
                                         #   clamped to this, and the firmware has the same shoulder limit.
HOME_POSE = (5.8, 54.0, 68.0)          # home pose unless you save another (home_pose.json)
ROLL_SLIDER = 42.0                     # wrist rotate the safety map was drawn for (hand level)
ROLL_TOL = 10.0                        # wrist rotate must be within this of ROLL_SLIDER
ROLL_RANGE = (0.0, 112.0)              # wrist rotate limits (firmware 4.18 enforces the same): above 112 the hand hits itself
BASE_LIMIT_DEG = 179.0                 # the base never goes further than this from its zero (cables)
GRIP_TIME_S = 1.5                      # seconds to open or close the gripper

# ---------------- safe motion ----------------
MARGIN_MM = 40.0                       # clear space the arm must keep to everything, at all times
FINISH_DEG = 8.0                       # the last few degrees of a measured move are done slowly, so the arm lands
FINISH_SPEED = 10.0                    #   the same way every time (real deg/s for that last part)
RUN_SPEED = 30.0                       # real deg/s during the automatic runs
MAX_SPEED = 100.0                      # never faster than this
FW_SPEED_BASE = (0.8, 1.0, 1.5)        # firmware speedBase for shoulder/elbow/wrist: its cap is speedBase x Speed setting (deg/s)
SURVEY_SPEED = 100.0                   # survey travel (the ESP32 still caps the shoulder lower); the last FINISH_DEG
                                       # of every survey pose is still at FINISH_SPEED
SPEED_SETTING = 200                    # the ESP32's Speed setting during a run
NORMAL_SPEED = 50                      # what the Speed slider on the arm page is normally at; restored after a run
TICK_CAP_DEG = 0.8 * SPEED_SETTING / 100 * 0.85   # most a joint may turn per ESP32 step and still keep all joints together
ARRIVE_TOL = 0.3                       # deg: "arrived" when every joint is this close to its target
STALL_S = 3.0                          # a move is abandoned if no joint changes for this long
MOVE_LIMIT_S = 120.0                   # and in any case after this long
OFF_PATH_DEG = 6.0                     # stop if the arm is this far from the planned straight path
WATCH_SLACK_MM = 10.0                  # stop if clear space falls this far below the margin mid-move

# ---------------- measuring with the tower camera ----------------
STILL_MM = 0.5                         # "stopped" = two camera frames in a row within this distance
READS = 5                              # camera measurements averaged per pose
READ_TIMEOUT_S = 4.0                   # give up looking for the tag after this long

# ---------------- finding the cube ----------------
BASE_DPS = 12.0                        # firmware 4.11+: EVERY automatic base move and pan runs smoothly at this speed

# ---------------- survey (data for the big solve) ----------------
SURVEY_FRAMES = 3                      # frames recorded from each camera at each pose
SURVEY_BASE_DEFAULT = 45.0             # base range (+-) when no scan range has been taught

# ---------------- base homing ----------------
HOMING_STAGE = (12.9, 54.0, 50.0)      # the arm goes to and from the homing pose through here, one joint at a time:
                                       #   in = shoulder, elbow, wrist;  out = wrist up to 50, then elbow, then shoulder
HOME_TAG_ID = 0                        # the tag on the plywood the gripper camera homes the base from
HOME_SEARCH_DEG = 30.0                 # never searches further than this either way for the tag
HOME_FINISH_DEG = 1.5                  # every base move ends with this much travel in the + direction (worm slack)
HOME_TOL_DEG = 0.05                    # done when the tag is within this much of its taught spot

# ---------------- grasp ----------------
PICK_SLOW = 8.0                        # deg/s for moves beside the block: the way in, the lift, the set-down

# ---------------- files in the data folder ----------------
MAP_FILE = "arm_map.npz"               # the safety map (made on the laptop by build_arm_map.py)
HOME_FILE = "home_pose.json"           # home: the home pose
TIMING_FILE = "timing.json"            # how slow the old firmware's moves are
BASE_HOME_FILE = "base_home.json"      # basehome --teach: the pose and picture position the base is homed from
SCAN_FILE = "scan.json"                # survey: the look poses, and how far the base may turn
