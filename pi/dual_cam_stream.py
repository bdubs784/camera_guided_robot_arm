#!/usr/bin/env python3
"""
Arm Eyes v4: Pi cameras (+ an optional USB webcam) with AprilTag tracking in millimetres (Pi 5)
-------------------------------------------------------------
Streams the cameras and finds the printed tags in each view. If a camera has
been calibrated (calibrate_cam.py saved ~/calib_<camera>.json), each tag's
position is reported in mm relative to that camera: x right, y down, z ahead.

    http://<pi>:8080              the views with tag boxes + live numbers
    http://<pi>:8080/torre        MJPEG stream, torre cam (boxes drawn on tags)
    http://<pi>:8080/gripper      MJPEG stream, gripper cam
    http://<pi>:8080/stream/<cam> MJPEG stream of any camera (also webcam)
    http://<pi>:8080/tags         JSON: every tag each camera sees right now + stats
    http://<pi>:8080/tags/reset   zero the "how often is the hand tag visible" counter
    http://<pi>:8080/frames       JSON: EVERY detected frame since ?since=<n>, per camera (new in v4)
    http://<pi>:8080/snap/<cam>   one small JPEG with the boxes drawn
    http://<pi>:8080/raw/<cam>    one full-size clean JPEG (for calibration later)

New in v4 (nothing old was removed or renamed, so arm_lab.py keeps working):
  - Each frame carries the moment it was captured ("t_cap", Unix seconds; from the sensor's own
    timestamp on the Pi cameras), the frame number, a sharpness score and a motion score.
  - /frames keeps the last ~900 detected frames per camera, so a logger polling now and then
    still gets every frame (needed to measure wobble and lag).
  - "sharp": blur check (higher = sharper). "motion": how much the picture changed since the
    previous detected frame (0 = identical). Both work without any tag in view.
  - An optional USB webcam ("webcam") starts automatically if one is plugged in.
  - Frame rate per camera (CAMERAS[...]["fps"]). The torre now runs at 30 fps for finer timing.

Cameras are picked by sensor type, not by port number, so it does not matter
which ribbon is in which port:
    torre   = ov5647 (older 5 MP, fixed focus)
    gripper = imx708 (Camera Module 3, autofocus)
    webcam  = the first USB camera found (optional)

This runs as the "armcams" service. After copying a new version to the Pi:
    sudo systemctl restart armcams.service
    journalctl -u armcams.service -n 30        # if the page does not come up
"""

import glob
import json
import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

# OpenCV splits every call over all 4 cores and its worker threads spin while waiting. With three cameras each
# running their own detection thread that pinned the Pi at 100% (load 7-9, 85 C). One OpenCV thread per call;
# the cameras already run in parallel with each other.
cv2.setNumThreads(int(os.environ.get("ARMCAMS_CV_THREADS", "1")))
from flask import Flask, Response, jsonify, request

try:                                    # only needed for the Pi cameras
    from picamera2 import Picamera2
    from libcamera import controls
except ImportError:                     # lets the file be tested on a machine without them
    Picamera2 = None
    controls = None

# ---------------- settings ----------------
CAMERAS = {
    # model  : which sensor this view uses ("usb" = a USB webcam)
    # sensor : sensor mode to run. Chosen to keep the same framing the 640x480
    #          stream had, but with more pixels for the tag detector.
    # detect : frame size the detector works on (same 4:3 shape as the stream)
    # hfov   : nominal horizontal field of view of that frame, degrees. Only used
    #          for the rough distance readout until the camera is calibrated.
    # fps    : frames per second asked of the camera
    "torre":   {"model": "ov5647", "sensor": (1296, 972), "detect": (1280, 960), "hfov": 53.5, "fps": 15},
    "gripper": {"model": "imx708", "sensor": (2304, 1296), "detect": (1152, 864), "hfov": None, "fps": 15},
    "webcam":  {"model": "usb", "detect": (1920, 1080), "hfov": None, "fps": 10, "optional": True},
}
STREAM_SIZE = (640, 480)     # what the browser gets
STREAM_FPS  = 15             # browser stream rate (the cameras themselves may run faster)
JPEG_Q      = 75
PORT        = 8080
FRAME_BUFFER = 900           # detected frames kept per camera for /frames (30 s at 30 fps)
SCORE_WIDTH  = 320           # sharpness and motion are measured on a picture this wide
# rotation per camera: None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE
ROTATE = {"torre": None, "gripper": None, "webcam": None}
# Most tag reads per second each camera may do. Detection is the heavy part; without a cap three cameras keep
# all four Pi cores at 100% (load 7, 84 C, throttling). The gripper camera steers the grab: it gets most.
MAX_DETECT_HZ = {"gripper": 15.0, "torre": 8.0, "webcam": 5.0}
# When nothing has asked for tags lately (no arm program running, no page open), read tags only this often.
# The first request wakes everything up again within a fraction of a second.
IDLE_DETECT_HZ = 2.0
IDLE_AFTER_S = 15.0
last_request = [0.0]
rate_scale = [1.0]           # /rate?scale=0.35: the survey eases the cameras off while the Pi runs hot
paused_until = {}            # /rate?pause=gripper&secs=60: no tag detection on that camera until then (arm.py's
                             # search: the gripper camera sees only table there and takes the webcam's CPU).
                             # /rate?pause= resumes all; a pause runs out by itself if the arm program dies

# ---------------- tags ----------------
# id: (name, size in mm of the black square's outer edge)
TAGS = {
    7:  ("target", 30.0),
    9:  ("hand", 19.5),   # back of the gripper camera mount
    31: ("finger L", 11.7),   # left finger, as the gripper camera sees it
    32: ("finger R", 11.7),   # right finger
    0:  ("home", 39.0),   # homing tag on the fixed base
    # side tags can be added here later, e.g.  10: ("hand left", 20.0),
}
HAND_TAGS   = [9]            # any of these visible to the torre cam = "hand is tracked"
FINGER_TAGS = (31, 32)       # the pair used for the finger-gap readout

# Detector tuning. Decimate > 1 is faster but needs bigger tags; 0 = most thorough.
QUAD_DECIMATE = 1.5

# ---------------- tag detection ----------------
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def make_detector():
    p = cv2.aruco.DetectorParameters()
    # The AprilTag method reads tags seen at a steep slant (the finger tags)
    # that the default method misses.
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    p.aprilTagQuadDecimate = QUAD_DECIMATE
    return cv2.aruco.ArucoDetector(DICT, p)


def load_calibration(name, size):
    """Lens numbers saved by calibrate_cam.py, or None if that camera is not calibrated."""
    path = os.path.expanduser(f"~/calib_{name}.json")
    try:
        with open(path) as f:
            c = json.load(f)
    except (FileNotFoundError, ValueError):
        return None
    if tuple(c.get("image_size", ())) != tuple(size):
        print(f"[{name}] calibration file is for {c.get('image_size')}, camera runs {list(size)}: ignoring it")
        return None
    return np.array(c["camera_matrix"], np.float64), np.array(c["dist_coeffs"], np.float64)


def tag_pose(pts, size_mm, calib):
    """Where a tag is relative to the camera, in mm: x right, y down, z straight ahead."""
    K, dist = calib
    h = size_mm / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, pts.astype(np.float32), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return tvec.flatten(), rvec.flatten()


def detect_tags(detector, bgr, hfov_deg=None, calib=None, gray=None):
    """Return a list of dicts, one per tag found in this frame."""
    if gray is None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    out = []
    if ids is None:
        return out
    width = bgr.shape[1]
    focal_px = (width / 2) / math.tan(math.radians(hfov_deg / 2)) if hfov_deg else None
    for c, i in zip(corners, ids.flatten()):
        pts = c[0]
        sides = [float(np.linalg.norm(pts[k] - pts[(k + 1) % 4])) for k in range(4)]
        tid = int(i)
        name, size_mm = TAGS.get(tid, (f"tag {tid}", None))
        det = {
            "id": tid,
            "name": name,
            "center": [round(float(v), 1) for v in pts.mean(axis=0)],
            "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in pts],
            "side_px": round(max(sides), 1),      # longest edge = least foreshortened
            "known": tid in TAGS,
        }
        pose = tag_pose(pts, size_mm, calib) if (calib and size_mm) else None
        if pose is not None:
            t, r = pose
            det["pos_mm"] = [round(float(v), 1) for v in t]
            det["dist_mm"] = round(float(np.linalg.norm(t)), 1)
            det["rvec"] = [round(float(v), 4) for v in r]
        elif focal_px and size_mm:
            # Rough only: nominal lens numbers, used until the camera is calibrated.
            det["approx_dist_mm"] = round(focal_px * size_mm / max(sides))
        out.append(det)
    return out


def finger_gap_mm(dets):
    """Center-to-center distance between the two finger tags, in mm (approximate)."""
    by_id = {d["id"]: d for d in dets}
    a, b = by_id.get(FINGER_TAGS[0]), by_id.get(FINGER_TAGS[1])
    if not a or not b:
        return None
    size_mm = TAGS[FINGER_TAGS[0]][1]
    px = math.dist(a["center"], b["center"])
    mm_per_px = size_mm / ((a["side_px"] + b["side_px"]) / 2)
    return round(px * mm_per_px, 1)


def draw_overlay(small, dets, scale, banner):
    for d in dets:
        pts = (np.array(d["corners"]) * scale).astype(np.int32)
        color = (80, 255, 120) if d["known"] else (0, 200, 255)
        cv2.polylines(small, [pts], True, color, 2)
        x, y = pts.min(axis=0)
        cv2.putText(small, d["name"], (int(x), max(12, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(small, d["name"], (int(x), max(12, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if banner:
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(small, (0, 0), (tw + 16, th + 14), (20, 20, 20), -1)
        cv2.putText(small, banner, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return small


def picture_scores(gray, prev_small):
    """Sharpness (variance of the Laplacian) and motion (mean change vs the previous frame), on a small copy."""
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (SCORE_WIDTH, max(1, int(h * SCORE_WIDTH / w))), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(small, cv2.CV_32F).var())
    motion = None
    if prev_small is not None and prev_small.shape == small.shape:
        motion = float(cv2.absdiff(small, prev_small).mean())
    return round(sharp, 1), (round(motion, 2) if motion is not None else None), small


def boottime_ns():
    try:
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    except (AttributeError, OSError):
        return time.monotonic_ns()


# ---------------- camera workers ----------------
def find_camera(model):
    for info in Picamera2.global_camera_info():
        if model in info.get("Model", ""):
            return info["Num"]
    raise SystemExit(f"No camera with sensor '{model}' found. "
                     "Run: rpicam-hello --list-cameras")


def find_usb_camera():
    """The first USB camera, by its stable name (the Pi cameras never show up there)."""
    found = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    return found[0] if found else None


FOCUS_FILE = os.path.expanduser("~/arm_data/gripper_focus.txt")


def saved_focus(default=10.0):
    """Lens position for the gripper camera: 1 / distance in metres. Remembered between restarts."""
    try:
        return float(open(FOCUS_FILE).read().strip())
    except (OSError, ValueError):
        return default


class Cam:
    """One camera: a capture thread (stream) and a detector thread (tags)."""

    def __init__(self, name):
        self.name = name
        self.cfg = CAMERAS[name]
        self.lock = threading.Lock()
        self.frame = None            # newest full-size frame
        self.frame_t = None          # when it was captured (Unix seconds)
        self.frame_meta = {}
        self.frame_no = 0            # frames captured so far
        self.jpeg = None             # newest small JPEG with boxes drawn
        self.dets = []               # newest detections
        self.det_info = {}           # t_cap, frame, sharp, motion of the newest detected frame
        self.detect_hz = 0.0
        self.recent = deque(maxlen=400)   # (time, hand_visible) for the 10 s figure
        self.total = 0
        self.total_visible = 0
        self.seen_count = {}
        self.buffer = deque(maxlen=FRAME_BUFFER)   # every detected frame, for /frames
        self.detector = make_detector()
        self.size = tuple(self.cfg["detect"])
        self._open()
        self.calib = load_calibration(name, self.size)
        print(f"[{name}] " + ("calibration loaded: positions are in mm" if self.calib
                              else "no calibration file: distances are rough estimates"))
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()

    # --- camera specific ---
    def _open(self):
        if Picamera2 is None:
            raise SystemExit("picamera2 is not installed")
        self.cam = Picamera2(find_camera(self.cfg["model"]))
        main = {"size": self.cfg["detect"], "format": "RGB888"}
        ctrl = {"FrameRate": self.cfg.get("fps", 15)}
        try:
            cfg = self.cam.create_video_configuration(
                main=main, controls=ctrl,
                sensor={"output_size": self.cfg["sensor"], "bit_depth": 10})
            self.cam.configure(cfg)
        except Exception as e:      # older picamera2: let it pick the sensor mode itself
            print(f"[{self.name}] could not force sensor mode ({e}); using automatic mode")
            self.cam.configure(self.cam.create_video_configuration(main=main, controls=ctrl))
        self.cam.start()
        # Camera Module 3 has autofocus; the older camera does not. Only ask if supported.
        if "AfMode" in self.cam.camera_controls:
            self.cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": saved_focus()})

    def _grab(self):
        """One frame: (image, capture time in Unix seconds, metadata)."""
        req = self.cam.capture_request()
        try:
            img = req.make_array("main")          # BGR-ordered array, ready for OpenCV
            md = req.get_metadata() or {}
        finally:
            req.release()
        now = time.time()
        t_cap = now
        ts = md.get("SensorTimestamp")             # ns, start of exposure, on the boot-time clock
        if ts:
            age = (boottime_ns() - ts) / 1e9
            if 0.0 <= age < 2.0:
                t_cap = now - age
        meta = {"exp_us": md.get("ExposureTime"), "gain": md.get("AnalogueGain"), "lens": md.get("LensPosition")}
        return img, t_cap, meta

    # --- stream ---
    def _capture_loop(self):
        last_jpeg = 0.0
        while True:
            try:
                img, t_cap, meta = self._grab()
            except Exception as e:
                print(f"[{self.name}] capture failed: {e}")
                time.sleep(0.5)
                continue
            if ROTATE.get(self.name) is not None:
                img = cv2.rotate(img, ROTATE[self.name])
            with self.lock:
                self.frame = img
                self.frame_t = t_cap
                self.frame_meta = meta
                self.frame_no += 1
                dets = self.dets
            now = time.time()
            if now - last_jpeg < 1.0 / STREAM_FPS:      # the browser stream does not need every frame
                continue
            last_jpeg = now
            scale = STREAM_SIZE[0] / img.shape[1]            # keep the picture's own shape (a 16:9 webcam stays 16:9),
            small = cv2.resize(img, (STREAM_SIZE[0], max(1, int(round(img.shape[0] * scale)))),   # so boxes land on the tags
                               interpolation=cv2.INTER_AREA)
            small = draw_overlay(small, dets, scale, self._banner(dets))
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if ok:
                with self.lock:
                    self.jpeg = buf.tobytes()

    def _banner(self, dets):
        if self.name == "torre":
            hand = next((d for d in dets if d["id"] in HAND_TAGS), None)
            pct = self.visible_pct_total()
            seen = f"   seen {pct:.0f}% of the time" if pct is not None else ""
            if hand and "pos_mm" in hand:
                x, y, z = hand["pos_mm"]
                return f"hand at x {x:.0f}  y {y:.0f}  z {z:.0f} mm{seen}"
            return f"hand tag: {'YES' if hand else 'no'}{seen}"
        if self.name == "gripper":
            gap = finger_gap_mm(dets)
            n = sum(1 for d in dets if d["id"] in FINGER_TAGS)
            return f"finger gap about {gap:.0f} mm" if gap is not None else f"finger tags: {n} of 2"
        return f"{len(dets)} tags"

    # --- tags ---
    def _detect_loop(self):
        last = None
        last_t0 = 0.0
        prev_small = None
        while True:
            with self.lock:
                img, t_cap, meta, no = self.frame, self.frame_t, self.frame_meta, self.frame_no
            if time.time() < paused_until.get(self.name, 0.0):
                with self.lock:
                    self.dets = []                  # nothing stale from before the pause
                time.sleep(0.02)
                continue
            cap_hz = MAX_DETECT_HZ.get(self.name)
            if time.time() - last_request[0] > IDLE_AFTER_S:
                cap_hz = min(cap_hz or IDLE_DETECT_HZ, IDLE_DETECT_HZ)       # nobody is asking: idle
            elif rate_scale[0] < 1.0:
                cap_hz = max(1.0, (cap_hz or self.cfg.get("fps", 30)) * rate_scale[0])
            if cap_hz and last is not None and time.time() - last_t0 < 1.0 / cap_hz:
                time.sleep(0.005)                       # rate cap: leave CPU for the other cameras (and SSH)
                continue
            if img is None or img is last:
                time.sleep(0.003)
                continue
            last = img
            t0 = time.time()
            last_t0 = t0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detect_tags(self.detector, img, self.cfg.get("hfov"), self.calib, gray=gray)
            sharp, motion, prev_small = picture_scores(gray, prev_small)
            dt = time.time() - t0
            visible = any(d["id"] in HAND_TAGS for d in dets)
            with self.lock:
                self.dets = dets
                self.detect_hz = 0.9 * self.detect_hz + 0.1 * (1.0 / max(dt, 1e-3))
                self.recent.append((t0, visible))
                self.total += 1
                self.total_visible += int(visible)
                for d in dets:
                    self.seen_count[d["id"]] = self.seen_count.get(d["id"], 0) + 1
                info = {"n": self.total, "frame": no, "t_cap": round(t_cap, 4) if t_cap else None,
                        "t_det": round(time.time(), 4), "sharp": sharp, "motion": motion}
                info.update({k: v for k, v in meta.items() if v is not None})
                self.det_info = info
                self.buffer.append(dict(info, tags=dets))

    def visible_pct_total(self):
        return 100.0 * self.total_visible / self.total if self.total else None

    def visible_pct_recent(self, seconds=10.0):
        now = time.time()
        vals = [v for t, v in self.recent if now - t <= seconds]
        return 100.0 * sum(vals) / len(vals) if vals else None

    def reset_stats(self):
        with self.lock:
            self.recent.clear()
            self.total = self.total_visible = 0
            self.seen_count = {}
            self.buffer.clear()

    def status(self):
        with self.lock:
            dets = list(self.dets)
            out = {
                "size": list(self.size),
                "calibrated": self.calib is not None,
                "tags": dets,
                "detect_hz": round(self.detect_hz, 1),
                "frames_checked": self.total,
                "seen_count": dict(self.seen_count),
            }
            out.update({k: v for k, v in self.det_info.items() if k in ("t_cap", "frame", "sharp", "motion")})
            pct_total, pct_recent = self.visible_pct_total(), self.visible_pct_recent()
        if self.name == "torre":
            out["hand_visible"] = any(d["id"] in HAND_TAGS for d in dets)
            out["hand_visible_pct_total"] = None if pct_total is None else round(pct_total, 1)
            out["hand_visible_pct_10s"] = None if pct_recent is None else round(pct_recent, 1)
        elif self.name == "gripper":
            out["finger_gap_mm"] = finger_gap_mm(dets)
        return out

    def frames_since(self, since, limit):
        with self.lock:
            rows = [f for f in self.buffer if f["n"] > since]
            last = self.total
        return rows[-limit:] if limit else rows, last

    def get_jpeg(self):
        with self.lock:
            return self.jpeg

    def get_raw_jpeg(self):
        with self.lock:
            img = self.frame
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buf.tobytes() if ok else None


WEBCAM_CONTROLS = os.path.expanduser("~/arm_data/webcam_controls.txt")


def apply_webcam_controls(dev):
    """
    Settings for the USB webcam (exposure, gain, ...) from ~/arm_data/webcam_controls.txt, one per line, like
        auto_exposure=1
        exposure_time_absolute=150
    (names and ranges: v4l2-ctl -d /dev/video0 --list-ctrls). Applied every time the camera starts, because a
    webcam forgets them when it is unplugged. Lines starting with # are ignored.
    """
    try:
        lines = [l.strip() for l in open(WEBCAM_CONTROLS) if l.strip() and not l.strip().startswith("#")]
    except OSError:
        return
    import subprocess
    for line in lines:                                   # one at a time: some only work after another is set
        try:
            r = subprocess.run(["v4l2-ctl", "-d", dev, "-c", line], capture_output=True, text=True, timeout=5)
            print(f"[webcam] {line}: " + ("ok" if r.returncode == 0 else (r.stderr.strip() or "failed")))
        except (OSError, subprocess.SubprocessError) as e:
            print(f"[webcam] {line}: {e}")


class UsbCam(Cam):
    """A USB webcam (UVC). Its capture time is when the frame arrived, so it runs a few ms late."""

    def _open(self):
        dev = self.cfg.get("device") or find_usb_camera()
        if not dev:
            raise SystemExit("no USB camera plugged in")
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open {dev}")
        apply_webcam_controls(dev)
        w, h = self.cfg["detect"]
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.get("fps", 30))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, img = self.cap.read()
        if not ok:
            raise SystemExit(f"{dev} gave no picture")
        self.size = (img.shape[1], img.shape[0])        # what the webcam actually gives
        print(f"[{self.name}] {dev} at {self.size[0]}x{self.size[1]}")

    def _grab(self):
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError("webcam read failed")
        return img, time.time(), {}


cams = {}


def start_cameras():
    for name, cfg in CAMERAS.items():
        try:
            cams[name] = (UsbCam if cfg["model"] == "usb" else Cam)(name)
        except (SystemExit, Exception) as e:
            if cfg.get("optional"):
                print(f"[{name}] not started ({e})")
            else:
                print(f"[{name}] CAMERA NOT STARTED: {e}")
    if not cams:
        raise SystemExit("No cameras found at all.")


# ---------------- web server ----------------
app = Flask(__name__)


def mjpeg(cam):
    period = 1.0 / STREAM_FPS
    while True:
        jpg = cam.get_jpeg()
        if jpg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(period)


def stream_of(name):
    if name not in cams:
        return f"{name} camera is not connected", 503
    return Response(mjpeg(cams[name]), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/torre")
def torre():
    return stream_of("torre")


@app.route("/gripper")
def gripper():
    return stream_of("gripper")


@app.route("/stream/<name>")
def stream(name):
    return stream_of(name)


@app.route("/snap/<name>")
def snap(name):
    cam = cams.get(name)
    if not cam or not cam.get_jpeg():
        return "no frame yet", 404
    return Response(cam.get_jpeg(), mimetype="image/jpeg")


@app.route("/raw/<name>")
def raw(name):
    cam = cams.get(name)
    jpg = cam.get_raw_jpeg() if cam else None
    if not jpg:
        return "no frame yet", 404
    return Response(jpg, mimetype="image/jpeg")


@app.route("/focus")
def focus():
    """/focus shows the gripper camera's lens position; /focus?v=12 sets it and remembers it."""
    cam = cams.get("gripper")
    if cam is None or not hasattr(cam, "cam") or "LensPosition" not in cam.cam.camera_controls:
        return Response(json.dumps({"ok": 0, "error": "gripper camera is not running or has no focus control"}), mimetype="application/json")
    lo, hi, _ = cam.cam.camera_controls["LensPosition"]
    v = request.args.get("v")
    if v is not None:
        v = max(float(lo), min(float(hi), float(v)))
        cam.cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": v})
        os.makedirs(os.path.dirname(FOCUS_FILE), exist_ok=True)
        open(FOCUS_FILE, "w").write(str(v))
    now = saved_focus()
    return Response(json.dumps({"ok": 1, "lens_position": now, "sharpest_at_cm": (round(100 / now, 1) if now > 0 else "far away"),
                                "allowed": [float(lo), float(hi)]}), mimetype="application/json")


@app.route("/rate")
def rate():
    if "pause" in request.args:                 # per camera; leaves the scale alone
        try:
            until = time.time() + min(120.0, max(0.0, float(request.args.get("secs", 60.0))))
        except ValueError:
            until = time.time() + 60.0
        names = [n for n in request.args.get("pause", "").split(",") if n]
        paused_until.clear()
        paused_until.update({n: until for n in names})
        return jsonify({"ok": 1, "scale": rate_scale[0], "paused": names})
    try:
        rate_scale[0] = min(1.0, max(0.05, float(request.args.get("scale", 1.0))))
    except ValueError:
        pass
    return jsonify({"ok": 1, "scale": rate_scale[0]})


@app.route("/tags")
def tags():
    if request.args.get("view") is None:       # the camera page polls with ?view=1: watching does not wake full speed
        last_request[0] = time.time()
    resp = jsonify({name: cam.status() for name, cam in cams.items()})
    resp.headers["Access-Control-Allow-Origin"] = "*"   # lets the arm page read it too
    return resp


@app.route("/frames")
def frames():
    """
    Every detected frame since ?since=<n> (per camera, n counts detected frames; 0 = all kept).
    ?cam=torre limits it to one camera; ?limit=200 caps how many come back (newest kept).
    Reply: {"torre": {"last": n, "frames": [{n, frame, t_cap, t_det, sharp, motion, exp_us, tags}, ...]}, ...}
    """
    last_request[0] = time.time()
    since = int(request.args.get("since", 0))
    limit = int(request.args.get("limit", 0))
    pick = request.args.get("cam")
    out = {}
    for name, cam in cams.items():
        if pick and name != pick:
            continue
        rows, last = cam.frames_since(since, limit)
        out[name] = {"last": last, "frames": rows}
    resp = jsonify(out)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/tags/reset", methods=["GET", "POST"])
def tags_reset():
    for cam in cams.values():
        cam.reset_stats()
    resp = jsonify({"ok": 1})
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


PAGE = """<!doctype html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arm Eyes</title><style>
body{background:#111;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:12px}
h2{text-align:center;margin:6px 0 12px}
.wrap{display:flex;flex-wrap:wrap;gap:12px;justify-content:center}
.cam{background:#1d1d1f;border-radius:12px;padding:8px;flex:1 1 300px;max-width:640px}
.cam img{width:100%;border-radius:8px;display:block}
.lbl{font-weight:600;margin:4px 2px 8px;color:#7fd}
.stat{font-size:14px;line-height:1.6;margin:8px 2px 2px;font-variant-numeric:tabular-nums}
.stat b{color:#7fd;font-weight:600}.dim{color:#888}
button{padding:8px 14px;border:0;border-radius:8px;background:#2a2a2e;color:#eee;font-size:14px}
.bar{text-align:center;margin-top:12px}
</style></head><body>
<h2>Arm Eyes</h2>
<div class="wrap">
  <div class="cam"><div class="lbl">Torre (overview)</div><img src="/torre">
    <div class="stat" id="st">waiting for tags...</div></div>
  <div class="cam"><div class="lbl">Gripper (hand)</div><img src="/gripper">
    <div class="stat" id="sg">waiting for tags...</div></div>
  <div class="cam" id="wcbox" style="display:none"><div class="lbl">Webcam</div><img id="wcimg">
    <div class="stat" id="sw">waiting for tags...</div></div>
</div>
<div class="bar"><button onclick="fetch('/tags/reset')">Reset visibility counter</button></div>
<script>
function list(tags){
  if(!tags.length) return '<span class="dim">no tags in view</span>';
  return tags.map(t=>`${t.name} <span class="dim">(id ${t.id}, ${Math.round(t.side_px)} px`+
    (t.pos_mm?`, x ${t.pos_mm[0].toFixed(0)} y ${t.pos_mm[1].toFixed(0)} z ${t.pos_mm[2].toFixed(0)} mm, `+
       `${t.dist_mm.toFixed(0)} mm away`:(t.approx_dist_mm?`, ~${t.approx_dist_mm} mm away`:''))+
    `)</span>`).join('<br>');
}
function scores(c){ return `<span class="dim">sharp ${c.sharp==null?'-':Math.round(c.sharp)}, motion ${c.motion==null?'-':c.motion.toFixed(1)}</span>`; }
async function tick(){
  try{
    const d=await (await fetch('/tags?view=1')).json(), t=d.torre, g=d.gripper, w=d.webcam;
    const pct=v=>v==null?'-':v.toFixed(0)+'%';
    if(t) st.innerHTML=`Hand tag visible: <b>${t.hand_visible?'YES':'no'}</b><br>`+
      `Seen <b>${pct(t.hand_visible_pct_total)}</b> of the time since reset `+
      `<span class="dim">(last 10 s: ${pct(t.hand_visible_pct_10s)}, ${t.frames_checked} checks, `+
      `${t.detect_hz}/s)</span> ${scores(t)}<br>`+list(t.tags);
    if(g) sg.innerHTML=`Finger gap: <b>${g.finger_gap_mm==null?'-':'~'+g.finger_gap_mm+' mm'}</b> `+
      `<span class="dim">(tag center to tag center, ${g.detect_hz}/s)</span> ${scores(g)}<br>`+list(g.tags);
    if(w){ if(wcbox.style.display==='none'){ wcbox.style.display=''; wcimg.src='/stream/webcam'; }
      sw.innerHTML=`<span class="dim">${w.detect_hz}/s</span> ${scores(w)}<br>`+list(w.tags); }
  }catch(e){ st.textContent='lost connection to the Pi'; }
}
setInterval(tick,500); tick();
</script></body></html>"""


@app.route("/")
def index():
    return PAGE


if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)   # do not log every request: that filled the SD card
    start_cameras()
    print(f"Streaming with tag tracking on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
