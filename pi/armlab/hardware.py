"""
Talking to the hardware over HTTP: the ESP32 arm controller and the camera service.
"""
import json
import time
import urllib.error
import urllib.request

from . import link
from .errors import ArmError
from .state import S


def http_json(url, timeout=3.0, tries=3):
    if link.CABLE is not None and url.startswith(S.esp_url):      # the Pi-ESP32 cable: ~1 ms, no WiFi hiccups
        path = url[len(S.esp_url):] or "/"
        for _ in range(3):                                         # a message damaged by noise is sent again
            cable = link.CABLE
            if cable is None:
                break
            try:
                got = cable.ask(path, min(timeout, 1.5))
                link.ok()
                return got
            except (OSError, ValueError) as e:
                link.failed(e)                             # several in a row: WiFi for the rest of the run
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = e
            time.sleep(0.2)
    raise ArmError(f"no answer from {url} ({last})")


def arm_state():
    """Everything the ESP32 knows: name, limits, angle, target and power for each joint."""
    return http_json(S.esp_url + "/state")["joints"]


def arm_angles():
    """Where the ESP32 has driven each joint to, right now: [shoulder, elbow, wrist, rotate, gripper, base]."""
    return [float(v) for v in http_json(S.esp_url + "/setall?a=")["a"]]


def arm_stop():
    """Hold position. Safe to call at any time."""
    try:
        http_json(S.esp_url + "/stop", tries=2)
        return True
    except ArmError:
        return False


def all_cameras():
    return http_json(S.cam_url + "/tags")


def camera_status():
    cams = all_cameras()
    if "torre" not in cams:
        raise ArmError("The camera program is running but the tower camera is not. Check its ribbon.")
    return cams["torre"]


def base_angle():
    return arm_angles()[5]
