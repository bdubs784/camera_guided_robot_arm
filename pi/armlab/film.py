"""
Film the cameras while something runs: ~/arm_data/film/<time>/<camera>_<t>.jpg, FILM_HZ frames a second each.

Reads the camera server only; it never talks to the arm. show --film starts it with the show and stops it when the
show ends, however it ends (a separate recorder needed its length guessed up front).

    f = Film().start()
    ...
    f.stop()      # -> (folder, frames saved)
"""
import os
import threading
import time
import urllib.request

import numpy as np

from .state import S

FILM_HZ = 2.0                    # frames a second per camera: enough to watch a round back, small on the SD card
FILM_WIDTH = 640                 # saved this wide
FILM_STREAMS = (("torre", "torre"), ("gripper", "gripper"))   # MJPEG streams on the camera server
FILM_STILLS = (("webcam", "raw/webcam"),)                      # the webcam: one picture per request


class Film:
    def __init__(self, root=None):
        self.out = os.path.join(root or os.path.join(S.data_dir, "film"), time.strftime("%Y%m%d_%H%M%S"))
        self._stop = threading.Event()
        self._threads = []
        self.frames = 0
        self._lock = threading.Lock()

    def start(self):
        os.makedirs(self.out, exist_ok=True)
        self._threads = [threading.Thread(target=self._stream, args=c, daemon=True) for c in FILM_STREAMS] + \
                        [threading.Thread(target=self._stills, args=c, daemon=True) for c in FILM_STILLS]
        for t in self._threads:
            t.start()
        return self

    def stop(self):
        """Stop filming; (folder, frames saved)."""
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)
        return self.out, self.frames

    def _save(self, cam, jpg):
        import cv2
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        h, w = img.shape[:2]
        img = cv2.resize(img, (FILM_WIDTH, int(h * FILM_WIDTH / w)))
        cv2.imwrite(os.path.join(self.out, "%s_%.2f.jpg" % (cam, time.time())), img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with self._lock:
            self.frames += 1

    def _stream(self, cam, path):                # MJPEG: keep reading, save the newest frame FILM_HZ times a second
        while not self._stop.is_set():
            try:
                r = urllib.request.urlopen(S.cam_url + "/" + path, timeout=10)
                buf, last = b"", 0.0
                while not self._stop.is_set():
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    buf += chunk
                    a, b = buf.find(b"\xff\xd8"), buf.find(b"\xff\xd9")
                    while a != -1 and b != -1 and b > a:
                        jpg, buf = buf[a:b + 2], buf[b + 2:]
                        if time.time() - last >= 1.0 / FILM_HZ:
                            self._save(cam, jpg)
                            last = time.time()
                        a, b = buf.find(b"\xff\xd8"), buf.find(b"\xff\xd9")
                    if len(buf) > 4_000_000:
                        buf = buf[-1_000_000:]
            except Exception:
                self._stop.wait(1.0)             # the camera server hiccuped: try again, never stop the show

    def _stills(self, cam, path):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._save(cam, urllib.request.urlopen(S.cam_url + "/" + path, timeout=5).read())
            except Exception:
                pass
            self._stop.wait(max(0.0, 1.0 / FILM_HZ - (time.time() - t0)))
