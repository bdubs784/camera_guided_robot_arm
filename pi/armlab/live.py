"""
The demo page: what the arm is doing right now, for a screen recording next to the camera views.

    show / grasp serve it while they run:   http://<pi>:8090        (the cameras stay on the armcams service, :8080)

The grab calls update(...) as it goes -- the step it is on, where it thinks the block is and how sure it is, how well
the cameras agree, the slack it measured, the squeeze, where it is putting the block, how long each part took -- and
the page polls /live. The camera pictures come straight from the armcams streams; the page only draws the block's
outline over them (from /tags) so it stands out from every other tag box. Nothing here moves anything, and a failure
in here never stops a grab.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8090
_LOCK = threading.Lock()
STATE = {"step": "Waiting to start", "t": time.time(), "events": []}
_SERVER = [None]
_LAST_POLL = [0.0]                # when the page last asked: the telemetry poll runs only while someone is watching
TEL_HZ = 5.0                      # the ESP32's /tel for the page's joints panel: current and target angle per joint,
                                  # shoulder temperature, its main loop's worst time, servo write time, base speed.
                                  # Over the cable it is a few ms a read and the link's lock keeps it one at a time
BASE_SHOWN = (-60.0, 90.0)        # the base's track on the page (its working range is -57..90)

# what the page says while each part runs (the grab books a part when it ENDS, so this is the one that comes next)
NEXT_STEP = {
    "find": "Planning the grab", "search": "Planning the grab", "plan": "Looking from more angles",
    "looks": "Moving above the block", "to hover": "Close-up look", "close look": "Re-planning from up close",
    "re-plan": "Reaching in", "in": "Closing the gripper", "close": "Lifting it", "lift": "Checking the grip",
    "squeeze": "Choosing where to put it", "carry plan": "Carrying it", "carry": "Setting it down",
    "down": "Letting go", "let go": "Checking where it landed", "check": "Going home",
    "home": "Round done", "home check": "Round done",
}


def update(**kw):
    """Merge fields into the state the page shows. Never raises."""
    try:
        with _LOCK:
            STATE.update(kw)
            STATE["t"] = time.time()
    except Exception:
        pass


def event(text):
    """A line for the page's short "what just happened" list (the last eight)."""
    try:
        with _LOCK:
            ev = STATE.setdefault("events", [])
            ev.append([round(time.time(), 1), str(text)])
            del ev[:-8]
            STATE["t"] = time.time()
    except Exception:
        pass


def lap(name, rows):
    """From Laps.lap: a part just ended -- say what comes next, and keep the timings."""
    update(step=NEXT_STEP.get(name, name), laps=[list(r) for r in rows])


def check(gap, margin):
    """From motion.move_to: a move's safety check -- the tightest clear space on its line against the margin. The page
    shows the last one, the tightest this round and how many (cleared at the round's start). Never raises."""
    try:
        with _LOCK:
            s = STATE.get("safety") or {}
            n = int(s.get("n", 0)) + 1
            low = min(float(gap), float(s.get("min", gap)))
            STATE["safety"] = {"last": float(gap), "min": low, "margin": float(margin), "n": n}
    except Exception:
        pass


def slack_pair(measured, predicted):
    """From the close look: the slack the cameras measured and the slack model's prediction made before it, for the
    page's running score (shoulder, elbow, wrist; the last 40). Never raises."""
    try:
        with _LOCK:
            log = STATE.setdefault("slack_log", [])
            log.append([[float(v) for v in measured[:3]], [float(predicted[k]) for k in ("shoulder", "elbow", "wrist")]])
            del log[:-40]
            STATE["slack_pred"] = {k: float(predicted[k]) for k in ("shoulder", "elbow", "wrist")}
    except Exception:
        pass


def snapshot():
    with _LOCK:
        out = json.loads(json.dumps(STATE, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    out["now"] = time.time()                        # the Pi's clock, for how old a camera frame is
    try:                                            # the ESP32 cable: commands answered, round trip, damaged
        from . import link
        c = link.CABLE
        out["link"] = {"kind": "cable", "count": c.count, "rtt_ms": c.rtt_ms, "bad": c.bad, "checked": c.checked} \
            if c is not None else {"kind": "wifi"}
    except Exception:
        pass
    try:                                            # the trace's CPU sampler (trace.py): the latest reading
        from . import trace
        if trace.CPU:
            t, busy, own, temp = trace.CPU[-1]
            out["cpu"] = {"busy": busy, "own": own, "temp": temp}
    except Exception:
        pass
    return out


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                      # quiet: a log line per poll would fill the SD card
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if self.path.startswith("/live"):
                _LAST_POLL[0] = time.time()
                self._send(json.dumps(snapshot()).encode(), "application/json")
            elif self.path == "/" or self.path.startswith("/?"):
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            else:
                self.send_error(404)
        except Exception:
            pass


def _tel_loop():
    """While the page is open: read the ESP32's telemetry TEL_HZ times a second into STATE["tel"]. Never raises."""
    from .hardware import http_json
    from .state import S
    lim, n, t_rate = None, 0, time.time()
    while True:
        try:
            if time.time() - _LAST_POLL[0] > 5.0 or not getattr(S, "esp_url", None):
                time.sleep(0.5)
                continue
            if lim is None:
                js = http_json(S.esp_url + "/state", timeout=1.0, tries=1)["joints"]
                lim = [[float(j["min"]), float(j["max"])] for j in js if not j.get("stepper")] + [list(BASE_SHOWN)]
            d = http_json(S.esp_url + "/tel", timeout=1.0, tries=1)
            n += 1
            now = time.time()
            tel = {"p": list(d.get("p") or []) + [d.get("b")], "tg": list(d.get("tg") or []) + [d.get("bt")],
                   "mv": list(d.get("mv") or []) + [d.get("br")], "lim": lim, "temp": d.get("st"),
                   "loop_ms": d.get("lp"), "write_us": d.get("wu"), "base_dps": d.get("bv"), "t": now}
            if now - t_rate >= 2.0:
                tel["hz"] = n / (now - t_rate)
                n, t_rate = 0, now
            with _LOCK:
                tel.setdefault("hz", (STATE.get("tel") or {}).get("hz"))
                STATE["tel"] = tel
        except Exception:
            time.sleep(1.0)
        time.sleep(1.0 / TEL_HZ)


def serve(port=PORT):
    """Start the page's server in the background (once). Returns the URL, or None if the port is taken."""
    if _SERVER[0] is not None:
        return _SERVER[0].url
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    except OSError:
        return None
    srv.daemon_threads = True
    srv.url = f"http://{os.uname().nodename if hasattr(os, 'uname') else 'localhost'}:{port}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    threading.Thread(target=_tel_loop, daemon=True).start()
    _SERVER[0] = srv
    return srv.url


PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_page.html"), encoding="utf-8").read() \
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_page.html")) else "<h1>live_page.html missing</h1>"
