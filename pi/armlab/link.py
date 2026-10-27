"""
The Pi-ESP32 cable (UART on Pi GPIO14/15 = header pins 8/10 -> ESP32 GPIO18/19, 921600 baud, firmware 4.7+).
Every ESP32 command is sent as its URL path on one line and the answer comes back as one JSON line, so it is the
same command set as WiFi, a few ms instead of 5-30 ms (and no WiFi hiccups). If the cable stops answering, the
program carries on over WiFi by itself and says so once.

Firmware 4.17+: every message is checked. The Pi sends <command>~<n>*<XX> (n = message number, XX = XOR of the
characters before the *), the ESP32 refuses any line whose check fails, and answers <json>~<n>*<XX>. A bit flipped by
motor noise is caught (and the command sent again) instead of becoming a different move, and a late answer to an
earlier message can never be taken for this one.
Firmware 4.18+: a cable line that has lost its check (noise ate the *XX) is refused instead of run unchecked, and a
junk number in a move ("~123") refuses the command instead of reading as 0 (that once sent the shoulder to 0).

    --link auto   (default) the cable if it answers, else WiFi
    --link cable  the cable or stop with an error
    --link wifi   WiFi only
"""
import json
import os
import threading
import time

from .errors import ArmError

PORT = os.environ.get("ARM_UART", "/dev/ttyAMA0")
BAUD = 921600
GIVE_UP_AFTER = 5            # this many failed commands in a row: switch to WiFi for the rest of the run

CABLE = None                 # the open cable, or None = WiFi


class Cable:
    def __init__(self, port=PORT):
        import serial                                    # python3-serial
        self.s = serial.Serial(port, BAUD, timeout=0.5)
        self.lock = threading.Lock()                     # the base watch thread talks too: one command at a time
        self.fails = 0
        self.checked = False                             # firmware 4.17+: checked messages (set by setup)
        self.n = 0
        self.bad = 0                                     # messages lost to noise (refused or garbled), for linktest
        self.count = 0                                   # commands answered, and their round trip (ms, smoothed):
        self.rtt_ms = None                               #   the demo page's link line

    def ask(self, path, timeout=1.5):
        """Send one command (URL path + query) and return its JSON answer."""
        with self.lock:
            t0 = time.time()
            self.s.reset_input_buffer()                  # nothing stale from an earlier, timed-out command
            if self.checked:
                self.n = (self.n + 1) % 100000
                body = f"{path}~{self.n}".encode()
                self.s.write(body + b"*%02X\n" % xor(body))
            else:
                self.s.write((path + "\n").encode())
            t_end = time.time() + timeout
            while time.time() < t_end:
                self.s.timeout = max(0.05, t_end - time.time())
                line = self.s.readline().strip()
                if not line:
                    break
                if self.checked:
                    got = self._checked_answer(line)
                    if got is None:
                        continue                         # garbled, or an answer to an earlier message
                else:
                    if not line.startswith(b"{"):
                        continue
                    try:
                        got = json.loads(line.decode())
                    except ValueError:
                        self.bad += 1
                        raise OSError(f"garbled answer on the cable to {path}")
                if got.get("ok") == 0 and got.get("error") in ("checksum", "unchecked", "unknown command"):
                    self.bad += 1                        # the command arrived damaged: send it again
                    raise OSError(f"the ESP32 got a damaged copy of {path}")
                ms = (time.time() - t0) * 1000.0
                self.count += 1
                self.rtt_ms = ms if self.rtt_ms is None else 0.9 * self.rtt_ms + 0.1 * ms
                return got
            raise OSError(f"no answer on the cable to {path}")

    def _checked_answer(self, line):
        """<json>~<n>*<XX> -> the json if the check matches and n is this message's number; a refusal; else None."""
        if b"*" not in line:
            return None
        body, _, ck = line.rpartition(b"*")
        try:
            if xor(body) != int(ck, 16):
                self.bad += 1
                return None
            if b"~" not in body:                         # the ESP32's "your message arrived damaged" has no number
                return json.loads(body.decode())
            js, _, n = body.rpartition(b"~")
            if int(n) != self.n:
                return None
            return json.loads(js.decode())
        except ValueError:
            self.bad += 1
            return None


def xor(b):
    x = 0
    for c in b:
        x ^= c
    return x


def setup(mode="auto"):
    """Open the cable if it is there and answers. Returns 'cable' or 'wifi'."""
    global CABLE
    CABLE = None
    if mode == "wifi":
        return "wifi"
    why = None
    try:
        c = Cable()
        ver = {}
        for _ in range(3):                               # (a first message can arrive damaged)
            try:
                ver = c.ask("/ver", 0.6)
                break
            except OSError as e:
                why = str(e)
        if ver.get("ok"):
            c.checked = bool(ver.get("ck"))
            CABLE = c
            return "cable"
        why = why or "it did not answer /ver"
    except ImportError:
        why = "python3-serial is not installed (sudo apt install python3-serial)"
    except Exception as e:                                # no port, no permission, no answer...
        why = str(e)
    if mode == "cable":
        raise ArmError(f"--link cable: the cable is not working ({why}).")
    return "wifi"


def ok():
    if CABLE is not None:
        CABLE.fails = 0


def failed(e):
    """One cable command failed. After several in a row, stop using the cable for this run."""
    global CABLE
    if CABLE is None:
        return
    CABLE.fails += 1
    if CABLE.fails >= GIVE_UP_AFTER:
        print(f"   (the cable to the ESP32 stopped answering: {e}. Carrying on over WiFi.)")
        CABLE = None
