"""
What the arm was doing, moment by moment, for finding where a grab's time goes. Laps say how long each phase took;
this says what filled it.

    with trace.span("base", "leg -12.0"):     # every arm glide, base leg, hand move, wait, camera look, plan
        ...
    trace.sample(arm_angles())                # the joint readings the moves already poll

finish_run writes trace.dump() into the run file; tools/timeline.py splits each phase into moving, base, hand,
looking, computing, waiting -- and the time nothing at all was recorded doing anything.
"""
import contextlib
import threading
import time

EVENTS = []                       # [t0, t1, kind, what]
SAMPLES = []                      # [t, shoulder, elbow, wrist, rotate, gripper, base]
CPU = []                          # [t, whole Pi busy %, this program's %, CPU temp C] every CPU_EVERY_S (Linux only)
CPU_EVERY_S = 0.5                 # worth watching: the camera server alone can use ~335% of the Pi's 400%
_LOCK = threading.Lock()          # the carry is planned on another thread
_CPU_THREAD = []


def reset():
    with _LOCK:
        EVENTS.clear()
        SAMPLES.clear()
        CPU.clear()
    if not _CPU_THREAD:
        t = threading.Thread(target=_cpu_loop, daemon=True)
        _CPU_THREAD.append(t)
        t.start()


def _cpu_now():
    """(total jiffies, idle jiffies, this process's jiffies, temp C or None), or None off Linux."""
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        with open("/proc/self/stat") as f:
            s = f.read().rsplit(")", 1)[1].split()
        own = int(s[11]) + int(s[12])                    # utime + stime
    except (OSError, ValueError, IndexError):
        return None
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read()) / 1000.0
    except (OSError, ValueError):
        temp = None
    return sum(v), v[3] + (v[4] if len(v) > 4 else 0), own, temp


def _cpu_loop():
    import os
    ncpu = os.cpu_count() or 1
    last = _cpu_now()
    while last is not None:
        time.sleep(CPU_EVERY_S)
        now = _cpu_now()
        if now is None:
            return
        dt, di = now[0] - last[0], now[1] - last[1]
        if dt > 0:
            row = [round(time.time(), 3), round(100.0 * (dt - di) / dt, 1),
                   round(100.0 * ncpu * (now[2] - last[2]) / dt, 1), now[3]]
            with _LOCK:
                CPU.append(row)
        last = now


@contextlib.contextmanager
def span(kind, what=""):
    t0 = time.time()
    try:
        yield
    finally:
        with _LOCK:
            EVENTS.append([round(t0, 3), round(time.time(), 3), kind, str(what)])


def sample(a):
    try:
        row = [round(time.time(), 3)] + [round(float(v), 2) for v in list(a)[:6]]
    except (TypeError, ValueError):
        return
    with _LOCK:
        SAMPLES.append(row)


def dump():
    with _LOCK:
        return {"events": [list(e) for e in EVENTS], "samples": [list(s) for s in SAMPLES], "cpu": [list(c) for c in CPU]}
