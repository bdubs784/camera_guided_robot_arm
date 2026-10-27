# Firmware changelog

Each version fixed something the arm actually did. The `vX.Y` tags in the code comments point here.

**4.20 — the shoulder sensor is a 10k NTC thermistor** (not a TMP36): 3.3 V – 10k resistor – GPIO 35 – NTC – GND.
Beta model (10k at 25 °C, B 3950). An open or shorted sensor reports `"st": null`. Measured 11.78k at ~21 °C.

**4.19 — shoulder temperature.** A sensor taped to the shoulder servo on GPIO 35, read every 50 ms, smoothed over
~2 s, reported in `/tel` as `"st"` (°C). The Pi's survey rests the arm limp when the shoulder runs hot; two servos
had already burned out on that joint.

**4.18 — no unchecked moves on the cable.** A survey poll (`/setall?a=…~n*XX`) lost its `*XX` to motor noise, arrived
looking like an old-style unchecked line, and `~n` was read as the number 0: the shoulder went to 0 and the fingers
dug into the table. Now a line on the Pi cable without a valid check is refused (`{"ok":0,"error":"unchecked"}`)
unless it only reads (`/ver`, `/tel`), and every number in a move must parse as a number: `"~123"` or `"4x"` refuses
the whole command (`"bad number"`) instead of reading as 0. Wrist rotate limited to 0–112 (was 130) for every
source: above 112 the hand runs into itself.

**4.17 — checked cable messages.** A cable command may end in `~<n>*<XX>`: n is the Pi's message number, XX the XOR
of every character before the `*` in hex (like NMEA/GPS sentences). A line whose check does not match is refused,
so a bit flipped by motor noise can never become a different move. The answer comes back as `<json>~<n>*<XX>`, so
the Pi can tell it apart from a late answer to an earlier message.

**4.16 — faster cable answers.** The loop still runs every ~10 ms (the old glides are timed by it), but during the
10 ms pause the Pi cable is polled every millisecond, so a command is answered within ~1 ms instead of up to 10+ ms.
Waking at power-up now goes shoulder first.

**4.15 — rest pose.** A pose the arm can sit in limp without falling (saved in flash). At power-up every joint is
commanded to the rest pose instead of 90/90/90, so switching the servos on holds the arm where it already sits
instead of flinging it.

    /rest                     show the rest pose and the wake setting
    /rest?set=1               the current pose becomes the rest pose
    /rest?a=104,79,62,42,80   set it by numbers (shoulder, elbow, wrist, wrist rotate, gripper)
    /rest?wake=1              switch the servos on at power-up, holding the rest pose
    /relax?ms=3000            glide to the rest pose, then servos off

**4.14 — test switches** for "the base turns more than its count while the arm moves" (saved in flash):
`/diag?sw=20` writes each servo at most every 20 ms, `/diag?core=0` runs the step driver on core 0,
`/diag?drv=pcnt` picks the MCPWM/PCNT driver. `/tel` adds the longest servo write (`wu`) and the write count (`wn`).
These found the cause: each servo write blocked the loop for 29.5 ms, and the step driver's timing ran on the same
core, so it was starved whenever the arm moved. With the step driver on core 0 a 45° turn during an arm swing came
out within 1% of the count. Core 0 is the default now; servo writes are also spaced 50 ms apart by default (4.15).

**4.13 — measurement tools** for the same problem: `/tel` adds the base speed the step library believes (`bv`), the
longest gap between loop runs (`lp`), pulses counted on a test input (`ec`) and driver state (`da`). `/diag?echo=1`
counts rising edges on GPIO 34; a jumper from the STEP wire counts the pulses really on that wire.

**4.12 — RMT step driver.** With the arm moving, the base really turned ~2× what its count said (count 45, base at
90). The first suspect was the step library's MCPWM/PCNT driver, which counts its own pulses back and could miss
some, so the base moved to the RMT driver, which emits an exact number of pulses (`BASE_USE_RMT`). It didn't cure
the drift (the cause is under 4.14), but RMT stayed.

**4.11 — smooth slow base moves.** `/setall` takes `bs=` (deg/s) and `ba=` (deg/s²) for the base, so a slow pan is one
constant-speed move instead of a new target five times a second (that stop-start made the stepper squeal).

**4.10 — base driver sleeps when idle.** The worm gear holds the base by itself, so the driver is switched off 0.4 s
after every base move and back on just before the next. That way nothing on the STEP line can turn the base while
it's meant to stand still.

**4.9 — base memory.** The worm can't be back-driven, so the base stays put with the power off. Its step count is
saved to flash once the base has stood still for a second and restored at power-up.

**4.8 — per-joint trim.** After a servo swap the horn sits a few degrees off; `/trim?j=0&d=-4.5` shifts the pulse so
the same command gives the same physical angle and no program has to change its numbers.

**4.7 — measurement and the cable.** `/tel` (telemetry with ESP timestamps), `/ver`, `id=` on any move,
`prof=1` clock-timed minimum-jerk moves (`prof=2` adds input shaping: `f=` wobble Hz, `z=` damping). The same
commands work over USB serial (115200) and the Pi UART (GPIO 18 RX / 19 TX, 921600 baud), one line in, one JSON line
out; `/telstream?hz=50` streams telemetry. Serial is read without blocking. WiFi power saving off (it caused 200 ms
answer delays). Wireless uploads (ArduinoOTA).

**4.6** — the game controller drives the base by speed, one smooth ramp, reversing only from a standstill.
**4.5** — elbow and wrist lower limits opened to 0.
**4.4** — controller moves glide between updates instead of stepping.
**4.3** — game controller support in the page (browser Gamepad API), a STOP button, `/setall` and `/stop`.
**4.2** — the page shows both Pi camera streams.
**4.0** — the base stepper (NEMA 17 + 40:1 worm) as joint 6, and per-joint power on/off.
