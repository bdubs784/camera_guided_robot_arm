# 2. Every move is checked, in three layers

In the demo loop the arm runs on its own for minutes at a time. It shares the table with its own camera tower and a
stepper base, and it has already burned out two shoulder servos. For this arm a collision isn't just a failed grab,
it's broken hardware, so nothing moves without being checked first.

## Layer 1: the safety map (`pi/armlab/safety.py`, `tools/build_arm_map.py`)

This is a grid computed ahead of time over the shoulder, elbow and wrist angles. Each cell stores how much clearance
the arm has from the tower, the table and itself in that pose. Looking it up is nearly free, so every planned path
is checked at every step, and while the arm moves a watcher stops it if the clearance drops more than 10 mm below
the 40 mm margin. Its weakness is that it was built from tape-measure kinematics and one fixed hand pose, which is
why layer 2 exists.

## Layer 2: the exact table check (`pi/armlab/tablecheck.py`)

Touch tests showed the map claiming 20–57 mm of room in poses where the hand was actually on the table. A closed
gripper reaches about 20 mm further than an open one, twisting the wrist changes the hand's shape, and very low
reaches were outside what the map had really computed. So the exact check builds the real hand (with the fingers
where the gripper actually is), the wrist and the links from the world model, at the real base angle, and measures
their distance to the table, the plywood and the base.

Twice it refused moves that were fine, and both times the fix was to make it more accurate:

- I first modelled the plywood and base as one profile spun around the base axis, using the tallest thing at each
  distance. That turned the 69 mm stepper housing, which only sits on one side, into a wall all the way round. Now
  it's a map by direction and distance.
- Comparing each point against its own direction bin and the neighbouring ones was too coarse: 190 mm out, one bin
  is 50 mm wide. Now each cell is a small box and the distance to it is measured properly.

It also had to be fast, because the planner calls it thousands of times per plan. Measuring every one of about 2,000
points on the arm against each of the 727 cells took 134 ms per call, and a whole plan took 115 s. The trick: the old
spun profile can never be farther away than the real obstacles, so it gives a cheap lower bound on the distance. The
check sorts the points by that bound and stops as soon as the bound is bigger than the closest real distance found
so far. The answer is identical and most of the work is skipped. Compiled with numba, a call takes about 5 ms.

## Layer 3: the firmware

The ESP32 clamps every joint to its own limits, whatever sent the command: the Pi, the web page, the game
controller, or its own move to the rest pose after a power cut.

The serial cable between the Pi and the ESP32 is checked as well. Once, electrical noise from the motors corrupted a
command on its way down the cable. A garbled number in it was read as 0, the shoulder drove to 0, and the fingers
dug into the table. Now every line on the cable carries a message number and an XOR checksum, a line without them
can't move anything, and a field that isn't a valid number gets the whole command refused.

## The rule I kept

Never loosen a check to make a plan pass. When a check refused something that looked fine, the fix was always to
make the check more *accurate* (the direction map, the real finger positions, the measured slack), never more
lenient.
