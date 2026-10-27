# 1. A world model solved from the cameras, not measured by hand

## The problem

Hobby servos don't report where they are. You ask for an angle and the servo gets close, but gear play, the
counterbalance spring and whatever the arm is holding can leave a joint a few degrees off. That matters more than it
sounds. A 2–3° error at the shoulder moves the fingertips 10–15 mm, and the open gripper (65 mm) only leaves about
12 mm of room on each side of the 39.7 mm block. So "roughly right" kinematics miss the block.

I started with link lengths from a tape measure plus a small learned correction. It had to be
redone every time a servo was swapped, and it was never good enough at the edge of reach.

## What I did instead

**Survey, then solve for everything at once.** `arm.py survey` drives the arm through hundreds of random safe poses
and base angles (in chunks, resting in between so the servos cool down) and records every AprilTag corner that every
camera sees. Then `tools/big_solve.py` runs a bundle adjustment: one big least-squares fit that adjusts all the
unknowns together until the model predicts every recorded tag corner as closely as possible. It solves for:

- the link lengths and joint gearing, the base axis and the wrist-rotate axis
- where all three cameras are (two on the tower, one on the hand)
- where every tag is, and which part of the arm or table it's stuck to (worked out automatically)
- how far off each joint was in each pose (the slack), so the fit doesn't bend the arm's geometry to explain it

To check that the solver can be trusted, I ran it on fake survey data where I knew the true answer: 250 poses, with
0.4 px of noise on every tag corner and 0.4° of gear play on the joints. It assigned every tag to the right part and
put the hand within 0.2 mm of the truth (median). The model the arm uses came from a real 623-pose survey.

## What the solve found that I wouldn't have measured

- **A wrist convention that wasn't pinned down.** The solver could turn the hand's frame a few degrees around the
  wrist axis and shift the wrist joint's zero the opposite way, and the result described exactly the same arm. The
  data can't tell those apart, so two solves landed 5–6° apart, and the planner's pitch limits (which assume one
  convention) quietly stopped meaning what they said. `tools/regauge_wrist.py` now fixes the convention after every
  solve.
- **The middle of the grip was 10 mm off.** In film of held blocks (`tools/finger_center.py`), the block sat
  −0.3 ± 0.6 mm from centre in the gripper camera, but the model put the centre of the grip 10.2 mm toward one
  finger. Close-in misses had been closing on the block's far edge. I fixed it in the model rather than adding an
  offset somewhere else.

**Touch calibration** (`arm.py touch`, then `fit_touch`) is an independent check: the arm touches points on the
table, and the hand model agrees with them to about ±3 mm.

## What it costs

The solve runs on the laptop and takes minutes, so it can't run on the Pi. Re-surveying after a hardware change takes
about 45 minutes of arm time. The Pi's system numpy (which the camera stack depends on) is too old for current scipy,
so rather than fight that, the Pi side doesn't use scipy: the small fits it does run during a grab
(`world._robust_fit`) are written out in plain numpy.
