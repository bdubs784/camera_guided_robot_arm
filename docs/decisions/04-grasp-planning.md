# 4. Planning a grab: from 85 seconds to under 4

`pi/armlab/grasp.py` takes "the block is here, turned this much" and turns it into a path: a hover above the block,
a way in, the grasp itself, and a way back out, with every pose checked (see [safety](02-safety.md)). What makes this
hard on this arm is how little room it has. The shoulder only has about 70° of usable travel, and the wrist only
twists about 112°, not a full turn, so not every orientation of the block can be matched just by twisting the wrist.
Close to the base the arm has to fold up so tightly that the only way to reach the block is with the hand pointing
straight down.

## What decides a grasp

- **Square to a face.** The gripper closes on two opposite faces of the block's 39.7 mm square top. Both missed
  pickups on one day had been planned about 20° off square, with the wrist twisted nearly to the end of its range,
  and on film the fingers closed on the corners. So the allowed angle came down from 20° to 12°. A looser 17° pass is
  tried only when nothing squares up: the open fingers still clear a block turned 17°, and the squeeze straightens it.
- **Pull it in.** At the edge of reach, there may be no square grasp on the middle of the block, but 5 mm closer to
  the base there are three. So the planner tries a ladder of "grab this many mm closer to the base than the centre".
- **The hover ladder.** Further than about 300 mm from the base, a hover 60 mm above the block plans easily. Inside
  280 mm, nothing plans until the hover comes down to about 10 mm. So the planner tries hovers from high to low and
  keeps the first one that works.

## The way in: straight lines beat joint space

For a servo arm the natural way to go from the hover down to the block is in joint space: move every joint from its
start angle to its end angle at once. That plans successfully more often close to the base. But on the arm, none of
the joint-space grabs on record held the block. Film and replays showed two reasons. Its hovers end up beside the
block, where the gripper camera sees the tag almost edge-on, and over the last 10 mm the fingertips slide about 15 mm
sideways along the block's face.

So the planner prefers a straight line down. Finding a joint-space plan doesn't stop it from looking for a straight
one, and any joint-space move near the block ends in a short straight vertical tail (25 mm if it fits, then 20,
15...), so the last part of the approach is always straight.

## Making it fast

The first version took **85 s** to plan a block 272 mm out on the laptop, and **139 s** on the Pi. Profiling showed
96% of that time inside the inverse-kinematics solver: 943,000 forward-kinematics calls per plan, each doing tiny 3×3
matrix sums in Python, where the cost of each call is mostly overhead rather than math.

1. **Stop asking the same question.** The hover ladder ran the same grasp search once per rung, six identical
   searches. The fingertip position was recomputed 275,000 times per plan for a gripper that doesn't move. The table
   check rebuilt a constant list on every call. Each now gets cached.
2. **Compile the hot loop.** `pi/armlab/fastik.py` is the same solver (damped Gauss-Newton) on the same kinematics,
   compiled with numba and written so it creates no new arrays inside the loop. It's about 15× faster, and a test
   checks its answers against the Python version.
3. **Don't plan from scratch on the Pi.** `tools/build_plan_table.py` and `grow_plan_table.py` precompute plans on the
   laptop for a grid of block positions and turns across the table (9,957 entries). During a grab the Pi takes the
   nearest entry and re-plans it for the block it actually sees: about 1.4 s instead of 15.6 s.

From scratch, a plan now takes about 3.7 s on the laptop. On the Pi, planning runs in the background while the arm is
already moving, and in the best show the arm spent only about 0.3 s per round waiting for a plan.

## Re-planning

After the close look at the hover, the block's position is known to within a millimetre or two, so the grab re-plans
starting from its first plan rather than from scratch. The plan carries its own settings (grasp depth, pull-in and
angle tolerance), so the re-plan keeps them.
