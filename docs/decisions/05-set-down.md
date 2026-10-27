# 5. Setting the block down where you meant to

Picking up got reliable before putting down did. The demo loop carries the block to a new distance from the base
every round (270 → 305 → 340 → 365 mm and back), so wherever a set-down lands is where the next round has to pick
it up. A set-down 20 mm off makes the next grab harder, and one that lands out of reach ends the show.

## Measure first: an offline replay

Every round logs where the fingers let go and where the next look found the block. `tools/replay_place.py` re-runs
the set-down planner on those recorded rounds without the arm, so I could check a change against every past round in
seconds instead of running a whole show.

It found the biggest bug right away. When looking for a set-down, `reachable_at` only tried holding the block the
same way it was picked up, with no change of direction and only 5–10° of change in the hand's tilt. Carrying the
block 40–70 mm to the next distance needs up to 20°. So every new distance was refused, and on the arm 8 of 24 rounds
simply put the block back where it was. After the fix, all 24 recorded rounds found a set-down in replay.

## Why the block slid outward

Across every set-down on record, blocks landed about 12–16 mm further from the base than aimed, and more the further
out they were. That turned out to have several causes, found one at a time from film and logs:

- **Dragging on release.** Opening the gripper straight to wide swung each finger out along its arc while it was
  still touching the block, and each set-down turned the block about 10° more the same way (it went 15.7° → 25° →
  32° → 40° over four rounds). Now the gripper opens just ~4 mm per side, lifts, and only then opens fully.
- **Lifting along the joint arc.** After a joint-space set-down, lifting back along the same arc dragged the
  fingertips 14–22 mm outward while still within 10 mm of the block's top, and blocks were found 14–60 mm out.
  Set-downs that came straight in and lifted straight up landed within about 8 mm. So every set-down now ends with a
  short straight approach and lifts straight up.
- **Measuring from the wrong point.** The let-go point was computed from the fingertip as if the gripper were
  closed, and a closed fingertip sits 21.6 mm lower than an open one. So every logged release looked about 20 mm under
  the spot.
- **Joints at the end of their range.** Set-downs with a joint at its limit, or with the shoulder far forward,
  landed 20–61 mm out. With room to spare on every joint they landed within 7 mm. The planner now prefers those.

What was left after these fixes was a steady outward drift that grows with distance: fitting a line to 16 set-downs
gave +7 mm at 300 mm from the base and +21 mm at 375 mm. So set-downs now aim a little closer to the base to cancel
it: 7.3 mm closer at 300 mm, plus 0.19 mm for every mm beyond that (about 20 mm closer at 365 mm). The far end of the
walk also came in from 375 to 365 mm, after one set-down at 375 slid out to 410 mm, past where anything can be
planned.

## Result

The three shows before the last fixes landed 13–25 mm from the aimed spot on average (worst 43 mm). The best show
after them carried and set down 8 of 8 rounds, landing **6.2 mm on average from the aimed spot** (worst 12.6 mm).
Measured along the line out from the base, they landed on average 3.9 mm short of the aim, give or take 5.5 mm, so
the outward drift is gone. Source: [`results/2026-10-23_best_show`](../../results/2026-10-23_best_show), via
`tools/show_report.py`.
