# 3. The base: a worm gear that doesn't land where you tell it

The base turns on a NEMA 17 stepper through a 40:1 worm gear. A worm can't be turned backwards by the load, which is
great for holding position with the power off. The downside is play. Where the base stops depends on which way it was
turning: the two directions end up about 0.85° apart on average, and up to 1.8°. Even repeating the same move, it
lands anywhere within ±0.6°. That sounds small, but 300 mm out, 1° at the base moves the fingers 5 mm, and the
gripper only has about 12 mm of room on each side of the block.

## Making the step count honest

The stepper has no position sensor, so the firmware counts the steps it sends. Early on that count was badly
wrong: whenever the arm moved at the same time, the base turned about twice as far as the count said (once the count
read 45° with the base really at 90°).

To find out why, I added test tools to the firmware. A jumper from the stepper's STEP wire back into a spare input
counted the pulses that actually went out, and switches let me move the step driver to the other CPU core or swap
step drivers without reflashing. The cause turned out to be timing. Each servo write blocked the ESP32 for 29.5 ms,
and the stepper's pulse timing ran on that same core, so every servo update starved it.

The ESP32 has two cores, so the fix was to give the stepper its own. With the step driver on core 0 and the servo
writes on core 1, a 45° turn while the arm was swinging came out within 1% of the count. The firmware's
[changelog](../../firmware/esp32_arm/CHANGELOG.md) (4.12–4.14) has the full hunt.

## Closing the loop with the cameras (`pi/armlab/baseloop.py`)

Even with an honest count, the worm's play means the base doesn't land exactly where it was told. The tower camera
and the webcam turn with the base and see dozens of fixed tags on the table, so with the world model they can measure
the base's real angle to about 0.1°. The loop is simple: move, measure, nudge by whatever is left, measure again.

Tuning it was mostly about understanding the worm:

- **Small nudges from a standstill did nothing.** A +1.25° nudge didn't move the base at all, while +2.2° did: the
  gear has to wind up before it turns. So every nudge now backs off 3° and comes forward in one move, and the gear is
  already turning when it reaches the target.
- **I asked for more precision than the gear has.** At a 0.1° target the loop used up all five of its tries in most
  moves (12–32 s each time), chasing an angle the worm can't reliably hit. 0.3° is about 1.5 mm at 300 mm out, and
  the gripper camera's close look later in the grab corrects whatever is left.

## Then, using the loop only where it pays off

Once the loop worked, I measured what it cost. Closing it on the way to the hover added about **9.5 s to every grab**
(73 closed-loop moves took 381 s in total over the recorded runs). And there it bought nothing, because when the arm
reaches the hover the gripper camera takes a close look at the target to grab, re-measures the base angle and re-plans from that anyway.

So now the base simply trusts its step count on the way to the hover and during the carry (open-loop moves saved
another 1.5–3 s per carry). It closes the loop only once, just before the fingers go in, and only if the close look
says the base is 0.5° or more off.

## Zeroing

The step count needs a zero. `basehome --world-only` finds it from the same table tags the world model is built on,
seen by the tower cameras, so the base's zero and the planner's picture of the world agree by construction. The
firmware saves the count to flash whenever the base stands still, so it survives a power cut.
