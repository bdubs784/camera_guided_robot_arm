# 7. Seeing the block

There are three cameras: a Pi camera and a USB webcam on the tower, which turn with the base, and a Pi camera on the
hand. The block has an AprilTag on top.

## One look isn't a measurement

The same webcam once put the same block 19 mm apart when it looked from two base angles 12° apart. So the grab takes
several looks and combines them in a small Kalman filter that tracks the block's x, y, height and turn
(`pi/armlab/cube.py`, `BlockFilter`). Each new look nudges the estimate in proportion to how much it's trusted.

A look far from the estimate usually means someone moved the block, so the filter starts over from it. But when
nothing has touched the block since the last look, a far reading is just a bad reading, so it gets averaged in like
any other and can't replace a good estimate by itself.

Each camera also misjudges the height of the block's top in its own way. Measured on 31 blocks between 194 and 445 mm
from the base, the tower camera reads 4.2 mm high, the webcam 9.0 mm high, and the gripper camera's error changes
with distance (9 mm low close in, 11 mm high at 450 mm). The grab's depth is measured from the top of the block, so
those corrections are part of the world model.

## Where to look from

- **The search pose.** The tower cameras don't move with the arm, so the arm can only get in their way. I tried
  candidate search poses against 88 spots on the table, and the one I chose was the only one that hides none of them
  from the tower camera. Compared with an old pose, the tower camera sees the block from 33 spots instead of 27, and
  the webcam from 67 instead of 65.
- **Where to wait between rounds.** I checked every look on record against one measure: in the image, how many
  pixels separate the block's tag from the nearest part of the arm. Waiting up high, the arm stands right in front of
  the webcam, which saw the block in only 4 looks out of 60. With 200 px or more clear, the tower camera saw the block
  11 times out of 11; with 100–200 px, only 1 in 4. So the arm waits where there's at least 200 px clear.
- **The close look at the hover.** The model said the tag would be 200 px inside the gripper camera's frame, yet run
  after run it wasn't in the picture at all. Rather than trust the prediction, the hover now searches: if the gripper
  camera doesn't see the tag, it sweeps the wrist, then the elbow, until it does.

## Fresh frames

Looks used to use frames 0.2–0.8 s old, sometimes taken while the arm was still moving. Twice the webcam "saw
nothing" right after a move, with 253 px clear, while its own recording showed the tag plainly. Now every look waits
for a frame captured after the arm stopped, and only from the cameras it needs. Frames take time to arrive: about
417 ms after capture from the webcam and about 121 ms from the gripper camera.

## Confirming the grab

Three independent checks:

- **The squeeze.** Starting from a light close, an empty hand closes about 5 mm further, while a hand holding the
  block barely moves.
- **The lift.** A tower camera watches the block itself rise.
- **The next round's look.** The block is where it was put.
