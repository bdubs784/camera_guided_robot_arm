#!/usr/bin/env python3
"""
A printable AprilTag mat: small tags, packed close, for the side of the table the grabs happen on.

    python tools/make_mat.py --out ~/Downloads

The mat already in service is 30 mm tags about 60 mm apart. From the gripper camera, 150-250 mm up, that is two
or three tags in the whole picture, and a camera that can only see two or three tags at a time is weakly tied
down by them. This sheet is 15 mm tags on a 25 mm pitch: the same patch of table carries about six times as
many, so the gripper camera has a crowd of them in view at once wherever it looks, which is what a solve needs
to pin a camera's position rather than just its aim.

The ids start at 400, well clear of everything in use (the mat is 100-135, the wedges run to 380), and
tag36h11 has 587 of them, so there is no risk of a new tag being mistaken for one already on the arm.

Printing: at 100% scale, no "fit to page". The sheet prints a 100 mm ruler -- measure it before taping anything
down. If it does not read 100 mm the tags are the wrong size and every distance the solve learns from them
will be wrong by the same factor.

The tags mean nothing until they are surveyed in: lay the mat, run a survey, then the solve works out where
each one is. Nothing has to be measured by hand.
"""
import argparse
import os

import cv2
from PIL import Image, ImageDraw, ImageFont

DPI = 600                      # plenty for a 15 mm tag: one tag cell comes out about 44 dots
CELLS = 8                      # tag36h11 is 6x6 of data inside a one-cell black border; that 8x8 is "the tag"


def mm(x, dpi=DPI):
    return int(round(x / 25.4 * dpi))


def sheet(ids, tag_mm, pitch_mm, page_mm, margin_mm, label):
    """One page: a grid of tags with their ids beside them, and a ruler to check the print scale."""
    W, H = mm(page_mm[0]), mm(page_mm[1])
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", mm(3.2))
        small = ImageFont.truetype("arial.ttf", mm(2.6))
    except Exception:
        font = small = ImageFont.load_default()

    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    usable_w = page_mm[0] - 2 * margin_mm
    usable_h = page_mm[1] - 2 * margin_mm - 18.0          # 18 mm kept at the bottom for the ruler and the caption
    cols = int(usable_w // pitch_mm)
    rows = int(usable_h // pitch_mm)
    px = mm(tag_mm)
    px -= px % CELLS                                       # whole dots per cell, so no cell is a dot wider than another
    used = []
    for i, tag in enumerate(ids[:cols * rows]):
        r, c = divmod(i, cols)
        x = mm(margin_mm + c * pitch_mm)
        y = mm(margin_mm + r * pitch_mm)
        a = cv2.aruco.generateImageMarker(dic, int(tag), px)
        img.paste(Image.fromarray(a), (x, y))
        d.text((x, y + px + mm(0.6)), str(tag), font=small, fill=0)
        used.append(tag)

    # the ruler: measure it after printing
    y0 = mm(page_mm[1] - margin_mm - 10.0)
    x0 = mm(margin_mm)
    d.line([(x0, y0), (x0 + mm(100), y0)], fill=0, width=mm(0.4))
    for k in range(11):
        t = x0 + mm(10 * k)
        d.line([(t, y0 - mm(2 if k % 5 else 3.5)), (t, y0 + mm(2 if k % 5 else 3.5))], fill=0, width=mm(0.3))
    d.text((x0, y0 + mm(4.5)), "this line is 100 mm - measure it before taping anything down", font=font, fill=0)
    d.text((x0, y0 - mm(9)), label, font=font, fill=0)
    return img, used, cols, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads"))
    ap.add_argument("--first-id", type=int, default=400)
    ap.add_argument("--tag-mm", type=float, default=15.0)
    ap.add_argument("--pitch-mm", type=float, default=25.0)
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--paper", choices=("letter", "a4"), default="letter")
    ap.add_argument("--margin-mm", type=float, default=10.0)
    args = ap.parse_args()

    page = (215.9, 279.4) if args.paper == "letter" else (210.0, 297.0)
    os.makedirs(args.out, exist_ok=True)
    pages, all_used = [], []
    nxt = args.first_id
    for p in range(args.pages):
        ids = list(range(nxt, nxt + 200))
        label = ("left-side mat, page %d of %d: %.0f mm AprilTag 36h11, %.0f mm pitch, ids from %d"
                 % (p + 1, args.pages, args.tag_mm, args.pitch_mm, nxt))
        img, used, cols, rows = sheet(ids, args.tag_mm, args.pitch_mm, page, args.margin_mm, label)
        pages.append(img.convert("RGB"))
        all_used += used
        nxt = used[-1] + 1
    path = os.path.join(args.out, "mat_left_%dmm_%dpitch_id%d.pdf"
                        % (args.tag_mm, args.pitch_mm, args.first_id))
    pages[0].save(path, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
    print("%d tags, ids %d-%d, %d per page (%d x %d), %.0f mm tags on a %.0f mm pitch"
          % (len(all_used), all_used[0], all_used[-1], len(all_used) // len(pages), cols, rows,
             args.tag_mm, args.pitch_mm))
    print("-> %s" % path)
    print("\nPrint at 100%% scale (no fit-to-page) and measure the 100 mm ruler before taping it down.")
    print("Then add to armlab/config.py so the code knows how big they are:")
    print("    TAG_MM.update({i: %.1f for i in range(%d, %d)})   # the left-side mat"
          % (args.tag_mm, all_used[0], all_used[-1] + 1))


if __name__ == "__main__":
    main()
