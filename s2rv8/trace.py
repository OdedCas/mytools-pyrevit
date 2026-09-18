"""S2Rv8 tracing prototype: find wall bands in a raster floor plan.

Walls in these drawings are thick dark bands (black outline + grey poche).
Dimension lines, text, furniture and door swings are thin. Removing
everything thinner than a wall leaves the walls.
"""
import sys

import cv2
import numpy as np


def load_gray(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise IOError("cannot read image: {}".format(path))
    return gray


def wall_mask(gray, dark_threshold=215, min_wall_px=7):
    """Dark pixels that survive an opening wider than any thin line."""
    dark = (gray < dark_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_wall_px, min_wall_px))
    return cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)


def axis_segments(mask, min_len_px=20):
    """Split the wall mask into horizontal and vertical bands.

    Opening with a long thin kernel keeps only pixels that belong to a run
    at least min_len_px long in that direction. Each connected band then
    becomes one segment: centerline along its long axis, thickness from its
    short axis.
    """
    segments = []
    for axis in ("h", "v"):
        size = (min_len_px, 1) if axis == "h" else (1, min_len_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        band = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=4)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            length, thick = (w, h) if axis == "h" else (h, w)
            if length < min_len_px or thick < 3:
                continue
            # a band thicker than it is long is a blob, not a wall
            if thick > length:
                continue
            if axis == "h":
                cy = y + h / 2.0
                segments.append(("h", x, cy, x + w, cy, h))
            else:
                cx = x + w / 2.0
                segments.append(("v", cx, y, cx, y + h, w))
    return segments


def _band(seg):
    """(axis, lo_along, hi_along, lo_perp, hi_perp) for a segment."""
    axis, x1, y1, x2, y2, t = seg
    if axis == "h":
        return axis, min(x1, x2), max(x1, x2), y1 - t / 2.0, y1 + t / 2.0
    return axis, min(y1, y2), max(y1, y2), x1 - t / 2.0, x1 + t / 2.0


def _seg(axis, a0, a1, p0, p1):
    c = (p0 + p1) / 2.0
    t = p1 - p0
    if axis == "h":
        return ("h", a0, c, a1, c, t)
    return ("v", c, a0, c, a1, t)


def merge_sandwich(segments, max_cavity_px=24, min_overlap=0.6):
    """Merge parallel bands separated by a thin cavity into one wall.

    Exterior walls here are two poche layers with a white gap between.
    Two same-axis bands merge when the gap between their faces is at most
    max_cavity_px and they overlap along their length by at least
    min_overlap of the shorter one.
    """
    bands = [_band(s) for s in segments]
    merged = True
    while merged:
        merged = False
        for i in range(len(bands)):
            for j in range(i + 1, len(bands)):
                ai, a0, a1, p0, p1 = bands[i]
                aj, b0, b1, q0, q1 = bands[j]
                if ai != aj:
                    continue
                gap = max(q0 - p1, p0 - q1)
                if gap < 0 or gap > max_cavity_px:
                    continue
                overlap = min(a1, b1) - max(a0, b0)
                shorter = min(a1 - a0, b1 - b0)
                if shorter <= 0 or overlap < min_overlap * shorter:
                    continue
                bands[i] = (ai, min(a0, b0), max(a1, b1), min(p0, q0), max(p1, q1))
                del bands[j]
                merged = True
                break
            if merged:
                break
    return [_seg(*b) for b in bands]


def main(path, out_prefix):
    gray = load_gray(path)
    mask = wall_mask(gray)
    cv2.imwrite(out_prefix + "_mask.png", mask)

    segments = axis_segments(mask)
    before = len(segments)
    segments = merge_sandwich(segments)
    print("sandwich merge: {} -> {} segments".format(before, len(segments)))

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = (overlay * 0.45 + 140).astype(np.uint8)  # fade the drawing
    colors = {"h": (40, 40, 220), "v": (200, 90, 20)}  # red / blue
    for axis, x1, y1, x2, y2, thick in segments:
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                 colors[axis], max(1, int(thick)))
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 1)
    cv2.imwrite(out_prefix + "_segments.png", overlay)

    thick = sorted(s[5] for s in segments)
    print("segments: {}  (h={}, v={})".format(
        len(segments),
        sum(1 for s in segments if s[0] == "h"),
        sum(1 for s in segments if s[0] == "v")))
    if thick:
        print("thickness px: min={} median={} max={}".format(
            thick[0], thick[len(thick) // 2], thick[-1]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
