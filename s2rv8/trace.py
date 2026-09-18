"""S2Rv8 tracing prototype: find wall bands in a raster floor plan.

Walls in these drawings are thick dark bands (black outline + grey poche).
Dimension lines, text, furniture and door swings are thin. Removing
everything thinner than a wall leaves the walls.

Runs on numpy + pillow only. OpenCV is deliberately NOT used: the Windows
Python that pyRevit will shell out to has numpy and pillow but no cv2, and
installing it there is not always possible. The morphology, connected
component labelling and overlay drawing below replace the cv2 calls that
this file used to make.
"""
import sys

import numpy as np
from PIL import Image


def load_gray(path):
    """Greyscale, with transparency flattened onto WHITE.

    The sample plan is RGBA. Left to itself a converter reads a fully
    transparent pixel as black, which the wall threshold below then counts as
    poche. Paper is white, so composite onto white first and the empty areas
    stay empty.
    """
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        img = img.convert("RGBA")
        paper = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(paper, img)
    gray = np.asarray(img.convert("L"))
    if gray.size == 0:
        raise IOError("cannot read image: {}".format(path))
    return gray


def _slide_1d(arr, k, axis, pad_value, reduce_max):
    """Sliding min/max of width k along axis, anchored like cv2 (k // 2)."""
    if k <= 1:
        return arr
    anchor = k // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (anchor, k - 1 - anchor)
    padded = np.pad(arr, pad, mode="constant", constant_values=pad_value)
    windows = np.lib.stride_tricks.sliding_window_view(padded, k, axis=axis)
    return windows.max(axis=-1) if reduce_max else windows.min(axis=-1)


def _erode(mask, kw, kh):
    # cv2 erodes with the border held at the max value, so edges do not eat in.
    out = _slide_1d(mask, kw, 1, 255, False)
    return _slide_1d(out, kh, 0, 255, False)


def _dilate(mask, kw, kh):
    out = _slide_1d(mask, kw, 1, 0, True)
    return _slide_1d(out, kh, 0, 0, True)


def morph_open(mask, kw, kh):
    """Erode then dilate with a kw x kh rectangle (a separable opening)."""
    return _dilate(_erode(mask, kw, kh), kw, kh)


def wall_mask(gray, dark_threshold=215, min_wall_px=7):
    """Dark pixels that survive an opening wider than any thin line."""
    dark = (gray < dark_threshold).astype(np.uint8) * 255
    return morph_open(dark, min_wall_px, min_wall_px)


def connected_components(mask):
    """4-connected components of a binary mask -> list of (x, y, w, h, area).

    Row runs are unioned with the overlapping runs of the row above, which is
    the standard run-length labelling cv2 uses internally. Only foreground
    runs are visited, so this stays fast on the sparse band images here.
    """
    h, w = mask.shape
    solid = mask > 0
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    prev_runs = []
    runs = []  # (label, row, start, end_exclusive)
    for y in range(h):
        row = solid[y]
        if not row.any():
            prev_runs = []
            continue
        # run boundaries on this row
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.uint8), [0]))))
        starts, ends = edges[0::2], edges[1::2]
        cur_runs = []
        for s, e in zip(starts, ends):
            label = 0
            for pl, _py, ps, pe in prev_runs:
                if ps < e and s < pe:  # column overlap => 4-connected
                    if label == 0:
                        label = pl
                    else:
                        union(label, pl)
            if label == 0:
                label = len(parent)
                parent.append(label)
            cur_runs.append((label, y, s, e))
        runs.extend(cur_runs)
        prev_runs = cur_runs

    stats = {}
    for label, y, s, e in runs:
        root = find(label)
        box = stats.get(root)
        if box is None:
            stats[root] = [s, y, e, y + 1, e - s]
        else:
            box[0] = min(box[0], s)
            box[1] = min(box[1], y)
            box[2] = max(box[2], e)
            box[3] = max(box[3], y + 1)
            box[4] += e - s
    return [(x0, y0, x1 - x0, y1 - y0, area)
            for x0, y0, x1, y1, area in stats.values()]


def axis_segments(mask, min_len_px=20):
    """Split the wall mask into horizontal and vertical bands.

    Opening with a long thin kernel keeps only pixels that belong to a run
    at least min_len_px long in that direction. Each connected band then
    becomes one segment: centerline along its long axis, thickness from its
    short axis.
    """
    segments = []
    for axis in ("h", "v"):
        kw, kh = (min_len_px, 1) if axis == "h" else (1, min_len_px)
        band = morph_open(mask, kw, kh)
        for x, y, w, h, area in connected_components(band):
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


def draw_segments(gray, segments):
    """Fade the drawing and paint each band as its measured rectangle.

    The band is drawn at its true extent, with a one pixel outline. The old
    cv2.line call drew round caps, which overshot each end by half the
    thickness and made every wall read as longer and fatter than it is.
    """
    overlay = np.stack([gray] * 3, axis=-1).astype(np.float32)
    overlay = (overlay * 0.45 + 140).clip(0, 255).astype(np.uint8)
    h, w = gray.shape
    colors = {"h": (220, 40, 40), "v": (20, 90, 200)}  # RGB: red / blue
    for axis, x1, y1, x2, y2, thick in segments:
        half = max(thick / 2.0, 0.5)
        if axis == "h":
            c0, c1 = int(round(x1)), int(round(x2))
            r0, r1 = int(round(y1 - half)), int(round(y1 + half))
        else:
            c0, c1 = int(round(x1 - half)), int(round(x1 + half))
            r0, r1 = int(round(y1)), int(round(y2))
        r0, r1 = max(r0, 0), min(max(r1, r0 + 1), h)
        c0, c1 = max(c0, 0), min(max(c1, c0 + 1), w)
        overlay[r0:r1, c0:c1] = colors[axis]
        overlay[r0:r1, c0:min(c0 + 1, w)] = (0, 0, 0)
        overlay[r0:r1, max(c1 - 1, c0):c1] = (0, 0, 0)
        overlay[r0:min(r0 + 1, h), c0:c1] = (0, 0, 0)
        overlay[max(r1 - 1, r0):r1, c0:c1] = (0, 0, 0)
    return overlay


def main(path, out_prefix):
    gray = load_gray(path)
    mask = wall_mask(gray)
    Image.fromarray(mask).save(out_prefix + "_mask.png")

    segments = axis_segments(mask)
    before = len(segments)
    segments = merge_sandwich(segments)
    print("sandwich merge: {} -> {} segments".format(before, len(segments)))

    Image.fromarray(draw_segments(gray, segments)).save(out_prefix + "_segments.png")

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
