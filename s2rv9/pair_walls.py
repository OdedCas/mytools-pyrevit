"""S2Rv9 step 2: pair the scan's lines into walls with CreateFromCADV2's code.

A wall is two parallel lines (DRAWING_RULES.md). CreateFromCADV2 already
pairs CAD lines that way (_find_wall_pairs, 4-50 cm apart by default), so the
scan's lines are converted to centimetres and handed to that same, tested
code instead of re-implementing it.

Usage:
    python pair_walls.py scan.jpeg out.png --px-per-cm 0.78
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw

import lines as L

HERE = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.join(HERE, "..", "MyTools.tab", "Create.panel", "CreateFromCADV2.pushbutton")
sys.path.insert(0, os.path.normpath(V2_DIR))

from v2_cad_extract import default_config          # noqa: E402
from v2_cad_recognition import _find_wall_pairs     # noqa: E402


def to_cad_lines(px_lines, px_per_cm, height_px):
    """Scan lines (px, y down) -> CAD-style line dicts (cm, y up)."""
    out = []
    for ln in px_lines:
        if ln["axis"] == "h":
            x1, y1, x2, y2 = ln["a0"], ln["pos"], ln["a1"], ln["pos"]
        else:
            x1, y1, x2, y2 = ln["pos"], ln["a0"], ln["pos"], ln["a1"]
        out.append({
            "type": "line", "layer": "0",
            "x1": x1 / px_per_cm, "y1": (height_px - y1) / px_per_cm,
            "x2": x2 / px_per_cm, "y2": (height_px - y2) / px_per_cm,
        })
    return out


def gap_brightness(gray, a, b):
    """Median grey level strictly between two parallel lines, over their overlap."""
    import numpy as np
    lo_p, hi_p = sorted((a["pos"], b["pos"]))
    p0, p1 = int(lo_p) + 3, int(hi_p) - 2   # skip the strokes themselves
    s0, s1 = int(max(a["a0"], b["a0"])), int(min(a["a1"], b["a1"]))
    if p1 <= p0 or s1 <= s0:
        return 0  # strokes touch: nothing between them to test, treat as filled
    region = gray[p0:p1, s0:s1] if a["axis"] == "h" else gray[s0:s1, p0:p1]
    return int(np.median(region))


def keep_filled(gray, px_lines, pairs, dists, paper=240):
    """Drop pairs with bare paper between them - dimension lines, boundaries.

    DRAWING_RULES.md: a wall is drawn with grey fill or hatch between its two
    lines. Two dimension lines, or a dimension line beside a dashed boundary,
    have white paper between them.
    """
    kept, kept_d, dropped = [], [], []
    for (i, j), d in zip(pairs, dists):
        if gap_brightness(gray, px_lines[i], px_lines[j]) < paper:
            kept.append((i, j))
            kept_d.append(d)
        else:
            dropped.append((i, j))
    return kept, kept_d, dropped


def draw_pairs(gray, px_lines, pairs, out_png):
    img = Image.fromarray(gray).convert("RGB")
    img = Image.eval(img, lambda v: int(v * 0.35 + 165))
    d = ImageDraw.Draw(img)

    def seg(ln):
        if ln["axis"] == "h":
            return [(ln["a0"], ln["pos"]), (ln["a1"], ln["pos"])]
        return [(ln["pos"], ln["a0"]), (ln["pos"], ln["a1"])]

    paired = set()
    for i, j in pairs:
        paired.add(i)
        paired.add(j)
    for k, ln in enumerate(px_lines):  # unpaired lines faint grey
        if k not in paired:
            d.line(seg(ln), fill=(170, 170, 170), width=1)
    for i, j in pairs:  # each wall: its two faces + a fill between them
        a, b = px_lines[i], px_lines[j]
        if a["axis"] == "h":
            x0, x1 = max(a["a0"], b["a0"]), min(a["a1"], b["a1"])
            y0, y1 = sorted((a["pos"], b["pos"]))
            d.rectangle([x0, y0, x1, y1], fill=(250, 170, 170))
        else:
            y0, y1 = max(a["a0"], b["a0"]), min(a["a1"], b["a1"])
            x0, x1 = sorted((a["pos"], b["pos"]))
            d.rectangle([x0, y0, x1, y1], fill=(170, 190, 250))
        d.line(seg(a), fill=(180, 0, 0), width=2)
        d.line(seg(b), fill=(180, 0, 0), width=2)
    img.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("out_png")
    ap.add_argument("--px-per-cm", type=float, required=True)
    args = ap.parse_args()

    gray = L.load_gray(args.scan)
    all_lines = L.join_collinear(L.find_lines(gray))
    px_lines = L.wall_pen_lines(gray, all_lines)
    print("heavy-pen lines: {} of {}".format(len(px_lines), len(all_lines)))
    cad = to_cad_lines(px_lines, args.px_per_cm, gray.shape[0])

    pairs, dists, _used = _find_wall_pairs(cad, default_config())
    draw_pairs(gray, px_lines, pairs, args.out_png)
    print("walls (pairs): {}".format(len(pairs)))
    if dists:
        ds = sorted(dists)
        print("wall thickness cm: min {:.0f}  median {:.0f}  max {:.0f}".format(
            ds[0], ds[len(ds) // 2], ds[-1]))
        buckets = {}
        for d_ in ds:
            b = int(round(d_ / 5.0) * 5)
            buckets[b] = buckets.get(b, 0) + 1
        print("thickness histogram (5 cm bins):",
              "  ".join("{}cm:{}".format(k, v) for k, v in sorted(buckets.items())))


if __name__ == "__main__":
    main()
