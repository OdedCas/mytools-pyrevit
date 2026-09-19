"""S2Rv9 step 1: find thin straight horizontal and vertical lines in a scan.

In a scanned plan a wall is two thin black lines with light grey between
them (see DRAWING_RULES.md), so walls are found by first finding lines and
then pairing them. This module does the first half.

numpy + pillow only: the Windows Python on the Revit machine has no OpenCV.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw


def load_gray(path):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        white = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    return np.asarray(img.convert("L"), dtype=np.uint8)


def estimate_skew(gray, ink=160, max_deg=2.0, step=0.05):
    """Rotation (degrees) that makes the plan's lines most axis-aligned.

    A phone scan is rarely perfectly square. For each candidate angle the
    ink is rotated and projected onto rows and columns; straight lines give
    sharp spikes, so the angle with the highest projection variance wins.
    Done on a half-size image for speed.
    """
    small = Image.fromarray(((gray < ink) * 255).astype(np.uint8))
    small = small.resize((small.width // 2, small.height // 2))
    best, best_score = 0.0, -1.0
    for deg in np.arange(-max_deg, max_deg + step / 2, step):
        rot = np.asarray(small.rotate(deg, resample=Image.BILINEAR, fillcolor=0), dtype=np.float32)
        score = rot.sum(axis=1).var() + rot.sum(axis=0).var()
        if score > best_score:
            best, best_score = float(deg), score
    return round(best, 3)


def deskew(gray, deg):
    if abs(deg) < 1e-6:
        return gray
    img = Image.fromarray(gray).rotate(deg, resample=Image.BICUBIC, fillcolor=255)
    return np.asarray(img, dtype=np.uint8)


def _row_runs(dark_row, min_len, max_gap):
    """(start, end) of dark runs in one row, bridging gaps up to max_gap."""
    idx = np.flatnonzero(dark_row)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    keep = (ends - starts + 1) >= min_len
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


def _stack_runs(runs_by_row, max_row_gap=1, min_overlap=0.6):
    """Join runs in neighbouring rows into lines with a stroke thickness.

    A scanned stroke is 2-3 px thick, so the same line shows up as a run in
    several consecutive rows. Runs join when their rows are adjacent and they
    overlap along the line by min_overlap of the shorter one.
    """
    open_lines = []   # each: [row0, row1, a0, a1]
    done = []
    for row in sorted(runs_by_row):
        still_open = []
        used = set()
        for ln in open_lines:
            if row - ln[1] > max_row_gap + 1:
                done.append(ln)
                continue
            matched = False
            for k, (a, b) in enumerate(runs_by_row[row]):
                if k in used:
                    continue
                ov = min(b, ln[3]) - max(a, ln[2])
                if ov >= min_overlap * min(b - a, ln[3] - ln[2]):
                    ln[1] = row
                    ln[2] = min(ln[2], a)
                    ln[3] = max(ln[3], b)
                    used.add(k)
                    matched = True
                    break
            still_open.append(ln)
        for k, (a, b) in enumerate(runs_by_row[row]):
            if k not in used:
                still_open.append([row, row, a, b])
        open_lines = still_open
    return done + open_lines


def ridge_mask(gray, across_axis, reach=3, contrast=25, ink=200):
    """Pixels darker than their surroundings on both sides, across a line.

    A wall's edge line has paper on one side and grey fill on the other, and
    is darker than both. The fill between the two edge lines is not darker
    than its neighbours - they are the black edge lines - so it drops out.
    That holds whether the scan made the fill light or nearly black, which a
    plain "dark = ink" threshold does not.

    across_axis=0 compares pixels above and below (finds horizontal lines);
    across_axis=1 compares left and right (finds vertical lines).
    """
    g = gray.astype(np.int16)
    pad = np.pad(g, reach, mode="edge")
    h, w = g.shape
    if across_axis == 0:
        before = pad[0:h, reach:reach + w]
        after = pad[2 * reach:2 * reach + h, reach:reach + w]
    else:
        before = pad[reach:reach + h, 0:w]
        after = pad[reach:reach + h, 2 * reach:2 * reach + w]
    return (g < ink) & (g <= np.minimum(before, after) - contrast)


def find_lines(gray, min_len=20, max_gap=2, max_stroke=20, solid_ink=100):
    """Horizontal and vertical lines as dicts, each with a stroke width in px.

    Two kinds of ink count:
      - ridges: a pen line darker than both sides (see ridge_mask) - the
        thin double lines of interior walls;
      - solid dark stripes (< solid_ink): several lines packed so tight they
        print as one band, e.g. an exterior wall's structure line with its
        cladding lines against it. Inside such a stripe nothing is darker
        than its neighbours, so ridge_mask alone loses the whole facade.

    min_len:    shorter runs are text, ticks or noise.
    max_gap:    scan dropouts bridged along a line.
    max_stroke: wider than this is a filled area (a room hatch, a logo).
    """
    out = []
    solid = gray < solid_ink
    for axis, grid in (("h", ridge_mask(gray, 0) | solid), ("v", (ridge_mask(gray, 1) | solid).T)):
        runs = {}
        for r in range(grid.shape[0]):
            rr = _row_runs(grid[r], min_len, max_gap)
            if rr:
                runs[r] = rr
        for r0, r1, a0, a1 in _stack_runs(runs):
            stroke = r1 - r0 + 1
            if stroke > max_stroke:
                continue
            pos = (r0 + r1) / 2.0
            out.append({"axis": axis, "pos": pos, "a0": a0, "a1": a1,
                        "len": a1 - a0 + 1, "stroke": stroke})
    return out


def join_collinear(lines, pos_tol=2.0, max_gap=8, min_len=30):
    """Join pieces of the same line that the scan broke apart.

    Pieces on the same axis whose positions differ by at most pos_tol px and
    whose ends are at most max_gap px apart become one line. Whatever is
    still shorter than min_len afterwards is text, ticks or noise.
    """
    out = []
    for axis in ("h", "v"):
        group = sorted((l for l in lines if l["axis"] == axis), key=lambda l: l["pos"])
        # bucket by position, then sweep each bucket along the line
        buckets = []
        for ln in group:
            if buckets and ln["pos"] - buckets[-1][-1]["pos"] <= pos_tol:
                buckets[-1].append(ln)
            else:
                buckets.append([ln])
        for bucket in buckets:
            bucket.sort(key=lambda l: l["a0"])

            def close(cur, members, inked):
                cur["pos"] = sum(members) / len(members)
                cur["len"] = cur["a1"] - cur["a0"] + 1
                cur["solid"] = min(1.0, inked / float(cur["len"]))
                out.append(cur)

            cur = dict(bucket[0])
            members = [cur["pos"]]
            inked = cur["len"]
            for ln in bucket[1:]:
                if ln["a0"] - cur["a1"] <= max_gap:
                    inked += max(0, ln["a1"] - max(ln["a0"], cur["a1"] + 1) + 1)
                    cur["a1"] = max(cur["a1"], ln["a1"])
                    cur["stroke"] = max(cur["stroke"], ln["stroke"])
                    members.append(ln["pos"])
                else:
                    close(cur, members, inked)
                    cur = dict(ln)
                    members = [cur["pos"]]
                    inked = cur["len"]
            close(cur, members, inked)
    return [l for l in out if l["len"] >= min_len]


def pen_darkness(gray, ln, reach=3, step=3):
    """How black the pen is along a line: median of its 2 darkest pixels across.

    Walls are drawn with a heavy black pen (8-74 on the sample scan);
    dimension lines and furniture with a lighter one (107-164).
    """
    p = int(round(ln["pos"]))
    h, w = gray.shape
    vals = []
    for a in range(int(ln["a0"]), int(ln["a1"]) + 1, step):
        if ln["axis"] == "h":
            prof = gray[max(0, p - reach):p + reach + 1, min(a, w - 1)]
        else:
            prof = gray[min(a, h - 1), max(0, p - reach):p + reach + 1]
        if prof.size >= 2:
            vals.append(np.sort(prof)[:2].mean())
    return float(np.median(vals)) if vals else 255.0


def wall_pen_lines(gray, lines, max_darkness=90.0):
    """Keep only lines drawn with a wall's heavy pen."""
    kept = []
    for ln in lines:
        ln["pen"] = pen_darkness(gray, ln)
        if ln["pen"] <= max_darkness:
            kept.append(ln)
    return kept


def draw_lines(gray, lines, out_png):
    img = Image.fromarray(gray).convert("RGB")
    img = Image.eval(img, lambda v: int(v * 0.45 + 140))
    d = ImageDraw.Draw(img)
    for ln in lines:
        c = (220, 40, 40) if ln["axis"] == "h" else (20, 90, 200)
        if ln["axis"] == "h":
            d.line([(ln["a0"], ln["pos"]), (ln["a1"], ln["pos"])], fill=c, width=2)
        else:
            d.line([(ln["pos"], ln["a0"]), (ln["pos"], ln["a1"])], fill=c, width=2)
    img.save(out_png)


def main(path, out_png):
    gray = load_gray(path)
    skew = estimate_skew(gray)
    gray = deskew(gray, skew)
    print("scan rotation corrected: {:+.2f} degrees".format(skew))
    raw = find_lines(gray)
    lines = join_collinear(raw)
    print("raw pieces: {}  ->  joined lines: {}".format(len(raw), len(lines)))
    draw_lines(gray, lines, out_png)
    h = [l for l in lines if l["axis"] == "h"]
    v = [l for l in lines if l["axis"] == "v"]
    print("lines: {}  (horizontal {}, vertical {})".format(len(lines), len(h), len(v)))
    lens = sorted(l["len"] for l in lines)
    print("length px: min {} median {} max {}".format(lens[0], lens[len(lens) // 2], lens[-1]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
