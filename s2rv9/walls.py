"""S2Rv9 step 2: group the scan's lines into walls, following DRAWING_RULES.md.

Why not CreateFromCADV2's _find_wall_pairs: it scores a pair by how close its
spacing is to a default 20 cm wall and by how much of the SHORTER line is
overlapped. That suits CAD, but on a scan it prefers a short stray stroke at
25 cm over a wall's real 8.7 cm partner (the 815 wall on the sample). The
shared function is left untouched because C2Rv7_C depends on it.

Here:
  1. nearest parallel neighbour first - a wall's two faces are adjacent lines
     with nothing between them; each line is used once;
  2. pairs whose faces touch are merged into one wall, so cladding,
     structure and plaster (rules 4-5) become a single exterior wall.

All distances are in cm; lines come from lines.py in px.
"""


def _overlap(a, b):
    return min(a["a1"], b["a1"]) - max(a["a0"], b["a0"])


def pair_nearest(px_lines, px_per_cm, min_cm=4.0, max_cm=50.0,
                 min_overlap_cm=30.0, min_overlap_ratio=0.5):
    """Greedy pairing, closest spacing first. Returns list of (i, j, gap_cm)."""
    cands = []
    for i in range(len(px_lines)):
        a = px_lines[i]
        for j in range(i + 1, len(px_lines)):
            b = px_lines[j]
            if a["axis"] != b["axis"]:
                continue
            gap = abs(a["pos"] - b["pos"]) / px_per_cm
            if gap < min_cm or gap > max_cm:
                continue
            ov = _overlap(a, b)
            shorter = min(a["len"], b["len"])
            if ov / px_per_cm < min_overlap_cm or ov < min_overlap_ratio * shorter:
                continue
            # a wall's faces have nothing between them: reject the pair if a
            # third line lies between the two and overlaps both
            lo, hi = sorted((a["pos"], b["pos"]))
            blocked = False
            for k, c in enumerate(px_lines):
                if k in (i, j) or c["axis"] != a["axis"]:
                    continue
                if lo < c["pos"] < hi and _overlap(c, a) > 0 and _overlap(c, b) > 0:
                    blocked = True
                    break
            if not blocked:
                cands.append((gap, i, j))
    cands.sort()
    used, pairs = set(), []
    for gap, i, j in cands:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs.append((i, j, gap))
    return pairs


def _pair_band(px_lines, i, j):
    a, b = px_lines[i], px_lines[j]
    lo, hi = sorted((a["pos"], b["pos"]))
    return {"axis": a["axis"], "p0": lo, "p1": hi,
            "s0": max(a["a0"], b["a0"]), "s1": min(a["a1"], b["a1"]),
            "lines": [i, j]}


def merge_touching(bands, px_per_cm, max_gap_cm=6.0, min_overlap_ratio=0.5):
    """Join bands whose faces are at most max_gap_cm apart into one wall."""
    bands = [dict(b) for b in bands]
    changed = True
    while changed:
        changed = False
        for x in range(len(bands)):
            for y in range(x + 1, len(bands)):
                a, b = bands[x], bands[y]
                if a["axis"] != b["axis"]:
                    continue
                gap = max(b["p0"] - a["p1"], a["p0"] - b["p1"]) / px_per_cm
                if gap > max_gap_cm:
                    continue
                ov = min(a["s1"], b["s1"]) - max(a["s0"], b["s0"])
                if ov <= 0 or ov < min_overlap_ratio * min(a["s1"] - a["s0"], b["s1"] - b["s0"]):
                    continue
                bands[x] = {"axis": a["axis"],
                            "p0": min(a["p0"], b["p0"]), "p1": max(a["p1"], b["p1"]),
                            "s0": min(a["s0"], b["s0"]), "s1": max(a["s1"], b["s1"]),
                            "lines": a["lines"] + b["lines"]}
                del bands[y]
                changed = True
                break
            if changed:
                break
    for b in bands:
        b["thick_cm"] = (b["p1"] - b["p0"]) / px_per_cm
        b["len_cm"] = (b["s1"] - b["s0"]) / px_per_cm
    return bands


def stripe_bands(px_lines, px_per_cm, min_stripe_px=6):
    """A stroke this wide is a wall face on its own - packed structure and
    cladding lines - so it becomes a band without needing a partner."""
    bands, used = [], set()
    for k, ln in enumerate(px_lines):
        if ln["stroke"] >= min_stripe_px:
            half = ln["stroke"] / 2.0
            bands.append({"axis": ln["axis"], "p0": ln["pos"] - half, "p1": ln["pos"] + half,
                          "s0": ln["a0"], "s1": ln["a1"], "lines": [k]})
            used.add(k)
    return bands, used


def find_walls(px_lines, px_per_cm):
    stripes, used = stripe_bands(px_lines, px_per_cm)
    thin = [ln if k not in used else dict(ln, axis="x") for k, ln in enumerate(px_lines)]
    pairs = pair_nearest(thin, px_per_cm)
    bands = stripes + [_pair_band(px_lines, i, j) for i, j, _ in pairs]
    return merge_touching(bands, px_per_cm)
