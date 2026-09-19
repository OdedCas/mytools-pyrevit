"""S2Rv9 step 4: windows in the exterior walls, and the balcony from them.

A window sits inside a wall (DRAWING_RULES.md rule 2). Along an exterior
wall the heavy pen and poche run the wall's full thickness; across a window
only thin frame / glazing lines remain. So the count of dark pixels across
the wall drops sharply over a window's width - on the sample from ~12 to
~2-6 - and the stretch where it drops is the window.

Width and sill come from the dimension strings (e.g. `100 UK60`), which the
user confirms; the widths found here are for comparison (rule: written
numbers win, the drawing says where).

The balcony (rule 7: outside the window wall): where a window wall has an
enclosed space on BOTH sides, one of them is not the apartment - the smaller
one is the balcony.

numpy only.
"""
import numpy as np


def _profile(gray, wall, dark=90):
    """Dark pixels across the wall's thickness, at each point along it."""
    p0, p1 = int(wall["p0"]), int(np.ceil(wall["p1"])) + 1
    s0, s1 = int(wall["s0"]), int(wall["s1"]) + 1
    if wall["axis"] == "h":
        return (gray[p0:p1, s0:s1] < dark).sum(axis=0).astype(float)
    return (gray[s0:s1, p0:p1] < dark).sum(axis=1).astype(float)


def _smooth(v, k=7):
    if v.size < k:
        return v
    return np.convolve(v, np.ones(k) / k, mode="same")


def find_windows(gray, walls, px_per_cm, min_cm=40.0, max_cm=260.0,
                 seed_ratio=0.62, extend_ratio=0.8, end_margin_cm=15.0,
                 smooth_px=15, merge_px=32):
    """Windows along each exterior wall, as dicts with position and width.

    Two levels, because windows differ in how much frame they show: a
    window must dip below seed_ratio x the wall's median somewhere, and it
    extends as far as the count stays below extend_ratio. A single cut-off
    either missed the sample's 3N window (it only drops to ~75%) or picked
    up noise. The stretch must be min_cm..max_cm long and not touch the
    wall's ends (a wall that simply stops is not a window).
    """
    found = []
    for wi, w in enumerate(walls):
        if not w.get("exterior"):
            continue
        prof = _smooth(_profile(gray, w), smooth_px)
        if prof.size < min_cm * px_per_cm:
            continue
        base = np.median(prof)
        if base < 4:
            continue  # too thin to tell a window from the wall
        low = prof < extend_ratio * base
        deep = prof < seed_ratio * base
        idx = np.flatnonzero(low)
        if idx.size == 0:
            continue
        # a mullion, a sill mark or the window's own tag (a hexagon drawn
        # inside the opening, rule 9) lifts the count mid-window; up to
        # merge_px (~40 cm) of that does not split the window
        breaks = np.flatnonzero(np.diff(idx) > merge_px)
        starts = np.concatenate(([idx[0]], idx[breaks + 1]))
        ends = np.concatenate((idx[breaks], [idx[-1]]))
        margin = end_margin_cm * px_per_cm
        for a, b in zip(starts, ends):
            if not deep[a:b + 1].any():
                continue
            width_cm = (b - a + 1) / px_per_cm
            if not (min_cm <= width_cm <= max_cm):
                continue
            if a < margin or b > prof.size - 1 - margin:
                continue
            found.append({"wall": wi, "axis": w["axis"],
                          "s0": w["s0"] + int(a), "s1": w["s0"] + int(b),
                          "p0": w["p0"], "p1": w["p1"],
                          "width_cm": round(width_cm, 1), "how": "dip"})
    found += gaps_between(walls, px_per_cm, min_cm, max_cm)
    return found


def gaps_between(walls, px_per_cm, min_cm=40.0, max_cm=260.0, min_perp_overlap=0.5,
                 thin_tol_px=4):
    """Openings where an exterior wall stops and carries on along the same line.

    Tracing often splits a wall at its window, so the window is the gap
    between two pieces rather than a dip inside one. Only the nearest piece
    on each side counts, so a gap is never measured across a third piece.
    One exterior side is enough: the piece past a window can be a cladding
    strip the outline counts as inside.
    """
    out = []
    for a in walls:
        best = None
        for b in walls:
            if b is a or b["axis"] != a["axis"] or b["s0"] <= a["s1"]:
                continue
            p0, p1 = max(a["p0"], b["p0"]), min(a["p1"], b["p1"])
            thin = min(a["p1"] - a["p0"], b["p1"] - b["p0"])
            # thin cladding-line pieces of one facade can sit a few px out of
            # line with each other
            if p1 - p0 < min_perp_overlap * thin and p1 - p0 < -thin_tol_px + 0.0:
                continue
            if p1 - p0 < min_perp_overlap * thin and thin > 2 * thin_tol_px:
                continue
            if best is None or b["s0"] < best["s0"]:
                best = b
        if best is None or not (a.get("exterior") or best.get("exterior")):
            continue
        gap_cm = (best["s0"] - a["s1"]) / px_per_cm
        if min_cm <= gap_cm <= max_cm:
            out.append({"wall": walls.index(a), "axis": a["axis"],
                        "s0": int(a["s1"]), "s1": int(best["s0"]),
                        "p0": min(a["p0"], best["p0"]), "p1": max(a["p1"], best["p1"]),
                        "width_cm": round(gap_cm, 1), "how": "gap"})
    return out


def glazing_share(gray, opening, pad_px=3, min_probe_cm=50.0, px_per_cm=0.78):
    """Share of an opening's length that has glazing lines running along it.

    A window shows thin frame/glass lines parallel to the wall inside the
    opening (DRAWING_RULES.md: openings sit inside walls). A doorway is empty
    there, or has a leaf and swing arc crossing it. At each point along the
    opening, count separate ink strokes across the wall's thickness; two or
    more is glazing.
    """
    import lines as L
    # probe at least min_probe_cm across: a traced piece can be only the inner
    # face of the wall, with the window frame lines further out
    thick = opening["p1"] - opening["p0"]
    pad_px = max(pad_px, int((min_probe_cm * px_per_cm - thick) / 2.0))
    p0 = max(0, int(opening["p0"]) - pad_px)
    p1 = int(np.ceil(opening["p1"])) + pad_px + 1
    s0, s1 = int(opening["s0"]), int(opening["s1"]) + 1
    if opening["axis"] == "h":
        ridge = L.ridge_mask(gray, 0)[p0:p1, s0:s1]          # lines along x
        cols = [ridge[:, i] for i in range(ridge.shape[1])]
    else:
        ridge = L.ridge_mask(gray, 1)[s0:s1, p0:p1]          # lines along y
        cols = [ridge[i, :] for i in range(ridge.shape[0])]
    glazed = 0
    for c in cols:
        strokes = int(np.count_nonzero(np.diff(c.astype(np.int8)) == 1) + (1 if c.size and c[0] else 0))
        if strokes >= 2:
            glazed += 1
    return glazed / float(max(1, len(cols)))


def classify_openings(gray, openings, px_per_cm, min_glazing=0.5):
    """Mark each opening window or door by its glazing lines."""
    for o in openings:
        o["glazing"] = round(glazing_share(gray, o, px_per_cm=px_per_cm), 2)
        o["kind"] = "window" if o["glazing"] >= min_glazing else "door"
    return openings


def balcony_from_windows(windows, walls, labels, spaces, px_per_cm, reach_cm=60.0):
    """Spaces on the far side of a window wall, per DRAWING_RULES.md rule 7.

    For each window, look a short way out from both faces of its wall. If
    both sides land in an enclosed space, the smaller space is outside the
    window wall: a balcony.
    """
    area = {s["label"]: s["area_m2"] for s in spaces}
    h, w_ = labels.shape
    reach = int(round(reach_cm * px_per_cm))
    balconies = set()
    for win in windows:
        mid = int((win["s0"] + win["s1"]) / 2)
        sides = []
        for p in (int(win["p0"]) - reach, int(np.ceil(win["p1"])) + reach):
            y, x = (p, mid) if win["axis"] == "h" else (mid, p)
            if 0 <= y < h and 0 <= x < w_:
                sides.append(int(labels[y, x]))
        spaces_both = [s for s in sides if s in area]
        if len(spaces_both) == 2 and spaces_both[0] != spaces_both[1]:
            balconies.add(min(spaces_both, key=lambda s: area[s]))
    return sorted(balconies)


# ------------------------------------------------ windows from the dimensions

def _side_of(o, labels, px_per_cm, reach_cm=60.0):
    """Which way an opening faces the outside: 'top', 'bottom', 'left',
    'right', or None if neither face looks onto the outside.

    Probes a short way beyond each face of the wall at the opening's middle.
    Facing the outside (label 1) is what makes it a facade window; the 5N
    glass wall looks onto the balcony, not the outside, so it is not part of
    any facade's dimension string.
    """
    h, w = labels.shape
    r = int(round(reach_cm * px_per_cm))
    mid = int((o["s0"] + o["s1"]) / 2)
    before, after = int(o["p0"]) - r, int(np.ceil(o["p1"])) + r

    def is_out(p):
        y, x = (p, mid) if o["axis"] == "h" else (mid, p)
        return (not (0 <= y < h and 0 <= x < w)) or labels[y, x] == 1

    if o["axis"] == "h":
        return "top" if is_out(before) else ("bottom" if is_out(after) else None)
    return "left" if is_out(before) else ("right" if is_out(after) else None)


def chain_windows(string, px_per_cm):
    """Window intervals of a dimension string, in px from the string's start."""
    out, cur = [], 0.0
    for i, seg in enumerate(string["segments"]):
        if seg.get("window"):
            out.append({"seg": i, "a": cur * px_per_cm, "b": (cur + seg["cm"]) * px_per_cm,
                        "cm": seg["cm"], "sill_cm": seg.get("sill_cm"), "tag": seg.get("tag")})
        cur += seg["cm"]
    return out


def place_from_dimensions(openings, dims, labels, px_per_cm, match_tol_cm=60.0):
    """Place windows by the written numbers; line each string up by the drawing.

    The dimension string decides where windows are and how wide (the
    written numbers win); the windows found in the drawing only fix where the
    string starts. For each side the start offset is chosen so the string's
    windows land nearest the found ones; a found opening that no string
    window claims is dropped (bathtub edges, shower glass), and a string
    window with nothing found near it is still placed, flagged.
    Returns one row per string window with both positions for comparison.
    """
    tol = match_tol_cm * px_per_cm
    found = [o for o in openings if o["kind"] == "window"]
    rows = []
    for side, string in dims["strings"].items():
        axis = "h" if side in ("top", "bottom") else "v"
        mine = [o for o in found if o["axis"] == axis and _side_of(o, labels, px_per_cm) == side]
        cw = chain_windows(string, px_per_cm)
        if not cw:
            continue
        # candidate offsets: every pairing of a string window with a found one
        best = None
        for c in cw:
            for o in mine:
                off = o["s0"] - c["a"]
                cost, used = 0.0, 0
                for c2 in cw:
                    d = min((abs((c2["a"] + off) - o2["s0"]) + abs((c2["b"] + off) - o2["s1"])
                             for o2 in mine), default=None)
                    if d is not None and d < 2 * tol:
                        cost += d
                        used += 1
                key = (-used, cost)
                if best is None or key < best[0]:
                    best = (key, off)
        off = best[1] if best else None
        for c in cw:
            row = {"side": side, "tag": c["tag"], "written_cm": c["cm"], "sill_cm": c["sill_cm"],
                   "placed": None, "found": None}
            if off is not None:
                row["placed"] = (c["a"] + off, c["b"] + off)
                near = [o for o in mine
                        if abs(o["s0"] - (c["a"] + off)) < tol and abs(o["s1"] - (c["b"] + off)) < tol]
                if near:
                    o = min(near, key=lambda o: abs(o["s0"] - (c["a"] + off)))
                    row["found"] = (o["s0"], o["s1"])
                    o["claimed"] = True
            rows.append(row)
    dropped = [o for o in found if not o.get("claimed")]
    return rows, dropped
