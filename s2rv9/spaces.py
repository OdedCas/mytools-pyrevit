"""S2Rv9 step 3: enclosed spaces (rooms, balcony) and the outside.

Walls from walls.py are painted into a mask, door and window gaps are closed,
and the free space is split into:
  - outside: reachable from the edge of the image,
  - spaces:  everything else, one label per enclosed area.

DRAWING_RULES.md rule 7 then decides which spaces are the apartment: the
balcony is whatever lies outside a window wall. That needs windows, which
come next; until then the balcony can be marked by hand.

numpy + pillow only.
"""
import numpy as np
from PIL import Image, ImageDraw


def _slide_max(a, k, axis):
    """Max over a centred window of k along axis (k odd), edge-padded."""
    r = k // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="constant", constant_values=0)
    out = np.zeros_like(a)
    n = a.shape[axis]
    for off in range(k):
        sl = [slice(None), slice(None)]
        sl[axis] = slice(off, off + n)
        np.maximum(out, p[tuple(sl)], out=out)
    return out


def dilate(mask, k):
    m = mask.astype(np.uint8)
    return _slide_max(_slide_max(m, k, 0), k, 1).astype(bool)


def erode(mask, k):
    return ~dilate(~mask, k)


def wall_mask(shape, walls):
    m = np.zeros(shape, bool)
    for w in walls:
        if w["axis"] == "h":
            m[int(w["p0"]):int(np.ceil(w["p1"])) + 1, int(w["s0"]):int(w["s1"]) + 1] = True
        else:
            m[int(w["s0"]):int(w["s1"]) + 1, int(w["p0"]):int(np.ceil(w["p1"])) + 1] = True
    return m


def bridge_openings(walls, px_per_cm, max_open_cm=160.0, min_perp_overlap=0.5):
    """Close doorways and windows by spanning gaps along a wall's own line.

    An opening sits inside a wall (DRAWING_RULES.md rule 2): the wall stops
    and continues on the same line. Two bands on the same axis that overlap
    across their thickness and end at most max_open_cm apart get a bridge.
    Only the gap is filled, so narrow rooms (a 105 cm corridor) stay free.
    """
    bridges = []
    for i, a in enumerate(walls):
        for b in walls[i + 1:]:
            if a["axis"] != b["axis"]:
                continue
            p0, p1 = max(a["p0"], b["p0"]), min(a["p1"], b["p1"])
            thin = min(a["p1"] - a["p0"], b["p1"] - b["p0"])
            if p1 - p0 < min_perp_overlap * thin:
                continue
            first, second = (a, b) if a["s1"] <= b["s0"] else (b, a)
            gap = second["s0"] - first["s1"]
            if gap <= 0 or gap / px_per_cm > max_open_cm:
                continue
            bridges.append({"axis": a["axis"], "p0": p0, "p1": p1,
                            "s0": first["s1"], "s1": second["s0"], "bridge": True})
    return bridges


def label_spaces(walls, shape, px_per_cm, bridge_cm=160.0, min_area_m2=0.8,
                 seal_cm=12.0):
    """Returns (label image, spaces). Label 0 = wall, 1 = outside, 2.. = spaces.

    Openings are closed with bridge_openings (gaps along a wall's line up to
    bridge_cm). A small closing of seal_cm then seals hairline corner gaps
    where the scan left two walls not quite touching - small enough that no
    room is filled.
    """
    k = int(round(seal_cm * px_per_cm)) | 1
    painted = wall_mask(shape, list(walls) + bridge_openings(walls, px_per_cm, bridge_cm))
    barrier = erode(dilate(painted, k), k)
    # .copy(): an image built straight from a numpy array does not keep
    # floodfill's changes on Pillow 12
    free = Image.fromarray(np.where(barrier, 0, 255).astype(np.uint8), "L").copy()
    h, w = shape
    # outside: flood from every border pixel that is free
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if free.getpixel((x, y)) == 255:
            ImageDraw.floodfill(free, (x, y), 1)
    arr = np.array(free)
    spaces = []
    label = 2
    cm2_per_px = (1.0 / px_per_cm) ** 2
    ys, xs = np.nonzero(arr == 255)
    seen = np.zeros(shape, bool)
    for y, x in zip(ys, xs):
        if seen[y, x] or free.getpixel((int(x), int(y))) != 255:
            continue
        ImageDraw.floodfill(free, (int(x), int(y)), label)
        cur = np.array(free) == label
        seen |= cur
        area_m2 = cur.sum() * cm2_per_px / 10000.0
        if area_m2 >= min_area_m2:
            yy, xx = np.nonzero(cur)
            spaces.append({"label": label, "area_m2": round(area_m2, 1),
                           "centre": (float(xx.mean()), float(yy.mean())),
                           "bbox": (int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max()))})
        label += 1
        if label > 250:
            break
    return np.array(free), spaces


def apartment_outline(walls, shape, px_per_cm, balcony_labels=(), envelope_bridge_cm=160.0):
    """Outline of the apartment + which walls are exterior.

    The envelope comes from a big closing (every opening up to
    envelope_bridge_cm sealed). That closing also fills rooms narrower than
    the bridge - the corridor, the bathrooms - but those are still INSIDE the
    envelope, which is all the outline needs; Revit finds the rooms itself
    once the walls exist.

    balcony_labels: spaces the user marked as balcony. DRAWING_RULES.md rule 7
    (balcony = outside the window wall) needs windows to apply automatically;
    a geometric guess was tried and scored the balcony, a corner bedroom and
    the living room almost equally (0.63 / 0.64 / 0.58), so it is not used.
    """
    labels, spaces = label_spaces(walls, shape, px_per_cm, bridge_cm=envelope_bridge_cm,
                                  seal_cm=envelope_bridge_cm)
    outside = labels == 1
    balcony_mask = np.isin(labels, list(balcony_labels)) if balcony_labels else np.zeros(shape, bool)
    footprint = ~outside & ~balcony_mask
    kept = clip_to_apartment(walls, footprint & ~wall_mask(shape, walls), shape)
    ext_side = dilate(outside | balcony_mask, 5)
    for w in kept:
        w["exterior"] = bool((wall_mask(shape, [w]) & ext_side).any())
    return footprint, balcony_mask, labels, spaces, kept


def clip_to_apartment(walls, inside, shape, reach_px=6, min_piece_px=15, junction_px=30):
    """Keep only the stretches of each wall that have the apartment beside them.

    A wall bounds the apartment on at least one side. Where a band has only
    balcony or outside on both sides it is a railing (DRAWING_RULES.md rule
    7) - e.g. the bottom facade band running on under the balcony - so that
    stretch is cut away, and a band with no apartment beside it at all is
    dropped. Gaps up to junction_px (~38 cm) are where another wall meets
    this one and do not split it.
    """
    h, w = shape
    out = []
    for wl in walls:
        lo, hi = int(wl["p0"]) - reach_px, int(np.ceil(wl["p1"])) + reach_px
        lo, hi = max(0, lo), min((h if wl["axis"] == "h" else w) - 1, hi)
        s0, s1 = int(wl["s0"]), int(wl["s1"])
        if wl["axis"] == "h":
            strip = inside[lo:hi + 1, s0:s1 + 1].any(axis=0)
        else:
            strip = inside[s0:s1 + 1, lo:hi + 1].any(axis=1)
        idx = np.flatnonzero(strip)
        if idx.size == 0:
            continue
        # where another wall meets this one there is wall, not apartment, on
        # that side for the width of the meeting wall - not a real break
        breaks = np.flatnonzero(np.diff(idx) > junction_px)
        starts = np.concatenate(([idx[0]], idx[breaks + 1]))
        ends = np.concatenate((idx[breaks], [idx[-1]]))
        for a, b in zip(starts, ends):
            if b - a + 1 >= min_piece_px:
                piece = dict(wl, s0=s0 + int(a), s1=s0 + int(b))
                piece["len_cm"] = wl["len_cm"] * (b - a + 1) / max(1.0, s1 - s0 + 1)
                out.append(piece)
    return out


def draw_outline(gray, footprint, balcony_mask, walls, out_png):
    img = np.asarray(Image.fromarray(gray).convert("RGB")).astype(np.float32) * 0.55 + 100
    img[footprint] = img[footprint] * 0.55 + np.array([150, 220, 150]) * 0.45
    img[balcony_mask] = img[balcony_mask] * 0.55 + np.array([240, 200, 110]) * 0.45
    out = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    for w in walls:
        col = (210, 30, 30) if w.get("exterior") else (40, 90, 210)
        box = ([w["s0"], w["p0"], w["s1"], w["p1"]] if w["axis"] == "h"
               else [w["p0"], w["s0"], w["p1"], w["s1"]])
        d.rectangle(box, fill=col)
    out.save(out_png)


def draw_spaces(gray, labels, spaces, out_png):
    rng = np.random.RandomState(7)
    img = np.asarray(Image.fromarray(gray).convert("RGB")).astype(np.float32) * 0.5 + 110
    for s in spaces:
        col = rng.randint(80, 230, 3)
        img[labels == s["label"]] = img[labels == s["label"]] * 0.3 + col * 0.7
    img[labels == 1] = [235, 235, 235]
    out = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    for s in spaces:
        cx, cy = s["centre"]
        d.text((cx - 12, cy - 6), "{}\n{:.1f}m2".format(s["label"], s["area_m2"]), fill=(0, 0, 0))
    out.save(out_png)
