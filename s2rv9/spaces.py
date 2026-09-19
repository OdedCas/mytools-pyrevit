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


def label_spaces(walls, shape, px_per_cm, bridge_cm=160.0, min_area_m2=0.8):
    """Returns (label image, spaces). Label 0 = wall, 1 = outside, 2.. = spaces.

    bridge_cm must exceed the widest opening (doors, windows, the 144 cm
    glass opening on the sample) so closing seals every room.
    """
    k = int(round(bridge_cm * px_per_cm)) | 1
    barrier = erode(dilate(wall_mask(shape, walls), k), k)
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
