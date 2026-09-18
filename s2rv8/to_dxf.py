"""S2Rv8 step 3: traced wall bands -> a DXF that C2Rv7_C can read.

C2Rv7_C is layer-first and pairs wall FACES: its rules are "raw wall-face
gaps are bridged before wall-face pairing" and "centerlines are generated
from paired faces only". So each traced band is written as its TWO faces,
not as a centerline -- a centerline has nothing to pair with and would
produce no walls at all.

Output is plain DXF R12 text. ezdxf is deliberately not used: the Windows
Python that pyRevit shells out to does not have it, and R12 with LINE
entities on named layers is all C2Rv7_C reads.

Coordinates are centimetres, Y up (image rows count downwards, so the Y
axis is flipped here).

    python to_dxf.py samples/sample_plan.png out/plan.dxf [--px-per-cm 0.7675]
"""
import sys

from trace import axis_segments, load_gray, merge_sandwich, wall_mask

# From s2rv8/README.md: the bottom dimension string totals 886 cm over about
# 680 px on the sample. This is step 2's job (find a dimension string and ask
# the user to confirm it); until that exists the value is an UNVERIFIED
# default and every exported length inherits its error.
DEFAULT_PX_PER_CM = 0.7675

LAYER_EXT = "A-WALL-EXT"
LAYER_INT = "A-WALL-INT"
LAYER_COLORS = {LAYER_EXT: 1, LAYER_INT: 3, "A-WALL-CORE": 5,
                "A-DOORS": 2, "A-WINDOWS": 4}


def wall_faces(segments):
    """Each band -> its two face lines, as (p1, p2) pairs in pixel space."""
    faces = []
    for axis, x1, y1, x2, y2, thick in segments:
        half = thick / 2.0
        if axis == "h":
            faces.append((((x1, y1 - half), (x2, y1 - half)),
                          ((x1, y1 + half), (x2, y1 + half)), thick))
        else:
            faces.append((((x1 - half, y1), (x1 - half, y2)),
                          ((x1 + half, y1), (x1 + half, y2)), thick))
    return faces


def classify(segments):
    """Exterior = bands lying on the outer envelope of the wall set.

    Derived from the drawing's own extents and each band's own thickness, so
    it carries no fixed distance and rescales with any plan. It is still a
    placeholder: a plan with a courtyard or a deep recess will mislabel the
    walls that face it, and a real classifier should flood-fill the outside
    the way CreateFromCADV2 does.
    """
    if not segments:
        return {}
    xs = [x for s in segments for x in (s[1], s[3])]
    ys = [y for s in segments for y in (s[2], s[4])]
    x_lo, x_hi, y_lo, y_hi = min(xs), max(xs), min(ys), max(ys)
    out = {}
    for i, (axis, x1, y1, x2, y2, thick) in enumerate(segments):
        if axis == "h":
            near = min(abs(y1 - y_lo), abs(y1 - y_hi))
        else:
            near = min(abs(x1 - x_lo), abs(x1 - x_hi))
        out[i] = LAYER_EXT if near <= thick else LAYER_INT
    return out


def _tag(code, value):
    return "{}\n{}\n".format(code, value)


def write_dxf(path, lines_by_layer, layers):
    """Minimal DXF R12: header units, a LAYER table, then LINE entities."""
    parts = []
    parts.append(_tag(0, "SECTION") + _tag(2, "HEADER"))
    parts.append(_tag(9, "$INSUNITS") + _tag(70, 5))  # 5 = centimetres
    parts.append(_tag(0, "ENDSEC"))

    parts.append(_tag(0, "SECTION") + _tag(2, "TABLES"))
    parts.append(_tag(0, "TABLE") + _tag(2, "LAYER") + _tag(70, len(layers)))
    for name in layers:
        parts.append(_tag(0, "LAYER") + _tag(2, name) + _tag(70, 0)
                     + _tag(62, LAYER_COLORS.get(name, 7)) + _tag(6, "CONTINUOUS"))
    parts.append(_tag(0, "ENDTAB") + _tag(0, "ENDSEC"))

    parts.append(_tag(0, "SECTION") + _tag(2, "ENTITIES"))
    for layer, segs in lines_by_layer.items():
        for (ax, ay), (bx, by) in segs:
            parts.append(
                _tag(0, "LINE") + _tag(8, layer)
                + _tag(10, "{:.4f}".format(ax)) + _tag(20, "{:.4f}".format(ay))
                + _tag(30, "0.0")
                + _tag(11, "{:.4f}".format(bx)) + _tag(21, "{:.4f}".format(by))
                + _tag(31, "0.0"))
    parts.append(_tag(0, "ENDSEC") + _tag(0, "EOF"))

    with open(path, "w") as fh:
        fh.write("".join(parts))


def convert(image_path, dxf_path, px_per_cm=DEFAULT_PX_PER_CM):
    gray = load_gray(image_path)
    segments = merge_sandwich(axis_segments(wall_mask(gray)))
    layer_of = classify(segments)
    height_px = gray.shape[0]

    def to_cm(pt):
        x, y = pt
        # image rows increase downwards; DXF Y increases upwards
        return (x / px_per_cm, (height_px - y) / px_per_cm)

    by_layer = {LAYER_EXT: [], LAYER_INT: []}
    for i, (f1, f2, thick) in enumerate(wall_faces(segments)):
        layer = layer_of[i]
        by_layer[layer].append((to_cm(f1[0]), to_cm(f1[1])))
        by_layer[layer].append((to_cm(f2[0]), to_cm(f2[1])))

    # C2Rv7_C reads five layers; create them all so the import is predictable
    # even though doors and windows are not detected yet.
    layers = [LAYER_EXT, LAYER_INT, "A-WALL-CORE", "A-DOORS", "A-WINDOWS"]
    write_dxf(dxf_path, by_layer, layers)

    thicks_cm = sorted(s[5] / px_per_cm for s in segments)
    print("walls: {}  ({} exterior, {} interior)".format(
        len(segments),
        sum(1 for v in layer_of.values() if v == LAYER_EXT),
        sum(1 for v in layer_of.values() if v == LAYER_INT)))
    print("lines written: {}".format(sum(len(v) for v in by_layer.values())))
    print("scale: {} px/cm  (UNVERIFIED - step 2 calibration not built)".format(px_per_cm))
    print("wall thickness cm: min={:.1f} median={:.1f} max={:.1f}".format(
        thicks_cm[0], thicks_cm[len(thicks_cm) // 2], thicks_cm[-1]))
    xs = [p[0] for v in by_layer.values() for ln in v for p in ln]
    ys = [p[1] for v in by_layer.values() for ln in v for p in ln]
    print("extent cm: {:.1f} x {:.1f}".format(max(xs) - min(xs), max(ys) - min(ys)))


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    px_per_cm = DEFAULT_PX_PER_CM
    if "--px-per-cm" in argv:
        px_per_cm = float(argv[argv.index("--px-per-cm") + 1])
    convert(argv[1], argv[2], px_per_cm)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
