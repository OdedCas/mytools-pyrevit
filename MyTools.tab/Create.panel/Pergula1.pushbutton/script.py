# -*- coding: utf-8 -*-
"""Pergula 1

Mono-pitch (single-slope) wood pergola built in the active Revit model as
DirectShape solids:

  - two rows of posts, spacing set by POST_SPACING_M
  - a continuous purlin along the top of each post row
  - straight rafters spanning the width, spacing set by RAFTER_SPACING_M
  - X dividers on the roof, FLUSH with the rafters (same plane, framed
    between them) - one X per field
  - a single continuous sloped fabric sheet laid on top of that plane
  - no bracing on the side walls (by design)

Edit the PARAMETERS block below, then run from pyRevit with a project open.
IronPython 2.7 compatible.
"""

__title__ = "Pergula 1"
__doc__ = "Build a mono-pitch wood pergola (posts, purlins, rafters, flush roof dividers, fabric) as DirectShapes."

from System.Collections.Generic import List

from Autodesk.Revit.DB import (
    Transaction, XYZ, Line, CurveLoop, GeometryCreationUtilities,
    DirectShape, ElementId, BuiltInCategory, GeometryObject,
)
from Autodesk.Revit.UI import TaskDialog

import clr
clr.AddReference("System.Windows.Forms")
clr.AddReference("System.Drawing")
from System.Windows.Forms import (
    Form, Label, TextBox, ComboBox, Button, DialogResult,
    FormBorderStyle, FormStartPosition, ComboBoxStyle,
)
from System.Drawing import Point, Size

doc = __revit__.ActiveUIDocument.Document  # noqa: F821  (provided by pyRevit)

# ---------------------------------------------------------------------------
# PARAMETERS (meters) -- edit these to match the real balcony
# ---------------------------------------------------------------------------
WIDTH_M = 2.0            # net width of the corridor / balcony (Y direction)
LENGTH_M = 10.75        # total run length (X direction)

# --- height of each side, controlled independently -------------------------
# HEIGHT_REF picks what the two numbers below mean:
#   "clear"    = clear height under the purlin (walking headroom)   <- default
#   "roof_top" = top of the fabric
HEIGHT_REF = "clear"
HEIGHT_LOW_M = 2.80      # low side
HEIGHT_HIGH_M = 3.10     # high side

# --- spacings --------------------------------------------------------------
POST_SPACING_M = 4.0     # columns every ~4 m
RAFTER_SPACING_M = 2.0   # rafters every ~2 m (snapped to divide the post bay)

# --- sections (width across, depth) ----------------------------------------
POST_SECTION_M = 0.14                 # 14/14 posts
PURLIN_SECTION_M = (0.10, 0.15)       # 10/15 top plate, spans post to post;
                                      # depth matched to the rafters so the
                                      # soffit is flush as well as the top
RAFTER_SECTION_M = (0.07, 0.15)       # 7/15 beams spanning the width
DIVIDER_SECTION_M = (0.04, 0.10)      # 4/10 X dividers - light members, they
                                      # only brace the roof and carry the cloth.
                                      # Top face still lands in the roof plane;
                                      # being shallower they just sit above the
                                      # rafter soffit.
FABRIC_THICKNESS_M = 0.01             # thin sheet representing the canvas

ORIGIN_M = (0.0, 0.0, 0.0)  # placement of the low-side / start corner

M2FT = 3.280839895013123  # meters -> feet (Revit internal units)


def m(v):
    return v * M2FT


# ---------------------------------------------------------------------------
# geometry helpers  (all return Solid objects; nothing touches the model)
# ---------------------------------------------------------------------------
def _loop(points):
    loop = CurveLoop()
    n = len(points)
    for i in range(n):
        loop.Append(Line.CreateBound(points[i], points[(i + 1) % n]))
    return loop


def _extrude(loop, direction, distance):
    loops = List[CurveLoop]()
    loops.Add(loop)
    return GeometryCreationUtilities.CreateExtrusionGeometry(
        loops, direction, distance)


def beam_solid(p1, p2, w, h, offset_up=0.0, ref_up=None):
    """Straight prismatic beam between two points.

    Cross-section is w along the local 'right' axis and h along the local
    'up' axis.  ref_up sets which way 'up' leans - pass the roof normal for
    roof members so they stay square to the roof plane.  offset_up shifts
    the member along its local up axis (use -h/2 to hang it under a plane).
    """
    v = p2 - p1
    length = v.GetLength()
    if length < 1e-9:
        return None
    direction = v.Normalize()

    if ref_up is None:
        ref_up = XYZ(0, 0, 1)
    if abs(direction.DotProduct(ref_up)) > 0.999:
        ref_up = XYZ(1, 0, 0)

    right = direction.CrossProduct(ref_up).Normalize()
    up = right.CrossProduct(direction).Normalize()

    base = p1 + up * offset_up
    hw, hh = w / 2.0, h / 2.0
    corners = [
        base + right * hw + up * hh,
        base - right * hw + up * hh,
        base - right * hw - up * hh,
        base + right * hw - up * hh,
    ]
    return _extrude(_loop(corners), direction, length)


def panel_solid(corners, thickness):
    """Flat quad panel extruded `thickness` along its own normal."""
    e1 = corners[1] - corners[0]
    e2 = corners[3] - corners[0]
    normal = e1.CrossProduct(e2).Normalize()
    if normal.Z < 0:
        normal = normal.Negate()
    return _extrude(_loop(corners), normal, thickness)


def place(solids, bic, name):
    """Create one DirectShape holding all the given solids."""
    geo = List[GeometryObject]()
    for s in solids:
        if s is not None:
            geo.Add(s)
    if geo.Count == 0:
        return None
    ds = DirectShape.CreateElement(doc, ElementId(bic))
    ds.ApplicationId = "PergulaBuilder"
    ds.ApplicationDataId = name
    ds.SetShape(geo)
    ds.Name = name
    return ds



# ---------------------------------------------------------------------------
# input dialog
# ---------------------------------------------------------------------------
def prompt_inputs():
    """Ask for the dimensions. Returns a dict, or None if cancelled."""
    fields = [
        ("LENGTH_M",         "Length (m)",          LENGTH_M),
        ("WIDTH_M",          "Width (m)",           WIDTH_M),
        ("HEIGHT_LOW_M",     "Low side height (m)", HEIGHT_LOW_M),
        ("HEIGHT_HIGH_M",    "High side height (m)", HEIGHT_HIGH_M),
        ("POST_SPACING_M",   "Post spacing (m)",    POST_SPACING_M),
        ("RAFTER_SPACING_M", "Rafter spacing (m)",  RAFTER_SPACING_M),
    ]

    form = Form()
    form.Text = "Pergula 1"
    form.FormBorderStyle = FormBorderStyle.FixedDialog
    form.StartPosition = FormStartPosition.CenterScreen
    form.MinimizeBox = False
    form.MaximizeBox = False
    form.ClientSize = Size(300, 40 * (len(fields) + 1) + 60)

    boxes = {}
    y = 15
    for key, caption, default in fields:
        lab = Label()
        lab.Text = caption
        lab.Location = Point(12, y + 3)
        lab.Size = Size(165, 20)
        form.Controls.Add(lab)

        box = TextBox()
        box.Text = str(default)
        box.Location = Point(185, y)
        box.Size = Size(100, 20)
        form.Controls.Add(box)
        boxes[key] = box
        y += 32

    lab = Label()
    lab.Text = "Heights are..."
    lab.Location = Point(12, y + 3)
    lab.Size = Size(165, 20)
    form.Controls.Add(lab)

    combo = ComboBox()
    combo.DropDownStyle = ComboBoxStyle.DropDownList
    combo.Items.Add("clear headroom")
    combo.Items.Add("top of fabric")
    combo.SelectedIndex = 0 if HEIGHT_REF == "clear" else 1
    combo.Location = Point(185, y)
    combo.Size = Size(100, 20)
    form.Controls.Add(combo)
    y += 45

    ok = Button()
    ok.Text = "Build"
    ok.DialogResult = DialogResult.OK
    ok.Location = Point(115, y)
    ok.Size = Size(80, 26)
    form.Controls.Add(ok)

    cancel = Button()
    cancel.Text = "Cancel"
    cancel.DialogResult = DialogResult.Cancel
    cancel.Location = Point(205, y)
    cancel.Size = Size(80, 26)
    form.Controls.Add(cancel)

    form.AcceptButton = ok
    form.CancelButton = cancel

    while True:
        if form.ShowDialog() != DialogResult.OK:
            return None
        values = {}
        bad = []
        for key, caption, _default in fields:
            try:
                v = float(boxes[key].Text.replace(",", "."))
            except ValueError:
                bad.append(caption)
                continue
            if v <= 0:
                bad.append(caption + " (must be > 0)")
                continue
            values[key] = v
        if bad:
            TaskDialog.Show("Pergula 1",
                            "Please check these values:\n\n- " +
                            "\n- ".join(bad))
            continue
        values["HEIGHT_REF"] = ("clear" if combo.SelectedIndex == 0
                                else "roof_top")
        return values


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build_pergola():
    ox, oy, oz = m(ORIGIN_M[0]), m(ORIGIN_M[1]), m(ORIGIN_M[2])

    width = m(WIDTH_M)
    length = m(LENGTH_M)

    post = m(POST_SECTION_M)
    pur_w, pur_h = m(PURLIN_SECTION_M[0]), m(PURLIN_SECTION_M[1])
    raf_w, raf_h = m(RAFTER_SECTION_M[0]), m(RAFTER_SECTION_M[1])
    div_w, div_h = m(DIVIDER_SECTION_M[0]), m(DIVIDER_SECTION_M[1])
    fabric_t = m(FABRIC_THICKNESS_M)

    # --- what the two height numbers mean --------------------------------
    # roof_top = the single roof plane: top face of the purlins, rafters AND
    # dividers alike, and the underside of the fabric.  In "clear" mode the
    # given height is the walking headroom under the purlin, which is the
    # deepest member, so only its depth is added on top.
    stack = pur_h
    if HEIGHT_REF == "clear":
        z_low = m(HEIGHT_LOW_M) + stack
        z_high = m(HEIGHT_HIGH_M) + stack
    else:
        z_low = m(HEIGHT_LOW_M)
        z_high = m(HEIGHT_HIGH_M)

    # --- bay division ----------------------------------------------------
    # Posts divide the run into whole bays; rafters then subdivide each post
    # bay a whole number of times, so every Nth rafter lands on a column.
    n_post_bays = max(1, int(round(length / m(POST_SPACING_M))))
    post_bay = length / float(n_post_bays)
    per_bay = max(1, int(round(post_bay / m(RAFTER_SPACING_M))))
    n_raf_bays = n_post_bays * per_bay

    xs_post = [ox + length * i / float(n_post_bays)
               for i in range(n_post_bays + 1)]
    xs_raf = [ox + length * i / float(n_raf_bays)
              for i in range(n_raf_bays + 1)]

    # --- roof plane (top face of rafters AND dividers) --------------------
    slope = (z_high - z_low) / width          # dz per unit Y

    def roof_z(y_local):
        return oz + z_low + slope * y_local

    nlen = (1.0 + slope * slope) ** 0.5
    roof_n = XYZ(0.0, -slope / nlen, 1.0 / nlen)   # unit normal, pointing up

    def roof_pt(x, y_local):
        return XYZ(x, oy + y_local, roof_z(y_local))

    # post-row centre lines, kept inside the footprint
    y_lo = oy + post / 2.0
    y_hi = oy + width - post / 2.0

    posts, purlins, rafters, dividers, fabric = [], [], [], [], []

    # --- purlins: top face ON the roof plane, tilted to match it ----------
    # ref_up = roof_n leans the section with the slope, so the top face lies
    # exactly in the plane instead of only touching it along one line.
    y_lo_local, y_hi_local = y_lo - oy, y_hi - oy
    for y_local in (y_lo_local, y_hi_local):
        purlins.append(beam_solid(roof_pt(ox, y_local),
                                  roof_pt(ox + length, y_local),
                                  pur_w, pur_h,
                                  offset_up=-pur_h / 2.0, ref_up=roof_n))

    # --- posts: ground up to the underside of the purlin ------------------
    # the purlin is tilted, so its soffit sits pur_h along the roof normal
    # below the plane - roof_n.Z converts that back to a vertical drop.
    for x in xs_post:
        for y_local in (y_lo_local, y_hi_local):
            top = roof_z(y_local) - pur_h * roof_n.Z
            posts.append(beam_solid(XYZ(x, oy + y_local, oz),
                                    XYZ(x, oy + y_local, top),
                                    post, post))

    # --- rafters: framed BETWEEN the purlins, top face on the same plane ---
    y_in_lo = y_lo_local + pur_w / 2.0
    y_in_hi = y_hi_local - pur_w / 2.0
    for x in xs_raf:
        rafters.append(beam_solid(roof_pt(x, y_in_lo), roof_pt(x, y_in_hi),
                                  raf_w, raf_h,
                                  offset_up=-raf_h / 2.0, ref_up=roof_n))

    # --- X dividers: SAME plane as the rafters, framed between them -------
    # Same depth as a rafter and the same -h/2 offset, so purlins, rafters
    # and dividers all share one top plane and the fabric lands on a single
    # flat surface.  The diagonals stop at the rafter and purlin faces.
    inset = raf_w / 2.0
    for i in range(n_raf_bays):
        x0 = xs_raf[i] + inset
        x1 = xs_raf[i + 1] - inset
        if x1 <= x0:
            continue
        a, b = roof_pt(x0, y_in_lo), roof_pt(x1, y_in_hi)
        c, d = roof_pt(x1, y_in_lo), roof_pt(x0, y_in_hi)
        dividers.append(beam_solid(a, b, div_w, div_h,
                                   offset_up=-div_h / 2.0, ref_up=roof_n))
        dividers.append(beam_solid(c, d, div_w, div_h,
                                   offset_up=-div_h / 2.0, ref_up=roof_n))

    # --- fabric: one continuous sheet laid on the roof plane --------------
    fabric.append(panel_solid([roof_pt(ox, 0.0),
                               roof_pt(ox + length, 0.0),
                               roof_pt(ox + length, width),
                               roof_pt(ox, width)], fabric_t))

    created = []
    t = Transaction(doc, "Build Pergula 1")
    t.Start()
    try:
        gm = BuiltInCategory.OST_GenericModel
        created.append(place(posts, gm, "Pergola - Posts"))
        created.append(place(purlins, gm, "Pergola - Purlins"))
        created.append(place(rafters, gm, "Pergola - Rafters"))
        created.append(place(dividers, gm, "Pergola - Roof Dividers"))
        created.append(place(fabric, gm, "Pergola - Fabric"))
        t.Commit()
    except Exception:
        if t.HasStarted() and not t.HasEnded():
            t.RollBack()
        raise

    return (n_post_bays, post_bay / M2FT, n_raf_bays,
            length / M2FT / n_raf_bays, [e for e in created if e is not None])


if doc is None or doc.IsFamilyDocument:
    TaskDialog.Show("Pergula 1",
                    "Open a project document (not a family) and run again.")
else:
    _values = prompt_inputs()
    if _values is not None:
        globals().update(_values)

        # Post spacing drives two things: the base moment on the unbraced
        # cantilever posts, and the purlin span.  The 10/15 purlin is the
        # first to go - its deflection passes at 4.5 m but not at 5 m.
        if POST_SPACING_M > 4.5:
            TaskDialog.Show(
                "Pergula 1",
                "Post spacing of {0:.2f} m is past the checked limit of "
                "4.5 m.\n\nAt that span the 10/15 purlin exceeds its "
                "deflection limit, and the 14/14 post base moment grows "
                "too.\n\nThe pergola will still be built, but deepen the "
                "purlin and re-check the post and its anchors."
                .format(POST_SPACING_M))

        n_pb, post_bay_m, n_rb, raf_sp_m, elements = build_pergola()
        TaskDialog.Show(
            "Pergula 1",
            "Pergola created.\n\n"
            "Run: {0:.2f} x {1:.2f} m\n"
            "Posts: {2} per row x 2 rows, bay {3:.2f} m\n"
            "Rafters: {4}, spacing {5:.2f} m\n"
            "Roof fields: {6} (each with one X)\n"
            "Heights ({7}): low {8:.2f} m, high {9:.2f} m\n"
            "DirectShapes: {10}".format(
                LENGTH_M, WIDTH_M, n_pb + 1, post_bay_m, n_rb + 1, raf_sp_m,
                n_rb, HEIGHT_REF, HEIGHT_LOW_M, HEIGHT_HIGH_M, len(elements)))
