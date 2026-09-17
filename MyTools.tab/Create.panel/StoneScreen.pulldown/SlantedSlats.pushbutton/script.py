# -*- coding: utf-8 -*-
"""Stone slats leaning at random angles, like books on a shelf.

Mullions cannot lean - they always run along their grid line.  Slanted
structural columns are the one native Revit element that tilts freely and
stays parametric, so each slat is placed as one.

Select the wall (or a line) that defines the run, then press this button.
The slats are placed in front of it, each leaning a random angle IN THE
PLANE of the wall.

IronPython 2.7 compatible.
"""

from Autodesk.Revit.DB import (
    FamilyInstance, ElementId, StorageType, WallType, MullionType, PanelType,
    Transaction, Line, XYZ, Wall, FilteredElementCollector, BuiltInParameter,
    BuiltInCategory, FamilySymbol, Element, CurveElement, Level, Material,
    DirectShape, GeometryCreationUtilities, CurveLoop, SolidOptions,
    GeometryObject, Curve
)
from Autodesk.Revit.DB.Structure import StructuralType
from Autodesk.Revit.UI import TaskDialog
from Autodesk.Revit.UI.Selection import ObjectType
import clr, random, math, os
clr.AddReference('System.Windows.Forms')
clr.AddReference('System.Drawing')
import System.Windows.Forms as WF
import System.Drawing as SD
from System.Collections.Generic import List

doc = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument

MM = 1.0 / 304.8

# ----------------------------------------------------------------- slat setup
SLAT_WIDTH      = 100.0   # mm, across the face of the wall
SLAT_DEPTH      = 300.0   # mm, projecting out from the wall
SLAT_SPACING_MIN = 1200.0  # mm, closest two slats may sit, centre to centre
SLAT_SPACING_MAX = 2400.0  # mm, furthest apart
                           # set both the SAME for perfectly even spacing
LEAN_MAX        = 12.0   # IGNORED - the max lean is now derived from the
                          # clear gap and height so slats cannot intersect
LEAN_MIN        = 3.0     # degrees - min lean, so none look accidentally plumb
AIR_GAP         = 200.0   # mm, wall face -> slat centreline
SLAT_HEIGHT     = 0.0     # mm; 0 = copy the host wall's height
STONE_MATERIAL  = "stone"
# Slats are built as DirectShape solids, not family instances: this Revit
# install has no family library on disk, and the one loaded column family has
# no dimension parameters.  DirectShape needs nothing loaded.
SLAT_CATEGORY   = "OST_StructuralColumns"   # or "OST_Walls", "OST_GenericModel"
SHOW_DIALOG     = True    # False to run silently with the values above
SETTINGS_FILE   = os.path.join(
    os.environ.get("APPDATA", "C:\\"), "slanted_slats_settings.txt")
RANDOM_SEED     = 1       # change to reshuffle; fixed = repeatable
FLIP_SIDE       = 1.0     # -1.0 if the slats land on the wrong side
REPLACE_EXISTING = True   # rerun replaces this run's slats instead of stacking
SLAT_TAG        = "STONE_SLAT_SCREEN"   # written to Comments to find them again

# --- the wall behind the slats ----------------------------------------------
CONVERT_HOST    = True      # retype the selected wall into the glazing
GLASS_TYPE      = "CW - Thin Mullion"
GLASS_MODULE    = 1200.0    # mm - vertical grid spacing of the glazing
MULL_SIDE       = 20.0      # mm - each side, so a 40 mm sightline
MULL_THICK      = 60.0      # mm - mullion depth

log = []
_NP = clr.GetClrType(Element).GetProperty("Name")



# ------------------------------------------------------------ settings dialog
FIELDS = [
    ("SLAT_WIDTH",       "Slat width (mm, across the wall face)"),
    ("SLAT_DEPTH",       "Slat depth (mm, projecting out)"),
    ("SLAT_SPACING_MIN", "Spacing min (mm, centre to centre)"),
    ("SLAT_SPACING_MAX", "Spacing max (mm) - equal to min = even spacing"),
    ("AIR_GAP",          "Gap from wall face to slat centre (mm)"),
    ("SLAT_HEIGHT",      "Slat height (mm, 0 = copy the wall)"),
    ("LEAN_MIN",         "Minimum lean (degrees)"),
    ("RANDOM_SEED",      "Random seed (change to reshuffle)"),
]


def load_saved():
    vals = {}
    try:
        if os.path.isfile(SETTINGS_FILE):
            f = open(SETTINGS_FILE, "r")
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    vals[k] = v
            f.close()
    except Exception:
        pass
    return vals


def save_settings(vals):
    try:
        f = open(SETTINGS_FILE, "w")
        for k in vals:
            f.write("{0}={1}\n".format(k, vals[k]))
        f.close()
    except Exception:
        pass


def ask_settings():
    """A dialog for the slat parameters.  Returns False if cancelled."""
    saved = load_saved()

    form = WF.Form()
    form.Text = "Slanted Slats - settings"
    form.Width = 430
    form.Height = 30 * len(FIELDS) + 210
    form.StartPosition = WF.FormStartPosition.CenterScreen
    form.FormBorderStyle = WF.FormBorderStyle.FixedDialog
    form.MaximizeBox = False
    form.MinimizeBox = False

    boxes = {}
    y = 15
    for key, label in FIELDS:
        lab = WF.Label()
        lab.Text = label
        lab.Left = 12
        lab.Top = y + 3
        lab.Width = 290
        form.Controls.Add(lab)

        tb = WF.TextBox()
        tb.Left = 310
        tb.Top = y
        tb.Width = 90
        tb.Text = str(saved.get(key, globals()[key]))
        form.Controls.Add(tb)
        boxes[key] = tb
        y += 30

    y += 5
    flip = WF.CheckBox()
    flip.Text = "Slats on the other side of the wall"
    flip.Left = 12
    flip.Top = y
    flip.Width = 380
    flip.Checked = str(saved.get("FLIP_SIDE", FLIP_SIDE)).startswith("-")
    form.Controls.Add(flip)
    y += 26

    conv = WF.CheckBox()
    conv.Text = "Convert the selected wall to the thin-mullion curtain wall"
    conv.Left = 12
    conv.Top = y
    conv.Width = 380
    conv.Checked = str(saved.get("CONVERT_HOST", CONVERT_HOST)) != "False"
    form.Controls.Add(conv)
    y += 30

    lab2 = WF.Label()
    lab2.Text = "Category"
    lab2.Left = 12
    lab2.Top = y + 3
    lab2.Width = 290
    form.Controls.Add(lab2)
    cat = WF.ComboBox()
    cat.Left = 240
    cat.Top = y
    cat.Width = 160
    cat.DropDownStyle = WF.ComboBoxStyle.DropDownList
    for c in ("OST_StructuralColumns", "OST_Walls", "OST_GenericModel"):
        cat.Items.Add(c)
    cat.SelectedItem = saved.get("SLAT_CATEGORY", SLAT_CATEGORY)
    if cat.SelectedIndex < 0:
        cat.SelectedIndex = 0
    form.Controls.Add(cat)
    y += 40

    ok = WF.Button()
    ok.Text = "OK"
    ok.Left = 220
    ok.Top = y
    ok.Width = 85
    ok.DialogResult = WF.DialogResult.OK
    form.Controls.Add(ok)
    form.AcceptButton = ok

    cancel = WF.Button()
    cancel.Text = "Cancel"
    cancel.Left = 315
    cancel.Top = y
    cancel.Width = 85
    cancel.DialogResult = WF.DialogResult.Cancel
    form.Controls.Add(cancel)
    form.CancelButton = cancel

    if form.ShowDialog() != WF.DialogResult.OK:
        return False

    out = {}
    for key, label in FIELDS:
        raw = boxes[key].Text.strip().replace(",", ".")
        try:
            val = float(raw)
        except Exception:
            WF.MessageBox.Show("'{0}' is not a number for:\n{1}".format(
                raw, label), "Slanted Slats")
            return False
        globals()[key] = val
        out[key] = val
    globals()["RANDOM_SEED"] = int(globals()["RANDOM_SEED"])
    out["RANDOM_SEED"] = globals()["RANDOM_SEED"]

    globals()["FLIP_SIDE"] = -1.0 if flip.Checked else 1.0
    out["FLIP_SIDE"] = globals()["FLIP_SIDE"]
    globals()["CONVERT_HOST"] = bool(conv.Checked)
    out["CONVERT_HOST"] = globals()["CONVERT_HOST"]
    globals()["SLAT_CATEGORY"] = str(cat.SelectedItem)
    out["SLAT_CATEGORY"] = globals()["SLAT_CATEGORY"]

    if globals()["SLAT_SPACING_MAX"] < globals()["SLAT_SPACING_MIN"]:
        globals()["SLAT_SPACING_MIN"], globals()["SLAT_SPACING_MAX"] = \
            globals()["SLAT_SPACING_MAX"], globals()["SLAT_SPACING_MIN"]

    # spacing is centre to centre, so it must exceed the slat width or the
    # slats would overlap before they even lean
    if globals()["SLAT_SPACING_MIN"] <= globals()["SLAT_WIDTH"]:
        WF.MessageBox.Show(
            "Spacing must be larger than the slat width.\n\n"
            "Slat width:  {0} mm\n"
            "Spacing min: {1} mm\n\n"
            "Spacing is measured centre to centre, so at {1} mm the slats "
            "would touch or overlap. Try a spacing of at least {2} mm."
            .format(globals()["SLAT_WIDTH"], globals()["SLAT_SPACING_MIN"],
                    int(globals()["SLAT_WIDTH"] * 3)),
            "Slanted Slats")
        return False
    save_settings(out)
    note("settings: width {0}, depth {1}, spacing {2}-{3}, gap {4}, seed {5}"
         .format(SLAT_WIDTH, SLAT_DEPTH, SLAT_SPACING_MIN, SLAT_SPACING_MAX,
                 AIR_GAP, RANDOM_SEED))
    return True


def note(m):
    log.append(m)


def ename(e):
    if e is None:
        return ""
    try:
        return e.Name
    except Exception:
        pass
    try:
        return _NP.GetValue(e, None) or ""
    except Exception:
        return ""


def set_by_name(elem, names, value):
    """Set the first parameter matching any of these display names."""
    for p in elem.Parameters:
        try:
            if p.Definition.Name in names and not p.IsReadOnly:
                p.Set(value)
                return p.Definition.Name
        except Exception:
            pass
    return None


def describe(e):
    if e is None:
        return "nothing"
    cat = "?"
    try:
        cat = e.Category.Name
    except Exception:
        pass
    tn = ""
    try:
        tn = ename(doc.GetElement(e.GetTypeId()))
    except Exception:
        pass
    return "{0} '{1}' (id {2}, {3})".format(cat, tn, e.Id, e.GetType().Name)


def curve_of(e):
    """Location curve of a wall or line, or None."""
    if e is None:
        return None
    try:
        c = e.Location.Curve
        if c is not None:
            return c
    except Exception:
        pass
    c = getattr(e, "GeometryCurve", None)
    return c


def resolve_run_element(e):
    """Accept a wall or line; if given a mullion/panel, use its host wall."""
    if curve_of(e) is not None:
        return e
    # a curtain panel or mullion knows the wall it belongs to
    for attr in ("Host", "HostId"):
        try:
            h = getattr(e, attr, None)
            if h is None:
                continue
            if not isinstance(h, Wall):
                h = doc.GetElement(h)
            if isinstance(h, Wall) and curve_of(h) is not None:
                note("picked a {0} - using its host wall instead".format(
                    e.GetType().Name))
                return h
        except Exception:
            pass
    return None


def get_run():
    # anything usable already selected?
    for eid in uidoc.Selection.GetElementIds():
        cand = resolve_run_element(doc.GetElement(eid))
        if cand is not None:
            note("using selected " + describe(cand))
            return (curve_of(cand).GetEndPoint(0),
                    curve_of(cand).GetEndPoint(1),
                    cand if isinstance(cand, Wall) else None)

    # otherwise ask, and allow a couple of goes
    last = None
    for attempt in range(3):
        ref = uidoc.Selection.PickObject(
            ObjectType.Element,
            "Select the WALL or a LINE for the slat run")
        e = doc.GetElement(ref.ElementId)
        cand = resolve_run_element(e)
        if cand is not None:
            note("using picked " + describe(cand))
            c = curve_of(cand)
            return (c.GetEndPoint(0), c.GetEndPoint(1),
                    cand if isinstance(cand, Wall) else None)
        last = e
        TaskDialog.Show(
            "Not usable",
            "You picked {0}.\n\nThat has no location line to follow.\n\n"
            "Pick the WALL itself, or a Model/Detail Line. If you clicked a "
            "curtain panel or mullion, press Tab before clicking to select "
            "the wall behind it.".format(describe(e)))
    raise Exception("No usable wall or line picked. Last was " +
                    describe(last))



def stone_material_id():
    for m in FilteredElementCollector(doc).OfClass(Material):
        if STONE_MATERIAL and STONE_MATERIAL.lower() in ename(m).lower():
            note("material: " + ename(m))
            return m.Id
    note("!! no material matching '{0}' - slats left with no material".format(
        STONE_MATERIAL))
    return ElementId.InvalidElementId


def slat_solid(p_base, d, n, shift, height, mat_id):
    """One leaning slat as a solid.

    The base rectangle lies flat; extruding it along a tilted vector makes
    the slat lean.  Depth of extrusion is the true slope length, so the top
    lands at exactly `height`.
    """
    hw = (SLAT_WIDTH * MM) / 2.0     # half width, along the run
    hd = (SLAT_DEPTH * MM) / 2.0     # half depth, across the run
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        pts.append(p_base + d.Multiply(sx * hw) + n.Multiply(sy * hd))
    curves = List[Curve]()
    for k in range(4):
        curves.Add(Line.CreateBound(pts[k], pts[(k + 1) % 4]))
    loop = CurveLoop.Create(curves)
    loops = List[CurveLoop]()
    loops.Add(loop)

    vec = d.Multiply(shift) + XYZ(0, 0, height)
    dist = vec.GetLength()
    direction = vec.Normalize()
    opts = SolidOptions(mat_id, ElementId.InvalidElementId)
    return GeometryCreationUtilities.CreateExtrusionGeometry(
        loops, direction, dist, opts)


def place_slat(solid, key, cat_id):
    ds = DirectShape.CreateElement(doc, cat_id)
    shapes = List[GeometryObject]()
    shapes.Add(solid)
    ds.SetShape(shapes)
    try:
        ds.Name = "Stone Slat"
    except Exception:
        pass
    p = ds.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
    if p and not p.IsReadOnly:
        p.Set(key)
    return ds


def make_positions(length_mm, seed):
    """Slat positions along the run, each gap drawn independently."""
    import random as _r
    _r.seed(seed + 77)
    lo, hi = SLAT_SPACING_MIN, SLAT_SPACING_MAX
    if hi < lo:
        lo, hi = hi, lo          # tolerate them being the wrong way round
    pos = [0.0]
    while True:
        nxt = pos[-1] + _r.uniform(lo, hi)
        if nxt > length_mm:
            break
        pos.append(nxt)
    return pos


def make_angles_for(positions, height, width, seed):
    """Lean angles for slats at these positions.

    Spacing varies, so the no-intersection limit is worked out per PAIR from
    that pair's own clear gap - a tight pair leans less than a wide one.

    The rule that prevents intersection is the relative one:
        |shift[i] - shift[i-1]| <= gap
    Two slats leaning the same way never converge, so each may lean up to the
    full gap in absolute terms.

    A second rule stops neighbours looking alike:
        |shift[i] - shift[i-1]| >= 0.35 * gap
    """
    import random as _r
    import math as _m
    _r.seed(seed)
    n = len(positions)
    shifts = []
    prev = None
    caps = []
    for i in range(n):
        if i == 0:
            gap = (positions[1] - positions[0] - width) if n > 1 else 1000.0
        else:
            gap = positions[i] - positions[i - 1] - width
        gap = max(gap, 1.0)
        caps.append(gap)
        dmin = 0.35 * gap
        pick = None
        for _ in range(400):
            v = _r.uniform(-gap, gap)
            if prev is None:
                pick = v
                break
            if dmin <= abs(v - prev) <= gap:
                pick = v
                break
        if pick is None:
            pick = prev - dmin if prev > 0 else prev + dmin
            pick = max(-gap, min(gap, pick))
        shifts.append(pick)
        prev = pick
    angles = [_m.degrees(_m.atan(sh / height)) if height > 0 else 0.0
              for sh in shifts]
    max_lean = _m.degrees(_m.atan(max(caps) / height)) if height > 0 else 0.0
    return angles, max_lean


def run_key(a, b):
    """Identifies one slat run, so a rerun replaces only its own slats."""
    return "{0} {1},{2}->{3},{4}".format(
        SLAT_TAG,
        int(round(a.X * 304.8)), int(round(a.Y * 304.8)),
        int(round(b.X * 304.8)), int(round(b.Y * 304.8)))


def purge_previous(key):
    """Delete slats this script placed on the same run before."""
    doomed = []
    for e in FilteredElementCollector(doc)\
            .OfCategory(getattr(BuiltInCategory, SLAT_CATEGORY))\
            .WhereElementIsNotElementType():
        try:
            p = e.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
            if p and p.AsString() == key:
                doomed.append(e.Id)
        except Exception:
            pass
    for eid in doomed:
        try:
            doc.Delete(eid)
        except Exception:
            pass
    if doomed:
        note("replaced {0} slats from a previous run".format(len(doomed)))
    return len(doomed)


def bip(*names):
    for n in names:
        v = getattr(BuiltInParameter, n, None)
        if v is not None:
            return v
    return None


def set_bip(elem, value, *names):
    b = bip(*names)
    if b is None:
        return False
    p = elem.get_Parameter(b)
    if p is None or p.IsReadOnly:
        return False
    p.Set(value)
    return True


def find_type(cls, name):
    for e in FilteredElementCollector(doc).OfClass(cls):
        if ename(e) == name:
            return e
    return None


def glazing_type():
    """The thin-mullion curtain wall that sits behind the slats."""
    thin = find_type(MullionType, "Thin 40x60")
    if thin is None:
        base = None
        for c in FilteredElementCollector(doc).OfClass(MullionType):
            fn = ""
            try:
                fn = ename(c.Family)
            except Exception:
                pass
            if "rect" in (fn + " " + ename(c)).lower():
                base = c
                break
        if base is None:
            raise Exception("No rectangular mullion type to duplicate.")
        thin = base.Duplicate("Thin 40x60")
    set_bip(thin, MULL_SIDE * MM, "RECT_MULLION_WIDTH1")
    set_bip(thin, MULL_SIDE * MM, "RECT_MULLION_WIDTH2")
    set_bip(thin, MULL_THICK * MM, "RECT_MULLION_THICK", "MULLION_THICKNESS")

    wt = find_type(WallType, GLASS_TYPE)
    if wt is None:
        base = None
        for w in FilteredElementCollector(doc).OfClass(WallType):
            if str(w.Kind) == "Curtain":
                base = w
                break
        if base is None:
            raise Exception("No curtain wall type to duplicate.")
        wt = base.Duplicate(GLASS_TYPE)
    for pt in FilteredElementCollector(doc).OfClass(PanelType):
        if "glaz" in ename(pt).lower():
            set_bip(wt, pt.Id, "AUTO_PANEL_WALL", "AUTO_PANEL")
            break
    set_bip(wt, 1, "SPACING_LAYOUT_VERT")
    set_bip(wt, GLASS_MODULE * MM, "SPACING_LENGTH_VERT")
    set_bip(wt, 0, "SPACING_LAYOUT_HORIZ")
    for nm in ("AUTO_MULLION_INTERIOR_VERT",
               "AUTO_MULLION_BORDER1_VERT",
               "AUTO_MULLION_BORDER2_VERT"):
        set_bip(wt, thin.Id, nm)
    note("glazing type '{0}': {1} mm grid, 40 mm sightline".format(
        GLASS_TYPE, GLASS_MODULE))
    return wt


def main():
    if SHOW_DIALOG and not ask_settings():
        return

    a, b, host = get_run()
    run = b - a
    length = run.GetLength()
    if length < 0.01:
        raise Exception("That run has no length.")
    d = run.Normalize()
    n = d.CrossProduct(XYZ.BasisZ).Normalize().Multiply(AIR_GAP * MM * FLIP_SIDE)

    # height
    h = SLAT_HEIGHT * MM
    if h <= 0:
        if host is not None:
            hp = host.get_Parameter(BuiltInParameter.WALL_USER_HEIGHT_PARAM)
            if hp:
                h = hp.AsDouble()
        if h <= 0:
            h = 3000.0 * MM
    note("run {0} mm, slat height {1} mm".format(
        round(length * 304.8, 1), round(h * 304.8, 1)))

    # level
    level = None
    if host is not None:
        level = doc.GetElement(host.LevelId)
    if level is None:
        lv = sorted(FilteredElementCollector(doc).OfClass(Level),
                    key=lambda x: x.Elevation)
        if not lv:
            raise Exception("The project has no levels.")
        level = lv[0]
    base_z = level.Elevation

    new_ids = []
    if min(SLAT_SPACING_MIN, SLAT_SPACING_MAX) <= SLAT_WIDTH:
        raise Exception(
            "Slat spacing must exceed the slat width.\n\n"
            "slat width  = {0} mm\n"
            "spacing min = {1} mm\n"
            "spacing max = {2} mm\n\n"
            "Spacing is centre to centre.".format(
                SLAT_WIDTH, SLAT_SPACING_MIN, SLAT_SPACING_MAX))
    positions = make_positions(length * 304.8, RANDOM_SEED)
    count = len(positions)
    angle_list, max_lean = make_angles_for(positions, h * 304.8,
                                           SLAT_WIDTH, RANDOM_SEED)
    gaps = [int(round(positions[k] - positions[k - 1]))
            for k in range(1, len(positions))]
    note("{0} slats, spacing {1}-{2} mm (actual {3}{4})".format(
        count, SLAT_SPACING_MIN, SLAT_SPACING_MAX, gaps[:10],
        " ..." if len(gaps) > 10 else ""))
    note("max lean {0} deg ({1} to {2} from horizontal); limit computed "
         "per pair from that pair's own gap".format(
             round(max_lean, 2), round(90 - max_lean, 1),
             round(90 + max_lean, 1)))

    t = Transaction(doc, "Slanted stone slats")
    t.Start()
    try:
        cat_id = ElementId(getattr(BuiltInCategory, SLAT_CATEGORY))
        mat_id = stone_material_id()
        note("slats built as DirectShape solids in category " + SLAT_CATEGORY)

        key = run_key(a, b)
        if REPLACE_EXISTING:
            purge_previous(key)
        else:
            note("REPLACE_EXISTING is off - duplicates are possible")

        made = 0
        angles = []
        new_ids = []
        for i in range(count):
            dist = positions[i] * MM
            if dist > length:
                break
            # random lean, never below LEAN_MIN, either direction
            ang = angle_list[i]
            angles.append(round(ang, 2))
            # lean happens IN the plane of the wall -> shift the top along d
            shift = h * math.tan(math.radians(ang))
            p_base = a + d.Multiply(dist) + n + XYZ(0, 0, base_z - a.Z)
            try:
                solid = slat_solid(p_base, d, n.Normalize(), shift, h, mat_id)
                ds = place_slat(solid, key, cat_id)
                new_ids.append(ds.Id)
                made += 1
            except Exception as ex:
                note("!! slat {0} failed: {1}".format(i, ex))
                if made == 0:
                    break
        note("placed {0} of {1} slats".format(made, count))

        # make sure they are actually visible in this view
        try:
            v = doc.ActiveView
            cat = doc.Settings.Categories.get_Item(
                getattr(BuiltInCategory, SLAT_CATEGORY))
            if v.GetCategoryHidden(cat.Id):
                v.SetCategoryHidden(cat.Id, False)
                note(SLAT_CATEGORY + " was HIDDEN in this view - unhidden")
            else:
                note(SLAT_CATEGORY + " already visible in this view")
        except Exception as ex:
            note("could not check category visibility: {0}".format(ex))

        # the wall behind: turn it into the thin-mullion glazing
        if CONVERT_HOST and host is not None:
            was = ename(host.WallType)
            host.WallType = glazing_type()
            note("selected wall retyped: '{0}' -> '{1}'".format(
                was, GLASS_TYPE))
        elif CONVERT_HOST:
            note("!! no wall was selected, so nothing to convert "
                 "(you picked a line)")
        note("lean angles: " + str(angles[:20]) +
             (" ..." if len(angles) > 20 else ""))
        t.Commit()
    except Exception as ex:
        if t.HasStarted() and not t.HasEnded():
            t.RollBack()
        TaskDialog.Show("Failed - rolled back",
                        "{0}\n\n{1}".format(ex, "\n".join(log)))
        raise

    # select what we made and zoom to it, so it cannot be "invisible"
    try:
        if new_ids:
            idlist = List[ElementId](new_ids)
            uidoc.Selection.SetElementIds(idlist)
            uidoc.ShowElements(idlist)
            note("selected and zoomed to the {0} new slats".format(len(new_ids)))
    except Exception as ex:
        note("could not zoom to the slats: {0}".format(ex))

    TaskDialog.Show("Done", "\n".join(log) +
                    "\n\nAdjust LEAN_MIN / LEAN_MAX for more or less tilt, "
                    "SLAT_WIDTH / SLAT_DEPTH for size, RANDOM_SEED to reshuffle.")


main()
