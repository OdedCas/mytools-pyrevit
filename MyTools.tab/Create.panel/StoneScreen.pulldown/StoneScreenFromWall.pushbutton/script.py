# -*- coding: utf-8 -*-
"""Turn a wall you drew with Revit's own tool into the stone screen assembly.

Draw the wall normally (Architecture > Wall) - you get the crosshair, the
rubber band, the live length readout and snapping, none of which the Revit
API can reproduce.  Then select it and press this button.

The selected wall becomes the glazing; a second wall carrying the stone
slats is created 200 mm to the exterior on the same line.

IronPython 2.7 compatible.
"""

from Autodesk.Revit.DB import (
    Transaction, Line, XYZ, Wall, WallType, FilteredElementCollector,
    BuiltInParameter, MullionType, PanelType, Material, Element, StorageType
)
from Autodesk.Revit.UI import TaskDialog
from Autodesk.Revit.UI.Selection import ObjectType
import clr
import random
import math


# shared settings dialog (extension lib folder)
try:
    from slatui import ask_settings
except ImportError:
    import sys
    _p = os.path.join(os.environ.get("APPDATA", ""), "pyRevit",
                      "Extensions", "MyTools.extension", "lib")
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
    from slatui import ask_settings

doc = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument

MM = 1.0 / 304.8
# Millimetres in, feet derived - so the dialog and the code agree.
SLAT_W_MM       = 100.0
SLAT_D_MM       = 300.0
SLAT_SPACING_MM = 600.0
AIR_GAP_MM      = 200.0
GLASS_MODULE_MM = 1200.0
MULL_SIDE_MM    = 20.0
MULL_THICK_MM   = 60.0
FLIP_SIDE    = 1.0
SHOW_DIALOG  = True


def apply_units():
    g = globals()
    g["SLAT_W"] = SLAT_W_MM * MM
    g["SLAT_D"] = SLAT_D_MM * MM
    g["SLAT_SPACING"] = SLAT_SPACING_MM * MM
    g["AIR_GAP"] = AIR_GAP_MM * MM
    g["GLASS_MODULE"] = GLASS_MODULE_MM * MM
    g["MULL_SIDE"] = MULL_SIDE_MM * MM
    g["MULL_THICK"] = MULL_THICK_MM * MM


SLAT_W = SLAT_D = SLAT_SPACING = AIR_GAP = 0.0
GLASS_MODULE = MULL_SIDE = MULL_THICK = 0.0
apply_units()

# --- slat variation ---------------------------------------------------------
# Width and angle live on the mullion TYPE, not the instance, so varying them
# per slat means making a set of types and assigning them one by one.
SLAT_VARIANTS   = 14      # how many distinct slat types to generate
SLAT_W_MIN      = 80.0    # mm - narrowest slat
SLAT_W_MAX      = 220.0   # mm - widest slat
SLAT_ANGLE_MAX  = 35.0    # degrees - max rotation either way about the slat axis
SLAT_DEPTH_VARY = 0.0     # mm - +/- variation on the 300 mm projection, 0 = none
RANDOM_SEED     = 1       # change for a different arrangement; fixed = repeatable

GLASS_TYPE = "CW - Thin Mullion"
STONE_TYPE = "CW - Stone Screen"
STONE_MATERIAL = "stone"

log = []
_NP = clr.GetClrType(Element).GetProperty("Name")



def settings():
    return ask_settings(
        "Screen From Wall - settings", globals(),
        fields=[
            ("SLAT_D_MM",       "Slat depth (mm, projecting out)", "float"),
            ("SLAT_SPACING_MM", "Slat spacing (mm, centre to centre)", "float"),
            ("AIR_GAP_MM",      "Gap, stone back to glass (mm)", "float"),
            ("GLASS_MODULE_MM", "Glazing grid spacing (mm)", "float"),
            ("MULL_SIDE_MM",    "Mullion half width (mm)", "float"),
            ("MULL_THICK_MM",   "Mullion depth (mm)", "float"),
            ("SLAT_W_MIN",      "Slat width min (mm)", "float"),
            ("SLAT_W_MAX",      "Slat width max (mm)", "float"),
            ("SLAT_ANGLE_MAX",  "Max slat rotation (degrees)", "float"),
            ("SLAT_VARIANTS",   "How many slat variants", "int"),
            ("RANDOM_SEED",     "Random seed", "int"),
        ],
        checks=[("FLIP_SIDE", "Screen on the other side of the wall",
                 -1.0, 1.0)],
        store="fromwall")


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


def bip(*names):
    for n in names:
        v = getattr(BuiltInParameter, n, None)
        if v is not None:
            return v
    return None


def set_bip(elem, value, *names):
    b = bip(*names)
    if b is None:
        note("!! no BuiltInParameter matched " + str(names))
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


def build_mullion(name, side, thick, material=None):
    mt = find_type(MullionType, name)
    if mt is None:
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
        mt = base.Duplicate(name)
    set_bip(mt, side, "RECT_MULLION_WIDTH1")
    set_bip(mt, side, "RECT_MULLION_WIDTH2")
    set_bip(mt, thick, "RECT_MULLION_THICK", "MULLION_THICKNESS")
    if material:
        for m in FilteredElementCollector(doc).OfClass(Material):
            if material.lower() in ename(m).lower():
                set_bip(mt, m.Id, "MATERIAL_ID_PARAM")
                break
    return mt


def set_by_name(elem, pname, value):
    """Set a parameter by its display name - used for 'Angle', which has no
    BuiltInParameter we can rely on across versions."""
    for p in elem.Parameters:
        try:
            if p.Definition.Name == pname and not p.IsReadOnly:
                p.Set(value)
                return True
        except Exception:
            pass
    return False


def build_slat_variants(material=None):
    """A family of slat types differing in width and rotation."""
    random.seed(RANDOM_SEED)
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

    variants = []
    angles_used = []
    for i in range(SLAT_VARIANTS):
        w = random.uniform(SLAT_W_MIN, SLAT_W_MAX)
        ang = random.uniform(-SLAT_ANGLE_MAX, SLAT_ANGLE_MAX)
        dep = 300.0 + random.uniform(-SLAT_DEPTH_VARY, SLAT_DEPTH_VARY)
        nm = "Stone Slat V{0:02d} w{1}".format(i + 1, int(round(w)))
        mt = find_type(MullionType, nm)
        if mt is None:
            mt = base.Duplicate(nm)
        set_bip(mt, (w / 2.0) * MM, "RECT_MULLION_WIDTH1")
        set_bip(mt, (w / 2.0) * MM, "RECT_MULLION_WIDTH2")
        set_bip(mt, dep * MM, "RECT_MULLION_THICK", "MULLION_THICKNESS")
        # Angle is stored in radians
        if not set_by_name(mt, "Angle", math.radians(ang)):
            set_bip(mt, math.radians(ang), "MULLION_ANGLE")
        if material:
            for m in FilteredElementCollector(doc).OfClass(Material):
                if material.lower() in ename(m).lower():
                    set_bip(mt, m.Id, "MATERIAL_ID_PARAM")
                    break
        variants.append(mt)
        angles_used.append((int(round(w)), round(ang, 1)))
    note("{0} slat variants (width mm, angle deg): {1}".format(
        len(variants), angles_used))
    return variants


def scatter_slats(wall, variants):
    """Assign the variant types randomly to the wall's mullions."""
    doc.Regenerate()
    cg = wall.CurtainGrid
    if cg is None:
        note("!! screen wall has no curtain grid - nothing to scatter")
        return
    mids = list(cg.GetMullionIds())
    if not mids:
        note("!! screen wall has no mullions yet - nothing to scatter")
        return
    random.seed(RANDOM_SEED + 1)
    tally = {}
    done = 0
    for mid in mids:
        m = doc.GetElement(mid)
        if m is None:
            continue
        mt = variants[random.randrange(len(variants))]
        try:
            m.ChangeTypeId(mt.Id)
            tally[ename(mt)] = tally.get(ename(mt), 0) + 1
            done += 1
        except Exception as ex:
            note("!! could not retype mullion {0}: {1}".format(mid, ex))
            break
    note("scattered {0} of {1} slats across {2} types".format(
        done, len(mids), len(tally)))


def build_wall_type(name, spacing, mullion, panel_key):
    wt = find_type(WallType, name)
    if wt is None:
        base = None
        for w in FilteredElementCollector(doc).OfClass(WallType):
            if str(w.Kind) == "Curtain":
                base = w
                break
        if base is None:
            raise Exception("No curtain wall type to duplicate.")
        wt = base.Duplicate(name)

    panel = None
    for pt in FilteredElementCollector(doc).OfClass(PanelType):
        if panel_key.lower() in ename(pt).lower():
            panel = pt
            break
    if panel is not None:
        set_bip(wt, panel.Id, "AUTO_PANEL_WALL", "AUTO_PANEL")
        note("{0}: panel = {1}".format(name, ename(panel)))
    else:
        note("!! {0}: no panel matching '{1}'. Available: {2}".format(
            name, panel_key,
            ", ".join(ename(x) for x in
                      FilteredElementCollector(doc).OfClass(PanelType))))

    set_bip(wt, 1, "SPACING_LAYOUT_VERT")
    set_bip(wt, spacing, "SPACING_LENGTH_VERT")
    set_bip(wt, 0, "SPACING_LAYOUT_HORIZ")
    ok = 0
    for nm in ("AUTO_MULLION_INTERIOR_VERT",
               "AUTO_MULLION_BORDER1_VERT",
               "AUTO_MULLION_BORDER2_VERT"):
        if set_bip(wt, mullion.Id, nm):
            ok += 1
    note("{0}: spacing {1} mm, mullion '{2}' on {3}/3 slots".format(
        name, round(spacing * 304.8, 1), ename(mullion), ok))
    return wt


def get_source_wall():
    for eid in uidoc.Selection.GetElementIds():
        e = doc.GetElement(eid)
        if isinstance(e, Wall):
            return e
    ref = uidoc.Selection.PickObject(
        ObjectType.Element,
        "Select the wall you drew (it becomes the glazing)")
    e = doc.GetElement(ref.ElementId)
    if not isinstance(e, Wall):
        raise Exception("That is not a wall.")
    return e


def main():
    if SHOW_DIALOG:
        if not settings():
            return
        apply_units()
        note("settings: slat {0}-{1} wide x {2} deep @ {3}, glazing {4}".format(
            SLAT_W_MIN, SLAT_W_MAX, SLAT_D_MM, SLAT_SPACING_MM,
            GLASS_MODULE_MM))

    src = get_source_wall()
    crv = src.Location.Curve
    a = crv.GetEndPoint(0)
    b = crv.GetEndPoint(1)
    # flatten - a wall drawn in plan is level, but never assume it
    a = XYZ(a.X, a.Y, 0)
    b = XYZ(b.X, b.Y, 0)
    if a.DistanceTo(b) < 0.01:
        raise Exception("That wall has no length.")
    note("source wall {0}, length {1} mm".format(
        src.Id, round(a.DistanceTo(b) * 304.8, 1)))

    level_id = src.LevelId
    t = Transaction(doc, "Stone screen from wall")
    t.Start()
    try:
        variants = build_slat_variants(STONE_MATERIAL)
        slat = variants[0]
        thin = build_mullion("Thin 40x60", MULL_SIDE, MULL_THICK)
        glass_wt = build_wall_type(GLASS_TYPE, GLASS_MODULE, thin, "glazed")
        stone_wt = build_wall_type(STONE_TYPE, SLAT_SPACING, slat, "empty")

        # retype the drawn wall into the glazing
        src.WallType = glass_wt
        note("source wall retyped to " + GLASS_TYPE)

        # stone screen on a parallel line, offset to the exterior
        d = (b - a).Normalize()
        n = d.CrossProduct(XYZ.BasisZ).Normalize().Multiply(AIR_GAP * FLIP_SIDE)
        screen = Wall.Create(doc, Line.CreateBound(a + n, b + n),
                             stone_wt.Id, level_id, 3000.0 * MM, 0.0,
                             False, False)

        # copy the drawn wall's height constraints onto the screen
        for bn in ("WALL_HEIGHT_TYPE", "WALL_USER_HEIGHT_PARAM",
                   "WALL_TOP_OFFSET", "WALL_BASE_OFFSET"):
            bp = bip(bn)
            if bp is None:
                continue
            sp = src.get_Parameter(bp)
            dp = screen.get_Parameter(bp)
            if sp is None or dp is None or dp.IsReadOnly:
                continue
            try:
                if sp.StorageType == StorageType.ElementId:
                    dp.Set(sp.AsElementId())
                elif sp.StorageType == StorageType.Double:
                    dp.Set(sp.AsDouble())
                elif sp.StorageType == StorageType.Integer:
                    dp.Set(sp.AsInteger())
            except Exception:
                pass
        note("stone screen wall {0} created, height copied from source".format(
            screen.Id))
        scatter_slats(screen, variants)
        t.Commit()
    except Exception as ex:
        if t.HasStarted() and not t.HasEnded():
            t.RollBack()
        TaskDialog.Show("Failed - rolled back",
                        "{0}\n\n{1}".format(ex, "\n".join(log)))
        raise

    TaskDialog.Show("Done", "\n".join(log) +
                    "\n\nIf the stone landed inside, set FLIP_SIDE = -1.0 "
                    "and undo/rerun.")


main()
