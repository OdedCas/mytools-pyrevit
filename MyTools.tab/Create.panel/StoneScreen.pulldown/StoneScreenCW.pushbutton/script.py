# -*- coding: utf-8 -*-
"""Stone slat screen + thin-mullion curtain wall.

Creates:
  1. A mullion profile family (100 wide x 300 deep) for the stone slats.
  2. Wall type "CW - Thin Mullion"  : glazed, 1200 mm vertical grid, 40 mm sightline.
  3. Wall type "CW - Stone Screen"  : empty panels, 600 mm vertical grid, stone slat mullions.
  4. Both walls between two points you pick, stone offset 200 mm to the exterior.

IronPython 2.7 compatible.  Run from pyRevit in a PLAN view.
"""

from Autodesk.Revit.DB import (
    Transaction, Line, XYZ, Wall, WallType, FilteredElementCollector,
    BuiltInCategory, BuiltInParameter, ElementId, Level, MullionType, Material,
    PanelType, FamilySymbol, IFamilyLoadOptions, FamilySource, SaveAsOptions,
    CurveArray, SketchPlane, Plane, View, ViewPlan, Element
)
from Autodesk.Revit.UI import (TaskDialog, TaskDialogCommandLinkId,
                               TaskDialogCommonButtons, TaskDialogResult)
from Autodesk.Revit.UI.Selection import ObjectType
from Autodesk.Revit.DB import CurveElement
import os, glob, time

# shared settings dialog (extension lib folder)
try:
    from slatui import ask_settings
except ImportError:
    import sys
    for _p in (os.path.join(os.environ.get("APPDATA", ""), "pyRevit",
                            "Extensions", "MyTools.extension", "lib"),):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.append(_p)
    from slatui import ask_settings


import clr

doc = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument
app = doc.Application

# ----------------------------------------------------------------- parameters
MM = 1.0 / 304.8

# Everything is entered in MILLIMETRES; the _ft values below are derived and
# re-derived after the dialog, so the dialog and the code never disagree.
SLAT_W_MM        = 100.0    # stone thickness across the face
SLAT_D_MM        = 300.0    # stone projection out from the screen plane
SLAT_SPACING_MM  = 600.0    # centre to centre
AIR_GAP_MM       = 200.0    # back of stone -> glass plane
GLASS_MODULE_MM  = 1200.0
MULL_SIDE_MM     = 20.0     # each side -> 40 mm total sightline
MULL_THICK_MM    = 60.0
FLIP_SIDE     = 1.0          # -1.0 puts the screen on the other side


def apply_units():
    """Refresh the feet-based values from the millimetre ones."""
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

USE_PROFILE_FAMILY = False   # True only if the slat is not a plain rectangle
STONE_MATERIAL     = "stone"  # substring match against project materials

FAM_NAME   = "Stone Slat 100x300"
FAM_DIR    = "C:\\Users\\cassu\\dev\\revit_profiles"
SHOW_DIALOG = True
GLASS_TYPE = "CW - Thin Mullion"
STONE_TYPE = "CW - Stone Screen"

log = []



def settings():
    return ask_settings(
        "Stone Screen CW - settings", globals(),
        fields=[
            ("SLAT_W_MM",       "Slat width (mm)", "float"),
            ("SLAT_D_MM",       "Slat depth (mm)", "float"),
            ("SLAT_SPACING_MM", "Slat spacing (mm, centre to centre)", "float"),
            ("GLASS_MODULE_MM", "Glazing grid spacing (mm)", "float"),
            ("MULL_SIDE_MM",    "Mullion half width (mm, 20 = 40 sightline)", "float"),
            ("MULL_THICK_MM",   "Mullion depth (mm)", "float"),
            ("AIR_GAP_MM",      "Gap, stone back to glass (mm)", "float"),
        ],
        checks=[("FLIP_SIDE", "Screen on the other side of the wall",
                 -1.0, 1.0),
                ("USE_PROFILE_FAMILY",
                 "Use a custom profile family (only for non-rectangular slats)",
                 True, False)],
        store="cw")


def note(msg):
    log.append(msg)


# ------------------------------------------------------------------- helpers
_ELEM_NAME_PROP = clr.GetClrType(Element).GetProperty("Name")


def ename(e):
    """Element.Name is hidden on some derived types under IronPython and
    raises MissingMemberException - fall back to reflection, then params."""
    if e is None:
        return ""
    try:
        return e.Name
    except Exception:
        pass
    try:
        v = _ELEM_NAME_PROP.GetValue(e, None)
        if v:
            return v
    except Exception:
        pass
    for bn in ("SYMBOL_NAME_PARAM", "ALL_MODEL_TYPE_NAME"):
        b = getattr(BuiltInParameter, bn, None)
        if b is None:
            continue
        try:
            pr = e.get_Parameter(b)
            if pr:
                v = pr.AsString()
                if v:
                    return v
        except Exception:
            pass
    return ""


def fam_name(e):
    """Family name of a type element, safely."""
    try:
        return ename(e.Family)
    except Exception:
        return ""


def bip(*names):
    """Return the first BuiltInParameter that exists under these names."""
    for n in names:
        v = getattr(BuiltInParameter, n, None)
        if v is not None:
            return v
    return None


def set_bip(elem, value, *names):
    b = bip(*names)
    if b is None:
        note("!! no BuiltInParameter matched {0}".format(names))
        return False
    p = elem.get_Parameter(b)
    if p is None or p.IsReadOnly:
        note("!! parameter {0} missing or read-only".format(names[0]))
        return False
    p.Set(value)
    return True


def find_type(cls, name):
    for e in FilteredElementCollector(doc).OfClass(cls):
        if ename(e) == name:
            return e
    return None


def first_type(cls):
    for e in FilteredElementCollector(doc).OfClass(cls):
        return e
    return None


class Loader(IFamilyLoadOptions):
    def OnFamilyFound(self, inUse, overwriteParameterValues):
        overwriteParameterValues.Value = True
        return True

    def OnSharedFamilyFound(self, sharedFamily, familyInUse, source, overwriteParameterValues):
        source.Value = FamilySource.Family
        overwriteParameterValues.Value = True
        return True


def close_stale_family_docs():
    """An aborted run can leave our family document open, locking the .rfa."""
    for d in list(app.Documents):
        try:
            if not d.IsFamilyDocument:
                continue
            pn = d.PathName or ""
            if pn.lower().startswith(FAM_DIR.lower()) or FAM_NAME in (d.Title or ""):
                d.Close(False)
                note("closed orphaned family document: " + (d.Title or pn))
        except Exception:
            pass


def family_sketch_view(fdoc):
    """A family made with NewFamilyDocument has no ActiveView - go find one."""
    views = []
    for v in FilteredElementCollector(fdoc).OfClass(View):
        try:
            if v.IsTemplate:
                continue
        except Exception:
            pass
        views.append(v)
    # a profile template carries a single 2D view; prefer a plan if several
    for v in views:
        if isinstance(v, ViewPlan):
            return v
    if views:
        return views[0]
    raise Exception("The profile family template has no usable view to sketch in.")


# ------------------------------------------------- 1. build the profile family
def build_profile():
    existing = None
    for fs in FilteredElementCollector(doc).OfClass(FamilySymbol):
        if fs.Family and fam_name(fs) == FAM_NAME:
            existing = fs
            break
    if existing:
        note("profile family already present, reusing it")
        return existing

    tpath = app.FamilyTemplatePath
    hits = []
    for pat in ("*Profile*Mullion*.rft", "*Mullion*Profile*.rft"):
        hits += glob.glob(os.path.join(tpath, pat))
        hits += glob.glob(os.path.join(tpath, "*", pat))
    if not hits:
        raise Exception(
            "Could not find a mullion profile template (.rft) under:\n" + tpath +
            "\nAuthor the profile manually: 100 wide x 300 deep rectangle, "
            "centred on the vertical ref plane, running from the ref-plane "
            "intersection outward."
        )
    template = hits[0]
    note("profile template: " + os.path.basename(template))

    close_stale_family_docs()

    fdoc = app.NewFamilyDocument(template)
    view = family_sketch_view(fdoc)
    ft = Transaction(fdoc, "sketch slat profile")
    ft.Start()
    # Origin = curtain grid line.  X = along the wall, Y = out toward exterior.
    hw = SLAT_W / 2.0
    pts = [XYZ(-hw, 0, 0), XYZ(hw, 0, 0), XYZ(hw, SLAT_D, 0), XYZ(-hw, SLAT_D, 0)]
    for i in range(4):
        seg = Line.CreateBound(pts[i], pts[(i + 1) % 4])
        fdoc.FamilyCreate.NewDetailCurve(view, seg)
    ft.Commit()

    if not os.path.isdir(FAM_DIR):
        os.makedirs(FAM_DIR)

    opts = SaveAsOptions()
    opts.OverwriteExistingFile = True

    fam = None
    try:
        path = os.path.join(FAM_DIR, FAM_NAME + ".rfa")
        try:
            fdoc.SaveAs(path, opts)
        except Exception as ex:
            # A previous failed run can leave the .rfa locked by an orphaned
            # in-memory family document.  Sidestep it with a fresh filename.
            note("!! clean save failed ({0}) - using a unique filename".format(ex))
            path = os.path.join(
                FAM_DIR, "{0}_{1}.rfa".format(FAM_NAME, int(time.time())))
            fdoc.SaveAs(path, opts)
        note("profile family saved to " + path)
        fam = fdoc.LoadFamily(doc, Loader())
    finally:
        try:
            fdoc.Close(False)
        except Exception:
            pass

    if fam is None:
        raise Exception("The profile family did not load into the project.")
    for sid in fam.GetFamilySymbolIds():
        return doc.GetElement(sid)
    return None


# ------------------------------------------------------ 2. mullion + wall types
def build_mullion(name, profile_symbol=None, side=None, thick=None, material=None):
    mt = find_type(MullionType, name)
    if mt:
        note("mullion type '{0}' already exists, reusing".format(name))
        return mt
    # Prefer a rectangular base type: the width/thickness parameters only
    # exist on rectangular mullions, not circular or profile-driven ones.
    base = None
    for mt2 in FilteredElementCollector(doc).OfClass(MullionType):
        fam = fam_name(mt2)
        if "rect" in (fam + " " + ename(mt2)).lower():
            base = mt2
            break
    if base is None:
        base = first_type(MullionType)
        note("!! no rectangular mullion found - duplicating '{0}' instead".format(
            ename(base) if base else "nothing"))
    if base is None:
        raise Exception("No mullion type in the project to duplicate from.")
    mt = base.Duplicate(name)

    if profile_symbol is not None:
        set_bip(mt, profile_symbol.Id, "MULLION_PROFILE_PARAM", "MULLION_PROFILE")
    if side is not None:
        set_bip(mt, side, "RECT_MULLION_WIDTH1")
        set_bip(mt, side, "RECT_MULLION_WIDTH2")
    if thick is not None:
        set_bip(mt, thick, "RECT_MULLION_THICK", "MULLION_THICKNESS")
    if material:
        mat = None
        for m in FilteredElementCollector(doc).OfClass(Material):
            if material.lower() in ename(m).lower():
                mat = m
                break
        if mat is not None:
            set_bip(mt, mat.Id, "MATERIAL_ID_PARAM")
            note("material '{0}' applied to {1}".format(ename(mat), name))
        else:
            note("!! no material matching '{0}' - set it by hand".format(material))
    return mt


def build_wall_type(name, spacing, mullion, panel_name):
    """Find or create the type, then ALWAYS (re)apply its settings - an
    existing type from a previous run still needs configuring."""
    wt = find_type(WallType, name)
    if wt:
        note("wall type '{0}' exists - reapplying settings".format(name))
    else:
        base = None
        for w in FilteredElementCollector(doc).OfClass(WallType):
            if str(w.Kind) == "Curtain":
                base = w
                break
        if base is None:
            raise Exception("No curtain wall type in the project to duplicate from.")
        wt = base.Duplicate(name)

    # curtain panel
    panel = None
    for pt in FilteredElementCollector(doc).OfClass(PanelType):
        pn = ename(pt).lower()
        if panel_name.lower() in pn:
            panel = pt
            break
    if panel is not None:
        if set_bip(wt, panel.Id, "AUTO_PANEL_WALL", "AUTO_PANEL"):
            note("{0}: panel = {1}".format(name, ename(panel)))
    else:
        avail = ", ".join(ename(pt) for pt in
                          FilteredElementCollector(doc).OfClass(PanelType))
        note("!! {0}: no panel matching '{1}'. Available: {2}".format(
            name, panel_name, avail))

    # vertical grid: fixed distance at `spacing`; horizontal: none
    if set_bip(wt, 1, "SPACING_LAYOUT_VERT", "AUTO_SPACING_LAYOUT_VERT"):
        note("{0}: vertical layout = Fixed Distance".format(name))
    if set_bip(wt, spacing, "SPACING_LENGTH_VERT", "AUTO_SPACING_VERT"):
        note("{0}: vertical spacing = {1} mm".format(name, round(spacing * 304.8, 1)))
    set_bip(wt, 0, "SPACING_LAYOUT_HORIZ", "AUTO_SPACING_LAYOUT_HORIZ")

    # mullions on every vertical line, including both ends
    ok = 0
    for nm in ("AUTO_MULLION_INTERIOR_VERT",
               "AUTO_MULLION_BORDER1_VERT",
               "AUTO_MULLION_BORDER2_VERT"):
        if set_bip(wt, mullion.Id, nm):
            ok += 1
    note("{0}: mullion '{1}' set on {2}/3 slots".format(name, ename(mullion), ok))
    return wt


# ------------------------------------------------------------------ 3. drive it
def curve_from_element(e):
    """Location curve of a line or wall, flattened to horizontal."""
    try:
        c = e.Location.Curve
    except Exception:
        try:
            c = e.GeometryCurve
        except Exception:
            return None
    if c is None:
        return None
    a = c.GetEndPoint(0)
    b = c.GetEndPoint(1)
    return XYZ(a.X, a.Y, 0), XYZ(b.X, b.Y, 0)


def get_run():
    """Where the wall run goes.

    Revit's API has no rubber-band picker - PickPoint draws nothing and
    reports no length.  So prefer reading a line the user drew with Revit's
    own Line tool, which gives full rubber band, live length and snapping.
    """
    # 1. something already selected?
    for eid in uidoc.Selection.GetElementIds():
        e = doc.GetElement(eid)
        if isinstance(e, (CurveElement, Wall)):
            r = curve_from_element(e)
            if r:
                note("using the selected {0} as the wall run".format(
                    e.GetType().Name))
                return r

    # 2. offer to pick a line
    td = TaskDialog("Stone screen - where does the wall go?")
    td.MainInstruction = "Pick a line, or pick two points?"
    td.MainContent = (
        "A LINE gives you Revit's rubber band, live length and snapping "
        "while you draw it.\n\n"
        "Draw one first with Annotate > Detail Line or Architecture > "
        "Model Line, then choose 'Pick a line'.\n\n"
        "Two points is quicker but shows no length while picking - the "
        "Revit API cannot draw a rubber band.")
    td.AddCommandLink(TaskDialogCommandLinkId.CommandLink1,
                      "Pick a line I already drew")
    td.AddCommandLink(TaskDialogCommandLinkId.CommandLink2,
                      "Pick two points")
    td.CommonButtons = TaskDialogCommonButtons.Cancel
    res = td.Show()

    if res == TaskDialogResult.Cancel:
        raise Exception("Cancelled.")

    if res == TaskDialogResult.CommandLink1:
        ref = uidoc.Selection.PickObject(
            ObjectType.Element, "Select the line for the wall run")
        e = doc.GetElement(ref.ElementId)
        r = curve_from_element(e)
        if not r:
            raise Exception("That element has no usable line.")
        note("using picked line: " + e.GetType().Name)
        return r

    p1 = uidoc.Selection.PickPoint("Start of wall")
    p2 = uidoc.Selection.PickPoint("End of wall")
    return XYZ(p1.X, p1.Y, 0), XYZ(p2.X, p2.Y, 0)


def main():
    if SHOW_DIALOG:
        if not settings():
            return
        apply_units()
        note("settings: slat {0}x{1} @ {2}, glazing {3}, gap {4}".format(
            SLAT_W_MM, SLAT_D_MM, SLAT_SPACING_MM, GLASS_MODULE_MM,
            AIR_GAP_MM))

    view = doc.ActiveView
    level = None
    if hasattr(view, "GenLevel") and view.GenLevel is not None:
        level = view.GenLevel
    if level is None:
        # No level in this view (3D, elevation) - fall back to the lowest,
        # rather than refusing to run at all.
        lv = sorted(FilteredElementCollector(doc).OfClass(Level),
                    key=lambda l: l.Elevation)
        if not lv:
            raise Exception("The project has no levels.")
        level = lv[0]
        note("view has no level - using '{0}'".format(ename(level)))

    p1, p2 = get_run()
    if p1.DistanceTo(p2) < 0.01:
        raise Exception("The two points are the same.")
    note("wall run length = {0} mm".format(round(p1.DistanceTo(p2) * 304.8, 1)))

    # top constraint: level above, else 3000 mm unconnected
    levels = sorted(FilteredElementCollector(doc).OfClass(Level),
                    key=lambda l: l.Elevation)
    top = None
    for l in levels:
        if l.Elevation > level.Elevation + 0.01:
            top = l
            break

    # Family creation and LoadFamily must happen with NO transaction open on
    # doc - LoadFamily opens its own internally and throws if one is active.
    prof = build_profile() if USE_PROFILE_FAMILY else None

    t = Transaction(doc, "Stone slat screen + curtain wall")
    t.Start()
    try:
        if USE_PROFILE_FAMILY:
            slat_mull = build_mullion("Stone Slat 100x300", profile_symbol=prof,
                                      material=STONE_MATERIAL)
        else:
            # a plain rectangle needs no profile family - the stock rectangular
            # mullion is parametric: widths across the face, thickness = depth
            slat_mull = build_mullion("Stone Slat 100x300",
                                      side=SLAT_W / 2.0, thick=SLAT_D,
                                      material=STONE_MATERIAL)
        thin_mull = build_mullion("Thin 40x60", side=MULL_SIDE, thick=MULL_THICK)

        glass_wt = build_wall_type(GLASS_TYPE, GLASS_MODULE, thin_mull, "glazed")
        stone_wt = build_wall_type(STONE_TYPE, SLAT_SPACING, slat_mull, "empty")

        # offset the screen to the exterior
        d = (p2 - p1).Normalize()
        n = d.CrossProduct(XYZ.BasisZ).Normalize().Multiply(AIR_GAP * FLIP_SIDE)

        glass_line = Line.CreateBound(p1, p2)
        stone_line = Line.CreateBound(p1 + n, p2 + n)

        for ln, wt, tag in ((glass_line, glass_wt, "glazing"),
                            (stone_line, stone_wt, "stone screen")):
            existing = [x for x in FilteredElementCollector(doc).OfClass(Wall)
                        if x.GetTypeId() == wt.Id]
            if existing:
                note("{0}: {1} wall(s) already exist - reusing, not duplicating"
                     .format(tag, len(existing)))
                continue
            w = Wall.Create(doc, ln, wt.Id, level.Id, 3000.0 * MM, 0.0, False, False)
            if top is not None:
                pr = w.get_Parameter(BuiltInParameter.WALL_HEIGHT_TYPE)
                if pr and not pr.IsReadOnly:
                    pr.Set(top.Id)
            note("created {0}: id {1}".format(tag, w.Id))

        t.Commit()
    except Exception as ex:
        if t.HasStarted() and not t.HasEnded():
            t.RollBack()
        TaskDialog.Show("Failed - rolled back",
                        "{0}\n\nProgress before the failure:\n{1}".format(
                            ex, "\n".join(log)))
        raise

    msg = "\n".join(log)
    msg += "\n\nTop constraint: " + (ename(top) if top else "unconnected 3000 mm - set it by hand")
    msg += "\n\nCheck which side the stone landed on. If it is inside, set FLIP_SIDE = -1.0 and rerun."
    TaskDialog.Show("Done", msg)


main()
