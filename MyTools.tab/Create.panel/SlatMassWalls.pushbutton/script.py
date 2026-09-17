# -*- coding: utf-8 -*-
"""Leaning stone slats as Walls, via a conceptual mass + Wall by Face.

Revit will not tilt an ordinary wall.  Wall by Face on a slanted mass face
is the only route to a genuinely leaning wall, so:

  1. build a mass family holding one thin leaning prism per slat
  2. load and place it
  3. create a Wall by Face on one face of each prism

The mass stays in the model (walls by face are hosted by it) but is hidden
unless Show Mass is on.  Edit the mass and use Update to Face to refresh.

IronPython 2.7 compatible.
"""

from Autodesk.Revit.DB import (
    Transaction, Line, XYZ, Wall, WallType, FilteredElementCollector,
    BuiltInParameter, BuiltInCategory, FamilySymbol, Element, CurveElement,
    Level, Material, Plane, SketchPlane, ReferenceArray, Options,
    GeometryInstance, Solid, PlanarFace, SaveAsOptions, IFamilyLoadOptions,
    FamilySource, CurveArray, ViewDetailLevel, MullionType, PanelType
)
from Autodesk.Revit.DB.Structure import StructuralType
from Autodesk.Revit.UI import TaskDialog, RevitCommandId, PostableCommand
from Autodesk.Revit.UI.Selection import ObjectType
import clr, random, math, os, glob, time

doc = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument
app = doc.Application

MM = 1.0 / 304.8

# ----------------------------------------------------------------- slat setup
SLAT_WIDTH   = 100.0    # mm - wall thickness of each slat
SLAT_DEPTH   = 300.0    # mm - how far it projects from the wall
SLAT_SPACING_MIN = 1200.0  # mm - closest two slats may sit, centre to centre
SLAT_SPACING_MAX = 2400.0  # mm - furthest apart
SLAT_SPACING = 1800.0   # kept only as a fallback if MIN/MAX are equal
LEAN_MAX     = 12.0   # IGNORED - the max lean is now derived from the
                          # clear gap and height so slats cannot intersect
LEAN_MIN     = 3.0      # degrees - min lean, so none read as failed verticals
AIR_GAP      = 200.0    # mm - wall face -> slat centre plane
SLAT_HEIGHT  = 0.0      # mm; 0 = copy the host wall height
MASS_THICK   = 20.0     # mm - thickness of the prisms inside the mass
MAX_SLATS    = 200      # safety cap
POST_WALL_BY_FACE = True  # launch the Wall by Face tool when the script ends
STONE_MATERIAL = "stone"
RANDOM_SEED  = 1
FLIP_SIDE    = 1.0

MASS_NAME = "Stone Slat Mass"
MASS_DIR  = "C:\\Users\\cassu\\dev\\revit_profiles"
WALL_NAME = "Stone Slat 100mm"

# --- the original wall becomes the glazing behind the slats ------------------
CONVERT_HOST = True
GLASS_TYPE   = "CW - Thin Mullion"
GLASS_MODULE = 1200.0   # mm - vertical grid of the glazing
MULL_SIDE    = 20.0     # mm - each side, so a 40 mm sightline
MULL_THICK   = 60.0     # mm - mullion depth
HORIZ_MULLION_AT = [240.0]   # mm above the wall base; [] for none
HORIZ_MULLION_TOP = True     # also put one along the top of the wall

log = []
_NP = clr.GetClrType(Element).GetProperty("Name")


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


class Loader(IFamilyLoadOptions):
    def OnFamilyFound(self, inUse, overwrite):
        overwrite.Value = True
        return True

    def OnSharedFamilyFound(self, fam, inUse, source, overwrite):
        source.Value = FamilySource.Family
        overwrite.Value = True
        return True



def make_positions(length_mm, seed):
    """Slat positions along the run, with a random gap between each pair."""
    import random as _r
    _r.seed(seed + 77)
    lo, hi = SLAT_SPACING_MIN, SLAT_SPACING_MAX
    if hi <= lo:
        lo = hi = SLAT_SPACING
    pos = [0.0]
    while True:
        nxt = pos[-1] + _r.uniform(lo, hi)
        if nxt > length_mm:
            break
        pos.append(nxt)
    return pos


def make_angles_for(positions, height, width, seed):
    """Lean angles for slats at these positions.

    Spacing varies, so the no-intersection limit is computed per PAIR from
    that pair's own clear gap, not from one global spacing.
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
        cap = gap
        dmin = 0.35 * cap
        pick = None
        for _ in range(400):
            v = _r.uniform(-cap, cap)
            if prev is None:
                pick = v
                break
            d = abs(v - prev)
            if dmin <= d <= cap:
                pick = v
                break
        if pick is None:
            pick = prev - dmin if prev > 0 else prev + dmin
            pick = max(-cap, min(cap, pick))
        shifts.append(pick)
        prev = pick
    angles = [_m.degrees(_m.atan(sh / height)) if height > 0 else 0.0
              for sh in shifts]
    max_lean = _m.degrees(_m.atan(max(caps) / height)) if height > 0 else 0.0
    return angles, max_lean


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



# ------------------------------------------------------------ 1. the mass
def build_mass(a, b, height, base_z):
    """A mass family with one thin leaning prism per slat."""
    # The conceptual mass template sits two levels down, e.g.
    #   Family Templates\English\Conceptual Mass\Metric Mass.rft
    # so walk the tree rather than globbing a fixed depth.
    root = app.FamilyTemplatePath
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".rft"):
                continue
            full = os.path.join(dirpath, fn)
            low = full.lower()
            if "conceptual mass" in low or fn.lower() in ("metric mass.rft",
                                                          "mass.rft"):
                found.append(full)
    if not found:
        raise Exception(
            "No conceptual mass template (.rft) found under:\n" + root +
            "\nExpected something like English\\Conceptual Mass\\Metric Mass.rft")

    def rank(pth):
        low = pth.lower()
        score = 0
        if "metric mass.rft" in low:
            score -= 4                      # exact metric template
        if os.sep + "english" + os.sep in low:
            score -= 2                      # prefer English over other locales
        if "imperial" in low:
            score += 3                      # this project is metric
        return score

    found.sort(key=rank)
    template = found[0]
    note("mass template: " + template.replace(root, "..."))

    # close an orphaned copy from a previous run
    for d in list(app.Documents):
        try:
            if d.IsFamilyDocument and MASS_NAME in (d.Title or ""):
                d.Close(False)
        except Exception:
            pass

    fdoc = app.NewFamilyDocument(template)

    run = b - a
    length = run.GetLength()
    dirv = run.Normalize()
    nrm = dirv.CrossProduct(XYZ.BasisZ).Normalize()
    off = nrm.Multiply(AIR_GAP * MM * FLIP_SIDE)

    positions = make_positions(length * 304.8, RANDOM_SEED)[:MAX_SLATS]
    count = len(positions)
    if min(SLAT_SPACING_MIN, SLAT_SPACING_MAX) <= SLAT_WIDTH:
        raise Exception("slat spacing must exceed SLAT_WIDTH.")
    angle_list, max_lean = make_angles_for(positions, height * 304.8,
                                           SLAT_WIDTH, RANDOM_SEED)
    gaps = [round(positions[k] - positions[k - 1], 0)
            for k in range(1, len(positions))]
    note("{0} slats, spacing {1}-{2} mm (actual {3}{4})".format(
        count, SLAT_SPACING_MIN, SLAT_SPACING_MAX, gaps[:10],
        " ..." if len(gaps) > 10 else ""))
    note("max lean {0} deg ({1} to {2} from horizontal); "
         "limit is computed per pair from that pair's own gap".format(
             round(max_lean, 2), round(90 - max_lean, 1),
             round(90 + max_lean, 1)))
    half_d = (SLAT_DEPTH * MM) / 2.0
    half_t = (MASS_THICK * MM) / 2.0

    ft = Transaction(fdoc, "slat forms")
    ft.Start()
    made = 0
    angles = []
    try:
        sp = SketchPlane.Create(
            fdoc, Plane.CreateByNormalAndOrigin(XYZ.BasisZ, XYZ.Zero))
        for i in range(count):
            dist = positions[i] * MM
            if dist > length:
                break
            ang = angle_list[i]
            angles.append(round(ang, 2))

            # centre of this slat at the base, in family coordinates
            c0 = (a - a) + dirv.Multiply(dist) + off + XYZ(0, 0, 0)
            # rectangle: SLAT_DEPTH across the normal, MASS_THICK along the run
            p = []
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p.append(c0 + nrm.Multiply(sx * half_d) +
                         dirv.Multiply(sy * half_t))
            ra = ReferenceArray()
            for k in range(4):
                ln = Line.CreateBound(p[k], p[(k + 1) % 4])
                mc = fdoc.FamilyCreate.NewModelCurve(ln, sp)
                ra.Append(mc.GeometryCurve.Reference)

            # extrude along a slanted vector -> the slat leans in the wall plane
            shift = height * math.tan(math.radians(ang))
            vec = dirv.Multiply(shift) + XYZ(0, 0, height)
            fdoc.FamilyCreate.NewExtrusionForm(True, ra, vec)
            made += 1
        ft.Commit()
    except Exception as ex:
        if ft.HasStarted() and not ft.HasEnded():
            ft.RollBack()
        try:
            fdoc.Close(False)
        except Exception:
            pass
        raise Exception("building slat forms failed at slat {0}: {1}".format(
            made, ex))

    note("mass: {0} slat forms, angles {1}{2}".format(
        made, angles[:15], " ..." if len(angles) > 15 else ""))

    if not os.path.isdir(MASS_DIR):
        os.makedirs(MASS_DIR)
    path = os.path.join(MASS_DIR, "{0}_{1}.rfa".format(MASS_NAME,
                                                       int(time.time())))
    opts = SaveAsOptions()
    opts.OverwriteExistingFile = True
    fam = None
    try:
        fdoc.SaveAs(path, opts)
        fam = fdoc.LoadFamily(doc, Loader())
    finally:
        try:
            fdoc.Close(False)
        except Exception:
            pass
    if fam is None:
        raise Exception("the mass family did not load")
    note("mass family loaded from " + os.path.basename(path))
    for sid in fam.GetFamilySymbolIds():
        return doc.GetElement(sid)
    raise Exception("the mass family has no type")


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


def add_horizontal_gridlines(wall, heights_mm):
    """Add horizontal curtain grid lines at these heights above the base.

    A wall curtain grid calls its two directions U and V, and which one is
    horizontal is not fixed - so add the line, measure the curve it produced,
    and if it came out vertical, undo it and use the other direction.
    """
    cg = wall.CurtainGrid
    if cg is None:
        note("!! wall has no curtain grid - no horizontal mullions added")
        return 0
    try:
        crv = wall.Location.Curve
    except Exception:
        note("!! wall has no location curve")
        return 0
    a = crv.GetEndPoint(0)
    b = crv.GetEndPoint(1)
    mid = XYZ((a.X + b.X) / 2.0, (a.Y + b.Y) / 2.0, 0)
    base_z = a.Z
    bp = wall.get_Parameter(BuiltInParameter.WALL_BASE_OFFSET)
    if bp:
        base_z += bp.AsDouble()

    added = 0
    for hmm in heights_mm:
        z = base_z + hmm * MM
        pt = XYZ(mid.X, mid.Y, z)
        placed = False
        for is_u in (True, False):
            gl = None
            try:
                gl = cg.AddGridLine(is_u, pt, False)
            except Exception as ex:
                continue
            if gl is None:
                continue
            horizontal = False
            try:
                c = gl.FullCurve
                d = (c.GetEndPoint(1) - c.GetEndPoint(0)).Normalize()
                horizontal = abs(d.Z) < 0.1
            except Exception:
                pass
            if horizontal:
                note("horizontal grid line added at {0} mm".format(hmm))
                added += 1
                placed = True
                break
            try:
                doc.Delete(gl.Id)
            except Exception:
                pass
        if not placed:
            note("!! could not add a horizontal grid line at {0} mm".format(hmm))
    return added


def glazing_type():
    """The thin-mullion curtain wall the original wall becomes."""
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
    # horizontal mullions: interior carries the 240 line, border2 the top
    set_bip(wt, thin.Id, "AUTO_MULLION_INTERIOR_HORIZ")
    if HORIZ_MULLION_TOP:
        set_bip(wt, thin.Id, "AUTO_MULLION_BORDER2_HORIZ")
        note("top horizontal mullion enabled")
    note("glazing type '{0}': {1} mm grid, 40 mm sightline".format(
        GLASS_TYPE, GLASS_MODULE))
    return wt


# ------------------------------------------------------- 2. the wall type
def slat_wall_type():
    wt = None
    for w in FilteredElementCollector(doc).OfClass(WallType):
        if ename(w) == WALL_NAME:
            wt = w
            break
    if wt is None:
        base = None
        for w in FilteredElementCollector(doc).OfClass(WallType):
            if str(w.Kind) == "Basic":
                base = w
                break
        if base is None:
            raise Exception("No basic wall type to duplicate.")
        wt = base.Duplicate(WALL_NAME)
    try:
        cs = wt.GetCompoundStructure()
        if cs is not None:
            cs.SetLayerWidth(0, SLAT_WIDTH * MM)
            mat = None
            for m in FilteredElementCollector(doc).OfClass(Material):
                if STONE_MATERIAL.lower() in ename(m).lower():
                    mat = m
                    break
            if mat is not None:
                cs.SetMaterialId(0, mat.Id)
            wt.SetCompoundStructure(cs)
            note("wall type '{0}' set to {1} mm{2}".format(
                WALL_NAME, SLAT_WIDTH,
                ", material " + ename(mat) if mat else ""))
    except Exception as ex:
        note("!! could not set wall thickness: {0}".format(ex))
    return wt


# --------------------------------------------------- 3. walls on the faces
def walls_on_faces(inst, wt, dirv):
    """One wall per slat: the largest planar face of each solid.

    Picking by area rather than by normal direction, because a leaning slat's
    face normal is tilted by the lean angle and any fixed tolerance on the
    normal eventually rejects the very slats we want.
    """
    opts = Options()
    opts.ComputeReferences = True
    opts.IncludeNonVisibleObjects = False
    opts.DetailLevel = ViewDetailLevel.Fine

    solids = []
    for g in inst.get_Geometry(opts):
        items = [g]
        if isinstance(g, GeometryInstance):
            items = list(g.GetInstanceGeometry())
        for it in items:
            if isinstance(it, Solid) and it.Faces.Size > 0 and it.Volume > 0:
                solids.append(it)
    note("mass geometry: {0} solids".format(len(solids)))
    if not solids:
        note("!! no solids found in the mass - cannot make walls by face")
        return 0, 0

    picked = []
    for sol in solids:
        best = None
        best_area = 0.0
        for f in sol.Faces:
            if not isinstance(f, PlanarFace):
                continue
            if f.Area > best_area:
                best_area = f.Area
                best = f
        if best is not None:
            picked.append(best)
    note("picked {0} faces (largest per solid, avg {1} m2)".format(
        len(picked),
        round(sum(f.Area for f in picked) / max(1, len(picked)) * 0.092903, 2)))

    # Creation.Document.NewWall(Face,...) was removed from modern Revit, so
    # discover what this build actually offers rather than assuming.
    def make_wall(face):
        ref = None
        try:
            ref = face.Reference
        except Exception:
            pass
        attempts = []
        if ref is not None:
            attempts.append(("Wall.Create(doc, ref, typeId, structural)",
                             lambda: Wall.Create(doc, ref, wt.Id, False)))
            attempts.append(("Wall.Create(doc, ref, typeId)",
                             lambda: Wall.Create(doc, ref, wt.Id)))
            attempts.append(("doc.Create.NewWall(ref, wt, False)",
                             lambda: doc.Create.NewWall(ref, wt, False)))
        attempts.append(("doc.Create.NewWall(face, wt, False)",
                         lambda: doc.Create.NewWall(face, wt, False)))
        last = None
        for label, fn in attempts:
            try:
                w = fn()
                if w is not None:
                    return w, label
            except Exception as ex:
                last = "{0} -> {1}".format(label, ex)
        raise Exception(last or "no wall-by-face call available")

    made = 0
    errs = 0
    used = None
    for f in picked:
        try:
            w, label = make_wall(f)
            if used is None:
                used = label
                note("wall by face via " + label)
            made += 1
        except Exception as ex:
            errs += 1
            if errs <= 2:
                note("!! wall by face failed: {0}".format(ex))

    if made == 0:
        note("Wall by Face is NOT in the Revit API - every Wall.Create "
             "overload takes a Curve, and Creation.Document.NewWall is gone. "
             "It exists only as an interactive tool, so the faces must be "
             "picked by hand.")

    if errs > 2:
        note("!! plus {0} more failures".format(errs - 2))
    return made, len(picked)


def main():
    a, b, host = get_run()
    if (b - a).GetLength() < 0.01:
        raise Exception("That run has no length.")

    h = SLAT_HEIGHT * MM
    if h <= 0:
        if host is not None:
            hp = host.get_Parameter(BuiltInParameter.WALL_USER_HEIGHT_PARAM)
            if hp:
                h = hp.AsDouble()
        if h <= 0:
            h = 3000.0 * MM

    level = doc.GetElement(host.LevelId) if host is not None else None
    if level is None:
        lv = sorted(FilteredElementCollector(doc).OfClass(Level),
                    key=lambda x: x.Elevation)
        if not lv:
            raise Exception("The project has no levels.")
        level = lv[0]

    note("run {0} mm, height {1} mm, level '{2}'".format(
        round((b - a).GetLength() * 304.8, 1), round(h * 304.8, 1),
        ename(level)))

    wall_count = 0
    # family work must happen with no transaction open on doc
    sym = build_mass(a, b, h, level.Elevation)

    t = Transaction(doc, "Slat mass + walls by face")
    t.Start()
    try:
        if not sym.IsActive:
            sym.Activate()
            doc.Regenerate()
        inst = doc.Create.NewFamilyInstance(
            XYZ(a.X, a.Y, level.Elevation), sym, level,
            StructuralType.NonStructural)
        doc.Regenerate()
        note("mass instance {0} placed".format(inst.Id))

        wt = slat_wall_type()
        made, total = walls_on_faces(inst, wt, (b - a).Normalize())
        wall_count = made
        note("created {0} of {1} walls by face".format(made, total))

        # the original wall becomes the thin-mullion glazing behind the slats
        if CONVERT_HOST and host is not None:
            was = ename(host.WallType)
            host.WallType = glazing_type()
            note("original wall retyped: '{0}' -> '{1}'".format(was, GLASS_TYPE))
            doc.Regenerate()
            if HORIZ_MULLION_AT:
                add_horizontal_gridlines(host, HORIZ_MULLION_AT)
        elif CONVERT_HOST:
            note("!! a line was picked, not a wall - nothing to convert")
        t.Commit()
    except Exception as ex:
        if t.HasStarted() and not t.HasEnded():
            t.RollBack()
        TaskDialog.Show("Failed - rolled back",
                        "{0}\n\n{1}".format(ex, "\n".join(log)))
        raise

    if POST_WALL_BY_FACE and wall_count == 0:
        try:
            cid = RevitCommandId.LookupPostableCommandId(
                PostableCommand.WallByFace)
            if cid is not None:
                __revit__.PostCommand(cid)
                note("Wall by Face tool will open when you close this dialog")
        except Exception as ex:
            note("could not launch Wall by Face: {0}".format(ex))

    TaskDialog.Show("Done", "\n".join(log) +
                    "\n\nWALL BY FACE cannot be scripted - the API has no "
                    "Face overload. The tool is now open: set the Type "
                    "Selector to 'Stone Slat 100mm', click each slat face "
                    "(or drag a box over them), then press Create.")


main()
