# -*- coding: utf-8 -*-
__title__ = u"AutoDimension 2"
__doc__ = u"""Auto-dimensioning for exterior and interior walls.
v9 - automatic exterior wall detection + one-click popup workflow.
Scratch copy of AutoDimension1 to sidestep __persistentengine__ code caching
while debugging -- delete once AutoDimension1 is confirmed fixed and merge
any further changes back into it.

Usage:
- Click AutoDimension 2. A popup offers:
    * Create dimensions now  -> everything is dimensioned in one run.
    * Place guide lines to adjust first -> guide lines are placed so you can
      move/copy them, then run AutoDimension 1 again to dimension along them.
- If guide lines already exist in the view, the popup is skipped and the
  dimensions are built straight away.

Exterior tiers (inner -> outer):
  1. openings (windows + doors)
  2. facade segments, merged across openings (breaks only at corners / jogs)
  3. the entire facade (one overall dimension)
Each tier references only its own (near) facade -- never across the building.
"""

# Keep the IronPython engine alive after main() returns so the modeless window
# and its Idling handler survive to run the Create Dimensions button.
__persistentengine__ = True

import math
import os
import clr

clr.AddReference("RevitAPI")
clr.AddReference("RevitAPIUI")

from Autodesk.Revit.DB import (
    FilteredElementCollector, BuiltInCategory,
    Dimension, DimensionType, Grid, FamilyInstance, Wall,
    XYZ, Line, Reference, ReferenceArray,
    ElementId, Transaction, TransactionGroup,
    Options, ViewPlan,
    PlanarFace, Solid, GeometryInstance,
    FailureProcessingResult, IFailuresPreprocessor,
    DatumEnds,
    HostObjectUtils, ShellLayerType,
    FamilyInstanceReferenceType,
    BuiltInParameter, CurveElement, SketchPlane, Plane,
    ModelLine, Category, GraphicsStyle, GraphicsStyleType,
    TextNote, TextNoteType, ElementTypeGroup,
    Color,
)
from Autodesk.Revit.UI import ExternalEvent, IExternalEventHandler
from Autodesk.Revit.UI.Selection import ObjectType
from pyrevit import forms, script, revit


class DimFailureSwallower(IFailuresPreprocessor):
    """Automatically deletes problematic dimensions instead of showing a dialog."""

    def __init__(self):
        self.had_errors = []

    def PreprocessFailures(self, failuresAccessor):
        failures = failuresAccessor.GetFailureMessages()
        for f in failures:
            try:
                sev = f.GetSeverity()
                desc = f.GetDescriptionText()
                if sev == sev.Error:
                    self.had_errors.append(desc)
                    ids = f.GetFailingElementIds()
                    if ids and ids.Count > 0:
                        failuresAccessor.DeleteElements(ids)
                    else:
                        failuresAccessor.ResolveFailure(f)
                elif sev == sev.Warning:
                    failuresAccessor.DeleteWarning(f)
            except Exception:
                try:
                    failuresAccessor.ResolveFailure(f)
                except Exception:
                    pass
        return FailureProcessingResult.Continue


# Use pyrevit.revit accessors (not __revit__) so this module is import-safe
# (top-level runs without a UI command context).
doc = revit.doc
uidoc = revit.uidoc
view = revit.active_view
output = script.get_output()

OFFSET_1_MM = 1000   # Tier 1: continuous openings/details chain
OFFSET_2_MM = 2000   # Tier 2: continuous wall-plane chain
OFFSET_3_MM = 3000   # Tier 3: overall side-to-side chain

OFFSET_CHAIN_1_MM = 1500
OFFSET_CHAIN_GAP_MM = 700

ZERO_TOL_MM = 5
INTERSECT_TOL_MM = 50
MAX_SNAP_DIST_MM = 10000

# Marker written to a guide line's Comments so the tool can recognize it later.
# Exterior format: "AUTODIM_GUIDE|<axis>|<tier>|<side>".
# Interior format: "AUTODIM_IGUIDE|<axis>".
GUIDE_MARK = u"AUTODIM_GUIDE"
IGUIDE_MARK = u"AUTODIM_IGUIDE"

# Also drop a text note at each exterior opening with its sill/head height
# above the level (elevation data a plan dimension cannot show).
ANNOTATE_OPENING_HEIGHTS = True

# How far inside the wall the UK=<sill> note sits, in mm. Raise it if the note
# clashes with what is drawn just inside the opening.
OPENING_NOTE_OFFSET_MM = 700

# Click the interior guides into place instead of seeding two at the building
# centre: left click drops one, right-click/Esc ends vertical placement and
# starts horizontal, a second cancel finishes. Set False for the old two-seed
# copy/move behaviour.
INTERACTIVE_INTERIOR_GUIDES = True

# Find exterior walls by flood-filling the building footprint instead of the
# old "is any wall further out?" overlap test, which dropped walls on steps,
# recesses and U-shaped notches. Set to False to fall back to the old test.
USE_FOOTPRINT_EXTERIOR_DETECTION = True

# Interior chains intentionally KEEP each wall's two faces, so the string reads
# room / wall thickness / room -- that thickness is wanted information here.
# (The exterior tiers still drop thickness slivers; that is a separate filter.)
FILTER_INTERIOR_THICKNESS = False

# Perp positions where the interior guides were auto-seeded (building center).
# A guide left untouched at its seed position is a template, not a request, so it
# is skipped when dimensioning. Drag-copying a seed leaves the original here and
# places the copy elsewhere -> only the copy gets dimensioned. Set on placement,
# read on dimensioning (same persistent engine).
_AD_INT_SEEDS = {"x": [], "y": []}

DEBUG = False


def mm_to_ft(mm):
    return mm / 304.8


def ft_to_mm(ft):
    return ft * 304.8


# --- Silent diagnostic log (writes to a file, never opens the output window) --
def _seg_log_path():
    try:
        return os.path.join(os.environ.get("USERPROFILE", ""), "dev",
                            "autodim2_seg.log")
    except Exception:
        return None


def _seg_log_reset():
    p = _seg_log_path()
    if not p:
        return
    try:
        f = open(p, "w")
        f.close()
    except Exception:
        pass


def _seg_log(msg):
    p = _seg_log_path()
    if not p:
        return
    try:
        f = open(p, "a")
        try:
            f.write(msg + "\n")
        finally:
            f.close()
    except Exception:
        pass


# Separate log for guide PLACEMENT (the dimensioning log reset would wipe it).
def _glog_path():
    try:
        return os.path.join(os.environ.get("USERPROFILE", ""), "dev",
                            "autodim2_guides.log")
    except Exception:
        return None


def _glog_reset():
    p = _glog_path()
    if not p:
        return
    try:
        open(p, "w").close()
    except Exception:
        pass


def _glog(msg):
    p = _glog_path()
    if not p:
        return
    try:
        f = open(p, "a")
        try:
            f.write(msg + "\n")
        finally:
            f.close()
    except Exception:
        pass


def collect_grids_from_selection(selected_elements):
    """Collects grids from selected elements. Determines bubble side."""
    grids = []
    for g in selected_elements:
        if not isinstance(g, Grid):
            continue
        try:
            crv = g.Curve
            if not isinstance(crv, Line):
                continue
            d = crv.Direction.Normalize()
            p0 = crv.GetEndPoint(0)
            p1 = crv.GetEndPoint(1)
            if abs(d.Y) < 0.1:
                orientation = "horizontal"
                coord = (p0.Y + p1.Y) / 2.0
            elif abs(d.X) < 0.1:
                orientation = "vertical"
                coord = (p0.X + p1.X) / 2.0
            else:
                continue

            bubble_end = _get_bubble_end(g)

            grids.append({
                "element": g, "name": g.Name,
                "orientation": orientation, "coord_ft": coord,
                "p0": p0, "p1": p1,
                "bubble_end": bubble_end,
            })
        except Exception:
            continue
    return grids


def _get_bubble_end(grid):
    """Determines which end of the grid has the bubble."""
    try:
        b0 = grid.IsBubbleVisibleInView(DatumEnds.End0, view)
        b1 = grid.IsBubbleVisibleInView(DatumEnds.End1, view)
        if b1 and not b0:
            return "p1"
        return "p0"
    except Exception as ex:
        return "p0"


def collect_elements_from_selection(selected_elements):
    """Collects walls and columns from selected elements."""
    elements = []
    seen_ids = set()
    wall_cat_id = ElementId(BuiltInCategory.OST_Walls)
    str_col_cat_id = ElementId(BuiltInCategory.OST_StructuralColumns)
    col_cat_id = ElementId(BuiltInCategory.OST_Columns)

    for e in selected_elements:
        if isinstance(e, Grid):
            continue
        eid = e.Id.IntegerValue
        if eid in seen_ids:
            continue
        seen_ids.add(eid)

        try:
            cat_id = e.Category.Id if e.Category else None
        except Exception:
            continue

        if cat_id == wall_cat_id:
            info = _bbox(e, "Wall")
            if info:
                elements.append(info)
        elif cat_id == str_col_cat_id or cat_id == col_cat_id:
            if isinstance(e, FamilyInstance) and e.SuperComponent is not None:
                continue
            info = _bbox(e, "Column")
            if info:
                elements.append(info)
    return elements


def collect_walls_in_view():
    """Collect all visible walls for the optional interior dimension pass."""
    walls = []
    try:
        visible_walls = FilteredElementCollector(doc, view.Id).OfClass(Wall).ToElements()
    except Exception:
        visible_walls = []

    for wall in visible_walls:
        info = _bbox(wall, "Wall")
        if info:
            walls.append(info)
    return walls


def _bbox(elem, cat):
    try:
        bb = elem.get_BoundingBox(view) or elem.get_BoundingBox(None)
        if not bb:
            return None
        w = abs(bb.Max.X - bb.Min.X)
        d = abs(bb.Max.Y - bb.Min.Y)
        if ft_to_mm(w) < 50 or ft_to_mm(d) < 50:
            return None
        return {
            "element": elem, "category": cat,
            "min_x": bb.Min.X, "max_x": bb.Max.X,
            "min_y": bb.Min.Y, "max_y": bb.Max.Y,
            "cx": (bb.Min.X + bb.Max.X) / 2.0,
            "cy": (bb.Min.Y + bb.Max.Y) / 2.0,
            "w_ft": w, "d_ft": d,
        }
    except Exception:
        return None


def get_faces(elem, axis):
    """Gets face references of an element."""
    is_family = isinstance(elem, FamilyInstance)

    # --- Walls: use HostObjectUtils for exact exterior/interior face refs ---
    if isinstance(elem, Wall):
        try:
            # wall.Orientation is the exterior face normal.
            # Only dimension walls whose faces are perpendicular to this axis —
            # same filter the old geometry code applied via abs(n.X) > 0.9.
            orientation = elem.Orientation
            if axis == "x" and abs(orientation.X) < 0.7:
                return None, None, None, None  # wall runs E-W, irrelevant for X axis
            if axis == "y" and abs(orientation.Y) < 0.7:
                return None, None, None, None  # wall runs N-S, irrelevant for Y axis

            ext_refs = list(HostObjectUtils.GetSideFaces(elem, ShellLayerType.Exterior))
            int_refs = list(HostObjectUtils.GetSideFaces(elem, ShellLayerType.Interior))
            if not ext_refs or not int_refs:
                if DEBUG:
                    output.print_md(u"   ⚠ HostObjectUtils returned no refs (id={})".format(
                        elem.Id.IntegerValue))
                return None, None, None, None

            ext_ref = ext_refs[0]
            int_ref = int_refs[0]

            # use bbox for coordinates — reliable for axis-aligned walls,
            # no geometry traversal needed
            bb = elem.get_BoundingBox(view) or elem.get_BoundingBox(None)
            if bb is None:
                return None, None, None, None

            if axis == "x":
                lo_c, hi_c = bb.Min.X, bb.Max.X
                # orientation.X > 0 means exterior is on the +X (hi) side
                if orientation.X > 0:
                    lo_ref, hi_ref = int_ref, ext_ref
                else:
                    lo_ref, hi_ref = ext_ref, int_ref
            else:
                lo_c, hi_c = bb.Min.Y, bb.Max.Y
                if orientation.Y > 0:
                    lo_ref, hi_ref = int_ref, ext_ref
                else:
                    lo_ref, hi_ref = ext_ref, int_ref

            if DEBUG:
                output.print_md(u"   🔍 get_faces wall axis={}: lo={:.0f}mm hi={:.0f}mm (id={})".format(
                    axis, ft_to_mm(lo_c), ft_to_mm(hi_c), elem.Id.IntegerValue))
            return lo_ref, hi_ref, lo_c, hi_c

        except Exception as ex:
            if DEBUG:
                output.print_md(u"   ⚠ HostObjectUtils failed: {} (id={})".format(
                    str(ex), elem.Id.IntegerValue))
            return None, None, None, None

    # --- Family instances (columns): use GetInstanceGeometry, no string hacking ---
    # opt.View omitted intentionally — plan-cut geometry returns null References
    opt = Options()
    opt.ComputeReferences = True
    opt.IncludeNonVisibleObjects = False
    geo = elem.get_Geometry(opt)
    if not geo:
        if DEBUG:
            output.print_md(u"   ⚠ get_faces: no geometry (id={})".format(elem.Id.IntegerValue))
        return None, None, None, None

    faces = []

    for item in geo:
        try:
            if isinstance(item, GeometryInstance) and is_family:
                # GetInstanceGeometry() returns world-space geometry with
                # instance-scoped references — no stable-string hacking needed
                inst_geo = item.GetInstanceGeometry()
                if not inst_geo:
                    continue
                for inst_item in inst_geo:
                    if not isinstance(inst_item, Solid) or inst_item.Faces.Size == 0:
                        continue
                    for face in inst_item.Faces:
                        if not isinstance(face, PlanarFace):
                            continue
                        ref = face.Reference
                        if ref is None:
                            continue
                        n = face.FaceNormal
                        if abs(n.Z) > 0.9:
                            continue
                        if axis == "x" and abs(n.X) > 0.9:
                            faces.append((ref, face.Origin.X))
                        elif axis == "y" and abs(n.Y) > 0.9:
                            faces.append((ref, face.Origin.Y))

            elif isinstance(item, Solid) and item.Faces.Size > 0:
                for face in item.Faces:
                    if not isinstance(face, PlanarFace):
                        continue
                    ref = face.Reference
                    if ref is None:
                        continue
                    n = face.FaceNormal
                    if axis == "x" and abs(n.X) > 0.9:
                        faces.append((ref, face.Origin.X))
                    elif axis == "y" and abs(n.Y) > 0.9:
                        faces.append((ref, face.Origin.Y))
        except Exception as ex:
            if DEBUG:
                output.print_md(u"   ⚠ scan error: {}".format(str(ex)))
            continue

    if DEBUG:
        output.print_md(u"   🔍 get_faces axis={}: {} faces, family={} (id={})".format(
            axis, len(faces), is_family, elem.Id.IntegerValue))

    if len(faces) < 2:
        if DEBUG:
            _dump_normals(geo, is_family)
        return None, None, None, None

    faces.sort(key=lambda x: x[1])
    if DEBUG:
        output.print_md(u"   📏 lo={:.0f}mm, hi={:.0f}mm".format(
            ft_to_mm(faces[0][1]), ft_to_mm(faces[-1][1])))
    return faces[0][0], faces[-1][0], faces[0][1], faces[-1][1]




def _dump_normals(geo, is_family):
    """Dumps all normals for debugging when issues occur."""
    all_n = []
    for item in geo:
        try:
            if isinstance(item, GeometryInstance) and is_family:
                xf = item.Transform
                for si in item.GetSymbolGeometry():
                    if isinstance(si, Solid):
                        for f in si.Faces:
                            if isinstance(f, PlanarFace):
                                wn = xf.OfVector(f.FaceNormal)
                                all_n.append(u"({:.2f},{:.2f},{:.2f})".format(wn.X, wn.Y, wn.Z))
            elif isinstance(item, Solid):
                for f in item.Faces:
                    if isinstance(f, PlanarFace):
                        all_n.append(u"({:.2f},{:.2f},{:.2f})".format(
                            f.FaceNormal.X, f.FaceNormal.Y, f.FaceNormal.Z))
        except Exception:
            pass
    output.print_md(u"   🧊 Normals: {}".format(u", ".join(all_n[:12])))


def get_grid_ref(grid):
    try:
        opt = Options()
        opt.ComputeReferences = True
        opt.IncludeNonVisibleObjects = True
        opt.View = view
        geo = grid.get_Geometry(opt)
        if geo:
            for item in geo:
                if isinstance(item, Line) and item.Reference:
                    return item.Reference
        crv = grid.Curve
        if crv and crv.Reference:
            return crv.Reference
    except Exception:
        pass
    return None


def make_dim(refs, p0, p1, label=""):
    if len(refs) < 2:
        return None
    ra = ReferenceArray()
    for r in refs:
        ra.Append(r)
    try:
        ln = Line.CreateBound(p0, p1)
        if DEBUG:
            d = ln.Direction.Normalize()
            output.print_md(
                u"   📐 make_dim [{}]: {} refs, line ({:.0f},{:.0f})->({:.0f},{:.0f}), dir=({:.2f},{:.2f})".format(
                    label, len(refs),
                    ft_to_mm(p0.X), ft_to_mm(p0.Y),
                    ft_to_mm(p1.X), ft_to_mm(p1.Y),
                    d.X, d.Y))
        dim = doc.Create.NewDimension(view, ln, ra)
        # Tag with the AUTODIM dimension type so a later run can find and delete
        # its own dimensions (accumulation cleanup).
        if dim is not None and _AD_DIM_TYPE_ID:
            try:
                dim.ChangeTypeId(_AD_DIM_TYPE_ID)
            except Exception:
                pass
        if DEBUG and dim:
            output.print_md(u"   ✅ Dimension created (id={})".format(dim.Id.IntegerValue))
        return dim
    except Exception as e:
        if DEBUG:
            output.print_md(u"   ❌ make_dim ERROR [{}]: **{}**".format(label, str(e)))
        return None


def _displace_small_texts(dim):
    """Move the labels of segments too narrow to hold their own text.

    Pushing them all the same way along the dimension line only works for a
    lone narrow segment: a RUN of them (thin walls, a jamb next to a jog) all
    shift by the same amount and land on top of each other. So offset
    PERPENDICULAR to the line instead, alternating side and stepping a row
    further out every two, which keeps each label over its own segment and lets
    Revit draw the leader. Row height and the width threshold both scale with
    the view, so this holds at any plot scale."""
    try:
        scale = view.Scale
    except Exception:
        scale = 100

    # Roughly the model-space width/height of one line of dimension text.
    text_width_mm = 5.0 * scale
    row_mm = 3.5 * scale

    try:
        crv = dim.Curve
        if not crv or not isinstance(crv, Line):
            return
        direction = crv.Direction.Normalize()
    except Exception:
        return
    # In-plane normal of the dimension line.
    perp = XYZ(-direction.Y, direction.X, 0.0)

    def _shift(target, run_index):
        """Offset one label off the line; returns True if it was moved."""
        try:
            if not target.IsTextPositionAdjustable():
                return False
            tp = target.TextPosition
            if tp is None:
                return False
            side = 1.0 if (run_index % 2 == 0) else -1.0
            level = (run_index // 2) + 1
            off = mm_to_ft(row_mm * level) * side
            target.TextPosition = XYZ(tp.X + perp.X * off,
                                      tp.Y + perp.Y * off,
                                      tp.Z)
            return True
        except Exception:
            return False

    try:
        segs = list(dim.Segments)
        if segs and len(segs) > 0:
            run = 0
            for seg in segs:
                try:
                    val = seg.Value
                    if val is None:
                        run = 0
                        continue
                    if ft_to_mm(val) >= text_width_mm:
                        run = 0          # wide enough: breaks the crowded run
                        continue
                    if _shift(seg, run):
                        run += 1
                except Exception:
                    continue
            return
    except Exception:
        pass

    try:
        val = dim.Value
        if val is None:
            return
        if ft_to_mm(val) >= text_width_mm:
            return
        _shift(dim, 0)
    except Exception:
        pass


def _opening_half_width(elem):
    """Returns half the opening width in feet."""
    for name in ("Width", "Rough Width", "Ширина"):
        try:
            p = elem.LookupParameter(name)
            if p and p.AsDouble() > 0:
                return p.AsDouble() / 2.0
        except Exception:
            pass
    return mm_to_ft(450)


def _get_opening_refs_geometry(elem, run_axis):
    """
    Geometry fallback: returns face refs from the opening whose normals
    are aligned with run_axis. No opt.View so references are stable.
    """
    opt = Options()
    opt.ComputeReferences = True
    geo = elem.get_Geometry(opt)
    if not geo:
        return []
    refs = []
    for item in geo:
        if not isinstance(item, GeometryInstance):
            continue
        inst_geo = item.GetInstanceGeometry()
        if not inst_geo:
            continue
        for inst_item in inst_geo:
            if not isinstance(inst_item, Solid) or inst_item.Faces.Size == 0:
                continue
            for face in inst_item.Faces:
                if not isinstance(face, PlanarFace):
                    continue
                ref = face.Reference
                if ref is None:
                    continue
                n = face.FaceNormal
                if abs(n.Z) > 0.9:
                    continue
                if run_axis == "y" and abs(n.Y) > 0.7:
                    refs.append((ref, face.Origin.Y))
                elif run_axis == "x" and abs(n.X) > 0.7:
                    refs.append((ref, face.Origin.X))
    if len(refs) >= 2:
        refs.sort(key=lambda x: x[1])
        return [refs[0], refs[-1]]
    return refs


_HOSTED_OPENINGS_CACHE = None


def _reset_hosted_openings_cache():
    global _HOSTED_OPENINGS_CACHE
    _HOSTED_OPENINGS_CACHE = None


def _hosted_openings_map():
    """Wall element id -> list of hosted door/window FamilyInstances.

    Built from each opening's own ``Host`` property rather than
    ``wall.GetDependentElements()``. The Revit API docs call
    GetDependentElements best-effort/incomplete, and in practice it misses
    openings hosted on walls that live inside a Group (e.g. a repeated
    apartment unit) -- Host still resolves correctly there.
    """
    global _HOSTED_OPENINGS_CACHE
    if _HOSTED_OPENINGS_CACHE is not None:
        return _HOSTED_OPENINGS_CACHE
    result = {}
    for cat in (BuiltInCategory.OST_Doors, BuiltInCategory.OST_Windows):
        try:
            insts = FilteredElementCollector(doc, view.Id).OfCategory(cat) \
                .WhereElementIsNotElementType().ToElements()
        except Exception:
            continue
        for elem in insts:
            if not isinstance(elem, FamilyInstance):
                continue
            try:
                host = elem.Host
            except Exception:
                host = None
            if host is None:
                continue
            result.setdefault(host.Id.IntegerValue, []).append(elem)
    _HOSTED_OPENINGS_CACHE = result
    return result


def _collect_opening_face_refs(wall, run_axis):
    """
    Returns (ref, coord) for Left/Right jamb references of every hosted
    door/window.  Primary: FamilyInstanceReferenceType.Left/Right.
    Fallback: geometry traversal without opt.View.
    """
    results = []
    dep_elems = _hosted_openings_map().get(wall.Id.IntegerValue, [])

    for elem in dep_elems:
        try:
            loc = elem.Location
            pt = loc.Point if hasattr(loc, "Point") else loc.Curve.Evaluate(0.5, True)
        except Exception:
            continue

        half_w = _opening_half_width(elem)
        pair = []  # [(ref, coord), ...]

        # Primary: FamilyInstanceReferenceType Left/Right
        for ref_type, sign in ((FamilyInstanceReferenceType.Left, -1.0),
                                (FamilyInstanceReferenceType.Right, 1.0)):
            try:
                refs_list = list(elem.GetReferences(ref_type))
                if refs_list:
                    coord = (pt.Y if run_axis == "y" else pt.X) + sign * half_w
                    pair.append((refs_list[0], coord))
            except Exception:
                pass

        # Fallback: geometry traversal
        if len(pair) < 2:
            geo_refs = _get_opening_refs_geometry(elem, run_axis)
            pair = geo_refs  # already (ref, coord) tuples

        for item in pair:
            results.append(item)

    if DEBUG and results:
        output.print_md(u"   🚪 openings on wall {}: {} refs".format(
            wall.Id.IntegerValue, len(results)))
    return results


def _get_wall_end_edge_refs(wall, orient, run_axis):
    """
    Returns (ref, coord) for vertical edge references on the exterior face of wall.
    opt.View is intentionally omitted — plan-cut geometry returns null References
    on end-cap faces; element-level geometry gives stable references.
    """
    opt = Options()
    opt.ComputeReferences = True
    # no opt.View — critical: view-scoped geometry breaks end-cap face References
    geo = wall.get_Geometry(opt)
    if not geo:
        return []

    edges = []
    for item in geo:
        if not isinstance(item, Solid) or item.Faces.Size == 0:
            continue
        for face in item.Faces:
            if not isinstance(face, PlanarFace):
                continue
            n = face.FaceNormal
            if n.DotProduct(orient) < 0.85:
                continue  # only the exterior face
            try:
                loops = list(face.EdgeLoops)
            except Exception:
                loops = []
            # The exterior face has an outer boundary loop (the wall's real ends)
            # plus one inner loop per window/door hole. The hole's jamb edges
            # would duplicate the opening references (collected separately) and
            # produce a doubled dimension at each opening -- "one for the wall and
            # one for the window". Keep only the outer loop (widest span along the
            # run axis); inner loops are always narrower.
            outer = None
            outer_span = -1.0
            for loop in loops:
                lo = hi = None
                for edge in loop:
                    c = edge.AsCurve()
                    if c is None:
                        continue
                    for k in (0, 1):
                        p = c.GetEndPoint(k)
                        v = p.Y if run_axis == "y" else p.X
                        if lo is None or v < lo:
                            lo = v
                        if hi is None or v > hi:
                            hi = v
                if lo is not None and (hi - lo) > outer_span:
                    outer_span = hi - lo
                    outer = loop
            if outer is None:
                continue
            try:
                for edge in outer:
                    curve = edge.AsCurve()
                    if curve is None:
                        continue
                    ref = edge.Reference
                    if ref is None:
                        continue
                    ep0 = curve.GetEndPoint(0)
                    ep1 = curve.GetEndPoint(1)
                    d = (ep1 - ep0).Normalize()
                    if abs(d.Z) < 0.7:
                        continue  # skip horizontal (top/bottom) edges
                    coord = ep0.Y if run_axis == "y" else ep0.X
                    edges.append((ref, coord))
            except Exception:
                pass
    return edges


def _opening_sill_head_mm(elem):
    """(sill_mm, head_mm) above the level for a hosted window/door, or None."""
    try:
        ps = elem.get_Parameter(BuiltInParameter.INSTANCE_SILL_HEIGHT_PARAM)
        ph = elem.get_Parameter(BuiltInParameter.INSTANCE_HEAD_HEIGHT_PARAM)
        sill = ps.AsDouble() if ps else None
        head = ph.AsDouble() if ph else None
        if sill is None and head is None:
            return None
        s_mm = int(round(ft_to_mm(sill))) if sill is not None else None
        h_mm = int(round(ft_to_mm(head))) if head is not None else None
        return (s_mm, h_mm)
    except Exception:
        return None


def _opening_note_type_id():
    """A small text type for opening-height notes (2 mm paper), duplicated from
    the default once and reused. Falls back to the default type."""
    default_id = None
    try:
        default_id = doc.GetDefaultElementTypeId(ElementTypeGroup.TextNoteType)
    except Exception:
        default_id = None
    if default_id is None or default_id == ElementId.InvalidElementId:
        return None
    name = u"AUTODIM_OPENING_2mm"
    try:
        for t in FilteredElementCollector(doc).OfClass(TextNoteType):
            try:
                if t.Name == name:
                    return t.Id
            except Exception:
                continue
        base = doc.GetElement(default_id)
        dup = base.Duplicate(name)
        p = dup.get_Parameter(BuiltInParameter.TEXT_SIZE)
        if p is not None and not p.IsReadOnly:
            p.Set(mm_to_ft(2.0))
        return dup.Id
    except Exception:
        return default_id


# The dimension type used for every AutoDimension dimension, so a later run can
# find and delete its own dimensions without touching the user's or grid dims.
AUTODIM_DIM_TYPE_NAME = u"AUTODIM"
_AD_DIM_TYPE_ID = None


def _autodim_dim_type_id():
    """Id of the AUTODIM dimension type, duplicated once from the current default
    linear dimension type so its appearance matches. Falls back to None."""
    try:
        for t in FilteredElementCollector(doc).OfClass(DimensionType):
            try:
                if t.Name == AUTODIM_DIM_TYPE_NAME:
                    return t.Id
            except Exception:
                continue
    except Exception:
        pass
    base = None
    try:
        bid = doc.GetDefaultElementTypeId(ElementTypeGroup.LinearDimensionType)
        if bid and bid != ElementId.InvalidElementId:
            base = doc.GetElement(bid)
    except Exception:
        base = None
    if base is None:
        try:
            for t in FilteredElementCollector(doc).OfClass(DimensionType):
                base = t
                break
        except Exception:
            base = None
    if base is None:
        return None
    try:
        return base.Duplicate(AUTODIM_DIM_TYPE_NAME).Id
    except Exception:
        return None


def _dim_references_grid(d):
    """True if the dimension measures at least one grid (a grid dimension)."""
    try:
        refs = d.References
        if refs is None:
            return False
        for r in refs:
            try:
                el = doc.GetElement(r.ElementId)
            except Exception:
                el = None
            if isinstance(el, Grid):
                return True
    except Exception:
        pass
    return False


def _delete_previous_autodim():
    """Remove wall/opening dimensions and opening-height notes so successive runs
    do not stack overlapping strings. A dimension is removed when it is our
    AUTODIM type OR when it references no grid (i.e. it measures walls/openings --
    the kind this tool makes, including untagged ones from older runs). Grid
    dimensions are always kept, and model lines are left untouched."""
    try:
        for d in list(FilteredElementCollector(doc, view.Id)
                      .OfClass(Dimension).ToElements()):
            try:
                is_autodim = False
                try:
                    is_autodim = d.DimensionType.Name.startswith(
                        AUTODIM_DIM_TYPE_NAME)
                except Exception:
                    is_autodim = False
                if is_autodim or not _dim_references_grid(d):
                    doc.Delete(d.Id)
            except Exception:
                continue
    except Exception:
        pass
    try:
        for tn in list(FilteredElementCollector(doc, view.Id)
                       .OfClass(TextNote).ToElements()):
            try:
                tt = doc.GetElement(tn.GetTypeId())
                if tt is not None and tt.Name.startswith(u"AUTODIM"):
                    doc.Delete(tn.Id)
            except Exception:
                continue
    except Exception:
        pass


def annotate_opening_heights(all_elems, ext_wall_ids):
    """Place a small text note just inside each exterior opening showing its
    sill/head height above the level -- elevation data a plan dimension cannot
    express. The note is offset off the wall toward the interior so it does not
    sit on the wall or the dimension tiers."""
    type_id = _opening_note_type_id()
    if type_id is None:
        return 0

    ext_elems = [ei for ei in all_elems
                 if ei["element"].Id.IntegerValue in ext_wall_ids]
    if not ext_elems:
        return 0
    bcx = (min(e["min_x"] for e in ext_elems) + max(e["max_x"] for e in ext_elems)) / 2.0
    bcy = (min(e["min_y"] for e in ext_elems) + max(e["max_y"] for e in ext_elems)) / 2.0

    # Just clear of the wall, toward the interior. The exterior tiers sit on the
    # OTHER side of the wall, so this never has to clear them -- what used to
    # drop the note onto a dimension string was the offset SIGN, handled below.
    inward = mm_to_ft(OPENING_NOTE_OFFSET_MM)
    created = 0
    seen = set()
    for ei in all_elems:
        wall = ei["element"]
        if ei["element"].Id.IntegerValue not in ext_wall_ids:
            continue
        if not isinstance(wall, Wall):
            continue
        try:
            orient = wall.Orientation  # gives the axis, not a trustworthy sign
        except Exception:
            orient = None
        for elem in _hosted_openings_map().get(wall.Id.IntegerValue, []):
            key = elem.Id.IntegerValue
            if key in seen:
                continue
            seen.add(key)
            sh = _opening_sill_head_mm(elem)
            if sh is None:
                continue
            try:
                loc = elem.Location
                pt = loc.Point if hasattr(loc, "Point") \
                    else loc.Curve.Evaluate(0.5, True)
            except Exception:
                continue
            # Nudge the note off the wall, toward the building interior. The
            # axis comes from wall.Orientation, but the SIGN is derived from
            # the wall's position relative to the building's overall center --
            # Wall.Orientation's sign is unreliable for walls that live inside
            # a mirrored Group (e.g. a repeated apartment unit).
            if orient is not None:
                sign_x = 1.0 if ei["cx"] >= bcx else -1.0
                sign_y = 1.0 if ei["cy"] >= bcy else -1.0
                ox = abs(orient.X) * sign_x
                oy = abs(orient.Y) * sign_y
                pt = XYZ(pt.X - ox * inward, pt.Y - oy * inward, pt.Z)
            s_mm, h_mm = sh
            if s_mm is None:
                continue
            txt = u"UK={}".format(s_mm)
            try:
                TextNote.Create(doc, view.Id, pt, txt, type_id)
                created += 1
            except Exception:
                continue
    if DEBUG:
        output.print_md(u"🔺 opening-height notes: {}".format(created))
    return created


def dim_wall_with_openings(ei, dims_to_adjust):
    """
    Creates a dimension line along the wall showing hosted door/window positions.
    Placed on the exterior side of the wall, offset by OFFSET_1_MM.
    """
    wall = ei["element"]
    if not isinstance(wall, Wall):
        return 0

    try:
        orient = wall.Orientation
    except Exception:
        return 0

    if abs(orient.X) > 0.7:
        run_axis = "y"
    elif abs(orient.Y) > 0.7:
        run_axis = "x"
    else:
        return 0

    opening_refs = _collect_opening_face_refs(wall, run_axis)
    if not opening_refs:
        return 0  # no openings on this wall

    end_edge_refs = _get_wall_end_edge_refs(wall, orient, run_axis)

    combined = opening_refs + end_edge_refs
    combined.sort(key=lambda x: x[1])

    tol = mm_to_ft(1)
    deduped = [combined[0]]
    for ref, coord in combined[1:]:
        if abs(coord - deduped[-1][1]) > tol:
            deduped.append((ref, coord))

    if len(deduped) < 2:
        return 0

    refs   = [r for r, _ in deduped]
    c_lo   = deduped[0][1]
    c_hi   = deduped[-1][1]
    off    = mm_to_ft(OFFSET_1_MM)

    if run_axis == "y":
        perp = (ei["max_x"] + off) if orient.X > 0 else (ei["min_x"] - off)
        p0 = XYZ(perp, c_lo, 0)
        p1 = XYZ(perp, c_hi, 0)
    else:
        perp = (ei["max_y"] + off) if orient.Y > 0 else (ei["min_y"] - off)
        p0 = XYZ(c_lo, perp, 0)
        p1 = XYZ(c_hi, perp, 0)

    if DEBUG:
        output.print_md(u"🚪 along-wall {}: {} refs, run={}, perp={:.0f}mm".format(
            wall.Id.IntegerValue, len(deduped), run_axis, ft_to_mm(perp)))

    dim = make_dim(refs, p0, p1, "along-wall-{}".format(wall.Id.IntegerValue))
    if dim:
        dims_to_adjust.append(dim)
        return 1
    return 0


def _pick_reference_points(prompt):
    """Pick multiple points; Revit Finish ends this selection phase."""
    try:
        point_refs = uidoc.Selection.PickObjects(
            ObjectType.PointOnElement,
            prompt
        )
        points = []
        for point_ref in point_refs:
            point = point_ref.GlobalPoint
            if point is not None:
                points.append(point)
        return points
    except Exception as ex:
        # Cancel/Escape cancels the current phase. Finish returns the
        # selected references from PickObjects. Do not hide a real API error.
        try:
            output.print_md(u"⚠ Point selection cancelled or failed: {}".format(
                str(ex)))
        except Exception:
            pass
        return []


def _interior_dimension_at_point(point, axis, wall_infos, dims_to_adjust, index):
    """Dimension every wall face crossed by one user-selected reference line."""
    face_pairs = []
    crossed = []
    tolerance = mm_to_ft(INTERSECT_TOL_MM)

    for ei in wall_infos:
        wall = ei["element"]
        try:
            orient = wall.Orientation
            if axis == "x":
                # A horizontal reference line crosses walls running north/south.
                if abs(orient.X) < 0.7:
                    continue
                if point.Y < ei["min_y"] - tolerance or point.Y > ei["max_y"] + tolerance:
                    continue
            else:
                # A vertical reference line crosses walls running east/west.
                if abs(orient.Y) < 0.7:
                    continue
                if point.X < ei["min_x"] - tolerance or point.X > ei["max_x"] + tolerance:
                    continue

            ref_lo, ref_hi, c_lo, c_hi = get_faces(wall, axis)
            if ref_lo is None or ref_hi is None:
                continue
            face_pairs.append((ref_lo, c_lo))
            face_pairs.append((ref_hi, c_hi))
            crossed.append(ei)
        except Exception:
            continue

    if len(face_pairs) < 2:
        if DEBUG:
            output.print_md(u"   ⏭ Interior {} point: fewer than two wall faces".format(axis))
        return 0

    face_pairs.sort(key=lambda item: item[1])
    deduped = []
    # Merge near-coincident faces (e.g. two walls meeting at a T-junction) so the
    # interior chain never emits a 0-length (or micro) segment. Kept below the
    # thinnest real wall (~125 mm) so a wall's own two faces are never merged.
    dedup_tolerance = mm_to_ft(50)
    for ref, coord in face_pairs:
        if not deduped or abs(coord - deduped[-1][1]) > dedup_tolerance:
            deduped.append((ref, coord))

    if len(deduped) < 2:
        return 0

    # A wall's own two faces sit exactly one thickness apart, so leaving both in
    # the chain reports wall thicknesses as if they were rooms. Collapse them
    # with the same filter the exterior tiers use.
    n_before = len(deduped)
    if FILTER_INTERIOR_THICKNESS and crossed:
        ttol = _max_wall_thickness(crossed) * 1.1
        deduped = _drop_thickness_refs(deduped, ttol)
        if len(deduped) < 2:
            return 0

    refs = [ref for ref, unused_coord in deduped]
    low = deduped[0][1]
    high = deduped[-1][1]
    _seg_log(u"  int dim axis={} at_perp_mm={} faces={}->{} low_mm={} high_mm={}".format(
        axis, int(round(ft_to_mm(point.X if axis == "y" else point.Y))),
        n_before, len(deduped), int(round(ft_to_mm(low))),
        int(round(ft_to_mm(high)))))
    if axis == "x":
        p0 = XYZ(low, point.Y, 0)
        p1 = XYZ(high, point.Y, 0)
    else:
        p0 = XYZ(point.X, low, 0)
        p1 = XYZ(point.X, high, 0)

    dim = make_dim(refs, p0, p1, "interior-{}-{}".format(axis, index))
    if dim:
        dims_to_adjust.append(dim)
        return 1
    return 0


def make_interior_dimensions(horizontal_points, vertical_points, wall_infos, dims_to_adjust):
    """Create one dimension chain for every selected horizontal/vertical point."""
    created = 0
    for index, point in enumerate(horizontal_points):
        created += _interior_dimension_at_point(point, "x", wall_infos, dims_to_adjust, index + 1)
    for index, point in enumerate(vertical_points):
        created += _interior_dimension_at_point(point, "y", wall_infos, dims_to_adjust, index + 1)
    return created


def create_interior_phase(points, axis, wall_infos, failure_handler):
    """Create and commit one interior direction immediately after Enter."""
    dims_to_adjust = []
    transaction = Transaction(doc, u"Interior {} dimensions".format(axis))
    options = transaction.GetFailureHandlingOptions()
    options.SetFailuresPreprocessor(failure_handler)
    transaction.SetFailureHandlingOptions(options)
    transaction.Start()
    try:
        created = make_interior_dimensions(
            points if axis == "x" else [],
            points if axis == "y" else [],
            wall_infos,
            dims_to_adjust,
        )
        doc.Regenerate()
        for dim in dims_to_adjust:
            _displace_small_texts(dim)
        transaction.Commit()
        return created
    except Exception:
        transaction.RollBack()
        raise


def _find_exterior_face_refs(all_elems, axis):
    """
    Returns (ref, coord) pairs for faces that are on the exterior building
    envelope along the given axis.

    A face is exterior if no other wall extends further outward (in that axis
    direction) while sharing at least MIN_OVERLAP of perpendicular extent.
    This filters out core/interior walls that are sandwiched between outer walls.
    """
    MIN_OVERLAP = mm_to_ft(300)

    walls = []
    for ei in all_elems:
        if not isinstance(ei["element"], Wall):
            continue
        try:
            orient = ei["element"].Orientation
            if axis == "x" and abs(orient.X) < 0.7:
                continue
            if axis == "y" and abs(orient.Y) < 0.7:
                continue
        except Exception:
            continue

        ref_lo, ref_hi, c_lo, c_hi = get_faces(ei["element"], axis)
        if ref_lo is None:
            continue

        if axis == "x":
            perp_lo, perp_hi = ei["min_y"], ei["max_y"]
        else:
            perp_lo, perp_hi = ei["min_x"], ei["max_x"]

        walls.append({
            "ref_lo": ref_lo, "ref_hi": ref_hi,
            "c_lo": c_lo, "c_hi": c_hi,
            "center": (c_lo + c_hi) / 2.0,
            "perp_lo": perp_lo, "perp_hi": perp_hi,
        })

    faces = []
    for w in walls:
        lo_blocked = any(
            o is not w
            and o["center"] < w["center"]
            and min(w["perp_hi"], o["perp_hi"]) - max(w["perp_lo"], o["perp_lo"]) >= MIN_OVERLAP
            for o in walls
        )
        hi_blocked = any(
            o is not w
            and o["center"] > w["center"]
            and min(w["perp_hi"], o["perp_hi"]) - max(w["perp_lo"], o["perp_lo"]) >= MIN_OVERLAP
            for o in walls
        )
        if not lo_blocked:
            faces.append((w["ref_lo"], w["c_lo"]))
        if not hi_blocked:
            faces.append((w["ref_hi"], w["c_hi"]))

    if DEBUG:
        output.print_md(u"🔍 exterior-detect axis={}: {}/{} wall faces kept".format(
            axis, len(faces), len(walls) * 2))
    return faces


def _wall_thickness_ft(elem):
    """Wall thickness in feet, or None."""
    try:
        w = elem.Width
        if w and w > 0:
            return w
    except Exception:
        pass
    try:
        w = elem.WallType.Width
        if w and w > 0:
            return w
    except Exception:
        pass
    return None


def _compute_exterior_wall_ids_footprint(all_elems):
    """Exterior walls = the walls on the boundary of the building FOOTPRINT.

    The old test ("is any wall further out sharing perpendicular overlap?")
    only holds for a rectangular box. On an L-shape, a step or a U-shaped
    notch, genuinely exterior walls sit between other walls and were dropped,
    so their openings never reached tier 1 and their facade never broke tier 2.

    Instead, rasterize the plan, flood-fill the empty space inward from
    outside the bounding box, and keep every wall that touches space reachable
    from outside. That is a footprint-boundary test, so it handles any
    footprint shape. The grid step is derived from the thinnest wall present
    (never a hardcoded distance) so no wall can be thinner than a cell and
    leak the fill through itself.

    Returns None when the model is too degenerate to rasterize, so the caller
    can fall back to the old heuristic.
    """
    walls = []
    for ei in all_elems:
        if not isinstance(ei["element"], Wall):
            continue
        walls.append(ei)
    if not walls:
        return None

    # Cell size: half the thinnest wall, so every wall spans >= 2 cells and
    # stays watertight against the flood fill.
    thin = None
    for ei in walls:
        t = _wall_thickness_ft(ei["element"])
        if t is None:
            continue
        if thin is None or t < thin:
            thin = t
    if thin is None or thin <= 0:
        return None
    step = thin / 2.0

    min_x = min(ei["min_x"] for ei in walls)
    max_x = max(ei["max_x"] for ei in walls)
    min_y = min(ei["min_y"] for ei in walls)
    max_y = max(ei["max_y"] for ei in walls)

    # One empty ring around the model so the fill always has a seed outside.
    min_x -= step
    max_x += step
    min_y -= step
    max_y += step

    nx = int(math.ceil((max_x - min_x) / step)) + 1
    ny = int(math.ceil((max_y - min_y) / step)) + 1
    # Guard against a pathological model turning this into a huge raster.
    if nx < 3 or ny < 3 or nx * ny > 4000000:
        return None

    # solid[i][j] = a wall occupies this cell.
    solid = [[False] * ny for _ in range(nx)]
    cells_of = {}
    for ei in walls:
        wid = ei["element"].Id.IntegerValue
        i0 = int(math.floor((ei["min_x"] - min_x) / step))
        i1 = int(math.ceil((ei["max_x"] - min_x) / step))
        j0 = int(math.floor((ei["min_y"] - min_y) / step))
        j1 = int(math.ceil((ei["max_y"] - min_y) / step))
        i0 = max(0, i0)
        j0 = max(0, j0)
        i1 = min(nx - 1, i1)
        j1 = min(ny - 1, j1)
        own = []
        for i in range(i0, i1 + 1):
            row = solid[i]
            for j in range(j0, j1 + 1):
                row[j] = True
                own.append((i, j))
        cells_of.setdefault(wid, []).extend(own)

    # Flood fill the open space from the corner (guaranteed empty by the ring).
    outside = [[False] * ny for _ in range(nx)]
    stack = [(0, 0)]
    outside[0][0] = True
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni = i + di
            nj = j + dj
            if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
                continue
            if outside[ni][nj] or solid[ni][nj]:
                continue
            outside[ni][nj] = True
            stack.append((ni, nj))

    # A wall is exterior when any of its cells touches outside-reachable space.
    ext_ids = set()
    for wid, cells in cells_of.items():
        for (i, j) in cells:
            hit = False
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni = i + di
                nj = j + dj
                if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
                    continue
                if outside[ni][nj]:
                    hit = True
                    break
            if hit:
                ext_ids.add(wid)
                break

    return ext_ids or None


def _compute_exterior_wall_ids(all_elems):
    """
    Returns a set of wall element IDs that have at least one exterior face
    (checked against both axes).  Used to exclude core/interior walls from
    the opening-dimension pass.
    """
    if USE_FOOTPRINT_EXTERIOR_DETECTION:
        ids = _compute_exterior_wall_ids_footprint(all_elems)
        if ids:
            return ids
        _glog(u"footprint exterior detection unavailable -- using overlap test")

    MIN_OVERLAP = mm_to_ft(300)
    ext_ids = set()

    for axis in ("x", "y"):
        walls = []
        for ei in all_elems:
            if not isinstance(ei["element"], Wall):
                continue
            try:
                orient = ei["element"].Orientation
                if axis == "x" and abs(orient.X) < 0.7:
                    continue
                if axis == "y" and abs(orient.Y) < 0.7:
                    continue
            except Exception:
                continue
            if axis == "x":
                center = (ei["min_x"] + ei["max_x"]) / 2.0
                perp_lo, perp_hi = ei["min_y"], ei["max_y"]
            else:
                center = (ei["min_y"] + ei["max_y"]) / 2.0
                perp_lo, perp_hi = ei["min_x"], ei["max_x"]
            walls.append({
                "id": ei["element"].Id.IntegerValue,
                "center": center, "perp_lo": perp_lo, "perp_hi": perp_hi,
            })

        for w in walls:
            lo_blocked = any(
                o is not w and o["center"] < w["center"]
                and min(w["perp_hi"], o["perp_hi"]) - max(w["perp_lo"], o["perp_lo"]) >= MIN_OVERLAP
                for o in walls
            )
            hi_blocked = any(
                o is not w and o["center"] > w["center"]
                and min(w["perp_hi"], o["perp_hi"]) - max(w["perp_lo"], o["perp_lo"]) >= MIN_OVERLAP
                for o in walls
            )
            if not lo_blocked or not hi_blocked:
                ext_ids.add(w["id"])

    return ext_ids


def _diag_dump_model(all_elems, ext_wall_ids):
    """One-shot diagnostic dump to the guides log: every door/window in the
    view (with its host and whether that host was detected as exterior) and
    every exterior wall (kind, orientation, group membership). Pure logging --
    no model changes."""
    try:
        wall_ids_in_view = set(ei["element"].Id.IntegerValue
                               for ei in all_elems)
        _glog(u"--- DIAG walls: {} in view, {} exterior ---".format(
            len(all_elems), len(ext_wall_ids)))
        for ei in all_elems:
            w = ei["element"]
            wid = w.Id.IntegerValue
            if wid not in ext_wall_ids:
                continue
            try:
                kind = str(w.WallType.Kind)
            except Exception:
                kind = u"?"
            try:
                o = w.Orientation
                orient = u"({:.2f},{:.2f})".format(o.X, o.Y)
            except Exception:
                orient = u"?"
            try:
                gid = w.GroupId.IntegerValue
            except Exception:
                gid = -1
            _glog(u"DIAG ext wall id={} kind={} orient={} group={} "
                  u"bbox_x=({},{}) bbox_y=({},{})".format(
                      wid, kind, orient, gid,
                      int(round(ft_to_mm(ei["min_x"]))),
                      int(round(ft_to_mm(ei["max_x"]))),
                      int(round(ft_to_mm(ei["min_y"]))),
                      int(round(ft_to_mm(ei["max_y"])))))
        n_open = 0
        for host_id, elems in _hosted_openings_map().items():
            for elem in elems:
                n_open += 1
                try:
                    cat = elem.Category.Name
                except Exception:
                    cat = u"?"
                try:
                    fam = elem.Symbol.Family.Name
                except Exception:
                    fam = u"?"
                try:
                    gid = elem.GroupId.IntegerValue
                except Exception:
                    gid = -1
                try:
                    loc = elem.Location
                    pt = loc.Point if hasattr(loc, "Point") \
                        else loc.Curve.Evaluate(0.5, True)
                    at = u"({},{})".format(int(round(ft_to_mm(pt.X))),
                                           int(round(ft_to_mm(pt.Y))))
                except Exception:
                    at = u"?"
                _glog(u"DIAG opening id={} cat={} fam='{}' group={} at={} "
                      u"host={} host_in_view={} host_is_ext={}".format(
                          elem.Id.IntegerValue, cat, fam, gid, at, host_id,
                          host_id in wall_ids_in_view,
                          host_id in ext_wall_ids))
        _glog(u"--- DIAG openings total: {} ---".format(n_open))
    except Exception as e:
        _glog(u"DIAG dump EXCEPTION: {}".format(str(e)))


def _dedupe_dimension_pairs(pairs):
    """Sort reference/coordinate pairs and remove coincident references."""
    if not pairs:
        return []
    pairs = sorted(pairs, key=lambda item: item[1])
    tolerance = mm_to_ft(1)
    result = []
    for ref, coord in pairs:
        if not result or abs(coord - result[-1][1]) > tolerance:
            result.append((ref, coord))
    return result


def _continuous_exterior_pairs(all_elems, ext_wall_ids, axis, include_openings,
                               side=None, perp_mid=None):
    """Build continuous references for one exterior dimension direction.

    When ``side`` ("min" or "max") and ``perp_mid`` are given, the per-wall end
    edges and openings are restricted to the wall on that side of the building,
    so a guide near one facade does NOT reach across and pick up the opposite
    facade's openings.

    ``_find_exterior_face_refs`` returns EVERY exterior face along the axis --
    both facades plus interior jog faces. Only the two building-END extremes are
    legitimate shared endpoints for a side chain; the rest are either the far
    facade (which produced cross-facade segments) or jog faces one wall-thickness
    from a corner (which produced thickness slivers). So when a side is requested
    we keep only those two extremes and let the side-filtered wall end edges
    supply every real break in between.
    """
    envelope = _find_exterior_face_refs(all_elems, axis)
    if side is not None and perp_mid is not None:
        detail_pairs = []
        if envelope:
            detail_pairs = [min(envelope, key=lambda rc: rc[1]),
                            max(envelope, key=lambda rc: rc[1])]
    else:
        detail_pairs = list(envelope)
    opening_pairs = []

    for ei in all_elems:
        wall = ei["element"]
        if ei["element"].Id.IntegerValue not in ext_wall_ids:
            continue
        if not isinstance(wall, Wall):
            continue
        try:
            orient = wall.Orientation
            runs_on_axis = (
                axis == "x" and abs(orient.Y) > 0.7
            ) or (
                axis == "y" and abs(orient.X) > 0.7
            )
            if not runs_on_axis:
                continue

            # Restrict to the near-side facade when a side is requested.
            if side is not None and perp_mid is not None:
                if axis == "x":
                    wc = (ei["min_y"] + ei["max_y"]) / 2.0
                else:
                    wc = (ei["min_x"] + ei["max_x"]) / 2.0
                w_side = "min" if wc < perp_mid else "max"
                if w_side != side:
                    continue

            # End edges preserve every break in the continuous exterior line.
            detail_pairs.extend(_get_wall_end_edge_refs(wall, orient, axis))
            if include_openings:
                opening_pairs.extend(_collect_opening_face_refs(wall, axis))
        except Exception:
            continue

    if include_openings:
        return _dedupe_dimension_pairs(detail_pairs + opening_pairs)
    return _dedupe_dimension_pairs(detail_pairs)


def _facade_segment_pairs(all_elems, ext_wall_ids, axis, side, perp_mid):
    """Tier-2 references: the building facade SEGMENTS, ignoring openings.

    Walls that lie on the same facade line (same perpendicular position) are
    merged into one run even if the model splits them at an opening, so the
    dimension breaks only at real corners / jogs (a change in the facade's
    perpendicular position) and at the building ends. Openings are not used.
    """
    # Envelope perpendicular faces. Only the two EXTREME ones (building ends)
    # are shared endpoints for this side; middle faces belong to a jog on ONE
    # facade and must not be injected into the other facade's chain.
    envelope = _find_exterior_face_refs(all_elems, axis)
    envelope_ends = []
    if envelope:
        envelope_ends = [
            min(envelope, key=lambda rc: rc[1]),
            max(envelope, key=lambda rc: rc[1]),
        ]
    # Construction-noise tolerance only -- any real jog/protrusion (a wall
    # stepping out even by less than a wall thickness) must still break the
    # run, so this stays tight rather than a generous "looks like a corner"
    # margin.
    perp_tol = mm_to_ft(INTERSECT_TOL_MM)
    match_tol = mm_to_ft(300)

    pieces = []
    endrefs = []
    for ei in all_elems:
        wall = ei["element"]
        if ei["element"].Id.IntegerValue not in ext_wall_ids:
            continue
        if not isinstance(wall, Wall):
            continue
        try:
            orient = wall.Orientation
            runs_on_axis = (
                axis == "x" and abs(orient.Y) > 0.7
            ) or (
                axis == "y" and abs(orient.X) > 0.7
            )
            if not runs_on_axis:
                continue
            if axis == "x":
                wc = (ei["min_y"] + ei["max_y"]) / 2.0
                c_lo, c_hi = ei["min_x"], ei["max_x"]
            else:
                wc = (ei["min_x"] + ei["max_x"]) / 2.0
                c_lo, c_hi = ei["min_y"], ei["max_y"]
            if perp_mid is not None and side is not None:
                if ("min" if wc < perp_mid else "max") != side:
                    continue
            pieces.append({"c_lo": c_lo, "c_hi": c_hi, "perp": wc})
            for ref, coord in _get_wall_end_edge_refs(wall, orient, axis):
                endrefs.append((ref, coord))
        except Exception:
            continue

    if not pieces:
        return _dedupe_dimension_pairs(envelope_ends)

    # Sweep left-to-right; same-perp neighbours extend the run (openings vanish),
    # a perp change starts a new run (a real jog / corner).
    pieces.sort(key=lambda p: p["c_lo"])
    runs = []
    for p in pieces:
        if runs and abs(p["perp"] - runs[-1]["perp"]) < perp_tol:
            runs[-1]["c_hi"] = max(runs[-1]["c_hi"], p["c_hi"])
            runs[-1]["c_lo"] = min(runs[-1]["c_lo"], p["c_lo"])
        else:
            runs.append({"c_lo": p["c_lo"], "c_hi": p["c_hi"],
                         "perp": p["perp"]})

    break_coords = []
    for r in runs:
        break_coords.append(r["c_lo"])
        break_coords.append(r["c_hi"])

    # Map each break coordinate to the nearest real face reference. Only the
    # near-side wall end edges and the two building-end corners are candidates,
    # so a far-facade jog can never appear in this side's chain.
    all_refs = endrefs + envelope_ends
    pairs = []
    for bc in break_coords:
        best = None
        best_d = None
        for ref, coord in all_refs:
            d = abs(coord - bc)
            if best_d is None or d < best_d:
                best_d = d
                best = ref
        if best is not None and best_d <= match_tol:
            pairs.append((best, bc))
    pairs.extend(envelope_ends)  # guarantee the outer building corners
    return _dedupe_dimension_pairs(pairs)


def make_continuous_exterior_chains(all_elems, ext_wall_ids, dims_to_adjust):
    """Create three continuous exterior tiers for X and Y directions."""
    created = 0

    for axis in ("x", "y"):
        tier1 = _continuous_exterior_pairs(all_elems, ext_wall_ids, axis, True)
        tier2 = _continuous_exterior_pairs(all_elems, ext_wall_ids, axis, False)
        if len(tier2) < 2:
            continue

        # Always measure from the outermost selected exterior wall side.
        exterior_elems = [
            ei for ei in all_elems
            if ei["element"].Id.IntegerValue in ext_wall_ids
        ]
        if not exterior_elems:
            continue
        perp_base = min(
            ei["min_y"] if axis == "x" else ei["min_x"]
            for ei in exterior_elems
        )

        c_min = tier2[0][1]
        c_max = tier2[-1][1]

        def create_tier(pairs, offset_mm, label):
            if len(pairs) < 2:
                return 0
            refs = [ref for ref, unused_coord in pairs]
            offset = mm_to_ft(offset_mm)
            perp = perp_base - offset
            if axis == "x":
                p0 = XYZ(c_min, perp, 0)
                p1 = XYZ(c_max, perp, 0)
            else:
                p0 = XYZ(perp, c_min, 0)
                p1 = XYZ(perp, c_max, 0)
            dim = make_dim(refs, p0, p1, label)
            if dim:
                dims_to_adjust.append(dim)
                return 1
            return 0

        created += create_tier(tier1, OFFSET_1_MM, "continuous-tier1-" + axis)
        created += create_tier(tier2, OFFSET_2_MM, "continuous-tier2-" + axis)
        created += create_tier(
            [tier2[0], tier2[-1]], OFFSET_3_MM,
            "continuous-overall-" + axis
        )

    return created


# ---------------------------------------------------------------------------
# Guide-line workflow (two-click exterior dimensioning)
# ---------------------------------------------------------------------------
# Click 1: draw three model lines per axis (opening / facade / overall) at the
#          default offsets, then stop so the user can move them.
# Click 2: read each guide line's position and place the matching dimension tier
#          along it, then delete the guide lines.

def _guide_comment(axis, tier, side):
    return u"{}|{}|{}|{}".format(GUIDE_MARK, axis, tier, side)


def _get_comment(elem):
    """Locale-independent read of the Comments parameter (Hebrew Revit safe)."""
    try:
        p = elem.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
        if p:
            return p.AsString()
    except Exception:
        pass
    return None


# --- Guide tagging via Line Styles -----------------------------------------
# Model lines do NOT reliably carry the Comments parameter, so tagging guides
# through Comments silently fails. Their Line Style (a subcategory of "Lines")
# always exists, persists across copy/move, and is locale-independent, so it is
# used as the durable tag. Names encode the guide role; "|" is illegal in Revit
# category names, so "_" is used as the separator.
#   exterior: AUTODIM_GUIDE_<axis>_<tier>_<side>   e.g. AUTODIM_GUIDE_x_1_min
#   interior: AUTODIM_IGUIDE_<axis>                e.g. AUTODIM_IGUIDE_x

def _guide_style_name(axis, tier, side):
    return u"{}_{}_{}_{}".format(GUIDE_MARK, axis, tier, side)


def _iguide_style_name(axis):
    return u"{}_{}".format(IGUIDE_MARK, axis)


def _get_or_create_line_style(name):
    """Return the projection GraphicsStyle for the "Lines" subcategory `name`,
    creating the subcategory if it does not exist. Must run in a transaction."""
    cats = doc.Settings.Categories
    lines_cat = cats.get_Item(BuiltInCategory.OST_Lines)
    subs = lines_cat.SubCategories
    sub = None
    try:
        if subs.Contains(name):
            sub = subs.get_Item(name)
    except Exception:
        sub = None
    if sub is None:
        for sc in subs:
            if sc.Name == name:
                sub = sc
                break
    if sub is None:
        sub = cats.NewSubcategory(lines_cat, name)
    # Force the guide style to be bright red and bold so the guide lines are
    # always clearly visible (a freshly recreated subcategory is otherwise a thin
    # near-invisible default).
    try:
        sub.LineColor = Color(255, 0, 0)
    except Exception:
        pass
    try:
        sub.SetLineWeight(6, GraphicsStyleType.Projection)
    except Exception:
        pass
    return sub.GetGraphicsStyle(GraphicsStyleType.Projection)


def _apply_line_style(mc, name):
    """Tag a model line with the named line style; ignore if unsupported."""
    try:
        mc.LineStyle = _get_or_create_line_style(name)
    except Exception:
        pass


def _purge_guide_line_styles():
    """Delete every AUTODIM_* line style subcategory from Object Styles. Any
    lines still using one revert to the default style. Must run in a
    transaction; call only after the guide lines themselves are deleted."""
    removed = 0
    try:
        lines_cat = doc.Settings.Categories.get_Item(BuiltInCategory.OST_Lines)
        victims = []
        for sc in lines_cat.SubCategories:
            try:
                nm = sc.Name
            except Exception:
                continue
            if nm and (nm.startswith(GUIDE_MARK) or nm.startswith(IGUIDE_MARK)):
                victims.append(sc.Id)
        for sid in victims:
            try:
                doc.Delete(sid)
                removed += 1
            except Exception:
                pass
    except Exception:
        pass
    return removed


def _line_style_name(ce):
    """Name of a curve element's line style, or None."""
    try:
        gs = ce.LineStyle
        if gs is None:
            return None
        try:
            return gs.Name
        except Exception:
            return gs.GraphicsStyleCategory.Name
    except Exception:
        return None


def _collect_model_lines():
    """Guide lines are DETAIL curves in the active view (view-specific, so they
    are always visible in the plan regardless of level elevation, view range,
    crop, or worksets). Collect every curve element in this view."""
    out = []
    try:
        col = FilteredElementCollector(doc, view.Id).OfCategory(
            BuiltInCategory.OST_Lines).WhereElementIsNotElementType()
        for ce in col:
            if isinstance(ce, CurveElement):
                out.append(ce)
    except Exception:
        pass
    return out


def _level_elevation():
    """Active plan's level elevation. Unlike _make_sketch_plane this creates
    nothing, so it is safe to call with no transaction open (the interactive
    guide picking needs the elevation between transactions)."""
    try:
        gl = view.GenLevel
        if gl is not None:
            return gl.Elevation
    except Exception:
        pass
    return 0.0


def _make_sketch_plane():
    """Horizontal sketch plane at the active plan's level elevation."""
    elev = 0.0
    gl_name = "none"
    try:
        gl = view.GenLevel
        if gl is not None:
            elev = gl.Elevation
            try:
                gl_name = gl.Name
            except Exception:
                gl_name = "?"
    except Exception:
        pass
    _glog(u"sketch plane: level='{}' elev_mm={}".format(
        gl_name, int(round(ft_to_mm(elev)))))
    plane = Plane.CreateByNormalAndOrigin(XYZ.BasisZ, XYZ(0, 0, elev))
    return SketchPlane.Create(doc, plane), elev


def _max_wall_thickness(exterior_elems):
    """Largest exterior wall thickness (feet), for the thickness-sliver filter.
    Falls back to a modest default if no width can be read."""
    widths = []
    for ei in exterior_elems:
        try:
            w = ei["element"].Width
            if w and w > 0:
                widths.append(w)
        except Exception:
            continue
    if widths:
        return max(widths)
    return mm_to_ft(300)


def _drop_thickness_refs(pairs, min_ft):
    """Remove chain references that would create a segment at or below one wall
    thickness, so the exterior string never shows a wall-thickness dimension.

    Keeps the outermost reference on each end (the true building extent) and
    drops the inner face of the corner/jog that sits a thickness away."""
    if len(pairs) <= 2:
        return pairs
    ordered = sorted(pairs, key=lambda item: item[1])
    kept = [ordered[0]]
    for p in ordered[1:-1]:
        if abs(p[1] - kept[-1][1]) > min_ft:
            kept.append(p)
    last = ordered[-1]
    if abs(last[1] - kept[-1][1]) > min_ft:
        kept.append(last)
    else:
        # A sliver at the far end: keep the outer face, drop the inner one.
        kept[-1] = last
    return kept


def _exterior_tier_geometry(all_elems, ext_wall_ids, axis):
    """Shared geometry for both guide creation and dimensioning along one axis.

    Returns a dict (c_min, c_max, perp_base, tier1/tier2/overall ref pairs) or
    None when there is nothing to dimension on this axis.
    """
    tier2 = _continuous_exterior_pairs(all_elems, ext_wall_ids, axis, False)
    if len(tier2) < 2:
        return None
    exterior_elems = [
        ei for ei in all_elems
        if ei["element"].Id.IntegerValue in ext_wall_ids
    ]
    if not exterior_elems:
        return None
    # Two perpendicular extremes so tiers can be placed on BOTH sides:
    #   axis "x": perp_min = bottom (min Y), perp_max = top (max Y)
    #   axis "y": perp_min = left  (min X), perp_max = right (max X)
    perp_min = min(
        ei["min_y"] if axis == "x" else ei["min_x"]
        for ei in exterior_elems
    )
    perp_max = max(
        ei["max_y"] if axis == "x" else ei["max_x"]
        for ei in exterior_elems
    )
    perp_mid = (perp_min + perp_max) / 2.0

    # Wall-thickness tolerance, derived from the actual exterior walls (no
    # hardcoded distance). Any exterior chain segment at or below this length is
    # a wall-thickness sliver and is dropped, so we never dimension a thickness.
    # Factor > 1 covers mitered/joined corners, where the end-cap face sits a
    # bit further than one nominal thickness from the outer envelope face.
    thick_tol = _max_wall_thickness(exterior_elems) * 1.6

    # Side-aware tiers: a guide on the "min" facade references only min-side
    # walls/openings, and likewise for "max" — so dimensions never reach across
    # the building to the opposite facade. Envelope endpoints are shared.
    def _pairs(open_, side):
        return _continuous_exterior_pairs(
            all_elems, ext_wall_ids, axis, open_, side=side, perp_mid=perp_mid)

    return {
        "axis": axis,
        "c_min": tier2[0][1],
        "c_max": tier2[-1][1],
        "perp_min": perp_min,
        "perp_max": perp_max,
        "thick_tol": thick_tol,
        "tier1_min": _pairs(True, "min"),
        "tier1_max": _pairs(True, "max"),
        "tier2_min": _facade_segment_pairs(
            all_elems, ext_wall_ids, axis, "min", perp_mid),
        "tier2_max": _facade_segment_pairs(
            all_elems, ext_wall_ids, axis, "max", perp_mid),
        "overall": [tier2[0], tier2[-1]],
    }


# Tier number -> default offset from the outer exterior face (mm).
GUIDE_TIER_OFFSETS = ((1, OFFSET_1_MM), (2, OFFSET_2_MM), (3, OFFSET_3_MM))


def _create_guide_line(sp, axis, tier, side, c_min, c_max, perp, elev):
    if axis == "x":
        p0 = XYZ(c_min, perp, elev)
        p1 = XYZ(c_max, perp, elev)
    else:
        p0 = XYZ(perp, c_min, elev)
        p1 = XYZ(perp, c_max, elev)
    ln = Line.CreateBound(p0, p1)
    # Detail curve (view-specific) instead of model line, so guides always show
    # in this plan regardless of level elevation / view range / crop / worksets.
    mc = doc.Create.NewDetailCurve(view, ln)
    # Durable tag = line style. Comments is also set as a best-effort fallback.
    _apply_line_style(mc, _guide_style_name(axis, tier, side))
    p = mc.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
    if p is not None and not p.IsReadOnly:
        p.Set(_guide_comment(axis, tier, side))
    applied = _line_style_name(mc)
    _glog(u"guide {}-t{}-{}: ({},{})->({},{}) z_mm={} style='{}'".format(
        axis, tier, side,
        int(round(ft_to_mm(p0.X))), int(round(ft_to_mm(p0.Y))),
        int(round(ft_to_mm(p1.X))), int(round(ft_to_mm(p1.Y))),
        int(round(ft_to_mm(elev))), applied))
    return mc


def create_exterior_guides(all_elems, ext_wall_ids):
    """Click 1: place three guide lines per axis on BOTH sides (all 4 sides)."""
    sp, elev = _make_sketch_plane()
    created = 0
    for axis in ("x", "y"):
        geo = _exterior_tier_geometry(all_elems, ext_wall_ids, axis)
        if not geo:
            continue
        for side in ("min", "max"):
            for tier, off_mm in GUIDE_TIER_OFFSETS:
                if side == "min":
                    perp = geo["perp_min"] - mm_to_ft(off_mm)
                else:
                    perp = geo["perp_max"] + mm_to_ft(off_mm)
                _create_guide_line(
                    sp, axis, tier, side, geo["c_min"], geo["c_max"], perp, elev)
                created += 1
    return created


def _find_guide_lines():
    """Return existing guide lines as dicts (elem, axis, tier, perp).

    ``perp`` is read from the line's current midpoint, so it reflects any move
    the user made between click 1 and click 2.
    """
    guides = []
    for ce in _collect_model_lines():
        axis = tier = side = None
        # Primary tag: line style name  AUTODIM_GUIDE_<axis>_<tier>_<side>
        sn = _line_style_name(ce)
        if sn and sn.startswith(GUIDE_MARK + u"_"):
            rest = sn[len(GUIDE_MARK) + 1:].split(u"_")
            if len(rest) == 3:
                axis = rest[0]
                try:
                    tier = int(rest[1])
                except Exception:
                    tier = None
                side = rest[2]
        # Fallback tag: Comments  AUTODIM_GUIDE|<axis>|<tier>|<side>
        if axis is None:
            c = _get_comment(ce)
            if c and c.startswith(GUIDE_MARK + u"|"):
                parts = c.split(u"|")
                if len(parts) == 4:
                    axis = parts[1]
                    try:
                        tier = int(parts[2])
                    except Exception:
                        tier = None
                    side = parts[3]
        if axis is None or tier is None:
            continue
        try:
            crv = ce.GeometryCurve
            mid = crv.Evaluate(0.5, True)
            perp = mid.Y if axis == "x" else mid.X
        except Exception:
            continue
        guides.append({"elem": ce, "axis": axis, "tier": tier,
                       "side": side, "perp": perp})
    return guides


def dimension_along_guides(all_elems, ext_wall_ids, guides, dims_to_adjust):
    """Click 2: place a dimension at each guide line's position (all sides)."""
    created = 0
    geo_cache = {}
    _seg_log_reset()
    for g in guides:
        axis = g["axis"]
        if axis not in geo_cache:
            geo_cache[axis] = _exterior_tier_geometry(
                all_elems, ext_wall_ids, axis)
        geo = geo_cache[axis]
        if not geo:
            continue
        tier = g["tier"]
        side = g.get("side")
        if side not in ("min", "max"):
            perp_mid = (geo["perp_min"] + geo["perp_max"]) / 2.0
            side = "min" if g["perp"] < perp_mid else "max"
        if tier == 1:
            pairs = geo["tier1_max"] if side == "max" else geo["tier1_min"]
        elif tier == 2:
            pairs = geo["tier2_max"] if side == "max" else geo["tier2_min"]
        else:
            pairs = geo["overall"]
        # Exterior rule: never dimension a wall thickness — drop thickness slivers.
        raw_mm = [int(round(ft_to_mm(c))) for unused_r, c in
                  sorted(pairs, key=lambda it: it[1])]
        ttol = geo.get("thick_tol", mm_to_ft(330))
        pairs = _drop_thickness_refs(pairs, ttol)
        kept_mm = [int(round(ft_to_mm(c))) for unused_r, c in pairs]
        segs_mm = [kept_mm[i + 1] - kept_mm[i] for i in range(len(kept_mm) - 1)]
        _seg_log(u"axis={} tier={} side={} thick_tol_mm={} n_raw={} n_kept={}"
                 u"\n  raw={}\n  kept={}\n  segs={}".format(
                     axis, tier, side, int(round(ft_to_mm(ttol))),
                     len(raw_mm), len(kept_mm), raw_mm, kept_mm, segs_mm))
        if len(pairs) < 2:
            continue
        refs = [ref for ref, unused_coord in pairs]
        perp = g["perp"]
        if axis == "x":
            a = XYZ(geo["c_min"], perp, 0)
            b = XYZ(geo["c_max"], perp, 0)
        else:
            a = XYZ(perp, geo["c_min"], 0)
            b = XYZ(perp, geo["c_max"], 0)
        # Revit places dimension text on the +90deg side of the line direction.
        # For the lower (bottom, axis-x min) string, drawing left->right puts the
        # text on the building side; reverse it so the text sits away from the
        # facade wall, matching the (already-correct) upper string.
        if axis == "x" and side == "min":
            p0, p1 = b, a
        else:
            p0, p1 = a, b
        dim = make_dim(refs, p0, p1, "guide-t{}-{}-{}".format(
            tier, axis, g.get("side", "")))
        if dim:
            dims_to_adjust.append(dim)
            created += 1
    return created


def _delete_guides(guides):
    for g in guides:
        try:
            doc.Delete(g["elem"].Id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Interior guide lines (one horizontal + one vertical, user copies/moves them)
# ---------------------------------------------------------------------------

def _iguide_comment(axis):
    return u"{}|{}".format(IGUIDE_MARK, axis)


def _create_interior_guide(axis, perp, lo, hi, elev):
    """Create one interior guide line in its own transaction, so it appears on
    the canvas the moment the user clicks rather than at the end of the run."""
    if axis == "y":
        p0 = XYZ(perp, lo, elev)
        p1 = XYZ(perp, hi, elev)
    else:
        p0 = XYZ(lo, perp, elev)
        p1 = XYZ(hi, perp, elev)
    t = Transaction(doc, u"AutoDimension interior guide")
    t.Start()
    try:
        mc = doc.Create.NewDetailCurve(view, Line.CreateBound(p0, p1))
        _apply_line_style(mc, _iguide_style_name(axis))
        p = mc.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
        if p is not None and not p.IsReadOnly:
            p.Set(_iguide_comment(axis))
        t.Commit()
        return True
    except Exception as e:
        t.RollBack()
        _glog(u"interior guide create FAILED axis={} perp_mm={}: {}".format(
            axis, int(round(ft_to_mm(perp))), str(e)))
        return False


def pick_and_create_interior_guides(all_elems):
    """Click the interior guides into place, drawing each one immediately.

    Phase 1 places VERTICAL guides, phase 2 HORIZONTAL ones. Left click drops a
    guide (drawn at once); Esc -- or right-click then Cancel on Revit's context
    menu -- ends the phase. Ending phase 2 leaves placement.

    Must run in a command context with NO transaction open: PickPoint cannot be
    called inside a transaction, nor from the modeless window at all. Each guide
    therefore gets its own short transaction between picks.

    Returns the number of guides created.
    """
    if not all_elems:
        return 0
    min_x = min(ei["min_x"] for ei in all_elems)
    max_x = max(ei["max_x"] for ei in all_elems)
    min_y = min(ei["min_y"] for ei in all_elems)
    max_y = max(ei["max_y"] for ei in all_elems)
    elev = _level_elevation()

    # Hand focus back to the drawing area before picking. Straight after the
    # ribbon command and the exterior-guide commit, Revit tends to eat the first
    # click or two just to activate the view, and those clicks never reach
    # PickPoint at all (confirmed from the log: every point that DID arrive
    # produced a line).
    try:
        uidoc.RefreshActiveView()
    except Exception:
        pass

    created = 0
    # axis "y" = vertical line (spans Y, positioned by X); "x" = horizontal.
    phases = (
        ("y", u"VERTICAL interior guides: click to place, Esc when done"),
        ("x", u"HORIZONTAL interior guides: click to place, Esc when done"),
    )
    for axis, prompt in phases:
        n_axis = 0
        n_points = 0
        while True:
            try:
                pt = uidoc.Selection.PickPoint(prompt)
            except Exception:
                # Esc / right-click-Cancel raises OperationCanceledException.
                # Any other failure (no work plane, view cannot pick) also just
                # ends this phase instead of killing the whole run.
                break
            if pt is None:
                break
            n_points += 1
            perp = pt.X if axis == "y" else pt.Y
            if axis == "y":
                ok = _create_interior_guide(axis, perp, min_y, max_y, elev)
            else:
                ok = _create_interior_guide(axis, perp, min_x, max_x, elev)
            if ok:
                created += 1
                n_axis += 1
        # points vs placed: if they match but you clicked more times than that,
        # the extra clicks were swallowed by Revit before reaching PickPoint.
        _glog(u"interior guides axis {}: {} points picked, {} placed".format(
            axis, n_points, n_axis))

    # Every line here is an explicit request, so there is no seed to skip later.
    _AD_INT_SEEDS["x"] = []
    _AD_INT_SEEDS["y"] = []
    return created


def create_interior_guides(all_elems):
    """Seed one horizontal + one vertical interior guide line through the
    building center, for the user to copy/move. Used when interactive placement
    is turned off (INTERACTIVE_INTERIOR_GUIDES = False).

    axis "x": horizontal line -> measures the N/S walls it crosses.
    axis "y": vertical line   -> measures the E/W walls it crosses.
    """
    if not all_elems:
        return 0
    sp, elev = _make_sketch_plane()
    min_x = min(ei["min_x"] for ei in all_elems)
    max_x = max(ei["max_x"] for ei in all_elems)
    min_y = min(ei["min_y"] for ei in all_elems)
    max_y = max(ei["max_y"] for ei in all_elems)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    # Remember the seed positions so untouched seeds are skipped when
    # dimensioning (perp is Y for a horizontal "x" guide, X for a "y" one).
    _AD_INT_SEEDS["x"] = [cy]
    _AD_INT_SEEDS["y"] = [cx]
    lines = [
        ("x", XYZ(min_x, cy, elev), XYZ(max_x, cy, elev)),
        ("y", XYZ(cx, min_y, elev), XYZ(cx, max_y, elev)),
    ]

    created = 0
    for axis, p0, p1 in lines:
        try:
            ln = Line.CreateBound(p0, p1)
            mc = doc.Create.NewDetailCurve(view, ln)
            _apply_line_style(mc, _iguide_style_name(axis))
            p = mc.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
            if p is not None and not p.IsReadOnly:
                p.Set(_iguide_comment(axis))
            created += 1
        except Exception:
            continue
    return created


def _find_interior_guides():
    """Return existing interior guide lines as dicts (elem, axis, perp).

    Any copies the user made are picked up too, since they inherit the marker.
    """
    guides = []
    for ce in _collect_model_lines():
        axis = None
        # Primary tag: line style name  AUTODIM_IGUIDE_<axis>
        sn = _line_style_name(ce)
        if sn and sn.startswith(IGUIDE_MARK + u"_"):
            axis = sn[len(IGUIDE_MARK) + 1:]
        # Fallback tag: Comments  AUTODIM_IGUIDE|<axis>
        if axis is None:
            c = _get_comment(ce)
            if c and c.startswith(IGUIDE_MARK + u"|"):
                parts = c.split(u"|")
                if len(parts) == 2:
                    axis = parts[1]
        if axis not in ("x", "y"):
            continue
        try:
            crv = ce.GeometryCurve
            mid = crv.Evaluate(0.5, True)
            perp = mid.Y if axis == "x" else mid.X
        except Exception:
            continue
        guides.append({"elem": ce, "axis": axis, "perp": perp})
    return guides


def dimension_along_interior_guides(all_elems, guides, dims_to_adjust):
    """Create an interior dimension along each interior guide line, reusing the
    existing per-line wall-crossing logic (_interior_dimension_at_point)."""
    created = 0
    _seg_log(u"--- interior guides found: {} ---".format(len(guides)))
    seed_tol = mm_to_ft(30)

    def _is_seed(axis, perp):
        return any(abs(perp - s) <= seed_tol for s in _AD_INT_SEEDS.get(axis, []))

    # An axis is "active" if it has at least one guide the user moved/copied away
    # from the seed. Only then do we treat the leftover seed as a copy-template to
    # skip. If the user left the single seed where it is, it IS the request.
    moved_axes = set(g["axis"] for g in guides if not _is_seed(g["axis"], g["perp"]))

    done_perp = {"x": [], "y": []}
    for i, g in enumerate(guides):
        axis = g["axis"]
        perp = g["perp"]
        if _is_seed(axis, perp) and axis in moved_axes:
            _seg_log(u"int guide #{} axis={} perp_mm={} SKIPPED (leftover copy seed)".format(
                i + 1, axis, int(round(ft_to_mm(perp)))))
            continue
        # De-duplicate guides landing on the same line (e.g. a copy dropped on top
        # of another), so the same interior dimension is not created twice.
        if any(abs(perp - p) <= seed_tol for p in done_perp.get(axis, [])):
            _seg_log(u"int guide #{} axis={} perp_mm={} SKIPPED (duplicate line)".format(
                i + 1, axis, int(round(ft_to_mm(perp)))))
            continue
        done_perp[axis].append(perp)
        _seg_log(u"int guide #{} axis={} perp_mm={}".format(
            i + 1, axis, int(round(ft_to_mm(perp)))))
        point = XYZ(0, perp, 0) if axis == "x" else XYZ(perp, 0, 0)
        created += _interior_dimension_at_point(
            point, axis, all_elems, dims_to_adjust, i + 1)
    return created


def make_building_chains(all_elems, dims_to_adjust):
    """
    No-grid fallback.  Creates two chains per axis:
    - Layer 2 (OFFSET_2_MM): all exterior face positions — facade detail
    - Layer 3 (OFFSET_3_MM): only the two outer faces — overall building span
    Both are placed on the south/west side of the building.
    Returns total number of dimensions created.
    """
    created = 0

    for axis in ("x", "y"):
        faces = _find_exterior_face_refs(all_elems, axis)
        if len(faces) < 2:
            # fallback: use all faces
            faces = []
            for ei in all_elems:
                ref_lo, ref_hi, c_lo, c_hi = get_faces(ei["element"], axis)
                if ref_lo is None:
                    continue
                faces.append((ref_lo, c_lo))
                faces.append((ref_hi, c_hi))
        if len(faces) < 2:
            continue

        faces.sort(key=lambda x: x[1])
        tol = mm_to_ft(1)
        deduped = [faces[0]]
        for ref, coord in faces[1:]:
            if abs(coord - deduped[-1][1]) > tol:
                deduped.append((ref, coord))
        if len(deduped) < 2:
            continue

        c_min = deduped[0][1]
        c_max = deduped[-1][1]

        if axis == "x":
            perp_base = min(ei["min_y"] for ei in all_elems)
        else:
            perp_base = min(ei["min_x"] for ei in all_elems)

        # --- Layer 2: all exterior face positions ---
        off2 = mm_to_ft(OFFSET_2_MM)
        refs2 = [r for r, _ in deduped]
        if axis == "x":
            p0 = XYZ(c_min, perp_base - off2, 0)
            p1 = XYZ(c_max, perp_base - off2, 0)
        else:
            p0 = XYZ(perp_base - off2, c_min, 0)
            p1 = XYZ(perp_base - off2, c_max, 0)
        if DEBUG:
            output.print_md(u"🔗 layer2 axis={}: {} refs, perp={:.0f}mm".format(
                axis, len(refs2), ft_to_mm(perp_base - off2)))
        dim2 = make_dim(refs2, p0, p1, "layer2-" + axis)
        if dim2:
            dims_to_adjust.append(dim2)
            created += 1

        # --- Layer 3: overall building span ---
        off3 = mm_to_ft(OFFSET_3_MM)
        refs3 = [deduped[0][0], deduped[-1][0]]
        if axis == "x":
            p0 = XYZ(c_min, perp_base - off3, 0)
            p1 = XYZ(c_max, perp_base - off3, 0)
        else:
            p0 = XYZ(perp_base - off3, c_min, 0)
            p1 = XYZ(perp_base - off3, c_max, 0)
        if DEBUG:
            output.print_md(u"🔗 layer3 axis={}: perp={:.0f}mm".format(
                axis, ft_to_mm(perp_base - off3)))
        dim3 = make_dim(refs3, p0, p1, "layer3-" + axis)
        if dim3:
            dims_to_adjust.append(dim3)
            created += 1

    return created


def dim_along_axis(ei, axis, grids_perpendicular, grids_parallel, all_elems, dims_to_adjust, forced_side=None,
                   occupied_zones=None):
    elem = ei["element"]
    created = 0

    elem_name = elem.Name if hasattr(elem, "Name") else "?"
    if DEBUG:
        output.print_md(u"---")
        output.print_md(u"### {} (id={}) axis={}  cat={}".format(
            elem_name, elem.Id.IntegerValue, axis, ei["category"]))
        output.print_md(u"   bbox: X[{:.0f}..{:.0f}] Y[{:.0f}..{:.0f}] mm".format(
            ft_to_mm(ei["min_x"]), ft_to_mm(ei["max_x"]),
            ft_to_mm(ei["min_y"]), ft_to_mm(ei["max_y"])))

    ref_lo, ref_hi, c_lo, c_hi = get_faces(elem, axis)
    if ref_lo is None:
        if DEBUG:
            output.print_md(u"   ⏭ Skipped — no faces found")
        return 0

    if axis == "x":
        perp_lo = ei["min_y"]
        perp_hi = ei["max_y"]
    else:
        perp_lo = ei["min_x"]
        perp_hi = ei["max_x"]

    side = _pick_side(ei, axis, grids_parallel, forced_side)

    off1 = mm_to_ft(OFFSET_1_MM)
    off2 = mm_to_ft(OFFSET_2_MM)

    if side < 0:
        line_row1 = perp_lo - off1
        line_row2 = perp_lo - off2
    else:
        line_row1 = perp_hi + off1
        line_row2 = perp_hi + off2

    best_grid, best_dist_ft = _find_nearest_grid(ei, axis, grids_perpendicular)

    if DEBUG:
        if best_grid:
            output.print_md(u"   🎯 Nearest grid: **{}** (dist={:.0f}mm), coord={:.0f}mm".format(
                best_grid["name"], ft_to_mm(best_dist_ft), ft_to_mm(best_grid["coord_ft"])))
        else:
            output.print_md(u"   ⚠ No suitable grid found")

    if best_grid is None or ft_to_mm(best_dist_ft) > MAX_SNAP_DIST_MM:
        if DEBUG:
            output.print_md(u"   → Overall only (no grid / too far)")
        dim_g = _dim_overall(ref_lo, ref_hi, c_lo, c_hi, axis, line_row1)
        if dim_g:
            dims_to_adjust.append(dim_g)
            created += 1
        return created

    grid_coord = best_grid["coord_ft"]
    grid_ref = get_grid_ref(best_grid["element"])
    if grid_ref is None:
        if DEBUG:
            output.print_md(u"   ❌ Failed to get Reference for grid {}".format(best_grid["name"]))
        dim_g = _dim_overall(ref_lo, ref_hi, c_lo, c_hi, axis, line_row1)
        if dim_g:
            dims_to_adjust.append(dim_g)
            created += 1
        return created

    tol_zero = mm_to_ft(ZERO_TOL_MM)
    tol_inter = mm_to_ft(INTERSECT_TOL_MM)

    intersects = (c_lo - tol_inter) < grid_coord < (c_hi + tol_inter)

    if DEBUG:
        output.print_md(u"   face_lo={:.0f}mm, face_hi={:.0f}mm, grid={:.0f}mm, intersects={}".format(
            ft_to_mm(c_lo), ft_to_mm(c_hi), ft_to_mm(grid_coord), intersects))
        output.print_md(u"   side={}, line_row1={:.0f}mm, line_row2={:.0f}mm".format(
            side, ft_to_mm(line_row1), ft_to_mm(line_row2)))

    if intersects:
        d_lo = abs(c_lo - grid_coord)
        d_hi = abs(c_hi - grid_coord)

        is_on_lo = d_lo <= tol_zero
        is_on_hi = d_hi <= tol_zero

        if DEBUG:
            output.print_md(u"   → INTERSECTS: d_lo={:.0f}mm, d_hi={:.0f}mm, on_lo={}, on_hi={}".format(
                ft_to_mm(d_lo), ft_to_mm(d_hi), is_on_lo, is_on_hi))

        if is_on_lo or is_on_hi:
            refs_snap = [grid_ref]
            if not is_on_lo:
                refs_snap.append(ref_lo)
            if not is_on_hi:
                refs_snap.append(ref_hi)

            if occupied_zones is not None:
                line_row1 = _adjust_perp_for_collisions(
                    axis, c_lo, c_hi, line_row1, side, occupied_zones, grid_coord)

            p0, p1 = _line_pts(c_lo, c_hi, axis, line_row1, grid_coord)
            dim1 = make_dim(refs_snap, p0, p1, "on-edge")
            if dim1:
                dims_to_adjust.append(dim1)
                created += 1
                if occupied_zones is not None:
                    _register_zone(axis, c_lo, c_hi, line_row1, occupied_zones, grid_coord)
        else:
            refs_snap = [ref_lo, grid_ref, ref_hi]

            if occupied_zones is not None:
                line_row1 = _adjust_perp_for_collisions(
                    axis, c_lo, c_hi, line_row1, side, occupied_zones, grid_coord)

            p0, p1 = _line_pts(c_lo, c_hi, axis, line_row1, grid_coord)
            dim1 = make_dim(refs_snap, p0, p1, "inside-EGE")
            if dim1:
                dims_to_adjust.append(dim1)
                created += 1
                if occupied_zones is not None:
                    _register_zone(axis, c_lo, c_hi, line_row1, occupied_zones, grid_coord)

            min_gap = mm_to_ft(OFFSET_2_MM - OFFSET_1_MM)
            if side < 0:
                line_row2 = min(line_row1 - min_gap, line_row2)
            else:
                line_row2 = max(line_row1 + min_gap, line_row2)
            if occupied_zones is not None:
                line_row2 = _adjust_perp_for_collisions(
                    axis, c_lo, c_hi, line_row2, side, occupied_zones)

            dim_g = _dim_overall(ref_lo, ref_hi, c_lo, c_hi, axis, line_row2)
            if dim_g:
                dims_to_adjust.append(dim_g)
                created += 1
                if occupied_zones is not None:
                    _register_zone(axis, c_lo, c_hi, line_row2, occupied_zones)

    else:
        if DEBUG:
            output.print_md(u"   → OUTSIDE: chain G->E->E")
        refs_chain = [grid_ref, ref_lo, ref_hi]

        span_min = min(c_lo, c_hi, grid_coord)
        span_max = max(c_lo, c_hi, grid_coord)

        if occupied_zones is not None:
            line_row1 = _adjust_perp_for_collisions(
                axis, span_min, span_max, line_row1, side, occupied_zones)

        safe_line_row1 = _avoid_collision(
            ei, line_row1, span_min, span_max,
            axis, side, all_elems or []
        )

        p0, p1 = _line_pts(c_lo, c_hi, axis, safe_line_row1, grid_coord)
        dim_chain = make_dim(refs_chain, p0, p1, "outside-GEE")
        if dim_chain:
            dims_to_adjust.append(dim_chain)
            created += 1
            if occupied_zones is not None:
                _register_zone(axis, span_min, span_max, safe_line_row1, occupied_zones)

    return created


def _dim_overall(ref_lo, ref_hi, c_lo, c_hi, axis, perp_pos):
    p0, p1 = _line_pts(c_lo, c_hi, axis, perp_pos)
    return make_dim([ref_lo, ref_hi], p0, p1, "overall")


def _line_pts(coord_lo, coord_hi, axis, perp_pos, grid_coord=None):
    lo = min(coord_lo, coord_hi)
    hi = max(coord_lo, coord_hi)
    if grid_coord is not None:
        lo = min(lo, grid_coord)
        hi = max(hi, grid_coord)

    if axis == "x":
        return XYZ(lo, perp_pos, 0), XYZ(hi, perp_pos, 0)
    else:
        return XYZ(perp_pos, lo, 0), XYZ(perp_pos, hi, 0)


def _pick_side(ei, axis, grids_parallel, forced_side=None):
    if forced_side is not None:
        if DEBUG:
            output.print_md(u"   📍 _pick_side axis={}: forced side={}".format(axis, forced_side))
        return forced_side

    if not grids_parallel:
        return -1

    if axis == "x":
        elem_center = ei["cy"]
    else:
        elem_center = ei["cx"]

    best_grid = None
    best_d = None
    best_sign = 0
    for g in grids_parallel:
        d = g["coord_ft"] - elem_center
        abs_d = abs(d)
        if best_d is None or abs_d < best_d:
            best_d = abs_d
            best_grid = g
            best_sign = d

    if best_grid is None:
        return -1

    side = -1 if best_sign < 0 else +1
    if DEBUG:
        output.print_md(u"   📍 _pick_side axis={}: parallel grid **{}** ({:.0f}mm), center={:.0f}mm → side={}".format(
            axis, best_grid["name"], ft_to_mm(best_grid["coord_ft"]),
            ft_to_mm(elem_center), side))
    return side


def _find_nearest_grid(ei, axis, grids):
    if axis == "x":
        elem_center = ei["cx"]
    else:
        elem_center = ei["cy"]
    best = None
    best_d = None
    for g in grids:
        d = abs(elem_center - g["coord_ft"])
        if best_d is None or d < best_d:
            best_d = d
            best = g
    if DEBUG:
        output.print_md(u"   🔎 _find_nearest_grid axis={}: {} grids available, center={:.0f}mm".format(
            axis, len(grids), ft_to_mm(elem_center)))
    if best_d is None:
        return None, None
    return best, best_d


def _avoid_collision(ei, perp_pos, coord_lo, coord_hi, axis, side, all_elems):
    margin = mm_to_ft(200)
    my_id = ei["element"].Id.IntegerValue
    lo = min(coord_lo, coord_hi)
    hi = max(coord_lo, coord_hi)

    for other in all_elems:
        if other["element"].Id.IntegerValue == my_id:
            continue

        if axis == "x":
            if other["max_x"] < lo or other["min_x"] > hi:
                continue
            if other["min_y"] - margin < perp_pos < other["max_y"] + margin:
                if side < 0:
                    perp_pos = min(perp_pos, other["min_y"] - margin)
                else:
                    perp_pos = max(perp_pos, other["max_y"] + margin)
        else:
            if other["max_y"] < lo or other["min_y"] > hi:
                continue
            if other["min_x"] - margin < perp_pos < other["max_x"] + margin:
                if side < 0:
                    perp_pos = min(perp_pos, other["min_x"] - margin)
                else:
                    perp_pos = max(perp_pos, other["max_x"] + margin)

    return perp_pos


COLLISION_SHIFT_MM = 300
COLLISION_MAX_PASSES = 3


def _make_zone(axis, coord_lo, coord_hi, perp_pos, height_ft=None):
    """Creates a rectangular zone for dimension collision checking."""
    if height_ft is None:
        height_ft = mm_to_ft(200)
    lo = min(coord_lo, coord_hi)
    hi = max(coord_lo, coord_hi)
    if axis == "x":
        return (lo, perp_pos - height_ft, hi, perp_pos + height_ft)
    else:
        return (perp_pos - height_ft, lo, perp_pos + height_ft, hi)


def _zone_overlaps(zone, occupied):
    """Checks if zone intersects with any of the occupied zones."""
    for oz in occupied:
        if zone[2] <= oz[0] or oz[2] <= zone[0]:
            continue
        if zone[3] <= oz[1] or oz[3] <= zone[1]:
            continue
        return True
    return False


def _adjust_perp_for_collisions(axis, coord_lo, coord_hi, perp_pos, side, occupied, grid_coord=None):
    """Shifts perp_pos until the dimension zone no longer overlaps with occupied zones."""
    lo = min(coord_lo, coord_hi)
    hi = max(coord_lo, coord_hi)
    if grid_coord is not None:
        lo = min(lo, grid_coord)
        hi = max(hi, grid_coord)

    shift = mm_to_ft(COLLISION_SHIFT_MM)
    original_perp = perp_pos
    for attempt in range(COLLISION_MAX_PASSES):
        zone = _make_zone(axis, lo, hi, perp_pos)
        if not _zone_overlaps(zone, occupied):
            break
        if DEBUG:
            output.print_md(u"   ⚠ COLLISION pass {}: perp={:.0f}mm overlaps, shifting by {}mm".format(
                attempt + 1, ft_to_mm(perp_pos), COLLISION_SHIFT_MM * (1 if side > 0 else -1)))
        perp_pos += shift * side
    if DEBUG and perp_pos != original_perp:
        output.print_md(u"   ↔ SHIFTED: {:.0f}mm → {:.0f}mm".format(
            ft_to_mm(original_perp), ft_to_mm(perp_pos)))
    return perp_pos


def _register_zone(axis, coord_lo, coord_hi, perp_pos, occupied, grid_coord=None):
    """Registers an occupied zone after a dimension is created."""
    lo = min(coord_lo, coord_hi)
    hi = max(coord_lo, coord_hi)
    if grid_coord is not None:
        lo = min(lo, grid_coord)
        hi = max(hi, grid_coord)
    zone = _make_zone(axis, lo, hi, perp_pos)
    occupied.append(zone)


def _grid_chain_exists(grids_sorted, measure_axis):
    """Checks if a dimension chain between the given grids already exists."""
    grid_ids = set()
    for g in grids_sorted:
        grid_ids.add(g["element"].Id.IntegerValue)

    if len(grid_ids) < 2:
        return False

    try:
        dims_on_view = FilteredElementCollector(doc, view.Id).OfClass(Dimension).ToElements()
    except Exception:
        return False

    for dim in dims_on_view:
        try:
            refs = dim.References
            if refs is None or refs.Size < 2:
                continue

            dim_ref_ids = set()
            for ref in refs:
                eid = ref.ElementId.IntegerValue
                dim_ref_ids.add(eid)

            if grid_ids.issubset(dim_ref_ids):
                if DEBUG:
                    output.print_md(u"   ⏭ Grid chain already exists (dim id={})".format(
                        dim.Id.IntegerValue))
                return True
        except Exception:
            continue

    return False


def make_grid_chain(grids_sorted, measure_axis, offset_mm):
    """Creates a dimension chain between grids."""
    if len(grids_sorted) < 2:
        return 0

    if _grid_chain_exists(grids_sorted, measure_axis):
        return 0

    refs = []
    for g in grids_sorted:
        r = get_grid_ref(g["element"])
        if r:
            refs.append(r)
    if len(refs) < 2:
        return 0

    bubble_coord, bubble_side = _get_bubble_baseline(grids_sorted, measure_axis)
    existing_offset_ft = _find_existing_grid_dim_offset(grids_sorted, measure_axis, bubble_side)

    off = mm_to_ft(offset_mm)

    if bubble_side > 0:
        base = max(bubble_coord, existing_offset_ft) if existing_offset_ft is not None else bubble_coord
        perp = base + off
    else:
        base = min(bubble_coord, existing_offset_ft) if existing_offset_ft is not None else bubble_coord
        perp = base - off

    if measure_axis == "x":
        p0 = XYZ(grids_sorted[0]["coord_ft"], perp, 0)
        p1 = XYZ(grids_sorted[-1]["coord_ft"], perp, 0)
    else:
        p0 = XYZ(perp, grids_sorted[0]["coord_ft"], 0)
        p1 = XYZ(perp, grids_sorted[-1]["coord_ft"], 0)

    if DEBUG:
        output.print_md(u"   📏 chain {}: perp={:.0f}mm, bubble_side={}, base={:.0f}mm (exist={})".format(
            measure_axis, ft_to_mm(perp), bubble_side, ft_to_mm(bubble_coord),
            u"{:.0f}mm".format(ft_to_mm(existing_offset_ft)) if existing_offset_ft is not None else "none"))

    ra = ReferenceArray()
    for r in refs:
        ra.Append(r)
    try:
        dim = doc.Create.NewDimension(view, Line.CreateBound(p0, p1), ra)
        return 1 if dim else 0
    except Exception as e:
        if DEBUG:
            output.print_md(u"⚠ chain: {}".format(str(e)))
        return 0


def _get_bubble_baseline(grids_sorted, measure_axis):
    """Determines the bubble-end coordinate and shift direction."""
    bubble_coords = []
    non_bubble_coords = []

    for g in grids_sorted:
        be = g.get("bubble_end", "p0")
        bp = g[be]
        nbp = g["p1"] if be == "p0" else g["p0"]

        if measure_axis == "x":
            bubble_coords.append(bp.Y)
            non_bubble_coords.append(nbp.Y)
        else:
            bubble_coords.append(bp.X)
            non_bubble_coords.append(nbp.X)

    avg_bubble = sum(bubble_coords) / len(bubble_coords)
    avg_non_bubble = sum(non_bubble_coords) / len(non_bubble_coords)

    if avg_bubble > avg_non_bubble:
        bubble_edge = max(bubble_coords)
        return bubble_edge, -1
    else:
        bubble_edge = min(bubble_coords)
        return bubble_edge, +1


def _find_existing_grid_dim_offset(grids_sorted, measure_axis, side=1):
    """Finds the position of existing grid dimension chains to avoid overlap."""
    grid_ids = set(g["element"].Id.IntegerValue for g in grids_sorted)

    try:
        dims_on_view = FilteredElementCollector(doc, view.Id).OfClass(Dimension).ToElements()
    except Exception:
        return None

    best_perp = None

    for dim in dims_on_view:
        try:
            refs = dim.References
            if refs is None or refs.Size < 2:
                continue

            match_count = 0
            for ref in refs:
                if ref.ElementId.IntegerValue in grid_ids:
                    match_count += 1

            if match_count < 2:
                continue

            crv = dim.Curve
            if crv and isinstance(crv, Line):
                if measure_axis == "x":
                    if side > 0:
                        perp = max(crv.GetEndPoint(0).Y, crv.GetEndPoint(1).Y)
                        if best_perp is None or perp > best_perp:
                            best_perp = perp
                    else:
                        perp = min(crv.GetEndPoint(0).Y, crv.GetEndPoint(1).Y)
                        if best_perp is None or perp < best_perp:
                            best_perp = perp
                else:
                    if side > 0:
                        perp = max(crv.GetEndPoint(0).X, crv.GetEndPoint(1).X)
                        if best_perp is None or perp > best_perp:
                            best_perp = perp
                    else:
                        perp = min(crv.GetEndPoint(0).X, crv.GetEndPoint(1).X)
                        if best_perp is None or perp < best_perp:
                            best_perp = perp
        except Exception:
            continue

    return best_perp


def place_all_guides(notify=True):
    """Auto-detect exterior walls and place every guide line: 3 exterior tiers
    on all 4 sides + one horizontal and one vertical interior guide. With
    ``notify`` the user is told to adjust the guides and re-run; when called as
    part of the one-run path it stays silent. Returns (n_ext, n_int) or None on
    failure."""
    _glog_reset()
    _glog(u"=== place_all_guides ===")
    _reset_hosted_openings_cache()
    all_elems = collect_walls_in_view()
    _glog(u"walls in view: {}".format(len(all_elems)))
    if not all_elems:
        forms.alert(u"No walls found in this view.", title=__title__)
        return None

    ext_wall_ids = _compute_exterior_wall_ids(all_elems)
    _glog(u"exterior wall ids: {}".format(len(ext_wall_ids)))
    _diag_dump_model(all_elems, ext_wall_ids)

    # Clear any leftover guides from a previous run, then place the exterior
    # ones. Committed on its own so they are visible while the user clicks the
    # interior positions -- and because PickPoint cannot run inside an open
    # transaction.
    stale = _find_guide_lines() + _find_interior_guides()
    _glog(u"stale guides cleared: {}".format(len(stale)))
    t = Transaction(doc, u"AutoDimension exterior guides")
    t.Start()
    try:
        if stale:
            _delete_guides(stale)
        n_ext = create_exterior_guides(all_elems, ext_wall_ids)
        t.Commit()
    except Exception as e:
        t.RollBack()
        _glog(u"guide creation EXCEPTION: {}".format(str(e)))
        forms.alert(u"Failed to create guide lines:\n{}".format(str(e)),
                    title=__title__)
        return None

    # Now the interior guides. Interactive placement draws each line as it is
    # clicked and runs its own per-line transactions, so nothing may be open
    # here (PickPoint refuses to run inside a transaction).
    n_int = 0
    if INTERACTIVE_INTERIOR_GUIDES:
        try:
            n_int = pick_and_create_interior_guides(all_elems)
        except Exception as e:
            _glog(u"interior pick EXCEPTION: {}".format(str(e)))
            n_int = 0
    else:
        t = Transaction(doc, u"AutoDimension interior guides")
        t.Start()
        try:
            n_int = create_interior_guides(all_elems)
            t.Commit()
        except Exception as e:
            t.RollBack()
            _glog(u"interior guide EXCEPTION: {}".format(str(e)))
            n_int = 0
    _glog(u"guides created: ext={} int={}".format(n_ext, n_int))

    if notify:
        forms.alert(
            u"Placed {} exterior guide lines (3 tiers x 4 sides) and {} interior "
            u"guide lines (1 horizontal + 1 vertical).\n\n"
            u"Move or copy the guides where you want dimensions, then run "
            u"AutoDimension 1 again to create the dimensions.".format(
                n_ext, n_int),
            title=__title__)
    return (n_ext, n_int)


def create_all_dimensions(notify=True):
    """Read the guide lines and build exterior + interior dimensions along them,
    then delete the guides. Runs as an ordinary command, so the transaction
    context is always valid (no modeless / ExternalEvent)."""
    if not isinstance(view, ViewPlan):
        forms.alert(u"Please open a plan view.", title=__title__)
        return

    _reset_hosted_openings_cache()
    all_elems = collect_walls_in_view()
    ext_guides = _find_guide_lines()
    int_guides = _find_interior_guides()
    if not ext_guides and not int_guides:
        forms.alert(u"No guide lines found. Run AutoDimension 1 to place "
                    u"guides first.", title=__title__)
        return

    ext_wall_ids = _compute_exterior_wall_ids(all_elems)
    fh = DimFailureSwallower()
    t = Transaction(doc, u"AutoDimension create")
    opts = t.GetFailureHandlingOptions()
    opts.SetFailuresPreprocessor(fh)
    t.SetFailureHandlingOptions(opts)
    t.Start()
    try:
        global _AD_DIM_TYPE_ID
        _AD_DIM_TYPE_ID = _autodim_dim_type_id()
        # Remove this tool's dimensions/notes from earlier runs so strings don't
        # stack (leaves the user's own and grid dimensions untouched).
        _delete_previous_autodim()
        dims = []
        n_ext = 0
        n_int = 0
        if ext_guides:
            n_ext = dimension_along_guides(
                all_elems, ext_wall_ids, ext_guides, dims)
        if int_guides:
            n_int = dimension_along_interior_guides(all_elems, int_guides, dims)
        if ANNOTATE_OPENING_HEIGHTS:
            annotate_opening_heights(all_elems, ext_wall_ids)
        _delete_guides(ext_guides + int_guides)
        doc.Regenerate()
        for d in dims:
            _displace_small_texts(d)
        # Purge the AUTODIM_* line styles so they don't accumulate in the model.
        _purge_guide_line_styles()
        t.Commit()
    except Exception as e:
        t.RollBack()
        forms.alert(u"Error creating dimensions:\n{}".format(str(e)),
                    title=__title__)
        return

    if notify:
        forms.alert(
            u"Created {} exterior + {} interior dimensions.".format(
                n_ext, n_int),
            title=__title__)


def _cleanup_guides_only():
    """Delete all guide lines and purge their styles without dimensioning
    (the Cancel path). Runs in its own transaction."""
    guides = _find_guide_lines() + _find_interior_guides()
    if not guides:
        return
    t = Transaction(doc, u"AutoDimension cancel")
    t.Start()
    try:
        _delete_guides(guides)
        _purge_guide_line_styles()
        t.Commit()
    except Exception:
        t.RollBack()


# ---------------------------------------------------------------------------
# One-run modeless flow: place guides -> floating window waits while the user
# adjusts them -> button creates the dimensions. Revit API calls from a modeless
# window must run in a valid context, so the button only sets a request flag and
# Revit's own Idling event (always an active context) does the transaction on the
# next tick. Needs __persistentengine__ so the handler survives past main().
# ---------------------------------------------------------------------------

_AD_WINDOW = None
_AD_REQUEST = None          # None | "create" | "cancel"
_AD_IDLING_HOOKED = False


def _ad_uiapp():
    from pyrevit import HOST_APP
    return HOST_APP.uiapp


def _on_idling(sender, args):
    """Runs on every Revit idle tick; does work only when a button set a
    request. Idling is a valid context for a transaction."""
    global _AD_REQUEST, _AD_WINDOW
    req = _AD_REQUEST
    if req is None:
        return
    _AD_REQUEST = None  # consume before doing work, so it runs once
    try:
        if req == u"cancel":
            _cleanup_guides_only()
        else:
            create_all_dimensions(notify=False)
    except Exception as ex:
        try:
            forms.alert(u"Error creating dimensions:\n{}".format(str(ex)),
                        title=__title__)
        except Exception:
            pass
    try:
        if _AD_WINDOW is not None:
            _AD_WINDOW.Close()
    except Exception:
        pass
    _AD_WINDOW = None


def _hook_idling():
    """Subscribe to Idling once for the engine lifetime."""
    global _AD_IDLING_HOOKED
    if _AD_IDLING_HOOKED:
        return
    _ad_uiapp().Idling += _on_idling
    _AD_IDLING_HOOKED = True


class AutoDimWindow(forms.WPFWindow):
    """Modeless floating window with Create Dimensions / Cancel buttons. The
    buttons only set a request flag -- the Idling handler does the model work."""

    def __init__(self, xaml_path):
        forms.WPFWindow.__init__(self, xaml_path)

    def _request(self, action, busy_text):
        global _AD_REQUEST
        try:
            self.status.Text = busy_text
        except Exception:
            pass
        _AD_REQUEST = action

    def create_click(self, sender, args):
        self._request(u"create", u"Creating dimensions...")

    def cancel_click(self, sender, args):
        self._request(u"cancel", u"Removing guides...")


def _open_modeless_window():
    """Show the modeless window and arm the Idling handler. Returns True on
    success."""
    global _AD_WINDOW
    try:
        _hook_idling()
        xaml_path = os.path.join(os.path.dirname(__file__), u"AutoDimWindow.xaml")
        win = AutoDimWindow(xaml_path)
        _AD_WINDOW = win
        win.show()
        return True
    except Exception as ex:
        try:
            output.print_md(u"modeless window failed: {}".format(str(ex)))
        except Exception:
            pass
        return False


def main():
    """One-run floating-window flow: place the bold-red guide lines, then open a
    modeless window that waits while the user moves/copies the guides. Clicking
    Create Dimensions builds the dimensions and deletes the guides."""
    if not isinstance(view, ViewPlan):
        forms.alert(u"Please open a plan view.", title=__title__)
        return

    # If the window is already open, don't disturb the guides being adjusted --
    # just bring it forward. If it was closed via the X, start a fresh cycle.
    global _AD_WINDOW
    if _AD_WINDOW is not None:
        still_open = False
        try:
            still_open = bool(_AD_WINDOW.IsVisible)
        except Exception:
            still_open = False
        if still_open:
            try:
                _AD_WINDOW.Activate()
            except Exception:
                pass
            return
        _AD_WINDOW = None

    placed = place_all_guides(notify=False)
    if placed is None:
        return
    n_ext, n_int = placed

    if not _open_modeless_window():
        # Window unavailable: leave the guides in place (don't delete them).
        forms.alert(
            u"Placed {} exterior + {} interior guide lines, but the floating "
            u"button could not open. Reload pyRevit and run AutoDimension 1 "
            u"again.".format(n_ext, n_int),
            title=__title__)


if __name__ == "__main__":
    main()
