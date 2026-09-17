# -*- coding: utf-8 -*-
__title__ = u"ext_dim_oai"
__doc__ = u"""Auto-dimensioning for exterior and interior walls.
v6 - exterior three-tier dimensions plus multi-line interior dimensions.

Usage:
1. Click the button
2. Select the exterior walls (grids and columns may also be selected)
3. Press Enter
4. Pick multiple horizontal reference points, then press Enter
5. Pick multiple vertical reference points, then press Enter

Scenarios:
1. Grid inside element -> snap E->G->E (row 1) + overall E->E (row 2)
2. Grid on edge -> snap G->E (acts as overall, row 1)
3. Grid outside element -> single chain G->E->E (row 1)
"""

import clr

clr.AddReference("RevitAPI")
clr.AddReference("RevitAPIUI")

from Autodesk.Revit.DB import (
    FilteredElementCollector, BuiltInCategory,
    Dimension, Grid, FamilyInstance, Wall,
    XYZ, Line, Reference, ReferenceArray,
    ElementId, Transaction, TransactionGroup,
    Options, ViewPlan,
    PlanarFace, Solid, GeometryInstance,
    FailureProcessingResult, IFailuresPreprocessor,
    DatumEnds,
    HostObjectUtils, ShellLayerType,
    FamilyInstanceReferenceType,
)
from Autodesk.Revit.UI.Selection import ObjectType
from pyrevit import forms, script


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


doc = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument
view = doc.ActiveView
output = script.get_output()

OFFSET_1_MM = 800    # Layer 1: opening dims, offset from wall face
OFFSET_2_MM = 1400   # Layer 2: facade detail chain, offset from building edge
OFFSET_3_MM = 2200   # Layer 3: overall chain, offset from building edge

OFFSET_CHAIN_1_MM = 1500
OFFSET_CHAIN_GAP_MM = 700

ZERO_TOL_MM = 5
INTERSECT_TOL_MM = 50
MAX_SNAP_DIST_MM = 10000

DEBUG = True


def mm_to_ft(mm):
    return mm / 304.8


def ft_to_mm(ft):
    return ft * 304.8


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
    except Exception:
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
        if DEBUG and dim:
            output.print_md(u"   ✅ Dimension created (id={})".format(dim.Id.IntegerValue))
        return dim
    except Exception as e:
        if DEBUG:
            output.print_md(u"   ❌ make_dim ERROR [{}]: **{}**".format(label, str(e)))
        return None


def _displace_small_texts(dim):
    try:
        scale = view.Scale
    except Exception:
        scale = 100

    text_width_mm = 5.0 * scale
    displace_mm = text_width_mm

    try:
        crv = dim.Curve
        if not crv or not isinstance(crv, Line):
            return
        direction = crv.Direction.Normalize()
    except Exception:
        return

    try:
        segs = list(dim.Segments)
        if segs and len(segs) > 0:
            for i, seg in enumerate(segs):
                try:
                    val = seg.Value
                    if val is None:
                        continue
                    val_mm = ft_to_mm(val)
                    if val_mm >= text_width_mm:
                        continue

                    if not seg.IsTextPositionAdjustable():
                        continue

                    tp = seg.TextPosition
                    if tp is None:
                        continue

                    sign = -1.0 if i == 0 else 1.0
                    offset_ft = mm_to_ft(displace_mm)
                    new_tp = XYZ(
                        tp.X + direction.X * offset_ft * sign,
                        tp.Y + direction.Y * offset_ft * sign,
                        tp.Z,
                    )
                    seg.TextPosition = new_tp
                except Exception:
                    continue
            return
    except Exception:
        pass

    try:
        val = dim.Value
        if val is None:
            return
        val_mm = ft_to_mm(val)
        if val_mm >= text_width_mm:
            return

        if not dim.IsTextPositionAdjustable():
            return

        tp = dim.TextPosition
        if tp is None:
            return

        offset_ft = mm_to_ft(displace_mm)
        new_tp = XYZ(
            tp.X + direction.X * offset_ft,
            tp.Y + direction.Y * offset_ft,
            tp.Z,
        )
        dim.TextPosition = new_tp
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


def _collect_opening_face_refs(wall, run_axis):
    """
    Returns (ref, coord) for Left/Right jamb references of every hosted
    door/window.  Primary: FamilyInstanceReferenceType.Left/Right.
    Fallback: geometry traversal without opt.View.
    """
    results = []
    try:
        dep_ids = wall.GetDependentElements(None)
    except Exception:
        return results

    door_cat = int(BuiltInCategory.OST_Doors)
    win_cat  = int(BuiltInCategory.OST_Windows)

    for dep_id in dep_ids:
        elem = doc.GetElement(dep_id)
        if elem is None or not isinstance(elem, FamilyInstance):
            continue
        try:
            cat_id = elem.Category.Id.IntegerValue
        except Exception:
            continue
        if cat_id not in (door_cat, win_cat):
            continue

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
                for loop in face.EdgeLoops:
                    for edge in loop:
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
    """Pick points until Enter/Escape cancels the current point prompt."""
    points = []
    while True:
        try:
            points.append(uidoc.Selection.PickPoint(prompt))
        except Exception:
            break
    return points


def _interior_dimension_at_point(point, axis, wall_infos, dims_to_adjust, index):
    """Dimension every wall face crossed by one user-selected reference line."""
    face_pairs = []
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
        except Exception:
            continue

    if len(face_pairs) < 2:
        if DEBUG:
            output.print_md(u"   ⏭ Interior {} point: fewer than two wall faces".format(axis))
        return 0

    face_pairs.sort(key=lambda item: item[1])
    deduped = []
    dedup_tolerance = mm_to_ft(1)
    for ref, coord in face_pairs:
        if not deduped or abs(coord - deduped[-1][1]) > dedup_tolerance:
            deduped.append((ref, coord))

    if len(deduped) < 2:
        return 0

    refs = [ref for ref, unused_coord in deduped]
    low = deduped[0][1]
    high = deduped[-1][1]
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


def _compute_exterior_wall_ids(all_elems):
    """
    Returns a set of wall element IDs that have at least one exterior face
    (checked against both axes).  Used to exclude core/interior walls from
    the opening-dimension pass.
    """
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


def main():
    if not isinstance(view, ViewPlan):
        forms.alert(u"Please open a plan view.", title=__title__)
        return

    try:
        sel_refs = uidoc.Selection.PickObjects(
            ObjectType.Element,
            u"Select the exterior walls (grids/columns optional), then press Enter"
        )
    except Exception:
        return

    if not sel_refs:
        forms.alert(u"Nothing selected.", title=__title__)
        return

    selected_elements = [doc.GetElement(r.ElementId) for r in sel_refs]

    all_grids = collect_grids_from_selection(selected_elements)
    all_elems = collect_elements_from_selection(selected_elements)

    h_grids = sorted([g for g in all_grids if g["orientation"] == "horizontal"],
                     key=lambda g: g["coord_ft"])
    v_grids = sorted([g for g in all_grids if g["orientation"] == "vertical"],
                     key=lambda g: g["coord_ft"])

    if DEBUG:
        n_walls = sum(1 for e in all_elems if e["category"] == "Wall")
        n_cols = sum(1 for e in all_elems if e["category"] == "Column")
        output.print_md(u"## Data (from selection)")
        output.print_md(u"- H grids: **{}** ({})".format(
            len(h_grids), u", ".join(g["name"] for g in h_grids)))
        output.print_md(u"- V grids: **{}** ({})".format(
            len(v_grids), u", ".join(g["name"] for g in v_grids)))
        output.print_md(u"- Elements: **{}** (walls: {}, columns: {})".format(
            len(all_elems), n_walls, n_cols))

    if not all_elems:
        forms.alert(u"No walls or columns in the selection.", title=__title__)
        return

    # Point picking happens before the transaction because Revit does not allow
    # interactive selection while a document transaction is open.
    interior_walls = collect_walls_in_view()
    horizontal_points = _pick_reference_points(
        u"Pick horizontal interior dimension lines; press Enter to switch to vertical"
    )
    vertical_points = _pick_reference_points(
        u"Pick vertical interior dimension lines; press Enter to finish"
    )

    if DEBUG:
        output.print_md(u"- Interior horizontal lines: **{}**".format(len(horizontal_points)))
        output.print_md(u"- Interior vertical lines: **{}**".format(len(vertical_points)))
        output.print_md(u"- Visible walls available for interior dimensions: **{}**".format(
            len(interior_walls)))

    tg = TransactionGroup(doc, __title__)
    tg.Start()
    total = 0
    failure_handler = DimFailureSwallower()

    try:
        t1 = Transaction(doc, u"Chains")
        opts1 = t1.GetFailureHandlingOptions()
        opts1.SetFailuresPreprocessor(failure_handler)
        t1.SetFailureHandlingOptions(opts1)
        t1.Start()
        n_chains = 0
        if len(v_grids) >= 2:
            n_chains += make_grid_chain(v_grids, "x", OFFSET_CHAIN_1_MM)
            if len(v_grids) > 2:
                n_chains += make_grid_chain(
                    [v_grids[0], v_grids[-1]], "x",
                    OFFSET_CHAIN_1_MM + OFFSET_CHAIN_GAP_MM)
        if len(h_grids) >= 2:
            n_chains += make_grid_chain(h_grids, "y", OFFSET_CHAIN_1_MM)
            if len(h_grids) > 2:
                n_chains += make_grid_chain(
                    [h_grids[0], h_grids[-1]], "y",
                    OFFSET_CHAIN_1_MM + OFFSET_CHAIN_GAP_MM)
        t1.Commit()
        total += n_chains
        if DEBUG:
            output.print_md(u"✅ Grid chains: **{}**".format(n_chains))

        t2 = Transaction(doc, u"Snaps+Overalls")
        opts2 = t2.GetFailureHandlingOptions()
        opts2.SetFailuresPreprocessor(failure_handler)
        t2.SetFailureHandlingOptions(opts2)
        t2.Start()
        n_x = 0
        n_y = 0
        dims_to_adjust = []
        occupied_zones = []

        # Identify exterior walls once — used to filter both chains and openings
        ext_wall_ids = _compute_exterior_wall_ids(all_elems)
        if DEBUG:
            output.print_md(u"🏢 Exterior walls detected: **{}** / {}".format(
                len(ext_wall_ids), len(all_elems)))

        if not h_grids and not v_grids:
            # No grids — Layer 2 (facade detail) + Layer 3 (overall) chains
            n_chains = make_building_chains(all_elems, dims_to_adjust)
            n_x += n_chains
        else:
            for ei in all_elems:
                side_x = _pick_side(ei, "x", h_grids)
                side_y = side_x

                try:
                    n_x += dim_along_axis(ei, "x", v_grids, h_grids, all_elems, dims_to_adjust,
                                          forced_side=side_x, occupied_zones=occupied_zones)
                except Exception as e:
                    if DEBUG:
                        output.print_md(u"⚠ Error on X axis: {}".format(str(e)))
                try:
                    n_y += dim_along_axis(ei, "y", h_grids, v_grids, all_elems, dims_to_adjust,
                                          forced_side=side_y, occupied_zones=occupied_zones)
                except Exception as e:
                    if DEBUG:
                        output.print_md(u"⚠ Error on Y axis: {}".format(str(e)))

        # Layer 1: along-wall opening dimensions — exterior walls only
        n_openings = 0
        for ei in all_elems:
            if ei["element"].Id.IntegerValue not in ext_wall_ids:
                continue  # skip core/interior walls
            try:
                n_openings += dim_wall_with_openings(ei, dims_to_adjust)
            except Exception as e:
                if DEBUG:
                    output.print_md(u"⚠ along-wall error (id={}): {}".format(
                        ei["element"].Id.IntegerValue, str(e)))
        n_x += n_openings

        # Interior pass: one chain for each horizontal and vertical reference
        # point selected by the user.
        n_interior = make_interior_dimensions(
            horizontal_points, vertical_points, interior_walls, dims_to_adjust
        )

        doc.Regenerate()

        for d in dims_to_adjust:
            _displace_small_texts(d)

        t2.Commit()
        total += n_x + n_y + n_interior
        if DEBUG:
            output.print_md(u"✅ Dimensions along X: **{}**, along Y: **{}**".format(n_x, n_y))
            output.print_md(u"✅ Interior dimensions: **{}**".format(n_interior))
            if occupied_zones:
                output.print_md(u"📦 Reserved zones: **{}**".format(len(occupied_zones)))
            if failure_handler.had_errors:
                output.print_md(u"⚠ Revit errors (auto-resolved): **{}**".format(len(failure_handler.had_errors)))
                for err_msg in failure_handler.had_errors:
                    output.print_md(u"   - {}".format(err_msg))

        tg.Assimilate()

    except Exception as e:
        tg.RollBack()
        forms.alert(u"Error:\n{}".format(str(e)), title=__title__)
        return

    output.print_md(u"---")
    output.print_md(u"## Result: **{}** dimensions created".format(total))


if __name__ == "__main__":
    main()
