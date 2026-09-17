# -*- coding: utf-8 -*-
"""
create_dimensions_util.py
Programmatic dimension creation matching the MCP create_dimensions tool interface.

Usage:
    from create_dimensions_util import create_dimensions

    results = create_dimensions(doc, view, [
        {
            "startPoint": {"x": 0,    "y": 0, "z": 0},
            "endPoint":   {"x": 5000, "y": 0, "z": 0},
            # optional:
            "linePoint":  {"x": 2500, "y": 800, "z": 0},
            "elementIds": [123, 456],   # dimension between specific elements
            "dimensionStyleId": -1,     # -1 = default style
            "viewId": -1,               # -1 = active view
        }
    ])
    # returns list of element IDs of created dimensions

All coordinates in millimetres. IronPython 2.7 compatible.
"""

import clr
clr.AddReference("RevitAPI")

from Autodesk.Revit.DB import (
    FilteredElementCollector,
    XYZ, Line, Reference, ReferenceArray,
    ElementId, Transaction, Options,
    PlanarFace, Solid, GeometryInstance,
    FamilyInstance, Wall,
    DimensionType,
    LocationPoint, LocationCurve,
)
import math

MM_TO_FT = 1.0 / 304.8
_SNAP_TOLERANCE_FT = 5.0  # 5-foot search radius for auto-detect


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_dimensions(doc, view, dimensions):
    """
    Create dimensions from a list of dicts.

    Each dict may contain:
        startPoint      dict {x, y, z} in mm  (required)
        endPoint        dict {x, y, z} in mm  (required)
        linePoint       dict {x, y, z} in mm  (optional, defaults to midpoint+offset)
        elementIds      list[int]              (optional, auto-detect if omitted)
        dimensionStyleId int                   (optional, -1 = default)
        viewId          int                    (optional, -1 = active view)

    Returns list of integer element IDs of successfully created Dimension elements.
    """
    created_ids = []

    for dim_info in dimensions:
        # resolve view
        target_view = view
        view_id = dim_info.get("viewId", -1)
        if view_id and view_id > 0:
            elem = doc.GetElement(ElementId(view_id))
            if elem is not None:
                target_view = elem

        with Transaction(doc, "Create Dimension") as t:
            t.Start()
            try:
                dim = _create_single(doc, target_view, dim_info)
                if dim is not None:
                    created_ids.append(dim.Id.IntegerValue)
                t.Commit()
            except Exception:
                t.RollBack()
                raise

    return created_ids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pt(coord_dict):
    """Convert mm dict to Revit XYZ (feet)."""
    return XYZ(
        coord_dict["x"] * MM_TO_FT,
        coord_dict["y"] * MM_TO_FT,
        coord_dict["z"] * MM_TO_FT,
    )


def _default_line_point(start, end):
    """Midpoint offset 1 foot in Y (same default as C# source)."""
    return XYZ(
        (start.X + end.X) / 2.0,
        (start.Y + end.Y) / 2.0 + 1.0,
        (start.Z + end.Z) / 2.0,
    )


def _create_single(doc, view, dim_info):
    start = _pt(dim_info["startPoint"])
    end = _pt(dim_info["endPoint"])

    line_pt_raw = dim_info.get("linePoint")
    line_pt = _pt(line_pt_raw) if line_pt_raw else _default_line_point(start, end)

    element_ids = dim_info.get("elementIds") or []
    dim_direction = (end - start).Normalize()

    ref_array = ReferenceArray()

    if element_ids:
        # --- element-based: extract face references from specified elements ---
        for eid in element_ids:
            elem = doc.GetElement(ElementId(eid))
            if elem is None:
                continue
            for ref in _get_element_references(elem, view, dim_direction):
                ref_array.Append(ref)
    else:
        # --- point-based: auto-detect nearest element at each endpoint ---
        start_ref = _find_reference_at_point(doc, view, start, dim_direction)
        end_ref = _find_reference_at_point(doc, view, end, dim_direction)
        if start_ref:
            ref_array.Append(start_ref)
        if end_ref:
            ref_array.Append(end_ref)

    if ref_array.Size < 2:
        return None

    line = Line.CreateBound(start, end)
    dim = doc.Create.NewDimension(view, line, ref_array)

    if dim is None:
        return None

    # apply dimension style if specified
    style_id = dim_info.get("dimensionStyleId", -1)
    if style_id and style_id > 0:
        dim_type = doc.GetElement(ElementId(style_id))
        if isinstance(dim_type, DimensionType):
            dim.DimensionType = dim_type

    return dim


def _get_element_references(elem, view, dim_direction=None):
    """
    Return a list of face References for an element, preferring the face
    whose normal is most aligned with dim_direction (for walls & families).
    Falls back to a generic Reference if geometry is unavailable.
    """
    refs = []

    if isinstance(elem, (Wall, FamilyInstance)):
        opt = Options()
        opt.View = view
        opt.ComputeReferences = True
        geo = elem.get_Geometry(opt)
        if geo is None:
            refs.append(Reference(elem))
            return refs

        best_ref = None
        best_score = -1.0

        for obj in geo:
            solids = []
            if isinstance(obj, Solid) and obj.Faces.Size > 0:
                solids.append(obj)
            elif isinstance(obj, GeometryInstance) and isinstance(elem, FamilyInstance):
                sym_geo = obj.GetSymbolGeometry()
                if sym_geo:
                    for sub in sym_geo:
                        if isinstance(sub, Solid) and sub.Faces.Size > 0:
                            solids.append(sub)

            for solid in solids:
                for face in solid.Faces:
                    if not isinstance(face, PlanarFace):
                        continue
                    ref = face.Reference
                    if ref is None:
                        continue
                    normal = face.FaceNormal

                    # skip top/bottom faces
                    if abs(normal.Z) > 0.9:
                        continue

                    if dim_direction is not None:
                        score = abs(normal.DotProduct(dim_direction))
                        if score > best_score:
                            best_score = score
                            best_ref = ref
                    else:
                        # no direction hint — take first vertical face
                        refs.append(ref)
                        return refs

        if best_ref is not None:
            refs.append(best_ref)
        elif not refs:
            refs.append(Reference(elem))

    else:
        refs.append(Reference(elem))

    return refs


def _find_reference_at_point(doc, view, point, dim_direction=None):
    """
    Find a face reference by locating the nearest element to *point* in the view.
    Search radius: _SNAP_TOLERANCE_FT feet.
    """
    collector = (
        FilteredElementCollector(doc, view.Id)
        .WhereElementIsNotElementType()
        .ToElements()
    )

    closest_elem = None
    min_dist = _SNAP_TOLERANCE_FT

    for elem in collector:
        loc = elem.Location
        if loc is None:
            continue

        elem_pt = None
        if isinstance(loc, LocationPoint):
            elem_pt = loc.Point
        elif isinstance(loc, LocationCurve):
            try:
                result = loc.Curve.Project(point)
                elem_pt = result.XYZPoint
            except Exception:
                continue
        else:
            continue

        dist = point.DistanceTo(elem_pt)
        if dist < min_dist:
            min_dist = dist
            closest_elem = elem

    if closest_elem is None:
        return None

    refs = _get_element_references(closest_elem, view, dim_direction)
    return refs[0] if refs else None
