# -*- coding: utf-8 -*-
__title__ = "D_W_detection"
__doc__ = "Pick a DWG import, detect A-DOORS/A-WINDOWS blocks, and draw opening markers."

import math
import os
import re
import sys

from Autodesk.Revit.DB import (
    BuiltInCategory,
    FilteredElementCollector,
    ImportInstance,
    Line,
    Dimension,
    ReferenceArray,
    TextNote,
    TextNoteType,
    Transaction,
    XYZ,
)
from Autodesk.Revit.Exceptions import OperationCanceledException
from Autodesk.Revit.UI import TaskDialog
from Autodesk.Revit.UI.Selection import ISelectionFilter, ObjectType


uidoc = __revit__.ActiveUIDocument
doc = uidoc.Document

CM_PER_FT = 30.48


def cm_to_ft(v):
    return float(v) / CM_PER_FT


def _vlen(x, y):
    return math.sqrt((x * x) + (y * y))


def _vnorm(x, y):
    ln = _vlen(x, y)
    if ln <= 1.0e-9:
        return (1.0, 0.0)
    return (x / ln, y / ln)


def _pt_dist(a, b):
    return _vlen(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _closest_point_on_segment(px, py, ax, ay, bx, by):
    vx = float(bx) - float(ax)
    vy = float(by) - float(ay)
    vv = (vx * vx) + (vy * vy)
    if vv <= 1.0e-9:
        return (float(ax), float(ay), 0.0)
    t = (((float(px) - float(ax)) * vx) + ((float(py) - float(ay)) * vy)) / vv
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    qx = float(ax) + (vx * t)
    qy = float(ay) + (vy * t)
    return (qx, qy, t)


def _line_bbox(ln):
    x1 = float(ln.get("x1", 0.0))
    y1 = float(ln.get("y1", 0.0))
    x2 = float(ln.get("x2", 0.0))
    y2 = float(ln.get("y2", 0.0))
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _arc_bbox(arc):
    cx = float(arc.get("cx", 0.0))
    cy = float(arc.get("cy", 0.0))
    r = abs(float(arc.get("r", 0.0)))
    return (cx - r, cy - r, cx + r, cy + r)


def _bbox_center(bb):
    return ((bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5)


def _bbox_size(bb):
    return (max(0.0, bb[2] - bb[0]), max(0.0, bb[3] - bb[1]))


def _merge_bbox(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _bbox_dist(a, b):
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return _vlen(dx, dy)


def _segment_intersection(ax, ay, bx, by, cx, cy, dx, dy):
    ax = float(ax)
    ay = float(ay)
    bx = float(bx)
    by = float(by)
    cx = float(cx)
    cy = float(cy)
    dx = float(dx)
    dy = float(dy)

    r1x = bx - ax
    r1y = by - ay
    r2x = dx - cx
    r2y = dy - cy
    den = (r1x * r2y) - (r1y * r2x)
    if abs(den) <= 1.0e-9:
        return None

    qpx = cx - ax
    qpy = cy - ay
    t = ((qpx * r2y) - (qpy * r2x)) / den
    u = ((qpx * r1y) - (qpy * r1x)) / den
    if t < -1.0e-6 or t > 1.0 + 1.0e-6:
        return None
    if u < -1.0e-6 or u > 1.0 + 1.0e-6:
        return None
    return (ax + (t * r1x), ay + (t * r1y))


def _cluster_points(points, tol_cm):
    out = []
    for p in (points or []):
        x, y = float(p[0]), float(p[1])
        hit = False
        for i in range(len(out)):
            d = _pt_dist((x, y), out[i])
            if d <= tol_cm:
                out[i] = ((out[i][0] + x) * 0.5, (out[i][1] + y) * 0.5)
                hit = True
                break
        if not hit:
            out.append((x, y))
    return out


def _collect_wall_corners(wall_lines):
    pts = []
    lines = []
    for ln in (wall_lines or []):
        x1 = float(ln.get("x1", 0.0))
        y1 = float(ln.get("y1", 0.0))
        x2 = float(ln.get("x2", 0.0))
        y2 = float(ln.get("y2", 0.0))
        if _vlen(x2 - x1, y2 - y1) < 20.0:
            continue
        pts.append((x1, y1))
        pts.append((x2, y2))
        lines.append((x1, y1, x2, y2))

    # Add intersection corners between non-parallel wall segments.
    for i in range(len(lines)):
        ax, ay, bx, by = lines[i]
        u1 = _vnorm(bx - ax, by - ay)
        for j in range(i + 1, len(lines)):
            cx, cy, dx, dy = lines[j]
            u2 = _vnorm(dx - cx, dy - cy)
            if abs((u1[0] * u2[0]) + (u1[1] * u2[1])) > 0.95:
                continue
            p = _segment_intersection(ax, ay, bx, by, cx, cy, dx, dy)
            if p is not None:
                pts.append(p)

    return _cluster_points(pts, tol_cm=12.0)


def _nearest_corner(pt, corners):
    best = None
    for c in (corners or []):
        d = _pt_dist(pt, c)
        if (best is None) or (d < best[1]):
            best = (c, d)
    return best


class _ImportSelectionFilter(ISelectionFilter):
    def __init__(self, active_view):
        self._view = active_view

    def AllowElement(self, element):
        try:
            if not isinstance(element, ImportInstance):
                return False
            return element.get_BoundingBox(self._view) is not None
        except Exception:
            return False

    def AllowReference(self, reference, point):
        return False


def _pick_target_import(active_view):
    instances = get_imported_cad_instances(doc, active_view)
    if not instances:
        raise ValueError("No imported DWG/DXF was found in the active view.")

    try:
        ref = uidoc.Selection.PickObject(
            ObjectType.Element,
            _ImportSelectionFilter(active_view),
            "Select the DWG import for D_W_detection",
        )
    except OperationCanceledException:
        raise ValueError("DWG selection was canceled.")

    inst = doc.GetElement(ref.ElementId) if ref is not None else None
    if not isinstance(inst, ImportInstance):
        raise ValueError("Selected element is not a DWG import instance.")
    if inst.get_BoundingBox(active_view) is None:
        raise ValueError("Selected DWG is not visible in the active view.")
    return inst


def _append_unique_pattern(layer_map, key, pattern):
    arr = list((layer_map or {}).get(key, []))
    if pattern not in arr:
        arr.append(pattern)
    layer_map[key] = arr


def _patch_layer_map(layer_map):
    layer_map = dict(layer_map or {})
    _append_unique_pattern(layer_map, "doors", r"^A[-_ ]?DOORS?$")
    _append_unique_pattern(layer_map, "windows", r"^A[-_ ]?WINDOWS?$")
    _append_unique_pattern(layer_map, "walls", r"^A[-_ ]?WALL(S)?$")
    _append_unique_pattern(layer_map, "walls", r"^A[-_ ]?WALL[-_ ]?EXT$")
    _append_unique_pattern(layer_map, "walls", r"^A[-_ ]?WALL[-_ ]?INT$")
    return layer_map


def _norm_layer(layer_name):
    return re.sub(r"[^A-Z0-9]", "", str(layer_name or "").upper())


def _is_target_layer(layer_name, kind):
    n = _norm_layer(layer_name)
    if kind == "door":
        return n.endswith("ADOOR") or n.endswith("ADOORS")
    if kind == "window":
        return n.endswith("AWINDOW") or n.endswith("AWINDOWS")
    return False


def _collect_target_layer_entities(raw):
    doors = {"lines": [], "arcs": []}
    windows = {"lines": [], "arcs": []}

    for ln in (raw.get("lines") or []):
        layer = ln.get("layer", "")
        if _is_target_layer(layer, "door"):
            doors["lines"].append(ln)
        elif _is_target_layer(layer, "window"):
            windows["lines"].append(ln)

    for arc in (raw.get("arcs") or []):
        layer = arc.get("layer", "")
        if _is_target_layer(layer, "door"):
            doors["arcs"].append(arc)
        elif _is_target_layer(layer, "window"):
            windows["arcs"].append(arc)

    return doors, windows


def _entity_from_line(ln):
    x1 = float(ln.get("x1", 0.0))
    y1 = float(ln.get("y1", 0.0))
    x2 = float(ln.get("x2", 0.0))
    y2 = float(ln.get("y2", 0.0))
    dx = x2 - x1
    dy = y2 - y1
    ln_len = _vlen(dx, dy)
    if ln_len <= 1.0e-9:
        return None
    return {
        "bbox": _line_bbox(ln),
        "center": ((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        "axis": _vnorm(dx, dy),
        "span": ln_len,
    }


def _entity_from_arc(arc):
    cx = float(arc.get("cx", 0.0))
    cy = float(arc.get("cy", 0.0))
    r = abs(float(arc.get("r", 0.0)))
    if r <= 1.0e-9:
        return None
    sx = float(arc.get("sx", cx + r))
    sy = float(arc.get("sy", cy))
    ex = float(arc.get("ex", cx - r))
    ey = float(arc.get("ey", cy))
    dx = ex - sx
    dy = ey - sy
    return {
        "bbox": _arc_bbox(arc),
        "center": (cx, cy),
        "axis": _vnorm(dx, dy),
        "span": max(20.0, 2.0 * r),
    }


def _group_entities_to_block_markers(kind, lines, arcs):
    entities = []
    for ln in (lines or []):
        e = _entity_from_line(ln)
        if e:
            entities.append(e)
    for arc in (arcs or []):
        e = _entity_from_arc(arc)
        if e:
            entities.append(e)

    if not entities:
        return []

    parent = list(range(len(entities)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    # Pass 1: connect touching/near primitives (a single CAD block usually forms one cluster).
    connect_tol_cm = 12.0
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if _bbox_dist(entities[i]["bbox"], entities[j]["bbox"]) <= connect_tol_cm:
                union(i, j)

    comp = {}
    for idx, ent in enumerate(entities):
        root = find(idx)
        comp.setdefault(root, []).append(ent)

    comps = []
    for group in comp.values():
        bb = group[0]["bbox"]
        axis_sum = [0.0, 0.0]
        span_max = 0.0
        for ent in group:
            bb = _merge_bbox(bb, ent["bbox"])
            axis_sum[0] += float(ent["axis"][0])
            axis_sum[1] += float(ent["axis"][1])
            span_max = max(span_max, float(ent["span"]))
        comps.append({
            "bbox": bb,
            "center": _bbox_center(bb),
            "axis": _vnorm(axis_sum[0], axis_sum[1]),
            "span": span_max,
        })

    # Pass 2: merge very-close clusters from the same block.
    parent2 = list(range(len(comps)))

    def find2(i):
        while parent2[i] != i:
            parent2[i] = parent2[parent2[i]]
            i = parent2[i]
        return i

    def union2(i, j):
        ri = find2(i)
        rj = find2(j)
        if ri != rj:
            parent2[rj] = ri

    merge_tol_cm = 55.0
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if _bbox_dist(comps[i]["bbox"], comps[j]["bbox"]) <= merge_tol_cm:
                union2(i, j)

    grouped2 = {}
    for idx, c in enumerate(comps):
        root = find2(idx)
        grouped2.setdefault(root, []).append(c)

    markers = []
    for group in grouped2.values():
        bb = group[0]["bbox"]
        axis_sum = [0.0, 0.0]
        span_max = 0.0
        for c in group:
            bb = _merge_bbox(bb, c["bbox"])
            axis_sum[0] += float(c["axis"][0])
            axis_sum[1] += float(c["axis"][1])
            span_max = max(span_max, float(c["span"]))

        axis = _vnorm(axis_sum[0], axis_sum[1])
        bw, bh = _bbox_size(bb)
        if abs(axis[0]) + abs(axis[1]) < 1.0e-6:
            axis = (1.0, 0.0) if bw >= bh else (0.0, 1.0)

        width_cm = max(40.0, min(300.0, max(span_max, bw, bh)))
        markers.append({
            "kind": kind,
            "center_cm": _bbox_center(bb),
            "axis_uv": axis,
            "width_cm": width_cm,
            "source": "layer_block",
        })

    # final dedupe
    out = []
    for mk in markers:
        c = mk["center_cm"]
        duplicate = False
        for ex in out:
            if _pt_dist(c, ex["center_cm"]) <= 40.0:
                duplicate = True
                break
        if not duplicate:
            out.append(mk)
    return out


def _attach_door_arc_candidates(markers, door_arcs):
    starts = []
    for arc in (door_arcs or []):
        starts.append((float(arc.get("sx", arc.get("cx", 0.0))), float(arc.get("sy", arc.get("cy", 0.0)))))

    if not starts:
        return list(markers or [])

    out = []
    for mk in (markers or []):
        center = mk.get("center_cm")
        if center is None:
            out.append(dict(mk))
            continue
        cx, cy = float(center[0]), float(center[1])
        best = None
        for p in starts:
            d = _pt_dist((cx, cy), p)
            if d > 220.0:
                continue
            if (best is None) or (d < best[1]):
                best = (p, d)
        m2 = dict(mk)
        if best is not None:
            m2["edge_start_cm"] = best[0]
        out.append(m2)
    return out


def _room_polygon_corners(room_data):
    pts = list((room_data or {}).get("room_polygon_cm") or [])
    if len(pts) < 3:
        return []
    corners = []
    for p in pts:
        try:
            corners.append((float(p[0]), float(p[1])))
        except Exception:
            continue
    return _cluster_points(corners, tol_cm=8.0)


def _opening_axis_from_polygon(opening, room_polygon):
    if not room_polygon or len(room_polygon) < 2:
        return (1.0, 0.0)
    try:
        edge_idx = int(opening.get("host_edge", -1))
    except Exception:
        edge_idx = -1
    if edge_idx < 0:
        return (1.0, 0.0)
    n = len(room_polygon)
    a = room_polygon[edge_idx % n]
    b = room_polygon[(edge_idx + 1) % n]
    return _vnorm(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def _opening_center_from_polygon(opening, room_polygon):
    cx = opening.get("center_x_cm")
    cy = opening.get("center_y_cm")
    if (cx is not None) and (cy is not None):
        return (float(cx), float(cy))
    if not room_polygon or len(room_polygon) < 2:
        return None
    try:
        edge_idx = int(opening.get("host_edge", -1))
    except Exception:
        edge_idx = -1
    if edge_idx < 0:
        return None
    n = len(room_polygon)
    a = room_polygon[edge_idx % n]
    b = room_polygon[(edge_idx + 1) % n]
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    edge_len = _vlen(dx, dy)
    if edge_len <= 1.0e-9:
        return None
    s = float(opening.get("start_cm", 0.0))
    e = float(opening.get("end_cm", s))
    if e < s:
        s, e = e, s
    t = (s + e) * 0.5
    ux, uy = (dx / edge_len, dy / edge_len)
    return (float(a[0]) + (ux * t), float(a[1]) + (uy * t))


def _markers_from_topology(room_data):
    room_polygon = list((room_data or {}).get("room_polygon_cm") or [])
    openings = list((room_data or {}).get("openings") or [])
    door = []
    window = []
    for op in openings:
        typ = str(op.get("type", "")).lower().strip()
        if typ not in ("door", "window"):
            continue
        center = _opening_center_from_polygon(op, room_polygon)
        if center is None:
            continue
        axis = _opening_axis_from_polygon(op, room_polygon)
        width_cm = max(40.0, min(300.0, float(op.get("width_cm", 100.0))))
        rec = {
            "kind": typ,
            "center_cm": center,
            "axis_uv": axis,
            "width_cm": width_cm,
            "source": "topology",
        }
        if typ == "door":
            door.append(rec)
        else:
            window.append(rec)
    return door, window


def _marker_text_points(markers, label):
    points = []
    for mk in (markers or []):
        cx, cy = mk["center_cm"]
        ux, uy = _vnorm(float(mk["axis_uv"][0]), float(mk["axis_uv"][1]))

        nx, ny = (uy, -ux)
        if ny > 0.0:
            nx, ny = (-nx, -ny)

        width = max(40.0, min(300.0, float(mk.get("width_cm", 100.0))))
        if str(label).upper() == "DD":
            # Doors: keep text close to the opening so it does not drift too low.
            off = max(8.0, min(16.0, width * 0.10))
        else:
            off = max(14.0, min(24.0, width * 0.15))
        mx = cx + (nx * off)
        my = cy + (ny * off)
        points.append(((mx, my), label))
    return points


def _snap_markers_to_walls(markers, wall_lines, max_snap_cm, sources=None):
    if not markers or not wall_lines:
        return list(markers or [])

    walls = []
    for ln in (wall_lines or []):
        x1 = float(ln.get("x1", 0.0))
        y1 = float(ln.get("y1", 0.0))
        x2 = float(ln.get("x2", 0.0))
        y2 = float(ln.get("y2", 0.0))
        dx = x2 - x1
        dy = y2 - y1
        seg_len = _vlen(dx, dy)
        if seg_len < 20.0:
            continue
        walls.append({
            "a": (x1, y1),
            "b": (x2, y2),
            "axis": _vnorm(dx, dy),
        })

    if not walls:
        return list(markers or [])

    out = []
    allowed = set(sources or [])
    for mk in (markers or []):
        c = mk.get("center_cm")
        if c is None:
            out.append(dict(mk))
            continue
        if allowed and (str(mk.get("source", "")) not in allowed):
            out.append(dict(mk))
            continue

        cx, cy = float(c[0]), float(c[1])
        best = None
        for w in walls:
            qx, qy, _t = _closest_point_on_segment(cx, cy, w["a"][0], w["a"][1], w["b"][0], w["b"][1])
            d = _vlen(cx - qx, cy - qy)
            if (best is None) or (d < best["dist"]):
                best = {"dist": d, "pt": (qx, qy), "axis": w["axis"]}

        if (best is not None) and (best["dist"] <= max_snap_cm):
            snapped = dict(mk)
            snapped["center_cm"] = best["pt"]
            snapped["axis_uv"] = best["axis"]
            out.append(snapped)
        else:
            out.append(dict(mk))
    return out


def _first_text_type_id():
    txt_types = list(FilteredElementCollector(doc).OfClass(TextNoteType))
    if not txt_types:
        return None
    return txt_types[0].Id


def _draw_marker_texts(active_view, text_points):
    if not text_points:
        return 0

    type_id = _first_text_type_id()
    if type_id is None:
        raise ValueError("No Text Note type exists in this project.")

    z = 0.0
    try:
        lvl = active_view.GenLevel
        if lvl is not None:
            z = float(lvl.Elevation)
    except Exception:
        z = 0.0

    created = 0
    tx = Transaction(doc, "D_W_detection")
    tx.Start()
    try:
        for p_cm, label in text_points:
            p = XYZ(cm_to_ft(p_cm[0]), cm_to_ft(p_cm[1]), z)
            try:
                TextNote.Create(doc, active_view.Id, p, str(label), type_id)
                created += 1
            except Exception:
                continue
        tx.Commit()
    except Exception:
        tx.RollBack()
        raise
    return created


def _door_dimension_points(door_markers, wall_corners):
    out = []
    for mk in (door_markers or []):
        edge_candidates = []
        edge_start = mk.get("edge_start_cm")
        if edge_start is not None:
            edge_candidates.append((float(edge_start[0]), float(edge_start[1])))
        center = mk.get("center_cm")
        if (not edge_candidates) and (center is not None):
            edge_candidates.append((float(center[0]), float(center[1])))
        if not edge_candidates:
            continue

        best = None
        for p in edge_candidates:
            near = _nearest_corner(p, wall_corners)
            if near is None:
                continue
            corner, dist = near
            if dist < 25.0:
                continue
            if (best is None) or (dist < best["dist"]):
                best = {"edge": p, "corner": corner, "dist": dist}
        if best is not None:
            out.append(best)
    return out


def _draw_door_dimensions(active_view, dim_pairs):
    if not dim_pairs:
        return {"attempted": 0, "created": 0, "failed": 0}

    z = 0.0
    try:
        lvl = active_view.GenLevel
        if lvl is not None:
            z = float(lvl.Elevation)
    except Exception:
        z = 0.0

    created = 0
    failed = 0
    attempted = 0
    tx = Transaction(doc, "D_W_detection Door Dimensions")
    tx.Start()
    try:
        # Make sure dimension category is visible in this view.
        try:
            dim_cat = doc.Settings.Categories.get_Item(BuiltInCategory.OST_Dimensions)
            if dim_cat is not None and active_view.CanCategoryBeHidden(dim_cat.Id):
                active_view.SetCategoryHidden(dim_cat.Id, False)
        except Exception:
            pass

        for rec in dim_pairs:
            attempted += 1
            p1 = rec["edge"]
            p2 = rec["corner"]
            dx = float(p2[0]) - float(p1[0])
            dy = float(p2[1]) - float(p1[1])
            ln = _vlen(dx, dy)
            if ln < 8.0:
                failed += 1
                continue
            ux, uy = (dx / ln, dy / ln)
            nx, ny = (-uy, ux)

            helper_half = 5.0
            dim_off = 10.0

            a1 = XYZ(cm_to_ft(p1[0] - (nx * helper_half)), cm_to_ft(p1[1] - (ny * helper_half)), z)
            b1 = XYZ(cm_to_ft(p1[0] + (nx * helper_half)), cm_to_ft(p1[1] + (ny * helper_half)), z)
            a2 = XYZ(cm_to_ft(p2[0] - (nx * helper_half)), cm_to_ft(p2[1] - (ny * helper_half)), z)
            b2 = XYZ(cm_to_ft(p2[0] + (nx * helper_half)), cm_to_ft(p2[1] + (ny * helper_half)), z)

            try:
                w1 = doc.Create.NewDetailCurve(active_view, Line.CreateBound(a1, b1))
                w2 = doc.Create.NewDetailCurve(active_view, Line.CreateBound(a2, b2))
            except Exception:
                failed += 1
                continue

            # Ensure geometry references are available before creating dimensions.
            try:
                doc.Regenerate()
            except Exception:
                pass

            d1 = XYZ(cm_to_ft(p1[0] + (nx * dim_off)), cm_to_ft(p1[1] + (ny * dim_off)), z)
            d2 = XYZ(cm_to_ft(p2[0] + (nx * dim_off)), cm_to_ft(p2[1] + (ny * dim_off)), z)

            try:
                dim_line = Line.CreateBound(d1, d2)
                refs = ReferenceArray()
                ref1 = None
                ref2 = None
                try:
                    ref1 = w1.GeometryCurve.Reference
                except Exception:
                    ref1 = None
                try:
                    ref2 = w2.GeometryCurve.Reference
                except Exception:
                    ref2 = None
                if ref1 is None:
                    try:
                        ref1 = w1.GeometryCurve.GetEndPointReference(0)
                    except Exception:
                        ref1 = None
                if ref2 is None:
                    try:
                        ref2 = w2.GeometryCurve.GetEndPointReference(0)
                    except Exception:
                        ref2 = None
                if (ref1 is None) or (ref2 is None):
                    failed += 1
                    continue
                refs.Append(ref1)
                refs.Append(ref2)
                dim = doc.Create.NewDimension(active_view, dim_line, refs)
                if dim is not None:
                    created += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
                continue

        tx.Commit()
    except Exception:
        tx.RollBack()
        raise
    return {"attempted": attempted, "created": created, "failed": failed}


def _count_view_dimensions(active_view):
    try:
        cnt = 0
        col = (
            FilteredElementCollector(doc, active_view.Id)
            .OfClass(Dimension)
            .WhereElementIsNotElementType()
        )
        for _d in col:
            cnt += 1
        return cnt
    except Exception:
        return -1


def main():
    active_view = uidoc.ActiveView
    if active_view is None:
        TaskDialog.Show(__title__, "No active view.")
        return

    inst = _pick_target_import(active_view)

    cfg = load_config(os.path.join(V2_DIR, "cad_config.json"))
    layer_map = _patch_layer_map(load_layer_map(os.path.join(V2_DIR, "cad_layer_map.json")))

    raw = extract_cad_from_view(doc, active_view, cfg, target_instance_id=inst.Id.IntegerValue)
    classified = classify_entities(raw.get("lines") or [], raw.get("arcs") or [], layer_map, cfg)

    door_layer, win_layer = _collect_target_layer_entities(raw)
    layer_doors = _group_entities_to_block_markers("door", door_layer["lines"], door_layer["arcs"])
    layer_windows = _group_entities_to_block_markers("window", win_layer["lines"], win_layer["arcs"])

    need_topology_doors = len(layer_doors) == 0
    need_topology_windows = len(layer_windows) == 0
    topo_doors = []
    topo_windows = []
    topo_error = None

    if need_topology_doors or need_topology_windows:
        try:
            room_data, _debug = recognize_topology(classified, cfg)
            topo_doors, topo_windows = _markers_from_topology(room_data)
        except Exception as ex:
            topo_error = str(ex)
            room_data = None
    else:
        room_data = None

    # Always try topology for corner extraction used by door dimensions.
    if room_data is None:
        try:
            room_data, _debug2 = recognize_topology(classified, cfg)
        except Exception:
            room_data = None

    final_doors = layer_doors if layer_doors else topo_doors
    final_windows = layer_windows if layer_windows else topo_windows
    final_doors = _attach_door_arc_candidates(final_doors, door_layer["arcs"])
    final_doors = _snap_markers_to_walls(
        final_doors,
        classified.get("wall_lines") or [],
        max_snap_cm=95.0,
        sources={"topology"},
    )
    final_windows = _snap_markers_to_walls(
        final_windows,
        classified.get("wall_lines") or [],
        max_snap_cm=70.0,
        sources={"topology"},
    )

    wall_corners = _room_polygon_corners(room_data)
    if not wall_corners:
        wall_corners = _collect_wall_corners(classified.get("wall_lines") or [])
    door_dim_pairs = _door_dimension_points(final_doors, wall_corners)
    dim_stats = _draw_door_dimensions(active_view, door_dim_pairs)

    door_text_points = _marker_text_points(final_doors, "DD")
    window_text_points = _marker_text_points(final_windows, "ww")
    door_text_created = _draw_marker_texts(active_view, door_text_points)
    window_text_created = _draw_marker_texts(active_view, window_text_points)

    summary = []
    summary.append("DWG instance id: {}".format(inst.Id.IntegerValue))
    summary.append(
        "A-DOORS entities: {} lines, {} arcs -> {} door block(s)".format(
            len(door_layer["lines"]), len(door_layer["arcs"]), len(layer_doors)
        )
    )
    summary.append(
        "A-WINDOWS entities: {} lines, {} arcs -> {} window block(s)".format(
            len(win_layer["lines"]), len(win_layer["arcs"]), len(layer_windows)
        )
    )
    summary.append("Detected doors: {} ({})".format(len(final_doors), "layer-block" if layer_doors else "topology"))
    summary.append("Detected windows: {} ({})".format(len(final_windows), "layer-block" if layer_windows else "topology"))
    summary.append(
        "Door dimensions: {}/{} created ({} failed)".format(
            int(dim_stats.get("created", 0)),
            int(dim_stats.get("attempted", 0)),
            int(dim_stats.get("failed", 0)),
        )
    )
    summary.append("Visible dimensions in this view: {}".format(_count_view_dimensions(active_view)))
    summary.append("Door text markers created: {}".format(door_text_created))
    summary.append("Window text markers created: {}".format(window_text_created))
    summary.append("Door annotation: linear dimension (arch edge -> closest corner)")
    summary.append("Door marker text: DD")
    summary.append("Window marker text: ww")
    if topo_error:
        summary.append("Topology note: {}".format(topo_error))

    TaskDialog.Show(__title__, "\n".join(summary))


# ---------------------------------------------------------------------------
# Load CAD helpers from sibling CreateFromCADV2 command.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(__file__)
PANEL_DIR = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(PANEL_DIR, "CreateFromCADV2.pushbutton")
if V2_DIR not in sys.path:
    sys.path.append(V2_DIR)

from v2_cad_extract import extract_cad_from_view, get_imported_cad_instances, load_config, load_layer_map
from v2_cad_classify import classify_entities
from v2_cad_recognition import recognize_topology


if __name__ == "__main__":
    try:
        main()
    except Exception as run_err:
        TaskDialog.Show(__title__, str(run_err))
