# -*- coding: utf-8 -*-
"""Extract line/arc primitives from an imported CAD instance in active Revit view.

This module keeps Autodesk imports inside runtime functions so unit tests can run
outside Revit.
"""

import json
import math
import os

CM_PER_FT = 30.48


def ft_to_cm(v):
    return float(v) * CM_PER_FT


def default_layer_map():
    return {
        "walls": [r"^A?-?WALL(S)?$", r"^WALL_.*", r"^A-WALL$"],
        "doors": [r"^A?-?DOOR(S)?$", r"^DOOR_.*", r"^A-DOOR$"],
        "windows": [r"^A?-?WIN(DOW)?S?$", r"^WINDOW_.*", r"^A-WIND$"],
        "ignore": [r"^DEFPOINTS$", r"^HATCH.*", r"^TEXT.*", r"^DIM.*"],
    }


def default_cad_config():
    return {
        "endpoint_snap_mm": 2.0,
        "join_tol_mm": 5.0,
        "collinear_angle_deg": 0.5,
        "parallel_angle_deg": 0.5,
        "wall_spacing_tol_mm": 5.0,
        "min_segment_mm": 5.0,
        "opening_host_distance_mm": 30.0,
        "opening_gap_min_cm": 45.0,
        "opening_gap_max_cm": 220.0,
        "default_wall_thickness_cm": 30.0,
        "fallback_min_wall_line_cm": 5.0,
        "polygon_min_area_cm2": 4000.0,
        "polygon_max_cycle_len": 40,
        "polygon_max_cycles": 200,
        "prefer_polygon_mode": False,
        "default_door_width_cm": 100.0,
        "default_door_height_cm": 210.0,
        "default_window_width_cm": 100.0,
        "default_window_height_cm": 100.0,
        "default_window_sill_cm": 105.0,
        "default_window_right_offset_cm": 30.0,
        "default_door_left_offset_cm": 90.0,
    }


def _merge(base, patch):
    out = dict(base)
    for k, v in (patch or {}).items():
        out[k] = v
    return out


def load_layer_map(path):
    data = default_layer_map()
    try:
        if path and os.path.isfile(path):
            with open(path, "r") as f:
                parsed = json.load(f)
            if isinstance(parsed, dict):
                data = _merge(data, parsed)
    except Exception:
        pass
    return data


def load_cad_config(path):
    data = default_cad_config()
    try:
        if path and os.path.isfile(path):
            with open(path, "r") as f:
                parsed = json.load(f)
            if isinstance(parsed, dict):
                data = _merge(data, parsed)
    except Exception:
        pass
    return data


def _round_pt(x, y, nd=4):
    return (round(float(x), nd), round(float(y), nd))


def _safe_layer_name(doc, geom_obj):
    try:
        gs_id = geom_obj.GraphicsStyleId
        if gs_id and gs_id.IntegerValue != -1:
            gs = doc.GetElement(gs_id)
            if gs and gs.GraphicsStyleCategory:
                return gs.GraphicsStyleCategory.Name or "UNSPECIFIED"
    except Exception:
        pass
    return "UNSPECIFIED"


def _apply_tf(pt, tf):
    if tf is None:
        return pt
    try:
        return tf.OfPoint(pt)
    except Exception:
        return pt


def _compose_tf(parent_tf, child_tf):
    if parent_tf is None:
        return child_tf
    if child_tf is None:
        return parent_tf
    try:
        return parent_tf.Multiply(child_tf)
    except Exception:
        try:
            return child_tf.Multiply(parent_tf)
        except Exception:
            return child_tf


def _append_line(out, p1, p2, layer, min_len_cm, source):
    x1 = ft_to_cm(p1.X)
    y1 = ft_to_cm(p1.Y)
    x2 = ft_to_cm(p2.X)
    y2 = ft_to_cm(p2.Y)
    dx = x2 - x1
    dy = y2 - y1
    ln = math.sqrt((dx * dx) + (dy * dy))
    if ln < min_len_cm:
        return
    out["lines"].append({
        "type": "line",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "length_cm": ln,
        "layer": layer,
        "source": source,
    })


def _append_arc(out, arc, layer, tf, source):
    center = _apply_tf(arc.Center, tf)
    start = _apply_tf(arc.GetEndPoint(0), tf)
    end = _apply_tf(arc.GetEndPoint(1), tf)
    cx = ft_to_cm(center.X)
    cy = ft_to_cm(center.Y)
    sx = ft_to_cm(start.X)
    sy = ft_to_cm(start.Y)
    ex = ft_to_cm(end.X)
    ey = ft_to_cm(end.Y)
    radius_cm = ft_to_cm(arc.Radius)
    start_ang = math.atan2(sy - cy, sx - cx)
    end_ang = math.atan2(ey - cy, ex - cx)

    out["arcs"].append({
        "type": "arc",
        "cx": cx,
        "cy": cy,
        "r": radius_cm,
        "sx": sx,
        "sy": sy,
        "ex": ex,
        "ey": ey,
        "start_ang": start_ang,
        "end_ang": end_ang,
        "layer": layer,
        "source": source,
    })


def _walk_geom(doc, geom_enum, tf, out, cfg):
    from Autodesk.Revit.DB import GeometryInstance, Line, Arc, PolyLine

    min_len_cm = float(cfg.get("min_segment_mm", 5.0)) / 10.0

    for obj in geom_enum:
        if isinstance(obj, GeometryInstance):
            nxt_tf = _compose_tf(tf, obj.Transform)
            try:
                _walk_geom(doc, obj.GetInstanceGeometry(), nxt_tf, out, cfg)
            except Exception:
                pass
            try:
                _walk_geom(doc, obj.GetSymbolGeometry(), nxt_tf, out, cfg)
            except Exception:
                pass
            continue

        layer = _safe_layer_name(doc, obj)

        if isinstance(obj, Line):
            p1 = _apply_tf(obj.GetEndPoint(0), tf)
            p2 = _apply_tf(obj.GetEndPoint(1), tf)
            _append_line(out, p1, p2, layer, min_len_cm, "cad_line")
            continue

        if isinstance(obj, Arc):
            _append_arc(out, obj, layer, tf, "cad_arc")
            continue

        if isinstance(obj, PolyLine):
            try:
                pts = obj.GetCoordinates()
            except Exception:
                pts = []
            for i in range(0, max(0, len(pts) - 1)):
                p1 = _apply_tf(pts[i], tf)
                p2 = _apply_tf(pts[i + 1], tf)
                _append_line(out, p1, p2, layer, min_len_cm, "cad_polyline")


def _dedupe_entities(entities):
    out = []
    seen = set()
    for e in entities:
        if e.get("type") == "line":
            p1 = _round_pt(e["x1"], e["y1"])
            p2 = _round_pt(e["x2"], e["y2"])
            key = ("line", tuple(sorted([p1, p2])), e.get("layer", ""))
        else:
            key = (
                "arc",
                round(e.get("cx", 0.0), 4),
                round(e.get("cy", 0.0), 4),
                round(e.get("r", 0.0), 4),
                round(e.get("sx", 0.0), 4),
                round(e.get("sy", 0.0), 4),
                round(e.get("ex", 0.0), 4),
                round(e.get("ey", 0.0), 4),
                e.get("layer", ""),
            )
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def get_imported_cad_instances(doc, view):
    from Autodesk.Revit.DB import FilteredElementCollector, ImportInstance

    out = []
    for inst in FilteredElementCollector(doc).OfClass(ImportInstance).WhereElementIsNotElementType():
        try:
            bb = inst.get_BoundingBox(view)
            if bb is None:
                continue
            out.append(inst)
        except Exception:
            continue
    return out


def extract_cad_from_view(doc, view, cfg, target_instance_id=None):
    from Autodesk.Revit.DB import Options

    instances = get_imported_cad_instances(doc, view)
    if len(instances) == 0:
        raise ValueError("No imported CAD instance found in active plan view")

    inst = None
    if target_instance_id is not None:
        for candidate in instances:
            try:
                if candidate.Id.IntegerValue == int(target_instance_id):
                    inst = candidate
                    break
            except Exception:
                continue
        if inst is None:
            raise ValueError("Selected CAD instance was not found in active plan view")
    else:
        if len(instances) > 1:
            raise ValueError("Multiple imported CAD instances found. Keep one visible CAD import in the view")
        inst = instances[0]

    out = {
        "meta": {
            "instance_id": inst.Id.IntegerValue,
            "view_id": view.Id.IntegerValue,
            "units": "cm",
        },
        "lines": [],
        "arcs": [],
    }

    opts = Options()
    opts.View = view
    opts.IncludeNonVisibleObjects = False
    opts.ComputeReferences = False

    geom = inst.get_Geometry(opts)
    _walk_geom(doc, geom, None, out, cfg)

    out["lines"] = _dedupe_entities(out["lines"])
    out["arcs"] = _dedupe_entities(out["arcs"])
    out["meta"]["line_count"] = len(out["lines"])
    out["meta"]["arc_count"] = len(out["arcs"])
    return out
