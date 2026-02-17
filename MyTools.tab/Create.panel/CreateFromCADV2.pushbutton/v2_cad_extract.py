# -*- coding: utf-8 -*-
"""Extract raw line/arc primitives from CAD ImportInstance in active Revit view."""

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
        "dimensions": [r"^DIM.*", r"^A-ANNO-DIM.*", r".*DIMENSION.*", r"^MEASURE.*"],
        "ignore": [r"^DEFPOINTS$", r"^HATCH.*", r"^TEXT.*"],
    }


def default_config():
    return {
        "endpoint_snap_mm": 4.0,
        "min_segment_mm": 8.0,
        "opening_host_distance_mm": 70.0,
        "opening_gap_min_cm": 55.0,
        "opening_gap_max_cm": 260.0,
        "wall_gap_bridge_min_cm": 4.0,
        "wall_gap_bridge_max_cm": 220.0,
        "wall_gap_bridge_angle_deg": 16.0,
        "polygon_min_area_cm2": 6000.0,
        "polygon_max_cycle_len": 80,
        "polygon_max_cycles": 600,
        "default_wall_thickness_cm": 20.0,
        "default_door_width_cm": 100.0,
        "default_door_height_cm": 210.0,
        "default_window_width_cm": 100.0,
        "default_window_height_cm": 100.0,
        "default_window_sill_cm": 105.0,
        "double_wall_pair_min_cm": 4.0,
        "double_wall_pair_max_cm": 50.0,
        "double_wall_pair_angle_deg": 8.0,
        "double_wall_pair_min_overlap_cm": 35.0,
        "double_wall_pair_overlap_ratio": 0.60,
        "dimension_hint_host_distance_cm": 180.0,
        "dimension_hint_angle_deg": 18.0,
        "dimension_hint_center_tol_cm": 120.0,
        "dimension_span_ratio_min": 0.75,
        "dimension_span_ratio_max": 2.5,
        "dimension_scale_min": 0.5,
        "dimension_scale_max": 2.0,
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


def load_config(path):
    data = default_config()
    try:
        if path and os.path.isfile(path):
            with open(path, "r") as f:
                parsed = json.load(f)
            if isinstance(parsed, dict):
                data = _merge(data, parsed)
    except Exception:
        pass
    return data


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


def _append_line(out, p1, p2, layer, min_len_cm):
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
    })


def _append_arc(out, arc, layer, tf):
    center = _apply_tf(arc.Center, tf)
    start = _apply_tf(arc.GetEndPoint(0), tf)
    end = _apply_tf(arc.GetEndPoint(1), tf)

    out["arcs"].append({
        "type": "arc",
        "cx": ft_to_cm(center.X),
        "cy": ft_to_cm(center.Y),
        "r": ft_to_cm(arc.Radius),
        "sx": ft_to_cm(start.X),
        "sy": ft_to_cm(start.Y),
        "ex": ft_to_cm(end.X),
        "ey": ft_to_cm(end.Y),
        "layer": layer,
    })


def _walk_geom(doc, geom_enum, tf, out, cfg):
    from Autodesk.Revit.DB import GeometryInstance, Line, Arc, PolyLine

    min_len_cm = float(cfg.get("min_segment_mm", 8.0)) / 10.0

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
            _append_line(out, p1, p2, layer, min_len_cm)
            continue

        if isinstance(obj, Arc):
            _append_arc(out, obj, layer, tf)
            continue

        if isinstance(obj, PolyLine):
            try:
                pts = obj.GetCoordinates()
            except Exception:
                pts = []
            for i in range(0, max(0, len(pts) - 1)):
                p1 = _apply_tf(pts[i], tf)
                p2 = _apply_tf(pts[i + 1], tf)
                _append_line(out, p1, p2, layer, min_len_cm)


def _round_pt(x, y, nd=4):
    return (round(float(x), nd), round(float(y), nd))


def _dedupe_lines(lines):
    out = []
    seen = set()
    for ln in (lines or []):
        p1 = _round_pt(ln["x1"], ln["y1"])
        p2 = _round_pt(ln["x2"], ln["y2"])
        key = tuple(sorted([p1, p2]))
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return out


def _dedupe_arcs(arcs):
    out = []
    seen = set()
    for a in (arcs or []):
        key = (
            round(a.get("cx", 0.0), 4),
            round(a.get("cy", 0.0), 4),
            round(a.get("r", 0.0), 4),
            round(a.get("sx", 0.0), 4),
            round(a.get("sy", 0.0), 4),
            round(a.get("ex", 0.0), 4),
            round(a.get("ey", 0.0), 4),
            a.get("layer", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
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
            raise ValueError("Multiple CAD imports found. Select one or hide others.")
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

    out["lines"] = _dedupe_lines(out["lines"])
    out["arcs"] = _dedupe_arcs(out["arcs"])
    out["meta"]["line_count"] = len(out["lines"])
    out["meta"]["arc_count"] = len(out["arcs"])
    return out
