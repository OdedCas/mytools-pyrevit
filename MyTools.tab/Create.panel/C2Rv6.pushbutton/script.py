# -*- coding: utf-8 -*-
__title__ = "C2Rv6"
__doc__ = "CAD to Revit V6. Select one DWG, then build continuous walls by A-WALL-EXT / A-WALL-INT layers only."

import imp
import os
import sys
import math

from Autodesk.Revit import DB
from Autodesk.Revit.DB import ImportInstance
from Autodesk.Revit.Exceptions import OperationCanceledException
from Autodesk.Revit.UI import TaskDialog
from Autodesk.Revit.UI.Selection import ISelectionFilter, ObjectType


SCRIPT_DIR = os.path.dirname(__file__)
PANEL_DIR = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(PANEL_DIR, "CreateFromCADV2.pushbutton")
_REC_MOD = None

if V2_DIR not in sys.path:
    sys.path.append(V2_DIR)


def _load_v2_module():
    path = os.path.join(V2_DIR, "script.py")
    try:
        return imp.load_source("c2rv6_v2_delegate", path)
    except Exception as ex:
        raise Exception("Failed loading CreateFromCADV2 module: {} ({})".format(path, ex))


def _load_recognition_helpers():
    global _REC_MOD
    if _REC_MOD is not None:
        return _REC_MOD
    path = os.path.join(V2_DIR, "v2_cad_recognition.py")
    try:
        _REC_MOD = imp.load_source("c2rv6_v2_recognition_helpers", path)
    except Exception as ex:
        raise Exception("Failed loading recognition helpers: {} ({})".format(path, ex))
    return _REC_MOD


class _DwgImportFilter(ISelectionFilter):
    def AllowElement(self, elem):
        try:
            return isinstance(elem, ImportInstance)
        except Exception:
            return False

    def AllowReference(self, reference, point):
        return False


def _pick_dwg_import(uidoc):
    try:
        picked = uidoc.Selection.PickObject(
            ObjectType.Element,
            _DwgImportFilter(),
            "Select DWG import/link for C2Rv6",
        )
    except OperationCanceledException:
        return None
    if picked is None:
        return None
    elem = uidoc.Document.GetElement(picked.ElementId)
    if not isinstance(elem, ImportInstance):
        return None
    return elem


def _apply_selected_import_scope(v2, selected_import):
    def _get_imported_selected(doc, view):
        try:
            inst = doc.GetElement(selected_import.Id)
            if inst is None:
                return []
            if inst.get_BoundingBox(view) is None:
                return []
            return [inst]
        except Exception:
            return []

    # Force CAD-from-existing mode and force the chosen import only.
    v2.get_imported_cad_instances = _get_imported_selected
    v2.choose_input_kind = lambda: "cad"
    v2.choose_cad_source_mode = lambda has_existing: "existing"
    v2.choose_post_action = lambda: "keep"


def _line_len_cm(ln):
    dx = float(ln.get("x2", 0.0)) - float(ln.get("x1", 0.0))
    dy = float(ln.get("y2", 0.0)) - float(ln.get("y1", 0.0))
    return math.sqrt((dx * dx) + (dy * dy))


def _line_key(ln):
    p1 = (round(float(ln.get("x1", 0.0)), 3), round(float(ln.get("y1", 0.0)), 3))
    p2 = (round(float(ln.get("x2", 0.0)), 3), round(float(ln.get("y2", 0.0)), 3))
    return (p1, p2) if p1 <= p2 else (p2, p1)


def _line_mid_cm(ln):
    return (
        (float(ln.get("x1", 0.0)) + float(ln.get("x2", 0.0))) * 0.5,
        (float(ln.get("y1", 0.0)) + float(ln.get("y2", 0.0))) * 0.5,
    )


def _line_axis_data(ln):
    x1 = float(ln.get("x1", 0.0))
    y1 = float(ln.get("y1", 0.0))
    x2 = float(ln.get("x2", 0.0))
    y2 = float(ln.get("y2", 0.0))
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt((dx * dx) + (dy * dy))
    if length <= 1.0e-9:
        return None
    ux = dx / length
    uy = dy / length
    if ux < -1.0e-9 or (abs(ux) < 1.0e-9 and uy < 0.0):
        ux = -ux
        uy = -uy
    return (x1, y1, x2, y2, ux, uy, length)


def _overlap_ratio_parallel(a, b):
    ad = _line_axis_data(a)
    bd = _line_axis_data(b)
    if ad is None or bd is None:
        return 0.0, 0.0
    ux = ad[4]
    uy = ad[5]
    a1 = ad[0] * ux + ad[1] * uy
    a2 = ad[2] * ux + ad[3] * uy
    b1 = bd[0] * ux + bd[1] * uy
    b2 = bd[2] * ux + bd[3] * uy
    amin = min(a1, a2)
    amax = max(a1, a2)
    bmin = min(b1, b2)
    bmax = max(b1, b2)
    overlap = min(amax, bmax) - max(amin, bmin)
    if overlap <= 0.0:
        return 0.0, 0.0
    base = min(ad[6], bd[6])
    if base <= 1.0e-9:
        return 0.0, 0.0
    return overlap / base, overlap


def _parallel_offset_cm(a, b):
    ad = _line_axis_data(a)
    bd = _line_axis_data(b)
    if ad is None or bd is None:
        return 1.0e9
    dot = abs((ad[4] * bd[4]) + (ad[5] * bd[5]))
    if dot < 0.995:
        return 1.0e9
    nx = -ad[5]
    ny = ad[4]
    mx, my = _line_mid_cm(b)
    return abs(((mx - ad[0]) * nx) + ((my - ad[1]) * ny))


def _dedupe_lines(lines, min_len_cm):
    out = []
    seen = set()
    for ln in (lines or []):
        if _line_len_cm(ln) < float(min_len_cm):
            continue
        k = _line_key(ln)
        if k in seen:
            continue
        seen.add(k)
        out.append(ln)
    return out


def _build_endpoint_map(lines, tol_cm):
    endpoint_map = {}
    for i, ln in enumerate(lines):
        k1 = _pt_key_cm(ln.get("x1", 0.0), ln.get("y1", 0.0), tol_cm)
        k2 = _pt_key_cm(ln.get("x2", 0.0), ln.get("y2", 0.0), tol_cm)
        endpoint_map.setdefault(k1, []).append(i)
        endpoint_map.setdefault(k2, []).append(i)
    return endpoint_map


def _prune_short_leaf_lines(lines, tol_cm, max_len_cm):
    lines = list(lines or [])
    if not lines:
        return lines

    keep = [True] * len(lines)
    changed = True
    while changed:
        changed = False
        active_lines = [lines[i] for i in range(len(lines)) if keep[i]]
        if not active_lines:
            break
        endpoint_map = _build_endpoint_map(active_lines, tol_cm)
        key_to_degree = {}
        for key, ids in endpoint_map.items():
            key_to_degree[key] = len(ids)

        active_idx = 0
        for i in range(len(lines)):
            if not keep[i]:
                continue
            ln = active_lines[active_idx]
            active_idx += 1
            if _line_len_cm(ln) > max_len_cm:
                continue
            k1 = _pt_key_cm(ln.get("x1", 0.0), ln.get("y1", 0.0), tol_cm)
            k2 = _pt_key_cm(ln.get("x2", 0.0), ln.get("y2", 0.0), tol_cm)
            d1 = key_to_degree.get(k1, 0)
            d2 = key_to_degree.get(k2, 0)
            if d1 <= 1 or d2 <= 1:
                keep[i] = False
                changed = True

    return [lines[i] for i in range(len(lines)) if keep[i]]


def _suppress_parallel_duplicates(lines, perp_tol_cm, overlap_ratio_min):
    lines = list(lines or [])
    if len(lines) <= 1:
        return lines

    keep = [True] * len(lines)
    for i in range(len(lines)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(lines)):
            if not keep[j]:
                continue
            off = _parallel_offset_cm(lines[i], lines[j])
            if off > perp_tol_cm:
                continue
            overlap_ratio, overlap_len = _overlap_ratio_parallel(lines[i], lines[j])
            if overlap_ratio < overlap_ratio_min or overlap_len <= 0.0:
                continue
            len_i = _line_len_cm(lines[i])
            len_j = _line_len_cm(lines[j])
            if len_i >= len_j:
                keep[j] = False
            else:
                keep[i] = False
                break
    return [lines[i] for i in range(len(lines)) if keep[i]]


def _bridge_raw_wall_faces(rec, lines, gap_cm, perp_tol_cm):
    lines = list(lines or [])
    if len(lines) <= 1:
        return lines
    bridged = rec._merge_collinear_overlapping(lines, perp_tol=perp_tol_cm, gap_tol=gap_cm)
    return _dedupe_lines(bridged, 1.0)


def _pt_key_cm(x, y, tol_cm):
    if tol_cm <= 1.0e-9:
        return (round(float(x), 4), round(float(y), 4))
    return (int(round(float(x) / tol_cm)), int(round(float(y) / tol_cm)))


def _largest_component_lines(lines, tol_cm):
    lines = list(lines or [])
    if len(lines) <= 1:
        return lines

    endpoint_map = {}
    for i, ln in enumerate(lines):
        k1 = _pt_key_cm(ln.get("x1", 0.0), ln.get("y1", 0.0), tol_cm)
        k2 = _pt_key_cm(ln.get("x2", 0.0), ln.get("y2", 0.0), tol_cm)
        endpoint_map.setdefault(k1, []).append(i)
        endpoint_map.setdefault(k2, []).append(i)

    adj = {}
    for i in range(len(lines)):
        adj[i] = set()
    for ids in endpoint_map.values():
        if len(ids) <= 1:
            continue
        for a in ids:
            for b in ids:
                if a != b:
                    adj[a].add(b)

    visited = set()
    best = None
    best_len = -1.0
    for i in range(len(lines)):
        if i in visited:
            continue
        stack = [i]
        visited.add(i)
        comp = []
        total_len = 0.0
        while stack:
            cur = stack.pop()
            comp.append(cur)
            total_len += _line_len_cm(lines[cur])
            for nb in adj.get(cur, []):
                if nb in visited:
                    continue
                visited.add(nb)
                stack.append(nb)
        if total_len > best_len:
            best_len = total_len
            best = comp

    if not best:
        return lines
    return [lines[i] for i in best]


def _connected_components(lines, tol_cm):
    lines = list(lines or [])
    if not lines:
        return []

    endpoint_map = _build_endpoint_map(lines, tol_cm)
    adj = {}
    for i in range(len(lines)):
        adj[i] = set()
    for ids in endpoint_map.values():
        if len(ids) <= 1:
            continue
        for a in ids:
            for b in ids:
                if a != b:
                    adj[a].add(b)

    comps = []
    seen = set()
    for i in range(len(lines)):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj.get(cur, []):
                if nb in seen:
                    continue
                seen.add(nb)
                stack.append(nb)
        comps.append([lines[j] for j in comp])
    return comps


def _bbox_of_lines(lines):
    pts = []
    for ln in (lines or []):
        pts.append((float(ln.get("x1", 0.0)), float(ln.get("y1", 0.0))))
        pts.append((float(ln.get("x2", 0.0)), float(ln.get("y2", 0.0))))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _remove_small_components(lines, tol_cm, max_total_len_cm, max_dim_cm):
    out = []
    for comp in _connected_components(lines, tol_cm):
        bbox = _bbox_of_lines(comp)
        if not bbox:
            continue
        total_len = sum([_line_len_cm(ln) for ln in comp])
        dim_x = bbox[2] - bbox[0]
        dim_y = bbox[3] - bbox[1]
        max_dim = max(dim_x, dim_y)
        if total_len <= max_total_len_cm and max_dim <= max_dim_cm:
            continue
        out.extend(comp)
    return out


def _remove_small_interior_fragments(lines, tol_cm, max_lines, max_total_len_cm, max_dim_cm):
    out = []
    for comp in _connected_components(lines, tol_cm):
        bbox = _bbox_of_lines(comp)
        if not bbox:
            continue
        total_len = sum([_line_len_cm(ln) for ln in comp])
        dim_x = bbox[2] - bbox[0]
        dim_y = bbox[3] - bbox[1]
        max_dim = max(dim_x, dim_y)
        if len(comp) <= max_lines and total_len <= max_total_len_cm and max_dim <= max_dim_cm:
            continue
        out.extend(comp)
    return out


def _remove_tiny_through_segments(lines, tol_cm, max_len_cm):
    lines = list(lines or [])
    if not lines:
        return lines

    endpoint_map = _build_endpoint_map(lines, tol_cm)
    endpoint_degree = {}
    for key, ids in endpoint_map.items():
        endpoint_degree[key] = len(ids)

    out = []
    for ln in lines:
        seg_len = _line_len_cm(ln)
        k1 = _pt_key_cm(ln.get("x1", 0.0), ln.get("y1", 0.0), tol_cm)
        k2 = _pt_key_cm(ln.get("x2", 0.0), ln.get("y2", 0.0), tol_cm)
        d1 = endpoint_degree.get(k1, 0)
        d2 = endpoint_degree.get(k2, 0)
        if seg_len <= max_len_cm and d1 >= 2 and d2 >= 2:
            continue
        out.append(ln)
    return out


def _keep_lines_in_bbox(lines, bbox, margin_cm):
    if not bbox:
        return list(lines or [])
    minx = bbox[0] - margin_cm
    miny = bbox[1] - margin_cm
    maxx = bbox[2] + margin_cm
    maxy = bbox[3] + margin_cm
    out = []
    for ln in (lines or []):
        mx, my = _line_mid_cm(ln)
        if minx <= mx <= maxx and miny <= my <= maxy:
            out.append(ln)
    return out


def _median(vals, fallback):
    arr = sorted([float(v) for v in (vals or []) if float(v) > 0.0])
    if not arr:
        return float(fallback)
    n = len(arr)
    m = n // 2
    if (n % 2) == 1:
        return float(arr[m])
    return float((arr[m - 1] + arr[m]) * 0.5)


def _estimate_thickness_cm(collapse_dbg, fallback_cm):
    if isinstance(collapse_dbg, dict):
        est = collapse_dbg.get("estimated_wall_thickness_cm")
        try:
            if est is not None and float(est) > 0.0:
                return float(est)
        except Exception:
            pass
        dists = collapse_dbg.get("pair_distances_cm") or []
        return _median(dists, fallback_cm)
    return float(fallback_cm)


def _collapse_paired_centerlines_only(rec, lines, cfg, fallback_cm):
    lines = list(lines or [])
    if len(lines) < 2:
        return [], {
            "input_count": len(lines),
            "output_count": 0,
            "paired_count": 0,
            "estimated_wall_thickness_cm": float(fallback_cm),
            "pair_distances_cm": [],
        }

    pairs, pair_dists, used = rec._find_wall_pairs(lines, cfg)
    if not pairs:
        return [], {
            "input_count": len(lines),
            "output_count": 0,
            "paired_count": 0,
            "estimated_wall_thickness_cm": float(fallback_cm),
            "pair_distances_cm": [],
        }

    center = rec._collapse_to_centerlines(lines, pairs, include_unpaired=False)
    est = _median(pair_dists, fallback_cm)
    return center, {
        "input_count": len(lines),
        "output_count": len(center),
        "paired_count": len(pairs),
        "estimated_wall_thickness_cm": float(est),
        "pair_distances_cm": [float(v) for v in pair_dists],
    }


def _wall_type_name(v2, wt):
    try:
        p = wt.get_Parameter(v2.BuiltInParameter.SYMBOL_NAME_PARAM)
        if p is not None:
            n = p.AsString()
            if n:
                return n
    except Exception:
        pass
    try:
        return wt.Name
    except Exception:
        return ""


def _pick_wall_type(v2, target_cm, name_tokens):
    wall_types = [wt for wt in v2.FilteredElementCollector(v2.doc).OfClass(v2.WallType) if wt.Kind == v2.WallKind.Basic]
    if not wall_types:
        return None

    target_ft = v2.cm_to_ft(float(target_cm))
    typed = []
    for wt in wall_types:
        nm = _wall_type_name(v2, wt).upper()
        for tok in (name_tokens or []):
            if str(tok).upper() in nm:
                typed.append(wt)
                break

    candidates = typed if typed else wall_types
    best = None
    best_delta = None
    for wt in candidates:
        try:
            w = float(wt.Width)
        except Exception:
            continue
        d = abs(w - target_ft)
        if best is None or d < best_delta:
            best = wt
            best_delta = d

    if best is not None:
        return best
    return v2.get_wall_type_nearest(target_ft)


def _create_walls_from_lines(v2, lines_cm, level, wall_type, min_len_cm):
    ids = []
    if wall_type is None:
        return ids
    for ln in (lines_cm or []):
        if _line_len_cm(ln) < float(min_len_cm):
            continue
        p0 = v2.XYZ(v2.cm_to_ft(float(ln["x1"])), v2.cm_to_ft(float(ln["y1"])), 0.0)
        p1 = v2.XYZ(v2.cm_to_ft(float(ln["x2"])), v2.cm_to_ft(float(ln["y2"])), 0.0)
        try:
            wall = v2.Wall.Create(v2.doc, v2.Line.CreateBound(p0, p1), wall_type.Id, level.Id, v2.cm_to_ft(300.0), 0.0, False, False)
            try:
                v2.set_wall_location_centerline(wall)
            except Exception:
                pass
            ids.append(wall.Id.IntegerValue)
        except Exception:
            continue
    return ids


def _safe_layer_name(doc, geom_obj):
    try:
        gs_id = geom_obj.GraphicsStyleId
        if gs_id and gs_id != DB.ElementId.InvalidElementId:
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


def _append_line_cm(out_lines, p1, p2, layer, min_len_cm, v2):
    x1 = v2.ft_to_cm(p1.X)
    y1 = v2.ft_to_cm(p1.Y)
    x2 = v2.ft_to_cm(p2.X)
    y2 = v2.ft_to_cm(p2.Y)
    ln = {
        "type": "line",
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "length_cm": math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
        "layer": layer,
    }
    if ln["length_cm"] < min_len_cm:
        return
    out_lines.append(ln)


def _walk_geom_symbol_only(doc, geom_enum, tf, out_lines, min_len_cm, v2):
    for obj in geom_enum:
        if isinstance(obj, DB.GeometryInstance):
            nxt_tf = _compose_tf(tf, obj.Transform)
            try:
                # Use symbol geometry path only to avoid duplicated translated copies.
                sub = obj.GetSymbolGeometry()
                _walk_geom_symbol_only(doc, sub, nxt_tf, out_lines, min_len_cm, v2)
            except Exception:
                pass
            continue

        layer = _safe_layer_name(doc, obj)
        if isinstance(obj, DB.Line):
            p1 = _apply_tf(obj.GetEndPoint(0), tf)
            p2 = _apply_tf(obj.GetEndPoint(1), tf)
            _append_line_cm(out_lines, p1, p2, layer, min_len_cm, v2)
            continue

        if isinstance(obj, DB.PolyLine):
            try:
                pts = obj.GetCoordinates()
            except Exception:
                pts = []
            for i in range(0, max(0, len(pts) - 1)):
                p1 = _apply_tf(pts[i], tf)
                p2 = _apply_tf(pts[i + 1], tf)
                _append_line_cm(out_lines, p1, p2, layer, min_len_cm, v2)


def _extract_from_selected_import(v2, selected_import, view, cfg):
    inst = selected_import
    if inst is None:
        raise Exception("Selected import is not available.")
    if inst.get_BoundingBox(view) is None:
        raise Exception("Selected import is not visible in active view.")

    min_len_cm = float(cfg.get("min_segment_mm", 8.0)) / 10.0
    out = {
        "meta": {
            "instance_id": inst.Id.IntegerValue,
            "view_id": view.Id.IntegerValue,
            "units": "cm",
        },
        "lines": [],
        "arcs": [],
    }

    opts = DB.Options()
    opts.View = view
    opts.IncludeNonVisibleObjects = False
    opts.ComputeReferences = False

    geom = inst.get_Geometry(opts)
    _walk_geom_symbol_only(v2.doc, geom, None, out["lines"], min_len_cm, v2)
    out["lines"] = _dedupe_lines(out["lines"], min_len_cm)
    out["meta"]["line_count"] = len(out["lines"])
    out["meta"]["arc_count"] = 0
    return out


def _apply_layer_first_wall_mode(v2, selected_import):
    rec = _load_recognition_helpers()

    # Force extraction from the selected import only and avoid duplicate geometry path.
    def _extract_selected_only(doc, view, cfg, target_instance_id=None):
        return _extract_from_selected_import(v2, selected_import, view, cfg)

    v2.extract_cad_from_view = _extract_selected_only

    # Strict layer map: only requested wall layers.
    orig_load_layer_map = v2.load_layer_map

    def _load_layer_map_strict(path):
        lm = dict(orig_load_layer_map(path) or {})
        lm["walls"] = [r"^A-WALL-EXT$", r"^A-WALL-INT$"]
        lm["doors"] = []
        lm["windows"] = []
        return lm

    v2.load_layer_map = _load_layer_map_strict

    # Keep only wall lines from these layers and ignore all openings.
    orig_classify = v2.classify_entities

    def _classify_strict(lines, arcs, layer_map, cfg=None):
        out = dict(orig_classify(lines, arcs, layer_map, cfg=cfg) or {})
        wl = list(out.get("wall_lines") or []) + list(out.get("unclassified_lines") or [])
        strict = []
        for ln in wl:
            layer = str(ln.get("layer", "")).strip().upper()
            if layer in ("A-WALL-EXT", "A-WALL-INT"):
                strict.append(ln)
        out["wall_lines"] = strict
        out["door_lines"] = []
        out["window_lines"] = []
        out["door_arcs"] = []
        out["window_arcs"] = []
        out["unclassified_lines"] = []
        out["unclassified_arcs"] = []
        out["all_line_candidates"] = list(strict)
        return out

    v2.classify_entities = _classify_strict

    # Recognition output is intentionally minimal; wall creation is layer-first below.
    def _recognize_minimal(classified, cfg):
        return {
            "room_polygon_cm": [],
            "wall_segments_cm": [],
            "internal_walls_cm": [],
            "openings": [],
            "measurements_cm": {},
        }

    v2.recognize_topology = _recognize_minimal

    # Layer-first wall builder: continuous walls, no door/window cuts.
    def _build_layer_first(level, topology, cfg, snapshot, classified=None):
        cfg = dict(cfg or {})
        classified = dict(classified or {})

        min_len_cm = float(cfg.get("model_wall_min_length_cm", 20.0))
        close_gap_ext_cm = float(cfg.get("continuous_gap_close_cm_ext", 180.0))
        close_gap_int_cm = float(cfg.get("continuous_gap_close_cm_int", 140.0))
        cleanup_tol_cm = float(cfg.get("continuous_cleanup_snap_cm", 6.0))
        raw_dup_tol_cm = float(cfg.get("continuous_raw_duplicate_tol_cm", 1.5))

        wall_lines = list(classified.get("wall_lines") or [])
        wall_lines = _dedupe_lines(wall_lines, min_len_cm)

        # Safety: if CAD came in duplicated (translated copy), keep one model footprint.
        ext_all = [ln for ln in wall_lines if str(ln.get("layer", "")).strip().upper() == "A-WALL-EXT"]
        ext_main = _largest_component_lines(ext_all, tol_cm=6.0)
        ext_bbox = _bbox_of_lines(ext_main)
        if ext_bbox:
            wall_lines = _keep_lines_in_bbox(wall_lines, ext_bbox, margin_cm=180.0)

        ext_raw = [ln for ln in wall_lines if str(ln.get("layer", "")).strip().upper() == "A-WALL-EXT"]
        int_raw = [ln for ln in wall_lines if str(ln.get("layer", "")).strip().upper() == "A-WALL-INT"]

        ext_raw = _dedupe_lines(ext_raw, min_len_cm)
        int_raw = _dedupe_lines(int_raw, min_len_cm)

        # Bridge wall-face gaps before pairing so openings do not break the wall run.
        ext_raw = _suppress_parallel_duplicates(ext_raw, raw_dup_tol_cm, 0.92)
        int_raw = _suppress_parallel_duplicates(int_raw, raw_dup_tol_cm, 0.92)
        ext_raw = _bridge_raw_wall_faces(rec, ext_raw, close_gap_ext_cm, max(6.0, raw_dup_tol_cm * 2.0))
        int_raw = _bridge_raw_wall_faces(rec, int_raw, close_gap_int_cm, max(6.0, raw_dup_tol_cm * 2.0))

        ext_center, ext_dbg = _collapse_paired_centerlines_only(
            rec,
            ext_raw,
            cfg,
            float(cfg.get("default_wall_thickness_cm", 20.0)),
        )
        int_center, int_dbg = _collapse_paired_centerlines_only(
            rec,
            int_raw,
            cfg,
            float(cfg.get("default_internal_wall_thickness_cm", 15.0)),
        )

        ext_thick_cm = _estimate_thickness_cm(ext_dbg, float(cfg.get("default_wall_thickness_cm", 20.0)))
        int_fallback_cm = float(cfg.get("default_internal_wall_thickness_cm", min(15.0, max(10.0, ext_thick_cm * 0.7))))
        int_thick_cm = _estimate_thickness_cm(int_dbg, int_fallback_cm)

        # Interior traces still need pruning for door-swing/jamb leftovers.
        int_center = _prune_short_leaf_lines(int_center, cleanup_tol_cm, max(70.0, int_thick_cm * 2.0))
        ext_center = _largest_component_lines(ext_center, cleanup_tol_cm)
        int_center = _remove_small_components(int_center, cleanup_tol_cm, max(220.0, int_thick_cm * 6.0), max(110.0, int_thick_cm * 2.5))
        int_center = _remove_small_interior_fragments(
            int_center,
            cleanup_tol_cm,
            2,
            max(140.0, int_thick_cm * 3.0),
            max(60.0, int_thick_cm * 1.4),
        )

        # Final consolidation on centerlines after pairing.
        ext_center = rec._merge_collinear_overlapping(ext_center, perp_tol=6.0, gap_tol=4.0)
        int_center = rec._merge_collinear_overlapping(int_center, perp_tol=6.0, gap_tol=4.0)
        ext_center = rec._extend_to_intersections(ext_center, ext_tol=max(50.0, close_gap_ext_cm))
        int_center = rec._extend_to_intersections(int_center, ext_tol=max(40.0, close_gap_int_cm))

        # After bridging/intersections, remove near-parallel duplicate centerlines.
        ext_center = _suppress_parallel_duplicates(ext_center, max(8.0, ext_thick_cm * 0.35), 0.75)
        int_center = _suppress_parallel_duplicates(int_center, max(6.0, int_thick_cm * 0.35), 0.70)
        int_center = _remove_tiny_through_segments(
            int_center,
            cleanup_tol_cm,
            max(45.0, int_thick_cm * 1.25),
        )

        ext_center = _dedupe_lines(ext_center, min_len_cm)
        int_center = _dedupe_lines(int_center, min_len_cm)

        ext_type = _pick_wall_type(v2, ext_thick_cm, ["EXTERIOR", "EXT"])
        int_type = _pick_wall_type(v2, int_thick_cm, ["INTERIOR", "INT", "PARTITION"])
        if ext_type is None:
            raise Exception("No Basic wall type found for exterior walls.")
        if int_type is None:
            int_type = ext_type

        wall_ids = []
        internal_wall_ids = []
        t = v2.Transaction(v2.doc, "Create Model From CAD V2 (C2Rv6 Layer-First Walls)")
        t.Start()
        try:
            wall_ids = _create_walls_from_lines(v2, ext_center, level, ext_type, min_len_cm)
            internal_wall_ids = _create_walls_from_lines(v2, int_center, level, int_type, min_len_cm)
            t.Commit()
        except Exception:
            try:
                t.RollBack()
            except Exception:
                pass
            raise

        if snapshot is not None:
            try:
                snapshot.log("Layer-first walls created: ext={} int={}".format(len(wall_ids), len(internal_wall_ids)))
                snapshot.save_json("08_geometry_summary.json", {
                    "geometry": {
                        "wall_ids": wall_ids,
                        "internal_wall_ids": internal_wall_ids,
                        "door_ids": [],
                        "window_ids": [],
                        "opening_errors": [],
                        "perimeter_wall_thickness_cm": float(ext_thick_cm),
                        "internal_wall_thickness_cm": float(int_thick_cm),
                    }
                })
            except Exception:
                pass

        return {
            "geometry": {
                "wall_ids": wall_ids,
                "internal_wall_ids": internal_wall_ids,
                "door_ids": [],
                "window_ids": [],
                "opening_errors": [],
                "perimeter_wall_thickness_cm": float(ext_thick_cm),
                "internal_wall_thickness_cm": float(int_thick_cm),
            },
            "dimensions": {
                "ok": False,
                "note": "Dimensions disabled in C2Rv6 layer-first wall mode.",
            }
        }

    v2.build_model_from_topology = _build_layer_first


def main():
    uidoc = __revit__.ActiveUIDocument
    if uidoc is None:
        TaskDialog.Show("C2Rv6", "No active Revit document.")
        return

    selected_import = _pick_dwg_import(uidoc)
    if selected_import is None:
        TaskDialog.Show("C2Rv6", "Canceled: no DWG import selected.")
        return

    v2 = _load_v2_module()
    if not hasattr(v2, "run_command"):
        raise Exception("CreateFromCADV2 script is missing run_command().")

    _apply_selected_import_scope(v2, selected_import)
    _apply_layer_first_wall_mode(v2, selected_import)
    v2.run_command()


if __name__ == "__main__":
    main()
