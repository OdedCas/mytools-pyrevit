# -*- coding: utf-8 -*-
__title__ = "C2Rv6_C"
__doc__ = "CAD to Revit V6C. Improved stepped wall handling for A-WALL-EXT / A-WALL-INT layers."

import imp
import os
import sys
import math

from Autodesk.Revit import DB
from Autodesk.Revit.DB import (
    BuiltInCategory,
    ElementId,
    FamilySymbol,
    FilteredElementCollector,
    ImportInstance,
    WallType,
)
from Autodesk.Revit.DB.Structure import StructuralType
from Autodesk.Revit.Exceptions import OperationCanceledException
from Autodesk.Revit.UI import TaskDialog
from Autodesk.Revit.UI.Selection import ISelectionFilter, ObjectType

import re


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


def _remove_tiny_through_segments(lines, tol_cm, max_len_cm, protected_indices=None):
    lines = list(lines or [])
    if not lines:
        return lines
    _prot = protected_indices or set()

    endpoint_map = _build_endpoint_map(lines, tol_cm)
    endpoint_degree = {}
    for key, ids in endpoint_map.items():
        endpoint_degree[key] = len(ids)

    out = []
    for i, ln in enumerate(lines):
        if i in _prot:
            out.append(ln)
            continue
        seg_len = _line_len_cm(ln)
        k1 = _pt_key_cm(ln.get("x1", 0.0), ln.get("y1", 0.0), tol_cm)
        k2 = _pt_key_cm(ln.get("x2", 0.0), ln.get("y2", 0.0), tol_cm)
        d1 = endpoint_degree.get(k1, 0)
        d2 = endpoint_degree.get(k2, 0)
        if seg_len <= max_len_cm and d1 >= 2 and d2 >= 2:
            continue
        out.append(ln)
    return out


def _build_node_edge_graph(lines, tol_cm):
    nodes = {}
    edge_nodes = []
    for i, ln in enumerate(lines or []):
        k1 = _pt_key_cm(ln.get("x1", 0.0), ln.get("y1", 0.0), tol_cm)
        k2 = _pt_key_cm(ln.get("x2", 0.0), ln.get("y2", 0.0), tol_cm)
        edge_nodes.append((k1, k2))
        nodes.setdefault(k1, set()).add(i)
        nodes.setdefault(k2, set()).add(i)
    return nodes, edge_nodes


def _biconnected_edge_blocks(lines, tol_cm):
    lines = list(lines or [])
    if not lines:
        return []

    nodes, edge_nodes = _build_node_edge_graph(lines, tol_cm)
    adj = {}
    for node, edge_ids in nodes.items():
        nbrs = []
        for edge_id in edge_ids:
            a, b = edge_nodes[edge_id]
            other = b if node == a else a
            nbrs.append((other, edge_id))
        adj[node] = nbrs

    disc = {}
    low = {}
    parent = {}
    time_ref = [0]
    edge_stack = []
    blocks = []

    def _dfs(u):
        time_ref[0] += 1
        disc[u] = time_ref[0]
        low[u] = time_ref[0]

        for v, edge_id in adj.get(u, []):
            if v not in disc:
                parent[v] = u
                edge_stack.append(edge_id)
                _dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] >= disc[u]:
                    block = set()
                    while edge_stack:
                        last = edge_stack.pop()
                        block.add(last)
                        if last == edge_id:
                            break
                    if block:
                        blocks.append(block)
            elif parent.get(u) != v and disc[v] < disc[u]:
                edge_stack.append(edge_id)
                low[u] = min(low[u], disc[v])

    for node in nodes:
        if node in disc:
            continue
        _dfs(node)
        if edge_stack:
            blocks.append(set(edge_stack))
            edge_stack[:] = []

    return blocks


def _ext_proximity_set(int_lines, ext_lines, proximity_cm):
    """Return set of int_lines indices that have an endpoint near any ext_lines endpoint."""
    if not ext_lines or not int_lines:
        return set()
    ext_pts = []
    for ln in ext_lines:
        ext_pts.append((float(ln.get("x1", 0.0)), float(ln.get("y1", 0.0))))
        ext_pts.append((float(ln.get("x2", 0.0)), float(ln.get("y2", 0.0))))
    near = set()
    prox2 = proximity_cm * proximity_cm
    for i, ln in enumerate(int_lines):
        pts = [
            (float(ln.get("x1", 0.0)), float(ln.get("y1", 0.0))),
            (float(ln.get("x2", 0.0)), float(ln.get("y2", 0.0))),
        ]
        for ix, iy in pts:
            for ex, ey in ext_pts:
                if (ix - ex) ** 2 + (iy - ey) ** 2 <= prox2:
                    near.add(i)
                    break
            if i in near:
                break
    return near


def _collapse_small_attached_cycles(lines, tol_cm, max_edges, max_total_len_cm, max_dim_cm, protected_indices=None):
    lines = list(lines or [])
    if len(lines) < 4:
        return lines

    nodes, edge_nodes = _build_node_edge_graph(lines, tol_cm)
    remove_ids = set()

    for block in _biconnected_edge_blocks(lines, tol_cm):
        edge_ids = sorted(list(block))
        if len(edge_ids) < 3 or len(edge_ids) > int(max_edges):
            continue

        block_lines = [lines[i] for i in edge_ids]
        bbox = _bbox_of_lines(block_lines)
        if not bbox:
            continue
        total_len = sum([_line_len_cm(ln) for ln in block_lines])
        dim_x = bbox[2] - bbox[0]
        dim_y = bbox[3] - bbox[1]
        max_dim = max(dim_x, dim_y)
        if total_len > float(max_total_len_cm) or max_dim > float(max_dim_cm):
            continue

        block_edge_set = set(edge_ids)
        external_nodes = set()
        for edge_id in edge_ids:
            for node in edge_nodes[edge_id]:
                incident = nodes.get(node, set())
                if any([(other_id not in block_edge_set) for other_id in incident]):
                    external_nodes.add(node)

        if not external_nodes:
            continue

        # Skip this block if any edge is protected (near exterior wall).
        _prot = protected_indices or set()
        if any(eid in _prot for eid in edge_ids):
            continue

        kept = []
        removable = []
        for edge_id in edge_ids:
            n1, n2 = edge_nodes[edge_id]
            ext1 = n1 in external_nodes
            ext2 = n2 in external_nodes
            if ext1 and ext2:
                kept.append(edge_id)
            else:
                removable.append(edge_id)

        if not kept or not removable:
            continue

        for edge_id in removable:
            remove_ids.add(edge_id)

    return [lines[i] for i in range(len(lines)) if i not in remove_ids]


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


def _measure_local_thickness(centerline, raw_lines, max_search_cm=50.0):
    """Measure wall thickness at a centerline by finding the two closest
    parallel raw wall-face lines on opposite sides."""
    cl_data = _line_axis_data(centerline)
    if cl_data is None:
        return None
    cx, cy = _line_mid_cm(centerline)
    cl_ux, cl_uy = cl_data[4], cl_data[5]
    # Normal direction
    nx, ny = -cl_uy, cl_ux

    pos_dist = None  # closest on positive normal side
    neg_dist = None  # closest on negative normal side

    for raw in raw_lines:
        rd = _line_axis_data(raw)
        if rd is None:
            continue
        # Check parallel
        dot = abs(cl_ux * rd[4] + cl_uy * rd[5])
        if dot < 0.95:
            continue
        # Check overlap along the line axis
        ratio, _ = _overlap_ratio_parallel(centerline, raw)
        if ratio < 0.3:
            continue
        # Perpendicular signed distance from centerline to raw line midpoint
        rmx, rmy = _line_mid_cm(raw)
        perp = (rmx - cx) * nx + (rmy - cy) * ny
        d = abs(perp)
        if d < 1.0 or d > max_search_cm:
            continue  # skip nearly-coincident lines
        if perp > 0:
            if pos_dist is None or d < pos_dist:
                pos_dist = d
        else:
            if neg_dist is None or d < neg_dist:
                neg_dist = d

    if pos_dist is not None and neg_dist is not None:
        return pos_dist + neg_dist
    return None


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


def _create_walls_per_thickness(v2, lines_cm, raw_lines, level, fallback_thick_cm,
                                 name_tokens, min_len_cm, snapshot=None):
    """Create interior walls with per-segment thickness from the raw DWG data."""
    if not lines_cm:
        return []

    # Measure local thickness for each centerline
    thickness_map = {}
    for i, cl in enumerate(lines_cm):
        t = _measure_local_thickness(cl, raw_lines)
        if t is not None:
            # Round to nearest cm for grouping
            t_rounded = round(t)
            t_rounded = max(5, t_rounded)
        else:
            t_rounded = int(round(fallback_thick_cm))
        thickness_map[i] = t_rounded

    # Group lines by thickness
    groups = {}
    for i, cl in enumerate(lines_cm):
        t = thickness_map[i]
        groups.setdefault(t, []).append(cl)

    if snapshot:
        try:
            counts = {}
            for t in thickness_map.values():
                counts[t] = counts.get(t, 0) + 1
            snapshot.log("Interior wall thickness groups: {}".format(
                ", ".join("{}cm x{}".format(t, n) for t, n in sorted(counts.items()))))
        except Exception:
            pass

    # Create walls per thickness group
    ids = []
    for t_cm, group_lines in sorted(groups.items()):
        wt = _pick_wall_type(v2, t_cm, name_tokens)
        if wt is None:
            continue
        for ln in group_lines:
            if _line_len_cm(ln) < float(min_len_cm):
                continue
            p0 = v2.XYZ(v2.cm_to_ft(float(ln["x1"])), v2.cm_to_ft(float(ln["y1"])), 0.0)
            p1 = v2.XYZ(v2.cm_to_ft(float(ln["x2"])), v2.cm_to_ft(float(ln["y2"])), 0.0)
            try:
                wall = v2.Wall.Create(v2.doc, v2.Line.CreateBound(p0, p1), wt.Id, level.Id, v2.cm_to_ft(300.0), 0.0, False, False)
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
    # Cache raw data so opening detection can access A-DOORS/A-WINDOWS lines later.
    _RAW_CAD_CACHE["data"] = out
    return out


# ---------------------------------------------------------------------------
# Opening detection (from A-DOORS / A-WINDOWS layers)
# ---------------------------------------------------------------------------
_RAW_CAD_CACHE = {"data": None}


def _norm_layer_name(name):
    return re.sub(r"[^A-Z0-9]", "", str(name or "").upper())


def _is_opening_layer(layer_name, kind):
    n = _norm_layer_name(layer_name)
    if kind == "door":
        return "ADOOR" in n
    if kind == "window":
        return "AWINDOW" in n
    return False


def _opening_bbox(lines):
    if not lines:
        return None
    xs = []
    ys = []
    for ln in lines:
        xs.append(float(ln.get("x1", 0)))
        xs.append(float(ln.get("x2", 0)))
        ys.append(float(ln.get("y1", 0)))
        ys.append(float(ln.get("y2", 0)))
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_center_cm(bb):
    return ((bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5)


def _bbox_max_dim(bb):
    return max(bb[2] - bb[0], bb[3] - bb[1])


def _bbox_dist_cm(a, b):
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.sqrt(dx * dx + dy * dy)


def _merge_bbox(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _detect_opening_markers(raw_lines, kind):
    """Cluster lines from A-DOORS or A-WINDOWS into opening markers.
    Returns list of {center_cm, width_cm, kind}."""
    filtered = []
    for ln in (raw_lines or []):
        if _is_opening_layer(ln.get("layer", ""), kind):
            x1 = float(ln.get("x1", 0))
            y1 = float(ln.get("y1", 0))
            x2 = float(ln.get("x2", 0))
            y2 = float(ln.get("y2", 0))
            bb = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            seg_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            filtered.append({"ln": ln, "bbox": bb, "len": seg_len})

    if not filtered:
        return []

    # Single-pass union-find: merge all primitives within 120cm.
    # Door blocks have two jamb frames ~100cm apart + a swing line,
    # so 120cm is needed to merge all parts of one door.
    merge_dist = 120.0 if kind == "door" else 60.0
    parent = list(range(len(filtered)))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i, j):
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(filtered)):
        for j in range(i + 1, len(filtered)):
            if _bbox_dist_cm(filtered[i]["bbox"], filtered[j]["bbox"]) <= merge_dist:
                _union(i, j)

    groups = {}
    for idx in range(len(filtered)):
        groups.setdefault(_find(idx), []).append(filtered[idx])

    markers = []
    for group in groups.values():
        # Merge all bboxes in the group.
        bb = group[0]["bbox"]
        for ent in group[1:]:
            bb = _merge_bbox(bb, ent["bbox"])

        # Separate frame/jamb lines (short, <= 50cm) from swing arcs (long).
        frame_xs = []
        frame_ys = []
        swing_line = None
        max_seg_len = 0.0
        for ent in group:
            if ent["len"] > max_seg_len:
                max_seg_len = ent["len"]
                swing_line = ent["ln"]
            if ent["len"] <= 50.0:  # frame/jamb lines only
                frame_xs.append(float(ent["ln"].get("x1", 0)))
                frame_xs.append(float(ent["ln"].get("x2", 0)))
                frame_ys.append(float(ent["ln"].get("y1", 0)))
                frame_ys.append(float(ent["ln"].get("y2", 0)))

        bb_w = bb[2] - bb[0]
        bb_h = bb[3] - bb[1]

        if kind == "door":
            # Width = swing arc length = actual door leaf width.
            # The jamb-to-jamb bbox is wider (includes jamb blocks on both sides).
            # Swing arc is the longest line in the cluster (>= 80cm typically).
            if max_seg_len > 50.0:
                width = max(40.0, min(200.0, max_seg_len))
            elif frame_xs and frame_ys:
                # Fallback: jamb-to-jamb minus estimated jamb thickness (~5cm each side)
                jw = max(max(frame_xs) - min(frame_xs),
                         max(frame_ys) - min(frame_ys))
                width = max(40.0, min(200.0, jw - 10.0))
            else:
                width = max(40.0, min(200.0, max_seg_len))
        else:
            # Window: shorter bbox dim is the frame width along the wall.
            width = max(30.0, min(300.0, min(bb_w, bb_h) if min(bb_w, bb_h) > 10.0 else max(bb_w, bb_h)))

        # Center: midpoint of all frame line endpoints = center between jambs.
        if frame_xs and frame_ys:
            center = ((min(frame_xs) + max(frame_xs)) * 0.5,
                      (min(frame_ys) + max(frame_ys)) * 0.5)
        else:
            center = _bbox_center_cm(bb)

        # Swing arc data for door orientation matching.
        swing_data = None
        if kind == "door" and swing_line is not None and max_seg_len > 50.0:
            sx1 = float(swing_line.get("x1", 0))
            sy1 = float(swing_line.get("y1", 0))
            sx2 = float(swing_line.get("x2", 0))
            sy2 = float(swing_line.get("y2", 0))
            # The swing arc goes from hinge (near a jamb) toward open position.
            # Midpoint of the arc indicates which side the door swings into.
            swing_mid = ((sx1 + sx2) * 0.5, (sy1 + sy2) * 0.5)
            # Hinge is the arc endpoint closest to center (between jambs).
            d1 = math.sqrt((sx1 - center[0]) ** 2 + (sy1 - center[1]) ** 2)
            d2 = math.sqrt((sx2 - center[0]) ** 2 + (sy2 - center[1]) ** 2)
            if d1 <= d2:
                hinge_pt = (sx1, sy1)
                open_pt = (sx2, sy2)
            else:
                hinge_pt = (sx2, sy2)
                open_pt = (sx1, sy1)
            swing_data = {
                "hinge_cm": hinge_pt,
                "open_cm": open_pt,
                "swing_mid_cm": swing_mid,
            }

        markers.append({
            "kind": kind,
            "center_cm": center,
            "width_cm": width,
            "swing": swing_data,
        })

    # Dedupe markers closer than 60cm
    out = []
    for mk in markers:
        dup = False
        for ex in out:
            dx = mk["center_cm"][0] - ex["center_cm"][0]
            dy = mk["center_cm"][1] - ex["center_cm"][1]
            if math.sqrt(dx * dx + dy * dy) <= 60.0:
                dup = True
                break
        if not dup:
            out.append(mk)
    return out


def _find_host_wall(v2, all_wall_ids, center_cm, level_elevation_ft):
    """Find the Revit wall whose location line is closest to center_cm.
    Returns (wall, point_on_wall_ft, distance_ft) or (None, None, 1e9)."""
    cx = v2.cm_to_ft(float(center_cm[0]))
    cy = v2.cm_to_ft(float(center_cm[1]))
    pt = v2.XYZ(cx, cy, level_elevation_ft)

    best_wall = None
    best_pt = None
    best_dist = 1.0e9
    for wid in all_wall_ids:
        try:
            wall = v2.doc.GetElement(ElementId(wid))
        except Exception:
            continue
        if wall is None:
            continue
        loc = wall.Location
        if loc is None:
            continue
        try:
            curve = loc.Curve
        except Exception:
            continue
        if curve is None:
            continue
        try:
            result = curve.Project(pt)
            if result is not None and result.Distance < best_dist:
                best_dist = result.Distance
                best_wall = wall
                best_pt = result.XYZPoint
        except Exception:
            continue
    return best_wall, best_pt, best_dist


def _get_family_types(v2, category):
    """Get all loaded FamilySymbol entries for a BuiltInCategory."""
    out = []
    try:
        col = FilteredElementCollector(v2.doc).OfCategory(category).OfClass(FamilySymbol)
        for fs in col:
            out.append(fs)
    except Exception:
        pass
    return out


def _pick_family_type_by_width(v2, family_types, target_width_cm):
    """Pick the family type whose width parameter is closest to target."""
    target_ft = v2.cm_to_ft(float(target_width_cm))
    best = None
    best_delta = 1.0e9
    for fs in family_types:
        try:
            # Try common width parameters
            w = None
            for pname in ["Width", "width", "Rough Width"]:
                p = fs.LookupParameter(pname)
                if p is not None:
                    w = p.AsDouble()
                    break
            if w is None:
                w = target_ft  # assume match if no width param
            d = abs(w - target_ft)
            if d < best_delta:
                best = fs
                best_delta = d
        except Exception:
            if best is None:
                best = fs
    return best


def _set_sill_height(v2, inst, height_cm):
    """Set sill height on a window instance."""
    height_ft = v2.cm_to_ft(float(height_cm))
    for pname in ["Sill Height", "sill height", "Sill_Height"]:
        try:
            p = inst.LookupParameter(pname)
            if p is not None and not p.IsReadOnly:
                p.Set(height_ft)
                return True
        except Exception:
            continue
    # Try built-in sill height parameter
    try:
        p = inst.get_Parameter(DB.BuiltInParameter.INSTANCE_SILL_HEIGHT_PARAM)
        if p is not None and not p.IsReadOnly:
            p.Set(height_ft)
            return True
    except Exception:
        pass
    return False


def _pick_door_type_for_wall(v2, family_types, target_width_cm, is_interior, snapshot=None):
    """Pick a door family type appropriate for interior or exterior walls."""
    target_ft = v2.cm_to_ft(float(target_width_cm))

    # Split into interior and exterior candidates by name heuristic.
    # Check EXTERIOR first since names like "Exterior-Single" contain "Single".
    int_types = []
    ext_types = []
    for fs in family_types:
        try:
            name = (fs.Family.Name or "").upper()
        except Exception:
            name = ""
        if "EXTERIOR" in name or "EXT-" in name or "-EXT" in name:
            ext_types.append(fs)
        else:
            int_types.append(fs)

    if snapshot:
        try:
            int_names = [fs.Family.Name for fs in int_types]
            ext_names = [fs.Family.Name for fs in ext_types]
            snapshot.log("Door types: {} interior ({}), {} exterior ({})".format(
                len(int_types), ", ".join(set(int_names))[:200],
                len(ext_types), ", ".join(set(ext_names))[:200]))
        except Exception:
            pass

    candidates = int_types if is_interior else ext_types
    if not candidates:
        candidates = family_types  # fallback to all

    # Prefer user/custom families over Revit defaults (M_ prefix).
    # If any custom families exist, use ONLY those.
    custom = []
    default = []
    for fs in candidates:
        try:
            nm = (fs.Family.Name or "").upper()
        except Exception:
            nm = ""
        if nm.startswith("M_") or nm.startswith("M "):
            default.append(fs)
        else:
            custom.append(fs)
    ranked = custom if custom else default

    # Pick by closest width, preferring earlier (custom) entries on ties
    best = None
    best_delta = 1.0e9
    for fs in ranked:
        try:
            w = None
            for pname in ["Width", "width", "Rough Width"]:
                p = fs.LookupParameter(pname)
                if p is not None:
                    w = p.AsDouble()
                    break
            if w is None:
                w = target_ft
            d = abs(w - target_ft)
            if d < best_delta:
                best = fs
                best_delta = d
        except Exception:
            if best is None:
                best = fs
    return best


def _match_door_swing(v2, inst, wall, swing, center_cm, snapshot=None):
    """Flip door hand/facing to match DWG swing arc direction.

    swing dict has: hinge_cm, open_cm, swing_mid_cm.
    We compare the DWG swing direction against Revit's actual door orientation
    and flip as needed.
    """
    try:
        curve = wall.Location.Curve
        ws = curve.GetEndPoint(0)
        we = curve.GetEndPoint(1)
    except Exception:
        return
    wall_dx = we.X - ws.X
    wall_dy = we.Y - ws.Y
    wall_len = math.sqrt(wall_dx * wall_dx + wall_dy * wall_dy)
    if wall_len < 1e-9:
        return
    # Wall unit tangent and normal
    wtx = wall_dx / wall_len
    wty = wall_dy / wall_len
    wnx = -wty  # normal (left of wall direction)
    wny = wtx

    hinge = swing["hinge_cm"]
    open_pt = swing["open_cm"]
    cx, cy = center_cm

    # --- DWG swing direction ---
    # Vector from door center to the open endpoint of the swing arc.
    # This tells us which side of the wall the door swings into.
    open_dx = v2.cm_to_ft(open_pt[0] - cx)
    open_dy = v2.cm_to_ft(open_pt[1] - cy)
    # Project onto wall normal: positive = normal side, negative = opposite
    dwg_facing_dot = open_dx * wnx + open_dy * wny
    # Project onto wall tangent: tells us which direction the door opens along the wall
    dwg_tang_dot = open_dx * wtx + open_dy * wty

    # --- DWG hinge side ---
    # Project hinge position onto wall tangent relative to door center.
    hinge_dx = v2.cm_to_ft(hinge[0] - cx)
    hinge_dy = v2.cm_to_ft(hinge[1] - cy)
    dwg_hinge_tang = hinge_dx * wtx + hinge_dy * wty

    # --- Revit current orientation ---
    # FacingOrientation: direction the door faces (normal side)
    # HandOrientation: direction from hinge to latch along the wall
    try:
        facing_dir = inst.FacingOrientation
        hand_dir = inst.HandOrientation
    except Exception:
        return

    # Compare DWG facing side with Revit facing side
    # If the DWG open-point is on the opposite side from Revit's facing, flip facing
    revit_facing_dot = facing_dir.X * wnx + facing_dir.Y * wny
    need_flip_facing = (dwg_facing_dot * revit_facing_dot < 0)

    # Compare DWG hinge side with Revit hand direction
    # HandOrientation points from hinge toward latch.
    # DWG hinge is at dwg_hinge_tang along wall; latch is opposite.
    # So DWG "hand direction" along wall tangent = -sign(dwg_hinge_tang)
    revit_hand_tang = hand_dir.X * wtx + hand_dir.Y * wty
    dwg_hand_dir = -1 if dwg_hinge_tang > 0 else 1
    need_flip_hand = (dwg_hand_dir * revit_hand_tang < 0)

    if need_flip_facing:
        inst.flipFacing()
    if need_flip_hand:
        inst.flipHand()

    if snapshot:
        try:
            snapshot.log("  swing: facing_dot={:.2f} vs revit={:.2f} flip_facing={}, "
                         "hinge_tang={:.2f} hand_tang={:.2f} flip_hand={}".format(
                dwg_facing_dot, revit_facing_dot, need_flip_facing,
                dwg_hinge_tang, revit_hand_tang, need_flip_hand))
        except Exception:
            pass


def _place_openings_in_walls(v2, level, ext_wall_ids, int_wall_ids, markers,
                              category, snapshot, sill_height_cm=None):
    """Place door or window family instances into host walls.
    sill_height_cm: for windows, the sill height above floor (e.g. 105).
    Returns list of created element IDs."""
    all_wall_ids = list(ext_wall_ids or []) + list(int_wall_ids or [])
    if not markers or not all_wall_ids:
        return []

    int_wall_set = set(int_wall_ids or [])
    cat_name = "door" if category == BuiltInCategory.OST_Doors else "window"
    family_types = _get_family_types(v2, category)
    if not family_types:
        if snapshot:
            try:
                snapshot.log("No {} family types loaded, skipping placement.".format(cat_name))
            except Exception:
                pass
        return []

    # Get level elevation for correct Z coordinate
    level_elev_ft = 0.0
    try:
        level_elev_ft = float(level.Elevation)
    except Exception:
        pass

    max_snap_ft = v2.cm_to_ft(150.0)  # max 150cm snap distance
    ids = []
    errors = []

    if snapshot:
        try:
            snapshot.log("Placing {} {}s, {} family types available, level elev={:.2f}ft".format(
                len(markers), cat_name, len(family_types), level_elev_ft))
        except Exception:
            pass

    for mk in markers:
        wall, pt, dist = _find_host_wall(v2, all_wall_ids, mk["center_cm"], level_elev_ft)
        if wall is None or dist > max_snap_ft:
            errors.append("No host wall within 150cm for {} at ({:.0f}, {:.0f}), best dist={:.1f}ft".format(
                cat_name, mk["center_cm"][0], mk["center_cm"][1], dist))
            continue

        is_interior = wall.Id.IntegerValue in int_wall_set

        if category == BuiltInCategory.OST_Doors:
            fs = _pick_door_type_for_wall(v2, family_types, mk["width_cm"], is_interior, snapshot)
        else:
            fs = _pick_family_type_by_width(v2, family_types, mk["width_cm"])

        if fs is None:
            errors.append("No {} family type available".format(cat_name))
            continue

        try:
            if not fs.IsActive:
                fs.Activate()
                v2.doc.Regenerate()
        except Exception:
            pass

        try:
            # Place at level elevation Z
            place_pt = v2.XYZ(pt.X, pt.Y, level_elev_ft)
            inst = v2.doc.Create.NewFamilyInstance(
                place_pt, fs, wall, level, StructuralType.NonStructural)

            # Set sill height for windows
            if sill_height_cm is not None and category == BuiltInCategory.OST_Windows:
                _set_sill_height(v2, inst, sill_height_cm)

            # Match door swing direction from DWG
            if category == BuiltInCategory.OST_Doors and mk.get("swing"):
                try:
                    _match_door_swing(v2, inst, wall, mk["swing"], mk["center_cm"], snapshot)
                except Exception:
                    pass

            ids.append(inst.Id.IntegerValue)
            if snapshot:
                try:
                    wall_loc = wall.Location.Curve
                    w_start = wall_loc.GetEndPoint(0)
                    w_end = wall_loc.GetEndPoint(1)
                    snapshot.log(
                        "{} placed id={} at ({:.0f},{:.0f})cm on {} wall {} "
                        "({:.0f},{:.0f})->({:.0f},{:.0f})ft dist={:.1f}cm width={:.0f}cm type={}".format(
                            cat_name, inst.Id.IntegerValue,
                            mk["center_cm"][0], mk["center_cm"][1],
                            "INT" if is_interior else "EXT",
                            wall.Id.IntegerValue,
                            w_start.X, w_start.Y, w_end.X, w_end.Y,
                            v2.ft_to_cm(dist),
                            mk["width_cm"],
                            fs.Family.Name))
                except Exception:
                    pass
        except Exception as ex:
            errors.append("Failed placing {} at ({:.0f}, {:.0f}): {}".format(
                cat_name, mk["center_cm"][0], mk["center_cm"][1], str(ex)))

    if snapshot:
        try:
            snapshot.log("Placed {} of {} {}s".format(len(ids), len(markers), cat_name))
            for e in errors:
                snapshot.log(e)
        except Exception:
            pass

    return ids


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
        raw_min_len_cm = float(cfg.get("raw_dedup_min_len_cm", 5.0))
        close_gap_ext_cm = float(cfg.get("continuous_gap_close_cm_ext", 180.0))
        close_gap_int_cm = float(cfg.get("continuous_gap_close_cm_int", 140.0))
        cleanup_tol_cm = float(cfg.get("continuous_cleanup_snap_cm", 6.0))
        raw_dup_tol_cm = float(cfg.get("continuous_raw_duplicate_tol_cm", 1.5))

        wall_lines = list(classified.get("wall_lines") or [])
        wall_lines = _dedupe_lines(wall_lines, raw_min_len_cm)

        # Safety: if CAD came in duplicated (translated copy), keep one model footprint.
        ext_all = [ln for ln in wall_lines if str(ln.get("layer", "")).strip().upper() == "A-WALL-EXT"]
        ext_main = _largest_component_lines(ext_all, tol_cm=6.0)
        ext_bbox = _bbox_of_lines(ext_main)
        if ext_bbox:
            wall_lines = _keep_lines_in_bbox(wall_lines, ext_bbox, margin_cm=180.0)

        ext_raw = [ln for ln in wall_lines if str(ln.get("layer", "")).strip().upper() == "A-WALL-EXT"]
        int_raw = [ln for ln in wall_lines if str(ln.get("layer", "")).strip().upper() == "A-WALL-INT"]

        ext_raw = _dedupe_lines(ext_raw, raw_min_len_cm)
        int_raw = _dedupe_lines(int_raw, raw_min_len_cm)

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

        def _dump_lines(label, lines):
            if snapshot is None:
                return
            try:
                data = []
                for ln in (lines or []):
                    data.append({
                        "x1": round(float(ln.get("x1", 0)), 2),
                        "y1": round(float(ln.get("y1", 0)), 2),
                        "x2": round(float(ln.get("x2", 0)), 2),
                        "y2": round(float(ln.get("y2", 0)), 2),
                        "len": round(_line_len_cm(ln), 2),
                    })
                snapshot.save_json("debug_{}.json".format(label), {"count": len(data), "lines": data})
            except Exception:
                pass

        _dump_lines("00_int_raw", int_raw)
        _dump_lines("01_int_after_pair", int_center)

        # Interior traces still need pruning for door-swing/jamb leftovers.
        int_center = _prune_short_leaf_lines(int_center, cleanup_tol_cm, max(70.0, int_thick_cm * 2.0))
        _dump_lines("02_int_after_prune_leaf", int_center)
        ext_center = _largest_component_lines(ext_center, cleanup_tol_cm)
        int_center = _remove_small_components(int_center, cleanup_tol_cm, max(220.0, int_thick_cm * 6.0), max(110.0, int_thick_cm * 2.5))
        _dump_lines("03_int_after_remove_small", int_center)
        int_center = _remove_small_interior_fragments(
            int_center,
            cleanup_tol_cm,
            2,
            max(140.0, int_thick_cm * 3.0),
            max(60.0, int_thick_cm * 1.4),
        )
        _dump_lines("04_int_after_remove_frag", int_center)

        # Final consolidation on centerlines after pairing.
        ext_center = rec._merge_collinear_overlapping(ext_center, perp_tol=6.0, gap_tol=4.0)
        int_center = rec._merge_collinear_overlapping(int_center, perp_tol=6.0, gap_tol=4.0)
        _dump_lines("05_int_after_merge", int_center)
        ext_center = rec._extend_to_intersections(ext_center, ext_tol=max(50.0, close_gap_ext_cm))
        # NOTE: Do NOT extend interior centerlines a second time.
        # The first collapse already did extend+split. A second extend
        # over-reaches past T-junctions (e.g. step walls) creating false boxes.
        _dump_lines("06_int_after_extend", int_center)

        # After bridging/intersections, remove near-parallel duplicate centerlines.
        ext_center = _suppress_parallel_duplicates(ext_center, max(8.0, ext_thick_cm * 0.35), 0.75)
        int_center = _suppress_parallel_duplicates(int_center, max(6.0, int_thick_cm * 0.35), 0.70)
        _dump_lines("07_int_after_dedup_par", int_center)

        # Protect interior segments near exterior walls from aggressive cleanup.
        _prot = _ext_proximity_set(int_center, ext_center, max(50.0, ext_thick_cm * 2.0))

        int_center = _remove_tiny_through_segments(
            int_center,
            cleanup_tol_cm,
            max(45.0, int_thick_cm * 1.25),
            protected_indices=_prot,
        )
        _dump_lines("08_int_after_tiny_thru", int_center)
        int_center = _collapse_small_attached_cycles(
            int_center,
            cleanup_tol_cm,
            6,
            max(220.0, int_thick_cm * 8.0),
            max(90.0, int_thick_cm * 3.0),
            protected_indices=_prot,
        )
        _dump_lines("09_int_final", int_center)

        ext_center = _dedupe_lines(ext_center, min_len_cm)
        int_center = _dedupe_lines(int_center, min_len_cm)

        ext_type = _pick_wall_type(v2, ext_thick_cm, ["EXTERIOR", "EXT"])
        if ext_type is None:
            raise Exception("No Basic wall type found for exterior walls.")

        wall_ids = []
        internal_wall_ids = []
        t = v2.Transaction(v2.doc, "Create Model From CAD V2 (C2Rv6_C Layer-First Walls)")
        t.Start()
        try:
            wall_ids = _create_walls_from_lines(v2, ext_center, level, ext_type, min_len_cm)
            # Interior walls: per-segment thickness measured from raw DWG pairs
            internal_wall_ids = _create_walls_per_thickness(
                v2, int_center, int_raw, level, int_thick_cm,
                ["INTERIOR", "INT", "PARTITION"], min_len_cm, snapshot)
            t.Commit()
        except Exception:
            try:
                t.RollBack()
            except Exception:
                pass
            raise

        # --- Opening detection and placement ---
        door_ids = []
        window_ids = []
        opening_errors = []
        raw_data = _RAW_CAD_CACHE.get("data")
        if raw_data is not None:
            raw_lines = raw_data.get("lines") or []
            door_markers = _detect_opening_markers(raw_lines, "door")
            window_markers = _detect_opening_markers(raw_lines, "window")

            if snapshot:
                try:
                    snapshot.log("Opening markers detected: {} doors, {} windows".format(
                        len(door_markers), len(window_markers)))
                    for i, mk in enumerate(door_markers):
                        sw = mk.get("swing") or {}
                        snapshot.log("  door[{}] center=({:.0f},{:.0f}) width={:.0f}cm hinge={} open={}".format(
                            i, mk["center_cm"][0], mk["center_cm"][1], mk["width_cm"],
                            sw.get("hinge_cm", "?"), sw.get("open_cm", "?")))
                    for i, mk in enumerate(window_markers):
                        snapshot.log("  win[{}] center=({:.0f},{:.0f}) width={:.0f}cm".format(
                            i, mk["center_cm"][0], mk["center_cm"][1], mk["width_cm"]))
                except Exception:
                    pass

            if door_markers or window_markers:
                t2 = v2.Transaction(v2.doc, "C2Rv6_C Place Openings")
                t2.Start()
                try:
                    if door_markers:
                        door_ids = _place_openings_in_walls(
                            v2, level, wall_ids, internal_wall_ids,
                            door_markers,
                            BuiltInCategory.OST_Doors, snapshot,
                            sill_height_cm=0.0)
                    if window_markers:
                        window_ids = _place_openings_in_walls(
                            v2, level, wall_ids, internal_wall_ids,
                            window_markers,
                            BuiltInCategory.OST_Windows, snapshot,
                            sill_height_cm=105.0)
                    t2.Commit()
                except Exception as ex:
                    opening_errors.append(str(ex))
                    try:
                        t2.RollBack()
                    except Exception:
                        pass

        if snapshot is not None:
            try:
                snapshot.log("Layer-first walls created: ext={} int={}, doors={} windows={}".format(
                    len(wall_ids), len(internal_wall_ids), len(door_ids), len(window_ids)))
                snapshot.save_json("08_geometry_summary.json", {
                    "geometry": {
                        "wall_ids": wall_ids,
                        "internal_wall_ids": internal_wall_ids,
                        "door_ids": door_ids,
                        "window_ids": window_ids,
                        "opening_errors": opening_errors,
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
                "door_ids": door_ids,
                "window_ids": window_ids,
                "opening_errors": opening_errors,
                "perimeter_wall_thickness_cm": float(ext_thick_cm),
                "internal_wall_thickness_cm": float(int_thick_cm),
            },
            "dimensions": {
                "ok": False,
                "note": "Dimensions disabled in C2Rv6_C layer-first wall mode.",
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
