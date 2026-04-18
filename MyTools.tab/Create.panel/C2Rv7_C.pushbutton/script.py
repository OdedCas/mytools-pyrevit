# -*- coding: utf-8 -*-
__title__ = "C2Rv7_C"
__doc__ = "CAD to Revit V7C. Layer-first walls (A-WALL-EXT/A-WALL-INT) + doors, windows, floors, rooms, dimensions. Optional LLM QA on interior fragments."

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

from Autodesk.Revit.DB import IFailuresPreprocessor, FailureProcessingResult


class _SwallowTypeErrors(IFailuresPreprocessor):
    """Auto-delete failing elements so 'Can't make type' dialogs don't appear."""

    def PreprocessFailures(self, failuresAccessor):
        for msg in failuresAccessor.GetFailureMessages():
            try:
                sev = msg.GetSeverity()
                # Delete element on errors; dismiss warnings
                if sev.ToString() == "Error":
                    ids = msg.GetFailingElementIds()
                    if ids and ids.Count > 0:
                        failuresAccessor.DeleteElements(ids)
                    else:
                        failuresAccessor.DeleteWarning(msg)
                else:
                    failuresAccessor.DeleteWarning(msg)
            except Exception:
                try:
                    failuresAccessor.DeleteWarning(msg)
                except Exception:
                    pass
        return FailureProcessingResult.Continue


SCRIPT_DIR = os.path.dirname(__file__)
PANEL_DIR = os.path.dirname(SCRIPT_DIR)
V2_DIR = os.path.join(PANEL_DIR, "CreateFromCADV2.pushbutton")
_REC_MOD = None
_LLM_QA_MOD = None

if V2_DIR not in sys.path:
    sys.path.append(V2_DIR)


def _load_v2_module():
    path = os.path.join(V2_DIR, "script.py")
    try:
        return imp.load_source("c2rv7_v2_delegate", path)
    except Exception as ex:
        raise Exception("Failed loading CreateFromCADV2 module: {} ({})".format(path, ex))


def _load_recognition_helpers():
    global _REC_MOD
    if _REC_MOD is not None:
        return _REC_MOD
    path = os.path.join(V2_DIR, "v2_cad_recognition.py")
    try:
        _REC_MOD = imp.load_source("c2rv7_v2_recognition_helpers", path)
    except Exception as ex:
        raise Exception("Failed loading recognition helpers: {} ({})".format(path, ex))
    return _REC_MOD


def _load_llm_qa():
    """Load the LLM QA sidecar if present. Returns module or None; never raises."""
    global _LLM_QA_MOD
    if _LLM_QA_MOD is not None:
        return _LLM_QA_MOD
    path = os.path.join(SCRIPT_DIR, "c2rv7_llm_qa.py")
    if not os.path.isfile(path):
        return None
    try:
        _LLM_QA_MOD = imp.load_source("c2rv7_llm_qa", path)
    except Exception:
        _LLM_QA_MOD = None
    return _LLM_QA_MOD


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
            "Select DWG import/link for C2Rv7_C",
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


def _dedupe_arcs(arcs):
    out = []
    seen = set()
    for arc in (arcs or []):
        key = (
            round(float(arc.get("cx", 0.0)), 4),
            round(float(arc.get("cy", 0.0)), 4),
            round(abs(float(arc.get("r", 0.0))), 4),
            round(float(arc.get("sx", 0.0)), 4),
            round(float(arc.get("sy", 0.0)), 4),
            round(float(arc.get("ex", 0.0)), 4),
            round(float(arc.get("ey", 0.0)), 4),
            str(arc.get("layer", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(arc)
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


def _append_arc_cm(out_arcs, arc, layer, tf, v2):
    center = _apply_tf(arc.Center, tf)
    start = _apply_tf(arc.GetEndPoint(0), tf)
    end = _apply_tf(arc.GetEndPoint(1), tf)
    # Compute radius from transformed center-to-start distance.
    # arc.Radius is in symbol space and ignores block-insert scaling,
    # so for scaled blocks it would be wrong (e.g. 10x block -> 10x too small).
    dx = start.X - center.X
    dy = start.Y - center.Y
    dz = start.Z - center.Z
    r_ft = math.sqrt(dx * dx + dy * dy + dz * dz)
    out_arcs.append({
        "type": "arc",
        "cx": v2.ft_to_cm(center.X),
        "cy": v2.ft_to_cm(center.Y),
        "r": abs(v2.ft_to_cm(r_ft)),
        "sx": v2.ft_to_cm(start.X),
        "sy": v2.ft_to_cm(start.Y),
        "ex": v2.ft_to_cm(end.X),
        "ey": v2.ft_to_cm(end.Y),
        "layer": layer,
    })


def _walk_geom_symbol_only(doc, geom_enum, tf, out_lines, out_arcs, min_len_cm, v2):
    for obj in geom_enum:
        if isinstance(obj, DB.GeometryInstance):
            nxt_tf = _compose_tf(tf, obj.Transform)
            try:
                # Use symbol geometry path only to avoid duplicated translated copies.
                sub = obj.GetSymbolGeometry()
                _walk_geom_symbol_only(doc, sub, nxt_tf, out_lines, out_arcs, min_len_cm, v2)
            except Exception:
                pass
            continue

        layer = _safe_layer_name(doc, obj)
        if isinstance(obj, DB.Line):
            p1 = _apply_tf(obj.GetEndPoint(0), tf)
            p2 = _apply_tf(obj.GetEndPoint(1), tf)
            _append_line_cm(out_lines, p1, p2, layer, min_len_cm, v2)
            continue

        if isinstance(obj, DB.Arc):
            _append_arc_cm(out_arcs, obj, layer, tf, v2)
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
    _walk_geom_symbol_only(v2.doc, geom, None, out["lines"], out["arcs"], min_len_cm, v2)
    out["lines"] = _dedupe_lines(out["lines"], min_len_cm)
    out["arcs"] = _dedupe_arcs(out["arcs"])
    out["meta"]["line_count"] = len(out["lines"])
    out["meta"]["arc_count"] = len(out["arcs"])
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


def _arc_bbox_cm(arc):
    cx = float(arc.get("cx", 0.0))
    cy = float(arc.get("cy", 0.0))
    r = abs(float(arc.get("r", 0.0)))
    return (cx - r, cy - r, cx + r, cy + r)


def _pick_best_door_arc(arcs):
    best = None
    best_radius = -1.0
    for arc in (arcs or []):
        try:
            r = abs(float(arc.get("r", 0.0)))
        except Exception:
            r = 0.0
        if r <= 1.0e-6:
            continue
        if best is None or r > best_radius:
            best = arc
            best_radius = r
    return best


def _detect_opening_markers(raw_data, kind):
    """Cluster lines from A-DOORS or A-WINDOWS into opening markers.
    Returns list of {center_cm, width_cm, kind}."""
    raw_lines = []
    raw_arcs = []
    if isinstance(raw_data, dict):
        raw_lines = list(raw_data.get("lines") or [])
        raw_arcs = list(raw_data.get("arcs") or [])

    filtered = []
    for ln in (raw_lines or []):
        if _is_opening_layer(ln.get("layer", ""), kind):
            x1 = float(ln.get("x1", 0))
            y1 = float(ln.get("y1", 0))
            x2 = float(ln.get("x2", 0))
            y2 = float(ln.get("y2", 0))
            bb = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            seg_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            filtered.append({"entity": "line", "ln": ln, "bbox": bb, "len": seg_len})

    for arc in (raw_arcs or []):
        if _is_opening_layer(arc.get("layer", ""), kind):
            bb = _arc_bbox_cm(arc)
            radius = abs(float(arc.get("r", 0.0)))
            filtered.append({
                "entity": "arc",
                "arc": arc,
                "bbox": bb,
                "len": max(20.0, radius * 2.0),
                "radius": radius,
            })

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

        # Separate frame/jamb lines (short, <= 50cm) from door swing geometry.
        frame_xs = []
        frame_ys = []
        swing_line = None
        max_seg_len = 0.0
        group_arcs = []
        for ent in group:
            if ent["entity"] == "arc":
                group_arcs.append(ent["arc"])
                continue
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
        if frame_xs and frame_ys:
            center = ((min(frame_xs) + max(frame_xs)) * 0.5,
                      (min(frame_ys) + max(frame_ys)) * 0.5)
        else:
            center = _bbox_center_cm(bb)

        is_double_door = False
        if kind == "door":
            # Check for double door: 2 arcs with similar radius, centers apart.
            valid_arcs = [a for a in group_arcs
                          if abs(float(a.get("r", 0.0))) > 1.0]
            if len(valid_arcs) >= 2:
                # Sort by radius descending, take top 2
                valid_arcs.sort(key=lambda a: abs(float(a.get("r", 0.0))), reverse=True)
                a1, a2 = valid_arcs[0], valid_arcs[1]
                r1 = abs(float(a1.get("r", 0.0)))
                r2 = abs(float(a2.get("r", 0.0)))
                c1 = (float(a1.get("cx", 0.0)), float(a1.get("cy", 0.0)))
                c2 = (float(a2.get("cx", 0.0)), float(a2.get("cy", 0.0)))
                hinge_dist = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
                # Double door: similar radius (within 20%) and hinges ~width apart
                if r2 > r1 * 0.7 and hinge_dist > r1 * 0.5:
                    is_double_door = True

            best_arc = _pick_best_door_arc(group_arcs)
            if is_double_door and len(valid_arcs) >= 2:
                # Double door: total width = distance between outer jamb edges.
                # Each leaf width = arc radius. Total = hinge distance + no overlap,
                # but better: use jamb-to-jamb if available, else 2 * leaf radius.
                leaf1 = abs(float(valid_arcs[0].get("r", 0.0)))
                leaf2 = abs(float(valid_arcs[1].get("r", 0.0)))
                if frame_xs and frame_ys:
                    jw = max(max(frame_xs) - min(frame_xs),
                             max(frame_ys) - min(frame_ys))
                    width = max(80.0, min(400.0, jw - 10.0))
                    width_source = "double_jamb"
                else:
                    width = max(80.0, min(400.0, hinge_dist))
                    width_source = "double_hinge_dist"
            elif best_arc is not None:
                # Prefer the real DWG swing arc radius; that is the leaf width.
                width = max(40.0, min(200.0, abs(float(best_arc.get("r", 0.0)))))
                width_source = "arc_radius"
            elif max_seg_len > 50.0:
                width = max(40.0, min(200.0, max_seg_len))
                width_source = "longest_line"
            elif frame_xs and frame_ys:
                # Fallback: jamb-to-jamb minus estimated jamb thickness (~5cm each side)
                jw = max(max(frame_xs) - min(frame_xs),
                         max(frame_ys) - min(frame_ys))
                width = max(40.0, min(200.0, jw - 10.0))
                width_source = "jamb_bbox"
            else:
                width = max(40.0, min(200.0, max_seg_len))
                width_source = "fallback_line"
        else:
            # Window: the longer bbox dim is the opening width along the wall.
            # The shorter dim is wall thickness direction (~11-30cm).
            width = max(30.0, min(300.0, max(bb_w, bb_h)))
            width_source = "bbox"

        # Swing arc data for door orientation matching.
        swing_data = None
        if kind == "door" and best_arc is not None:
            swing_data = {
                "source": "arc",
                "hinge_cm": (float(best_arc.get("cx", 0.0)), float(best_arc.get("cy", 0.0))),
                "arc_endpoints_cm": [
                    (float(best_arc.get("sx", 0.0)), float(best_arc.get("sy", 0.0))),
                    (float(best_arc.get("ex", 0.0)), float(best_arc.get("ey", 0.0))),
                ],
                "radius_cm": abs(float(best_arc.get("r", 0.0))),
            }
        elif kind == "door" and swing_line is not None and max_seg_len > 50.0:
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
                "source": "line",
                "hinge_cm": hinge_pt,
                "open_cm": open_pt,
                "swing_mid_cm": swing_mid,
            }

        markers.append({
            "kind": kind,
            "center_cm": center,
            "width_cm": width,
            "width_source": width_source,
            "swing": swing_data,
            "is_double": is_double_door,
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


def _lookup_first_param(elem, names):
    for pname in (names or []):
        try:
            p = elem.LookupParameter(pname)
        except Exception:
            p = None
        if p is not None:
            return p
    return None


def _safe_str(s):
    """Convert a string to ASCII-safe bytes for logging (IronPython 2.7).

    In IronPython 2.7, str is bytes and unicode is text.  .format() with
    any unicode argument produces unicode output which then fails when
    snapshot.log() writes it.  So we must return pure ASCII *bytes*.
    """
    if s is None:
        return b"None"
    # .NET strings and Python unicode -> encode to ASCII bytes, replacing non-ASCII
    try:
        # Works for unicode / System.String
        return s.encode("ascii", "replace")
    except (AttributeError, UnicodeDecodeError, UnicodeEncodeError):
        pass
    # Already bytes / str
    try:
        return str(s)
    except Exception:
        return b"<?>"


def _family_symbol_name(fs):
    try:
        name = fs.Name
        if name:
            return name
    except Exception:
        pass
    try:
        p = fs.get_Parameter(DB.BuiltInParameter.SYMBOL_NAME_PARAM)
        if p is not None:
            name = p.AsString()
            if name:
                return name
    except Exception:
        pass
    return "Type"


def _family_symbol_width_ft(fs):
    p = _lookup_first_param(fs, ["Width", "width", "Rough Width"])
    if p is None:
        return None
    try:
        return p.AsDouble()
    except Exception:
        return None


def _ensure_family_type_width(v2, family_types, base_fs, target_width_cm, snapshot=None):
    if base_fs is None:
        return None

    target_ft = v2.cm_to_ft(float(target_width_cm))
    tol_ft = v2.cm_to_ft(2.0)  # 2cm tolerance
    base_width = _family_symbol_width_ft(base_fs)
    if base_width is None:
        return base_fs
    if abs(base_width - target_ft) <= tol_ft:
        return base_fs

    # --- First: scan ALL existing types of the same family for an exact match ---
    # This avoids unnecessary duplication and type corruption.
    try:
        fam = base_fs.Family
        fam_type_ids = fam.GetFamilySymbolIds()
        for tid in fam_type_ids:
            try:
                existing_fs = v2.doc.GetElement(tid)
                if existing_fs is None:
                    continue
                w = _family_symbol_width_ft(existing_fs)
                if w is not None and abs(w - target_ft) <= tol_ft:
                    if snapshot:
                        try:
                            snapshot.log("  found existing type {} width={:.0f}cm for target={:.0f}cm".format(
                                _safe_str(_family_symbol_name(existing_fs)),
                                v2.ft_to_cm(w), float(target_width_cm)))
                        except Exception:
                            pass
                    return existing_fs
            except Exception:
                continue
    except Exception:
        pass

    # --- No existing match: duplicate from base and set width ---
    # NEVER modify the base type in-place — that corrupts it for other doors.
    base_param = _lookup_first_param(base_fs, ["Width", "width", "Rough Width"])
    if base_param is None or base_param.IsReadOnly:
        return base_fs

    desired_name = "{} {:.0f}cm".format(_family_symbol_name(base_fs), float(target_width_cm))
    dup = None
    for idx in range(0, 10):
        try_name = desired_name if idx == 0 else "{} {}".format(desired_name, idx + 1)
        try:
            dup = base_fs.Duplicate(try_name)
            break
        except Exception:
            continue
    if dup is None:
        return base_fs

    dup_param = _lookup_first_param(dup, ["Width", "width", "Rough Width"])
    if dup_param is None or dup_param.IsReadOnly:
        return base_fs

    try:
        dup_param.Set(target_ft)
        v2.doc.Regenerate()
        verify_w = dup_param.AsDouble()
        verify_cm = verify_w * 30.48
        if snapshot:
            try:
                ok = abs(verify_cm - float(target_width_cm)) <= 2.0
                snapshot.log("  resized type {} -> {:.0f}cm {}".format(
                    _safe_str(_family_symbol_name(dup)), float(target_width_cm),
                    "verified" if ok else "WARNING: actual={:.0f}cm".format(verify_cm)))
            except Exception:
                pass
        return dup
    except Exception:
        return base_fs


def _pick_family_type_by_width(v2, family_types, target_width_cm):
    """Pick the family type whose width parameter is closest to target."""
    # Exclude mamad (safe room) and curtain wall families
    family_types = [fs for fs in family_types if not _is_mamad_family(fs)]
    family_types = [fs for fs in family_types if not _is_curtain_wall_family(fs)]
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


def _is_mamad_family(fs):
    """Check if a family symbol belongs to a mamad (safe room) family."""
    try:
        name = (fs.Family.Name or "")
    except Exception:
        name = ""
    try:
        tname = (fs.Name or "")
    except Exception:
        tname = ""
    combined = name + " " + tname
    return u"\u05de\u05de\u05d3" in combined or u'\u05de\u05de"\u05d3' in combined


def _is_double_door_family(fs):
    """Check if a family symbol is a double door family."""
    try:
        name = (fs.Family.Name or "").upper()
    except Exception:
        name = ""
    return "DOUBLE" in name or "DBL" in name or u"\u05db\u05e4\u05d5\u05dc" in name


def _is_curtain_wall_family(fs):
    """Check if a family is a curtain-wall-hosted family (not suitable for basic walls)."""
    try:
        name = (fs.Family.Name or "").upper()
    except Exception:
        name = ""
    # "CW" surrounded by spaces/underscores/start/end, or full phrase
    if " CW " in name or " CW_" in name or "_CW " in name or "_CW_" in name:
        return True
    if name.startswith("CW ") or name.startswith("CW_"):
        return True
    if name.endswith(" CW") or name.endswith("_CW"):
        return True
    if "CURTAIN WALL" in name or "CURTAIN_WALL" in name or "CURTAINWALL" in name:
        return True
    return False


def _pick_door_type_for_wall(v2, family_types, target_width_cm, is_interior, snapshot=None,
                              is_double=False):
    """Pick a door family type appropriate for interior or exterior walls."""
    target_ft = v2.cm_to_ft(float(target_width_cm))

    # Exclude mamad (safe room) families — reserved for mamad openings only.
    family_types = [fs for fs in family_types if not _is_mamad_family(fs)]

    # Exclude curtain wall families — they can't be hosted on basic walls.
    family_types = [fs for fs in family_types if not _is_curtain_wall_family(fs)]

    # Filter by single vs double door
    if is_double:
        # Double door: ONLY use double-door families. If none available, return None
        # to skip placement (placing a 200cm single door would break the wall).
        double_types = [fs for fs in family_types if _is_double_door_family(fs)]
        if not double_types:
            if snapshot:
                try:
                    snapshot.log("  WARNING: no double-door family loaded, skipping double door (width={:.0f}cm)".format(
                        float(target_width_cm)))
                except Exception:
                    pass
            return None
        family_types = double_types
    else:
        # For single doors, exclude double door families
        family_types = [fs for fs in family_types if not _is_double_door_family(fs)]

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
            int_names = [_safe_str(fs.Family.Name) for fs in int_types]
            ext_names = [_safe_str(fs.Family.Name) for fs in ext_types]
            snapshot.log("Door types: {} interior ({}), {} exterior ({})".format(
                len(int_types), ", ".join(set(int_names))[:200],
                len(ext_types), ", ".join(set(ext_names))[:200]))
        except Exception as ex:
            snapshot.log("Door types log error: {}".format(ex))

    candidates = int_types if is_interior else ext_types
    if not candidates:
        candidates = family_types  # fallback to all

    # Pick by closest width from ALL candidates (custom + default).
    # Don't prefer custom families since they may have fixed geometry
    # that doesn't resize parametrically.
    best = None
    best_delta = 1.0e9
    for fs in candidates:
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

    if snapshot:
        try:
            if best is not None:
                bw = _family_symbol_width_ft(best)
                bw_cm = v2.ft_to_cm(bw) if bw is not None else -1
                snapshot.log("  picked {} door: {} / {} width={:.0f}cm target={:.0f}cm delta={:.0f}cm double={}".format(
                    "INT" if is_interior else "EXT",
                    _safe_str(best.Family.Name), _safe_str(_family_symbol_name(best)), bw_cm, float(target_width_cm),
                    v2.ft_to_cm(best_delta), is_double))
            else:
                snapshot.log("  WARNING: no door picked for target={:.0f}cm int={} double={}".format(
                    float(target_width_cm), is_interior, is_double))
        except Exception as ex:
            snapshot.log("  pick_log_err: {}".format(ex))
    return best


def _match_door_swing(v2, inst, wall, swing, center_cm, snapshot=None):
    """Flip door hand/facing to match DWG swing arc direction.

    Uses world-coordinate comparison between DWG arc geometry and
    Revit FacingOrientation / HandOrientation — no wall tangent needed.

    Arc geometry:
      - hinge = arc center (pivot point)
      - closed_pt = arc endpoint along the wall (door leaf closed position)
      - open_pt = arc endpoint perpendicular to wall (door swung open)

    Revit orientations:
      - FacingOrientation = direction the door opens into (perpendicular to wall)
      - HandOrientation = direction from hinge toward latch (along wall)
    """
    hinge = swing.get("hinge_cm")
    if hinge is None:
        return

    # Get arc endpoints
    endpoints = list(swing.get("arc_endpoints_cm") or [])
    open_cm = swing.get("open_cm")

    if open_cm is not None:
        # open_cm provided directly (line-source swing)
        open_pt = open_cm
        closed_pt = None
    elif len(endpoints) >= 2:
        # Determine which endpoint is open (perpendicular) vs closed (along wall).
        # The open endpoint is FURTHER from the wall curve.
        try:
            curve = wall.Location.Curve
        except Exception:
            return

        best_open = None
        best_closed = None
        best_open_dist = -1.0
        best_closed_dist = 1.0e9
        for pt in endpoints:
            test_pt = v2.XYZ(v2.cm_to_ft(float(pt[0])), v2.cm_to_ft(float(pt[1])), 0.0)
            try:
                result = curve.Project(test_pt)
                d = result.Distance if result is not None else 1.0e9
            except Exception:
                d = 1.0e9
            if d > best_open_dist:
                best_open_dist = d
                best_open = pt
            if d < best_closed_dist:
                best_closed_dist = d
                best_closed = pt
        open_pt = best_open
        closed_pt = best_closed
    else:
        return

    if open_pt is None:
        return

    # --- DWG open direction (facing) ---
    # Vector from hinge to open endpoint: the direction the door swings into.
    dwg_open_dx = v2.cm_to_ft(float(open_pt[0]) - float(hinge[0]))
    dwg_open_dy = v2.cm_to_ft(float(open_pt[1]) - float(hinge[1]))
    dwg_open_len = math.sqrt(dwg_open_dx ** 2 + dwg_open_dy ** 2)
    if dwg_open_len < 1e-9:
        return

    # --- DWG hand direction ---
    # Vector from door center toward hinge: Revit HandOrientation points
    # toward the hinge side of the opening, so we compare with center->hinge.
    dwg_hand_dx = v2.cm_to_ft(float(hinge[0]) - float(center_cm[0]))
    dwg_hand_dy = v2.cm_to_ft(float(hinge[1]) - float(center_cm[1]))
    has_hand = (dwg_hand_dx ** 2 + dwg_hand_dy ** 2) > 1e-9

    # --- Revit current orientation ---
    try:
        facing_dir = inst.FacingOrientation
        hand_dir = inst.HandOrientation
    except Exception:
        return

    # Compare facing: dot product of DWG open direction with Revit FacingOrientation.
    facing_dot = dwg_open_dx * facing_dir.X + dwg_open_dy * facing_dir.Y
    need_flip_facing = (facing_dot < 0)

    if need_flip_facing:
        inst.flipFacing()
        try:
            v2.doc.Regenerate()
        except Exception:
            pass

    # Re-read hand after any facing flip (facing flip can affect hand).
    try:
        hand_dir = inst.HandOrientation
    except Exception:
        return

    # Compare hand: dot product of DWG hand direction with Revit HandOrientation.
    need_flip_hand = False
    if has_hand:
        hand_dot = dwg_hand_dx * hand_dir.X + dwg_hand_dy * hand_dir.Y
        need_flip_hand = (hand_dot < 0)
    if need_flip_hand:
        inst.flipHand()

    if snapshot:
        try:
            snapshot.log("  swing({}): open=({:.0f},{:.0f}) closed={} "
                         "facing_dot={:.2f} flip_facing={}, "
                         "hand_dot={:.2f} flip_hand={}".format(
                swing.get("source", "unknown"),
                float(open_pt[0]), float(open_pt[1]),
                "({:.0f},{:.0f})".format(float(closed_pt[0]), float(closed_pt[1])) if closed_pt else "?",
                facing_dot, need_flip_facing,
                (dwg_hand_dx * hand_dir.X + dwg_hand_dy * hand_dir.Y) if has_hand else 0.0,
                need_flip_hand))
        except Exception:
            pass


def _bridge_gaps_at_doors(centerlines, door_markers, snapshot=None):
    """Merge collinear centerline segments that have a gap where a door is located.

    When the DWG has a door opening, the wall lines stop at the jambs,
    creating a gap in the centerlines. This function bridges those gaps
    so the Revit wall is continuous and can host the door.
    """
    if not door_markers or not centerlines:
        return centerlines

    door_centers = []
    for mk in door_markers:
        cx, cy = mk["center_cm"]
        half_w = float(mk["width_cm"]) * 0.5 + 30.0  # extra margin for jambs
        door_centers.append((float(cx), float(cy), half_w))

    merged = True
    result = list(centerlines)
    while merged:
        merged = False
        new_result = []
        used = set()
        for i in range(len(result)):
            if i in used:
                continue
            li = result[i]
            x1i, y1i = float(li.get("x1", 0)), float(li.get("y1", 0))
            x2i, y2i = float(li.get("x2", 0)), float(li.get("y2", 0))
            dxi = x2i - x1i
            dyi = y2i - y1i
            leni = math.sqrt(dxi * dxi + dyi * dyi)
            if leni < 1e-6:
                new_result.append(li)
                continue

            best_j = None
            best_gap = 1e9
            for j in range(i + 1, len(result)):
                if j in used:
                    continue
                lj = result[j]
                x1j, y1j = float(lj.get("x1", 0)), float(lj.get("y1", 0))
                x2j, y2j = float(lj.get("x2", 0)), float(lj.get("y2", 0))
                dxj = x2j - x1j
                dyj = y2j - y1j
                lenj = math.sqrt(dxj * dxj + dyj * dyj)
                if lenj < 1e-6:
                    continue

                # Check collinearity: same direction (dot product close to ±1)
                dot = (dxi * dxj + dyi * dyj) / (leni * lenj)
                if abs(abs(dot) - 1.0) > 0.05:
                    continue

                # Check perpendicular distance between lines (must be close)
                mid_jx = (x1j + x2j) * 0.5
                mid_jy = (y1j + y2j) * 0.5
                # Project midpoint of j onto line i
                t_proj = ((mid_jx - x1i) * dxi + (mid_jy - y1i) * dyi) / (leni * leni)
                proj_x = x1i + t_proj * dxi
                proj_y = y1i + t_proj * dyi
                perp_dist = math.sqrt((mid_jx - proj_x) ** 2 + (mid_jy - proj_y) ** 2)
                if perp_dist > 15.0:  # max 15cm perpendicular offset
                    continue

                # Find the gap between the two segments.
                # Check all 4 endpoint-to-endpoint distances, find the smallest.
                pts_i = [(x1i, y1i), (x2i, y2i)]
                pts_j = [(x1j, y1j), (x2j, y2j)]
                min_gap = 1e9
                gap_mid = None
                for pi in pts_i:
                    for pj in pts_j:
                        g = math.sqrt((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2)
                        if g < min_gap:
                            min_gap = g
                            gap_mid = ((pi[0] + pj[0]) * 0.5, (pi[1] + pj[1]) * 0.5)

                if min_gap < 10.0:
                    continue  # already touching, not a door gap
                if min_gap > 350.0:
                    continue  # too far apart

                # Check if a door marker is in this gap
                door_in_gap = False
                for dcx, dcy, dhw in door_centers:
                    d_to_gap = math.sqrt((dcx - gap_mid[0]) ** 2 + (dcy - gap_mid[1]) ** 2)
                    if d_to_gap < dhw:
                        door_in_gap = True
                        break

                if door_in_gap and min_gap < best_gap:
                    best_j = j
                    best_gap = min_gap

            if best_j is not None:
                # Merge lines i and best_j into one spanning both + gap.
                lj = result[best_j]
                all_pts = [
                    (x1i, y1i), (x2i, y2i),
                    (float(lj.get("x1", 0)), float(lj.get("y1", 0))),
                    (float(lj.get("x2", 0)), float(lj.get("y2", 0))),
                ]
                # Project all points onto line direction and take extremes
                nx = dxi / leni
                ny = dyi / leni
                projs = [(p[0] * nx + p[1] * ny, p) for p in all_pts]
                projs.sort(key=lambda x: x[0])
                p_min = projs[0][1]
                p_max = projs[-1][1]
                merged_line = dict(li)
                merged_line["x1"] = p_min[0]
                merged_line["y1"] = p_min[1]
                merged_line["x2"] = p_max[0]
                merged_line["y2"] = p_max[1]
                new_result.append(merged_line)
                used.add(i)
                used.add(best_j)
                merged = True
                if snapshot:
                    try:
                        snapshot.log("  bridged gap {:.0f}cm at ({:.0f},{:.0f}) -> wall ({:.0f},{:.0f})->({:.0f},{:.0f})".format(
                            best_gap, gap_mid[0], gap_mid[1],
                            p_min[0], p_min[1], p_max[0], p_max[1]))
                    except Exception:
                        pass
            else:
                new_result.append(li)
        # Add any remaining unused lines
        for k in range(len(result)):
            if k not in used and result[k] not in new_result:
                new_result.append(result[k])
        result = new_result

    return result


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

        # Verify the opening fits within the wall segment.
        # The projected point must not be at the wall edge with the opening
        # extending beyond the wall ends.
        skip = False
        try:
            curve = wall.Location.Curve
            ws = curve.GetEndPoint(0)
            we = curve.GetEndPoint(1)
            wall_len_ft = math.sqrt((we.X - ws.X) ** 2 + (we.Y - ws.Y) ** 2)
            half_width_ft = v2.cm_to_ft(float(mk["width_cm"]) * 0.5)
            # Distance from projected point to each wall endpoint
            d_start = math.sqrt((pt.X - ws.X) ** 2 + (pt.Y - ws.Y) ** 2)
            d_end = math.sqrt((pt.X - we.X) ** 2 + (pt.Y - we.Y) ** 2)
            if d_start < half_width_ft or d_end < half_width_ft:
                errors.append("{} at ({:.0f},{:.0f}) width={:.0f}cm does not fit in wall "
                              "(len={:.0f}cm, {:.0f}cm/{:.0f}cm from ends), skipping".format(
                    cat_name, mk["center_cm"][0], mk["center_cm"][1], mk["width_cm"],
                    v2.ft_to_cm(wall_len_ft), v2.ft_to_cm(d_start), v2.ft_to_cm(d_end)))
                skip = True
        except Exception:
            pass
        if skip:
            continue

        is_interior = wall.Id.IntegerValue in int_wall_set
        if snapshot:
            try:
                snapshot.log("  {} at ({:.0f},{:.0f}) -> wall {} dist={:.1f}cm int={} double={}".format(
                    cat_name, mk["center_cm"][0], mk["center_cm"][1],
                    wall.Id.IntegerValue, v2.ft_to_cm(dist), is_interior, mk.get("is_double", False)))
            except Exception:
                pass

        is_double = mk.get("is_double", False)
        if category == BuiltInCategory.OST_Doors:
            fs = _pick_door_type_for_wall(v2, family_types, mk["width_cm"], is_interior, snapshot,
                                           is_double=is_double)
            fs = _ensure_family_type_width(v2, family_types, fs, mk["width_cm"], snapshot)
        else:
            fs = _pick_family_type_by_width(v2, family_types, mk["width_cm"])

        if fs is None:
            errors.append("No {} family type available".format(cat_name))
            continue

        # Activate the type; if it fails, try other types from the same family
        activated = fs.IsActive
        if not activated:
            try:
                fs.Activate()
                v2.doc.Regenerate()
                activated = True
            except Exception:
                pass
        if not activated:
            # Type can't activate (corrupted?) — try other types from same
            # family first, then any other family type as last resort.
            fallback = None
            failed_id = fs.Id
            # Pass 1: same family
            try:
                for alt_id in fs.Family.GetFamilySymbolIds():
                    alt = v2.doc.GetElement(alt_id)
                    if alt is None or alt.Id == failed_id:
                        continue
                    if alt.IsActive:
                        fallback = alt
                        break
                    try:
                        alt.Activate()
                        v2.doc.Regenerate()
                        fallback = alt
                        break
                    except Exception:
                        continue
            except Exception:
                pass
            # Pass 2: any other loaded family type (sorted by width proximity)
            if fallback is None:
                target_ft = v2.cm_to_ft(float(mk["width_cm"]))
                alt_candidates = []
                for alt_fs in family_types:
                    try:
                        if alt_fs.Id == failed_id:
                            continue
                        w = _family_symbol_width_ft(alt_fs)
                        delta = abs(w - target_ft) if w is not None else 1e9
                        alt_candidates.append((delta, alt_fs))
                    except Exception:
                        continue
                alt_candidates.sort(key=lambda x: x[0])
                for _, alt_fs in alt_candidates:
                    if alt_fs.IsActive:
                        fallback = alt_fs
                        break
                    try:
                        alt_fs.Activate()
                        v2.doc.Regenerate()
                        fallback = alt_fs
                        break
                    except Exception:
                        continue
            if fallback is not None:
                if snapshot:
                    try:
                        snapshot.log("  fallback type: {} / {} (original failed activation)".format(
                            _safe_str(fallback.Family.Name),
                            _safe_str(_family_symbol_name(fallback))))
                    except Exception:
                        pass
                fs = fallback
            else:
                if snapshot:
                    try:
                        snapshot.log("  WARNING: no activatable type for {}, skipping".format(
                            _safe_str(_family_symbol_name(fs))))
                    except Exception:
                        pass
                errors.append("Can't activate {} type".format(cat_name))
                continue

        try:
            # XYZ is absolute coordinates; Z = level elevation for floor-level placement.
            place_pt = v2.XYZ(pt.X, pt.Y, level_elev_ft)
            inst = v2.doc.Create.NewFamilyInstance(
                place_pt, fs, wall, level, StructuralType.NonStructural)

            # Regenerate so Revit processes the wall cut and hosting
            try:
                v2.doc.Regenerate()
            except Exception:
                pass

            # Set sill height for windows
            if sill_height_cm is not None and category == BuiltInCategory.OST_Windows:
                _set_sill_height(v2, inst, sill_height_cm)

            # Match door swing direction from DWG (single doors only)
            if category == BuiltInCategory.OST_Doors and mk.get("swing") and not is_double:
                try:
                    _match_door_swing(v2, inst, wall, mk["swing"], mk["center_cm"], snapshot)
                except Exception as ex:
                    if snapshot:
                        try:
                            snapshot.log("  swing match error: {}".format(ex))
                        except Exception:
                            pass

            ids.append(inst.Id.IntegerValue)
            if snapshot:
                try:
                    wall_loc = wall.Location.Curve
                    w_start = wall_loc.GetEndPoint(0)
                    w_end = wall_loc.GetEndPoint(1)
                except Exception as ex:
                    snapshot.log("{} placed id={} at ({:.0f},{:.0f})cm (wall loc error: {})".format(
                        cat_name, inst.Id.IntegerValue,
                        mk["center_cm"][0], mk["center_cm"][1], ex))
                    w_start = w_end = None
                try:
                    actual_width_ft = _family_symbol_width_ft(fs)
                    if actual_width_ft is None:
                        actual_width_txt = "n/a"
                    else:
                        actual_width_txt = "{:.0f}cm".format(v2.ft_to_cm(actual_width_ft))
                except Exception:
                    actual_width_txt = "err"
                # Check hosting
                host_info = "no_host"
                try:
                    h = inst.Host
                    if h is not None:
                        host_info = "hosted={}".format(h.Id.IntegerValue)
                    else:
                        host_info = "host=None"
                except Exception:
                    host_info = "host_check_err"
                # Log placement - separate try so we always get output
                try:
                    if w_start is not None:
                        snapshot.log(
                            "{} placed id={} at ({:.0f},{:.0f})cm on {} wall {} "
                            "({:.0f},{:.0f})->({:.0f},{:.0f})ft dist={:.1f}cm width={:.0f}cm({}) "
                            "type={} actual={} {} double={}".format(
                                cat_name, inst.Id.IntegerValue,
                                mk["center_cm"][0], mk["center_cm"][1],
                                "INT" if is_interior else "EXT",
                                wall.Id.IntegerValue,
                                w_start.X, w_start.Y, w_end.X, w_end.Y,
                                v2.ft_to_cm(dist),
                                mk["width_cm"],
                                mk.get("width_source", "?"),
                                _safe_str(fs.Family.Name),
                                actual_width_txt,
                                host_info,
                                is_double))
                except Exception as ex2:
                    snapshot.log("{} placed id={} LOG_ERR: {}".format(
                        cat_name, inst.Id.IntegerValue, ex2))
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


# ---------------------------------------------------------------------------
#  Floor creation helpers
# ---------------------------------------------------------------------------

def _collect_floor_types(doc):
    """Return all FloorType elements in the document."""
    from Autodesk.Revit.DB import FloorType as _FT
    return list(FilteredElementCollector(doc).OfCategory(
        BuiltInCategory.OST_Floors).OfClass(_FT))


def _get_element_name(el):
    """Get element name safely, handling IronPython unicode issues."""
    # Try .NET Element.Name property directly
    try:
        n = el.Name
        if n and len(n) > 0:
            return n
    except Exception:
        pass
    # Try via Parameter API
    try:
        p = el.get_Parameter(DB.BuiltInParameter.SYMBOL_NAME_PARAM)
        if p is not None:
            v = p.AsString()
            if v and len(v) > 0:
                return v
    except Exception:
        pass
    try:
        p = el.get_Parameter(DB.BuiltInParameter.ALL_MODEL_TYPE_NAME)
        if p is not None:
            v = p.AsString()
            if v and len(v) > 0:
                return v
    except Exception:
        pass
    # Last resort: .ToString()
    try:
        s = el.ToString()
        if s:
            return s
    except Exception:
        pass
    return "? (id={})".format(el.Id.IntegerValue)


def _pick_floor_types(v2):
    """Show a dialog asking the user to choose concrete and tile floor types.

    Returns (concrete_floor_type, tile_floor_type) or None if canceled.
    """
    import clr
    clr.AddReference("System.Windows.Forms")
    clr.AddReference("System.Drawing")
    from System.Windows.Forms import (
        Form, Label, Button, ListBox, FormBorderStyle, DialogResult,
        SelectionMode,
    )
    from System.Drawing import Size, Point

    floor_types = _collect_floor_types(v2.doc)
    if not floor_types:
        TaskDialog.Show("C2Rv7_C", "No floor types found in the project.")
        return None

    # Get names using .NET interop to preserve Hebrew
    names = []
    for ft in floor_types:
        names.append(_get_element_name(ft))

    # Pre-select indices by name keyword match
    concrete_idx = 0
    tile_idx = 0
    for i, n in enumerate(names):
        try:
            nu = n.upper()
        except Exception:
            nu = ""
        if "CONCRETE" in nu or "STRUCTURAL" in nu or "BETON" in nu:
            concrete_idx = i
            break
    for i, n in enumerate(names):
        try:
            nu = n.upper()
        except Exception:
            nu = ""
        if "TILE" in nu or "FINISH" in nu or "CERAMIC" in nu:
            tile_idx = i
            break

    form = Form()
    form.Text = "Choose Floor Types"
    form.Width = 500
    form.Height = 450
    form.FormBorderStyle = FormBorderStyle.FixedDialog
    form.MinimizeBox = False
    form.MaximizeBox = False
    try:
        form.StartPosition = 1  # CenterScreen
    except Exception:
        pass

    lbl1 = Label()
    lbl1.Text = "Concrete Structural Floor:"
    lbl1.Location = Point(12, 12)
    lbl1.Size = Size(460, 20)
    form.Controls.Add(lbl1)

    lb1 = ListBox()
    lb1.Location = Point(12, 34)
    lb1.Size = Size(460, 150)
    lb1.SelectionMode = SelectionMode.One
    for n in names:
        lb1.Items.Add(n)
    lb1.SelectedIndex = concrete_idx
    form.Controls.Add(lb1)

    lbl2 = Label()
    lbl2.Text = "Tile Finish Floor (5cm offset):"
    lbl2.Location = Point(12, 194)
    lbl2.Size = Size(460, 20)
    form.Controls.Add(lbl2)

    lb2 = ListBox()
    lb2.Location = Point(12, 216)
    lb2.Size = Size(460, 150)
    lb2.SelectionMode = SelectionMode.One
    for n in names:
        lb2.Items.Add(n)
    lb2.SelectedIndex = tile_idx
    form.Controls.Add(lb2)

    btn_ok = Button()
    btn_ok.Text = "OK"
    btn_ok.Location = Point(290, 378)
    btn_ok.Size = Size(80, 30)
    btn_ok.DialogResult = DialogResult.OK
    form.Controls.Add(btn_ok)
    form.AcceptButton = btn_ok

    btn_cancel = Button()
    btn_cancel.Text = "Cancel"
    btn_cancel.Location = Point(380, 378)
    btn_cancel.Size = Size(80, 30)
    btn_cancel.DialogResult = DialogResult.Cancel
    form.Controls.Add(btn_cancel)
    form.CancelButton = btn_cancel

    result = form.ShowDialog()
    if result != DialogResult.OK:
        return None

    ci = lb1.SelectedIndex
    ti = lb2.SelectedIndex
    if ci < 0 or ti < 0:
        return None

    return (floor_types[ci], floor_types[ti])


def _convex_hull_pts(points):
    """Compute convex hull of 2D points using Andrew's monotone chain.

    Returns list of (x, y) tuples in CCW order, or empty list if < 3 unique points.
    """
    pts = sorted(set(points))
    if len(pts) < 3:
        return []

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _chain_ext_wall_loop(curves_ft, tol_ft=0.5, snapshot=None):
    """Chain exterior wall segments into a closed polygon.

    Returns list of (x, y) tuples or empty list on failure.
    Falls back to convex hull if the chain walk fails.
    """
    if len(curves_ft) < 3:
        return []

    # --- Union-find endpoint merge ---
    all_pts = []
    parent = []

    def _uf_find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _uf_union(a, b):
        ra, rb = _uf_find(a), _uf_find(b)
        if ra != rb:
            parent[rb] = ra

    def _add_point(x, y):
        tol2 = tol_ft * tol_ft
        for j in range(len(all_pts)):
            dx = all_pts[j][0] - x
            dy = all_pts[j][1] - y
            if dx * dx + dy * dy <= tol2:
                idx = len(all_pts)
                all_pts.append((x, y))
                parent.append(idx)
                _uf_union(idx, j)
                return idx
        idx = len(all_pts)
        all_pts.append((x, y))
        parent.append(idx)
        return idx

    # Register all curve endpoints and store their root indices
    curve_roots = []  # (root0, root1) per curve
    for p0, p1 in curves_ft:
        i0 = _add_point(p0[0], p0[1])
        i1 = _add_point(p1[0], p1[1])
        curve_roots.append((_uf_find(i0), _uf_find(i1)))

    # Build averaged coordinates per root
    root_sum = {}
    root_cnt = {}
    for i in range(len(all_pts)):
        r = _uf_find(i)
        if r in root_sum:
            root_sum[r] = (root_sum[r][0] + all_pts[i][0],
                           root_sum[r][1] + all_pts[i][1])
            root_cnt[r] += 1
        else:
            root_sum[r] = (all_pts[i][0], all_pts[i][1])
            root_cnt[r] = 1
    pts_map = {}
    for r in root_sum:
        c = root_cnt[r]
        pts_map[r] = (root_sum[r][0] / c, root_sum[r][1] / c)

    # Build adjacency using stored roots (no second _add_point call)
    ep_map = {}
    seen_edges = set()
    for r0_raw, r1_raw in curve_roots:
        r0 = _uf_find(r0_raw)
        r1 = _uf_find(r1_raw)
        if r0 == r1:
            continue
        edge_key = (min(r0, r1), max(r0, r1))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        ep_map.setdefault(r0, []).append(r1)
        ep_map.setdefault(r1, []).append(r0)

    if not ep_map:
        if snapshot:
            try:
                snapshot.log("  chain: no adjacency, falling back to convex hull")
            except Exception:
                pass
        all_xy = [(p[0], p[1]) for p0, p1 in curves_ft for p in (p0, p1)]
        return _convex_hull_pts(all_xy)

    # --- Bridge degree-1 nodes (gap from door openings) ---
    max_bridge_ft = 50.0
    for _round in range(200):
        deg1 = [k for k in ep_map if len(ep_map[k]) == 1]
        if not deg1:
            break
        best_d2 = max_bridge_ft * max_bridge_ft + 1.0
        best_a, best_b = None, None
        for k in deg1:
            ax, ay = pts_map[k]
            neighbors = set(ep_map.get(k, []))
            for other in ep_map:
                if other == k or other in neighbors:
                    continue
                bx, by = pts_map[other]
                d2 = (ax - bx) ** 2 + (ay - by) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_a, best_b = k, other
        if best_a is None or best_d2 > max_bridge_ft * max_bridge_ft:
            break
        ep_map[best_a].append(best_b)
        ep_map[best_b].append(best_a)
        if snapshot:
            try:
                ax, ay = pts_map[best_a]
                bx, by = pts_map[best_b]
                snapshot.log("  bridge(deg1) {:.1f}ft: ({:.1f},{:.1f})->({:.1f},{:.1f})".format(
                    best_d2 ** 0.5, ax, ay, bx, by))
            except Exception:
                pass

    # --- Prune dead ends ---
    changed = True
    while changed:
        changed = False
        dead = [k for k in ep_map if len(ep_map[k]) <= 1]
        for k in dead:
            changed = True
            for other in ep_map.get(k, []):
                if other in ep_map:
                    ep_map[other] = [o for o in ep_map[other] if o != k]
            if k in ep_map:
                del ep_map[k]

    if not ep_map:
        if snapshot:
            try:
                snapshot.log("  chain: no nodes after prune, falling back to convex hull")
            except Exception:
                pass
        all_xy = [(p[0], p[1]) for p0, p1 in curves_ft for p in (p0, p1)]
        return _convex_hull_pts(all_xy)

    if snapshot:
        try:
            snapshot.log("  chain: {} nodes after bridge+prune".format(len(ep_map)))
            for k in ep_map:
                x, y = pts_map[k]
                snapshot.log("    node({:.1f},{:.1f}) deg={}".format(x, y, len(ep_map[k])))
        except Exception:
            pass

    # --- Walk the loop ---
    start = min(ep_map.keys(), key=lambda k: (pts_map[k][0], pts_map[k][1]))
    polygon = [pts_map[start]]
    visited_pairs = set()
    current = start
    prev_angle = math.atan2(-1.0, 0.0)

    for _step in range(len(curves_ft) * 4 + 10):
        cx, cy = pts_map[current]
        candidates = []
        for other in ep_map.get(current, []):
            pair = (min(current, other), max(current, other))
            if pair in visited_pairs:
                continue
            ox, oy = pts_map[other]
            angle = math.atan2(oy - cy, ox - cx)
            rel = prev_angle - angle
            while rel <= 0:
                rel += 2.0 * math.pi
            while rel > 2.0 * math.pi:
                rel -= 2.0 * math.pi
            candidates.append((rel, other))

        if not candidates:
            break

        candidates.sort()
        _, nxt = candidates[-1]

        pair = (min(current, nxt), max(current, nxt))
        visited_pairs.add(pair)

        nx, ny = pts_map[nxt]
        prev_angle = math.atan2(ny - cy, nx - cx) + math.pi
        while prev_angle > math.pi:
            prev_angle -= 2.0 * math.pi
        while prev_angle <= -math.pi:
            prev_angle += 2.0 * math.pi

        current = nxt
        if current == start:
            break
        polygon.append(pts_map[current])

    # Remove pinch points
    changed = True
    while changed:
        changed = False
        seen = {}
        for i, pt in enumerate(polygon):
            key = (round(pt[0], 4), round(pt[1], 4))
            if key in seen:
                j = seen[key]
                polygon = polygon[:j] + polygon[i:]
                changed = True
                break
            seen[key] = i

    if len(polygon) >= 3:
        if snapshot:
            try:
                snapshot.log("  floor polygon (chain): {} vertices".format(len(polygon)))
                for i, (x, y) in enumerate(polygon):
                    snapshot.log("  floor vtx[{}] ({:.2f},{:.2f})ft".format(i, x, y))
            except Exception:
                pass
        return polygon

    # --- Fallback: convex hull ---
    if snapshot:
        try:
            snapshot.log("  chain walk failed ({} vertices), falling back to convex hull".format(
                len(polygon)))
        except Exception:
            pass
    all_xy = [(p[0], p[1]) for p0, p1 in curves_ft for p in (p0, p1)]
    hull = _convex_hull_pts(all_xy)
    if snapshot and hull:
        try:
            snapshot.log("  floor polygon (hull fallback): {} vertices".format(len(hull)))
            for i, (x, y) in enumerate(hull):
                snapshot.log("  floor vtx[{}] ({:.2f},{:.2f})ft".format(i, x, y))
        except Exception:
            pass
    return hull


def _create_rooms_and_tags(v2, level, snapshot=None):
    """Place a Room and RoomTag in every enclosed space on the given level.

    Uses Revit's PlanTopology to find circuits (enclosed regions bounded
    by room-bounding walls), then creates a Room in each unoccupied circuit
    and a RoomTag at the room's location point.

    Returns (room_ids, tag_ids).
    """
    from Autodesk.Revit.DB import UV, ViewPlan, Transaction

    room_ids = []
    tag_ids = []

    # --- Create rooms from plan topology circuits ---
    t = Transaction(v2.doc, "C2Rv7_C Create Rooms")
    t.Start()
    try:
        plan_topo = v2.doc.get_PlanTopology(level)
        for circuit in plan_topo.Circuits:
            if circuit.IsRoomLocated:
                continue
            room = v2.doc.Create.NewRoom(None, circuit)
            if room is not None:
                room_ids.append(room.Id.IntegerValue)
                # Set room name to "Room" and number to sequential index
                try:
                    from Autodesk.Revit.DB import BuiltInParameter
                    p_name = room.get_Parameter(BuiltInParameter.ROOM_NAME)
                    if p_name and not p_name.IsReadOnly:
                        p_name.Set("Room")
                    p_num = room.get_Parameter(BuiltInParameter.ROOM_NUMBER)
                    if p_num and not p_num.IsReadOnly:
                        p_num.Set(str(len(room_ids)))
                except Exception:
                    pass
        t.Commit()
    except Exception as ex:
        if snapshot:
            try:
                snapshot.log("Room creation error: {}".format(ex))
            except Exception:
                pass
        try:
            t.RollBack()
        except Exception:
            pass
        return room_ids, tag_ids

    if not room_ids:
        if snapshot:
            try:
                snapshot.log("No enclosed circuits found for rooms")
            except Exception:
                pass
        return room_ids, tag_ids

    # --- Find a floor plan view for this level (needed for room tags) ---
    plan_view = None
    try:
        for vp in FilteredElementCollector(v2.doc).OfClass(ViewPlan).ToElements():
            if vp.GenLevel is not None and vp.GenLevel.Id == level.Id:
                if not vp.IsTemplate:
                    plan_view = vp
                    break
    except Exception:
        pass

    if plan_view is None:
        if snapshot:
            try:
                snapshot.log("No floor plan view found for level, skipping room tags")
            except Exception:
                pass
        return room_ids, tag_ids

    # --- Place room tags ---
    t2 = Transaction(v2.doc, "C2Rv7_C Create Room Tags")
    t2.Start()
    try:
        from Autodesk.Revit.DB import LinkElementId
        for rid in room_ids:
            try:
                room = v2.doc.GetElement(ElementId(rid))
                if room is None:
                    continue
                loc = room.Location
                if loc is None:
                    continue
                pt = loc.Point
                uv = UV(pt.X, pt.Y)
                tag = v2.doc.Create.NewRoomTag(
                    LinkElementId(room.Id), uv, plan_view.Id)
                if tag is not None:
                    tag_ids.append(tag.Id.IntegerValue)
            except Exception:
                continue

        # --- Configure room tag type: show name ON, rounding 0 places ---
        tag_types_done = set()
        for tid in tag_ids:
            try:
                tag_elem = v2.doc.GetElement(ElementId(tid))
                if tag_elem is None:
                    continue
                tag_type_id = tag_elem.GetTypeId()
                if tag_type_id in tag_types_done:
                    continue
                tag_types_done.add(tag_type_id)
                tag_type = v2.doc.GetElement(tag_type_id)
                if tag_type is None:
                    continue
                # Iterate type parameters — look for name toggle and rounding
                for p in tag_type.Parameters:
                    try:
                        pname = p.Definition.Name.lower() if p.Definition else ""
                        if p.IsReadOnly:
                            continue
                        # Enable "Show Room Name" / "Name" toggle
                        if ("name" in pname and
                                ("show" in pname or "visible" in pname
                                 or pname in ("name",))):
                            if p.StorageType.ToString() == "Integer":
                                p.Set(1)
                            elif p.StorageType.ToString() == "String":
                                p.Set("Yes")
                        # Set rounding / decimal places to 0
                        if ("round" in pname or "decimal" in pname
                                or "precision" in pname or "places" in pname):
                            if p.StorageType.ToString() == "Integer":
                                p.Set(0)
                            elif p.StorageType.ToString() == "Double":
                                p.Set(0.0)
                    except Exception:
                        continue
            except Exception:
                continue

        t2.Commit()
    except Exception as ex:
        if snapshot:
            try:
                snapshot.log("Room tag creation error: {}".format(ex))
            except Exception:
                pass
        try:
            t2.RollBack()
        except Exception:
            pass

    if snapshot:
        try:
            snapshot.log("Rooms created: {} rooms, {} tags".format(
                len(room_ids), len(tag_ids)))
        except Exception:
            pass

    return room_ids, tag_ids


def _create_exterior_dimensions(v2, level, wall_ids, door_ids, window_ids,
                                ext_thick_cm, snapshot=None):
    """Create 3 rows of dimension chains parallel to each exterior wall side.

    Row 1 (closest):  Opening dimensions — measures each door/window.
                       Only created if the side has openings.
                       Chain runs from side start to side end.
    Row 2:            Segment dimensions — measures each wall segment.
    Row 3 (farthest): Overall dimension of the entire side.

    Rows are spaced at equal intervals outward from the exterior face.
    """
    from Autodesk.Revit.DB import (
        Options, XYZ, Line, ReferenceArray, ViewPlan, Transaction,
    )
    try:
        from Autodesk.Revit.DB import FamilyInstanceReferenceType
    except Exception:
        FamilyInstanceReferenceType = None

    dim_ids = []

    # --- Get floor plan view ---
    plan_view = None
    for vp in FilteredElementCollector(v2.doc).OfClass(ViewPlan).ToElements():
        if vp.GenLevel is not None and vp.GenLevel.Id == level.Id:
            if not vp.IsTemplate:
                plan_view = vp
                break
    if plan_view is None:
        if snapshot:
            try:
                snapshot.log("Dimensions: no plan view found")
            except Exception:
                pass
        return dim_ids

    # --- Helper: get face references by normal direction ---
    def _wall_faces_by_normal(wall, target_dir):
        """Return [(pos_along_target_dir, Reference), ...] for faces whose
        normal is roughly parallel to target_dir.

        For a wall PARALLEL to the measurement direction: returns end faces.
        For a wall PERPENDICULAR to it: returns side faces (exterior face).
        """
        opts = Options()
        opts.ComputeReferences = True
        opts.View = plan_view
        refs = []
        try:
            geom = wall.get_Geometry(opts)
        except Exception:
            return refs
        if geom is None:
            return refs
        for gobj in geom:
            solids = []
            try:
                if gobj.Faces.Size > 0:
                    solids.append(gobj)
            except Exception:
                pass
            try:
                for sub in gobj.GetInstanceGeometry():
                    try:
                        if sub.Faces.Size > 0:
                            solids.append(sub)
                    except Exception:
                        pass
            except Exception:
                pass
            for solid in solids:
                try:
                    for face in solid.Faces:
                        if face.Reference is None:
                            continue
                        try:
                            fn = face.FaceNormal
                            dot = abs(fn.X * target_dir[0] + fn.Y * target_dir[1])
                            if dot < 0.8:
                                continue
                            bb = face.GetBoundingBox()
                            from Autodesk.Revit.DB import UV as _UV
                            mid_uv = _UV(
                                (bb.Min.U + bb.Max.U) / 2.0,
                                (bb.Min.V + bb.Max.V) / 2.0)
                            pt = face.Evaluate(mid_uv)
                            pos = target_dir[0] * pt.X + target_dir[1] * pt.Y
                            refs.append((pos, face.Reference))
                        except Exception:
                            continue
                except Exception:
                    continue
        refs.sort()
        return refs

    # --- Helper: get opening left/right references ---
    def _opening_edge_refs(inst):
        """Return (left_ref, right_ref) for a door/window instance."""
        if FamilyInstanceReferenceType is None:
            return None, None
        try:
            lrefs = inst.GetReferences(FamilyInstanceReferenceType.Left)
            rrefs = inst.GetReferences(FamilyInstanceReferenceType.Right)
            lr = lrefs[0] if lrefs and lrefs.Count > 0 else None
            rr = rrefs[0] if rrefs and rrefs.Count > 0 else None
            return lr, rr
        except Exception:
            return None, None

    # --- Collect wall data ---
    walls = []
    cx_sum, cy_sum = 0.0, 0.0
    for wid in (wall_ids or []):
        try:
            wall = v2.doc.GetElement(ElementId(wid))
            if wall is None:
                continue
            curve = wall.Location.Curve
            p0 = curve.GetEndPoint(0)
            p1 = curve.GetEndPoint(1)
            dx = p1.X - p0.X
            dy = p1.Y - p0.Y
            length = (dx * dx + dy * dy) ** 0.5
            if length < 0.01:
                continue
            dir_x = dx / length
            dir_y = dy / length
            mx = (p0.X + p1.X) / 2.0
            my = (p0.Y + p1.Y) / 2.0
            cx_sum += mx
            cy_sum += my
            # Angle in [0, pi)
            angle = math.atan2(dir_y, dir_x)
            while angle < 0:
                angle += math.pi
            while angle >= math.pi:
                angle -= math.pi
            perp_dist = -dir_y * mx + dir_x * my
            walls.append({
                "id": wid, "wall": wall,
                "p0": (p0.X, p0.Y), "p1": (p1.X, p1.Y),
                "mid": (mx, my), "angle": angle,
                "perp_dist": perp_dist,
                "dir": (dir_x, dir_y),
                "normal": (-dir_y, dir_x),
                "length": length,
            })
        except Exception:
            continue

    if not walls:
        return dim_ids

    centroid = (cx_sum / len(walls), cy_sum / len(walls))

    # --- Classify walls into 4 cardinal groups (N/S/E/W) ---
    # Each wall's outward normal (away from centroid) determines its group.
    # N = normal predominantly +Y, S = -Y, E = +X, W = -X.
    # This ensures only 4 dimension sides, each on a single straight line
    # outside the building — never following L-shape contours inward.
    cardinal_groups = {"N": [], "S": [], "E": [], "W": []}
    for w in walls:
        nx, ny = w["normal"]
        to_c = (centroid[0] - w["mid"][0], centroid[1] - w["mid"][1])
        if nx * to_c[0] + ny * to_c[1] > 0:
            nx, ny = -nx, -ny
        w["out_normal"] = (nx, ny)
        if abs(ny) >= abs(nx):
            key = "N" if ny > 0 else "S"
        else:
            key = "E" if nx > 0 else "W"
        cardinal_groups[key].append(w)

    # --- Map openings to host walls ---
    opening_by_wall = {}
    for oid in list(door_ids or []) + list(window_ids or []):
        try:
            inst = v2.doc.GetElement(ElementId(oid))
            if inst is None:
                continue
            host = inst.Host
            if host is None:
                continue
            lr, rr = _opening_edge_refs(inst)
            pt = inst.Location.Point
            opening_by_wall.setdefault(host.Id.IntegerValue, []).append({
                "pos": (pt.X, pt.Y), "left": lr, "right": rr,
            })
        except Exception:
            continue

    # --- Dimension spacing ---
    row_spacing_ft = v2.cm_to_ft(90.0)   # 90cm from outer wall and between detail rows
    overall_gap_ft = v2.cm_to_ft(50.0)   # 50cm from last detail row to overall dim

    # --- Create dimensions per cardinal side ---
    sides_created = 0
    t = Transaction(v2.doc, "C2Rv7_C Create Dimensions")
    t.Start()
    try:
        for card_key in ("N", "S", "E", "W"):
            group_walls = cardinal_groups[card_key]
            if not group_walls:
                continue
            try:
                # Fixed side direction and outward normal per cardinal key.
                # N: walls face north → horizontal walls, dim line above
                # S: walls face south → horizontal walls, dim line below
                # E: walls face east  → vertical walls, dim line right
                # W: walls face west  → vertical walls, dim line left
                if card_key == "N":
                    sdir = (1.0, 0.0)
                    snorm = (0.0, 1.0)
                elif card_key == "S":
                    sdir = (1.0, 0.0)
                    snorm = (0.0, -1.0)
                elif card_key == "E":
                    sdir = (0.0, 1.0)
                    snorm = (1.0, 0.0)
                else:  # W
                    sdir = (0.0, 1.0)
                    snorm = (-1.0, 0.0)

                # Find the most exterior wall endpoint along outward normal.
                # ALL dimension chains for this side sit at fixed offset from here.
                all_pts = []
                for w in group_walls:
                    all_pts.append(w["p0"])
                    all_pts.append(w["p1"])

                max_ext_perp = max(
                    snorm[0] * pt[0] + snorm[1] * pt[1] for pt in all_pts)

                # --- Collect refs using exterior faces of perpendicular walls
                # at corners, plus parallel wall end faces elsewhere ---
                #
                # At corners: the perpendicular wall's exterior side face
                # is what you actually see from outside — gives the true
                # corner position with no wall-thickness gap.
                # At intermediate points (T-junctions): use parallel wall
                # end faces since there's no perpendicular exterior wall.

                perp_keys = ["E", "W"] if card_key in ("N", "S") else ["N", "S"]

                # Parallel wall endpoints for junction matching
                par_ep_list = []
                for w in group_walls:
                    par_ep_list.append(w["p0"])
                    par_ep_list.append(w["p1"])

                ep_tol = v2.cm_to_ft(ext_thick_cm * 2.0)
                centroid_sdir = sdir[0] * centroid[0] + sdir[1] * centroid[1]

                # Step 1: Perpendicular wall exterior face refs at corners
                corner_refs = []
                corner_positions = []
                for pk in perp_keys:
                    for pw in cardinal_groups.get(pk, []):
                        # Must share an endpoint with a parallel wall
                        at_corner = False
                        for pep in par_ep_list:
                            for wep in [pw["p0"], pw["p1"]]:
                                if (abs(pep[0] - wep[0]) < ep_tol and
                                        abs(pep[1] - wep[1]) < ep_tol):
                                    at_corner = True
                                    break
                            if at_corner:
                                break
                        if not at_corner:
                            continue
                        side_refs = _wall_faces_by_normal(pw["wall"], sdir)
                        if not side_refs:
                            continue
                        # Exterior face = further from centroid along sdir
                        pw_sdir = sdir[0] * pw["mid"][0] + sdir[1] * pw["mid"][1]
                        if pw_sdir < centroid_sdir:
                            ext_ref = side_refs[0]   # leftmost face
                        else:
                            ext_ref = side_refs[-1]  # rightmost face
                        corner_refs.append(ext_ref)
                        corner_positions.append(ext_ref[0])

                # Step 2: Parallel wall end face refs (skip those near corners)
                par_refs = []
                for w in group_walls:
                    w_refs = _wall_faces_by_normal(w["wall"], sdir)
                    if not w_refs:
                        continue
                    # Keep refs closest to actual endpoints (t~0 and t~wlen)
                    wlen = w["length"]
                    p0 = w["p0"]
                    wdir_xy = w["dir"]
                    base_proj = sdir[0] * p0[0] + sdir[1] * p0[1]
                    dir_dot = sdir[0] * wdir_xy[0] + sdir[1] * wdir_xy[1]
                    start_t = 0.0
                    end_t = wlen
                    start_ref = min(w_refs, key=lambda r: abs(
                        r[0] - (base_proj + start_t * dir_dot)))
                    end_ref = min(w_refs, key=lambda r: abs(
                        r[0] - (base_proj + end_t * dir_dot)))
                    for candidate in [start_ref, end_ref]:
                        # Skip if near a corner ref (perp wall takes priority)
                        near_corner = False
                        for cp in corner_positions:
                            if abs(candidate[0] - cp) < ep_tol:
                                near_corner = True
                                break
                        if not near_corner:
                            par_refs.append(candidate)

                # Combine and sort
                seg_refs = corner_refs + par_refs
                seg_refs.sort()

                # Deduplicate truly overlapping refs
                if len(seg_refs) > 2:
                    deduped = [seg_refs[0]]
                    for k in range(1, len(seg_refs)):
                        if abs(seg_refs[k][0] - deduped[-1][0]) > 0.05:
                            deduped.append(seg_refs[k])
                    seg_refs = deduped

                # Collect opening refs for this side
                side_opening_refs = []
                has_openings = False
                for w in group_walls:
                    ops = opening_by_wall.get(w["id"], [])
                    for op in ops:
                        has_openings = True
                        pos = sdir[0] * op["pos"][0] + sdir[1] * op["pos"][1]
                        if op["left"]:
                            side_opening_refs.append((pos - 0.001, op["left"]))
                        if op["right"]:
                            side_opening_refs.append((pos + 0.001, op["right"]))

                # For opening row: bookend with first/last segment refs
                if has_openings and seg_refs:
                    side_opening_refs.append(seg_refs[0])
                    side_opening_refs.append(seg_refs[-1])
                    side_opening_refs.sort()

                # Helper to create a dimension line at fixed offset from
                # the most exterior wall — always straight, always outside.
                def _make_dim_line(offset_ft, _snorm=snorm, _sdir=sdir,
                                   _max_perp=max_ext_perp):
                    ox = _snorm[0] * (_max_perp + offset_ft)
                    oy = _snorm[1] * (_max_perp + offset_ft)
                    return Line.CreateBound(
                        XYZ(ox + _sdir[0] * (-500.0), oy + _sdir[1] * (-500.0), 0.0),
                        XYZ(ox + _sdir[0] * 500.0, oy + _sdir[1] * 500.0, 0.0))

                def _place_dim(refs_list, offset_ft):
                    if len(refs_list) < 2:
                        return
                    ra = ReferenceArray()
                    for _, ref in refs_list:
                        ra.Append(ref)
                    if ra.Size < 2:
                        return
                    dim_line = _make_dim_line(offset_ft)
                    try:
                        dim = v2.doc.Create.NewDimension(plan_view, dim_line, ra)
                        if dim is not None:
                            dim_ids.append(dim.Id.IntegerValue)
                    except Exception:
                        pass

                cur_offset = row_spacing_ft  # first row 90cm from outer wall

                # Row 1: Opening dimensions (only if openings exist on this side)
                if has_openings and len(side_opening_refs) >= 2:
                    _place_dim(side_opening_refs, cur_offset)
                    cur_offset += row_spacing_ft

                # Row 2: Segment dimensions (only if more than one segment)
                if len(seg_refs) >= 3:
                    _place_dim(seg_refs, cur_offset)
                    cur_offset += overall_gap_ft  # 50cm gap to overall

                # Row 3: Overall dimension (always — start to end)
                if len(seg_refs) >= 2:
                    overall = [seg_refs[0], seg_refs[-1]]
                    _place_dim(overall, cur_offset)

                sides_created += 1
            except Exception:
                continue

        t.Commit()
    except Exception as ex:
        if snapshot:
            try:
                snapshot.log("Dimension creation error: {}".format(ex))
            except Exception:
                pass
        try:
            t.RollBack()
        except Exception:
            pass

    if snapshot:
        try:
            snapshot.log("Dimensions created: {} dims across {} sides".format(
                len(dim_ids), sides_created))
        except Exception:
            pass

    return dim_ids


def _extract_inner_face_lines(rec, ext_raw, cfg, snapshot=None):
    """From raw exterior wall DWG lines, extract only the inner face lines.

    For each wall pair (inner/outer), the inner face is the line whose
    midpoint is closer to the centroid of all exterior wall midpoints
    (i.e. closer to the building interior).

    Returns list of line dicts (in cm) representing the inner face.
    """
    if not ext_raw or len(ext_raw) < 2:
        return list(ext_raw or [])

    pairs, pair_dists, used = rec._find_wall_pairs(ext_raw, cfg)
    if not pairs:
        if snapshot:
            try:
                snapshot.log("  inner face: no wall pairs found, using all ext lines")
            except Exception:
                pass
        return list(ext_raw)

    # Compute centroid of all exterior wall midpoints
    cx, cy = 0.0, 0.0
    for ln in ext_raw:
        mx = (float(ln["x1"]) + float(ln["x2"])) / 2.0
        my = (float(ln["y1"]) + float(ln["y2"])) / 2.0
        cx += mx
        cy += my
    cx /= len(ext_raw)
    cy /= len(ext_raw)

    # For each pair, pick the line closer to centroid (= inner face)
    inner_indices = set()
    for i, j in pairs:
        mi = ((float(ext_raw[i]["x1"]) + float(ext_raw[i]["x2"])) / 2.0,
              (float(ext_raw[i]["y1"]) + float(ext_raw[i]["y2"])) / 2.0)
        mj = ((float(ext_raw[j]["x1"]) + float(ext_raw[j]["x2"])) / 2.0,
              (float(ext_raw[j]["y1"]) + float(ext_raw[j]["y2"])) / 2.0)
        di = (mi[0] - cx) ** 2 + (mi[1] - cy) ** 2
        dj = (mj[0] - cx) ** 2 + (mj[1] - cy) ** 2
        inner_indices.add(i if di < dj else j)

    # Also include unpaired lines (they might be single-line wall segments)
    used_indices = set()
    for i, j in pairs:
        used_indices.add(i)
        used_indices.add(j)

    inner_lines = []
    for idx in inner_indices:
        inner_lines.append(ext_raw[idx])

    if snapshot:
        try:
            snapshot.log("  inner face: {} pairs -> {} inner lines, centroid=({:.0f},{:.0f})cm".format(
                len(pairs), len(inner_lines), cx, cy))
        except Exception:
            pass

    return inner_lines


def _create_floors(v2, level, ext_inner_lines_cm, concrete_type, tile_type,
                   snapshot=None):
    """Create concrete structural floor + tile finish floors.

    Uses DWG inner face lines of exterior walls (in cm) to build the
    floor polygon — this gives the exact interior boundary.
    """
    from Autodesk.Revit.DB import (
        CurveLoop as _CL,
        Floor as _Floor,
        BuiltInParameter as _BIP,
    )

    floor_ids = []
    tile_floor_ids = []

    # Read concrete slab thickness for logging
    concrete_thickness_ft = 0.0
    if concrete_type is not None:
        try:
            cs = concrete_type.GetCompoundStructure()
            if cs is not None:
                concrete_thickness_ft = cs.GetWidth()  # feet
        except Exception:
            pass
        if snapshot:
            try:
                snapshot.log("Concrete slab thickness: {:.1f}cm ({:.4f}ft)".format(
                    concrete_thickness_ft * 30.48, concrete_thickness_ft))
            except Exception:
                pass

    # Convert inner face lines from cm to feet curves
    inner_curves_ft = []
    for ln in (ext_inner_lines_cm or []):
        x1 = v2.cm_to_ft(float(ln["x1"]))
        y1 = v2.cm_to_ft(float(ln["y1"]))
        x2 = v2.cm_to_ft(float(ln["x2"]))
        y2 = v2.cm_to_ft(float(ln["y2"]))
        inner_curves_ft.append(((x1, y1), (x2, y2)))

    if snapshot:
        try:
            snapshot.log("  floor: {} inner face curves (cm->ft)".format(len(inner_curves_ft)))
        except Exception:
            pass

    polygon = _chain_ext_wall_loop(inner_curves_ft, tol_ft=0.5, snapshot=snapshot)

    if snapshot and polygon:
        try:
            for i, (x, y) in enumerate(polygon):
                snapshot.log("  floor vtx[{}] ({:.2f},{:.2f})ft".format(i, x, y))
        except Exception:
            pass

    def _make_floor(v2, polygon, floor_type, level, label, offset_ft, snapshot):
        """Create a single floor from polygon points. Returns element id or None."""
        pts = [v2.XYZ(x, y, 0.0) for x, y in polygon]
        t = v2.Transaction(v2.doc, "C2Rv7_C Create {} Floor".format(label))
        t.Start()
        try:
            loop = v2.build_loop(pts)
            from System.Collections.Generic import List
            loop_list = List[_CL]()
            loop_list.Add(loop)
            floor = _Floor.Create(v2.doc, loop_list, floor_type.Id, level.Id)
            if offset_ft > 0.0:
                p = floor.get_Parameter(_BIP.FLOOR_HEIGHTABOVELEVEL_PARAM)
                if p and not p.IsReadOnly:
                    p.Set(offset_ft)
            if snapshot:
                try:
                    snapshot.log("{} floor created: id={} ({} vertices, offset={:.1f}cm)".format(
                        label, floor.Id.IntegerValue, len(pts), offset_ft * 30.48))
                except Exception:
                    pass
            t.Commit()
            return floor.Id.IntegerValue
        except Exception as ex:
            if snapshot:
                try:
                    snapshot.log("{} floor FAILED: {}".format(label, ex))
                except Exception:
                    pass
            try:
                t.RollBack()
            except Exception:
                pass
            return None

    # --- Concrete floor (at level, thickness goes DOWN) ---
    if len(polygon) >= 3 and concrete_type is not None:
        fid = _make_floor(v2, polygon, concrete_type, level, "Concrete", 0.0, snapshot)
        if fid is not None:
            floor_ids.append(fid)

    # --- Tile floor (at level, offset=0 — sits on concrete top surface) ---
    if len(polygon) >= 3 and tile_type is not None:
        fid = _make_floor(v2, polygon, tile_type, level, "Tile", 0.0, snapshot)
        if fid is not None:
            tile_floor_ids.append(fid)

    return floor_ids, tile_floor_ids


def _line_keys_set(lines):
    return set(_line_key(ln) for ln in (lines or []))


def _diff_removed_lines(before, after):
    after_keys = _line_keys_set(after)
    return [ln for ln in (before or []) if _line_key(ln) not in after_keys]


def _group_removed_as_fragments(removed_lines, tol_cm):
    if not removed_lines:
        return []
    return _connected_components(removed_lines, tol_cm)


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

        # Prompt user for floor type selection early (before any transactions)
        floor_choice = _pick_floor_types(v2)
        concrete_type = floor_choice[0] if floor_choice else None
        tile_type = floor_choice[1] if floor_choice else None

        llm_qa = _load_llm_qa()
        qa_report = {
            "enabled": bool(llm_qa is not None and llm_qa.qa_enabled(cfg)),
            "envelope": None,
            "fragments": None,
            "thickness": None,
            "restored_fragment_lines": 0,
        }
        if snapshot is not None:
            try:
                snapshot.log("LLM-QA enabled={}".format(qa_report["enabled"]))
            except Exception:
                pass

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

        # Hook 1: validate exterior envelope looks like a real building footprint.
        if qa_report["enabled"]:
            try:
                env_res = llm_qa.validate_exterior_envelope(cfg, ext_main, snapshot=snapshot)
                qa_report["envelope"] = env_res
                if env_res.get("verdict") == "reject" and float(env_res.get("confidence", 0.0)) >= 0.7:
                    if snapshot is not None:
                        try:
                            snapshot.log("LLM-QA envelope REJECT (conf {:.2f}): {}".format(
                                float(env_res.get("confidence", 0.0)),
                                env_res.get("reason", "")[:160],
                            ))
                        except Exception:
                            pass
            except Exception:
                qa_report["envelope"] = {"verdict": "no_opinion", "error": "exception"}

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

        # Hook 3: thickness sanity check (informational only, never changes numbers).
        if qa_report["enabled"]:
            try:
                thk_res = llm_qa.sanity_check_thicknesses(
                    cfg, ext_thick_cm, int_thick_cm, ext_center, int_center, snapshot=snapshot,
                )
                qa_report["thickness"] = thk_res
            except Exception:
                qa_report["thickness"] = {"verdict": "no_opinion", "error": "exception"}

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
        int_before_removal = list(int_center)
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

        # Hook 2: ask Claude whether any removed fragment was a legitimate short wall.
        if qa_report["enabled"]:
            try:
                removed = _diff_removed_lines(int_before_removal, int_center)
                fragments = _group_removed_as_fragments(removed, cleanup_tol_cm)
                fragments = [
                    frag for frag in fragments
                    if frag and sum(_line_len_cm(ln) for ln in frag) >= max(10.0, 0.5 * cleanup_tol_cm)
                ]
                if fragments:
                    frag_res = llm_qa.classify_interior_fragments(
                        cfg, int_center, fragments, snapshot=snapshot,
                    )
                    qa_report["fragments"] = {
                        "count_considered": len(fragments),
                        "verdicts": frag_res.get("verdicts", {}),
                        "confidences": frag_res.get("confidences", {}),
                        "reasons": frag_res.get("reasons", {}),
                    }
                    restored = 0
                    for i, frag in enumerate(fragments):
                        verdict = frag_res.get("verdicts", {}).get(str(i), "unsure")
                        conf = float(frag_res.get("confidences", {}).get(str(i), 0.0))
                        if verdict == "keep" and conf >= 0.6:
                            int_center.extend(frag)
                            restored += len(frag)
                    qa_report["restored_fragment_lines"] = restored
                    if snapshot is not None:
                        try:
                            snapshot.log("LLM-QA fragments: reviewed={} restored_lines={}".format(
                                len(fragments), restored,
                            ))
                        except Exception:
                            pass
            except Exception:
                qa_report["fragments"] = {"error": "exception"}

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

        # --- Detect doors BEFORE creating walls ---
        # If a door sits in a gap between two collinear wall segments,
        # bridge the gap so the wall is continuous and can host the door.
        raw_data = _RAW_CAD_CACHE.get("data")
        door_markers = []
        window_markers = []
        if raw_data is not None:
            door_markers = _detect_opening_markers(raw_data, "door")
            window_markers = _detect_opening_markers(raw_data, "window")

            # Bridge centerline gaps at door locations
            ext_center = _bridge_gaps_at_doors(ext_center, door_markers, snapshot)
            int_center = _bridge_gaps_at_doors(int_center, door_markers, snapshot)

        ext_type = _pick_wall_type(v2, ext_thick_cm, ["EXTERIOR", "EXT"])
        if ext_type is None:
            raise Exception("No Basic wall type found for exterior walls.")

        wall_ids = []
        internal_wall_ids = []
        t = v2.Transaction(v2.doc, "Create Model From CAD V2 (C2Rv7_C Layer-First Walls)")
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

        # --- Opening placement (markers already detected above) ---
        door_ids = []
        window_ids = []
        opening_errors = []
        if raw_data is not None:

            if snapshot:
                try:
                    snapshot.log("Opening markers detected: {} doors, {} windows ({} raw lines, {} raw arcs)".format(
                        len(door_markers), len(window_markers),
                        len(raw_data.get("lines") or []), len(raw_data.get("arcs") or [])))
                    for i, mk in enumerate(door_markers):
                        sw = mk.get("swing") or {}
                        snapshot.log("  door[{}] center=({:.0f},{:.0f}) width={:.0f}cm({}) {} swing={} hinge={} open={}".format(
                            i, mk["center_cm"][0], mk["center_cm"][1], mk["width_cm"],
                            mk.get("width_source", "?"),
                            "DOUBLE" if mk.get("is_double") else "single",
                            sw.get("source", "?"),
                            sw.get("hinge_cm", "?"), sw.get("open_cm", "?")))
                    for i, mk in enumerate(window_markers):
                        snapshot.log("  win[{}] center=({:.0f},{:.0f}) width={:.0f}cm({})".format(
                            i, mk["center_cm"][0], mk["center_cm"][1], mk["width_cm"], mk.get("width_source", "?")))
                except Exception:
                    pass

            if door_markers or window_markers:
                t2 = v2.Transaction(v2.doc, "C2Rv7_C Place Openings")
                # Suppress "Can't make type" dialogs — auto-delete failing elements
                fho = t2.GetFailureHandlingOptions()
                fho.SetFailuresPreprocessor(_SwallowTypeErrors())
                t2.SetFailureHandlingOptions(fho)
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

        # --- Extract inner face lines from ext wall pairs for floor ---
        ext_inner_lines_cm = _extract_inner_face_lines(rec, ext_raw, cfg, snapshot)

        # --- Floor creation ---
        concrete_floor_ids = []
        tile_floor_ids = []
        if concrete_type is not None or tile_type is not None:
            try:
                concrete_floor_ids, tile_floor_ids = _create_floors(
                    v2, level, ext_inner_lines_cm,
                    concrete_type, tile_type,
                    snapshot=snapshot)
            except Exception as ex:
                if snapshot:
                    try:
                        import traceback
                        snapshot.log("Floor creation error: {}\n{}".format(ex, traceback.format_exc()))
                    except Exception:
                        snapshot.log("Floor creation error: {}".format(ex))

        # --- Room + Room Tag creation ---
        room_ids = []
        tag_ids = []
        try:
            room_ids, tag_ids = _create_rooms_and_tags(v2, level, snapshot)
        except Exception as ex:
            if snapshot:
                try:
                    snapshot.log("Room creation error: {}".format(ex))
                except Exception:
                    pass

        # --- Exterior dimensions ---
        ext_dim_ids = []
        try:
            ext_dim_ids = _create_exterior_dimensions(
                v2, level, wall_ids, door_ids, window_ids,
                ext_thick_cm, snapshot)
        except Exception as ex:
            if snapshot:
                try:
                    snapshot.log("Dimension creation error: {}".format(ex))
                except Exception:
                    pass

        if snapshot is not None:
            try:
                snapshot.log("Layer-first walls created: ext={} int={}, doors={} windows={}, "
                             "concrete_floors={} tile_floors={}, rooms={} tags={}, dims={}".format(
                    len(wall_ids), len(internal_wall_ids), len(door_ids), len(window_ids),
                    len(concrete_floor_ids), len(tile_floor_ids),
                    len(room_ids), len(tag_ids), len(ext_dim_ids)))
                snapshot.save_json("08_geometry_summary.json", {
                    "geometry": {
                        "wall_ids": wall_ids,
                        "internal_wall_ids": internal_wall_ids,
                        "door_ids": door_ids,
                        "window_ids": window_ids,
                        "concrete_floor_ids": concrete_floor_ids,
                        "tile_floor_ids": tile_floor_ids,
                        "room_ids": room_ids,
                        "tag_ids": tag_ids,
                        "ext_dim_ids": ext_dim_ids,
                        "opening_errors": opening_errors,
                        "perimeter_wall_thickness_cm": float(ext_thick_cm),
                        "internal_wall_thickness_cm": float(int_thick_cm),
                    },
                    "llm_qa": qa_report,
                })
            except Exception:
                pass

        return {
            "geometry": {
                "wall_ids": wall_ids,
                "internal_wall_ids": internal_wall_ids,
                "door_ids": door_ids,
                "window_ids": window_ids,
                "concrete_floor_ids": concrete_floor_ids,
                "tile_floor_ids": tile_floor_ids,
                "room_ids": room_ids,
                "tag_ids": tag_ids,
                "ext_dim_ids": ext_dim_ids,
                "opening_errors": opening_errors,
                "perimeter_wall_thickness_cm": float(ext_thick_cm),
                "internal_wall_thickness_cm": float(int_thick_cm),
            },
            "llm_qa": qa_report,
        }

    v2.build_model_from_topology = _build_layer_first


def main():
    uidoc = __revit__.ActiveUIDocument
    if uidoc is None:
        TaskDialog.Show("C2Rv7_C", "No active Revit document.")
        return

    selected_import = _pick_dwg_import(uidoc)
    if selected_import is None:
        TaskDialog.Show("C2Rv7_C", "Canceled: no DWG import selected.")
        return

    v2 = _load_v2_module()
    if not hasattr(v2, "run_command"):
        raise Exception("CreateFromCADV2 script is missing run_command().")

    _apply_selected_import_scope(v2, selected_import)
    _apply_layer_first_wall_mode(v2, selected_import)
    v2.run_command()


if __name__ == "__main__":
    main()
