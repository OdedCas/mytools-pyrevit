# -*- coding: utf-8 -*-
"""Room topology + opening extraction from CAD linework.

Supports:
- rectilinear mode (legacy rectangle logic)
- polygon mode (closed loop from arbitrary angled linework)
"""

import math


def _abs(v):
    return v if v >= 0.0 else -v


def _line_angle_deg(ln):
    dx = float(ln["x2"]) - float(ln["x1"])
    dy = float(ln["y2"]) - float(ln["y1"])
    return math.degrees(math.atan2(dy, dx))


def _line_length_cm(ln):
    dx = float(ln["x2"]) - float(ln["x1"])
    dy = float(ln["y2"]) - float(ln["y1"])
    return math.sqrt((dx * dx) + (dy * dy))


def _is_horizontal(ln, tol_deg):
    a = _abs(_line_angle_deg(ln))
    return (a <= tol_deg) or (_abs(a - 180.0) <= tol_deg)


def _is_vertical(ln, tol_deg):
    a = _abs(_line_angle_deg(ln))
    return _abs(a - 90.0) <= tol_deg


def _angle_delta_deg(a, b):
    d = _abs(a - b) % 180.0
    if d > 90.0:
        d = 180.0 - d
    return d


def _cluster_values(vals, tol_cm):
    if not vals:
        return []
    s = sorted(float(v) for v in vals)
    groups = [[s[0]]]
    for v in s[1:]:
        if _abs(v - groups[-1][-1]) <= tol_cm:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / float(len(g)) for g in groups]


def _pick_inner_pair(coords, default_thk_cm):
    vals = sorted(coords)
    if len(vals) >= 4:
        lo_outer = vals[0]
        lo_inner = vals[1]
        hi_inner = vals[-2]
        hi_outer = vals[-1]
        thk = ((lo_inner - lo_outer) + (hi_outer - hi_inner)) * 0.5
        return lo_inner, hi_inner, max(1.0, thk)
    if len(vals) == 3:
        a, b, c = vals[0], vals[1], vals[2]
        if (c - b) > (b - a):
            return b, c, max(1.0, b - a)
        return a, b, max(1.0, c - b)
    if len(vals) == 2:
        return vals[0], vals[1], default_thk_cm
    raise ValueError("Not enough wall traces to infer room bounds")


def _opening_from_lines(lines, target_y, host_tol_cm, min_w, max_w):
    best = None
    for ln in (lines or []):
        y = (float(ln["y1"]) + float(ln["y2"])) * 0.5
        if _abs(y - target_y) > host_tol_cm:
            continue
        x0 = min(float(ln["x1"]), float(ln["x2"]))
        x1 = max(float(ln["x1"]), float(ln["x2"]))
        w = x1 - x0
        if w < min_w or w > max_w:
            continue
        score = w
        if (best is None) or (score > best["score"]):
            best = {"left": x0, "right": x1, "score": score}
    return best


def _door_from_arcs(arcs, target_y, host_tol_cm, min_w, max_w):
    best = None
    for a in (arcs or []):
        cy = float(a.get("cy", 0.0))
        if _abs(cy - target_y) > host_tol_cm:
            continue
        r = float(a.get("r", 0.0))
        w = 2.0 * r
        if w < min_w or w > max_w:
            continue
        cx = float(a.get("cx", 0.0))
        cand = {"left": cx - (w * 0.5), "right": cx + (w * 0.5), "score": w}
        if (best is None) or (cand["score"] > best["score"]):
            best = cand
    return best


def _line_key(ln):
    p1 = (round(float(ln["x1"]), 4), round(float(ln["y1"]), 4))
    p2 = (round(float(ln["x2"]), 4), round(float(ln["y2"]), 4))
    if p1 <= p2:
        return (p1, p2)
    return (p2, p1)


def _merge_unique_lines(base, extra):
    out = list(base or [])
    seen = set(_line_key(ln) for ln in out)
    for ln in (extra or []):
        k = _line_key(ln)
        if k in seen:
            continue
        seen.add(k)
        out.append(ln)
    return out


def _fallback_wall_lines_from_unclassified(classified, ang_tol, min_len_cm):
    out = []
    for ln in (classified.get("unclassified_lines") or []):
        if _line_length_cm(ln) < min_len_cm:
            continue
        if _is_horizontal(ln, ang_tol) or _is_vertical(ln, ang_tol):
            out.append(ln)
    return out


def _snap_pt(x, y, tol_cm):
    if tol_cm <= 1e-6:
        return (float(x), float(y))
    return (
        round(float(x) / tol_cm) * tol_cm,
        round(float(y) / tol_cm) * tol_cm,
    )


def _build_graph_segments(lines, snap_cm, min_len_cm):
    segs = []
    seen = set()
    for ln in (lines or []):
        if _line_length_cm(ln) < min_len_cm:
            continue
        p1 = _snap_pt(ln["x1"], ln["y1"], snap_cm)
        p2 = _snap_pt(ln["x2"], ln["y2"], snap_cm)
        if p1 == p2:
            continue
        key = (p1, p2) if p1 <= p2 else (p2, p1)
        if key in seen:
            continue
        seen.add(key)
        segs.append((p1, p2))

    node_ids = {}
    nodes = []

    def get_id(p):
        i = node_ids.get(p)
        if i is not None:
            return i
        nid = len(nodes)
        node_ids[p] = nid
        nodes.append(p)
        return nid

    adj = {}
    for p1, p2 in segs:
        a = get_id(p1)
        b = get_id(p2)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    return nodes, adj


def _canonical_cycle(path):
    n = len(path)
    if n == 0:
        return tuple()

    def rotate(lst, k):
        return lst[k:] + lst[:k]

    candidates = []
    for seq in (path, list(reversed(path))):
        m = min(seq)
        for i in range(n):
            if seq[i] == m:
                candidates.append(tuple(rotate(seq, i)))
    return min(candidates)


def _find_cycles(nodes, adj, max_len, max_cycles):
    found = set()

    def dfs(start, curr, parent, path, visited):
        if len(found) >= max_cycles:
            return
        for nxt in adj.get(curr, []):
            if nxt == parent:
                continue
            if nxt == start and len(path) >= 3:
                cyc = _canonical_cycle(path)
                if cyc:
                    found.add(cyc)
                continue
            if nxt in visited:
                continue
            if len(path) >= max_len:
                continue
            if nxt < start:
                continue
            visited.add(nxt)
            dfs(start, nxt, curr, path + [nxt], visited)
            visited.remove(nxt)

    for start in range(len(nodes)):
        visited = set([start])
        dfs(start, start, -1, [start], visited)
        if len(found) >= max_cycles:
            break
    return [list(c) for c in found]


def _poly_area_signed(points):
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += (x1 * y2) - (x2 * y1)
    return 0.5 * s


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _dist_pt_seg(px, py, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    vv = (vx * vx) + (vy * vy)
    if vv <= 1e-9:
        dx = px - ax
        dy = py - ay
        return math.sqrt((dx * dx) + (dy * dy)), 0.0
    t = ((wx * vx) + (wy * vy)) / vv
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + (vx * t)
    cy = ay + (vy * t)
    dx = px - cx
    dy = py - cy
    return math.sqrt((dx * dx) + (dy * dy)), t


def _project_t_cm(px, py, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    ln = math.sqrt((vx * vx) + (vy * vy))
    if ln <= 1e-9:
        return 0.0
    ux = vx / ln
    uy = vy / ln
    return ((px - ax) * ux) + ((py - ay) * uy)


def _edge_list(poly):
    out = []
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        ln = math.sqrt((dx * dx) + (dy * dy))
        if ln <= 1e-6:
            continue
        out.append({
            "idx": i,
            "a": a,
            "b": b,
            "len": ln,
            "ang": math.degrees(math.atan2(dy, dx)),
        })
    return out


def _pick_polygon_cycle(cycles, nodes, classified, cfg):
    min_area = float(cfg.get("polygon_min_area_cm2", 4000.0))
    host_tol = float(cfg.get("opening_host_distance_mm", 30.0)) / 10.0
    opening_lines = list(classified.get("door_lines") or []) + list(classified.get("window_lines") or [])
    opening_arcs = list(classified.get("door_arcs") or []) + list(classified.get("window_arcs") or [])

    best = None
    for cyc in cycles:
        pts = [nodes[i] for i in cyc]
        area = _abs(_poly_area_signed(pts))
        if area < min_area:
            continue
        edges = _edge_list(pts)
        if len(edges) < 3:
            continue

        support = 0
        for ln in opening_lines:
            mx = (float(ln["x1"]) + float(ln["x2"])) * 0.5
            my = (float(ln["y1"]) + float(ln["y2"])) * 0.5
            dmin = 1.0e20
            for e in edges:
                d, _ = _dist_pt_seg(mx, my, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
                if d < dmin:
                    dmin = d
            if dmin <= (host_tol * 2.0):
                support += 1
        for a in opening_arcs:
            mx = float(a.get("cx", 0.0))
            my = float(a.get("cy", 0.0))
            dmin = 1.0e20
            for e in edges:
                d, _ = _dist_pt_seg(mx, my, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
                if d < dmin:
                    dmin = d
            if dmin <= (host_tol * 3.0):
                support += 1

        score = (support * 1000000.0) - area
        cand = {"poly": pts, "score": score, "area": area, "support": support}
        if (best is None) or (cand["score"] > best["score"]):
            best = cand
    return best


def _opening_candidates_from_lines(lines, edges, kind, host_tol_cm, min_w, max_w, ang_tol_deg):
    out = []
    for ln in (lines or []):
        x1 = float(ln["x1"])
        y1 = float(ln["y1"])
        x2 = float(ln["x2"])
        y2 = float(ln["y2"])
        w = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
        if w < min_w or w > max_w:
            continue
        lang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        mx = (x1 + x2) * 0.5
        my = (y1 + y2) * 0.5

        best = None
        for e in edges:
            d, _ = _dist_pt_seg(mx, my, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            if d > host_tol_cm:
                continue
            if _angle_delta_deg(lang, e["ang"]) > ang_tol_deg:
                continue
            t1 = _project_t_cm(x1, y1, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            t2 = _project_t_cm(x2, y2, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            s = max(0.0, min(t1, t2))
            t = min(e["len"], max(t1, t2))
            if (t - s) < min_w:
                continue
            cand = {
                "type": kind,
                "host_edge": e["idx"],
                "start_cm": s,
                "end_cm": t,
                "width_cm": t - s,
                "confidence": max(0.0, 1.0 - (d / max(host_tol_cm, 1e-6))),
            }
            if (best is None) or (cand["confidence"] > best["confidence"]):
                best = cand
        if best:
            out.append(best)
    return out


def _door_candidates_from_arcs(arcs, edges, host_tol_cm, min_w, max_w):
    out = []
    for a in (arcs or []):
        cx = float(a.get("cx", 0.0))
        cy = float(a.get("cy", 0.0))
        r = float(a.get("r", 0.0))
        w = 2.0 * r
        if w < min_w or w > max_w:
            continue
        best = None
        for e in edges:
            d, _ = _dist_pt_seg(cx, cy, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            if d > (host_tol_cm * 2.0):
                continue
            tc = _project_t_cm(cx, cy, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            s = max(0.0, tc - (w * 0.5))
            t = min(e["len"], tc + (w * 0.5))
            cand = {
                "type": "door",
                "host_edge": e["idx"],
                "start_cm": s,
                "end_cm": t,
                "width_cm": t - s,
                "confidence": max(0.0, 1.0 - (d / max(host_tol_cm * 2.0, 1e-6))),
            }
            if (best is None) or (cand["confidence"] > best["confidence"]):
                best = cand
        if best:
            out.append(best)
    return out


def _merge_openings(cands, min_sep_cm):
    out = []
    groups = {}
    for c in (cands or []):
        key = (c.get("type"), int(c.get("host_edge", -1)))
        groups.setdefault(key, []).append(c)
    for _, arr in groups.items():
        arr = sorted(arr, key=lambda x: (x["start_cm"], -x.get("confidence", 0.0)))
        merged = []
        for c in arr:
            if not merged:
                merged.append(dict(c))
                continue
            prev = merged[-1]
            if c["start_cm"] <= (prev["end_cm"] + min_sep_cm):
                prev["start_cm"] = min(prev["start_cm"], c["start_cm"])
                prev["end_cm"] = max(prev["end_cm"], c["end_cm"])
                prev["width_cm"] = prev["end_cm"] - prev["start_cm"]
                prev["confidence"] = max(prev.get("confidence", 0.0), c.get("confidence", 0.0))
            else:
                merged.append(dict(c))
        out.extend(merged)
    return out


def _infer_gap_openings(edges, support_lines, host_tol_cm, min_w, max_w):
    gaps = []
    for e in edges:
        intervals = []
        for ln in (support_lines or []):
            x1 = float(ln["x1"])
            y1 = float(ln["y1"])
            x2 = float(ln["x2"])
            y2 = float(ln["y2"])
            lang = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if _angle_delta_deg(lang, e["ang"]) > 12.0:
                continue
            mx = (x1 + x2) * 0.5
            my = (y1 + y2) * 0.5
            d, _ = _dist_pt_seg(mx, my, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            if d > host_tol_cm:
                continue
            t1 = _project_t_cm(x1, y1, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            t2 = _project_t_cm(x2, y2, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            s = max(0.0, min(t1, t2))
            t = min(e["len"], max(t1, t2))
            if (t - s) >= 1.0:
                intervals.append((s, t))
        if not intervals:
            continue
        intervals.sort()
        merged = [list(intervals[0])]
        for s, t in intervals[1:]:
            if s <= merged[-1][1]:
                if t > merged[-1][1]:
                    merged[-1][1] = t
            else:
                merged.append([s, t])
        cursor = 0.0
        for s, t in merged:
            if s > cursor:
                gw = s - cursor
                if gw >= min_w and gw <= max_w:
                    gaps.append({"host_edge": e["idx"], "start_cm": cursor, "end_cm": s, "width_cm": gw})
            cursor = max(cursor, t)
        if cursor < e["len"]:
            gw = e["len"] - cursor
            if gw >= min_w and gw <= max_w:
                gaps.append({"host_edge": e["idx"], "start_cm": cursor, "end_cm": e["len"], "width_cm": gw})
    return gaps


def _rectilinear_topology(classified, cfg, wall_lines):
    ang_tol = float(cfg.get("parallel_angle_deg", 0.5))
    cluster_tol_cm = float(cfg.get("join_tol_mm", 5.0)) / 10.0
    vertical = [ln for ln in wall_lines if _is_vertical(ln, ang_tol)]
    horizontal = [ln for ln in wall_lines if _is_horizontal(ln, ang_tol)]
    if len(vertical) < 2 or len(horizontal) < 2:
        raise ValueError("Wall recognition failed: need horizontal and vertical walls")

    x_vals = _cluster_values([((ln["x1"] + ln["x2"]) * 0.5) for ln in vertical], cluster_tol_cm)
    y_vals = _cluster_values([((ln["y1"] + ln["y2"]) * 0.5) for ln in horizontal], cluster_tol_cm)
    default_thk = float(cfg.get("default_wall_thickness_cm", 30.0))
    left_in, right_in, thk_x = _pick_inner_pair(x_vals, default_thk)
    bot_in, top_in, thk_y = _pick_inner_pair(y_vals, default_thk)
    room_w = right_in - left_in
    room_h = top_in - bot_in
    if room_w <= 1.0 or room_h <= 1.0:
        raise ValueError("Invalid inner room bounds from CAD")

    wall_thk = max(1.0, (thk_x + thk_y) * 0.5)
    host_tol = float(cfg.get("opening_host_distance_mm", 30.0)) / 10.0

    door = _opening_from_lines(classified.get("door_lines"), bot_in, host_tol, 40.0, 200.0)
    if door is None:
        door = _door_from_arcs(classified.get("door_arcs"), bot_in, host_tol, 40.0, 200.0)
    window = _opening_from_lines(classified.get("window_lines"), top_in, host_tol, 30.0, 300.0)

    dflt_door_w = float(cfg.get("default_door_width_cm", 100.0))
    dflt_door_h = float(cfg.get("default_door_height_cm", 210.0))
    dflt_win_w = float(cfg.get("default_window_width_cm", 100.0))
    dflt_win_h = float(cfg.get("default_window_height_cm", 100.0))
    dflt_win_sill = float(cfg.get("default_window_sill_cm", 105.0))
    dflt_win_right = float(cfg.get("default_window_right_offset_cm", 30.0))
    dflt_door_left = float(cfg.get("default_door_left_offset_cm", 90.0))

    if door is None:
        door_left = dflt_door_left
        door_width = dflt_door_w
        door_detected = False
    else:
        door_left = max(0.0, float(door["left"]) - left_in)
        door_width = max(1.0, float(door["right"]) - float(door["left"]))
        door_detected = True

    if window is None:
        window_width = dflt_win_w
        window_right = dflt_win_right
        window_detected = False
    else:
        window_width = max(1.0, float(window["right"]) - float(window["left"]))
        window_right = max(0.0, right_in - float(window["right"]))
        window_detected = True

    door_left = max(0.0, min(door_left, max(0.0, room_w - door_width)))
    window_right = max(0.0, min(window_right, max(0.0, room_w - window_width)))

    meas = {
        "room_width_cm": room_w,
        "room_height_cm": room_h,
        "wall_thickness_cm": wall_thk,
        "door_width_cm": door_width,
        "door_height_cm": dflt_door_h,
        "door_left_offset_cm": door_left,
        "window_width_cm": window_width,
        "window_height_cm": dflt_win_h,
        "window_right_offset_cm": window_right,
        "window_sill_cm": dflt_win_sill,
    }

    openings = []
    if door_detected:
        openings.append({"type": "door", "host_edge": 0, "start_cm": door_left, "end_cm": door_left + door_width, "width_cm": door_width, "confidence": 1.0})
    if window_detected:
        ws = max(0.0, room_w - window_right - window_width)
        openings.append({"type": "window", "host_edge": 2, "start_cm": ws, "end_cm": ws + window_width, "width_cm": window_width, "confidence": 1.0})

    return {
        "inner_sw_cm": [left_in, bot_in],
        "inner_se_cm": [right_in, bot_in],
        "inner_ne_cm": [right_in, top_in],
        "inner_nw_cm": [left_in, top_in],
        "room_polygon_cm": [[left_in, bot_in], [right_in, bot_in], [right_in, top_in], [left_in, top_in]],
        "wall_segments_cm": [
            {"start": [left_in, bot_in], "end": [right_in, bot_in], "length_cm": room_w},
            {"start": [right_in, bot_in], "end": [right_in, top_in], "length_cm": room_h},
            {"start": [right_in, top_in], "end": [left_in, top_in], "length_cm": room_w},
            {"start": [left_in, top_in], "end": [left_in, bot_in], "length_cm": room_h},
        ],
        "openings": openings,
        "measurements_cm": meas,
        "debug": {
            "mode": "rectilinear",
            "x_clusters": x_vals,
            "y_clusters": y_vals,
            "x_cluster_count": len(x_vals),
            "y_cluster_count": len(y_vals),
            "door_detected": door_detected,
            "window_detected": window_detected,
        },
    }


def _polygon_topology(classified, cfg, wall_lines):
    snap_cm = float(cfg.get("endpoint_snap_mm", 2.0)) / 10.0
    min_len_cm = float(cfg.get("fallback_min_wall_line_cm", 40.0))
    max_cycle_len = int(cfg.get("polygon_max_cycle_len", 40))
    max_cycles = int(cfg.get("polygon_max_cycles", 200))
    nodes, adj = _build_graph_segments(wall_lines, snap_cm, min_len_cm)
    if len(nodes) < 3:
        raise ValueError("Polygon recognition failed: not enough graph nodes")
    cycles = _find_cycles(nodes, adj, max_cycle_len, max_cycles)
    picked = _pick_polygon_cycle(cycles, nodes, classified, cfg)
    if not picked:
        raise ValueError("Polygon recognition failed: no valid closed room loop")

    poly = picked["poly"]
    edges = _edge_list(poly)
    host_tol = float(cfg.get("opening_host_distance_mm", 30.0)) / 10.0
    poly_host_tol = max(host_tol, float(cfg.get("polygon_opening_host_tol_cm", 15.0)))
    min_gap = float(cfg.get("opening_gap_min_cm", 45.0))
    max_gap = float(cfg.get("opening_gap_max_cm", 220.0))

    door_candidates = []
    door_candidates.extend(_opening_candidates_from_lines(classified.get("door_lines"), edges, "door", poly_host_tol, 40.0, 200.0, 16.0))
    door_candidates.extend(_door_candidates_from_arcs(classified.get("door_arcs"), edges, poly_host_tol, 40.0, 200.0))
    window_candidates = _opening_candidates_from_lines(classified.get("window_lines"), edges, "window", poly_host_tol, 30.0, 400.0, 16.0)

    support_lines = _merge_unique_lines(list(classified.get("wall_lines") or []), list(classified.get("unclassified_lines") or []))
    inferred = _infer_gap_openings(edges, support_lines, poly_host_tol, min_gap, max_gap)
    if (not door_candidates) and inferred:
        g = max(inferred, key=lambda x: x["width_cm"])
        d = dict(g)
        d["type"] = "door"
        d["confidence"] = 0.4
        door_candidates.append(d)
    if (not window_candidates) and inferred:
        for g in inferred:
            skip = False
            for d in door_candidates:
                if int(d.get("host_edge", -1)) != int(g.get("host_edge", -2)):
                    continue
                if _abs(float(d.get("start_cm", 0.0)) - float(g.get("start_cm", 0.0))) <= 1.0 and _abs(float(d.get("end_cm", 0.0)) - float(g.get("end_cm", 0.0))) <= 1.0:
                    skip = True
                    break
            if skip:
                continue
            w = dict(g)
            w["type"] = "window"
            w["confidence"] = 0.35
            window_candidates.append(w)

    door_candidates = _merge_openings(door_candidates, 2.0)
    window_candidates = _merge_openings(window_candidates, 2.0)

    dflt_door_w = float(cfg.get("default_door_width_cm", 100.0))
    dflt_door_h = float(cfg.get("default_door_height_cm", 210.0))
    dflt_win_w = float(cfg.get("default_window_width_cm", 100.0))
    dflt_win_h = float(cfg.get("default_window_height_cm", 100.0))
    dflt_win_sill = float(cfg.get("default_window_sill_cm", 105.0))
    dflt_win_right = float(cfg.get("default_window_right_offset_cm", 30.0))
    dflt_door_left = float(cfg.get("default_door_left_offset_cm", 90.0))

    openings = []
    for d in door_candidates:
        o = dict(d)
        o["height_cm"] = dflt_door_h
        openings.append(o)
    for w in window_candidates:
        o = dict(w)
        o["height_cm"] = dflt_win_h
        o["sill_cm"] = dflt_win_sill
        openings.append(o)
    openings = sorted(openings, key=lambda x: (x.get("host_edge", -1), x.get("start_cm", 0.0), x.get("type", "")))

    minx, miny, maxx, maxy = _bbox(poly)
    room_w = maxx - minx
    room_h = maxy - miny

    first_door = None
    first_window = None
    for o in openings:
        if (first_door is None) and o.get("type") == "door":
            first_door = o
        if (first_window is None) and o.get("type") == "window":
            first_window = o

    if first_door is None:
        door_width = dflt_door_w
        door_left = dflt_door_left
    else:
        door_width = max(1.0, float(first_door.get("width_cm", dflt_door_w)))
        door_left = max(0.0, min(float(first_door.get("start_cm", dflt_door_left)), max(0.0, room_w - door_width)))

    if first_window is None:
        window_width = dflt_win_w
        window_right = dflt_win_right
    else:
        window_width = max(1.0, float(first_window.get("width_cm", dflt_win_w)))
        window_right = max(0.0, min(room_w - float(first_window.get("end_cm", room_w - dflt_win_right)), max(0.0, room_w - window_width)))

    meas = {
        "room_width_cm": max(1.0, room_w),
        "room_height_cm": max(1.0, room_h),
        "wall_thickness_cm": float(cfg.get("default_wall_thickness_cm", 30.0)),
        "door_width_cm": door_width,
        "door_height_cm": dflt_door_h,
        "door_left_offset_cm": door_left,
        "window_width_cm": window_width,
        "window_height_cm": dflt_win_h,
        "window_right_offset_cm": window_right,
        "window_sill_cm": dflt_win_sill,
    }

    wall_segments = []
    for e in edges:
        wall_segments.append({"start": [e["a"][0], e["a"][1]], "end": [e["b"][0], e["b"][1]], "length_cm": e["len"]})

    return {
        "inner_sw_cm": [minx, miny],
        "inner_se_cm": [maxx, miny],
        "inner_ne_cm": [maxx, maxy],
        "inner_nw_cm": [minx, maxy],
        "room_polygon_cm": [[p[0], p[1]] for p in poly],
        "wall_segments_cm": wall_segments,
        "openings": openings,
        "measurements_cm": meas,
        "debug": {
            "mode": "polygon",
            "polygon_area_cm2": picked["area"],
            "polygon_opening_support": picked["support"],
            "cycle_count": len(cycles),
            "door_detected": len(door_candidates) > 0,
            "window_detected": len(window_candidates) > 0,
            "door_candidates": door_candidates,
            "window_candidates": window_candidates,
        },
    }


def derive_room_from_cad(classified, cfg):
    cfg = cfg or {}
    ang_tol = float(cfg.get("parallel_angle_deg", 0.5))
    min_wall_len_cm = float(cfg.get("fallback_min_wall_line_cm", 40.0))
    wall_lines = list(classified.get("wall_lines") or [])
    fallback = _fallback_wall_lines_from_unclassified(classified, ang_tol, min_wall_len_cm)
    if fallback:
        wall_lines = _merge_unique_lines(wall_lines, fallback)

    rect_out = None
    rect_err = None
    try:
        rect_out = _rectilinear_topology(classified, cfg, wall_lines)
    except Exception as ex:
        rect_err = str(ex)

    poly_out = None
    poly_err = None
    try:
        poly_out = _polygon_topology(classified, cfg, wall_lines)
    except Exception as ex:
        poly_err = str(ex)

    prefer_poly = bool(cfg.get("prefer_polygon_mode", True))

    # Auto-pick polygon for complex orthogonal layouts (e.g. stepped/non-rectangular rooms),
    # while keeping rectilinear mode for clear rectangle traces.
    auto_prefer_poly = False
    if rect_out is not None:
        dbg = rect_out.get("debug", {})
        x_count = int(dbg.get("x_cluster_count", len(dbg.get("x_clusters", []))))
        y_count = int(dbg.get("y_cluster_count", len(dbg.get("y_clusters", []))))
        rect_is_simple_box = (x_count in (2, 4)) and (y_count in (2, 4))
        if (not rect_is_simple_box) and (x_count >= 3 or y_count >= 3):
            auto_prefer_poly = True

    if poly_out is not None and (prefer_poly or auto_prefer_poly or rect_out is None):
        poly_out.setdefault("debug", {})
        poly_out["debug"]["rectilinear_error"] = rect_err
        poly_out["debug"]["fallback_wall_from_unclassified"] = bool(fallback)
        poly_out["debug"]["auto_prefer_polygon"] = auto_prefer_poly
        return poly_out
    if rect_out is not None:
        rect_out.setdefault("debug", {})
        rect_out["debug"]["polygon_error"] = poly_err
        rect_out["debug"]["fallback_wall_from_unclassified"] = bool(fallback)
        return rect_out

    raise ValueError("Wall recognition failed: rectilinear={} ; polygon={}".format(rect_err or "n/a", poly_err or "n/a"))
