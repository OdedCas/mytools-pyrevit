# -*- coding: utf-8 -*-
"""Generic CAD recognition for irregular line-based DWG plans."""

import math


def _abs(v):
    return v if v >= 0.0 else -v


def _line_len(ln):
    dx = float(ln["x2"]) - float(ln["x1"])
    dy = float(ln["y2"]) - float(ln["y1"])
    return math.sqrt((dx * dx) + (dy * dy))


def _line_mid(ln):
    return (
        (float(ln["x1"]) + float(ln["x2"])) * 0.5,
        (float(ln["y1"]) + float(ln["y2"])) * 0.5,
    )


def _line_axis(ln):
    x1 = float(ln["x1"])
    y1 = float(ln["y1"])
    x2 = float(ln["x2"])
    y2 = float(ln["y2"])
    dx = x2 - x1
    dy = y2 - y1
    ln_len = math.sqrt((dx * dx) + (dy * dy))
    if ln_len <= 1.0e-9:
        return None
    return (x1, y1, x2, y2, dx / ln_len, dy / ln_len, ln_len)


def _line_axis_angle_deg(ln):
    ax = _line_axis(ln)
    if ax is None:
        return 0.0
    return _angle_of(ax[4], ax[5])


def _project_on_axis(x, y, ux, uy):
    return (x * ux) + (y * uy)


def _line_overlap_ratio(ln_a, ln_b, ux, uy):
    a1 = _project_on_axis(float(ln_a["x1"]), float(ln_a["y1"]), ux, uy)
    a2 = _project_on_axis(float(ln_a["x2"]), float(ln_a["y2"]), ux, uy)
    b1 = _project_on_axis(float(ln_b["x1"]), float(ln_b["y1"]), ux, uy)
    b2 = _project_on_axis(float(ln_b["x2"]), float(ln_b["y2"]), ux, uy)

    amin = min(a1, a2)
    amax = max(a1, a2)
    bmin = min(b1, b2)
    bmax = max(b1, b2)

    overlap = min(amax, bmax) - max(amin, bmin)
    if overlap <= 0.0:
        return 0.0, 0.0

    la = max(1.0e-9, _line_len(ln_a))
    lb = max(1.0e-9, _line_len(ln_b))
    ratio = overlap / min(la, lb)
    return overlap, ratio


def _centroid_of_line_mids(lines):
    if not lines:
        return (0.0, 0.0)
    sx = 0.0
    sy = 0.0
    for ln in lines:
        mx, my = _line_mid(ln)
        sx += mx
        sy += my
    n = float(len(lines))
    return (sx / n, sy / n)


def _collapse_double_wall_lines(lines, cfg):
    lines = list(lines or [])
    if len(lines) < 2:
        return lines, {
            "input_count": len(lines),
            "output_count": len(lines),
            "paired_count": 0,
            "estimated_wall_thickness_cm": None,
        }

    default_wall = float((cfg or {}).get("default_wall_thickness_cm", 20.0))
    min_dist = float((cfg or {}).get("double_wall_pair_min_cm", max(4.0, default_wall * 0.25)))
    max_dist = float((cfg or {}).get("double_wall_pair_max_cm", max(45.0, default_wall * 2.5)))
    angle_tol = float((cfg or {}).get("double_wall_pair_angle_deg", 8.0))
    min_overlap_cm = float((cfg or {}).get("double_wall_pair_min_overlap_cm", 35.0))
    min_overlap_ratio = float((cfg or {}).get("double_wall_pair_overlap_ratio", 0.60))

    mids_centroid = _centroid_of_line_mids(lines)

    candidates = []
    for i in range(len(lines)):
        li = lines[i]
        axi = _line_axis(li)
        if axi is None:
            continue
        ai = _line_axis_angle_deg(li)

        for j in range(i + 1, len(lines)):
            lj = lines[j]
            axj = _line_axis(lj)
            if axj is None:
                continue
            aj = _line_axis_angle_deg(lj)

            ad = _angle_delta(ai, aj)
            if ad > 90.0:
                ad = 180.0 - ad
            if ad > angle_tol:
                continue

            mj = _line_mid(lj)
            d, _ = _dist_pt_seg(mj[0], mj[1], float(li["x1"]), float(li["y1"]), float(li["x2"]), float(li["y2"]))
            if d < min_dist or d > max_dist:
                continue

            overlap_cm, overlap_ratio = _line_overlap_ratio(li, lj, axi[4], axi[5])
            if overlap_cm < min_overlap_cm or overlap_ratio < min_overlap_ratio:
                continue

            score = (abs(d - default_wall) / max(default_wall, 1.0)) + (ad / max(angle_tol, 1.0)) + (1.0 - overlap_ratio)
            candidates.append({
                "i": i,
                "j": j,
                "dist": d,
                "score": score,
            })

    if not candidates:
        return lines, {
            "input_count": len(lines),
            "output_count": len(lines),
            "paired_count": 0,
            "estimated_wall_thickness_cm": None,
        }

    candidates.sort(key=lambda x: x["score"])
    used = set()
    selected_pairs = []
    pair_dists = []

    for c in candidates:
        i = c["i"]
        j = c["j"]
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        selected_pairs.append((i, j))
        pair_dists.append(float(c["dist"]))

    out = []
    used_in_pairs = set()
    for i, j in selected_pairs:
        used_in_pairs.add(i)
        used_in_pairs.add(j)
        li = lines[i]
        lj = lines[j]
        mi = _line_mid(li)
        mj = _line_mid(lj)

        dci = math.sqrt(((mi[0] - mids_centroid[0]) ** 2) + ((mi[1] - mids_centroid[1]) ** 2))
        dcj = math.sqrt(((mj[0] - mids_centroid[0]) ** 2) + ((mj[1] - mids_centroid[1]) ** 2))
        # Keep the inner trace (closer to room centroid) and drop its parallel outer mate.
        keep = li if dci <= dcj else lj
        out.append(keep)

    for idx, ln in enumerate(lines):
        if idx in used_in_pairs:
            continue
        out.append(ln)

    est = None
    if pair_dists:
        ds = sorted(pair_dists)
        m = len(ds) // 2
        if len(ds) % 2 == 1:
            est = ds[m]
        else:
            est = (ds[m - 1] + ds[m]) * 0.5

    return out, {
        "input_count": len(lines),
        "output_count": len(out),
        "paired_count": len(selected_pairs),
        "estimated_wall_thickness_cm": est,
    }


def _line_key(ln):
    p1 = (round(float(ln["x1"]), 4), round(float(ln["y1"]), 4))
    p2 = (round(float(ln["x2"]), 4), round(float(ln["y2"]), 4))
    if p1 <= p2:
        return (p1, p2)
    return (p2, p1)


def _merge_unique_lines(base, extra):
    out = list(base or [])
    seen = set()
    for ln in out:
        seen.add(_line_key(ln))
    for ln in (extra or []):
        k = _line_key(ln)
        if k in seen:
            continue
        seen.add(k)
        out.append(ln)
    return out


def _snap_pt(x, y, tol_cm):
    if tol_cm <= 1.0e-9:
        return (float(x), float(y))
    return (
        round(float(x) / tol_cm) * tol_cm,
        round(float(y) / tol_cm) * tol_cm,
    )


def _angle_of(vx, vy):
    return math.degrees(math.atan2(vy, vx))


def _angle_delta(a, b):
    d = _abs((a - b) % 360.0)
    if d > 180.0:
        d = 360.0 - d
    return d


def _build_graph(lines, snap_cm, min_len_cm):
    node_ids = {}
    nodes = []
    segs = []
    seen = set()

    def get_id(pt):
        nid = node_ids.get(pt)
        if nid is not None:
            return nid
        nid = len(nodes)
        node_ids[pt] = nid
        nodes.append(pt)
        return nid

    for ln in (lines or []):
        if _line_len(ln) < min_len_cm:
            continue
        p1 = _snap_pt(ln["x1"], ln["y1"], snap_cm)
        p2 = _snap_pt(ln["x2"], ln["y2"], snap_cm)
        if p1 == p2:
            continue
        key = (p1, p2) if p1 <= p2 else (p2, p1)
        if key in seen:
            continue
        seen.add(key)
        a = get_id(p1)
        b = get_id(p2)
        segs.append({"a": a, "b": b, "bridged": False})

    return nodes, segs


def _adj_from_segs(segs, n_nodes):
    adj = {}
    deg = [0] * n_nodes
    for i in range(n_nodes):
        adj[i] = set()
    for s in (segs or []):
        a = int(s["a"])
        b = int(s["b"])
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
        deg[a] += 1
        deg[b] += 1
    return adj, deg


def _endpoint_dir(node_id, adj, nodes):
    neigh = list(adj.get(node_id, []))
    if not neigh:
        return None
    n = neigh[0]
    x1, y1 = nodes[node_id]
    x2, y2 = nodes[n]
    vx = x2 - x1
    vy = y2 - y1
    ln = math.sqrt((vx * vx) + (vy * vy))
    if ln <= 1.0e-9:
        return None
    return (vx / ln, vy / ln)


def _seg_exists(a, b, segs):
    x = a if a <= b else b
    y = b if a <= b else a
    for s in (segs or []):
        p = int(s["a"])
        q = int(s["b"])
        if p > q:
            p, q = q, p
        if p == x and q == y:
            return True
    return False


def _bridge_open_gaps(nodes, segs, cfg):
    snap_cm = float(cfg.get("endpoint_snap_mm", 4.0)) / 10.0
    min_gap = float(cfg.get("wall_gap_bridge_min_cm", 4.0))
    max_gap = float(cfg.get("wall_gap_bridge_max_cm", 220.0))
    angle_tol = float(cfg.get("wall_gap_bridge_angle_deg", 16.0))

    adj, deg = _adj_from_segs(segs, len(nodes))
    ends = []
    for i, d in enumerate(deg):
        if d == 1:
            ends.append(i)

    used = set()
    added = []
    for i in range(len(ends)):
        a = ends[i]
        if a in used:
            continue
        ax, ay = nodes[a]
        adir = _endpoint_dir(a, adj, nodes)

        best = None
        best_dist = None
        for j in range(i + 1, len(ends)):
            b = ends[j]
            if b in used:
                continue
            if a == b:
                continue
            if _seg_exists(a, b, segs) or _seg_exists(a, b, added):
                continue
            bx, by = nodes[b]
            dx = bx - ax
            dy = by - ay
            dist = math.sqrt((dx * dx) + (dy * dy))
            if dist < min_gap or dist > max_gap:
                continue

            bdir = _endpoint_dir(b, adj, nodes)
            if adir is not None:
                aang = _angle_of(adir[0], adir[1])
                gang = _angle_of(dx, dy)
                if min(_angle_delta(aang, gang), _angle_delta(aang + 180.0, gang)) > angle_tol:
                    continue
            if bdir is not None:
                bang = _angle_of(bdir[0], bdir[1])
                g2ang = _angle_of(-dx, -dy)
                if min(_angle_delta(bang, g2ang), _angle_delta(bang + 180.0, g2ang)) > angle_tol:
                    continue

            if best is None or dist < best_dist:
                best = b
                best_dist = dist

        if best is not None:
            used.add(a)
            used.add(best)
            added.append({"a": a, "b": best, "bridged": True, "width_cm": best_dist})

    if not added:
        return segs, []

    all_segs = list(segs) + added
    # Re-snap sanity for nearly coincident endpoints after bridging.
    if snap_cm > 0.0:
        pass
    return all_segs, added


def _find_room_cycle_from_lines(wall_lines, classified, cfg, relax_gap=False):
    snap_cm = float(cfg.get("endpoint_snap_mm", 4.0)) / 10.0
    min_len_cm = float(cfg.get("min_segment_mm", 8.0)) / 10.0

    nodes, segs = _build_graph(wall_lines, snap_cm, min_len_cm)
    if len(nodes) < 3 or len(segs) < 3:
        return None

    bridge_cfg = dict(cfg or {})
    if relax_gap:
        curr = float(bridge_cfg.get("wall_gap_bridge_max_cm", 220.0))
        bridge_cfg["wall_gap_bridge_max_cm"] = max(curr, float(cfg.get("fallback_wall_gap_bridge_max_cm", 700.0)))
        bridge_cfg["wall_gap_bridge_angle_deg"] = max(float(bridge_cfg.get("wall_gap_bridge_angle_deg", 16.0)), 22.0)

    segs2, bridges = _bridge_open_gaps(nodes, segs, bridge_cfg)
    adj, _ = _adj_from_segs(segs2, len(nodes))

    cycles = _find_cycles(
        nodes,
        adj,
        int(cfg.get("polygon_max_cycle_len", 80)),
        int(cfg.get("polygon_max_cycles", 600)),
    )
    picked = _pick_room_cycle(cycles, nodes, classified, cfg)
    if not picked:
        return None

    return {
        "picked": picked,
        "nodes": nodes,
        "segs": segs,
        "bridges": bridges,
        "cycles": cycles,
    }


def _canonical_cycle(path):
    n = len(path)
    if n == 0:
        return tuple()

    def rot(arr, k):
        return arr[k:] + arr[:k]

    candidates = []
    for seq in (path, list(reversed(path))):
        m = min(seq)
        for i in range(n):
            if seq[i] == m:
                candidates.append(tuple(rot(seq, i)))
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

    out = []
    for c in found:
        out.append(list(c))
    return out


def _poly_area_signed(points):
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += (x1 * y2) - (x2 * y1)
    return 0.5 * s


def _poly_centroid(points):
    n = len(points)
    if n == 0:
        return (0.0, 0.0)
    sx = 0.0
    sy = 0.0
    for p in points:
        sx += p[0]
        sy += p[1]
    return (sx / float(n), sy / float(n))


def _inner_probe_point(poly, centroid):
    if not poly:
        return centroid
    vx, vy = poly[0]
    cx, cy = centroid
    # Pull slightly from a vertex toward centroid to avoid boundary ambiguity.
    return ((vx * 0.85) + (cx * 0.15), (vy * 0.85) + (cy * 0.15))


def _point_in_poly(px, py, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        inter = ((yi > py) != (yj > py))
        if inter:
            den = (yj - yi)
            if _abs(den) < 1e-12:
                den = 1e-12
            x_cross = (xj - xi) * (py - yi) / den + xi
            if px < x_cross:
                inside = not inside
        j = i
    return inside


def _dist_pt_seg(px, py, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    vv = (vx * vx) + (vy * vy)
    if vv <= 1e-12:
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
    if ln <= 1.0e-12:
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
        if ln <= 1.0e-9:
            continue
        out.append({
            "idx": i,
            "a": a,
            "b": b,
            "len": ln,
            "ang": math.degrees(math.atan2(dy, dx)),
        })
    return out


def _angle_delta_axis(a, b):
    d = _angle_delta(a, b)
    if d > 90.0:
        d = 180.0 - d
    return d


def _dominant_axes_from_edges(edges):
    if not edges:
        return (1.0, 0.0), (0.0, 1.0), 0.0
    longest = None
    for e in edges:
        if longest is None or float(e.get("len", 0.0)) > float(longest.get("len", 0.0)):
            longest = e
    ang = float(longest.get("ang", 0.0))
    rad = math.radians(ang)
    ux = math.cos(rad)
    uy = math.sin(rad)
    vx = -uy
    vy = ux
    return (ux, uy), (vx, vy), ang


def _poly_span_on_axis(poly, ux, uy):
    vals = []
    for p in (poly or []):
        vals.append((float(p[0]) * ux) + (float(p[1]) * uy))
    if not vals:
        return 0.0, 0.0, 0.0
    mn = min(vals)
    mx = max(vals)
    return mn, mx, (mx - mn)


def _line_len_ang_mid(ln):
    x1 = float(ln.get("x1", 0.0))
    y1 = float(ln.get("y1", 0.0))
    x2 = float(ln.get("x2", 0.0))
    y2 = float(ln.get("y2", 0.0))
    dx = x2 - x1
    dy = y2 - y1
    ln_len = math.sqrt((dx * dx) + (dy * dy))
    ang = _angle_of(dx, dy)
    mx = (x1 + x2) * 0.5
    my = (y1 + y2) * 0.5
    return ln_len, ang, mx, my


def _pick_dimension_span_hints(poly, edges, dim_lines, cfg):
    out = {
        "target_u_cm": None,
        "target_v_cm": None,
        "current_u_cm": None,
        "current_v_cm": None,
        "axis_u": (1.0, 0.0),
        "axis_v": (0.0, 1.0),
        "used_lines": 0,
    }

    if not poly or not edges or (not dim_lines):
        return out

    angle_tol = float((cfg or {}).get("dimension_hint_angle_deg", 18.0))
    host_tol = float((cfg or {}).get("dimension_hint_host_distance_cm", 180.0))
    span_ratio_min = float((cfg or {}).get("dimension_span_ratio_min", 0.75))
    span_ratio_max = float((cfg or {}).get("dimension_span_ratio_max", 2.5))

    axis_u, axis_v, ang_u = _dominant_axes_from_edges(edges)
    ang_v = _angle_of(axis_v[0], axis_v[1])
    out["axis_u"] = axis_u
    out["axis_v"] = axis_v

    _, _, span_u = _poly_span_on_axis(poly, axis_u[0], axis_u[1])
    _, _, span_v = _poly_span_on_axis(poly, axis_v[0], axis_v[1])
    out["current_u_cm"] = span_u
    out["current_v_cm"] = span_v

    cands_u = []
    cands_v = []

    for ln in (dim_lines or []):
        ln_len, ln_ang, mx, my = _line_len_ang_mid(ln)
        if ln_len <= 1.0:
            continue

        du = _angle_delta_axis(ln_ang, ang_u)
        dv = _angle_delta_axis(ln_ang, ang_v)
        axis_kind = None
        if du <= angle_tol or dv <= angle_tol:
            axis_kind = "u" if du <= dv else "v"
        else:
            continue

        best_edge_dist = None
        for e in (edges or []):
            ea = float(e.get("ang", 0.0))
            if axis_kind == "u" and _angle_delta_axis(ea, ang_u) > angle_tol:
                continue
            if axis_kind == "v" and _angle_delta_axis(ea, ang_v) > angle_tol:
                continue
            d, _ = _dist_pt_seg(mx, my, float(e["a"][0]), float(e["a"][1]), float(e["b"][0]), float(e["b"][1]))
            if best_edge_dist is None or d < best_edge_dist:
                best_edge_dist = d

        if best_edge_dist is None or best_edge_dist > host_tol:
            continue

        span_ref = span_u if axis_kind == "u" else span_v
        if span_ref <= 1.0e-6:
            continue
        ratio_to_span = ln_len / max(span_ref, 1.0)
        if ratio_to_span < span_ratio_min or ratio_to_span > span_ratio_max:
            continue

        rel = abs(ln_len - span_ref) / max(span_ref, 1.0)
        score = rel + (best_edge_dist / max(host_tol, 1.0)) * 0.25
        rec = {"length_cm": ln_len, "score": score, "edge_dist": best_edge_dist}
        if axis_kind == "u":
            cands_u.append(rec)
        else:
            cands_v.append(rec)

    if cands_u:
        cands_u.sort(key=lambda x: x["score"])
        out["target_u_cm"] = float(cands_u[0]["length_cm"])
        out["used_lines"] += 1
    if cands_v:
        cands_v.sort(key=lambda x: x["score"])
        out["target_v_cm"] = float(cands_v[0]["length_cm"])
        out["used_lines"] += 1

    return out


def _scale_poly_on_axes(poly, axis_u, axis_v, scale_u, scale_v):
    if not poly:
        return []
    cx, cy = _poly_centroid(poly)
    ux, uy = axis_u
    vx, vy = axis_v
    out = []
    for p in poly:
        px = float(p[0]) - cx
        py = float(p[1]) - cy
        du = (px * ux) + (py * uy)
        dv = (px * vx) + (py * vy)
        nx = cx + ((du * scale_u) * ux) + ((dv * scale_v) * vx)
        ny = cy + ((du * scale_u) * uy) + ((dv * scale_v) * vy)
        out.append((nx, ny))
    return out


def _scale_openings_by_edge_ratio(openings, edges_before, edges_after):
    map_before = {}
    map_after = {}
    for e in (edges_before or []):
        map_before[int(e["idx"])] = float(e["len"])
    for e in (edges_after or []):
        map_after[int(e["idx"])] = float(e["len"])

    out = []
    for op in (openings or []):
        c = dict(op)
        idx = int(c.get("host_edge", -1))
        l0 = map_before.get(idx)
        l1 = map_after.get(idx)
        if l0 is None or l1 is None or l0 <= 1.0e-9:
            out.append(c)
            continue

        ratio = l1 / l0
        s = float(c.get("start_cm", 0.0)) * ratio
        t = float(c.get("end_cm", s)) * ratio
        if t < s:
            s, t = t, s
        s = max(0.0, min(l1, s))
        t = max(s, min(l1, t))

        c["start_cm"] = s
        c["end_cm"] = t
        c["width_cm"] = max(1.0, t - s)
        out.append(c)

    return out


def _apply_dimension_hints_to_openings(openings, edges, dim_lines, cfg):
    if not openings or not edges or not dim_lines:
        return openings, 0

    host_tol = float((cfg or {}).get("dimension_hint_host_distance_cm", 180.0))
    angle_tol = float((cfg or {}).get("dimension_hint_angle_deg", 18.0))
    center_tol = float((cfg or {}).get("dimension_hint_center_tol_cm", 120.0))

    edge_map = {}
    for e in edges:
        edge_map[int(e["idx"])] = e

    out = []
    used = 0
    for op in (openings or []):
        c = dict(op)
        idx = int(c.get("host_edge", -1))
        e = edge_map.get(idx)
        if e is None:
            out.append(c)
            continue

        ex1 = float(e["a"][0])
        ey1 = float(e["a"][1])
        ex2 = float(e["b"][0])
        ey2 = float(e["b"][1])
        eang = float(e.get("ang", 0.0))
        elen = float(e.get("len", 0.0))

        s0 = float(c.get("start_cm", 0.0))
        t0 = float(c.get("end_cm", s0))
        if t0 < s0:
            s0, t0 = t0, s0
        center0 = (s0 + t0) * 0.5

        best = None
        for ln in (dim_lines or []):
            ln_len, ln_ang, mx, my = _line_len_ang_mid(ln)
            if ln_len <= 1.0:
                continue
            if _angle_delta_axis(ln_ang, eang) > angle_tol:
                continue
            d, _ = _dist_pt_seg(mx, my, ex1, ey1, ex2, ey2)
            if d > host_tol:
                continue

            tc = _project_t_cm(mx, my, ex1, ey1, ex2, ey2)
            if tc < -center_tol or tc > (elen + center_tol):
                continue

            curr_w = max(1.0, float(c.get("width_cm", 1.0)))
            if ln_len < (curr_w * 0.5) or ln_len > (curr_w * 2.2):
                continue

            if str(c.get("type", "")).lower() == "door":
                if ln_len < 50.0 or ln_len > 220.0:
                    continue
            elif str(c.get("type", "")).lower() == "window":
                if ln_len < 30.0 or ln_len > 420.0:
                    continue

            score = (abs(tc - center0) / max(center_tol, 1.0)) + (d / max(host_tol, 1.0))
            if best is None or score < best["score"]:
                best = {"length_cm": ln_len, "score": score}

        if best is not None:
            new_w = float(best["length_cm"])
            center = center0
            ns = max(0.0, center - (new_w * 0.5))
            nt = min(elen, center + (new_w * 0.5))
            if (nt - ns) < 1.0:
                ns = max(0.0, min(elen - 1.0, center - 0.5))
                nt = min(elen, ns + 1.0)
            c["start_cm"] = ns
            c["end_cm"] = nt
            c["width_cm"] = max(1.0, nt - ns)
            c["confidence"] = max(float(c.get("confidence", 0.0)), 0.85)
            c["from_dimension_hint"] = True
            used += 1

        out.append(c)

    return out, used


def _opening_from_arc(arcs, edges, host_tol_cm, min_w, max_w):
    out = []
    for a in (arcs or []):
        cx = float(a.get("cx", 0.0))
        cy = float(a.get("cy", 0.0))
        w = 2.0 * float(a.get("r", 0.0))
        if w < min_w or w > max_w:
            continue

        best = None
        for e in edges:
            d, _ = _dist_pt_seg(cx, cy, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            if d > host_tol_cm:
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
                "confidence": max(0.0, 1.0 - (d / max(host_tol_cm, 1e-6))),
            }
            if best is None or cand["confidence"] > best["confidence"]:
                best = cand
        if best:
            out.append(best)
    return out


def _opening_from_lines(lines, edges, kind, host_tol_cm, min_w, max_w):
    out = []
    for ln in (lines or []):
        x1 = float(ln["x1"])
        y1 = float(ln["y1"])
        x2 = float(ln["x2"])
        y2 = float(ln["y2"])
        w = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
        if w < min_w or w > max_w:
            continue

        mx = (x1 + x2) * 0.5
        my = (y1 + y2) * 0.5

        best = None
        for e in edges:
            d, _ = _dist_pt_seg(mx, my, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            if d > host_tol_cm:
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
            if best is None or cand["confidence"] > best["confidence"]:
                best = cand
        if best:
            out.append(best)
    return out


def _opening_from_bridges(bridges, nodes, edges):
    out = []
    for b in (bridges or []):
        a = nodes[int(b["a"])]
        c = nodes[int(b["b"])]
        x1, y1 = a
        x2, y2 = c
        w = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
        if w < 45.0 or w > 280.0:
            continue
        kind = None
        if w <= 130.0:
            kind = "door"
        else:
            kind = "window"

        mx = (x1 + x2) * 0.5
        my = (y1 + y2) * 0.5
        best = None
        for e in edges:
            d, _ = _dist_pt_seg(mx, my, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            if d > 60.0:
                continue
            t1 = _project_t_cm(x1, y1, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            t2 = _project_t_cm(x2, y2, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
            s = max(0.0, min(t1, t2))
            t = min(e["len"], max(t1, t2))
            cand = {
                "type": kind,
                "host_edge": e["idx"],
                "start_cm": s,
                "end_cm": t,
                "width_cm": max(1.0, t - s),
                "confidence": 0.55,
            }
            if best is None or cand["width_cm"] > best["width_cm"]:
                best = cand
        if best:
            out.append(best)
    return out


def _merge_openings(cands, min_sep_cm):
    groups = {}
    for c in (cands or []):
        k = (c.get("type"), int(c.get("host_edge", -1)))
        groups.setdefault(k, []).append(c)

    out = []
    for _, arr in groups.items():
        arr = sorted(arr, key=lambda x: (x["start_cm"], -x.get("confidence", 0.0)))
        merged = []
        for c in arr:
            if not merged:
                merged.append(dict(c))
                continue
            p = merged[-1]
            if c["start_cm"] <= (p["end_cm"] + min_sep_cm):
                p["start_cm"] = min(p["start_cm"], c["start_cm"])
                p["end_cm"] = max(p["end_cm"], c["end_cm"])
                p["width_cm"] = p["end_cm"] - p["start_cm"]
                p["confidence"] = max(p.get("confidence", 0.0), c.get("confidence", 0.0))
            else:
                merged.append(dict(c))
        out.extend(merged)

    return sorted(out, key=lambda x: (x.get("host_edge", -1), x.get("start_cm", 0.0), x.get("type", "")))


def _opening_from_edge_gaps(edges, support_lines, host_tol_cm, min_w, max_w):
    out = []
    for e in (edges or []):
        intervals = []
        for ln in (support_lines or []):
            x1 = float(ln["x1"])
            y1 = float(ln["y1"])
            x2 = float(ln["x2"])
            y2 = float(ln["y2"])

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
                    kind = "door" if gw <= 130.0 else "window"
                    out.append({
                        "type": kind,
                        "host_edge": e["idx"],
                        "start_cm": cursor,
                        "end_cm": s,
                        "width_cm": gw,
                        "confidence": 0.45,
                        "from_gap": True,
                    })
            cursor = max(cursor, t)

        if cursor < e["len"]:
            gw = e["len"] - cursor
            if gw >= min_w and gw <= max_w:
                kind = "door" if gw <= 130.0 else "window"
                out.append({
                    "type": kind,
                    "host_edge": e["idx"],
                    "start_cm": cursor,
                    "end_cm": e["len"],
                    "width_cm": gw,
                    "confidence": 0.45,
                    "from_gap": True,
                })

    return out


def _pick_room_cycle(cycles, nodes, classified, cfg):
    min_area = float(cfg.get("polygon_min_area_cm2", 6000.0))

    door_lines = classified.get("door_lines") or []
    win_lines = classified.get("window_lines") or []
    all_opening_lines = list(door_lines) + list(win_lines)
    all_opening_arcs = list(classified.get("door_arcs") or []) + list(classified.get("window_arcs") or [])

    scored = []
    for cyc in cycles:
        pts = [nodes[i] for i in cyc]
        area = _abs(_poly_area_signed(pts))
        if area < min_area:
            continue
        edges = _edge_list(pts)
        if len(edges) < 3:
            continue

        support = 0
        for ln in all_opening_lines:
            mx = (float(ln["x1"]) + float(ln["x2"])) * 0.5
            my = (float(ln["y1"]) + float(ln["y2"])) * 0.5
            dmin = 1e20
            for e in edges:
                d, _ = _dist_pt_seg(mx, my, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
                if d < dmin:
                    dmin = d
            if dmin <= 80.0:
                support += 1

        for a in all_opening_arcs:
            mx = float(a.get("cx", 0.0))
            my = float(a.get("cy", 0.0))
            dmin = 1e20
            for e in edges:
                d, _ = _dist_pt_seg(mx, my, e["a"][0], e["a"][1], e["b"][0], e["b"][1])
                if d < dmin:
                    dmin = d
            if dmin <= 100.0:
                support += 1

        scored.append({
            "poly": pts,
            "area": area,
            "support": support,
            "centroid": _poly_centroid(pts),
            "probe": _inner_probe_point(pts, _poly_centroid(pts)),
        })

    if not scored:
        return None

    # Determine nesting: inner room loops are usually contained by an outer wall loop.
    for i in range(len(scored)):
        contains_count = 0
        px, py = scored[i]["probe"]
        for j in range(len(scored)):
            if i == j:
                continue
            # A polygon can only be contained by a larger-area polygon.
            if scored[j]["area"] <= scored[i]["area"]:
                continue
            if _point_in_poly(px, py, scored[j]["poly"]):
                contains_count += 1
        scored[i]["contains_count"] = contains_count

    # For nested double-line walls, inner room boundary is typically a contained loop.
    nested = [s for s in scored if s.get("contains_count", 0) > 0]
    if nested:
        max_support_nested = max([s["support"] for s in nested])
        nested = [s for s in nested if s["support"] == max_support_nested]
        nested.sort(key=lambda x: x["area"], reverse=True)
        return nested[0]

    # Prefer loops that are supported by opening geometry.
    max_support = max([s["support"] for s in scored])
    if max_support > 0:
        cands = [s for s in scored if s["support"] == max_support]
        cands.sort(key=lambda x: x["area"], reverse=True)
        return cands[0]

    # Otherwise pick the largest room-like loop.
    scored.sort(key=lambda x: x["area"], reverse=True)
    return scored[0]


def recognize_topology(classified, cfg):
    cfg = cfg or {}

    wall_lines_raw = _merge_unique_lines(
        list(classified.get("wall_lines") or []),
        list(classified.get("unclassified_lines") or []),
    )
    wall_lines, collapse_dbg = _collapse_double_wall_lines(wall_lines_raw, cfg)
    if not wall_lines:
        raise ValueError("Wall recognition failed: no line candidates")

    solve_mode = "collapsed"
    solve_out = _find_room_cycle_from_lines(wall_lines, classified, cfg, relax_gap=False)

    if solve_out is None and wall_lines_raw:
        solve_mode = "raw_fallback"
        solve_out = _find_room_cycle_from_lines(wall_lines_raw, classified, cfg, relax_gap=False)
        if solve_out is not None:
            wall_lines = list(wall_lines_raw)

    if solve_out is None and wall_lines_raw:
        solve_mode = "raw_relaxed_bridge"
        solve_out = _find_room_cycle_from_lines(wall_lines_raw, classified, cfg, relax_gap=True)
        if solve_out is not None:
            wall_lines = list(wall_lines_raw)

    if solve_out is None:
        raise ValueError("Wall recognition failed: no closed room loop")

    picked = solve_out["picked"]
    nodes = solve_out["nodes"]
    segs = solve_out["segs"]
    bridges = solve_out["bridges"]
    cycles = solve_out["cycles"]

    poly_raw = picked["poly"]
    edges_raw = _edge_list(poly_raw)

    default_wall = float(cfg.get("default_wall_thickness_cm", 20.0))
    estimated_wall = collapse_dbg.get("estimated_wall_thickness_cm")
    wall_thickness_cm = float(estimated_wall) if estimated_wall is not None else default_wall

    host_tol_cm = float(cfg.get("opening_host_distance_mm", 70.0)) / 10.0
    host_tol_cm = max(host_tol_cm, (wall_thickness_cm * 0.6) + 2.0)
    openings = []
    openings.extend(_opening_from_arc(classified.get("door_arcs") or [], edges_raw, host_tol_cm * 2.0, 60.0, 180.0))
    openings.extend(_opening_from_lines(classified.get("door_lines") or [], edges_raw, "door", host_tol_cm * 1.5, 60.0, 180.0))
    openings.extend(_opening_from_lines(classified.get("window_lines") or [], edges_raw, "window", host_tol_cm * 1.5, 40.0, 350.0))
    openings.extend(_opening_from_bridges(bridges, nodes, edges_raw))
    support_lines = _merge_unique_lines(
        list(classified.get("wall_lines") or []),
        list(classified.get("unclassified_lines") or []),
    )
    openings.extend(
        _opening_from_edge_gaps(
            edges_raw,
            support_lines,
            host_tol_cm=max(host_tol_cm * 1.5, 20.0),
            min_w=float(cfg.get("opening_gap_min_cm", 55.0)),
            max_w=float(cfg.get("opening_gap_max_cm", 260.0)),
        )
    )
    openings = _merge_openings(openings, 2.0)

    dim_lines = list(classified.get("dimension_lines") or [])
    dim_hints = _pick_dimension_span_hints(poly_raw, edges_raw, dim_lines, cfg)
    scale_u = 1.0
    scale_v = 1.0
    tu = dim_hints.get("target_u_cm")
    tv = dim_hints.get("target_v_cm")
    cu = dim_hints.get("current_u_cm")
    cv = dim_hints.get("current_v_cm")
    scale_min = float(cfg.get("dimension_scale_min", 0.5))
    scale_max = float(cfg.get("dimension_scale_max", 2.0))
    if tu is not None and cu is not None and cu > 1.0e-6:
        scale_u = max(scale_min, min(scale_max, float(tu) / float(cu)))
    if tv is not None and cv is not None and cv > 1.0e-6:
        scale_v = max(scale_min, min(scale_max, float(tv) / float(cv)))

    poly = poly_raw
    edges = edges_raw
    if abs(scale_u - 1.0) > 1.0e-6 or abs(scale_v - 1.0) > 1.0e-6:
        poly = _scale_poly_on_axes(poly_raw, dim_hints.get("axis_u", (1.0, 0.0)), dim_hints.get("axis_v", (0.0, 1.0)), scale_u, scale_v)
        edges = _edge_list(poly)
        openings = _scale_openings_by_edge_ratio(openings, edges_raw, edges)

    openings, opening_dim_used = _apply_dimension_hints_to_openings(openings, edges, dim_lines, cfg)
    openings = _merge_openings(openings, 2.0)

    minx = min([p[0] for p in poly])
    miny = min([p[1] for p in poly])
    maxx = max([p[0] for p in poly])
    maxy = max([p[1] for p in poly])
    room_w = maxx - minx
    room_h = maxy - miny

    default_door_w = float(cfg.get("default_door_width_cm", 100.0))
    default_door_h = float(cfg.get("default_door_height_cm", 210.0))
    default_win_w = float(cfg.get("default_window_width_cm", 100.0))
    default_win_h = float(cfg.get("default_window_height_cm", 100.0))
    default_win_sill = float(cfg.get("default_window_sill_cm", 105.0))

    first_door = None
    first_window = None
    for o in openings:
        if first_door is None and o.get("type") == "door":
            first_door = o
        if first_window is None and o.get("type") == "window":
            first_window = o

    door_width = default_door_w
    door_left = 90.0
    if first_door is not None:
        door_width = max(1.0, float(first_door.get("width_cm", default_door_w)))
        door_left = max(0.0, float(first_door.get("start_cm", 90.0)))

    window_width = default_win_w
    window_right = 30.0
    if first_window is not None:
        window_width = max(1.0, float(first_window.get("width_cm", default_win_w)))
        edge_len = 0.0
        for e in edges:
            if int(e["idx"]) == int(first_window.get("host_edge", -1)):
                edge_len = e["len"]
                break
        if edge_len > 0.0:
            window_right = max(0.0, edge_len - float(first_window.get("end_cm", edge_len)))

    wall_segments = []
    for e in edges:
        wall_segments.append({
            "start": [e["a"][0], e["a"][1]],
            "end": [e["b"][0], e["b"][1]],
            "length_cm": e["len"],
        })

    # Ensure deterministic defaults when no windows/doors are found.
    if not any([o for o in openings if o.get("type") == "window"]):
        # On longest edge, create a default 100cm window 30cm from one side.
        longest = None
        for e in edges:
            if longest is None or e["len"] > longest["len"]:
                longest = e
        if longest and longest["len"] > (default_win_w + 30.0):
            s = max(0.0, longest["len"] - 30.0 - default_win_w)
            openings.append({
                "type": "window",
                "host_edge": longest["idx"],
                "start_cm": s,
                "end_cm": s + default_win_w,
                "width_cm": default_win_w,
                "height_cm": default_win_h,
                "sill_cm": default_win_sill,
                "confidence": 0.2,
                "synthetic": True,
            })

    measurements_cm = {
        "room_width_cm": max(1.0, room_w),
        "room_height_cm": max(1.0, room_h),
        "wall_thickness_cm": wall_thickness_cm,
        "door_width_cm": door_width,
        "door_height_cm": default_door_h,
        "door_left_offset_cm": door_left,
        "window_width_cm": window_width,
        "window_height_cm": default_win_h,
        "window_right_offset_cm": window_right,
        "window_sill_cm": default_win_sill,
    }

    return {
        "room_polygon_cm": [[p[0], p[1]] for p in poly],
        "wall_segments_cm": wall_segments,
        "openings": openings,
        "measurements_cm": measurements_cm,
        "debug": {
            "mode": "polygon_v2",
            "solve_mode": solve_mode,
            "wall_line_candidates": len(wall_lines),
            "wall_line_candidates_raw": len(wall_lines_raw),
            "paired_wall_line_count": int(collapse_dbg.get("paired_count") or 0),
            "estimated_wall_thickness_cm": collapse_dbg.get("estimated_wall_thickness_cm"),
            "dimension_line_count": len(dim_lines),
            "dimension_span_hint_used_count": int(dim_hints.get("used_lines") or 0),
            "dimension_opening_hint_used_count": int(opening_dim_used),
            "dimension_target_u_cm": dim_hints.get("target_u_cm"),
            "dimension_target_v_cm": dim_hints.get("target_v_cm"),
            "dimension_scale_u": scale_u,
            "dimension_scale_v": scale_v,
            "graph_node_count": len(nodes),
            "graph_segment_count": len(segs),
            "bridged_count": len(bridges),
            "cycle_count": len(cycles),
            "picked_area_cm2": picked["area"],
            "picked_support": picked["support"],
        },
    }
