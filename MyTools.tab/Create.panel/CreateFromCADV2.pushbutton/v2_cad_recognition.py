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


def _merge_collinear_overlapping(lines, perp_tol=5.0, gap_tol=2.0):
    """Merge overlapping collinear lines into maximal segments.

    Groups lines by direction (H/V/diagonal) and perpendicular position.
    Lines in the same group that overlap or nearly touch are merged into
    one line spanning their combined extent.
    """
    if not lines or len(lines) < 2:
        return list(lines or [])

    # For each line, compute: axis unit vector, perpendicular offset, projection range
    entries = []
    for idx, ln in enumerate(lines):
        x1 = float(ln["x1"])
        y1 = float(ln["y1"])
        x2 = float(ln["x2"])
        y2 = float(ln["y2"])
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.5:
            entries.append(None)
            continue
        # Normalize direction: always point in positive-x direction
        # (or positive-y if vertical)
        ux = dx / length
        uy = dy / length
        if ux < -1e-9 or (abs(ux) < 1e-9 and uy < 0):
            ux, uy = -ux, -uy
        # Perpendicular offset from origin (signed distance)
        perp = -uy * x1 + ux * y1
        # Projection range along axis
        proj1 = ux * x1 + uy * y1
        proj2 = ux * x2 + uy * y2
        pmin = min(proj1, proj2)
        pmax = max(proj1, proj2)
        # Angle bucket (rounded to 1 degree)
        ang = round(math.degrees(math.atan2(uy, ux)))
        entries.append({
            "idx": idx, "ux": ux, "uy": uy,
            "perp": perp, "pmin": pmin, "pmax": pmax,
            "ang": ang, "ln": ln,
        })

    # Group by angle bucket (within 2 degrees)
    from collections import defaultdict
    angle_groups = defaultdict(list)
    for e in entries:
        if e is not None:
            angle_groups[e["ang"]].append(e)

    merged_out = []
    used = set()

    for ang_key, group in angle_groups.items():
        # Sort by perpendicular offset
        group.sort(key=lambda e: e["perp"])
        # Sub-group by perpendicular proximity
        subgroups = [[group[0]]]
        for k in range(1, len(group)):
            if abs(group[k]["perp"] - subgroups[-1][-1]["perp"]) <= perp_tol:
                subgroups[-1].append(group[k])
            else:
                subgroups.append([group[k]])

        for sg in subgroups:
            if len(sg) == 1:
                # No merge needed
                continue
            # Sort by projection start
            sg.sort(key=lambda e: e["pmin"])
            # Merge overlapping/touching intervals
            merged_intervals = []
            curr_min = sg[0]["pmin"]
            curr_max = sg[0]["pmax"]
            curr_members = [sg[0]]
            for k in range(1, len(sg)):
                if sg[k]["pmin"] <= curr_max + gap_tol:
                    curr_max = max(curr_max, sg[k]["pmax"])
                    curr_members.append(sg[k])
                else:
                    merged_intervals.append((curr_min, curr_max, curr_members))
                    curr_min = sg[k]["pmin"]
                    curr_max = sg[k]["pmax"]
                    curr_members = [sg[k]]
            merged_intervals.append((curr_min, curr_max, curr_members))

            for pmin, pmax, members in merged_intervals:
                if len(members) < 2:
                    continue
                # Mark originals as used, emit one merged line
                for m in members:
                    used.add(m["idx"])
                # Average the perpendicular offset
                avg_perp = sum(m["perp"] for m in members) / len(members)
                ux = members[0]["ux"]
                uy = members[0]["uy"]
                # Reconstruct endpoints from axis projection + perp offset
                # Point = proj * (ux, uy) + perp * (-uy, ux)  [since perp = -uy*x + ux*y]
                # Actually: x = ux*proj + (-uy)*perp, y = uy*proj + ux*perp
                nx = -uy  # perp direction x
                ny = ux   # perp direction y
                x1 = ux * pmin + nx * avg_perp
                y1 = uy * pmin + ny * avg_perp
                x2 = ux * pmax + nx * avg_perp
                y2 = uy * pmax + ny * avg_perp
                merged_out.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "layer": members[0]["ln"].get("layer", ""),
                    "type": "line",
                })

    # Add non-merged lines
    for idx, ln in enumerate(lines):
        if idx not in used:
            merged_out.append(ln)

    return merged_out


def _extend_to_intersections(lines, ext_tol=50.0):
    """Extend/trim centerlines to meet at intersections with perpendicular lines.

    For each pair of roughly-perpendicular lines whose theoretical intersection
    is close to both endpoints, extend both lines to the intersection point.
    This connects walls at corners and T-junctions.
    """
    if not lines or len(lines) < 2:
        return list(lines or [])

    # Parse lines into (x1,y1,x2,y2,ux,uy,length) tuples
    parsed = []
    for ln in lines:
        x1 = float(ln["x1"])
        y1 = float(ln["y1"])
        x2 = float(ln["x2"])
        y2 = float(ln["y2"])
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1.0:
            parsed.append(None)
            continue
        parsed.append((x1, y1, x2, y2, dx / length, dy / length, length))

    # For each line, track extension points at each end
    # end_extensions[i] = {"start": [(x,y),...], "end": [(x,y),...]}
    end_ext = [{"start": [], "end": []} for _ in lines]

    for i in range(len(lines)):
        if parsed[i] is None:
            continue
        x1i, y1i, x2i, y2i, uxi, uyi, leni = parsed[i]
        for j in range(i + 1, len(lines)):
            if parsed[j] is None:
                continue
            x1j, y1j, x2j, y2j, uxj, uyj, lenj = parsed[j]

            # Check roughly perpendicular (angle between 60-120 degrees)
            dot = abs(uxi * uxj + uyi * uyj)
            if dot > 0.5:  # cos(60°) = 0.5
                continue

            # Find intersection of infinite lines
            # Line i: P = (x1i,y1i) + t*(uxi,uyi)
            # Line j: P = (x1j,y1j) + s*(uxj,uyj)
            # Solve: x1i + t*uxi = x1j + s*uxj  and  y1i + t*uyi = y1j + s*uyj
            denom = uxi * uyj - uyi * uxj
            if abs(denom) < 1e-9:
                continue
            dx0 = x1j - x1i
            dy0 = y1j - y1i
            t = (dx0 * uyj - dy0 * uxj) / denom
            s = (dx0 * uyi - dy0 * uxi) / denom

            ix = x1i + t * uxi
            iy = y1i + t * uyi

            # Check if intersection is near the end of line i (within ext_tol)
            # t=0 is start, t=leni is end
            if t < -ext_tol or t > leni + ext_tol:
                continue
            if s < -ext_tol or s > lenj + ext_tol:
                continue

            # Extend line i: if intersection is beyond start/end
            if t < 0:
                end_ext[i]["start"].append((ix, iy, -t))
            elif t > leni:
                end_ext[i]["end"].append((ix, iy, t - leni))
            # Similarly for line j
            if s < 0:
                end_ext[j]["start"].append((ix, iy, -s))
            elif s > lenj:
                end_ext[j]["end"].append((ix, iy, s - lenj))

    # Apply extensions: pick the closest intersection at each end
    out = []
    for i, ln in enumerate(lines):
        if parsed[i] is None:
            out.append(ln)
            continue
        x1, y1, x2, y2 = parsed[i][0], parsed[i][1], parsed[i][2], parsed[i][3]
        # Extend start (closest intersection)
        starts = end_ext[i]["start"]
        if starts:
            best = min(starts, key=lambda p: p[2])
            if best[2] <= ext_tol:
                x1, y1 = best[0], best[1]
        # Extend end
        ends = end_ext[i]["end"]
        if ends:
            best = min(ends, key=lambda p: p[2])
            if best[2] <= ext_tol:
                x2, y2 = best[0], best[1]
        out.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "layer": ln.get("layer", ""),
            "type": "line",
        })
    return out


def _split_at_crossings(lines, tol=2.0):
    """Split lines at mutual intersection points to create a connected graph.

    When two non-parallel lines cross each other (not at endpoints), both
    are split at the crossing point so the graph builder can connect them.
    """
    if not lines or len(lines) < 2:
        return list(lines or [])

    parsed = []
    for ln in lines:
        x1 = float(ln["x1"])
        y1 = float(ln["y1"])
        x2 = float(ln["x2"])
        y2 = float(ln["y2"])
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        parsed.append((x1, y1, x2, y2, dx, dy, length))

    # For each line, collect split parameters (0..1)
    splits = [[] for _ in lines]

    for i in range(len(lines)):
        x1i, y1i, x2i, y2i, dxi, dyi, leni = parsed[i]
        if leni < 1.0:
            continue
        for j in range(i + 1, len(lines)):
            x1j, y1j, x2j, y2j, dxj, dyj, lenj = parsed[j]
            if lenj < 1.0:
                continue

            # Check not parallel
            denom = dxi * dyj - dyi * dxj
            if abs(denom) < 1e-9:
                continue

            dx0 = x1j - x1i
            dy0 = y1j - y1i
            t = (dx0 * dyj - dy0 * dxj) / denom
            s = (dx0 * dyi - dy0 * dxi) / denom

            # Check intersection is on each segment (with end margin).
            # A line is only split if the crossing is in its interior
            # (not at its endpoints). But if line A's interior crosses
            # line B at B's endpoint, A is still split — only B is skipped.
            end_margin_i = tol / max(leni, 1.0)
            t_interior = end_margin_i < t < 1.0 - end_margin_i
            end_margin_j = tol / max(lenj, 1.0)
            s_interior = end_margin_j < s < 1.0 - end_margin_j
            # At least one must be interior; the other must be on-segment
            t_on = -0.01 <= t <= 1.01
            s_on = -0.01 <= s <= 1.01
            if not (t_on and s_on):
                continue
            if not (t_interior or s_interior):
                continue

            if t_interior:
                splits[i].append(t)
            if s_interior:
                splits[j].append(s)

    # Build output with split lines
    out = []
    for i, ln in enumerate(lines):
        if not splits[i]:
            out.append(ln)
            continue
        x1, y1, x2, y2 = parsed[i][0], parsed[i][1], parsed[i][2], parsed[i][3]
        # Sort split params and add endpoints
        params = sorted(set(splits[i]))
        params = [0.0] + params + [1.0]
        layer = ln.get("layer", "")
        for k in range(len(params) - 1):
            t0 = params[k]
            t1 = params[k + 1]
            sx1 = x1 + (x2 - x1) * t0
            sy1 = y1 + (y2 - y1) * t0
            sx2 = x1 + (x2 - x1) * t1
            sy2 = y1 + (y2 - y1) * t1
            seg_len = math.sqrt((sx2 - sx1) ** 2 + (sy2 - sy1) ** 2)
            if seg_len >= 1.0:
                out.append({
                    "x1": sx1, "y1": sy1, "x2": sx2, "y2": sy2,
                    "layer": layer, "type": "line",
                })
    return out


def _find_wall_pairs(lines, cfg):
    """Find parallel line pairs that represent double-line walls.

    Returns (selected_pairs, pair_dists, used_indices) where each pair is (i, j).
    """
    default_wall = float((cfg or {}).get("default_wall_thickness_cm", 20.0))
    min_dist = float((cfg or {}).get("double_wall_pair_min_cm", max(4.0, default_wall * 0.25)))
    max_dist = float((cfg or {}).get("double_wall_pair_max_cm", max(45.0, default_wall * 2.5)))
    angle_tol = float((cfg or {}).get("double_wall_pair_angle_deg", 8.0))
    min_overlap_cm = float((cfg or {}).get("double_wall_pair_min_overlap_cm", 35.0))
    min_overlap_ratio = float((cfg or {}).get("double_wall_pair_overlap_ratio", 0.60))

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
        return [], [], set()

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

    return selected_pairs, pair_dists, used


def _collapse_to_centerlines(lines, pairs, include_unpaired=True):
    """Collapse pairs to centerlines (average of the two parallel lines).

    When include_unpaired is False, only centerlines are returned (no noise
    from door frames, crossbars, or other non-wall geometry).
    """
    used_in_pairs = set()
    for i, j in pairs:
        used_in_pairs.add(i)
        used_in_pairs.add(j)

    out = []
    for i, j in pairs:
        li = lines[i]
        lj = lines[j]
        ix1, iy1 = float(li["x1"]), float(li["y1"])
        ix2, iy2 = float(li["x2"]), float(li["y2"])
        jx1, jy1 = float(lj["x1"]), float(lj["y1"])
        jx2, jy2 = float(lj["x2"]), float(lj["y2"])

        # Compute centerline position (perpendicular midpoint)
        dx = ix2 - ix1
        dy = iy2 - iy1
        length_i = math.sqrt(dx * dx + dy * dy)
        if length_i < 1.0:
            continue
        ux = dx / length_i
        uy = dy / length_i

        # Centerline perpendicular position: average of both lines' midpoints
        # projected onto the normal
        nx = -uy
        ny = ux
        mi = _line_mid(li)
        mj = _line_mid(lj)
        perp_i = mi[0] * nx + mi[1] * ny
        perp_j = mj[0] * nx + mj[1] * ny
        center_perp = (perp_i + perp_j) * 0.5

        # Project ALL 4 endpoints onto the axis direction and take max extent
        projs = [
            ix1 * ux + iy1 * uy,
            ix2 * ux + iy2 * uy,
            jx1 * ux + jy1 * uy,
            jx2 * ux + jy2 * uy,
        ]
        proj_min = min(projs)
        proj_max = max(projs)

        # Reconstruct endpoints on centerline
        cx1 = ux * proj_min + nx * center_perp
        cy1 = uy * proj_min + ny * center_perp
        cx2 = ux * proj_max + nx * center_perp
        cy2 = uy * proj_max + ny * center_perp

        out.append({
            "x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2,
            "layer": li.get("layer", ""),
            "type": "line",
        })

    # Merge overlapping collinear centerlines into maximal segments.
    # This handles cases where multiple pairs generate overlapping
    # centerlines for the same physical wall (common in multi-room plans).
    out = _merge_collinear_overlapping(out, perp_tol=5.0, gap_tol=2.0)

    # Extend centerlines to meet at perpendicular intersections.
    # This connects walls at corners and T-junctions where endpoints
    # don't exactly match due to inner/outer wall length differences.
    out = _extend_to_intersections(out, ext_tol=50.0)

    # Split lines at crossing points so the graph builder connects them.
    # After merge+extend, centerlines may cross each other in the middle
    # (e.g., a long wall passing through a perpendicular wall's midpoint).
    out = _split_at_crossings(out, tol=2.0)

    if include_unpaired:
        for idx, ln in enumerate(lines):
            if idx not in used_in_pairs:
                out.append(ln)
    return out


def _collapse_pick_inner(lines, pairs, centroid):
    """Pick the inner line from each pair.

    Primary heuristic: the shorter line is the inner wall (outer walls wrap
    around corners and are typically longer).
    Tiebreaker: the line closer to *centroid* is the inner wall.
    """
    used_in_pairs = set()
    for i, j in pairs:
        used_in_pairs.add(i)
        used_in_pairs.add(j)

    out = []
    cx, cy = centroid
    for i, j in pairs:
        li = lines[i]
        lj = lines[j]
        len_i = _line_len(li)
        len_j = _line_len(lj)
        # If length difference > 5%, shorter = inner
        if len_i < len_j * 0.95:
            out.append(li)
        elif len_j < len_i * 0.95:
            out.append(lj)
        else:
            # Same length: use centroid distance
            mi = _line_mid(li)
            mj = _line_mid(lj)
            dci = math.sqrt((mi[0] - cx) ** 2 + (mi[1] - cy) ** 2)
            dcj = math.sqrt((mj[0] - cx) ** 2 + (mj[1] - cy) ** 2)
            out.append(li if dci <= dcj else lj)

    for idx, ln in enumerate(lines):
        if idx not in used_in_pairs:
            out.append(ln)
    return out


def _collapse_double_wall_lines(lines, cfg):
    """Two-pass double-wall collapse that handles L/T/U shapes.

    Pass 1: Collapse to centerlines, find cycle, get polygon centroid.
    Pass 2: Use centroid to pick correct inner line from each pair.
    """
    lines = list(lines or [])
    if len(lines) < 2:
        return lines, {
            "input_count": len(lines),
            "output_count": len(lines),
            "paired_count": 0,
            "estimated_wall_thickness_cm": None,
            "pairs": [],
        }

    selected_pairs, pair_dists, used = _find_wall_pairs(lines, cfg)

    if not selected_pairs:
        return lines, {
            "input_count": len(lines),
            "output_count": len(lines),
            "paired_count": 0,
            "estimated_wall_thickness_cm": None,
            "pairs": [],
        }

    # Use centerlines as initial collapse (works for any shape)
    centerline_out = _collapse_to_centerlines(lines, selected_pairs)

    est = None
    if pair_dists:
        ds = sorted(pair_dists)
        m = len(ds) // 2
        if len(ds) % 2 == 1:
            est = ds[m]
        else:
            est = (ds[m - 1] + ds[m]) * 0.5

    return centerline_out, {
        "input_count": len(lines),
        "output_count": len(centerline_out),
        "paired_count": len(selected_pairs),
        "estimated_wall_thickness_cm": est,
        "pairs": selected_pairs,
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
    """Merge overlapping detections of the same opening, but preserve
    distinct openings separated by > min_sep_cm.

    After merging overlaps, split back apart if the merged interval
    is suspiciously wide (> 1.8x the widest contributing candidate).
    """
    groups = {}
    for c in (cands or []):
        k = (c.get("type"), int(c.get("host_edge", -1)))
        groups.setdefault(k, []).append(c)

    out = []
    for _, arr in groups.items():
        arr = sorted(arr, key=lambda x: (x["start_cm"], -x.get("confidence", 0.0)))
        merged = []
        # Track contributing candidates for each merged group
        contrib = []
        for c in arr:
            if not merged:
                merged.append(dict(c))
                contrib.append([dict(c)])
                continue
            p = merged[-1]
            if c["start_cm"] <= (p["end_cm"] + min_sep_cm):
                p["start_cm"] = min(p["start_cm"], c["start_cm"])
                p["end_cm"] = max(p["end_cm"], c["end_cm"])
                p["width_cm"] = p["end_cm"] - p["start_cm"]
                p["confidence"] = max(p.get("confidence", 0.0), c.get("confidence", 0.0))
                contrib[-1].append(dict(c))
            else:
                merged.append(dict(c))
                contrib.append([dict(c)])

        # Split-back: if merged width is >1.8x widest contributor, keep individuals
        for i, m in enumerate(merged):
            max_contrib_w = max(c.get("width_cm", 1.0) for c in contrib[i])
            if m["width_cm"] > max_contrib_w * 1.8 and len(contrib[i]) >= 2:
                # Restore individual candidates instead of merged blob
                for c in contrib[i]:
                    out.append(c)
            else:
                out.append(m)

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
        edge_gaps = []
        for s, t in merged:
            if s > cursor:
                gw = s - cursor
                if gw >= min_w and gw <= max_w:
                    edge_gaps.append({
                        "type": "door",  # provisional, refined below
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
                edge_gaps.append({
                    "type": "door",
                    "host_edge": e["idx"],
                    "start_cm": cursor,
                    "end_cm": e["len"],
                    "width_cm": gw,
                    "confidence": 0.45,
                    "from_gap": True,
                })

        # Multiple gaps on the same edge are likely all windows, not doors.
        # A single gap <= 130cm is a door; wider is a window.
        if len(edge_gaps) >= 2:
            for g in edge_gaps:
                g["type"] = "window"
        else:
            for g in edge_gaps:
                g["type"] = "door" if g["width_cm"] <= 130.0 else "window"

        out.extend(edge_gaps)

    return out


def _opening_from_window_pattern(all_lines, edges, cfg):
    """Detect windows from clusters of short lines perpendicular to wall edges.

    Israeli DWG windows typically appear as 2+ short perpendicular lines
    (glass panes / mullions) within a gap in the wall.
    """
    perp_tol = float(cfg.get("window_pattern_perp_angle_tol_deg", 20.0))
    min_count = int(cfg.get("window_pattern_min_line_count", 2))
    host_dist = float(cfg.get("window_pattern_host_distance_cm", 30.0))
    min_w = float(cfg.get("window_pattern_min_width_cm", 40.0))
    max_w = float(cfg.get("window_pattern_max_width_cm", 200.0))
    max_line_len = float(cfg.get("window_pattern_max_line_length_cm", 25.0))

    out = []
    for e in (edges or []):
        eang = float(e.get("ang", 0.0))
        ea = e["a"]
        eb = e["b"]
        elen = float(e.get("len", 0.0))
        if elen < min_w:
            continue

        # Find short lines perpendicular to this edge and close to it
        perp_projs = []
        for ln in (all_lines or []):
            length = _line_len(ln)
            if length > max_line_len or length < 1.0:
                continue
            ang = _line_axis_angle_deg(ln)
            delta = _angle_delta_axis(ang, eang)
            if abs(delta - 90.0) > perp_tol:
                continue

            mx, my = _line_mid(ln)
            d, _ = _dist_pt_seg(mx, my, ea[0], ea[1], eb[0], eb[1])
            if d > host_dist:
                continue

            proj = _project_t_cm(mx, my, ea[0], ea[1], eb[0], eb[1])
            if proj < -5.0 or proj > elen + 5.0:
                continue
            perp_projs.append(proj)

        if len(perp_projs) < min_count:
            continue

        # Cluster by projection along edge
        perp_projs.sort()
        clusters = [[perp_projs[0]]]
        for k in range(1, len(perp_projs)):
            if perp_projs[k] - perp_projs[k - 1] <= 20.0:
                clusters[-1].append(perp_projs[k])
            else:
                clusters.append([perp_projs[k]])

        for cluster in clusters:
            if len(cluster) < min_count:
                continue
            span_start = cluster[0]
            span_end = cluster[-1]
            width = span_end - span_start
            if width < min_w * 0.5:
                # Narrow cluster — estimate window width from pane positions
                # Assume panes are inside the window, add margins
                width = max(width + 20.0, min_w)
                span_start = max(0.0, span_start - 10.0)
                span_end = min(elen, span_start + width)
                width = span_end - span_start
            if width < min_w or width > max_w:
                continue
            out.append({
                "type": "window",
                "host_edge": e["idx"],
                "start_cm": max(0.0, span_start),
                "end_cm": min(elen, span_end),
                "width_cm": max(1.0, span_end - span_start),
                "confidence": min(0.75, 0.3 + 0.1 * len(cluster)),
                "from_window_pattern": True,
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


def _match_dimensions_to_edges(edges, dim_lines, cfg):
    """Match dimension lines to individual polygon edges by parallelism + proximity.

    Returns dict: edge_idx -> measured_length_cm
    """
    angle_tol = float((cfg or {}).get("dimension_hint_angle_deg", 18.0))
    host_tol = float((cfg or {}).get("dimension_hint_host_distance_cm", 180.0))

    result = {}
    used_dims = set()

    for e in (edges or []):
        eang = float(e.get("ang", 0.0))
        elen = float(e.get("len", 0.0))
        if elen < 1.0:
            continue
        ea = e["a"]
        eb = e["b"]
        best = None

        for di, ln in enumerate(dim_lines or []):
            if di in used_dims:
                continue
            ln_len, ln_ang, mx, my = _line_len_ang_mid(ln)
            if ln_len <= 1.0:
                continue
            if _angle_delta_axis(ln_ang, eang) > angle_tol:
                continue
            d, _ = _dist_pt_seg(mx, my, ea[0], ea[1], eb[0], eb[1])
            if d > host_tol:
                continue
            # Dimension length should be within reasonable ratio of edge
            ratio = ln_len / max(elen, 1.0)
            if ratio < 0.3 or ratio > 3.0:
                continue
            score = d + abs(ln_len - elen) * 0.1
            if best is None or score < best[0]:
                best = (score, di, ln_len)

        if best is not None:
            used_dims.add(best[1])
            result[int(e["idx"])] = best[2]

    return result


def _match_dimensions_to_openings(openings, edges, dim_lines, cfg):
    """Match short dimension lines near opening centers to refine opening widths.

    Returns updated openings list and count of matched dimensions.
    """
    angle_tol = float((cfg or {}).get("dimension_hint_angle_deg", 18.0))
    host_tol = float((cfg or {}).get("dimension_hint_host_distance_cm", 180.0))

    edge_map = {}
    for e in (edges or []):
        edge_map[int(e["idx"])] = e

    out = []
    matched = 0
    for op in (openings or []):
        c = dict(op)
        idx = int(c.get("host_edge", -1))
        e = edge_map.get(idx)
        if e is None:
            out.append(c)
            continue

        ea = e["a"]
        eb = e["b"]
        eang = float(e.get("ang", 0.0))
        elen = float(e.get("len", 0.0))

        s0 = float(c.get("start_cm", 0.0))
        t0 = float(c.get("end_cm", s0))
        center0 = (s0 + t0) * 0.5
        curr_w = max(1.0, float(c.get("width_cm", 1.0)))

        best = None
        for ln in (dim_lines or []):
            ln_len, ln_ang, mx, my = _line_len_ang_mid(ln)
            if ln_len <= 1.0 or ln_len > 350.0:
                continue
            if _angle_delta_axis(ln_ang, eang) > angle_tol:
                continue
            d, _ = _dist_pt_seg(mx, my, ea[0], ea[1], eb[0], eb[1])
            if d > host_tol:
                continue

            tc = _project_t_cm(mx, my, ea[0], ea[1], eb[0], eb[1])
            # Dimension should be near the opening center
            if abs(tc - center0) > curr_w * 1.5:
                continue
            # Dimension length should be reasonable for an opening
            if ln_len < curr_w * 0.4 or ln_len > curr_w * 2.5:
                continue

            score = abs(tc - center0) + d * 0.5
            if best is None or score < best[0]:
                best = (score, ln_len)

        if best is not None:
            new_w = best[1]
            ns = max(0.0, center0 - new_w * 0.5)
            nt = min(elen, center0 + new_w * 0.5)
            if (nt - ns) >= 1.0:
                c["start_cm"] = ns
                c["end_cm"] = nt
                c["width_cm"] = max(1.0, nt - ns)
                c["from_edge_dimension"] = True
                matched += 1

        out.append(c)

    return out, matched


def _smooth_wall_zigzag(poly, wall_thickness_cm):
    """Legacy zigzag smoother — kept as fallback for non-paired walls."""
    if len(poly) < 4:
        return poly
    threshold = wall_thickness_cm * 1.8
    changed = True
    result = list(poly)
    for _pass in range(5):
        if not changed or len(result) < 4:
            break
        changed = False
        new_poly = []
        n = len(result)
        for i in range(n):
            a = result[(i - 1) % n]
            b = result[i]
            c = result[(i + 1) % n]
            ab = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
            bc = math.sqrt((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2)
            if min(ab, bc) > threshold:
                new_poly.append(b)
                continue
            abx, aby = b[0] - a[0], b[1] - a[1]
            bcx, bcy = c[0] - b[0], c[1] - b[1]
            if ab < 1e-9 or bc < 1e-9:
                new_poly.append(b)
                continue
            dot = (abx * bcx + aby * bcy) / (ab * bc)
            dot = max(-1.0, min(1.0, dot))
            turn_deg = math.degrees(math.acos(dot))
            if (turn_deg < 35.0 or turn_deg > 145.0) and min(ab, bc) < threshold:
                changed = True
                continue
            new_poly.append(b)
        if changed:
            result = new_poly
    return result


def _snap_poly_to_wall_centers(poly, wall_lines_raw, pairs, wall_thickness_cm):
    """Snap polygon vertices to wall pair centerlines to remove zigzag.

    When a polygon mixes inner and outer wall traces, it creates a zigzag
    pattern with ~wall_thickness offsets.  By projecting every vertex onto
    the centerline of its nearest wall pair, the polygon collapses to a
    consistent centerline position (ideal for Revit wall placement).

    Corner vertices near two perpendicular wall pairs accumulate both
    perpendicular shifts so they land at the correct intersection.
    """
    if not pairs or len(poly) < 3:
        return poly

    # Pre-compute pair geometry
    pair_data = []
    for pi, pj in pairs:
        li = wall_lines_raw[pi]
        lj = wall_lines_raw[pj]
        ix1, iy1 = float(li["x1"]), float(li["y1"])
        ix2, iy2 = float(li["x2"]), float(li["y2"])
        jx1, jy1 = float(lj["x1"]), float(lj["y1"])
        jx2, jy2 = float(lj["x2"]), float(lj["y2"])
        # Align endpoints
        d_same = ((ix1 - jx1) ** 2 + (iy1 - jy1) ** 2
                  + (ix2 - jx2) ** 2 + (iy2 - jy2) ** 2)
        d_flip = ((ix1 - jx2) ** 2 + (iy1 - jy2) ** 2
                  + (ix2 - jx1) ** 2 + (iy2 - jy1) ** 2)
        if d_flip < d_same:
            jx1, jy1, jx2, jy2 = jx2, jy2, jx1, jy1

        # Wall direction unit vector
        dx = ix2 - ix1
        dy = iy2 - iy1
        wlen = math.sqrt(dx * dx + dy * dy)
        if wlen < 1.0:
            continue
        ux = dx / wlen
        uy = dy / wlen
        # Wall normal (perpendicular, pointing from i toward j)
        nx = -uy
        ny = ux
        # Signed perpendicular distance from i-midpoint to j-midpoint
        li_mx = (ix1 + ix2) * 0.5
        li_my = (iy1 + iy2) * 0.5
        lj_mx = (jx1 + jx2) * 0.5
        lj_my = (jy1 + jy2) * 0.5
        d_perp = (lj_mx - li_mx) * nx + (lj_my - li_my) * ny
        half_d = d_perp * 0.5  # shift from line-i to centerline

        pair_data.append({
            "li": (ix1, iy1, ix2, iy2),
            "lj": (jx1, jy1, jx2, jy2),
            "nx": nx, "ny": ny,
            "shift_i_x": nx * half_d,
            "shift_i_y": ny * half_d,
            "shift_j_x": -nx * half_d,
            "shift_j_y": -ny * half_d,
            "ang": math.degrees(math.atan2(uy, ux)),
        })

    if not pair_data:
        return poly

    half_wall = wall_thickness_cm * 0.75
    result = []
    for vx, vy in poly:
        # Accumulate shifts from all nearby wall pairs (one per unique angle)
        shifts = {}  # keyed by rounded wall angle -> (shift_x, shift_y)
        for pd in pair_data:
            ix1, iy1, ix2, iy2 = pd["li"]
            di, _ = _dist_pt_seg(vx, vy, ix1, iy1, ix2, iy2)
            jx1, jy1, jx2, jy2 = pd["lj"]
            dj, _ = _dist_pt_seg(vx, vy, jx1, jy1, jx2, jy2)

            matched_shift = None
            d_min = min(di, dj)
            if d_min > half_wall:
                continue
            if di <= dj:
                matched_shift = (pd["shift_i_x"], pd["shift_i_y"])
            else:
                matched_shift = (pd["shift_j_x"], pd["shift_j_y"])

            # Only keep one shift per wall direction (closest match wins)
            ang_key = round(pd["ang"] % 180.0)
            prev = shifts.get(ang_key)
            if prev is None or d_min < prev[2]:
                shifts[ang_key] = (matched_shift[0], matched_shift[1], d_min)

        if shifts:
            sx = sum(v[0] for v in shifts.values())
            sy = sum(v[1] for v in shifts.values())
            result.append((vx + sx, vy + sy))
        else:
            result.append((vx, vy))

    # Fold removal: when zigzag U-turns snap to the same centerline point,
    # the polygon revisits a position, creating a fold (A→...→A).
    # Only remove the loop if it has near-zero area (back-and-forth trace),
    # not if it contains a large enclosed area.
    fold_tol = 5.0  # cm
    changed = True
    while changed and len(result) >= 3:
        changed = False
        for i in range(len(result)):
            for j in range(i + 2, len(result)):
                dx = result[i][0] - result[j][0]
                dy = result[i][1] - result[j][1]
                if dx * dx + dy * dy < fold_tol * fold_tol:
                    # Check if the loop i→j has negligible area
                    loop = result[i:j + 1]
                    loop_area = 0.0
                    for k in range(len(loop)):
                        x1, y1 = loop[k]
                        x2, y2 = loop[(k + 1) % len(loop)]
                        loop_area += x1 * y2 - x2 * y1
                    loop_area = abs(loop_area) * 0.5
                    # Only remove if loop area is tiny (< wall_thickness^2)
                    if loop_area < wall_thickness_cm * wall_thickness_cm:
                        result = result[:i + 1] + result[j + 1:]
                        changed = True
                        break
            if changed:
                break

    # Clean up: remove near-duplicate consecutive vertices
    if len(result) < 3:
        return result
    cleaned = [result[0]]
    for k in range(1, len(result)):
        d = math.sqrt((result[k][0] - cleaned[-1][0]) ** 2
                      + (result[k][1] - cleaned[-1][1]) ** 2)
        if d > 2.0:
            cleaned.append(result[k])
    if len(cleaned) > 2:
        d = math.sqrt((cleaned[-1][0] - cleaned[0][0]) ** 2
                      + (cleaned[-1][1] - cleaned[0][1]) ** 2)
        if d < 2.0:
            cleaned.pop()

    # Remove collinear points
    if len(cleaned) > 3:
        final = []
        nc = len(cleaned)
        for i in range(nc):
            p = cleaned[(i - 1) % nc]
            q = cleaned[i]
            r = cleaned[(i + 1) % nc]
            cross = abs((q[0] - p[0]) * (r[1] - q[1])
                        - (q[1] - p[1]) * (r[0] - q[0]))
            if cross > 5.0:
                final.append(q)
        if len(final) >= 3:
            cleaned = final

    return cleaned


def _find_internal_walls(poly, nodes, segs, bridges, snap_tol=2.0, min_len_cm=30.0,
                         perimeter_parallel_tol_cm=10.0, perimeter_parallel_angle_deg=8.0,
                         endpoint_snap_cm=6.0, dangling_max_cm=35.0, with_debug=False):
    """Find graph edges that lie inside the outer polygon but are not on its boundary.

    These are internal partition walls dividing rooms.

    Returns a list of dicts:
        [{"start": (x, y), "end": (x, y)}, ...]
    Each represents a wall segment in cm coordinates.
    """
    stats = {
        "candidate_count": 0,
        "rejected_boundary": 0,
        "rejected_outside": 0,
        "rejected_short": 0,
        "rejected_parallel_perimeter": 0,
        "rejected_dangling": 0,
        "accepted_premerge": 0,
        "accepted_count": 0,
    }
    if len(poly) < 3:
        if with_debug:
            return [], stats
        return []

    # Build set of polygon boundary edges as coordinate pairs (snapped)
    def _snap(v):
        return round(v / snap_tol) * snap_tol

    boundary_edges = set()
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        key = ((_snap(ax), _snap(ay)), (_snap(bx), _snap(by)))
        rev = (key[1], key[0])
        boundary_edges.add(key)
        boundary_edges.add(rev)

    # Collect all graph edges (original + bridged)
    all_segs = list(segs or [])
    for br in (bridges or []):
        all_segs.append(br)

    # Node degree helps reject dangling/noise internal edges.
    node_deg = {}
    for seg in all_segs:
        a_idx = int(seg["a"])
        b_idx = int(seg["b"])
        node_deg[a_idx] = int(node_deg.get(a_idx, 0)) + 1
        node_deg[b_idx] = int(node_deg.get(b_idx, 0)) + 1

    perimeter_edges = _edge_list(poly)

    internal = []
    for seg in all_segs:
        stats["candidate_count"] += 1
        a_idx = int(seg["a"])
        b_idx = int(seg["b"])
        if a_idx >= len(nodes) or b_idx >= len(nodes):
            continue
        ax, ay = nodes[a_idx]
        bx, by = nodes[b_idx]

        # Skip if this edge is on the polygon boundary
        key = ((_snap(ax), _snap(ay)), (_snap(bx), _snap(by)))
        if key in boundary_edges:
            stats["rejected_boundary"] += 1
            continue

        # Both endpoints must be inside or on the polygon boundary
        # Use a slightly inflated polygon test — check midpoint is inside
        mx = (ax + bx) * 0.5
        my = (ay + by) * 0.5
        if not _point_in_poly(mx, my, poly):
            stats["rejected_outside"] += 1
            continue

        # Skip very short segments (noise, split artifacts)
        dx = bx - ax
        dy = by - ay
        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len < float(min_len_cm):
            stats["rejected_short"] += 1
            continue

        seg_ang = math.degrees(math.atan2(dy, dx))
        near_parallel_perimeter = False
        min_dist_a = None
        min_dist_b = None
        for pe in perimeter_edges:
            pa = pe["a"]
            pb = pe["b"]
            d_mid, _ = _dist_pt_seg(mx, my, pa[0], pa[1], pb[0], pb[1])
            if d_mid <= float(perimeter_parallel_tol_cm):
                if _angle_delta_axis(seg_ang, pe.get("ang", 0.0)) <= float(perimeter_parallel_angle_deg):
                    near_parallel_perimeter = True
                    break
            d_a, _ = _dist_pt_seg(ax, ay, pa[0], pa[1], pb[0], pb[1])
            d_b, _ = _dist_pt_seg(bx, by, pa[0], pa[1], pb[0], pb[1])
            if min_dist_a is None or d_a < min_dist_a:
                min_dist_a = d_a
            if min_dist_b is None or d_b < min_dist_b:
                min_dist_b = d_b

        if near_parallel_perimeter:
            stats["rejected_parallel_perimeter"] += 1
            continue

        a_on_perimeter = (min_dist_a is not None and min_dist_a <= float(endpoint_snap_cm))
        b_on_perimeter = (min_dist_b is not None and min_dist_b <= float(endpoint_snap_cm))
        deg_a = int(node_deg.get(a_idx, 0))
        deg_b = int(node_deg.get(b_idx, 0))

        # Reject short dangling stubs that do not connect to perimeter/junctions.
        if (deg_a <= 1 or deg_b <= 1) and seg_len < float(dangling_max_cm) and (not a_on_perimeter) and (not b_on_perimeter):
            stats["rejected_dangling"] += 1
            continue
        if (deg_a <= 1 and deg_b <= 1) and (not a_on_perimeter) and (not b_on_perimeter):
            stats["rejected_dangling"] += 1
            continue

        internal.append({
            "start": (ax, ay),
            "end": (bx, by),
            "length_cm": seg_len,
        })
    stats["accepted_premerge"] = len(internal)

    # Merge collinear connected internal walls into single segments
    if len(internal) < 2:
        stats["accepted_count"] = len(internal)
        if with_debug:
            return internal, stats
        return internal

    merged = True
    while merged:
        merged = False
        for i in range(len(internal)):
            for j in range(i + 1, len(internal)):
                wi = internal[i]
                wj = internal[j]
                # Check if they share an endpoint (within snap_tol)
                pts_i = [wi["start"], wi["end"]]
                pts_j = [wj["start"], wj["end"]]
                shared = False
                for pi in pts_i:
                    for pj in pts_j:
                        dx = pi[0] - pj[0]
                        dy = pi[1] - pj[1]
                        if dx * dx + dy * dy < snap_tol * snap_tol * 4:
                            shared = True
                            break
                    if shared:
                        break
                if not shared:
                    continue
                # Check collinearity: angle between the two segments
                dx_i = wi["end"][0] - wi["start"][0]
                dy_i = wi["end"][1] - wi["start"][1]
                dx_j = wj["end"][0] - wj["start"][0]
                dy_j = wj["end"][1] - wj["start"][1]
                li = math.sqrt(dx_i * dx_i + dy_i * dy_i)
                lj = math.sqrt(dx_j * dx_j + dy_j * dy_j)
                if li < 1e-6 or lj < 1e-6:
                    continue
                dot = _abs(dx_i * dx_j + dy_i * dy_j) / (li * lj)
                if dot < 0.996:  # ~5 degrees
                    continue
                # Merge: project all 4 endpoints onto the axis and take extremes
                ux = dx_i / li
                uy = dy_i / li
                all_pts = [wi["start"], wi["end"], wj["start"], wj["end"]]
                projs = [(p[0] * ux + p[1] * uy, p) for p in all_pts]
                projs.sort(key=lambda x: x[0])
                new_start = projs[0][1]
                new_end = projs[-1][1]
                dx_n = new_end[0] - new_start[0]
                dy_n = new_end[1] - new_start[1]
                new_len = math.sqrt(dx_n * dx_n + dy_n * dy_n)
                internal[i] = {"start": new_start, "end": new_end, "length_cm": new_len}
                internal.pop(j)
                merged = True
                break
            if merged:
                break

    # Dedupe near-identical segments after merge.
    deduped = []
    seen = set()
    for w in internal:
        p1 = (round(float(w["start"][0]), 1), round(float(w["start"][1]), 1))
        p2 = (round(float(w["end"][0]), 1), round(float(w["end"][1]), 1))
        key = (p1, p2) if p1 <= p2 else (p2, p1)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(w)

    stats["accepted_count"] = len(deduped)
    if with_debug:
        return deduped, stats
    return deduped


def _annotate_opening_centers(openings, edges):
    edge_map = {}
    for e in (edges or []):
        edge_map[int(e.get("idx", -1))] = e

    out = []
    for op in (openings or []):
        o = dict(op)
        host_edge = int(o.get("host_edge", -1))
        e = edge_map.get(host_edge)
        if e is not None:
            start_cm = float(o.get("start_cm", 0.0))
            end_cm = float(o.get("end_cm", start_cm))
            if end_cm < start_cm:
                start_cm, end_cm = end_cm, start_cm
            edge_len = max(1.0e-9, float(e.get("len", 1.0)))
            center_t = max(0.0, min(edge_len, (start_cm + end_cm) * 0.5))
            ax, ay = e["a"]
            bx, by = e["b"]
            ux = (bx - ax) / edge_len
            uy = (by - ay) / edge_len
            o["center_x_cm"] = ax + (ux * center_t)
            o["center_y_cm"] = ay + (uy * center_t)
        out.append(o)
    return out


def recognize_topology(classified, cfg):
    cfg = cfg or {}

    wall_lines_raw = _merge_unique_lines(
        list(classified.get("wall_lines") or []),
        list(classified.get("unclassified_lines") or []),
    )
    wall_lines, collapse_dbg = _collapse_double_wall_lines(wall_lines_raw, cfg)
    if not wall_lines:
        raise ValueError("Wall recognition failed: no line candidates")

    pairs = collapse_dbg.get("pairs", [])
    solve_mode = "collapsed"
    solve_out = None

    # Strategy 1: Centerlines-only (no unpaired noise like door frames, crossbars).
    # This gives the cleanest graph for double-wall rooms.
    if pairs:
        centerlines_only = _collapse_to_centerlines(wall_lines_raw, pairs, include_unpaired=False)
        if centerlines_only:
            solve_out = _find_room_cycle_from_lines(centerlines_only, classified, cfg, relax_gap=False)
            if solve_out is not None:
                solve_mode = "centerlines_only"

    # Strategy 2: Inner-wall-only + unpaired lines.
    # For each pair, pick the shorter line (inner wall is shorter than outer).
    # This removes outer wall lines that cause zigzag cycles.
    if solve_out is None and pairs:
        lines_centroid = _centroid_of_line_mids(wall_lines_raw)
        inner_lines = _collapse_pick_inner(wall_lines_raw, pairs, lines_centroid)
        solve_out = _find_room_cycle_from_lines(inner_lines, classified, cfg, relax_gap=False)
        if solve_out is not None:
            solve_mode = "inner_walls"
            wall_lines = inner_lines

    # Strategy 3: Inner walls with relaxed bridging.
    if solve_out is None and pairs:
        lines_centroid = _centroid_of_line_mids(wall_lines_raw)
        inner_lines = _collapse_pick_inner(wall_lines_raw, pairs, lines_centroid)
        solve_out = _find_room_cycle_from_lines(inner_lines, classified, cfg, relax_gap=True)
        if solve_out is not None:
            solve_mode = "inner_walls_relaxed"
            wall_lines = inner_lines

    # Strategy 4: Centerlines + unpaired lines (original collapsed approach).
    if solve_out is None:
        solve_out = _find_room_cycle_from_lines(wall_lines, classified, cfg, relax_gap=False)
        if solve_out is not None:
            solve_mode = "collapsed"

    # Strategy 5: Raw lines fallback (both inner and outer wall traces).
    if solve_out is None and wall_lines_raw:
        solve_mode = "raw_fallback"
        solve_out = _find_room_cycle_from_lines(wall_lines_raw, classified, cfg, relax_gap=False)
        if solve_out is not None:
            wall_lines = list(wall_lines_raw)

    # Strategy 6: Raw lines with relaxed bridge tolerance.
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

    # Post-process: remove double-wall zigzag jogs from the polygon.
    # When inner/outer wall traces both appear, the polygon has short segments
    # (~wall thickness) connecting them.
    default_wall = float(cfg.get("default_wall_thickness_cm", 20.0))
    estimated_wall = collapse_dbg.get("estimated_wall_thickness_cm")
    wall_thickness_cm = float(estimated_wall) if estimated_wall is not None else default_wall

    # Snap to pair centerlines only when the cycle came from raw traces
    # (not already on centerlines). Centerlines_only polygons are already
    # at the correct midpoint position — snapping would push them to
    # inner/outer wall traces.
    if pairs and solve_mode in ("collapsed", "raw_fallback", "raw_relaxed_bridge"):
        poly_raw = _snap_poly_to_wall_centers(
            poly_raw, wall_lines_raw, pairs, wall_thickness_cm)
    elif solve_mode not in ("centerlines_only",):
        poly_raw = _smooth_wall_zigzag(poly_raw, wall_thickness_cm)

    edges_raw = _edge_list(poly_raw)
    minx_raw = min(p[0] for p in poly_raw)
    miny_raw = min(p[1] for p in poly_raw)
    maxx_raw = max(p[0] for p in poly_raw)
    maxy_raw = max(p[1] for p in poly_raw)

    host_tol_cm = float(cfg.get("opening_host_distance_mm", 70.0)) / 10.0
    host_tol_cm = max(host_tol_cm, (wall_thickness_cm * 0.6) + 2.0)
    openings = []
    openings.extend(_opening_from_arc(classified.get("door_arcs") or [], edges_raw, host_tol_cm * 2.0, 60.0, 180.0))
    openings.extend(_opening_from_lines(classified.get("door_lines") or [], edges_raw, "door", host_tol_cm * 1.5, 60.0, 180.0))
    openings.extend(_opening_from_lines(classified.get("window_lines") or [], edges_raw, "window", host_tol_cm * 1.5, 40.0, 350.0))
    openings.extend(_opening_from_bridges(bridges, nodes, edges_raw))

    # Edge gap and window pattern detection: skip for centerlines_only mode
    # because centerline gaps are topology artifacts (wall pairs don't cover
    # full extent), not real openings. These methods produce false windows.
    if solve_mode != "centerlines_only":
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
        # Window pattern detection from perpendicular line clusters
        all_input_lines = _merge_unique_lines(
            list(classified.get("wall_lines") or []),
            _merge_unique_lines(
                list(classified.get("unclassified_lines") or []),
                list(classified.get("window_lines") or []),
            ),
        )
        openings.extend(_opening_from_window_pattern(all_input_lines, edges_raw, cfg))

    # Reclassify bridge-detected openings: multiple "doors" on collinear edges
    # are likely windows (you rarely have 3+ doors in a row on the same wall).
    # Also, a bridge opening near a door arc stays as door; others become windows.
    door_arcs = classified.get("door_arcs") or []
    edge_ang_map = {}
    for e in edges_raw:
        edge_ang_map[int(e["idx"])] = float(e.get("ang", 0.0))
    ang_groups = {}
    for oi, o in enumerate(openings):
        if o.get("confidence", 0) != 0.55:  # bridge openings have conf=0.55
            continue
        eidx = int(o.get("host_edge", -1))
        eang = edge_ang_map.get(eidx, 0.0) % 180.0
        akey = round(eang / 10.0) * 10.0
        ang_groups.setdefault(akey, []).append(oi)
    for akey, indices in ang_groups.items():
        if len(indices) < 2:
            continue
        # Multiple bridge openings on same wall direction → windows
        for oi in indices:
            # Check if a door arc is near this opening (within 50cm of edge midpoint)
            o = openings[oi]
            eidx = int(o.get("host_edge", -1))
            e = None
            for ed in edges_raw:
                if int(ed["idx"]) == eidx:
                    e = ed
                    break
            has_door_arc = False
            if e is not None:
                emx = (e["a"][0] + e["b"][0]) * 0.5
                emy = (e["a"][1] + e["b"][1]) * 0.5
                for arc in door_arcs:
                    acx = float(arc.get("cx", 0.0))
                    acy = float(arc.get("cy", 0.0))
                    d = math.sqrt((acx - emx) ** 2 + (acy - emy) ** 2)
                    if d < 80.0:
                        has_door_arc = True
                        break
            if not has_door_arc:
                openings[oi]["type"] = "window"

    openings = _merge_openings(openings, float(cfg.get("opening_merge_min_sep_cm", 15.0)))

    # Filter dimension lines to only those near the room polygon bounding box.
    # DWGs often contain title block / sheet dims at coordinates far from the room.
    dim_lines_raw = list(classified.get("dimension_lines") or [])
    dim_margin = 500.0  # cm margin around room bbox
    dim_lines = []
    for ln in dim_lines_raw:
        mx = (float(ln.get("x1", 0)) + float(ln.get("x2", 0))) * 0.5
        my = (float(ln.get("y1", 0)) + float(ln.get("y2", 0))) * 0.5
        if (minx_raw - dim_margin <= mx <= maxx_raw + dim_margin and
                miny_raw - dim_margin <= my <= maxy_raw + dim_margin):
            dim_lines.append(ln)
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
    # Per-segment dimension matching for individual opening widths
    openings, edge_dim_matched = _match_dimensions_to_openings(openings, edges, dim_lines, cfg)
    edge_dimensions = _match_dimensions_to_edges(edges, dim_lines, cfg)
    openings = _merge_openings(openings, float(cfg.get("opening_merge_min_sep_cm", 15.0)))
    openings = _annotate_opening_centers(openings, edges)

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
        seg = {
            "start": [e["a"][0], e["a"][1]],
            "end": [e["b"][0], e["b"][1]],
            "length_cm": e["len"],
        }
        measured = edge_dimensions.get(int(e["idx"]))
        if measured is not None:
            seg["measured_length_cm"] = measured
        wall_segments.append(seg)

    # Optional synthetic fallback for demos/debug only.
    if bool(cfg.get("enable_synthetic_window_fallback", False)) and (not any([o for o in openings if o.get("type") == "window"])):
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

    # Detect internal partition walls (edges inside the outer polygon)
    all_bridge_segs = [{"a": br["a"], "b": br["b"], "bridged": True} for br in bridges]
    internal_walls, internal_wall_stats = _find_internal_walls(
        poly,
        nodes,
        segs,
        all_bridge_segs,
        snap_tol=float(cfg.get("internal_wall_snap_tol_cm", 2.0)),
        min_len_cm=float(cfg.get("internal_wall_min_length_cm", 30.0)),
        perimeter_parallel_tol_cm=float(cfg.get("internal_wall_perimeter_parallel_tol_cm", 10.0)),
        perimeter_parallel_angle_deg=float(cfg.get("internal_wall_perimeter_parallel_angle_deg", 8.0)),
        endpoint_snap_cm=float(cfg.get("internal_wall_endpoint_snap_cm", 6.0)),
        dangling_max_cm=float(cfg.get("internal_wall_dangling_max_cm", 35.0)),
        with_debug=True,
    )

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

    room_data = {
        "room_polygon_cm": [[p[0], p[1]] for p in poly],
        "wall_segments_cm": wall_segments,
        "internal_walls_cm": [[w["start"][0], w["start"][1], w["end"][0], w["end"][1]] for w in internal_walls],
        "openings": openings,
        "measurements_cm": measurements_cm,
    }

    debug = {
        "mode": "polygon_v2",
        "solve_mode": solve_mode,
        "wall_line_candidates": len(wall_lines),
        "wall_line_candidates_raw": len(wall_lines_raw),
        "paired_wall_line_count": int(collapse_dbg.get("paired_count") or 0),
        "estimated_wall_thickness_cm": collapse_dbg.get("estimated_wall_thickness_cm"),
        "dimension_line_count": len(dim_lines),
        "dimension_span_hint_used_count": int(dim_hints.get("used_lines") or 0),
        "dimension_opening_hint_used_count": int(opening_dim_used),
        "dimension_edge_matched_count": len(edge_dimensions),
        "dimension_opening_edge_matched_count": int(edge_dim_matched),
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
        "internal_wall_count": len(internal_walls),
        "internal_wall_candidate_count": int(internal_wall_stats.get("candidate_count", 0)),
        "internal_wall_rejected_boundary_count": int(internal_wall_stats.get("rejected_boundary", 0)),
        "internal_wall_rejected_outside_count": int(internal_wall_stats.get("rejected_outside", 0)),
        "internal_wall_rejected_short_count": int(internal_wall_stats.get("rejected_short", 0)),
        "internal_wall_rejected_parallel_perimeter_count": int(internal_wall_stats.get("rejected_parallel_perimeter", 0)),
        "internal_wall_rejected_dangling_count": int(internal_wall_stats.get("rejected_dangling", 0)),
    }

    result = dict(room_data)
    result["rooms"] = [room_data]
    result["debug"] = debug
    return result
