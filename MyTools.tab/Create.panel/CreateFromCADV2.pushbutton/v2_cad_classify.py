# -*- coding: utf-8 -*-
"""Classification that stays permissive for unlabeled CAD."""

import math
import re


def _compile_regex(patterns):
    out = []
    for p in (patterns or []):
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except Exception:
            continue
    return out


def _match_any(text, compiled):
    t = str(text or "")
    for rx in compiled:
        if rx.search(t):
            return True
    return False


# ---------------------------------------------------------------------------
# Geometry helpers for pre-classification
# ---------------------------------------------------------------------------

def _line_len(ln):
    dx = float(ln["x2"]) - float(ln["x1"])
    dy = float(ln["y2"]) - float(ln["y1"])
    return math.sqrt(dx * dx + dy * dy)


def _line_angle_deg(ln):
    dx = float(ln["x2"]) - float(ln["x1"])
    dy = float(ln["y2"]) - float(ln["y1"])
    return math.degrees(math.atan2(dy, dx))


def _angle_delta(a, b):
    d = abs((a - b) % 360.0)
    if d > 180.0:
        d = 360.0 - d
    return d


def _angle_delta_axis(a, b):
    d = _angle_delta(a, b)
    if d > 90.0:
        d = 180.0 - d
    return d


def _line_endpoint_min_dist(ln, lines):
    """Min distance from either endpoint of *ln* to any endpoint of *lines*."""
    x1, y1 = float(ln["x1"]), float(ln["y1"])
    x2, y2 = float(ln["x2"]), float(ln["y2"])
    best = 1e20
    for other in lines:
        for ox, oy in [(float(other["x1"]), float(other["y1"])),
                        (float(other["x2"]), float(other["y2"]))]:
            for px, py in [(x1, y1), (x2, y2)]:
                d = math.sqrt((px - ox) ** 2 + (py - oy) ** 2)
                if d < best:
                    best = d
    return best


def _is_dashed_sequence(lines, max_seg_cm, max_gap_cm, min_count):
    """Check if a list of collinear short segments looks like a dashed line."""
    if len(lines) < min_count:
        return False
    # All must be short
    for ln in lines:
        if _line_len(ln) > max_seg_cm:
            return False
    # Check angles are consistent
    base_ang = _line_angle_deg(lines[0])
    for ln in lines[1:]:
        if _angle_delta_axis(_line_angle_deg(ln), base_ang) > 8.0:
            return False
    # Check gaps between consecutive segments are small and regular
    # Sort by projection on the shared axis
    rad = math.radians(base_ang)
    ux, uy = math.cos(rad), math.sin(rad)

    def proj(ln):
        return min(
            float(ln["x1"]) * ux + float(ln["y1"]) * uy,
            float(ln["x2"]) * ux + float(ln["y2"]) * uy,
        )
    sorted_lines = sorted(lines, key=proj)
    for i in range(len(sorted_lines) - 1):
        cur_end = max(
            float(sorted_lines[i]["x1"]) * ux + float(sorted_lines[i]["y1"]) * uy,
            float(sorted_lines[i]["x2"]) * ux + float(sorted_lines[i]["y2"]) * uy,
        )
        nxt_start = min(
            float(sorted_lines[i + 1]["x1"]) * ux + float(sorted_lines[i + 1]["y1"]) * uy,
            float(sorted_lines[i + 1]["x2"]) * ux + float(sorted_lines[i + 1]["y2"]) * uy,
        )
        gap = nxt_start - cur_end
        if gap < -1.0 or gap > max_gap_cm:
            return False
    return True


def _geometric_reclassify(out, cfg, all_lines):
    """Move unclassified entities to typed buckets based on geometry patterns.

    This runs after layer-based classification to rescue entities when
    layer names are uninformative (e.g. '0', 'Layer1').
    """
    cfg = cfg or {}
    arc_min_r = float(cfg.get("geom_classify_arc_min_r_cm", 30.0))
    arc_max_r = float(cfg.get("geom_classify_arc_max_r_cm", 110.0))
    min_useful = float(cfg.get("geom_classify_min_useful_line_cm", 3.0))
    dashed_seg_max = float(cfg.get("geom_classify_dashed_segment_max_cm", 8.0))
    dashed_gap_max = float(cfg.get("geom_classify_dashed_gap_max_cm", 8.0))
    win_max_line_len = float(cfg.get("geom_classify_window_max_line_len_cm", 25.0))
    win_min_parallel = int(cfg.get("geom_classify_window_min_parallel_count", 2))
    win_spacing_max = float(cfg.get("geom_classify_window_spacing_max_cm", 15.0))

    # --- 1. Door arcs: unclassified arcs with radius in door-swing range ---
    # Only reclassify if no door_arcs were found by layer name
    if not out["door_arcs"]:
        promoted_arcs = []
        remaining_arcs = []
        for arc in out["unclassified_arcs"]:
            r = float(arc.get("r", 0.0))
            if arc_min_r <= r <= arc_max_r:
                promoted_arcs.append(arc)
            else:
                remaining_arcs.append(arc)
        if promoted_arcs:
            out["door_arcs"] = promoted_arcs
            out["unclassified_arcs"] = remaining_arcs

    # --- 2. Tiny lines → ignored (tick marks, annotation lines) ---
    kept_lines = []
    tiny_count = 0
    for ln in out["unclassified_lines"]:
        if _line_len(ln) < min_useful:
            tiny_count += 1
        else:
            kept_lines.append(ln)
    out["unclassified_lines"] = kept_lines
    out["ignored"] += tiny_count

    # --- 1b. Outlier lines → ignored (far from main cluster) ---
    # DWGs often contain north arrows, title block marks, or revision symbols
    # at coordinates very far from the actual room geometry.
    remaining = list(out["unclassified_lines"])
    if len(remaining) > 10:
        mids_x = []
        mids_y = []
        for ln in remaining:
            mx = (float(ln["x1"]) + float(ln["x2"])) * 0.5
            my = (float(ln["y1"]) + float(ln["y2"])) * 0.5
            mids_x.append(mx)
            mids_y.append(my)
        mids_x_s = sorted(mids_x)
        mids_y_s = sorted(mids_y)
        n4 = len(mids_x_s) // 4
        q1x = mids_x_s[max(0, n4)]
        q3x = mids_x_s[min(len(mids_x_s) - 1, len(mids_x_s) - 1 - n4)]
        q1y = mids_y_s[max(0, n4)]
        q3y = mids_y_s[min(len(mids_y_s) - 1, len(mids_y_s) - 1 - n4)]
        iqr_x = max(q3x - q1x, 100.0)
        iqr_y = max(q3y - q1y, 100.0)
        fence = 4.0
        lo_x = q1x - fence * iqr_x
        hi_x = q3x + fence * iqr_x
        lo_y = q1y - fence * iqr_y
        hi_y = q3y + fence * iqr_y
        new_remaining = []
        outlier_count = 0
        for i, ln in enumerate(remaining):
            mx = mids_x[i]
            my = mids_y[i]
            if mx < lo_x or mx > hi_x or my < lo_y or my > hi_y:
                outlier_count += 1
            else:
                new_remaining.append(ln)
        if outlier_count > 0:
            out["unclassified_lines"] = new_remaining
            out["ignored"] += outlier_count

    # --- 2a. Dimension tick marks → ignored ---
    # DWGs often have dimension arrow/tick marks on layer "0" (not "dimension").
    # These are ~21cm diagonal lines at 45 degrees near dimension line endpoints.
    dim_lines = out.get("dimension_lines", [])
    if dim_lines:
        dim_tick_max = float(cfg.get("geom_classify_dim_tick_max_cm", 25.0))
        dim_tick_min = float(cfg.get("geom_classify_dim_tick_min_cm", 15.0))
        remaining = list(out["unclassified_lines"])
        tick_indices = set()
        for i, ln in enumerate(remaining):
            length = _line_len(ln)
            if length < dim_tick_min or length > dim_tick_max:
                continue
            # Must be at ~45 degrees
            ang = _line_angle_deg(ln) % 180.0
            if not (35.0 <= ang <= 55.0 or 125.0 <= ang <= 145.0):
                continue
            # Must be near a dimension line endpoint
            mx = (float(ln["x1"]) + float(ln["x2"])) * 0.5
            my = (float(ln["y1"]) + float(ln["y2"])) * 0.5
            near_dim = False
            for dl in dim_lines:
                for dx, dy in [(float(dl["x1"]), float(dl["y1"])),
                               (float(dl["x2"]), float(dl["y2"]))]:
                    d = math.sqrt((mx - dx) ** 2 + (my - dy) ** 2)
                    if d < 30.0:
                        near_dim = True
                        break
                if near_dim:
                    break
            if near_dim:
                tick_indices.add(i)
        if tick_indices:
            new_remaining = []
            for i, ln in enumerate(remaining):
                if i in tick_indices:
                    out["ignored"] += 1
                else:
                    new_remaining.append(ln)
            out["unclassified_lines"] = new_remaining

    # --- 2b. Small closed rectangles → ignored (window frame details) ---
    # In real DWGs, window/door frames appear as small rectangles (e.g. 11cm x 5cm).
    # These are groups of 4 short lines forming a closed box that pollute the wall graph.
    small_rect_max = float(cfg.get("geom_classify_small_rect_max_cm", 20.0))
    remaining = list(out["unclassified_lines"])
    short_lines = [(i, ln) for i, ln in enumerate(remaining) if _line_len(ln) <= small_rect_max]
    rect_indices = set()
    # Group short lines by proximity of their midpoints
    used_short = set()
    for si, (idx_a, ln_a) in enumerate(short_lines):
        if idx_a in used_short:
            continue
        mx_a = (float(ln_a["x1"]) + float(ln_a["x2"])) * 0.5
        my_a = (float(ln_a["y1"]) + float(ln_a["y2"])) * 0.5
        cluster = [idx_a]
        for sj in range(si + 1, len(short_lines)):
            idx_b, ln_b = short_lines[sj]
            if idx_b in used_short:
                continue
            mx_b = (float(ln_b["x1"]) + float(ln_b["x2"])) * 0.5
            my_b = (float(ln_b["y1"]) + float(ln_b["y2"])) * 0.5
            dist = math.sqrt((mx_a - mx_b) ** 2 + (my_a - my_b) ** 2)
            if dist <= small_rect_max * 2:
                cluster.append(idx_b)
        # A closed rectangle = exactly 4 lines, 2 pairs of parallel lines
        if len(cluster) == 4:
            angles = []
            for ci in cluster:
                angles.append(_line_angle_deg(remaining[ci]) % 180.0)
            angles.sort()
            # Check for 2 pairs of similar angles (2 horizontal + 2 vertical)
            if (abs(angles[0] - angles[1]) < 10.0 and abs(angles[2] - angles[3]) < 10.0
                    and abs(angles[1] - angles[2]) > 30.0):
                for ci in cluster:
                    rect_indices.add(ci)
                    used_short.add(ci)

    if rect_indices:
        new_remaining = []
        for i, ln in enumerate(remaining):
            if i in rect_indices:
                out["ignored"] += 1
            else:
                new_remaining.append(ln)
        out["unclassified_lines"] = new_remaining

    # --- 2c. Glass pane lines → ignored ---
    # Window glass panes appear as 3+ tightly-spaced parallel lines (< 3cm apart).
    # They create noise in the wall graph and can incorrectly pair with wall lines.
    glass_spacing_max = float(cfg.get("geom_classify_glass_spacing_max_cm", 3.0))
    glass_min_count = int(cfg.get("geom_classify_glass_min_count", 3))
    remaining = list(out["unclassified_lines"])
    glass_indices = set()
    # Group lines by angle and proximity of midpoints
    for i in range(len(remaining)):
        if i in glass_indices:
            continue
        ln_i = remaining[i]
        len_i = _line_len(ln_i)
        if len_i < 10.0:
            continue
        ang_i = _line_angle_deg(ln_i) % 180.0
        mx_i = (float(ln_i["x1"]) + float(ln_i["x2"])) * 0.5
        my_i = (float(ln_i["y1"]) + float(ln_i["y2"])) * 0.5
        cluster = [i]
        for j in range(i + 1, len(remaining)):
            if j in glass_indices:
                continue
            ln_j = remaining[j]
            len_j = _line_len(ln_j)
            # Must be similar length
            if abs(len_j - len_i) > len_i * 0.15:
                continue
            ang_j = _line_angle_deg(ln_j) % 180.0
            if _angle_delta_axis(ang_i, ang_j) > 5.0:
                continue
            mx_j = (float(ln_j["x1"]) + float(ln_j["x2"])) * 0.5
            my_j = (float(ln_j["y1"]) + float(ln_j["y2"])) * 0.5
            dist = math.sqrt((mx_i - mx_j) ** 2 + (my_i - my_j) ** 2)
            if dist <= glass_spacing_max * (len(cluster)):
                cluster.append(j)
        if len(cluster) >= glass_min_count:
            for ci in cluster:
                glass_indices.add(ci)
    if glass_indices:
        new_remaining = []
        for i, ln in enumerate(remaining):
            if i in glass_indices:
                out["ignored"] += 1
            else:
                new_remaining.append(ln)
        out["unclassified_lines"] = new_remaining

    # --- 3. Dashed line detection: collinear short segments with gaps ---
    # Group unclassified lines by angle (within 8 degrees) and proximity
    remaining = list(out["unclassified_lines"])
    dashed_indices = set()
    # Build angle groups
    angle_groups = {}
    for i, ln in enumerate(remaining):
        length = _line_len(ln)
        if length > dashed_seg_max * 3:
            continue  # Only consider short-ish lines as dashed candidates
        ang = round(_line_angle_deg(ln) % 180.0 / 10.0) * 10.0
        angle_groups.setdefault(ang, []).append(i)

    for ang_key, indices in angle_groups.items():
        if len(indices) < 3:
            continue
        group_lines = [remaining[i] for i in indices]
        if _is_dashed_sequence(group_lines, dashed_seg_max, dashed_gap_max, 3):
            for i in indices:
                dashed_indices.add(i)

    if dashed_indices:
        new_remaining = []
        for i, ln in enumerate(remaining):
            if i in dashed_indices:
                out["ignored"] += 1
            else:
                new_remaining.append(ln)
        out["unclassified_lines"] = new_remaining

    # --- 4. Window pattern: clusters of short perpendicular lines near long lines ---
    # Find long lines (potential wall candidates) from all lines
    long_lines = [ln for ln in all_lines if _line_len(ln) > 50.0]
    if not long_lines:
        return

    short_unclassified = [ln for ln in out["unclassified_lines"] if _line_len(ln) <= win_max_line_len]
    if len(short_unclassified) < win_min_parallel:
        return

    window_indices = set()
    unclass_list = list(out["unclassified_lines"])

    for wall_ln in long_lines:
        wall_ang = _line_angle_deg(wall_ln)
        # Find short lines perpendicular to this wall and close to it
        perp_candidates = []
        for i, ln in enumerate(unclass_list):
            if i in window_indices:
                continue
            length = _line_len(ln)
            if length > win_max_line_len or length < 1.0:
                continue
            ang = _line_angle_deg(ln)
            # Check perpendicularity (within 20 degrees of 90)
            delta = _angle_delta_axis(ang, wall_ang)
            if abs(delta - 90.0) > 20.0:
                continue
            # Check proximity to wall line
            mx = (float(ln["x1"]) + float(ln["x2"])) * 0.5
            my = (float(ln["y1"]) + float(ln["y2"])) * 0.5
            wx1, wy1 = float(wall_ln["x1"]), float(wall_ln["y1"])
            wx2, wy2 = float(wall_ln["x2"]), float(wall_ln["y2"])
            # Distance from midpoint to wall line segment
            vx, vy = wx2 - wx1, wy2 - wy1
            vv = vx * vx + vy * vy
            if vv < 1e-12:
                continue
            t = max(0.0, min(1.0, ((mx - wx1) * vx + (my - wy1) * vy) / vv))
            cx, cy = wx1 + vx * t, wy1 + vy * t
            dist = math.sqrt((mx - cx) ** 2 + (my - cy) ** 2)
            if dist > win_spacing_max * 2:
                continue
            # Project midpoint onto wall axis for clustering
            wall_len = math.sqrt(vv)
            proj_t = ((mx - wx1) * vx + (my - wy1) * vy) / vv * wall_len
            perp_candidates.append({"idx": i, "proj": proj_t, "dist": dist})

        if len(perp_candidates) < win_min_parallel:
            continue

        # Cluster perpendicular lines by projection along wall axis
        perp_candidates.sort(key=lambda c: c["proj"])
        clusters = [[perp_candidates[0]]]
        for k in range(1, len(perp_candidates)):
            gap = perp_candidates[k]["proj"] - perp_candidates[k - 1]["proj"]
            if gap <= win_spacing_max:
                clusters[-1].append(perp_candidates[k])
            else:
                clusters.append([perp_candidates[k]])

        for cluster in clusters:
            if len(cluster) >= win_min_parallel:
                span = cluster[-1]["proj"] - cluster[0]["proj"]
                if 5.0 <= span <= 200.0:
                    for c in cluster:
                        window_indices.add(c["idx"])

    if window_indices:
        new_unclass = []
        for i, ln in enumerate(unclass_list):
            if i in window_indices:
                out["window_lines"].append(ln)
            else:
                new_unclass.append(ln)
        out["unclassified_lines"] = new_unclass


def classify_entities(lines, arcs, layer_map, cfg=None):
    wall_rx = _compile_regex((layer_map or {}).get("walls", []))
    door_rx = _compile_regex((layer_map or {}).get("doors", []))
    win_rx = _compile_regex((layer_map or {}).get("windows", []))
    dim_rx = _compile_regex((layer_map or {}).get("dimensions", []))
    ign_rx = _compile_regex((layer_map or {}).get("ignore", []))

    out = {
        "wall_lines": [],
        "door_lines": [],
        "window_lines": [],
        "door_arcs": [],
        "window_arcs": [],
        "dimension_lines": [],
        "dimension_arcs": [],
        "unclassified_lines": [],
        "unclassified_arcs": [],
        "ignored": 0,
    }

    for ln in (lines or []):
        layer = ln.get("layer", "")
        if _match_any(layer, dim_rx):
            out["dimension_lines"].append(ln)
            continue
        if _match_any(layer, ign_rx):
            out["ignored"] += 1
            continue
        if _match_any(layer, wall_rx):
            out["wall_lines"].append(ln)
            continue
        if _match_any(layer, door_rx):
            out["door_lines"].append(ln)
            continue
        if _match_any(layer, win_rx):
            out["window_lines"].append(ln)
            continue
        out["unclassified_lines"].append(ln)

    for arc in (arcs or []):
        layer = arc.get("layer", "")
        if _match_any(layer, dim_rx):
            out["dimension_arcs"].append(arc)
            continue
        if _match_any(layer, ign_rx):
            out["ignored"] += 1
            continue
        if _match_any(layer, door_rx):
            out["door_arcs"].append(arc)
            continue
        if _match_any(layer, win_rx):
            out["window_arcs"].append(arc)
            continue
        out["unclassified_arcs"].append(arc)

    # Geometry-based reclassification for unlabeled entities
    all_lines = list(lines or [])
    _geometric_reclassify(out, cfg, all_lines)

    # Permissive defaults: unlabeled arcs are door candidates.
    if (not out["door_arcs"]) and out["unclassified_arcs"]:
        out["door_arcs"] = list(out["unclassified_arcs"])

    # Treat unlabeled lines as potential walls (recognizer will decide).
    out["all_line_candidates"] = list(out["wall_lines"]) + list(out["unclassified_lines"])
    return out
