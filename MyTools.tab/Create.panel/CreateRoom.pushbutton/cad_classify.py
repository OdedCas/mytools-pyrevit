# -*- coding: utf-8 -*-
"""Layer-first CAD entity classification with strict fallbacks."""

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


def classify_entities(lines, arcs, layer_map):
    wall_rx = _compile_regex((layer_map or {}).get("walls", []))
    door_rx = _compile_regex((layer_map or {}).get("doors", []))
    win_rx = _compile_regex((layer_map or {}).get("windows", []))
    ign_rx = _compile_regex((layer_map or {}).get("ignore", []))

    out = {
        "wall_lines": [],
        "door_lines": [],
        "window_lines": [],
        "door_arcs": [],
        "window_arcs": [],
        "unclassified_lines": [],
        "unclassified_arcs": [],
        "ignored": 0,
        "fallback": {
            "door_arcs_from_unclassified": 0,
            "window_lines_from_geometry": 0,
        },
    }

    for ln in (lines or []):
        layer = ln.get("layer", "")
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

    # Strict fallback: unlabeled arcs are treated as potential doors.
    if (not out["door_arcs"]) and out["unclassified_arcs"]:
        out["door_arcs"].extend(out["unclassified_arcs"])
        out["fallback"]["door_arcs_from_unclassified"] = len(out["unclassified_arcs"])

    return out
