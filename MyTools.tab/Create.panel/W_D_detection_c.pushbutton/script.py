# -*- coding: utf-8 -*-
__title__ = "W_D_detection_c"
__doc__ = "Detect doors and windows in an imported DWG by layer and mark with W/D text."

import os
import sys
import math
import re
import clr

from Autodesk.Revit.DB import (
    FilteredElementCollector,
    GeometryInstance,
    ImportInstance,
    Line as RvtLine,
    Arc as RvtArc,
    PolyLine as RvtPolyLine,
    Options,
    TextNote,
    TextNoteType,
    Transaction,
    ViewPlan,
    XYZ,
)
from Autodesk.Revit.UI import TaskDialog
from Autodesk.Revit.UI.Selection import ObjectType

uidoc = __revit__.ActiveUIDocument
doc = uidoc.Document

CM_PER_FT = 30.48
TEXT_OFFSET_CM = 25.0
CLUSTER_DIST_CM = 80.0  # geometry closer than this = same opening

DOOR_RX = re.compile(r"door", re.IGNORECASE)
WINDOW_RX = re.compile(r"wind|window|glazing", re.IGNORECASE)


def ft_to_cm(v):
    return float(v) * CM_PER_FT


def cm_to_ft(v):
    return float(v) / CM_PER_FT


def _get_text_note_type():
    collector = FilteredElementCollector(doc).OfClass(TextNoteType)
    for t in collector:
        return t.Id
    return None


def _safe_layer_name(geom_obj):
    try:
        gs_id = geom_obj.GraphicsStyleId
        if gs_id and gs_id.IntegerValue != -1:
            gs = doc.GetElement(gs_id)
            if gs and gs.GraphicsStyleCategory:
                return gs.GraphicsStyleCategory.Name or ""
    except Exception:
        pass
    return ""


def _compose_tf(parent_tf, child_tf):
    if parent_tf is None:
        return child_tf
    if child_tf is None:
        return parent_tf
    try:
        return parent_tf.Multiply(child_tf)
    except Exception:
        return child_tf


def _apply_tf(pt, tf):
    if tf is None:
        return pt
    try:
        return tf.OfPoint(pt)
    except Exception:
        return pt


def _collect_geometry_by_layer(doc, view, import_instance):
    """Walk all geometry and collect points grouped by layer name.
    Returns dict: layer_name -> list of (x_cm, y_cm) points."""
    opts = Options()
    opts.View = view
    opts.IncludeNonVisibleObjects = False

    geom = import_instance.get_Geometry(opts)
    if geom is None:
        return {}

    layer_points = {}

    def _walk(geom_enum, tf):
        for obj in geom_enum:
            if isinstance(obj, GeometryInstance):
                nxt_tf = _compose_tf(tf, obj.Transform)
                try:
                    _walk(obj.GetInstanceGeometry(), nxt_tf)
                except Exception:
                    pass
                continue

            layer = _safe_layer_name(obj)
            if not layer:
                continue

            points = []
            try:
                if isinstance(obj, RvtLine):
                    p1 = _apply_tf(obj.GetEndPoint(0), tf)
                    p2 = _apply_tf(obj.GetEndPoint(1), tf)
                    points.append((ft_to_cm(p1.X), ft_to_cm(p1.Y)))
                    points.append((ft_to_cm(p2.X), ft_to_cm(p2.Y)))
                elif isinstance(obj, RvtArc):
                    center = _apply_tf(obj.Center, tf)
                    points.append((ft_to_cm(center.X), ft_to_cm(center.Y)))
                    s = _apply_tf(obj.GetEndPoint(0), tf)
                    e = _apply_tf(obj.GetEndPoint(1), tf)
                    points.append((ft_to_cm(s.X), ft_to_cm(s.Y)))
                    points.append((ft_to_cm(e.X), ft_to_cm(e.Y)))
                elif isinstance(obj, RvtPolyLine):
                    pts = obj.GetCoordinates()
                    for pt in pts:
                        p = _apply_tf(pt, tf)
                        points.append((ft_to_cm(p.X), ft_to_cm(p.Y)))
            except Exception:
                pass

            if points:
                if layer not in layer_points:
                    layer_points[layer] = []
                layer_points[layer].extend(points)

    _walk(geom, None)
    return layer_points


def _dist_2d(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _cluster_points(points, max_dist):
    """Cluster points by proximity. Returns list of lists of point indices."""
    n = len(points)
    assigned = [False] * n
    clusters = []
    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        queue = [i]
        while queue:
            cur = queue.pop(0)
            cx, cy = points[cur]
            for j in range(n):
                if assigned[j]:
                    continue
                if _dist_2d(cx, cy, points[j][0], points[j][1]) <= max_dist:
                    assigned[j] = True
                    cluster.append(j)
                    queue.append(j)
        clusters.append(cluster)
    return clusters


def _centroid(points):
    if not points:
        return None
    sx = 0.0
    sy = 0.0
    for x, y in points:
        sx += x
        sy += y
    n = float(len(points))
    return (sx / n, sy / n)


def main():
    view = doc.ActiveView
    if not isinstance(view, ViewPlan):
        TaskDialog.Show("W_D_detection_c", "Please run this tool from a plan view.")
        return

    text_type_id = _get_text_note_type()
    if text_type_id is None:
        TaskDialog.Show("W_D_detection_c", "No TextNoteType found in the document.")
        return

    try:
        ref = uidoc.Selection.PickObject(
            ObjectType.Element,
            "Select the imported DWG to analyze"
        )
    except Exception:
        return

    picked_el = doc.GetElement(ref.ElementId)
    if not isinstance(picked_el, ImportInstance):
        TaskDialog.Show("W_D_detection_c", "Selected element is not a CAD import.")
        return

    # Collect all geometry points grouped by layer
    layer_points = _collect_geometry_by_layer(doc, view, picked_el)

    if not layer_points:
        TaskDialog.Show("W_D_detection_c", "No geometry found in DWG.")
        return

    # Find door and window layers
    door_layers = []
    window_layers = []
    all_layers = sorted(layer_points.keys())

    for lyr in all_layers:
        if DOOR_RX.search(lyr):
            door_layers.append(lyr)
        elif WINDOW_RX.search(lyr):
            window_layers.append(lyr)

    # Collect all door points and window points
    door_points = []
    for lyr in door_layers:
        door_points.extend(layer_points[lyr])

    window_points = []
    for lyr in window_layers:
        window_points.extend(layer_points[lyr])

    # Cluster into individual openings
    door_clusters = _cluster_points(door_points, CLUSTER_DIST_CM) if door_points else []
    window_clusters = _cluster_points(window_points, CLUSTER_DIST_CM) if window_points else []

    markers = []
    for cluster in door_clusters:
        pts = [door_points[i] for i in cluster]
        c = _centroid(pts)
        if c:
            markers.append({"type": "door", "cx": c[0], "cy": c[1]})

    for cluster in window_clusters:
        pts = [window_points[i] for i in cluster]
        c = _centroid(pts)
        if c:
            markers.append({"type": "window", "cx": c[0], "cy": c[1]})

    if not markers:
        TaskDialog.Show(
            "W_D_detection_c",
            "No doors or windows detected.\n\n"
            "All layers in DWG:\n%s\n\n"
            "Door layers matched: %s\n"
            "Window layers matched: %s" % (
                "\n".join(all_layers),
                ", ".join(door_layers) if door_layers else "(none)",
                ", ".join(window_layers) if window_layers else "(none)",
            )
        )
        return

    door_count = 0
    window_count = 0

    t = Transaction(doc, "W_D_detection_c - Mark Openings")
    t.Start()
    try:
        for m in markers:
            px = cm_to_ft(m["cx"])
            py = cm_to_ft(m["cy"] - TEXT_OFFSET_CM)
            pos = XYZ(px, py, 0.0)

            if m["type"] == "door":
                TextNote.Create(doc, view.Id, pos, "DD", text_type_id)
                door_count += 1
            else:
                TextNote.Create(doc, view.Id, pos, "WW", text_type_id)
                window_count += 1

        t.Commit()
    except Exception as ex:
        t.RollBack()
        TaskDialog.Show("W_D_detection_c", "Error: %s" % str(ex))
        return

    summary = (
        "Doors (D): %d\nWindows (W): %d\n\n"
        "Door layers: %s\nWindow layers: %s\n\n"
        "All layers: %s\n\n"
        "Door points: %d, Window points: %d"
    ) % (
        door_count, window_count,
        ", ".join(door_layers), ", ".join(window_layers),
        ", ".join(all_layers),
        len(door_points), len(window_points),
    )
    TaskDialog.Show("W_D_detection_c - Results", summary)


main()
