# -*- coding: utf-8 -*-
__title__ = "Create From CAD V2"
__doc__ = "Read DWG/DXF linework, recognize walls/windows/doors, and build a Revit model."

import os
import sys
import math
import clr

clr.AddReference("System.Windows.Forms")
import System.Windows.Forms as WinForms

from System.Windows.Forms import (
    OpenFileDialog,
    DialogResult,
    Form,
    Label,
    Button,
    FormBorderStyle,
)

from Autodesk.Revit.DB import (
    BuiltInCategory,
    BuiltInParameter,
    CurveLoop,
    ElementId,
    FamilySymbol,
    FilteredElementCollector,
    Floor,
    FloorType,
    Level,
    Line,
    Structure,
    Transaction,
    ViewPlan,
    ViewType,
    Wall,
    WallKind,
    WallType,
    DWGImportOptions,
    XYZ,
)
from Autodesk.Revit.UI import TaskDialog
from Autodesk.Revit.Exceptions import OperationCanceledException

from v2_snapshot import SnapshotRun
from v2_cad_extract import (
    load_config,
    load_layer_map,
    get_imported_cad_instances,
    extract_cad_from_view,
)
from v2_cad_classify import classify_entities
from v2_cad_recognition import recognize_topology


uidoc = __revit__.ActiveUIDocument
doc = uidoc.Document

CM_PER_FT = 30.48


def cm_to_ft(v):
    return float(v) / CM_PER_FT


def ft_to_cm(v):
    return float(v) * CM_PER_FT


def first_or_none(iterable):
    for it in iterable:
        return it
    return None


def set_form_center(form):
    try:
        form.StartPosition = WinForms.FormStartPosition.CenterScreen
    except Exception:
        pass


def choose_input_kind():
    form = Form()
    form.Text = "Choose Source"
    form.Width = 460
    form.Height = 170
    form.FormBorderStyle = FormBorderStyle.FixedDialog
    set_form_center(form)
    form.MinimizeBox = False
    form.MaximizeBox = False

    lbl = Label()
    lbl.Left = 12
    lbl.Top = 12
    lbl.Width = 430
    lbl.Height = 22
    lbl.Text = "Select input source:"
    form.Controls.Add(lbl)

    btn_cad = Button()
    btn_cad.Text = "CAD DWG/DXF"
    btn_cad.Left = 28
    btn_cad.Top = 56
    btn_cad.Width = 180

    btn_scan = Button()
    btn_scan.Text = "Scan / Sketch"
    btn_scan.Left = 236
    btn_scan.Top = 56
    btn_scan.Width = 180

    choice = {"kind": None}

    def on_cad(sender, args):
        choice["kind"] = "cad"
        form.DialogResult = DialogResult.OK
        form.Close()

    def on_scan(sender, args):
        choice["kind"] = "scan"
        form.DialogResult = DialogResult.OK
        form.Close()

    btn_cad.Click += on_cad
    btn_scan.Click += on_scan
    form.Controls.Add(btn_cad)
    form.Controls.Add(btn_scan)

    if form.ShowDialog() != DialogResult.OK:
        return None
    return choice["kind"]


def choose_cad_source_mode(has_existing):
    form = Form()
    form.Text = "Choose CAD Source"
    form.Width = 520
    form.Height = 200
    form.FormBorderStyle = FormBorderStyle.FixedDialog
    set_form_center(form)
    form.MinimizeBox = False
    form.MaximizeBox = False

    lbl = Label()
    lbl.Left = 12
    lbl.Top = 12
    lbl.Width = 480
    lbl.Height = 44
    if has_existing:
        lbl.Text = "Use existing imported CAD in this view, or import a new DWG/DXF."
    else:
        lbl.Text = "No imported CAD found in this view. Import a new DWG/DXF file."
    form.Controls.Add(lbl)

    btn_existing = Button()
    btn_existing.Text = "Use Existing in View"
    btn_existing.Left = 28
    btn_existing.Top = 84
    btn_existing.Width = 200
    btn_existing.Enabled = bool(has_existing)

    btn_import = Button()
    btn_import.Text = "Import DWG/DXF File"
    btn_import.Left = 260
    btn_import.Top = 84
    btn_import.Width = 200

    choice = {"source": None}

    def on_existing(sender, args):
        choice["source"] = "existing"
        form.DialogResult = DialogResult.OK
        form.Close()

    def on_import(sender, args):
        choice["source"] = "import"
        form.DialogResult = DialogResult.OK
        form.Close()

    btn_existing.Click += on_existing
    btn_import.Click += on_import

    form.Controls.Add(btn_existing)
    form.Controls.Add(btn_import)

    if form.ShowDialog() != DialogResult.OK:
        return None
    return choice["source"]


def choose_post_action():
    form = Form()
    form.Text = "DWG After Build"
    form.Width = 500
    form.Height = 190
    form.FormBorderStyle = FormBorderStyle.FixedDialog
    set_form_center(form)
    form.MinimizeBox = False
    form.MaximizeBox = False

    lbl = Label()
    lbl.Left = 12
    lbl.Top = 12
    lbl.Width = 460
    lbl.Height = 44
    lbl.Text = "After model generation, keep imported CAD or replace it with model?"
    form.Controls.Add(lbl)

    btn_keep = Button()
    btn_keep.Text = "Keep DWG + Model"
    btn_keep.Left = 28
    btn_keep.Top = 84
    btn_keep.Width = 200

    btn_replace = Button()
    btn_replace.Text = "Replace DWG with Model"
    btn_replace.Left = 260
    btn_replace.Top = 84
    btn_replace.Width = 200

    choice = {"mode": None}

    def on_keep(sender, args):
        choice["mode"] = "keep"
        form.DialogResult = DialogResult.OK
        form.Close()

    def on_replace(sender, args):
        choice["mode"] = "replace"
        form.DialogResult = DialogResult.OK
        form.Close()

    btn_keep.Click += on_keep
    btn_replace.Click += on_replace

    form.Controls.Add(btn_keep)
    form.Controls.Add(btn_replace)

    if form.ShowDialog() != DialogResult.OK:
        return None
    return choice["mode"]


def pick_cad_path():
    dialog = OpenFileDialog()
    dialog.Filter = "CAD Files|*.dwg;*.dxf|DWG Files|*.dwg|DXF Files|*.dxf"
    dialog.Multiselect = False
    dialog.Title = "Select CAD file to import"
    if dialog.ShowDialog() == DialogResult.OK:
        return dialog.FileName
    return None


def get_plan_view():
    active = uidoc.ActiveView
    if isinstance(active, ViewPlan) and (not active.IsTemplate) and active.ViewType == ViewType.FloorPlan:
        return active
    return None


def get_level_from_view(plan_view):
    if plan_view and hasattr(plan_view, "GenLevel") and plan_view.GenLevel:
        return plan_view.GenLevel
    return first_or_none(FilteredElementCollector(doc).OfClass(Level))


def get_floor_type():
    return first_or_none(FilteredElementCollector(doc).OfCategory(BuiltInCategory.OST_Floors).OfClass(FloorType))


def get_wall_type_nearest(thickness_ft):
    wall_types = [wt for wt in FilteredElementCollector(doc).OfClass(WallType) if wt.Kind == WallKind.Basic]
    if not wall_types:
        return None
    return min(wall_types, key=lambda wt: abs(wt.Width - thickness_ft))


def set_param(element, bip, value):
    if element is None:
        return False
    p = element.get_Parameter(bip)
    if p and (not p.IsReadOnly):
        p.Set(value)
        return True
    return False


def set_param_by_names(element, names, value):
    if element is None:
        return False
    for name in names:
        try:
            p = element.LookupParameter(name)
            if p and (not p.IsReadOnly):
                p.Set(value)
                return True
        except Exception:
            continue
    return False


def set_opening_size(instance, width_ft, height_ft, width_bip, height_bip):
    w_ok = set_param(instance, width_bip, width_ft)
    h_ok = set_param(instance, height_bip, height_ft)
    if not w_ok:
        set_param_by_names(instance, ["Width", "width", "Rough Width"], width_ft)
    if not h_ok:
        set_param_by_names(instance, ["Height", "height", "Rough Height"], height_ft)


def get_symbol(category):
    return first_or_none(FilteredElementCollector(doc).OfCategory(category).OfClass(FamilySymbol))


def _symbol_display_name(sym):
    try:
        fam = sym.Family.Name
    except Exception:
        fam = ""
    try:
        typ = sym.Name
    except Exception:
        typ = ""
    if fam and typ:
        return "{} : {}".format(fam, typ)
    return fam or typ or "<unnamed>"


def _symbol_width_ft(sym):
    for name in ["Width", "width", "Rough Width"]:
        try:
            p = sym.LookupParameter(name)
            if p and p.HasValue:
                return p.AsDouble()
        except Exception:
            pass
    for bip in [BuiltInParameter.DOOR_WIDTH, BuiltInParameter.WINDOW_WIDTH,
                BuiltInParameter.GENERIC_WIDTH]:
        try:
            p = sym.get_Parameter(bip)
            if p and p.HasValue:
                return p.AsDouble()
        except Exception:
            pass
    return None


def _symbol_has_visible_bbox(sym):
    """Best-effort visibility check. None means unknown, not a hard fail."""
    try:
        bb = sym.get_BoundingBox(None)
        if bb is None:
            return None
        dx = abs(float(bb.Max.X) - float(bb.Min.X))
        dy = abs(float(bb.Max.Y) - float(bb.Min.Y))
        dz = abs(float(bb.Max.Z) - float(bb.Min.Z))
        return (dx > 1.0e-6) or (dy > 1.0e-6) or (dz > 1.0e-6)
    except Exception:
        return None


def get_symbol_by_width(category, target_width_ft, cfg=None, audit=None):
    """Pick a family symbol by width while filtering non-visual opening-only types."""
    cfg = cfg or {}
    reject_patterns = list(cfg.get("opening_symbol_reject_name_patterns", ["opening"]) or ["opening"])
    reject_patterns = [str(p).strip().lower() for p in reject_patterns if str(p).strip()]
    require_visible = bool(cfg.get("opening_symbol_require_visible_geometry", True))

    symbols = list(FilteredElementCollector(doc).OfCategory(category).OfClass(FamilySymbol))
    if not symbols:
        return None

    ranked = []
    for sym in symbols:
        width_ft = _symbol_width_ft(sym)
        if width_ft is None:
            diff = 1.0e9
        else:
            diff = abs(width_ft - float(target_width_ft))
        ranked.append((diff, sym, width_ft))
    ranked.sort(key=lambda x: x[0])

    fallback = ranked[0][1]
    for diff, sym, width_ft in ranked:
        name = _symbol_display_name(sym)
        lname = name.lower()
        blocked = None
        for pat in reject_patterns:
            if pat and (pat in lname):
                blocked = "name_pattern:{}".format(pat)
                break
        if blocked is None and require_visible:
            vis = _symbol_has_visible_bbox(sym)
            if vis is False:
                blocked = "no_visible_geometry"

        if blocked is not None:
            if audit is not None:
                audit.append({
                    "symbol": name,
                    "reason": blocked,
                    "width_ft": width_ft,
                    "target_width_ft": float(target_width_ft),
                })
            continue
        return sym

    # If all were rejected, use nearest-width fallback to avoid total failure.
    if audit is not None and fallback is not None:
        audit.append({
            "symbol": _symbol_display_name(fallback),
            "reason": "fallback_after_all_rejected",
            "width_ft": _symbol_width_ft(fallback),
            "target_width_ft": float(target_width_ft),
        })
    return fallback


def build_loop(points):
    loop = CurveLoop()
    n = len(points)
    for i in range(n):
        loop.Append(Line.CreateBound(points[i], points[(i + 1) % n]))
    return loop


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
    if ln <= 1.0e-9:
        return 0.0
    ux = vx / ln
    uy = vy / ln
    return ((px - ax) * ux) + ((py - ay) * uy)


def _edge_list_from_poly_cm(poly_cm):
    out = []
    n = len(poly_cm)
    for i in range(n):
        a = poly_cm[i]
        b = poly_cm[(i + 1) % n]
        dx = float(b[0]) - float(a[0])
        dy = float(b[1]) - float(a[1])
        ln = math.sqrt((dx * dx) + (dy * dy))
        if ln <= 1.0e-9:
            continue
        out.append({"idx": i, "a": a, "b": b, "len": ln})
    return out


def _angle_delta_axis_deg(a, b):
    d = abs((float(a) - float(b)) % 360.0)
    if d > 180.0:
        d = 360.0 - d
    if d > 90.0:
        d = 180.0 - d
    return d


def _build_wall_runs_from_polygon(poly_cm, cfg):
    merge_tol_deg = float(cfg.get("model_wall_merge_angle_deg", 2.0))
    join_tol_cm = float(cfg.get("model_wall_join_tol_cm", 1.0))
    jog_skip_cm = float(cfg.get("model_wall_jog_skip_cm", 20.0))
    jog_bridge_join_tol_cm = float(cfg.get("model_wall_jog_bridge_join_tol_cm", 12.0))

    edges = []
    n = len(poly_cm)
    for i in range(n):
        a = poly_cm[i]
        b = poly_cm[(i + 1) % n]
        dx = float(b[0]) - float(a[0])
        dy = float(b[1]) - float(a[1])
        ln = math.sqrt((dx * dx) + (dy * dy))
        if ln <= 1.0e-9:
            continue
        edges.append({
            "idx": i,
            "a": (float(a[0]), float(a[1])),
            "b": (float(b[0]), float(b[1])),
            "len_cm": ln,
            "dir_x": dx / ln,
            "dir_y": dy / ln,
            "ang": math.degrees(math.atan2(dy, dx)),
        })

    runs = []
    edge_to_run = {}
    skipped_short_since_last = False

    for e in edges:
        if float(e.get("len_cm", 0.0)) < jog_skip_cm:
            if runs:
                edge_to_run[e["idx"]] = {
                    "run_idx": len(runs) - 1,
                    "offset_cm": max(0.0, float(runs[-1]["len_cm"])),
                    "len_cm": e["len_cm"],
                    "from_short_jog": True,
                }
            skipped_short_since_last = True
            continue

        if not runs:
            runs.append({
                "start": e["a"],
                "end": e["b"],
                "len_cm": e["len_cm"],
                "source_edges": [{
                    "edge_idx": e["idx"],
                    "offset_cm": 0.0,
                    "len_cm": e["len_cm"],
                }],
            })
            edge_to_run[e["idx"]] = {"run_idx": 0, "offset_cm": 0.0, "len_cm": e["len_cm"]}
            continue

        last = runs[-1]
        lx2, ly2 = last["end"]
        ex1, ey1 = e["a"]
        join_dist = math.sqrt(((lx2 - ex1) ** 2) + ((ly2 - ey1) ** 2))

        ldx = last["end"][0] - last["start"][0]
        ldy = last["end"][1] - last["start"][1]
        lln = math.sqrt((ldx * ldx) + (ldy * ldy))
        if lln <= 1.0e-9:
            ldir_x = e["dir_x"]
            ldir_y = e["dir_y"]
            lang = e["ang"]
        else:
            ldir_x = ldx / lln
            ldir_y = ldy / lln
            lang = math.degrees(math.atan2(ldy, ldx))

        dot = (ldir_x * e["dir_x"]) + (ldir_y * e["dir_y"])
        ang_ok = _angle_delta_axis_deg(lang, e["ang"]) <= merge_tol_deg
        forward_ok = dot > 0.2
        effective_join_tol_cm = jog_bridge_join_tol_cm if skipped_short_since_last else join_tol_cm

        if join_dist <= effective_join_tol_cm and ang_ok and forward_ok:
            offset_cm = float(last["len_cm"])
            last["end"] = e["b"]
            last["len_cm"] = float(last["len_cm"]) + float(e["len_cm"])
            last["source_edges"].append({
                "edge_idx": e["idx"],
                "offset_cm": offset_cm,
                "len_cm": e["len_cm"],
            })
            edge_to_run[e["idx"]] = {"run_idx": len(runs) - 1, "offset_cm": offset_cm, "len_cm": e["len_cm"]}
        else:
            run_idx = len(runs)
            runs.append({
                "start": e["a"],
                "end": e["b"],
                "len_cm": e["len_cm"],
                "source_edges": [{
                    "edge_idx": e["idx"],
                    "offset_cm": 0.0,
                    "len_cm": e["len_cm"],
                }],
            })
            edge_to_run[e["idx"]] = {"run_idx": run_idx, "offset_cm": 0.0, "len_cm": e["len_cm"]}
        skipped_short_since_last = False

    return {
        "edges": edges,
        "runs": runs,
        "edge_to_run": edge_to_run,
    }


def _project_point_to_wall_axis(wall, xyz):
    try:
        loc = getattr(wall, "Location", None)
        crv = getattr(loc, "Curve", None)
        if crv is not None:
            res = crv.Project(xyz)
            if res is not None and getattr(res, "XYZPoint", None) is not None:
                q = res.XYZPoint
                return XYZ(q.X, q.Y, xyz.Z)
    except Exception:
        pass
    return xyz


def _pt_seg_dist_cm(px, py, ax, ay, bx, by):
    return _dist_pt_seg(px, py, ax, ay, bx, by)[0]


def _safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def _nearest_wall_index_for_point_cm(px_cm, py_cm, wall_meta, max_dist_cm=None):
    best_idx = None
    best_dist = None
    for idx, wm in enumerate(wall_meta):
        d = _pt_seg_dist_cm(
            px_cm,
            py_cm,
            ft_to_cm(wm["center_start"].X),
            ft_to_cm(wm["center_start"].Y),
            ft_to_cm(wm["center_end"].X),
            ft_to_cm(wm["center_end"].Y),
        )
        if best_idx is None or d < best_dist:
            best_idx = idx
            best_dist = d
    if best_idx is None:
        return None, None
    if max_dist_cm is not None and best_dist is not None and best_dist > float(max_dist_cm):
        return None, best_dist
    return best_idx, best_dist


def _manual_pick_opening_pairs(kind):
    pairs = []
    while True:
        try:
            p1 = uidoc.Selection.PickPoint("Pick {} LEFT edge (ESC to finish {})".format(kind.upper(), kind.upper()))
        except OperationCanceledException:
            break
        p2 = uidoc.Selection.PickPoint("Pick {} RIGHT edge".format(kind.upper()))
        pairs.append((p1, p2))
    return pairs


def _convert_pair_to_opening(p1_ft, p2_ft, kind, edges, cfg):
    p1 = (ft_to_cm(p1_ft.X), ft_to_cm(p1_ft.Y))
    p2 = (ft_to_cm(p2_ft.X), ft_to_cm(p2_ft.Y))
    mx = (p1[0] + p2[0]) * 0.5
    my = (p1[1] + p2[1]) * 0.5

    best = None
    best_dist = None
    for e in edges:
        d, _ = _dist_pt_seg(mx, my, float(e["a"][0]), float(e["a"][1]), float(e["b"][0]), float(e["b"][1]))
        if best is None or d < best_dist:
            best = e
            best_dist = d
    if best is None:
        return None

    t1 = _project_t_cm(p1[0], p1[1], float(best["a"][0]), float(best["a"][1]), float(best["b"][0]), float(best["b"][1]))
    t2 = _project_t_cm(p2[0], p2[1], float(best["a"][0]), float(best["a"][1]), float(best["b"][0]), float(best["b"][1]))
    s = max(0.0, min(t1, t2))
    t = min(float(best["len"]), max(t1, t2))
    width = max(1.0, t - s)

    op = {
        "type": kind,
        "host_edge": int(best["idx"]),
        "start_cm": s,
        "end_cm": t,
        "width_cm": width,
        "confidence": 1.0,
        "manual": True,
    }
    if kind == "door":
        op["height_cm"] = float(cfg.get("default_door_height_cm", 210.0))
    elif kind == "window":
        op["height_cm"] = float(cfg.get("default_window_height_cm", 100.0))
        op["sill_cm"] = float(cfg.get("default_window_sill_cm", 105.0))
    return op


def maybe_manual_openings(topology, cfg, snapshot):
    openings = list(topology.get("openings") or [])
    door_count = len([o for o in openings if str(o.get("type", "")).lower() == "door"])
    window_count = len([o for o in openings if str(o.get("type", "")).lower() == "window"])

    if door_count >= 1 and window_count >= 1:
        return topology

    poly_cm = list(topology.get("room_polygon_cm") or [])
    edges = _edge_list_from_poly_cm(poly_cm)
    if not edges:
        return topology

    TaskDialog.Show(
        "Create From CAD V2",
        "Auto opening recognition is incomplete (doors={}, windows={}).\\n"
        "You can now pick openings manually on the CAD.\\n"
        "Pick points in pairs (left/right). Press ESC to finish each type.".format(door_count, window_count),
    )

    manual = []
    door_pairs = _manual_pick_opening_pairs("door")
    for p1, p2 in door_pairs:
        op = _convert_pair_to_opening(p1, p2, "door", edges, cfg)
        if op:
            manual.append(op)

    window_pairs = _manual_pick_opening_pairs("window")
    for p1, p2 in window_pairs:
        op = _convert_pair_to_opening(p1, p2, "window", edges, cfg)
        if op:
            manual.append(op)

    if not manual:
        snapshot.log("Manual opening fallback: no manual picks")
        return topology

    keep = []
    if not any([m for m in manual if m["type"] == "door"]):
        keep.extend([o for o in openings if str(o.get("type", "")).lower() == "door"])
    if not any([m for m in manual if m["type"] == "window"]):
        keep.extend([o for o in openings if str(o.get("type", "")).lower() == "window"])

    topology["openings"] = keep + manual
    snapshot.log(
        "Manual opening fallback applied: doors={}, windows={}".format(
            len([o for o in topology["openings"] if o.get("type") == "door"]),
            len([o for o in topology["openings"] if o.get("type") == "window"]),
        )
    )
    return topology


def import_cad_file_to_view(view, cad_path):
    t_imp = Transaction(doc, "Import CAD (V2)")
    try:
        t_imp.Start()
        opts = DWGImportOptions()
        opts.ThisViewOnly = True
        opts.OrientToView = True
        opts.VisibleLayersOnly = True

        imported_ref = clr.Reference[ElementId](ElementId.InvalidElementId)
        ok = doc.Import(cad_path, opts, view, imported_ref)
        if not ok:
            t_imp.RollBack()
            return None, "Revit failed to import CAD file."

        t_imp.Commit()
        imported_id = imported_ref.Value
        inst = doc.GetElement(imported_id) if imported_id else None
        if inst is None:
            return None, "CAD imported but ImportInstance not found."
        return inst, None
    except Exception as ex:
        try:
            if t_imp.HasStarted():
                t_imp.RollBack()
        except Exception:
            pass
        return None, str(ex)


def cleanup_imported_cad_instance(instance_id, snapshot):
    if instance_id is None:
        return
    t_clean = Transaction(doc, "Remove Source CAD (V2)")
    try:
        t_clean.Start()
        elem = doc.GetElement(ElementId(int(instance_id)))
        if elem is not None:
            doc.Delete(elem.Id)
        t_clean.Commit()
        snapshot.log("Removed CAD import instance {}".format(instance_id))
    except Exception as ex:
        try:
            if t_clean.HasStarted():
                t_clean.RollBack()
        except Exception:
            pass
        snapshot.log("Failed removing CAD {}: {}".format(instance_id, ex))


def build_model_from_topology(level, topology, cfg, snapshot):
    poly_cm = list(topology.get("room_polygon_cm") or [])
    if len(poly_cm) < 3:
        raise Exception("Invalid recognized room polygon")

    wall_thickness_cm = float(topology.get("measurements_cm", {}).get("wall_thickness_cm", cfg.get("default_wall_thickness_cm", 20.0)))
    wall_type = get_wall_type_nearest(cm_to_ft(wall_thickness_cm))
    floor_type = get_floor_type()
    if wall_type is None or floor_type is None:
        raise Exception("Missing WallType or FloorType in project")

    area2 = 0.0
    for i in range(len(poly_cm)):
        x1, y1 = poly_cm[i]
        x2, y2 = poly_cm[(i + 1) % len(poly_cm)]
        area2 += (x1 * y2) - (x2 * y1)
    if abs(area2) <= 1e-6:
        raise Exception("Recognized polygon has near-zero area")

    inner_pts = [XYZ(cm_to_ft(p[0]), cm_to_ft(p[1]), 0.0) for p in poly_cm]
    wall_runs_data = _build_wall_runs_from_polygon(poly_cm, cfg)
    wall_runs = list(wall_runs_data.get("runs") or [])
    edge_to_run = dict(wall_runs_data.get("edge_to_run") or {})

    wall_height_ft = cm_to_ft(300.0)
    half = wall_type.Width * 0.5
    min_wall_len_cm = float(cfg.get("model_wall_min_length_cm", 20.0))
    create_roof = bool(cfg.get("model_create_roof", False))
    skip_low_conf = bool(cfg.get("model_skip_low_confidence_openings", False))
    opening_host_fallback_max_cm = float(cfg.get("opening_host_fallback_max_dist_cm", 180.0))

    walls = []
    wall_meta = []
    run_to_wall_idx = {}
    door_ids = []
    window_ids = []

    t_geo = Transaction(doc, "Create Model From CAD V2")
    t_geo.Start()
    try:
        # Build perimeter walls from merged collinear runs to avoid wall slicing.
        for run_idx, run in enumerate(wall_runs):
            if float(run.get("len_cm", 0.0)) < min_wall_len_cm:
                continue

            p0 = XYZ(cm_to_ft(run["start"][0]), cm_to_ft(run["start"][1]), 0.0)
            p1 = XYZ(cm_to_ft(run["end"][0]), cm_to_ft(run["end"][1]), 0.0)
            dx = p1.X - p0.X
            dy = p1.Y - p0.Y
            ln = math.sqrt((dx * dx) + (dy * dy))
            if ln <= 1e-9:
                continue

            if area2 > 0.0:
                ox = dy / ln
                oy = -dx / ln
            else:
                ox = -dy / ln
                oy = dx / ln

            c0 = XYZ(p0.X + (ox * half), p0.Y + (oy * half), 0.0)
            c1 = XYZ(p1.X + (ox * half), p1.Y + (oy * half), 0.0)

            wall = Wall.Create(doc, Line.CreateBound(c0, c1), wall_type.Id, level.Id, wall_height_ft, 0.0, False, False)
            run_to_wall_idx[run_idx] = len(walls)
            walls.append(wall)
            wall_meta.append({
                "run_idx": run_idx,
                "center_start": c0,
                "center_end": c1,
                "dir_x": dx / ln,
                "dir_y": dy / ln,
                "len_ft": ln,
                "len_cm": float(run.get("len_cm", 0.0)),
            })

        if len(walls) < 3:
            raise Exception("Failed to create enough walls")

        # Create internal partition walls (detected centerline edges inside the polygon)
        internal_walls_cm = list(topology.get("internal_walls_cm") or [])
        internal_wall_ids = []
        internal_wall_rejected = 0
        internal_wall_errors = []
        min_internal_len_cm = float(cfg.get("internal_wall_min_length_cm", 30.0))
        perimeter_dup_tol_cm = float(cfg.get("internal_wall_perimeter_duplicate_tol_cm", 12.0))
        perimeter_parallel_tol_cm = float(cfg.get("internal_wall_perimeter_parallel_tol_cm", perimeter_dup_tol_cm))
        perimeter_parallel_angle_deg = float(cfg.get("internal_wall_perimeter_parallel_angle_deg", 8.0))
        internal_wall_reject_breakdown = {
            "too_short": 0,
            "near_parallel_perimeter": 0,
            "error": 0,
        }
        for iw in internal_walls_cm:
            try:
                x0 = float(iw[0])
                y0 = float(iw[1])
                x1 = float(iw[2])
                y1 = float(iw[3])
                iw_dx_cm = x1 - x0
                iw_dy_cm = y1 - y0
                iw_len_cm = math.sqrt((iw_dx_cm * iw_dx_cm) + (iw_dy_cm * iw_dy_cm))
                if iw_len_cm < min_internal_len_cm:
                    internal_wall_rejected += 1
                    internal_wall_reject_breakdown["too_short"] += 1
                    continue

                mx = (x0 + x1) * 0.5
                my = (y0 + y1) * 0.5
                iw_ang = math.degrees(math.atan2(iw_dy_cm, iw_dx_cm))
                near_parallel_perimeter = False
                for pm in wall_meta:
                    pm_x0 = ft_to_cm(pm["center_start"].X)
                    pm_y0 = ft_to_cm(pm["center_start"].Y)
                    pm_x1 = ft_to_cm(pm["center_end"].X)
                    pm_y1 = ft_to_cm(pm["center_end"].Y)
                    d = _pt_seg_dist_cm(
                        mx,
                        my,
                        pm_x0,
                        pm_y0,
                        pm_x1,
                        pm_y1,
                    )
                    if d > perimeter_parallel_tol_cm:
                        continue
                    pm_ang = math.degrees(math.atan2(pm_y1 - pm_y0, pm_x1 - pm_x0))
                    if _angle_delta_axis_deg(iw_ang, pm_ang) <= perimeter_parallel_angle_deg:
                        near_parallel_perimeter = True
                        break
                if near_parallel_perimeter:
                    internal_wall_rejected += 1
                    internal_wall_reject_breakdown["near_parallel_perimeter"] += 1
                    continue

                iw_p0 = XYZ(cm_to_ft(x0), cm_to_ft(y0), 0.0)
                iw_p1 = XYZ(cm_to_ft(x1), cm_to_ft(y1), 0.0)
                iw_dx = iw_p1.X - iw_p0.X
                iw_dy = iw_p1.Y - iw_p0.Y
                iw_ln = math.sqrt(iw_dx * iw_dx + iw_dy * iw_dy)
                if iw_ln <= 1e-9:
                    internal_wall_rejected += 1
                    internal_wall_reject_breakdown["too_short"] += 1
                    continue
                iw_wall = Wall.Create(doc, Line.CreateBound(iw_p0, iw_p1), wall_type.Id, level.Id, wall_height_ft, 0.0, False, False)
                internal_wall_ids.append(iw_wall.Id.IntegerValue)
            except Exception as ex:
                internal_wall_errors.append(str(ex))
                internal_wall_reject_breakdown["error"] += 1

        Floor.Create(doc, [build_loop(inner_pts)], floor_type.Id, level.Id)
        if create_roof:
            roof = Floor.Create(doc, [build_loop(inner_pts)], floor_type.Id, level.Id)
            set_param(roof, BuiltInParameter.FLOOR_HEIGHTABOVELEVEL_PARAM, wall_height_ft)

        openings = list(topology.get("openings") or [])
        min_opening_conf = float(cfg.get("model_min_opening_confidence", 0.45))
        end_clear_cm = float(cfg.get("model_opening_end_clearance_cm", 15.0))
        end_clear_ft = cm_to_ft(end_clear_cm)
        place_synthetic = bool(cfg.get("model_place_synthetic_openings", False))
        opening_errors = []
        opening_attempts = []
        opening_family_choices = []
        opening_family_rejects = []
        opening_host_fallback_count = 0

        for op in openings:
            otype = str(op.get("type", "")).lower()
            if otype not in ("door", "window"):
                continue

            host_edge = int(op.get("host_edge", -1))
            attempt = {
                "type": otype,
                "host_edge": host_edge,
                "width_cm": float(op.get("width_cm", 0.0)),
                "confidence": float(op.get("confidence", 1.0)),
                "synthetic": bool(op.get("synthetic", False)),
            }

            if attempt["synthetic"] and not place_synthetic:
                attempt["status"] = "skipped"
                attempt["reason"] = "synthetic_disabled"
                opening_attempts.append(attempt)
                continue

            if skip_low_conf and (not bool(op.get("manual", False))) and attempt["confidence"] < min_opening_conf:
                attempt["status"] = "skipped"
                attempt["reason"] = "low_confidence"
                opening_attempts.append(attempt)
                continue

            anchor_x_cm = _safe_float(op.get("center_x_cm"), None)
            anchor_y_cm = _safe_float(op.get("center_y_cm"), None)
            if anchor_x_cm is not None and anchor_y_cm is not None:
                attempt["anchor_xy_cm"] = [anchor_x_cm, anchor_y_cm]

            run_rec = edge_to_run.get(host_edge)
            wall_idx = None
            if run_rec is not None:
                wall_idx = run_to_wall_idx.get(int(run_rec["run_idx"]))

            used_host_fallback = False
            host_fallback_dist_cm = None
            if (wall_idx is None or wall_idx < 0 or wall_idx >= len(walls)) and anchor_x_cm is not None and anchor_y_cm is not None:
                wall_idx, host_fallback_dist_cm = _nearest_wall_index_for_point_cm(
                    anchor_x_cm,
                    anchor_y_cm,
                    wall_meta,
                    max_dist_cm=opening_host_fallback_max_cm,
                )
                if wall_idx is not None:
                    used_host_fallback = True
                    opening_host_fallback_count += 1

            if wall_idx is None or wall_idx < 0 or wall_idx >= len(walls):
                attempt["status"] = "failed"
                if run_rec is None:
                    attempt["reason"] = "host_edge_not_mapped"
                    opening_errors.append("{} edge {}: host edge not mapped".format(otype, host_edge))
                else:
                    attempt["reason"] = "host_wall_missing"
                    opening_errors.append("{} edge {}: host wall missing".format(otype, host_edge))
                opening_attempts.append(attempt)
                continue

            meta = wall_meta[wall_idx]
            if anchor_x_cm is not None and anchor_y_cm is not None:
                px = cm_to_ft(anchor_x_cm)
                py = cm_to_ft(anchor_y_cm)
            elif run_rec is not None:
                s_cm = float(op.get("start_cm", 0.0))
                e_cm = float(op.get("end_cm", s_cm))
                if e_cm < s_cm:
                    s_cm, e_cm = e_cm, s_cm
                edge_len_cm = max(1.0, float(run_rec.get("len_cm", 1.0)))
                center_edge_cm = min(edge_len_cm, max(0.0, (s_cm + e_cm) * 0.5))
                center_run_cm = float(run_rec.get("offset_cm", 0.0)) + center_edge_cm
                center_t_ft = cm_to_ft(center_run_cm)
                if meta["len_ft"] > (2.0 * end_clear_ft):
                    center_t_ft = min(meta["len_ft"] - end_clear_ft, max(end_clear_ft, center_t_ft))
                else:
                    center_t_ft = meta["len_ft"] * 0.5
                px = meta["center_start"].X + (meta["dir_x"] * center_t_ft)
                py = meta["center_start"].Y + (meta["dir_y"] * center_t_ft)
            else:
                attempt["status"] = "failed"
                attempt["reason"] = "missing_anchor"
                opening_errors.append("{} edge {}: no anchor point".format(otype, host_edge))
                opening_attempts.append(attempt)
                continue

            attempt["mapped_wall_idx"] = int(wall_idx)
            attempt["point_xy_ft"] = [px, py]
            attempt["host_fallback"] = bool(used_host_fallback)
            if host_fallback_dist_cm is not None:
                attempt["host_fallback_dist_cm"] = float(host_fallback_dist_cm)

            try:
                if otype == "door":
                    dw = cm_to_ft(float(op.get("width_cm", cfg.get("default_door_width_cm", 100.0))))
                    dh = cm_to_ft(float(op.get("height_cm", cfg.get("default_door_height_cm", 210.0))))
                    sym = get_symbol_by_width(BuiltInCategory.OST_Doors, dw, cfg=cfg, audit=opening_family_rejects)
                    if sym is None:
                        attempt["status"] = "failed"
                        attempt["reason"] = "door_symbol_missing"
                        opening_errors.append("door edge {}: no door family symbol".format(host_edge))
                        opening_attempts.append(attempt)
                        continue
                    sym_name = _symbol_display_name(sym)
                    attempt["symbol"] = sym_name
                    if not sym.IsActive:
                        sym.Activate()
                        doc.Regenerate()
                    raw_pt = XYZ(px, py, 0.0)
                    place_pt = _project_point_to_wall_axis(walls[wall_idx], raw_pt)
                    inst = doc.Create.NewFamilyInstance(place_pt, sym, walls[wall_idx], level, Structure.StructuralType.NonStructural)
                    set_opening_size(inst, dw, dh, BuiltInParameter.DOOR_WIDTH, BuiltInParameter.DOOR_HEIGHT)
                    door_ids.append(inst.Id.IntegerValue)
                    attempt["status"] = "placed"
                    attempt["instance_id"] = inst.Id.IntegerValue
                    opening_family_choices.append({
                        "type": "door",
                        "instance_id": inst.Id.IntegerValue,
                        "symbol": sym_name,
                        "width_ft": dw,
                    })

                elif otype == "window":
                    ww = cm_to_ft(float(op.get("width_cm", cfg.get("default_window_width_cm", 100.0))))
                    wh = cm_to_ft(float(op.get("height_cm", cfg.get("default_window_height_cm", 100.0))))
                    sill = cm_to_ft(float(op.get("sill_cm", cfg.get("default_window_sill_cm", 105.0))))
                    sym = get_symbol_by_width(BuiltInCategory.OST_Windows, ww, cfg=cfg, audit=opening_family_rejects)
                    if sym is None:
                        attempt["status"] = "failed"
                        attempt["reason"] = "window_symbol_missing"
                        opening_errors.append("window edge {}: no window family symbol".format(host_edge))
                        opening_attempts.append(attempt)
                        continue
                    sym_name = _symbol_display_name(sym)
                    attempt["symbol"] = sym_name
                    if not sym.IsActive:
                        sym.Activate()
                        doc.Regenerate()
                    raw_pt = XYZ(px, py, sill)
                    place_pt = _project_point_to_wall_axis(walls[wall_idx], raw_pt)
                    inst = doc.Create.NewFamilyInstance(place_pt, sym, walls[wall_idx], level, Structure.StructuralType.NonStructural)
                    set_opening_size(inst, ww, wh, BuiltInParameter.WINDOW_WIDTH, BuiltInParameter.WINDOW_HEIGHT)
                    set_param(inst, BuiltInParameter.INSTANCE_SILL_HEIGHT_PARAM, sill)
                    window_ids.append(inst.Id.IntegerValue)
                    attempt["status"] = "placed"
                    attempt["instance_id"] = inst.Id.IntegerValue
                    opening_family_choices.append({
                        "type": "window",
                        "instance_id": inst.Id.IntegerValue,
                        "symbol": sym_name,
                        "width_ft": ww,
                    })

            except Exception as ex:
                attempt["status"] = "failed"
                attempt["reason"] = str(ex)
                opening_errors.append("{} edge {}: {}".format(otype, host_edge, str(ex)))

            opening_attempts.append(attempt)

        t_geo.Commit()

        placed_attempts = [a for a in opening_attempts if a.get("status") == "placed"]
        failed_attempts = [a for a in opening_attempts if a.get("status") == "failed"]
        skipped_attempts = [a for a in opening_attempts if a.get("status") == "skipped"]

        summary = {
            "geometry": {
                "mode": "polygon_v2",
                "wall_ids": [w.Id.IntegerValue for w in walls],
                "internal_wall_ids": internal_wall_ids,
                "internal_wall_rejected_count": internal_wall_rejected,
                "internal_wall_error_count": len(internal_wall_errors),
                "internal_wall_reject_breakdown": internal_wall_reject_breakdown,
                "door_ids": door_ids,
                "window_ids": window_ids,
                "opening_errors": opening_errors,
                "opening_host_fallback_count": int(opening_host_fallback_count),
                "opening_family_choices": opening_family_choices[:120],
                "opening_family_rejects": opening_family_rejects[:240],
                "opening_attempt_count": len(opening_attempts),
                "opening_placed_count": len(placed_attempts),
                "opening_failed_count": len(failed_attempts),
                "opening_skipped_count": len(skipped_attempts),
                "opening_attempts": opening_attempts[:120],
            },
            "dimensions": {
                "ok": False,
                "note": "Dimensions are intentionally disabled in V2 while CAD recognition is stabilized.",
            },
        }
        snapshot.save_json("08_geometry_summary.json", summary)
        return summary

    except Exception:
        t_geo.RollBack()
        raise


def run_command():
    plan_view = get_plan_view()
    if plan_view is None:
        TaskDialog.Show("Create From CAD V2", "Run this command in a Floor Plan view.")
        return

    level = get_level_from_view(plan_view)
    if level is None:
        TaskDialog.Show("Create From CAD V2", "No valid Level found.")
        return

    snapshot = SnapshotRun()
    snapshot.log("Run started")

    kind = choose_input_kind()
    if kind is None:
        TaskDialog.Show("Create From CAD V2", "Canceled.")
        return

    if kind == "scan":
        TaskDialog.Show("Create From CAD V2", "Scan/sketch flow is handled by the legacy command: 'Create Room From Image'.")
        return

    ext_dir = os.path.dirname(__file__)
    cfg = load_config(os.path.join(ext_dir, "cad_config.json"))
    layer_map = load_layer_map(os.path.join(ext_dir, "cad_layer_map.json"))

    try:
        cad_instances = get_imported_cad_instances(doc, plan_view)
    except Exception:
        cad_instances = []

    source_mode = choose_cad_source_mode(len(cad_instances) > 0)
    if source_mode is None:
        TaskDialog.Show("Create From CAD V2", "Canceled.")
        return

    post_action = choose_post_action()
    if post_action is None:
        TaskDialog.Show("Create From CAD V2", "Canceled.")
        return

    selected_instance = None
    cad_path = None

    if source_mode == "import":
        cad_path = pick_cad_path()
        if not cad_path:
            TaskDialog.Show("Create From CAD V2", "Canceled.")
            return
        selected_instance, err = import_cad_file_to_view(plan_view, cad_path)
        if selected_instance is None:
            TaskDialog.Show("Create From CAD V2", "CAD import failed.\n{}".format(err or "Unknown error"))
            return
    else:
        if len(cad_instances) == 0:
            TaskDialog.Show("Create From CAD V2", "No imported CAD found in active view.")
            return
        if len(cad_instances) > 1:
            TaskDialog.Show("Create From CAD V2", "Multiple CAD imports found. Keep one visible import or use 'Import DWG/DXF File'.")
            return
        selected_instance = cad_instances[0]

    selected_id = selected_instance.Id.IntegerValue if selected_instance else None
    snapshot.save_json("00_input_meta.json", {
        "source_kind": "cad",
        "source_mode": source_mode,
        "post_action": post_action,
        "cad_path": cad_path,
        "cad_instance_id": selected_id,
        "view_name": plan_view.Name,
        "units": "cm",
    })

    model_created = False
    try:
        raw = extract_cad_from_view(doc, plan_view, cfg, target_instance_id=selected_id)
        classified = classify_entities(raw.get("lines", []), raw.get("arcs", []), layer_map, cfg=cfg)
        topology = recognize_topology(classified, cfg)
        topology = maybe_manual_openings(topology, cfg, snapshot)

        snapshot.save_json("01_raw_cad.json", raw)
        snapshot.save_json("02_classified.json", {
            "wall_lines": len(classified.get("wall_lines", [])),
            "door_lines": len(classified.get("door_lines", [])),
            "window_lines": len(classified.get("window_lines", [])),
            "door_arcs": len(classified.get("door_arcs", [])),
            "window_arcs": len(classified.get("window_arcs", [])),
            "dimension_lines": len(classified.get("dimension_lines", [])),
            "dimension_arcs": len(classified.get("dimension_arcs", [])),
            "unclassified_lines": len(classified.get("unclassified_lines", [])),
            "unclassified_arcs": len(classified.get("unclassified_arcs", [])),
            "ignored": classified.get("ignored", 0),
        })
        snapshot.save_json("03_topology.json", topology)

        out = build_model_from_topology(level, topology, cfg, snapshot)
        model_created = True

        if post_action == "replace":
            cleanup_imported_cad_instance(selected_id, snapshot)

        geo = out.get("geometry", {})
        err_list = geo.get("opening_errors", [])
        msg = "Model generated from CAD.\nWalls: {}\nInternal walls: {}\nDoors: {}\nWindows: {}".format(
            len(geo.get("wall_ids", [])),
            len(geo.get("internal_wall_ids", [])),
            len(geo.get("door_ids", [])),
            len(geo.get("window_ids", [])),
        )
        if err_list:
            msg += "\n\nOpening warnings:\n" + "\n".join(err_list[:5])
        msg += "\n\nSnapshots:\n{}".format(snapshot.run_dir)
        TaskDialog.Show("Create From CAD V2", msg)

    except OperationCanceledException:
        snapshot.log("Canceled by user")
        TaskDialog.Show("Create From CAD V2", "Canceled.\nSnapshots:\n{}".format(snapshot.run_dir))
    except Exception as ex:
        snapshot.save_error("v2_pipeline", ex)
        TaskDialog.Show("Create From CAD V2", "CAD recognition failed.\n{}\n\nSnapshots:\n{}".format(ex, snapshot.run_dir))
        # If we imported this run and failed, keep CAD so user can inspect/debug.


if __name__ == "__main__":
    run_command()
