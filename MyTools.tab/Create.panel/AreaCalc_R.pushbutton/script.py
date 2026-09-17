# -*- coding: utf-8 -*-
"""AreaCalc_R – Export Revit area polygons to DXF for the Israeli robot (רישוי זמין).

Outputs:
  <name>.dxf  – area polygons with usage_type codes on RZ_AREA / RZ_FLOOR / RZ_FRAME layers
  <name>.dat  – DWFX_SCALE value

Upload both files + a matching DWFX (exported from Revit area plan views) to the robot.
"""

import clr
import math

clr.AddReference('RevitAPI')
clr.AddReference('System.Windows.Forms')
clr.AddReference('System.Drawing')

from Autodesk.Revit.DB import (
    FilteredElementCollector, ViewPlan, ViewSheet, ViewType,
    SpatialElement, Area, SpatialElementBoundaryOptions,
    UnitTypeId, UnitUtils,
)
from pyrevit import forms as pf, revit
from System.Windows.Forms import (
    Button, CheckBox, CheckedListBox, DialogResult, Form,
    FormBorderStyle, Label, Panel, SaveFileDialog, TextBox,
)

doc = revit.doc

try:
    text_type = unicode
except NameError:
    text_type = str


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def to_text(value):
    if value is None:
        return u""
    if isinstance(value, text_type):
        return value
    if isinstance(value, str):
        return value
    return text_type(value)


def normalize_text(value):
    text = to_text(value)
    for marker in [u"‎", u"‏", u"‪", u"‫", u"‬", u"﻿"]:
        text = text.replace(marker, u"")
    return u" ".join(text.split())


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def choose_items(title, prompt, items):
    form = Form()
    form.Text = title
    form.Width = 520
    form.Height = 640
    form.FormBorderStyle = FormBorderStyle.FixedDialog
    form.MinimizeBox = False
    form.MaximizeBox = False
    form.TopMost = True
    try:
        form.StartPosition = 1
    except Exception:
        pass

    lbl = Label()
    lbl.Text = prompt
    lbl.Left = 12
    lbl.Top = 12
    lbl.Width = 480
    lbl.Height = 40
    form.Controls.Add(lbl)

    clb = CheckedListBox()
    clb.Left = 12
    clb.Top = 56
    clb.Width = 480
    clb.Height = 500
    clb.CheckOnClick = True
    clb.HorizontalScrollbar = True
    clb.IntegralHeight = False
    for item in items:
        clb.Items.Add(item)
    form.Controls.Add(clb)

    ok_btn = Button()
    ok_btn.Text = u"המשך"
    ok_btn.Left = 322
    ok_btn.Top = 568
    ok_btn.Width = 80
    ok_btn.DialogResult = DialogResult.OK
    form.Controls.Add(ok_btn)

    cancel_btn = Button()
    cancel_btn.Text = u"ביטול"
    cancel_btn.Left = 412
    cancel_btn.Top = 568
    cancel_btn.Width = 80
    cancel_btn.DialogResult = DialogResult.Cancel
    form.Controls.Add(cancel_btn)

    form.AcceptButton = ok_btn
    form.CancelButton = cancel_btn

    if form.ShowDialog() != DialogResult.OK:
        return None
    return [to_text(item) for item in clb.CheckedItems]


# ---------------------------------------------------------------------------
# Revit data helpers
# ---------------------------------------------------------------------------

_area_view_names = {}


def collect_sheets():
    sheets = FilteredElementCollector(doc).OfClass(ViewSheet).ToElements()
    matches = [s for s in sheets if not s.IsPlaceholder and u"שטחים" in normalize_text(s.Name)]
    matches.sort(key=lambda s: (to_text(s.SheetNumber), to_text(s.Name)))
    return matches


def pick_sheets(sheets):
    items = [u"{} | {}".format(to_text(s.SheetNumber), to_text(s.Name)) for s in sheets]
    sel = choose_items(u"בחירת גיליונות שטחים", u"סמן גיליון אחד או יותר:", items)
    if not sel:
        return None
    label_map = dict(zip(items, sheets))
    return [label_map[l] for l in sel]


def get_area_views(sheets):
    views = {}
    for sheet in sheets:
        for vid in sheet.GetAllPlacedViews():
            v = doc.GetElement(vid)
            if isinstance(v, ViewPlan) and v.ViewType == ViewType.AreaPlan and not v.IsTemplate:
                views[v.Id.IntegerValue] = v
    return list(views.values())


def collect_areas(views):
    global _area_view_names
    _area_view_names = {}
    found = {}
    for view in views:
        vname = to_text(view.Name).strip()
        for el in FilteredElementCollector(doc, view.Id).OfClass(SpatialElement).ToElements():
            if isinstance(el, Area) and el.Area > 0:
                eid = el.Id.IntegerValue
                if eid not in found:
                    found[eid] = el
                    _area_view_names[eid] = vname
    return list(found.values())


def get_level_name(area):
    lv = doc.GetElement(area.LevelId)
    return to_text(lv.Name) if lv else u""


def get_level_elevation_m(area):
    lv = doc.GetElement(area.LevelId)
    if lv is None:
        return 0.0
    return round(UnitUtils.ConvertFromInternalUnits(lv.Elevation, UnitTypeId.Meters), 2)


def get_building(area):
    vname = _area_view_names.get(area.Id.IntegerValue, u"").strip()
    if vname and vname[0].upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return vname[0].upper()
    lname = get_level_name(area).strip()
    return lname[0].upper() if lname else u"?"


def pick_buildings(areas):
    buildings = sorted(set(get_building(a) for a in areas))
    if len(buildings) == 1:
        return buildings
    return choose_items(u"בחירת מבנים", u"סמן מבנה אחד או יותר:", buildings)


def get_usage_type(area):
    for pname in ["usage_type", "UsageType", "Usage_Type", u"קוד שימוש"]:
        p = area.LookupParameter(pname)
        if p and p.HasValue:
            try:
                return int(p.AsInteger())
            except Exception:
                try:
                    return int(p.AsDouble())
                except Exception:
                    pass
    return 10  # default: עיקרי generic


# ---------------------------------------------------------------------------
# Floor names dialog  (Option A: same layout as AreaCalc3 but text box = floor names)
# ---------------------------------------------------------------------------

def pick_levels_with_floor_names(areas, selected_buildings):
    """One dialog per building.
    Returns {(building, level_name): "EL00,EL01,EL02"} or None if cancelled.
    Each checked level becomes one RZ_FLOOR polygon.
    The text box value is written verbatim into FLOOR=... in the DXF.
    """
    sel_set = set(selected_buildings)
    bld_levels = {}
    level_elev = {}
    for area in areas:
        b = get_building(area)
        if b not in sel_set:
            continue
        lv = get_level_name(area)
        if b not in bld_levels:
            bld_levels[b] = set()
        bld_levels[b].add(lv)
        level_elev[lv] = get_level_elevation_m(area)

    result = {}
    for b in sorted(sel_set):
        levels = sorted(bld_levels.get(b, []), key=lambda l: level_elev.get(l, 0.0))
        if not levels:
            continue

        form = Form()
        form.Text = u"שמות קומות לרובוט – מבנה {}".format(b)
        form.Width = 580
        form.FormBorderStyle = FormBorderStyle.FixedDialog
        form.MinimizeBox = False
        form.MaximizeBox = False
        form.TopMost = True
        try:
            form.StartPosition = 1
        except Exception:
            pass

        lbl = Label()
        lbl.Text = u"סמן קומה ומלא שמות קומות זהות מופרדות בפסיק (ברירת מחדל: שם הקומה בלבד):"
        lbl.Left = 12
        lbl.Top = 12
        lbl.Width = 546
        lbl.Height = 36
        form.Controls.Add(lbl)

        hdr_lv = Label()
        hdr_lv.Text = u"קומה"
        hdr_lv.Left = 36
        hdr_lv.Top = 52
        hdr_lv.Width = 200
        hdr_lv.Height = 18
        form.Controls.Add(hdr_lv)

        hdr_names = Label()
        hdr_names.Text = u"שמות קומות (לרובוט)"
        hdr_names.Left = 262
        hdr_names.Top = 52
        hdr_names.Width = 280
        hdr_names.Height = 18
        form.Controls.Add(hdr_names)

        row_h = 30
        panel_h = min(len(levels) * row_h + 4, 420)

        pnl = Panel()
        pnl.Left = 12
        pnl.Top = 74
        pnl.Width = 546
        pnl.Height = panel_h
        pnl.AutoScroll = True
        form.Controls.Add(pnl)

        checkboxes = []
        textboxes = []

        def make_handler(tb):
            def on_change(s, e):
                tb.Enabled = s.Checked
            return on_change

        for i, lv in enumerate(levels):
            cb = CheckBox()
            cb.Text = lv
            cb.Left = 4
            cb.Top = i * row_h + 4
            cb.Width = 240
            cb.Height = 22

            tb = TextBox()
            tb.Text = lv
            tb.Left = 256
            tb.Top = i * row_h + 4
            tb.Width = 278
            tb.Height = 22
            tb.Enabled = False

            cb.CheckedChanged += make_handler(tb)
            pnl.Controls.Add(cb)
            pnl.Controls.Add(tb)
            checkboxes.append(cb)
            textboxes.append((tb, lv))

        bottom = 74 + panel_h + 16
        form.Height = bottom + 60

        ok_btn = Button()
        ok_btn.Text = u"המשך"
        ok_btn.Left = 374
        ok_btn.Top = bottom
        ok_btn.Width = 80
        ok_btn.DialogResult = DialogResult.OK
        form.Controls.Add(ok_btn)

        cancel_btn = Button()
        cancel_btn.Text = u"ביטול"
        cancel_btn.Left = 464
        cancel_btn.Top = bottom
        cancel_btn.Width = 80
        cancel_btn.DialogResult = DialogResult.Cancel
        form.Controls.Add(cancel_btn)

        form.AcceptButton = ok_btn
        form.CancelButton = cancel_btn

        if form.ShowDialog() != DialogResult.OK:
            return None

        for cb, (tb, lv) in zip(checkboxes, textboxes):
            if cb.Checked:
                names = to_text(tb.Text).strip() or lv
                result[(b, lv)] = names

    return result


# ---------------------------------------------------------------------------
# Scale dialog
# ---------------------------------------------------------------------------

def pick_scale():
    form = Form()
    form.Text = u"קנה מידה DWFX"
    form.Width = 360
    form.Height = 160
    form.FormBorderStyle = FormBorderStyle.FixedDialog
    form.MinimizeBox = False
    form.MaximizeBox = False
    form.TopMost = True
    try:
        form.StartPosition = 1
    except Exception:
        pass

    lbl = Label()
    lbl.Text = u"ערך DWFX_SCALE (קנה מידה שייצאת את ה-DWFX, לדוגמה 10, 100, 200):"
    lbl.Left = 12
    lbl.Top = 14
    lbl.Width = 330
    lbl.Height = 36
    form.Controls.Add(lbl)

    tb = TextBox()
    tb.Text = u"10"
    tb.Left = 12
    tb.Top = 56
    tb.Width = 100
    tb.Height = 24
    form.Controls.Add(tb)

    ok_btn = Button()
    ok_btn.Text = u"אישור"
    ok_btn.Left = 194
    ok_btn.Top = 86
    ok_btn.Width = 70
    ok_btn.DialogResult = DialogResult.OK
    form.Controls.Add(ok_btn)

    cancel_btn = Button()
    cancel_btn.Text = u"ביטול"
    cancel_btn.Left = 274
    cancel_btn.Top = 86
    cancel_btn.Width = 70
    cancel_btn.DialogResult = DialogResult.Cancel
    form.Controls.Add(cancel_btn)

    form.AcceptButton = ok_btn
    form.CancelButton = cancel_btn

    if form.ShowDialog() != DialogResult.OK:
        return None
    return to_text(tb.Text).strip() or u"10"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_boundary_mm(area):
    """Return list of (x_mm, y_mm) for the outer boundary, or None."""
    opts = SpatialElementBoundaryOptions()
    try:
        segments = area.GetBoundarySegments(opts)
    except Exception:
        return None
    if not segments:
        return None
    pts = []
    for seg in segments[0]:
        curve = seg.GetCurve()
        tess = list(curve.Tessellate())
        for pt in tess[:-1]:
            x = UnitUtils.ConvertFromInternalUnits(pt.X, UnitTypeId.Millimeters)
            y = UnitUtils.ConvertFromInternalUnits(pt.Y, UnitTypeId.Millimeters)
            pts.append((x, y))
    return pts if len(pts) >= 3 else None


def centroid(pts):
    n = float(len(pts))
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def poly_bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def floor_outline(poly_list, pad=200.0):
    """Bounding box of all area polygons on a floor, with padding."""
    all_pts = []
    for pts in poly_list:
        all_pts.extend(pts)
    mn_x, mn_y, mx_x, mx_y = poly_bbox(all_pts)
    return [
        (mn_x - pad, mn_y - pad),
        (mx_x + pad, mn_y - pad),
        (mx_x + pad, mx_y + pad),
        (mn_x - pad, mx_y + pad),
    ]


def frame_outline(floor_outlines, pad=500.0):
    all_pts = []
    for pts in floor_outlines:
        all_pts.extend(pts)
    mn_x, mn_y, mx_x, mx_y = poly_bbox(all_pts)
    return [
        (mn_x - pad, mn_y - pad),
        (mx_x + pad, mn_y - pad),
        (mx_x + pad, mx_y + pad),
        (mn_x - pad, mx_y + pad),
    ]


# ---------------------------------------------------------------------------
# ASCII DXF writer (R12 / AC1009)
# ---------------------------------------------------------------------------

def _fmt(v):
    return "{:.3f}".format(v)


def _polyline(layer, pts):
    """Closed POLYLINE + VERTEX entities (R12 compatible)."""
    lines = [
        "0", "POLYLINE",
        "8", layer,
        "66", "1",
        "10", "0.0", "20", "0.0", "30", "0.0",
        "70", "1",
    ]
    for x, y in pts:
        lines += [
            "0", "VERTEX",
            "8", layer,
            "10", _fmt(x), "20", _fmt(y), "30", "0.0",
        ]
    lines += ["0", "SEQEND"]
    return lines


def _text(layer, x, y, text, height=100.0):
    return [
        "0", "TEXT",
        "8", layer,
        "10", _fmt(x), "20", _fmt(y), "30", "0.0",
        "40", _fmt(height),
        "1", text,
    ]


def write_dxf(filepath, floor_groups):
    """Write ASCII R12 DXF.

    floor_groups is a list of dicts:
      building_no   str
      floor_names   str   e.g. "EL00" or "EL02,EL03,EL04"
      elevations    str   e.g. "19.20" or "30.95,34.15,37.35"
      is_underground int
      floor_polygon list of (x,y)
      areas         list of {usage_type: int, polygon: [(x,y)]}
    """
    L = []

    def emit(*pairs):
        it = iter(pairs)
        for code in it:
            val = next(it)
            L.append(str(code))
            L.append(str(val))

    def emitlines(lst):
        L.extend(lst)

    # HEADER
    emit(0, "SECTION", 2, "HEADER")
    emit(9, "$ACADVER", 1, "AC1009")
    emit(9, "$INSUNITS", 70, 0)
    emit(9, "$DWGCODEPAGE", 3, "ANSI_1255")
    emit(0, "ENDSEC")

    # TABLES – minimal LAYER table
    emit(0, "SECTION", 2, "TABLES")
    emit(0, "TABLE", 2, "LAYER", 70, 4)
    for lname in ["0", "RZ_FRAME", "RZ_FLOOR", "RZ_AREA"]:
        emit(0, "LAYER", 2, lname, 70, 0, 62, 7, 6, "Continuous")
    emit(0, "ENDTAB")
    emit(0, "ENDSEC")

    # ENTITIES
    emit(0, "SECTION", 2, "ENTITIES")

    # Frame – one polygon around everything
    all_floor_polys = [fg["floor_polygon"] for fg in floor_groups]
    frame_pts = frame_outline(all_floor_polys)
    emitlines(_polyline("RZ_FRAME", frame_pts))
    fcx, fcy = centroid(frame_pts)
    emitlines(_text("RZ_FRAME", fcx, fcy + 300, "PAGE_NO=1"))

    # Floors + areas
    for fg in floor_groups:
        emitlines(_polyline("RZ_FLOOR", fg["floor_polygon"]))
        pcx, pcy = centroid(fg["floor_polygon"])
        floor_label = "BUILDING_NO={}&&&FLOOR={}&&&LEVEL_ELEVATION={}&&&IS_UNDERGROUND={}".format(
            fg["building_no"], fg["floor_names"], fg["elevations"], fg["is_underground"]
        )
        emitlines(_text("RZ_FLOOR", pcx, pcy, floor_label))

        for a in fg["areas"]:
            emitlines(_polyline("RZ_AREA", a["polygon"]))
            acx, acy = centroid(a["polygon"])
            area_label = "USAGE_TYPE={}&&&USAGE_TYPE_OLD=&&&AREA=&&&ASSET=".format(
                a["usage_type"]
            )
            emitlines(_text("RZ_AREA", acx, acy, area_label))

    emit(0, "ENDSEC")
    emit(0, "EOF")

    content = u"\r\n".join(L) + u"\r\n"
    with open(filepath, "wb") as f:
        f.write(content.encode("cp1255", errors="replace"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sheets = collect_sheets()
    if not sheets:
        pf.alert(u"לא נמצאו גיליונות ששמם מכיל 'שטחים'.")
        return

    sel_sheets = pick_sheets(sheets)
    if not sel_sheets:
        return

    views = get_area_views(sel_sheets)
    if not views:
        pf.alert(u"לא נמצאו תוכניות שטחים בגיליונות שנבחרו.")
        return

    areas = collect_areas(views)
    if not areas:
        pf.alert(u"לא נמצאו שטחים (Area > 0).")
        return

    buildings = pick_buildings(areas)
    if not buildings:
        return

    floor_names_map = pick_levels_with_floor_names(areas, buildings)
    if not floor_names_map:
        return

    scale_str = pick_scale()
    if not scale_str:
        return

    # Build floor groups
    bld_level_areas = {}
    for area in areas:
        b = get_building(area)
        lv = get_level_name(area)
        if (b, lv) in floor_names_map:
            key = (b, lv)
            if key not in bld_level_areas:
                bld_level_areas[key] = []
            bld_level_areas[key].append(area)

    # Sort by elevation
    def sort_key(kv):
        representative = kv[1][0]
        return get_level_elevation_m(representative)

    floor_groups = []
    for (b, lv), area_list in sorted(bld_level_areas.items(), key=sort_key):
        floor_names_str = floor_names_map[(b, lv)]
        base_elev = get_level_elevation_m(area_list[0])

        # Build elevation string: one value per name in floor_names_str
        names = [n.strip() for n in floor_names_str.split(",")]
        elevations_str = ",".join(["{:.2f}".format(base_elev)] * len(names))

        # Extract area polygons
        area_data = []
        poly_list = []
        for area in area_list:
            poly = get_boundary_mm(area)
            if poly is None:
                continue
            usage = get_usage_type(area)
            area_data.append({"usage_type": usage, "polygon": poly})
            poly_list.append(poly)

        if not poly_list:
            continue

        floor_groups.append({
            "building_no": b,
            "floor_names": floor_names_str,
            "elevations": elevations_str,
            "is_underground": 0,
            "floor_polygon": floor_outline(poly_list),
            "areas": area_data,
        })

    if not floor_groups:
        pf.alert(u"לא נמצאו שטחים עם גבולות תקינים לייצוא.")
        return

    # Save dialog
    dlg = SaveFileDialog()
    dlg.Filter = u"DXF files (*.dxf)|*.dxf"
    dlg.DefaultExt = "dxf"
    dlg.Title = u"שמירת קובץ DXF לרובוט"
    if dlg.ShowDialog() != DialogResult.OK:
        return

    dxf_path = to_text(dlg.FileName)
    if not dxf_path.lower().endswith(".dxf"):
        dxf_path = dxf_path + ".dxf"
    dat_path = dxf_path[:-4] + ".dat"

    write_dxf(dxf_path, floor_groups)

    with open(dat_path, "wb") as f:
        f.write(("DWFX_SCALE\t{}\n".format(scale_str)).encode("ascii"))

    pf.alert(
        u"נשמר בהצלחה:\n{}\n{}\n\nאל תשכח לייצא גם DWFX מתוכניות השטחים ב-Revit.".format(
            dxf_path, dat_path
        )
    )


main()
