# -*- coding: utf-8 -*-
"""Area calculation report grouped by building (first letter of level name) and category.
Exports to XLSX with per-level typical-floor multipliers.
"""

import clr
import zipfile
from xml.sax.saxutils import escape

clr.AddReference('RevitAPI')
clr.AddReference('System.Windows.Forms')
clr.AddReference('System.Drawing')

from Autodesk.Revit.DB import (
    FilteredElementCollector, ViewPlan, ViewSheet, ViewType,
    SpatialElement, Area, BuiltInParameter, UnitTypeId, UnitUtils,
)
from pyrevit import forms, revit, script
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
        return ""
    if isinstance(value, text_type):
        return value
    if isinstance(value, str):
        return value
    return text_type(value)


def normalize_text(value):
    text = to_text(value)
    for marker in [u"‎", u"‏", u"‪", u"‫", u"‬", u"﻿"]:
        text = text.replace(marker, "")
    return " ".join(text.split())


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

    label = Label()
    label.Text = prompt
    label.Left = 12
    label.Top = 12
    label.Width = 480
    label.Height = 40
    form.Controls.Add(label)

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

def collect_sheets():
    sheets = FilteredElementCollector(doc).OfClass(ViewSheet).ToElements()
    matches = [s for s in sheets if not s.IsPlaceholder and u"שטחים" in normalize_text(s.Name)]
    matches.sort(key=lambda s: (to_text(s.SheetNumber), to_text(s.Name)))
    return matches


def pick_sheets(sheets):
    items = ["{} | {}".format(to_text(s.SheetNumber), to_text(s.Name)) for s in sheets]
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


_area_view_names = {}  # area element id -> view name (for building detection)

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
    return to_text(lv.Name) if lv else ""


def get_building(area):
    view_name = _area_view_names.get(area.Id.IntegerValue, "").strip()
    if view_name and view_name[0].upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return view_name[0].upper()
    level_name = get_level_name(area).strip()
    return level_name[0].upper() if level_name else u"?"


def pick_buildings(areas):
    buildings = sorted(set(get_building(a) for a in areas))
    if len(buildings) == 1:
        return buildings
    return choose_items(u"בחירת מבנים", u"סמן מבנה אחד או יותר:", buildings)


def pick_levels_with_multipliers(areas, selected_buildings):
    """One dialog per building: checkbox per level + multiplier text box on the right.
    Returns {(building, level): multiplier} or None if cancelled."""
    sel_set = set(selected_buildings)
    bld_levels = {}
    for area in areas:
        b = get_building(area)
        if b not in sel_set:
            continue
        lv = get_level_name(area)
        if b not in bld_levels:
            bld_levels[b] = set()
        bld_levels[b].add(lv)

    result = {}
    for b in sorted(sel_set):
        levels = sorted(bld_levels.get(b, []))
        if not levels:
            continue

        form = Form()
        form.Text = u"קומות ומספר חזרות - מבנה {}".format(b)
        form.Width = 500
        form.FormBorderStyle = FormBorderStyle.FixedDialog
        form.MinimizeBox = False
        form.MaximizeBox = False
        form.TopMost = True
        try:
            form.StartPosition = 1
        except Exception:
            pass

        lbl = Label()
        lbl.Text = u"סמן קומה ומלא מספר חזרות:"
        lbl.Left = 12
        lbl.Top = 12
        lbl.Width = 460
        lbl.Height = 22
        form.Controls.Add(lbl)

        hdr_lv = Label()
        hdr_lv.Text = u"קומה"
        hdr_lv.Left = 32
        hdr_lv.Top = 38
        hdr_lv.Width = 300
        hdr_lv.Height = 18
        form.Controls.Add(hdr_lv)

        hdr_mult = Label()
        hdr_mult.Text = u"חזרות"
        hdr_mult.Left = 390
        hdr_mult.Top = 38
        hdr_mult.Width = 60
        hdr_mult.Height = 18
        form.Controls.Add(hdr_mult)

        cb_all = CheckBox()
        cb_all.Text = u"בחר הכל"
        cb_all.Left = 12
        cb_all.Top = 58
        cb_all.Width = 120
        cb_all.Height = 22
        form.Controls.Add(cb_all)

        row_h = 30
        panel_h = min(len(levels) * row_h + 4, 420)

        pnl = Panel()
        pnl.Left = 12
        pnl.Top = 84
        pnl.Width = 460
        pnl.Height = panel_h
        pnl.AutoScroll = True
        form.Controls.Add(pnl)

        checkboxes = []
        textboxes = []

        for i, lv in enumerate(levels):
            cb = CheckBox()
            cb.Text = lv
            cb.Left = 4
            cb.Top = i * row_h + 4
            cb.Width = 340
            cb.Height = 22
            pnl.Controls.Add(cb)

            tb = TextBox()
            tb.Text = "1"
            tb.Left = 368
            tb.Top = i * row_h + 4
            tb.Width = 72
            tb.Height = 22
            tb.Enabled = False
            pnl.Controls.Add(tb)

            checkboxes.append(cb)
            textboxes.append(tb)

            def make_handler(t):
                def on_change(sender, e):
                    t.Enabled = sender.Checked
                return on_change
            cb.CheckedChanged += make_handler(tb)

        def make_select_all(cbs, tbs):
            def on_select_all(sender, e):
                for c, t in zip(cbs, tbs):
                    c.Checked = sender.Checked
                    t.Enabled = sender.Checked
            return on_select_all
        cb_all.CheckedChanged += make_select_all(checkboxes, textboxes)

        btn_top = 84 + panel_h + 12
        form.Height = btn_top + 72

        ok_btn = Button()
        ok_btn.Text = u"המשך"
        ok_btn.Left = 300
        ok_btn.Top = btn_top
        ok_btn.Width = 80
        ok_btn.DialogResult = DialogResult.OK
        form.Controls.Add(ok_btn)

        cancel_btn = Button()
        cancel_btn.Text = u"ביטול"
        cancel_btn.Left = 392
        cancel_btn.Top = btn_top
        cancel_btn.Width = 80
        cancel_btn.DialogResult = DialogResult.Cancel
        form.Controls.Add(cancel_btn)

        form.AcceptButton = ok_btn
        form.CancelButton = cancel_btn

        if form.ShowDialog() != DialogResult.OK:
            return None

        for i, lv in enumerate(levels):
            if checkboxes[i].Checked:
                try:
                    m = max(1, int(textboxes[i].Text.strip()))
                except (ValueError, TypeError):
                    m = 1
                result[(b, lv)] = m

    return result if result else None


def get_area_name(area):
    p = area.get_Parameter(BuiltInParameter.ROOM_NAME)
    if p and p.HasValue:
        v = p.AsString()
        if v:
            return v
    return u"Unnamed"


def get_area_sqm(area):
    try:
        internal = area.Area
    except Exception:
        internal = None
    if internal is None:
        p = area.get_Parameter(BuiltInParameter.ROOM_AREA)
        if p:
            internal = p.AsDouble()
    if internal is None:
        return 0.0
    try:
        return round(UnitUtils.ConvertFromInternalUnits(internal, UnitTypeId.SquareMeters), 2)
    except Exception:
        return round(internal * 0.09290304, 2)


def get_param_text(param):
    if param is None or not param.HasValue:
        return ""
    for getter in ["AsString", "AsValueString"]:
        try:
            v = getattr(param, getter)()
            if v:
                return to_text(v)
        except Exception:
            pass
    return ""


_USAGE_CODE_PARAMS = [
    "usage_type", "UsageType", "Usage_Type",
    u"קוד שימוש", u"קוד_שימוש", u"סוג שימוש",
]


def get_usage_code(area):
    for param_name in _USAGE_CODE_PARAMS:
        try:
            p = area.LookupParameter(param_name)
        except Exception:
            p = None
        if p is None or not p.HasValue:
            continue
        for getter in ["AsInteger", "AsDouble"]:
            try:
                return int(getattr(p, getter)())
            except Exception:
                pass
        raw = get_param_text(p).strip()
        if raw:
            try:
                return int(float(raw))
            except (ValueError, TypeError):
                pass
    return None


def classify_by_code(code):
    if code == 300:
        return u"הורדה"
    if 250 <= code <= 302:
        return u"אחר"
    if 101 <= code <= 130:
        return u"שירות"
    if 1 <= code <= 33:
        return u"עיקרי"
    return u"אחר"


def classify_by_name(area_name):
    source = normalize_text(area_name)
    if u"הורד" in source:
        return u"הורדה"
    service_keywords = [
        u"ממ\"ד", u"ממד", u"מרחב מוגן", u"מקלט",
        u"מעלית", u"מדרגות", u"מבואה",
        u"קומת עמודים", u"מפולשת", u"מעברים",
        u"מערכות טכניות", u"מבני שירות",
        u"חדרי שירות", u"שירות משותף",
        u"עובי קירות", u"גגון", u"קירוי", u"בליטות",
        u"מבני שמירה",
    ]
    for kw in service_keywords:
        if kw in source:
            return u"שירות"
    if u"שירות" in source or u"שרות" in source:
        return u"שירות"
    main_keywords = [
        u"מגורים", u"דירה", u"מסחר", u"חנות", u"הסעדה",
        u"תעשייה", u"מלאכה", u"חקלאות",
        u"משרד", u"מלון", u"אירוח",
        u"נופש", u"ספורט", u"ציבור", u"דת",
        u"פנאי", u"תרבות", u"חינוך", u"בריאות",
        u"עיקרי",
    ]
    for kw in main_keywords:
        if kw in source:
            return u"עיקרי"
    return u"אחר"


def classify_category(area, area_name):
    code = get_usage_code(area)
    if code is not None:
        return classify_by_code(code)
    return classify_by_name(area_name)


# ---------------------------------------------------------------------------
# XLSX helpers
# ---------------------------------------------------------------------------

_BUILDING_COLORS = [
    "FFD9E1F2",
    "FFE2EFDA",
    "FFFFF2CC",
    "FFFCE4D6",
    "FFE2D9F3",
    "FFFFD7D7",
    "FFD9F2F2",
    "FFFFE0CC",
]


def xml_text(value):
    return escape(to_text(value))


def utf8_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, text_type):
        return value.encode("utf-8")
    return text_type(value).encode("utf-8")


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def col_name(index):
    result = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def build_styles_xml(fill_colors):
    fill_lines = [
        '<fill><patternFill patternType="none"/></fill>',
        '<fill><patternFill patternType="gray125"/></fill>',
    ]
    for color in fill_colors:
        fill_lines.append(
            '<fill><patternFill patternType="solid">'
            '<fgColor rgb="{0}"/><bgColor indexed="64"/>'
            '</patternFill></fill>'.format(color)
        )
    n = len(fill_colors)
    xf_lines = [
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>',          # 0: default
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>',  # 1: bold
    ]
    for i in range(n):  # 2..n+1: fill only (data rows)
        xf_lines.append(
            '<xf numFmtId="0" fontId="0" fillId="{0}" borderId="0" xfId="0" applyFill="1"/>'.format(i + 2)
        )
    for i in range(n):  # n+2..2n+1: bold+fill (title rows)
        xf_lines.append(
            '<xf numFmtId="0" fontId="1" fillId="{0}" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'.format(i + 2)
        )
    # 2n+2: bold + wrap + center/center (column header row)
    xf_lines.append(
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1">'
        '<alignment horizontal="center" vertical="center" wrapText="1"/>'
        '</xf>'
    )
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        '<fonts count="2">',
        '<font><sz val="11"/><name val="Calibri"/></font>',
        '<font><b/><sz val="11"/><name val="Calibri"/></font>',
        '</fonts>',
        '<fills count="{0}">'.format(n + 2),
        "\n".join(fill_lines),
        '</fills>',
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>',
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>',
        '<cellXfs count="{0}">'.format(2 + n * 2 + 1),
        "\n".join(xf_lines),
        '</cellXfs>',
        '</styleSheet>',
    ])


def build_sheet_xml(rows, row_styles=None, freeze_header=False, col_width=18.0,
                    row_height_pt=18.75, header_height_pt=45.0):
    if row_styles is None:
        row_styles = {}
    max_cols = max([len(row) for row in rows] or [1])
    last_ref = "{0}{1}".format(col_name(max_cols), max(len(rows), 1))
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        '<dimension ref="A1:{0}"/>'.format(last_ref),
    ]
    # Always RTL; add freeze pane if requested
    if freeze_header:
        lines += [
            '<sheetViews><sheetView tabSelected="1" workbookViewId="0" rightToLeft="1">',
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>',
            '<selection pane="bottomLeft"/>',
            '</sheetView></sheetViews>',
        ]
    else:
        lines += [
            '<sheetViews><sheetView tabSelected="1" workbookViewId="0" rightToLeft="1"/>',
            '</sheetViews>',
        ]
    lines += [
        '<cols><col min="1" max="{0}" width="{1}" customWidth="1"/></cols>'.format(max_cols, col_width),
        '<sheetData>',
    ]
    for row_index, row in enumerate(rows, 1):
        xf = row_styles.get(row_index, 0)
        style_attr = ' s="{0}"'.format(xf) if xf else ""
        row_fmt = (' s="{0}" customFormat="1"'.format(xf)) if xf else ""
        ht = header_height_pt if row_index == 1 else row_height_pt
        lines.append('<row r="{0}" customHeight="1" ht="{1}"{2}>'.format(row_index, ht, row_fmt))
        for col_index, value in enumerate(row, 1):
            cell_ref = "{0}{1}".format(col_name(col_index), row_index)
            if is_number(value):
                lines.append('<c r="{0}"{1}><v>{2}</v></c>'.format(cell_ref, style_attr, value))
            elif to_text(value).startswith("="):
                formula = xml_text(to_text(value)[1:])
                lines.append('<c r="{0}"{1}><f>{2}</f><v>0</v></c>'.format(cell_ref, style_attr, formula))
            else:
                s = to_text(value)
                if s:
                    lines.append(
                        '<c r="{0}" t="inlineStr"{1}><is><t xml:space="preserve">{2}</t></is></c>'.format(
                            cell_ref, style_attr, xml_text(s)
                        )
                    )
        lines.append('</row>')
    lines.extend(['</sheetData>', '</worksheet>'])
    return "\n".join(lines)


def build_workbook_xml(sheet_names):
    sheet_lines = []
    for index, name in enumerate(sheet_names, 1):
        sheet_lines.append(
            '<sheet name="{0}" sheetId="{1}" r:id="rId{1}"/>'.format(xml_text(name), index)
        )
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        '<sheets>',
        "\n".join(sheet_lines),
        '</sheets>',
        '</workbook>',
    ])


def build_workbook_rels_xml(sheet_names):
    rel_lines = [
        '<Relationship Id="rId0" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>',
    ]
    for index, _ in enumerate(sheet_names, 1):
        rel_lines.append(
            '<Relationship Id="rId{0}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet{0}.xml"/>'.format(index)
        )
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        "\n".join(rel_lines),
        '</Relationships>',
    ])


def build_root_rels_xml():
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>',
        '</Relationships>',
    ])


def build_content_types_xml(sheet_count):
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for index in range(1, sheet_count + 1):
        overrides.append(
            '<Override PartName="/xl/worksheets/sheet{0}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'.format(index)
        )
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        "\n".join(overrides),
        '</Types>',
    ])


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report_data(results, multipliers):
    all_buildings = sorted(set(r["building"] for r in results))
    N = len(all_buildings)
    used_colors = []
    building_data_xf = {}
    building_title_xf = {}
    for i, b in enumerate(all_buildings):
        ci = i % len(_BUILDING_COLORS)
        used_colors.append(_BUILDING_COLORS[ci])
        building_data_xf[b] = i + 2
        building_title_xf[b] = i + 2 + N

    HEADER_XF = 1
    TITLE_XF = 2 + 2 * N  # bold + wrap + center/center — column header row only

    # --- Detail sheet ---
    detail_rows = []
    detail_styles = {}
    detail_rows.append([
        u"מבנה", u"קומה", u"שם שטח", u"קטגוריה",
        u'שטח קומה אחת (מ"ר)', u"מספר חזרות", u'סה"כ (מ"ר)',
    ])
    detail_styles[1] = TITLE_XF

    results.sort(key=lambda r: (r["building"], r["level"], r["name"]))
    current_building = None
    for row in results:
        b = row["building"]
        if b != current_building:
            if current_building is not None:
                detail_rows.append([])
            current_building = b
            ri = len(detail_rows) + 1
            detail_rows.append([u"Building " + b, "", "", "", "", "", ""])
            detail_styles[ri] = building_title_xf[b]
        m = multipliers.get((b, row["level"]), 1)
        ri = len(detail_rows) + 1
        detail_rows.append([
            b, row["level"], row["name"], row["category"],
            row["sqm"], m,
            "=E{0}*F{0}".format(ri),
        ])
        detail_styles[ri] = building_data_xf[b]

    # --- Summary sheet ---
    summary_rows = []
    summary_styles = {}
    summary_rows.append([
        u"מבנה", u"קומה",
        u'עיקרי - קומה אחת', u'שירות - קומה אחת', u'סה"כ (עיקרי+שירות) קומה אחת',
        u'אחר (לא נכלל)', u"מספר חזרות",
        u'עיקרי - סה"כ', u'שירות - סה"כ', u'סה"כ (עיקרי+שירות)',
        u'אחר - סה"כ (לא נכלל)',
    ])
    summary_styles[1] = TITLE_XF

    summary = {}
    for row in results:
        b = row["building"]
        lv = row["level"]
        cat = row["category"]
        if cat == u"הורדה":
            continue
        if b not in summary:
            summary[b] = {}
        if lv not in summary[b]:
            summary[b][lv] = {u"עיקרי": 0.0, u"שירות": 0.0, u"אחר": 0.0}
        if cat in summary[b][lv]:
            summary[b][lv][cat] += row["sqm"]
        else:
            summary[b][lv][u"אחר"] += row["sqm"]

    building_total_rows = {}  # b -> excel row index of building total row

    for b in sorted(summary.keys()):
        ri = len(summary_rows) + 1
        summary_rows.append([u"Building " + b, "", "", "", "", "", "", "", "", "", ""])
        summary_styles[ri] = building_title_xf[b]

        first_data_ri = len(summary_rows) + 1

        for lv in sorted(summary[b].keys()):
            m = multipliers.get((b, lv), 1)
            cats = summary[b][lv]
            ri = len(summary_rows) + 1
            c_val = round(cats[u"עיקרי"], 2)
            d_val = round(cats[u"שירות"], 2)
            f_val = round(cats[u"אחר"], 2)
            summary_rows.append([
                b, lv,
                c_val, d_val,
                "=C{0}+D{0}".format(ri),
                f_val, m,
                "=C{0}*G{0}".format(ri),
                "=D{0}*G{0}".format(ri),
                "=H{0}+I{0}".format(ri),
                "=F{0}*G{0}".format(ri),
            ])
            summary_styles[ri] = building_data_xf[b]

        last_data_ri = len(summary_rows)
        total_ri = len(summary_rows) + 1
        building_total_rows[b] = total_ri
        summary_rows.append([
            u'סה"כ Building ' + b, "",
            "=SUM(C{0}:C{1})".format(first_data_ri, last_data_ri),
            "=SUM(D{0}:D{1})".format(first_data_ri, last_data_ri),
            "=C{0}+D{0}".format(total_ri),
            "=SUM(F{0}:F{1})".format(first_data_ri, last_data_ri),
            "",
            "=SUM(H{0}:H{1})".format(first_data_ri, last_data_ri),
            "=SUM(I{0}:I{1})".format(first_data_ri, last_data_ri),
            "=H{0}+I{0}".format(total_ri),
            "=SUM(K{0}:K{1})".format(first_data_ri, last_data_ri),
        ])
        summary_styles[total_ri] = building_title_xf[b]
        summary_rows.append([])

    # Grand total — sum building total rows
    grand_ri = len(summary_rows) + 1
    t_rows = sorted(building_total_rows.values())

    def cell_sum(col, row_indices):
        if not row_indices:
            return 0
        return "=" + "+".join("{0}{1}".format(col, r) for r in row_indices)

    summary_rows.append([
        u'סה"כ כולל', "",
        cell_sum("C", t_rows),
        cell_sum("D", t_rows),
        "=C{0}+D{0}".format(grand_ri),
        cell_sum("F", t_rows),
        "",
        cell_sum("H", t_rows),
        cell_sum("I", t_rows),
        "=H{0}+I{0}".format(grand_ri),
        cell_sum("K", t_rows),
    ])
    summary_styles[grand_ri] = HEADER_XF

    workbook_data = {u"רשימת שטחים": detail_rows, u"סיכום": summary_rows}
    row_styles = {u"רשימת שטחים": detail_styles, u"סיכום": summary_styles}
    return workbook_data, row_styles, used_colors


def export_xlsx(workbook_data, row_styles, fill_colors):
    dialog = SaveFileDialog()
    dialog.Filter = "Excel Workbook|*.xlsx"
    dialog.Title = u"שמור דוח שטחים"
    dialog.FileName = u"AreaCalc3_Report.xlsx"
    if dialog.ShowDialog() != DialogResult.OK:
        return

    path = dialog.FileName
    sheet_names = [u"רשימת שטחים", u"סיכום"]

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", utf8_bytes(build_content_types_xml(len(sheet_names))))
        zf.writestr("_rels/.rels", utf8_bytes(build_root_rels_xml()))
        zf.writestr("xl/workbook.xml", utf8_bytes(build_workbook_xml(sheet_names)))
        zf.writestr("xl/_rels/workbook.xml.rels", utf8_bytes(build_workbook_rels_xml(sheet_names)))
        zf.writestr("xl/styles.xml", utf8_bytes(build_styles_xml(fill_colors)))
        for index, name in enumerate(sheet_names, 1):
            zf.writestr(
                "xl/worksheets/sheet{0}.xml".format(index),
                utf8_bytes(build_sheet_xml(
                    workbook_data[name],
                    row_styles.get(name, {}),
                    freeze_header=True,
                    col_width=13.0,
                    row_height_pt=18.75,
                ))
            )

    forms.alert(u"הדוח נשמר:\n" + path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sheets = collect_sheets()
    if not sheets:
        forms.alert(u"לא נמצאו גיליונות המכילים 'שטחים' בשמם.")
        script.exit()

    selected_sheets = pick_sheets(sheets)
    if not selected_sheets:
        script.exit()

    views = get_area_views(selected_sheets)
    if not views:
        forms.alert(u"לא נמצאו מבטי שטחים בגיליונות שנבחרו.")
        script.exit()

    areas = collect_areas(views)
    if not areas:
        forms.alert(u"לא נמצאו שטחים.")
        script.exit()

    selected_buildings = pick_buildings(areas)
    if not selected_buildings:
        script.exit()

    multipliers = pick_levels_with_multipliers(areas, selected_buildings)
    if not multipliers:
        script.exit()

    selected_level_set = set(multipliers.keys())
    results = []
    for area in areas:
        b = get_building(area)
        lv = get_level_name(area)
        if (b, lv) not in selected_level_set:
            continue
        name = get_area_name(area)
        sqm = get_area_sqm(area)
        cat = classify_category(area, name)
        results.append({
            "building": b,
            "level": lv,
            "name": name,
            "sqm": sqm,
            "category": cat,
        })

    if not results:
        forms.alert(u"לא נמצאו שטחים בקומות שנבחרו.")
        script.exit()

    workbook_data, row_styles, fill_colors = build_report_data(results, multipliers)

    try:
        export_xlsx(workbook_data, row_styles, fill_colors)
    except Exception:
        import traceback
        forms.alert(u"שגיאה בייצוא:\n" + traceback.format_exc())


if __name__ == "__main__":
    main()
