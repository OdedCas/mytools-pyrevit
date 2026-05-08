#! python3
# -*- coding: utf-8 -*-
__title__ = "Import Sheets"
__doc__ = "Import sheets from an ArchiTask Revit Export SL workbook."

import os
import zipfile
import xml.etree.ElementTree as ET

import clr

clr.AddReference("System.Windows.Forms")
clr.AddReference("System.Drawing")

from System.Windows.Forms import (
    Button,
    CheckBox,
    ComboBox,
    ComboBoxStyle,
    DialogResult,
    Form,
    FormBorderStyle,
    Label,
    OpenFileDialog,
    TextBox,
)

from Autodesk.Revit.DB import (
    BuiltInCategory,
    BuiltInParameter,
    FamilySymbol,
    FilteredElementCollector,
    Level,
    ViewFamily,
    ViewFamilyType,
    ViewPlan,
    StorageType,
    Transaction,
    ViewDuplicateOption,
    View,
    ViewSheet,
    ViewType,
    Viewport,
    XYZ,
)
from Autodesk.Revit.UI import TaskDialog


uidoc = __revit__.ActiveUIDocument
doc = uidoc.Document

XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"

ROW_PARAM_MAP = {
    "Sheet Number": ["Sheet Number", "Drawing Number", "Number", "מספר גיליון"],
    "Sheet Name": ["Sheet Name", "Drawing Title", "Title", "שם גיליון"],
    "Chapter": ["Chapter", "Chapter Key", "פרק"],
    "Scale": ["AR_Scale", "Sheet Scale", "Drawing Scale", "Scale", 'קנ"מ'],
    "Version": ["Version", "Revision", "Rev", "Ver", "מספר גרסה", "גרסה"],
    "Level": ["AR_Level", "Sheet Level", "Drawing Level", "Level", "מפלס"],
    "Floor View": ["Floor View", "View Name", "Placed View", "שם מבט", "מבט קומה"],
    "Comments": ["Comments", "Sheet Comments", "Remarks", "הערות"],
}

META_PARAM_MAP = {
    "project_name": ["Project Name", "Project", "שם פרויקט"],
    "project_name_en": ["Project Name EN", "Project Name English", "English Project Name"],
    "client": ["Client Name", "Client", "שם לקוח"],
    "project_address": ["Project Address", "Project Location", "Address", "כתובת"],
    "organization_name": ["Organization Name", "Office Name", "Company Name", "שם משרד", "שם חברה"],
    "organization_address": ["Organization Address", "Office Address", "Company Address", "כתובת משרד"],
    "organization_phone": ["Organization Phone", "Office Phone", "Company Phone", "Phone", "טלפון"],
    "organization_phone_secondary": ["Organization Phone 2", "Secondary Phone", "Phone Secondary", "טלפון נוסף"],
    "text_scale_factor": ["Text Scale Factor", "Font Scale", "Text Scale", "Scale Factor", "מקדם טקסט"],
    "issue_date": ["Issue Date", "Date", "תאריך גיליון", "תאריך"],
    "building_code": ["Building Code", "Building Number", "קוד מבנה"],
    "discipline_code": ["Discipline", "Discipline Code", "דיסציפלינה"],
    "naming_convention": ["Naming Convention", "Sheet Naming", "קונבנציית שמות"],
}

# Built-in parameters to read from ProjectInformation and push to title blocks.
# Built dynamically so missing BIPs in older Revit versions are silently skipped.
def _build_project_info_bip_names():
    _candidates = [
        ("PROJECT_NAME",              ["Project Name", "שם פרויקט", "Project_Name"]),
        ("CLIENT_NAME",               ["Client Name", "Client", "Owner", "שם לקוח"]),
        ("AUTHOR",                    ["Author", "Drawn By", "מחבר"]),
        ("PROJECT_ADDRESS",           ["Project Address", "Address", "כתובת"]),
        ("PROJECT_NUMBER",            ["Project Number", "Project No", "מספר פרויקט"]),
        ("PROJECT_ORGANIZATION_NAME", ["Organization Name", "Company Name", "Office Name", "שם משרד", "שם חברה"]),
        ("PROJECT_ISSUE_DATE",        ["Issue Date", "Date", "תאריך"]),
    ]
    result = []
    for attr_name, names in _candidates:
        try:
            bip = getattr(BuiltInParameter, attr_name)
            result.append((bip, names))
        except AttributeError:
            pass
    return result

PROJECT_INFO_BIP_NAMES = _build_project_info_bip_names()


def set_form_center(form):
    try:
        import System.Windows.Forms as WinForms
        form.StartPosition = WinForms.FormStartPosition.CenterScreen
    except Exception:
        pass


def safe_string(value):
    if value is None:
        return ""
    try:
        return unicode(value).strip()
    except NameError:
        return str(value).strip()
    except Exception:
        try:
            return str(value).strip()
        except Exception:
            return ""


def normalize_path_part(path_text):
    return safe_string(path_text).replace("\\", "/").lstrip("/")


def get_collector_titleblock_symbols():
    symbols = []
    collector = FilteredElementCollector(doc).OfCategory(BuiltInCategory.OST_TitleBlocks).WhereElementIsElementType()
    for symbol in collector:
        if isinstance(symbol, FamilySymbol):
            symbols.append(symbol)
    return sorted(symbols, key=lambda s: get_titleblock_display_name(s).lower())


def get_symbol_type_name(symbol):
    for bip in [BuiltInParameter.SYMBOL_NAME_PARAM, BuiltInParameter.ALL_MODEL_TYPE_NAME]:
        try:
            param = symbol.get_Parameter(bip)
            if param:
                value = safe_string(param.AsString() or param.AsValueString())
                if value:
                    return value
        except Exception:
            pass
    return safe_string(symbol.Name)


def get_titleblock_display_name(symbol):
    family_name = ""
    try:
        family_name = safe_string(symbol.Family.Name)
    except Exception:
        family_name = ""
    type_name = get_symbol_type_name(symbol)
    if family_name and type_name:
        return "{} : {}".format(family_name, type_name)
    return family_name or type_name or "Titleblock"


def pick_excel_path():
    dialog = OpenFileDialog()
    dialog.Filter = "Excel Workbook|*.xlsx"
    dialog.Multiselect = False
    dialog.Title = "Select ArchiTask Revit Export SL workbook"
    if dialog.ShowDialog() == DialogResult.OK:
        return dialog.FileName
    return None


def choose_import_settings(symbols):
    form = Form()
    form.Text = "Import Sheets from ArchiTask"
    form.Width = 1220
    form.Height = 570
    form.FormBorderStyle = FormBorderStyle.FixedDialog
    form.MinimizeBox = False
    form.MaximizeBox = False
    set_form_center(form)

    lbl_file = Label()
    lbl_file.Left = 12
    lbl_file.Top = 18
    lbl_file.Width = 260
    lbl_file.Text = "Workbook (.xlsx)"
    form.Controls.Add(lbl_file)

    txt_file = TextBox()
    txt_file.Left = 12
    txt_file.Top = 38
    txt_file.Width = 1088
    form.Controls.Add(txt_file)

    btn_browse = Button()
    btn_browse.Text = "Browse..."
    btn_browse.Left = 1110
    btn_browse.Top = 36
    btn_browse.Width = 82
    form.Controls.Add(btn_browse)

    lbl_tb = Label()
    lbl_tb.Left = 12
    lbl_tb.Top = 76
    lbl_tb.Width = 240
    lbl_tb.Text = "Titleblock type for new sheets"
    form.Controls.Add(lbl_tb)

    cmb_tb = ComboBox()
    cmb_tb.Left = 12
    cmb_tb.Top = 96
    cmb_tb.Width = 1180
    cmb_tb.DropDownStyle = ComboBoxStyle.DropDownList
    for item in symbols:
        cmb_tb.Items.Add(get_titleblock_display_name(item))
    if cmb_tb.Items.Count > 0:
        cmb_tb.SelectedIndex = 0
    form.Controls.Add(cmb_tb)

    chk_update = CheckBox()
    chk_update.Left = 12
    chk_update.Top = 136
    chk_update.Width = 320
    chk_update.Text = "Update existing sheets with matching numbers"
    form.Controls.Add(chk_update)

    chk_overwrite = CheckBox()
    chk_overwrite.Left = 12
    chk_overwrite.Top = 160
    chk_overwrite.Width = 320
    chk_overwrite.Text = "Overwrite filled metadata parameters"
    form.Controls.Add(chk_overwrite)

    chk_create_missing = CheckBox()
    chk_create_missing.Left = 12
    chk_create_missing.Top = 188
    chk_create_missing.Width = 520
    chk_create_missing.Checked = True
    chk_create_missing.Text = "Create missing plan views from the sheet level"
    form.Controls.Add(chk_create_missing)

    chk_duplicate_cores = CheckBox()
    chk_duplicate_cores.Left = 12
    chk_duplicate_cores.Top = 214
    chk_duplicate_cores.Width = 560
    chk_duplicate_cores.Checked = True
    chk_duplicate_cores.Text = "Duplicate matching 1:100 views to 1:50 for core sheets"
    form.Controls.Add(chk_duplicate_cores)

    note = Label()
    note.Left = 12
    note.Top = 244
    note.Width = 1180
    note.Height = 170
    note.Text = (
        "The importer auto-matches views by Floor View, Sheet Number, Sheet Name, Level, and Chapter.\n"
        "You do not need to pick 46 views manually.\n"
        "When a 1:50 core sheet needs a 1:100 source view, the tool can duplicate the source view and place the copy on the sheet.\n"
        "If no matching plan view exists, the tool can create one from the sheet level and then place it.\n"
        "The importer sets Sheet Number, Sheet Name, Comments, Scale, Version, Level, and fills matching project or organization parameters."
    )
    form.Controls.Add(note)

    btn_ok = Button()
    btn_ok.Text = "Import"
    btn_ok.Left = 1010
    btn_ok.Top = 472
    btn_ok.Width = 82
    btn_ok.DialogResult = DialogResult.OK
    form.Controls.Add(btn_ok)

    btn_cancel = Button()
    btn_cancel.Text = "Cancel"
    btn_cancel.Left = 1100
    btn_cancel.Top = 472
    btn_cancel.Width = 82
    btn_cancel.DialogResult = DialogResult.Cancel
    form.Controls.Add(btn_cancel)

    def on_browse(sender, args):
        path = pick_excel_path()
        if path:
            txt_file.Text = path

    btn_browse.Click += on_browse

    form.AcceptButton = btn_ok
    form.CancelButton = btn_cancel

    if form.ShowDialog() != DialogResult.OK:
        return None

    path = safe_string(txt_file.Text)
    if (not path) or (not os.path.exists(path)):
        TaskDialog.Show(__title__, "Pick an existing .xlsx workbook first.")
        return None
    if cmb_tb.SelectedIndex < 0 or cmb_tb.SelectedIndex >= len(symbols):
        TaskDialog.Show(__title__, "No titleblock type selected.")
        return None

    return {
        "file_path": path,
        "titleblock_symbol": symbols[cmb_tb.SelectedIndex],
        "update_existing": bool(chk_update.Checked),
        "overwrite_existing": bool(chk_overwrite.Checked),
        "create_missing_views": bool(chk_create_missing.Checked),
        "duplicate_core_views": bool(chk_duplicate_cores.Checked),
    }


def read_zip_text(zip_file, member_name):
    with zip_file.open(member_name, "r") as handle:
        data = handle.read()
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return data


def load_shared_strings(zip_file):
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []
    root = ET.fromstring(read_zip_text(zip_file, "xl/sharedStrings.xml"))
    strings = []
    for item in root.findall(XML_NS + "si"):
        parts = []
        for text_node in item.iter():
            if text_node.tag == XML_NS + "t" and text_node.text is not None:
                parts.append(text_node.text)
        strings.append("".join(parts))
    return strings


def load_workbook_sheet_paths(zip_file):
    rels_root = ET.fromstring(read_zip_text(zip_file, "xl/_rels/workbook.xml.rels"))
    rel_targets = {}
    for rel in rels_root.findall(REL_NS + "Relationship"):
        rid = rel.attrib.get("Id")
        target = normalize_path_part(rel.attrib.get("Target"))
        if rid and target:
            rel_targets[rid] = "xl/" + target

    workbook_root = ET.fromstring(read_zip_text(zip_file, "xl/workbook.xml"))
    sheets = {}
    for sheet_node in workbook_root.findall(XML_NS + "sheets/" + XML_NS + "sheet"):
        name = safe_string(sheet_node.attrib.get("name"))
        rid = sheet_node.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = rel_targets.get(rid)
        if name and target:
            sheets[name] = target
    return sheets


def col_index_from_ref(cell_ref):
    letters = []
    for ch in safe_string(cell_ref):
        if ch.isalpha():
            letters.append(ch.upper())
        else:
            break
    value = 0
    for ch in letters:
        value = (value * 26) + (ord(ch) - ord("A") + 1)
    return value - 1


def parse_sheet_rows(zip_file, sheet_path, shared_strings):
    root = ET.fromstring(read_zip_text(zip_file, sheet_path))
    rows = []
    for row_node in root.findall(XML_NS + "sheetData/" + XML_NS + "row"):
        row_map = {}
        for cell in row_node.findall(XML_NS + "c"):
            ref = cell.attrib.get("r", "")
            idx = col_index_from_ref(ref)
            cell_type = cell.attrib.get("t")
            value = ""
            if cell_type == "inlineStr":
                tnode = cell.find(XML_NS + "is/" + XML_NS + "t")
                value = tnode.text if tnode is not None and tnode.text is not None else ""
            else:
                vnode = cell.find(XML_NS + "v")
                raw = vnode.text if vnode is not None and vnode.text is not None else ""
                if cell_type == "s":
                    try:
                        shared_index = int(raw)
                        value = shared_strings[shared_index]
                    except Exception:
                        value = ""
                else:
                    value = raw
            row_map[idx] = safe_string(value)
        if row_map:
            rows.append(row_map)
    return rows


def sheet_rows_to_dicts(raw_rows):
    if not raw_rows:
        return []
    max_col = max(raw_rows[0].keys())
    headers = []
    for col_idx in range(max_col + 1):
        headers.append(safe_string(raw_rows[0].get(col_idx, "")))
    out = []
    for row_map in raw_rows[1:]:
        item = {}
        has_value = False
        for col_idx, header in enumerate(headers):
            if not header:
                continue
            value = safe_string(row_map.get(col_idx, ""))
            if value:
                has_value = True
            item[header] = value
        if has_value:
            out.append(item)
    return out


def metadata_rows_to_dict(raw_rows):
    out = {}
    dict_rows = sheet_rows_to_dicts(raw_rows)
    for item in dict_rows:
        key = safe_string(item.get("Key"))
        value = safe_string(item.get("Value"))
        if key:
            out[key] = value
    return out


def load_architask_workbook(file_path):
    with zipfile.ZipFile(file_path, "r") as zip_file:
        shared_strings = load_shared_strings(zip_file)
        sheets = load_workbook_sheet_paths(zip_file)

        sheet_path = sheets.get("Revit Import")
        if not sheet_path:
            sheet_path = next(iter(sheets.values()), None)
        if not sheet_path:
            raise Exception("No worksheet found in workbook.")

        raw_sheet_rows = parse_sheet_rows(zip_file, sheet_path, shared_strings)
        sheet_rows = sheet_rows_to_dicts(raw_sheet_rows)

        meta = {}
        meta_path = sheets.get("ArchiTask Metadata")
        if meta_path:
            raw_meta_rows = parse_sheet_rows(zip_file, meta_path, shared_strings)
            meta = metadata_rows_to_dict(raw_meta_rows)

        return sheet_rows, meta


def get_existing_sheets_by_number():
    out = {}
    collector = FilteredElementCollector(doc).OfClass(ViewSheet)
    for sheet in collector:
        number = safe_string(sheet.SheetNumber)
        if number and number not in out:
            out[number] = sheet
    return out


def get_titleblock_instance(sheet):
    collector = FilteredElementCollector(doc, sheet.Id).OfCategory(BuiltInCategory.OST_TitleBlocks).WhereElementIsNotElementType()
    for element in collector:
        return element
    return None


def get_viewport_center(sheet):
    try:
        outline = sheet.Outline
        min_u = float(outline.Min.U)
        min_v = float(outline.Min.V)
        max_u = float(outline.Max.U)
        max_v = float(outline.Max.V)
        return XYZ((min_u + max_u) / 2.0, (min_v + max_v) / 2.0, 0.0)
    except Exception:
        return XYZ(0.5, 0.5, 0.0)


def sheet_has_viewports(sheet):
    try:
        return len(list(sheet.GetAllViewports())) > 0
    except Exception:
        return False


def normalize_for_match(value):
    cleaned = []
    for ch in safe_string(value).lower():
        if ch.isalnum():
            cleaned.append(ch)
    return ''.join(cleaned)


def parse_scale_denominator(scale_text):
    text = safe_string(scale_text)
    if not text:
        return 0
    if ':' in text:
        parts = text.split(':')
        text = parts[-1]
    try:
        return int(float(text))
    except Exception:
        return 0


def get_placeable_views():
    collector = FilteredElementCollector(doc).OfClass(View)
    allowed_types = set([
        ViewType.FloorPlan,
        ViewType.CeilingPlan,
        ViewType.EngineeringPlan,
        ViewType.AreaPlan,
    ])
    views = []
    for view in collector:
        try:
            if view.IsTemplate:
                continue
            if view.ViewType not in allowed_types:
                continue
            views.append(view)
        except Exception:
            pass
    return views


def get_view_family_type(view_family):
    collector = FilteredElementCollector(doc).OfClass(ViewFamilyType)
    for view_family_type in collector:
        try:
            if view_family_type.ViewFamily == view_family:
                return view_family_type
        except Exception:
            pass
    return None


def get_view_scale(view):
    try:
        return int(view.Scale)
    except Exception:
        return 0


def get_row_chapter(row):
    return safe_string(row.get("Chapter")).lower()


def get_row_level_value(row):
    return safe_string(row.get("Level")).upper()


def get_row_base_level_value(row):
    level = get_row_level_value(row)
    if not level:
        return ""
    if "-" in level and level[0].isdigit():
        return safe_string(level.split("-")[0]).upper()
    return level


def get_level_candidates(row):
    candidates = []
    level = get_row_base_level_value(row)
    if not level:
        return candidates
    candidates.append(level)
    candidates.append("Level {}".format(level))
    candidates.append("L{}".format(level))
    if level.isdigit():
        number = str(int(level))
        padded = "{:0>2}".format(number)
        candidates.extend([
            number,
            padded,
            "Level {}".format(number),
            "Level {}".format(padded),
            "L{}".format(number),
            "L{}".format(padded),
        ])
    if level.startswith("B") and len(level) > 1 and level[1:].isdigit():
        candidates.extend([
            "Basement {}".format(level[1:]),
            "B {}".format(level[1:]),
        ])
    elif level == "00":
        candidates.extend(["Ground floor", "Ground", "Level 0", "Level 00", "L0"])
    elif level == "RF":
        candidates.extend(["Roof", "Roof Level", "Roof plan"])
    elif level == "TRF":
        candidates.extend(["Technical roof", "Technical roof plan", "Roof"])
    return [item for item in candidates if item]


def find_best_level_match(row):
    candidates = [normalize_for_match(item) for item in get_level_candidates(row)]
    if not candidates:
        return None

    levels = list(FilteredElementCollector(doc).OfClass(Level))
    for candidate in candidates:
        for level in levels:
            try:
                level_name = normalize_for_match(level.Name)
                if level_name == candidate or level_name.startswith(candidate) or candidate.startswith(level_name):
                    return level
            except Exception:
                pass

    return None


def get_row_view_family(row):
    chapter = get_row_chapter(row)
    sheet_name = safe_string(row.get("Sheet Name")).lower()
    if chapter == "ch23" or "ceiling" in sheet_name:
        return ViewFamily.CeilingPlan
    return ViewFamily.FloorPlan


def get_row_target_view_name(row):
    floor_view = safe_string(row.get("Floor View"))
    if floor_view:
        return floor_view
    return "{} - {}".format(safe_string(row.get("Sheet Number")), safe_string(row.get("Sheet Name"))).strip(" -")


def get_row_semantic_tokens(row):
    tokens = []
    level = get_row_level_value(row)
    sheet_name = safe_string(row.get("Sheet Name")).lower()
    chapter = get_row_chapter(row)

    if chapter == "ch21" or "core" in sheet_name:
        tokens.append("core")
    if "typical" in sheet_name or "-" in level:
        tokens.append("typical")
    if level.startswith("B"):
        tokens.extend(["basement", "b{}".format(level[1:])])
    elif level == "00":
        tokens.extend(["ground", "groundfloor"])
    elif level == "RF":
        tokens.append("roof")
    elif level == "TRF":
        tokens.extend(["technicalroof", "roof"])
    elif level.isdigit():
        tokens.extend(["level", "level{}".format(str(int(level)))])

    return [normalize_for_match(item) for item in tokens if item]


def get_view_type_for_family(view_family):
    if view_family == ViewFamily.CeilingPlan:
        return ViewType.CeilingPlan
    return ViewType.FloorPlan


def get_views_for_row(row):
    expected_type = get_view_type_for_family(get_row_view_family(row))
    views = []
    for view in get_placeable_views():
        try:
            if view.ViewType == expected_type:
                views.append(view)
        except Exception:
            pass
    return views


def get_view_level_name(view):
    try:
        gen_level = view.GenLevel
        if gen_level is not None:
            return safe_string(gen_level.Name)
    except Exception:
        pass
    return ""


def view_matches_row_level(view, row):
    level_name = normalize_for_match(get_view_level_name(view))
    if not level_name:
        return False
    for candidate in get_level_candidates(row):
        candidate_norm = normalize_for_match(candidate)
        if not candidate_norm:
            continue
        if level_name == candidate_norm or level_name.startswith(candidate_norm) or candidate_norm.startswith(level_name):
            return True
    return False


def get_existing_view_names():
    names = set()
    collector = FilteredElementCollector(doc).OfClass(View)
    for view in collector:
        try:
            names.add(safe_string(view.Name))
        except Exception:
            pass
    return names


def make_unique_view_name(base_name):
    existing = get_existing_view_names()
    candidate = safe_string(base_name)
    if not candidate:
        candidate = "ArchiTask View"
    if candidate not in existing:
        return candidate
    idx = 2
    while True:
        next_name = "{} ({})".format(candidate, idx)
        if next_name not in existing:
            return next_name
        idx += 1


def get_default_plan_view_type():
    return get_view_family_type(ViewFamily.FloorPlan)


def create_missing_plan_view_for_row(row):
    level = find_best_level_match(row)
    if level is None:
        return None, "no matching level found for {}".format(safe_string(row.get("Level")))

    view_type = get_view_family_type(get_row_view_family(row))
    if view_type is None:
        return None, "no matching plan view type found in the model"

    desired_scale = parse_scale_denominator(row.get("Scale"))
    base_name = get_row_target_view_name(row)
    if not base_name.strip("- "):
        base_name = "ArchiTask Floor Plan"

    try:
        view = ViewPlan.Create(doc, view_type.Id, level.Id)
        if desired_scale > 0:
            view.Scale = desired_scale
        try:
            view.Name = make_unique_view_name(base_name)
        except Exception:
            pass
        return view, None
    except Exception as ex:
        return None, ex


def get_view_candidates(row):
    candidates = []
    sheet_number = safe_string(row.get("Sheet Number"))
    sheet_name = safe_string(row.get("Sheet Name"))
    level = safe_string(row.get("Level"))
    chapter = safe_string(row.get("Chapter"))
    floor_view = safe_string(row.get("Floor View"))

    if floor_view:
        candidates.append(floor_view)
    if sheet_number and sheet_name:
        candidates.append("{} - {}".format(sheet_number, sheet_name))
    if sheet_number:
        candidates.append(sheet_number)
    if sheet_name:
        candidates.append(sheet_name)
    if level:
        candidates.append(level)
        if level.startswith('B'):
            candidates.append("Basement {}".format(level[1:]))
        elif level == '00':
            candidates.append('Ground floor')
        elif level == 'RF':
            candidates.append('Roof plan')
        elif level == 'TRF':
            candidates.append('Technical roof plan')
    if chapter:
        candidates.append(chapter)
        if chapter == 'ch21':
            candidates.extend(['Core plans', 'Core'])
        elif chapter == 'ch20':
            candidates.extend(['Floor plans', 'Floor'])
    return [item for item in candidates if item]


def find_best_view_match(row):
    views = get_views_for_row(row)
    if not views:
        return None

    target_names = [normalize_for_match(item) for item in get_view_candidates(row)]
    semantic_tokens = get_row_semantic_tokens(row)
    desired_scale = parse_scale_denominator(row.get("Scale"))

    best_view = None
    best_score = -1

    for view in views:
        try:
            view_name_norm = normalize_for_match(view.Name)
            score = 0

            for target_name in target_names:
                if not target_name:
                    continue
                if view_name_norm == target_name:
                    score += 120
                elif view_name_norm.startswith(target_name) or target_name.startswith(view_name_norm):
                    score += 60
                elif target_name in view_name_norm:
                    score += 35

            if view_matches_row_level(view, row):
                score += 50

            for token in semantic_tokens:
                if token and token in view_name_norm:
                    score += 12

            view_scale = get_view_scale(view)
            if desired_scale > 0:
                if view_scale == desired_scale:
                    score += 18
                elif desired_scale == 50 and view_scale == 100:
                    score += 14

            if score > best_score:
                best_score = score
                best_view = view
        except Exception:
            pass

    if best_score >= 50:
        return best_view
    return None


def duplicate_view_for_row(source_view, row):
    desired_scale = parse_scale_denominator(row.get("Scale"))
    base_name = get_row_target_view_name(row)
    if not base_name.strip('- '):
        base_name = safe_string(source_view.Name)
    try:
        dup_id = source_view.Duplicate(ViewDuplicateOption.Duplicate)
        dup_view = doc.GetElement(dup_id)
        if desired_scale > 0:
            dup_view.Scale = desired_scale
        new_name = make_unique_view_name("{} [{}]".format(base_name, "1:{}".format(desired_scale) if desired_scale > 0 else "copy"))
        dup_view.Name = new_name
        return dup_view
    except Exception:
        return None


def try_place_view(sheet, view):
    if view is None:
        return False
    if not Viewport.CanAddViewToSheet(doc, sheet.Id, view.Id):
        return False
    Viewport.Create(doc, sheet.Id, view.Id, get_viewport_center(sheet))
    return True


def place_view_on_sheet_if_possible(sheet, row, allow_duplicate_core_views, create_missing_views, messages):
    view = find_best_view_match(row)
    desired_scale = parse_scale_denominator(row.get("Scale"))
    sheet_number = safe_string(sheet.SheetNumber)

    try:
        if sheet_has_viewports(sheet):
            return True

        if view is not None:
            needs_duplicate = False
            if desired_scale == 50 and get_view_scale(view) not in (0, desired_scale):
                needs_duplicate = allow_duplicate_core_views

            if not needs_duplicate and try_place_view(sheet, view):
                return True

            if allow_duplicate_core_views:
                duplicate = duplicate_view_for_row(view, row)
                if try_place_view(sheet, duplicate):
                    return True

        if create_missing_views:
            created_view, create_error = create_missing_plan_view_for_row(row)
            if created_view is not None:
                if try_place_view(sheet, created_view):
                    return True
                if allow_duplicate_core_views:
                    duplicate = duplicate_view_for_row(created_view, row)
                    if try_place_view(sheet, duplicate):
                        return True
            else:
                messages.append("{} -> could not create plan view: {}".format(sheet_number, create_error))

        if view is None:
            messages.append("{} -> no matching or creatable view found".format(sheet_number))
        else:
            messages.append("{} -> matched view exists but still could not be placed: {}".format(sheet_number, safe_string(view.Name)))
        return False
    except Exception as ex:
        messages.append("{} -> failed to place view: {}".format(sheet_number, ex))
        return False


def get_parameter_text(param):
    if param is None:
        return ""
    try:
        if param.StorageType == StorageType.String:
            return safe_string(param.AsString())
        if param.StorageType == StorageType.Integer:
            return safe_string(param.AsInteger())
        if param.StorageType == StorageType.Double:
            return safe_string(param.AsValueString() or param.AsDouble())
        if param.StorageType == StorageType.ElementId:
            return safe_string(param.AsValueString())
    except Exception:
        pass
    return ""


def try_set_parameter(param, value_text, overwrite_existing):
    if param is None or param.IsReadOnly:
        return False
    value_text = safe_string(value_text)
    if not value_text:
        return False
    if (not overwrite_existing) and get_parameter_text(param):
        return False
    try:
        if param.StorageType == StorageType.String:
            param.Set(value_text)
            return True
        if param.StorageType == StorageType.Integer:
            param.Set(int(float(value_text)))
            return True
        if param.StorageType == StorageType.Double:
            try:
                param.Set(float(value_text))
                return True
            except Exception:
                return False
    except Exception:
        return False
    return False


def try_set_parameter_names(element, names, value_text, overwrite_existing):
    if element is None:
        return False
    for name in names:
        try:
            param = element.LookupParameter(name)
            if try_set_parameter(param, value_text, overwrite_existing):
                return True
        except Exception:
            pass
    return False


def try_set_comments(sheet, value_text, overwrite_existing):
    if not value_text:
        return False
    try:
        param = sheet.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
        if try_set_parameter(param, value_text, overwrite_existing):
            return True
    except Exception:
        pass
    return try_set_parameter_names(sheet, ROW_PARAM_MAP["Comments"], value_text, overwrite_existing)


def apply_row_value(sheet, titleblock, names, value_text, overwrite_existing):
    if not value_text:
        return False
    changed = False
    if try_set_parameter_names(sheet, names, value_text, overwrite_existing):
        changed = True
    if try_set_parameter_names(titleblock, names, value_text, overwrite_existing):
        changed = True
    return changed


def resolve_meta_names(meta_key):
    """Map a raw workbook metadata key to the corresponding parameter name list."""
    if meta_key in META_PARAM_MAP:
        return META_PARAM_MAP[meta_key]
    meta_key_norm = normalize_for_match(meta_key)
    for internal_key, names in META_PARAM_MAP.items():
        if normalize_for_match(internal_key) == meta_key_norm:
            return names
        for name in names:
            if normalize_for_match(name) == meta_key_norm:
                return names
    # Fallback: try setting a parameter with the exact raw key name
    return [meta_key]


def apply_metadata_value(sheet, titleblock, project_info, meta_key, meta_value, overwrite_existing):
    names = resolve_meta_names(meta_key)
    if not meta_value:
        return False
    changed = False
    if try_set_parameter_names(titleblock, names, meta_value, overwrite_existing):
        changed = True
    if try_set_parameter_names(sheet, names, meta_value, overwrite_existing):
        changed = True
    # Always overwrite ProjectInformation — Revit fills those fields with placeholder
    # text (the param name itself) so overwrite_existing would wrongly block them.
    if try_set_parameter_names(project_info, names, meta_value, True):
        changed = True
    return changed


def apply_project_metadata_once(project_info, metadata, overwrite_existing):
    changed = 0
    for key, value in metadata.items():
        if apply_metadata_value(None, None, project_info, key, value, overwrite_existing):
            changed += 1
    return changed


def apply_sheet_data(sheet, titleblock, project_info, row, metadata, overwrite_existing):
    sheet_number = safe_string(row.get("Sheet Number"))
    sheet_name = safe_string(row.get("Sheet Name"))
    if sheet_number and (overwrite_existing or safe_string(sheet.SheetNumber) != sheet_number):
        sheet.SheetNumber = sheet_number
    if sheet_name and (overwrite_existing or safe_string(sheet.Name) != sheet_name):
        sheet.Name = sheet_name

    try_set_comments(sheet, row.get("Comments"), overwrite_existing)
    apply_row_value(sheet, titleblock, ROW_PARAM_MAP["Scale"], row.get("Scale"), overwrite_existing)
    apply_row_value(sheet, titleblock, ROW_PARAM_MAP["Version"], row.get("Version"), overwrite_existing)
    apply_row_value(sheet, titleblock, ROW_PARAM_MAP["Level"], row.get("Level"), overwrite_existing)
    apply_row_value(sheet, titleblock, ROW_PARAM_MAP["Floor View"], row.get("Floor View"), overwrite_existing)

    for key, value in metadata.items():
        apply_metadata_value(sheet, titleblock, project_info, key, value, overwrite_existing)


def push_project_info_to_titleblock(titleblock, project_info, overwrite_existing):
    """Copy built-in ProjectInformation values to matching title block parameters."""
    if titleblock is None or project_info is None:
        return
    for bip, names in PROJECT_INFO_BIP_NAMES:
        try:
            src = project_info.get_Parameter(bip)
            if src is None:
                continue
            value = safe_string(src.AsString() or src.AsValueString())
            if value:
                try_set_parameter_names(titleblock, names, value, overwrite_existing)
        except Exception:
            pass


def read_titleblock_text_scale(titleblock_symbol):
    """Sample the text_scale_factor from an existing instance of this title block type."""
    scale_names = META_PARAM_MAP.get("text_scale_factor", [])
    if not scale_names or titleblock_symbol is None:
        return None
    try:
        collector = FilteredElementCollector(doc).OfCategory(BuiltInCategory.OST_TitleBlocks).WhereElementIsNotElementType()
        for inst in collector:
            try:
                if inst.GetTypeId() != titleblock_symbol.Id:
                    continue
                for name in scale_names:
                    try:
                        param = inst.LookupParameter(name)
                        if param:
                            value = safe_string(param.AsString() or param.AsValueString())
                            if value:
                                return value
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    return None


def create_or_update_sheets(rows, metadata, titleblock_symbol, update_existing, overwrite_existing, duplicate_core_views, create_missing_views):
    existing_by_number = get_existing_sheets_by_number()
    project_info = doc.ProjectInformation
    summary = {"created": 0, "updated": 0, "skipped": 0, "failed": 0}
    messages = []
    seen_numbers = set()

    # Sample text scale from existing instances of this title block so new
    # sheets inherit the same font rendering as those already in the project.
    ref_text_scale = read_titleblock_text_scale(titleblock_symbol)

    transaction = Transaction(doc, "Import Sheets from ArchiTask")
    started = False
    transaction.Start()
    started = True
    try:
        if titleblock_symbol and not titleblock_symbol.IsActive:
            titleblock_symbol.Activate()
            doc.Regenerate()

        apply_project_metadata_once(project_info, metadata, overwrite_existing)

        for row in rows:
            sheet_number = safe_string(row.get("Sheet Number"))
            if not sheet_number:
                summary["skipped"] += 1
                messages.append("Skipped one row with empty Sheet Number.")
                continue
            if sheet_number in seen_numbers:
                summary["skipped"] += 1
                messages.append("Skipped duplicate workbook row: {}".format(sheet_number))
                continue
            seen_numbers.add(sheet_number)

            try:
                sheet = existing_by_number.get(sheet_number)
                was_existing = sheet is not None
                if sheet is not None:
                    if not update_existing:
                        summary["skipped"] += 1
                        continue
                else:
                    sheet = ViewSheet.Create(doc, titleblock_symbol.Id)
                    doc.Regenerate()
                    existing_by_number[sheet_number] = sheet
                    summary["created"] += 1

                titleblock = get_titleblock_instance(sheet)

                apply_sheet_data(sheet, titleblock, project_info, row, metadata, overwrite_existing)
                push_project_info_to_titleblock(titleblock, project_info, overwrite_existing)
                if ref_text_scale and titleblock:
                    try_set_parameter_names(
                        titleblock,
                        META_PARAM_MAP.get("text_scale_factor", []),
                        ref_text_scale,
                        False,
                    )
                place_view_on_sheet_if_possible(sheet, row, duplicate_core_views, create_missing_views, messages)
                if was_existing:
                    summary["updated"] += 1
            except Exception as row_ex:
                summary["failed"] += 1
                messages.append("{} -> {}".format(sheet_number, row_ex))

        transaction.Commit()
    except Exception:
        if started:
            transaction.RollBack()
        raise

    return summary, messages


def main():
    symbols = get_collector_titleblock_symbols()
    if not symbols:
        TaskDialog.Show(__title__, "No titleblock types were found in this Revit model.")
        return

    settings = choose_import_settings(symbols)
    if not settings:
        return

    try:
        rows, metadata = load_architask_workbook(settings["file_path"])
    except Exception as ex:
        TaskDialog.Show(__title__, "Failed to read workbook.\n{}".format(ex))
        return

    if not rows:
        TaskDialog.Show(__title__, "No sheet rows were found in the workbook.")
        return

    try:
        summary, messages = create_or_update_sheets(
            rows,
            metadata,
            settings["titleblock_symbol"],
            settings["update_existing"],
            settings["overwrite_existing"],
            settings["duplicate_core_views"],
            settings["create_missing_views"],
        )
    except Exception as ex:
        TaskDialog.Show(__title__, "Sheet import failed.\n{}".format(ex))
        return

    lines = [
        "Workbook: {}".format(os.path.basename(settings["file_path"])),
        "Created: {}".format(summary.get("created", 0)),
        "Updated: {}".format(summary.get("updated", 0)),
        "Skipped: {}".format(summary.get("skipped", 0)),
        "Failed: {}".format(summary.get("failed", 0)),
    ]
    if messages:
        lines.append("")
        lines.append("Details:")
        for item in messages[:12]:
            lines.append("- {}".format(item))
        if len(messages) > 12:
            lines.append("- ...and {} more".format(len(messages) - 12))

    TaskDialog.Show(__title__, "\n".join(lines))


if __name__ == "__main__":
    main()
