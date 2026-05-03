#! python3
# -*- coding: utf-8 -*-
__title__ = "Import Sheets V2"
__doc__ = "Create Revit sheets and views from an ArchiTask Export SL workbook."

import os
import zipfile
import xml.etree.ElementTree as ET
import sys

from pyrevit import forms
from Autodesk.Revit.DB import (
    BuiltInCategory,
    BuiltInParameter,
    FilteredElementCollector,
    Transaction,
    ViewSheet,
    ViewPlan,
    ViewFamily,
    ViewFamilyType,
    Level,
    Viewport,
    XYZ,
    View,
)

# --- Configuration ---
XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"

doc = __revit__.ActiveUIDocument.Document

def safe_str(val):
    if val is None: return ""
    return str(val).strip()

def norm(s):
    return "".join(c for c in s.lower() if c.isalnum())

def get_name(el):
    try:
        p = el.get_Parameter(BuiltInParameter.SYMBOL_NAME_PARAM)
        if p: return p.AsString()
        return el.Name
    except: return ""

def get_family_name(el):
    try:
        if hasattr(el, 'FamilyName'): return el.FamilyName
        return el.Family.Name
    except: return ""

def find_matching_level(name, sheet_num, chapter, cached_levels):
    search_targets = []
    if name:
        n = norm(name)
        search_targets.append(n)
        if n.startswith('b') and n[1:].isdigit(): search_targets.append("basement" + n[1:])
        if n in ["00", "0", "ground"]: search_targets.extend(["ground", "קרקע", "00", "0", "l0", "level0"])
        if n in ["rf", "roof"]: search_targets.extend(["roof", "גג", "level99", "l99", "rf"])

    ch_num = "".join(c for c in chapter if c.isdigit())
    if sheet_num and ch_num in ['20', '21']:
        digits = "".join([c for c in sheet_num if c.isdigit()])
        if len(digits) >= 3:
            try:
                val = int(digits[-3:])
                target_floor = val - 100
                if target_floor == 0: search_targets.extend(["ground", "קרקע", "00", "0", "l0", "level0"])
                elif target_floor < 0:
                    b_val = str(abs(target_floor))
                    search_targets.extend(["b" + b_val, "basement" + b_val, "p" + b_val])
                elif target_floor == 99: search_targets.extend(["roof", "גג", "rf"])
                else:
                    f_val = str(target_floor)
                    search_targets.extend(["l" + f_val, "level" + f_val, "floor" + f_val, f_val])
            except: pass

    for target in search_targets:
        for l in cached_levels:
            ln = norm(l.Name)
            if target == ln or target in ln or ln in target:
                return l
    
    if ch_num not in ['20', '21'] and cached_levels:
        return cached_levels[0]
    return None

def read_xlsx_sheets(path):
    with zipfile.ZipFile(path, 'r') as z:
        sheet_map = {}
        if 'xl/workbook.xml' in z.namelist():
            wb_root = ET.fromstring(z.read('xl/workbook.xml'))
            for s_node in wb_root.findall(".//" + XML_NS + "sheet"):
                name = s_node.attrib.get('name')
                s_id = s_node.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                sheet_map[name] = s_id
        path_map = {}
        if 'xl/_rels/workbook.xml.rels' in z.namelist():
            rels_root = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
            for r_node in rels_root.findall(REL_NS + "Relationship"):
                r_id = r_node.attrib.get('Id')
                target = r_node.attrib.get('Target')
                path_map[r_id] = "xl/" + target
        target_path = 'xl/worksheets/sheet1.xml'
        sl_id = sheet_map.get('Sheet List') or sheet_map.get('Revit Import')
        if sl_id and sl_id in path_map: target_path = path_map[sl_id]
        elif sheet_map:
            first_id = list(sheet_map.values())[0]
            if first_id in path_map: target_path = path_map[first_id]
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall(XML_NS + 'si'):
                parts = []
                for t in si.findall('.//' + XML_NS + 't'):
                    if t.text: parts.append(t.text)
                strings.append("".join(parts))
        root = ET.fromstring(z.read(target_path))
        rows = []
        for row_node in root.findall(XML_NS + 'sheetData/' + XML_NS + 'row'):
            row_data = {}
            for cell in row_node.findall(XML_NS + 'c'):
                ref = cell.attrib.get('r', '')
                col_letters = "".join([c for c in ref if c.isalpha()])
                col_idx = 0
                for char in col_letters: col_idx = col_idx * 26 + (ord(char.upper()) - ord('A') + 1)
                t = cell.attrib.get('t')
                v_node = cell.find(XML_NS + 'v')
                val = v_node.text if v_node is not None else ""
                if t == 's' and val:
                    try: val = strings[int(val)]
                    except: val = ""
                row_data[col_idx] = val
            rows.append(row_data)
        if not rows: return []
        header_row = rows[0]
        headers = {idx: safe_str(name) for idx, name in header_row.items()}
        data = []
        for r in rows[1:]:
            item = {}
            for idx, val in r.items():
                head = headers.get(idx)
                if head: item[head] = safe_str(val)
            if item.get('Sheet Number'): data.append(item)
        return data

def main():
    try:
        # 1. Collect Data & Titleblocks
        tbs = list(FilteredElementCollector(doc).OfCategory(BuiltInCategory.OST_TitleBlocks).WhereElementIsElementType())
        if not tbs:
            forms.alert("No Titleblocks found.", title="Error")
            return
        tb_options = { "{} : {}".format(get_family_name(x), get_name(x)): x for x in tbs }
        selected_tb_name = forms.SelectFromList.show(sorted(tb_options.keys()), title="Select Titleblock", multiselect=False)
        if not selected_tb_name: return
        selected_tb = tb_options[selected_tb_name]

        path = forms.pick_excel_file(title="Select ArchiTask Revit Export SL")
        if not path: return
        create_views = forms.alert("Automatically create and place views?", yes=True, no=True, title="Create Views?")
        
        print("--- Starting ArchiTask Import ---")
        sheet_data = read_xlsx_sheets(path)
        print("Loaded {} rows from Excel.".format(len(sheet_data)))
        
        # 2. Pre-scan Project (Cache Everything for Speed)
        cached_levels = sorted(list(FilteredElementCollector(doc).OfClass(Level)), key=lambda x: x.Elevation)
        cached_sheets = {s.SheetNumber: s for s in FilteredElementCollector(doc).OfClass(ViewSheet)}
        cached_views = {v.Name: v for v in FilteredElementCollector(doc).OfClass(View) if not v.IsTemplate}
        
        vft_collector = list(FilteredElementCollector(doc).OfClass(ViewFamilyType))
        floor_plan_type = next((v for v in vft_collector if v.ViewFamily == ViewFamily.FloorPlan), None)
        ceiling_plan_type = next((v for v in vft_collector if v.ViewFamily == ViewFamily.CeilingPlan), None)
        
        created_sheets, created_views, placed_views = 0, 0, 0
        
        with Transaction(doc, "ArchiTask: Fast Import") as t:
            t.Start()
            with forms.ProgressBar(title="Processing Sheets...", total=len(sheet_data)) as pb:
                for i, item in enumerate(sheet_data):
                    num = item.get('Sheet Number')
                    name = item.get('Sheet Name')
                    chapter = item.get('Chapter', '').lower()
                    scale_val = item.get('Scale', '100')
                    level_name = item.get('Level', '')
                    view_suggested = item.get('Floor View', '')
                    pb.update_progress(i + 1, len(sheet_data))
                    
                    if not chapter and "_" in num:
                        parts = num.split("_")
                        if len(parts) >= 2 and parts[1].isdigit(): chapter = parts[1]
                    
                    # 3. Sheet Handling
                    sheet = cached_sheets.get(num)
                    if not sheet:
                        try:
                            sheet = ViewSheet.Create(doc, selected_tb.Id)
                            sheet.SheetNumber = num
                            sheet.Name = name
                            created_sheets += 1
                        except: continue
                    
                    # 4. View Handling
                    if create_views:
                        ch_num = "".join(c for c in chapter if c.isdigit())
                        target_name = view_suggested if view_suggested else "{} - {}".format(num, name)
                        
                        if ch_num in ['20', '21', '22', '23', '50', '51', '90']:
                            level = find_matching_level(level_name, num, chapter, cached_levels)
                            
                            # Create missing level if needed (Plans only)
                            if not level and ch_num in ['20', '21']:
                                try:
                                    floor_num = 0
                                    digits = "".join([c for c in num if c.isdigit()])
                                    if len(digits) >= 3: floor_num = int(digits[-3:]) - 100
                                    if floor_num > 100: floor_num = 1
                                    elevation = floor_num * 11.4829
                                    level = Level.Create(doc, elevation)
                                    level.Name = "Level {:02d}".format(floor_num) if floor_num >= 0 else "Level B{}".format(abs(floor_num))
                                    cached_levels.append(level)
                                    print("Created Level: {}".format(level.Name))
                                except: pass
                            
                            if level:
                                v_type = ceiling_plan_type if ch_num == '23' else floor_plan_type
                                if v_type:
                                    view = cached_views.get(target_name)
                                    if not view:
                                        try:
                                            view = ViewPlan.Create(doc, v_type.Id, level.Id)
                                            try: view.Name = target_name
                                            except: view.Name = target_name + " (AT)"
                                            
                                            try:
                                                s = 100
                                                if ":" in scale_val: s = int(scale_val.split(":")[1])
                                                else: s = int(float(scale_val))
                                                view.Scale = s
                                            except: pass
                                            
                                            created_views += 1
                                            cached_views[view.Name] = view
                                        except: pass
                                    
                                    if view and Viewport.CanAddViewToSheet(doc, sheet.Id, view.Id):
                                        Viewport.Create(doc, sheet.Id, view.Id, XYZ(1.5, 1.0, 0))
                                        placed_views += 1

                        elif ch_num in ['30', '31', '32', '40']:
                            # Fast Match Existing Elevations/Sections
                            view = cached_views.get(target_name)
                            if not view:
                                # Loose match fallback
                                for vn, v in cached_views.items():
                                    if target_name in vn or vn in target_name:
                                        view = v; break
                            
                            if view and Viewport.CanAddViewToSheet(doc, sheet.Id, view.Id):
                                Viewport.Create(doc, sheet.Id, view.Id, XYZ(1.5, 1.0, 0))
                                placed_views += 1
            t.Commit()
            
        forms.alert("Import Complete!\n\nSheets: {}\nViews Created: {}\nViews Placed: {}".format(len(sheet_data), created_views, placed_views), title="Success")
    except Exception as e:
        print("FATAL ERROR: {}".format(e))
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
