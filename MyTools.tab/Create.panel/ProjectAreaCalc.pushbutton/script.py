# -*- coding: utf-8 -*-
"""Calculates project areas from an Area Plan (View) and exports to CSV.
Optimized for Revit 2025 (.NET 8)."""

import clr
clr.AddReference('RevitAPI')
clr.AddReference('RevitServices')

from Autodesk.Revit.DB import *
from pyrevit import forms, revit, script
import csv

doc = revit.doc

def get_area_sqm(area_element):
    """
    Converts internal units to Square Meters using modern ForgeTypeId.
    Compatible with Revit 2022-2025+.
    """
    # Get internal area value
    internal_area = area_element.get_Parameter(BuiltInParameter.ROOM_AREA).AsDouble()
    
    # Use UnitUtils for reliable conversion in Revit 2025
    try:
        # SpecTypeId.Area is the modern way to handle this
        sqm_val = UnitUtils.ConvertFromInternalUnits(internal_area, UnitTypeId.SquareMeters)
        return round(sqm_val, 2)
    except:
        # Fallback for older API versions if needed
        return round(internal_area * 0.09290304, 2)

def categorize_area(name):
    """Categorizes area based on Hebrew keywords."""
    if not name:
        return "Other"
    
    # Handle both IronPython and CPython string types
    name_str = str(name)
    if "עיקרי" in name_str:
        return "Main (עיקרי)"
    if "שירות" in name_str or "שרות" in name_str:
        return "Service (שירות)"
    return "Other"

def get_building_name(area_element):
    """Attempts to find a Building parameter."""
    params = ["Building", "Bldg", "בניין", "בנין"]
    for p_name in params:
        p = area_element.LookupParameter(p_name)
        if p and p.HasValue:
            return p.AsString()
    return "Default Building"

# 1. Selection: Choose Area Plan (View)
all_views = FilteredElementCollector(doc).OfClass(ViewPlan).ToElements()
area_plans = [v for v in all_views if v.ViewType == ViewType.AreaPlan and not v.IsTemplate]

if not area_plans:
    forms.alert("No Area Plans found in project.")
    script.exit()

view_dict = {"{} ({})".format(v.Name, v.AreaScheme.Name): v for v in area_plans}
selected_view_name = forms.SelectFromList.show(
    sorted(view_dict.keys()), 
    title="Select Area Plan (View) for Calculation", 
    multiselect=False
)

if not selected_view_name:
    script.exit()

selected_view = view_dict[selected_view_name]

# 2. Collection: Areas in that specific View
areas = FilteredElementCollector(doc, selected_view.Id).OfClass(Area).ToElements()

if not areas:
    forms.alert("No Areas found in view: {}".format(selected_view.Name))
    script.exit()

# 3. Processing
results = []
summary = {} 

for area in areas:
    name = area.get_Parameter(BuiltInParameter.ROOM_NAME).AsString() or "Unnamed"
    level_el = doc.GetElement(area.LevelId)
    level_name = level_el.Name if level_el else "Unknown Level"
    
    building = get_building_name(area)
    sqm = get_area_sqm(area)
    cat = categorize_area(name)
    
    results.append({
        'Building': building,
        'Level': level_name,
        'Name': name,
        'Category': cat,
        'Area': sqm
    })
    
    if building not in summary:
        summary[building] = {}
    if level_name not in summary[building]:
        summary[building][level_name] = {'Main (עיקרי)': 0.0, 'Service (שירות)': 0.0, 'Other': 0.0}
    
    summary[building][level_name][cat] += sqm

# 4. Prepare CSV Data
csv_rows = []
csv_rows.append(["Project Area Calculation Report (Revit 2025)"])
csv_rows.append(["Source View:", selected_view.Name])
csv_rows.append([])

csv_rows.append(["Detailed Area List"])
csv_rows.append(["Building", "Level", "Area Name", "Category", "Area (sqm)"])

results.sort(key=lambda x: (x['Building'], x['Level'], x['Name']))

for r in results:
    csv_rows.append([r['Building'], r['Level'], r['Name'], r['Category'], r['Area']])

csv_rows.append([])
csv_rows.append(["Summary Table (by Building & Level)"])
csv_rows.append(["Building", "Level", "Total Main (עיקרי)", "Total Service (שירות)", "Other", "Total Level"])

for bldg in sorted(summary.keys()):
    bldg_total_m = 0
    bldg_total_s = 0
    bldg_total_o = 0
    
    for lvl in sorted(summary[bldg].keys()):
        m = summary[bldg][lvl]['Main (עיקרי)']
        s = summary[bldg][lvl]['Service (שירות)']
        o = summary[bldg][lvl]['Other']
        total = m + s + o
        
        bldg_total_m += m
        bldg_total_s += s
        bldg_total_o += o
        
        csv_rows.append([bldg, lvl, round(m, 2), round(s, 2), round(o, 2), round(total, 2)])
    
    b_total = bldg_total_m + bldg_total_s + bldg_total_o
    csv_rows.append(["SUBTOTAL: " + bldg, "", round(bldg_total_m, 2), round(bldg_total_s, 2), round(bldg_total_o, 2), round(b_total, 2)])
    csv_rows.append([])

# 5. Export
output_file = forms.save_file(file_ext='csv', default_name='Project_Area_Report.csv')
if output_file:
    try:
        # Use 'w' with newline='' for Python 3/CPython (Revit 2025)
        import sys
        mode = 'wb' if sys.version_info[0] < 3 else 'w'
        kwargs = {} if sys.version_info[0] < 3 else {'newline': '', 'encoding': 'utf-8-sig'}
        
        with open(output_file, mode, **kwargs) as f:
            writer = csv.writer(f)
            for row in csv_rows:
                writer.writerow(row)
        forms.alert("Exported successfully to: " + output_file)
    except Exception as e:
        forms.alert("Failed to export: " + str(e))
