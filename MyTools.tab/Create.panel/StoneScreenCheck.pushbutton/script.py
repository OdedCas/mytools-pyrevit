# -*- coding: utf-8 -*-
"""Read-only diagnostic for the stone screen curtain walls. Changes nothing."""

from Autodesk.Revit.DB import (
    FilteredElementCollector, Wall, WallType, MullionType, Element,
    BuiltInParameter, StorageType, BuiltInCategory, FamilySymbol
)
import clr

doc = __revit__.ActiveUIDocument.Document
out = []

_NP = clr.GetClrType(Element).GetProperty("Name")


def ename(e):
    if e is None:
        return "<none>"
    try:
        return e.Name
    except Exception:
        pass
    try:
        return _NP.GetValue(e, None) or "<?>"
    except Exception:
        return "<?>"


def pval(p):
    try:
        if p.StorageType == StorageType.String:
            return p.AsString()
        if p.StorageType == StorageType.Integer:
            return p.AsInteger()
        if p.StorageType == StorageType.Double:
            return round(p.AsDouble() * 304.8, 1)
        if p.StorageType == StorageType.ElementId:
            return ename(doc.GetElement(p.AsElementId()))
    except Exception:
        pass
    return "?"


TARGETS = ("CW - Thin Mullion", "CW - Stone Screen")

out.append("=== MULLION TYPES IN PROJECT ===")
for mt in FilteredElementCollector(doc).OfClass(MullionType):
    out.append("  " + ename(mt))

out.append("")
out.append("=== WALL TYPES ===")
for wt in FilteredElementCollector(doc).OfClass(WallType):
    n = ename(wt)
    if n not in TARGETS:
        continue
    out.append("[" + n + "]  Kind=" + str(wt.Kind))
    # dump EVERY type parameter with its BuiltInParameter enum name - the
    # display names are ambiguous (vertical and horizontal both say "Layout")
    for p in wt.Parameters:
        try:
            pn = p.Definition.Name
        except Exception:
            continue
        try:
            bip = str(p.Definition.BuiltInParameter)
        except Exception:
            bip = "<not builtin>"
        out.append("    {0:<26} | {1:<40} = {2}".format(pn, bip, pval(p)))

out.append("")
out.append("=== WALLS ===")


def report_wall(w, tn):
    out.append("Wall id {0}  type '{1}'".format(w.Id, tn))
    try:
        out.append("    length = {0} mm".format(
            round(w.get_Parameter(BuiltInParameter.CURVE_ELEM_LENGTH).AsDouble() * 304.8, 1)))
    except Exception:
        pass
    # --- is the wall itself upright and level? -----------------------------
    try:
        lc = w.Location
        c = lc.Curve
        a = c.GetEndPoint(0)
        b = c.GetEndPoint(1)
        out.append("    location start = ({0}, {1}, {2}) mm".format(
            round(a.X * 304.8, 1), round(a.Y * 304.8, 1), round(a.Z * 304.8, 1)))
        out.append("    location end   = ({0}, {1}, {2}) mm".format(
            round(b.X * 304.8, 1), round(b.Y * 304.8, 1), round(b.Z * 304.8, 1)))
        dz = abs(a.Z - b.Z) * 304.8
        out.append("    location dZ    = {0} mm {1}".format(
            round(dz, 1), "<-- NOT LEVEL" if dz > 1.0 else "(level, OK)"))
        out.append("    curve type     = {0}".format(c.GetType().Name))
    except Exception as ex:
        out.append("    location = error: {0}".format(ex))
    for bn in ("WALL_BASE_OFFSET", "WALL_TOP_OFFSET", "WALL_USER_HEIGHT_PARAM",
               "WALL_BASE_CONSTRAINT", "WALL_HEIGHT_TYPE"):
        b2 = getattr(BuiltInParameter, bn, None)
        if b2 is None:
            continue
        try:
            pr = w.get_Parameter(b2)
            if pr:
                out.append("    {0:<26} = {1}".format(bn, pval(pr)))
        except Exception:
            pass

    cg = w.CurtainGrid
    if cg is None:
        out.append("    !! no CurtainGrid - this is NOT a curtain wall")
        return
    # CurtainGrid speaks U/V, not H/V; probe what this build exposes.
    for label in ("GetUGridLineIds", "GetVGridLineIds"):
        fn = getattr(cg, label, None)
        if fn is None:
            out.append("    {0} = <absent>".format(label))
            continue
        try:
            out.append("    {0} = {1}".format(label, len(list(fn()))))
        except Exception as ex:
            out.append("    {0} = error: {1}".format(label, ex))
    for nm in ("NumULines", "NumVLines"):
        v = getattr(cg, nm, None)
        if v is not None:
            out.append("    {0} = {1}".format(nm, v))
    try:
        out.append("    panels   = {0}".format(len(list(cg.GetPanelIds()))))
    except Exception as ex:
        out.append("    panels   = error: {0}".format(ex))
    try:
        mids = list(cg.GetMullionIds())
        out.append("    mullions = {0}".format(len(mids)))
        seen = {}
        for mid in mids:
            m = doc.GetElement(mid)
            k = ename(m.MullionType) if m else "?"
            seen[k] = seen.get(k, 0) + 1
        for k in seen:
            out.append("        {0} x {1}".format(seen[k], k))
        # are the mullions actually vertical?
        shown = 0
        for mid in mids:
            if shown >= 4:
                break
            m = doc.GetElement(mid)
            try:
                mc = m.Location.Curve
                s0 = mc.GetEndPoint(0)
                s1 = mc.GetEndPoint(1)
                v = s1 - s0
                ln = (v.X ** 2 + v.Y ** 2 + v.Z ** 2) ** 0.5
                vert = abs(v.Z) / ln if ln else 0
                import math
                ang = math.degrees(math.acos(min(1.0, vert)))
                out.append("        mullion {0}: len={1}mm  off-vertical={2} deg {3}".format(
                    mid, round(ln * 304.8, 1), round(ang, 2),
                    "<-- ANGLED" if ang > 0.5 else ""))
                shown += 1
            except Exception as ex:
                out.append("        mullion {0}: no curve ({1})".format(mid, ex))
                shown += 1
    except Exception as ex:
        out.append("    mullions = error: {0}".format(ex))


found = 0
for w in FilteredElementCollector(doc).OfClass(Wall):
    tn = ename(w.WallType)
    if tn not in TARGETS:
        continue
    found += 1
    try:
        report_wall(w, tn)
    except Exception as ex:
        out.append("    !! wall report failed: {0}".format(ex))

if not found:
    out.append("  none found - the walls were not created")


out.append("")
out.append("=== STRUCTURAL COLUMNS ===")
cols = list(FilteredElementCollector(doc)
            .OfCategory(BuiltInCategory.OST_StructuralColumns)
            .WhereElementIsNotElementType())
out.append("count = {0}".format(len(cols)))
shown = 0
for c in cols:
    if shown >= 6:
        out.append("  ... and {0} more".format(len(cols) - shown))
        break
    try:
        tn = ename(doc.GetElement(c.GetTypeId()))
    except Exception:
        tn = "?"
    line = "  id {0}  type '{1}'  slanted={2}".format(
        c.Id, tn, getattr(c, "IsSlantedColumn", "n/a"))
    try:
        loc = c.Location
        cv = getattr(loc, "Curve", None)
        if cv is not None:
            a2 = cv.GetEndPoint(0)
            b2 = cv.GetEndPoint(1)
            line += "  base=({0},{1},{2}) top=({3},{4},{5})".format(
                round(a2.X * 304.8), round(a2.Y * 304.8), round(a2.Z * 304.8),
                round(b2.X * 304.8), round(b2.Y * 304.8), round(b2.Z * 304.8))
        else:
            pt = getattr(loc, "Point", None)
            if pt is not None:
                line += "  point=({0},{1},{2}) VERTICAL".format(
                    round(pt.X * 304.8), round(pt.Y * 304.8), round(pt.Z * 304.8))
    except Exception as ex:
        line += "  loc error: {0}".format(ex)
    out.append(line)
    shown += 1

out.append("")
out.append("=== STRUCTURAL COLUMN TYPES LOADED ===")
syms = list(FilteredElementCollector(doc).OfClass(FamilySymbol)
            .OfCategory(BuiltInCategory.OST_StructuralColumns))
out.append("types available = {0}".format(len(syms)))
for sy in syms[:10]:
    out.append("  " + ename(sy))
if not syms:
    out.append("  NONE LOADED - that alone explains zero slats")

out.append("")
out.append("=== ACTIVE VIEW ===")
v = doc.ActiveView
out.append("view '{0}'  type={1}".format(ename(v), v.ViewType))
try:
    cat = doc.Settings.Categories.get_Item(BuiltInCategory.OST_StructuralColumns)
    out.append("Structural Columns hidden here = {0}".format(
        v.GetCategoryHidden(cat.Id)))
except Exception as ex:
    out.append("visibility check failed: {0}".format(ex))
try:
    out.append("discipline = {0}".format(v.Discipline))
except Exception:
    pass

text = "\n".join(out)
print(text)
try:
    from pyrevit import script
    script.get_output().print_md("```\n" + text + "\n```")
except Exception:
    pass
