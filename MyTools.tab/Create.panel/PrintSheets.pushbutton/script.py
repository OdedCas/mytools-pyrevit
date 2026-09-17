# -*- coding: utf-8 -*-
"""Print selected sheets with configurable print parameters."""

import clr
clr.AddReference('PresentationFramework')
clr.AddReference('PresentationCore')
clr.AddReference('WindowsBase')

import System.Windows as sw
import System.Windows.Controls as swc
import System.Windows.Media as swm

from pyrevit import revit, DB, forms, script

# These enums live directly in Autodesk.Revit.DB
PageOrientationType = DB.PageOrientationType
ZoomType            = DB.ZoomType
ColorDepthType      = DB.ColorDepthType
PrintRange          = DB.PrintRange

doc = revit.doc
logger = script.get_logger()


# ── helpers ────────────────────────────────────────────────────────────────

def get_all_sheets():
    return (
        DB.FilteredElementCollector(doc)
        .OfClass(DB.ViewSheet)
        .WhereElementIsNotElementType()
        .ToElements()
    )


def sheet_display_name(sheet):
    return "{} - {}".format(sheet.SheetNumber, sheet.Name)


# ── print-parameter dialog ─────────────────────────────────────────────────

class PrintOptions(object):
    def __init__(self):
        self.printer_name   = ""
        self.paper_size     = "A3"
        self.orientation    = "Landscape"
        self.zoom_type      = "FitToPage"
        self.zoom_percent   = 100
        self.color_depth    = "Color"
        self.hidden_views   = False
        self.combine        = False
        self.reverse_order  = False
        self.print_to_pdf   = False
        self.pdf_path       = ""
        self.in_session     = False   # True = CurrentWindow, False = Select
        self.use_sheet_size = False   # True = detect paper size from each sheet's title block


class PrintSettingsDialog(sw.Window):
    PAPER_SIZES  = ["A0", "A1", "A2", "A3", "A4", "Letter", "Legal", "Tabloid"]
    ORIENTATIONS = ["Landscape", "Portrait"]
    ZOOM_TYPES   = ["FitToPage", "Zoom"]
    COLOR_DEPTHS = ["Color", "BlackLine", "GrayScale"]

    def __init__(self, printer_names):
        self.Title  = "Print Settings"
        self.Width  = 400
        self.SizeToContent = sw.SizeToContent.Height
        self.WindowStartupLocation = sw.WindowStartupLocation.CenterScreen
        self.ResizeMode = sw.ResizeMode.NoResize
        self.result = None

        outer = swc.StackPanel()
        outer.Margin = sw.Thickness(16, 12, 16, 12)

        def section(label):
            lbl = swc.Label()
            lbl.Content = label
            lbl.FontWeight = sw.FontWeights.SemiBold
            lbl.Padding = sw.Thickness(0, 8, 0, 2)
            outer.Children.Add(lbl)

        def combo(items, default=None):
            cb = swc.ComboBox()
            cb.Margin = sw.Thickness(0, 0, 0, 2)
            for item in items:
                cb.Items.Add(item)
            cb.SelectedIndex = items.index(default) if default in items else 0
            outer.Children.Add(cb)
            return cb

        def checkbox(text, checked=False):
            chk = swc.CheckBox()
            chk.Content = text
            chk.IsChecked = checked
            chk.Margin = sw.Thickness(0, 6, 0, 0)
            outer.Children.Add(chk)
            return chk

        # ── Print Range ────────────────────────────────────────────────────
        section("Print Range")

        range_panel = swc.StackPanel()
        range_panel.Margin = sw.Thickness(0, 2, 0, 0)

        self._rb_selected = swc.RadioButton()
        self._rb_selected.Content  = "Selected sheets"
        self._rb_selected.IsChecked = True
        self._rb_selected.Margin   = sw.Thickness(0, 2, 0, 2)
        self._rb_selected.Checked += self._on_range_change

        self._rb_session = swc.RadioButton()
        self._rb_session.Content = "Current view (in session — print as shown)"
        self._rb_session.Margin  = sw.Thickness(0, 2, 0, 2)
        self._rb_session.Checked += self._on_range_change

        range_panel.Children.Add(self._rb_selected)
        range_panel.Children.Add(self._rb_session)
        outer.Children.Add(range_panel)

        sep0 = swc.Separator()
        sep0.Margin = sw.Thickness(0, 10, 0, 0)
        outer.Children.Add(sep0)

        # ── Printer ────────────────────────────────────────────────────────
        section("Printer")
        self._cb_printer = combo(printer_names)

        section("Paper Size")
        self._chk_sheet_size = swc.CheckBox()
        self._chk_sheet_size.Content = "Use each sheet's own paper size"
        self._chk_sheet_size.IsChecked = True
        self._chk_sheet_size.Margin = sw.Thickness(0, 2, 0, 4)
        self._chk_sheet_size.Checked   += self._on_sheet_size_change
        self._chk_sheet_size.Unchecked += self._on_sheet_size_change
        outer.Children.Add(self._chk_sheet_size)

        self._cb_paper = combo(self.PAPER_SIZES, "A3")
        self._cb_paper.IsEnabled = False

        section("Orientation")
        self._cb_orient = combo(self.ORIENTATIONS, "Landscape")
        self._cb_orient.IsEnabled = False

        section("Zoom")
        zoom_row = swc.StackPanel()
        zoom_row.Orientation = swc.Orientation.Horizontal
        zoom_row.Margin = sw.Thickness(0, 0, 0, 2)

        self._cb_zoom = swc.ComboBox()
        self._cb_zoom.Width = 120
        self._cb_zoom.Margin = sw.Thickness(0, 0, 12, 0)
        for zt in self.ZOOM_TYPES:
            self._cb_zoom.Items.Add(zt)
        self._cb_zoom.SelectedIndex = 0

        zoom_pct_lbl = swc.Label()
        zoom_pct_lbl.Content = "Zoom %:"
        zoom_pct_lbl.Padding = sw.Thickness(0, 3, 6, 0)

        self._tb_zoom = swc.TextBox()
        self._tb_zoom.Text = "100"
        self._tb_zoom.Width = 50
        self._tb_zoom.VerticalContentAlignment = sw.VerticalAlignment.Center

        zoom_row.Children.Add(self._cb_zoom)
        zoom_row.Children.Add(zoom_pct_lbl)
        zoom_row.Children.Add(self._tb_zoom)
        outer.Children.Add(zoom_row)

        section("Color Depth")
        self._cb_color = combo(self.COLOR_DEPTHS, "Color")

        sep = swc.Separator()
        sep.Margin = sw.Thickness(0, 12, 0, 4)
        outer.Children.Add(sep)

        self._chk_hidden  = checkbox("Hide scope boxes, crop boundaries, reference planes")

        # these only make sense for multi-sheet (selected) mode
        self._chk_combine = checkbox("Combine sheets into a single file")
        self._chk_reverse = checkbox("Reverse sheet order")

        self._chk_pdf     = checkbox("Print to PDF  (opens Save dialog)")

        sep2 = swc.Separator()
        sep2.Margin = sw.Thickness(0, 12, 0, 8)
        outer.Children.Add(sep2)

        btn_row = swc.StackPanel()
        btn_row.Orientation = swc.Orientation.Horizontal
        btn_row.HorizontalAlignment = sw.HorizontalAlignment.Right

        btn_ok = swc.Button()
        btn_ok.Content   = "Print"
        btn_ok.Width     = 90
        btn_ok.IsDefault = True
        btn_ok.Margin    = sw.Thickness(0, 0, 8, 0)
        btn_ok.Click    += self._on_ok

        btn_cancel = swc.Button()
        btn_cancel.Content  = "Cancel"
        btn_cancel.Width    = 90
        btn_cancel.IsCancel = True
        btn_cancel.Click   += self._on_cancel

        btn_row.Children.Add(btn_ok)
        btn_row.Children.Add(btn_cancel)
        outer.Children.Add(btn_row)

        self.Content = outer

    def _on_sheet_size_change(self, sender, e):
        use_sheet = self._checked(self._chk_sheet_size)
        self._cb_paper.IsEnabled  = not use_sheet
        self._cb_orient.IsEnabled = not use_sheet

    def _on_range_change(self, sender, e):
        in_session = self._checked(self._rb_session)
        self._chk_combine.IsEnabled     = not in_session
        self._chk_reverse.IsEnabled     = not in_session
        self._chk_sheet_size.IsEnabled  = not in_session
        if in_session:
            self._cb_paper.IsEnabled  = True
            self._cb_orient.IsEnabled = True
        else:
            self._on_sheet_size_change(None, None)

    def _checked(self, chk):
        v = chk.IsChecked
        return bool(v) if v is not None else False

    def _on_ok(self, sender, e):
        po = PrintOptions()
        po.in_session    = self._checked(self._rb_session)
        po.printer_name  = self._cb_printer.SelectedItem or ""
        po.paper_size    = self._cb_paper.SelectedItem   or "A3"
        po.orientation   = self._cb_orient.SelectedItem  or "Landscape"
        po.zoom_type     = self._cb_zoom.SelectedItem    or "FitToPage"
        try:
            po.zoom_percent = max(1, min(int(self._tb_zoom.Text), 100))
        except (ValueError, TypeError):
            po.zoom_percent = 100
        po.color_depth      = self._cb_color.SelectedItem   or "Color"
        po.hidden_views     = self._checked(self._chk_hidden)
        po.combine          = self._checked(self._chk_combine) and not po.in_session
        po.reverse_order    = self._checked(self._chk_reverse) and not po.in_session
        po.print_to_pdf     = self._checked(self._chk_pdf)
        po.use_sheet_size   = self._checked(self._chk_sheet_size) and not po.in_session
        self.result      = po
        self.DialogResult = True

    def _on_cancel(self, sender, e):
        self.DialogResult = False


def pick_pdf_path(sheet_count):
    import System.Windows.Forms as swf
    dlg = swf.SaveFileDialog()
    dlg.Title      = "Save PDF"
    dlg.Filter     = "PDF files (*.pdf)|*.pdf|All files (*.*)|*.*"
    dlg.DefaultExt = "pdf"
    dlg.FileName   = "sheets_{}_pages".format(sheet_count)
    if dlg.ShowDialog() != swf.DialogResult.OK or not dlg.FileName:
        script.exit()
    return dlg.FileName


def ask_print_options(printer_names):
    dlg = PrintSettingsDialog(list(printer_names))
    if not dlg.ShowDialog():
        script.exit()
    return dlg.result


# ── apply settings to PrintManager ─────────────────────────────────────────

COLOR_MAP = {
    "Color":      ColorDepthType.Color,
    "BlackLine":  ColorDepthType.BlackLine,
    "GrayScale":  ColorDepthType.GrayScale,
}

ORIENTATION_MAP = {
    "Landscape": PageOrientationType.Landscape,
    "Portrait":  PageOrientationType.Portrait,
}

ZOOM_MAP = {
    "FitToPage": ZoomType.FitToPage,
    "Zoom":      ZoomType.Zoom,
}


def find_paper_size(pm, name):
    for ps in pm.PaperSizes:
        if ps.Name.lower().startswith(name.lower()):
            return ps
    return None


def get_sheet_dims_mm(doc, sheet):
    """Return (width_mm, height_mm) from the sheet's title block, or None."""
    tbs = (DB.FilteredElementCollector(doc)
           .OfCategory(DB.BuiltInCategory.OST_TitleBlocks)
           .OwnedByView(sheet.Id)
           .ToElements())
    for tb in tbs:
        w_p = tb.get_Parameter(DB.BuiltInParameter.SHEET_WIDTH)
        h_p = tb.get_Parameter(DB.BuiltInParameter.SHEET_HEIGHT)
        if w_p and h_p:
            return (w_p.AsDouble() * 304.8, h_p.AsDouble() * 304.8)  # feet → mm
    return None


def find_paper_size_by_dims(pm, w_mm, h_mm, tol_mm=15.0):
    """Find the printer PaperSize matching the given sheet dimensions (mm).
    Queries the Windows printer driver for actual paper sizes with dimensions,
    then looks up the matched name in Revit's PaperSize list."""
    long_side  = max(w_mm, h_mm)
    short_side = min(w_mm, h_mm)

    try:
        import System.Drawing.Printing as sdp
        prt = sdp.PrinterSettings()
        prt.PrinterName = pm.PrinterName
        best_name = None
        best_diff = float('inf')
        for ps in prt.PaperSizes:
            # Width/Height are in hundredths of an inch
            pw_mm = ps.Width  / 100.0 * 25.4
            ph_mm = ps.Height / 100.0 * 25.4
            pl = max(pw_mm, ph_mm)
            ps_short = min(pw_mm, ph_mm)
            diff = abs(pl - long_side) + abs(ps_short - short_side)
            if diff < best_diff:
                best_diff = diff
                best_name = ps.PaperName
        if best_diff > tol_mm * 2 or best_name is None:
            return None
        # match back to Revit's PaperSize list by name
        for rps in pm.PaperSizes:
            if rps.Name.lower() == best_name.lower():
                return rps
        # fallback: startswith match
        return find_paper_size(pm, best_name)
    except Exception as ex:
        logger.warning("Driver paper-size query failed ({}), falling back to ISO table.".format(ex))

    # ISO fallback table
    iso = [("A0",1189,841),("A1",841,594),("A2",594,420),("A3",420,297),("A4",297,210)]
    best_name, best_diff = None, float('inf')
    for name, l, s in iso:
        diff = abs(long_side - l) + abs(short_side - s)
        if diff < best_diff:
            best_diff, best_name = diff, name
    if best_diff > tol_mm * 2 or best_name is None:
        return None
    return find_paper_size(pm, best_name)


def _safe_set(obj, attr, val):
    try:
        setattr(obj, attr, val)
    except (AttributeError, Exception):
        pass


def apply_options(pm, po, paper_size_override=None, orientation_override=None):
    pp = pm.PrintSetup.CurrentPrintSetting.PrintParameters

    if paper_size_override is not None:
        pp.PaperSize = paper_size_override
    elif not po.use_sheet_size:
        paper = find_paper_size(pm, po.paper_size)
        if paper:
            pp.PaperSize = paper
        else:
            logger.warning("Paper size '{}' not found — using printer default.".format(po.paper_size))

    if orientation_override is not None:
        pp.PageOrientation = orientation_override
    elif not po.use_sheet_size:
        pp.PageOrientation = ORIENTATION_MAP.get(po.orientation, PageOrientationType.Landscape)

    pp.ZoomType = ZOOM_MAP.get(po.zoom_type, ZoomType.FitToPage)
    if po.zoom_type == "Zoom":
        pp.Zoom = po.zoom_percent

    pp.ColorDepth = COLOR_MAP.get(po.color_depth, ColorDepthType.Color)

    for _attr in ("HideScopeBoxes", "HideCropBoundaries", "HideReferencePlane",
                  "HideReferencePlanes", "HideUnreferencedViewTags"):
        try:
            setattr(pp, _attr, po.hidden_views)
        except AttributeError:
            pass

    if po.in_session:
        pm.PrintRange = PrintRange.CurrentWindow
    else:
        pm.PrintRange = PrintRange.Select
        _safe_set(pm, "CombinedFile",  po.combine)
        _safe_set(pm, "ReverseOrder",  po.reverse_order)

    _safe_set(pm, "PrintToFile", po.print_to_pdf)
    if po.print_to_pdf and po.pdf_path:
        _safe_set(pm, "PrintToFileName", po.pdf_path)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    # 1. set up PrintManager first to get printer list
    pm = doc.PrintManager
    pm.SelectNewPrintDriver(pm.PrinterName)

    printer_names = [pm.PrinterName]
    try:
        import System.Drawing.Printing as sdp
        printer_names = list(sdp.PrinterSettings.InstalledPrinters)
    except Exception:
        pass

    if not printer_names:
        forms.alert("No printers found.", exitscript=True)

    # 2. show settings dialog
    po = ask_print_options(printer_names)

    # 3. for "in session" we skip the sheet picker entirely
    if po.in_session:
        sheets_to_print = []
    else:
        all_sheets = list(get_all_sheets())
        if not all_sheets:
            forms.alert("No sheets found in the model.", exitscript=True)
        all_sheets.sort(key=lambda s: s.SheetNumber)

        selected = forms.SelectFromList.show(
            [sheet_display_name(s) for s in all_sheets],
            title="Select Sheets to Print",
            multiselect=True,
            button_name="Next: Confirm & Print"
        )
        if not selected:
            script.exit()

        sheets_to_print = [
            s for s in all_sheets if sheet_display_name(s) in selected
        ]

    # 4. if PDF output, ask for file path
    if po.print_to_pdf:
        po.pdf_path = pick_pdf_path(len(sheets_to_print) if sheets_to_print else 1)

    # 5. switch printer if needed
    if po.printer_name and po.printer_name != pm.PrinterName:
        try:
            pm.SelectNewPrintDriver(po.printer_name)
        except Exception as e:
            logger.warning("Could not switch printer: {}".format(e))

    # 6. confirm
    if po.in_session:
        range_line  = "Range   : Current view (in session)"
        sheet_lines = ""
        paper_line  = "Paper   : {}  {}".format(po.paper_size, po.orientation)
    else:
        range_line  = "Range   : {} sheet(s)".format(len(sheets_to_print))
        sheet_lines = "\n".join("  " + sheet_display_name(s) for s in sheets_to_print) + "\n\n"
        paper_line  = ("Paper   : each sheet's own size"
                       if po.use_sheet_size
                       else "Paper   : {}  {}".format(po.paper_size, po.orientation))

    output_line = (
        "Output  : {}".format(po.pdf_path)
        if po.print_to_pdf
        else "Printer : {}".format(po.printer_name)
    )
    msg = (
        "{}\n\n"
        "{}"
        "{}\n"
        "{}\n"
        "Zoom    : {}{}\n"
        "Color   : {}"
    ).format(
        range_line,
        sheet_lines,
        output_line,
        paper_line,
        po.zoom_type,
        " {}%".format(po.zoom_percent) if po.zoom_type == "Zoom" else "",
        po.color_depth,
    )

    if not forms.alert(msg, yes=True, no=True, title="Confirm Print"):
        script.exit()

    # 7. print
    if po.in_session or not po.use_sheet_size:
        # single SubmitPrint for all sheets at once
        with revit.Transaction("Configure Print Settings"):
            apply_options(pm, po)
            if not po.in_session:
                vss = pm.ViewSheetSetting
                view_set = DB.ViewSet()
                for sheet in sheets_to_print:
                    view_set.Insert(sheet)
                vss.CurrentViewSheetSet.Views = view_set
                vss.SaveAs("_PrintSheets_Temp")
        pm.SubmitPrint()
    else:
        # per-sheet printing: detect each sheet's paper size from its title block
        for sheet in sheets_to_print:
            dims = get_sheet_dims_mm(doc, sheet)
            paper_override = None
            orient_override = None
            if dims:
                w_mm, h_mm = dims
                paper_override = find_paper_size_by_dims(pm, w_mm, h_mm)
                if paper_override is None:
                    logger.warning(
                        "No matching paper size for sheet {} ({:.0f}x{:.0f}mm) — skipping.".format(
                            sheet.SheetNumber, w_mm, h_mm))
                    continue
                orient_override = (PageOrientationType.Landscape
                                   if w_mm >= h_mm
                                   else PageOrientationType.Portrait)
            else:
                logger.warning("No title block on sheet {} — skipping.".format(sheet.SheetNumber))
                continue

            with revit.Transaction("Print {}".format(sheet.SheetNumber)):
                apply_options(pm, po,
                              paper_size_override=paper_override,
                              orientation_override=orient_override)
                vss = pm.ViewSheetSetting
                view_set = DB.ViewSet()
                view_set.Insert(sheet)
                vss.CurrentViewSheetSet.Views = view_set
                try:
                    vss.SaveAs("_PrintSheets_Temp")
                except Exception:
                    vss.Save()

            pm.SubmitPrint()

    if po.print_to_pdf:
        forms.alert("PDF saved: {}".format(po.pdf_path))
    else:
        forms.alert("Print job sent.")


main()
