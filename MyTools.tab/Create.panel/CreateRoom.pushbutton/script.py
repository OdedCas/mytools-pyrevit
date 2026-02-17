# -*- coding: utf-8 -*-
__title__ = "Create Room From Image"
__doc__ = "Rectify sketch, parse dimensions (cm), and generate room + openings + dimensions."

import os
import sys
import clr

clr.AddReference("System.Windows.Forms")
clr.AddReference("System.Drawing")
import System.Windows.Forms as WinForms

from System.Windows.Forms import (
    OpenFileDialog,
    DialogResult,
    Form,
    Label,
    TextBox,
    Button,
    FormBorderStyle,
)
from System.Drawing import Image as DrawingImage

from snapshot_manager import SnapshotRun
from rectification import build_rectification, apply_homography, distance, RectificationError
from ocr_extract import run_easyocr
from measurement_parser import parse_ocr_tokens, merge_measurements_cm, DEFAULTS_CM
from geometry_builder import build_room_cm
from overlay_layout import compute_overlay_and_note_layout
from cad_extract import load_layer_map, load_cad_config, get_imported_cad_instances, extract_cad_from_view
from cad_classify import classify_entities
from cad_topology import derive_room_from_cad
from cad_report import save_cad_snapshot_stages

from Autodesk.Revit.DB import (
    BuiltInCategory,
    BuiltInParameter,
    CurveLoop,
    ElementTypeGroup,
    DrawLayer,
    FamilyInstanceReferenceType,
    FamilySymbol,
    FilteredElementCollector,
    Floor,
    FloorType,
    DimensionType,
    FormatOptions,
    UnitTypeId,
    SpecTypeId,
    HostObjectUtils,
    Level,
    Line,
    ReferenceArray,
    ShellLayerType,
    Structure,
    TextNote,
    TextNoteType,
    Transaction,
    ViewPlan,
    ViewType,
    Wall,
    WallKind,
    WallType,
    ImageType,
    ImageTypeOptions,
    ImageTypeSource,
    ImageInstance,
    ImagePlacementOptions,
    BoxPlacement,
    PlanViewPlane,
    ElementId,
    DWGImportOptions,
    XYZ,
)
from Autodesk.Revit.Exceptions import OperationCanceledException
from Autodesk.Revit.UI import TaskDialog


uidoc = __revit__.ActiveUIDocument
doc = uidoc.Document


CM_PER_FT = 30.48
KNOWN_CALIBRATION_CM = 100.0
_CM_DIM_TYPE_ID = None


def ft_to_cm(value_ft):
    return float(value_ft) * CM_PER_FT


def cm_to_ft(value_cm):
    return float(value_cm) / CM_PER_FT


def xyz_ft_to_cm_tuple(pt):
    return (ft_to_cm(pt.X), ft_to_cm(pt.Y), ft_to_cm(pt.Z))


def xy_ft_to_cm_tuple(pt):
    return (ft_to_cm(pt.X), ft_to_cm(pt.Y))


def xy_cm_to_xyz_ft(pt, z_ft):
    return XYZ(cm_to_ft(pt[0]), cm_to_ft(pt[1]), z_ft)


def first_or_none(iterable):
    for item in iterable:
        return item
    return None


def add(a, b):
    return XYZ(a.X + b.X, a.Y + b.Y, a.Z + b.Z)


def clamp_value(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def get_plan_view():
    active = uidoc.ActiveView
    if isinstance(active, ViewPlan) and (not active.IsTemplate) and active.ViewType == ViewType.FloorPlan:
        return active
    return None


def configure_plan_view_for_digitizing(plan_view):
    t_view = Transaction(doc, "Prepare Plan View")
    t_view.Start()
    try:
        # Scope Box = None
        try:
            p_scope = plan_view.get_Parameter(BuiltInParameter.VIEWER_VOLUME_OF_INTEREST_CROP)
            if p_scope and (not p_scope.IsReadOnly):
                p_scope.Set(ElementId.InvalidElementId)
        except Exception:
            pass

        # Remove crop/clip limitations where possible.
        try:
            plan_view.CropBoxActive = False
        except Exception:
            pass
        try:
            plan_view.CropBoxVisible = False
        except Exception:
            pass
        try:
            p_ann_crop = plan_view.get_Parameter(BuiltInParameter.VIEWER_ANNOTATION_CROP_ACTIVE)
            if p_ann_crop and (not p_ann_crop.IsReadOnly):
                p_ann_crop.Set(0)
        except Exception:
            pass

        # View range: cut plane at 120 cm and relaxed top/bottom/view-depth.
        try:
            vr = plan_view.GetViewRange()
            vr.SetOffset(PlanViewPlane.CutPlane, cm_to_ft(120.0))
            try:
                vr.SetOffset(PlanViewPlane.TopClipPlane, cm_to_ft(1000.0))
            except Exception:
                pass
            try:
                vr.SetOffset(PlanViewPlane.BottomClipPlane, cm_to_ft(-1000.0))
            except Exception:
                pass
            try:
                vr.SetOffset(PlanViewPlane.ViewDepthPlane, cm_to_ft(-1000.0))
            except Exception:
                pass
            plan_view.SetViewRange(vr)
        except Exception:
            pass

        t_view.Commit()
    except Exception:
        try:
            t_view.RollBack()
        except Exception:
            pass
        raise


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


def build_loop(points):
    loop = CurveLoop()
    for i in range(len(points)):
        loop.Append(Line.CreateBound(points[i], points[(i + 1) % len(points)]))
    return loop


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
        p = element.LookupParameter(name)
        if p and (not p.IsReadOnly):
            p.Set(value)
            return True
    return False


def set_opening_size(instance, width_ft, height_ft, width_bip, height_bip):
    width_ok = set_param(instance, width_bip, width_ft)
    height_ok = set_param(instance, height_bip, height_ft)

    if not width_ok:
        width_ok = set_param_by_names(instance, ["Width", "width", "Rough Width"], width_ft)
    if not height_ok:
        height_ok = set_param_by_names(instance, ["Height", "height", "Rough Height"], height_ft)

    symbol = instance.Symbol if hasattr(instance, "Symbol") else None
    if (not width_ok) and symbol:
        set_param(symbol, width_bip, width_ft) or set_param_by_names(symbol, ["Width", "width", "Rough Width"], width_ft)
    if (not height_ok) and symbol:
        set_param(symbol, height_bip, height_ft) or set_param_by_names(symbol, ["Height", "height", "Rough Height"], height_ft)


def get_param_double(element, bip, names):
    try:
        p = element.get_Parameter(bip)
        if p and p.HasValue:
            return p.AsDouble()
    except Exception:
        pass

    for name in names:
        try:
            p = element.LookupParameter(name)
            if p and p.HasValue:
                return p.AsDouble()
        except Exception:
            pass

    return None


def ensure_sized_symbol(base_symbol, width_ft, height_ft, width_bip, height_bip, name_prefix):
    if base_symbol is None:
        return None

    fam_id = None
    try:
        fam_id = base_symbol.Family.Id
    except Exception:
        fam_id = None

    # Reuse an existing symbol in same family with matching size if available.
    try:
        for sym in FilteredElementCollector(doc).OfClass(FamilySymbol).OfCategoryId(base_symbol.Category.Id):
            if fam_id and hasattr(sym, "Family"):
                try:
                    if sym.Family.Id.IntegerValue != fam_id.IntegerValue:
                        continue
                except Exception:
                    pass

            w = get_param_double(sym, width_bip, ["Width", "width", "Rough Width"])
            h = get_param_double(sym, height_bip, ["Height", "height", "Rough Height"])
            if w is None or h is None:
                continue
            if abs(w - width_ft) <= cm_to_ft(0.5) and abs(h - height_ft) <= cm_to_ft(0.5):
                return sym
    except Exception:
        pass

    # Create a dedicated symbol type with requested size.
    target_name = "{}_{:.0f}x{:.0f}cm".format(name_prefix, ft_to_cm(width_ft), ft_to_cm(height_ft))
    sym = base_symbol
    try:
        for i in range(0, 50):
            nm = target_name if i == 0 else "{}_{}".format(target_name, i)
            try:
                new_id = base_symbol.Duplicate(nm)
                maybe = doc.GetElement(new_id)
                if maybe:
                    sym = maybe
                break
            except Exception:
                continue
    except Exception:
        pass

    try:
        set_param(sym, width_bip, width_ft) or set_param_by_names(sym, ["Width", "width", "Rough Width"], width_ft)
    except Exception:
        pass
    try:
        set_param(sym, height_bip, height_ft) or set_param_by_names(sym, ["Height", "height", "Rough Height"], height_ft)
    except Exception:
        pass

    return sym


def get_symbol(category):
    return first_or_none(FilteredElementCollector(doc).OfCategory(category).OfClass(FamilySymbol))


def get_wall_face_ref(wall, side_type):
    refs = HostObjectUtils.GetSideFaces(wall, side_type)
    if refs and len(refs) > 0:
        return refs[0]
    return None


def _face_distance_to_point(face_ref, point):
    try:
        host = doc.GetElement(face_ref)
        face = host.GetGeometryObjectFromReference(face_ref)
        if face:
            ir = face.Project(point)
            if ir:
                return ir.Distance
    except Exception:
        pass
    return float('inf')


def get_wall_face_ref_by_room(wall, room_center, prefer_inside):
    refs = []
    for side_type in (ShellLayerType.Interior, ShellLayerType.Exterior):
        try:
            side_refs = HostObjectUtils.GetSideFaces(wall, side_type)
            if side_refs:
                for r in side_refs:
                    refs.append(r)
        except Exception:
            pass

    if not refs:
        return None

    # Deduplicate references.
    uniq = []
    seen = set()
    for r in refs:
        try:
            key = r.ConvertToStableRepresentation(doc)
        except Exception:
            key = str(r)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    scored = []
    for r in uniq:
        d = _face_distance_to_point(r, room_center)
        scored.append((d, r))

    scored = [sr for sr in scored if sr[0] < 1.0e20]
    if not scored:
        return uniq[0]

    scored.sort(key=lambda sr: sr[0])
    if prefer_inside:
        return scored[0][1]
    return scored[-1][1]


def get_instance_ref(instance, ref_type):
    try:
        refs = instance.GetReferences(ref_type)
        if refs and len(refs) > 0:
            return refs[0]
    except Exception:
        return None
    return None


def ensure_cm_dimension_type():
    global _CM_DIM_TYPE_ID

    dim_types = [dt for dt in FilteredElementCollector(doc).OfClass(DimensionType)]
    if not dim_types:
        return None

    dim_type = None
    if _CM_DIM_TYPE_ID:
        try:
            dim_type = doc.GetElement(_CM_DIM_TYPE_ID)
        except Exception:
            dim_type = None

    target_name = "__pyrevit_cm_dim"
    if dim_type is None:
        for dt in dim_types:
            try:
                if dt.Name == target_name:
                    dim_type = dt
                    break
            except Exception:
                pass

    if dim_type is None:
        base = dim_types[0]
        dim_type = base
    try:
        new_id = base.Duplicate(target_name)
        maybe = doc.GetElement(new_id)
        if maybe:
            dim_type = maybe
    except Exception:
        pass

    # Try to force cm display on the dimension type (Revit API variants).
    applied = False
    try:
        fmt = FormatOptions(UnitTypeId.Centimeters)
        try:
            fmt.UseDefault = False
        except Exception:
            pass

        try:
            dim_type.SetUnitsFormatOptions(fmt)
            applied = True
        except Exception:
            pass

        if not applied:
            try:
                dim_type.SetFormatOptions(fmt)
                applied = True
            except Exception:
                pass

        if not applied:
            try:
                units = doc.GetUnits()
                units.SetFormatOptions(SpecTypeId.Length, fmt)
                doc.SetUnits(units)
                applied = True
            except Exception:
                pass
    except Exception:
        pass

    # Dimension text size = 1.8 mm (0.18 cm)
    try:
        set_param(dim_type, BuiltInParameter.TEXT_SIZE, cm_to_ft(0.18))
    except Exception:
        pass

    _CM_DIM_TYPE_ID = dim_type.Id
    return _CM_DIM_TYPE_ID


def create_dimension(view, p1, p2, ref_a, ref_b):
    if not (view and ref_a and ref_b):
        return None

    refs = ReferenceArray()
    refs.Append(ref_a)
    refs.Append(ref_b)

    try:
        dim = doc.Create.NewDimension(view, Line.CreateBound(p1, p2), refs)
    except Exception:
        return None

    cm_dim_type_id = ensure_cm_dimension_type()
    if dim and cm_dim_type_id:
        try:
            dim.ChangeTypeId(cm_dim_type_id)
        except Exception:
            pass

    return dim


def create_dimension_chain(view, p1, p2, refs_list):
    if not (view and refs_list):
        return None

    refs = ReferenceArray()
    count = 0
    for r in refs_list:
        if r is None:
            continue
        refs.Append(r)
        count += 1

    if count < 2:
        return None

    try:
        dim = doc.Create.NewDimension(view, Line.CreateBound(p1, p2), refs)
    except Exception:
        return None

    cm_dim_type_id = ensure_cm_dimension_type()
    if dim and cm_dim_type_id:
        try:
            dim.ChangeTypeId(cm_dim_type_id)
        except Exception:
            pass

    return dim


def pick_point(prompt):
    return uidoc.Selection.PickPoint(prompt)


def pick_image_path():
    dialog = OpenFileDialog()
    dialog.Filter = "Image Files|*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"
    dialog.Multiselect = False
    dialog.Title = "Select sketch image to overlay (optional)"
    if dialog.ShowDialog() == DialogResult.OK:
        return dialog.FileName
    return None


def set_form_center(form):
    try:
        form.StartPosition = WinForms.FormStartPosition.CenterScreen
        return
    except Exception:
        pass
    try:
        form.StartPosition = WinForms.StartPosition.CenterScreen
    except Exception:
        pass


def choose_input_mode():
    form = Form()
    form.Text = "Choose Input Source"
    form.Width = 460
    form.Height = 170
    form.FormBorderStyle = FormBorderStyle.FixedDialog
    set_form_center(form)
    form.MinimizeBox = False
    form.MaximizeBox = False

    lbl = Label()
    lbl.Left = 12
    lbl.Top = 12
    lbl.Width = 420
    lbl.Height = 24
    lbl.Text = "Select source for room generation:"
    form.Controls.Add(lbl)

    btn_cad = Button()
    btn_cad.Text = "Imported CAD DWG"
    btn_cad.Left = 28
    btn_cad.Top = 56
    btn_cad.Width = 180

    btn_img = Button()
    btn_img.Text = "Scanned Image"
    btn_img.Left = 236
    btn_img.Top = 56
    btn_img.Width = 180

    choice = {"mode": None}

    def on_cad(sender, args):
        choice["mode"] = "cad"
        form.DialogResult = DialogResult.OK
        form.Close()

    def on_img(sender, args):
        choice["mode"] = "image"
        form.DialogResult = DialogResult.OK
        form.Close()

    btn_cad.Click += on_cad
    btn_img.Click += on_img

    form.Controls.Add(btn_cad)
    form.Controls.Add(btn_img)

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
        lbl.Text = "Use existing imported CAD in this view, or import a new DWG/DXF file."
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


def import_cad_file_to_view(view, cad_path):
    t_imp = Transaction(doc, "Import CAD (DWG/DXF)")
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
            return None, "CAD imported but ImportInstance was not found."
        return inst, None
    except Exception as ex:
        try:
            if t_imp.HasStarted():
                t_imp.RollBack()
        except Exception:
            pass
        return None, str(ex)


def get_view_bounds(view):
    min_pt = None
    max_pt = None
    for ui_view in uidoc.GetOpenUIViews():
        if ui_view.ViewId.IntegerValue == view.Id.IntegerValue:
            corners = ui_view.GetZoomCorners()
            if corners and len(corners) == 2:
                p1 = corners[0]
                p2 = corners[1]
                min_pt = XYZ(min(p1.X, p2.X), min(p1.Y, p2.Y), min(p1.Z, p2.Z))
                max_pt = XYZ(max(p1.X, p2.X), max(p1.Y, p2.Y), max(p1.Z, p2.Z))
                break

    if min_pt is None or max_pt is None:
        min_pt = XYZ(-cm_to_ft(500.0), -cm_to_ft(500.0), 0.0)
        max_pt = XYZ(cm_to_ft(500.0), cm_to_ft(500.0), 0.0)

    center = XYZ((min_pt.X + max_pt.X) * 0.5, (min_pt.Y + max_pt.Y) * 0.5, (min_pt.Z + max_pt.Z) * 0.5)
    return min_pt, max_pt, center


def get_image_aspect(image_path):
    if not image_path:
        return 1.0
    try:
        img = DrawingImage.FromFile(image_path)
        try:
            w = float(img.Width)
            h = float(img.Height)
        finally:
            img.Dispose()
        if w > 0 and h > 0:
            return w / h
    except Exception:
        pass
    return 1.0


def create_image_overlay(view, image_path, center_pt):
    if not image_path:
        return None, None, None

    image_type = None
    image_inst = None
    err = None

    try:
        img_opts = None
        ctor_errors = []
        for args in (
            (image_path, False, ImageTypeSource.Import),
            (image_path, False),
            (image_path, True, ImageTypeSource.Import),
            (image_path, True),
        ):
            try:
                img_opts = ImageTypeOptions(*args)
                break
            except Exception as ex_ctor:
                ctor_errors.append("args{} -> {}".format(len(args), ex_ctor))

        if img_opts is None:
            raise Exception("Unable to create ImageTypeOptions. " + " | ".join(ctor_errors))

        image_type = ImageType.Create(doc, img_opts)
        place = ImagePlacementOptions(center_pt, BoxPlacement.Center)

        try:
            image_inst = ImageInstance.Create(doc, view, image_type.Id, place)
        except Exception:
            image_inst = ImageInstance.Create(doc, view.Id, image_type.Id, place)
    except Exception as ex:
        image_type = None
        image_inst = None
        err = str(ex)

    return image_type, image_inst, err


def create_temp_text_type():
    base_type_id = doc.GetDefaultElementTypeId(ElementTypeGroup.TextNoteType)
    base_type = doc.GetElement(base_type_id)
    if not isinstance(base_type, TextNoteType):
        return base_type_id, None

    base_name = "__pyrevit_image_pick_note"
    existing_names = set()
    for tt in FilteredElementCollector(doc).OfClass(TextNoteType):
        try:
            existing_names.add(tt.Name)
        except Exception:
            pass

    name = base_name
    idx = 1
    while name in existing_names:
        name = "{}_{}".format(base_name, idx)
        idx += 1

    temp_type = base_type.Duplicate(name)
    # Keep guidance readable at typical zoom levels.
    set_param(temp_type, BuiltInParameter.TEXT_SIZE, cm_to_ft(1.20))
    return temp_type.Id, temp_type


def create_instruction_note(view, origin, text_type_id, width_ft):
    text = (
        "Create Room From Image (cm)\n"
        "1) Pick inner corners: NW, NE, SE, SW\n"
        "2) Pick calibration: start + end of 100cm line\n"
        "3) Pick door: LEFT edge, RIGHT edge\n"
        "4) Pick window: LEFT edge, RIGHT edge\n"
        "Tip: pick along inner wall faces for best accuracy\n"
        "ESC = cancel"
    )
    try:
        return TextNote.Create(doc, view.Id, origin, width_ft, text, text_type_id)
    except Exception:
        return TextNote.Create(doc, view.Id, origin, text, text_type_id)


def fit_image_overlay(image_inst, target_layout):
    if image_inst is None:
        return

    try:
        image_inst.DrawLayer = DrawLayer.Background
    except Exception:
        pass

    try:
        image_inst.LockProportions = True
    except Exception:
        pass

    target_center = XYZ(target_layout["image_center"][0], target_layout["image_center"][1], 0.0)
    target_max_width = target_layout["image_width"]
    target_max_height = target_layout["image_height"]

    current_width = None
    current_height = None
    try:
        current_width = image_inst.Width
    except Exception:
        pass
    try:
        current_height = image_inst.Height
    except Exception:
        pass

    if (not current_width) or (not current_height):
        try:
            bbox = image_inst.get_BoundingBox(doc.ActiveView)
            if bbox:
                current_width = abs(bbox.Max.X - bbox.Min.X)
                current_height = abs(bbox.Max.Y - bbox.Min.Y)
        except Exception:
            pass

    scale_ratio = None
    final_width = target_max_width
    if current_width and current_height and current_width > 1e-9 and current_height > 1e-9:
        scale_ratio = min(target_max_width / current_width, target_max_height / current_height)
        final_width = current_width * scale_ratio

    scaled = False
    try:
        image_inst.Width = final_width
        scaled = True
    except Exception:
        pass

    if (not scaled) and scale_ratio:
        try:
            if hasattr(image_inst, "WidthScale"):
                image_inst.WidthScale = image_inst.WidthScale * scale_ratio
                if hasattr(image_inst, "HeightScale"):
                    image_inst.HeightScale = image_inst.HeightScale * scale_ratio
        except Exception:
            pass

    try:
        image_inst.SetLocation(target_center, BoxPlacement.Center)
    except Exception:
        pass


def cleanup_temp_elements(temp_ids):
    if not temp_ids:
        return
    t_clean = Transaction(doc, "Cleanup overlay")
    t_clean.Start()
    for element_id in temp_ids:
        try:
            if element_id and doc.GetElement(element_id):
                doc.Delete(element_id)
        except Exception:
            pass
    t_clean.Commit()


def _parse_float_text(value_text, fallback):
    try:
        return float(str(value_text).replace(",", ".").strip())
    except Exception:
        return float(fallback)


def confirm_measurements_dialog(meas_cm):
    fields = [
        ("room_width_cm", "Room Width (cm)"),
        ("room_height_cm", "Room Height (cm)"),
        ("wall_thickness_cm", "Wall Thickness (cm)"),
        ("door_width_cm", "Door Width (cm)"),
        ("door_height_cm", "Door Height (cm)"),
        ("door_left_offset_cm", "Door Left Offset (cm)"),
        ("window_width_cm", "Window Width (cm)"),
        ("window_height_cm", "Window Height (cm)"),
        ("window_right_offset_cm", "Window Right Offset (cm)"),
        ("window_sill_cm", "Window Sill (cm)"),
    ]

    form = Form()
    form.Text = "Confirm Measurements (cm)"
    form.Width = 460
    form.Height = 420
    form.FormBorderStyle = FormBorderStyle.FixedDialog
    set_form_center(form)
    form.MinimizeBox = False
    form.MaximizeBox = False

    labels = {}
    y = 16
    for key, title in fields:
        lb = Label()
        lb.Left = 12
        lb.Top = y + 4
        lb.Width = 210
        lb.Text = title
        form.Controls.Add(lb)

        tb = TextBox()
        tb.Left = 230
        tb.Top = y
        tb.Width = 180
        tb.Text = str(round(float(meas_cm.get(key, DEFAULTS_CM.get(key, 0.0))), 3))
        form.Controls.Add(tb)
        labels[key] = tb
        y += 30

    btn_ok = Button()
    btn_ok.Text = "Generate"
    btn_ok.Left = 230
    btn_ok.Top = y + 10
    btn_ok.Width = 85
    btn_ok.DialogResult = DialogResult.OK

    btn_cancel = Button()
    btn_cancel.Text = "Cancel"
    btn_cancel.Left = 325
    btn_cancel.Top = y + 10
    btn_cancel.Width = 85
    btn_cancel.DialogResult = DialogResult.Cancel

    form.Controls.Add(btn_ok)
    form.Controls.Add(btn_cancel)
    form.AcceptButton = btn_ok
    form.CancelButton = btn_cancel

    if form.ShowDialog() != DialogResult.OK:
        return None

    out = dict(meas_cm)
    for key, _ in fields:
        out[key] = _parse_float_text(labels[key].Text, meas_cm.get(key, DEFAULTS_CM.get(key, 0.0)))

    return merge_measurements_cm(out, {}, prefer_ocr=False)


def compute_pick_measurements_cm(rect, scale_cm_per_unit, door_l_rect, door_r_rect, win_l_rect, win_r_rect):
    width_cm = rect["rect_width_units"] * scale_cm_per_unit
    height_cm = rect["rect_height_units"] * scale_cm_per_unit

    dl = min(door_l_rect[0], door_r_rect[0]) * scale_cm_per_unit
    dr = max(door_l_rect[0], door_r_rect[0]) * scale_cm_per_unit

    wl = min(win_l_rect[0], win_r_rect[0]) * scale_cm_per_unit
    wr = max(win_l_rect[0], win_r_rect[0]) * scale_cm_per_unit

    out = dict(DEFAULTS_CM)
    out["room_width_cm"] = width_cm
    out["room_height_cm"] = height_cm
    out["door_left_offset_cm"] = dl
    out["door_width_cm"] = max(1.0, dr - dl)
    out["window_width_cm"] = max(1.0, wr - wl)
    out["window_right_offset_cm"] = max(0.0, width_cm - wr)
    return out


def merge_with_ocr_guardrails(pick_cm, ocr_parsed):
    merged = merge_measurements_cm(pick_cm, {}, prefer_ocr=False)

    for key, value in (ocr_parsed or {}).items():
        if key not in merged:
            continue
        try:
            v = float(value)
        except Exception:
            continue

        # Allow OCR to directly fill these.
        if key in ("wall_thickness_cm", "door_height_cm", "window_height_cm", "window_sill_cm"):
            merged[key] = v
            continue

        # For geometric lengths, only accept OCR if reasonably close to picked geometry.
        base = merged.get(key, 0.0)
        if base <= 1e-6:
            merged[key] = v
            continue

        rel = abs(v - base) / base
        if rel <= 0.35:
            merged[key] = v

    return merge_measurements_cm(merged, {}, prefer_ocr=False)


def run_cad_mode(plan_view, level, snapshot, target_instance_id=None):
    ext_dir = os.path.dirname(__file__)
    layer_map = load_layer_map(os.path.join(ext_dir, "cad_layer_map.json"))
    cad_cfg = load_cad_config(os.path.join(ext_dir, "cad_config.json"))

    raw = extract_cad_from_view(doc, plan_view, cad_cfg, target_instance_id=target_instance_id)
    classified = classify_entities(raw.get("lines", []), raw.get("arcs", []), layer_map)
    topology = derive_room_from_cad(classified, cad_cfg)
    meas_cm = topology.get("measurements_cm", {})

    save_cad_snapshot_stages(snapshot, raw, classified, topology, meas_cm)

    merged = merge_measurements_cm(meas_cm, {}, prefer_ocr=False)
    confirmed = confirm_measurements_dialog(merged)
    if confirmed is None:
        raise OperationCanceledException()

    snapshot.save_json("05_measurements_confirmed.json", confirmed)

    inner_sw_cm = topology.get("inner_sw_cm", [0.0, 0.0])
    origin_ft = XYZ(cm_to_ft(inner_sw_cm[0]), cm_to_ft(inner_sw_cm[1]), 0.0)
    build_out = build_model_and_dimensions(confirmed, plan_view, level, snapshot, origin_override_ft=origin_ft)

    return {
        "model_created": True,
        "dimensions_created": bool(build_out.get("dimensions", {}).get("ok")),
        "mode": "imported_dwg",
    }


def build_model_and_dimensions(meas_cm, plan_view, level, snapshot, origin_override_ft=None):
    room = build_room_cm(meas_cm)

    wall_type = get_wall_type_nearest(cm_to_ft(meas_cm["wall_thickness_cm"]))
    floor_type = get_floor_type()
    if wall_type is None or floor_type is None:
        raise Exception("Missing WallType/FloorType")

    if origin_override_ft is None:
        view_min, view_max, _ = get_view_bounds(plan_view)
        origin_x = view_min.X + (view_max.X - view_min.X) * 0.70
        origin_y = view_min.Y + (view_max.Y - view_min.Y) * 0.30
    else:
        origin_x = origin_override_ft.X
        origin_y = origin_override_ft.Y

    inner_sw = XYZ(origin_x, origin_y, 0)
    inner_se = XYZ(origin_x + cm_to_ft(meas_cm["room_width_cm"]), origin_y, 0)
    inner_ne = XYZ(origin_x + cm_to_ft(meas_cm["room_width_cm"]), origin_y + cm_to_ft(meas_cm["room_height_cm"]), 0)
    inner_nw = XYZ(origin_x, origin_y + cm_to_ft(meas_cm["room_height_cm"]), 0)

    half = wall_type.Width * 0.5
    cl_sw = XYZ(inner_sw.X - half, inner_sw.Y - half, 0)
    cl_se = XYZ(inner_se.X + half, inner_se.Y - half, 0)
    cl_ne = XYZ(inner_ne.X + half, inner_ne.Y + half, 0)
    cl_nw = XYZ(inner_nw.X - half, inner_nw.Y + half, 0)

    floor_height_ft = cm_to_ft(300.0)

    result = {
        "geometry": {},
        "dimensions": {"ok": False, "error": None},
    }

    walls = []
    door = None
    window = None

    t_geo = Transaction(doc, "Create Room Geometry (cm)")
    t_geo.Start()
    try:
        walls.append(Wall.Create(doc, Line.CreateBound(cl_sw, cl_se), wall_type.Id, level.Id, floor_height_ft, 0.0, False, False))  # south
        walls.append(Wall.Create(doc, Line.CreateBound(cl_se, cl_ne), wall_type.Id, level.Id, floor_height_ft, 0.0, False, False))  # east
        walls.append(Wall.Create(doc, Line.CreateBound(cl_ne, cl_nw), wall_type.Id, level.Id, floor_height_ft, 0.0, False, False))  # north
        walls.append(Wall.Create(doc, Line.CreateBound(cl_nw, cl_sw), wall_type.Id, level.Id, floor_height_ft, 0.0, False, False))  # west

        Floor.Create(doc, [build_loop([cl_sw, cl_se, cl_ne, cl_nw])], floor_type.Id, level.Id)
        roof = Floor.Create(doc, [build_loop([cl_sw, cl_se, cl_ne, cl_nw])], floor_type.Id, level.Id)
        set_param(roof, BuiltInParameter.FLOOR_HEIGHTABOVELEVEL_PARAM, floor_height_ft)

        door_type = get_symbol(BuiltInCategory.OST_Doors)
        window_type = get_symbol(BuiltInCategory.OST_Windows)

        # If no exact 100cm window width exists, create one from current family symbol.
        desired_window_width_ft = cm_to_ft(100.0)
        desired_window_height_ft = cm_to_ft(meas_cm["window_height_cm"])
        if window_type:
            window_type = ensure_sized_symbol(
                window_type,
                desired_window_width_ft,
                desired_window_height_ft,
                BuiltInParameter.WINDOW_WIDTH,
                BuiltInParameter.WINDOW_HEIGHT,
                "__pyrevit_window",
            )

        if door_type and (not door_type.IsActive):
            door_type.Activate()
        if window_type and (not window_type.IsActive):
            window_type.Activate()
        doc.Regenerate()

        if door_type:
            door_center_x = inner_sw.X + cm_to_ft(meas_cm["door_left_offset_cm"] + (meas_cm["door_width_cm"] * 0.5))
            door_point = XYZ(door_center_x, inner_sw.Y - half, 0)
            door = doc.Create.NewFamilyInstance(door_point, door_type, walls[0], level, Structure.StructuralType.NonStructural)
            set_opening_size(door, cm_to_ft(meas_cm["door_width_cm"]), cm_to_ft(meas_cm["door_height_cm"]), BuiltInParameter.DOOR_WIDTH, BuiltInParameter.DOOR_HEIGHT)

        if window_type:
            window_width_cm = 100.0
            window_center_x = inner_sw.X + cm_to_ft(
                meas_cm["room_width_cm"] - meas_cm["window_right_offset_cm"] - (window_width_cm * 0.5)
            )
            window_point = XYZ(window_center_x, inner_nw.Y + half, cm_to_ft(meas_cm["window_sill_cm"]))
            window = doc.Create.NewFamilyInstance(window_point, window_type, walls[2], level, Structure.StructuralType.NonStructural)
            set_opening_size(window, cm_to_ft(window_width_cm), cm_to_ft(meas_cm["window_height_cm"]), BuiltInParameter.WINDOW_WIDTH, BuiltInParameter.WINDOW_HEIGHT)
            set_param(window, BuiltInParameter.INSTANCE_SILL_HEIGHT_PARAM, cm_to_ft(meas_cm["window_sill_cm"]))
            meas_cm["window_width_cm"] = window_width_cm

        t_geo.Commit()
        result["geometry"] = {
            "wall_ids": [w.Id.IntegerValue for w in walls],
            "door_id": door.Id.IntegerValue if door else None,
            "window_id": window.Id.IntegerValue if window else None,
        }
    except Exception:
        t_geo.RollBack()
        raise

    t_dim = Transaction(doc, "Create Room Dimensions (cm)")
    t_dim.Start()
    try:
        doc.Regenerate()

        room_center = XYZ((inner_sw.X + inner_ne.X) * 0.5, (inner_sw.Y + inner_ne.Y) * 0.5, 0)

        west_int_ref = get_wall_face_ref_by_room(walls[3], room_center, True)
        east_int_ref = get_wall_face_ref_by_room(walls[1], room_center, True)
        south_int_ref = get_wall_face_ref_by_room(walls[0], room_center, True)
        north_int_ref = get_wall_face_ref_by_room(walls[2], room_center, True)

        west_ext_ref = get_wall_face_ref_by_room(walls[3], room_center, False)
        east_ext_ref = get_wall_face_ref_by_room(walls[1], room_center, False)
        south_ext_ref = get_wall_face_ref_by_room(walls[0], room_center, False)
        north_ext_ref = get_wall_face_ref_by_room(walls[2], room_center, False)

        ext_off = cm_to_ft(60.0)

        # Outside: all 4 wall dimensions (north/south/east/west)
        create_dimension(plan_view, XYZ(inner_sw.X, inner_sw.Y - ext_off, 0), XYZ(inner_se.X, inner_se.Y - ext_off, 0), west_ext_ref, east_ext_ref)  # south outside
        create_dimension(plan_view, XYZ(inner_nw.X, inner_nw.Y + ext_off, 0), XYZ(inner_ne.X, inner_ne.Y + ext_off, 0), west_ext_ref, east_ext_ref)  # north outside
        create_dimension(plan_view, XYZ(inner_sw.X - ext_off, inner_sw.Y, 0), XYZ(inner_nw.X - ext_off, inner_nw.Y, 0), south_ext_ref, north_ext_ref)  # west outside
        create_dimension(plan_view, XYZ(inner_se.X + ext_off, inner_se.Y, 0), XYZ(inner_ne.X + ext_off, inner_ne.Y, 0), south_ext_ref, north_ext_ref)  # east outside

        # Inside: place each chain near what it measures while staying inside bounds.
        room_h = inner_nw.Y - inner_sw.Y
        near_face = cm_to_ft(10.0)
        min_gap = cm_to_ft(8.0)
        inside_min_y = inner_sw.Y + cm_to_ft(3.0)
        inside_max_y = inner_nw.Y - cm_to_ft(3.0)

        y_door = clamp_value(inner_sw.Y + near_face, inside_min_y, inside_max_y)
        y_win = clamp_value(inner_nw.Y - near_face, inside_min_y, inside_max_y)
        y_center = clamp_value(inner_sw.Y + (room_h * 0.50), inside_min_y, inside_max_y)

        # Keep guaranteed ordering inside room in short rooms.
        if y_center <= (y_door + min_gap):
            y_center = clamp_value(y_door + min_gap, inside_min_y, inside_max_y)
        if y_win <= (y_center + min_gap):
            y_win = clamp_value(y_center + min_gap, inside_min_y, inside_max_y)

        # 1) overall inner width at mid room
        create_dimension(
            plan_view,
            XYZ(inner_sw.X, y_center, 0),
            XYZ(inner_se.X, y_center, 0),
            west_int_ref,
            east_int_ref,
        )

        # 2) door chain near south inner wall
        if door:
            door_left_ref = get_instance_ref(door, FamilyInstanceReferenceType.Left)
            door_right_ref = get_instance_ref(door, FamilyInstanceReferenceType.Right)
            create_dimension_chain(
                plan_view,
                XYZ(inner_sw.X, y_door, 0),
                XYZ(inner_se.X, y_door, 0),
                [west_int_ref, door_left_ref, door_right_ref, east_int_ref],
            )

        # 3) window chain near north inner wall
        if window:
            window_left_ref = get_instance_ref(window, FamilyInstanceReferenceType.Left)
            window_right_ref = get_instance_ref(window, FamilyInstanceReferenceType.Right)
            create_dimension_chain(
                plan_view,
                XYZ(inner_sw.X, y_win, 0),
                XYZ(inner_se.X, y_win, 0),
                [west_int_ref, window_left_ref, window_right_ref, east_int_ref],
            )

        t_dim.Commit()
        result["dimensions"]["ok"] = True
    except Exception as ex:
        result["dimensions"]["ok"] = False
        result["dimensions"]["error"] = str(ex)
        t_dim.RollBack()

    snapshot.save_json("08_geometry_summary.json", result)
    return result


# ---------------------- Main flow ----------------------

def run_command():
    
    plan_view = get_plan_view()
    if plan_view is None:
        TaskDialog.Show("Create Room From Image", "Run this command in a Floor Plan view.")
        sys.exit()
    
    level = get_level_from_view(plan_view)
    if level is None:
        TaskDialog.Show("Create Room From Image", "No valid Level found.")
        sys.exit()

    try:
        configure_plan_view_for_digitizing(plan_view)
    except Exception:
        pass
    
    snapshot = SnapshotRun()
    snapshot.log("Run started")

    mode = choose_input_mode()
    if mode is None:
        snapshot.log("Canceled at source selection")
        TaskDialog.Show("Create Room", "Canceled.")
        return

    if mode == "cad":
        try:
            cad_instances = get_imported_cad_instances(doc, plan_view)
        except Exception:
            cad_instances = []

        cad_source_mode = choose_cad_source_mode(len(cad_instances) > 0)
        if cad_source_mode is None:
            snapshot.log("Canceled at CAD source selection")
            TaskDialog.Show("Create Room From CAD", "Canceled.")
            return

        cad_path = None
        selected_cad_instance = None

        if cad_source_mode == "import":
            cad_path = pick_cad_path()
            if not cad_path:
                snapshot.log("Canceled at CAD file selection")
                TaskDialog.Show("Create Room From CAD", "Canceled.")
                return

            imported_instance, import_err = import_cad_file_to_view(plan_view, cad_path)
            if imported_instance is None:
                TaskDialog.Show("Create Room From CAD", "Failed to import CAD file.\n{}".format(import_err or "Unknown error"))
                return

            selected_cad_instance = imported_instance
            try:
                doc.Regenerate()
            except Exception:
                pass
        else:
            if len(cad_instances) == 0:
                TaskDialog.Show("Create Room From CAD", "No imported CAD found in active plan view.")
                return
            if len(cad_instances) > 1:
                TaskDialog.Show(
                    "Create Room From CAD",
                    "Multiple CAD imports detected. Either keep one visible import, or choose 'Import DWG/DXF File'.",
                )
                return
            selected_cad_instance = cad_instances[0]

        snapshot.save_json("00_input_meta.json", {
            "units": "cm",
            "source_mode": "imported_dwg_file" if cad_source_mode == "import" else "existing_imported_dwg",
            "known_calibration_cm": KNOWN_CALIBRATION_CM,
            "view_name": plan_view.Name,
            "cad_path": cad_path,
            "cad_instance_id": selected_cad_instance.Id.IntegerValue if selected_cad_instance else None,
        })
        try:
            out = run_cad_mode(
                plan_view,
                level,
                snapshot,
                target_instance_id=(selected_cad_instance.Id.IntegerValue if selected_cad_instance else None),
            )
            if out.get("dimensions_created"):
                TaskDialog.Show("Create Room From CAD", "Model + dimensions generated from imported CAD.\nSnapshots saved to:\n{}".format(snapshot.run_dir))
            else:
                TaskDialog.Show("Create Room From CAD", "Model generated from imported CAD, but some dimensions failed.\nSnapshots saved to:\n{}".format(snapshot.run_dir))
            return
        except OperationCanceledException:
            snapshot.log("CAD mode canceled by user")
            TaskDialog.Show("Create Room From CAD", "Canceled.")
            return
        except Exception as ex:
            snapshot.save_error("cad_pipeline", ex)
            TaskDialog.Show("Create Room From CAD", "CAD recognition failed.\n{}".format(ex))
            return

    image_path = pick_image_path()
    if image_path:
        base = os.path.basename(image_path)
        ext = os.path.splitext(base)[1] or ".png"
        snapshot.copy_file(image_path, "01_raw_image{}".format(ext))
    
    snapshot.save_json("00_input_meta.json", {
        "units": "cm",
        "image_path": image_path,
        "known_calibration_cm": KNOWN_CALIBRATION_CM,
        "view_name": plan_view.Name,
    })
    
    # Temporary overlay elements to cleanup later.
    temp_ids = []
    model_created = False
    dimensions_created = False
    
    overlay_tx = Transaction(doc, "Overlay image + instructions")
    try:
        overlay_tx.Start()
        view_min, view_max, center = get_view_bounds(plan_view)
        image_aspect = get_image_aspect(image_path)
        overlay_layout = compute_overlay_and_note_layout(
            (view_min.X, view_min.Y),
            (view_max.X, view_max.Y),
            image_aspect,
            left_fraction=0.60,
        )
        snapshot.save_json("07_overlay_layout.json", overlay_layout)
    
        image_type, image_inst, image_err = create_image_overlay(plan_view, image_path, center)
        if image_inst:
            fit_image_overlay(image_inst, overlay_layout)
            temp_ids.append(image_inst.Id)
        if image_type:
            temp_ids.append(image_type.Id)
    
        text_type_id, text_type = create_temp_text_type()
        note_origin = XYZ(overlay_layout["note_origin"][0], overlay_layout["note_origin"][1], center.Z)
        note = create_instruction_note(plan_view, note_origin, text_type_id, overlay_layout["note_width"])
        if note:
            temp_ids.append(note.Id)
        if text_type:
            temp_ids.append(text_type.Id)
    
        overlay_tx.Commit()
    
        if image_path and (not image_inst):
            TaskDialog.Show("Create Room From Image", "Image overlay failed. Continue with picks.\nReason: {}".format(image_err if image_err else "Unknown"))
    
        try:
            uidoc.RefreshActiveView()
        except Exception:
            pass
    except Exception as ex_overlay:
        try:
            overlay_tx.RollBack()
        except Exception:
            pass
        snapshot.save_error("overlay", ex_overlay)
    
    try:
        # Four-corner rectification (inner room corners).
        nw_pick = pick_point("Pick INNER NW corner")
        ne_pick = pick_point("Pick INNER NE corner")
        se_pick = pick_point("Pick INNER SE corner")
        sw_pick = pick_point("Pick INNER SW corner")
    
        nw_cm = xy_ft_to_cm_tuple(nw_pick)
        ne_cm = xy_ft_to_cm_tuple(ne_pick)
        se_cm = xy_ft_to_cm_tuple(se_pick)
        sw_cm = xy_ft_to_cm_tuple(sw_pick)
    
        rect = build_rectification(nw_cm, ne_cm, se_cm, sw_cm)
    
        calib_a = pick_point("Pick START of known 100cm segment")
        calib_b = pick_point("Pick END of same 100cm segment")
        calib_a_r = apply_homography(rect["homography"], xy_ft_to_cm_tuple(calib_a))
        calib_b_r = apply_homography(rect["homography"], xy_ft_to_cm_tuple(calib_b))
        # Smooth calibration by using dominant-axis length in the rectified frame.
        dx = abs(calib_a_r[0] - calib_b_r[0])
        dy = abs(calib_a_r[1] - calib_b_r[1])
        calib_len_units = max(dx, dy)
        if calib_len_units < 1e-9:
            calib_len_units = distance(calib_a_r, calib_b_r)
        if calib_len_units < 1e-9:
            raise RectificationError("Calibration picks are too close.")
    
        scale_cm_per_unit = KNOWN_CALIBRATION_CM / calib_len_units
    
        door_left_pick = pick_point("Pick DOOR left edge on south wall")
        door_right_pick = pick_point("Pick DOOR right edge on south wall")
        window_left_pick = pick_point("Pick WINDOW left edge on north wall")
        window_right_pick = pick_point("Pick WINDOW right edge on north wall")
    
        door_left_r = apply_homography(rect["homography"], xy_ft_to_cm_tuple(door_left_pick))
        door_right_r = apply_homography(rect["homography"], xy_ft_to_cm_tuple(door_right_pick))
        window_left_r = apply_homography(rect["homography"], xy_ft_to_cm_tuple(window_left_pick))
        window_right_r = apply_homography(rect["homography"], xy_ft_to_cm_tuple(window_right_pick))
    
        pick_meas = compute_pick_measurements_cm(rect, scale_cm_per_unit, door_left_r, door_right_r, window_left_r, window_right_r)
    
        snapshot.save_json("02_rectified_transform.json", {
            "homography": rect["homography"],
            "rect_width_units": rect["rect_width_units"],
            "rect_height_units": rect["rect_height_units"],
            "scale_cm_per_unit": scale_cm_per_unit,
        })
    
        snapshot.save_json("06_pick_points.json", {
            "corners_src_ft": {
                "nw": [nw_pick.X, nw_pick.Y],
                "ne": [ne_pick.X, ne_pick.Y],
                "se": [se_pick.X, se_pick.Y],
                "sw": [sw_pick.X, sw_pick.Y],
            },
            "calibration_src_ft": {
                "a": [calib_a.X, calib_a.Y],
                "b": [calib_b.X, calib_b.Y],
            },
            "door_src_ft": {
                "left": [door_left_pick.X, door_left_pick.Y],
                "right": [door_right_pick.X, door_right_pick.Y],
            },
            "window_src_ft": {
                "left": [window_left_pick.X, window_left_pick.Y],
                "right": [window_right_pick.X, window_right_pick.Y],
            },
        })
    
        ocr_payload = {"engine": "easyocr", "available": False, "tokens": [], "errors": ["No image selected"]}
        if image_path:
            # Keep OCR bounded to avoid long UI stalls.
            ocr_payload = run_easyocr(image_path, timeout_sec=20)
        snapshot.save_json("03_ocr_boxes.json", ocr_payload)
    
        ocr_parsed = parse_ocr_tokens(ocr_payload.get("tokens", []))
        snapshot.save_json("04_measurements_raw.json", {
            "pick_measurements_cm": pick_meas,
            "ocr_parsed": ocr_parsed,
        })
    
        merged = merge_with_ocr_guardrails(pick_meas, ocr_parsed.get("parsed", {}))
        confirmed = confirm_measurements_dialog(merged)
        if confirmed is None:
            raise OperationCanceledException()
    
        snapshot.save_json("05_measurements_confirmed.json", confirmed)
    
        build_out = build_model_and_dimensions(confirmed, plan_view, level, snapshot, origin_override_ft=sw_pick)
        model_created = True
        dimensions_created = bool(build_out.get("dimensions", {}).get("ok"))
    
    except OperationCanceledException:
        snapshot.log("Canceled by user")
        TaskDialog.Show("Create Room From Image", "Canceled.")
    except RectificationError as ex_rect:
        snapshot.save_error("rectification", ex_rect)
        TaskDialog.Show("Create Room From Image", "Rectification failed.\n{}".format(ex_rect))
    except Exception as ex:
        snapshot.save_error("runtime", ex)
        TaskDialog.Show("Create Room From Image", "Failed to create model.\n{}".format(ex))
    finally:
        cleanup_temp_elements(temp_ids)
        snapshot.log("Overlay cleanup completed")
    
    if model_created:
        if dimensions_created:
            TaskDialog.Show("Create Room From Image", "Model + dimensions generated (cm workflow).\nSnapshots saved to:\n{}".format(snapshot.run_dir))
        else:
            TaskDialog.Show("Create Room From Image", "Model generated, but some dimensions failed.\nSnapshots saved to:\n{}".format(snapshot.run_dir))

# Avoid side effects when pyRevit inspects/imports this module at startup.
if __name__ == "__main__":
    run_command()
