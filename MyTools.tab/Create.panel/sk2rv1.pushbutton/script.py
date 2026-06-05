#! python3
# -*- coding: utf-8 -*-
__title__ = "sk2rv1"
__doc__ = "Sketch to Revit V1. Recognize thick orthogonal walls from a sketch image and place doors from swing-backed gaps."

import math
import os
import sys
import json
import re
import struct
import clr

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

clr.AddReference("System.Windows.Forms")
clr.AddReference("System.Drawing")
clr.AddReference("System")
import System.Windows.Forms as WinForms
from System.Diagnostics import Process, ProcessStartInfo

from System.Windows.Forms import (
    Button,
    DialogResult,
    Form,
    FormBorderStyle,
    Label,
    OpenFileDialog,
    TextBox,
)
from System.Drawing import Image as DrawingImage

from Autodesk.Revit.DB import (
    BoxPlacement,
    BuiltInCategory,
    BuiltInParameter,
    DrawLayer,
    ElementTypeGroup,
    FamilySymbol,
    FilteredElementCollector,
    ImageInstance,
    ImagePlacementOptions,
    ImageType,
    ImageTypeOptions,
    ImageTypeSource,
    Level,
    Line,
    TextNote,
    TextNoteType,
    Transaction,
    ViewPlan,
    ViewType,
    Wall,
    WallKind,
    WallType,
    XYZ,
    Structure,
)
from Autodesk.Revit.Exceptions import OperationCanceledException
from Autodesk.Revit.UI import TaskDialog


uidoc = __revit__.ActiveUIDocument
doc = uidoc.Document

CM_PER_FT = 30.48
DEFAULT_DOOR_WIDTH_CM = 90.0
DEFAULT_DOOR_HEIGHT_CM = 210.0
DEFAULT_WALL_THICKNESS_CM = 20.0
DEFAULT_WALL_HEIGHT_CM = 300.0
MIN_SEGMENT_CM = 8.0


def cm_to_ft(value_cm):
    return float(value_cm) / CM_PER_FT


def ft_to_cm(value_ft):
    return float(value_ft) * CM_PER_FT


def distance_xy(p1, p2):
    dx = float(p2.X) - float(p1.X)
    dy = float(p2.Y) - float(p1.Y)
    return math.sqrt((dx * dx) + (dy * dy))


def first_or_none(iterable):
    for item in iterable:
        return item
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


def parse_float_text(value_text, fallback):
    try:
        return float(str(value_text).replace(",", ".").strip())
    except Exception:
        return float(fallback)


def choose_sketch_settings():
    form = Form()
    form.Text = "sk2rv1 Settings"
    form.Width = 470
    form.Height = 265
    form.FormBorderStyle = FormBorderStyle.FixedDialog
    set_form_center(form)
    form.MinimizeBox = False
    form.MaximizeBox = False

    fields = [
        ("door_width_cm", "Measured door width (cm)", DEFAULT_DOOR_WIDTH_CM),
        ("door_height_cm", "Default door height (cm)", DEFAULT_DOOR_HEIGHT_CM),
        ("wall_thickness_cm", "Wall thickness (cm)", DEFAULT_WALL_THICKNESS_CM),
        ("wall_height_cm", "Wall height (cm)", DEFAULT_WALL_HEIGHT_CM),
    ]

    controls = {}
    top = 18
    for key, title, default in fields:
        label = Label()
        label.Left = 12
        label.Top = top + 4
        label.Width = 210
        label.Text = title
        form.Controls.Add(label)

        box = TextBox()
        box.Left = 235
        box.Top = top
        box.Width = 195
        box.Text = str(default)
        form.Controls.Add(box)
        controls[key] = box
        top += 34

    note = Label()
    note.Left = 12
    note.Top = top + 2
    note.Width = 418
    note.Height = 40
    note.Text = "The two clicks are only for calibration. Walls and doors are recognized from the image; doors require a swing arc near the gap."
    form.Controls.Add(note)

    btn_ok = Button()
    btn_ok.Text = "Continue"
    btn_ok.Left = 255
    btn_ok.Top = 182
    btn_ok.Width = 82
    btn_ok.DialogResult = DialogResult.OK
    form.Controls.Add(btn_ok)

    btn_cancel = Button()
    btn_cancel.Text = "Cancel"
    btn_cancel.Left = 348
    btn_cancel.Top = 182
    btn_cancel.Width = 82
    btn_cancel.DialogResult = DialogResult.Cancel
    form.Controls.Add(btn_cancel)

    form.AcceptButton = btn_ok
    form.CancelButton = btn_cancel

    if form.ShowDialog() != DialogResult.OK:
        return None

    out = {}
    for key, _, default in fields:
        out[key] = parse_float_text(controls[key].Text, default)
    return out


def pick_image_path():
    dialog = OpenFileDialog()
    dialog.Filter = "Image Files|*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"
    dialog.Multiselect = False
    dialog.Title = "Select sketch image"
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


def get_wall_type_nearest(thickness_ft):
    wall_types = [wt for wt in FilteredElementCollector(doc).OfClass(WallType) if wt.Kind == WallKind.Basic]
    if not wall_types:
        return None
    return min(wall_types, key=lambda wt: abs(float(wt.Width) - float(thickness_ft)))


def set_param(element, bip, value):
    if element is None:
        return False
    try:
        param = element.get_Parameter(bip)
        if param and (not param.IsReadOnly):
            param.Set(value)
            return True
    except Exception:
        pass
    return False


def set_param_by_names(element, names, value):
    if element is None:
        return False
    for name in names:
        try:
            param = element.LookupParameter(name)
            if param and (not param.IsReadOnly):
                param.Set(value)
                return True
        except Exception:
            pass
    return False


def get_param_double(element, bip, names):
    try:
        param = element.get_Parameter(bip)
        if param and param.HasValue:
            return param.AsDouble()
    except Exception:
        pass
    for name in names:
        try:
            param = element.LookupParameter(name)
            if param and param.HasValue:
                return param.AsDouble()
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
            if abs(w - width_ft) <= cm_to_ft(1.0) and abs(h - height_ft) <= cm_to_ft(1.0):
                return sym
    except Exception:
        pass

    target_name = "{}_{:.0f}x{:.0f}cm".format(name_prefix, ft_to_cm(width_ft), ft_to_cm(height_ft))
    sym = base_symbol
    try:
        for idx in range(0, 50):
            name = target_name if idx == 0 else "{}_{}".format(target_name, idx)
            try:
                new_id = base_symbol.Duplicate(name)
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


def set_opening_size(instance, width_ft, height_ft, width_bip, height_bip):
    width_ok = set_param(instance, width_bip, width_ft)
    height_ok = set_param(instance, height_bip, height_ft)
    if not width_ok:
        width_ok = set_param_by_names(instance, ["Width", "width", "Rough Width"], width_ft)
    if not height_ok:
        height_ok = set_param_by_names(instance, ["Height", "height", "Rough Height"], height_ft)

    symbol = instance.Symbol if hasattr(instance, "Symbol") else None
    if symbol:
        if not width_ok:
            set_param(symbol, width_bip, width_ft) or set_param_by_names(symbol, ["Width", "width", "Rough Width"], width_ft)
        if not height_ok:
            set_param(symbol, height_bip, height_ft) or set_param_by_names(symbol, ["Height", "height", "Rough Height"], height_ft)


def get_symbol(category):
    return first_or_none(FilteredElementCollector(doc).OfCategory(category).OfClass(FamilySymbol))


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
    width, height = get_image_size(image_path)
    if width > 0.0 and height > 0.0:
        return float(width) / float(height)
    return 1.0


def get_jpeg_exif_orientation(image_path):
    try:
        handle = open(image_path, "rb")
        try:
            data = handle.read()
        finally:
            handle.close()
    except Exception:
        return None

    if not data or len(data) < 4:
        return None

    try:
        if data[:2] != "\xFF\xD8":
            return None

        pos = 2
        total_len = len(data)
        while (pos + 4) < total_len:
            if data[pos] != "\xFF":
                break

            marker = ord(data[pos + 1])
            pos += 2

            if marker == 0xD8 or marker == 0xD9 or (0xD0 <= marker <= 0xD7):
                continue

            seg_len = struct.unpack(">H", data[pos:pos + 2])[0]
            if seg_len < 2 or (pos + seg_len) > total_len:
                break

            seg = data[pos + 2:pos + seg_len]
            pos += seg_len

            if marker != 0xE1 or not seg.startswith("Exif\x00\x00"):
                continue

            tiff = seg[6:]
            if len(tiff) < 8:
                return None

            endian = tiff[:2]
            if endian == "II":
                u16 = lambda chunk: struct.unpack("<H", chunk)[0]
                u32 = lambda chunk: struct.unpack("<I", chunk)[0]
            elif endian == "MM":
                u16 = lambda chunk: struct.unpack(">H", chunk)[0]
                u32 = lambda chunk: struct.unpack(">I", chunk)[0]
            else:
                return None

            if u16(tiff[2:4]) != 42:
                return None

            ifd0 = u32(tiff[4:8])
            if (ifd0 + 2) > len(tiff):
                return None

            count = u16(tiff[ifd0:ifd0 + 2])
            base = ifd0 + 2
            for idx in range(count):
                entry = tiff[base + (idx * 12):base + ((idx + 1) * 12)]
                if len(entry) < 12:
                    break
                tag = u16(entry[0:2])
                if tag == 0x0112:
                    return u16(entry[8:10])
            return None
    except Exception:
        return None

    return None


def get_image_size(image_path):
    if not image_path:
        return (0, 0)
    try:
        img = DrawingImage.FromFile(image_path)
        try:
            width = int(img.Width)
            height = int(img.Height)
        finally:
            img.Dispose()

        orientation = get_jpeg_exif_orientation(image_path)
        if orientation in (5, 6, 7, 8):
            return (height, width)
        return (width, height)
    except Exception:
        return (0, 0)


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


def fit_image_overlay(image_inst, target_width_ft, target_height_ft, target_center):
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

    current_width = None
    current_height = None
    try:
        current_width = float(image_inst.Width)
    except Exception:
        current_width = None
    try:
        current_height = float(image_inst.Height)
    except Exception:
        current_height = None

    if current_width is None or current_height is None or current_width <= 1.0e-9 or current_height <= 1.0e-9:
        bbox = None
        try:
            bbox = image_inst.get_BoundingBox(doc.ActiveView)
        except Exception:
            bbox = None
        if bbox:
            current_width = float(bbox.Max.X - bbox.Min.X)
            current_height = float(bbox.Max.Y - bbox.Min.Y)

    if current_width and current_height and current_width > 1.0e-9 and current_height > 1.0e-9:
        scale_ratio = min(float(target_width_ft) / current_width, float(target_height_ft) / current_height)
        final_width = current_width * scale_ratio
        try:
            image_inst.Width = final_width
        except Exception:
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


def scale_image_overlay(image_inst, scale_ratio, target_center):
    if image_inst is None or scale_ratio <= 1.0e-9:
        return
    try:
        image_inst.LockProportions = True
    except Exception:
        pass

    try:
        image_inst.Width = float(image_inst.Width) * float(scale_ratio)
    except Exception:
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


def get_image_bbox(image_inst, view):
    bbox = image_inst.get_BoundingBox(view)
    if bbox is None:
        raise Exception("Image bounding box is unavailable.")
    return (float(bbox.Min.X), float(bbox.Min.Y), float(bbox.Max.X), float(bbox.Max.Y))


def create_temp_text_type():
    base_type_id = doc.GetDefaultElementTypeId(ElementTypeGroup.TextNoteType)
    base_type = doc.GetElement(base_type_id)
    if not isinstance(base_type, TextNoteType):
        return base_type_id, None

    base_name = "__sk2rv1_note"
    existing_names = set()
    for text_type in FilteredElementCollector(doc).OfClass(TextNoteType):
        try:
            existing_names.add(text_type.Name)
        except Exception:
            pass

    name = base_name
    idx = 1
    while name in existing_names:
        name = "{}_{}".format(base_name, idx)
        idx += 1

    temp_type = base_type.Duplicate(name)
    set_param(temp_type, BuiltInParameter.TEXT_SIZE, cm_to_ft(0.3))
    return temp_type.Id, temp_type


def create_instruction_note(view, origin, text_type_id, width_ft):
    text = (
        "sk2rv1\n"
        "1) Pick the left and right edge of the measured door\n"
        "2) The tool recognizes thick walls from the sketch image\n"
        "3) Doors are placed only when a gap has nearby swing-arc evidence"
    )
    try:
        return TextNote.Create(doc, view.Id, origin, width_ft, text, text_type_id)
    except Exception:
        return TextNote.Create(doc, view.Id, origin, text, text_type_id)


def cleanup_temp_elements(ids_to_delete):
    if not ids_to_delete:
        return
    tx = Transaction(doc, "Cleanup sk2rv1 temp overlay")
    try:
        tx.Start()
        for elem_id in ids_to_delete:
            try:
                doc.Delete(elem_id)
            except Exception:
                pass
        tx.Commit()
    except Exception:
        try:
            if tx.HasStarted():
                tx.RollBack()
        except Exception:
            pass


def pick_point(prompt):
    return uidoc.Selection.PickPoint(prompt)


def read_text_file(path):
    try:
        handle = open(path, "r")
        try:
            return handle.read()
        finally:
            handle.close()
    except Exception:
        return None


def quote_arg(value):
    text = str(value or "")
    if not text:
        return '""'
    if re.search(r'[\s"]', text) is None:
        return text
    return '"' + text.replace('"', '\\"') + '"'


def extract_json_payload(text):
    payload = (text or "").strip()
    if not payload:
        return None

    try:
        json.loads(payload)
        return payload
    except Exception:
        pass

    start = payload.find("{")
    end = payload.rfind("}")
    if start >= 0 and end > start:
        candidate = payload[start:end + 1].strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    for line in reversed(payload.splitlines()):
        candidate = (line or "").strip()
        if not candidate:
            continue
        if not candidate.startswith("{"):
            continue
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    return None


def get_pyrevit_cpython_info():
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None, None, "APPDATA was not found."

    config_path = os.path.join(appdata, "pyRevit", "pyRevit_config.ini")
    config_text = read_text_file(config_path)
    if not config_text:
        return None, None, "pyRevit config was not found at {}.".format(config_path)

    clone_root = None
    clone_match = re.search(r'"master"\s*:\s*"([^"]+)"', config_text)
    if clone_match:
        clone_root = clone_match.group(1)
    if not clone_root:
        clone_root = os.path.join(appdata, "pyRevit-Master")

    engine_match = re.search(r"(?m)^\s*cpyengine\s*=\s*([0-9]+)\s*$", config_text)
    if not engine_match:
        return None, None, "cpyengine was not found in {}.".format(config_path)

    engine_code = engine_match.group(1)
    python_exe = os.path.join(clone_root, "bin", "cengines", "CPY{}".format(engine_code), "python.exe")
    site_packages = os.path.join(clone_root, "site-packages")

    if not os.path.isfile(python_exe):
        return None, None, "pyRevit CPython was not found at {}.".format(python_exe)
    if not os.path.isdir(site_packages):
        return None, None, "pyRevit site-packages was not found at {}.".format(site_packages)

    return python_exe, site_packages, None


def run_recognition_helper(image_path, door_hint_px):
    python_exe, site_packages, info_err = get_pyrevit_cpython_info()
    if info_err:
        raise Exception(info_err)

    helper_path = os.path.join(SCRIPT_DIR, "sketch_recognition.py")
    if not os.path.isfile(helper_path):
        raise Exception("Recognition helper was not found at {}.".format(helper_path))

    parts = [
        quote_arg(python_exe),
        quote_arg(helper_path),
        "--image",
        quote_arg(image_path),
        "--site-packages",
        quote_arg(site_packages),
    ]

    if isinstance(door_hint_px, dict):
        center = door_hint_px.get("center")
        if center is not None:
            parts.extend(["--hint-cx", str(float(center[0]))])
            parts.extend(["--hint-cy", str(float(center[1]))])
        width_px = door_hint_px.get("width_px")
        if width_px is not None:
            parts.extend(["--hint-width", str(float(width_px))])
        orientation = door_hint_px.get("preferred_orientation")
        if orientation:
            parts.extend(["--hint-orientation", str(orientation)])

    start_info = ProcessStartInfo()
    start_info.FileName = python_exe
    start_info.Arguments = " ".join(parts[1:])
    start_info.UseShellExecute = False
    start_info.RedirectStandardOutput = True
    start_info.RedirectStandardError = True
    start_info.CreateNoWindow = True
    start_info.WorkingDirectory = SCRIPT_DIR

    proc = Process()
    proc.StartInfo = start_info
    proc.Start()
    stdout = proc.StandardOutput.ReadToEnd()
    stderr = proc.StandardError.ReadToEnd()
    proc.WaitForExit()

    if proc.ExitCode != 0:
        raise Exception("Recognition helper failed with exit code {}.\n{}".format(proc.ExitCode, stderr.strip() or stdout.strip() or "No output."))

    payload = extract_json_payload(stdout)
    if not payload:
        combined = (stdout or "").strip()
        if stderr:
            combined = (combined + "\n" + stderr.strip()).strip() if combined else stderr.strip()
        raise Exception("Recognition helper returned no JSON output.\n{}".format(combined[:1200] or "No output."))
    try:
        return json.loads(payload)
    except Exception as ex:
        raise Exception("Recognition helper returned invalid JSON.\n{}\n{}".format(ex, payload[:1200]))


def revit_point_to_image_px(point, bbox_ft, img_w, img_h):
    width_ft = float(bbox_ft[2] - bbox_ft[0])
    height_ft = float(bbox_ft[3] - bbox_ft[1])
    if width_ft <= 1.0e-9 or height_ft <= 1.0e-9:
        raise Exception("Invalid image overlay size.")

    x = ((float(point.X) - bbox_ft[0]) / width_ft) * float(img_w)
    y = ((float(point.Y) - bbox_ft[1]) / height_ft) * float(img_h)
    return (x, y)


def image_px_to_revit_xyz(pt_px, bbox_ft, img_w, img_h, z_ft=0.0):
    width_ft = float(bbox_ft[2] - bbox_ft[0])
    height_ft = float(bbox_ft[3] - bbox_ft[1])
    x_ft = bbox_ft[0] + (float(pt_px[0]) / float(img_w)) * width_ft
    # Image y=0 is top.  Revit bbox Min.Y is bottom of the overlay.
    # The overlay is placed so that image top-left = (Min.X, Min.Y) in the
    # Revit plan view coordinate system, i.e. image Y grows in the same
    # direction as Revit Y.
    y_ft = bbox_ft[1] + (float(pt_px[1]) / float(img_h)) * height_ft
    return XYZ(x_ft, y_ft, z_ft)


def convert_segments_to_revit(segments_px, doors_px, bbox_ft, img_w, img_h):
    segments_ft = []
    for seg in segments_px or []:
        start = image_px_to_revit_xyz(seg["start"], bbox_ft, img_w, img_h, 0.0)
        end = image_px_to_revit_xyz(seg["end"], bbox_ft, img_w, img_h, 0.0)
        if distance_xy(start, end) <= cm_to_ft(MIN_SEGMENT_CM):
            continue
        segments_ft.append({
            "segment_index": seg["index"],
            "orientation": seg["orientation"],
            "start": start,
            "end": end,
        })

    doors_ft = []
    for door in doors_px or []:
        center = image_px_to_revit_xyz(door["center"], bbox_ft, img_w, img_h, 0.0)
        width_ft = (float(door["width_px"]) / float(img_w)) * float(bbox_ft[2] - bbox_ft[0]) if door["orientation"] == "h" else (float(door["width_px"]) / float(img_h)) * float(bbox_ft[3] - bbox_ft[1])
        doors_ft.append({
            "segment_index": door["segment_index"],
            "orientation": door["orientation"],
            "center": center,
            "width_ft": width_ft,
        })
    return segments_ft, doors_ft


def create_model(level, wall_type, wall_height_ft, door_height_ft, segments_ft, doors_ft):
    tx = Transaction(doc, "sk2rv1 Create Model")
    walls_by_segment = {}
    door_ids = []
    wall_ids = []
    skipped_doors = 0

    door_base_symbol = get_symbol(BuiltInCategory.OST_Doors)

    try:
        tx.Start()
        for seg in segments_ft:
            wall = Wall.Create(doc, Line.CreateBound(seg["start"], seg["end"]), wall_type.Id, level.Id, wall_height_ft, 0.0, False, False)
            wall_ids.append(wall.Id)
            walls_by_segment[seg["segment_index"]] = wall

        if door_base_symbol and (not door_base_symbol.IsActive):
            door_base_symbol.Activate()
            doc.Regenerate()

        for door in doors_ft:
            host_wall = walls_by_segment.get(door["segment_index"])
            if host_wall is None or door_base_symbol is None:
                skipped_doors += 1
                continue
            width_ft = max(door["width_ft"], cm_to_ft(50.0))
            door_symbol = ensure_sized_symbol(
                door_base_symbol,
                width_ft,
                door_height_ft,
                BuiltInParameter.DOOR_WIDTH,
                BuiltInParameter.DOOR_HEIGHT,
                "__sk2rv1_door",
            )
            if door_symbol is None:
                skipped_doors += 1
                continue
            if not door_symbol.IsActive:
                door_symbol.Activate()
                doc.Regenerate()
            inst = doc.Create.NewFamilyInstance(
                door["center"],
                door_symbol,
                host_wall,
                level,
                Structure.StructuralType.NonStructural,
            )
            set_opening_size(inst, width_ft, door_height_ft, BuiltInParameter.DOOR_WIDTH, BuiltInParameter.DOOR_HEIGHT)
            door_ids.append(inst.Id)

        tx.Commit()
    except Exception:
        try:
            if tx.HasStarted():
                tx.RollBack()
        except Exception:
            pass
        raise

    return {
        "wall_ids": wall_ids,
        "door_ids": door_ids,
        "skipped_doors": skipped_doors,
    }


def run_command():
    plan_view = get_plan_view()
    if plan_view is None:
        TaskDialog.Show("sk2rv1", "Open a floor plan view first.")
        return

    level = get_level_from_view(plan_view)
    if level is None:
        TaskDialog.Show("sk2rv1", "No level was found for the active plan view.")
        return

    settings = choose_sketch_settings()
    if settings is None:
        return

    image_path = pick_image_path()
    if not image_path:
        return

    wall_type = get_wall_type_nearest(cm_to_ft(settings["wall_thickness_cm"]))
    if wall_type is None:
        TaskDialog.Show("sk2rv1", "No basic wall type was found in this Revit model.")
        return

    image_ids = []
    note_ids = []
    overlay_err = None
    image_inst = None
    overlay_center = None

    tx_overlay = Transaction(doc, "sk2rv1 Overlay")
    try:
        tx_overlay.Start()
        view_min, view_max, center = get_view_bounds(plan_view)
        overlay_center = center
        view_w = abs(float(view_max.X) - float(view_min.X))
        view_h = abs(float(view_max.Y) - float(view_min.Y))
        aspect = get_image_aspect(image_path)
        target_w = view_w * 0.82
        target_h = view_h * 0.82
        if aspect > 1.0:
            target_h = min(target_h, target_w / aspect)
        else:
            target_w = min(target_w, target_h * aspect)

        image_type, image_inst, overlay_err = create_image_overlay(plan_view, image_path, center)
        if image_type:
            image_ids.append(image_type.Id)
        if image_inst:
            image_ids.append(image_inst.Id)
            fit_image_overlay(image_inst, target_w, target_h, center)

        note_type_id, note_type = create_temp_text_type()
        if note_type:
            note_ids.append(note_type.Id)

        note_width = view_w * 0.0875
        note_origin = XYZ(float(view_min.X) + (view_w * 0.03), float(view_max.Y) - (view_h * 0.12), 0.0)
        if image_inst:
            img_bbox = get_image_bbox(image_inst, plan_view)
            note_width = min(note_width, max(view_w * 0.05, (img_bbox[2] - img_bbox[0]) * 0.25))
            note_x = max(float(view_min.X) + (view_w * 0.02), float(img_bbox[0]) - note_width - (view_w * 0.015))
            note_y = float(img_bbox[3]) - (view_h * 0.02)
            note_origin = XYZ(note_x, note_y, 0.0)

        note = create_instruction_note(plan_view, note_origin, note_type_id, note_width)
        if note:
            note_ids.append(note.Id)
        tx_overlay.Commit()
    except Exception as ex:
        overlay_err = str(ex)
        try:
            if tx_overlay.HasStarted():
                tx_overlay.RollBack()
        except Exception:
            pass

    if image_inst is None:
        TaskDialog.Show("sk2rv1", "Image overlay failed.\n{}".format(overlay_err if overlay_err else "Unknown error"))
        cleanup_temp_elements(image_ids + note_ids)
        return

    try:
        img_w, img_h = get_image_size(image_path)
        if img_w <= 0 or img_h <= 0:
            raise Exception("Could not read image size.")

        calib_left = pick_point("Pick the LEFT edge of the measured door")
        calib_right = pick_point("Pick the RIGHT edge of the measured door")
        calib_dist_ft = distance_xy(calib_left, calib_right)
        if calib_dist_ft <= 1.0e-9:
            raise Exception("Calibration points are identical.")

        bbox_before = get_image_bbox(image_inst, plan_view)
        hint_left_px = revit_point_to_image_px(calib_left, bbox_before, img_w, img_h)
        hint_right_px = revit_point_to_image_px(calib_right, bbox_before, img_w, img_h)
        door_hint_px = {
            "center": ((hint_left_px[0] + hint_right_px[0]) * 0.5, (hint_left_px[1] + hint_right_px[1]) * 0.5),
            "width_px": math.sqrt(
                ((hint_right_px[0] - hint_left_px[0]) * (hint_right_px[0] - hint_left_px[0])) +
                ((hint_right_px[1] - hint_left_px[1]) * (hint_right_px[1] - hint_left_px[1]))
            ),
            "preferred_orientation": "h" if abs(hint_right_px[0] - hint_left_px[0]) >= abs(hint_right_px[1] - hint_left_px[1]) else "v",
        }

        desired_door_ft = cm_to_ft(settings["door_width_cm"])
        scale_ratio = desired_door_ft / calib_dist_ft

        tx_scale = Transaction(doc, "sk2rv1 Calibrate Overlay")
        tx_scale.Start()
        try:
            scale_image_overlay(image_inst, scale_ratio, overlay_center)
            tx_scale.Commit()
        except Exception:
            try:
                tx_scale.RollBack()
            except Exception:
                pass
            raise

        bbox_after = get_image_bbox(image_inst, plan_view)

        recognition = run_recognition_helper(image_path, door_hint_px)
        segments_px = recognition.get("segments_px", [])
        doors_px = recognition.get("doors_px", [])
        if not segments_px:
            raise Exception("No wall runs were recognized from the sketch.")

        # Use image dimensions from recognition (guaranteed to match segment coords).
        rec_w = recognition.get("img_w", 0)
        rec_h = recognition.get("img_h", 0)
        if rec_w > 0 and rec_h > 0:
            img_w, img_h = rec_w, rec_h

        segments_ft, doors_ft = convert_segments_to_revit(segments_px, doors_px, bbox_after, img_w, img_h)
        if not segments_ft:
            raise Exception("Wall recognition returned only degenerate segments.")

        # Write diagnostic log for debugging coordinate mapping.
        try:
            debug_dir = os.path.dirname(image_path)
            debug_lines = [
                "img_w={} img_h={}".format(img_w, img_h),
                "bbox_before=({:.3f}, {:.3f}, {:.3f}, {:.3f})".format(*bbox_before),
                "bbox_after=({:.3f}, {:.3f}, {:.3f}, {:.3f})".format(*bbox_after),
                "scale_ratio={:.6f}".format(scale_ratio),
            ]
            for seg in segments_px[:3]:
                debug_lines.append("seg_px: {} ({},{}) -> ({},{})".format(
                    seg["orientation"], seg["start"][0], seg["start"][1],
                    seg["end"][0], seg["end"][1]))
            for seg in segments_ft[:3]:
                debug_lines.append("seg_ft: {} ({:.3f},{:.3f}) -> ({:.3f},{:.3f})".format(
                    seg["orientation"], seg["start"].X, seg["start"].Y,
                    seg["end"].X, seg["end"].Y))
            log_path = os.path.join(debug_dir, "sk2rv1_debug.txt")
            handle = open(log_path, "w")
            try:
                handle.write("\n".join(debug_lines))
            finally:
                handle.close()
        except Exception:
            pass

        build = create_model(
            level,
            wall_type,
            cm_to_ft(settings["wall_height_cm"]),
            cm_to_ft(settings["door_height_cm"]),
            segments_ft,
            doors_ft,
        )

        cleanup_temp_elements(note_ids)
        TaskDialog.Show(
            "sk2rv1",
            "Created {} walls and {} doors.\nRecognized bars: {}. Swing arcs used: {}.\nOverlay kept in view for inspection.".format(
                len(build["wall_ids"]),
                len(build["door_ids"]),
                recognition.get("bar_count", 0),
                recognition.get("arc_count", 0),
            ),
        )
    except OperationCanceledException:
        cleanup_temp_elements(image_ids + note_ids)
        TaskDialog.Show("sk2rv1", "Canceled.")
    except Exception as ex:
        cleanup_temp_elements(image_ids + note_ids)
        TaskDialog.Show("sk2rv1", "Failed.\n{}".format(ex))


if __name__ == "__main__":
    run_command()
