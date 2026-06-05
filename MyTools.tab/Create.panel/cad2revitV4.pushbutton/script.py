# -*- coding: utf-8 -*-
from Autodesk.Revit.DB import *
import math
import os
import datetime
from collections import defaultdict
from System.Collections.Generic import List

try:
    import hashlib
except Exception:
    hashlib = None


doc = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument

# ============================================================
# PARAMETERS (TUNE)
# ============================================================
MM_TO_FT = 1.0 / 304.8

MIN_LEN_FT = 300 * MM_TO_FT
ANGLE_TOL_DEG = 1.0
OVERLAP_MIN = 0.60
THICK_MIN_FT = 70 * MM_TO_FT
THICK_MAX_FT = 600 * MM_TO_FT
THICK_CLUSTER_TOL_FT = 8 * MM_TO_FT

COLLINEAR_DIST_TOL_FT = 5 * MM_TO_FT
ENDPOINT_SNAP_TOL_FT = 10 * MM_TO_FT
INTERSECT_TOL_FT = 5 * MM_TO_FT

DEFAULT_WALL_HEIGHT_FT = 3000 * MM_TO_FT  # 3m
DEFAULT_SILL_FT = 1000 * MM_TO_FT  # window sill 1m (fallback)
DEFAULT_HEAD_FT = 2100 * MM_TO_FT  # window head 2.1m (fallback)
DEFAULT_DOOR_HEIGHT_FT = 2100 * MM_TO_FT  # door height 2.1m (fallback)

# Door arc detection
DOOR_ARC_R_MIN_FT = 300 * MM_TO_FT  # 0.3m
DOOR_ARC_R_MAX_FT = 2000 * MM_TO_FT  # 2.0m
DOOR_ARC_SWEEP_MIN_DEG = 20.0
DOOR_ARC_SWEEP_MAX_DEG = 175.0
DOOR_HOST_DIST_FT = 450 * MM_TO_FT  # max distance from arc center to wall centerline

# Gap -> opening detection (window/door fallback)
GAP_MIN_DOOR_FT = 550 * MM_TO_FT
GAP_MAX_DOOR_FT = 1600 * MM_TO_FT
# Arc-free door gaps up to ~1300mm are common enough to classify as doors
# even when some other doors in the DWG do have swing arcs.
GAP_DOOR_AUTO_MAX_FT = 1300 * MM_TO_FT
GAP_MIN_WIN_FT = 250 * MM_TO_FT
GAP_MAX_WIN_FT = 4000 * MM_TO_FT
ARC_NEAR_GAP_FT = 800 * MM_TO_FT  # tolerate hinge point offsets / short host wall segments
OPENING_AXIS_BUCKET_FT = 60 * MM_TO_FT  # tolerate CAD jitter when grouping openings
OPENING_HOST_FALLBACK_DIST_FT = 700 * MM_TO_FT
OUTER_EDGE_TOL_FT = 500 * MM_TO_FT  # keep close to facade centerlines; avoid nearby interior walls

# Preview / Build toggles
PREVIEW_CENTERLINES = False
BUILD_WALLS = True

PREVIEW_DOORS = True
BUILD_DOORS = True

PREVIEW_WINDOWS = True
BUILD_WINDOWS = True

# If CAD wall lines are broken around openings, stitch short collinear gaps
# so doors/windows can be hosted on continuous Revit walls.
BRIDGE_OPENING_GAPS = True
WALL_GAP_BRIDGE_MAX_FT = 4200 * MM_TO_FT  # 4.2m
WALL_GAP_AXIS_TOL_FT = 40 * MM_TO_FT  # relax axis matching only for gap stitching
WALL_GAP_ANGLE_TOL_DEG = 2.5

# Script/version marker shown in the lower corner of the detected DWG extents.
# Use a build label from the deployed script file so the note changes
# automatically when Revit is actually running a different script copy.
def _get_dwg_build_label():
    script_path = None
    try:
        script_path = __file__
    except Exception:
        script_path = None

    if not script_path:
        return "unknown"

    try:
        mtime = os.path.getmtime(script_path)
        stamp = datetime.datetime.fromtimestamp(mtime).strftime("%Y%m%d-%H%M%S")
    except Exception:
        stamp = "unknown-time"

    if hashlib is None:
        return stamp

    try:
        with open(script_path, "rb") as fh:
            digest = hashlib.md5(fh.read()).hexdigest()[:6]
        return "{}-{}".format(stamp, digest)
    except Exception:
        return stamp


DWG_VERSION = _get_dwg_build_label()


# ============================================================
# 2D helpers
# ============================================================
def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def vmul(a, s):
    return (a[0] * s, a[1] * s)


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def vlen(a):
    return math.sqrt(vdot(a, a))


def vnorm(a):
    L = vlen(a)
    return (a[0] / L, a[1] / L) if L > 1e-12 else (1.0, 0.0)


def xyz_to_xy(p):
    return (p.X, p.Y)


def xy_to_xyz(xy, z):
    return XYZ(xy[0], xy[1], z)


def angle_parallel(d1, d2, ang_deg):
    cos_tol = math.cos(math.radians(ang_deg))
    return abs(vdot(d1, d2)) >= cos_tol


def bbox_from_line_xy(a0, a1):
    return (min(a0[0], a1[0]), min(a0[1], a1[1]), max(a0[0], a1[0]), max(a0[1], a1[1]))


def bbox_expand(bb, r):
    return (bb[0] - r, bb[1] - r, bb[2] + r, bb[3] + r)


def dist_point_to_infinite_line_xy(p, l0, ldir):
    v = vsub(p, l0)
    t = vdot(v, ldir)
    proj = vadd(l0, vmul(ldir, t))
    dx = p[0] - proj[0]
    dy = p[1] - proj[1]
    return math.sqrt(dx * dx + dy * dy)


def closest_point_on_segment_xy(p, a, b):
    ab = vsub(b, a)
    L2 = vdot(ab, ab)
    if L2 < 1e-12:
        return a, 0.0
    t = vdot(vsub(p, a), ab) / L2
    t = max(0.0, min(1.0, t))
    return vadd(a, vmul(ab, t)), t


def dist_point_to_segment_xy(p, a, b):
    c, _ = closest_point_on_segment_xy(p, a, b)
    return vlen(vsub(p, c))


def project_scalar_on_axis(p, origin, axis_dir):
    return vdot(vsub(p, origin), axis_dir)


def overlap_ratio_1d(a0, a1, b0, b1):
    ov = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1e-12, min(a1 - a0, b1 - b0))
    return ov / denom


def line_intersection_param(a0, a1, b0, b1, tol=1e-12):
    ax, ay = a0
    bx, by = a1
    cx, cy = b0
    dx, dy = b1
    r = (bx - ax, by - ay)
    s = (dx - cx, dy - cy)
    rxs = r[0] * s[1] - r[1] * s[0]
    q_p = (cx - ax, cy - ay)
    if abs(rxs) < tol:
        return None
    t = (q_p[0] * s[1] - q_p[1] * s[0]) / rxs
    u = (q_p[0] * r[1] - q_p[1] * r[0]) / rxs
    return (t, u)


# ============================================================
# Spatial grid
# ============================================================
class SpatialGrid(object):
    def __init__(self, cell_size_ft):
        self.cs = cell_size_ft
        self.cells = defaultdict(list)

    def _keys_for_bbox(self, bb):
        x0 = int(math.floor(bb[0] / self.cs))
        y0 = int(math.floor(bb[1] / self.cs))
        x1 = int(math.floor(bb[2] / self.cs))
        y1 = int(math.floor(bb[3] / self.cs))
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                yield (ix, iy)

    def insert(self, obj, bb=None):
        bb = bb or obj.bb
        for k in self._keys_for_bbox(bb):
            self.cells[k].append(obj)

    def query(self, bb):
        seen = set()
        res = []
        for k in self._keys_for_bbox(bb):
            for obj in self.cells.get(k, []):
                oid = getattr(obj, "id", id(obj))
                if oid not in seen:
                    seen.add(oid)
                    res.append(obj)
        return res


# ============================================================
# Segments & center segments
# ============================================================
_SEG_ID = 0


class Seg(object):
    def __init__(self, revit_line):
        global _SEG_ID
        _SEG_ID += 1
        self.id = _SEG_ID
        self.rv = revit_line
        p0 = revit_line.GetEndPoint(0)
        p1 = revit_line.GetEndPoint(1)
        self.z = (p0.Z + p1.Z) / 2.0
        self.a = xyz_to_xy(p0)
        self.b = xyz_to_xy(p1)
        self.v = vsub(self.b, self.a)
        self.len = vlen(self.v)
        self.dir = vnorm(self.v)
        self.bb = bbox_from_line_xy(self.a, self.b)


class CenterSeg(object):
    def __init__(self, a_xy, b_xy, z, thick_ft):
        self.a = a_xy
        self.b = b_xy
        self.z = z
        self.v = vsub(self.b, self.a)
        self.len = vlen(self.v)
        self.dir = vnorm(self.v)
        self.bb = bbox_from_line_xy(self.a, self.b)
        self.thick = thick_ft


# ============================================================
# Extract geometry from ImportInstance
# ============================================================
def extract_lines_and_arcs(import_inst):
    opt = Options()
    opt.DetailLevel = ViewDetailLevel.Fine
    geo = import_inst.get_Geometry(opt)
    lines = []
    arcs = []

    def walk(g):
        for obj in g:
            if isinstance(obj, GeometryInstance):
                walk(obj.GetInstanceGeometry())
            elif isinstance(obj, Line):
                lines.append(obj)
            elif isinstance(obj, Arc):
                arcs.append(obj)
            elif isinstance(obj, PolyLine):
                pts = list(obj.GetCoordinates())
                if len(pts) >= 2:
                    for i in range(len(pts) - 1):
                        p0 = pts[i]
                        p1 = pts[i + 1]
                        if p0.DistanceTo(p1) > 1e-9:
                            lines.append(Line.CreateBound(p0, p1))

    walk(geo)
    return lines, arcs


# ============================================================
# Pairing
# ============================================================
def find_parallel_pairs(segs):
    CELL = 2.0 / 0.3048
    grid = SpatialGrid(CELL)
    for s in segs:
        grid.insert(s)

    pairs = []
    for a in segs:
        bbq = bbox_expand(a.bb, THICK_MAX_FT)
        neigh = grid.query(bbq)

        o = a.a
        d = a.dir
        a0 = project_scalar_on_axis(a.a, o, d)
        a1 = project_scalar_on_axis(a.b, o, d)
        if a0 > a1:
            a0, a1 = a1, a0

        for b in neigh:
            if b.id <= a.id:
                continue
            if not angle_parallel(a.dir, b.dir, ANGLE_TOL_DEG):
                continue

            b0 = project_scalar_on_axis(b.a, o, d)
            b1 = project_scalar_on_axis(b.b, o, d)
            if b0 > b1:
                b0, b1 = b1, b0

            if overlap_ratio_1d(a0, a1, b0, b1) < OVERLAP_MIN:
                continue

            t = dist_point_to_infinite_line_xy(b.a, a.a, a.dir)
            if t < THICK_MIN_FT or t > THICK_MAX_FT:
                continue

            pairs.append((a, b, t))
    return pairs


# ============================================================
# Thickness clustering + snap
# ============================================================
def cluster_thickness(values, tol_ft):
    clusters = []
    for v in sorted(values):
        placed = False
        for c in clusters:
            if abs(c[0] - v) <= tol_ft:
                c.append(v)
                placed = True
                break
        if not placed:
            clusters.append([v])
    reps = []
    for c in clusters:
        reps.append(sum(c) / len(c))
    return reps


def snap_to_cluster(v, reps):
    best = reps[0]
    bestd = abs(v - best)
    for r in reps[1:]:
        d = abs(v - r)
        if d < bestd:
            best = r
            bestd = d
    return best


# ============================================================
# Build centerline from a pair using overlap span
# ============================================================
def build_center_from_pair(a, b, thick_rep_ft):
    o = a.a
    d = a.dir

    a0 = project_scalar_on_axis(a.a, o, d)
    a1 = project_scalar_on_axis(a.b, o, d)
    b0 = project_scalar_on_axis(b.a, o, d)
    b1 = project_scalar_on_axis(b.b, o, d)
    if a0 > a1:
        a0, a1 = a1, a0
    if b0 > b1:
        b0, b1 = b1, b0

    s0 = max(a0, b0)
    s1 = min(a1, b1)
    if (s1 - s0) <= MIN_LEN_FT:
        return None

    pa0 = vadd(o, vmul(d, s0))
    pa1 = vadd(o, vmul(d, s1))

    vb = vsub(b.a, pa0)
    along = vmul(d, vdot(vb, d))
    n = vsub(vb, along)
    nlen = vlen(n)
    if nlen < 1e-9:
        n = (-d[1], d[0])
        nlen = 1.0
    n = (n[0] / nlen, n[1] / nlen)

    shift = thick_rep_ft * 0.5
    c0 = vadd(pa0, vmul(n, shift))
    c1 = vadd(pa1, vmul(n, shift))
    return CenterSeg(c0, c1, a.z, thick_rep_ft)


# ============================================================
# Merge collinear
# ============================================================
def are_collinear_same_axis(s1, s2, axis_tol_ft=COLLINEAR_DIST_TOL_FT, angle_tol_deg=ANGLE_TOL_DEG):
    if not angle_parallel(s1.dir, s2.dir, angle_tol_deg):
        return False
    d0 = dist_point_to_infinite_line_xy(s2.a, s1.a, s1.dir)
    d1 = dist_point_to_infinite_line_xy(s2.b, s1.a, s1.dir)
    return d0 <= axis_tol_ft and d1 <= axis_tol_ft


def merge_collinear(
    centers,
    max_join_gap_ft=ENDPOINT_SNAP_TOL_FT,
    axis_tol_ft=COLLINEAR_DIST_TOL_FT,
    angle_tol_deg=ANGLE_TOL_DEG,
    respect_thickness=True,
):
    buckets = defaultdict(list)
    if respect_thickness:
        for s in centers:
            key = int(round(s.thick / THICK_CLUSTER_TOL_FT))
            buckets[key].append(s)
    else:
        buckets[0] = list(centers)

    merged_all = []
    for key, segs in buckets.items():
        grid = SpatialGrid(2.0 / 0.3048)
        for s in segs:
            grid.insert(s)

        used = set()
        for s in segs:
            if id(s) in used:
                continue
            chain = [s]
            used.add(id(s))

            grown = True
            while grown:
                grown = False
                # chain bbox
                bb = chain[0].bb
                for c in chain[1:]:
                    bb = (
                        min(bb[0], c.bb[0]),
                        min(bb[1], c.bb[1]),
                        max(bb[2], c.bb[2]),
                        max(bb[3], c.bb[3]),
                    )
                neigh = grid.query(bbox_expand(bb, max_join_gap_ft))

                for n in neigh:
                    if id(n) in used:
                        continue
                    if respect_thickness and abs(n.thick - s.thick) > THICK_CLUSTER_TOL_FT:
                        continue
                    if not are_collinear_same_axis(s, n, axis_tol_ft, angle_tol_deg):
                        continue

                    o = s.a
                    d = s.dir

                    def t(p):
                        return project_scalar_on_axis(p, o, d)

                    ts = []
                    for cc in chain:
                        ts.extend([t(cc.a), t(cc.b)])
                    cmin, cmax = min(ts), max(ts)
                    nmin, nmax = sorted([t(n.a), t(n.b)])

                    if nmin <= cmax + max_join_gap_ft and nmax >= cmin - max_join_gap_ft:
                        chain.append(n)
                        used.add(id(n))
                        grown = True

            o = s.a
            d = s.dir
            ts = []
            for cc in chain:
                ts.extend([project_scalar_on_axis(cc.a, o, d), project_scalar_on_axis(cc.b, o, d)])
            t0, t1 = min(ts), max(ts)
            a_xy = vadd(o, vmul(d, t0))
            b_xy = vadd(o, vmul(d, t1))
            thick_ft = s.thick if respect_thickness else max([cc.thick for cc in chain])
            merged_all.append(CenterSeg(a_xy, b_xy, s.z, thick_ft))
    return merged_all


# ============================================================
# Split at intersections
# ============================================================
def split_at_intersections(centers):
    grid = SpatialGrid(2.0 / 0.3048)
    for s in centers:
        grid.insert(s)

    result = []
    for s in centers:
        params = [0.0, 1.0]
        neigh = grid.query(bbox_expand(s.bb, INTERSECT_TOL_FT))

        for o in neigh:
            if o is s:
                continue
            if angle_parallel(s.dir, o.dir, 1.0):
                continue

            inter = line_intersection_param(s.a, s.b, o.a, o.b)
            if not inter:
                continue
            ta, tb = inter
            if -0.01 <= ta <= 1.01 and -0.01 <= tb <= 1.01:
                tcl = max(0.0, min(1.0, ta))
                params.append(tcl)

        params = sorted(set([round(p, 6) for p in params]))
        for i in range(len(params) - 1):
            p0, p1 = params[i], params[i + 1]
            if (p1 - p0) <= 1e-6:
                continue
            a_xy = vadd(s.a, vmul(vsub(s.b, s.a), p0))
            b_xy = vadd(s.a, vmul(vsub(s.b, s.a), p1))
            if vlen(vsub(b_xy, a_xy)) < MIN_LEN_FT:
                continue
            result.append(CenterSeg(a_xy, b_xy, s.z, s.thick))
    return result


# ============================================================
# De-dup center segments (same thickness, same axis, same span)
# ============================================================
def key_center(seg):
    # canonical orientation: sort endpoints in axis scalar
    o = seg.a
    d = seg.dir
    t0 = project_scalar_on_axis(seg.a, o, d)
    t1 = project_scalar_on_axis(seg.b, o, d)
    if t0 > t1:
        # flip
        a, b = seg.b, seg.a
    else:
        a, b = seg.a, seg.b

    # quantize for stable hashing
    q = 5 * MM_TO_FT  # 5mm quant

    def qv(x):
        return int(round(x / q))

    # use a point near seg.a and direction to define axis
    return (
        int(round(seg.thick / THICK_CLUSTER_TOL_FT)),
        qv(a[0]),
        qv(a[1]),
        qv(b[0]),
        qv(b[1]),
        qv(seg.dir[0] * 1000.0),
        qv(seg.dir[1] * 1000.0),
    )


def dedup_centers(centers):
    seen = set()
    out = []
    for s in centers:
        k = key_center(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


# ============================================================
# Preview drawing
# ============================================================
def ensure_sketch_plane(z):
    plane = Plane.CreateByNormalAndOrigin(XYZ.BasisZ, XYZ(0, 0, z))
    return SketchPlane.Create(doc, plane)


def draw_model_lines(center_segs, name="Preview"):
    if not center_segs:
        return
    byz = defaultdict(list)
    for s in center_segs:
        byz[round(s.z, 6)].append(s)

    t = Transaction(doc, name)
    t.Start()
    for z, segs in byz.items():
        sp = ensure_sketch_plane(segs[0].z)
        for s in segs:
            ln = Line.CreateBound(xy_to_xyz(s.a, s.z), xy_to_xyz(s.b, s.z))
            doc.Create.NewModelCurve(ln, sp)
    t.Commit()


def draw_points_xy(points, z, name="PointsPreview"):
    if not points:
        return
    t = Transaction(doc, name)
    t.Start()
    sp = ensure_sketch_plane(z)
    for p in points:
        # tiny cross
        s = 150 * MM_TO_FT
        p1 = (p[0] - s, p[1])
        p2 = (p[0] + s, p[1])
        p3 = (p[0], p[1] - s)
        p4 = (p[0], p[1] + s)
        doc.Create.NewModelCurve(Line.CreateBound(xy_to_xyz(p1, z), xy_to_xyz(p2, z)), sp)
        doc.Create.NewModelCurve(Line.CreateBound(xy_to_xyz(p3, z), xy_to_xyz(p4, z)), sp)
    t.Commit()


# ============================================================
# WallType creation
# ============================================================
def _elem_name(elem):
    try:
        n = elem.Name
        if n:
            return n
    except Exception:
        pass
    try:
        n = Element.Name.GetValue(elem)
        if n:
            return n
    except Exception:
        pass
    return ""


def centers_xy_extents(center_segs):
    if not center_segs:
        return None
    xs = []
    ys = []
    for s in center_segs:
        xs.extend([s.a[0], s.b[0]])
        ys.extend([s.a[1], s.b[1]])
    return (min(xs), min(ys), max(xs), max(ys))


def is_outer_envelope_gap(p_xy, axis_dir, ext_xy, tol_ft=OUTER_EDGE_TOL_FT):
    if not ext_xy:
        return False
    dx = abs(axis_dir[0])
    dy = abs(axis_dir[1])
    if dx >= dy:
        return abs(p_xy[1] - ext_xy[1]) <= tol_ft or abs(p_xy[1] - ext_xy[3]) <= tol_ft
    return abs(p_xy[0] - ext_xy[0]) <= tol_ft or abs(p_xy[0] - ext_xy[2]) <= tol_ft


def place_dwg_version_note(view, ext_xy, level):
    if not ext_xy:
        return
    if not isinstance(view, ViewPlan):
        print("DWG version note skipped: open a plan view.")
        return

    note_text = "DWG v{}".format(DWG_VERSION)
    pad = 250 * MM_TO_FT
    note_pt = XYZ(ext_xy[0] + pad, ext_xy[1] + pad, level.Elevation if level else 0.0)

    t = Transaction(doc, "DWG Version Note")
    t.Start()
    try:
        # Keep only one DWG version note per active view.
        for tn in FilteredElementCollector(doc, view.Id).OfClass(TextNote):
            txt = ""
            try:
                txt = tn.Text
            except Exception:
                txt = ""
            if txt and txt.startswith("DWG v"):
                doc.Delete(tn.Id)

        tnt = FilteredElementCollector(doc).OfClass(TextNoteType).FirstElement()
        if tnt:
            TextNote.Create(doc, view.Id, note_pt, note_text, tnt.Id)
            print("DWG note:", note_text)
        else:
            print("DWG version note skipped: no TextNoteType.")
        t.Commit()
    except Exception as ex:
        t.RollBack()
        print("Could not place DWG version note:", ex)


def focus_elements_in_ui(elems):
    if not elems:
        return
    try:
        ids = List[ElementId]()
        for e in elems:
            ids.Add(e.Id)
        uidoc.Selection.SetElementIds(ids)
        uidoc.ShowElements(ids)
        try:
            uidoc.RefreshActiveView()
        except Exception:
            pass
        print("Focused on created elements.")
    except Exception as ex:
        print("Could not focus on created elements:", ex)


def get_or_create_walltype(doc, thick_ft):
    base = FilteredElementCollector(doc).OfClass(WallType).FirstElement()
    mm = thick_ft * 304.8
    name = "CAD_Auto_{:.0f}mm".format(mm)

    for wt in FilteredElementCollector(doc).OfClass(WallType):
        if _elem_name(wt) == name:
            return wt

    dup_result = base.Duplicate(name)
    # Revit API return type can vary by version/binding:
    # some return ElementId, others return the duplicated ElementType.
    if isinstance(dup_result, ElementId):
        new_type = doc.GetElement(dup_result)
    else:
        new_type = dup_result
    cs = new_type.GetCompoundStructure()
    if cs is None:
        return new_type
    layers = list(cs.GetLayers())
    if not layers:
        return new_type

    # Keep membrane layers at zero and scale the physical layers to target width.
    # If this specific wall template rejects the resulting structure, keep the
    # duplicated wall type unchanged instead of failing the whole command.
    try:
        tiny = 0.1 * MM_TO_FT  # 0.1 mm minimum positive width
        non_mem_idxs = []
        non_mem_total = 0.0

        for i, lyr in enumerate(layers):
            is_mem = False
            try:
                is_mem = lyr.Function == MaterialFunctionAssignment.Membrane
            except Exception:
                is_mem = False

            if is_mem:
                lyr.Width = 0.0
            else:
                non_mem_idxs.append(i)
                non_mem_total += max(0.0, lyr.Width)

        if non_mem_idxs:
            if non_mem_total > 1e-9:
                scale = thick_ft / non_mem_total
                for i in non_mem_idxs:
                    layers[i].Width = max(tiny, layers[i].Width * scale)
            else:
                primary = non_mem_idxs[0]
                for i in non_mem_idxs:
                    layers[i].Width = 0.0
                layers[primary].Width = max(tiny, thick_ft)

            cs.SetLayers(layers)
            new_type.SetCompoundStructure(cs)
    except Exception:
        pass
    return new_type


# ============================================================
# Create walls
# ============================================================
def create_walls(center_segs, level, height_ft):
    t = Transaction(doc, "CAD -> Revit Walls")
    t.Start()

    wt_cache = {}
    created = []
    for s in center_segs:
        thick_key = int(round(s.thick / THICK_CLUSTER_TOL_FT))
        if thick_key not in wt_cache:
            wt_cache[thick_key] = get_or_create_walltype(doc, s.thick)
        wt = wt_cache[thick_key]

        ln = Line.CreateBound(xy_to_xyz(s.a, level.Elevation), xy_to_xyz(s.b, level.Elevation))
        w = Wall.Create(doc, ln, wt.Id, level.Id, height_ft, 0.0, False, False)
        created.append(w)
    t.Commit()
    return created


# ============================================================
# Find nearest wall to a point (for hosting)
# ============================================================
class WallRef2D(object):
    def __init__(self, wall):
        self.wall = wall
        loc = wall.Location
        self.curve = loc.Curve  # Revit curve
        p0 = self.curve.GetEndPoint(0)
        p1 = self.curve.GetEndPoint(1)
        self.z = p0.Z
        self.a = xyz_to_xy(p0)
        self.b = xyz_to_xy(p1)
        self.dir = vnorm(vsub(self.b, self.a))
        self.bb = bbox_from_line_xy(self.a, self.b)


def nearest_wall(wall_refs, p_xy, max_dist_ft):
    best = None
    bestd = 1e9
    for wr in wall_refs:
        # quick bbox check
        if not (
            wr.bb[0] - max_dist_ft <= p_xy[0] <= wr.bb[2] + max_dist_ft
            and wr.bb[1] - max_dist_ft <= p_xy[1] <= wr.bb[3] + max_dist_ft
        ):
            continue
        d = dist_point_to_segment_xy(p_xy, wr.a, wr.b)
        if d < bestd:
            bestd = d
            best = wr
    if best and bestd <= max_dist_ft:
        return best, bestd
    return None, None


# ============================================================
# Door detection from arcs (point = arc center projected to wall)
# ============================================================
def arc_center_xy(arc):
    c = arc.Center
    return (c.X, c.Y)


def is_door_arc_candidate(a):
    r = a.Radius
    if r < DOOR_ARC_R_MIN_FT or r > DOOR_ARC_R_MAX_FT:
        return False
    try:
        sweep_deg = abs((a.Length / max(r, 1e-9)) * 180.0 / math.pi)
        if sweep_deg < DOOR_ARC_SWEEP_MIN_DEG or sweep_deg > DOOR_ARC_SWEEP_MAX_DEG:
            return False
    except Exception:
        pass
    return True


def collect_door_points_from_arcs(arcs, wall_refs):
    door_pts = []
    for a in arcs:
        if not is_door_arc_candidate(a):
            continue
        p = arc_center_xy(a)
        wr, d = nearest_wall(wall_refs, p, DOOR_HOST_DIST_FT)
        if not wr:
            wr, d = nearest_wall(wall_refs, p, OPENING_HOST_FALLBACK_DIST_FT)
        if not wr:
            continue
        # project center to finite wall segment (clamped to wall extents)
        proj, _ = closest_point_on_segment_xy(p, wr.a, wr.b)
        door_pts.append((proj, wr.wall))
    return door_pts


def collect_arc_centers_for_doors(arcs):
    pts = []
    for a in arcs:
        if not is_door_arc_candidate(a):
            continue
        pts.append(arc_center_xy(a))
    return pts


# ============================================================
# Window detection from gaps
# ============================================================
def quant_dir(d):
    # quantize direction to bucket axes (treat d and -d same)
    # choose canonical sign
    if d[0] < 0 or (abs(d[0]) < 1e-9 and d[1] < 0):
        d = (-d[0], -d[1])
    q = 0.01
    return (int(round(d[0] / q)), int(round(d[1] / q)))


def axis_key_for_center(seg, include_thickness=True):
    # axis = (direction bucket + offset)
    d = seg.dir
    if d[0] < 0 or (abs(d[0]) < 1e-9 and d[1] < 0):
        d = (-d[0], -d[1])
    db = quant_dir(d)
    # normal
    n = (-d[1], d[0])
    # signed distance of a point from origin along normal
    off = vdot(seg.a, n)
    qoff = int(round(off / OPENING_AXIS_BUCKET_FT))
    if include_thickness:
        return (db[0], db[1], qoff, int(round(seg.thick / THICK_CLUSTER_TOL_FT)))
    return (db[0], db[1], qoff)


def merge_intervals(intervals, tol=ENDPOINT_SNAP_TOL_FT):
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def intervals_gaps(merged_intervals, min_gap, max_gap):
    gaps = []
    for i in range(len(merged_intervals) - 1):
        e0 = merged_intervals[i][1]
        s1 = merged_intervals[i + 1][0]
        g = s1 - e0
        if min_gap <= g <= max_gap:
            gaps.append((e0, s1))
    return gaps


def collect_opening_candidates_from_gaps(center_segs, door_points_xy):
    # door_points_xy: arc-derived candidate points for door-vs-window classification
    # Build interval coverage per axis from center_segs
    axis_map = defaultdict(list)
    ext_xy = centers_xy_extents(center_segs)
    has_arc_hints = len(door_points_xy) > 0
    for s in center_segs:
        o = s.a
        d = s.dir
        t0 = project_scalar_on_axis(s.a, o, d)
        t1 = project_scalar_on_axis(s.b, o, d)
        if t0 > t1:
            t0, t1 = t1, t0
        # Ignore thickness bucket for opening detection; thickness drift in CAD
        # pairing should not block gap detection on the same wall axis.
        axis_map[axis_key_for_center(s, False)].append((o, d, t0, t1, s.z, s.thick))

    door_pts_xy = []
    win_pts_xy = []
    bridge_segs = []

    # For each axis group, merge intervals in a shared coordinate system:
    # We'll pick first segment's origin+dir as reference and re-project all endpoints to it (since collinear).
    for k, items in axis_map.items():
        if len(items) < 2:
            continue
        o0, d0, _, _, z0, _ = items[0]
        group_thick = max([it[5] for it in items])

        intervals = []
        for (o, d, t0, t1, z, thick) in items:
            # re-project endpoints into ref axis
            # endpoints:
            pA = vadd(o, vmul(d, t0))
            pB = vadd(o, vmul(d, t1))
            s0 = project_scalar_on_axis(pA, o0, d0)
            s1 = project_scalar_on_axis(pB, o0, d0)
            if s0 > s1:
                s0, s1 = s1, s0
            intervals.append((s0, s1, thick))

        merged = merge_intervals([(s0, s1) for (s0, s1, _) in intervals])
        # gaps candidates for openings (door or window)
        gaps = intervals_gaps(merged, min(GAP_MIN_DOOR_FT, GAP_MIN_WIN_FT), max(GAP_MAX_DOOR_FT, GAP_MAX_WIN_FT))

        for g0, g1 in gaps:
            mid = (g0 + g1) * 0.5
            p = vadd(o0, vmul(d0, mid))
            a_gap = vadd(o0, vmul(d0, g0))
            b_gap = vadd(o0, vmul(d0, g1))

            # classify by width + proximity to arc-derived door hints
            is_near_door = False
            for dp in door_points_xy:
                if dist_point_to_segment_xy(dp, a_gap, b_gap) <= ARC_NEAR_GAP_FT:
                    is_near_door = True
                    break
            gap_w = abs(g1 - g0)
            is_door_sized_gap = GAP_MIN_DOOR_FT <= gap_w <= GAP_MAX_DOOR_FT
            is_auto_door_gap = GAP_MIN_DOOR_FT <= gap_w <= GAP_DOOR_AUTO_MAX_FT
            is_window_sized_gap = GAP_MIN_WIN_FT <= gap_w <= GAP_MAX_WIN_FT
            is_arc_door_gap = is_near_door and gap_w <= 2200 * MM_TO_FT
            is_outer_gap = is_outer_envelope_gap(p, d0, ext_xy)

            classify_door = False
            classify_window = False

            if is_arc_door_gap or (is_door_sized_gap and is_near_door):
                classify_door = True
            elif is_outer_gap and is_window_sized_gap:
                classify_window = True
            elif is_auto_door_gap:
                classify_door = True
            elif is_window_sized_gap and not has_arc_hints and is_outer_gap:
                classify_window = True

            # Always bridge opening gaps when we later build walls so the wall stays
            # continuous and family placement creates the actual opening.
            if classify_door or classify_window:
                if vlen(vsub(b_gap, a_gap)) > 10 * MM_TO_FT:
                    adj_thicks = []
                    edge_tol = max(ENDPOINT_SNAP_TOL_FT, 20 * MM_TO_FT)
                    for s0, s1, thick in intervals:
                        if abs(s1 - g0) <= edge_tol or abs(s0 - g1) <= edge_tol:
                            adj_thicks.append(thick)
                    bridge_thick = (sum(adj_thicks) / float(len(adj_thicks))) if adj_thicks else group_thick
                    bridge_segs.append(CenterSeg(a_gap, b_gap, z0, bridge_thick))

            if classify_door:
                door_pts_xy.append(p)
            elif classify_window:
                win_pts_xy.append((p, gap_w))
    return door_pts_xy, win_pts_xy, bridge_segs


def host_door_points(door_pts_xy, wall_refs):
    hosted = []
    for p in door_pts_xy:
        wr, dist = nearest_wall(wall_refs, p, DOOR_HOST_DIST_FT)
        if not wr:
            wr, dist = nearest_wall(wall_refs, p, OPENING_HOST_FALLBACK_DIST_FT)
        if wr:
            hosted.append((p, wr.wall))
    return hosted


def merge_hosted_doors(gap_hosted, arc_hosted, merge_tol_ft=1200 * MM_TO_FT):
    out = list(gap_hosted or [])
    for ap, aw in (arc_hosted or []):
        keep = True
        for gp, gw in out:
            same_wall = False
            try:
                same_wall = gw.Id.IntegerValue == aw.Id.IntegerValue
            except Exception:
                same_wall = gw.Id == aw.Id
            if same_wall and vlen(vsub(ap, gp)) <= merge_tol_ft:
                keep = False
                break
        if keep:
            out.append((ap, aw))
    return out


def host_window_points(win_pts_xy, wall_refs):
    hosted = []
    for p, w in win_pts_xy:
        wr, dist = nearest_wall(wall_refs, p, DOOR_HOST_DIST_FT)
        if not wr:
            wr, dist = nearest_wall(wall_refs, p, OPENING_HOST_FALLBACK_DIST_FT)
        if wr:
            hosted.append((p, wr.wall, w))
    return hosted


def filter_windows_overlapping_doors(win_pts_hosted, door_pts_hosted, tol_ft=700 * MM_TO_FT):
    if not win_pts_hosted or not door_pts_hosted:
        return win_pts_hosted

    def _same_wall(a, b):
        try:
            return a.Id.IntegerValue == b.Id.IntegerValue
        except Exception:
            return a.Id == b.Id

    out = []
    removed = 0
    for wp, ww, width_ft in win_pts_hosted:
        keep = True
        for dp, dw in door_pts_hosted:
            if not _same_wall(ww, dw):
                continue
            if vlen(vsub(wp, dp)) <= tol_ft:
                keep = False
                removed += 1
                break
        if keep:
            out.append((wp, ww, width_ft))

    if removed:
        print("Suppressed windows near doors:", removed)
    return out


# ============================================================
# Place Family Instances (Door/Window)
# ============================================================
def get_first_symbol_of_category(bic):
    sym = FilteredElementCollector(doc).OfClass(FamilySymbol).OfCategory(bic).FirstElement()
    return sym


def ensure_symbol_active(sym):
    if sym and not sym.IsActive:
        sym.Activate()
        doc.Regenerate()


def host_point_candidates(p_xy, host_wall, edge_clear_ft):
    try:
        loc = host_wall.Location
        crv = loc.Curve
        p0 = crv.GetEndPoint(0)
        p1 = crv.GetEndPoint(1)
        a = xyz_to_xy(p0)
        b = xyz_to_xy(p1)
        ab = vsub(b, a)
        L = vlen(ab)
        if L < 1e-9:
            return [p_xy]

        d = (ab[0] / L, ab[1] / L)
        t0 = project_scalar_on_axis(p_xy, a, d)
        t0 = max(0.0, min(L, t0))

        clear = min(max(0.0, edge_clear_ft), max(0.0, L * 0.49))
        offsets = [
            0.0,
            150 * MM_TO_FT,
            -150 * MM_TO_FT,
            300 * MM_TO_FT,
            -300 * MM_TO_FT,
            clear,
            -clear,
        ]
        out = []
        seen = set()
        for off in offsets:
            tt = max(0.0, min(L, t0 + off))
            p = vadd(a, vmul(d, tt))
            k = (round(p[0], 6), round(p[1], 6))
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
        return out or [p_xy]
    except Exception:
        return [p_xy]


def place_doors(door_pts_hosted, level):
    sym = get_first_symbol_of_category(BuiltInCategory.OST_Doors)
    if not sym:
        raise Exception("אין FamilySymbol לדלתות בפרויקט.")
    t = Transaction(doc, "Place Doors")
    t.Start()
    ensure_symbol_active(sym)

    placed = 0
    failed = 0
    for (p_xy, host_wall) in door_pts_hosted:
        ok = False
        for cand_xy in host_point_candidates(p_xy, host_wall, 120 * MM_TO_FT):
            try:
                pt = xy_to_xyz(cand_xy, level.Elevation)
                doc.Create.NewFamilyInstance(pt, sym, host_wall, level, Structure.StructuralType.NonStructural)
                placed += 1
                ok = True
                break
            except Exception:
                pass
        if not ok:
            failed += 1

    t.Commit()
    if failed:
        print("Door insert failed:", failed)
    return placed


def place_windows(win_pts_hosted, level):
    sym = get_first_symbol_of_category(BuiltInCategory.OST_Windows)
    if not sym:
        raise Exception("אין FamilySymbol לחלונות בפרויקט.")
    t = Transaction(doc, "Place Windows")
    t.Start()
    ensure_symbol_active(sym)

    placed = 0
    failed = 0
    for (p_xy, host_wall, width_ft) in win_pts_hosted:
        edge_clear = max(250 * MM_TO_FT, min(1200 * MM_TO_FT, 0.5 * width_ft + 50 * MM_TO_FT))
        fi = None
        for cand_xy in host_point_candidates(p_xy, host_wall, edge_clear):
            try:
                pt = xy_to_xyz(cand_xy, level.Elevation)
                fi = doc.Create.NewFamilyInstance(pt, sym, host_wall, level, Structure.StructuralType.NonStructural)
                break
            except Exception:
                fi = None
        if fi is None:
            failed += 1
            continue

        # optional: set sill height if parameter exists
        p_sill = fi.LookupParameter("Sill Height")
        if p_sill and not p_sill.IsReadOnly:
            p_sill.Set(DEFAULT_SILL_FT)

        placed += 1

    t.Commit()
    if failed:
        print("Window insert failed:", failed)
    return placed


# ============================================================
# MAIN
# ============================================================
sel = list(uidoc.Selection.GetElementIds())
if len(sel) != 1:
    raise Exception("בחר ImportInstance אחד (CAD מיובא)")

imp = doc.GetElement(sel[0])
if not isinstance(imp, ImportInstance):
    raise Exception("הבחירה אינה ImportInstance")

raw_lines, raw_arcs = extract_lines_and_arcs(imp)

# segments
segs = [Seg(l) for l in raw_lines if l.Length >= MIN_LEN_FT]

pairs = find_parallel_pairs(segs)
if not pairs:
    raise Exception("לא נמצאו קירות. בדוק יחידות/דיוק/כפילות קווים.")

clusters = cluster_thickness([t for (_, _, t) in pairs], THICK_CLUSTER_TOL_FT)

centers = []
for a, b, t in pairs:
    t_rep = snap_to_cluster(t, clusters)
    cs = build_center_from_pair(a, b, t_rep)
    if cs:
        centers.append(cs)

centers = merge_collinear(centers)
centers = split_at_intersections(centers)
centers = dedup_centers(centers)
opening_source_centers = list(centers)

# Detect opening candidates from raw gap geometry before wall build.
arc_door_hint_xy = collect_arc_centers_for_doors(raw_arcs)
gap_door_pts_xy, gap_win_pts_xy, bridge_segs = collect_opening_candidates_from_gaps(
    opening_source_centers, arc_door_hint_xy
)

wall_centers = opening_source_centers
if BRIDGE_OPENING_GAPS:
    wall_centers = wall_centers + bridge_segs
    wall_centers = merge_collinear(
        wall_centers,
        WALL_GAP_BRIDGE_MAX_FT,
        WALL_GAP_AXIS_TOL_FT,
        WALL_GAP_ANGLE_TOL_DEG,
        False,
    )
    wall_centers = dedup_centers(wall_centers)
    print("Wall gap stitching:", len(opening_source_centers), "->", len(wall_centers), "bridges:", len(bridge_segs))

# level
view = doc.ActiveView
level = getattr(view, "GenLevel", None)
if level is None:
    level = FilteredElementCollector(doc).OfClass(Level).FirstElement()
print("Target level:", _elem_name(level), "elev_m:", round(level.Elevation * 0.3048, 3))

ext = centers_xy_extents(wall_centers)
if ext:
    print(
        "Center extents XY (m):",
        (
            round(ext[0] * 0.3048, 2),
            round(ext[1] * 0.3048, 2),
            round(ext[2] * 0.3048, 2),
            round(ext[3] * 0.3048, 2),
        ),
    )
place_dwg_version_note(view, ext, level)

if PREVIEW_CENTERLINES:
    draw_model_lines(wall_centers, name="Preview Centerlines")
    print("Centerlines:", len(wall_centers))
    print("Thickness clusters (mm):", [round(c * 304.8, 1) for c in clusters])

walls_created = []
if BUILD_WALLS:
    walls_created = create_walls(wall_centers, level, DEFAULT_WALL_HEIGHT_FT)
    print("Walls created:", len(walls_created))
    focus_elements_in_ui(walls_created)

# build wall refs for hosting
wall_refs = [WallRef2D(w) for w in walls_created] if walls_created else []
if not wall_refs:
    # if you already have walls in model and didn't build now, you can collect them:
    # wall_refs = [WallRef2D(w) for w in FilteredElementCollector(doc).OfClass(Wall).ToElements()]
    pass

# -------- OPENINGS --------
door_pts_hosted = []
gap_door_pts_hosted = []
win_pts_hosted = []
if wall_refs:
    gap_door_pts_hosted = host_door_points(gap_door_pts_xy, wall_refs)
    win_pts_hosted = host_window_points(gap_win_pts_xy, wall_refs)

arc_door_pts_hosted = []
if wall_refs:
    arc_door_pts_hosted = collect_door_points_from_arcs(raw_arcs, wall_refs)
door_pts_hosted = merge_hosted_doors(gap_door_pts_hosted, arc_door_pts_hosted)
win_pts_hosted = filter_windows_overlapping_doors(win_pts_hosted, door_pts_hosted)

door_points_xy_only = [p for (p, _) in door_pts_hosted]

if PREVIEW_DOORS and door_points_xy_only:
    draw_points_xy(door_points_xy_only, level.Elevation, name="Preview Door Points")
    print(
        "Door points:",
        len(door_points_xy_only),
        "gap_raw:",
        len(gap_door_pts_xy),
        "gap_hosted:",
        len(gap_door_pts_hosted),
        "arc_hosted:",
        len(arc_door_pts_hosted),
    )

if BUILD_DOORS and door_pts_hosted:
    n = place_doors(door_pts_hosted, level)
    print("Doors placed:", n)

win_points_xy_only = [p for (p, _, _) in win_pts_hosted]

if PREVIEW_WINDOWS and win_points_xy_only:
    draw_points_xy(win_points_xy_only, level.Elevation, name="Preview Window Points")
    print("Window points:", len(win_points_xy_only), "raw:", len(gap_win_pts_xy))

if BUILD_WINDOWS and win_pts_hosted:
    n = place_windows(win_pts_hosted, level)
    print("Windows placed:", n)

print("Done.")
print("Thickness clusters (mm):", [round(c * 304.8, 1) for c in clusters])
