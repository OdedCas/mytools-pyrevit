# -*- coding: utf-8 -*-
__title__ = "C2Rv5"
__doc__ = "CAD to Revit V5 – detect double-wall pairs, build walls/doors/windows/curtain."

from Autodesk.Revit.DB import *
from Autodesk.Revit.DB.Structure import StructuralType
import math
import os

try:
    from collections import defaultdict
except ImportError:
    class defaultdict(dict):
        def __init__(self, factory):
            self._factory = factory
            dict.__init__(self)
        def __missing__(self, key):
            self[key] = self._factory()
            return self[key]

doc = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument

# ============================================================
# PARAMETERS (TUNE)
# ============================================================
MM_TO_FT = 1.0 / 304.8

# Filtering
# Keep short CAD segments (door jambs, small returns) in the pairing stage,
# but still require a larger minimum when creating wall axes/fragments.
MIN_RAW_LEN_FT   = 80  * MM_TO_FT
MIN_AXIS_LEN_FT  = 120 * MM_TO_FT
ANGLE_TOL_DEG    = 1.0

# Pairing wall side lines
OVERLAP_MIN      = 0.55
THICK_MIN_FT     = 5   * MM_TO_FT   # lowered: CAD may use thin symbolic lines (10-20mm apart)
THICK_MAX_FT     = 600 * MM_TO_FT
THICK_CLUSTER_TOL_FT = 8 * MM_TO_FT
DEFAULT_WALL_THICK_FT = 200 * MM_TO_FT  # if detected thickness < 50mm, override to this

# Make walls continuous across openings
MERGE_GAP_MAX_FT = 3000 * MM_TO_FT   # allow merging across openings up to 3m
COLLINEAR_DIST_TOL_FT = 5 * MM_TO_FT
AXIS_INTERVAL_MERGE_TOL_FT = 3 * MM_TO_FT   # avoid smoothing small CAD offsets

# Split at intersections
INTERSECT_TOL_FT      = 5 * MM_TO_FT
OPENING_SPLIT_SKIP_TOL_FT = 20 * MM_TO_FT   # don't cut wall axes through known openings
OPENING_CLIP_TOL_FT       = 2  * MM_TO_FT   # tolerate floating-point drift when transferring openings

# Wall creation
DEFAULT_WALL_HEIGHT_FT = 3000 * MM_TO_FT

# Door arc detection
DOOR_ARC_R_MIN_FT = 450 * MM_TO_FT
DOOR_ARC_R_MAX_FT = 1500 * MM_TO_FT
HOST_SEARCH_DIST_FT = 350 * MM_TO_FT
BRIDGE_JAMB_MAX_LEN_FT = 1800 * MM_TO_FT   # short returns/jambs for recessed-door fallback
BRIDGE_JAMB_PAR_TOL_DEG = 6.0
BRIDGE_GAP_PERP_DOT_MAX = 0.35             # gap vector should be ~perpendicular to jambs
BRIDGE_HOST_EXTEND_FT   = 60 * MM_TO_FT    # extend synthetic host into jambs a bit

# Gaps classification
GAP_MIN_DOOR_FT   = 650 * MM_TO_FT
GAP_MAX_DOOR_FT   = 1400 * MM_TO_FT
GAP_MIN_WIN_FT    = 450 * MM_TO_FT
GAP_MAX_WIN_FT    = 4500 * MM_TO_FT
ARC_NEAR_GAP_FT   = 450 * MM_TO_FT
WINDOW_HOST_MIN_THICK_FT = 60 * MM_TO_FT   # avoid hosting on thin symbolic line-pair "walls"
WINDOW_HOST_THIN_PENALTY_FT = 500 * MM_TO_FT
WINDOW_HOST_SHORT_PENALTY_FT = 200 * MM_TO_FT

# Curtain threshold (gap width)
CURTAIN_THRESHOLD_FT = 2400 * MM_TO_FT  # 2.4m+: treat as curtain wall/glazing

DEFAULT_SILL_FT   = 1000 * MM_TO_FT

# Envelope graph tolerances
NODE_MERGE_TOL_FT = 25 * MM_TO_FT      # tighter node snapping avoids collapsing small corners
MAX_LOOP_NODES    = 220               # safety limit for loop search

# Toggles
PREVIEW_WALL_AXES = False
PREVIEW_EXTERIOR_AXES = True

BUILD_WALLS   = True
BUILD_DOORS   = True
BUILD_WINDOWS = True
BUILD_CURTAIN_WALLS = True

PREVIEW_DOOR_POINTS   = True
PREVIEW_WINDOW_POINTS = True

# ============================================================
# 2D helpers
# ============================================================
def vsub(a,b): return (a[0]-b[0], a[1]-b[1])
def vadd(a,b): return (a[0]+b[0], a[1]+b[1])
def vmul(a,s): return (a[0]*s, a[1]*s)
def vdot(a,b): return a[0]*b[0] + a[1]*b[1]
def vlen(a): return math.sqrt(vdot(a,a))
def vnorm(a):
    L = vlen(a)
    return (a[0]/L, a[1]/L) if L > 1e-12 else (1.0,0.0)

def xyz_to_xy(p): return (p.X, p.Y)
def xy_to_xyz(xy, z): return XYZ(xy[0], xy[1], z)

def angle_parallel(d1, d2, ang_deg):
    cos_tol = math.cos(math.radians(ang_deg))
    return abs(vdot(d1, d2)) >= cos_tol

def bbox_from_line_xy(a0, a1):
    return (min(a0[0],a1[0]), min(a0[1],a1[1]), max(a0[0],a1[0]), max(a0[1],a1[1]))

def bbox_expand(bb, r):
    return (bb[0]-r, bb[1]-r, bb[2]+r, bb[3]+r)

def dist_point_to_infinite_line_xy(p, l0, ldir):
    v = vsub(p, l0)
    t = vdot(v, ldir)
    proj = vadd(l0, vmul(ldir, t))
    dx = p[0]-proj[0]; dy = p[1]-proj[1]
    return math.sqrt(dx*dx + dy*dy)

def project_scalar_on_axis(p, origin, axis_dir):
    return vdot(vsub(p, origin), axis_dir)

def overlap_ratio_1d(a0,a1,b0,b1):
    ov = max(0.0, min(a1,b1) - max(a0,b0))
    denom = max(1e-12, min(a1-a0, b1-b0))
    return ov / denom

def line_intersection_param(a0, a1, b0, b1, tol=1e-12):
    ax, ay = a0; bx, by = a1
    cx, cy = b0; dx, dy = b1
    r = (bx-ax, by-ay)
    s = (dx-cx, dy-cy)
    rxs = r[0]*s[1] - r[1]*s[0]
    q_p = (cx-ax, cy-ay)
    if abs(rxs) < tol:
        return None
    t = (q_p[0]*s[1] - q_p[1]*s[0]) / rxs
    u = (q_p[0]*r[1] - q_p[1]*r[0]) / rxs
    return (t, u)

def cross2(a,b): return a[0]*b[1] - a[1]*b[0]

def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def closest_point_on_segment_xy(p, a, b):
    ab = vsub(b, a)
    ab2 = vdot(ab, ab)
    if ab2 <= 1e-12:
        return a
    t = vdot(vsub(p, a), ab) / ab2
    t = clamp(t, 0.0, 1.0)
    return vadd(a, vmul(ab, t))

def dist_point_to_segment_xy(p, a, b):
    q = closest_point_on_segment_xy(p, a, b)
    return vlen(vsub(p, q))

def segment_param_xy(p, a, b):
    ab = vsub(b, a)
    ab2 = vdot(ab, ab)
    if ab2 <= 1e-12:
        return 0.0
    return vdot(vsub(p, a), ab) / ab2

def scalar_in_intervals(t, intervals, tol=0.0):
    for (a, b) in intervals:
        lo = min(a, b) - tol
        hi = max(a, b) + tol
        if lo <= t <= hi:
            return True
    return False

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
        for ix in range(x0, x1+1):
            for iy in range(y0, y1+1):
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
# Segments and axes
# ============================================================
_SEG_ID = 0
class Seg(object):
    def __init__(self, revit_line):
        global _SEG_ID
        _SEG_ID += 1
        self.id = _SEG_ID
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
        self.a=a_xy; self.b=b_xy; self.z=z; self.thick=thick_ft
        self.v=vsub(self.b,self.a)
        self.len=vlen(self.v)
        self.dir=vnorm(self.v)
        self.bb=bbox_from_line_xy(self.a,self.b)

class WallAxis(object):
    def __init__(self, origin, axis_dir, z, thick_ft, tmin, tmax):
        self.o=origin; self.d=axis_dir; self.z=z; self.thick=thick_ft
        self.tmin=tmin; self.tmax=tmax
        self.openings=[]          # (g0,g1) axis scalars
        self.is_exterior=False

    @property
    def a(self): return vadd(self.o, vmul(self.d, self.tmin))
    @property
    def b(self): return vadd(self.o, vmul(self.d, self.tmax))
    @property
    def bb(self): return bbox_from_line_xy(self.a, self.b)

# ============================================================
# Extract geometry
# ============================================================
def extract_lines_and_arcs(import_inst):
    opt = Options()
    opt.DetailLevel = ViewDetailLevel.Fine
    geo = import_inst.get_Geometry(opt)
    lines=[]; arcs=[]

    def add_polyline_segments(pl):
        try:
            pts = list(pl.GetCoordinates())
        except Exception:
            return
        if not pts or len(pts) < 2:
            return
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            try:
                if p0.DistanceTo(p1) > 1e-9:
                    lines.append(Line.CreateBound(p0, p1))
            except Exception:
                continue

    def walk(g):
        for obj in g:
            if isinstance(obj, GeometryInstance):
                walk(obj.GetInstanceGeometry())
            elif isinstance(obj, PolyLine):
                add_polyline_segments(obj)
            elif isinstance(obj, Line):
                lines.append(obj)
            elif isinstance(obj, Arc):
                arcs.append(obj)
    walk(geo)
    return lines, arcs

# ============================================================
# Pairing wall side lines -> candidates
# ============================================================
def find_parallel_pairs(segs):
    CELL = 2.0 / 0.3048
    grid = SpatialGrid(CELL)
    for s in segs:
        grid.insert(s)

    pairs=[]
    for a in segs:
        neigh = grid.query(bbox_expand(a.bb, THICK_MAX_FT))
        o=a.a; d=a.dir
        a0=project_scalar_on_axis(a.a,o,d); a1=project_scalar_on_axis(a.b,o,d)
        if a0>a1: a0,a1=a1,a0

        for b in neigh:
            if b.id <= a.id:
                continue
            if not angle_parallel(a.dir,b.dir,ANGLE_TOL_DEG):
                continue

            b0=project_scalar_on_axis(b.a,o,d); b1=project_scalar_on_axis(b.b,o,d)
            if b0>b1: b0,b1=b1,b0

            if overlap_ratio_1d(a0,a1,b0,b1) < OVERLAP_MIN:
                continue

            t=dist_point_to_infinite_line_xy(b.a,a.a,a.dir)
            if t<THICK_MIN_FT or t>THICK_MAX_FT:
                continue

            pairs.append((a,b,t))
    return pairs

# ============================================================
# Thickness clustering
# ============================================================
def cluster_thickness(values, tol_ft):
    clusters=[]
    for v in sorted(values):
        placed=False
        for c in clusters:
            if abs(c[0]-v) <= tol_ft:
                c.append(v); placed=True; break
        if not placed:
            clusters.append([v])
    return [sum(c)/len(c) for c in clusters]

def snap_to_cluster(v, reps):
    best=reps[0]; bestd=abs(v-best)
    for r in reps[1:]:
        d=abs(v-r)
        if d<bestd: best=r; bestd=d
    return best

# ============================================================
# Build raw center segments from pairs (overlap span)
# ============================================================
def build_center_from_pair(a,b,thick):
    o=a.a; d=a.dir

    a0=project_scalar_on_axis(a.a,o,d); a1=project_scalar_on_axis(a.b,o,d)
    b0=project_scalar_on_axis(b.a,o,d); b1=project_scalar_on_axis(b.b,o,d)
    if a0>a1: a0,a1=a1,a0
    if b0>b1: b0,b1=b1,b0

    s0=max(a0,b0); s1=min(a1,b1)
    if (s1-s0) <= MIN_AXIS_LEN_FT:
        return None

    pa0=vadd(o, vmul(d,s0))
    pa1=vadd(o, vmul(d,s1))

    vb=vsub(b.a, pa0)
    along=vmul(d, vdot(vb,d))
    n=vsub(vb, along)
    nlen=vlen(n)
    if nlen < 1e-9:
        n=(-d[1], d[0]); nlen=1.0
    n=(n[0]/nlen, n[1]/nlen)

    c0=vadd(pa0, vmul(n, thick*0.5))
    c1=vadd(pa1, vmul(n, thick*0.5))
    return CenterSeg(c0,c1,a.z,thick)

# ============================================================
# Build continuous WallAxes (merge across gaps) + openings list
# ============================================================
def quant_dir(d):
    if d[0] < 0 or (abs(d[0]) < 1e-9 and d[1] < 0):
        d=(-d[0], -d[1])
    q=0.01
    return (int(round(d[0]/q)), int(round(d[1]/q)))

def axis_bucket_key(seg):
    d=seg.dir
    db=quant_dir(d)
    n=(-d[1], d[0])
    off=vdot(seg.a, n)
    qoff=int(round(off / (10*MM_TO_FT))) # 10mm bucket keeps close parallel axes distinct
    tkey=int(round(seg.thick / THICK_CLUSTER_TOL_FT))
    return (db[0], db[1], qoff, tkey, round(seg.z,6))

def merge_intervals(intervals, tol):
    if not intervals:
        return []
    intervals=sorted(intervals)
    out=[list(intervals[0])]
    for s,e in intervals[1:]:
        if s <= out[-1][1] + tol:
            out[-1][1]=max(out[-1][1], e)
        else:
            out.append([s,e])
    return [(a,b) for a,b in out]

def build_wall_axes(center_segs):
    buckets=defaultdict(list)
    for s in center_segs:
        buckets[axis_bucket_key(s)].append(s)

    axes=[]
    for k, segs in buckets.items():
        ref=segs[0]
        o0=ref.a; d0=ref.dir; z0=ref.z; thick=ref.thick

        intervals=[]
        for s in segs:
            s0=project_scalar_on_axis(s.a,o0,d0)
            s1=project_scalar_on_axis(s.b,o0,d0)
            if s0>s1: s0,s1=s1,s0
            intervals.append((s0,s1))

        merged=merge_intervals(intervals, AXIS_INTERVAL_MERGE_TOL_FT)
        if not merged:
            continue

        run_start, run_end = merged[0]
        run_open=[]
        for i in range(len(merged)-1):
            a_end=merged[i][1]
            b_start=merged[i+1][0]
            gap=b_start-a_end

            if gap <= MERGE_GAP_MAX_FT:
                if gap >= GAP_MIN_WIN_FT:
                    run_open.append((a_end,b_start))
                run_end=merged[i+1][1]
            else:
                wa=WallAxis(o0,d0,z0,thick,run_start,run_end)
                wa.openings=list(run_open)
                axes.append(wa)
                run_start, run_end = merged[i+1]
                run_open=[]

        wa=WallAxis(o0,d0,z0,thick,run_start,run_end)
        wa.openings=list(run_open)
        axes.append(wa)

    return axes

def split_axes_at_intersections(axes):
    grid=SpatialGrid(2.0/0.3048)
    for s in axes:
        grid.insert(s)

    out=[]
    for s in axes:
        cuts=[s.tmin, s.tmax]
        neigh=grid.query(bbox_expand(s.bb, INTERSECT_TOL_FT))

        for o in neigh:
            if o is s:
                continue
            if angle_parallel(s.d, o.d, 1.0):
                continue
            inter=line_intersection_param(s.a,s.b,o.a,o.b)
            if not inter:
                continue
            ta, tb = inter
            if -0.01 <= ta <= 1.01 and -0.01 <= tb <= 1.01:
                p=vadd(s.a, vmul(vsub(s.b,s.a), max(0.0,min(1.0,ta))))
                t=project_scalar_on_axis(p, s.o, s.d)
                if s.tmin+1e-6 < t < s.tmax-1e-6:
                    # Preserve opening spans (doors/windows) on the host wall axis.
                    # Intersections from jamb/return axes can otherwise split the axis
                    # and drop the opening metadata due to rounding.
                    if s.openings and scalar_in_intervals(t, s.openings, OPENING_SPLIT_SKIP_TOL_FT):
                        continue
                    cuts.append(t)

        cuts=sorted(set([round(c,6) for c in cuts]))
        for i in range(len(cuts)-1):
            t0,t1=cuts[i],cuts[i+1]
            if (t1-t0) < MIN_AXIS_LEN_FT:
                continue
            wa=WallAxis(s.o,s.d,s.z,s.thick,t0,t1)
            # Clip overlapping openings onto each split segment to survive
            # small floating-point drift in intersection cuts.
            clipped=[]
            for (a,b) in s.openings:
                oa=max(a, t0 - OPENING_CLIP_TOL_FT)
                ob=min(b, t1 + OPENING_CLIP_TOL_FT)
                if (ob - oa) > (1 * MM_TO_FT):
                    oa=max(oa, t0)
                    ob=min(ob, t1)
                    if (ob - oa) > (1 * MM_TO_FT):
                        clipped.append((oa,ob))
            wa.openings=merge_intervals(clipped, OPENING_CLIP_TOL_FT)
            out.append(wa)
    return out

# ============================================================
# V5 Envelope: Graph -> loops -> choose max area loop
# ============================================================
def merge_node(nodes, p, tol):
    for i, n in enumerate(nodes):
        if vlen(vsub(n, p)) <= tol:
            return i
    nodes.append(p)
    return len(nodes)-1

def polygon_area_xy(pts):
    area=0.0
    for i in range(len(pts)):
        x1,y1=pts[i]
        x2,y2=pts[(i+1)%len(pts)]
        area += (x1*y2 - x2*y1)
    return abs(area)*0.5

def classify_exterior_axes_v5(axes):
    nodes=[]
    edges=[]  # (n0,n1,axis)
    adj=defaultdict(list)

    for ax in axes:
        n0 = merge_node(nodes, ax.a, NODE_MERGE_TOL_FT)
        n1 = merge_node(nodes, ax.b, NODE_MERGE_TOL_FT)
        edges.append((n0,n1,ax))
        adj[n0].append((n1,ax))
        adj[n1].append((n0,ax))

    if len(nodes) < 4:
        for ax in axes:
            ax.is_exterior = True
        return

    # IronPython 2.7: no nonlocal — use mutable container
    best = [0.0, None]  # [best_area, best_loop_axes]

    def dfs(start, current, path_nodes, path_axes, visited_set):
        if len(path_nodes) > MAX_LOOP_NODES:
            return

        for (nxt, ax) in adj[current]:
            if nxt == start and len(path_nodes) >= 4:
                pts = [nodes[i] for i in path_nodes]
                area = polygon_area_xy(pts)
                if area > best[0]:
                    best[0] = area
                    best[1] = list(path_axes)
                continue

            if nxt in visited_set:
                continue

            if nxt < start:
                continue

            visited_set.add(nxt)
            path_nodes.append(nxt)
            path_axes.append(ax)
            dfs(start, nxt, path_nodes, path_axes, visited_set)
            path_axes.pop()
            path_nodes.pop()
            visited_set.remove(nxt)

    for s in range(len(nodes)):
        visited=set([s])
        dfs(s, s, [s], [], visited)

    best_loop_axes = best[1]

    if best_loop_axes is None:
        bb=[1e9,1e9,-1e9,-1e9]
        for ax in axes:
            bb[0]=min(bb[0], ax.bb[0]); bb[1]=min(bb[1], ax.bb[1])
            bb[2]=max(bb[2], ax.bb[2]); bb[3]=max(bb[3], ax.bb[3])
        for ax in axes:
            touch = (abs(ax.bb[0]-bb[0])<300*MM_TO_FT or abs(ax.bb[2]-bb[2])<300*MM_TO_FT or
                     abs(ax.bb[1]-bb[1])<300*MM_TO_FT or abs(ax.bb[3]-bb[3])<300*MM_TO_FT)
            ax.is_exterior = bool(touch)
        return

    loop_set = set(id(a) for a in best_loop_axes)
    for ax in axes:
        ax.is_exterior = (id(ax) in loop_set)

# ============================================================
# Preview drawing
# ============================================================
def ensure_sketch_plane(z):
    plane = Plane.CreateByNormalAndOrigin(XYZ.BasisZ, XYZ(0,0,z))
    return SketchPlane.Create(doc, plane)

def draw_axes(axes, name):
    if not axes:
        return
    byz=defaultdict(list)
    for w in axes:
        byz[round(w.z,6)].append(w)

    t=Transaction(doc, name)
    t.Start()
    try:
        for z, items in byz.items():
            sp=ensure_sketch_plane(items[0].z)
            for w in items:
                ln=Line.CreateBound(xy_to_xyz(w.a,w.z), xy_to_xyz(w.b,w.z))
                doc.Create.NewModelCurve(ln, sp)
        t.Commit()
    except Exception as e:
        t.RollBack()
        print("draw_axes '%s' failed: %s" % (name, str(e)))

def draw_points(points_xy, z, name):
    if not points_xy:
        return
    t=Transaction(doc, name)
    t.Start()
    try:
        sp=ensure_sketch_plane(z)
        s=150*MM_TO_FT
        for p in points_xy:
            p1=(p[0]-s,p[1]); p2=(p[0]+s,p[1])
            p3=(p[0],p[1]-s); p4=(p[0],p[1]+s)
            doc.Create.NewModelCurve(Line.CreateBound(xy_to_xyz(p1,z), xy_to_xyz(p2,z)), sp)
            doc.Create.NewModelCurve(Line.CreateBound(xy_to_xyz(p3,z), xy_to_xyz(p4,z)), sp)
        t.Commit()
    except Exception as e:
        t.RollBack()
        print("draw_points '%s' failed: %s" % (name, str(e)))

# ============================================================
# WallTypes and Walls
# ============================================================
def _find_basic_walltype():
    """Find a Basic (layered) WallType suitable for duplication."""
    for wt in FilteredElementCollector(doc).OfClass(WallType):
        try:
            if wt.Kind == WallKind.Basic:
                return wt
        except Exception:
            continue
    # fallback: just use first
    return FilteredElementCollector(doc).OfClass(WallType).FirstElement()

def _safe_elem_name(elem):
    """IronPython-safe element name getter (WallType.Name can raise `Name`)."""
    try:
        return elem.Name
    except Exception:
        pass
    try:
        return Element.Name.GetValue(elem)
    except Exception:
        pass
    try:
        p = elem.get_Parameter(BuiltInParameter.SYMBOL_NAME_PARAM)
        if p:
            return p.AsString()
    except Exception:
        pass
    return None

def get_or_create_walltype(doc, thick_ft, prefix="CAD_Auto"):
    mm = thick_ft * 304.8
    name = "%s_%dmm" % (prefix, int(round(mm)))

    for wt in FilteredElementCollector(doc).OfClass(WallType):
        if _safe_elem_name(wt) == name:
            return wt

    base = _find_basic_walltype()
    new_type = base.Duplicate(name)
    try:
        cs = new_type.GetCompoundStructure()
        if cs and cs.LayerCount > 0:
            cs.SetLayerWidth(0, thick_ft)
            for i in range(1, cs.LayerCount):
                cs.SetLayerWidth(i, 0.0)
            new_type.SetCompoundStructure(cs)
    except Exception as e:
        print("Warning: could not set wall thickness: %s" % str(e))
    return new_type

def create_walls(axes, level, height_ft):
    wt_cache={}
    out=[]
    t=Transaction(doc, "V5 Create Walls")
    t.Start()
    try:
        for w in axes:
            key=int(round(w.thick/THICK_CLUSTER_TOL_FT))
            if key not in wt_cache:
                wt_cache[key]=get_or_create_walltype(doc, w.thick, prefix="CAD_Auto")
            wt=wt_cache[key]
            ln=Line.CreateBound(xy_to_xyz(w.a, level.Elevation), xy_to_xyz(w.b, level.Elevation))
            wall=Wall.Create(doc, ln, wt.Id, level.Id, height_ft, 0.0, False, False)
            out.append((w, wall))
        t.Commit()
    except Exception as e:
        t.RollBack()
        raise Exception("create_walls failed: %s" % str(e))
    return out

# ============================================================
# Hosting search for doors/windows
# ============================================================
class WallRef2D(object):
    def __init__(self, axis, wall):
        self.axis=axis
        self.wall=wall
        try:
            self.wall_id = wall.Id.IntegerValue
        except Exception:
            self.wall_id = str(wall.Id)
        crv=wall.Location.Curve
        p0=crv.GetEndPoint(0); p1=crv.GetEndPoint(1)
        self.a=xyz_to_xy(p0); self.b=xyz_to_xy(p1)
        self.dir=vnorm(vsub(self.b,self.a))
        self.bb=bbox_from_line_xy(self.a,self.b)

def snap_host_point_xy(wr, p_xy):
    return closest_point_on_segment_xy(p_xy, wr.a, wr.b)

def has_near_point(points_xy, p_xy, tol_ft):
    for q in points_xy:
        if vlen(vsub(q, p_xy)) <= tol_ft:
            return True
    return False

def has_near_hosted_item(items, wr, p_xy, tol_ft):
    for it in items:
        wr2 = it.get("wr", None)
        if wr2 is None:
            continue
        if getattr(wr2, "wall_id", None) != getattr(wr, "wall_id", None):
            continue
        if vlen(vsub(it.get("pt"), p_xy)) <= tol_ft:
            return True
    return False

def nearest_wall(wall_refs, p_xy, max_dist_ft, exterior_only=None):
    best=None; bestd=1e9
    for wr in wall_refs:
        if exterior_only is True and not wr.axis.is_exterior:
            continue
        if exterior_only is False and wr.axis.is_exterior:
            continue
        if not (wr.bb[0]-max_dist_ft <= p_xy[0] <= wr.bb[2]+max_dist_ft and
                wr.bb[1]-max_dist_ft <= p_xy[1] <= wr.bb[3]+max_dist_ft):
            continue
        d=dist_point_to_infinite_line_xy(p_xy, wr.a, wr.dir)
        if d < bestd:
            bestd=d; best=wr
    if best and bestd <= max_dist_ft:
        return best
    return None

def best_host_wall_for_opening_point(wall_refs, p_xy, axis_dir=None, preferred_wr=None,
                                     exterior_only=None, max_dist_ft=None,
                                     min_thick_ft=None, target_opening_gap_ft=None):
    """
    Pick the best wall host for an opening point, preferring a wall segment that
    actually contains the point (not one that only matches by infinite-line distance).
    This avoids endpoint clamping when walls were split into short segments.
    """
    if max_dist_ft is None:
        max_dist_ft = HOST_SEARCH_DIST_FT

    best = None
    best_score = 1e18

    for wr in wall_refs:
        if exterior_only is True and not wr.axis.is_exterior:
            continue
        if exterior_only is False and wr.axis.is_exterior:
            continue
        if axis_dir is not None and not angle_parallel(axis_dir, wr.dir, 5.0):
            continue
        if not (wr.bb[0]-max_dist_ft <= p_xy[0] <= wr.bb[2]+max_dist_ft and
                wr.bb[1]-max_dist_ft <= p_xy[1] <= wr.bb[3]+max_dist_ft):
            continue

        q = closest_point_on_segment_xy(p_xy, wr.a, wr.b)
        d = vlen(vsub(p_xy, q))
        if d > max_dist_ft:
            continue

        t = segment_param_xy(p_xy, wr.a, wr.b)
        seg_len = max(vlen(vsub(wr.b, wr.a)), 1e-9)
        wr_thick = getattr(getattr(wr, "axis", None), "thick", 0.0)
        if t < 0.0:
            overrun = (-t) * seg_len
        elif t > 1.0:
            overrun = (t - 1.0) * seg_len
        else:
            overrun = 0.0

        # Strongly prefer a segment that actually contains the opening point.
        score = d + (overrun * 8.0)

        if preferred_wr is not None and getattr(wr, "wall_id", None) != getattr(preferred_wr, "wall_id", None):
            score += 5 * MM_TO_FT

        # Window/cw placement: penalize thin symbolic segments and very short hosts.
        if min_thick_ft is not None and wr_thick < min_thick_ft:
            score += WINDOW_HOST_THIN_PENALTY_FT
        if target_opening_gap_ft is not None and seg_len < (target_opening_gap_ft + 150 * MM_TO_FT):
            score += WINDOW_HOST_SHORT_PENALTY_FT

        if score < best_score:
            best_score = score
            best = wr

    return best

def nearest_wall_for_door_arc(wall_refs, arc, exterior_only=None):
    # Primary probe: chord midpoint (normal case)
    e0 = xyz_to_xy(arc.GetEndPoint(0))
    e1 = xyz_to_xy(arc.GetEndPoint(1))
    chord_mid = ((e0[0]+e1[0])/2.0, (e0[1]+e1[1])/2.0)
    wr = nearest_wall(wall_refs, chord_mid, HOST_SEARCH_DIST_FT, exterior_only=exterior_only)
    if wr:
        return wr

    # Fallback probes: hinge/arc center and endpoints. This recovers doors in
    # recessed openings where the midpoint falls inside a gap between split hosts.
    probes = [xyz_to_xy(arc.Center), e0, e1]
    probe_dists = [HOST_SEARCH_DIST_FT, max(HOST_SEARCH_DIST_FT, 650 * MM_TO_FT)]

    for p in probes:
        for dmax in probe_dists:
            wr = nearest_wall(wall_refs, p, dmax, exterior_only=exterior_only)
            if wr:
                return wr
    return None

def _quant_pt_key(p, mm=25.0):
    q = mm * MM_TO_FT
    return (int(round(p[0] / q)), int(round(p[1] / q)))

def door_data_from_bridge_jamb_pairs(wall_refs, existing_door_points_xy, cad_segs):
    """
    Fallback for recessed/internal openings where no wall axis exists across the mouth
    (e.g. U-shaped wall returns with a door swing drawn between the jambs).
    """
    out = []
    local_pts = []

    n = len(wall_refs)
    for i in range(n):
        wr1 = wall_refs[i]
        if wr1.axis.is_exterior:
            continue
        if vlen(vsub(wr1.a, wr1.b)) > BRIDGE_JAMB_MAX_LEN_FT:
            continue

        for j in range(i + 1, n):
            wr2 = wall_refs[j]
            if wr2.axis.is_exterior:
                continue
            if vlen(vsub(wr2.a, wr2.b)) > BRIDGE_JAMB_MAX_LEN_FT:
                continue
            if not angle_parallel(wr1.dir, wr2.dir, BRIDGE_JAMB_PAR_TOL_DEG):
                continue

            # Similar wall thicknesses reduce false matches.
            if abs(getattr(wr1.axis, "thick", DEFAULT_WALL_THICK_FT) -
                   getattr(wr2.axis, "thick", DEFAULT_WALL_THICK_FT)) > (120 * MM_TO_FT):
                continue

            for p1 in (wr1.a, wr1.b):
                for p2 in (wr2.a, wr2.b):
                    gap_vec = vsub(p2, p1)
                    gap = vlen(gap_vec)
                    if gap < GAP_MIN_DOOR_FT or gap > GAP_MAX_DOOR_FT:
                        continue
                    gdir = vnorm(gap_vec)

                    # We want the vector between jamb ends, not along the jambs.
                    if abs(vdot(gdir, wr1.dir)) > BRIDGE_GAP_PERP_DOT_MAX:
                        continue

                    mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)

                    if has_near_point(existing_door_points_xy, mid, ARC_NEAR_GAP_FT):
                        continue
                    if has_near_point(local_pts, mid, 250 * MM_TO_FT):
                        continue

                    # Require CAD evidence of a door leaf near the opening.
                    has_leaf = (opening_has_door_leaf_evidence(cad_segs, wr1, mid, gap) or
                                opening_has_door_leaf_evidence(cad_segs, wr2, mid, gap))
                    if not has_leaf:
                        continue

                    # Normalize endpoint order for stable dedupe.
                    k1 = _quant_pt_key(p1)
                    k2 = _quant_pt_key(p2)
                    if k2 < k1:
                        p1, p2 = p2, p1
                        gap_vec = vsub(p2, p1)
                        gdir = vnorm(gap_vec)

                    ext = BRIDGE_HOST_EXTEND_FT
                    a_host = vadd(p1, vmul(gdir, -ext))
                    b_host = vadd(p2, vmul(gdir,  ext))
                    thick = 0.5 * (getattr(wr1.axis, "thick", DEFAULT_WALL_THICK_FT) +
                                   getattr(wr2.axis, "thick", DEFAULT_WALL_THICK_FT))

                    out.append({
                        "pt": mid,
                        "gap": gap,
                        "side": 0.0,
                        "ccw": False,
                        "source": "bridge_jambs",
                        "bridge_host": {
                            "a": a_host,
                            "b": b_host,
                            "thick": thick
                        }
                    })
                    local_pts.append(mid)
    return out

# ============================================================
# Door data from arcs
# ============================================================
def door_data_from_arcs(arcs, wall_refs):
    out=[]
    for a in arcs:
        r=a.Radius
        if r < DOOR_ARC_R_MIN_FT or r > DOOR_ARC_R_MAX_FT:
            continue

        e0=xyz_to_xy(a.GetEndPoint(0))
        e1=xyz_to_xy(a.GetEndPoint(1))
        chord_mid=((e0[0]+e1[0])/2.0, (e0[1]+e1[1])/2.0)

        wr = nearest_wall_for_door_arc(wall_refs, a, exterior_only=None)
        if not wr:
            continue

        o=wr.a; d=wr.dir
        t=project_scalar_on_axis(chord_mid, o, d)
        proj=vadd(o, vmul(d,t))
        proj = snap_host_point_xy(wr, proj)

        c=xyz_to_xy(a.Center)
        n=(-d[1], d[0])
        side = vdot(vsub(c, proj), n)

        v0 = vsub(e0, c)
        v1 = vsub(e1, c)
        z = cross2(v0, v1)
        ccw = (z > 0)

        if has_near_hosted_item(out, wr, proj, 200 * MM_TO_FT):
            continue

        out.append({
            "pt": proj,
            "wall": wr.wall,
            "wr": wr,
            "side": side,
            "ccw": ccw,
            "source": "arc"
        })
    return out

def opening_has_door_leaf_evidence(cad_segs, wr, p_xy, gap_ft):
    # Door leafs are usually a medium-length line crossing/angling from the wall.
    # Windows are more often represented with short parallel linework.
    radius = max(ARC_NEAR_GAP_FT, gap_ft * 0.9)
    min_leaf = max(300 * MM_TO_FT, gap_ft * 0.35)
    max_leaf = min(2500 * MM_TO_FT, gap_ft * 1.8)
    for s in cad_segs:
        if s.len < min_leaf or s.len > max_leaf:
            continue
        if not (min(s.bb[0], s.bb[2]) - radius <= p_xy[0] <= max(s.bb[0], s.bb[2]) + radius and
                min(s.bb[1], s.bb[3]) - radius <= p_xy[1] <= max(s.bb[1], s.bb[3]) + radius):
            continue
        if dist_point_to_segment_xy(p_xy, s.a, s.b) > radius:
            continue
        if angle_parallel(s.dir, wr.dir, 12.0):
            continue
        da = dist_point_to_infinite_line_xy(s.a, wr.a, wr.dir)
        db = dist_point_to_infinite_line_xy(s.b, wr.a, wr.dir)
        if min(da, db) > HOST_SEARCH_DIST_FT * 1.5:
            continue
        return True
    return False

def door_data_from_opening_gaps(wall_refs, existing_door_points_xy, cad_segs):
    out = []
    for wr in wall_refs:
        ax = wr.axis
        for (g0, g1) in ax.openings:
            gap = g1 - g0
            if gap < GAP_MIN_DOOR_FT or gap > GAP_MAX_DOOR_FT:
                continue
            mid = (g0 + g1) * 0.5
            p = vadd(ax.o, vmul(ax.d, mid))
            p = snap_host_point_xy(wr, p)

            if has_near_point(existing_door_points_xy, p, ARC_NEAR_GAP_FT):
                continue
            if has_near_hosted_item(out, wr, p, 200 * MM_TO_FT):
                continue

            has_leaf = opening_has_door_leaf_evidence(cad_segs, wr, p, gap)

            # Exterior openings need CAD evidence (door leaf/polyline fragment)
            # to avoid converting narrow windows into doors.
            if ax.is_exterior and not has_leaf:
                continue

            # Interior openings are often doors; allow common-width gaps even if
            # the arc was missed, but avoid very wide gaps without evidence.
            if (not ax.is_exterior) and (not has_leaf) and gap > (1100 * MM_TO_FT):
                continue

            out.append({
                "pt": p,
                "wall": wr.wall,
                "wr": wr,
                "side": 0.0,
                "ccw": False,
                "gap": gap,
                "source": "gap"
            })
    return out

# ============================================================
# Windows from openings on EXTERIOR axes only + curtain walls
# ============================================================
def openings_from_exterior_axes(wall_refs, door_points_xy):
    wins=[]
    curtains=[]
    win_pts_all = []
    curtain_pts_all = []
    for wr in wall_refs:
        ax = wr.axis
        if not ax.is_exterior:
            continue
        for (g0,g1) in ax.openings:
            gap = g1-g0
            if gap < GAP_MIN_WIN_FT:
                continue
            mid=(g0+g1)*0.5
            p=vadd(ax.o, vmul(ax.d, mid))
            host_wr = best_host_wall_for_opening_point(
                wall_refs, p, axis_dir=ax.d, preferred_wr=wr, exterior_only=True,
                max_dist_ft=max(HOST_SEARCH_DIST_FT, 300 * MM_TO_FT),
                min_thick_ft=WINDOW_HOST_MIN_THICK_FT,
                target_opening_gap_ft=gap
            ) or wr

            # If we still ended up on a thin symbolic host, make one more pass without
            # preferred-wall bias and with a wider search to catch the real exterior wall.
            if getattr(host_wr.axis, "thick", 0.0) < WINDOW_HOST_MIN_THICK_FT:
                host_wr2 = best_host_wall_for_opening_point(
                    wall_refs, p, axis_dir=ax.d, preferred_wr=None, exterior_only=True,
                    max_dist_ft=max(HOST_SEARCH_DIST_FT, 600 * MM_TO_FT),
                    min_thick_ft=WINDOW_HOST_MIN_THICK_FT,
                    target_opening_gap_ft=gap
                )
                if host_wr2 is not None:
                    host_wr = host_wr2
            p = snap_host_point_xy(host_wr, p)

            near=False
            for dp in door_points_xy:
                if vlen(vsub(dp,p)) <= ARC_NEAR_GAP_FT:
                    near=True; break
            if near:
                continue

            if gap >= CURTAIN_THRESHOLD_FT:
                if has_near_point(curtain_pts_all, p, 120 * MM_TO_FT):
                    continue
                if not has_near_hosted_item(curtains, host_wr, p, 200 * MM_TO_FT):
                    curtains.append({"pt": p, "wall": host_wr.wall, "wr": host_wr, "gap": gap})
                    curtain_pts_all.append(p)
            else:
                if gap <= GAP_MAX_WIN_FT:
                    if has_near_point(win_pts_all, p, 120 * MM_TO_FT):
                        continue
                    if not has_near_hosted_item(wins, host_wr, p, 200 * MM_TO_FT):
                        wins.append({"pt": p, "wall": host_wr.wall, "wr": host_wr, "gap": gap})
                        win_pts_all.append(p)
    return wins, curtains

# ============================================================
# Symbol selection by width
# ============================================================
def get_symbols(bic):
    return list(FilteredElementCollector(doc).OfClass(FamilySymbol).OfCategory(bic))

def get_symbol_by_width(symbols, target_ft):
    if not symbols:
        return None
    param_names = ["Width", "Rough Width", "Nominal Width"]
    best = symbols[0]
    bestd = 1e9
    for s in symbols:
        w = None
        for pn in param_names:
            p = s.LookupParameter(pn)
            if p and p.StorageType == StorageType.Double:
                w = p.AsDouble()
                break
        if w is None:
            continue
        d = abs(w - target_ft)
        if d < bestd:
            bestd = d
            best = s
    return best

def symbol_is_opening_only(sym):
    parts = []
    try:
        parts.append(safe_strtype(sym.FamilyName))
    except Exception:
        pass
    try:
        parts.append(safe_strtype(_safe_elem_name(sym)))
    except Exception:
        pass
    txt = " ".join(parts).lower()
    return ("opening" in txt) or ("void" in txt) or (u"פתח" in txt)

def choose_door_symbol(symbols, target_ft):
    real_door_symbols = [s for s in symbols if not symbol_is_opening_only(s)]
    if real_door_symbols:
        return get_symbol_by_width(real_door_symbols, target_ft) or real_door_symbols[0]
    return get_symbol_by_width(symbols, target_ft) or (symbols[0] if symbols else None)

def ensure_active(sym):
    if sym and not sym.IsActive:
        sym.Activate()
        doc.Regenerate()

# ============================================================
# Place Doors
# ============================================================
def place_doors(door_data, level):
    symbols = get_symbols(BuiltInCategory.OST_Doors)
    if not symbols:
        raise Exception("No Door FamilySymbol in project.")
    sym_default = symbols[0]
    bridge_walltype_cache = {}
    bridge_host_cache = {}

    t=Transaction(doc, "V5 Place Doors")
    t.Start()
    try:
        placed=0
        for d in door_data:
            try:
                target_w = d.get("gap", 900*MM_TO_FT)
                sym = choose_door_symbol(symbols, target_w) or sym_default
                ensure_active(sym)

                wr = d.get("wr", None)
                host_wall = d.get("wall", None)

                if (host_wall is None) and d.get("bridge_host"):
                    bh = d["bridge_host"]
                    a = bh["a"]; b = bh["b"]
                    thick = bh.get("thick", DEFAULT_WALL_THICK_FT)
                    kpts = sorted([_quant_pt_key(a, 10.0), _quant_pt_key(b, 10.0)])
                    k = (kpts[0], kpts[1], int(round(thick / THICK_CLUSTER_TOL_FT)))

                    if k in bridge_host_cache:
                        host_wall, wr = bridge_host_cache[k]
                    else:
                        wt_key = int(round(thick / THICK_CLUSTER_TOL_FT))
                        if wt_key not in bridge_walltype_cache:
                            bridge_walltype_cache[wt_key] = get_or_create_walltype(doc, thick, prefix="CAD_Auto")
                        wt = bridge_walltype_cache[wt_key]

                        ln = Line.CreateBound(xy_to_xyz(a, level.Elevation), xy_to_xyz(b, level.Elevation))
                        host_wall = Wall.Create(doc, ln, wt.Id, level.Id, DEFAULT_WALL_HEIGHT_FT, 0.0, False, False)

                        axis_dir = vnorm(vsub(b, a))
                        axis_len = vlen(vsub(b, a))
                        fake_axis = WallAxis(a, axis_dir, level.Elevation, thick, 0.0, axis_len)
                        fake_axis.is_exterior = False
                        wr = WallRef2D(fake_axis, host_wall)
                        bridge_host_cache[k] = (host_wall, wr)

                    d["wall"] = host_wall
                    d["wr"] = wr

                if host_wall is None:
                    continue

                pt_xy = d["pt"]
                if wr is not None:
                    pt_xy = snap_host_point_xy(wr, pt_xy)
                pt = xy_to_xyz(pt_xy, level.Elevation)
                fi = doc.Create.NewFamilyInstance(pt, sym, host_wall, level,
                                                  StructuralType.NonStructural)

                try:
                    if d["side"] > 0:
                        fi.flipFacing()
                except Exception:
                    pass

                try:
                    if d["ccw"]:
                        fi.flipHand()
                except Exception:
                    pass

                placed += 1
            except Exception as door_err:
                print("Skipping door (%s): %s" % (d.get("source", "?"), str(door_err)))
                continue
        t.Commit()
    except Exception as e:
        t.RollBack()
        print("place_doors failed: %s" % str(e))
        placed = 0
    return placed

# ============================================================
# Place Windows
# ============================================================
def place_windows(win_data, level):
    symbols = get_symbols(BuiltInCategory.OST_Windows)
    if not symbols:
        raise Exception("No Window FamilySymbol in project.")
    sym_default = symbols[0]

    t=Transaction(doc, "V5 Place Windows")
    t.Start()
    try:
        placed=0
        for w in win_data:
            sym = get_symbol_by_width(symbols, w["gap"]) or sym_default
            ensure_active(sym)

            pt_xy = w["pt"]
            wr = w.get("wr", None)
            if wr is not None:
                pt_xy = snap_host_point_xy(wr, pt_xy)
            pt = xy_to_xyz(pt_xy, level.Elevation)
            fi = doc.Create.NewFamilyInstance(pt, sym, w["wall"], level,
                                              StructuralType.NonStructural)

            p_sill = fi.LookupParameter("Sill Height") or fi.LookupParameter("Sill")
            if p_sill and not p_sill.IsReadOnly:
                p_sill.Set(DEFAULT_SILL_FT)

            placed += 1
        t.Commit()
    except Exception as e:
        t.RollBack()
        print("place_windows failed: %s" % str(e))
        placed = 0
    return placed

# ============================================================
# Curtain Wall creation
# ============================================================
def get_any_curtain_walltype():
    for wt in FilteredElementCollector(doc).OfClass(WallType):
        try:
            if wt.Kind == WallKind.Curtain:
                return wt
        except Exception:
            pass
    return None

def create_curtain_walls(curtain_data, level):
    wt = get_any_curtain_walltype()
    if not wt:
        print("No Curtain WallType found. Skipping curtain walls.")
        return 0

    t=Transaction(doc, "V5 Create Curtain Walls")
    t.Start()
    try:
        created=0
        for c in curtain_data:
            wall = c["wall"]
            crv = wall.Location.Curve
            p0 = crv.GetEndPoint(0); p1 = crv.GetEndPoint(1)
            d = vnorm(vsub(xyz_to_xy(p1), xyz_to_xy(p0)))

            half = c["gap"] * 0.5
            mid = c["pt"]
            a = vadd(mid, vmul(d, -half))
            b = vadd(mid, vmul(d,  half))

            ln = Line.CreateBound(xy_to_xyz(a, level.Elevation), xy_to_xyz(b, level.Elevation))
            Wall.Create(doc, ln, wt.Id, level.Id, DEFAULT_WALL_HEIGHT_FT, 0.0, False, False)
            created += 1
        t.Commit()
    except Exception as e:
        t.RollBack()
        print("create_curtain_walls failed: %s" % str(e))
        created = 0
    return created

# ============================================================
# MAIN
# ============================================================
sel = list(uidoc.Selection.GetElementIds())
if len(sel) != 1:
    raise Exception("Select one ImportInstance (CAD link) and run again.")

imp = doc.GetElement(sel[0])
if not isinstance(imp, ImportInstance):
    raise Exception("Selection is not an ImportInstance.")

raw_lines, raw_arcs = extract_lines_and_arcs(imp)

diag = []
diag.append("Raw lines: %d, Raw arcs: %d" % (len(raw_lines), len(raw_arcs)))

if raw_lines:
    lengths_ft = sorted([l.Length for l in raw_lines])
    diag.append("Line len range: %.4f..%.4f ft (%.1f..%.1f mm)" % (
        lengths_ft[0], lengths_ft[-1], lengths_ft[0]*304.8, lengths_ft[-1]*304.8))
    diag.append("MIN_RAW_LEN_FT=%.4f ft (%.1f mm)" % (MIN_RAW_LEN_FT, MIN_RAW_LEN_FT*304.8))
    diag.append("MIN_AXIS_LEN_FT=%.4f ft (%.1f mm)" % (MIN_AXIS_LEN_FT, MIN_AXIS_LEN_FT*304.8))
    for i, l in enumerate(raw_lines[:3]):
        p0 = l.GetEndPoint(0); p1 = l.GetEndPoint(1)
        diag.append("  L%d: (%.2f,%.2f)->(%.2f,%.2f) len=%.4f ft" % (
            i, p0.X, p0.Y, p1.X, p1.Y, l.Length))

segs = [Seg(l) for l in raw_lines if l.Length >= MIN_RAW_LEN_FT]
diag.append("Segs after filter: %d" % len(segs))

if segs:
    sample_dists = []
    for i in range(min(len(segs), 20)):
        for j in range(i+1, min(len(segs), 20)):
            if angle_parallel(segs[i].dir, segs[j].dir, ANGLE_TOL_DEG):
                d = dist_point_to_infinite_line_xy(segs[j].a, segs[i].a, segs[i].dir)
                if d > 0.001:
                    sample_dists.append(d)
    if sample_dists:
        sample_dists.sort()
        diag.append("Parallel dists (ft): %s" % [round(d,4) for d in sample_dists[:10]])
        diag.append("Parallel dists (mm): %s" % [round(d*304.8,1) for d in sample_dists[:10]])
    diag.append("THICK range: %.4f..%.4f ft (%.1f..%.1f mm)" % (
        THICK_MIN_FT, THICK_MAX_FT, THICK_MIN_FT*304.8, THICK_MAX_FT*304.8))

# Write diagnostics to file regardless of outcome
_diag_path = os.path.join(os.environ.get("TEMP", "C:\\\\Temp"), "c2rv5_diag.txt")
try:
    with open(_diag_path, "w") as _f:
        _f.write("\n".join(diag))
except Exception:
    pass

pairs = find_parallel_pairs(segs)
if not pairs:
    msg = "No wall pairs found.\n" + "\n".join(diag)
    raise Exception(msg)

clusters = cluster_thickness([t for (_,_,t) in pairs], THICK_CLUSTER_TOL_FT)

# If all detected thicknesses are tiny (symbolic lines), override to real wall thickness
if all(c < 50 * MM_TO_FT for c in clusters):
    print("Detected symbolic line spacing (%.1f mm). Overriding wall thickness to %.0f mm." % (
        clusters[0]*304.8, DEFAULT_WALL_THICK_FT*304.8))
    clusters = [DEFAULT_WALL_THICK_FT]

center_segs=[]
for a,b,t in pairs:
    cs = build_center_from_pair(a,b,snap_to_cluster(t,clusters))
    if cs:
        center_segs.append(cs)

axes = build_wall_axes(center_segs)
axes = split_axes_at_intersections(axes)

# V5 exterior classification
classify_exterior_axes_v5(axes)

# level from view
view = doc.ActiveView
level = getattr(view, "GenLevel", None)
if level is None:
    level = FilteredElementCollector(doc).OfClass(Level).FirstElement()

# previews
if PREVIEW_WALL_AXES:
    draw_axes(axes, "V5 Preview Wall Axes")

if PREVIEW_EXTERIOR_AXES:
    ext = [a for a in axes if a.is_exterior]
    draw_axes(ext, "V5 Preview Exterior Axes")

# create walls
walls_axes_and_walls=[]
if BUILD_WALLS:
    walls_axes_and_walls = create_walls(axes, level, DEFAULT_WALL_HEIGHT_FT)
    print("Walls created: %d" % len(walls_axes_and_walls))

wall_refs = [WallRef2D(axis, wall) for (axis, wall) in walls_axes_and_walls]

# doors
door_data = door_data_from_arcs(raw_arcs, wall_refs)
gap_door_data = door_data_from_opening_gaps(wall_refs, [d["pt"] for d in door_data], segs)
if gap_door_data:
    door_data.extend(gap_door_data)
    print("Doors inferred from gaps (arc missed): %d" % len(gap_door_data))
bridge_door_data = door_data_from_bridge_jamb_pairs(wall_refs, [d["pt"] for d in door_data], segs)
if bridge_door_data:
    door_data.extend(bridge_door_data)
    print("Doors inferred from jamb pairs (recessed openings): %d" % len(bridge_door_data))
door_pts = [d["pt"] for d in door_data]

if PREVIEW_DOOR_POINTS:
    draw_points(door_pts, level.Elevation, "V5 Preview Door Points")

if BUILD_DOORS and door_data:
    n = place_doors(door_data, level)
    print("Doors placed: %d" % n)

# windows + curtains from openings (EXTERIOR ONLY)
win_data, curtain_data = openings_from_exterior_axes(wall_refs, door_pts)
win_pts = [w["pt"] for w in win_data]

if PREVIEW_WINDOW_POINTS:
    draw_points(win_pts, level.Elevation, "V5 Preview Window Points")

if BUILD_WINDOWS and win_data:
    n = place_windows(win_data, level)
    print("Windows placed: %d" % n)

if BUILD_CURTAIN_WALLS and curtain_data:
    n = create_curtain_walls(curtain_data, level)
    print("Curtain walls created: %d" % n)

print("Exterior axes: %d / %d" % (sum(1 for a in axes if a.is_exterior), len(axes)))
print("Thickness clusters (mm): %s" % [round(c*304.8,1) for c in clusters])
print("Done V5.")
