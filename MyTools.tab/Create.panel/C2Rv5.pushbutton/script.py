# -*- coding: utf-8 -*-
__title__ = "C2Rv5.2"
__doc__ = "CAD to Revit V5.2 – V5.1 + single-line interior walls + window jamb-pair detection."

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
BUILD_ID = "C2Rv5.2-build-2026-03-01-0225"

# ============================================================
# PARAMETERS (TUNE)
# ============================================================
MM_TO_FT = 1.0 / 304.8

# Filtering
MIN_RAW_LEN_FT   = 80  * MM_TO_FT
MIN_AXIS_LEN_FT  = 120 * MM_TO_FT
ORPHAN_AXIS_MIN_LEN_FT = 300 * MM_TO_FT
ANGLE_TOL_DEG    = 1.0

# Pairing wall side lines
OVERLAP_MIN      = 0.55
THICK_MIN_FT     = 70  * MM_TO_FT
THICK_MAX_FT     = 600 * MM_TO_FT
THICK_CLUSTER_TOL_FT = 8 * MM_TO_FT
PAIR_THICK_SNAP_TOL_FT = 25 * MM_TO_FT

DEFAULT_WALL_THICK_FT = 200 * MM_TO_FT
DEFAULT_PARTITION_THICK_FT = 100 * MM_TO_FT

# Real-world wall thickness window used to suppress symbolic micro-pairs.
REAL_WALL_MIN_THICK_FT = 70 * MM_TO_FT
REAL_WALL_MAX_THICK_FT = 450 * MM_TO_FT
MAX_WALL_THICK_CLUSTERS = 3
THIN_WALL_CLUSTER_MAX_FT = 145 * MM_TO_FT
WALL_CLUSTER_MIN_SHARE = 0.06
WALL_CLUSTER_MIN_COUNT = 2
WALL_STD_THICKS_MM = [100.0, 120.0, 150.0, 200.0, 250.0, 300.0]
WALL_STD_SNAP_TOL_MM = 18.0

# Make walls continuous across openings
MERGE_GAP_MAX_FT = 3000 * MM_TO_FT
AXIS_INTERVAL_MERGE_TOL_FT = 3 * MM_TO_FT

# Split at intersections
INTERSECT_TOL_FT      = 5 * MM_TO_FT
OPENING_SPLIT_SKIP_TOL_FT = 20 * MM_TO_FT
OPENING_CLIP_TOL_FT       = 2  * MM_TO_FT

# Wall creation
DEFAULT_WALL_HEIGHT_FT = 3000 * MM_TO_FT

# Door arc detection
DOOR_ARC_R_MIN_FT = 450 * MM_TO_FT
DOOR_ARC_R_MAX_FT = 1500 * MM_TO_FT
HOST_SEARCH_DIST_FT = 350 * MM_TO_FT

BRIDGE_JAMB_MAX_LEN_FT = 1800 * MM_TO_FT
BRIDGE_JAMB_PAR_TOL_DEG = 6.0
BRIDGE_GAP_PERP_DOT_MAX = 0.35
BRIDGE_HOST_EXTEND_FT   = 60 * MM_TO_FT
DOOR_JAMB_MAX_LEN_FT = 650 * MM_TO_FT
DOOR_JAMB_SEARCH_RAD_FT = 450 * MM_TO_FT
DOOR_JAMB_PERP_TOL_DEG = 18.0

# Gaps classification
GAP_MIN_DOOR_FT   = 650 * MM_TO_FT
GAP_MAX_DOOR_FT   = 1400 * MM_TO_FT

GAP_MIN_WIN_FT    = 450 * MM_TO_FT
GAP_MAX_WIN_FT    = 4500 * MM_TO_FT
ARC_NEAR_GAP_FT   = 450 * MM_TO_FT

WINDOW_HOST_MIN_THICK_FT = 60 * MM_TO_FT
WINDOW_HOST_THIN_PENALTY_FT = 500 * MM_TO_FT
WINDOW_HOST_SHORT_PENALTY_FT = 200 * MM_TO_FT

# Curtain threshold (gap width)
CURTAIN_THRESHOLD_FT = 2400 * MM_TO_FT

DEFAULT_SILL_FT   = 1050 * MM_TO_FT
DEFAULT_DOOR_BASE_OFFSET_FT = 0.0

# Envelope graph tolerances
NODE_MERGE_TOL_FT = 80 * MM_TO_FT
MAX_LOOP_NODES    = 220

# V5.2: Single-line interior walls fallback
SINGLE_LINE_AXIS_MIN_LEN_FT = 900 * MM_TO_FT
SINGLE_LINE_NEAR_AXIS_TOL_FT = 120 * MM_TO_FT
SINGLE_LINE_DEFAULT_THICK_FT = 100 * MM_TO_FT

# V5.2: Window jamb-pair detection (symbols, not gaps)
WIN_JAMB_MAX_LEN_FT = 500 * MM_TO_FT
WIN_JAMB_SEARCH_RAD_FT = 350 * MM_TO_FT
WIN_GAP_MIN_FT = 500 * MM_TO_FT
WIN_GAP_MAX_FT = 3000 * MM_TO_FT
WIN_JAMB_PERP_TOL_DEG = 15.0
ENABLE_WINDOW_JAMB_PAIR_FALLBACK = True

# True-duplicate tolerance (do not collapse distinct close windows).
OPENING_DUP_CENTER_TOL_FT = 45 * MM_TO_FT
OPENING_DUP_GAP_TOL_FT = 90 * MM_TO_FT
DOOR_DUP_CENTER_TOL_FT = 220 * MM_TO_FT
WINDOW_DUP_CENTER_TOL_FT = 220 * MM_TO_FT
WINDOW_GAP_MAX_STRICT_FT = 2200 * MM_TO_FT
WINDOW_NEAR_DOOR_CLEARANCE_FT = 800 * MM_TO_FT
WINDOW_JAMB_CLUSTER_SPACING_FT = 500 * MM_TO_FT
WINDOW_JAMB_MAX_TOTAL = 2

DOOR_MAX_WIDTH_FT = 1300 * MM_TO_FT
DOOR_REJECT_NAME_TOKENS = [
    "double", "double-flush", "2leaf", "2-leaf", "bifold", "bi-fold", "fold", "passage", "opening", "void", u"כפול", u"מעבר", u"פתח"
]
DOOR_PREFERRED_FAMILY_TOKEN = u"עץ-ציר"
DOOR_DEBUG_LIST_ON_PREFERRED_MISS = True
DOOR_END_CLEARANCE_FT = 120 * MM_TO_FT
AXIS_OVERLAP_SUPPRESS_DIST_FT = 90 * MM_TO_FT
OPENING_CONFLICT_CLEARANCE_FT = 900 * MM_TO_FT
DOOR_REHOST_MAX_DIST_FT = 700 * MM_TO_FT
ENABLE_AXIS_SPLIT_AT_INTERSECTIONS = False
STRICT_OPENING_HOST_MODE = True
STRICT_ARC_DOOR_REQUIRE_AXIS_GAP = True

# Keep strict double-line wall mode by default.
ENABLE_SINGLE_LINE_FALLBACK = False

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
    if v < lo: return lo
    if v > hi: return hi
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
# Extract geometry (includes PolyLine)
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

def remap_symbolic_clusters(clusters):
    if len(clusters) <= 1:
        return [DEFAULT_WALL_THICK_FT]
    clusters = sorted(clusters)
    n = len(clusters)
    result = []
    for i in range(n):
        frac = float(i) / float(n - 1)
        mapped = DEFAULT_PARTITION_THICK_FT + frac * (DEFAULT_WALL_THICK_FT - DEFAULT_PARTITION_THICK_FT)
        result.append(mapped)
    return result

def cluster_thickness_with_counts(values, tol_ft):
    groups = []
    for v in sorted(values):
        placed = False
        for g in groups:
            if abs(g["rep"] - v) <= tol_ft:
                n = g["count"]
                g["rep"] = ((g["rep"] * n) + v) / float(n + 1)
                g["count"] = n + 1
                placed = True
                break
        if not placed:
            groups.append({"rep": v, "count": 1})
    return groups

def select_wall_thickness_clusters(pairs):
    if not pairs:
        return [], []
    groups = cluster_thickness_with_counts([t for (_, _, t) in pairs], THICK_CLUSTER_TOL_FT)
    real_groups = [g for g in groups if REAL_WALL_MIN_THICK_FT <= g["rep"] <= REAL_WALL_MAX_THICK_FT]
    if not real_groups:
        real_groups = list(groups)
    real_groups.sort(key=lambda g: g["count"], reverse=True)
    total = sum(g["count"] for g in real_groups)
    min_count = max(WALL_CLUSTER_MIN_COUNT, int(round(float(total) * WALL_CLUSTER_MIN_SHARE)))

    significant = [g for g in real_groups if g["count"] >= min_count]
    if not significant:
        significant = real_groups[:MAX_WALL_THICK_CLUSTERS]

    # Force-keep two thin partition families when available (e.g. 100/150mm).
    thin = [g for g in real_groups if g["rep"] <= THIN_WALL_CLUSTER_MAX_FT]
    thin = sorted(thin, key=lambda g: g["count"], reverse=True)
    for g in thin[:2]:
        if g not in significant:
            significant.append(g)

    # Keep strongest thick family for exterior walls.
    thick = [g for g in real_groups if g["rep"] > THIN_WALL_CLUSTER_MAX_FT]
    if thick and (thick[0] not in significant):
        significant.append(thick[0])

    def _snap_std(ft_val):
        mm = ft_val * 304.8
        best = mm
        bestd = 1e9
        for s in WALL_STD_THICKS_MM:
            d = abs(mm - s)
            if d < bestd:
                bestd = d
                best = s
        if bestd <= WALL_STD_SNAP_TOL_MM:
            return best / 304.8
        return ft_val

    chosen = [_snap_std(g["rep"]) for g in significant]
    chosen = sorted(chosen)

    dedup = []
    for c in chosen:
        if not dedup or abs(c - dedup[-1]) > THICK_CLUSTER_TOL_FT:
            dedup.append(c)
    chosen = dedup[:(MAX_WALL_THICK_CLUSTERS + 2)]
    return chosen, groups

# ============================================================
# Build raw center segments from pairs
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
# Build continuous WallAxes + openings
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
    qoff=int(round(off / (10*MM_TO_FT)))
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
        o0=ref.a; d0=ref.dir; z0=ref.z

        # thickness = median of bucket (stable)
        thicks = sorted([s.thick for s in segs])
        thick = thicks[len(thicks)//2]

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

def merge_node(nodes, p, tol):
    for i, n in enumerate(nodes):
        if vlen(vsub(n, p)) <= tol:
            return i
    nodes.append(p)
    return len(nodes)-1

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
                    if s.openings and scalar_in_intervals(t, s.openings, OPENING_SPLIT_SKIP_TOL_FT):
                        continue
                    cuts.append(t)

        cuts=sorted(set([round(c,6) for c in cuts]))
        for i in range(len(cuts)-1):
            t0,t1=cuts[i],cuts[i+1]
            if (t1-t0) < MIN_AXIS_LEN_FT:
                continue
            wa=WallAxis(s.o,s.d,s.z,s.thick,t0,t1)

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
# Filter orphan short axes
# ============================================================
def filter_orphan_short_axes(axes, min_len_ft):
    nodes = []
    node_ids = []
    for ax in axes:
        n0 = merge_node(nodes, ax.a, NODE_MERGE_TOL_FT)
        n1 = merge_node(nodes, ax.b, NODE_MERGE_TOL_FT)
        node_ids.append((n0, n1))

    degree = defaultdict(int)
    for (n0, n1) in node_ids:
        degree[n0] += 1
        degree[n1] += 1

    out = []
    removed = 0
    for i, ax in enumerate(axes):
        ax_len = vlen(vsub(ax.a, ax.b))
        n0, n1 = node_ids[i]
        if ax_len >= min_len_ft or (degree[n0] >= 2 and degree[n1] >= 2):
            out.append(ax)
        else:
            removed += 1
    if removed:
        print("Filtered %d orphan short axes" % removed)
    return out

# ============================================================
# Dedup overlapping axes (true duplicates only)
# ============================================================
def dedup_overlapping_axes(axes):
    removed = set()
    n = len(axes)

    for i in range(n):
        if i in removed:
            continue
        ai = axes[i]
        for j in range(i + 1, n):
            if j in removed:
                continue
            aj = axes[j]
            if not angle_parallel(ai.d, aj.d, ANGLE_TOL_DEG):
                continue

            tol_i = max(20 * MM_TO_FT, min(60 * MM_TO_FT, getattr(ai, "thick", DEFAULT_WALL_THICK_FT) * 0.25))
            tol_j = max(20 * MM_TO_FT, min(60 * MM_TO_FT, getattr(aj, "thick", DEFAULT_WALL_THICK_FT) * 0.25))
            fixed_tol = max(tol_i, tol_j)
            dist = dist_point_to_infinite_line_xy(aj.a, ai.a, ai.d)
            if dist > fixed_tol:
                continue

            o = ai.a; d = ai.d
            ai0 = project_scalar_on_axis(ai.a, o, d)
            ai1 = project_scalar_on_axis(ai.b, o, d)
            aj0 = project_scalar_on_axis(aj.a, o, d)
            aj1 = project_scalar_on_axis(aj.b, o, d)
            if ai0 > ai1: ai0, ai1 = ai1, ai0
            if aj0 > aj1: aj0, aj1 = aj1, aj0
            ov = overlap_ratio_1d(ai0, ai1, aj0, aj1)
            if ov < 0.7:
                continue

            li = ai1 - ai0
            lj = aj1 - aj0
            if li > lj:
                removed.add(j)
            elif lj > li:
                removed.add(i)
                break
            else:
                if ai.thick >= aj.thick:
                    removed.add(j)
                else:
                    removed.add(i)
                    break

    out = [axes[i] for i in range(n) if i not in removed]
    if removed:
        print("Deduped %d overlapping axes (true duplicates only)" % len(removed))
    return out

def merge_collinear_axes_for_creation(axes):
    buckets = defaultdict(list)
    for ax in axes:
        db = quant_dir(ax.d)
        n = (-ax.d[1], ax.d[0])
        off = vdot(ax.a, n)
        qoff = int(round(off / (20 * MM_TO_FT)))
        tkey = int(round(ax.thick / THICK_CLUSTER_TOL_FT))
        buckets[(db[0], db[1], qoff, tkey, round(ax.z, 6))].append(ax)

    out = []
    for _, group in buckets.items():
        ref = group[0]
        o = ref.o
        d = ref.d
        z = ref.z
        thicks = sorted([g.thick for g in group])
        thick = thicks[len(thicks) // 2]

        intervals = []
        all_openings = []
        for g in group:
            t0 = project_scalar_on_axis(g.a, o, d)
            t1 = project_scalar_on_axis(g.b, o, d)
            if t0 > t1:
                t0, t1 = t1, t0
            intervals.append((t0, t1))
            all_openings.extend(list(getattr(g, "openings", []) or []))

        merged = merge_intervals(intervals, AXIS_INTERVAL_MERGE_TOL_FT)
        merged_open = merge_intervals(all_openings, OPENING_CLIP_TOL_FT)
        ext_flag = any(getattr(g, "is_exterior", False) for g in group)

        for (t0, t1) in merged:
            if (t1 - t0) < MIN_AXIS_LEN_FT:
                continue
            wa = WallAxis(o, d, z, thick, t0, t1)
            wa.is_exterior = ext_flag
            clipped = []
            for (a, b) in merged_open:
                oa = max(a, t0)
                ob = min(b, t1)
                if (ob - oa) > (1 * MM_TO_FT):
                    clipped.append((oa, ob))
            wa.openings = merge_intervals(clipped, OPENING_CLIP_TOL_FT)
            out.append(wa)
    return out

def suppress_overlapping_axes(axes):
    if not axes:
        return axes

    ranked = sorted(
        axes,
        key=lambda a: (getattr(a, "thick", 0.0), vlen(vsub(a.a, a.b))),
        reverse=True
    )
    grid = SpatialGrid(2.0 / 0.3048)
    kept = []
    removed = 0

    for ax in ranked:
        dup = False
        neigh = grid.query(bbox_expand(ax.bb, AXIS_OVERLAP_SUPPRESS_DIST_FT))
        for k in neigh:
            if not angle_parallel(ax.d, k.d, 2.0):
                continue
            dist = dist_point_to_infinite_line_xy(ax.a, k.a, k.d)
            if dist > AXIS_OVERLAP_SUPPRESS_DIST_FT:
                continue

            o = k.a
            d = k.d
            a0 = project_scalar_on_axis(ax.a, o, d)
            a1 = project_scalar_on_axis(ax.b, o, d)
            b0 = project_scalar_on_axis(k.a, o, d)
            b1 = project_scalar_on_axis(k.b, o, d)
            if a0 > a1:
                a0, a1 = a1, a0
            if b0 > b1:
                b0, b1 = b1, b0
            ov = overlap_ratio_1d(a0, a1, b0, b1)
            if ov >= 0.65:
                dup = True
                break

        if dup:
            removed += 1
            continue
        kept.append(ax)
        grid.insert(ax)

    if removed:
        print("Suppressed %d overlapping axes before wall creation." % removed)
    return kept

# ============================================================
# V5.2: Add single-line interior axes (fallback)
# ============================================================
def add_single_line_interior_axes(segs, axes):
    grid = SpatialGrid(2.0/0.3048)
    for ax in axes:
        grid.insert(ax)

    added = 0
    for s in segs:
        if s.len < SINGLE_LINE_AXIS_MIN_LEN_FT:
            continue

        neigh = grid.query(bbox_expand(s.bb, SINGLE_LINE_NEAR_AXIS_TOL_FT))
        too_close = False
        for ax in neigh:
            if not angle_parallel(s.dir, ax.d, 2.0):
                continue
            d = dist_point_to_infinite_line_xy(s.a, ax.a, ax.d)
            if d <= SINGLE_LINE_NEAR_AXIS_TOL_FT:
                o=ax.a; ddir=ax.d
                a0=project_scalar_on_axis(ax.a,o,ddir); a1=project_scalar_on_axis(ax.b,o,ddir)
                b0=project_scalar_on_axis(s.a,o,ddir);  b1=project_scalar_on_axis(s.b,o,ddir)
                if a0>a1: a0,a1=a1,a0
                if b0>b1: b0,b1=b1,b0
                if overlap_ratio_1d(a0,a1,b0,b1) > 0.35:
                    too_close = True
                    break
        if too_close:
            continue

        o0 = s.a
        d0 = s.dir
        ax_new = WallAxis(o0, d0, s.z, SINGLE_LINE_DEFAULT_THICK_FT, 0.0, s.len)
        ax_new.is_exterior = False
        axes.append(ax_new)
        added += 1

    if added:
        print("Added single-line interior axes: %d" % added)
    return axes

# ============================================================
# Exterior classification: Max-area loop
# ============================================================
def polygon_area_xy(pts):
    area=0.0
    for i in range(len(pts)):
        x1,y1=pts[i]
        x2,y2=pts[(i+1)%len(pts)]
        area += (x1*y2 - x2*y1)
    return abs(area)*0.5

def _classify_exterior_fallback(axes):
    bb = [1e9, 1e9, -1e9, -1e9]
    for ax in axes:
        bb[0] = min(bb[0], ax.bb[0]); bb[1] = min(bb[1], ax.bb[1])
        bb[2] = max(bb[2], ax.bb[2]); bb[3] = max(bb[3], ax.bb[3])
    for ax in axes:
        touch = (abs(ax.bb[0]-bb[0]) < 300*MM_TO_FT or abs(ax.bb[2]-bb[2]) < 300*MM_TO_FT or
                 abs(ax.bb[1]-bb[1]) < 300*MM_TO_FT or abs(ax.bb[3]-bb[3]) < 300*MM_TO_FT)
        ax.is_exterior = bool(touch)

def classify_exterior_axes_v5(axes):
    nodes=[]
    adj=defaultdict(list)

    for ax in axes:
        n0 = merge_node(nodes, ax.a, NODE_MERGE_TOL_FT)
        n1 = merge_node(nodes, ax.b, NODE_MERGE_TOL_FT)
        adj[n0].append((n1, ax))
        adj[n1].append((n0, ax))

    if len(nodes) < 4:
        for ax in axes:
            ax.is_exterior = True
        return

    state = {"best_area": 0.0, "best_axes": None}

    def dfs(start, current, path_nodes, path_axes, visited):
        if len(path_nodes) > MAX_LOOP_NODES:
            return
        for (nxt, ax) in adj[current]:
            if nxt == start and len(path_nodes) >= 4:
                pts = [nodes[i] for i in path_nodes]
                area = polygon_area_xy(pts)
                if area > state["best_area"]:
                    state["best_area"] = area
                    state["best_axes"] = list(path_axes)
                continue
            if nxt in visited:
                continue
            if nxt < start:
                continue
            visited.add(nxt)
            path_nodes.append(nxt)
            path_axes.append(ax)
            dfs(start, nxt, path_nodes, path_axes, visited)
            path_axes.pop()
            path_nodes.pop()
            visited.remove(nxt)

    for s in range(len(nodes)):
        visited=set([s])
        dfs(s, s, [s], [], visited)

    best_axes = state["best_axes"]
    if not best_axes:
        print("Exterior loop not found. Using bbox fallback.")
        _classify_exterior_fallback(axes)
        return

    loop_set = set(id(a) for a in best_axes)
    for ax in axes:
        ax.is_exterior = (id(ax) in loop_set)

    ext_open = sum(len(a.openings) for a in axes if a.is_exterior)
    print("Exterior loop edges: %d | exterior openings: %d" % (len(best_axes), ext_open))

def normalize_exterior_wall_thickness(axes, clusters):
    ext = [a for a in axes if getattr(a, "is_exterior", False)]
    if not ext:
        return
    thick_candidates = [c for c in clusters if c > THIN_WALL_CLUSTER_MAX_FT]
    if thick_candidates:
        target = max(thick_candidates)
    else:
        target = max(clusters) if clusters else None
    if target is None:
        return
    tol = 20 * MM_TO_FT
    changed = 0
    for a in ext:
        if abs(a.thick - target) > tol and a.thick < target:
            a.thick = target
            changed += 1
    if changed:
        print("Normalized exterior wall thickness on %d axes to %.0f mm" % (changed, target * 304.8))

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
    for wt in FilteredElementCollector(doc).OfClass(WallType):
        try:
            if wt.Kind == WallKind.Basic:
                return wt
        except Exception:
            continue
    return FilteredElementCollector(doc).OfClass(WallType).FirstElement()

def _safe_elem_name(elem):
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

def _find_material_by_tokens(token_list):
    mats = list(FilteredElementCollector(doc).OfClass(Material))
    for m in mats:
        nm = (_safe_elem_name(m) or "").lower()
        for tok in token_list:
            if tok in nm:
                return m
    return None

def _pick_wall_material(is_exterior, thick_ft):
    if is_exterior:
        m = _find_material_by_tokens(["concrete", "beton", u"בטון"])
        return m, "CONC"
    # User rule: all interior walls are CMU (including 100/150/200).
    m = _find_material_by_tokens(["cmu", "block", "blk", u"בלוק"])
    if m is not None:
        return m, "CMU"
    # fallback: concrete if CMU material is unavailable in project.
    m2 = _find_material_by_tokens(["concrete", "beton", u"בטון"])
    if m2 is not None:
        return m2, "CMU"
    return None, "GEN"

def _set_walltype_structure(wt, thick_ft, mat_id=None):
    cs = wt.GetCompoundStructure()
    if cs:
        # Preferred path: replace wall layers with one structural layer.
        try:
            from System.Collections.Generic import List
            layers = List[CompoundStructureLayer]()
            mid = mat_id if mat_id is not None else ElementId.InvalidElementId
            layers.Add(CompoundStructureLayer(thick_ft, MaterialFunctionAssignment.Structure, mid))
            cs.SetLayers(layers)
            wt.SetCompoundStructure(cs)
            return
        except Exception:
            pass

        # Fallback: set widest layer width only (avoid invalid zero-width edits).
        try:
            lc = cs.LayerCount
            idx = 0
            best = -1.0
            for i in range(lc):
                w = cs.GetLayerWidth(i)
                if w > best:
                    best = w
                    idx = i
            cs.SetLayerWidth(idx, thick_ft)
            if mat_id is not None:
                try:
                    cs.SetMaterialId(idx, mat_id)
                except Exception:
                    pass
            wt.SetCompoundStructure(cs)
            return
        except Exception:
            pass

    # Final fallback if compound-structure writes are blocked.
    p = wt.get_Parameter(BuiltInParameter.WALL_ATTR_WIDTH_PARAM)
    if p and (not p.IsReadOnly):
        p.Set(thick_ft)
        return

    raise Exception("Unable to set wall type structure/width for {}".format(_safe_elem_name(wt) or "Unknown"))

def _get_walltype_width(wt):
    try:
        cs = wt.GetCompoundStructure()
        if cs:
            total = 0.0
            for i in range(cs.LayerCount):
                total += cs.GetLayerWidth(i)
            return total
    except Exception:
        pass
    try:
        p = wt.get_Parameter(BuiltInParameter.WALL_ATTR_WIDTH_PARAM)
        if p:
            return p.AsDouble()
    except Exception:
        pass
    return None

def _walltype_structure_summary(wt):
    total = 0.0
    nz = 0
    mat_names = []
    try:
        cs = wt.GetCompoundStructure()
        if cs:
            for i in range(cs.LayerCount):
                w = cs.GetLayerWidth(i)
                total += w
                if w > (0.5 * MM_TO_FT):
                    nz += 1
                    try:
                        mid = cs.GetMaterialId(i)
                        m = doc.GetElement(mid) if mid else None
                        mat_names.append(_safe_elem_name(m) if m is not None else "None")
                    except Exception:
                        mat_names.append("Unknown")
    except Exception:
        pass
    return total, nz, mat_names

def get_or_create_walltype(doc, thick_ft, prefix="CAD_Auto", is_exterior=False, mat_id=None, mat_tag="GEN"):
    mm = thick_ft * 304.8
    role = "EXT" if is_exterior else "INT"
    base_name = "%s_%s_%s_%dmm" % (prefix, role, mat_tag, int(round(mm)))

    all_types = list(FilteredElementCollector(doc).OfClass(WallType))
    existing_names = set([_safe_elem_name(wt) for wt in all_types if _safe_elem_name(wt)])

    base = _find_basic_walltype()
    new_name = base_name
    if new_name in existing_names:
        idx = 2
        while ("%s_v%d" % (base_name, idx)) in existing_names:
            idx += 1
        new_name = "%s_v%d" % (base_name, idx)

    new_type = base.Duplicate(new_name)
    _set_walltype_structure(new_type, thick_ft, mat_id=mat_id)
    return new_type

def create_walls(axes, level, height_ft):
    wt_cache={}
    out=[]
    _mat_log = set()
    t=Transaction(doc, "V5.2 Create Walls")
    t.Start()
    try:
        for w in axes:
            m, mtag = _pick_wall_material(getattr(w, "is_exterior", False), w.thick)
            mid = m.Id if m is not None else None
            key=(int(round(w.thick/THICK_CLUSTER_TOL_FT)), 1 if getattr(w, "is_exterior", False) else 0, mtag)
            if key not in wt_cache:
                wt_cache[key]=get_or_create_walltype(
                    doc,
                    w.thick,
                    prefix="CAD_Auto",
                    is_exterior=getattr(w, "is_exterior", False),
                    mat_id=mid,
                    mat_tag=mtag
                )
                wt_total, wt_nz, wt_mats = _walltype_structure_summary(wt_cache[key])
                if abs(wt_total - w.thick) > (1.5 * MM_TO_FT):
                    raise Exception("Wall type thickness mismatch for %s: expected %.0fmm got %.1fmm" % (
                        _safe_elem_name(wt_cache[key]) or "Unknown", w.thick * 304.8, wt_total * 304.8))
                if wt_nz != 1:
                    raise Exception("Wall type structure invalid for %s: expected 1 active layer, got %d" % (
                        _safe_elem_name(wt_cache[key]) or "Unknown", wt_nz))
                ml = _safe_elem_name(m) if m is not None else "Default"
                if (mtag, ml) not in _mat_log:
                    _mat_log.add((mtag, ml))
                    print("Wall material role %s -> %s" % (mtag, ml))
                print("Wall type %s | %.0fmm | active_layers=%d | mats=%s" % (
                    _safe_elem_name(wt_cache[key]) or "Unknown",
                    wt_total * 304.8,
                    wt_nz,
                    wt_mats))
            wt=wt_cache[key]
            ln=Line.CreateBound(xy_to_xyz(w.a, level.Elevation), xy_to_xyz(w.b, level.Elevation))
            wall=Wall.Create(doc, ln, wt.Id, level.Id, height_ft, 0.0, False, False)
            out.append((w, wall))
        st = t.Commit()
        if st != TransactionStatus.Committed:
            raise Exception("wall transaction did not commit: %s" % str(st))
    except Exception as e:
        t.RollBack()
        raise Exception("create_walls failed: %s" % str(e))

    valid = []
    invalid_count = 0
    for (ax, wall) in out:
        try:
            wid = wall.Id
            w2 = doc.GetElement(wid)
            if w2 is None:
                invalid_count += 1
                continue
            _ = w2.Id.IntegerValue
            valid.append((ax, w2))
        except Exception:
            invalid_count += 1
    if invalid_count:
        print("Skipped %d invalid walls after create transaction." % invalid_count)
    return valid

# ============================================================
# Hosting search helpers
# ============================================================
class WallRef2D(object):
    def __init__(self, axis, wall):
        self.axis=axis
        self.wall=wall
        try:
            self.wall_id = wall.Id.IntegerValue
        except Exception:
            try:
                self.wall_id = str(wall.Id)
            except Exception:
                raise Exception("invalid wall reference")
        try:
            crv=wall.Location.Curve
            p0=crv.GetEndPoint(0); p1=crv.GetEndPoint(1)
        except Exception:
            raise Exception("wall geometry unavailable for wall id {}".format(self.wall_id))
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

def is_duplicate_opening(items, wr, p_xy, gap_ft, pt_tol_ft=None, gap_tol_ft=None):
    if wr is None:
        return False
    if pt_tol_ft is None:
        pt_tol_ft = OPENING_DUP_CENTER_TOL_FT
    if gap_tol_ft is None:
        gap_tol_ft = OPENING_DUP_GAP_TOL_FT

    for it in items:
        wr2 = it.get("wr", None)
        if wr2 is None:
            continue
        if getattr(wr2, "wall_id", None) != getattr(wr, "wall_id", None):
            continue
        if vlen(vsub(it.get("pt"), p_xy)) > pt_tol_ft:
            continue
        gap2 = it.get("gap", gap_ft)
        if abs(gap2 - gap_ft) <= gap_tol_ft:
            return True
    return False

def _wall_refs_geom_equivalent(wr1, wr2):
    if (wr1 is None) or (wr2 is None):
        return False
    if getattr(wr1, "wall_id", None) == getattr(wr2, "wall_id", None):
        return True
    if not angle_parallel(wr1.dir, wr2.dir, 3.0):
        return False
    if dist_point_to_infinite_line_xy(wr1.a, wr2.a, wr2.dir) > (120 * MM_TO_FT):
        return False
    m1 = ((wr1.a[0] + wr1.b[0]) * 0.5, (wr1.a[1] + wr1.b[1]) * 0.5)
    m2 = ((wr2.a[0] + wr2.b[0]) * 0.5, (wr2.a[1] + wr2.b[1]) * 0.5)
    return vlen(vsub(m1, m2)) <= (400 * MM_TO_FT)

def dedupe_openings(items, kind):
    if not items:
        return []
    if kind == "door":
        pt_tol = DOOR_DUP_CENTER_TOL_FT
        gap_tol = 180 * MM_TO_FT
        priority = {"arc": 0, "gap": 1, "bridge_jambs": 2, "cad_jamb_pair": 3}
    else:
        pt_tol = WINDOW_DUP_CENTER_TOL_FT
        gap_tol = OPENING_DUP_GAP_TOL_FT
        priority = {"gap": 0, "jamb_pair": 1}

    ordered = sorted(items, key=lambda x: priority.get(str(x.get("source", "")), 99))
    kept = []
    for it in ordered:
        wr = it.get("wr", None)
        p = it.get("pt", None)
        if p is None:
            continue
        g = it.get("gap", 0.0)
        dup = False
        for ex in kept:
            wrx = ex.get("wr", None)
            if not _wall_refs_geom_equivalent(wr, wrx):
                continue
            if kind == "door":
                s1, _ = _opening_axis_scalar(wr, p)
                s2, _ = _opening_axis_scalar(wrx, ex.get("pt"))
                if s1 is not None and s2 is not None:
                    h1 = 0.5 * g
                    h2 = 0.5 * ex.get("gap", g)
                    ov0 = max(s1 - h1, s2 - h2)
                    ov1 = min(s1 + h1, s2 + h2)
                    if ov1 >= ov0 - (60 * MM_TO_FT):
                        dup = True
                        break
            if vlen(vsub(ex.get("pt"), p)) > pt_tol:
                continue
            if abs(ex.get("gap", g) - g) > gap_tol:
                continue
            dup = True
            break
        if not dup:
            kept.append(it)
    return kept

def _door_class_from_data(d):
    cls = d.get("door_class", None)
    if cls in ("exterior", "interior"):
        return cls
    src = str(d.get("source", "")).lower()
    if src in ("bridge_jambs", "cad_jamb_pair"):
        return "interior"
    wr = d.get("wr", None)
    if wr is not None:
        return "exterior" if getattr(getattr(wr, "axis", None), "is_exterior", False) else "interior"
    return "interior"

def rehost_and_validate_doors(door_data, wall_refs):
    if not door_data:
        return []
    out = []
    rej = defaultdict(int)
    for d in door_data:
        wr = d.get("wr", None)
        pt = d.get("pt", None)
        if wr is None or pt is None:
            rej["missing_host_or_point"] += 1
            continue

        dcls = _door_class_from_data(d)
        d["door_class"] = dcls
        want_ext = True if dcls == "exterior" else False
        wr_is_ext = bool(getattr(getattr(wr, "axis", None), "is_exterior", False))
        class_ok = ((want_ext and wr_is_ext) or ((not want_ext) and (not wr_is_ext)))
        host = wr

        # Keep original host when it already matches class and point is close to host.
        # Only rehost when class mismatch or host geometry is clearly invalid.
        if class_ok:
            dline = dist_point_to_infinite_line_xy(pt, wr.a, wr.dir)
            if dline > max(HOST_SEARCH_DIST_FT, DOOR_REHOST_MAX_DIST_FT):
                class_ok = False

        if not class_ok:
            axis_dir = getattr(wr, "dir", None)
            host = best_host_wall_for_opening_point(
                wall_refs, pt, axis_dir=axis_dir, preferred_wr=None,
                exterior_only=want_ext,
                max_dist_ft=max(HOST_SEARCH_DIST_FT, DOOR_REHOST_MAX_DIST_FT),
                min_thick_ft=REAL_WALL_MIN_THICK_FT,
                target_opening_gap_ft=d.get("gap", 900 * MM_TO_FT)
            )
            if host is None:
                rej["no_host_after_reclass"] += 1
                continue
            if want_ext and (not getattr(host.axis, "is_exterior", False)):
                rej["host_not_exterior"] += 1
                continue
            if (not want_ext) and getattr(host.axis, "is_exterior", False):
                rej["host_not_interior"] += 1
                continue

        if STRICT_OPENING_HOST_MODE:
            dline = dist_point_to_infinite_line_xy(pt, host.a, host.dir)
            band = max(140 * MM_TO_FT, getattr(host.axis, "thick", DEFAULT_WALL_THICK_FT) * 0.9)
            if dline > band:
                rej["outside_host_band"] += 1
                continue
            host_len = max(vlen(vsub(host.b, host.a)), 1e-9)
            gap = d.get("gap", 900 * MM_TO_FT)
            if host_len < max(500 * MM_TO_FT, gap * 0.70):
                rej["host_too_short"] += 1
                continue
            t_host = segment_param_xy(pt, host.a, host.b)
            end_clear = min(t_host, 1.0 - t_host) * host_len
            min_end_clear = max((160 * MM_TO_FT) if getattr(host.axis, "is_exterior", False) else (80 * MM_TO_FT), 0.12 * gap)
            if end_clear < min_end_clear:
                rej["too_near_wall_end"] += 1
                continue

        d["wr"] = host
        d["wall"] = host.wall
        out.append(d)

    if rej:
        print("Door host validation rejects: %s" % dict(rej))
    if out:
        src = defaultdict(int)
        cls = defaultdict(int)
        for d in out:
            src[str(d.get("source", "?"))] += 1
            cls[str(d.get("door_class", "?"))] += 1
        print("Door candidates after host-validate by source: %s" % dict(src))
        print("Door candidates after host-validate by class: %s" % dict(cls))
    return out

def _opening_axis_scalar(wr, p_xy):
    if wr is None or p_xy is None:
        return None, None
    host_len = max(vlen(vsub(wr.b, wr.a)), 1e-9)
    t = segment_param_xy(p_xy, wr.a, wr.b)
    s = t * host_len
    return s, host_len

def _openings_conflict_on_host(dwr, dpt, dgap, owr, opt, ogap):
    if (dwr is None) or (owr is None) or (dpt is None) or (opt is None):
        return False
    if not _wall_refs_geom_equivalent(dwr, owr):
        return False

    # Fast radial test.
    if vlen(vsub(dpt, opt)) <= OPENING_CONFLICT_CLEARANCE_FT:
        return True

    ds, dlen = _opening_axis_scalar(dwr, dpt)
    os, olen = _opening_axis_scalar(owr, opt)
    if ds is None or os is None:
        return False

    # Interval overlap on host axis (door/window widths + clearance).
    dhalf = max((dgap or 0.0) * 0.5, 350 * MM_TO_FT)
    ohalf = max((ogap or 0.0) * 0.5, 250 * MM_TO_FT)
    need = dhalf + ohalf + (120 * MM_TO_FT)
    return abs(ds - os) <= need

def resolve_opening_conflicts(door_data, win_data, curtain_data):
    if not door_data:
        return win_data, curtain_data

    kept_wins = []
    rej_win = 0
    for w in (win_data or []):
        conflict = False
        for d in door_data:
            if _openings_conflict_on_host(
                d.get("wr"), d.get("pt"), d.get("gap", 0.0),
                w.get("wr"), w.get("pt"), w.get("gap", 0.0)
            ):
                conflict = True
                break
        if conflict:
            rej_win += 1
        else:
            kept_wins.append(w)

    kept_curt = []
    rej_curt = 0
    for c in (curtain_data or []):
        conflict = False
        for d in door_data:
            if _openings_conflict_on_host(
                d.get("wr"), d.get("pt"), d.get("gap", 0.0),
                c.get("wr"), c.get("pt"), c.get("gap", 0.0)
            ):
                conflict = True
                break
        if conflict:
            rej_curt += 1
        else:
            kept_curt.append(c)

    if rej_win or rej_curt:
        print("Opening conflict resolver: rejected_near_door_same_host windows=%d curtains=%d" % (rej_win, rej_curt))
    return kept_wins, kept_curt

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

        score = d + (overrun * 8.0)

        if preferred_wr is not None and getattr(wr, "wall_id", None) != getattr(preferred_wr, "wall_id", None):
            score += 5 * MM_TO_FT

        if min_thick_ft is not None and wr_thick < min_thick_ft:
            score += WINDOW_HOST_THIN_PENALTY_FT
        if target_opening_gap_ft is not None and seg_len < (target_opening_gap_ft + 150 * MM_TO_FT):
            score += WINDOW_HOST_SHORT_PENALTY_FT

        if score < best_score:
            best_score = score
            best = wr

    return best

# ============================================================
# Door data from arcs + gaps + recessed jamb-bridge
# ============================================================
def nearest_wall_for_door_arc(wall_refs, arc, exterior_only=None):
    e0 = xyz_to_xy(arc.GetEndPoint(0))
    e1 = xyz_to_xy(arc.GetEndPoint(1))
    chord_mid = ((e0[0]+e1[0])/2.0, (e0[1]+e1[1])/2.0)
    wr = nearest_wall(wall_refs, chord_mid, HOST_SEARCH_DIST_FT, exterior_only=exterior_only)
    if wr:
        return wr

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

def opening_has_door_leaf_evidence(cad_segs, wr, p_xy, gap_ft):
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

def opening_has_door_arc_evidence(cad_arcs, wr, p_xy, gap_ft):
    if not cad_arcs or wr is None or p_xy is None:
        return False
    center_tol = max(260 * MM_TO_FT, gap_ft * 0.5)
    host_band = max(220 * MM_TO_FT, getattr(wr.axis, "thick", DEFAULT_WALL_THICK_FT) * 1.6 + 120 * MM_TO_FT)
    for a in cad_arcs:
        try:
            r = a.Radius
        except Exception:
            continue
        if r < DOOR_ARC_R_MIN_FT or r > DOOR_ARC_R_MAX_FT:
            continue
        e0 = xyz_to_xy(a.GetEndPoint(0))
        e1 = xyz_to_xy(a.GetEndPoint(1))
        chord = vsub(e1, e0)
        chord_len = vlen(chord)
        if chord_len < GAP_MIN_DOOR_FT or chord_len > GAP_MAX_DOOR_FT:
            continue
        chord_mid = ((e0[0] + e1[0]) * 0.5, (e0[1] + e1[1]) * 0.5)
        if vlen(vsub(chord_mid, p_xy)) > center_tol:
            continue
        if not angle_parallel(vnorm(chord), wr.dir, 20.0):
            continue
        de0 = dist_point_to_infinite_line_xy(e0, wr.a, wr.dir)
        de1 = dist_point_to_infinite_line_xy(e1, wr.a, wr.dir)
        if max(de0, de1) > host_band:
            continue
        return True
    return False

def axis_has_opening_near_point(wr, p_xy, gap_ft, center_tol_ft=None, gap_tol_ft=None):
    if wr is None or p_xy is None:
        return False
    ax = getattr(wr, "axis", None)
    if ax is None:
        return False
    opens = list(getattr(ax, "openings", []) or [])
    if not opens:
        return False
    if center_tol_ft is None:
        center_tol_ft = max(220 * MM_TO_FT, gap_ft * 0.45)
    if gap_tol_ft is None:
        gap_tol_ft = max(140 * MM_TO_FT, gap_ft * 0.45)

    s = project_scalar_on_axis(p_xy, ax.o, ax.d)
    for (g0, g1) in opens:
        mid = (g0 + g1) * 0.5
        g = abs(g1 - g0)
        if abs(s - mid) <= center_tol_ft and abs(g - gap_ft) <= gap_tol_ft:
            return True
    return False

def opening_has_window_jamb_evidence(cad_segs, wr, p_xy, gap_ft):
    radius = max(250 * MM_TO_FT, min(700 * MM_TO_FT, gap_ft * 0.7))
    min_jamb = max(45 * MM_TO_FT, wr.axis.thick * 0.30)
    max_jamb = WIN_JAMB_MAX_LEN_FT
    count = 0
    for s in cad_segs:
        if s.len < min_jamb or s.len > max_jamb:
            continue
        if not (min(s.bb[0], s.bb[2]) - radius <= p_xy[0] <= max(s.bb[0], s.bb[2]) + radius and
                min(s.bb[1], s.bb[3]) - radius <= p_xy[1] <= max(s.bb[1], s.bb[3]) + radius):
            continue
        if dist_point_to_segment_xy(p_xy, s.a, s.b) > radius:
            continue
        if angle_parallel(s.dir, wr.dir, 70.0):
            continue
        dm = dist_point_to_infinite_line_xy(s.a, wr.a, wr.dir)
        if dm > (wr.axis.thick * 0.8 + 80 * MM_TO_FT):
            continue
        count += 1
        if count >= 2:
            return True
    return False

def door_data_from_bridge_jamb_pairs(wall_refs, existing_door_points_xy, cad_segs):
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

                    if abs(vdot(gdir, wr1.dir)) > BRIDGE_GAP_PERP_DOT_MAX:
                        continue

                    mid = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)

                    if has_near_point(existing_door_points_xy, mid, ARC_NEAR_GAP_FT):
                        continue
                    if has_near_point(local_pts, mid, 250 * MM_TO_FT):
                        continue

                    has_leaf = (opening_has_door_leaf_evidence(cad_segs, wr1, mid, gap) or
                                opening_has_door_leaf_evidence(cad_segs, wr2, mid, gap))
                    if not has_leaf:
                        continue

                    k1 = _quant_pt_key(p1)
                    k2 = _quant_pt_key(p2)
                    if k2 < k1:
                        p1, p2 = p2, p1
                        gap_vec = vsub(p2, p1)
                        gdir = vnorm(gap_vec)

                    host_wr = best_host_wall_for_opening_point(
                        wall_refs, mid, axis_dir=gdir, preferred_wr=None,
                        exterior_only=False,
                        max_dist_ft=max(HOST_SEARCH_DIST_FT, 600 * MM_TO_FT),
                        min_thick_ft=REAL_WALL_MIN_THICK_FT,
                        target_opening_gap_ft=gap
                    )
                    if host_wr is None:
                        continue
                    mid = snap_host_point_xy(host_wr, mid)

                    out.append({
                        "pt": mid,
                        "gap": gap,
                        "side": 0.0,
                        "ccw": False,
                        "source": "bridge_jambs",
                        "door_class": "interior",
                        "wall": host_wr.wall,
                        "wr": host_wr
                    })
                    local_pts.append(mid)
    return out

def door_data_from_cad_jamb_pairs(wall_refs, existing_door_points_xy, cad_segs):
    out = []
    for wr in wall_refs:
        if getattr(wr.axis, "thick", 0.0) < REAL_WALL_MIN_THICK_FT:
            continue

        bb = bbox_expand(wr.bb, DOOR_JAMB_SEARCH_RAD_FT)
        near = []
        for s in cad_segs:
            if s.len > DOOR_JAMB_MAX_LEN_FT:
                continue
            if not (bb[0] <= s.bb[2] and bb[2] >= s.bb[0] and bb[1] <= s.bb[3] and bb[3] >= s.bb[1]):
                continue
            dm = dist_point_to_infinite_line_xy(s.a, wr.a, wr.dir)
            if dm > (wr.axis.thick * 0.95 + 90 * MM_TO_FT):
                continue
            if angle_parallel(s.dir, wr.dir, 90 - DOOR_JAMB_PERP_TOL_DEG):
                continue
            near.append(s)

        if len(near) < 2:
            continue

        local = []
        for i in range(len(near)):
            for j in range(i + 1, len(near)):
                s1 = near[i]
                s2 = near[j]
                m1 = ((s1.a[0] + s1.b[0]) * 0.5, (s1.a[1] + s1.b[1]) * 0.5)
                m2 = ((s2.a[0] + s2.b[0]) * 0.5, (s2.a[1] + s2.b[1]) * 0.5)
                vec = vsub(m2, m1)
                gap = vlen(vec)
                if gap < GAP_MIN_DOOR_FT or gap > GAP_MAX_DOOR_FT:
                    continue
                if not angle_parallel(vnorm(vec), wr.dir, 12.0):
                    continue

                mid = ((m1[0] + m2[0]) * 0.5, (m1[1] + m2[1]) * 0.5)
                mid = snap_host_point_xy(wr, mid)
                if has_near_point(existing_door_points_xy, mid, DOOR_DUP_CENTER_TOL_FT):
                    continue
                if has_near_point(local, mid, DOOR_DUP_CENTER_TOL_FT):
                    continue
                if not opening_has_door_leaf_evidence(cad_segs, wr, mid, gap):
                    continue

                out.append({
                    "pt": mid,
                    "wall": wr.wall,
                    "wr": wr,
                    "side": 0.0,
                    "ccw": False,
                    "gap": gap,
                    "source": "cad_jamb_pair",
                    "door_class": "interior"
                })
                local.append(mid)
    return out

def door_data_from_arcs(arcs, wall_refs, cad_segs, existing_door_points_xy):
    out=[]
    for a in arcs:
        r=a.Radius
        if r < DOOR_ARC_R_MIN_FT or r > DOOR_ARC_R_MAX_FT:
            continue

        e0=xyz_to_xy(a.GetEndPoint(0))
        e1=xyz_to_xy(a.GetEndPoint(1))
        chord = vsub(e1, e0)
        chord_len = vlen(chord)
        if chord_len < GAP_MIN_DOOR_FT or chord_len > GAP_MAX_DOOR_FT:
            continue
        chord_dir = vnorm(chord)
        chord_mid=((e0[0]+e1[0])/2.0, (e0[1]+e1[1])/2.0)

        wr = best_host_wall_for_opening_point(
            wall_refs, chord_mid, axis_dir=chord_dir, preferred_wr=None,
            exterior_only=None,
            max_dist_ft=max(HOST_SEARCH_DIST_FT, 650 * MM_TO_FT),
            min_thick_ft=REAL_WALL_MIN_THICK_FT,
            target_opening_gap_ft=chord_len
        )
        if wr is None:
            wr = nearest_wall_for_door_arc(wall_refs, a, exterior_only=None)
        if not wr:
            continue
        if not angle_parallel(chord_dir, wr.dir, 20.0):
            continue

        o=wr.a; d=wr.dir
        t=project_scalar_on_axis(chord_mid, o, d)
        proj=vadd(o, vmul(d,t))
        proj = snap_host_point_xy(wr, proj)

        if has_near_point(existing_door_points_xy, proj, DOOR_DUP_CENTER_TOL_FT):
            continue
        if has_near_hosted_item(out, wr, proj, DOOR_DUP_CENTER_TOL_FT):
            continue

        # Arc must be geometrically tied to host wall.
        de0 = dist_point_to_infinite_line_xy(e0, wr.a, wr.dir)
        de1 = dist_point_to_infinite_line_xy(e1, wr.a, wr.dir)
        host_band = max(220 * MM_TO_FT, getattr(wr.axis, "thick", DEFAULT_WALL_THICK_FT) * 1.6 + 120 * MM_TO_FT)
        if max(de0, de1) > host_band:
            continue

        c=xyz_to_xy(a.Center)
        n=(-d[1], d[0])
        side = vdot(vsub(c, proj), n)

        v0 = vsub(e0, c)
        v1 = vsub(e1, c)
        z = cross2(v0, v1)
        ccw = (z > 0)

        if has_near_hosted_item(out, wr, proj, 200 * MM_TO_FT):
            continue

        door_w = chord_len
        if door_w < GAP_MIN_DOOR_FT:
            door_w = GAP_MIN_DOOR_FT
        elif door_w > GAP_MAX_DOOR_FT:
            door_w = GAP_MAX_DOOR_FT

        has_leaf = opening_has_door_leaf_evidence(cad_segs, wr, proj, door_w)
        has_gap = axis_has_opening_near_point(wr, proj, door_w)
        if STRICT_OPENING_HOST_MODE and STRICT_ARC_DOOR_REQUIRE_AXIS_GAP and (not has_gap) and getattr(wr.axis, "is_exterior", False):
            continue
        if (not has_leaf) and (not has_gap):
            continue

        out.append({
            "pt": proj,
            "wall": wr.wall,
            "wr": wr,
            "side": side,
            "ccw": ccw,
            "source": "arc",
            "gap": door_w,
            "door_class": ("exterior" if getattr(wr.axis, "is_exterior", False) else "interior")
        })
    return out

def door_data_from_opening_gaps(wall_refs, existing_door_points_xy, cad_segs, cad_arcs):
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
            has_arc = opening_has_door_arc_evidence(cad_arcs, wr, p, gap)

            if ax.is_exterior:
                # Exterior gap-doors must match a door-swing arc to avoid facade false positives.
                if STRICT_OPENING_HOST_MODE and (not has_arc):
                    continue
                if (not STRICT_OPENING_HOST_MODE) and (not (has_leaf or has_arc)):
                    continue
            if (not ax.is_exterior) and (not has_leaf) and gap > (1100 * MM_TO_FT):
                continue

            out.append({
                "pt": p,
                "wall": wr.wall,
                "wr": wr,
                "side": 0.0,
                "ccw": False,
                "gap": gap,
                "source": "gap",
                "door_class": ("exterior" if ax.is_exterior else "interior")
            })
    return out

# ============================================================
# Windows from gaps + curtains
# ============================================================
def openings_from_axes(wall_refs, door_points_xy, cad_segs, exterior_only=True):
    wins=[]
    curtains=[]

    for wr in wall_refs:
        ax = wr.axis
        if exterior_only and (not ax.is_exterior):
            continue

        for (g0,g1) in ax.openings:
            gap = g1-g0
            if gap < GAP_MIN_WIN_FT:
                continue
            if gap > WINDOW_GAP_MAX_STRICT_FT:
                continue
            mid=(g0+g1)*0.5
            p=vadd(ax.o, vmul(ax.d, mid))

            host_wr = best_host_wall_for_opening_point(
                wall_refs, p, axis_dir=ax.d, preferred_wr=wr,
                exterior_only=(True if exterior_only else None),
                max_dist_ft=max(HOST_SEARCH_DIST_FT, 300 * MM_TO_FT),
                min_thick_ft=WINDOW_HOST_MIN_THICK_FT,
                target_opening_gap_ft=gap
            ) or wr

            if getattr(host_wr.axis, "thick", 0.0) < WINDOW_HOST_MIN_THICK_FT:
                host_wr2 = best_host_wall_for_opening_point(
                    wall_refs, p, axis_dir=ax.d, preferred_wr=None,
                    exterior_only=(True if exterior_only else None),
                    max_dist_ft=max(HOST_SEARCH_DIST_FT, 600 * MM_TO_FT),
                    min_thick_ft=WINDOW_HOST_MIN_THICK_FT,
                    target_opening_gap_ft=gap
                )
                if host_wr2 is not None:
                    host_wr = host_wr2

            p = snap_host_point_xy(host_wr, p)

            too_near_door=False
            for dp in door_points_xy:
                if vlen(vsub(dp,p)) <= WINDOW_NEAR_DOOR_CLEARANCE_FT:
                    too_near_door=True; break
            if too_near_door:
                continue

            host_len = max(vlen(vsub(host_wr.a, host_wr.b)), 1e-9)
            t_host = segment_param_xy(p, host_wr.a, host_wr.b)
            if min(t_host, 1.0 - t_host) * host_len < (180 * MM_TO_FT):
                continue

            if not opening_has_window_jamb_evidence(cad_segs, host_wr, p, gap):
                continue

            if gap >= CURTAIN_THRESHOLD_FT:
                if not is_duplicate_opening(curtains, host_wr, p, gap, pt_tol_ft=WINDOW_DUP_CENTER_TOL_FT, gap_tol_ft=220 * MM_TO_FT):
                    curtains.append({"pt": p, "wall": host_wr.wall, "wr": host_wr, "gap": gap})
            else:
                if gap <= GAP_MAX_WIN_FT:
                    if not is_duplicate_opening(wins, host_wr, p, gap, pt_tol_ft=WINDOW_DUP_CENTER_TOL_FT, gap_tol_ft=OPENING_DUP_GAP_TOL_FT):
                        wins.append({"pt": p, "wall": host_wr.wall, "wr": host_wr, "gap": gap, "source": "gap"})

    return wins, curtains

# ============================================================
# V5.2: Window jamb-pair detection (symbol-based)
# ============================================================
def windows_from_jamb_pairs(wall_refs, cad_segs, door_points_xy):
    by_wall = defaultdict(list)
    target_gap = 1200 * MM_TO_FT
    for wr in wall_refs:
        if not wr.axis.is_exterior:
            continue
        if getattr(wr.axis, "thick", 0.0) < REAL_WALL_MIN_THICK_FT:
            continue

        bb = bbox_expand(wr.bb, WIN_JAMB_SEARCH_RAD_FT)
        near = []
        for s in cad_segs:
            if s.len > WIN_JAMB_MAX_LEN_FT:
                continue
            if s.len < max(40 * MM_TO_FT, wr.axis.thick * 0.45):
                continue
            if not (bb[0] <= s.bb[2] and bb[2] >= s.bb[0] and bb[1] <= s.bb[3] and bb[3] >= s.bb[1]):
                continue

            dm = dist_point_to_infinite_line_xy(s.a, wr.a, wr.dir)
            if dm > (wr.axis.thick * 0.75 + 60 * MM_TO_FT):
                continue

            # want jamb ~perpendicular to wall
            if angle_parallel(s.dir, wr.dir, 90 - WIN_JAMB_PERP_TOL_DEG):
                continue

            near.append(s)

        if len(near) < 2:
            continue

        for i in range(len(near)):
            for j in range(i+1, len(near)):
                s1 = near[i]; s2 = near[j]
                if abs(s1.len - s2.len) > max(120 * MM_TO_FT, 0.8 * wr.axis.thick):
                    continue
                m1 = ((s1.a[0]+s1.b[0])*0.5, (s1.a[1]+s1.b[1])*0.5)
                m2 = ((s2.a[0]+s2.b[0])*0.5, (s2.a[1]+s2.b[1])*0.5)

                vec = vsub(m2, m1)
                gap = vlen(vec)
                if gap < WIN_GAP_MIN_FT or gap > WIN_GAP_MAX_FT:
                    continue

                gdir = vnorm(vec)
                if not angle_parallel(gdir, wr.dir, 10.0):
                    continue

                mid = ((m1[0]+m2[0])*0.5, (m1[1]+m2[1])*0.5)
                mid = snap_host_point_xy(wr, mid)

                too_near_door=False
                for dp in door_points_xy:
                    if vlen(vsub(dp, mid)) <= ARC_NEAR_GAP_FT:
                        too_near_door=True; break
                if too_near_door:
                    continue

                wall_key = getattr(wr, "wall_id", id(wr))
                if is_duplicate_opening(by_wall.get(wall_key, []), wr, mid, gap, pt_tol_ft=WINDOW_DUP_CENTER_TOL_FT, gap_tol_ft=OPENING_DUP_GAP_TOL_FT):
                    continue

                wall_len = max(vlen(vsub(wr.b, wr.a)), 1e-9)
                t_along = segment_param_xy(mid, wr.a, wr.b) * wall_len
                score = min(s1.len, s2.len) - (0.25 * abs(gap - target_gap))
                by_wall[wall_key].append({
                    "pt": mid,
                    "wall": wr.wall,
                    "wr": wr,
                    "gap": gap,
                    "source": "jamb_pair",
                    "_t": t_along,
                    "_score": score
                })

    wins = []
    for _, cands in by_wall.items():
        cands = sorted(cands, key=lambda c: c.get("_score", 0.0), reverse=True)
        kept = []
        for c in cands:
            too_close = False
            for k in kept:
                if abs(c.get("_t", 0.0) - k.get("_t", 0.0)) < WINDOW_JAMB_CLUSTER_SPACING_FT:
                    too_close = True
                    break
            if too_close:
                continue
            kept.append(c)
        wins.extend(kept)

    if len(wins) > WINDOW_JAMB_MAX_TOTAL:
        wins = sorted(wins, key=lambda c: c.get("_score", 0.0), reverse=True)[:WINDOW_JAMB_MAX_TOTAL]

    cleaned = []
    for w in wins:
        w2 = dict(w)
        if "_t" in w2:
            del w2["_t"]
        if "_score" in w2:
            del w2["_score"]
        cleaned.append(w2)

    if cleaned:
        print("Windows inferred from jamb-pairs: %d" % len(cleaned))
    return cleaned

# ============================================================
# Symbol selection by width
# ============================================================
def get_symbols(bic):
    return list(FilteredElementCollector(doc).OfClass(FamilySymbol).OfCategory(bic))

def symbol_width_ft(sym):
    param_names = ["Width", "Rough Width", "Nominal Width"]
    for pn in param_names:
        try:
            p = sym.LookupParameter(pn)
            if p and p.StorageType == StorageType.Double:
                return p.AsDouble()
        except Exception:
            continue
    return None

def symbol_text(sym):
    parts = []
    try:
        parts.append((sym.FamilyName or "").lower())
    except Exception:
        pass
    try:
        parts.append((_safe_elem_name(sym) or "").lower())
    except Exception:
        pass
    return " ".join(parts).lower()

def normalize_token_text(txt):
    if txt is None:
        return ""
    try:
        s = txt.lower()
    except Exception:
        try:
            s = str(txt).lower()
        except Exception:
            return ""
    bad = [u"\u200e", u"\u200f", u"\u202a", u"\u202b", u"\u202c", u"\u202d", u"\u202e", "-", "_", " ", "\t", "\r", "\n"]
    for b in bad:
        s = s.replace(b, "")
    return s

def symbol_matches_preferred_door(sym):
    txt = symbol_text(sym)
    n = normalize_token_text(txt)
    pref = normalize_token_text(DOOR_PREFERRED_FAMILY_TOKEN)
    if pref and (pref in n):
        return True
    pref_rev = normalize_token_text(DOOR_PREFERRED_FAMILY_TOKEN[::-1])
    if pref_rev and (pref_rev in n):
        return True
    # Robust Hebrew fallback for bidi/reversed storage.
    if ((u"עץ" in txt or u"ץע" in txt) and (u"ציר" in txt or u"ריצ" in txt)):
        return True
    return False

def log_door_symbol_inventory(symbols, limit=20):
    try:
        uniq = []
        seen = set()
        for s in symbols:
            t = symbol_text(s)
            if not t or t in seen:
                continue
            seen.add(t)
            uniq.append(t)
            if len(uniq) >= limit:
                break
        if uniq:
            print("Door families/types available (sample): %s" % uniq)
    except Exception:
        pass

def get_symbol_by_width(symbols, target_ft):
    if not symbols:
        return None
    best = None
    bestd = 1e9
    for s in symbols:
        w = symbol_width_ft(s)
        if w is None:
            continue
        d = abs(w - target_ft)
        if d < bestd:
            bestd = d
            best = s
    if best is not None:
        return best
    return symbols[0]

def ensure_active(sym):
    if sym and not sym.IsActive:
        sym.Activate()
        doc.Regenerate()

def symbol_is_opening_only(sym):
    txt = symbol_text(sym)
    return ("opening" in txt) or ("void" in txt) or (u"פתח" in txt)

def symbol_is_bad_door(sym):
    txt = symbol_text(sym)
    for tok in DOOR_REJECT_NAME_TOKENS:
        if tok in txt:
            return True
    w = symbol_width_ft(sym)
    if (w is not None) and (w > DOOR_MAX_WIDTH_FT):
        return True
    return False

def choose_door_symbol(symbols, target_ft):
    if not symbols:
        return None

    candidates = [s for s in symbols if (not symbol_is_opening_only(s)) and (not symbol_is_bad_door(s))]
    if not candidates:
        candidates = [s for s in symbols if not symbol_is_bad_door(s)]
    if not candidates:
        candidates = list(symbols)

    best = get_symbol_by_width(candidates, target_ft)
    return best if best is not None else candidates[0]

def choose_door_symbol_candidates(symbols, target_ft):
    if not symbols:
        return []
    candidates = [s for s in symbols if (not symbol_is_opening_only(s)) and (not symbol_is_bad_door(s))]
    if not candidates:
        candidates = [s for s in symbols if not symbol_is_bad_door(s)]
    if not candidates:
        candidates = list(symbols)

    preferred = []
    others = []
    for s in candidates:
        if symbol_matches_preferred_door(s):
            preferred.append(s)
        else:
            others.append(s)

    preferred = sorted(preferred, key=lambda s: abs((symbol_width_ft(s) or target_ft) - target_ft))
    others = sorted(others, key=lambda s: abs((symbol_width_ft(s) or target_ft) - target_ft))

    ordered = preferred + others
    if preferred:
        print("Door symbol preference: using '%s' family first." % DOOR_PREFERRED_FAMILY_TOKEN)
    else:
        print("Door symbol preference '%s' not found; using fallback door families." % DOOR_PREFERRED_FAMILY_TOKEN)
        if DOOR_DEBUG_LIST_ON_PREFERRED_MISS:
            log_door_symbol_inventory(candidates, limit=15)
    return ordered

def door_point_on_host_with_clearance(wr, pt_xy, door_w_ft):
    if wr is None:
        return None
    host_len = max(vlen(vsub(wr.b, wr.a)), 1e-9)
    min_clear = 2 * MM_TO_FT
    if host_len <= min_clear:
        return None
    extra = max(host_len - door_w_ft, 0.0)
    avail_clear = 0.5 * extra
    if avail_clear <= 0.0:
        return vadd(wr.a, vmul(vsub(wr.b, wr.a), 0.5))
    desired = max(DOOR_END_CLEARANCE_FT, 0.12 * door_w_ft)
    clear = min(desired, max(min_clear, avail_clear - (1 * MM_TO_FT)))

    t = segment_param_xy(pt_xy, wr.a, wr.b)
    tmin = clear / host_len
    tmax = 1.0 - tmin
    if t < tmin:
        t = tmin
    if t > tmax:
        t = tmax
    return vadd(wr.a, vmul(vsub(wr.b, wr.a), t))

def set_instance_param_ft(fi, bip, val_ft):
    try:
        p = fi.get_Parameter(bip)
        if p and (not p.IsReadOnly):
            p.Set(val_ft)
            return True
    except Exception:
        pass
    return False

def set_door_base_offset(fi):
    if set_instance_param_ft(fi, BuiltInParameter.INSTANCE_SILL_HEIGHT_PARAM, DEFAULT_DOOR_BASE_OFFSET_FT):
        return
    if set_instance_param_ft(fi, BuiltInParameter.INSTANCE_FREE_HOST_OFFSET_PARAM, DEFAULT_DOOR_BASE_OFFSET_FT):
        return
    p = fi.LookupParameter("Sill Height") or fi.LookupParameter("Offset from Level") or fi.LookupParameter("Base Offset")
    if p and (not p.IsReadOnly):
        try:
            p.Set(DEFAULT_DOOR_BASE_OFFSET_FT)
        except Exception:
            pass

def set_window_sill(fi):
    if set_instance_param_ft(fi, BuiltInParameter.INSTANCE_SILL_HEIGHT_PARAM, DEFAULT_SILL_FT):
        return
    p = fi.LookupParameter("Sill Height") or fi.LookupParameter("Sill")
    if p and (not p.IsReadOnly):
        try:
            p.Set(DEFAULT_SILL_FT)
        except Exception:
            pass

# ============================================================
# Place Doors / Windows
# ============================================================
def place_doors(door_data, level):
    symbols = get_symbols(BuiltInCategory.OST_Doors)
    if not symbols:
        raise Exception("No Door FamilySymbol in project.")
    sym_default = symbols[0]

    t=Transaction(doc, "V5.2 Place Doors")
    t.Start()
    placed=0
    try:
        for d in door_data:
            try:
                target_w = d.get("gap", 900*MM_TO_FT)
                door_candidates = choose_door_symbol_candidates(symbols, target_w)
                if not door_candidates:
                    door_candidates = [sym_default]

                wr = d.get("wr", None)
                host_wall = d.get("wall", None)

                if host_wall is None:
                    continue

                pt_xy = d["pt"]
                if wr is not None:
                    pt_xy = snap_host_point_xy(wr, pt_xy)
                    pt_adj = door_point_on_host_with_clearance(wr, pt_xy, target_w)
                    if pt_adj is None:
                        print("Skipping door (%s): host wall too short for %.0fmm" % (d.get("source", "?"), target_w * 304.8))
                        continue
                    pt_xy = pt_adj
                pt = xy_to_xyz(pt_xy, level.Elevation)

                fi = None
                last_err = None
                for sym in door_candidates[:8]:
                    try:
                        ensure_active(sym)
                        fi = doc.Create.NewFamilyInstance(pt, sym, host_wall, level, StructuralType.NonStructural)
                        break
                    except Exception as e_try:
                        last_err = e_try
                        fi = None
                        continue
                if fi is None:
                    raise Exception("no compatible door type for %.0fmm (%s)" % (target_w * 304.8, str(last_err)))
                set_door_base_offset(fi)

                try:
                    if d.get("side", 0.0) > 0:
                        fi.flipFacing()
                except Exception:
                    pass

                try:
                    if d.get("ccw", False):
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

def place_windows(win_data, level):
    symbols = get_symbols(BuiltInCategory.OST_Windows)
    if not symbols:
        raise Exception("No Window FamilySymbol in project.")
    sym_default = symbols[0]

    t=Transaction(doc, "V5.2 Place Windows")
    t.Start()
    placed=0
    try:
        for w in win_data:
            sym = get_symbol_by_width(symbols, w.get("gap", 1200*MM_TO_FT)) or sym_default
            ensure_active(sym)

            pt_xy = w["pt"]
            wr = w.get("wr", None)
            if wr is not None:
                pt_xy = snap_host_point_xy(wr, pt_xy)
            pt = xy_to_xyz(pt_xy, level.Elevation)

            fi = doc.Create.NewFamilyInstance(pt, sym, w["wall"], level, StructuralType.NonStructural)
            set_window_sill(fi)

            placed += 1
        t.Commit()
    except Exception as e:
        t.RollBack()
        print("place_windows failed: %s" % str(e))
        placed = 0
    return placed

# ============================================================
# Curtain Walls
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

    t=Transaction(doc, "V5.2 Create Curtain Walls")
    t.Start()
    created=0
    try:
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
print("Running %s" % BUILD_ID)
if len(sel) != 1:
    raise Exception("Select one ImportInstance (CAD link) and run again.")

imp = doc.GetElement(sel[0])
if not isinstance(imp, ImportInstance):
    raise Exception("Selection is not an ImportInstance.")

raw_lines, raw_arcs = extract_lines_and_arcs(imp)
segs = [Seg(l) for l in raw_lines if l.Length >= MIN_RAW_LEN_FT]

_diag_path = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "c2rv5_2_diag.txt")
try:
    with open(_diag_path, "w") as _f:
        _f.write("Raw lines=%d arcs=%d segs=%d\n" % (len(raw_lines), len(raw_arcs), len(segs)))
except Exception:
    pass

pairs = find_parallel_pairs(segs)
if pairs:
    clusters, raw_groups = select_wall_thickness_clusters(pairs)
    if not clusters:
        raise Exception("No valid wall-thickness clusters found.")
    print("Thickness groups (mm,count): %s" % [("%0.1f" % (g["rep"] * 304.8), g["count"]) for g in raw_groups])
    print("Selected wall thickness clusters (mm): %s" % [round(c * 304.8, 1) for c in clusters])

    center_segs = []
    for a, b, t in pairs:
        snap = snap_to_cluster(t, clusters)
        if abs(t - snap) > PAIR_THICK_SNAP_TOL_FT:
            continue
        cs = build_center_from_pair(a, b, snap)
        if cs:
            center_segs.append(cs)

    axes = build_wall_axes(center_segs)
    print("Axes before dedup: %d" % len(axes))
    axes = dedup_overlapping_axes(axes)
    if ENABLE_AXIS_SPLIT_AT_INTERSECTIONS:
        axes = split_axes_at_intersections(axes)
    axes = filter_orphan_short_axes(axes, ORPHAN_AXIS_MIN_LEN_FT)
    axes = merge_collinear_axes_for_creation(axes)
    axes = suppress_overlapping_axes(axes)
else:
    if not ENABLE_SINGLE_LINE_FALLBACK:
        raise Exception("No valid double-line wall pairs found. Check CAD scale/layers or enable single-line fallback.")
    print("No wall pairs found. Falling back to single-line axis mode.")
    clusters = [SINGLE_LINE_DEFAULT_THICK_FT]
    axes = add_single_line_interior_axes(segs, [])
    axes = dedup_overlapping_axes(axes)
    if ENABLE_AXIS_SPLIT_AT_INTERSECTIONS:
        axes = split_axes_at_intersections(axes)
    axes = filter_orphan_short_axes(axes, ORPHAN_AXIS_MIN_LEN_FT)
    axes = merge_collinear_axes_for_creation(axes)
    axes = suppress_overlapping_axes(axes)

if not axes:
    raise Exception("No candidate wall axes found. Check CAD import scale and linework quality.")

print("Axes after pipeline (+single-line): %d" % len(axes))
try:
    ax_hist = defaultdict(int)
    for a in axes:
        ax_hist[int(round(a.thick * 304.8))] += 1
    print("Axis thickness histogram (mm->count): %s" % sorted(ax_hist.items()))
except Exception:
    pass

# classify exterior
classify_exterior_axes_v5(axes)
normalize_exterior_wall_thickness(axes, clusters)

# level from view
view = doc.ActiveView
level = getattr(view, "GenLevel", None)
if level is None:
    level = FilteredElementCollector(doc).OfClass(Level).FirstElement()

# previews
if PREVIEW_WALL_AXES:
    draw_axes(axes, "V5.2 Preview Wall Axes")

if PREVIEW_EXTERIOR_AXES:
    ext = [a for a in axes if a.is_exterior]
    draw_axes(ext, "V5.2 Preview Exterior Axes")

# create walls
walls_axes_and_walls=[]
if BUILD_WALLS:
    walls_axes_and_walls = create_walls(axes, level, DEFAULT_WALL_HEIGHT_FT)
    print("Walls created: %d" % len(walls_axes_and_walls))

wall_refs = []
invalid_wall_refs = 0
for (axis, wall) in walls_axes_and_walls:
    try:
        wall_refs.append(WallRef2D(axis, wall))
    except Exception as ex:
        invalid_wall_refs += 1
        print("Skipping invalid wall ref: %s" % str(ex))
if invalid_wall_refs:
    print("Skipped %d wall refs due to invalid objects." % invalid_wall_refs)
if not wall_refs:
    raise Exception("No valid walls available for opening hosting.")

# doors (reliable sources first, arcs as fallback)
door_data = []
gap_door_data = door_data_from_opening_gaps(wall_refs, [], segs, raw_arcs)
if gap_door_data:
    door_data.extend(gap_door_data)
    print("Doors inferred from gaps (arc missed): %d" % len(gap_door_data))
bridge_door_data = door_data_from_bridge_jamb_pairs(wall_refs, [d["pt"] for d in door_data], segs)
if bridge_door_data:
    door_data.extend(bridge_door_data)
    print("Doors inferred from jamb pairs (recessed openings): %d" % len(bridge_door_data))
cad_jamb_door_data = door_data_from_cad_jamb_pairs(wall_refs, [d["pt"] for d in door_data], segs)
if cad_jamb_door_data:
    door_data.extend(cad_jamb_door_data)
    print("Doors inferred from CAD jamb pairs: %d" % len(cad_jamb_door_data))
arc_door_data = door_data_from_arcs(raw_arcs, wall_refs, segs, [d["pt"] for d in door_data])
if arc_door_data:
    door_data.extend(arc_door_data)
    print("Doors inferred from arcs: %d" % len(arc_door_data))
door_data = dedupe_openings(door_data, "door")
door_data = rehost_and_validate_doors(door_data, wall_refs)
door_data = dedupe_openings(door_data, "door")
door_pts = [d["pt"] for d in door_data]

if PREVIEW_DOOR_POINTS:
    draw_points(door_pts, level.Elevation, "V5.2 Preview Door Points")

if BUILD_DOORS and door_data:
    n = place_doors(door_data, level)
    print("Doors placed: %d" % n)

# windows + curtains from gaps first
ext_openings = sum(len(a.openings) for a in axes if a.is_exterior)
ext_count = sum(1 for a in axes if a.is_exterior)

if ext_openings == 0:
    print("WARNING: exterior openings = 0 (DWG probably uses window symbols). Skipping gap-windows and trying jamb fallback.")
    win_data, curtain_data = [], []
else:
    win_data, curtain_data = openings_from_axes(wall_refs, door_pts, segs, exterior_only=True)

# V5.2: if no window found from geometric gaps, try jamb-pair detection as fallback
if ENABLE_WINDOW_JAMB_PAIR_FALLBACK and len(win_data) == 0:
    win_data2 = windows_from_jamb_pairs(wall_refs, segs, door_pts)
    win_data.extend(win_data2)

win_data = dedupe_openings(win_data, "window")
win_data, curtain_data = resolve_opening_conflicts(door_data, win_data, curtain_data)

win_pts = [w["pt"] for w in win_data]
if PREVIEW_WINDOW_POINTS:
    draw_points(win_pts, level.Elevation, "V5.2 Preview Window Points")

if BUILD_WINDOWS and win_data:
    n = place_windows(win_data, level)
    print("Windows placed: %d" % n)

if BUILD_CURTAIN_WALLS and curtain_data:
    n = create_curtain_walls(curtain_data, level)
    print("Curtain walls created: %d" % n)

int_count = sum(1 for a in axes if not a.is_exterior)
print("Exterior axes: %d / %d total (with %d openings)" % (ext_count, len(axes), ext_openings))
print("Interior axes: %d" % int_count)
print("Windows found: %d, Curtains: %d" % (len(win_data), len(curtain_data)))
print("Thickness clusters (mm): %s" % [round(c*304.8,1) for c in clusters])
print("Done V5.2.")
