# -*- coding: utf-8 -*-
"""
apartment_planner.py  —  ApartmentPlanner module for C2Rv7

Reads an existing Revit Area (apartment boundary), runs a Rule Engine to
determine the room programme, partitions the polygon geometrically, and
draws Model Lines in the active Revit view.

Usage (called from script.py after STEP 3.5):
    from apartment_planner import run_apartment_planner
    run_apartment_planner(v2, level, area_element, snapshot=snapshot)

Design rules (mirrors c2rv7_llm_qa.py style):
  - Pure stdlib + IronPython-safe (no shapely, no numpy).
  - Never raises into the host script.
  - All decisions logged to snapshot.
  - Units: internal geometry in centimetres (cm).
    Revit API calls use feet (multiply cm * CM_TO_FEET).

Israeli Residential Building Code rules encoded:
  - Room min area  : 9.00 m²  (90,000 cm²)
  - Room min width : 260 cm
  - Master bedroom : min area 12.00 m² (120,000 cm²), min width 320 cm
  - Living room    : min width 350 cm, near entrance
  - Bathroom       : min area 4.00 m² (40,000 cm²), min width 180 cm
  - En-suite       : 140 x 240 cm
  - Guest WC       : 90 x 140 cm  (3+ bedroom apartments only)
  - Apartment size → room count table (see ROOM_TABLE)
"""

import math
import os

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CM_TO_FEET = 1.0 / 30.48

# Room-count table: (min_sqm, max_sqm, bedrooms)
# bedrooms = total sleeping rooms INCLUDING master
ROOM_TABLE = [
    (0,   50,  1),   # studio/1-room
    (50,  75,  2),   # 2-room
    (75,  100, 3),   # 3-room
    (100, 130, 4),   # 4-room
    (130, 999, 5),   # 5-room
]

# Minimum areas / widths in cm
MIN_ROOM_AREA_CM2    = 9.0   * 10000   # 9 m²
MIN_ROOM_WIDTH_CM    = 260.0
MIN_MASTER_AREA_CM2  = 12.0  * 10000   # 12 m²
MIN_MASTER_WIDTH_CM  = 320.0
MIN_LIVING_WIDTH_CM  = 350.0
MIN_BATH_AREA_CM2    = 4.0   * 10000   # 4 m²
MIN_BATH_WIDTH_CM    = 180.0
ENSUITE_W_CM         = 140.0
ENSUITE_D_CM         = 240.0
GUEST_WC_W_CM        = 90.0
GUEST_WC_D_CM        = 140.0
SHAFT_AREA_CM2       = 2500.0          # 0.25 m²  (50x50 cm shaft)
PANTRY_MIN_AREA_CM2  = 1.0 * 10000    # 1 m² — optional

# Zones
ZONE_PUBLIC  = "public"   # living + kitchen + dining
ZONE_PRIVATE = "private"  # bedrooms + bathrooms


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log(snapshot, msg):
    if snapshot is None:
        return
    try:
        snapshot.log("[ApartmentPlanner] " + str(msg))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Geometry helpers  (all in cm, IronPython-safe)
# ---------------------------------------------------------------------------

def _pt(x, y):
    return (float(x), float(y))


def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1]


def _cross2d(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _length(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1])


def _norm(v):
    l = _length(v)
    if l < 1e-9:
        return (0.0, 0.0)
    return (v[0] / l, v[1] / l)


def _poly_area_signed(pts):
    """Signed area via shoelace (positive = CCW)."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += pts[i][0] * pts[j][1]
        s -= pts[j][0] * pts[i][1]
    return s * 0.5


def _poly_area(pts):
    return abs(_poly_area_signed(pts))


def _poly_centroid(pts):
    n = len(pts)
    if n == 0:
        return (0.0, 0.0)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    return (cx, cy)


def _bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _poly_width_height(pts):
    minx, miny, maxx, maxy = _bbox(pts)
    return (maxx - minx), (maxy - miny)


def _ensure_ccw(pts):
    if _poly_area_signed(pts) < 0:
        return list(reversed(pts))
    return list(pts)


def _point_in_poly(pt, pts):
    """Ray-casting point-in-polygon."""
    x, y = pt
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _clip_poly_by_line(pts, lx1, ly1, lx2, ly2):
    """
    Sutherland-Hodgman clip: keep the LEFT side of the directed line
    from (lx1,ly1) to (lx2,ly2).
    """
    def inside(p):
        return _cross2d(_vec((lx1, ly1), (lx2, ly2)),
                        _vec((lx1, ly1), p)) >= 0.0

    def intersect(a, b):
        da = _vec((lx1, ly1), a)
        db = _vec((lx1, ly1), b)
        ca = _cross2d(_vec((lx1, ly1), (lx2, ly2)), da)
        cb = _cross2d(_vec((lx1, ly1), (lx2, ly2)), db)
        denom = cb - ca
        if abs(denom) < 1e-9:
            return a
        t = -ca / denom
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    output = list(pts)
    for i in range(len(pts)):
        if not output:
            break
        input_pts = output
        output = []
        s = input_pts[-1]
        for e in input_pts:
            if inside(e):
                if not inside(s):
                    output.append(intersect(s, e))
                output.append(e)
            elif inside(s):
                output.append(intersect(s, e))
            s = e
    return output


def _split_poly_horizontal(pts, y_cut):
    """Split polygon into (bottom_pts, top_pts) at horizontal line y=y_cut."""
    # clip below: keep where y <= y_cut  → left of line going RIGHT at y_cut
    bottom = _clip_poly_by_line(pts,
                                pts[0][0] - 1e6, y_cut,
                                pts[0][0] + 1e6, y_cut)
    # clip above: keep where y >= y_cut  → left of line going LEFT at y_cut
    top = _clip_poly_by_line(pts,
                             pts[0][0] + 1e6, y_cut,
                             pts[0][0] - 1e6, y_cut)
    return bottom, top


def _split_poly_vertical(pts, x_cut):
    """Split polygon into (left_pts, right_pts) at vertical line x=x_cut."""
    left = _clip_poly_by_line(pts,
                              x_cut, pts[0][1] - 1e6,
                              x_cut, pts[0][1] + 1e6)
    right = _clip_poly_by_line(pts,
                               x_cut, pts[0][1] + 1e6,
                               x_cut, pts[0][1] - 1e6)
    return left, right


def _poly_min_width(pts):
    """Approximate minimum width via rotating calipers (bbox of rotated polygon)."""
    if len(pts) < 3:
        return 0.0
    min_w = 1e12
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        edge = _norm(_vec(pts[i], pts[j]))
        perp = (-edge[1], edge[0])
        proj = [_dot(p, perp) for p in pts]
        w = max(proj) - min(proj)
        if w < min_w:
            min_w = w
    return min_w


# ---------------------------------------------------------------------------
# Revit Area  →  polygon in cm
# ---------------------------------------------------------------------------

def _area_polygon_cm(area_element):
    """
    Extract the boundary polygon of a Revit Area as a list of (x_cm, y_cm).
    Returns list of points or None on failure.
    Uses SpatialElementBoundaryOptions + GetBoundarySegments.
    Coordinates come from Revit in FEET → convert to cm.
    """
    try:
        from Autodesk.Revit.DB import SpatialElementBoundaryOptions, SpatialElementBoundaryLocation
        opts = SpatialElementBoundaryOptions()
        opts.SpatialElementBoundaryLocation = SpatialElementBoundaryLocation.Center
        boundary_loops = area_element.GetBoundarySegments(opts)
        if not boundary_loops or boundary_loops.Count == 0:
            return None
        # Take the outermost loop (largest area)
        best_pts = None
        best_area = -1.0
        for loop in boundary_loops:
            pts_cm = []
            for seg in loop:
                c = seg.GetCurve().GetEndPoint(0)
                pts_cm.append((c.X * 30.48, c.Y * 30.48))
            a = _poly_area(pts_cm)
            if a > best_area:
                best_area = a
                best_pts = pts_cm
        return best_pts
    except Exception:
        return None


def _area_sqm(area_element):
    """Return area in m² directly from Revit Area parameter (more reliable)."""
    try:
        from Autodesk.Revit.DB import BuiltInParameter
        param = area_element.get_Parameter(BuiltInParameter.ROOM_AREA)
        if param is not None:
            sq_feet = param.AsDouble()
            return sq_feet * 0.0929  # ft² → m²
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Rule Engine  —  determine room programme
# ---------------------------------------------------------------------------

class RoomProgramme(object):
    """
    Holds the decided room list for one apartment.
    Each room is a dict:
        {
          "name": str,
          "zone": "public" | "private",
          "min_area_cm2": float,
          "min_width_cm": float,
          "needs_exterior_wall": bool,   # must touch boundary
          "needs_window": bool,
          "notes": str,
        }
    """
    def __init__(self, rooms, total_sqm, n_bedrooms):
        self.rooms = rooms
        self.total_sqm = total_sqm
        self.n_bedrooms = n_bedrooms

    def __repr__(self):
        names = [r["name"] for r in self.rooms]
        return "RoomProgramme({}sqm, rooms={})".format(
            round(self.total_sqm, 1), names)


def _bedroom_count(sqm):
    for min_s, max_s, beds in ROOM_TABLE:
        if min_s <= sqm < max_s:
            return beds
    return ROOM_TABLE[-1][2]


def build_programme(sqm, snapshot=None):
    """
    Given apartment area in m², return a RoomProgramme.
    """
    n_bedrooms = _bedroom_count(sqm)
    _log(snapshot, "Area={:.1f}sqm -> {} bedrooms".format(sqm, n_bedrooms))

    rooms = []

    # ---- PUBLIC ZONE ----

    # Living room
    rooms.append({
        "name": u"סלון",   # סלון
        "zone": ZONE_PUBLIC,
        "min_area_cm2": MIN_LIVING_WIDTH_CM * MIN_LIVING_WIDTH_CM,  # ~12 m²
        "min_width_cm": MIN_LIVING_WIDTH_CM,
        "needs_exterior_wall": True,
        "needs_window": True,
        "notes": u"קרוב לכניסה",  # קרוב לכניסה
    })

    # Kitchen
    rooms.append({
        "name": u"מטבח",   # מטבח
        "zone": ZONE_PUBLIC,
        "min_area_cm2": 7.0 * 10000,
        "min_width_cm": 220.0,
        "needs_exterior_wall": True,
        "needs_window": True,
        "notes": u"צורת L, פתוח לסלון",
    })

    # Dining
    rooms.append({
        "name": u"פינת אוכל",   # פינת אוכל
        "zone": ZONE_PUBLIC,
        "min_area_cm2": 4.0 * 10000,
        "min_width_cm": 200.0,
        "needs_exterior_wall": False,
        "needs_window": False,
        "notes": u"פתוח, קשור למטבח/סלון",
    })

    # ---- PRIVATE ZONE ----

    # Master bedroom
    rooms.append({
        "name": u"חדר הורים",   # חדר הורים
        "zone": ZONE_PRIVATE,
        "min_area_cm2": MIN_MASTER_AREA_CM2,
        "min_width_cm": MIN_MASTER_WIDTH_CM,
        "needs_exterior_wall": True,
        "needs_window": True,
        "notes": u"כולל שירותים צמודים",
    })
    # En-suite
    rooms.append({
        "name": u"שירותי הורים",   # שירותי הורים
        "zone": ZONE_PRIVATE,
        "min_area_cm2": ENSUITE_W_CM * ENSUITE_D_CM,
        "min_width_cm": ENSUITE_W_CM,
        "needs_exterior_wall": False,
        "needs_window": False,
        "notes": "ensuite {}x{}cm".format(int(ENSUITE_W_CM), int(ENSUITE_D_CM)),
    })

    # Additional bedrooms
    for i in range(n_bedrooms - 1):
        rooms.append({
            "name": u"חדר {}".format(i + 1),   # חדר N
            "zone": ZONE_PRIVATE,
            "min_area_cm2": MIN_ROOM_AREA_CM2,
            "min_width_cm": MIN_ROOM_WIDTH_CM,
            "needs_exterior_wall": True,
            "needs_window": True,
            "notes": "",
        })

    # Mark last bedroom as saferoom (ממ"ד)
    for r in reversed(rooms):
        if r["zone"] == ZONE_PRIVATE and r["name"].startswith(u"חדר"):
            r["name"] += u' (ממ"ד)'
            r["notes"] += u" | ממ\"ד"
            break

    # Main bathroom
    rooms.append({
        "name": u"חדר אמבטיה",   # חדר אמבטיה
        "zone": ZONE_PRIVATE,
        "min_area_cm2": MIN_BATH_AREA_CM2,
        "min_width_cm": MIN_BATH_WIDTH_CM,
        "needs_exterior_wall": False,
        "needs_window": False,
        "notes": u"עדיפות לחלון חיצוני, אחרת פיר 50x50",
    })

    # Guest WC — only for 3+ bedrooms
    if n_bedrooms >= 3:
        rooms.append({
            "name": u"שירותי אורחים",   # שירותי אורחים
            "zone": ZONE_PUBLIC,
            "min_area_cm2": GUEST_WC_W_CM * GUEST_WC_D_CM,
            "min_width_cm": GUEST_WC_W_CM,
            "needs_exterior_wall": False,
            "needs_window": False,
            "notes": "{}x{}cm".format(int(GUEST_WC_W_CM), int(GUEST_WC_D_CM)),
        })

    # Optional pantry — only if area > 90 m²
    if sqm > 90:
        rooms.append({
            "name": u"מזווה",   # מזווה
            "zone": ZONE_PUBLIC,
            "min_area_cm2": PANTRY_MIN_AREA_CM2,
            "min_width_cm": 80.0,
            "needs_exterior_wall": False,
            "needs_window": False,
            "notes": u"אופציונלי, ליד מטבח",
        })

    _log(snapshot, "Programme: {}".format([r["name"] for r in rooms]))
    return RoomProgramme(rooms, sqm, n_bedrooms)


# ---------------------------------------------------------------------------
# Space Partitioner  —  divide polygon into room sub-polygons
# ---------------------------------------------------------------------------

class PartitionResult(object):
    def __init__(self):
        self.rooms = []          # list of {"name":, "poly": [(x,y)...], "area_cm2":}
        self.cut_lines = []      # list of {"x1","y1","x2","y2"}  in cm
        self.warnings = []

    def add_room(self, name, poly):
        area = _poly_area(poly)
        self.rooms.append({"name": name, "poly": poly, "area_cm2": area})

    def add_cut(self, x1, y1, x2, y2):
        self.cut_lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    def warn(self, msg):
        self.warnings.append(msg)


def _find_best_split(poly, target_area_cm2, prefer_vertical=True):
    """
    Binary-search for a horizontal or vertical cut line that gives a sub-polygon
    of approximately target_area_cm2.
    Returns (cut_coord, is_vertical) or None.
    """
    if not poly or len(poly) < 3:
        return None
    total = _poly_area(poly)
    if total < 1.0:
        return None
    ratio = target_area_cm2 / total
    ratio = max(0.05, min(0.95, ratio))

    minx, miny, maxx, maxy = _bbox(poly)
    w = maxx - minx
    h = maxy - miny

    def area_left_of_cut(coord, vertical):
        if vertical:
            left, _ = _split_poly_vertical(poly, coord)
            return _poly_area(left) if left else 0.0
        else:
            bottom, _ = _split_poly_horizontal(poly, coord)
            return _poly_area(bottom) if bottom else 0.0

    def bisect(lo, hi, vertical):
        for _ in range(30):
            mid = (lo + hi) * 0.5
            a = area_left_of_cut(mid, vertical)
            if a / total < ratio:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5

    results = []
    if w > MIN_ROOM_WIDTH_CM * 2:
        coord = bisect(minx, maxx, True)
        left, right = _split_poly_vertical(poly, coord)
        wl = _poly_min_width(left) if left else 0.0
        wr = _poly_min_width(right) if right else 0.0
        if wl >= MIN_ROOM_WIDTH_CM * 0.8 and wr >= MIN_ROOM_WIDTH_CM * 0.8:
            results.append((coord, True, min(wl, wr)))

    if h > MIN_ROOM_WIDTH_CM * 2:
        coord = bisect(miny, maxy, False)
        bottom, top = _split_poly_horizontal(poly, coord)
        wb = _poly_min_width(bottom) if bottom else 0.0
        wt = _poly_min_width(top) if top else 0.0
        if wb >= MIN_ROOM_WIDTH_CM * 0.8 and wt >= MIN_ROOM_WIDTH_CM * 0.8:
            results.append((coord, False, min(wb, wt)))

    if not results:
        return None
    if prefer_vertical and w >= h:
        vert_results = [r for r in results if r[1]]
        if vert_results:
            return vert_results[0][0], vert_results[0][1]
    horiz_results = [r for r in results if not r[1]]
    if horiz_results:
        return horiz_results[0][0], horiz_results[0][1]
    return results[0][0], results[0][1]


def _cut_line_from_split(poly, coord, is_vertical):
    """Return the (x1,y1,x2,y2) cut line clipped to the polygon bbox."""
    minx, miny, maxx, maxy = _bbox(poly)
    pad = max(1.0, (maxx - minx + maxy - miny) * 0.001)
    if is_vertical:
        return coord, miny - pad, coord, maxy + pad
    else:
        return minx - pad, coord, maxx + pad, coord


def partition_apartment(poly_cm, programme, entrance_side="left", snapshot=None):
    """
    Main partitioner.
    poly_cm       : list of (x,y) in cm — apartment boundary (CCW)
    programme     : RoomProgramme
    entrance_side : 'left' | 'right' | 'bottom' | 'top'
    Returns PartitionResult.
    """
    result = PartitionResult()
    poly = _ensure_ccw(poly_cm)
    total_area = _poly_area(poly)

    if total_area < 1000:
        result.warn("Polygon too small: {:.0f}cm2".format(total_area))
        return result

    _log(snapshot, "Partitioning {:.0f}cm2 ({:.1f}m2), entrance={}".format(
        total_area, total_area / 10000.0, entrance_side))

    public_rooms  = [r for r in programme.rooms if r["zone"] == ZONE_PUBLIC]
    private_rooms = [r for r in programme.rooms if r["zone"] == ZONE_PRIVATE]

    total_public_area  = sum(r["min_area_cm2"] for r in public_rooms)
    total_private_area = sum(r["min_area_cm2"] for r in private_rooms)
    total_min = total_public_area + total_private_area

    scale = total_area / max(total_min, 1.0)
    scale = max(1.0, min(scale, 3.0))

    public_target  = total_public_area  * scale
    private_target = total_private_area * scale

    if public_target + private_target > total_area * 0.98:
        public_target  = total_area * (total_public_area  / max(total_min, 1.0))
        private_target = total_area * (total_private_area / max(total_min, 1.0))

    # Split into PUBLIC / PRIVATE zones
    prefer_vert = entrance_side in ("left", "right")

    split = _find_best_split(poly, public_target, prefer_vertical=prefer_vert)

    if split is None:
        _log(snapshot, "Warning: could not split into zones, using full polygon")
        result.warn("Could not split public/private zones")
        public_poly  = poly
        private_poly = []
    else:
        coord, is_vert = split
        if is_vert:
            public_poly, private_poly = _split_poly_vertical(poly, coord)
            cx1, cy1, cx2, cy2 = _cut_line_from_split(poly, coord, True)
        else:
            public_poly, private_poly = _split_poly_horizontal(poly, coord)
            cx1, cy1, cx2, cy2 = _cut_line_from_split(poly, coord, False)

        result.add_cut(cx1, cy1, cx2, cy2)
        _log(snapshot, "Zone split: {} public={:.1f}m2 private={:.1f}m2".format(
            "vertical" if is_vert else "horizontal",
            _poly_area(public_poly) / 10000.0 if public_poly else 0,
            _poly_area(private_poly) / 10000.0 if private_poly else 0))

    _subdivide_zone(public_poly,  public_rooms,  result, snapshot)
    _subdivide_zone(private_poly, private_rooms, result, snapshot)

    _log(snapshot, "Partition done: {} rooms, {} cut lines, {} warnings".format(
        len(result.rooms), len(result.cut_lines), len(result.warnings)))

    return result


def _subdivide_zone(poly, rooms, result, snapshot):
    """
    Recursively slice the polygon and assign one room per slice.
    Simple greedy approach: slice off rooms one by one from one end.
    """
    if not poly or len(poly) < 3 or not rooms:
        return

    remaining_poly = _ensure_ccw(list(poly))

    for i, room in enumerate(rooms):
        if not remaining_poly or len(remaining_poly) < 3:
            result.warn("Ran out of polygon for room: {}".format(room["name"]))
            break

        is_last = (i == len(rooms) - 1)

        if is_last:
            result.add_room(room["name"], remaining_poly)
            _log(snapshot, "  Room '{}': {:.1f}m2".format(
                room["name"], _poly_area(remaining_poly) / 10000.0))
            remaining_poly = []
        else:
            target = room["min_area_cm2"]
            total_remaining = _poly_area(remaining_poly)

            # Don't take more than 70% in one cut to leave space for others
            max_take = total_remaining * 0.70
            target = min(target, max_take)
            target = max(target, room["min_area_cm2"])

            minx, miny, maxx, maxy = _bbox(remaining_poly)
            w = maxx - minx
            h = maxy - miny
            prefer_vert = (w >= h)

            split = _find_best_split(remaining_poly, target,
                                     prefer_vertical=prefer_vert)

            if split is None:
                result.warn("Can't split for '{}', assigning remaining".format(
                    room["name"]))
                result.add_room(room["name"], remaining_poly)
                _log(snapshot, "  Room '{}': assigned remaining {:.1f}m2".format(
                    room["name"], _poly_area(remaining_poly) / 10000.0))
                remaining_poly = []
                break
            else:
                coord, is_vert = split
                # Capture bbox before modifying remaining_poly
                pre_split_poly = list(remaining_poly)
                if is_vert:
                    room_poly, remaining_poly = _split_poly_vertical(
                        remaining_poly, coord)
                    cx1, cy1, cx2, cy2 = _cut_line_from_split(
                        pre_split_poly, coord, True)
                else:
                    room_poly, remaining_poly = _split_poly_horizontal(
                        remaining_poly, coord)
                    cx1, cy1, cx2, cy2 = _cut_line_from_split(
                        pre_split_poly, coord, False)

                if room_poly:
                    result.add_room(room["name"], room_poly)
                    result.add_cut(cx1, cy1, cx2, cy2)
                    _log(snapshot, "  Room '{}': {:.1f}m2".format(
                        room["name"], _poly_area(room_poly) / 10000.0))

    if remaining_poly and len(remaining_poly) >= 3:
        leftover_area = _poly_area(remaining_poly)
        if leftover_area > 1000:
            result.warn("Leftover unassigned area: {:.1f}m2".format(
                leftover_area / 10000.0))
            result.add_room(u"שטח לא מוקצה", remaining_poly)


# ---------------------------------------------------------------------------
# Validation — check programme against rules
# ---------------------------------------------------------------------------

def validate_programme(result, programme, snapshot=None):
    """
    Check each assigned room against its minimum constraints.
    Returns list of violation strings.
    """
    violations = []
    room_map = {}
    for r in programme.rooms:
        room_map[r["name"]] = r

    for assigned in result.rooms:
        name = assigned["name"]
        area = assigned["area_cm2"]
        poly = assigned["poly"]

        spec = room_map.get(name)
        if spec is None:
            continue

        min_area = spec["min_area_cm2"]
        min_width = spec["min_width_cm"]
        actual_width = _poly_min_width(poly)

        if area < min_area * 0.95:
            msg = "{}: {:.1f}m2 < min {:.1f}m2".format(
                name, area / 10000.0, min_area / 10000.0)
            violations.append(msg)
            _log(snapshot, "VIOLATION: " + msg)

        if actual_width < min_width * 0.90:
            msg = "{}: width {:.0f}cm < min {:.0f}cm".format(
                name, actual_width, min_width)
            violations.append(msg)
            _log(snapshot, "VIOLATION: " + msg)

    return violations


# ---------------------------------------------------------------------------
# Revit Model Line writer
# ---------------------------------------------------------------------------

def _cm_to_feet(cm):
    return float(cm) / 30.48


def _find_floor_plan_view(doc, level):
    """Return a floor plan ViewPlan for the given level, or None."""
    try:
        from Autodesk.Revit.DB import FilteredElementCollector, ViewPlan, ViewType
        for vp in FilteredElementCollector(doc).OfClass(ViewPlan).ToElements():
            try:
                if vp.IsTemplate:
                    continue
                if str(vp.ViewType) == "FloorPlan" and vp.GenLevel is not None:
                    if vp.GenLevel.Id == level.Id:
                        return vp
            except Exception:
                continue
    except Exception:
        pass
    return None


def _draw_model_lines(doc, cut_lines_cm, view, snapshot=None, level=None):
    """
    Draw a list of cut lines as Revit Model Lines.
    If view is an Area Plan (which doesn't support NewModelCurve), finds the
    floor plan view for the same level automatically.
    cut_lines_cm: list of {"x1","y1","x2","y2"} in cm
    Returns list of created element IDs.
    """
    from Autodesk.Revit.DB import (
        Line, XYZ, SketchPlane, Plane
    )
    from Autodesk.Revit.DB import Transaction

    created_ids = []
    if not cut_lines_cm:
        return created_ids

    # If current view is an Area Plan, switch to the floor plan view
    draw_view = view
    try:
        if str(view.ViewType) == "AreaPlan" and level is not None:
            fp = _find_floor_plan_view(doc, level)
            if fp is not None:
                draw_view = fp
                _log(snapshot, "Switched draw target to floor plan view: {}".format(
                    fp.Name))
            else:
                _log(snapshot, "No floor plan view found for level; trying active view")
    except Exception:
        pass

    try:
        sp = draw_view.SketchPlane
        if sp is None:
            raise Exception("no sketch plane")
    except Exception:
        try:
            plane = Plane.CreateByNormalAndOrigin(XYZ.BasisZ, XYZ.Zero)
            sp = SketchPlane.Create(doc, plane)
        except Exception as ex:
            _log(snapshot, "Could not create sketch plane: {}".format(ex))
            return created_ids

    t = Transaction(doc, "ApartmentPlanner: Draw Partition Lines")
    t.Start()
    try:
        for seg in cut_lines_cm:
            try:
                x1 = _cm_to_feet(seg["x1"])
                y1 = _cm_to_feet(seg["y1"])
                x2 = _cm_to_feet(seg["x2"])
                y2 = _cm_to_feet(seg["y2"])
                dx = x2 - x1
                dy = y2 - y1
                if math.sqrt(dx*dx + dy*dy) < 1e-6:
                    continue
                ln = Line.CreateBound(XYZ(x1, y1, 0.0), XYZ(x2, y2, 0.0))
                mc = doc.Create.NewModelCurve(ln, sp)  # requires floor plan view
                if mc is not None:
                    created_ids.append(mc.Id.IntegerValue)
            except Exception as ex:
                _log(snapshot, "Line draw error: {}".format(ex))
        t.Commit()
        _log(snapshot, "Drew {} model lines".format(len(created_ids)))
    except Exception as ex:
        _log(snapshot, "Transaction error: {}".format(ex))
        try:
            t.RollBack()
        except Exception:
            pass

    return created_ids


# ---------------------------------------------------------------------------
# UI Dialogs
# ---------------------------------------------------------------------------

def _show_programme_dialog(programme, sqm):
    """Show the proposed room programme to the user and ask for confirmation."""
    from Autodesk.Revit.UI import TaskDialog, TaskDialogCommonButtons, TaskDialogResult

    lines = [
        u"{:.1f} מ\"ר → {} חדרי שינה".format(
            sqm, programme.n_bedrooms),
        "",
        u"תוכנית חדרים מוצעת:",
    ]
    for r in programme.rooms:
        zone_label = u"ציבורי" if r["zone"] == ZONE_PUBLIC else u"פרטי"
        lines.append(u"  * {} ({}) - min {:.0f}m2, w {:.0f}cm".format(
            r["name"], zone_label,
            r["min_area_cm2"] / 10000.0,
            r["min_width_cm"]))

    td = TaskDialog("ApartmentPlanner")
    td.MainInstruction = u"תוכנית חדרים לדירה"
    td.MainContent = "\n".join(lines)
    td.CommonButtons = (TaskDialogCommonButtons.Yes |
                        TaskDialogCommonButtons.No)
    td.DefaultButton = TaskDialogResult.Yes
    return td.Show()


def _show_violations_dialog(violations):
    from Autodesk.Revit.UI import TaskDialog, TaskDialogCommonButtons
    if not violations:
        return
    td = TaskDialog(u"ApartmentPlanner - אזהרות")
    td.MainInstruction = u"חריגות מתקנים:"
    td.MainContent = "\n".join("* " + v for v in violations)
    td.CommonButtons = TaskDialogCommonButtons.Ok
    td.Show()


def _show_result_dialog(result, violations):
    from Autodesk.Revit.UI import TaskDialog, TaskDialogCommonButtons
    lines = [u"חולקו {} חדרים:".format(len(result.rooms))]
    for r in result.rooms:
        lines.append(u"  * {}: {:.1f} m2".format(
            r["name"], r["area_cm2"] / 10000.0))
    lines.append("")
    lines.append(u"שורטטו {} קווי מחיצה".format(len(result.cut_lines)))
    if violations:
        lines.append("")
        lines.append(u"[!] {} חריגות".format(len(violations)))

    td = TaskDialog(u"ApartmentPlanner - תוצאה")
    td.MainInstruction = u"חלוקת הדירה הושלמה"
    td.MainContent = "\n".join(lines)
    td.CommonButtons = TaskDialogCommonButtons.Ok
    td.Show()


# ---------------------------------------------------------------------------
# Entrance detection
# ---------------------------------------------------------------------------

def _find_entrance_point(doc, area_elem, poly_cm, snapshot=None):
    """
    Find the entry door by looking for Door FamilyInstances whose location
    point is within 60cm of any edge of the Area boundary polygon.
    Returns (x_cm, y_cm) or None.
    """
    ENTRY_SNAP_CM = 60.0

    try:
        from Autodesk.Revit.DB import FilteredElementCollector, BuiltInCategory, FamilyInstance

        doors = FilteredElementCollector(doc)\
            .OfCategory(BuiltInCategory.OST_Doors)\
            .OfClass(FamilyInstance)\
            .ToElements()

        candidates = []
        n = len(poly_cm)

        for door in doors:
            try:
                loc = door.Location
                if loc is None:
                    continue
                pt_ft = loc.Point
                dx_cm = pt_ft.X * 30.48
                dy_cm = pt_ft.Y * 30.48

                min_dist = 1e12
                for i in range(n):
                    j = (i + 1) % n
                    ax, ay = poly_cm[i]
                    bx, by = poly_cm[j]
                    ex, ey = bx - ax, by - ay
                    seg_len = math.sqrt(ex*ex + ey*ey)
                    if seg_len < 1e-6:
                        continue
                    t = ((dx_cm - ax) * ex + (dy_cm - ay) * ey) / (seg_len * seg_len)
                    t = max(0.0, min(1.0, t))
                    cx2 = ax + t * ex
                    cy2 = ay + t * ey
                    dist = math.sqrt((dx_cm - cx2)**2 + (dy_cm - cy2)**2)
                    if dist < min_dist:
                        min_dist = dist

                if min_dist <= ENTRY_SNAP_CM:
                    candidates.append((min_dist, dx_cm, dy_cm, door.Id))
            except Exception:
                continue

        if not candidates:
            _log(snapshot, "No entry door found on Area boundary")
            return None

        candidates.sort(key=lambda c: c[0])
        best = candidates[0]
        _log(snapshot, "Entry door at ({:.0f}, {:.0f})cm dist={:.1f}cm".format(
            best[1], best[2], best[0]))
        return (best[1], best[2])

    except Exception as ex:
        _log(snapshot, "Entrance detection error: {}".format(ex))
        return None


def _entrance_side(poly_cm, entry_pt):
    """
    Determine which side of the bounding box the entrance is on.
    Returns 'left' | 'right' | 'bottom' | 'top'.
    """
    if entry_pt is None:
        return "left"

    minx, miny, maxx, maxy = _bbox(poly_cm)
    cx = (minx + maxx) * 0.5
    cy = (miny + maxy) * 0.5
    ex, ey = entry_pt

    dx = ex - cx
    dy = ey - cy
    w = maxx - minx
    h = maxy - miny

    nx = dx / max(w * 0.5, 1.0)
    ny = dy / max(h * 0.5, 1.0)

    if abs(nx) >= abs(ny):
        return "left" if nx < 0 else "right"
    else:
        return "bottom" if ny < 0 else "top"


# ---------------------------------------------------------------------------
# Area selector
# ---------------------------------------------------------------------------

def _pick_area(uidoc, snapshot=None):
    """Prompt user to select an Area element. Returns Area element or None."""
    try:
        from Autodesk.Revit.UI.Selection import ISelectionFilter, ObjectType
        from Autodesk.Revit.DB import BuiltInCategory

        class _AreaFilter(ISelectionFilter):
            def AllowElement(self, elem):
                try:
                    return (elem.Category is not None and
                            elem.Category.Id.IntegerValue ==
                            int(BuiltInCategory.OST_Areas))
                except Exception:
                    return False

            def AllowReference(self, reference, point):
                return False

        picked = uidoc.Selection.PickObject(
            ObjectType.Element,
            _AreaFilter(),
            u"בחר Area של דירה לתכנון")
        if picked is None:
            return None
        return uidoc.Document.GetElement(picked.ElementId)
    except Exception as ex:
        _log(snapshot, "Area pick error: {}".format(ex))
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_apartment_planner(v2, level, area_elem=None, snapshot=None, uidoc=None):
    """
    Full pipeline:
      1. Use provided area_elem, or prompt user to pick one
      2. Rule Engine builds programme
      3. User confirms programme
      4. Partitioner runs
      5. Model Lines drawn
      6. Validation shown

    v2        : the loaded CreateFromCADV2 module (for doc attribute)
    level     : Revit Level element
    area_elem : optional Revit Area element; if None, user is prompted to pick
    snapshot  : optional snapshot logger
    uidoc     : UIDocument; if None, falls back to v2 attributes or __revit__
    """
    try:
        if uidoc is None:
            try:
                uidoc = v2.uidoc
            except Exception:
                pass
        if uidoc is None:
            try:
                uidoc = __revit__.ActiveUIDocument
            except Exception:
                pass

        if area_elem is None:
            if uidoc is None:
                _log(snapshot, "No UIDocument available")
                return None
            area_elem = _pick_area(uidoc, snapshot)
            if area_elem is None:
                _log(snapshot, "No area selected, cancelled")
                return None

        sqm = _area_sqm(area_elem)
        if sqm is None or sqm < 10:
            _log(snapshot, "Could not read area sqm")
            from Autodesk.Revit.UI import TaskDialog
            TaskDialog.Show("ApartmentPlanner", u"לא ניתן לקרוא שטח Area.")
            return None

        poly_cm = _area_polygon_cm(area_elem)
        if poly_cm is None or len(poly_cm) < 3:
            _log(snapshot, "Could not read area boundary polygon")
            from Autodesk.Revit.UI import TaskDialog
            TaskDialog.Show("ApartmentPlanner",
                            u"לא ניתן לקרוא את גבולות ה-Area.")
            return None

        _log(snapshot, "Area selected: {:.1f}m2, polygon={} pts".format(
            sqm, len(poly_cm)))

        if uidoc is not None:
            doc = uidoc.Document
        else:
            doc = v2.doc
        entry_pt = _find_entrance_point(doc, area_elem, poly_cm, snapshot)
        side = _entrance_side(poly_cm, entry_pt)
        _log(snapshot, "Entrance side: {}".format(side))

        programme = build_programme(sqm, snapshot)

        from Autodesk.Revit.UI import TaskDialogResult
        response = _show_programme_dialog(programme, sqm)
        if response != TaskDialogResult.Yes:
            _log(snapshot, "User cancelled programme")
            return None

        part_result = partition_apartment(
            poly_cm, programme,
            entrance_side=side,
            snapshot=snapshot)

        violations = validate_programme(part_result, programme, snapshot)

        view = uidoc.ActiveView if uidoc is not None else v2.doc.ActiveView
        created_ids = _draw_model_lines(doc, part_result.cut_lines, view, snapshot, level=level)

        _show_result_dialog(part_result, violations)
        if violations:
            _show_violations_dialog(violations)

        if snapshot:
            try:
                snapshot.save_json("apartment_planner_result.json", {
                    "sqm": sqm,
                    "n_bedrooms": programme.n_bedrooms,
                    "rooms": [
                        {"name": r["name"],
                         "area_m2": round(r["area_cm2"] / 10000.0, 2)}
                        for r in part_result.rooms
                    ],
                    "cut_lines": len(part_result.cut_lines),
                    "model_line_ids": created_ids,
                    "violations": violations,
                    "warnings": part_result.warnings,
                })
            except Exception:
                pass

        return {
            "programme": programme,
            "partition": part_result,
            "model_line_ids": created_ids,
            "violations": violations,
        }

    except Exception as ex:
        _log(snapshot, "run_apartment_planner FAILED: {}".format(ex))
        try:
            from Autodesk.Revit.UI import TaskDialog
            TaskDialog.Show("ApartmentPlanner", "Error: {}".format(str(ex)))
        except Exception:
            pass
        return None
