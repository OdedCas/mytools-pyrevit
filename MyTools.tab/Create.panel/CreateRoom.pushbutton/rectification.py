"""Projective rectification helpers (pure Python)."""

import math


class RectificationError(ValueError):
    pass


def _pt2(p):
    return (float(p[0]), float(p[1]))


def distance(p1, p2):
    a = _pt2(p1)
    b = _pt2(p2)
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt((dx * dx) + (dy * dy))


def _solve_linear_8x8(A, b):
    # Gaussian elimination with partial pivoting.
    n = 8
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot = col
        max_abs = abs(M[col][col])
        for r in range(col + 1, n):
            v = abs(M[r][col])
            if v > max_abs:
                max_abs = v
                pivot = r

        if max_abs < 1e-12:
            raise RectificationError("Singular system while solving homography.")

        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]

        pivot_val = M[col][col]
        for c in range(col, n + 1):
            M[col][c] = M[col][c] / pivot_val

        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if abs(factor) < 1e-18:
                continue
            for c in range(col, n + 1):
                M[r][c] = M[r][c] - (factor * M[col][c])

    return [M[i][n] for i in range(n)]


def compute_homography(src_pts, dst_pts):
    """Compute H mapping src->dst. h33 is fixed to 1.

    src_pts / dst_pts: 4 points each, [(x,y), ...]
    Returns list of 9 values row-major.
    """
    if len(src_pts) != 4 or len(dst_pts) != 4:
        raise RectificationError("Need exactly 4 source and 4 destination points.")

    A = []
    b = []
    for i in range(4):
        x, y = _pt2(src_pts[i])
        u, v = _pt2(dst_pts[i])

        A.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        b.append(u)

        A.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        b.append(v)

    h = _solve_linear_8x8(A, b)
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]


def apply_homography(H, p):
    x, y = _pt2(p)
    w = (H[6] * x) + (H[7] * y) + H[8]
    if abs(w) < 1e-12:
        raise RectificationError("Invalid homography denominator.")

    u = ((H[0] * x) + (H[1] * y) + H[2]) / w
    v = ((H[3] * x) + (H[4] * y) + H[5]) / w
    return (u, v)


def estimate_rect_size(src_nw, src_ne, src_se, src_sw):
    top = distance(src_nw, src_ne)
    bottom = distance(src_sw, src_se)
    left = distance(src_nw, src_sw)
    right = distance(src_ne, src_se)

    width = (top + bottom) * 0.5
    height = (left + right) * 0.5
    if width < 1e-9 or height < 1e-9:
        raise RectificationError("Degenerate corner picks.")

    return width, height


def build_rectification(src_nw, src_ne, src_se, src_sw):
    """Build homography from picked quadrilateral to orthogonal rectangle.

    Destination frame:
      SW=(0,0), SE=(W,0), NE=(W,H), NW=(0,H)
    """
    width, height = estimate_rect_size(src_nw, src_ne, src_se, src_sw)

    src = [
        _pt2(src_nw),
        _pt2(src_ne),
        _pt2(src_se),
        _pt2(src_sw),
    ]
    dst = [
        (0.0, height),
        (width, height),
        (width, 0.0),
        (0.0, 0.0),
    ]

    H = compute_homography(src, dst)

    return {
        "homography": H,
        "rect_width_units": width,
        "rect_height_units": height,
        "dst_corners": {
            "nw": (0.0, height),
            "ne": (width, height),
            "se": (width, 0.0),
            "sw": (0.0, 0.0),
        },
    }


def rectify_points(points, H):
    return [apply_homography(H, p) for p in points]
