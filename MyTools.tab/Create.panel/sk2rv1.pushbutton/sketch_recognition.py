import argparse
import json
import math
import os
import sys


cv2 = None
np = None


def ensure_dependencies(extra_site_packages=None):
    global cv2
    global np
    if cv2 is not None and np is not None:
        return
    if extra_site_packages and extra_site_packages not in sys.path:
        sys.path.append(extra_site_packages)
    import cv2 as _cv2
    import numpy as _np
    cv2 = _cv2
    np = _np


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _get_exif_orientation(image_path):
    """Read EXIF orientation tag from a JPEG file."""
    try:
        with open(image_path, "rb") as f:
            data = f.read(65536)
    except Exception:
        return 1
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return 1
    import struct
    pos = 2
    while pos + 4 < len(data):
        if data[pos] != 0xFF:
            break
        marker = data[pos + 1]
        pos += 2
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        seg_len = (data[pos] << 8) | data[pos + 1]
        if seg_len < 2 or pos + seg_len > len(data):
            break
        seg = data[pos + 2:pos + seg_len]
        pos += seg_len
        if marker != 0xE1 or not seg[:6] == b"Exif\x00\x00":
            continue
        tiff = seg[6:]
        if len(tiff) < 8:
            return 1
        endian = tiff[:2]
        if endian == b"II":
            u16 = lambda c: struct.unpack("<H", c)[0]
            u32 = lambda c: struct.unpack("<I", c)[0]
        elif endian == b"MM":
            u16 = lambda c: struct.unpack(">H", c)[0]
            u32 = lambda c: struct.unpack(">I", c)[0]
        else:
            return 1
        if u16(tiff[2:4]) != 42:
            return 1
        ifd0 = u32(tiff[4:8])
        if ifd0 + 2 > len(tiff):
            return 1
        count = u16(tiff[ifd0:ifd0 + 2])
        for i in range(count):
            entry = tiff[ifd0 + 2 + i * 12:ifd0 + 2 + (i + 1) * 12]
            if len(entry) < 12:
                break
            if u16(entry[0:2]) == 0x0112:
                return u16(entry[8:10])
        return 1
    return 1


def _apply_exif_rotation(gray, orientation):
    """Rotate/flip image to match EXIF orientation.

    Note: for orientations 6 and 8, we also flip horizontally because
    Revit's .NET image rendering interprets EXIF rotation with opposite
    handedness compared to OpenCV's cv2.rotate.
    """
    if orientation == 1:
        return gray
    if orientation == 3:
        return cv2.rotate(gray, cv2.ROTATE_180)
    if orientation == 6:
        return cv2.flip(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE), 1)
    if orientation == 8:
        return cv2.flip(cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    if orientation == 2:
        return cv2.flip(gray, 1)
    if orientation == 4:
        return cv2.flip(gray, 0)
    if orientation == 5:
        return cv2.flip(cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE), 1)
    if orientation == 7:
        return cv2.flip(cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    return gray


def _prep_binary(image_path, max_dim=1200):
    """Load image, resize, threshold to get ink mask."""
    ensure_dependencies()
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise Exception("OpenCV failed to load image: {}".format(image_path))

    # Apply EXIF rotation so coordinates match what Revit/viewers display.
    orientation = _get_exif_orientation(image_path)
    gray = _apply_exif_rotation(gray, orientation)

    # Resize large images.
    h, w = gray.shape[:2]
    if max(h, w) > max_dim:
        scale = float(max_dim) / float(max(h, w))
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 7,
    )
    otsu = cv2.bitwise_or(otsu, adaptive)

    # Remove tiny specks.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(otsu, connectivity=8)
    cleaned = np.zeros_like(otsu)
    for idx in range(1, num_labels):
        if int(stats[idx, cv2.CC_STAT_AREA]) < 30:
            continue
        cleaned[labels == idx] = 255

    return cleaned


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------

def _skeletonize(binary_img, max_iter=300):
    """Morphological skeletonization (erosion/opening subtraction)."""
    ensure_dependencies()
    skel = np.zeros_like(binary_img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary_img.copy()
    for _ in range(max_iter):
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------

def _merge_collinear_segments(segments, perp_tol, gap_tol):
    """Merge overlapping/nearby collinear axis-aligned segments.

    Sort-based: sort by perpendicular position, then merge along the axis.
    No bucketing, so no boundary issues.
    """
    h_segs = [dict(s) for s in (segments or []) if s["orientation"] == "h"]
    v_segs = [dict(s) for s in (segments or []) if s["orientation"] == "v"]

    merged = []
    idx = 0

    # Merge horizontal: sort by Y, then by X.
    h_segs.sort(key=lambda s: (s["start"][1], s["start"][0]))
    groups = []
    for seg in h_segs:
        placed = False
        for grp in groups:
            if abs(grp[0]["start"][1] - seg["start"][1]) <= perp_tol:
                grp.append(seg)
                placed = True
                break
        if not placed:
            groups.append([seg])

    for grp in groups:
        grp.sort(key=lambda s: s["start"][0])
        cur = None
        for seg in grp:
            if cur is None:
                cur = dict(seg)
                continue
            if seg["start"][0] <= cur["end"][0] + gap_tol:
                avg_y = (cur["start"][1] + seg["start"][1]) * 0.5
                cur["start"] = (cur["start"][0], avg_y)
                cur["end"] = (max(cur["end"][0], seg["end"][0]), avg_y)
            else:
                cur["index"] = idx; idx += 1
                merged.append(cur)
                cur = dict(seg)
        if cur is not None:
            cur["index"] = idx; idx += 1
            merged.append(cur)

    # Merge vertical: sort by X, then by Y.
    v_segs.sort(key=lambda s: (s["start"][0], s["start"][1]))
    groups = []
    for seg in v_segs:
        placed = False
        for grp in groups:
            if abs(grp[0]["start"][0] - seg["start"][0]) <= perp_tol:
                grp.append(seg)
                placed = True
                break
        if not placed:
            groups.append([seg])

    for grp in groups:
        grp.sort(key=lambda s: s["start"][1])
        cur = None
        for seg in grp:
            if cur is None:
                cur = dict(seg)
                continue
            if seg["start"][1] <= cur["end"][1] + gap_tol:
                avg_x = (cur["start"][0] + seg["start"][0]) * 0.5
                cur["start"] = (avg_x, cur["start"][1])
                cur["end"] = (avg_x, max(cur["end"][1], seg["end"][1]))
            else:
                cur["index"] = idx; idx += 1
                merged.append(cur)
                cur = dict(seg)
        if cur is not None:
            cur["index"] = idx; idx += 1
            merged.append(cur)

    return merged


def _has_ink_along_path(binary_img, x1, y1, x2, y2, half_width=8, min_density=0.04):
    """Check if there's ink in the binary image along a straight path."""
    if binary_img is None:
        return True
    img_h, img_w = binary_img.shape[:2]
    if abs(x2 - x1) < 2 and abs(y2 - y1) < 2:
        return True
    # Sample a rectangular band along the path.
    rx1 = max(0, int(round(min(x1, x2) - half_width)))
    ry1 = max(0, int(round(min(y1, y2) - half_width)))
    rx2 = min(img_w, int(round(max(x1, x2) + half_width)))
    ry2 = min(img_h, int(round(max(y1, y2) + half_width)))
    if rx2 <= rx1 or ry2 <= ry1:
        return False
    patch = binary_img[ry1:ry2, rx1:rx2]
    density = float(cv2.countNonZero(patch)) / float(max(patch.size, 1))
    return density >= min_density


def _extend_segments_to_intersections(segments, ext_tol, binary_img=None):
    """Extend axis-aligned segments to meet at perpendicular intersections.

    If binary_img is provided, long extensions are only accepted if there is
    ink evidence along the extension path.
    """
    ensure_dependencies()
    result = []
    for s in (segments or []):
        r = dict(s)
        r["start"] = list(r["start"])
        r["end"] = list(r["end"])
        result.append(r)

    h_indices = [i for i, s in enumerate(result) if s["orientation"] == "h"]
    v_indices = [i for i, s in enumerate(result) if s["orientation"] == "v"]

    # Short extensions (within short_tol) are always accepted.
    # Longer extensions (up to ext_tol) require ink evidence,
    # UNLESS the extension connects to the other segment's endpoint
    # (forming a corner — corners must always connect).
    short_tol = min(ext_tol * 0.4, 50.0)
    corner_tol = 15.0  # how close to an endpoint counts as "corner"

    for hi in h_indices:
        for vi in v_indices:
            h = result[hi]
            v = result[vi]
            yh = h["start"][1]
            hx1, hx2 = h["start"][0], h["end"][0]
            xv = v["start"][0]
            vy1, vy2 = v["start"][1], v["end"][1]

            if xv < hx1 - ext_tol or xv > hx2 + ext_tol:
                continue
            if yh < vy1 - ext_tol or yh > vy2 + ext_tol:
                continue

            # Check if yh is near V's endpoint (corner connection).
            y_is_corner = abs(yh - vy1) <= corner_tol or abs(yh - vy2) <= corner_tol
            # Check if xv is near H's endpoint (corner connection).
            x_is_corner = abs(xv - hx1) <= corner_tol or abs(xv - hx2) <= corner_tol

            def _should_extend(gap, x1, y1, x2, y2, is_corner):
                if gap <= short_tol:
                    return True
                if is_corner:
                    return True
                return _has_ink_along_path(binary_img, x1, y1, x2, y2)

            # Extend H segment.
            if xv < hx1 and (hx1 - xv) <= ext_tol:
                if _should_extend(hx1 - xv, xv, yh, hx1, yh, y_is_corner):
                    result[hi]["start"][0] = xv
            elif xv > hx2 and (xv - hx2) <= ext_tol:
                if _should_extend(xv - hx2, hx2, yh, xv, yh, y_is_corner):
                    result[hi]["end"][0] = xv

            # Extend V segment.
            if yh < vy1 and (vy1 - yh) <= ext_tol:
                if _should_extend(vy1 - yh, xv, yh, xv, vy1, x_is_corner):
                    result[vi]["start"][1] = yh
            elif yh > vy2 and (yh - vy2) <= ext_tol:
                if _should_extend(yh - vy2, xv, vy2, xv, yh, x_is_corner):
                    result[vi]["end"][1] = yh

    for r in result:
        r["start"] = tuple(r["start"])
        r["end"] = tuple(r["end"])
    return result


def _split_segments_at_crossings(segments, tol=3.0):
    """Split axis-aligned segments at T-junctions."""
    h_segs = [(i, s) for i, s in enumerate(segments or []) if s["orientation"] == "h"]
    v_segs = [(i, s) for i, s in enumerate(segments or []) if s["orientation"] == "v"]
    splits = [[] for _ in (segments or [])]

    for hi, h in h_segs:
        yh = h["start"][1]
        hx1, hx2 = h["start"][0], h["end"][0]
        h_len = hx2 - hx1
        for vi, v in v_segs:
            xv = v["start"][0]
            vy1, vy2 = v["start"][1], v["end"][1]
            v_len = vy2 - vy1
            if not (hx1 - 0.5 <= xv <= hx2 + 0.5 and vy1 - 0.5 <= yh <= vy2 + 0.5):
                continue
            margin_h = tol / max(h_len, 1.0)
            t_h = (xv - hx1) / max(h_len, 1e-9)
            margin_v = tol / max(v_len, 1.0)
            t_v = (yh - vy1) / max(v_len, 1e-9)
            h_int = margin_h < t_h < 1.0 - margin_h
            v_int = margin_v < t_v < 1.0 - margin_v
            if not (h_int or v_int):
                continue
            if h_int:
                splits[hi].append(xv)
            if v_int:
                splits[vi].append(yh)

    out = []
    idx = 0
    for i, seg in enumerate(segments or []):
        if not splits[i]:
            seg_copy = dict(seg)
            seg_copy["index"] = idx; idx += 1
            out.append(seg_copy)
            continue
        if seg["orientation"] == "h":
            y = seg["start"][1]
            points = [seg["start"][0]] + sorted(set(splits[i])) + [seg["end"][0]]
            for k in range(len(points) - 1):
                if points[k + 1] - points[k] >= 3.0:
                    out.append({"index": idx, "orientation": "h",
                                "start": (points[k], y), "end": (points[k + 1], y)})
                    idx += 1
        else:
            x = seg["start"][0]
            points = [seg["start"][1]] + sorted(set(splits[i])) + [seg["end"][1]]
            for k in range(len(points) - 1):
                if points[k + 1] - points[k] >= 3.0:
                    out.append({"index": idx, "orientation": "v",
                                "start": (x, points[k]), "end": (x, points[k + 1])})
                    idx += 1
    return out


def _infer_missing_segments(segments, binary_img, max_gap):
    """Add missing H/V segments between dangling endpoints when ink exists.

    Finds V-segment endpoints that aren't connected to any H-segment and
    checks if there's another V-segment at a similar Y that we can bridge
    with a new H-segment (and vice versa).
    """
    if binary_img is None:
        return segments
    tol = 15.0  # How close endpoints need to be to count as "connected".

    def _is_connected(pt, segs, orientation, skip_idx):
        """Check if point is near the start/end of any segment with given orientation."""
        for s in segs:
            if s["index"] == skip_idx:
                continue
            if s["orientation"] != orientation:
                continue
            for ep in (s["start"], s["end"]):
                if abs(pt[0] - ep[0]) <= tol and abs(pt[1] - ep[1]) <= tol:
                    return True
        return False

    new_segs = list(segments)
    idx = max((s["index"] for s in segments), default=-1) + 1

    # Find V-segment endpoints that need an H connection.
    v_segs = [s for s in segments if s["orientation"] == "v"]
    for vs in v_segs:
        for ep_name in ("start", "end"):
            ep = vs[ep_name]
            if _is_connected(ep, segments, "h", vs["index"]):
                continue
            # Look for another V segment at a similar Y to bridge to.
            for vs2 in v_segs:
                if vs2["index"] == vs["index"]:
                    continue
                for ep2_name in ("start", "end"):
                    ep2 = vs2[ep2_name]
                    if abs(ep[1] - ep2[1]) > tol:
                        continue
                    h_gap = abs(ep[0] - ep2[0])
                    if h_gap < tol or h_gap > max_gap:
                        continue
                    avg_y = (ep[1] + ep2[1]) * 0.5
                    x_lo = min(ep[0], ep2[0])
                    x_hi = max(ep[0], ep2[0])
                    if _has_ink_along_path(binary_img, x_lo, avg_y, x_hi, avg_y,
                                           half_width=12, min_density=0.02):
                        new_segs.append({
                            "index": idx, "orientation": "h",
                            "start": (x_lo, avg_y), "end": (x_hi, avg_y),
                        })
                        idx += 1

    # Find H-segment endpoints that need a V connection.
    h_segs = [s for s in segments if s["orientation"] == "h"]
    for hs in h_segs:
        for ep_name in ("start", "end"):
            ep = hs[ep_name]
            if _is_connected(ep, segments, "v", hs["index"]):
                continue
            for hs2 in h_segs:
                if hs2["index"] == hs["index"]:
                    continue
                for ep2_name in ("start", "end"):
                    ep2 = hs2[ep2_name]
                    if abs(ep[0] - ep2[0]) > tol:
                        continue
                    v_gap = abs(ep[1] - ep2[1])
                    if v_gap < tol or v_gap > max_gap:
                        continue
                    avg_x = (ep[0] + ep2[0]) * 0.5
                    y_lo = min(ep[1], ep2[1])
                    y_hi = max(ep[1], ep2[1])
                    if _has_ink_along_path(binary_img, avg_x, y_lo, avg_x, y_hi,
                                           half_width=12, min_density=0.02):
                        new_segs.append({
                            "index": idx, "orientation": "v",
                            "start": (avg_x, y_lo), "end": (avg_x, y_hi),
                        })
                        idx += 1

    return new_segs


# ---------------------------------------------------------------------------
# Main recognition: fill hatching → skeletonize → Hough → segments
# ---------------------------------------------------------------------------

def _recognize_skeleton(binary_img):
    """Skeleton-based wall recognition for hand-drawn sketches.

    1. Heavy morphological close fills hatching into solid wall blobs.
    2. Skeletonize reduces each blob to a thin centerline.
    3. Hough lines on skeleton finds axis-aligned wall segments.
    4. Merge collinear, extend to intersections, split at T-junctions.
    """
    ensure_dependencies()
    img_h, img_w = binary_img.shape[:2]
    max_dim = max(img_h, img_w)

    # --- Step 1: Fill hatching ---
    # Directional close: fills hatching gaps along each axis without
    # merging walls across room interiors.
    k_long = max(30, int(round(max_dim * 0.055)))
    k_short = max(3, k_long // 5)

    # Directional close in 4 orientations: H, V, 45°, 135°.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_long, k_short))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_short, k_long))

    # Diagonal kernels for cross-hatching.
    diag_size = max(k_long // 2, 15)
    diag1 = np.zeros((diag_size, diag_size), dtype=np.uint8)
    diag2 = np.zeros((diag_size, diag_size), dtype=np.uint8)
    for di in range(diag_size):
        diag1[di, di] = 1
        diag2[di, diag_size - 1 - di] = 1
    # Widen the diagonal lines slightly.
    widen = np.ones((max(2, k_short // 2), max(2, k_short // 2)), dtype=np.uint8)
    diag1 = cv2.dilate(diag1, widen)
    diag2 = cv2.dilate(diag2, widen)

    h_closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, h_kernel, iterations=3)
    v_closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, v_kernel, iterations=3)
    d1_closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, diag1, iterations=2)
    d2_closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, diag2, iterations=2)
    closed = cv2.bitwise_or(h_closed, v_closed)
    closed = cv2.bitwise_or(closed, cv2.bitwise_or(d1_closed, d2_closed))

    # Square close to connect corners.
    corner_k = max(9, int(round(max_dim * 0.015)))
    closed = cv2.morphologyEx(
        closed, cv2.MORPH_CLOSE,
        np.ones((corner_k, corner_k), dtype=np.uint8),
        iterations=3,
    )

    # Remove small blobs that aren't walls.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_wall_area = max_dim * max_dim * 0.001
    wall_mask = np.zeros_like(closed)
    for idx in range(1, num_labels):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_wall_area:
            wall_mask[labels == idx] = 255

    # --- Step 2: Skeletonize ---
    skel = _skeletonize(wall_mask)

    # Clean skeleton: remove short branches by pruning endpoints.
    # A simple approach: dilate then re-thin to smooth out small spurs.
    skel = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    # Re-thin after close.
    skel = _skeletonize(skel, max_iter=20)

    # --- Step 3: Hough lines on skeleton ---
    min_line_len = max(15, int(round(max_dim * 0.025)))
    max_gap = max(25, int(round(max_dim * 0.04)))
    lines = cv2.HoughLinesP(
        skel, 1, math.pi / 180.0,
        threshold=max(8, int(round(max_dim * 0.01))),
        minLineLength=min_line_len,
        maxLineGap=max_gap,
    )

    if lines is None:
        return {"segments": [], "skel": skel, "wall_mask": wall_mask}

    # --- Step 4: Filter to axis-aligned segments ---
    raw_segments = []
    for line in lines:
        x1, y1, x2, y2 = [int(v) for v in line[0]]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = math.sqrt(float(dx * dx + dy * dy))
        if length < min_line_len:
            continue

        if dx < dy * 0.3:
            avg_x = (x1 + x2) * 0.5
            raw_segments.append({
                "index": len(raw_segments),
                "orientation": "v",
                "start": (avg_x, float(min(y1, y2))),
                "end": (avg_x, float(max(y1, y2))),
            })
        elif dy < dx * 0.3:
            avg_y = (y1 + y2) * 0.5
            raw_segments.append({
                "index": len(raw_segments),
                "orientation": "h",
                "start": (float(min(x1, x2)), avg_y),
                "end": (float(max(x1, x2)), avg_y),
            })

    # --- Step 5: Filter short/edge segments ---
    min_seg_len = max(20.0, max_dim * 0.04)
    edge_margin = max(15.0, max_dim * 0.02)
    filtered = []
    for seg in raw_segments:
        if seg["orientation"] == "h":
            length = abs(seg["end"][0] - seg["start"][0])
            pos = seg["start"][1]
        else:
            length = abs(seg["end"][1] - seg["start"][1])
            pos = seg["start"][0]
        if length < min_seg_len:
            continue
        # Skip segments too close to image edge (false detections).
        if pos < edge_margin or pos > max_dim - edge_margin:
            continue
        filtered.append(seg)

    # --- Step 6: Merge, extend, split ---
    perp_tol = max(25.0, max_dim * 0.04)
    gap_tol = max(25.0, max_dim * 0.05)
    segments = _merge_collinear_segments(filtered, perp_tol=perp_tol, gap_tol=gap_tol)
    segments = _merge_collinear_segments(segments, perp_tol=perp_tol, gap_tol=gap_tol)

    ext_tol = max(80.0, max_dim * 0.25)
    # Iterate extension until stable (chains of extensions may need multiple passes).
    for _pass in range(5):
        prev = [(s["start"], s["end"]) for s in segments]
        segments = _extend_segments_to_intersections(segments, ext_tol=ext_tol,
                                                      binary_img=binary_img)
        curr = [(s["start"], s["end"]) for s in segments]
        if curr == prev:
            break

    # Infer missing segments between dangling endpoints with ink evidence.
    segments = _infer_missing_segments(segments, binary_img, ext_tol)

    segments = _split_segments_at_crossings(segments, tol=3.0)

    # Post-split merge to unify segments at slightly different positions.
    segments = _merge_collinear_segments(segments, perp_tol=perp_tol, gap_tol=gap_tol)

    # Remove segments that touch image edges (x=0 or y=0).
    edge_margin2 = max(5.0, max_dim * 0.01)
    segments = [s for s in segments
                if not (s["orientation"] == "h" and s["start"][0] < edge_margin2)
                and not (s["orientation"] == "v" and s["start"][1] < edge_margin2)]

    # Remove very short segments that are noise after splitting.
    segments = [s for s in segments
                if (abs(s["end"][0] - s["start"][0]) if s["orientation"] == "h"
                    else abs(s["end"][1] - s["start"][1])) >= min_seg_len * 0.5]

    # Re-index.
    for i, s in enumerate(segments):
        s["index"] = i

    return {"segments": segments, "skel": skel, "wall_mask": wall_mask,
            "binary": binary_img}


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

def _save_debug_image(image_path, binary_img, wall_mask, skel, segments, debug_path,
                      binary_raw=None):
    """Save annotated debug image with 3 insets: binary, wall_mask, skeleton."""
    ensure_dependencies()
    original = cv2.imread(image_path)
    if original is None:
        return

    # Resize original to match binary dimensions.
    bh, bw = binary_img.shape[:2]
    canvas = cv2.resize(original, (bw, bh))

    # Draw wall segments (red, thick) with green endpoints.
    for seg in segments:
        sx, sy = int(round(seg["start"][0])), int(round(seg["start"][1]))
        ex, ey = int(round(seg["end"][0])), int(round(seg["end"][1]))
        cv2.line(canvas, (sx, sy), (ex, ey), (0, 0, 255), 3)
        cv2.circle(canvas, (sx, sy), 5, (0, 255, 0), -1)
        cv2.circle(canvas, (ex, ey), 5, (0, 255, 0), -1)

    scale = min(160.0 / max(bw, 1), 160.0 / max(bh, 1))
    tw, th = max(1, int(bw * scale)), max(1, int(bh * scale))
    y_off = 0

    # Inset 1: binary threshold result.
    if binary_raw is not None:
        thumb0 = cv2.cvtColor(cv2.resize(binary_raw, (tw, th)), cv2.COLOR_GRAY2BGR)
        if y_off + th < bh and tw < bw:
            canvas[y_off:y_off + th, bw - tw:bw] = thumb0
            y_off += th + 2

    # Inset 2: wall mask (after fill).
    if wall_mask is not None:
        thumb1 = cv2.cvtColor(cv2.resize(wall_mask, (tw, th)), cv2.COLOR_GRAY2BGR)
        if y_off + th < bh and tw < bw:
            canvas[y_off:y_off + th, bw - tw:bw] = thumb1
            y_off += th + 2

    # Inset 3: skeleton.
    if skel is not None:
        thumb2 = cv2.cvtColor(cv2.resize(skel, (tw, th)), cv2.COLOR_GRAY2BGR)
        if y_off + th < bh and tw < bw:
            canvas[y_off:y_off + th, bw - tw:bw] = thumb2

    cv2.imwrite(debug_path, canvas)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recognize_sketch(image_path, door_hint=None, site_packages=None, debug_path=None):
    ensure_dependencies(site_packages)
    binary_img = _prep_binary(image_path)
    if binary_img is None or binary_img.size == 0:
        return {"segments_px": [], "doors_px": [], "wall_thickness_px": 0.0,
                "bar_count": 0, "arc_count": 0, "img_w": 0, "img_h": 0}

    # Return actual processed image dimensions so script.py uses them
    # instead of re-reading EXIF (which may differ between CPython/IronPython).
    proc_h, proc_w = binary_img.shape[:2]

    result = _recognize_skeleton(binary_img)
    segments = result["segments"]

    if debug_path:
        try:
            _save_debug_image(
                image_path, binary_img,
                result.get("wall_mask"), result.get("skel"),
                segments, debug_path,
                binary_raw=result.get("binary"),
            )
        except Exception:
            pass

    return {
        "segments_px": segments,
        "doors_px": [],
        "wall_thickness_px": 20.0,
        "bar_count": len(segments),
        "arc_count": 0,
        "img_w": proc_w,
        "img_h": proc_h,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--site-packages", dest="site_packages")
    parser.add_argument("--hint-cx", type=float)
    parser.add_argument("--hint-cy", type=float)
    parser.add_argument("--hint-width", type=float)
    parser.add_argument("--hint-orientation")
    parser.add_argument("--debug", help="Path to save debug visualization image")
    args = parser.parse_args(argv)

    hint = None
    if args.hint_cx is not None and args.hint_cy is not None:
        hint = {
            "center": (float(args.hint_cx), float(args.hint_cy)),
            "width_px": float(args.hint_width or 0.0),
        }
        if args.hint_orientation:
            hint["preferred_orientation"] = args.hint_orientation

    result = recognize_sketch(
        args.image, door_hint=hint,
        site_packages=args.site_packages, debug_path=args.debug,
    )
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
