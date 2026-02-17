"""Pure geometry solver used by the pyRevit room-from-image command.

This module is intentionally Revit-free so it can be unit-tested outside Revit.
All points are tuples: (x, y) or (x, y, z).
"""

import math


class LayoutSolverError(ValueError):
    pass


def _pt3(p):
    if len(p) >= 3:
        return (float(p[0]), float(p[1]), float(p[2]))
    return (float(p[0]), float(p[1]), 0.0)


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mul(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def _dot(a, b):
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])


def _len_xy(v):
    return math.sqrt((v[0] * v[0]) + (v[1] * v[1]))


def _normalize_xy(v):
    ln = _len_xy(v)
    if ln < 1e-9:
        raise LayoutSolverError("Invalid point selection. Two points are too close.")
    return (v[0] / ln, v[1] / ln, 0.0)


def _rot_left_90(v):
    return (-v[1], v[0], 0.0)


def _project_scalar(origin, axis_unit, point):
    return _dot(_sub(point, origin), axis_unit)


def calibration_scale(calib_a, calib_b, known_len):
    if calib_a is None or calib_b is None:
        return 1.0
    a = _pt3(calib_a)
    b = _pt3(calib_b)
    ln = _len_xy(_sub(b, a))
    if ln < 1e-9:
        raise LayoutSolverError("Calibration points are too close.")
    if known_len <= 0:
        raise LayoutSolverError("Known calibration length must be positive.")
    return float(known_len) / ln


def solve_layout(
    inner_sw_raw,
    inner_se_raw,
    inner_ne_raw,
    inner_nw_raw,
    door_left_raw,
    door_right_raw,
    window_left_raw,
    window_right_raw,
    scale_factor,
    min_opening,
):
    sw = _pt3(inner_sw_raw)
    se = _pt3(inner_se_raw)
    ne = _pt3(inner_ne_raw)
    nw = _pt3(inner_nw_raw)

    d_l = _pt3(door_left_raw)
    d_r = _pt3(door_right_raw)
    w_l = _pt3(window_left_raw)
    w_r = _pt3(window_right_raw)

    x_axis = _normalize_xy(_sub(se, sw))
    y_axis = _rot_left_90(x_axis)
    if _dot(y_axis, _sub(nw, sw)) < 0:
        y_axis = _mul(y_axis, -1.0)

    len_x_1 = abs(_project_scalar(sw, x_axis, se))
    len_x_2 = abs(_project_scalar(nw, x_axis, ne))
    inner_x_len = ((len_x_1 + len_x_2) * 0.5) * scale_factor

    len_y_1 = abs(_project_scalar(sw, y_axis, nw))
    len_y_2 = abs(_project_scalar(se, y_axis, ne))
    inner_y_len = ((len_y_1 + len_y_2) * 0.5) * scale_factor

    if inner_x_len <= 0 or inner_y_len <= 0:
        raise LayoutSolverError("Computed room size is invalid.")

    # Keep SW anchor where the user clicked; lengths are scaled from calibration.
    inner_sw = sw
    inner_se = _add(inner_sw, _mul(x_axis, inner_x_len))
    inner_nw = _add(inner_sw, _mul(y_axis, inner_y_len))
    inner_ne = _add(inner_se, _mul(y_axis, inner_y_len))

    door_left_s = _project_scalar(inner_sw, x_axis, d_l) * scale_factor
    door_right_s = _project_scalar(inner_sw, x_axis, d_r) * scale_factor
    if door_left_s > door_right_s:
        door_left_s, door_right_s = door_right_s, door_left_s

    window_left_s = _project_scalar(inner_sw, x_axis, w_l) * scale_factor
    window_right_s = _project_scalar(inner_sw, x_axis, w_r) * scale_factor
    if window_left_s > window_right_s:
        window_left_s, window_right_s = window_right_s, window_left_s

    door_width = max(float(min_opening), abs(door_right_s - door_left_s))
    door_center_s = (door_left_s + door_right_s) * 0.5

    window_width = max(float(min_opening), abs(window_right_s - window_left_s))
    window_center_s = (window_left_s + window_right_s) * 0.5

    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "inner_x_len": inner_x_len,
        "inner_y_len": inner_y_len,
        "inner_sw": inner_sw,
        "inner_se": inner_se,
        "inner_nw": inner_nw,
        "inner_ne": inner_ne,
        "door_left_s": door_left_s,
        "door_right_s": door_right_s,
        "door_width": door_width,
        "door_center_s": door_center_s,
        "window_left_s": window_left_s,
        "window_right_s": window_right_s,
        "window_width": window_width,
        "window_center_s": window_center_s,
    }
