"""Pure geometry computation helpers for room layout (cm domain)."""


def build_room_cm(measurements):
    w = float(measurements["room_width_cm"])
    h = float(measurements["room_height_cm"])
    t = float(measurements["wall_thickness_cm"])

    if w <= 0 or h <= 0 or t <= 0:
        raise ValueError("Invalid room dimensions")

    half = t * 0.5

    # Inner corners in cm, SW origin.
    inner_sw = (0.0, 0.0)
    inner_se = (w, 0.0)
    inner_ne = (w, h)
    inner_nw = (0.0, h)

    # Wall centerline rectangle.
    cl_sw = (-half, -half)
    cl_se = (w + half, -half)
    cl_ne = (w + half, h + half)
    cl_nw = (-half, h + half)

    door_center_x = float(measurements["door_left_offset_cm"]) + (float(measurements["door_width_cm"]) * 0.5)
    window_center_x = w - float(measurements["window_right_offset_cm"]) - (float(measurements["window_width_cm"]) * 0.5)

    return {
        "inner": {
            "sw": inner_sw,
            "se": inner_se,
            "ne": inner_ne,
            "nw": inner_nw,
        },
        "centerline": {
            "sw": cl_sw,
            "se": cl_se,
            "ne": cl_ne,
            "nw": cl_nw,
        },
        "door_center_x_cm": door_center_x,
        "window_center_x_cm": window_center_x,
        "half_wall_cm": half,
    }
