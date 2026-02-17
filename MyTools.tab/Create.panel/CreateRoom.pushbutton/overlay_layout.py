"""Pure layout math for image overlay + instruction panel placement."""


def clamp(value, lo, hi):
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def compute_overlay_and_note_layout(
    view_min,
    view_max,
    image_aspect,
    left_fraction=0.60,
    margin_fraction=0.03,
    note_fraction=0.27,
):
    """Return deterministic layout rectangles.

    Args:
        view_min: (x, y)
        view_max: (x, y)
        image_aspect: width / height (>0)

    Returns:
        dict with:
            image_center (x, y)
            image_width
            image_height
            image_left
            image_right
            image_bottom
            image_top
            note_origin (x, y)
            note_width
            right_panel_left
    """
    x0, y0 = float(view_min[0]), float(view_min[1])
    x1, y1 = float(view_max[0]), float(view_max[1])

    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid view bounds")
    if image_aspect <= 0:
        raise ValueError("image_aspect must be > 0")

    vw = x1 - x0
    vh = y1 - y0

    left_fraction = clamp(left_fraction, 0.45, 0.85)
    margin_x = vw * clamp(margin_fraction, 0.01, 0.10)
    margin_y = vh * clamp(margin_fraction, 0.01, 0.10)

    right_panel_left = x0 + vw * left_fraction

    # Left image pane bounds.
    pane_x0 = x0 + margin_x
    pane_x1 = right_panel_left - margin_x
    pane_y0 = y0 + margin_y
    pane_y1 = y1 - margin_y

    pane_w = max(pane_x1 - pane_x0, vw * 0.20)
    pane_h = max(pane_y1 - pane_y0, vh * 0.20)

    # Fit image fully into pane while preserving aspect ratio.
    fit_w = pane_w
    fit_h = fit_w / image_aspect
    if fit_h > pane_h:
        fit_h = pane_h
        fit_w = fit_h * image_aspect

    cx = pane_x0 + (pane_w * 0.5)
    cy = pane_y0 + (pane_h * 0.5)

    image_left = cx - (fit_w * 0.5)
    image_right = cx + (fit_w * 0.5)
    image_bottom = cy - (fit_h * 0.5)
    image_top = cy + (fit_h * 0.5)

    # Right instruction panel.
    note_x = max(right_panel_left + margin_x, image_right + margin_x)
    note_y = y1 - margin_y
    note_width = max(vw * 0.12, (x1 - note_x - margin_x) * clamp(note_fraction / 0.27, 0.75, 1.50))
    note_width = min(note_width, max(vw * 0.10, x1 - note_x - margin_x))

    return {
        "image_center": (cx, cy),
        "image_width": fit_w,
        "image_height": fit_h,
        "image_left": image_left,
        "image_right": image_right,
        "image_bottom": image_bottom,
        "image_top": image_top,
        "note_origin": (note_x, note_y),
        "note_width": note_width,
        "right_panel_left": right_panel_left,
    }
