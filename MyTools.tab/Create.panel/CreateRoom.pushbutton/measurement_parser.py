"""Sketch measurement parsing and merge logic (cm domain)."""

import re


DEFAULTS_CM = {
    "room_width_cm": 370.0,
    "room_height_cm": 290.0,
    "wall_thickness_cm": 30.0,
    "door_width_cm": 100.0,
    "door_height_cm": 210.0,
    "door_left_offset_cm": 90.0,
    "window_width_cm": 100.0,
    "window_height_cm": 100.0,
    "window_right_offset_cm": 30.0,
    "window_sill_cm": 105.0,
}


def _to_float(text):
    try:
        return float(str(text).replace(",", "."))
    except Exception:
        return None


def extract_numeric_candidates(text):
    s = str(text or "")
    candidates = []

    for token in re.findall(r"\d+(?:[\.,]\d+)?", s):
        v = _to_float(token)
        if v is not None:
            candidates.append(v)

    digits = re.sub(r"\D", "", s)
    if len(digits) >= 4:
        # Sliding windows to recover merged OCR blobs like "61110" -> 61/110/111 etc.
        for win in (2, 3):
            for i in range(0, len(digits) - win + 1):
                seg = digits[i:i + win]
                v = _to_float(seg)
                if v is not None:
                    candidates.append(v)

    # Dedupe while preserving order.
    out = []
    seen = set()
    for v in candidates:
        k = round(v, 4)
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


def _best_by_target(values, lo, hi, target):
    vals = [v for v in values if lo <= v <= hi]
    if not vals:
        return None
    return min(vals, key=lambda v: abs(v - target))


def parse_ocr_tokens(tokens):
    numbers = []
    for tok in tokens or []:
        text = tok.get("text", "") if isinstance(tok, dict) else str(tok)
        numbers.extend(extract_numeric_candidates(text))

    out = {
        "raw_numbers": numbers,
        "parsed": {},
    }

    # Heuristic picks by expected ranges.
    room_candidates = sorted([v for v in numbers if 220 <= v <= 900], reverse=True)
    if room_candidates:
        out["parsed"]["room_width_cm"] = room_candidates[0]
    if len(room_candidates) > 1:
        out["parsed"]["room_height_cm"] = room_candidates[1]

    maybe_thickness = _best_by_target(numbers, 15, 60, 30)
    if maybe_thickness is not None:
        out["parsed"]["wall_thickness_cm"] = maybe_thickness

    door_w = _best_by_target(numbers, 60, 140, 100)
    if door_w is not None:
        out["parsed"]["door_width_cm"] = door_w

    door_h = _best_by_target(numbers, 180, 240, 210)
    if door_h is not None:
        out["parsed"]["door_height_cm"] = door_h

    window_w = _best_by_target(numbers, 60, 140, 100)
    if window_w is not None:
        out["parsed"]["window_width_cm"] = window_w

    window_h = _best_by_target(numbers, 60, 160, 100)
    if window_h is not None:
        out["parsed"]["window_height_cm"] = window_h

    door_left = _best_by_target(numbers, 40, 180, 90)
    if door_left is not None:
        out["parsed"]["door_left_offset_cm"] = door_left

    window_right = _best_by_target(numbers, 10, 120, 30)
    if window_right is not None:
        out["parsed"]["window_right_offset_cm"] = window_right

    sill = _best_by_target(numbers, 70, 140, 105)
    if sill is not None:
        out["parsed"]["window_sill_cm"] = sill

    return out


def _validate_positive(data, keys):
    for k in keys:
        v = data.get(k)
        if v is None:
            continue
        if v <= 0:
            data[k] = DEFAULTS_CM[k]


def merge_measurements_cm(pick_measurements, ocr_parsed, prefer_ocr=True):
    merged = dict(DEFAULTS_CM)

    for k, v in (pick_measurements or {}).items():
        if k in merged and v is not None:
            merged[k] = float(v)

    if prefer_ocr:
        for k, v in (ocr_parsed or {}).items():
            if k in merged and v is not None:
                merged[k] = float(v)

    _validate_positive(merged, list(DEFAULTS_CM.keys()))

    # Keep offsets bounded inside room width.
    room_w = merged["room_width_cm"]
    merged["door_left_offset_cm"] = max(0.0, min(merged["door_left_offset_cm"], max(0.0, room_w - merged["door_width_cm"])))
    merged["window_right_offset_cm"] = max(0.0, min(merged["window_right_offset_cm"], max(0.0, room_w - merged["window_width_cm"])))

    return merged
