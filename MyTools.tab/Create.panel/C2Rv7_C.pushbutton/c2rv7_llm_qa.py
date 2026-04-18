# -*- coding: utf-8 -*-
"""
C2Rv7 LLM QA sidecar.

Optional Claude-based sanity checks on the wall pipeline. Three hook points:

1. validate_exterior_envelope(lines)
   After ext_main is picked, render it and ask Claude whether it looks like a
   valid building outer envelope or a fragment / translated-duplicate artifact.

2. classify_interior_fragments(all_lines, candidate_fragments)
   Before _remove_small_components / _remove_small_interior_fragments fires,
   show Claude the full plan with each candidate fragment highlighted and
   ask per-fragment whether it is a real short wall or a CAD artifact.

3. sanity_check_thicknesses(ext_thick_cm, int_thick_cm, ext_lines, int_lines)
   After thickness estimation, ask Claude whether these numbers look plausible
   for the rendered plan, and whether the ext/int split looks right.

Design rules:
- Entirely optional. Missing API key, missing render library, or network
  failure => every function returns a pass-through "no opinion" result and
  logs a note. Never raises into the host script.
- Hard timeout per call (default 15s). Architects won't wait.
- All artifacts (rendered PNGs, prompts, raw responses) written to the
  snapshot folder for post-mortem debugging.
- Pure stdlib HTTP (urllib) so it works wherever pyRevit CPython runs.
- Rendering uses matplotlib if available, else Pillow, else skips.

Entry-point API key:
    env var ANTHROPIC_API_KEY

Configuration (passed via cfg dict):
    llm_qa_enabled            : bool, default False
    llm_qa_model              : str,  default "claude-opus-4-7"
    llm_qa_timeout_s          : int,  default 15
    llm_qa_max_fragments      : int,  default 12  (fragments per batch call)
    llm_qa_render_px          : int,  default 1400  (image longest side)
    llm_qa_envelope           : bool, default True
    llm_qa_fragments          : bool, default True
    llm_qa_thickness          : bool, default False  (informational only)
"""

import base64
import io
import json
import math
import os
import socket
import time

try:
    import urllib.request as _urlreq
    import urllib.error as _urlerr
except ImportError:  # pragma: no cover - IronPython safety
    _urlreq = None
    _urlerr = None


# ---------------------------------------------------------------------------
# Public result shapes
# ---------------------------------------------------------------------------

RESULT_NO_OPINION = "no_opinion"
RESULT_ACCEPT = "accept"
RESULT_REJECT = "reject"


def _blank_envelope_result(reason):
    return {
        "verdict": RESULT_NO_OPINION,
        "confidence": 0.0,
        "reason": reason,
        "raw": None,
    }


def _blank_fragments_result(candidates, reason):
    return {
        "verdicts": {str(i): RESULT_NO_OPINION for i in range(len(candidates or []))},
        "confidences": {str(i): 0.0 for i in range(len(candidates or []))},
        "reasons": {str(i): reason for i in range(len(candidates or []))},
        "raw": None,
    }


def _blank_thickness_result(reason):
    return {
        "verdict": RESULT_NO_OPINION,
        "notes": reason,
        "raw": None,
    }


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg_bool(cfg, key, default):
    v = (cfg or {}).get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(default)


def _cfg_int(cfg, key, default):
    try:
        return int((cfg or {}).get(key, default))
    except Exception:
        return int(default)


def _cfg_str(cfg, key, default):
    v = (cfg or {}).get(key, default)
    return default if v is None else str(v)


def qa_enabled(cfg):
    if not _cfg_bool(cfg, "llm_qa_enabled", False):
        return False
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    if _urlreq is None:
        return False
    return True


# ---------------------------------------------------------------------------
# Snapshot logging
# ---------------------------------------------------------------------------

def _log(snapshot, msg):
    if snapshot is None:
        return
    try:
        snapshot.log("[LLM-QA] " + str(msg))
    except Exception:
        pass


def _save_bytes(snapshot, name, payload):
    """Write a file into the snapshot folder. Returns path or None."""
    if snapshot is None:
        return None
    folder = None
    for attr in ("folder", "run_folder", "path", "dir"):
        try:
            folder = getattr(snapshot, attr, None)
            if folder and os.path.isdir(str(folder)):
                folder = str(folder)
                break
            folder = None
        except Exception:
            folder = None
    if not folder:
        return None
    try:
        full = os.path.join(folder, name)
        with open(full, "wb") as f:
            f.write(payload)
        return full
    except Exception:
        return None


def _save_text(snapshot, name, text):
    try:
        return _save_bytes(snapshot, name, text.encode("utf-8"))
    except Exception:
        return None


def _save_json(snapshot, name, obj):
    if snapshot is None:
        return None
    try:
        return snapshot.save_json(name, obj)
    except Exception:
        try:
            return _save_text(snapshot, name, json.dumps(obj, indent=2))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Geometry helpers (local to avoid import cycles with script.py)
# ---------------------------------------------------------------------------

def _line_len_cm(ln):
    dx = float(ln.get("x2", 0.0)) - float(ln.get("x1", 0.0))
    dy = float(ln.get("y2", 0.0)) - float(ln.get("y1", 0.0))
    return math.sqrt((dx * dx) + (dy * dy))


def _bbox_of_lines(lines):
    pts = []
    for ln in (lines or []):
        pts.append((float(ln.get("x1", 0.0)), float(ln.get("y1", 0.0))))
        pts.append((float(ln.get("x2", 0.0)), float(ln.get("y2", 0.0))))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _have_matplotlib():
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def _have_pillow():
    try:
        from PIL import Image  # noqa: F401
        from PIL import ImageDraw  # noqa: F401
        return True
    except Exception:
        return False


def _render_with_matplotlib(groups, bbox, px_long):
    """
    groups: list of {"lines": [...], "color": "#rrggbb", "width": float, "label": str}
    bbox:   (minx, miny, maxx, maxy) in cm
    Returns PNG bytes or None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
    except Exception:
        return None

    try:
        minx, miny, maxx, maxy = bbox
        pad_cm = max(50.0, 0.05 * max(maxx - minx, maxy - miny))
        minx -= pad_cm
        miny -= pad_cm
        maxx += pad_cm
        maxy += pad_cm
        span_x = max(1.0, maxx - minx)
        span_y = max(1.0, maxy - miny)
        aspect = span_x / span_y
        if aspect >= 1.0:
            w_in = max(4.0, float(px_long) / 150.0)
            h_in = w_in / aspect
        else:
            h_in = max(4.0, float(px_long) / 150.0)
            w_in = h_in * aspect

        fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=150)
        ax.set_facecolor("white")
        for grp in groups:
            color = grp.get("color", "#111111")
            width = float(grp.get("width", 1.0))
            for ln in grp.get("lines", []) or []:
                ax.plot(
                    [float(ln.get("x1", 0.0)), float(ln.get("x2", 0.0))],
                    [float(ln.get("y1", 0.0)), float(ln.get("y2", 0.0))],
                    color=color,
                    linewidth=width,
                    solid_capstyle="round",
                )
        # Legend entries
        handles = []
        for grp in groups:
            lbl = grp.get("label")
            if not lbl:
                continue
            handles.append(mlines.Line2D(
                [], [],
                color=grp.get("color", "#111111"),
                linewidth=float(grp.get("width", 1.0)),
                label=lbl,
            ))
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def _render_with_pillow(groups, bbox, px_long):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    try:
        minx, miny, maxx, maxy = bbox
        pad_cm = max(50.0, 0.05 * max(maxx - minx, maxy - miny))
        minx -= pad_cm
        miny -= pad_cm
        maxx += pad_cm
        maxy += pad_cm
        span_x = max(1.0, maxx - minx)
        span_y = max(1.0, maxy - miny)
        if span_x >= span_y:
            W = int(px_long)
            H = max(32, int(px_long * span_y / span_x))
        else:
            H = int(px_long)
            W = max(32, int(px_long * span_x / span_y))
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        def _xy(x, y):
            px = (float(x) - minx) / span_x * (W - 1)
            py = (H - 1) - (float(y) - miny) / span_y * (H - 1)
            return (px, py)

        for grp in groups:
            color_hex = grp.get("color", "#111111").lstrip("#")
            if len(color_hex) == 6:
                color = (
                    int(color_hex[0:2], 16),
                    int(color_hex[2:4], 16),
                    int(color_hex[4:6], 16),
                )
            else:
                color = (17, 17, 17)
            width = max(1, int(round(float(grp.get("width", 1.0)))))
            for ln in grp.get("lines", []) or []:
                p1 = _xy(ln.get("x1", 0.0), ln.get("y1", 0.0))
                p2 = _xy(ln.get("x2", 0.0), ln.get("y2", 0.0))
                draw.line([p1, p2], fill=color, width=width)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _render_png(groups, bbox, px_long):
    if _have_matplotlib():
        out = _render_with_matplotlib(groups, bbox, px_long)
        if out:
            return out, "matplotlib"
    if _have_pillow():
        out = _render_with_pillow(groups, bbox, px_long)
        if out:
            return out, "pillow"
    return None, None


# ---------------------------------------------------------------------------
# Claude API call (stdlib only)
# ---------------------------------------------------------------------------

def _call_claude(model, png_bytes, user_text, timeout_s, system_text=None, max_tokens=1024):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or _urlreq is None:
        return None, "no_api_key_or_urllib"

    content = []
    if png_bytes:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(png_bytes).decode("ascii"),
            },
        })
    content.append({"type": "text", "text": user_text})

    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "messages": [{"role": "user", "content": content}],
    }
    if system_text:
        payload["system"] = system_text

    data = json.dumps(payload).encode("utf-8")
    req = _urlreq.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with _urlreq.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except socket.timeout:
        return None, "timeout"
    except _urlerr.HTTPError as ex:
        try:
            body = ex.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return None, "http_error_{}: {}".format(ex.code, body[:400])
    except Exception as ex:
        return None, "network_error: {}".format(ex)

    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as ex:
        return None, "bad_json: {}".format(ex)

    try:
        text_out = ""
        for block in obj.get("content") or []:
            if block.get("type") == "text":
                text_out += block.get("text") or ""
        return {"text": text_out, "full": obj}, None
    except Exception as ex:
        return None, "response_shape: {}".format(ex)


def _extract_json_from_text(text):
    """Claude sometimes wraps JSON in prose or code fences. Pull the first object out."""
    if not text:
        return None
    s = text.strip()
    # Strip code fences
    if s.startswith("```"):
        nl = s.find("\n")
        if nl >= 0:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # Find first balanced {...}
    start = s.find("{")
    while start >= 0:
        depth = 0
        for i in range(start, len(s)):
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    frag = s[start:i + 1]
                    try:
                        return json.loads(frag)
                    except Exception:
                        break
        start = s.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------
# Hook 1: exterior envelope validation
# ---------------------------------------------------------------------------

ENVELOPE_SYSTEM = (
    "You are reviewing the output of an automated CAD-to-Revit wall conversion. "
    "You will be shown a rendered floor plan containing only the lines the tool "
    "has identified as the main exterior building envelope. Your job is to decide "
    "whether this looks like a reasonable building outer envelope (accept) or "
    "whether it looks like a fragment, a translated duplicate, or otherwise "
    "wrong (reject). Respond ONLY with a JSON object."
)

ENVELOPE_USER = (
    "Rendered in the image: the exterior wall lines the tool kept as the main "
    "building footprint.\n\n"
    "Decide one of: accept, reject, no_opinion.\n"
    "- accept: forms a single coherent outer envelope (closed or nearly closed outline).\n"
    "- reject: looks like only a portion of a building, a stray fragment, two "
    "disjoint building copies, a single wall with nothing else, or otherwise "
    "not a plausible whole-building exterior.\n"
    "- no_opinion: you cannot tell.\n\n"
    "Respond with ONLY this JSON shape and nothing else:\n"
    '{"verdict": "accept|reject|no_opinion", "confidence": 0.0-1.0, "reason": "<short>"}'
)


def validate_exterior_envelope(cfg, ext_main_lines, snapshot=None):
    if not qa_enabled(cfg):
        return _blank_envelope_result("qa_disabled")
    if not _cfg_bool(cfg, "llm_qa_envelope", True):
        return _blank_envelope_result("hook_disabled")
    if not ext_main_lines:
        return _blank_envelope_result("no_lines")

    bbox = _bbox_of_lines(ext_main_lines)
    if not bbox:
        return _blank_envelope_result("no_bbox")

    px = _cfg_int(cfg, "llm_qa_render_px", 1400)
    png, renderer = _render_png(
        [{"lines": ext_main_lines, "color": "#111111", "width": 2.0, "label": "ext envelope"}],
        bbox,
        px,
    )
    if not png:
        return _blank_envelope_result("no_renderer")

    _save_bytes(snapshot, "llm_qa_envelope.png", png)
    _log(snapshot, "envelope render ok via {}, {} lines".format(renderer, len(ext_main_lines)))

    t0 = time.time()
    resp, err = _call_claude(
        _cfg_str(cfg, "llm_qa_model", "claude-opus-4-7"),
        png,
        ENVELOPE_USER,
        _cfg_int(cfg, "llm_qa_timeout_s", 15),
        system_text=ENVELOPE_SYSTEM,
        max_tokens=300,
    )
    dt = time.time() - t0
    _log(snapshot, "envelope api dt={:.2f}s err={}".format(dt, err))

    if resp is None:
        return _blank_envelope_result("api_error: {}".format(err))

    _save_text(snapshot, "llm_qa_envelope_response.txt", resp.get("text") or "")
    parsed = _extract_json_from_text(resp.get("text") or "")
    if not isinstance(parsed, dict):
        return _blank_envelope_result("unparseable_response")

    verdict_raw = str(parsed.get("verdict") or "").strip().lower()
    if verdict_raw in (RESULT_ACCEPT, RESULT_REJECT, RESULT_NO_OPINION):
        verdict = verdict_raw
    else:
        verdict = RESULT_NO_OPINION
    try:
        conf = float(parsed.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    reason = str(parsed.get("reason") or "")[:400]

    out = {
        "verdict": verdict,
        "confidence": max(0.0, min(1.0, conf)),
        "reason": reason,
        "raw": resp.get("text"),
    }
    _save_json(snapshot, "llm_qa_envelope.json", out)
    return out


# ---------------------------------------------------------------------------
# Hook 2: interior fragment classification
# ---------------------------------------------------------------------------

FRAGMENT_SYSTEM = (
    "You are reviewing an automated CAD-to-Revit wall conversion. The rendered "
    "image shows a full floor plan with the current interior wall centerlines "
    "in dark blue, and a set of numbered candidate fragments highlighted in red. "
    "These candidates are currently flagged for removal because they are small "
    "or look like artifacts. For each numbered candidate, decide whether it is "
    "(a) a legitimate short wall that should be kept, or (b) a CAD artifact / "
    "door-swing residue / jamb leftover / duplicate that should be removed. "
    "Respond ONLY with a JSON object."
)

FRAGMENT_USER_TEMPLATE = (
    "The plan contains {n} numbered candidate fragments highlighted in red and "
    "labeled F0, F1, F2, etc. Each one is a small interior wall segment "
    "currently marked for removal.\n\n"
    "For each candidate, decide:\n"
    '- "keep": it looks like a legitimate short interior wall\n'
    '- "remove": it is an artifact, door-swing residue, duplicate, or stray line\n'
    '- "unsure": you cannot tell\n\n'
    "Respond with ONLY this JSON shape and nothing else:\n"
    "{{\n"
    '  "fragments": {{\n'
    '    "F0": {{"verdict": "keep|remove|unsure", "confidence": 0.0-1.0, "reason": "<short>"}},\n'
    '    "F1": {{ ... }}\n'
    "  }}\n"
    "}}"
)


def _label_points(lines):
    bbox = _bbox_of_lines(lines)
    if not bbox:
        return None
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def _render_fragments_with_labels(all_int_lines, fragment_groups, bbox, px_long):
    """Render base interior lines + highlighted fragments with text labels (F0, F1...)."""
    if _have_matplotlib():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.lines as mlines

            minx, miny, maxx, maxy = bbox
            pad_cm = max(50.0, 0.05 * max(maxx - minx, maxy - miny))
            minx -= pad_cm
            miny -= pad_cm
            maxx += pad_cm
            maxy += pad_cm
            span_x = max(1.0, maxx - minx)
            span_y = max(1.0, maxy - miny)
            aspect = span_x / span_y
            if aspect >= 1.0:
                w_in = max(5.0, float(px_long) / 150.0)
                h_in = w_in / aspect
            else:
                h_in = max(5.0, float(px_long) / 150.0)
                w_in = h_in * aspect

            fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=150)
            ax.set_facecolor("white")
            # Base interior lines
            for ln in all_int_lines or []:
                ax.plot(
                    [float(ln.get("x1", 0.0)), float(ln.get("x2", 0.0))],
                    [float(ln.get("y1", 0.0)), float(ln.get("y2", 0.0))],
                    color="#1f3b8a", linewidth=1.1, solid_capstyle="round",
                )
            # Fragments in red, labeled
            for idx, grp in enumerate(fragment_groups):
                frag_lines = grp.get("lines") or []
                label = grp.get("label") or "F{}".format(idx)
                for ln in frag_lines:
                    ax.plot(
                        [float(ln.get("x1", 0.0)), float(ln.get("x2", 0.0))],
                        [float(ln.get("y1", 0.0)), float(ln.get("y2", 0.0))],
                        color="#d11a1a", linewidth=3.0, solid_capstyle="round",
                    )
                lbl_pt = _label_points(frag_lines)
                if lbl_pt is not None:
                    ax.annotate(
                        label,
                        xy=lbl_pt,
                        xytext=(8, 8),
                        textcoords="offset points",
                        fontsize=11,
                        color="#8a0a0a",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="#fff0f0", ec="#8a0a0a", lw=0.8),
                    )

            handles = [
                mlines.Line2D([], [], color="#1f3b8a", linewidth=2.0, label="interior walls (kept)"),
                mlines.Line2D([], [], color="#d11a1a", linewidth=3.0, label="candidate fragments"),
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            return buf.getvalue(), "matplotlib"
        except Exception:
            try:
                import matplotlib.pyplot as plt
                plt.close("all")
            except Exception:
                pass

    # Pillow fallback (no text labels, just color coding)
    groups = [{"lines": all_int_lines, "color": "#1f3b8a", "width": 2.0, "label": None}]
    for grp in fragment_groups:
        groups.append({"lines": grp.get("lines") or [], "color": "#d11a1a", "width": 4.0, "label": None})
    png, renderer = _render_png(groups, bbox, px_long)
    return png, renderer


def classify_interior_fragments(cfg, all_int_lines, candidate_fragments, snapshot=None):
    """
    candidate_fragments: list of lists of lines. Each inner list is one fragment.
    Returns dict keyed "F0", "F1", ... to {"verdict": keep|remove|unsure, ...}
    plus flat results indexed by fragment position for convenience.
    """
    n = len(candidate_fragments or [])
    if not qa_enabled(cfg):
        return _blank_fragments_result(candidate_fragments, "qa_disabled")
    if not _cfg_bool(cfg, "llm_qa_fragments", True):
        return _blank_fragments_result(candidate_fragments, "hook_disabled")
    if n == 0:
        return _blank_fragments_result([], "no_candidates")

    batch_max = max(1, _cfg_int(cfg, "llm_qa_max_fragments", 12))
    if n > batch_max:
        # Process in batches; concat results
        combined = {
            "verdicts": {},
            "confidences": {},
            "reasons": {},
            "raw": [],
        }
        for start in range(0, n, batch_max):
            chunk = candidate_fragments[start:start + batch_max]
            chunk_res = classify_interior_fragments(cfg, all_int_lines, chunk, snapshot=snapshot)
            for i in range(len(chunk)):
                global_key = str(start + i)
                local_key = str(i)
                combined["verdicts"][global_key] = chunk_res["verdicts"].get(local_key, RESULT_NO_OPINION)
                combined["confidences"][global_key] = chunk_res["confidences"].get(local_key, 0.0)
                combined["reasons"][global_key] = chunk_res["reasons"].get(local_key, "")
            combined["raw"].append(chunk_res.get("raw"))
        return combined

    # Build combined bbox
    all_for_bbox = list(all_int_lines or [])
    for frag in candidate_fragments:
        all_for_bbox.extend(frag or [])
    bbox = _bbox_of_lines(all_for_bbox)
    if not bbox:
        return _blank_fragments_result(candidate_fragments, "no_bbox")

    fragment_groups = []
    for i, frag in enumerate(candidate_fragments):
        fragment_groups.append({"lines": frag, "label": "F{}".format(i)})

    px = _cfg_int(cfg, "llm_qa_render_px", 1400)
    png, renderer = _render_fragments_with_labels(all_int_lines, fragment_groups, bbox, px)
    if not png:
        return _blank_fragments_result(candidate_fragments, "no_renderer")

    _save_bytes(snapshot, "llm_qa_fragments.png", png)
    _log(snapshot, "fragments render ok via {}, {} candidates".format(renderer, n))

    user_text = FRAGMENT_USER_TEMPLATE.format(n=n)
    t0 = time.time()
    resp, err = _call_claude(
        _cfg_str(cfg, "llm_qa_model", "claude-opus-4-7"),
        png,
        user_text,
        _cfg_int(cfg, "llm_qa_timeout_s", 15),
        system_text=FRAGMENT_SYSTEM,
        max_tokens=1200,
    )
    dt = time.time() - t0
    _log(snapshot, "fragments api dt={:.2f}s err={}".format(dt, err))

    if resp is None:
        return _blank_fragments_result(candidate_fragments, "api_error: {}".format(err))

    _save_text(snapshot, "llm_qa_fragments_response.txt", resp.get("text") or "")
    parsed = _extract_json_from_text(resp.get("text") or "")
    if not isinstance(parsed, dict):
        return _blank_fragments_result(candidate_fragments, "unparseable_response")

    frags_obj = parsed.get("fragments") or {}
    verdicts = {}
    confidences = {}
    reasons = {}
    for i in range(n):
        key = "F{}".format(i)
        entry = frags_obj.get(key) if isinstance(frags_obj, dict) else None
        if not isinstance(entry, dict):
            verdicts[str(i)] = RESULT_NO_OPINION
            confidences[str(i)] = 0.0
            reasons[str(i)] = "missing_in_response"
            continue
        v = str(entry.get("verdict") or "").strip().lower()
        if v in ("keep", "remove", "unsure"):
            verdicts[str(i)] = v
        else:
            verdicts[str(i)] = "unsure"
        try:
            confidences[str(i)] = max(0.0, min(1.0, float(entry.get("confidence", 0.0))))
        except Exception:
            confidences[str(i)] = 0.0
        reasons[str(i)] = str(entry.get("reason") or "")[:400]

    out = {
        "verdicts": verdicts,
        "confidences": confidences,
        "reasons": reasons,
        "raw": resp.get("text"),
    }
    _save_json(snapshot, "llm_qa_fragments.json", out)
    return out


# ---------------------------------------------------------------------------
# Hook 3: thickness sanity check (informational only)
# ---------------------------------------------------------------------------

THICKNESS_SYSTEM = (
    "You are reviewing an automated CAD-to-Revit wall conversion. You will be "
    "shown a rendered plan with exterior walls in black and interior walls in blue. "
    "The tool has estimated wall thicknesses from paired CAD faces. Decide whether "
    "those thickness numbers look plausible for the drawn walls. Respond ONLY with JSON."
)

THICKNESS_USER_TEMPLATE = (
    "The tool estimated:\n"
    "- exterior wall thickness: {ext} cm\n"
    "- interior wall thickness: {int} cm\n\n"
    "Based only on the rendered plan proportions (not any absolute dimension you "
    "might try to read off the drawing), decide whether these numbers look "
    "reasonable for a residential/commercial Israeli plan. Typical exterior is "
    "20-30 cm, typical interior is 7-15 cm.\n\n"
    "Respond with ONLY this JSON:\n"
    '{{"verdict": "plausible|suspicious|no_opinion", "notes": "<short>"}}'
)


def sanity_check_thicknesses(cfg, ext_thick_cm, int_thick_cm, ext_lines, int_lines, snapshot=None):
    if not qa_enabled(cfg):
        return _blank_thickness_result("qa_disabled")
    if not _cfg_bool(cfg, "llm_qa_thickness", False):
        return _blank_thickness_result("hook_disabled")

    combined = list(ext_lines or []) + list(int_lines or [])
    bbox = _bbox_of_lines(combined)
    if not bbox:
        return _blank_thickness_result("no_bbox")

    px = _cfg_int(cfg, "llm_qa_render_px", 1400)
    png, renderer = _render_png(
        [
            {"lines": ext_lines, "color": "#111111", "width": 2.2, "label": "exterior"},
            {"lines": int_lines, "color": "#1f3b8a", "width": 1.3, "label": "interior"},
        ],
        bbox,
        px,
    )
    if not png:
        return _blank_thickness_result("no_renderer")

    _save_bytes(snapshot, "llm_qa_thickness.png", png)

    user_text = THICKNESS_USER_TEMPLATE.format(
        ext="{:.1f}".format(float(ext_thick_cm)),
        int="{:.1f}".format(float(int_thick_cm)),
    )
    t0 = time.time()
    resp, err = _call_claude(
        _cfg_str(cfg, "llm_qa_model", "claude-opus-4-7"),
        png,
        user_text,
        _cfg_int(cfg, "llm_qa_timeout_s", 15),
        system_text=THICKNESS_SYSTEM,
        max_tokens=300,
    )
    dt = time.time() - t0
    _log(snapshot, "thickness api dt={:.2f}s err={}".format(dt, err))

    if resp is None:
        return _blank_thickness_result("api_error: {}".format(err))

    _save_text(snapshot, "llm_qa_thickness_response.txt", resp.get("text") or "")
    parsed = _extract_json_from_text(resp.get("text") or "")
    if not isinstance(parsed, dict):
        return _blank_thickness_result("unparseable_response")

    v = str(parsed.get("verdict") or "").strip().lower()
    if v not in ("plausible", "suspicious", "no_opinion"):
        v = "no_opinion"
    out = {
        "verdict": v,
        "notes": str(parsed.get("notes") or "")[:400],
        "raw": resp.get("text"),
    }
    _save_json(snapshot, "llm_qa_thickness.json", out)
    return out
