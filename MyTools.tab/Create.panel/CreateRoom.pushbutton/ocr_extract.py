"""OCR extraction wrapper with EasyOCR backend when available.

This module is resilient: if EasyOCR is unavailable it returns a structured
failure payload instead of raising.
"""

import json
import subprocess
import time


_OCR_SNIPPET = r'''
import json
import sys
try:
    import easyocr
except Exception as e:
    print(json.dumps({"ok": False, "error": "easyocr import failed: {}".format(e), "tokens": []}))
    raise SystemExit(0)

image_path = sys.argv[1]
try:
    reader = easyocr.Reader(['en'], gpu=False)
    out = reader.readtext(image_path)
    tokens = []
    for item in out:
        bbox, text, conf = item
        norm_bbox = []
        for pt in bbox:
            norm_bbox.append([float(pt[0]), float(pt[1])])
        tokens.append({"text": text, "confidence": float(conf), "bbox": norm_bbox})
    print(json.dumps({"ok": True, "tokens": tokens}))
except Exception as e:
    print(json.dumps({"ok": False, "error": "easyocr runtime failed: {}".format(e), "tokens": []}))
'''


def _candidate_commands():
    # Ordered from most explicit to most common.
    return [
        ["python3"],
        ["python"],
        ["py", "-3"],
    ]


def _to_text(value):
    if value is None:
        return ""
    try:
        # Python 3 bytes
        return value.decode("utf-8", "ignore")
    except Exception:
        return str(value)


def run_easyocr(image_path, timeout_sec=90):
    result = {
        "engine": "easyocr",
        "available": False,
        "used_command": None,
        "tokens": [],
        "errors": [],
    }

    if not image_path:
        result["errors"].append("No image_path provided")
        return result

    for cmd_prefix in _candidate_commands():
        cmd = list(cmd_prefix) + ["-c", _OCR_SNIPPET, image_path]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            start = time.time()
            timed_out = False
            while proc.poll() is None:
                if (time.time() - start) > float(timeout_sec):
                    timed_out = True
                    break
                time.sleep(0.1)

            if timed_out:
                try:
                    proc.terminate()
                except Exception:
                    pass
                result["errors"].append("{}: timeout after {}s".format(" ".join(cmd_prefix), timeout_sec))
                continue

            stdout, stderr = proc.communicate()

            out_text = _to_text(stdout).strip()
            err_text = _to_text(stderr).strip()

            if not out_text:
                result["errors"].append("{}: empty output {}".format(" ".join(cmd_prefix), err_text))
                continue

            payload = json.loads(out_text)
            result["used_command"] = " ".join(cmd_prefix)
            if payload.get("ok"):
                result["available"] = True
                result["tokens"] = payload.get("tokens", [])
                return result

            err = payload.get("error", "unknown error")
            result["errors"].append("{}: {}".format(" ".join(cmd_prefix), err))
        except Exception as ex:
            result["errors"].append("{}: {}".format(" ".join(cmd_prefix), ex))

    return result
