"""Snapshot manager for sketch-to-CAD runs.

Designed to run under IronPython (pyRevit) and CPython tests.
"""

import datetime
import json
import os
import shutil
import traceback


def _ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def default_snapshot_roots():
    roots = []

    env_root = os.environ.get("CADGEN_SNAPSHOT_ROOT")
    if env_root:
        roots.append(env_root)

    roots.append("/home/cassu/dev/cadgeneration")
    roots.append(r"\\wsl.localhost\Ubuntu-24.04\home\cassu\dev\cadgeneration")

    home = os.path.expanduser("~")
    roots.append(os.path.join(home, "dev", "cadgeneration"))
    roots.append(os.path.join(home, "cadgeneration"))

    deduped = []
    seen = set()
    for r in roots:
        if not r:
            continue
        if r in seen:
            continue
        seen.add(r)
        deduped.append(r)
    return deduped


def pick_snapshot_root(preferred=None):
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(default_snapshot_roots())

    for root in candidates:
        try:
            _ensure_dir(root)
            return root
        except Exception:
            continue

    fallback = os.path.join(os.getcwd(), "cadgeneration")
    _ensure_dir(fallback)
    return fallback


def _run_id():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    micros = datetime.datetime.now().strftime("%f")
    return "{}_{}".format(ts, micros[-4:])


class SnapshotRun(object):
    def __init__(self, root=None, run_name=None):
        self.root = pick_snapshot_root(root)
        self.run_name = run_name or _run_id()
        self.run_dir = _ensure_dir(os.path.join(self.root, self.run_name))

    def path(self, name):
        return os.path.join(self.run_dir, name)

    def save_json(self, name, payload):
        path = self.path(name)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    def save_text(self, name, text, append=False):
        path = self.path(name)
        mode = "a" if append else "w"
        with open(path, mode) as f:
            f.write(text)
            if text and (not text.endswith("\n")):
                f.write("\n")
        return path

    def copy_file(self, src_path, target_name):
        if not src_path:
            return None
        if not os.path.isfile(src_path):
            return None
        target = self.path(target_name)
        shutil.copy2(src_path, target)
        return target

    def save_error(self, stage_name, exc):
        payload = {
            "stage": stage_name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        self.save_json("errors.json", payload)
        self.save_text("run_log.txt", "[ERROR] {}: {}".format(stage_name, exc), append=True)

    def log(self, message):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.save_text("run_log.txt", "[{}] {}".format(now, message), append=True)
