# -*- coding: utf-8 -*-
"""Snapshot manager for CAD-to-model V2 runs."""

import datetime
import json
import os
import traceback


def _ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def _default_root():
    candidates = []
    env_root = os.environ.get("CADGEN_SNAPSHOT_ROOT")
    if env_root:
        candidates.append(env_root)

    # Prefer deterministic local Windows path when running inside Revit/pyRevit.
    if os.name == "nt":
        home = os.path.expanduser("~")
        candidates.append(os.path.join(home, "dev", "cadgeneration"))
        candidates.append(r"C:\Users\cassu\dev\cadgeneration")

    # WSL/Linux candidates.
    candidates.append("/home/cassu/dev/cadgeneration")
    candidates.append(r"\\wsl.localhost\Ubuntu-24.04\home\cassu\dev\cadgeneration")
    home = os.path.expanduser("~")
    candidates.append(os.path.join(home, "dev", "cadgeneration"))

    for c in candidates:
        if not c:
            continue
        try:
            _ensure_dir(c)
            return c
        except Exception:
            continue

    fallback = os.path.join(os.getcwd(), "cadgeneration")
    return _ensure_dir(fallback)


def _run_id():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ms = datetime.datetime.now().strftime("%f")
    return "{}_{}".format(ts, ms[-4:])


class SnapshotRun(object):
    def __init__(self, root=None, run_name=None):
        self.root = _default_root() if root is None else _ensure_dir(root)
        self.run_name = run_name or _run_id()
        self.run_dir = _ensure_dir(os.path.join(self.root, self.run_name))

    def path(self, name):
        return os.path.join(self.run_dir, name)

    def save_json(self, name, payload):
        p = self.path(name)
        with open(p, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return p

    def log(self, text):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        p = self.path("run_log.txt")
        with open(p, "a") as f:
            f.write("[{}] {}\n".format(now, text))

    def save_error(self, stage, ex):
        payload = {
            "stage": stage,
            "error": str(ex),
            "traceback": traceback.format_exc(),
        }
        self.save_json("errors.json", payload)
        self.log("ERROR {}: {}".format(stage, ex))
