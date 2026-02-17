import json
import os
import shutil
import sys
import tempfile
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from snapshot_manager import SnapshotRun


class SnapshotManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="snaptest_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_folder_and_json(self):
        run = SnapshotRun(root=self.tmp, run_name="run_a")
        self.assertTrue(os.path.isdir(run.run_dir))

        p = run.save_json("a.json", {"x": 1})
        self.assertTrue(os.path.isfile(p))
        with open(p, "r") as f:
            data = json.load(f)
        self.assertEqual(data["x"], 1)

    def test_copy_file(self):
        src = os.path.join(self.tmp, "src.txt")
        with open(src, "w") as f:
            f.write("hello")

        run = SnapshotRun(root=self.tmp, run_name="run_b")
        dst = run.copy_file(src, "copied.txt")
        self.assertTrue(os.path.isfile(dst))

    def test_log_and_error(self):
        run = SnapshotRun(root=self.tmp, run_name="run_c")
        run.log("hello")
        self.assertTrue(os.path.isfile(run.path("run_log.txt")))

        try:
            raise RuntimeError("boom")
        except Exception as ex:
            run.save_error("test", ex)

        self.assertTrue(os.path.isfile(run.path("errors.json")))


if __name__ == "__main__":
    unittest.main()
