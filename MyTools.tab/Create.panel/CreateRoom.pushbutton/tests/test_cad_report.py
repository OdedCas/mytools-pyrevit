import os
import sys
import shutil
import tempfile
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from snapshot_manager import SnapshotRun
from cad_report import save_cad_snapshot_stages


class CADReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cadreport_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_snapshot_stage_files(self):
        run = SnapshotRun(root=self.tmp, run_name="cadrun")
        save_cad_snapshot_stages(
            run,
            {"lines": [], "arcs": [], "meta": {}},
            {"wall_lines": []},
            {"inner_sw_cm": [0, 0], "inner_se_cm": [1, 0], "inner_ne_cm": [1, 1], "inner_nw_cm": [0, 1], "measurements_cm": {}},
            {"room_width_cm": 370},
        )
        for name in (
            "10_cad_raw_entities.json",
            "11_cad_classified.json",
            "12_wall_pairs.json",
            "13_opening_assignment.json",
            "14_model_inputs.json",
        ):
            self.assertTrue(os.path.isfile(run.path(name)))


if __name__ == "__main__":
    unittest.main()
