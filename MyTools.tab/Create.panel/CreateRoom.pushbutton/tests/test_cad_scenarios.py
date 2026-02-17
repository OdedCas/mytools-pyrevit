import os
import shutil
import sys
import tempfile
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from geometry_builder import build_room_cm
from measurement_parser import parse_ocr_tokens, merge_measurements_cm
from rectification import build_rectification, apply_homography, distance
from snapshot_manager import SnapshotRun
from overlay_layout import compute_overlay_and_note_layout


class CADScenarioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cadscenario_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scenario_standard_room(self):
        meas = {
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
        out = build_room_cm(meas)
        self.assertEqual(out["door_center_x_cm"], 140.0)
        self.assertEqual(out["window_center_x_cm"], 290.0)

    def test_scenario_noisy_ocr_merge(self):
        tokens = [
            {"text": "61110"},
            {"text": "251"},
            {"text": "338"},
            {"text": "90"},
            {"text": "100"},
            {"text": "180"},
        ]
        parsed = parse_ocr_tokens(tokens)["parsed"]
        picked = {
            "room_width_cm": 370.0,
            "room_height_cm": 290.0,
            "door_width_cm": 100.0,
            "window_width_cm": 100.0,
            "door_left_offset_cm": 90.0,
            "window_right_offset_cm": 30.0,
        }
        merged = merge_measurements_cm(picked, parsed, prefer_ocr=False)
        self.assertEqual(merged["room_width_cm"], 370.0)
        self.assertEqual(merged["door_left_offset_cm"], 90.0)

    def test_scenario_rectification_and_snapshot(self):
        nw = (10.0, 100.0)
        ne = (210.0, 90.0)
        se = (220.0, 10.0)
        sw = (20.0, 0.0)
        rect = build_rectification(nw, ne, se, sw)

        # Calibration segment that should map to 100cm after scaling.
        a = (60.0, 50.0)
        b = (110.0, 50.0)
        ar = apply_homography(rect["homography"], a)
        br = apply_homography(rect["homography"], b)
        self.assertTrue(distance(ar, br) > 0)

        run = SnapshotRun(root=self.tmp, run_name="scenario_run")
        run.save_json("00_input_meta.json", {"units": "cm", "known_calibration_cm": 100.0})
        run.save_json("02_rectified_transform.json", rect)
        run.save_json("03_ocr_boxes.json", {"engine": "easyocr", "available": False, "tokens": []})
        run.save_json("04_measurements_raw.json", {"pick_measurements_cm": {"room_width_cm": 370}, "ocr_parsed": {}})
        run.save_json("05_measurements_confirmed.json", {"room_width_cm": 370, "room_height_cm": 290})
        run.save_json("08_geometry_summary.json", {"geometry": {"wall_ids": [1, 2, 3, 4]}, "dimensions": {"ok": True}})

        for name in (
            "00_input_meta.json",
            "02_rectified_transform.json",
            "03_ocr_boxes.json",
            "04_measurements_raw.json",
            "05_measurements_confirmed.json",
            "08_geometry_summary.json",
        ):
            self.assertTrue(os.path.isfile(run.path(name)), msg="missing snapshot {}".format(name))

    def test_scenario_overlay_layout_contract(self):
        out = compute_overlay_and_note_layout((0.0, 0.0), (338.0, 251.0), image_aspect=0.62)
        self.assertGreaterEqual(out["image_left"], 0.0)
        self.assertLessEqual(out["image_right"], 338.0)
        self.assertGreaterEqual(out["image_bottom"], 0.0)
        self.assertLessEqual(out["image_top"], 251.0)
        self.assertGreaterEqual(out["note_origin"][0], out["image_right"])


if __name__ == "__main__":
    unittest.main()
