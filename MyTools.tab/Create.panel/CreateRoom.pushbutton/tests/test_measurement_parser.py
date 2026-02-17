import os
import sys
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from measurement_parser import extract_numeric_candidates, parse_ocr_tokens, merge_measurements_cm


class MeasurementParserTests(unittest.TestCase):
    def test_extract_candidates_from_blob(self):
        vals = extract_numeric_candidates("61110")
        self.assertIn(61.0, vals)
        self.assertIn(110.0, vals)

    def test_parse_tokens(self):
        tokens = [
            {"text": "90"},
            {"text": "100"},
            {"text": "180"},
            {"text": "210"},
            {"text": "370"},
            {"text": "290"},
            {"text": "30"},
            {"text": "105"},
        ]
        parsed = parse_ocr_tokens(tokens)
        self.assertTrue("parsed" in parsed)
        self.assertTrue(parsed["parsed"]["door_width_cm"] >= 60)

    def test_merge_prefers_ocr(self):
        pick = {"door_width_cm": 95.0, "room_width_cm": 371.0}
        ocr = {"door_width_cm": 100.0}
        merged = merge_measurements_cm(pick, ocr, prefer_ocr=True)
        self.assertEqual(merged["door_width_cm"], 100.0)
        self.assertEqual(merged["room_width_cm"], 371.0)


if __name__ == "__main__":
    unittest.main()
