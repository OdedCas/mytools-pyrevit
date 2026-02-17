import os
import sys
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ocr_extract import run_easyocr


class OCRExtractTests(unittest.TestCase):
    def test_missing_image_path(self):
        out = run_easyocr(None)
        self.assertFalse(out["available"])
        self.assertTrue(len(out["errors"]) > 0)

    def test_timeout_path_is_handled(self):
        out = run_easyocr("/tmp/not_a_real_image.png", timeout_sec=0.001)
        self.assertIn("available", out)
        self.assertIn("errors", out)


if __name__ == "__main__":
    unittest.main()
