import os
import sys
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from overlay_layout import compute_overlay_and_note_layout


class OverlayLayoutTests(unittest.TestCase):
    def assert_le(self, a, b, msg=""):
        self.assertTrue(a <= b, msg or "{} <= {} failed".format(a, b))

    def assert_ge(self, a, b, msg=""):
        self.assertTrue(a >= b, msg or "{} >= {} failed".format(a, b))

    def check_layout(self, view_min, view_max, aspect):
        out = compute_overlay_and_note_layout(view_min, view_max, aspect)

        x0, y0 = view_min
        x1, y1 = view_max

        # Image fully in view pane
        self.assert_ge(out["image_left"], x0)
        self.assert_le(out["image_right"], x1)
        self.assert_ge(out["image_bottom"], y0)
        self.assert_le(out["image_top"], y1)

        # Note starts to the right of image
        self.assert_ge(out["note_origin"][0], out["image_right"])

        # Note width fits inside the view
        self.assert_le(out["note_origin"][0] + out["note_width"], x1)

    def test_landscape_image(self):
        self.check_layout((0.0, 0.0), (100.0, 60.0), aspect=1.6)

    def test_portrait_image(self):
        self.check_layout((0.0, 0.0), (100.0, 60.0), aspect=0.7)

    def test_square_image(self):
        self.check_layout((0.0, 0.0), (100.0, 60.0), aspect=1.0)

    def test_tall_view(self):
        self.check_layout((0.0, 0.0), (80.0, 140.0), aspect=1.3)

    def test_wide_view(self):
        self.check_layout((0.0, 0.0), (180.0, 80.0), aspect=1.3)

    def test_extreme_portrait_image(self):
        # Regression: image must still fit fully in pane (no clipping).
        self.check_layout((0.0, 0.0), (338.0, 251.0), aspect=0.62)

    def test_extreme_landscape_image(self):
        self.check_layout((0.0, 0.0), (338.0, 251.0), aspect=2.20)


if __name__ == "__main__":
    unittest.main()
