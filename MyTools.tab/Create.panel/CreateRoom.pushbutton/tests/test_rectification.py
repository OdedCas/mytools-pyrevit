import math
import os
import sys
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rectification import build_rectification, apply_homography, distance


class RectificationTests(unittest.TestCase):
    def assert_close(self, a, b, tol=1e-5):
        self.assertTrue(abs(a - b) <= tol, "{} != {}".format(a, b))

    def test_build_rectification_maps_corners(self):
        # Slightly skewed quadrilateral.
        nw = (10.0, 90.0)
        ne = (220.0, 80.0)
        se = (230.0, 10.0)
        sw = (20.0, 0.0)

        out = build_rectification(nw, ne, se, sw)
        H = out["homography"]

        nw_m = apply_homography(H, nw)
        ne_m = apply_homography(H, ne)
        se_m = apply_homography(H, se)
        sw_m = apply_homography(H, sw)

        self.assert_close(nw_m[0], 0.0)
        self.assert_close(nw_m[1], out["rect_height_units"])

        self.assert_close(ne_m[0], out["rect_width_units"])
        self.assert_close(ne_m[1], out["rect_height_units"])

        self.assert_close(se_m[0], out["rect_width_units"])
        self.assert_close(se_m[1], 0.0)

        self.assert_close(sw_m[0], 0.0)
        self.assert_close(sw_m[1], 0.0)

    def test_distance(self):
        self.assertAlmostEqual(distance((0, 0), (3, 4)), 5.0)


if __name__ == "__main__":
    unittest.main()
