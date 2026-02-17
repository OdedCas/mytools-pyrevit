import math
import os
import sys
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from layout_solver import LayoutSolverError, calibration_scale, solve_layout


def rotate_point(x, y, angle_deg):
    a = math.radians(angle_deg)
    c = math.cos(a)
    s = math.sin(a)
    return (x * c - y * s, x * s + y * c, 0.0)


class LayoutSolverTests(unittest.TestCase):
    def assert_close(self, actual, expected, tol=1e-6):
        self.assertTrue(abs(actual - expected) <= tol, "{} != {}".format(actual, expected))

    def test_axis_aligned_room(self):
        out = solve_layout(
            (0, 0, 0),
            (370, 0, 0),
            (370, 290, 0),
            (0, 290, 0),
            (90, 0, 0),
            (190, 0, 0),
            (240, 290, 0),
            (340, 290, 0),
            scale_factor=1.0,
            min_opening=40.0,
        )

        self.assert_close(out["inner_x_len"], 370.0)
        self.assert_close(out["inner_y_len"], 290.0)
        self.assert_close(out["door_width"], 100.0)
        self.assert_close(out["window_width"], 100.0)
        self.assert_close(out["door_center_s"], 140.0)
        self.assert_close(out["window_center_s"], 290.0)

    def test_rotated_room(self):
        angle = 32.0
        sw = rotate_point(0, 0, angle)
        se = rotate_point(370, 0, angle)
        ne = rotate_point(370, 290, angle)
        nw = rotate_point(0, 290, angle)

        d_l = rotate_point(90, 0, angle)
        d_r = rotate_point(190, 0, angle)
        w_l = rotate_point(240, 290, angle)
        w_r = rotate_point(340, 290, angle)

        out = solve_layout(sw, se, ne, nw, d_l, d_r, w_l, w_r, scale_factor=1.0, min_opening=40.0)

        self.assert_close(out["inner_x_len"], 370.0, tol=1e-4)
        self.assert_close(out["inner_y_len"], 290.0, tol=1e-4)
        self.assert_close(out["door_width"], 100.0, tol=1e-4)
        self.assert_close(out["window_width"], 100.0, tol=1e-4)

    def test_scaled_from_calibration(self):
        scale = calibration_scale((10, 10, 0), (60, 10, 0), 100.0)

        out = solve_layout(
            (0, 0, 0),
            (185, 0, 0),
            (185, 145, 0),
            (0, 145, 0),
            (45, 0, 0),
            (95, 0, 0),
            (120, 145, 0),
            (170, 145, 0),
            scale_factor=scale,
            min_opening=40.0,
        )

        self.assert_close(scale, 2.0)
        self.assert_close(out["inner_x_len"], 370.0)
        self.assert_close(out["inner_y_len"], 290.0)
        self.assert_close(out["door_width"], 100.0)
        self.assert_close(out["window_width"], 100.0)

    def test_min_opening_enforced(self):
        out = solve_layout(
            (0, 0, 0),
            (370, 0, 0),
            (370, 290, 0),
            (0, 290, 0),
            (100, 0, 0),
            (100, 0, 0),
            (250, 290, 0),
            (250, 290, 0),
            scale_factor=1.0,
            min_opening=40.0,
        )
        self.assert_close(out["door_width"], 40.0)
        self.assert_close(out["window_width"], 40.0)

    def test_invalid_calibration_raises(self):
        with self.assertRaises(LayoutSolverError):
            calibration_scale((10, 10, 0), (10, 10, 0), 100.0)


if __name__ == "__main__":
    unittest.main()
