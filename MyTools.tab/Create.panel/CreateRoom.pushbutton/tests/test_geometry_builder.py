import os
import sys
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from geometry_builder import build_room_cm


class GeometryBuilderTests(unittest.TestCase):
    def test_build_room(self):
        meas = {
            "room_width_cm": 370,
            "room_height_cm": 290,
            "wall_thickness_cm": 30,
            "door_width_cm": 100,
            "door_left_offset_cm": 90,
            "window_width_cm": 100,
            "window_right_offset_cm": 30,
        }
        out = build_room_cm(meas)

        self.assertEqual(out["inner"]["se"][0], 370.0)
        self.assertEqual(out["inner"]["nw"][1], 290.0)
        self.assertEqual(out["door_center_x_cm"], 140.0)
        self.assertEqual(out["window_center_x_cm"], 290.0)


if __name__ == "__main__":
    unittest.main()
