import os
import sys
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cad_topology import derive_room_from_cad
from cad_extract import default_cad_config


class CADTopologyTests(unittest.TestCase):
    def test_rectangular_room_with_openings(self):
        # Two traces each side -> wall thickness 30, inner = [30..400] x [30..320]
        walls = [
            {"x1": 0, "y1": 0, "x2": 0, "y2": 350},
            {"x1": 30, "y1": 0, "x2": 30, "y2": 350},
            {"x1": 400, "y1": 0, "x2": 400, "y2": 350},
            {"x1": 430, "y1": 0, "x2": 430, "y2": 350},
            {"x1": 0, "y1": 0, "x2": 430, "y2": 0},
            {"x1": 0, "y1": 30, "x2": 430, "y2": 30},
            {"x1": 0, "y1": 320, "x2": 430, "y2": 320},
            {"x1": 0, "y1": 350, "x2": 430, "y2": 350},
        ]

        doors = [{"x1": 120, "y1": 30, "x2": 220, "y2": 30}]
        wins = [{"x1": 270, "y1": 320, "x2": 370, "y2": 320}]

        classified = {
            "wall_lines": [{"type": "line", "layer": "A-WALL", **w} for w in walls],
            "door_lines": [{"type": "line", "layer": "DOOR", **d} for d in doors],
            "window_lines": [{"type": "line", "layer": "WINDOW", **w} for w in wins],
            "door_arcs": [],
        }

        out = derive_room_from_cad(classified, default_cad_config())
        meas = out["measurements_cm"]
        self.assertAlmostEqual(meas["room_width_cm"], 370.0, places=3)
        self.assertAlmostEqual(meas["room_height_cm"], 290.0, places=3)
        self.assertAlmostEqual(meas["door_width_cm"], 100.0, places=3)
        self.assertAlmostEqual(meas["window_width_cm"], 100.0, places=3)
        self.assertIn("room_polygon_cm", out)
        self.assertTrue(len(out.get("wall_segments_cm", [])) >= 4)

    def test_unclassified_lines_fallback_for_walls(self):
        # Same geometry as standard room, but all CAD lines are unlabeled.
        unclassified = [
            {"type": "line", "layer": "0", "x1": 0, "y1": 0, "x2": 0, "y2": 350},
            {"type": "line", "layer": "0", "x1": 30, "y1": 0, "x2": 30, "y2": 350},
            {"type": "line", "layer": "0", "x1": 400, "y1": 0, "x2": 400, "y2": 350},
            {"type": "line", "layer": "0", "x1": 430, "y1": 0, "x2": 430, "y2": 350},
            {"type": "line", "layer": "0", "x1": 0, "y1": 0, "x2": 430, "y2": 0},
            {"type": "line", "layer": "0", "x1": 0, "y1": 30, "x2": 430, "y2": 30},
            {"type": "line", "layer": "0", "x1": 0, "y1": 320, "x2": 430, "y2": 320},
            {"type": "line", "layer": "0", "x1": 0, "y1": 350, "x2": 430, "y2": 350},
            # Opening-like segments that should not break wall inference.
            {"type": "line", "layer": "0", "x1": 120, "y1": 30, "x2": 220, "y2": 30},
            {"type": "line", "layer": "0", "x1": 270, "y1": 320, "x2": 370, "y2": 320},
        ]
        classified = {
            "wall_lines": [],
            "door_lines": [],
            "window_lines": [],
            "door_arcs": [],
            "unclassified_lines": unclassified,
        }
        out = derive_room_from_cad(classified, default_cad_config())
        meas = out["measurements_cm"]
        self.assertAlmostEqual(meas["room_width_cm"], 370.0, places=3)
        self.assertAlmostEqual(meas["room_height_cm"], 290.0, places=3)
        self.assertTrue(out["debug"]["fallback_wall_from_unclassified"])

    def test_polygon_mode_for_non_rectangular_room(self):
        # Diamond-like rotated room contour (single-line wall representation).
        walls = [
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 50, "x2": 80, "y2": 0},
            {"type": "line", "layer": "A-WALL", "x1": 80, "y1": 0, "x2": 160, "y2": 50},
            {"type": "line", "layer": "A-WALL", "x1": 160, "y1": 50, "x2": 80, "y2": 100},
            {"type": "line", "layer": "A-WALL", "x1": 80, "y1": 100, "x2": 0, "y2": 50},
        ]
        # Opening hints on two different edges.
        door = [{"type": "line", "layer": "DOOR", "x1": 24, "y1": 35, "x2": 40, "y2": 25}]
        win = [{"type": "line", "layer": "WINDOW", "x1": 118, "y1": 72, "x2": 138, "y2": 60}]

        classified = {
            "wall_lines": walls,
            "door_lines": door,
            "window_lines": win,
            "door_arcs": [],
            "window_arcs": [],
            "unclassified_lines": [],
        }
        cfg = default_cad_config()
        cfg["prefer_polygon_mode"] = True
        out = derive_room_from_cad(classified, cfg)
        self.assertEqual(out["debug"]["mode"], "polygon")
        self.assertEqual(len(out.get("room_polygon_cm", [])), 4)
        self.assertIn("measurements_cm", out)

    def test_complex_orthogonal_shape_auto_prefers_polygon(self):
        # Stepped orthogonal room (not a rectangle), with short step segments.
        walls = [
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 0, "x2": 200, "y2": 0},
            {"type": "line", "layer": "A-WALL", "x1": 200, "y1": 0, "x2": 200, "y2": 20},
            {"type": "line", "layer": "A-WALL", "x1": 200, "y1": 20, "x2": 260, "y2": 20},
            {"type": "line", "layer": "A-WALL", "x1": 260, "y1": 20, "x2": 260, "y2": 140},
            {"type": "line", "layer": "A-WALL", "x1": 260, "y1": 140, "x2": 0, "y2": 140},
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 140, "x2": 0, "y2": 0},
        ]
        classified = {
            "wall_lines": walls,
            "door_lines": [],
            "window_lines": [],
            "door_arcs": [],
            "window_arcs": [],
            "unclassified_lines": [],
        }
        cfg = default_cad_config()
        cfg["prefer_polygon_mode"] = False
        out = derive_room_from_cad(classified, cfg)
        self.assertEqual(out["debug"]["mode"], "polygon")
        self.assertTrue(out["debug"].get("auto_prefer_polygon"))


if __name__ == "__main__":
    unittest.main()
