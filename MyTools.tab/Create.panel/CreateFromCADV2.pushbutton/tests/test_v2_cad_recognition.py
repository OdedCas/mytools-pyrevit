import os
import sys
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from v2_cad_extract import default_config
from v2_cad_recognition import recognize_topology


class CADRecognitionV2Tests(unittest.TestCase):
    def test_nested_double_line_walls_pick_inner_loop(self):
        # Outer + inner wall traces (typical DWG double-line wall drawing).
        outer = [
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 0, "x2": 430, "y2": 0},
            {"type": "line", "layer": "A-WALL", "x1": 430, "y1": 0, "x2": 430, "y2": 350},
            {"type": "line", "layer": "A-WALL", "x1": 430, "y1": 350, "x2": 0, "y2": 350},
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 350, "x2": 0, "y2": 0},
        ]
        inner = [
            {"type": "line", "layer": "A-WALL", "x1": 30, "y1": 30, "x2": 400, "y2": 30},
            {"type": "line", "layer": "A-WALL", "x1": 400, "y1": 30, "x2": 400, "y2": 320},
            {"type": "line", "layer": "A-WALL", "x1": 400, "y1": 320, "x2": 30, "y2": 320},
            {"type": "line", "layer": "A-WALL", "x1": 30, "y1": 320, "x2": 30, "y2": 30},
        ]

        classified = {
            "wall_lines": outer + inner,
            "door_lines": [],
            "window_lines": [],
            "door_arcs": [],
            "window_arcs": [],
            "unclassified_lines": [],
            "unclassified_arcs": [],
        }

        out = recognize_topology(classified, default_config())
        meas = out["measurements_cm"]
        self.assertAlmostEqual(float(meas["room_width_cm"]), 370.0, places=3)
        self.assertAlmostEqual(float(meas["room_height_cm"]), 290.0, places=3)
        self.assertEqual(int(out["debug"].get("wall_line_candidates_raw", 0)), 8)
        self.assertEqual(int(out["debug"].get("wall_line_candidates", 0)), 4)
        self.assertEqual(int(out["debug"].get("paired_wall_line_count", 0)), 4)
        self.assertAlmostEqual(float(out["debug"].get("estimated_wall_thickness_cm", 0.0)), 30.0, places=3)

    def test_nested_double_line_prefers_inner_even_with_outer_opening_mark(self):
        outer = [
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 0, "x2": 430, "y2": 0},
            {"type": "line", "layer": "A-WALL", "x1": 430, "y1": 0, "x2": 430, "y2": 350},
            {"type": "line", "layer": "A-WALL", "x1": 430, "y1": 350, "x2": 0, "y2": 350},
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 350, "x2": 0, "y2": 0},
        ]
        inner = [
            {"type": "line", "layer": "A-WALL", "x1": 30, "y1": 30, "x2": 400, "y2": 30},
            {"type": "line", "layer": "A-WALL", "x1": 400, "y1": 30, "x2": 400, "y2": 320},
            {"type": "line", "layer": "A-WALL", "x1": 400, "y1": 320, "x2": 30, "y2": 320},
            {"type": "line", "layer": "A-WALL", "x1": 30, "y1": 320, "x2": 30, "y2": 30},
        ]
        # Opening marker on outer boundary only.
        outer_opening = [{"type": "line", "layer": "WINDOW", "x1": 280, "y1": 350, "x2": 380, "y2": 350}]

        classified = {
            "wall_lines": outer + inner,
            "door_lines": [],
            "window_lines": outer_opening,
            "door_arcs": [],
            "window_arcs": [],
            "unclassified_lines": [],
            "unclassified_arcs": [],
        }

        out = recognize_topology(classified, default_config())
        meas = out["measurements_cm"]
        self.assertAlmostEqual(float(meas["room_width_cm"]), 370.0, places=3)
        self.assertAlmostEqual(float(meas["room_height_cm"]), 290.0, places=3)

    def test_irregular_polygon_from_unlabeled_lines(self):
        # L-shape room contour (single line wall traces).
        lines = [
            {"type": "line", "layer": "0", "x1": 0, "y1": 0, "x2": 280, "y2": 0},
            {"type": "line", "layer": "0", "x1": 280, "y1": 0, "x2": 280, "y2": 80},
            {"type": "line", "layer": "0", "x1": 280, "y1": 80, "x2": 360, "y2": 80},
            {"type": "line", "layer": "0", "x1": 360, "y1": 80, "x2": 360, "y2": 220},
            {"type": "line", "layer": "0", "x1": 360, "y1": 220, "x2": 0, "y2": 220},
            {"type": "line", "layer": "0", "x1": 0, "y1": 220, "x2": 0, "y2": 0},
        ]

        classified = {
            "wall_lines": [],
            "door_lines": [],
            "window_lines": [],
            "door_arcs": [],
            "window_arcs": [],
            "unclassified_lines": lines,
            "unclassified_arcs": [],
        }

        out = recognize_topology(classified, default_config())
        self.assertEqual(out["debug"]["mode"], "polygon_v2")
        self.assertGreaterEqual(len(out.get("room_polygon_cm", [])), 6)
        self.assertEqual(int(out["debug"].get("paired_wall_line_count", 0)), 0)

    def test_gap_bridge_closes_loop(self):
        # Rectangle with a gap on top edge (window/door-like opening in source linework).
        lines = [
            {"type": "line", "layer": "WALL", "x1": 0, "y1": 0, "x2": 300, "y2": 0},
            {"type": "line", "layer": "WALL", "x1": 300, "y1": 0, "x2": 300, "y2": 220},
            {"type": "line", "layer": "WALL", "x1": 300, "y1": 220, "x2": 190, "y2": 220},
            {"type": "line", "layer": "WALL", "x1": 90, "y1": 220, "x2": 0, "y2": 220},
            {"type": "line", "layer": "WALL", "x1": 0, "y1": 220, "x2": 0, "y2": 0},
        ]

        classified = {
            "wall_lines": lines,
            "door_lines": [],
            "window_lines": [],
            "door_arcs": [],
            "window_arcs": [],
            "unclassified_lines": [],
            "unclassified_arcs": [],
        }

        out = recognize_topology(classified, default_config())
        self.assertGreaterEqual(out["debug"].get("bridged_count", 0), 1)
        self.assertGreaterEqual(len(out.get("room_polygon_cm", [])), 4)

    def test_detects_door_and_window_candidates(self):
        walls = [
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 0, "x2": 370, "y2": 0},
            {"type": "line", "layer": "A-WALL", "x1": 370, "y1": 0, "x2": 370, "y2": 290},
            {"type": "line", "layer": "A-WALL", "x1": 370, "y1": 290, "x2": 0, "y2": 290},
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 290, "x2": 0, "y2": 0},
        ]
        door_arc = [{"type": "arc", "layer": "DOOR", "cx": 140.0, "cy": 0.0, "r": 50.0, "sx": 90.0, "sy": 0.0, "ex": 190.0, "ey": 0.0}]
        window_line = [{"type": "line", "layer": "WINDOW", "x1": 240.0, "y1": 290.0, "x2": 340.0, "y2": 290.0}]

        classified = {
            "wall_lines": walls,
            "door_lines": [],
            "window_lines": window_line,
            "door_arcs": door_arc,
            "window_arcs": [],
            "unclassified_lines": [],
            "unclassified_arcs": [],
        }

        out = recognize_topology(classified, default_config())
        types = [o.get("type") for o in out.get("openings", [])]
        self.assertIn("door", types)
        self.assertIn("window", types)

    def test_fallback_creates_default_100cm_window(self):
        walls = [
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 0, "x2": 300, "y2": 0},
            {"type": "line", "layer": "A-WALL", "x1": 300, "y1": 0, "x2": 300, "y2": 250},
            {"type": "line", "layer": "A-WALL", "x1": 300, "y1": 250, "x2": 0, "y2": 250},
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 250, "x2": 0, "y2": 0},
        ]
        classified = {
            "wall_lines": walls,
            "door_lines": [],
            "window_lines": [],
            "door_arcs": [],
            "window_arcs": [],
            "unclassified_lines": [],
            "unclassified_arcs": [],
        }

        out = recognize_topology(classified, default_config())
        windows = [o for o in out.get("openings", []) if o.get("type") == "window"]
        self.assertTrue(len(windows) >= 1)
        self.assertAlmostEqual(float(windows[0].get("width_cm", 0.0)), 100.0, places=3)

    def test_dimension_lines_adjust_room_and_opening_sizes(self):
        walls = [
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 0, "x2": 300, "y2": 0},
            {"type": "line", "layer": "A-WALL", "x1": 300, "y1": 0, "x2": 300, "y2": 200},
            {"type": "line", "layer": "A-WALL", "x1": 300, "y1": 200, "x2": 0, "y2": 200},
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 200, "x2": 0, "y2": 0},
        ]

        door_line = [{"type": "line", "layer": "DOOR", "x1": 100, "y1": 0, "x2": 180, "y2": 0}]
        window_line = [{"type": "line", "layer": "WINDOW", "x1": 120, "y1": 200, "x2": 220, "y2": 200}]

        # Long dims: guide room span; short dims near openings: guide opening widths.
        dims = [
            {"type": "line", "layer": "DIM", "x1": 0, "y1": 250, "x2": 450, "y2": 250},
            {"type": "line", "layer": "DIM", "x1": 340, "y1": 0, "x2": 340, "y2": 260},
            {"type": "line", "layer": "DIM", "x1": 90, "y1": -40, "x2": 230, "y2": -40},
            {"type": "line", "layer": "DIM", "x1": 90, "y1": 260, "x2": 270, "y2": 260},
        ]

        classified = {
            "wall_lines": walls,
            "door_lines": door_line,
            "window_lines": window_line,
            "door_arcs": [],
            "window_arcs": [],
            "dimension_lines": dims,
            "dimension_arcs": [],
            "unclassified_lines": [],
            "unclassified_arcs": [],
        }

        out = recognize_topology(classified, default_config())
        meas = out["measurements_cm"]

        self.assertAlmostEqual(float(meas["room_width_cm"]), 450.0, places=3)
        self.assertAlmostEqual(float(meas["room_height_cm"]), 260.0, places=3)
        self.assertAlmostEqual(float(meas["door_width_cm"]), 140.0, places=3)
        self.assertAlmostEqual(float(meas["window_width_cm"]), 180.0, places=3)

        dbg = out.get("debug", {})
        self.assertGreaterEqual(int(dbg.get("dimension_span_hint_used_count", 0)), 2)
        self.assertGreaterEqual(int(dbg.get("dimension_opening_hint_used_count", 0)), 2)


if __name__ == "__main__":
    unittest.main()
