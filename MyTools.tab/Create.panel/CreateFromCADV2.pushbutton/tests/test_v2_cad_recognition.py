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
        # Centerline dimensions: midpoint between inner (370x290) and outer (430x350)
        self.assertAlmostEqual(float(meas["room_width_cm"]), 400.0, delta=5.0)
        self.assertAlmostEqual(float(meas["room_height_cm"]), 320.0, delta=5.0)
        self.assertEqual(int(out["debug"].get("wall_line_candidates_raw", 0)), 8)
        # After centerline merge+split, segment count may increase; pairs stay 4
        self.assertGreaterEqual(int(out["debug"].get("wall_line_candidates", 0)), 4)
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
        # Centerline dimensions: midpoint between inner (370x290) and outer (430x350)
        self.assertAlmostEqual(float(meas["room_width_cm"]), 400.0, delta=5.0)
        self.assertAlmostEqual(float(meas["room_height_cm"]), 320.0, delta=5.0)

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

    def test_openings_include_center_anchor_coordinates(self):
        walls = [
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 0, "x2": 400, "y2": 0},
            {"type": "line", "layer": "A-WALL", "x1": 400, "y1": 0, "x2": 400, "y2": 260},
            {"type": "line", "layer": "A-WALL", "x1": 400, "y1": 260, "x2": 0, "y2": 260},
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 260, "x2": 0, "y2": 0},
        ]
        door_arc = [{"type": "arc", "layer": "DOOR", "cx": 140.0, "cy": 0.0,
                     "r": 50.0, "sx": 90.0, "sy": 0.0, "ex": 190.0, "ey": 0.0}]
        window_line = [{"type": "line", "layer": "WINDOW", "x1": 250.0, "y1": 260.0, "x2": 340.0, "y2": 260.0}]
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
        openings = list(out.get("openings", []))
        self.assertGreaterEqual(len(openings), 2, "Expected at least one door and one window")
        for op in openings:
            self.assertIn("center_x_cm", op)
            self.assertIn("center_y_cm", op)
            self.assertTrue(isinstance(float(op["center_x_cm"]), float))
            self.assertTrue(isinstance(float(op["center_y_cm"]), float))

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

        cfg = default_config()
        cfg["enable_synthetic_window_fallback"] = True
        out = recognize_topology(classified, cfg)
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


    def test_l_shaped_double_wall_room(self):
        """L-shaped room with double-line walls should produce 6+ vertex polygon."""
        # Outer L-shape
        outer = [
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 0, "x2": 500, "y2": 0},
            {"type": "line", "layer": "A-WALL", "x1": 500, "y1": 0, "x2": 500, "y2": 200},
            {"type": "line", "layer": "A-WALL", "x1": 500, "y1": 200, "x2": 300, "y2": 200},
            {"type": "line", "layer": "A-WALL", "x1": 300, "y1": 200, "x2": 300, "y2": 400},
            {"type": "line", "layer": "A-WALL", "x1": 300, "y1": 400, "x2": 0, "y2": 400},
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 400, "x2": 0, "y2": 0},
        ]
        # Inner L-shape (20cm wall thickness)
        inner = [
            {"type": "line", "layer": "A-WALL", "x1": 20, "y1": 20, "x2": 480, "y2": 20},
            {"type": "line", "layer": "A-WALL", "x1": 480, "y1": 20, "x2": 480, "y2": 180},
            {"type": "line", "layer": "A-WALL", "x1": 480, "y1": 180, "x2": 280, "y2": 180},
            {"type": "line", "layer": "A-WALL", "x1": 280, "y1": 180, "x2": 280, "y2": 380},
            {"type": "line", "layer": "A-WALL", "x1": 280, "y1": 380, "x2": 20, "y2": 380},
            {"type": "line", "layer": "A-WALL", "x1": 20, "y1": 380, "x2": 20, "y2": 20},
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
        poly = out.get("room_polygon_cm", [])
        self.assertGreaterEqual(len(poly), 6, "L-shape should have 6+ vertices, got %d" % len(poly))
        self.assertGreaterEqual(int(out["debug"].get("paired_wall_line_count", 0)), 4)

    def test_multiple_windows_same_wall(self):
        """Three windows on the same wall should be detected as separate openings."""
        walls = [
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 0, "x2": 600, "y2": 0},
            {"type": "line", "layer": "A-WALL", "x1": 600, "y1": 0, "x2": 600, "y2": 300},
            {"type": "line", "layer": "A-WALL", "x1": 600, "y1": 300, "x2": 0, "y2": 300},
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 300, "x2": 0, "y2": 0},
        ]
        # Three 80cm windows on the top wall (y=300), each labeled
        windows = [
            {"type": "line", "layer": "WINDOW", "x1": 60, "y1": 300, "x2": 140, "y2": 300},
            {"type": "line", "layer": "WINDOW", "x1": 240, "y1": 300, "x2": 320, "y2": 300},
            {"type": "line", "layer": "WINDOW", "x1": 420, "y1": 300, "x2": 500, "y2": 300},
        ]

        classified = {
            "wall_lines": walls,
            "door_lines": [],
            "window_lines": windows,
            "door_arcs": [],
            "window_arcs": [],
            "unclassified_lines": [],
            "unclassified_arcs": [],
        }

        out = recognize_topology(classified, default_config())
        win_openings = [o for o in out.get("openings", []) if o.get("type") == "window"]
        self.assertGreaterEqual(len(win_openings), 3,
                                "Expected 3 separate windows, got %d" % len(win_openings))

    def test_rooms_key_present_in_output(self):
        """Output should contain 'rooms' list for multi-room support."""
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
        self.assertIn("rooms", out)
        self.assertEqual(len(out["rooms"]), 1)
        room = out["rooms"][0]
        self.assertIn("room_polygon_cm", room)
        self.assertIn("openings", room)
        self.assertIn("measurements_cm", room)
        # Backward compat: flat fields should still exist
        self.assertIn("room_polygon_cm", out)
        self.assertIn("openings", out)

    def test_room2_real_data_simplified(self):
        """Simplified version of room2.dwg: L-shaped walls with window frame noise.

        Uses the inner wall trace from the actual raw data which forms a closed loop.
        """
        # Inner wall loop from raw CAD (forms a closed L-shape)
        # Path: 664.3,3093.4 → 664.3,1795.6 → 1830.05,1795.6 → 1830.67,1765.6
        #       → 634.3,1765.6 → 634.3,3123.4 → 2304.8,3123.4 → 2304.8,2887.7
        #       → 2974.9,2887.7 → 2974.9,2690.4 (window break)
        # We'll use the simpler inner loop that actually closes:
        inner_loop = [
            {"type": "line", "layer": "0", "x1": 664.3, "y1": 3093.4, "x2": 2274.8, "y2": 3093.4},
            {"type": "line", "layer": "0", "x1": 2274.8, "y1": 3093.4, "x2": 2274.8, "y2": 2857.7},
            {"type": "line", "layer": "0", "x1": 2274.8, "y1": 2857.7, "x2": 2944.9, "y2": 2857.7},
            {"type": "line", "layer": "0", "x1": 2944.9, "y1": 2857.7, "x2": 2944.9, "y2": 2196.4},
            {"type": "line", "layer": "0", "x1": 2944.9, "y1": 2196.4, "x2": 2274.8, "y2": 2196.4},
            {"type": "line", "layer": "0", "x1": 2274.8, "y1": 2196.4, "x2": 2274.8, "y2": 1795.6},
            {"type": "line", "layer": "0", "x1": 2274.8, "y1": 1795.6, "x2": 664.3, "y2": 1795.6},
            {"type": "line", "layer": "0", "x1": 664.3, "y1": 1795.6, "x2": 664.3, "y2": 3093.4},
        ]
        # Outer wall loop (30cm offset)
        outer_loop = [
            {"type": "line", "layer": "0", "x1": 634.3, "y1": 3123.4, "x2": 2304.8, "y2": 3123.4},
            {"type": "line", "layer": "0", "x1": 2304.8, "y1": 3123.4, "x2": 2304.8, "y2": 2887.7},
            {"type": "line", "layer": "0", "x1": 2304.8, "y1": 2887.7, "x2": 2974.9, "y2": 2887.7},
            {"type": "line", "layer": "0", "x1": 2974.9, "y1": 2887.7, "x2": 2974.9, "y2": 2166.4},
            {"type": "line", "layer": "0", "x1": 2974.9, "y1": 2166.4, "x2": 2304.8, "y2": 2166.4},
            {"type": "line", "layer": "0", "x1": 2304.8, "y1": 2166.4, "x2": 2304.8, "y2": 1765.6},
            {"type": "line", "layer": "0", "x1": 2304.8, "y1": 1765.6, "x2": 634.3, "y2": 1765.6},
            {"type": "line", "layer": "0", "x1": 634.3, "y1": 1765.6, "x2": 634.3, "y2": 3123.4},
        ]
        # Window frame noise (small rectangles)
        window_frames = [
            {"type": "line", "layer": "0", "x1": 2944.9, "y1": 2389.8, "x2": 2956.0, "y2": 2389.8},
            {"type": "line", "layer": "0", "x1": 2956.0, "y1": 2389.8, "x2": 2956.0, "y2": 2394.6},
            {"type": "line", "layer": "0", "x1": 2956.0, "y1": 2394.6, "x2": 2944.9, "y2": 2394.6},
            {"type": "line", "layer": "0", "x1": 2944.9, "y1": 2394.6, "x2": 2944.9, "y2": 2389.8},
        ]
        # Door arc on the inner wall bottom edge (y=1795.6)
        door_arc = [{"type": "arc", "layer": "0", "cx": 1200.0, "cy": 1795.6,
                      "r": 50.0, "sx": 1150.0, "sy": 1795.6, "ex": 1250.0, "ey": 1795.6}]

        from v2_cad_classify import classify_entities
        classified = classify_entities(
            inner_loop + outer_loop + window_frames, door_arc, {}, cfg=default_config())

        out = recognize_topology(classified, default_config())
        poly = out.get("room_polygon_cm", [])
        area = out["debug"]["picked_area_cm2"]
        # Room should be L-shaped, area should be reasonable
        self.assertGreater(area, 100000, "Room area too small: %f" % area)
        self.assertGreaterEqual(len(poly), 6, "L-shape needs 6+ vertices, got %d" % len(poly))
        # Door should be detected
        types = [o.get("type") for o in out.get("openings", [])]
        self.assertIn("door", types, "Door should be detected")

    def test_door_arc_on_l_shaped_room(self):
        """Door arc should be detected on an L-shaped room."""
        lines = [
            {"type": "line", "layer": "0", "x1": 0, "y1": 0, "x2": 280, "y2": 0},
            {"type": "line", "layer": "0", "x1": 280, "y1": 0, "x2": 280, "y2": 80},
            {"type": "line", "layer": "0", "x1": 280, "y1": 80, "x2": 360, "y2": 80},
            {"type": "line", "layer": "0", "x1": 360, "y1": 80, "x2": 360, "y2": 220},
            {"type": "line", "layer": "0", "x1": 360, "y1": 220, "x2": 0, "y2": 220},
            {"type": "line", "layer": "0", "x1": 0, "y1": 220, "x2": 0, "y2": 0},
        ]
        door_arc = [{"type": "arc", "layer": "DOOR", "cx": 140.0, "cy": 0.0,
                      "r": 45.0, "sx": 95.0, "sy": 0.0, "ex": 185.0, "ey": 0.0}]

        classified = {
            "wall_lines": [],
            "door_lines": [],
            "window_lines": [],
            "door_arcs": door_arc,
            "window_arcs": [],
            "unclassified_lines": lines,
            "unclassified_arcs": [],
        }

        out = recognize_topology(classified, default_config())
        types = [o.get("type") for o in out.get("openings", [])]
        self.assertIn("door", types, "Door should be detected on L-shaped room")


    def test_centerline_snap_removes_zigzag_folds(self):
        """Zigzag polygon alternating inner/outer traces should snap to centerlines."""
        # Room 400x300, wall 30cm thick.
        # Zigzag: inner-south→outer-south(fold)→outer-west→inner-north(fold)→
        #         inner-east→outer-east(fold)→back to start.
        # Wall pairs for all 4 sides.
        outer = [
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 0, "x2": 400, "y2": 0},
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 0, "x2": 0, "y2": 300},
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 300, "x2": 400, "y2": 300},
            {"type": "line", "layer": "A-WALL", "x1": 400, "y1": 0, "x2": 400, "y2": 300},
        ]
        inner = [
            {"type": "line", "layer": "A-WALL", "x1": 30, "y1": 30, "x2": 370, "y2": 30},
            {"type": "line", "layer": "A-WALL", "x1": 30, "y1": 30, "x2": 30, "y2": 270},
            {"type": "line", "layer": "A-WALL", "x1": 30, "y1": 270, "x2": 370, "y2": 270},
            {"type": "line", "layer": "A-WALL", "x1": 370, "y1": 30, "x2": 370, "y2": 270},
        ]
        classified = {
            "wall_lines": outer + inner,
            "door_lines": [],
            "window_lines": [],
            "door_arcs": [],
            "window_arcs": [],
            "dimension_lines": [],
            "dimension_arcs": [],
            "unclassified_lines": [],
            "unclassified_arcs": [],
        }
        out = recognize_topology(classified, default_config())
        poly = out["room_polygon_cm"]
        n = len(poly)
        area = abs(sum(
            poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
            for i in range(n)
        )) / 2.0
        # Centerlines form ~370x270 rectangle = ~99900 cm²
        # Allow some tolerance for averaging
        self.assertGreater(area, 80000, "Area should be close to centerline rectangle")
        self.assertLess(area, 120000)
        self.assertLessEqual(n, 6, "Should be clean rectangle (4 vertices) not zigzag")


    def test_internal_walls_detected_for_partitioned_room(self):
        """Room with an internal partition wall should report it in internal_walls_cm.

        When a partition connects both top and bottom walls, the cycle finder
        picks one sub-room as the polygon. The partition becomes a boundary
        edge of that room. But the OTHER side's partition-to-wall edge becomes
        an internal wall (it's inside the picked polygon).

        For a T-shaped partition (connects to one wall only), the partition
        centerline is fully inside the outer polygon and detected as internal.
        """
        # Outer walls: 600x400 rectangle with double lines (30cm thick)
        outer = [
            {"type": "line", "layer": "0", "x1": 0, "y1": 0, "x2": 600, "y2": 0},
            {"type": "line", "layer": "0", "x1": 600, "y1": 0, "x2": 600, "y2": 400},
            {"type": "line", "layer": "0", "x1": 600, "y1": 400, "x2": 0, "y2": 400},
            {"type": "line", "layer": "0", "x1": 0, "y1": 400, "x2": 0, "y2": 0},
        ]
        inner = [
            {"type": "line", "layer": "0", "x1": 30, "y1": 30, "x2": 570, "y2": 30},
            {"type": "line", "layer": "0", "x1": 570, "y1": 30, "x2": 570, "y2": 370},
            {"type": "line", "layer": "0", "x1": 570, "y1": 370, "x2": 30, "y2": 370},
            {"type": "line", "layer": "0", "x1": 30, "y1": 370, "x2": 30, "y2": 30},
        ]
        # Internal partition: T-junction, vertical wall at x=300, from bottom
        # wall (y=30) to y=250 (doesn't reach top wall). Both traces.
        partition_outer = [
            {"type": "line", "layer": "0", "x1": 285, "y1": 30, "x2": 285, "y2": 250},
        ]
        partition_inner = [
            {"type": "line", "layer": "0", "x1": 315, "y1": 30, "x2": 315, "y2": 250},
        ]

        classified = {
            "wall_lines": outer + inner + partition_outer + partition_inner,
            "unclassified_lines": [],
            "door_arcs": [],
            "door_lines": [],
            "window_lines": [],
            "window_arcs": [],
            "dimension_lines": [],
        }
        result = recognize_topology(classified, default_config())

        internal = result.get("internal_walls_cm", [])
        self.assertGreaterEqual(len(internal), 1,
            "Should detect at least 1 internal partition wall")

        # The partition should be roughly vertical at x~300, length >100cm
        found_partition = False
        for w in internal:
            dx = abs(w[2] - w[0])
            dy = abs(w[3] - w[1])
            length = (dx * dx + dy * dy) ** 0.5
            if length > 100 and dx < 50:  # roughly vertical, > 1m long
                found_partition = True
                break
        self.assertTrue(found_partition,
            "Should find a vertical partition wall >1m long, got: {}".format(internal))

    def test_no_internal_walls_for_single_room(self):
        """Simple single room should have zero internal walls."""
        outer = [
            {"type": "line", "layer": "0", "x1": 0, "y1": 0, "x2": 400, "y2": 0},
            {"type": "line", "layer": "0", "x1": 400, "y1": 0, "x2": 400, "y2": 300},
            {"type": "line", "layer": "0", "x1": 400, "y1": 300, "x2": 0, "y2": 300},
            {"type": "line", "layer": "0", "x1": 0, "y1": 300, "x2": 0, "y2": 0},
        ]
        inner = [
            {"type": "line", "layer": "0", "x1": 30, "y1": 30, "x2": 370, "y2": 30},
            {"type": "line", "layer": "0", "x1": 370, "y1": 30, "x2": 370, "y2": 270},
            {"type": "line", "layer": "0", "x1": 370, "y1": 270, "x2": 30, "y2": 270},
            {"type": "line", "layer": "0", "x1": 30, "y1": 270, "x2": 30, "y2": 30},
        ]
        classified = {
            "wall_lines": outer + inner,
            "unclassified_lines": [],
            "door_arcs": [],
            "door_lines": [],
            "window_lines": [],
            "window_arcs": [],
            "dimension_lines": [],
        }
        result = recognize_topology(classified, default_config())
        internal = result.get("internal_walls_cm", [])
        self.assertEqual(len(internal), 0,
            "Single room should have no internal walls, got: {}".format(internal))


if __name__ == "__main__":
    unittest.main()
