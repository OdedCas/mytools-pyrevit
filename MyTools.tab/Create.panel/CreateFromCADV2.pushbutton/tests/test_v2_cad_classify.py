import os
import sys
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from v2_cad_classify import classify_entities


class GeometricClassifyTests(unittest.TestCase):
    def test_arc_reclassified_as_door_when_no_layer(self):
        """Unclassified arc with radius in door-swing range => door_arcs."""
        lines = [
            {"type": "line", "layer": "0", "x1": 0, "y1": 0, "x2": 300, "y2": 0},
        ]
        arcs = [
            {"type": "arc", "layer": "0", "cx": 100, "cy": 0, "r": 45.0,
             "sx": 55, "sy": 0, "ex": 145, "ey": 0},
        ]
        out = classify_entities(lines, arcs, {}, cfg={})
        self.assertTrue(len(out["door_arcs"]) >= 1)
        self.assertEqual(out["door_arcs"][0]["r"], 45.0)

    def test_tiny_lines_ignored(self):
        """Lines < 3cm should be ignored as tick marks."""
        lines = [
            {"type": "line", "layer": "0", "x1": 0, "y1": 0, "x2": 300, "y2": 0},
            {"type": "line", "layer": "0", "x1": 50, "y1": 0, "x2": 50, "y2": 2},
            {"type": "line", "layer": "0", "x1": 100, "y1": 0, "x2": 100, "y2": 1.5},
        ]
        out = classify_entities(lines, [], {}, cfg={})
        # The two tiny lines should be ignored
        self.assertEqual(out["ignored"], 2)
        # The 300cm line should remain
        self.assertEqual(len(out["unclassified_lines"]), 1)

    def test_window_pattern_perpendicular_lines(self):
        """Clusters of short perpendicular lines near a long line => window_lines."""
        # Long wall line
        lines = [
            {"type": "line", "layer": "0", "x1": 0, "y1": 0, "x2": 400, "y2": 0},
        ]
        # Short perpendicular lines (window panes) near the wall
        window_panes = [
            {"type": "line", "layer": "0", "x1": 100, "y1": -5, "x2": 100, "y2": 15},
            {"type": "line", "layer": "0", "x1": 110, "y1": -5, "x2": 110, "y2": 15},
            {"type": "line", "layer": "0", "x1": 120, "y1": -5, "x2": 120, "y2": 15},
        ]
        all_lines = lines + window_panes
        out = classify_entities(all_lines, [], {}, cfg={})
        self.assertTrue(len(out["window_lines"]) >= 2,
                        "Expected perpendicular short lines to be classified as windows, got %d" % len(out["window_lines"]))

    def test_layer_classification_still_works(self):
        """Layer-based classification should still work when layers are named."""
        lines = [
            {"type": "line", "layer": "A-WALL", "x1": 0, "y1": 0, "x2": 300, "y2": 0},
            {"type": "line", "layer": "WINDOW", "x1": 100, "y1": 0, "x2": 200, "y2": 0},
        ]
        arcs = [
            {"type": "arc", "layer": "DOOR", "cx": 50, "cy": 0, "r": 40,
             "sx": 10, "sy": 0, "ex": 90, "ey": 0},
        ]
        layer_map = {
            "walls": ["^A-WALL$"],
            "windows": ["^WINDOW$"],
            "doors": ["^DOOR$"],
        }
        out = classify_entities(lines, arcs, layer_map, cfg={})
        self.assertEqual(len(out["wall_lines"]), 1)
        self.assertEqual(len(out["window_lines"]), 1)
        self.assertEqual(len(out["door_arcs"]), 1)

    def test_cfg_defaults_when_none(self):
        """Should work even without cfg parameter (backward compat)."""
        lines = [
            {"type": "line", "layer": "0", "x1": 0, "y1": 0, "x2": 300, "y2": 0},
        ]
        # Call without cfg (old API)
        out = classify_entities(lines, [], {})
        self.assertEqual(len(out["unclassified_lines"]) + len(out["wall_lines"]), 1)


if __name__ == "__main__":
    unittest.main()
