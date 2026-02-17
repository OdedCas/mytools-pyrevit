import os
import sys
import unittest

THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cad_classify import classify_entities
from cad_extract import default_layer_map


class CADClassifyTests(unittest.TestCase):
    def test_layer_first_classification(self):
        lines = [
            {"type": "line", "x1": 0, "y1": 0, "x2": 100, "y2": 0, "layer": "A-WALL"},
            {"type": "line", "x1": 10, "y1": 0, "x2": 20, "y2": 0, "layer": "DOOR"},
            {"type": "line", "x1": 30, "y1": 0, "x2": 50, "y2": 0, "layer": "WINDOW"},
        ]
        arcs = [{"type": "arc", "cx": 10, "cy": 5, "r": 50, "layer": "X-UNKNOWN"}]
        out = classify_entities(lines, arcs, default_layer_map())
        self.assertEqual(len(out["wall_lines"]), 1)
        self.assertEqual(len(out["door_lines"]), 1)
        self.assertEqual(len(out["window_lines"]), 1)
        self.assertEqual(len(out["door_arcs"]), 1)  # fallback from unclassified arcs


if __name__ == "__main__":
    unittest.main()
