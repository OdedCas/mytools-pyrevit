# C2Rv6_C

`C2Rv6_C` is a DWG-to-Revit wall builder forked from `C2Rv6`, with improved handling of stepped/L-shaped interior walls.

Current behavior:

- Forces the user to select one DWG import in the active view
- Processes only that selected DWG
- Reads wall geometry only from `A-WALL-EXT` and `A-WALL-INT`
- Ignores doors and windows for wall cutting
- Builds walls from paired wall faces instead of from every raw CAD line

Key differences from C2Rv6:

- Does not run a second extend-to-intersections on interior centerlines, preventing false box artifacts at stepped wall junctions
- Uses a lower dedup threshold (5cm vs 20cm) for raw wall lines to preserve short structural connectors
- Protects interior wall segments near exterior walls from aggressive cleanup removal

Pipeline:

1. Select one DWG import instance in Revit.
2. Extract geometry from that selected import only (symbol geometry path to avoid duplicates).
3. Filter lines by `A-WALL-EXT` and `A-WALL-INT`.
4. Bridge wall-face gaps before pairing.
5. Collapse paired wall faces into centerlines only.
6. Cleanup: prune leaves, remove small components/fragments, merge collinear, suppress parallel duplicates, collapse small attached cycles.
7. Create Revit walls from the resulting centerlines.

Important files:

- `script.py`: current implementation
- `bundle.yaml`: pyRevit button metadata
- `c2rv6_c.md`: implementation notes and change history
