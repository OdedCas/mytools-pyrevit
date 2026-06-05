# C2Rv6

`C2Rv6` is a DWG-to-Revit wall builder for the office CAD standard.

Current behavior:

- forces the user to select one DWG import in the active view
- processes only that selected DWG
- reads wall geometry only from `A-WALL-EXT` and `A-WALL-INT`
- ignores doors and windows for wall cutting
- builds walls from paired wall faces instead of from every raw CAD line

Current pipeline:

1. Select one DWG import instance in Revit.
2. Extract geometry from that selected import only.
3. Filter lines by `A-WALL-EXT` and `A-WALL-INT`.
4. Bridge wall-face gaps before pairing.
5. Collapse paired wall faces into centerlines only.
6. Run focused cleanup on duplicate lines, tiny false fragments, and small attached interior loops.
7. Create Revit walls from the resulting centerlines.

Known status:

- main perimeter and most walls are now much closer to the DWG
- current tuning specifically targets the false small box/pocket pattern that can appear on interior walls
- the center thick wall block is intentional and should remain

Important files:

- `script.py`: current `C2Rv6` implementation
- `bundle.yaml`: pyRevit button metadata
- `c2rv6.md`: current implementation notes and remaining work
