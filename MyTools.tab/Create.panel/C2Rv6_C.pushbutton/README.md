# C2Rv6_C

DWG-to-Revit wall + opening builder forked from `C2Rv6`, with improved handling of stepped/L-shaped interior walls, per-segment wall thickness, automatic door/window placement, and door swing matching.

## What it does

- Forces the user to select one DWG import in the active view
- Reads wall geometry from `A-WALL-EXT` and `A-WALL-INT` layers
- Reads door/window geometry from `A-DOORS` and `A-WINDOWS` layers
- Builds Revit walls from paired wall-face centerlines with per-segment thickness
- Places door and window family instances into the created walls
- Matches door swing direction to the DWG drawing

## Key differences from C2Rv6

- No second extend-to-intersections on interior centerlines (fixes box artifacts on stepped walls)
- Lower raw dedup threshold (5cm vs 20cm) preserves short structural connectors
- Exterior proximity protection prevents cleanup from removing wall-to-wall connections
- Per-wall-segment thickness: each centerline gets its own thickness measured from raw DWG wall faces, so different rooms can have different wall thicknesses
- Automatic door/window detection and placement from DWG opening layers
- Door swing orientation matched from DWG arc direction (facing + hand flip)

## Pipeline

1. Select one DWG import instance in Revit
2. Extract geometry from that selected import only (symbol geometry path)
3. Filter lines by `A-WALL-EXT` and `A-WALL-INT` for walls
4. Cache raw CAD data for opening detection
5. Bridge wall-face gaps, collapse paired faces to centerlines
6. Cleanup: prune leaves, remove small components/fragments, merge collinear, suppress parallel duplicates, collapse small cycles
7. Measure per-centerline wall thickness from raw DWG face pairs
8. Create Revit walls grouped by thickness, each with the matching wall type
9. Detect openings: cluster `A-DOORS` lines (120cm merge) and `A-WINDOWS` lines (60cm merge)
10. For each opening marker, find the nearest host wall and place a family instance
    - Doors: width from swing arc length, exterior families for ext walls, custom/interior families for int walls, placed at floor level, swing matched to DWG
    - Windows: width from bbox, best width match, sill height 105cm

## Files

- `script.py` — main implementation
- `bundle.yaml` — pyRevit button metadata
- `c2rv6_c.md` — implementation notes and change history
- `README.md` — this file
