# C2Rv6_C

This file records the current `C2Rv6_C` state, forked from `C2Rv6`.

## Scope

DWG-to-Revit wall + opening builder from a selected DWG import using `A-WALL-EXT` / `A-WALL-INT` layers for walls and `A-DOORS` / `A-WINDOWS` layers for openings.

## Changes from C2Rv6

### 1. No second extend on interior centerlines

C2Rv6 ran a second `_extend_to_intersections` on interior centerlines during cleanup. This over-extended past T-junctions on stepped/L-shaped walls, creating false rectangular boxes.

**Fix**: Removed the second extend for `int_center`. Exterior centerlines still get it since they form a single perimeter without internal T-junctions.

### 2. Lower raw dedup threshold

Changed initial line dedup from `min_len_cm=20` to `raw_min_len_cm=5`. The 20cm threshold was filtering out short structural connectors (e.g. 15cm step segments) before wall-face pairing. The 20cm threshold is still used for final Revit wall creation.

### 3. Exterior proximity protection

Added `_ext_proximity_set()` to identify interior centerline indices near exterior wall endpoints. These protected indices are excluded from `_remove_tiny_through_segments` and `_collapse_small_attached_cycles` to preserve wall-to-wall connections.

### 4. Per-wall-segment thickness

Instead of applying a single global median thickness to all interior walls, each centerline now gets its own thickness measured from the raw DWG wall-face lines.

`_measure_local_thickness()` finds the two closest parallel raw wall-face lines on opposite sides of each centerline and measures their perpendicular distance. `_create_walls_per_thickness()` groups centerlines by measured thickness (rounded to nearest cm) and picks the correct Revit wall type per group. This way different rooms can have different wall thicknesses (e.g. 10cm and 15cm) as drawn in the DWG.

### 5. Opening detection and placement

Doors and windows are detected from raw CAD data (cached during extraction) using layer-based filtering (`A-DOORS`, `A-WINDOWS`) and union-find clustering. `C2Rv6_C` now keeps real DWG arcs from the selected import instead of flattening everything to lines.

**Door detection**:
- Door primitives are clustered with 120cm merge distance.
- If a real DWG door swing arc exists, width comes from the arc radius, which is the door leaf width.
- If no arc exists, width falls back to the longest swing line or jamb bbox heuristic.
- Center stays at the midpoint between jamb/frame lines when those are available.

**Window detection**: Window primitives are clustered with 60cm merge distance. Width = shorter bbox dimension.

**Placement pipeline**:
1. Find the nearest Revit wall (by curve projection) within 150cm
2. Classify wall as interior/exterior based on which ID list it belongs to
3. Pick family type: exterior doors get `EXTERIOR`-named families, interior doors prefer custom-loaded families (non-M_ prefix) over Revit defaults
4. If the chosen door family has a writable width parameter, reuse an existing exact-width type from that family or duplicate the selected type and set its width to the DWG width
5. Place with `NewFamilyInstance(point, symbol, wall, level, NonStructural)`
6. Windows: sill height 105cm via `INSTANCE_SILL_HEIGHT_PARAM`
7. Doors: placed at floor level (0cm)

### 6. Door swing matching

After placement, `_match_door_swing()` adjusts the Revit door orientation to match the DWG:
- If a real DWG arc exists, the arc center is used as the hinge, and the open endpoint is inferred relative to the host wall axis.
- The open endpoint determines which side of the wall the door swings into. Compared against Revit's `FacingOrientation` — flips `flipFacing()` if mismatched.
- The hinge location determines which end of the opening the hinge sits at. Compared against Revit's `HandOrientation` — flips `flipHand()` if mismatched.
- If no arc exists, the older longest-line swing heuristic is still used as a fallback.

## Why the box artifact occurred (historical)

The stepped interior wall in the DWG has two parallel traces forming an L-shape. After pairing, the step vertical centerline was correctly at y=574-788. The second `_extend_to_intersections` extended it to y=900 (the top horizontal), creating a vertical wall through the room.

## File location

`MyTools.tab/Create.panel/C2Rv6_C.pushbutton/c2rv6_c.md`
