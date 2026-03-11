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

Doors and windows are detected from raw CAD data (cached during extraction) using layer-based filtering (`A-DOORS`, `A-WINDOWS`) and union-find clustering.

**Door detection**: Lines on `A-DOORS` layer are clustered with 120cm merge distance (door blocks have two jamb frames ~100cm apart + a swing arc line). Width = swing arc length (the longest line in the cluster, typically the door leaf width). Center = midpoint of all frame/jamb line endpoints (lines <= 50cm), excluding the swing arc.

**Window detection**: Lines on `A-WINDOWS` layer are clustered with 60cm merge distance. Width = shorter bbox dimension.

**Placement pipeline**:
1. Find the nearest Revit wall (by curve projection) within 150cm
2. Classify wall as interior/exterior based on which ID list it belongs to
3. Pick family type: exterior doors get `EXTERIOR`-named families, interior doors prefer custom-loaded families (non-M_ prefix) over Revit defaults. Width matching selects closest available size.
4. Place with `NewFamilyInstance(point, symbol, wall, level, NonStructural)`
5. Windows: sill height 105cm via `INSTANCE_SILL_HEIGHT_PARAM`
6. Doors: placed at floor level (0cm)

### 6. Door swing matching

After placement, `_match_door_swing()` reads the DWG swing arc direction and adjusts the Revit door orientation to match:
- The swing arc's open endpoint determines which side of the wall the door swings into. Compared against Revit's `FacingOrientation` — flips `flipFacing()` if mismatched.
- The swing arc's hinge endpoint (closest to door center) determines which end of the opening the hinge sits at. Compared against Revit's `HandOrientation` — flips `flipHand()` if mismatched.

## Why the box artifact occurred (historical)

The stepped interior wall in the DWG has two parallel traces forming an L-shape. After pairing, the step vertical centerline was correctly at y=574-788. The second `_extend_to_intersections` extended it to y=900 (the top horizontal), creating a vertical wall through the room.

## File location

`MyTools.tab/Create.panel/C2Rv6_C.pushbutton/c2rv6_c.md`
