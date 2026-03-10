# C2Rv6_C

This file records the current `C2Rv6_C` state, forked from `C2Rv6`.

## Scope

Same as C2Rv6: wall creation only from a selected DWG import using `A-WALL-EXT` / `A-WALL-INT` layers.

## Changes from C2Rv6

### 1. No second extend on interior centerlines

The first `_collapse_to_centerlines` (in V2 recognition) already runs extend-to-intersections + split-at-crossings internally. C2Rv6 ran a second `_extend_to_intersections` on interior centerlines during the cleanup phase. This second extend over-reached past T-junctions, extending step/L-shaped wall segments past their correct stop point to the next parallel line. The result was false rectangular boxes instead of correct stepped wall shapes.

**Fix**: Removed the second `_extend_to_intersections` for `int_center`. Exterior centerlines still get the second extend since they form a single perimeter without internal T-junctions.

### 2. Lower raw dedup threshold

Changed initial line dedup from `min_len_cm=20` to `raw_min_len_cm=5`. The 20cm threshold was filtering out short structural connectors (e.g. 15cm step segments) before wall-face pairing could process them. The 20cm threshold is still used for final Revit wall creation.

### 3. Exterior proximity protection

Added `_ext_proximity_set()` which identifies interior centerline indices whose endpoints are near exterior wall endpoints. These protected indices are passed to `_remove_tiny_through_segments` and `_collapse_small_attached_cycles` to prevent removal of structurally important segments that connect interior walls to the exterior boundary.

## Why the box artifact occurred

The stepped interior wall in the DWG has two parallel traces (inner/outer) forming an L-shape:

```
outer:  down x=2138, left y=753, down x=2061, right y=684 (15cm), up x=2076, right y=738, up x=2153
inner:  down x=2153, left y=907, ... up x=2184
```

After pairing, the step vertical centerline was correctly at x=2428, y=574-788. The second `_extend_to_intersections` extended it to y=900 (the top horizontal), creating a vertical wall through the room that formed a rectangular box with the existing top/right/step-horizontal centerlines.

## File location

`MyTools.tab/Create.panel/C2Rv6_C.pushbutton/c2rv6_c.md`
