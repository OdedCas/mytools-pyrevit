# CAD V2 Fix Plan: Windows, Doors, and Internal Walls

Date: 2026-02-19
Scope: `Create From CAD V2` pipeline (`v2_cad_classify.py`, `v2_cad_recognition.py`, `script.py`)

## Problem Summary

Current behavior in Revit:
1. Perimeter walls are created, but windows/doors are often missing.
2. Walls appear "sliced" into segments instead of getting hosted windows/doors.
3. Internal partition walls are partially detected but not consistently clean.

Desired behavior:
1. Keep walls continuous.
2. Place hosted door/window families on walls.
3. Let Revit cut the wall through family hosting, not through wall segmentation logic.
4. Detect and build meaningful internal walls only.

## Root Causes

1. Wall creation is edge-driven from polygon vertices, so over-segmented polygons create many short wall pieces.
2. Opening placement currently catches failures and continues, but host/family mismatch still causes silent misses.
3. Door/window family fallback between categories can choose invalid symbols for the target opening type.
4. Internal-wall extraction can include noise or boundary-adjacent artifacts.

## Implementation Plan

1. Add deterministic diagnostics in snapshots
- Extend `08_geometry_summary.json` with:
  - attempted openings
  - successfully placed openings
  - failed openings with reason
  - internal walls created
- Keep current `opening_errors`, but include edge id, type, width, and placement point.

2. Normalize perimeter walls before creation
- In `script.py`, build a merged wall-edge list from `room_polygon_cm`:
  - merge contiguous collinear edges
  - remove micro-segments under threshold
- Create Revit perimeter walls from merged edges only.
- Keep mapping from original edge index to merged wall host for opening placement.

3. Make opening recognition stricter and wall-safe
- In `v2_cad_recognition.py`, keep opening candidates as metadata only:
  - no topology operation should split wall geometry for openings
- Reduce false positives from gap-based methods when confidence is low.
- Keep synthetic fallback window only when no reliable opening exists and config allows it.

4. Hardening for window/door placement
- In `script.py`, place openings only with matching category symbols:
  - doors -> `OST_Doors`
  - windows -> `OST_Windows`
- Validate symbol is host-based and activatable before placement.
- Project opening center to host wall curve and clamp away from wall ends by clearance.
- On failure, keep wall intact, record a structured error, do not rollback model transaction.

5. Internal wall recognition cleanup
- In `v2_cad_recognition.py`:
  - keep only edges inside polygon and not on boundary
  - enforce minimum length threshold
  - merge connected collinear segments
  - dedupe near-overlapping segments
- Output stable `internal_walls_cm` list for model build.

6. Internal wall model build
- In `script.py`, create internal walls from `internal_walls_cm` after perimeter wall creation.
- Reject duplicates near perimeter walls by distance threshold.
- Include created internal wall ids in summary output.

7. Tests and acceptance gates
- Unit tests:
  - opening recognition does not increase wall segment count
  - internal wall extraction for positive and negative cases
  - host-edge mapping survives edge merge
- Pipeline tests:
  - doors/windows hosted and cut correctly
  - no wall slicing where an opening exists
- Acceptance criteria:
  - perimeter walls remain continuous
  - openings are represented by families (not gaps)
  - internal walls detected only where expected

## Documentation Deliverables

1. Update `README.md` section for CAD V2:
- opening detection behavior
- hosted opening placement behavior
- internal wall output behavior
- snapshot troubleshooting fields

2. Add troubleshooting note:
- how to inspect `03_topology.json` and `08_geometry_summary.json`
- how to identify family/hosting failures from `opening_errors`

## Rollout Plan

1. Implement in repo and pass tests.
2. Copy runtime files to live pyRevit extension.
3. Reload pyRevit and validate on known failing DWGs.
4. If regression occurs, restore from backup and re-run with debug snapshots.
