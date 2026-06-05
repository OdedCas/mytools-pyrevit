# cad2revitV4 - Codex Handoff Notes

## Goal

Maintain and debug the pyRevit `cad2revitV4` button that converts imported CAD wall/opening linework into Revit walls, doors, and windows.

## Runtime Constraint (Important)

- `script.py` is a pyRevit/Revit script, not a standalone Python script.
- It depends on Revit API modules (`Autodesk.Revit.DB`) and `__revit__`.
- Terminal execution with `python3 script.py` is expected to fail outside Revit.

## Current Working Files

- Source/edit file:
  - `mytools-pyrevit/MyTools.tab/Create.panel/cad2revitV4.pushbutton/script.py`
- Backup:
  - `mytools-pyrevit/MyTools.tab/Create.panel/cad2revitV4.pushbutton/script.py.bak_before_merge_20260222_0001`
- Installed pyRevit copy (Windows profile):
  - `/mnt/c/Users/cassu/AppData/Roaming/pyRevit/extensions/MyTools.extension/MyTools.tab/Create.panel/cad2revitV4.pushbutton/script.py`

## Verified Sync Status (Feb 23, 2026)

The source `script.py` and installed pyRevit `script.py` were verified identical (same SHA-256) during this session.

## Script Behavior Summary

- Requires exactly one selected `ImportInstance`
- Extracts CAD lines/arcs
- Infers wall centerlines from parallel pairs + thickness clustering
- Merges/splits/deduplicates centerlines
- Detects openings from:
  - arc-based door swings
  - gap-based door/window candidates
- Optionally bridges wall gaps around openings before wall creation
- Creates walls by inferred thickness
- Hosts doors/windows on nearest created walls
- Adds a DWG build/version note using deployed file mtime/hash

## Key Tuning / Feature Flags (Top of `script.py`)

- Build/preview:
  - `PREVIEW_CENTERLINES`
  - `BUILD_WALLS`
  - `PREVIEW_DOORS`
  - `BUILD_DOORS`
  - `PREVIEW_WINDOWS`
  - `BUILD_WINDOWS`
- Opening/wall detection tolerances:
  - `MIN_LEN_FT`
  - `ANGLE_TOL_DEG`
  - `THICK_*`
  - `GAP_*`
  - `DOOR_ARC_*`
  - `BRIDGE_OPENING_GAPS` + `WALL_GAP_*`

## Known Practical Risks

- Uses first available Door/Window `FamilySymbol` in project (not type-by-width matching)
- CAD quality/noise strongly affects wall pairing and opening detection
- If no walls are created, hosting doors/windows will be skipped
- Active view/level context influences placement elevation

## Edit Workflow (Recommended)

1. Edit the source file in `mytools-pyrevit/.../cad2revitV4.pushbutton/script.py`
2. Sync/copy to installed pyRevit extension if needed
3. Reload pyRevit (or restart Revit) if the button does not pick up changes
4. Test on a single selected imported CAD instance
5. Review pyRevit output counts (`Walls created`, `Doors placed`, `Windows placed`)

## Quick Debug Hints

- If terminal run fails with `Autodesk` import error: expected, test in Revit
- If no walls found: inspect CAD units, duplicate lines, and minimum length threshold
- If doors/windows miss hosts: inspect wall creation success and host distance tolerances
- Use preview toggles to visualize centerlines and opening points before building
