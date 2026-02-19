# codex.md

## Project
- Name: `mytools-pyrevit`
- Main command: `Create From CAD V2`
- Runtime: pyRevit (IronPython) + Revit API
- Dev repo: `/home/cassu/mytools-pyrevit`
- Live extension: `/mnt/c/Users/cassu/AppData/Roaming/pyRevit/extensions/MyTools.extension`

## Current Implementation Status (2026-02-19)
- Double-wall recognition: implemented.
- Perimeter wall creation: merged collinear runs before wall creation (continuous walls).
- Openings (door/window):
  - strict category placement (`door -> OST_Doors`, `window -> OST_Windows`)
  - projected to host wall axis
  - clamped away from wall ends
  - per-opening diagnostics recorded in snapshot summary
- Internal walls:
  - recognized as internal graph edges (inside polygon, non-boundary, min-length)
  - collinear merge + dedupe
  - created in model with duplicate-to-perimeter rejection

## Files That Matter Most
- `MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/script.py`
- `MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/v2_cad_recognition.py`
- `MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/v2_cad_extract.py`
- `MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/cad_config.json`
- `MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/tests/test_v2_cad_recognition.py`

## Snapshot Fields for Debug
- `03_topology.json`
  - `openings`
  - `internal_walls_cm`
  - `debug.solve_mode`
- `08_geometry_summary.json`
  - `opening_attempt_count`
  - `opening_placed_count`
  - `opening_failed_count`
  - `opening_skipped_count`
  - `opening_errors`
  - `opening_attempts`
  - `internal_wall_ids`
  - `internal_wall_rejected_count`

## Important Config Knobs
- `model_wall_merge_angle_deg`
- `model_wall_join_tol_cm`
- `model_wall_min_length_cm`
- `model_min_opening_confidence`
- `model_opening_end_clearance_cm`
- `model_place_synthetic_openings`
- `enable_synthetic_window_fallback`
- `internal_wall_min_length_cm`
- `internal_wall_perimeter_duplicate_tol_cm`

## Validation Commands
```bash
cd /home/cassu/mytools-pyrevit/MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/tests
python3 -m unittest -q
```

## Deploy to Live Extension
```bash
cp /home/cassu/mytools-pyrevit/MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/{script.py,v2_cad_recognition.py,v2_cad_extract.py,cad_config.json} \
  /mnt/c/Users/cassu/AppData/Roaming/pyRevit/extensions/MyTools.extension/MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/
```
