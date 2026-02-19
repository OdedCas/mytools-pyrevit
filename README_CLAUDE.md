# README_CLAUDE.md

This file is a quick handoff for Claude (or any coding agent) to continue work safely.

## What Was Implemented

1. Wall generation stability
- Perimeter walls are now built from merged collinear runs, not raw polygon micro-segments.
- Goal: avoid "sliced" wall behavior from fragmented edges.

2. Door/window hosting behavior
- Placement is strict by category:
  - doors use door symbols
  - windows use window symbols
- Opening points are projected onto wall axis and clamped away from wall ends.
- No cross-category fallback during placement.
- Per-opening diagnostics are recorded instead of failing the entire model transaction.

3. Internal wall recognition/modeling
- Internal walls are extracted from internal graph edges (inside polygon, non-boundary).
- Short-noise filtering + collinear merge + dedupe.
- Internal walls are created after perimeter walls.
- Internal walls close to perimeter walls are rejected as duplicates.

4. Diagnostics
- `08_geometry_summary.json` now includes:
  - opening attempt/placed/failed/skipped counts
  - `opening_attempts` details
  - `opening_errors`
  - internal wall created/rejected counts

## Key Files

- `MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/script.py`
- `MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/v2_cad_recognition.py`
- `MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/v2_cad_extract.py`
- `MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/cad_config.json`
- `MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/tests/test_v2_cad_recognition.py`

## Important Config Defaults

- `model_min_opening_confidence`: 0.45
- `model_opening_end_clearance_cm`: 15.0
- `model_place_synthetic_openings`: false
- `enable_synthetic_window_fallback`: false
- `internal_wall_min_length_cm`: 30.0
- `internal_wall_perimeter_duplicate_tol_cm`: 12.0

## Validate

```bash
cd /home/cassu/mytools-pyrevit/MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/tests
python3 -m unittest -q
```

## Deploy Runtime Files to Live pyRevit

```bash
cp /home/cassu/mytools-pyrevit/MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/{script.py,v2_cad_recognition.py,v2_cad_extract.py,cad_config.json} \
  /mnt/c/Users/cassu/AppData/Roaming/pyRevit/extensions/MyTools.extension/MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/
```

## Snapshots to Inspect in Revit Runs

- `03_topology.json`
- `08_geometry_summary.json`

