# CreateFromCADV2 - Codex Handoff Notes

## Goal
Stabilize DWG -> Revit generation for:
- internal wall recognition
- door/window hosting + family selection
- top slab output (remove duplicate roof slab by default)

## What Was Implemented

### 1) Single slab default
File: `script.py`
- Geometry creation now always creates one floor slab.
- Roof slab creation is optional via config:
  - `model_create_roof` (default `false`)

### 2) Opening family selection hardened
File: `script.py`
- `get_symbol_by_width(...)` now:
  - ranks by width proximity
  - rejects family/type names containing configured patterns (default: `opening`)
  - can require visible bounding box geometry
  - logs rejected candidates for diagnostics
- New config keys:
  - `opening_symbol_reject_name_patterns`
  - `opening_symbol_require_visible_geometry`

### 3) Opening host fallback improved
File: `script.py`
- If `host_edge` mapping fails, opening can fallback to nearest perimeter wall using opening center point.
- Fallback cap distance controlled by:
  - `opening_host_fallback_max_dist_cm`
- Summary output includes:
  - `opening_host_fallback_count`

### 4) Opening center anchors added to topology
File: `v2_cad_recognition.py`
- Each opening now includes:
  - `center_x_cm`
  - `center_y_cm`
- These are computed from host edge + start/end offsets and used by model placement.

### 5) Internal wall filtering strengthened
File: `v2_cad_recognition.py`
- `_find_internal_walls(...)` now rejects:
  - boundary edges
  - outside edges
  - short noise
  - edges parallel and too close to perimeter
  - dangling stubs based on node degree + perimeter proximity
- Still merges collinear segments and dedupes.
- New config keys:
  - `internal_wall_endpoint_snap_cm`
  - `internal_wall_dangling_max_cm`
  - `internal_wall_perimeter_parallel_tol_cm`
  - `internal_wall_perimeter_parallel_angle_deg`
  - `internal_wall_source_exclude_layer_tokens`
  - `internal_wall_include_unpaired_singles`
  - `internal_wall_unpaired_single_min_len_cm`

### 5b) Internal wall source graph improved
File: `v2_cad_recognition.py`
- Internal walls are no longer detected only from perimeter solve graph.
- Source now combines:
  - paired centerlines
  - unpaired-pair centerlines
  - optional long unpaired singles inside polygon
- Dimension/annotation layer tokens are excluded from internal wall source.
- Added debug fields:
  - `internal_wall_source_mode`
  - `internal_wall_source_line_count`
  - `internal_wall_source_unpaired_pair_count`
  - `internal_wall_source_unpaired_single_count`
  - `internal_wall_graph_node_count`
  - `internal_wall_graph_segment_count`

### 6) Internal wall rejection tuning in model stage
File: `script.py`
- Internal walls near and parallel to perimeter are rejected before creation.
- Summary includes:
  - `internal_wall_reject_breakdown`

### 6b) Tiny perimeter jog collapse in model stage
File: `script.py`
- Wall run builder now skips very short polygon jog segments and bridges the
  next collinear segment across them.
- This prevents 5-20cm recognition artifacts from creating visible wall kinks.
- New config keys:
  - `model_wall_jog_skip_cm`
  - `model_wall_jog_bridge_join_tol_cm`

### 7) Low-confidence opening policy switch
File: `script.py`
- Added flag:
  - `model_skip_low_confidence_openings` (default `false`)
- Default behavior now places low-confidence openings (with logging) instead of auto-skip.

## Output JSON Changes

### `03_topology.json`
- `openings[*].center_x_cm`
- `openings[*].center_y_cm`
- `debug.internal_wall_candidate_count`
- `debug.internal_wall_rejected_boundary_count`
- `debug.internal_wall_rejected_outside_count`
- `debug.internal_wall_rejected_short_count`
- `debug.internal_wall_rejected_parallel_perimeter_count`
- `debug.internal_wall_rejected_dangling_count`

### `08_geometry_summary.json`
- `geometry.opening_host_fallback_count`
- `geometry.opening_family_choices`
- `geometry.opening_family_rejects`
- `geometry.internal_wall_reject_breakdown`

## Validation
- Unit tests (`tests/test_v2_cad_recognition.py`) pass.
- Added test to assert opening center anchor fields are present.

## Known Runtime Constraint
- `script.py` behavior must be verified in Revit/pyRevit runtime because unit tests do not execute Revit API object creation.
