# Create From CAD V2

Converts CAD linework (DWG/DXF) into Revit model elements:
- perimeter walls
- internal walls
- doors/windows hosted on walls
- one floor slab (roof slab optional)

## Main files
- `script.py`: Revit-side creation pipeline
- `v2_cad_extract.py`: CAD extraction + config loading
- `v2_cad_classify.py`: primitive classification
- `v2_cad_recognition.py`: topology and opening/internal-wall recognition
- `cad_config.json`: tuning parameters

## Important config defaults
- `model_create_roof: false`
- `opening_symbol_reject_name_patterns: ["opening"]`
- `opening_symbol_require_visible_geometry: true`
- `opening_host_fallback_max_dist_cm: 180.0`
- `model_skip_low_confidence_openings: false`
- `internal_wall_perimeter_parallel_tol_cm: 10.0`
- `internal_wall_perimeter_parallel_angle_deg: 8.0`
- `internal_wall_endpoint_snap_cm: 6.0`
- `internal_wall_dangling_max_cm: 35.0`

## Snapshot outputs
Each run writes JSON snapshots under the configured snapshot root.

Key files:
- `03_topology.json`: recognized polygon/openings/internal walls + debug metrics
- `08_geometry_summary.json`: created element IDs + opening/internal-wall placement diagnostics

## Troubleshooting
If doors/windows are missing or wrong:
1. Check `08_geometry_summary.json` -> `opening_family_rejects`
2. Check `opening_host_fallback_count` and `opening_attempts`
3. If needed, adjust `opening_symbol_reject_name_patterns` and host fallback distance

If internal walls are noisy:
1. Increase `internal_wall_min_length_cm`
2. Increase `internal_wall_perimeter_parallel_tol_cm`
3. Reduce `internal_wall_dangling_max_cm`

