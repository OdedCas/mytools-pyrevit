# D_W_detection - Codex Notes

## Goal
Detect door/window locations from an inserted DWG and annotate:
- doors with `DD` text
- windows with `ww` text
- doors with Revit linear dimensions (door arch edge to nearest corner)

## Current behavior
1. User must pick the target DWG `ImportInstance`.
2. Layer-first detection from DWG layers:
- doors: `A-DOORS` variants
- windows: `A-WINDOWS` variants
3. Geometry is clustered to "one block = one opening marker".
4. Door dimension points are built from door-arc start points and nearest corners.
5. Door dimensions are created with Revit `NewDimension(...)`.
6. Summary reports detection counts and dimension creation stats.

## Main file
- `script.py`

## Notes
- If dimensions are created but not visible, check view template/annotation visibility.
- Script attempts to unhide `OST_Dimensions` in the active view before creation.
