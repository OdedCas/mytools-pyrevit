# r2cv6

This file records the current `C2Rv6` state for the layer-first wall workflow.

## Scope

`C2Rv6` is currently limited to wall creation.

- input: one selected DWG import in the active floor plan
- wall source layers: `A-WALL-EXT`, `A-WALL-INT`
- openings: ignored for wall cutting
- output: native Revit walls only

## Current rules

1. User must select the DWG import before conversion.
2. Only the selected DWG is processed.
3. Geometry is extracted from one geometry path to avoid duplicated translated copies.
4. Exterior wall traces are used to keep one main model footprint.
5. Raw wall-face gaps are bridged before wall-face pairing.
6. Centerlines are generated from paired faces only.
7. Interior-only cleanup removes tiny false fragments after pairing.

## Why this version exists

Earlier versions had these recurring failures:

- duplicated side-by-side models from double extraction
- walls cut by window and door geometry
- tiny wall strips created from opening jambs
- duplicated parallel wall runs that should collapse to one wall

The current version narrows the logic to the office CAD standard instead of trying to infer all CAD content.

## Remaining issue

For the current test DWG, one small false interior fragment still survives in the right-side room.

The next debugging step, if needed, should be:

1. export/log final `int_center` line segments before Revit wall creation
2. identify the exact surviving segment in coordinates
3. remove that case with a geometric rule instead of broad tolerance changes

## File location

`MyTools.tab/Create.panel/C2Rv6.pushbutton/r2cv6.md`
