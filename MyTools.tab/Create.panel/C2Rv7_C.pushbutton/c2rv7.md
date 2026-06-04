# C2Rv7

This file records the current `C2Rv7` state for the layer-first wall workflow.

## Scope

`C2Rv7` is focused on layer-first wall creation, with optional apartment-area
creation after the walls are generated.

- input: one selected DWG import in the active floor plan
- wall source layers: `A-WALL-EXT`, `A-WALL-INT`, `A-WALL-CORE`
- openings: ignored for wall cutting
- output: native Revit walls, optional doors/windows/floors/stairs/dimensions,
  and optional apartment `Area` objects

## Current rules

1. User must select the DWG import before conversion.
2. Only the selected DWG is processed.
3. Geometry is extracted from one geometry path to avoid duplicated translated copies.
4. Exterior wall traces are used to keep one main model footprint.
5. Raw wall-face gaps are bridged before wall-face pairing.
6. Centerlines are generated from paired faces only.
7. Interior cleanup removes tiny fragments and collapses small attached loop patterns.
8. Apartment areas are created from large enclosed cells after wall detection:
   - exterior wall is counted fully inside the apartment area by using the
     exterior wall outer face
   - apartment/apartment walls are counted to the wall centerline
   - core walls are counted to the wall centerline
   - apartment-separating walls must measure at least 20 cm
   - minimum apartment area is 35 sqm

## Why this version exists

Earlier versions had these recurring failures:

- duplicated side-by-side models from double extraction
- walls cut by window and door geometry
- tiny wall strips created from opening jambs
- duplicated parallel wall runs that should collapse to one wall
- small attached interior pockets that should simplify into the main wall chain

The current version narrows the logic to the office CAD standard instead of trying to infer all CAD content.

## Current limitation

If a DWG contains a complex interior pocket that is drawn as a valid wall loop and is larger than the cleanup thresholds, it may still survive.

Apartment-area detection assumes that apartment boundaries are already drawn as
wall boundaries. It does not create internal rooms inside an apartment; it
creates one area per apartment-sized enclosed space.

The next debugging step, if needed, should be:

1. inspect the final `int_center` segments for that component
2. compare them to the raw `A-WALL-INT` source lines
3. tighten the attached-loop collapse rule without changing the perimeter logic

## File location

`MyTools.tab/Create.panel/C2Rv7.pushbutton/c2rv7.md`
