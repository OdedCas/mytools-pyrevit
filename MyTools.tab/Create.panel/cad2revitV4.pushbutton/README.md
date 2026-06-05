# cad2revitV4

pyRevit button script that converts imported CAD linework (DWG `ImportInstance`) into Revit elements:

- walls (from paired parallel CAD lines)
- doors (from swing arcs and/or wall gaps)
- windows (from wall gaps)

## Runtime Requirement

This script must run inside Revit through pyRevit.

- `script.py` imports `Autodesk.Revit.DB`
- running with plain `python3` will fail (`ModuleNotFoundError: Autodesk`)

## How To Run

1. Open Revit and load pyRevit.
2. Select exactly one imported CAD object (`ImportInstance`) in the active view.
3. Run the `cad2revitV4` button.

The script prints progress and counts to the pyRevit output panel.

## What It Does (High Level)

1. Extracts CAD lines and arcs from the selected import
2. Finds parallel line pairs to infer wall thickness
3. Builds and cleans wall centerlines (merge/split/dedup)
4. Detects opening candidates from gaps and door swing arcs
5. Optionally stitches wall gaps around openings for better hosting
6. Creates Revit walls and places door/window family instances on hosts
7. Writes a DWG/script version note in the view (for deployed-script verification)

## Main Toggles (Top of `script.py`)

- `PREVIEW_CENTERLINES`
- `BUILD_WALLS`
- `PREVIEW_DOORS`
- `BUILD_DOORS`
- `PREVIEW_WINDOWS`
- `BUILD_WINDOWS`
- `BRIDGE_OPENING_GAPS`

## Important Assumptions

- A Door family symbol and a Window family symbol exist in the project
- The active view has a level (or at least one level exists in the model)
- CAD geometry is reasonably clean and uses parallel wall line pairs

## Troubleshooting

- `No module named Autodesk`: Run from pyRevit/Revit, not terminal Python
- `בחר ImportInstance אחד (CAD מיובא)`: Select exactly one imported CAD object
- `לא נמצאו קירות...`: CAD geometry/units/tolerances may not match expectations
- Door/window placement fails: confirm loaded family symbols and hostable walls

## Files

- `script.py` - active pyRevit script
- `script.py.bak_before_merge_20260222_0001` - backup snapshot before merge
