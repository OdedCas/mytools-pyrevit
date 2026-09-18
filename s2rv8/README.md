# S2Rv8 — PDF / image floor plan to Revit walls (prototype)

Status: **tracing prototype only.** No Revit button yet.

## Plan

1. Trace — find wall bands in the plan (`trace.py`)
2. Calibrate — find a dimension string, ask the user to confirm it, scale to cm
3. Write DXF on the layers `C2Rv7_C` reads: `A-WALL-EXT`, `A-WALL-INT`,
   `A-WALL-CORE`, `A-DOORS`, `A-WINDOWS` (coordinates in cm)
4. Import into Revit, hand off to the `C2Rv7_C` pipeline — walls first,
   then doors and windows

`trace.py` runs in normal CPython 3, **not** inside pyRevit: IronPython 2.7
has no imaging library. This is a deliberate exception to the repo's
"no external dependencies" rule — the Revit button itself will stay pure
IronPython and call this as a separate process.

## Run it

```
python -m venv .venv
.venv/bin/pip install -r s2rv8/requirements.txt        # Windows: .venv\Scripts\pip
.venv/bin/python s2rv8/trace.py s2rv8/samples/sample_plan.png out/step
```

Writes `out/step_mask.png` and `out/step_segments.png` (red = horizontal
walls, blue = vertical, band width = measured thickness).

## How tracing works

- Walls are thick dark bands; dimension lines, text, door swings and
  furniture are thin. A morphological opening wider than any thin line
  (`min_wall_px=7`) leaves only walls.
- Long thin kernels split the result into horizontal and vertical bands;
  each band becomes a centerline plus thickness.
- Exterior walls in the sample are **sandwich walls** — two poche layers
  with a white cavity (~16 px). `merge_sandwich` joins parallel bands with a
  gap up to 24 px into one wall.
- Doorways come out as gaps in the walls, which is what C2Rv7_C expects.

## Sample result (`samples/sample_plan.png`, 945 x 1664 px)

~0.77 px/cm (bottom dimensions total 886 cm over ~680 px). 62 walls found.

Known problems:
- Top-left exterior wall near the first `4N` window still traced as two
  thin lines instead of one thick wall.
- Blob near door `D4` — likely a shaft/column; unconfirmed.
- Balcony edge on the right broken into short pieces; unclear whether it
  should be a wall at all.
- Short stubs where walls meet window frames.
