# S2Rv9 — scanned plan to Revit walls (work in progress)

Goal: pick a scanned plan (image or PDF) → reference planes along the
outside faces → exterior walls → interior walls → windows placed from the
dimension strings. Built directly in Revit, no DXF, reusing C2Rv7_C's
wall/window functions. **Read `DRAWING_RULES.md` first** — the tracing
follows those rules.

Status: **tracing only.** No Revit code yet. Developed on the Itai-HP machine
(no Revit there); Revit testing happens on the other laptop.

## Pipeline so far

| Step | File | What it does |
|---|---|---|
| 1 | `lines.py` | Finds thin straight lines: ink darker than both sides (`ridge_mask`), plus solid dark stripes; joins broken pieces (`join_collinear`); keeps heavy-pen lines only (`wall_pen_lines`) |
| 2 | `walls.py` | Pairs parallel lines nearest-first into walls; a wide dark stripe is a wall face on its own; merges touching bands so cladding + structure + plaster become one wall |
| — | `pair_walls.py` | Earlier attempt using CreateFromCADV2's `_find_wall_pairs`. Kept for reference: its CAD-tuned scoring picks the wrong partner on scans (see `walls.py` docstring) |

Runs on numpy + pillow only (the Revit PC's Python has no OpenCV).

## Sample: `samples/original_scan.jpeg`

- Scale **0.78 px/cm**, confirmed by two dimensions Oded verified:
  `815` wall (636 px → 0.780) and living room `560` (437 px → 0.781).
- The `815` wall traces as **808 cm**; the `5N` wall as **20 cm** (as written).

## What was learned

- The earlier `s2rv8/samples/sample_plan.png` is **not** a faithful copy of
  this scan: numbers were changed and walls redrawn. Do not use it.
- In the scan, interior walls are two thin lines with light (sometimes dark)
  fill; the right facade is structure + cladding lines packed into one dark
  stripe. Both styles appear in the same drawing.
- **Pen weight separates walls from everything else**: wall lines 8–74
  (0 = black), dimension lines and furniture 107–164. Brightness *between*
  a wall's lines does not — many real walls are paper-white between.

## Known problems / next

1. Find the apartment outline first; drop anything outside it (balcony,
   site lines — balcony is not a wall).
2. Use the outline to check exterior walls are 20–50 cm (rule 3); `5N` wall
   currently over-merges to 47 cm.
3. Upper right facade only partly found.
4. Windows from dimension strings; then the Revit button.

## Run

```
.venv/bin/python s2rv9/lines.py s2rv9/samples/original_scan.jpeg out/lines.png
```
