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
| 3 | `spaces.py` | Apartment outline (big closing → outside flood), enclosed spaces, balcony excluded (marked by the user until windows are detected), walls clipped to where the apartment is beside them (railings drop out), exterior / interior split |
| 4 | `windows.py` | Openings in walls (the dark count across a wall drops at a window; or a gap between wall pieces on one line); window vs door by glazing lines inside the opening; **balcony found automatically** (smaller space beyond a window wall, rule 7); windows placed **from the dimension strings** (`samples/*.dims.json`), each string lined up by the windows found, with a numbers-vs-drawing comparison |
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

## Current result on the sample

Outline 113.6 m² (walls included), balcony 8.9 m² excluded, 16 exterior and
23 interior wall pieces. Left exterior wall 674 cm × 28 cm; `815` neighbour
wall 805 cm; `5N` wall 24 cm (written 20).

Windows: all 7 in the dimension strings placed; numbers vs drawing within
0-47 cm (table in the commit message). Candidates the strings don't
confirm (bathtub edges, shower glass, two on the left wall) are dropped.

## Known problems / next

1. Two candidates on the left wall (x=204, 108 cm and 76 cm, glazing 0.99 /
   0.88) face outside but have no dimension string - real bathroom windows,
   or bathtub lines? Ask Oded.
2. The 5N glass wall (505 cm) is listed in `glazed_walls` but not yet placed.
3. Rooms narrower than 1.6 m are not labelled as separate spaces (the big
   closing fills them). They are inside the outline, which is what the
   exterior/interior split needs; Revit makes the rooms from the walls.
4. Upper right facade only partly found; some short interior walls missing.
5. Reading the dimension strings with Claude (now typed into dims.json);
   then the Revit button.

## Run

```
.venv/bin/python s2rv9/lines.py s2rv9/samples/original_scan.jpeg out/lines.png
```
