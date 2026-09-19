# S2Rv9 — how to read an Israeli architectural floor plan

Rules given by Oded (architect) on 2026-09-18. The tracer and the Revit
builder must follow these. When code and this file disagree, this file wins.

## Walls

1. **Two parallel lines = a wall.**
2. **A door or window sits inside a wall.** The wall exists on both sides
   of the opening and usually continues to the next wall. An opening is never
   a reason to end a wall.
3. **A wall with a window is an exterior wall**, usually **20–50 cm** thick.
4. **Many parallel lines on the outside face** of an exterior wall are the
   cladding (stone or aluminium) — not separate walls.
5. **Extra parallel lines on the inside face** of an exterior wall are the
   inner finish (plaster or gypsum board) — not separate walls.
6. **Diagonal hatch inside a wall** = block (CMU) or concrete, depending on
   colour or pattern. Hatch that is only scan noise is ignored.

7. **A balcony is not a wall.** Its edge (railing / parapet) is drawn as a
   thin double line, but it is left out of the Revit model. The wall between
   the apartment and the balcony (e.g. the `5N` window wall) *is* a wall.
   **The balcony is whatever lies outside the window wall.**
8. **A thin wall on the outside edge can be a neighbour (party) wall**, e.g.
   the `815` wall (~10 cm). It is a real wall and part of the apartment's
   outline, even though it has no windows and is thinner than rule 3's
   20–50 cm.

## Wall material by thickness

| Thickness | Material |
|---|---|
| 7–15 cm | probably block (CMU) |
| 20 cm | concrete or block — depends on location |
| Mamad (safe room) walls | **always concrete** |

## Symbols

- **Small square or circle inside a wall** = a pipe (drain or sewer).
- **Circle or polygon next to a door or window** = its tag / number.
  Ignore for now.
- **`W`** in a bathroom label = WC (e.g. `W 63` = WC, 63 cm).

## Dimensions

- **Never assume what a dimension measures — follow its tick marks.**
  `195` in the middle bedroom is between the mamad's pipes, not a room size.
- **Some dimensions have no tick marks.** Kitchen `420` runs to the kitchen
  island; its ends can be worked out from neighbouring dimensions (the
  corridor `140`).
- **Dimension rows can be stacked.** A second row totals several segments of
  the row above: bottom facade `352` = `144` + `208`.
- **Don't trust a doubtful reading — measure it.** Once enough dimensions are
  confirmed, the scale is known and anything unclear (`15`, `385`) is measured
  from the drawing instead of read.
- **Window notation:** `100 UK60` = 100 cm wide, sill 60 cm. `4N`, `3N`, `5N`
  are window type tags.
- **Door notation:** `D8` + width, e.g. `D8 80` = 80 cm, `D3 70` = 70 cm.

## Source images

- **Always work from the original scan or PDF.** A cleaned-up copy of the
  sample (`s2rv8/samples/sample_plan.png`) had changed numbers and redrawn
  walls: `208` dropped from the bottom string, `124` became `128`, `182`
  became `166`, `550` became `596`, window `3N` became `4N`.
- In the real scan a wall is **two thin black lines with light grey between**
  (~10 px), the same stroke weight as dimension lines and text. Tracing by
  "remove everything thin" deletes the walls; walls must be found by pairing
  parallel lines.
- The sample scan is rotated 180° relative to its text (numbers read upside
  down). Geometry is unaffected.
