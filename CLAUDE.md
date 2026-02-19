# CLAUDE.md - Project Instructions

## Project Overview
PyRevit extension that reads DWG/DXF CAD drawings and generates Revit models (walls, doors, windows, floors). The main tool is **CreateFromCADV2**.

## Key Architecture

### Runtime
- Runs inside Revit via pyRevit's IronPython 2.7 runtime
- NO external dependencies allowed (no numpy, shapely, etc.)
- All math must be pure Python + `math` module
- Must be compatible with both IronPython 2.7 and CPython 3.x (for tests)

### File Locations
- **WSL dev repo**: `/home/cassu/mytools-pyrevit/`
- **Windows pyRevit extension**: `/mnt/c/Users/cassu/AppData/Roaming/pyRevit/extensions/MyTools.extension/`
- **Debug snapshots**: `/mnt/c/Users/cassu/dev/cadgeneration/<timestamp>/`
- After editing, sync both locations:
  ```bash
  cp <wsl_file> /mnt/c/Users/cassu/AppData/Roaming/pyRevit/extensions/MyTools.extension/MyTools.tab/Create.panel/CreateFromCADV2.pushbutton/<file>
  ```

### V2 Pipeline
```
DWG → Extract lines/arcs (v2_cad_extract.py)
    → Classify by layer (v2_cad_classify.py)
    → Recognize topology (v2_cad_recognition.py)
        1. Merge wall + unclassified lines
        2. Detect double-wall pairs (parallel lines ~30cm apart)
        3. Collapse to centerlines (merge overlapping, extend to intersections, split at crossings)
        4. Find room cycle (graph-based: snap endpoints → bridge gaps → find shortest cycle)
        5. Post-process polygon (fold removal, collinear cleanup)
        6. Detect openings (doors from arcs, windows from line patterns/gaps)
    → Build Revit model (script.py)
```

### Strategy Cascade (recognize_topology)
The tool tries strategies in order until one finds a closed cycle:
1. **centerlines_only** — only paired wall centerlines (cleanest graph)
2. **inner_walls** — inner line from each pair + unpaired lines
3. **inner_walls_relaxed** — same but with larger gap bridging
4. **collapsed** — centerlines + unpaired lines
5. **raw_fallback** — all original lines
6. **raw_relaxed_bridge** — all lines with max gap bridging

## Testing
```bash
cd /home/cassu/mytools-pyrevit/MyTools.tab/Create.panel/CreateFromCADV2.pushbutton
python3 -m unittest discover -s tests -v
```

## Test Data
- **room2.dwg** — L-shaped single room, double-line walls, layer "dimension" for dims
  - Snapshot: `20260218_081404_2000/`
  - Needs `layer_map={'dimensions': ['dimension']}` for correct classification
- **room3.dwg** — Multi-room apartment, all lines on layer "0", no layer info
  - Snapshot: `20260218_082158_4000/`
  - Works with empty layer_map (geometric classification only)

## Important Conventions
- Wall thickness is typically 30cm in test drawings
- Coordinates are in centimeters (CAD units, not Revit feet)
- The polygon represents the room boundary at wall centerlines
- Revit walls are placed at centerline positions with thickness parameter
- Snap tolerance: 0.4cm (endpoint_snap_mm=4.0)
- Graph nodes are snapped to 0.4cm grid

## Known Limitations
- Single-room cycle detection — for multi-room apartments, finds the outer perimeter only
- No hatching/fill detection
- No text/dimension OCR from DWG (relies on layer names or geometric patterns)
- Window detection is heuristic (perpendicular line clusters near walls)
