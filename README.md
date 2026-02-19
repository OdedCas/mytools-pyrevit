# MyTools - PyRevit Extension

A PyRevit extension that reads DWG/DXF CAD drawings and automatically generates a Revit model from the linework — creating walls, doors, windows, and floors.

## Tools

### Create From CAD V2 (`CreateFromCADV2.pushbutton`)
The current main tool. Reads a DWG/DXF file, recognizes architectural elements by layer, and builds a Revit model.

**Key features over V1:**
- Double-wall pair detection
- Wall gap bridging (connects discontinuous wall segments)
- Dimension-hint assisted measurements (reads dimension lines from the CAD)
- Higher polygon cycle limits for complex floor plans
- Option to keep or replace the imported DWG after model generation

### Create Room From Image (`CreateRoom.pushbutton`)
The original V1 tool. Supports both scanned image input (with OCR for dimension reading) and CAD DWG input.

**Image workflow:** rectifies a scanned sketch via homography, runs EasyOCR to extract dimension annotations, then builds the Revit model.
**CAD workflow:** same pipeline as V2 but with simpler topology recognition.

---

## How It Works

Both tools follow the same core pipeline:

```
DWG/DXF file
     │
     ▼
Import into Revit view (DWGImportOptions)
     │
     ▼
Extract geometry (lines, arcs, polylines)
Convert units: Revit feet → centimeters
     │
     ▼
Classify entities by layer name (regex patterns in cad_layer_map.json)
Categories: walls | doors | windows | dimensions | ignore
     │
     ▼
Topology recognition
  - Snap endpoints within tolerance
  - Detect room polygon (rectilinear or arbitrary)
  - Detect openings (explicit door/window elements or inferred gaps)
  - V2: bridge wall gaps, detect double-wall pairs, read dimension hints
     │
     ▼
Build Revit model
  - Create continuous walls from merged collinear polygon edges
  - Place hosted door/window families on walls (Revit performs the cut)
  - Create internal partition walls from detected internal wall centerlines
  - Create floor slab
  - Apply dimension annotations
```

---

## Module Structure

### V2 (`CreateFromCADV2.pushbutton/`)

| File | Purpose |
|---|---|
| `script.py` | Entry point — UI dialogs, Revit transaction, model assembly |
| `v2_cad_extract.py` | Walks Revit CAD geometry hierarchy, extracts lines/arcs, converts units |
| `v2_cad_classify.py` | Classifies entities by layer name using regex patterns |
| `v2_cad_recognition.py` | Topology engine — polygon detection, opening inference, double-wall & dimension-hint logic |
| `v2_snapshot.py` | Saves debug JSON snapshots of each processing stage |
| `cad_config.json` | Tunable parameters (tolerances, defaults for wall/door/window sizes) |
| `cad_layer_map.json` | Regex patterns mapping layer names to categories |

### V1 (`CreateRoom.pushbutton/`)

| File | Purpose |
|---|---|
| `script.py` | Entry point — handles both image and CAD workflows |
| `cad_extract.py` | CAD geometry extraction |
| `cad_classify.py` | Layer-based entity classification |
| `cad_topology.py` | Topology recognition (rectilinear + polygon modes) |
| `geometry_builder.py` | Builds Revit room geometry from parsed measurements |
| `cad_report.py` | Generates debug snapshots of CAD stages |
| `ocr_extract.py` | Runs EasyOCR on scanned images |
| `rectification.py` | Image rectification via homography (for scanned sketches) |
| `measurement_parser.py` | Parses OCR-extracted dimension tokens |
| `overlay_layout.py` | Computes image overlay positioning in the Revit view |
| `snapshot_manager.py` | Debug snapshot management |
| `layout_solver.py` | Solves room layout constraints from parsed measurements |

---

## Configuration

### `cad_layer_map.json`
Maps layer name regex patterns to element categories:
```json
{
  "walls":   ["^A?-?WALL(S)?$", "^WALL_.*", ...],
  "doors":   ["^A?-?DOOR(S)?$", "^DOOR_.*", ...],
  "windows": ["^A?-?WIN(DOW)?S?$", ...],
  "ignore":  ["^DEFPOINTS$", "^HATCH.*", "^DIM.*", ...]
}
```

### `cad_config.json` (V2 defaults)
| Parameter | Default | Description |
|---|---|---|
| `endpoint_snap_mm` | 4.0 | Max gap to snap endpoints together |
| `min_segment_mm` | 8.0 | Minimum wall segment length |
| `opening_host_distance_mm` | 70.0 | Max distance from opening to host wall |
| `opening_gap_min_cm` | 55.0 | Minimum gap width to treat as opening |
| `opening_gap_max_cm` | 260.0 | Maximum gap width to treat as opening |
| `wall_gap_bridge_max_cm` | 220.0 | Max gap to bridge between wall segments |
| `default_wall_thickness_cm` | 20.0 | Fallback wall thickness |
| `default_door_width_cm` | 100.0 | Fallback door width |
| `default_door_height_cm` | 210.0 | Fallback door height |
| `default_window_width_cm` | 100.0 | Fallback window width |
| `default_window_sill_cm` | 105.0 | Fallback window sill height |
| `model_wall_merge_angle_deg` | 2.0 | Max angle delta for collinear edge merge before wall creation |
| `model_wall_join_tol_cm` | 1.0 | Endpoint join tolerance for perimeter edge merge |
| `model_wall_min_length_cm` | 20.0 | Skip very short perimeter wall runs |
| `model_min_opening_confidence` | 0.45 | Minimum opening confidence for automatic placement |
| `model_opening_end_clearance_cm` | 15.0 | Clamp opening center away from wall ends |
| `model_place_synthetic_openings` | false | If false, synthetic fallback openings are not placed |
| `enable_synthetic_window_fallback` | false | If true, recognition can add fallback window metadata |
| `internal_wall_min_length_cm` | 30.0 | Minimum internal wall length to keep |
| `internal_wall_perimeter_duplicate_tol_cm` | 12.0 | Reject internal walls that overlap perimeter walls |

---

## Current Status & Roadmap

### What Works (Feb 2026)
- **Single-room DWG** (room2.dwg): L-shaped room with double-line walls, door arc, window patterns — fully recognized, correct centerline polygon, 3 openings detected
- **Multi-room apartment DWG** (room3.dwg): outer perimeter detected as single polygon (14-20 vertices), 5+ openings detected. Internal room subdivision not yet supported.

### Double-Wall Handling Pipeline
The core challenge: architectural DWG files draw each wall as **two parallel lines** (inner + outer traces, typically 30cm apart). The tool must collapse these to a single centerline for Revit wall placement.

**Current pipeline:**
1. **Pair detection** — find parallel lines 4-45cm apart with >60% overlap
2. **Centerline generation** — compute midpoint between each pair, using union extent (longest of inner/outer)
3. **Collinear merge** — combine overlapping centerlines from multiple pairs of the same wall
4. **Intersection extension** — extend centerlines to meet perpendicular walls at corners
5. **Crossing split** — split centerlines at T-junctions to create connected graph nodes
6. **Cycle detection** — find closed room polygon in the resulting graph

### Planned Improvements

| Phase | Description | Status |
|-------|------------|--------|
| 1 | Double-wall centerline collapse | Done |
| 2 | Fold/zigzag removal in polygon post-processing | Done |
| 3 | Geometric window pattern detection (no layer names needed) | Partial |
| 4 | Opening merge — multiple openings per wall edge | Not started |
| 5 | Per-segment dimension matching (read dimension lines for exact sizes) | Not started |
| 6 | **Multi-room support** — detect internal walls, find individual room cycles | Not started |
| 7 | Door swing direction from arc geometry | Not started |
| 8 | Curved wall support | Not started |

### Phase 6: Multi-Room Support (Next Major Feature)
For apartment floor plans with internal partition walls:
- Detect the outer perimeter (already works)
- Identify internal wall centerlines (T-junctions already detected in the graph)
- Find individual room cycles within the perimeter
- Generate separate Revit rooms with shared walls
- Handle door openings between rooms

---

## Requirements

- [pyRevit](https://github.com/eirannejad/pyRevit) installed
- Autodesk Revit (tested with Revit 2024/2025)
- Python 3.x (via pyRevit's IronPython 3 / CPython runtime)
- EasyOCR (V1 image workflow only): `pip install easyocr`

## Installation

1. Copy `MyTools.extension` into your pyRevit extensions folder:
   - Default: `%APPDATA%\pyRevit\Extensions\`
2. Reload pyRevit (pyRevit tab → Reload)
3. The **MyTools** tab will appear in the Revit ribbon with the **Create** panel

## Usage

1. Open or create a floor plan view in Revit
2. *(Optional)* Pre-import a DWG into the view, or let the tool prompt you to browse for one
3. Click **Create From CAD V2** in the MyTools ribbon tab
4. Follow the prompts — the tool will classify layers, recognize geometry, and build the model
5. Debug snapshots are saved alongside the script for troubleshooting

## Troubleshooting (CAD V2)

- Check `03_topology.json` first:
  - `openings` contains recognized opening candidates (`type`, `host_edge`, `width_cm`, `confidence`)
  - `internal_walls_cm` contains detected partition wall centerlines
- Check `08_geometry_summary.json` after model build:
  - `opening_attempt_count`, `opening_placed_count`, `opening_failed_count`, `opening_skipped_count`
  - `opening_errors` for concise failure reasons
  - `opening_attempts` for per-opening details (type, host edge, mapped wall, reason)
  - `internal_wall_ids` and `internal_wall_rejected_count` for partition wall build diagnostics
