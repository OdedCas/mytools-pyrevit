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
  - Create walls along recognized polygon edges
  - Place door/window families at detected openings
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
