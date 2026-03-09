# C2Rv6 - Codex Notes

## Purpose
Create a safer replacement path for C2Rv5 without deleting C2Rv5:

- use existing strong vector recognition from V2
- add explicit QA preview/export checkpoint before creation

## Reused modules
From `CreateFromCADV2.pushbutton`:
- `v2_cad_extract.py`
- `v2_cad_classify.py`
- `v2_cad_recognition.py`
- `v2_snapshot.py`
- `script.py` (`build_model_from_topology`)

## Key difference from C2Rv5
- C2Rv6 stops before build and asks for confirmation after QA export.
- QA export includes CSV + JSON written into the snapshot run folder.

## Runtime assumptions
- Must run inside Revit/pyRevit.
- Active view must be Floor Plan.
- One DWG import instance must be selected or picked.
