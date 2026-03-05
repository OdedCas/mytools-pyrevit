# codex.md

## Project
- Name: `mytools-pyrevit`
- Active command: `C2Rv5` (pyRevit pushbutton)
- Dev repo: `/home/cassu/mytools-pyrevit`
- Live pyRevit extension: `/mnt/c/Users/cassu/AppData/Roaming/pyRevit/extensions/MyTools.extension`

## Current Active Mode (2026-03-05)
- Script: `MyTools.tab/Create.panel/C2Rv5.pushbutton/script.py`
- Build id: `C2Rv5.2-build-2026-03-04-0021`
- `MARK_ONLY_MODE = True`
- `PHASE1_WALLS_ONLY = True`

Meaning:
- No walls are created.
- No doors/windows are placed.
- Tool places only text markers in active view:
  - `D` = detected door candidate
  - `W` = detected window candidate

## Detection Strategy (Mark-Only)
1. Topology-first:
   - infer wall axes from CAD wall layers
   - infer opening candidates from topology gaps/opening structure
   - classify candidates using gap and arc evidence
2. Layer fallback:
   - if topology gives none, derive markers from `A-DOORS` / `A-WINDOWS` primitives
   - fallback window guess can use `A-DOORS` when window layer is missing

## Required CAD Layer Contract
- Walls exterior: `A-WALL-EXT`
- Walls interior: `A-WALL-INT`
- Doors symbols: `A-DOORS`
- Windows symbols: `A-WINDOWS`

Normalized matching is used (case/space/hyphen/underscore insensitive).

## Key Files
- `MyTools.tab/Create.panel/C2Rv5.pushbutton/script.py`
- `README.md`
- `codex.md`

## Run / Validate in Revit
1. pyRevit -> Reload
2. Select one CAD `ImportInstance`
3. Run `C2Rv5`
4. Verify output contains:
   - `Running C2Rv5.2-build-2026-03-04-0021`
   - `Mode: MARK ONLY (no walls/doors/windows are created).`
   - `MARK-ONLY: topology candidates D=... W=...`
   - `MARK-ONLY mode: placed D marks=..., W marks=...`

## Deploy to Live pyRevit Extension
```bash
cp /home/cassu/mytools-pyrevit/MyTools.tab/Create.panel/C2Rv5.pushbutton/script.py \
  /mnt/c/Users/cassu/AppData/Roaming/pyRevit/extensions/MyTools.extension/MyTools.tab/Create.panel/C2Rv5.pushbutton/script.py
```

## Commit Scope Rule
When committing from this state, include only:
- `codex.md`
- `MyTools.tab/Create.panel/C2Rv5.pushbutton/script.py`

Do not include unrelated pending files unless explicitly requested.
