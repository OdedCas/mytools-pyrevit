# D_W_detection

pyRevit command for DWG-based opening annotation.

## What it creates
- Door text: `DD`
- Window text: `ww`
- Door linear dimensions (arch edge to nearest corner)

## How to use
1. Open a plan view with imported DWG.
2. Run `D_W_detection`.
3. Pick the DWG import when prompted.
4. Review summary dialog for:
- detected doors/windows
- created dimensions
- text marker counts

## Layer expectations
- Doors on `A-DOORS` (or close variants)
- Windows on `A-WINDOWS` (or close variants)

## File
- `script.py`
