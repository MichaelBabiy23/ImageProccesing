# Dendrite Annotation Tool

Draw skeleton centerlines on SEM images. The tool auto-expands each stroke into a YOLO polygon by detecting the dendrite boundary perpendicular to your stroke.

## Setup

```bash
pip install opencv-python scipy numpy
```

## Run

```bash
cd "final project/annotator"
python annotator.py
```

## Controls

| Key / Action | Effect |
|---|---|
| Left-click + drag | Draw a centerline stroke |
| Right-click | Undo last stroke |
| `W` | Increase probe distance (+5px) — expand width search |
| `S` | Decrease probe distance (-5px) — shrink width search |
| `R` | Reset all strokes on current image |
| `Enter` or `Space` | Save annotations + visual, go to next image |
| `Q` | Skip current image (no save) |

## Output

`annotator/output/` contains:
- `<stem>.txt` — YOLO segmentation format (normalized polygon coords)
- `<stem>_annotated.png` — visual overlay for validation (green polygon, red skeleton)

Already-annotated images are tracked in `output/done.json` and skipped on re-run.

## Tuning

- **Probe distance** (`W`/`S`): how far perpendicular to look for the dendrite edge. Increase for thick dendrites, decrease for thin ones.
- **DISPLAY_SCALE** in `annotator.py`: set to `0.5` or `0.75` if images are too large for your screen.
- **threshold multiplier** (line ~`threshold = center_intensity * 0.55`): if auto-width is wrong, adjust this fraction.
