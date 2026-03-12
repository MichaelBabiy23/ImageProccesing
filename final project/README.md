# SEM Dendrite Segmentation

Automatic pixel-wise semantic segmentation of lithium dendrites in Scanning Electron Microscope (SEM) images. Implements and compares two approaches: a classical computer vision pipeline and a deep learning (YOLO-Seg) pipeline.

Student ID: 323073734

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: numpy, opencv-python, scikit-image, ultralytics, PyQt5, PySide6.

## Project Structure

```
classic_pipeline.py       # Classic CV segmentation pipeline (Approach B)
yolo_pipeline.py          # YOLO-Seg deep learning pipeline (Approach A)
pipeline_gui.py           # Interactive GUI for classic pipeline parameter tuning
evaluate.py               # Evaluation metrics (Dice, IoU, Precision, Recall)
utils.py                  # Shared I/O, SEM cleaning, and visualization utilities
run_all.py                # End-to-end orchestrator for both pipelines
run_preset_single_image.py # Apply GUI preset to single image with comparison
annotator/app.py          # Polygon annotation tool for creating YOLO labels
yolo_viewer/viewer.py     # YOLO prediction viewer with GT overlay
yolo_training/
  prepare_and_train.py    # Dataset preparation + YOLO training launcher
  evaluate_yolo.py        # YOLO model evaluation from training results
requirements.txt          # Python dependencies
README.md                 # This file
```

## Interactive GUI

Parameter tuning GUI with real-time preview of all pipeline stages.

Windows:

```bash
run-gui.bat
```

macOS / Linux:

```bash
python3 pipeline_gui.py
```

## Classic Pipeline

Four-stage pipeline: pre-processing (histogram normalization, CLAHE, bilateral filter), segmentation (adaptive thresholding), post-processing (baseline cut, small component removal), and separation (distance transform + watershed). Plus skeletonization via Zhang-Suen thinning.

Single image:

```bash
python classic_pipeline.py <path/to/image.tif>
```

Batch (all images in a directory):

```bash
python classic_pipeline.py --input <images_dir> --output <output_dir>
```

Self-test with synthetic data:

```bash
python classic_pipeline.py
```

Intermediate images (10 stages + overlays) are saved per image under `output/classic/<image_name>/`.

## YOLO Pipeline

Uses YOLOv11 instance segmentation with transfer learning from COCO pretrained weights.

### Dataset Preparation and Training

Using the dedicated training script (handles data splitting and image conversion):

```bash
cd yolo_training
python prepare_and_train.py
```

Or using the generic pipeline script:

```bash
python yolo_pipeline.py train --data <path/to/data.yaml> [--epochs 100] [--model yolo11n-seg.pt]
```

### Inference

Single image:

```bash
python yolo_pipeline.py predict --model <weights.pt> --source <image.tif>
```

Batch:

```bash
python yolo_pipeline.py predict --model <weights.pt> --source <images_dir> --output <output_dir>
```

### YOLO Viewer

Interactive viewer with ground truth and prediction overlays:

```bash
cd yolo_viewer
python viewer.py
```

## Annotation Tool

Interactive polygon annotation tool for creating YOLO-format segmentation labels:

```bash
cd annotator
python app.py
```

Supports loading masks from the classic pipeline output as annotation seeds, draggable vertex editing, undo/redo, and export in YOLO polygon format.

## Full Pipeline (run_all.py)

The orchestrator runs both pipelines on the same images and produces all deliverables in one command.

Both pipelines with evaluation:

```bash
python run_all.py --images data/raw/Easy --gt data/ground_truth/ --yolo-model weights/best.pt --output output/
```

Classic pipeline only (no YOLO model):

```bash
python run_all.py --images data/raw/Easy --gt data/ground_truth/
```

Self-test with synthetic images:

```bash
python run_all.py
```

The orchestrator runs 5 stages:
1. Classic pipeline on all images
2. YOLO pipeline (if model provided)
3. Skeletonization of classic masks
4. 4-panel comparison figures (Source, Classic, YOLO, Skeleton)
5. Evaluation with metrics and failure analysis (if ground truth provided)

## Evaluation

Computes Dice, IoU, Precision, and Recall against ground truth masks. Generates 4-panel comparison figures (Source, Classic Mask, YOLO Mask, Skeleton) and a text summary with failure analysis.

### Failure Analysis

The evaluation module automatically performs failure analysis on images with Dice score below a configurable threshold (default: 0.5). Root cause characterization based on metric patterns:

- **Over-segmentation** (low precision, high recall) -- noise or background included as dendrite
- **Under-segmentation** (high precision, low recall) -- thin branches or low-contrast dendrites missed
- **Fundamental mismatch** (low precision, low recall) -- wrong region detected or severe artifacts
- Cross-method insights: if classic fails but YOLO succeeds, likely non-uniform illumination; if YOLO fails but classic succeeds, likely out-of-distribution sample

### Self-Test

```bash
python evaluate.py
```

## Output Structure

```
output/
  classic/
    <image_name>/
      01_original.png ... 10_skeleton.png
      overlay_mask_on_orig.png
      overlay_skel_on_orig.png
  yolo/
    train/dendrite_seg/weights/best.pt
    <image_name>_mask.png
    <image_name>_skeleton.png
  comparisons/
    <image_name>_comparison.png
  evaluation/
    <image_name>_comparison.png
    metrics_summary.txt
```
